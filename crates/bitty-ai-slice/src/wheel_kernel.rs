//! Unified WheelKernel facade coordinating storage, task DAG, context compiler, and action protocols (AI-0167).
//!
//! Durable path notes (AI-0178, formal):
//! - The database file is opened exactly once per [`WheelKernel`]; the single
//!   [`rusqlite::Connection`] is shared into [`ContentStore`] and [`TaskEngine`]
//!   (see `content_store` single-connection docs). Two connections on one file
//!   split pragmas and let a crash land between the blob write and the HEAD
//!   update; one connection plus one `transaction()` per commit keeps tree blob,
//!   checkpoint row, and HEAD/branch refs atomic.
//! - `commit_checkpoint` is atomic (see [`ContentStore::commit_checkpoint_atomic`]):
//!   HEAD advances only when the transaction commits.
//! - Recovery validates the HEAD -> checkpoint -> tree triple and fails closed
//!   on any partial triple (typed [`FacadeError`], no partial HEAD advance).
//! - `recent_actions` and `active_task_id` are explicitly non-persisted
//!   session state: they reset on every open and never survive a reopen.
//!   Durability covers HEAD, the checkpoint DAG, blobs/refs, and tasks only.
//!
//! Provides the boundary plane and runtime orchestrator for Wheel:
//! - Coordinates transactional SQLite [`ContentStore`] (data plane).
//! - Coordinates graph-theoretic [`TaskEngine`] (control plane).
//! - Coordinates Merkle [`ContextTree`] cognitive state (compilation plane).
//! - Coordinates [`ActionEngine`] auto-spillover pipeline (execution plane).
//! - Integrates streaming SSE sessions and three-zone context compilation under budget.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bitty_ai_session::sessions::{
    SessionBinding, SessionError, WheelSessionId, bind_session, bump_session_epoch, list_sessions,
    resolve_session,
};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::action_protocol::{ActionEngine, ActionOutcome, SpilloverConfig};
use crate::content_hash::ContentHash;
use crate::content_store::{Checkpoint, CheckpointDraft, ContentStore, Rationale};
use crate::context_compiler::{
    CompiledContext, CompilerBudgetConfig, ContextCompiler, ContextTree, EntryKind, TreeEntry,
};
use crate::facade::{AiStreamSession, FacadeError};
use crate::merge::{GcOptions, GcReport, MergeInput};
use crate::session_refs::{
    BranchName, PruneReport, ReflogEntry, commit_checkpoint_with_branch, create_branch,
    delete_branch, get_branch, list_branches, prune_reflog, read_reflog, rename_branch,
    update_branch,
};
use crate::task_dag::{TaskDraft, TaskEngine, TaskEngineError, TaskId, TaskNode, TaskView};

/// Maximum retained uncollapsed action outcomes in memory for Zone 3 compilation (64).
pub const MAX_RECENT_ACTIONS: usize = 64;

/// Reflog reason recorded for branch moves driven by [`WheelKernel::commit_checkpoint`].
const WHEEL_COMMIT_REFLOG_REASON: &str = "wheel commit";

/// Reflog actor recorded for branch moves driven by [`WheelKernel::commit_checkpoint`].
const WHEEL_COMMIT_REFLOG_ACTOR: &str = "wheel-kernel";

/// Fenced resume outcome (AI-0197).
///
/// `session_id` is `Some` on the session-id (fenced) path and `None` on the
/// bare branch/`HEAD` (unfenced read) path. `checkpoint` is the resolved live
/// tip. `generation` is the live task-engine generation re-read at resume
/// time, `0` when the checkpoint's task names no task row (informational
/// snapshot, never a fence). `pending_unknowns` is always empty with
/// `pending_log_absent = true`: no durable pending tool-call log exists, so
/// resume honestly reports HEAD plus generation and says so (a durable
/// pending log is a separate follow-up slice). `fence_token` is the admitted
/// epoch on the session path, `0` on the branch path (no fence admitted).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeReport {
    /// Bound session id, or `None` for unfenced branch/`HEAD` resumes.
    pub session_id: Option<WheelSessionId>,
    /// Branch followed (`heads/...`, or `HEAD` for direct HEAD resumes).
    pub branch: String,
    /// Resolved live-tip checkpoint.
    pub checkpoint: ContentHash,
    /// Live task-engine generation at resume time (`0` = no task binding).
    pub generation: u64,
    /// In-flight tool-call ids surviving the crash (always empty: no log).
    pub pending_unknowns: Vec<String>,
    /// Always `true`: no durable pending log exists (honest flag).
    pub pending_log_absent: bool,
    /// Admitted fencing epoch (session path) or `0` (branch path).
    pub fence_token: u64,
}

/// Map a sessions-plane error into the stringly-typed [`FacadeError`].
///
/// No new `FacadeError` variant: `Store` already carries every fail-closed
/// refusal shape on this boundary (the task directs reusing an existing
/// variant where one fits). The typed [`SessionError::StaleEpoch`] refusal
/// (mirroring the adoption-rule `StaleEpoch` field shape) survives in the
/// message text; callers needing typed discrimination use the sessions plane
/// directly.
fn map_session_err(err: SessionError) -> FacadeError {
    FacadeError::Store(err.to_string())
}

/// Unified runtime orchestrator coordinating all Wheel upstream layers.
pub struct WheelKernel {
    content_store: ContentStore,
    task_engine: TaskEngine,
    active_tree: ContextTree,
    action_engine: ActionEngine,
    spillover_config: SpilloverConfig,
    budget_config: CompilerBudgetConfig,
    active_task_id: Option<TaskId>,
    head_checkpoint: Option<ContentHash>,
    recent_actions: Vec<ActionOutcome>,
}

impl WheelKernel {
    /// Open a persistent WheelKernel backed by a SQLite database at `path` with default configs.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FacadeError> {
        Self::open_with_config(
            path,
            CompilerBudgetConfig::default(),
            SpilloverConfig::default(),
        )
    }

    /// Open a persistent WheelKernel with caller-supplied budget and spillover configurations.
    ///
    /// Opens the file exactly once, applies the durable pragma profile, admits
    /// content and task schemas, claims the single-writer lock, shares the
    /// handle into the content store and task engine, then validates the
    /// HEAD -> checkpoint -> tree triple (fail closed on partial).
    pub fn open_with_config(
        path: impl AsRef<Path>,
        budget_config: CompilerBudgetConfig,
        spillover_config: SpilloverConfig,
    ) -> Result<Self, FacadeError> {
        let path = path.as_ref();
        crate::content_store::check_sqlite_magic(path).map_err(FacadeError::from)?;
        let conn = Connection::open(path)
            .map_err(|e| FacadeError::Store(crate::content_store::map_busy_for_facade(e)))?;
        // Fail-fast admission: the busy timeout stays zero until the writer
        // claim succeeds, so a second open while the first lives returns
        // promptly instead of waiting out the steady-state timeout.
        conn.busy_timeout(Duration::ZERO)
            .map_err(|e| FacadeError::Store(crate::content_store::map_busy_for_facade(e)))?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(|e| FacadeError::Store(crate::content_store::map_busy_for_facade(e)))?;
        crate::content_store::admit_or_init(&conn).map_err(FacadeError::from)?;
        crate::task_dag::admit_task_or_init(&conn).map_err(FacadeError::from)?;
        crate::content_store::apply_durable_pragmas(&conn).map_err(FacadeError::from)?;
        conn.busy_timeout(Duration::ZERO)
            .map_err(|e| FacadeError::Store(crate::content_store::map_busy_for_facade(e)))?;
        crate::content_store::init_content_schema(&conn).map_err(FacadeError::from)?;
        {
            use crate::task_dag::TASK_SCHEMA;
            conn.execute_batch(TASK_SCHEMA)
                .map_err(|e| FacadeError::TaskDag(format!("task schema init: {e}")))?;
        }
        crate::content_store::verify_profile(&conn).map_err(FacadeError::from)?;
        crate::content_store::claim_writer_fast(&conn).map_err(FacadeError::from)?;
        let shared = Arc::new(Mutex::new(conn));
        let content_store =
            ContentStore::from_shared(Arc::clone(&shared)).map_err(FacadeError::from)?;
        let task_engine =
            TaskEngine::from_shared(Arc::clone(&shared)).map_err(FacadeError::from)?;
        let (head_checkpoint, active_tree) = Self::recover(&content_store)?;

        let action_engine = ActionEngine::new(spillover_config.clone());

        Ok(Self {
            content_store,
            task_engine,
            active_tree,
            action_engine,
            spillover_config,
            budget_config,
            active_task_id: None,
            head_checkpoint,
            recent_actions: Vec::new(),
        })
    }

    /// Open a durable WheelKernel (formal path, same profile as [`Self::open`]).
    pub fn open_durable(path: impl AsRef<Path>) -> Result<Self, FacadeError> {
        Self::open(path)
    }

    /// Validate the HEAD -> checkpoint -> tree triple after open.
    ///
    /// Returns the admitted HEAD plus the decoded tree, or fails closed with
    /// a typed error when any link is missing or undecodable. Never advances
    /// a partial HEAD: on error the caller receives `Err` and no kernel.
    fn recover(
        content_store: &ContentStore,
    ) -> Result<(Option<ContentHash>, ContextTree), FacadeError> {
        let head = content_store.get_ref("HEAD").map_err(FacadeError::from)?;
        let Some(ref head_hash) = head else {
            return Ok((None, ContextTree::new()));
        };
        let Some(checkpoint) = content_store
            .get_checkpoint(head_hash)
            .map_err(FacadeError::from)?
        else {
            return Err(FacadeError::Store(format!(
                "durable recovery: HEAD {head_hash} has no checkpoint row; refusing partial HEAD"
            )));
        };
        let Some(ref tree_hash) = checkpoint.tree_hash else {
            return Err(FacadeError::Store(format!(
                "durable recovery: checkpoint {head_hash} has no tree blob link; refusing partial HEAD"
            )));
        };
        let Some(blob_bytes) = content_store
            .get_blob(tree_hash)
            .map_err(FacadeError::from)?
        else {
            return Err(FacadeError::Store(format!(
                "durable recovery: checkpoint {head_hash} tree {tree_hash} blob missing; refusing partial HEAD"
            )));
        };
        let tree = ContextTree::from_canonical_bytes(&blob_bytes).map_err(|e| {
            FacadeError::Store(format!("durable recovery: tree decode failed: {e}"))
        })?;
        Ok((head, tree))
    }

    /// Open an in-memory WheelKernel with default configurations.
    pub fn open_in_memory() -> Result<Self, FacadeError> {
        Self::open_in_memory_with_config(
            CompilerBudgetConfig::default(),
            SpilloverConfig::default(),
        )
    }

    /// Open an in-memory WheelKernel with caller-supplied budget and spillover configurations.
    pub fn open_in_memory_with_config(
        budget_config: CompilerBudgetConfig,
        spillover_config: SpilloverConfig,
    ) -> Result<Self, FacadeError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| FacadeError::Store(crate::content_store::map_busy_for_facade(e)))?;
        crate::content_store::apply_durable_pragmas(&conn).map_err(FacadeError::from)?;
        crate::content_store::init_content_schema(&conn).map_err(FacadeError::from)?;
        {
            use crate::task_dag::TASK_SCHEMA;
            conn.execute_batch(TASK_SCHEMA)
                .map_err(|e| FacadeError::TaskDag(format!("task schema init: {e}")))?;
        }
        let shared = Arc::new(Mutex::new(conn));
        let content_store =
            ContentStore::from_shared(Arc::clone(&shared)).map_err(FacadeError::from)?;
        let task_engine =
            TaskEngine::from_shared(Arc::clone(&shared)).map_err(FacadeError::from)?;
        let action_engine = ActionEngine::new(spillover_config.clone());

        Ok(Self {
            content_store,
            task_engine,
            active_tree: ContextTree::new(),
            action_engine,
            spillover_config,
            budget_config,
            active_task_id: None,
            head_checkpoint: None,
            recent_actions: Vec::new(),
        })
    }

    // --- Task Lifecycle & Control Plane Operations ---

    /// Create a new task in the DAG control plane.
    pub fn create_task(&mut self, draft: TaskDraft, now_ms: u64) -> Result<TaskNode, FacadeError> {
        self.task_engine
            .create_task(draft, now_ms)
            .map_err(FacadeError::from)
    }

    /// Retrieve an immutable view of a task by its identifier.
    pub fn get_task(&self, id: &TaskId) -> Result<Option<TaskNode>, FacadeError> {
        match self.task_engine.get_task(id) {
            Ok(node) => Ok(Some(node)),
            Err(TaskEngineError::TaskNotFound(_)) => Ok(None),
            Err(e) => Err(FacadeError::from(e)),
        }
    }

    /// List all tasks currently managed in the DAG.
    pub fn list_tasks(&self) -> Result<Vec<TaskNode>, FacadeError> {
        self.task_engine.list_tasks().map_err(FacadeError::from)
    }

    /// Retrieve an enriched view of a task including its inlined prerequisite dependencies.
    pub fn get_task_view(&self, id: &TaskId) -> Result<Option<TaskView>, FacadeError> {
        match self.task_engine.get_task_view(id) {
            Ok(view) => Ok(Some(view)),
            Err(TaskEngineError::TaskNotFound(_)) => Ok(None),
            Err(e) => Err(FacadeError::from(e)),
        }
    }

    /// List all tasks currently managed in the DAG with inlined prerequisite dependencies.
    pub fn list_task_views(&self) -> Result<Vec<TaskView>, FacadeError> {
        self.task_engine
            .list_task_views()
            .map_err(FacadeError::from)
    }

    /// Set or clear the active task driving Zone 2 compilation.
    pub fn set_active_task(&mut self, id: Option<TaskId>) -> Result<(), FacadeError> {
        if let Some(ref task_id) = id {
            let exists = self
                .task_engine
                .has_task(task_id)
                .map_err(FacadeError::from)?;
            if !exists {
                return Err(FacadeError::TaskDag(format!(
                    "task '{}' not found",
                    task_id.as_str()
                )));
            }
        }
        self.active_task_id = id;
        Ok(())
    }

    /// Get the identifier of the active task, if one is designated.
    #[must_use]
    pub fn active_task(&self) -> Option<&TaskId> {
        self.active_task_id.as_ref()
    }

    /// Start a task, binding a worker identity and transitioning to Running.
    pub fn start_task(
        &mut self,
        id: &TaskId,
        worker_id: &str,
        now_ms: u64,
    ) -> Result<TaskNode, FacadeError> {
        self.task_engine
            .assign_task(id, worker_id, now_ms)
            .map_err(FacadeError::from)?;
        self.task_engine.get_task(id).map_err(FacadeError::from)
    }

    /// Complete a task successfully, triggering cascade readiness on downstream dependents.
    pub fn complete_task(
        &mut self,
        id: &TaskId,
        expected_generation: u64,
        checkpoint: Option<ContentHash>,
        now_ms: u64,
    ) -> Result<TaskNode, FacadeError> {
        self.task_engine
            .complete_task(id, expected_generation, checkpoint, now_ms)
            .map_err(FacadeError::from)
    }

    /// Mark a task failed, triggering cascade blocking across its downstream dependency graph.
    pub fn fail_task(
        &mut self,
        id: &TaskId,
        expected_generation: u64,
        error_message: &str,
        now_ms: u64,
    ) -> Result<TaskNode, FacadeError> {
        self.task_engine
            .fail_task(id, expected_generation, error_message, now_ms)
            .map_err(FacadeError::from)
    }

    /// Cancel a task, triggering cascade blocking across its downstream dependency graph.
    pub fn cancel_task(&mut self, id: &TaskId, now_ms: u64) -> Result<TaskNode, FacadeError> {
        self.task_engine
            .cancel_task(id, now_ms)
            .map_err(FacadeError::from)
    }

    /// Retry a previously failed or cancelled task.
    pub fn retry_task(&mut self, id: &TaskId, now_ms: u64) -> Result<TaskNode, FacadeError> {
        self.task_engine
            .retry_task(id, now_ms)
            .map_err(FacadeError::from)
    }

    /// Compute deterministic topological execution order of tasks using Kahn's algorithm.
    pub fn topological_sort(&self) -> Result<Vec<TaskId>, FacadeError> {
        self.task_engine
            .topological_sort()
            .map_err(FacadeError::from)
    }

    // --- Merkle Context Tree & Slot Operations ---

    /// Store a blob in the content store and insert/update the named slot in the active Merkle tree.
    pub fn put_slot(
        &mut self,
        name: &str,
        content: &[u8],
        now_ms: u64,
    ) -> Result<ContentHash, FacadeError> {
        let hash = self
            .content_store
            .put_blob(content, now_ms)
            .map_err(FacadeError::from)?;
        let entry = TreeEntry::new(name, hash, EntryKind::Blob, content.len())
            .map_err(|e| FacadeError::Compiler(e.to_string()))?;
        self.active_tree.insert(entry);
        Ok(hash)
    }

    /// Retrieve raw bytes of a named slot from the active Merkle tree and content store.
    pub fn get_slot(&self, name: &str) -> Result<Option<Vec<u8>>, FacadeError> {
        if let Some(entry) = self.active_tree.get(name) {
            self.content_store
                .get_blob(&entry.hash)
                .map_err(FacadeError::from)
        } else {
            Ok(None)
        }
    }

    /// Remove a named slot from the active Merkle tree.
    pub fn remove_slot(&mut self, name: &str) -> bool {
        self.active_tree.remove(name).is_some()
    }

    /// List all slot entries in the active Merkle tree.
    #[must_use]
    pub fn list_slots(&self) -> Vec<TreeEntry> {
        self.active_tree.entries().cloned().collect()
    }

    /// Read-only reference to the active Merkle context tree.
    #[must_use]
    pub fn active_tree(&self) -> &ContextTree {
        &self.active_tree
    }

    /// Compute the current deterministic Merkle digest of the active context tree.
    #[must_use]
    pub fn tree_hash(&self) -> ContentHash {
        self.active_tree.digest()
    }

    // --- Cognitive Checkpointing & Commit Plane Operations ---

    /// Commit a cognitive checkpoint linking the current Merkle tree, structured rationale,
    /// and parent reference into the immutable content store.
    ///
    /// Atomic (AI-0178): the tree blob, checkpoint row, and HEAD ref update
    /// commit in one SQLite `transaction()` via
    /// [`ContentStore::commit_checkpoint_atomic`] (no branch) or
    /// [`commit_checkpoint_with_branch`](crate::session_refs::commit_checkpoint_with_branch)
    /// (with branch). HEAD
    /// advances only when the transaction commits; a torn write (crash
    /// between blob insert and HEAD update) replays as the pre-crash HEAD on
    /// reopen, never a half-advanced HEAD.
    ///
    /// The `branch` move commits in the same transaction as the checkpoint
    /// and HEAD (AI-0180):
    /// [`commit_checkpoint_with_branch`](crate::session_refs::commit_checkpoint_with_branch)
    /// writes the blob, checkpoint, HEAD, branch ref, and reflog row
    /// atomically, so every branch move appends a reflog row. The checkpoint
    /// commit is authoritative, so the update is not fast-forward-only. A
    /// failure rolls back the whole commit (including HEAD); the in-memory
    /// HEAD mirrors the durable HEAD only on success.
    pub fn commit_checkpoint(
        &mut self,
        rationale: Rationale,
        branch: Option<&str>,
        now_ms: u64,
    ) -> Result<Checkpoint, FacadeError> {
        let tree_bytes = self.active_tree.canonical_bytes();
        let parents = match &self.head_checkpoint {
            Some(h) => vec![*h],
            None => Vec::new(),
        };
        let task_id = self
            .active_task_id
            .as_ref()
            .map(|id| id.as_str().to_string())
            .unwrap_or_else(|| "task-unassigned".to_string());
        let draft = CheckpointDraft {
            parents,
            task_id,
            agent_id: "wheel-kernel".to_string(),
            rationale,
            tree_hash: None,
            summary: "cognitive checkpoint".to_string(),
            timestamp_ms: now_ms,
        };
        let Some(branch_name) = branch else {
            let checkpoint = self
                .content_store
                .commit_checkpoint_atomic(&tree_bytes, now_ms, draft, &["HEAD"], now_ms)
                .map_err(FacadeError::from)?;
            self.head_checkpoint = Some(checkpoint.id);
            return Ok(checkpoint);
        };
        // Single-transaction HEAD + branch commit: namespace validation happens
        // inside the refs-plane function before any write, so a bad branch name fails
        // without persisting a checkpoint or advancing HEAD. A reflog failure
        // rolls back the whole commit (including HEAD).
        let checkpoint = commit_checkpoint_with_branch(
            &mut self.content_store,
            &tree_bytes,
            now_ms,
            draft,
            branch_name,
            WHEEL_COMMIT_REFLOG_REASON,
            WHEEL_COMMIT_REFLOG_ACTOR,
            now_ms,
        )
        .map_err(FacadeError::from)?;
        self.head_checkpoint = Some(checkpoint.id);
        Ok(checkpoint)
    }

    /// Retrieve a checkpoint by its content address.
    pub fn get_checkpoint(&self, hash: &ContentHash) -> Result<Option<Checkpoint>, FacadeError> {
        self.content_store
            .get_checkpoint(hash)
            .map_err(FacadeError::from)
    }

    /// Get the current HEAD checkpoint hash, if one has been committed.
    #[must_use]
    pub fn head_checkpoint(&self) -> Option<&ContentHash> {
        self.head_checkpoint.as_ref()
    }

    /// Traverse backward through parent commits starting from HEAD up to `max_depth`.
    pub fn log(&self, max_depth: usize) -> Result<Vec<Checkpoint>, FacadeError> {
        if let Some(ref head) = self.head_checkpoint {
            self.content_store
                .log(head, max_depth)
                .map_err(FacadeError::from)
        } else {
            Ok(Vec::new())
        }
    }

    // --- Typed Branch Verbs & Reflog (refs-plane v1, AI-0180) ---
    //
    // Thin passthroughs over the [`crate::session_refs`] refs-plane free
    // functions: no logic lives here, only error mapping into [`FacadeError`].

    /// Create a branch pointing at an existing checkpoint (no force).
    pub fn create_branch(
        &mut self,
        name: &str,
        target: &ContentHash,
        reason: &str,
        actor: &str,
        now_ms: u64,
    ) -> Result<BranchName, FacadeError> {
        create_branch(&mut self.content_store, name, target, reason, actor, now_ms)
            .map_err(FacadeError::from)
    }

    /// Move a branch to an existing checkpoint, recording history.
    pub fn update_branch(
        &mut self,
        name: &str,
        target: &ContentHash,
        reason: &str,
        actor: &str,
        now_ms: u64,
        fast_forward_only: bool,
    ) -> Result<BranchName, FacadeError> {
        update_branch(
            &mut self.content_store,
            name,
            target,
            reason,
            actor,
            now_ms,
            fast_forward_only,
        )
        .map_err(FacadeError::from)
    }

    /// Delete a branch, leaving a tombstone reflog row. Returns the deleted tip.
    pub fn delete_branch(
        &mut self,
        name: &str,
        reason: &str,
        actor: &str,
        now_ms: u64,
    ) -> Result<ContentHash, FacadeError> {
        delete_branch(&mut self.content_store, name, reason, actor, now_ms)
            .map_err(FacadeError::from)
    }

    /// Atomically rename a branch. Returns the moved tip.
    pub fn rename_branch(
        &mut self,
        old_name: &str,
        new_name: &str,
        reason: &str,
        actor: &str,
        now_ms: u64,
    ) -> Result<ContentHash, FacadeError> {
        rename_branch(
            &mut self.content_store,
            old_name,
            new_name,
            reason,
            actor,
            now_ms,
        )
        .map_err(FacadeError::from)
    }

    /// List all branches (`heads/` only), ordered by name.
    pub fn list_branches(&self) -> Result<Vec<(BranchName, ContentHash)>, FacadeError> {
        list_branches(&self.content_store).map_err(FacadeError::from)
    }

    /// Get a branch tip (`heads/` only). Missing branches return `None`.
    pub fn get_branch(&self, name: &str) -> Result<Option<ContentHash>, FacadeError> {
        get_branch(&self.content_store, name).map_err(FacadeError::from)
    }

    /// Read reflog history for any well-formed ref name, newest-first.
    pub fn read_reflog(&self, name: &str, limit: usize) -> Result<Vec<ReflogEntry>, FacadeError> {
        read_reflog(&self.content_store, name, limit).map_err(FacadeError::from)
    }

    /// Explicitly expire reflog history for one ref, bounded by age and count.
    ///
    /// Thin passthrough over [`prune_reflog`][crate::session_refs::prune_reflog]:
    /// no validation, policy, or clock selection lives here. The caller
    /// supplies every bound (`older_than_ms`, `max_rows`) and every clock
    /// (`tombstone_grace_ms`, `now_ms`); the floor
    /// ([`MIN_REFLOG_FLOOR`][crate::session_refs::MIN_REFLOG_FLOOR]) and the
    /// tombstone-grace rule come verbatim from the refs plane via
    /// [`FacadeError`].
    pub fn prune_reflog(
        &mut self,
        name: &str,
        older_than_ms: u64,
        max_rows: usize,
        tombstone_grace_ms: u64,
        now_ms: u64,
    ) -> Result<PruneReport, FacadeError> {
        prune_reflog(
            &mut self.content_store,
            name,
            older_than_ms,
            max_rows,
            tombstone_grace_ms,
            now_ms,
        )
        .map_err(FacadeError::from)
    }

    // --- Session Bindings & Resume (AI-0197) ---
    //
    // Durable caller-provided session ids bound to branch tips, plus the
    // fenced resume and cheap-fork verbs over them. Thin orchestration over
    // the [`bitty_ai_session::sessions`] plane: shape validation, fencing,
    // and row storage live there; what lives here is tip resolution (refs
    // plane), live task-generation reads (task engine), and working-tree
    // sync so the next commit continues from the resumed checkpoint.

    /// Read the live task-engine generation for a checkpoint's task.
    ///
    /// Returns the task row's `generation` when the checkpoint's `task_id`
    /// names a live task, else `0` (no task binding: `task-unassigned` or a
    /// removed task). Never fails: an unbound task is ordinary, not corrupt.
    fn live_task_generation(&self, checkpoint: &Checkpoint) -> u64 {
        let Ok(task_id) = TaskId::new(checkpoint.task_id.clone()) else {
            return 0;
        };
        match self.task_engine.get_task(&task_id) {
            Ok(node) => node.generation,
            Err(_) => 0,
        }
    }

    /// Bind a caller-provided session id to a branch tip (no force).
    ///
    /// The branch must exist; its live tip's checkpoint supplies the stored
    /// head and the generation snapshot (see
    /// [`bitty_ai_session::sessions`] for snapshot-vs-live semantics). The
    /// row mints epoch `1`. A bound id refuses with `AlreadyExists` (via
    /// [`FacadeError::Store`); the kernel never mints or reuses session ids.
    pub fn new_session(
        &mut self,
        session_id: &str,
        branch: &str,
        now_ms: u64,
    ) -> Result<SessionBinding, FacadeError> {
        let id = WheelSessionId::parse(session_id).map_err(map_session_err)?;
        let branch_name = BranchName::parse(branch).map_err(FacadeError::from)?;
        let tip = match get_branch(&self.content_store, branch_name.as_str())
            .map_err(FacadeError::from)?
        {
            Some(hash) => hash,
            None => return Err(map_session_err(SessionError::NotFound)),
        };
        let checkpoint = match self
            .content_store
            .get_checkpoint(&tip)
            .map_err(FacadeError::from)?
        {
            Some(checkpoint) => checkpoint,
            None => {
                return Err(FacadeError::Store(format!(
                    "session branch {branch_name} points at a missing checkpoint; refusing"
                )));
            }
        };
        let generation = self.live_task_generation(&checkpoint);
        bind_session(
            &self.content_store,
            &id,
            &branch_name,
            &tip,
            generation,
            1,
            now_ms,
        )
        .map_err(map_session_err)
    }

    /// Resume a session by id (fenced) or a branch/`HEAD` ref (unfenced read).
    ///
    /// Session-id path: the id must be bound; the claim is admitted only
    /// when `claim_epoch > stored epoch`, else [`SessionError::StaleEpoch`]
    /// refuses with zero writes. On success the row advances to the branch's
    /// live tip (sessions follow branch moves), the live generation, and the
    /// claimed epoch, and the report carries that epoch as `fence_token`. A
    /// bound session whose branch was deleted refuses as `NotFound`.
    ///
    /// Branch/`HEAD` path: resolves the tip with no fencing and no row
    /// write; `session_id` is `None` and `fence_token` is `0` (no fence
    /// admitted). Resume is read-only apart from the session-path epoch
    /// bump: it never replays tool calls and never re-executes effects.
    ///
    /// Both paths refuse when the working tree is dirty (zero writes,
    /// [`FacadeError::Store`] naming `uncommitted`): the gate runs before
    /// any store write -- including the session-path epoch bump -- so a
    /// dirty `slot.put` without commit never loses slots to the resume sync,
    /// matching the `merge_commit` / GC gates.
    ///
    /// `pending_unknowns` is always empty with `pending_log_absent = true`:
    /// no durable pending tool-call log exists anywhere in this tree (the
    /// runtime pending set is in-memory only), so an honest resume reports
    /// HEAD plus generation and says so. `generation` is the live
    /// task-engine generation re-read at resume time (`0` when the HEAD
    /// checkpoint's task names no task row).
    pub fn resume_session(
        &mut self,
        ref_or_branch: &str,
        claim_epoch: u64,
        now_ms: u64,
    ) -> Result<ResumeReport, FacadeError> {
        // Dirty gate first (both paths): a resume sync would blindly
        // overwrite the in-memory tree, so uncommitted slots refuse before
        // any store read or write, exactly like `merge_commit` / GC.
        if self.working_tree_dirty()? {
            return Err(FacadeError::Store(
                "resume refused: working tree has uncommitted changes; commit or discard before resuming"
                    .to_string(),
            ));
        }
        // Session namespace and branch namespace are disjoint by
        // construction (session ids never contain `/`, branches always do),
        // so a successful session-id parse selects the session path
        // exclusively: an unbound id is NotFound, never a branch retry.
        if let Ok(id) = WheelSessionId::parse(ref_or_branch) {
            let Some(binding) =
                resolve_session(&self.content_store, &id).map_err(map_session_err)?
            else {
                return Err(map_session_err(SessionError::NotFound));
            };
            let branch_name = BranchName::parse(&binding.branch)
                .map_err(|_| map_session_err(SessionError::Corrupt))?;
            let tip = match get_branch(&self.content_store, branch_name.as_str())
                .map_err(FacadeError::from)?
            {
                Some(hash) => hash,
                None => return Err(map_session_err(SessionError::NotFound)),
            };
            let checkpoint = match self
                .content_store
                .get_checkpoint(&tip)
                .map_err(FacadeError::from)?
            {
                Some(checkpoint) => checkpoint,
                None => {
                    return Err(FacadeError::Store(format!(
                        "session branch {branch_name} points at a missing checkpoint; refusing"
                    )));
                }
            };
            let generation = self.live_task_generation(&checkpoint);
            // Fallible tree load runs BEFORE the epoch bump: a missing tree
            // link, missing blob, or undecodable tree refuses here with zero
            // row writes, so the same claim stays admissible for retry.
            let tree = self.load_tree_for(&tip)?;
            let advanced = bump_session_epoch(
                &self.content_store,
                &id,
                claim_epoch,
                &tip,
                generation,
                now_ms,
            )
            .map_err(map_session_err)?;
            // Infallible assignment runs only after the bump succeeds.
            self.active_tree = tree;
            self.head_checkpoint = Some(tip);
            return Ok(ResumeReport {
                session_id: Some(id),
                branch: advanced.branch,
                checkpoint: tip,
                generation,
                pending_unknowns: Vec::new(),
                pending_log_absent: true,
                fence_token: advanced.epoch,
            });
        }
        let (branch_text, tip) = if ref_or_branch == "HEAD" {
            let head = match self
                .content_store
                .get_ref("HEAD")
                .map_err(FacadeError::from)?
            {
                Some(hash) => hash,
                None => return Err(map_session_err(SessionError::NotFound)),
            };
            ("HEAD".to_owned(), head)
        } else {
            let branch_name = BranchName::parse(ref_or_branch).map_err(FacadeError::from)?;
            let tip = match get_branch(&self.content_store, branch_name.as_str())
                .map_err(FacadeError::from)?
            {
                Some(hash) => hash,
                None => return Err(map_session_err(SessionError::NotFound)),
            };
            (branch_name.as_str().to_owned(), tip)
        };
        let checkpoint = match self
            .content_store
            .get_checkpoint(&tip)
            .map_err(FacadeError::from)?
        {
            Some(checkpoint) => checkpoint,
            None => {
                return Err(FacadeError::Store(format!(
                    "resume ref {branch_text} points at a missing checkpoint; refusing"
                )));
            }
        };
        let generation = self.live_task_generation(&checkpoint);
        self.sync_working_tree_to(&tip)?;
        Ok(ResumeReport {
            session_id: None,
            branch: branch_text,
            checkpoint: tip,
            generation,
            pending_unknowns: Vec::new(),
            pending_log_absent: true,
            fence_token: 0,
        })
    }

    /// Fork a branch at an existing checkpoint tip (cheap snapshot).
    ///
    /// Thin passthrough over the refs-plane `create_branch`: a fork is a new
    /// ref pointing at the same checkpoint, so parent blobs are shared by
    /// construction (no data is copied; both tips reach the same tree blob
    /// through content addressing). Missing targets refuse as
    /// `MissingTarget`; taken names as `AlreadyExists`.
    pub fn fork_branch(
        &mut self,
        name: &str,
        from_tip: &ContentHash,
        reason: &str,
        actor: &str,
        now_ms: u64,
    ) -> Result<BranchName, FacadeError> {
        create_branch(
            &mut self.content_store,
            name,
            from_tip,
            reason,
            actor,
            now_ms,
        )
        .map_err(FacadeError::from)
    }

    /// List all bound sessions, ordered by session id ascending.
    ///
    /// Global to the database file (no directory scoping, owner decision
    /// AI-0197). Thin passthrough over the sessions plane.
    pub fn list_sessions(&self) -> Result<Vec<SessionBinding>, FacadeError> {
        list_sessions(&self.content_store).map_err(map_session_err)
    }

    /// Load and decode the tree blob for a checkpoint tip.
    ///
    /// Fallible half of the resume sync: fails closed on a missing
    /// checkpoint row, tree link, blob, or undecodable bytes (same shape as
    /// `recover`). The session resume path calls this BEFORE
    /// `bump_session_epoch` so tree failures refuse with zero row writes;
    /// the infallible state assignment then runs only after the bump
    /// succeeds. [`Self::sync_working_tree_to`] is load-plus-assign for the
    /// paths (branch resume) that carry no epoch.
    fn load_tree_for(&self, tip: &ContentHash) -> Result<ContextTree, FacadeError> {
        let checkpoint = match self
            .content_store
            .get_checkpoint(tip)
            .map_err(FacadeError::from)?
        {
            Some(checkpoint) => checkpoint,
            None => {
                return Err(FacadeError::Store(format!(
                    "resume sync: checkpoint {tip} has no row; refusing"
                )));
            }
        };
        let Some(tree_hash) = checkpoint.tree_hash else {
            return Err(FacadeError::Store(format!(
                "resume sync: checkpoint {tip} has no tree blob link; refusing"
            )));
        };
        let Some(blob_bytes) = self
            .content_store
            .get_blob(&tree_hash)
            .map_err(FacadeError::from)?
        else {
            return Err(FacadeError::Store(format!(
                "resume sync: checkpoint {tip} tree {tree_hash} blob missing; refusing"
            )));
        };
        ContextTree::from_canonical_bytes(&blob_bytes)
            .map_err(|e| FacadeError::Store(format!("resume sync: tree decode failed: {e}")))
    }

    /// Sync the in-memory working tree and HEAD pointer to a resumed tip.
    ///
    /// Loads the checkpoint's tree blob and decodes it into the active tree
    /// (fail closed on a missing row, link, blob, or undecodable bytes, same
    /// shape as `recover`), so the next `commit_checkpoint` continues from
    /// the resumed lineage instead of forking history off a stale tree.
    fn sync_working_tree_to(&mut self, tip: &ContentHash) -> Result<(), FacadeError> {
        let tree = self.load_tree_for(tip)?;
        self.active_tree = tree;
        self.head_checkpoint = Some(*tip);
        Ok(())
    }

    // --- Merge + GC (session plane, AI-0190) ---
    //
    // Thin passthroughs over the [`crate::merge`] free functions: no
    // validation, policy, or taxonomy lives here. All fail-closed errors
    // (`NoCommonAncestor` / `CrissCross` / `Conflicts` zero-writes /
    // `StaleGeneration`; GC truncated-resume; `dry_run` / preview parity)
    // come verbatim from `merge.rs` via [`FacadeError`].

    /// Check whether the working tree has uncommitted changes relative to HEAD.
    ///
    /// Read-only: performs no store writes. With no HEAD, an empty tree is
    /// clean and any slot is dirty. Otherwise byte-compares
    /// `active_tree.canonical_bytes()` against the HEAD checkpoint tree blob.
    /// A missing checkpoint row, tree link, or blob fails closed via
    /// [`FacadeError::Store`] (same shape as `recover`).
    fn working_tree_dirty(&self) -> Result<bool, FacadeError> {
        let Some(head_hash) = &self.head_checkpoint else {
            return Ok(!self.active_tree.is_empty());
        };
        let Some(checkpoint) = self
            .content_store
            .get_checkpoint(head_hash)
            .map_err(FacadeError::from)?
        else {
            return Err(FacadeError::Store(format!(
                "working tree check: HEAD {head_hash} has no checkpoint row; refusing"
            )));
        };
        let Some(tree_hash) = checkpoint.tree_hash else {
            return Err(FacadeError::Store(format!(
                "working tree check: checkpoint {head_hash} has no tree blob link; refusing"
            )));
        };
        let Some(blob_bytes) = self
            .content_store
            .get_blob(&tree_hash)
            .map_err(FacadeError::from)?
        else {
            return Err(FacadeError::Store(format!(
                "working tree check: checkpoint {head_hash} tree {tree_hash} blob missing; refusing"
            )));
        };
        Ok(self.active_tree.canonical_bytes() != blob_bytes)
    }

    /// Merge two checkpoint tips into a two-parent commit on `target_branch`.
    ///
    /// Refuses when the working tree is dirty (zero writes, [`FacadeError::Store`]
    /// naming `uncommitted`): the gate runs before any store write so a dirty
    /// `slot.put` without commit never loses slots to a blind overwrite. Also
    /// refuses when HEAD exists and `input.ours` is not HEAD (zero writes):
    /// otherwise the sync would drop committed HEAD-only slots from the
    /// working tree and HEAD would jump to a lineage that does not descend
    /// from the old HEAD. On a clean merge the working tree syncs from the
    /// merged tree blob before the in-memory HEAD advances, so the next
    /// `commit_checkpoint` keeps theirs-only slots. Validation stays in
    /// `merge.rs`.
    pub fn merge_commit(&mut self, input: MergeInput) -> Result<Checkpoint, FacadeError> {
        if self.working_tree_dirty()? {
            return Err(FacadeError::Store(
                "merge refused: working tree has uncommitted changes; commit or discard before merging"
                    .to_string(),
            ));
        }
        if let Some(head) = &self.head_checkpoint {
            if input.ours != *head {
                return Err(FacadeError::Store(format!(
                    "merge refused: ours {} is not HEAD {head}; merge into HEAD only",
                    input.ours
                )));
            }
        }
        let checkpoint =
            crate::merge::merge_commit(&mut self.content_store, &self.task_engine, input)
                .map_err(FacadeError::from)?;
        let Some(tree_hash) = checkpoint.tree_hash else {
            return Err(FacadeError::Store(format!(
                "merge sync: checkpoint {} has no tree blob link; refusing",
                checkpoint.id
            )));
        };
        let Some(blob_bytes) = self
            .content_store
            .get_blob(&tree_hash)
            .map_err(FacadeError::from)?
        else {
            return Err(FacadeError::Store(format!(
                "merge sync: checkpoint {} tree {tree_hash} blob missing; refusing",
                checkpoint.id
            )));
        };
        let tree = ContextTree::from_canonical_bytes(&blob_bytes)
            .map_err(|e| FacadeError::Store(format!("merge sync: tree decode failed: {e}")))?;
        self.active_tree = tree;
        self.head_checkpoint = Some(checkpoint.id);
        Ok(checkpoint)
    }

    /// Read-only GC preview: byte-matches the next destructive batch.
    ///
    /// Refuses when the working tree is dirty (zero reads beyond the gate,
    /// [`FacadeError::Store`] naming `uncommitted`, same shape as
    /// [`Self::merge_commit`]): `slot.put` payloads live only in the active
    /// tree until `commit_checkpoint`, while GC roots cover refs and reflog
    /// only, so even a preview must not run while uncommitted payloads are
    /// invisible to reachability. The gate runs before any store read and is
    /// identical to the [`Self::collect_garbage`] gate, so preview==collect
    /// parity holds by construction. Thin passthrough over
    /// [`crate::merge::gc_preview`] once clean.
    pub fn gc_preview(&self, options: &GcOptions) -> Result<GcReport, FacadeError> {
        if self.working_tree_dirty()? {
            return Err(FacadeError::Store(
                "gc refused: working tree has uncommitted changes; commit or discard before collecting"
                    .to_string(),
            ));
        }
        crate::merge::gc_preview(&self.content_store, options).map_err(FacadeError::from)
    }

    /// Bounded destructive GC, or preview when `options.dry_run` is set.
    ///
    /// Same dirty gate as [`Self::gc_preview`] (identical call, identical
    /// [`FacadeError::Store`] refusal naming `uncommitted`): runs before any
    /// store read or write, so a dirty `slot.put` without commit can never
    /// lose its payload to collection. Thin passthrough over
    /// [`crate::merge::collect_garbage`] once clean.
    pub fn collect_garbage(&mut self, options: &GcOptions) -> Result<GcReport, FacadeError> {
        if self.working_tree_dirty()? {
            return Err(FacadeError::Store(
                "gc refused: working tree has uncommitted changes; commit or discard before collecting"
                    .to_string(),
            ));
        }
        crate::merge::collect_garbage(&mut self.content_store, options).map_err(FacadeError::from)
    }

    // --- Action Protocol & Auto-Spillover Operations ---

    /// Record an action outcome, auto-spilling oversized stdout/stderr into the blob store
    /// and retaining the bounded outcome for Zone 3 context compilation.
    #[allow(clippy::too_many_arguments)]
    pub fn record_action_outcome(
        &mut self,
        action_id: impl Into<String>,
        success: bool,
        exit_code: Option<i32>,
        duration_ms: u64,
        raw_stdout: &str,
        raw_stderr: &str,
        timestamp_ms: u64,
    ) -> Result<ActionOutcome, FacadeError> {
        let outcome = self
            .action_engine
            .process_outcome(
                action_id,
                success,
                exit_code,
                duration_ms,
                raw_stdout,
                raw_stderr,
                &mut self.content_store,
                timestamp_ms,
            )
            .map_err(FacadeError::from)?;

        if self.recent_actions.len() >= MAX_RECENT_ACTIONS {
            self.recent_actions.remove(0);
        }
        self.recent_actions.push(outcome.clone());

        Ok(outcome)
    }

    /// Read-only access to recent uncollapsed action outcomes.
    #[must_use]
    pub fn recent_actions(&self) -> &[ActionOutcome] {
        &self.recent_actions
    }

    /// Clear all retained uncollapsed action outcomes.
    pub fn clear_recent_actions(&mut self) {
        self.recent_actions.clear();
    }

    // --- Three-Zone Context Compilation ---

    /// Compile a three-zone context prompt combining invariant prefix (Zone 1),
    /// cognitive state and active task (Zone 2), and turn prompt with recent actions (Zone 3).
    pub fn compile_context(
        &self,
        system_instruction: &str,
        project_rules: &[&str],
        tool_schemas: &[&str],
        turn_prompt: &str,
    ) -> Result<CompiledContext, FacadeError> {
        let mut compiler = ContextCompiler::new();
        compiler.system_prompt = system_instruction.to_string();
        compiler.project_rules = project_rules.join("\n\n");
        compiler.tool_schemas = tool_schemas.join("\n\n");

        if let Some(ref task_id) = self.active_task_id {
            match self.task_engine.get_task(task_id) {
                Ok(node) => compiler.active_task = Some(node),
                Err(TaskEngineError::TaskNotFound(_)) => compiler.active_task = None,
                Err(e) => return Err(FacadeError::from(e)),
            }
        }

        let mut checkpoints = self.log(8)?;
        checkpoints.reverse();
        compiler.checkpoints = checkpoints;
        compiler.context_tree = self.active_tree.clone();
        compiler.turn_prompt = turn_prompt.to_string();
        compiler.uncollapsed_observations = self
            .recent_actions
            .iter()
            .map(|a| a.format_for_context())
            .collect();

        compiler
            .compile(&self.budget_config)
            .map_err(FacadeError::from)
    }

    // --- Subsystem Accessors & Helpers ---

    /// Read-only reference to the underlying task engine.
    #[must_use]
    pub fn task_engine(&self) -> &TaskEngine {
        &self.task_engine
    }

    /// Mutable reference to the underlying task engine.
    pub fn task_engine_mut(&mut self) -> &mut TaskEngine {
        &mut self.task_engine
    }

    /// Read-only reference to the underlying content store.
    #[must_use]
    pub fn content_store(&self) -> &ContentStore {
        &self.content_store
    }

    /// Mutable reference to the underlying content store.
    pub fn content_store_mut(&mut self) -> &mut ContentStore {
        &mut self.content_store
    }

    /// Read-only reference to the compiler budget config.
    #[must_use]
    pub fn budget_config(&self) -> &CompilerBudgetConfig {
        &self.budget_config
    }

    /// Read-only reference to the spillover config.
    #[must_use]
    pub fn spillover_config(&self) -> &SpilloverConfig {
        &self.spillover_config
    }

    /// Spawn a new streaming chat completion session.
    #[must_use]
    pub fn new_stream_session(&self) -> AiStreamSession {
        AiStreamSession::new()
    }
}

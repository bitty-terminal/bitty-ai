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

use rusqlite::Connection;

use crate::action_protocol::{ActionEngine, ActionOutcome, SpilloverConfig};
use crate::content_store::{Checkpoint, CheckpointDraft, ContentHash, ContentStore, Rationale};
use crate::context_compiler::{
    CompiledContext, CompilerBudgetConfig, ContextCompiler, ContextTree, EntryKind, TreeEntry,
};
use crate::facade::{AiStreamSession, FacadeError};
use crate::session_refs::{BranchName, ReflogEntry};
use crate::task_dag::{TaskDraft, TaskEngine, TaskEngineError, TaskId, TaskNode, TaskView};

/// Maximum retained uncollapsed action outcomes in memory for Zone 3 compilation (64).
pub const MAX_RECENT_ACTIONS: usize = 64;

/// Reflog reason recorded for branch moves driven by [`WheelKernel::commit_checkpoint`].
const WHEEL_COMMIT_REFLOG_REASON: &str = "wheel commit";

/// Reflog actor recorded for branch moves driven by [`WheelKernel::commit_checkpoint`].
const WHEEL_COMMIT_REFLOG_ACTOR: &str = "wheel-kernel";

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
    /// [`ContentStore::commit_checkpoint_atomic`]. HEAD advances only when the
    /// transaction commits; a torn write (crash between blob insert and HEAD
    /// update) replays as the pre-crash HEAD on reopen, never a half-advanced
    /// HEAD.
    ///
    /// The `branch` move routes through the refs-plane verbs (AI-0180):
    /// [`ContentStore::create_branch`] when the branch is new,
    /// [`ContentStore::update_branch`] otherwise, so every branch move
    /// appends a reflog row. The checkpoint commit is authoritative, so the
    /// update is not fast-forward-only. A branch-verb failure after the
    /// atomic commit still reports `Err`, with the in-memory HEAD mirroring
    /// the durably advanced HEAD.
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
        // Validate the namespace before any write so a bad branch name fails
        // without persisting a checkpoint or advancing HEAD.
        let parsed = BranchName::parse(branch_name).map_err(FacadeError::from)?;
        let checkpoint = self
            .content_store
            .commit_checkpoint_atomic(&tree_bytes, now_ms, draft, &["HEAD"], now_ms)
            .map_err(FacadeError::from)?;
        let branch_result = match self
            .content_store
            .get_branch(parsed.as_str())
            .map_err(FacadeError::from)?
        {
            Some(_) => self.content_store.update_branch(
                parsed.as_str(),
                &checkpoint.id,
                WHEEL_COMMIT_REFLOG_REASON,
                WHEEL_COMMIT_REFLOG_ACTOR,
                now_ms,
                false,
            ),
            None => self.content_store.create_branch(
                parsed.as_str(),
                &checkpoint.id,
                WHEEL_COMMIT_REFLOG_REASON,
                WHEEL_COMMIT_REFLOG_ACTOR,
                now_ms,
            ),
        };
        // Durable HEAD already advanced; mirror it before reporting.
        self.head_checkpoint = Some(checkpoint.id);
        branch_result.map_err(FacadeError::from)?;
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
    // Thin passthroughs over the [`ContentStore`] refs-plane verbs: no logic
    // lives here, only error mapping into [`FacadeError`].

    /// Create a branch pointing at an existing checkpoint (no force).
    pub fn create_branch(
        &mut self,
        name: &str,
        target: &ContentHash,
        reason: &str,
        actor: &str,
        now_ms: u64,
    ) -> Result<BranchName, FacadeError> {
        self.content_store
            .create_branch(name, target, reason, actor, now_ms)
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
        self.content_store
            .update_branch(name, target, reason, actor, now_ms, fast_forward_only)
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
        self.content_store
            .delete_branch(name, reason, actor, now_ms)
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
        self.content_store
            .rename_branch(old_name, new_name, reason, actor, now_ms)
            .map_err(FacadeError::from)
    }

    /// List all branches (`heads/` only), ordered by name.
    pub fn list_branches(&self) -> Result<Vec<(BranchName, ContentHash)>, FacadeError> {
        self.content_store
            .list_branches()
            .map_err(FacadeError::from)
    }

    /// Get a branch tip (`heads/` only). Missing branches return `None`.
    pub fn get_branch(&self, name: &str) -> Result<Option<ContentHash>, FacadeError> {
        self.content_store
            .get_branch(name)
            .map_err(FacadeError::from)
    }

    /// Read reflog history for any well-formed ref name, newest-first.
    pub fn read_reflog(&self, name: &str, limit: usize) -> Result<Vec<ReflogEntry>, FacadeError> {
        self.content_store
            .read_reflog(name, limit)
            .map_err(FacadeError::from)
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

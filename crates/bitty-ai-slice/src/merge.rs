//! Merge-commit + GC reachability (AI-0181, session plane v1).
//!
//! This module is the session follow-up to the AI-0180 refs plane. It adds two
//! durable operations over the existing [`ContentStore`][crate::content_store::ContentStore]
//! data plane plus the [`TaskEngine`][crate::task_dag::TaskEngine] control plane:
//!
//! - [`merge_commit`]: strict lowest-common-ancestor merge of two checkpoint
//!   tips into a two-parent commit written via
//!   [`ContentStore::commit_checkpoint_with_branch`].
//! - [`collect_garbage`] / [`gc_preview`]: bounded reachability garbage
//!   collection over checkpoints and tree blobs.
//!
//! ## Merge-commit
//!
//! [`merge_commit`] runs six ordered steps, all read-only except the last:
//!
//! 1. Validate `target_branch` with [`BranchName`][crate::session_refs::BranchName]
//!    parsing (zero writes).
//! 2. Optional [`TaskEngine`] generation fence: when
//!    [`MergeInput::expected_task_generation`] is `Some`, the live generation
//!    of [`MergeInput::task_id`] must equal the supplied value, else
//!    [`MergeError::StaleGeneration`] (zero writes).
//! 3. Strict LCA via [`strict_merge_base`]: full ancestor-set intersection
//!    plus minimal-element filter. No common ancestor yields
//!    [`MergeError::NoCommonAncestor`]; more than one minimal common ancestor
//!    yields [`MergeError::CrissCrossHistory`]. Same BFS shape as
//!    [`ContentStore::merge_base`][crate::content_store::ContentStore::merge_base]
//!    but strict instead of first-found.
//! 4. Materialize the three trees (base, ours, theirs) from durable storage
//!    only: `get_checkpoint` -> `tree_hash` -> `get_blob` (digest-verified) ->
//!    [`ContextTree::from_canonical_bytes`][crate::context_compiler::ContextTree::from_canonical_bytes].
//!    An absent checkpoint row or blob row is [`MergeError::MissingObject`];
//!    an absent `tree_hash` link, a digest mismatch, or undecodable tree bytes
//!    is [`MergeError::Corrupt`].
//! 5. [`ContextTree::merge_3way`][crate::context_compiler::ContextTree::merge_3way]:
//!    any slot conflict returns [`MergeError::Conflicts`] with zero writes.
//! 6. Clean merges persist one [`Checkpoint`] with `parents == [ours, theirs]`
//!    through [`ContentStore::commit_checkpoint_with_branch`], which advances
//!    HEAD, advances `target_branch`, and appends exactly one reflog row in a
//!    single SQLite transaction. An oversize `reason`/`actor` therefore rolls
//!    the whole commit back, including HEAD.
//!
//! Time/Space: steps 3-4 cost O(V + E) store reads over the reachable DAG
//! (V checkpoints, E parent edges); the minimal-element filter costs
//! O(|C| * (V + E)) for |C| common ancestors. Space is O(V) hashes.
//!
//! ## Caller duties (R5 owner, R3 tombstone filtering)
//!
//! - R5 owner: the caller owns `target_branch` selection and merge-result
//!   ownership. The commit path is authoritative (not fast-forward-only): it
//!   advances HEAD and `target_branch` together. Branch protection, review, or
//!   merge-policy decisions live outside this module; the caller must choose
//!   the intended owner branch before calling. `HEAD` itself cannot be a merge
//!   target here ([`BranchName`] refuses it); HEAD-only commits stay on the
//!   Wheel commit path.
//! - R3 tombstone filtering: reflog deletion tombstones (all-zero `new_hash`)
//!   carry no payload and are never merge inputs. Never pass the all-zero hash
//!   as `ours`/`theirs` (it has no checkpoint row and reports
//!   [`MergeError::MissingObject`]); application-level slot tombstones, if any,
//!   must be filtered by the caller before building merge trees.
//!
//! ## Garbage collection
//!
//! Roots are the union of all [`ContentStore::list_refs`] targets and every
//! reflog `old_hash`/`new_hash` with `at_ms >= now_ms - reflog_grace_ms`
//! (saturating). All-zero tombstone hashes are skipped as roots: they have no
//! payload to keep. Reachability is a BFS over `parents_json` from those
//! roots. Unreachable checkpoints are pruned; blobs referenced by no surviving
//! (reachable) checkpoint are pruned. A surviving checkpoint references its
//! `tree_hash` blob, and each surviving tree references every entry hash named
//! in its canonical bytes (slot-payload blobs stored via `put_blob`, for
//! example [`WheelKernel::put_slot`][crate::wheel_kernel::WheelKernel::put_slot]),
//! plus transitively every entry of each nested tree named via `EntryKind::Tree`
//! up to an explicit depth cap: GC materializes each surviving tree blob via
//! [`ContextTree::from_canonical_bytes`][crate::context_compiler::ContextTree::from_canonical_bytes]
//! and retains the union of all entry hashes, recursing into nested tree
//! blobs. A surviving checkpoint with a missing `tree_hash` blob row, a digest
//! mismatch, undecodable tree bytes, an undecodable nested tree blob, or a
//! nesting chain deeper than the cap is [`MergeError::Corrupt`] (fail closed,
//! no partial GC).
//!
//! Batching: one [`collect_garbage`] call deletes at most
//! [`GcOptions::max_deletes_per_call`] rows (checkpoints first, then blobs, in
//! ascending hash order) inside a single SQLite transaction and reports
//! [`GcReport::truncated`] when garbage remains, so the caller resumes by
//! calling again. [`GcOptions::max_deletes_per_call`] of `0` deletes nothing
//! and reports `truncated` whenever garbage remains (no progress; supply at
//! least `1` to reclaim). [`gc_preview`] is read-only and byte-matches the
//! next destructive call's sets under the same options (same order, same
//! bound). [`GcOptions::dry_run`] makes [`collect_garbage`] behave exactly
//! like [`gc_preview`] (no writes).
//!
//! Reflog rows are never pruned here; reflog expiry is a separate future
//! operation. GC writes no reflog rows and reads no blob payloads for
//! tombstones (zero-hash entries are skipped before any read).
//!
//! Standalone blobs are NOT retained: content blobs referenced by no surviving
//! tree — for example action auto-spillover payloads (stored via
//! [`BlobSink::store_blob`][crate::action_protocol::BlobSink::store_blob]
//! into the blob table but held only by an
//! [`ActionPayload::Spilled`][crate::action_protocol::ActionPayload]
//! [`BlobPointer`][crate::action_protocol::BlobPointer] outside the store) or
//! any orphan `put_blob` never committed into a slot entry — are treated as
//! garbage. Refs and reflog entries pin checkpoint hashes only and cannot pin
//! a bare blob hash: to retain such a blob, commit it into a slot entry of a
//! tree reachable from a live ref or an in-window reflog entry, or adopt a
//! separate retention policy (future work).
//!
//! ## Errors
//!
//! Every [`MergeError`] display string is a static literal except
//! [`MergeError::MissingObject`], which carries the fixed-size content hash,
//! and [`MergeError::StaleGeneration`], which carries the two generation
//! counters. Caller-supplied text (branch names, reasons, actors, summaries,
//! slot names) is never echoed. Storage details are dropped to static text so
//! error output stays caller-clock deterministic.
//!
//! Clocks are caller-supplied (`at_ms` / `now_ms`) throughout: no
//! `Instant`/`SystemTime`, no threads, no scheduler. Standard library plus the
//! existing `rusqlite` dependency only.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use rusqlite::params;

use crate::content_store::{
    Checkpoint, CheckpointDraft, ContentHash, ContentStore, ContentStoreError,
};
use crate::context_compiler::{ContextTree, EntryKind, SlotConflict};
use crate::facade::FacadeError;
use crate::session_refs::{BranchName, RefError};
use crate::task_dag::{TaskEngine, TaskEngineError, TaskId};

/// All-zero content hash marking a deletion tombstone (mirrors git).
///
/// GC skips this value as a root (it has no payload); merge never accepts it
/// as an input (it has no checkpoint row).
fn tombstone_hash() -> ContentHash {
    ContentHash::from_bytes([0u8; 32])
}

/// Input for [`merge_commit`].
#[derive(Debug, Clone)]
pub struct MergeInput {
    /// Our side tip (first merge parent).
    pub ours: ContentHash,
    /// Their side tip (second merge parent).
    pub theirs: ContentHash,
    /// Branch receiving the merge commit (`heads/` namespace, validated).
    pub target_branch: String,
    /// Task owning this merge (checkpoint field + generation-fence key).
    pub task_id: String,
    /// Agent authoring the merge commit.
    pub agent_id: String,
    /// Structured rationale for the merge commit.
    pub rationale: crate::content_store::Rationale,
    /// Short human-readable merge summary.
    pub summary: String,
    /// Optional worker generation fence against the live [`TaskEngine`] task.
    ///
    /// When `Some`, the live generation of [`Self::task_id`] must equal this
    /// value or [`merge_commit`] fails with [`MergeError::StaleGeneration`]
    /// before any write. When `None`, no fence is checked.
    pub expected_task_generation: Option<u64>,
    /// Reflog actor recorded for the target-branch move.
    pub actor: String,
    /// Reflog reason recorded for the target-branch move.
    pub reason: String,
    /// Caller-supplied timestamp (tree blob, checkpoint, and ref update).
    pub at_ms: u64,
}

/// Typed failures for [`merge_commit`], [`strict_merge_base`],
/// [`gc_preview`], and [`collect_garbage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeError {
    /// The two histories share no ancestor.
    NoCommonAncestor,
    /// More than one minimal common ancestor (criss-cross history).
    CrissCrossHistory,
    /// Slot-level merge conflicts; no write was performed.
    Conflicts(Vec<SlotConflict>),
    /// A referenced checkpoint or blob row does not exist.
    MissingObject(ContentHash),
    /// The live task generation differs from the fenced expectation.
    ///
    /// `expected` is the live [`TaskEngine`] generation, `found` is the
    /// caller-supplied [`MergeInput::expected_task_generation`] value,
    /// matching the existing engine convention (live first, supplied second).
    StaleGeneration {
        /// Live generation read from the task engine.
        expected: u64,
        /// Caller-supplied generation that failed the fence.
        found: u64,
    },
    /// Caller-supplied string violates its bound or namespace.
    InvalidName,
    /// Named task or branch does not exist.
    NotFound,
    /// Reserved for branch-create races; the merge path upserts.
    AlreadyExists,
    /// Stored data failed integrity validation. Fail closed, no partial state.
    Corrupt,
    /// Underlying storage failure. Static text; details are dropped.
    Storage,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCommonAncestor => write!(f, "no common ancestor for merge"),
            Self::CrissCrossHistory => {
                write!(f, "criss-cross history with multiple merge bases")
            }
            Self::Conflicts(_) => write!(f, "merge conflicts detected"),
            Self::MissingObject(hash) => {
                write!(f, "merge object {hash} does not exist")
            }
            Self::StaleGeneration { expected, found } => {
                write!(
                    f,
                    "stale task generation: expected {expected}, found {found}"
                )
            }
            Self::InvalidName => write!(f, "invalid merge name, reason, or actor string"),
            Self::NotFound => write!(f, "merge task or branch not found"),
            Self::AlreadyExists => write!(f, "merge target branch already exists"),
            Self::Corrupt => write!(f, "merge store corrupt or incompatible"),
            Self::Storage => write!(f, "merge storage error"),
        }
    }
}

impl std::error::Error for MergeError {}

impl From<MergeError> for FacadeError {
    fn from(err: MergeError) -> Self {
        Self::Store(err.to_string())
    }
}

/// Map a raw SQLite error to [`MergeError`] with static text.
fn map_sqlite(err: rusqlite::Error) -> MergeError {
    if let rusqlite::Error::SqliteFailure(failure, _) = &err {
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return MergeError::Corrupt;
        }
    }
    MergeError::Storage
}

/// Map a store error to [`MergeError`], dropping untrusted payloads.
///
/// Missing-link variants carry the hash into [`MergeError::MissingObject`];
/// everything else collapses to static text.
fn map_store(err: ContentStoreError) -> MergeError {
    match err {
        ContentStoreError::MissingTarget(hash)
        | ContentStoreError::MissingParent(hash)
        | ContentStoreError::MissingTree(hash) => MergeError::MissingObject(hash),
        ContentStoreError::CorruptData { .. }
        | ContentStoreError::CorruptCheckpoint { .. }
        | ContentStoreError::Corrupt { .. }
        | ContentStoreError::InvalidHash(_)
        | ContentStoreError::Json(_)
        | ContentStoreError::EmptyField(_)
        | ContentStoreError::InvalidRefName(_)
        | ContentStoreError::OversizedField { .. } => MergeError::Corrupt,
        ContentStoreError::WriterBusy => MergeError::Storage,
        ContentStoreError::Sqlite(err) => map_sqlite(err),
    }
}

/// Map a branch-verb error to [`MergeError`].
fn map_ref_err(err: RefError) -> MergeError {
    match err {
        RefError::InvalidName | RefError::ProtectedHead | RefError::ReservedNamespace => {
            MergeError::InvalidName
        }
        RefError::AlreadyExists => MergeError::AlreadyExists,
        RefError::NotFound => MergeError::NotFound,
        RefError::MissingTarget(hash) => MergeError::MissingObject(hash),
        RefError::Corrupt => MergeError::Corrupt,
        RefError::Storage(_) => MergeError::Storage,
    }
}

/// Map a task-engine lookup error to [`MergeError`].
///
/// [`TaskId`] parse failures (caller input) collapse to [`MergeError::InvalidName`];
/// a missing task is [`MergeError::NotFound`]; stored-data problems are
/// [`MergeError::Corrupt`]. Unreachable control-plane variants fail closed as
/// [`MergeError::Corrupt`].
fn map_task(err: TaskEngineError) -> MergeError {
    match err {
        TaskEngineError::TaskNotFound(_) => MergeError::NotFound,
        TaskEngineError::InvalidTaskId(_)
        | TaskEngineError::EmptyField(_)
        | TaskEngineError::OversizedField { .. } => MergeError::InvalidName,
        TaskEngineError::InvalidHash(_) => MergeError::Corrupt,
        TaskEngineError::Corrupt { .. } => MergeError::Corrupt,
        TaskEngineError::InvalidStatusTransition { .. }
        | TaskEngineError::StaleGeneration { .. }
        | TaskEngineError::CycleDetected { .. }
        | TaskEngineError::SelfDependency(_)
        | TaskEngineError::DuplicateTask(_)
        | TaskEngineError::DependencyNotFound { .. } => MergeError::Corrupt,
        TaskEngineError::WriterBusy => MergeError::Storage,
        TaskEngineError::Sqlite(err) => map_sqlite(err),
    }
}

/// Collect the full ancestor set of `root` (including `root`) via BFS.
///
/// Same traversal shape as
/// [`ContentStore::merge_base`][crate::content_store::ContentStore::merge_base].
/// An absent `root` row is [`MergeError::MissingObject`]; a dangling parent
/// link reached mid-traversal is [`MergeError::Corrupt`].
fn ancestor_set(
    store: &ContentStore,
    root: &ContentHash,
) -> Result<HashSet<ContentHash>, MergeError> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([*root]);
    while let Some(current) = queue.pop_front() {
        if !seen.insert(current) {
            continue;
        }
        let next = store.get_checkpoint(&current).map_err(map_store)?;
        let Some(checkpoint) = next else {
            if current == *root {
                return Err(MergeError::MissingObject(current));
            }
            return Err(MergeError::Corrupt);
        };
        for parent in checkpoint.parents {
            if !seen.contains(&parent) {
                queue.push_back(parent);
            }
        }
    }
    Ok(seen)
}

/// Strict lowest common ancestor of two checkpoints.
///
/// Intersects the full ancestor sets of `a` and `b`, then filters to minimal
/// elements: a common ancestor survives only when no other common ancestor
/// descends from it. Exactly one survivor is the merge base; none is
/// [`MergeError::NoCommonAncestor`]; several indicate criss-cross history
/// ([`MergeError::CrissCrossHistory`]). Equal tips return the tip itself when
/// its row exists, mirroring [`ContentStore::merge_base`][crate::content_store::ContentStore::merge_base].
///
/// Time O(|C| * (V + E)) for |C| common ancestors over V checkpoints and E
/// parent edges; space O(V).
pub fn strict_merge_base(
    store: &ContentStore,
    a: &ContentHash,
    b: &ContentHash,
) -> Result<ContentHash, MergeError> {
    if a == b {
        let present = store.get_checkpoint(a).map_err(map_store)?;
        if present.is_none() {
            return Err(MergeError::MissingObject(*a));
        }
        return Ok(*a);
    }
    let ancestors_a = ancestor_set(store, a)?;
    let ancestors_b = ancestor_set(store, b)?;
    let common: Vec<ContentHash> = ancestors_a.intersection(&ancestors_b).copied().collect();
    if common.is_empty() {
        return Err(MergeError::NoCommonAncestor);
    }
    if common.len() == 1 {
        return Ok(common[0]);
    }
    let mut per_node: HashMap<ContentHash, HashSet<ContentHash>> = HashMap::new();
    for candidate in &common {
        per_node.insert(*candidate, ancestor_set(store, candidate)?);
    }
    let mut is_strict_ancestor = HashSet::new();
    for other in &common {
        let ancestors_of_other = &per_node[other];
        for candidate in &common {
            if *candidate != *other && ancestors_of_other.contains(candidate) {
                is_strict_ancestor.insert(*candidate);
            }
        }
    }
    let mut minimal: Vec<ContentHash> = common
        .into_iter()
        .filter(|candidate| !is_strict_ancestor.contains(candidate))
        .collect();
    minimal.sort_unstable();
    match minimal.len() {
        0 => Err(MergeError::Corrupt),
        1 => Ok(minimal[0]),
        _ => Err(MergeError::CrissCrossHistory),
    }
}

/// Materialize one checkpoint's context tree from durable storage only.
///
/// Absent checkpoint/blob rows are [`MergeError::MissingObject`]; an absent
/// `tree_hash` link, a digest mismatch, or undecodable tree bytes are
/// [`MergeError::Corrupt`].
fn materialize_tree(store: &ContentStore, id: &ContentHash) -> Result<ContextTree, MergeError> {
    let checkpoint = store
        .get_checkpoint(id)
        .map_err(map_store)?
        .ok_or(MergeError::MissingObject(*id))?;
    let tree_hash = checkpoint.tree_hash.ok_or(MergeError::Corrupt)?;
    let bytes = store
        .get_blob(&tree_hash)
        .map_err(map_store)?
        .ok_or(MergeError::MissingObject(tree_hash))?;
    ContextTree::from_canonical_bytes(&bytes).map_err(|_| MergeError::Corrupt)
}

/// Merge two checkpoint tips into a two-parent commit on `target_branch`.
///
/// Runs the six ordered steps documented at the module level. Steps 1-5 are
/// read-only, so [`MergeError::NoCommonAncestor`],
/// [`MergeError::CrissCrossHistory`], [`MergeError::Conflicts`],
/// [`MergeError::MissingObject`], [`MergeError::StaleGeneration`],
/// [`MergeError::InvalidName`], and [`MergeError::NotFound`] all leave the
/// store untouched. Step 6 is one atomic transaction (tree blob, checkpoint
/// row, HEAD, branch ref, reflog row): an oversize `reason`/`actor` fails as
/// [`MergeError::InvalidName`] with HEAD and the branch fully rolled back.
pub fn merge_commit(
    store: &mut ContentStore,
    tasks: &TaskEngine,
    input: MergeInput,
) -> Result<Checkpoint, MergeError> {
    BranchName::parse(&input.target_branch).map_err(|err| match err {
        RefError::InvalidName | RefError::ProtectedHead | RefError::ReservedNamespace => {
            MergeError::InvalidName
        }
        RefError::AlreadyExists => MergeError::AlreadyExists,
        RefError::NotFound => MergeError::NotFound,
        RefError::MissingTarget(hash) => MergeError::MissingObject(hash),
        RefError::Corrupt => MergeError::Corrupt,
        RefError::Storage(_) => MergeError::Storage,
    })?;
    if let Some(fenced) = input.expected_task_generation {
        let task_id = TaskId::new(input.task_id.as_str()).map_err(|_| MergeError::InvalidName)?;
        let node = tasks.get_task(&task_id).map_err(map_task)?;
        if node.generation != fenced {
            return Err(MergeError::StaleGeneration {
                expected: node.generation,
                found: fenced,
            });
        }
    }
    let base_id = strict_merge_base(store, &input.ours, &input.theirs)?;
    let base_tree = materialize_tree(store, &base_id)?;
    let ours_tree = materialize_tree(store, &input.ours)?;
    let theirs_tree = materialize_tree(store, &input.theirs)?;
    let merged = ContextTree::merge_3way(&base_tree, &ours_tree, &theirs_tree);
    if !merged.conflicts.is_empty() {
        return Err(MergeError::Conflicts(merged.conflicts));
    }
    let merged_bytes = merged.merged.canonical_bytes();
    let draft = CheckpointDraft {
        parents: vec![input.ours, input.theirs],
        task_id: input.task_id,
        agent_id: input.agent_id,
        rationale: input.rationale,
        tree_hash: None,
        summary: input.summary,
        timestamp_ms: input.at_ms,
    };
    store
        .commit_checkpoint_with_branch(
            &merged_bytes,
            input.at_ms,
            draft,
            &input.target_branch,
            &input.reason,
            &input.actor,
            input.at_ms,
        )
        .map_err(map_ref_err)
}

/// Options for [`gc_preview`] and [`collect_garbage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcOptions {
    /// Caller-supplied clock reading selecting the reflog grace window.
    pub now_ms: u64,
    /// Reflog entries with `at_ms >= now_ms - reflog_grace_ms` pin their hashes.
    pub reflog_grace_ms: u64,
    /// Maximum checkpoint-plus-blob row deletes per destructive call (256 suggested).
    ///
    /// `0` deletes nothing and reports `truncated` whenever garbage remains
    /// (the caller makes no progress; supply at least `1` to reclaim).
    pub max_deletes_per_call: usize,
    /// When true, [`collect_garbage`] performs no writes (preview equivalent).
    pub dry_run: bool,
}

impl Default for GcOptions {
    fn default() -> Self {
        Self {
            now_ms: 0,
            reflog_grace_ms: 0,
            max_deletes_per_call: 256,
            dry_run: false,
        }
    }
}

/// Outcome of [`gc_preview`] or [`collect_garbage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcReport {
    /// Reachable checkpoints pinning live state (roots plus ancestors).
    pub reachable_checkpoints: usize,
    /// Checkpoints deleted by this call (preview: that would be deleted).
    pub deleted_checkpoints: usize,
    /// Blobs deleted by this call (preview: that would be deleted).
    pub deleted_blobs: usize,
    /// True when garbage remains beyond this call's bound; call again.
    pub truncated: bool,
}

/// Internal GC plan: full sorted unreachable sets plus the reachable count.
struct GcPlan {
    reachable_checkpoints: usize,
    unreachable_checkpoints: Vec<ContentHash>,
    unreferenced_blobs: Vec<ContentHash>,
}

/// Read every `(old_hash, new_hash)` pair in the reflog grace window.
///
/// Returns raw hex strings; the SQLite read lock is released before any
/// caller parses hashes or touches the checkpoint tables (single non-reentrant
/// connection lock). Bad hex in-window is [`MergeError::Corrupt`].
fn read_reflog_window(
    store: &ContentStore,
    threshold_i64: i64,
) -> Result<Vec<(Option<String>, String)>, MergeError> {
    let pairs: Vec<(Option<String>, String)> = {
        let guard = store.lock_conn().map_err(map_store)?;
        let mut stmt = guard
            .prepare("SELECT old_hash, new_hash FROM reflog WHERE at_ms >= ?1")
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map(params![threshold_i64], |row| {
                Ok((row.get::<_, Option<String>>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(map_sqlite)?;
        let mut out = Vec::new();
        for item in rows {
            out.push(item.map_err(map_sqlite)?);
        }
        out
    };
    Ok(pairs)
}

/// List every checkpoint hash, ordered ascending for deterministic batches.
fn list_all_checkpoint_hashes(store: &ContentStore) -> Result<Vec<ContentHash>, MergeError> {
    let hexes: Vec<String> = {
        let guard = store.lock_conn().map_err(map_store)?;
        let mut stmt = guard
            .prepare("SELECT hash FROM checkpoints ORDER BY hash ASC")
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(map_sqlite)?;
        let mut out = Vec::new();
        for item in rows {
            out.push(item.map_err(map_sqlite)?);
        }
        out
    };
    let mut hashes = Vec::with_capacity(hexes.len());
    for hex in hexes {
        hashes.push(ContentHash::from_hex(&hex).map_err(|_| MergeError::Corrupt)?);
    }
    Ok(hashes)
}

/// List every blob hash, ordered ascending for deterministic batches.
fn list_all_blob_hashes(store: &ContentStore) -> Result<Vec<ContentHash>, MergeError> {
    let hexes: Vec<String> = {
        let guard = store.lock_conn().map_err(map_store)?;
        let mut stmt = guard
            .prepare("SELECT hash FROM blobs ORDER BY hash ASC")
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(map_sqlite)?;
        let mut out = Vec::new();
        for item in rows {
            out.push(item.map_err(map_sqlite)?);
        }
        out
    };
    let mut hashes = Vec::with_capacity(hexes.len());
    for hex in hexes {
        hashes.push(ContentHash::from_hex(&hex).map_err(|_| MergeError::Corrupt)?);
    }
    Ok(hashes)
}

/// Maximum nested-tree traversal depth for GC blob retention.
///
/// `EntryKind::Tree` entries name another tree blob whose own entries must be
/// retained transitively. The bound caps chained `get_blob` reads per GC call;
/// sharing and cycles cost no extra reads via the visited set. Exceeding the
/// cap fails closed as [`MergeError::Corrupt`]. No production writer nests
/// today (`WheelKernel::put_slot` is Blob-only), so 8 is generous for any
/// foreseeable hierarchy use while keeping worst-case chained reads small.
const GC_MAX_NESTED_TREE_DEPTH: usize = 8;

/// Compute the full GC plan: reachable closure plus sorted garbage sets.
///
/// Roots are live ref targets (verified to exist; a dangling ref is
/// [`MergeError::Corrupt`]) plus in-window reflog hashes (stale pins for
/// already-collected rows are skipped; zero-hash tombstones are never roots).
/// A reachable checkpoint whose parent row is absent is [`MergeError::Corrupt`]
/// (dangling DAG link): the closure check after the lenient BFS enforces it.
/// Blob retention covers each surviving checkpoint's `tree_hash` blob plus
/// every entry hash named in that surviving tree (slot-payload blobs), and
/// transitively every entry of each nested tree named via `EntryKind::Tree`
/// up to [`GC_MAX_NESTED_TREE_DEPTH`]: each visited tree blob is materialized
/// via `ContextTree::from_canonical_bytes` and a missing tree blob row, a
/// digest mismatch, undecodable tree bytes, a nested tree blob that does not
/// decode, or a nesting chain deeper than the cap is [`MergeError::Corrupt`].
fn gc_compute(store: &ContentStore, options: &GcOptions) -> Result<GcPlan, MergeError> {
    let ref_pairs = store.list_refs().map_err(map_store)?;
    let mut ref_targets = HashSet::new();
    for (_, target) in &ref_pairs {
        ref_targets.insert(*target);
    }
    for target in &ref_targets {
        let exists = store.has_checkpoint(target).map_err(map_store)?;
        if !exists {
            return Err(MergeError::Corrupt);
        }
    }
    let threshold_u64 = options.now_ms.saturating_sub(options.reflog_grace_ms);
    let threshold_i64 = threshold_u64.min(i64::MAX as u64) as i64;
    let window = read_reflog_window(store, threshold_i64)?;
    let zero = tombstone_hash();
    let mut roots: HashSet<ContentHash> = ref_targets.clone();
    for (old_opt, new_hex) in &window {
        if let Some(old_hex) = old_opt {
            let hash = ContentHash::from_hex(old_hex).map_err(|_| MergeError::Corrupt)?;
            if hash != zero {
                roots.insert(hash);
            }
        }
        let hash = ContentHash::from_hex(new_hex).map_err(|_| MergeError::Corrupt)?;
        if hash != zero {
            roots.insert(hash);
        }
    }
    let mut reachable: HashSet<ContentHash> = HashSet::new();
    let mut surviving_trees: HashSet<ContentHash> = HashSet::new();
    let mut parents_of: HashMap<ContentHash, Vec<ContentHash>> = HashMap::new();
    let mut queue: VecDeque<ContentHash> = roots.into_iter().collect();
    while let Some(current) = queue.pop_front() {
        if reachable.contains(&current) {
            continue;
        }
        let next = store.get_checkpoint(&current).map_err(map_store)?;
        let Some(checkpoint) = next else {
            continue;
        };
        reachable.insert(current);
        if let Some(tree_hash) = checkpoint.tree_hash {
            surviving_trees.insert(tree_hash);
        }
        for parent in &checkpoint.parents {
            if !reachable.contains(parent) {
                queue.push_back(*parent);
            }
        }
        parents_of.insert(current, checkpoint.parents);
    }
    for (child, parents) in &parents_of {
        for parent in parents {
            if !reachable.contains(parent) {
                let _ = child;
                return Err(MergeError::Corrupt);
            }
        }
    }
    let all_checkpoints = list_all_checkpoint_hashes(store)?;
    let mut unreachable_checkpoints = Vec::new();
    for hash in all_checkpoints {
        if !reachable.contains(&hash) {
            unreachable_checkpoints.push(hash);
        }
    }
    let mut surviving_blobs: HashSet<ContentHash> = surviving_trees.clone();
    // Producibility evidence for `EntryKind::Tree` (choice (a): recurse):
    // `ContextTree::insert` accepts any `TreeEntry` with no kind gate
    // (context_compiler.rs), `TreeEntry::new` validates only the slot name,
    // `from_canonical_bytes` decodes kind byte 2 into `EntryKind::Tree`,
    // `merge_3way` propagates entries verbatim (kind preserved), and
    // `commit_checkpoint_atomic` stores arbitrary caller-supplied tree bytes
    // without canonical validation. No current production or test path builds
    // one (`WheelKernel::put_slot` and every test helper hardcode
    // `EntryKind::Blob`), but the reachable state is constructible through the
    // public API and durable bytes, so fail-closed rejection would turn a
    // constructible reachable state into a GC availability failure. GC
    // therefore traverses nested trees with an explicit depth cap; `Blob`
    // entries retain only their payload hash and are never traversed.
    // The visited set makes sharing and reference cycles (including
    // self-reference) terminate without extra reads.
    let mut visited_trees: HashSet<ContentHash> = HashSet::new();
    let mut tree_queue: VecDeque<(ContentHash, usize)> = surviving_trees
        .iter()
        .copied()
        .map(|hash| (hash, 0))
        .collect();
    while let Some((tree_hash, depth)) = tree_queue.pop_front() {
        if !visited_trees.insert(tree_hash) {
            continue;
        }
        let bytes = store
            .get_blob(&tree_hash)
            .map_err(map_store)?
            .ok_or(MergeError::Corrupt)?;
        let tree = ContextTree::from_canonical_bytes(&bytes).map_err(|_| MergeError::Corrupt)?;
        for entry in tree.entries() {
            surviving_blobs.insert(entry.hash);
            if entry.kind == EntryKind::Tree {
                if depth + 1 > GC_MAX_NESTED_TREE_DEPTH {
                    return Err(MergeError::Corrupt);
                }
                if !visited_trees.contains(&entry.hash) {
                    tree_queue.push_back((entry.hash, depth + 1));
                }
            }
        }
    }
    let all_blobs = list_all_blob_hashes(store)?;
    let mut unreferenced_blobs = Vec::new();
    for hash in all_blobs {
        if !surviving_blobs.contains(&hash) {
            unreferenced_blobs.push(hash);
        }
    }
    Ok(GcPlan {
        reachable_checkpoints: reachable.len(),
        unreachable_checkpoints,
        unreferenced_blobs,
    })
}

/// Bound a plan to one batch, checkpoints first in ascending hash order.
fn bound_report(plan: &GcPlan, max_deletes: usize) -> GcReport {
    let total = plan
        .unreachable_checkpoints
        .len()
        .saturating_add(plan.unreferenced_blobs.len());
    let budget = max_deletes.min(total);
    let deleted_checkpoints = plan.unreachable_checkpoints.len().min(budget);
    let deleted_blobs = budget.saturating_sub(deleted_checkpoints);
    GcReport {
        reachable_checkpoints: plan.reachable_checkpoints,
        deleted_checkpoints,
        deleted_blobs,
        truncated: total > budget,
    }
}

/// Read-only GC preview: byte-matches the next destructive batch.
///
/// Computes the same sorted garbage sets and applies the same
/// [`GcOptions::max_deletes_per_call`] bound as [`collect_garbage`], but
/// performs no writes. Reflog rows are only read, never pruned.
pub fn gc_preview(store: &ContentStore, options: &GcOptions) -> Result<GcReport, MergeError> {
    let plan = gc_compute(store, options)?;
    Ok(bound_report(&plan, options.max_deletes_per_call))
}

/// Bounded reachability garbage collection: prune unreachable checkpoints and
/// blobs referenced by no surviving checkpoint tree.
///
/// Retention covers each surviving checkpoint's `tree_hash` blob plus every
/// entry hash named in the surviving trees (slot-payload blobs), transitively
/// including nested-tree entries up to the GC depth cap; standalone
/// blobs referenced by no surviving tree (action-spillover payloads, orphan
/// `put_blob` rows) are pruned. Deletes at most [`GcOptions::max_deletes_per_call`] rows (checkpoints
/// first, then blobs, ascending hash order) in a single SQLite transaction.
/// When [`GcReport::truncated`] is true, garbage remains and the caller
/// resumes by calling again. With [`GcOptions::dry_run`], no writes occur and
/// the result equals [`gc_preview`]. Reflog rows are never deleted; no reflog
/// rows are written; zero-hash tombstones stay payload-free.
pub fn collect_garbage(
    store: &mut ContentStore,
    options: &GcOptions,
) -> Result<GcReport, MergeError> {
    if options.dry_run {
        return gc_preview(store, options);
    }
    let plan = gc_compute(store, options)?;
    let report = bound_report(&plan, options.max_deletes_per_call);
    let checkpoint_deletes = plan
        .unreachable_checkpoints
        .iter()
        .take(report.deleted_checkpoints);
    let blob_deletes = plan.unreferenced_blobs.iter().take(report.deleted_blobs);
    if report.deleted_checkpoints == 0 && report.deleted_blobs == 0 {
        return Ok(report);
    }
    let mut guard = store.lock_conn().map_err(map_store)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    for hash in checkpoint_deletes {
        tx.execute(
            "DELETE FROM checkpoints WHERE hash = ?1",
            params![hash.to_hex()],
        )
        .map_err(map_sqlite)?;
    }
    for hash in blob_deletes {
        tx.execute("DELETE FROM blobs WHERE hash = ?1", params![hash.to_hex()])
            .map_err(map_sqlite)?;
    }
    tx.commit().map_err(map_sqlite)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_compiler::{EntryKind, TreeEntry};
    use crate::session_refs::MAX_REFLOG_REASON_BYTES;
    use crate::task_dag::{TaskDraft, TaskId};

    fn test_rationale() -> crate::content_store::Rationale {
        crate::content_store::Rationale::new("merge why", "merge what")
    }

    fn slot_tree(pairs: &[(&str, &str)]) -> ContextTree {
        let mut tree = ContextTree::new();
        for (name, content) in pairs {
            let hash = ContentHash::compute(content.as_bytes());
            tree.insert(
                TreeEntry::new(*name, hash, EntryKind::Blob, content.len())
                    .expect("valid test slot"),
            );
        }
        tree
    }

    fn commit_tree(
        store: &mut ContentStore,
        parents: Vec<ContentHash>,
        tree: &ContextTree,
        summary: &str,
        at_ms: u64,
    ) -> Checkpoint {
        let bytes = tree.canonical_bytes();
        let draft = CheckpointDraft {
            parents,
            task_id: "AI-0181".to_string(),
            agent_id: "ctx-0181-test".to_string(),
            rationale: test_rationale(),
            tree_hash: None,
            summary: summary.to_string(),
            timestamp_ms: at_ms,
        };
        store
            .commit_checkpoint_atomic(&bytes, at_ms, draft, &[], at_ms)
            .expect("setup commit")
    }

    fn merge_input(ours: ContentHash, theirs: ContentHash, target_branch: &str) -> MergeInput {
        MergeInput {
            ours,
            theirs,
            target_branch: target_branch.to_string(),
            task_id: "AI-0181".to_string(),
            agent_id: "ctx-0181-test".to_string(),
            rationale: test_rationale(),
            summary: "merge summary".to_string(),
            expected_task_generation: None,
            actor: "tester".to_string(),
            reason: "merge test".to_string(),
            at_ms: 9000,
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StoreSnapshot {
        checkpoints: Vec<String>,
        blobs: Vec<String>,
        refs: Vec<String>,
        reflog: Vec<String>,
    }

    fn snapshot(store: &ContentStore) -> StoreSnapshot {
        let guard = store.lock_conn().expect("lock");
        let mut checkpoints = Vec::new();
        {
            let mut stmt = guard
                .prepare(
                    "SELECT hash, parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms
                     FROM checkpoints ORDER BY hash ASC",
                )
                .expect("prepare checkpoints");
            let rows = stmt
                .query_map([], |row| {
                    Ok(format!(
                        "{}|{}|{}|{}|{}|{}|{}|{}",
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, Option<String>>(5)?.as_deref().unwrap_or("-"),
                        row.get::<_, String>(6)?,
                        row.get::<_, i64>(7)?
                    ))
                })
                .expect("query checkpoints");
            for item in rows {
                checkpoints.push(item.expect("row"));
            }
        }
        let mut blobs = Vec::new();
        {
            let mut stmt = guard
                .prepare("SELECT hash, size, created_at_ms FROM blobs ORDER BY hash ASC")
                .expect("prepare blobs");
            let rows = stmt
                .query_map([], |row| {
                    Ok(format!(
                        "{}|{}|{}",
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?
                    ))
                })
                .expect("query blobs");
            for item in rows {
                blobs.push(item.expect("row"));
            }
        }
        let mut refs = Vec::new();
        {
            let mut stmt = guard
                .prepare("SELECT name, target_hash, updated_at_ms FROM refs ORDER BY name ASC")
                .expect("prepare refs");
            let rows = stmt
                .query_map([], |row| {
                    Ok(format!(
                        "{}|{}|{}",
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?
                    ))
                })
                .expect("query refs");
            for item in rows {
                refs.push(item.expect("row"));
            }
        }
        let mut reflog = Vec::new();
        {
            let mut stmt = guard
                .prepare(
                    "SELECT seq, ref_name, old_hash, new_hash, reason, actor, at_ms
                     FROM reflog ORDER BY seq ASC",
                )
                .expect("prepare reflog");
            let rows = stmt
                .query_map([], |row| {
                    Ok(format!(
                        "{}|{}|{}|{}|{}|{}|{}",
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?.as_deref().unwrap_or("-"),
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, i64>(6)?
                    ))
                })
                .expect("query reflog");
            for item in rows {
                reflog.push(item.expect("row"));
            }
        }
        StoreSnapshot {
            checkpoints,
            blobs,
            refs,
            reflog,
        }
    }

    fn setup_clean_pair(store: &mut ContentStore) -> (Checkpoint, Checkpoint) {
        let base = commit_tree(
            &mut *store,
            Vec::new(),
            &slot_tree(&[("shared", "v0")]),
            "base",
            1000,
        );
        let ours = commit_tree(
            &mut *store,
            vec![base.id],
            &slot_tree(&[("shared", "v0"), ("ours-only", "o")]),
            "ours",
            2000,
        );
        let theirs = commit_tree(
            &mut *store,
            vec![base.id],
            &slot_tree(&[("shared", "v0"), ("theirs-only", "t")]),
            "theirs",
            3000,
        );
        (ours, theirs)
    }

    fn setup_task_engine(task_id: &str, now_ms: u64) -> (TaskEngine, u64) {
        let mut engine = TaskEngine::open_in_memory().expect("task engine");
        let node = engine
            .create_task(
                TaskDraft {
                    id: TaskId::new(task_id).expect("task id"),
                    title: "merge task".to_string(),
                    description: String::new(),
                    priority: 0,
                    dependencies: Vec::new(),
                },
                now_ms,
            )
            .expect("create task");
        (engine, node.generation)
    }

    #[test]
    fn clean_merge_advances_head_and_branch_with_reflog() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let (ours, theirs) = setup_clean_pair(&mut store);
        let (engine, generation) = setup_task_engine("AI-0181", 1000);
        let mut input = merge_input(ours.id, theirs.id, "heads/main");
        input.expected_task_generation = Some(generation);
        let head_before = store.get_ref("HEAD").expect("head read");
        let merged = merge_commit(&mut store, &engine, input).expect("clean merge");
        assert_eq!(merged.parents, vec![ours.id, theirs.id]);
        assert_eq!(store.get_ref("HEAD").expect("head"), Some(merged.id));
        assert_eq!(
            store.get_branch("heads/main").expect("branch"),
            Some(merged.id)
        );
        assert_ne!(head_before, Some(merged.id));
        let history = store.read_reflog("heads/main", 10).expect("reflog");
        assert_eq!(history.len(), 1, "creation writes exactly one reflog row");
        assert_eq!(history[0].old_hash, None);
        assert_eq!(history[0].new_hash, merged.id);
        let tree = materialize_tree(&store, &merged.id).expect("merged tree");
        assert!(tree.get("shared").is_some());
        assert!(tree.get("ours-only").is_some());
        assert!(tree.get("theirs-only").is_some());
    }

    #[test]
    fn conflict_same_slot_both_edit_leaves_store_unchanged() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("s", "v0")]),
            "base",
            1000,
        );
        let ours = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("s", "v1")]),
            "ours",
            2000,
        );
        let theirs = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("s", "v2")]),
            "theirs",
            3000,
        );
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let err = merge_commit(
            &mut store,
            &engine,
            merge_input(ours.id, theirs.id, "heads/main"),
        )
        .expect_err("both-edit must conflict");
        match err {
            MergeError::Conflicts(conflicts) => {
                assert_eq!(conflicts.len(), 1);
                assert_eq!(conflicts[0].slot, "s");
            }
            other => panic!("expected Conflicts, got {other:?}"),
        }
        assert_eq!(snapshot(&store), before, "conflict must write nothing");
    }

    #[test]
    fn conflict_add_add_differ_leaves_store_unchanged() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(&mut store, Vec::new(), &slot_tree(&[]), "base", 1000);
        let ours = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("new", "a")]),
            "ours",
            2000,
        );
        let theirs = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("new", "b")]),
            "theirs",
            3000,
        );
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let err = merge_commit(
            &mut store,
            &engine,
            merge_input(ours.id, theirs.id, "heads/main"),
        )
        .expect_err("add/add-differ must conflict");
        match err {
            MergeError::Conflicts(conflicts) => {
                assert_eq!(conflicts.len(), 1);
                assert_eq!(conflicts[0].slot, "new");
            }
            other => panic!("expected Conflicts, got {other:?}"),
        }
        assert_eq!(snapshot(&store), before, "conflict must write nothing");
    }

    #[test]
    fn conflict_add_delete_leaves_store_unchanged() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("s", "v0")]),
            "base",
            1000,
        );
        let ours = commit_tree(&mut store, vec![base.id], &slot_tree(&[]), "delete", 2000);
        let theirs = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("s", "v1")]),
            "modify",
            3000,
        );
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let err = merge_commit(
            &mut store,
            &engine,
            merge_input(ours.id, theirs.id, "heads/main"),
        )
        .expect_err("delete/modify must conflict");
        match err {
            MergeError::Conflicts(conflicts) => {
                assert_eq!(conflicts.len(), 1);
                assert_eq!(conflicts[0].slot, "s");
            }
            other => panic!("expected Conflicts, got {other:?}"),
        }
        assert_eq!(snapshot(&store), before, "conflict must write nothing");
    }

    #[test]
    fn disjoint_histories_have_no_common_ancestor() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let left = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("slot", "left")]),
            "left",
            1000,
        );
        let right = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("slot", "right")]),
            "right",
            2000,
        );
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let err = merge_commit(
            &mut store,
            &engine,
            merge_input(left.id, right.id, "heads/main"),
        )
        .expect_err("disjoint histories must fail");
        assert_eq!(err, MergeError::NoCommonAncestor);
        assert_eq!(snapshot(&store), before, "failure must write nothing");
    }

    #[test]
    fn criss_cross_history_is_rejected() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("shared", "v0")]),
            "base",
            1000,
        );
        let x = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("shared", "v0"), ("x", "1")]),
            "x",
            2000,
        );
        let y = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("shared", "v0"), ("y", "1")]),
            "y",
            2500,
        );
        let m1 = commit_tree(
            &mut store,
            vec![x.id, y.id],
            &slot_tree(&[("shared", "v0"), ("x", "1"), ("y", "1"), ("m", "1")]),
            "m1",
            3000,
        );
        let m2 = commit_tree(
            &mut store,
            vec![y.id, x.id],
            &slot_tree(&[("shared", "v0"), ("x", "1"), ("y", "1"), ("m", "2")]),
            "m2",
            3500,
        );
        assert_ne!(m1.id, m2.id, "merges must differ to form criss-cross");
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let err = merge_commit(&mut store, &engine, merge_input(m1.id, m2.id, "heads/main"))
            .expect_err("criss-cross must fail");
        assert_eq!(err, MergeError::CrissCrossHistory);
        assert_eq!(snapshot(&store), before, "failure must write nothing");
    }

    #[test]
    fn oversize_reason_rolls_back_head_and_branch() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let (ours, theirs) = setup_clean_pair(&mut store);
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let head_before = store.get_ref("HEAD").expect("head");
        let mut input = merge_input(ours.id, theirs.id, "heads/main");
        input.reason = "r".repeat(MAX_REFLOG_REASON_BYTES + 1);
        let err = merge_commit(&mut store, &engine, input).expect_err("oversize must fail");
        assert_eq!(err, MergeError::InvalidName);
        assert_eq!(
            store.get_ref("HEAD").expect("head after"),
            head_before,
            "HEAD must roll back"
        );
        assert_eq!(
            store.get_branch("heads/main").expect("branch after"),
            None,
            "branch must not exist after rollback"
        );
        assert_eq!(snapshot(&store), before, "rollback must be total");
    }

    #[test]
    fn stale_generation_is_rejected_before_any_write() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let (ours, theirs) = setup_clean_pair(&mut store);
        let (engine, generation) = setup_task_engine("AI-0181", 1000);
        assert_eq!(generation, 0);
        let before = snapshot(&store);
        let mut input = merge_input(ours.id, theirs.id, "heads/main");
        input.expected_task_generation = Some(generation + 99);
        let err = merge_commit(&mut store, &engine, input).expect_err("stale fence must fail");
        assert_eq!(
            err,
            MergeError::StaleGeneration {
                expected: generation,
                found: generation + 99
            }
        );
        assert_eq!(snapshot(&store), before, "stale fence must write nothing");
    }

    #[test]
    fn missing_tip_reports_missing_object_without_writes() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let present = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("slot", "v")]),
            "present",
            1000,
        );
        let absent = ContentHash::compute(b"no-such-checkpoint");
        let engine = TaskEngine::open_in_memory().expect("tasks");
        let before = snapshot(&store);
        let err = merge_commit(
            &mut store,
            &engine,
            merge_input(present.id, absent, "heads/main"),
        )
        .expect_err("absent tip must fail");
        assert_eq!(err, MergeError::MissingObject(absent));
        assert_eq!(snapshot(&store), before, "failure must write nothing");
    }

    #[test]
    fn gc_preview_matches_bounded_destructive_batches_and_keeps_roots() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("slot", "base")]),
            "base",
            1000,
        );
        let live1 = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "base"), ("live", "1")]),
            "live1",
            2000,
        );
        let live2 = commit_tree(
            &mut store,
            vec![live1.id],
            &slot_tree(&[("slot", "base"), ("live", "2")]),
            "live2",
            3000,
        );
        store
            .update_refs_atomic(&[("HEAD", &live2.id), ("heads/main", &live2.id)], 3000)
            .expect("live refs");
        let fork1 = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "base"), ("fork", "1")]),
            "fork1",
            2000,
        );
        let fork2 = commit_tree(
            &mut store,
            vec![fork1.id],
            &slot_tree(&[("slot", "base"), ("fork", "2")]),
            "fork2",
            2500,
        );
        let pinned = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "base"), ("pinned", "1")]),
            "pinned",
            2000,
        );
        store
            .create_branch("heads/tmp", &pinned.id, "pin", "tester", 9000)
            .expect("pin branch");
        store
            .delete_branch("heads/tmp", "unpin", "tester", 9100)
            .expect("unpin branch");
        let full = GcOptions {
            now_ms: 10_000,
            reflog_grace_ms: 2000,
            max_deletes_per_call: 1000,
            dry_run: false,
        };
        let preview = gc_preview(&store, &full).expect("preview");
        assert_eq!(preview.reachable_checkpoints, 4);
        assert_eq!(preview.deleted_checkpoints, 2);
        assert_eq!(preview.deleted_blobs, 2);
        assert!(!preview.truncated);
        let tiny = GcOptions {
            max_deletes_per_call: 1,
            ..full.clone()
        };
        let first_preview = gc_preview(&store, &tiny).expect("bounded preview");
        assert!(first_preview.truncated);
        assert_eq!(
            first_preview.deleted_checkpoints + first_preview.deleted_blobs,
            1
        );
        let first = collect_garbage(&mut store, &tiny).expect("first batch");
        assert_eq!(first, first_preview, "preview must byte-match the batch");
        let mut total_checkpoints = first.deleted_checkpoints;
        let mut total_blobs = first.deleted_blobs;
        let mut calls = 1;
        loop {
            let report = collect_garbage(&mut store, &tiny).expect("resume batch");
            total_checkpoints += report.deleted_checkpoints;
            total_blobs += report.deleted_blobs;
            calls += 1;
            if !report.truncated {
                break;
            }
            assert!(calls < 10, "bounded resume must converge");
        }
        assert_eq!(total_checkpoints, 2);
        assert_eq!(total_blobs, 2);
        for kept in [base.id, live1.id, live2.id, pinned.id] {
            assert!(
                store.has_checkpoint(&kept).expect("kept check"),
                "reachable checkpoint must survive"
            );
        }
        for gone in [fork1.id, fork2.id] {
            assert!(
                !store.has_checkpoint(&gone).expect("gone check"),
                "abandoned fork must be pruned"
            );
        }
        assert_eq!(
            store.get_ref("heads/main").expect("main"),
            Some(live2.id),
            "live ref target must survive"
        );
    }

    #[test]
    fn gc_retains_live_slot_payload_blobs_and_prunes_orphans() {
        let mut store = ContentStore::open_in_memory().expect("store");
        // REAL slot-payload blob via `put_blob` (the `WheelKernel::put_slot`
        // path): the tree names the payload hash, but no checkpoint column
        // links the payload blob itself — only the tree blob is referenced.
        let payload: &[u8] = b"live slot payload";
        let payload_hash = store.put_blob(payload, 1000).expect("payload blob");
        let mut live_tree = ContextTree::new();
        live_tree.insert(
            TreeEntry::new("doc", payload_hash, EntryKind::Blob, payload.len())
                .expect("valid test slot"),
        );
        let live = commit_tree(&mut store, Vec::new(), &live_tree, "live", 1000);
        store
            .update_refs_atomic(&[("HEAD", &live.id), ("heads/main", &live.id)], 1000)
            .expect("live refs");
        // Abandoned fork with its own REAL payload blob: unreachable, so the
        // checkpoint, its tree blob, and its payload blob are all garbage.
        let fork_payload: &[u8] = b"abandoned fork payload";
        let fork_payload_hash = store
            .put_blob(fork_payload, 1500)
            .expect("fork payload blob");
        let mut fork_tree = ContextTree::new();
        fork_tree.insert(
            TreeEntry::new(
                "doc",
                fork_payload_hash,
                EntryKind::Blob,
                fork_payload.len(),
            )
            .expect("valid test slot"),
        );
        let fork = commit_tree(&mut store, Vec::new(), &fork_tree, "fork", 1500);
        // Orphan spillover-style blob: stored via `put_blob` but committed
        // into no tree (the `BlobSink::store_blob` / `BlobPointer` path).
        let orphan: &[u8] = b"spilled action output nobody committed";
        let orphan_hash = store.put_blob(orphan, 1600).expect("orphan blob");
        assert_eq!(
            store
                .get_blob(&payload_hash)
                .expect("payload pre-gc")
                .as_deref(),
            Some(payload),
            "setup must store the live payload"
        );
        let options = GcOptions {
            now_ms: 100_000,
            reflog_grace_ms: 1000,
            max_deletes_per_call: 1000,
            dry_run: false,
        };
        let report = collect_garbage(&mut store, &options).expect("gc");
        assert_eq!(report.reachable_checkpoints, 1);
        assert_eq!(report.deleted_checkpoints, 1);
        assert_eq!(report.deleted_blobs, 3);
        assert!(!report.truncated);
        // Live state survives with its payload dereferenceable: the defect
        // deleted this blob, leaving slot reads silently `None`.
        assert!(
            store.has_checkpoint(&live.id).expect("live check"),
            "live checkpoint survives"
        );
        assert_eq!(
            store
                .get_blob(&payload_hash)
                .expect("payload post-gc")
                .as_deref(),
            Some(payload),
            "reachable slot-payload blob must survive GC"
        );
        let tree = materialize_tree(&store, &live.id).expect("live tree materializes");
        assert_eq!(
            tree.get("doc").map(|entry| entry.hash),
            Some(payload_hash),
            "surviving tree still names the payload"
        );
        // True garbage is pruned: the fork checkpoint with its tree and
        // payload blobs, plus the orphan spillover blob.
        assert!(
            !store.has_checkpoint(&fork.id).expect("fork check"),
            "abandoned fork is pruned"
        );
        assert!(
            store
                .get_blob(&fork_payload_hash)
                .expect("fork payload")
                .is_none(),
            "unreachable slot payload is pruned"
        );
        assert!(
            store.get_blob(&orphan_hash).expect("orphan").is_none(),
            "orphan spillover blob is pruned"
        );
        assert_eq!(
            store.get_ref("HEAD").expect("head"),
            Some(live.id),
            "HEAD still pins the live checkpoint"
        );
    }

    #[test]
    fn gc_never_prunes_reflog_and_keeps_tombstone_payload_absent() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("slot", "base")]),
            "base",
            1000,
        );
        let live = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "live")]),
            "live",
            2000,
        );
        store
            .update_refs_atomic(&[("HEAD", &live.id), ("heads/main", &live.id)], 2000)
            .expect("live refs");
        let gone_tip = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "gone")]),
            "gone",
            1500,
        );
        store
            .create_branch("heads/gone", &gone_tip.id, "create", "tester", 1600)
            .expect("gone branch");
        store
            .delete_branch("heads/gone", "remove", "tester", 1700)
            .expect("delete branch");
        let zero = tombstone_hash();
        assert!(
            store.get_blob(&zero).expect("zero blob").is_none(),
            "tombstone has no blob payload"
        );
        assert!(
            store
                .get_checkpoint(&zero)
                .expect("zero checkpoint")
                .is_none(),
            "tombstone has no checkpoint payload"
        );
        let reflog_before = snapshot(&store).reflog;
        assert!(!reflog_before.is_empty(), "setup must leave reflog rows");
        let options = GcOptions {
            now_ms: 100_000,
            reflog_grace_ms: 1000,
            max_deletes_per_call: 1000,
            dry_run: false,
        };
        let report = collect_garbage(&mut store, &options).expect("gc");
        assert!(
            !store.has_checkpoint(&gone_tip.id).expect("gone check"),
            "unpinned tombstoned tip is pruned"
        );
        assert!(
            store.has_checkpoint(&live.id).expect("live check"),
            "live tip survives"
        );
        assert_eq!(
            snapshot(&store).reflog,
            reflog_before,
            "reflog rows are never pruned"
        );
        assert!(
            store.get_blob(&zero).expect("zero blob after").is_none(),
            "tombstone payload stays absent"
        );
        assert!(
            store
                .get_checkpoint(&zero)
                .expect("zero checkpoint after")
                .is_none(),
            "tombstone payload stays absent"
        );
        let _ = report;
    }

    #[test]
    fn gc_retains_nested_tree_payloads_transitively() {
        let mut store = ContentStore::open_in_memory().expect("store");
        // Nested-tree fixture: no production writer nests today
        // (`WheelKernel::put_slot` is Blob-only), but `ContextTree::insert`,
        // `merge_3way`, `from_canonical_bytes` (kind byte 2), and the commit
        // path all preserve `EntryKind::Tree`, so the state is constructible
        // and GC must traverse it. The shallow defect retained the nested
        // tree blob itself yet pruned this payload.
        let payload: &[u8] = b"nested payload";
        let payload_hash = store.put_blob(payload, 1000).expect("payload blob");
        let mut inner = ContextTree::new();
        inner.insert(
            TreeEntry::new("inner-doc", payload_hash, EntryKind::Blob, payload.len())
                .expect("valid inner slot"),
        );
        let inner_bytes = inner.canonical_bytes();
        let inner_hash = store.put_blob(&inner_bytes, 1000).expect("inner tree blob");
        let mut outer = ContextTree::new();
        outer.insert(
            TreeEntry::new("nested", inner_hash, EntryKind::Tree, inner_bytes.len())
                .expect("valid nested slot"),
        );
        let live = commit_tree(&mut store, Vec::new(), &outer, "live", 1000);
        store
            .update_refs_atomic(&[("HEAD", &live.id), ("heads/main", &live.id)], 1000)
            .expect("live refs");
        let options = GcOptions {
            now_ms: 100_000,
            reflog_grace_ms: 1000,
            max_deletes_per_call: 1000,
            dry_run: false,
        };
        let report = collect_garbage(&mut store, &options).expect("gc");
        assert_eq!(report.reachable_checkpoints, 1);
        assert_eq!(report.deleted_checkpoints, 0);
        assert_eq!(report.deleted_blobs, 0);
        assert!(!report.truncated);
        assert_eq!(
            store
                .get_blob(&payload_hash)
                .expect("payload post-gc")
                .as_deref(),
            Some(payload),
            "transitively reachable payload through a nested tree must survive GC"
        );
        assert_eq!(
            store
                .get_blob(&inner_hash)
                .expect("inner tree post-gc")
                .as_deref(),
            Some(inner_bytes.as_slice()),
            "nested tree blob itself must survive GC"
        );
    }

    #[test]
    fn gc_dry_run_writes_nothing() {
        let mut store = ContentStore::open_in_memory().expect("store");
        let base = commit_tree(
            &mut store,
            Vec::new(),
            &slot_tree(&[("slot", "base")]),
            "base",
            1000,
        );
        let live = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "live")]),
            "live",
            2000,
        );
        store
            .update_refs_atomic(&[("HEAD", &live.id)], 2000)
            .expect("head");
        let fork = commit_tree(
            &mut store,
            vec![base.id],
            &slot_tree(&[("slot", "fork")]),
            "fork",
            1500,
        );
        let options = GcOptions {
            now_ms: 50_000,
            reflog_grace_ms: 1000,
            max_deletes_per_call: 1000,
            dry_run: true,
        };
        let before = snapshot(&store);
        let preview = gc_preview(&store, &options).expect("preview");
        let dry = collect_garbage(&mut store, &options).expect("dry run");
        assert_eq!(dry, preview, "dry run must equal preview");
        assert_eq!(snapshot(&store), before, "dry run must write nothing");
        assert!(
            store
                .has_checkpoint(&fork.id)
                .expect("fork survives dry run"),
            "dry run deletes nothing"
        );
    }
}

//! Typed branch verbs with automatic reflog (AI-0180, refs-plane v1).
//!
//! This module is the refs plane: a narrow, typed namespace over the raw
//! `refs` table in [`crate::content_store::ContentStore`]. Every branch lives
//! under `heads/`; HEAD, tags, and bare names are refused with typed errors.
//! Every ref move (create, update, delete, rename) appends exactly one
//! reflog row **in the same SQLite transaction** as the move itself, so the
//! move and its history entry commit or roll back together.
//!
//! ## Namespace rules
//!
//! - [`BranchName::parse`] requires the `heads/` prefix. Length and charset
//!   follow the store rule ([`MAX_REF_NAME_BYTES`][crate::content_store::MAX_REF_NAME_BYTES]
//!   bytes, `[A-Za-z0-9/._-]`): violations are [`RefError::InvalidName`].
//! - The exact name `HEAD` is [`RefError::ProtectedHead`]; it is owned by the
//!   Wheel commit path, never by branch verbs.
//! - `tags/...` (and any other non-`heads/` namespaced name) is
//!   [`RefError::ReservedNamespace`]; a bare name without a namespace is
//!   [`RefError::InvalidName`].
//!
//! ## Reflog
//!
//! The `reflog` table records `(ref_name, old_hash, new_hash, reason, actor,
//! at_ms)` rows in append order (`seq INTEGER PRIMARY KEY AUTOINCREMENT`).
//! Branch creation stores `old_hash = NULL`; branch deletion stores a
//! tombstone row (`old_hash` = deleted tip, `new_hash` = all-zero hash,
//! mirroring the git convention) while history stays readable under the old
//! name. [`read_reflog`] returns rows newest-first and caps
//! every read at [`MAX_REFLOG_READ_LIMIT`] entries.
//!
//! Caller-supplied `reason`/`actor` strings are bounded
//! ([`MAX_REFLOG_REASON_BYTES`]/[`MAX_REFLOG_ACTOR_BYTES`]). The bounds are
//! enforced at append time, inside the same transaction as the ref move, so
//! a bound violation rolls the move back instead of leaving a moved ref with
//! no history row.
//!
//! ## Raw paths
//!
//! The pre-existing raw [`ContentStore`] ref paths (`update_ref`,
//! `update_refs_atomic`, `delete_ref`, `commit_checkpoint_atomic`) bypass the
//! reflog; they are kept for compatibility. New code must use the typed
//! verbs here. Stored-data validation failures (bad hex in `refs`/`reflog`,
//! unparseable stored branch names) fail closed as [`RefError::Corrupt`];
//! caller-input failures use the namespace/existence variants.
//!
//! Errors never echo caller-supplied text: every [`RefError`] display string
//! is a static literal except [`RefError::MissingTarget`], which carries the
//! fixed-size content hash (same reuse as
//! [`ContentStoreError::MissingTarget`][crate::content_store::ContentStoreError::MissingTarget]).

use std::fmt;

use rusqlite::{OptionalExtension, Transaction, params};

use crate::content_store::{
    Checkpoint, CheckpointDraft, ContentHash, ContentStore, ContentStoreError, MAX_AGENT_ID_BYTES,
    MAX_BLOB_BYTES, MAX_CHECKPOINT_PARENTS, MAX_SUMMARY_BYTES, MAX_TASK_ID_BYTES,
};
use crate::facade::FacadeError;

/// Maximum byte length for a reflog reason string (512 bytes).
pub const MAX_REFLOG_REASON_BYTES: usize = 512;

/// Maximum byte length for a reflog actor string (128 bytes).
pub const MAX_REFLOG_ACTOR_BYTES: usize = 128;

/// Maximum reflog entries returned by a single [`read_reflog`] call (1024).
pub const MAX_REFLOG_READ_LIMIT: usize = 1024;

/// Required branch namespace prefix.
const BRANCH_NAMESPACE: &str = "heads/";

/// Reserved tag namespace prefix (refused by branch verbs).
const TAG_NAMESPACE: &str = "tags/";

/// The protected HEAD ref name (never a branch).
const HEAD_NAME: &str = "HEAD";

/// Static storage refusal for non-fast-forward updates.
const NON_FAST_FORWARD_MSG: &str = "refusing non-fast-forward branch update";

/// Static message when another writer holds the single-writer lock.
const WRITER_BUSY_MSG: &str = "writer busy: another writer holds the single-writer lock";

/// A validated `heads/` branch name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BranchName(String);

impl BranchName {
    /// Parse and validate a branch name.
    ///
    /// Requires the `heads/` prefix with a non-empty leaf, the store length
    /// bound, and the store charset. The exact name `HEAD` is refused as
    /// [`RefError::ProtectedHead`], `tags/...` (and any other namespaced
    /// non-branch name) as [`RefError::ReservedNamespace`], and bare or
    /// malformed names as [`RefError::InvalidName`].
    pub fn parse(raw: &str) -> Result<Self, RefError> {
        if raw == HEAD_NAME {
            return Err(RefError::ProtectedHead);
        }
        ContentStore::validate_ref_name(raw).map_err(|_| RefError::InvalidName)?;
        if let Some(leaf) = raw.strip_prefix(BRANCH_NAMESPACE) {
            if leaf.is_empty() {
                return Err(RefError::InvalidName);
            }
            return Ok(Self(raw.to_owned()));
        }
        if raw.starts_with(TAG_NAMESPACE) || raw.contains('/') {
            return Err(RefError::ReservedNamespace);
        }
        Err(RefError::InvalidName)
    }

    /// Borrow the validated branch name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BranchName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Typed failures for branch verbs and reflog reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefError {
    /// Caller-supplied string violates its bound: malformed/oversize branch
    /// name, or oversize reflog reason/actor.
    InvalidName,
    /// Create or rename-target name already exists (no force).
    AlreadyExists,
    /// Named branch does not exist.
    NotFound,
    /// The HEAD ref was targeted by a branch verb.
    ProtectedHead,
    /// Correctly formed but non-branch namespace (`tags/...`, ...).
    ReservedNamespace,
    /// Ref target checkpoint does not exist (reuses the store concept).
    MissingTarget(ContentHash),
    /// Stored data failed integrity validation (bad hex, bad shape, bad
    /// stored name) or the schema is partial. Fail closed, no partial state.
    Corrupt,
    /// Underlying storage failure or store-level refusal (static text only).
    Storage(String),
}

impl fmt::Display for RefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName => write!(f, "invalid branch name, reason, or actor string"),
            Self::AlreadyExists => write!(f, "branch already exists"),
            Self::NotFound => write!(f, "branch not found"),
            Self::ProtectedHead => write!(f, "HEAD is protected from branch verbs"),
            Self::ReservedNamespace => write!(f, "ref namespace is reserved (not heads/)"),
            Self::MissingTarget(hash) => {
                write!(f, "target checkpoint {hash} for ref does not exist")
            }
            Self::Corrupt => write!(f, "branch store corrupt or incompatible"),
            Self::Storage(detail) => write!(f, "branch storage error: {detail}"),
        }
    }
}

impl std::error::Error for RefError {}

impl From<RefError> for FacadeError {
    fn from(err: RefError) -> Self {
        Self::Store(err.to_string())
    }
}

/// One reflog history entry, newest-first from [`read_reflog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflogEntry {
    /// Append-order sequence number (SQLite AUTOINCREMENT).
    pub seq: i64,
    /// Ref this row records (any well-formed ref name; survives deletion).
    pub ref_name: String,
    /// Previous tip (`None` on creation).
    pub old_hash: Option<ContentHash>,
    /// New tip (all-zero hash on deletion tombstones).
    pub new_hash: ContentHash,
    /// Caller-supplied reason (bounded by [`MAX_REFLOG_REASON_BYTES`]).
    pub reason: String,
    /// Caller-supplied actor (bounded by [`MAX_REFLOG_ACTOR_BYTES`]).
    pub actor: String,
    /// Caller-supplied timestamp in milliseconds since the unix epoch.
    pub at_ms: u64,
}

/// All-zero content hash marking a deletion tombstone (mirrors git).
fn tombstone_hash() -> ContentHash {
    ContentHash::from_bytes([0u8; 32])
}

/// Map a raw SQLite error to a typed ref error (static text only).
fn map_sqlite(err: rusqlite::Error) -> RefError {
    if let rusqlite::Error::SqliteFailure(failure, _) = &err {
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        ) {
            return RefError::Storage(WRITER_BUSY_MSG.to_owned());
        }
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return RefError::Corrupt;
        }
    }
    RefError::Storage(err.to_string())
}

/// Map a store error to a typed ref error, dropping untrusted payloads.
fn map_store_err(err: ContentStoreError) -> RefError {
    match err {
        ContentStoreError::MissingTarget(hash) => RefError::MissingTarget(hash),
        ContentStoreError::Corrupt { .. }
        | ContentStoreError::CorruptData { .. }
        | ContentStoreError::CorruptCheckpoint { .. }
        | ContentStoreError::InvalidHash(_)
        | ContentStoreError::MissingParent(_)
        | ContentStoreError::MissingTree(_) => RefError::Corrupt,
        ContentStoreError::InvalidRefName(_)
        | ContentStoreError::EmptyField(_)
        | ContentStoreError::OversizedField { .. } => RefError::InvalidName,
        ContentStoreError::WriterBusy => RefError::Storage(WRITER_BUSY_MSG.to_owned()),
        ContentStoreError::Sqlite(err) => map_sqlite(err),
        ContentStoreError::Json(err) => RefError::Storage(err.to_string()),
    }
}

/// Enforce reflog metadata bounds (length only; empty values are allowed).
fn check_reflog_meta(reason: &str, actor: &str) -> Result<(), RefError> {
    if reason.len() > MAX_REFLOG_REASON_BYTES || actor.len() > MAX_REFLOG_ACTOR_BYTES {
        return Err(RefError::InvalidName);
    }
    Ok(())
}

/// Append one reflog row inside the caller's transaction.
///
/// Bounds are enforced here (not upfront) so a violation aborts the whole
/// move transaction instead of leaving a moved ref without history.
fn append_reflog_tx(
    tx: &Transaction<'_>,
    ref_name: &str,
    old_hash: Option<&ContentHash>,
    new_hash: &ContentHash,
    reason: &str,
    actor: &str,
    at_ms: u64,
) -> Result<(), RefError> {
    check_reflog_meta(reason, actor)?;
    let old_hex: Option<String> = old_hash.map(ContentHash::to_hex);
    tx.execute(
        "INSERT INTO reflog (ref_name, old_hash, new_hash, reason, actor, at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            ref_name,
            old_hex,
            new_hash.to_hex(),
            reason,
            actor,
            at_ms as i64
        ],
    )
    .map_err(map_sqlite)?;
    Ok(())
}

/// Read one ref target inside a transaction, failing closed on bad hex.
fn read_ref_tx(tx: &Transaction<'_>, name: &str) -> Result<Option<ContentHash>, RefError> {
    let maybe_hex: Option<String> = tx
        .query_row(
            "SELECT target_hash FROM refs WHERE name = ?1",
            params![name],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(map_sqlite)?;
    match maybe_hex {
        Some(hex) => ContentHash::from_hex(&hex)
            .map(Some)
            .map_err(|_| RefError::Corrupt),
        None => Ok(None),
    }
}

/// Atomically persist tree blob, checkpoint, HEAD, branch, and reflog in one transaction.
///
/// Wheel durable commit path with branch (AI-0180): the tree blob insert,
/// the checkpoint insert, the HEAD upsert, the branch insert-or-update,
/// and the single branch reflog row commit or roll back together in one
/// SQLite `transaction()`. Reuses [`append_reflog_tx`] so an oversize
/// reason/actor aborts the whole commit (including HEAD), never a
/// half-advanced HEAD. Branch creation stores `old_hash = NULL`; updates
/// store the previous tip. The branch move is authoritative (not
/// fast-forward-only), matching the previous
/// `create_branch`/`update_branch(false)` success behavior.
#[allow(clippy::too_many_arguments)]
pub fn commit_checkpoint_with_branch(
    store: &mut ContentStore,
    tree_bytes: &[u8],
    tree_created_at_ms: u64,
    draft: CheckpointDraft,
    branch_name: &str,
    reason: &str,
    actor: &str,
    updated_at_ms: u64,
) -> Result<Checkpoint, RefError> {
    if tree_bytes.len() > MAX_BLOB_BYTES {
        return Err(map_store_err(ContentStoreError::OversizedField {
            field: "blob_data",
            size: tree_bytes.len(),
            max: MAX_BLOB_BYTES,
        }));
    }
    if draft.parents.len() > MAX_CHECKPOINT_PARENTS {
        return Err(map_store_err(ContentStoreError::OversizedField {
            field: "parents",
            size: draft.parents.len(),
            max: MAX_CHECKPOINT_PARENTS,
        }));
    }
    if draft.task_id.trim().is_empty() {
        return Err(map_store_err(ContentStoreError::EmptyField("task_id")));
    }
    if draft.task_id.len() > MAX_TASK_ID_BYTES {
        return Err(map_store_err(ContentStoreError::OversizedField {
            field: "task_id",
            size: draft.task_id.len(),
            max: MAX_TASK_ID_BYTES,
        }));
    }
    if draft.agent_id.trim().is_empty() {
        return Err(map_store_err(ContentStoreError::EmptyField("agent_id")));
    }
    if draft.agent_id.len() > MAX_AGENT_ID_BYTES {
        return Err(map_store_err(ContentStoreError::OversizedField {
            field: "agent_id",
            size: draft.agent_id.len(),
            max: MAX_AGENT_ID_BYTES,
        }));
    }
    if draft.summary.len() > MAX_SUMMARY_BYTES {
        return Err(map_store_err(ContentStoreError::OversizedField {
            field: "summary",
            size: draft.summary.len(),
            max: MAX_SUMMARY_BYTES,
        }));
    }
    draft.rationale.validate().map_err(map_store_err)?;
    let branch = BranchName::parse(branch_name)?;

    let tree_hash = ContentHash::compute(tree_bytes);
    let tree_hex = tree_hash.to_hex();
    let effective = CheckpointDraft {
        parents: draft.parents.clone(),
        task_id: draft.task_id.clone(),
        agent_id: draft.agent_id.clone(),
        rationale: draft.rationale.clone(),
        tree_hash: Some(tree_hash),
        summary: draft.summary.clone(),
        timestamp_ms: draft.timestamp_ms,
    };
    let id = effective.canonical_hash().map_err(map_store_err)?;
    let id_hex = id.to_hex();
    let rationale_json = serde_json::to_string(&effective.rationale)
        .map_err(|err| RefError::Storage(err.to_string()))?;
    let parents_json = serde_json::to_string(&effective.parents)
        .map_err(|err| RefError::Storage(err.to_string()))?;

    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    for parent in &effective.parents {
        let hex = parent.to_hex();
        let exists: bool = tx
            .query_row(
                "SELECT 1 FROM checkpoints WHERE hash = ?1",
                params![hex],
                |_| Ok(()),
            )
            .optional()
            .map_err(map_sqlite)?
            .is_some();
        if !exists {
            return Err(map_store_err(ContentStoreError::MissingParent(*parent)));
        }
    }
    tx.execute(
        "INSERT OR IGNORE INTO blobs (hash, size, data, created_at_ms) VALUES (?1, ?2, ?3, ?4)",
        params![
            tree_hex,
            tree_bytes.len() as i64,
            tree_bytes,
            tree_created_at_ms as i64
        ],
    )
    .map_err(map_sqlite)?;
    tx.execute(
            "INSERT OR IGNORE INTO checkpoints (hash, parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id_hex,
                parents_json,
                effective.task_id,
                effective.agent_id,
                rationale_json,
                tree_hex,
                effective.summary,
                effective.timestamp_ms as i64
            ],
        )
        .map_err(map_sqlite)?;
    tx.execute(
            "INSERT INTO refs (name, target_hash, updated_at_ms)
             VALUES ('HEAD', ?1, ?2)
             ON CONFLICT(name) DO UPDATE SET target_hash = excluded.target_hash, updated_at_ms = excluded.updated_at_ms",
            params![id_hex, updated_at_ms as i64],
        )
        .map_err(map_sqlite)?;
    let previous = read_ref_tx(&tx, branch.as_str())?;
    match previous {
        None => {
            tx.execute(
                "INSERT INTO refs (name, target_hash, updated_at_ms) VALUES (?1, ?2, ?3)",
                params![branch.as_str(), id_hex, updated_at_ms as i64],
            )
            .map_err(map_sqlite)?;
            append_reflog_tx(
                &tx,
                branch.as_str(),
                None,
                &id,
                reason,
                actor,
                updated_at_ms,
            )?;
        }
        Some(current) => {
            let affected = tx
                .execute(
                    "UPDATE refs SET target_hash = ?1, updated_at_ms = ?2 WHERE name = ?3",
                    params![id_hex, updated_at_ms as i64, branch.as_str()],
                )
                .map_err(map_sqlite)?;
            if affected != 1 {
                return Err(RefError::NotFound);
            }
            append_reflog_tx(
                &tx,
                branch.as_str(),
                Some(&current),
                &id,
                reason,
                actor,
                updated_at_ms,
            )?;
        }
    }
    tx.commit().map_err(map_sqlite)?;

    Ok(Checkpoint {
        id,
        parents: effective.parents,
        task_id: effective.task_id,
        agent_id: effective.agent_id,
        rationale: effective.rationale,
        tree_hash: effective.tree_hash,
        summary: effective.summary,
        timestamp_ms: effective.timestamp_ms,
    })
}

/// Create a branch pointing at an existing checkpoint (no force).
///
/// Refuses existing names with [`RefError::AlreadyExists`] and missing
/// targets with [`RefError::MissingTarget`]. Appends the creation row
/// (`old_hash = NULL`) in the same transaction as the insert.
pub fn create_branch(
    store: &mut ContentStore,
    name: &str,
    target: &ContentHash,
    reason: &str,
    actor: &str,
    at_ms: u64,
) -> Result<BranchName, RefError> {
    let branch = BranchName::parse(name)?;
    if !store.has_checkpoint(target).map_err(map_store_err)? {
        return Err(RefError::MissingTarget(*target));
    }
    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    if read_ref_tx(&tx, branch.as_str())?.is_some() {
        return Err(RefError::AlreadyExists);
    }
    tx.execute(
        "INSERT INTO refs (name, target_hash, updated_at_ms) VALUES (?1, ?2, ?3)",
        params![branch.as_str(), target.to_hex(), at_ms as i64],
    )
    .map_err(map_sqlite)?;
    append_reflog_tx(&tx, branch.as_str(), None, target, reason, actor, at_ms)?;
    tx.commit().map_err(map_sqlite)?;
    Ok(branch)
}

/// Move a branch to an existing checkpoint, recording history.
///
/// With `fast_forward_only`, the move is refused (store-level refusal,
/// [`RefError::Storage`]) unless the current tip is an ancestor of the
/// target per [`ContentStore::merge_base`](crate::content_store::ContentStore::merge_base)
/// (equal tips trivially pass).
/// Missing branches report [`RefError::NotFound`].
pub fn update_branch(
    store: &mut ContentStore,
    name: &str,
    target: &ContentHash,
    reason: &str,
    actor: &str,
    at_ms: u64,
    fast_forward_only: bool,
) -> Result<BranchName, RefError> {
    let branch = BranchName::parse(name)?;
    if !store.has_checkpoint(target).map_err(map_store_err)? {
        return Err(RefError::MissingTarget(*target));
    }
    let old_hash = match store.get_ref(branch.as_str()).map_err(map_store_err)? {
        Some(hash) => hash,
        None => return Err(RefError::NotFound),
    };
    if fast_forward_only {
        let base = store.merge_base(&old_hash, target).map_err(map_store_err)?;
        if base != Some(old_hash) {
            return Err(RefError::Storage(NON_FAST_FORWARD_MSG.to_owned()));
        }
    }
    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    let current = match read_ref_tx(&tx, branch.as_str())? {
        Some(hash) => hash,
        None => return Err(RefError::NotFound),
    };
    if fast_forward_only && current != old_hash {
        return Err(RefError::Storage(NON_FAST_FORWARD_MSG.to_owned()));
    }
    let affected = tx
        .execute(
            "UPDATE refs SET target_hash = ?1, updated_at_ms = ?2 WHERE name = ?3",
            params![target.to_hex(), at_ms as i64, branch.as_str()],
        )
        .map_err(map_sqlite)?;
    if affected != 1 {
        return Err(RefError::NotFound);
    }
    append_reflog_tx(
        &tx,
        branch.as_str(),
        Some(&current),
        target,
        reason,
        actor,
        at_ms,
    )?;
    tx.commit().map_err(map_sqlite)?;
    Ok(branch)
}

/// Delete a branch, leaving a tombstone reflog row under its name.
///
/// HEAD is refused ([`RefError::ProtectedHead`]); missing branches report
/// [`RefError::NotFound`]. Returns the deleted tip.
pub fn delete_branch(
    store: &mut ContentStore,
    name: &str,
    reason: &str,
    actor: &str,
    at_ms: u64,
) -> Result<ContentHash, RefError> {
    let branch = BranchName::parse(name)?;
    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    let old_hash = match read_ref_tx(&tx, branch.as_str())? {
        Some(hash) => hash,
        None => return Err(RefError::NotFound),
    };
    let affected = tx
        .execute("DELETE FROM refs WHERE name = ?1", params![branch.as_str()])
        .map_err(map_sqlite)?;
    if affected != 1 {
        return Err(RefError::NotFound);
    }
    append_reflog_tx(
        &tx,
        branch.as_str(),
        Some(&old_hash),
        &tombstone_hash(),
        reason,
        actor,
        at_ms,
    )?;
    tx.commit().map_err(map_sqlite)?;
    Ok(old_hash)
}

/// Atomically rename a branch: delete old, create new, two reflog rows.
///
/// One transaction holds all four writes (delete, insert, tombstone row
/// for the old name, creation row for the new name). Self-rename is
/// [`RefError::InvalidName`]; a taken new name is
/// [`RefError::AlreadyExists`]; a missing old name is
/// [`RefError::NotFound`]. Returns the moved tip.
pub fn rename_branch(
    store: &mut ContentStore,
    old_name: &str,
    new_name: &str,
    reason: &str,
    actor: &str,
    at_ms: u64,
) -> Result<ContentHash, RefError> {
    let old_branch = BranchName::parse(old_name)?;
    let new_branch = BranchName::parse(new_name)?;
    if old_branch == new_branch {
        return Err(RefError::InvalidName);
    }
    let tip = match store.get_ref(old_branch.as_str()).map_err(map_store_err)? {
        Some(hash) => hash,
        None => return Err(RefError::NotFound),
    };
    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    let current = match read_ref_tx(&tx, old_branch.as_str())? {
        Some(hash) => hash,
        None => return Err(RefError::NotFound),
    };
    if read_ref_tx(&tx, new_branch.as_str())?.is_some() {
        return Err(RefError::AlreadyExists);
    }
    let deleted = tx
        .execute(
            "DELETE FROM refs WHERE name = ?1",
            params![old_branch.as_str()],
        )
        .map_err(map_sqlite)?;
    if deleted != 1 {
        return Err(RefError::NotFound);
    }
    tx.execute(
        "INSERT INTO refs (name, target_hash, updated_at_ms) VALUES (?1, ?2, ?3)",
        params![new_branch.as_str(), tip.to_hex(), at_ms as i64],
    )
    .map_err(map_sqlite)?;
    append_reflog_tx(
        &tx,
        old_branch.as_str(),
        Some(&current),
        &tombstone_hash(),
        reason,
        actor,
        at_ms,
    )?;
    append_reflog_tx(&tx, new_branch.as_str(), None, &tip, reason, actor, at_ms)?;
    tx.commit().map_err(map_sqlite)?;
    Ok(tip)
}

/// List all branches (`heads/` only), ordered by name.
pub fn list_branches(store: &ContentStore) -> Result<Vec<(BranchName, ContentHash)>, RefError> {
    let pairs: Vec<(String, String)> = {
        let guard = store.lock_conn().map_err(map_store_err)?;
        let mut stmt = guard
            .prepare(
                "SELECT name, target_hash FROM refs WHERE name GLOB 'heads/*' ORDER BY name ASC",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(map_sqlite)?;
        let mut out = Vec::new();
        for item in rows {
            out.push(item.map_err(map_sqlite)?);
        }
        out
    };
    let mut result = Vec::with_capacity(pairs.len());
    for (name, hash_hex) in pairs {
        let branch = BranchName::parse(&name).map_err(|_| RefError::Corrupt)?;
        let hash = ContentHash::from_hex(&hash_hex).map_err(|_| RefError::Corrupt)?;
        result.push((branch, hash));
    }
    Ok(result)
}

/// Get a branch tip (`heads/` only). Missing branches return `None`.
pub fn get_branch(store: &ContentStore, name: &str) -> Result<Option<ContentHash>, RefError> {
    let branch = BranchName::parse(name)?;
    store.get_ref(branch.as_str()).map_err(map_store_err)
}

/// Read reflog history for any well-formed ref name, newest-first.
///
/// The name is validated for shape only (HEAD and deleted-branch names
/// stay readable). Every call returns at most
/// [`MAX_REFLOG_READ_LIMIT`] entries.
pub fn read_reflog(
    store: &ContentStore,
    name: &str,
    limit: usize,
) -> Result<Vec<ReflogEntry>, RefError> {
    ContentStore::validate_ref_name(name).map_err(|_| RefError::InvalidName)?;
    let capped = limit.min(MAX_REFLOG_READ_LIMIT) as i64;
    let guard = store.lock_conn().map_err(map_store_err)?;
    let mut stmt = guard
        .prepare(
            "SELECT seq, ref_name, old_hash, new_hash, reason, actor, at_ms
                 FROM reflog WHERE ref_name = ?1 ORDER BY seq DESC LIMIT ?2",
        )
        .map_err(map_sqlite)?;
    let rows = stmt
        .query_map(params![name, capped], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
            ))
        })
        .map_err(map_sqlite)?;
    let mut entries = Vec::new();
    for item in rows {
        let (seq, ref_name, old_hex, new_hex, reason, actor, at_ms) = item.map_err(map_sqlite)?;
        if ref_name != name {
            return Err(RefError::Corrupt);
        }
        let old_hash = match old_hex {
            Some(hex) => Some(ContentHash::from_hex(&hex).map_err(|_| RefError::Corrupt)?),
            None => None,
        };
        let new_hash = ContentHash::from_hex(&new_hex).map_err(|_| RefError::Corrupt)?;
        if at_ms < 0 {
            return Err(RefError::Corrupt);
        }
        entries.push(ReflogEntry {
            seq,
            ref_name,
            old_hash,
            new_hash,
            reason,
            actor,
            at_ms: at_ms as u64,
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content_store::{CheckpointDraft, Rationale};

    fn test_checkpoint(store: &mut ContentStore, summary: &str) -> ContentHash {
        store
            .commit_checkpoint(CheckpointDraft {
                parents: Vec::new(),
                task_id: "AI-0180".to_string(),
                agent_id: "ctx-0180-impl".to_string(),
                rationale: Rationale::new("Test why", "Test what"),
                tree_hash: None,
                summary: summary.to_string(),
                timestamp_ms: 1000,
            })
            .expect("commit test checkpoint")
            .id
    }

    #[test]
    fn corrupt_ref_hex_fails_closed() {
        let mut store = ContentStore::open_in_memory().expect("open");
        let target = test_checkpoint(&mut store, "tip");
        create_branch(&mut store, "heads/main", &target, "create", "tester", 1000).expect("create");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE refs SET target_hash = 'not-hex' WHERE name = 'heads/main'",
                    [],
                )
                .expect("plant corrupt hex");
        }
        assert!(matches!(
            get_branch(&store, "heads/main"),
            Err(RefError::Corrupt)
        ));
        assert!(matches!(list_branches(&store), Err(RefError::Corrupt)));
    }

    #[test]
    fn corrupt_reflog_hex_fails_closed() {
        let mut store = ContentStore::open_in_memory().expect("open");
        let target = test_checkpoint(&mut store, "tip");
        create_branch(&mut store, "heads/main", &target, "create", "tester", 1000).expect("create");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE reflog SET new_hash = 'ZZZ' WHERE ref_name = 'heads/main'",
                    [],
                )
                .expect("plant corrupt reflog hex");
        }
        assert!(matches!(
            read_reflog(&store, "heads/main", 10),
            Err(RefError::Corrupt)
        ));
    }

    #[test]
    fn oversize_reason_rolls_back_the_move() {
        let mut store = ContentStore::open_in_memory().expect("open");
        let target = test_checkpoint(&mut store, "tip");
        let long_reason = "r".repeat(MAX_REFLOG_REASON_BYTES + 1);
        let err = create_branch(
            &mut store,
            "heads/main",
            &target,
            &long_reason,
            "tester",
            1000,
        )
        .expect_err("oversize reason must fail");
        assert_eq!(err, RefError::InvalidName);
        // The ref insert rolled back with the rejected row: nothing persisted.
        assert_eq!(
            get_branch(&store, "heads/main").expect("get"),
            None,
            "failed create must leave no ref"
        );
        assert!(
            read_reflog(&store, "heads/main", 10)
                .expect("read")
                .is_empty(),
            "failed create must leave no reflog row"
        );
    }

    #[test]
    fn oversize_actor_rolls_back_update_and_rename() {
        let mut store = ContentStore::open_in_memory().expect("open");
        let first = test_checkpoint(&mut store, "first");
        let second = test_checkpoint(&mut store, "second");
        create_branch(&mut store, "heads/main", &first, "create", "tester", 1000).expect("create");
        let long_actor = "a".repeat(MAX_REFLOG_ACTOR_BYTES + 1);

        let err = update_branch(
            &mut store,
            "heads/main",
            &second,
            "update",
            &long_actor,
            2000,
            false,
        )
        .expect_err("oversize actor must fail");
        assert_eq!(err, RefError::InvalidName);
        assert_eq!(
            get_branch(&store, "heads/main").expect("get"),
            Some(first),
            "failed update must keep the old tip"
        );
        assert_eq!(
            read_reflog(&store, "heads/main", 10).expect("read").len(),
            1,
            "failed update must append no reflog row"
        );

        let err = rename_branch(
            &mut store,
            "heads/main",
            "heads/next",
            "rename",
            &long_actor,
            3000,
        )
        .expect_err("oversize actor must fail rename");
        assert_eq!(err, RefError::InvalidName);
        assert_eq!(
            get_branch(&store, "heads/main").expect("get old"),
            Some(first),
            "failed rename must keep the old branch"
        );
        assert_eq!(
            get_branch(&store, "heads/next").expect("get new"),
            None,
            "failed rename must not create the new branch"
        );
    }
}

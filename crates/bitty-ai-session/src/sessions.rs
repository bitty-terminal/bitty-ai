//! Durable session-id bindings (AI-0197, session slice v1).
//!
//! This module is the session-identity plane: a narrow, typed namespace mapping
//! caller-provided session ids to branch tips, beside the refs plane in
//! [`crate::session_refs`]. Every row lives in the `session_bindings` table:
//! `(session_id TEXT PRIMARY KEY, branch TEXT NOT NULL, head_hex TEXT NOT
//! NULL, generation INTEGER NOT NULL, epoch INTEGER NOT NULL, updated_at_ms
//! INTEGER NOT NULL)`.
//!
//! ## Two session-id types
//!
//! The runtime handle [`bitty_ai_runtime` `SessionId`] is a process-local
//! `u64` counter that must never be persisted as a stable external id. The
//! [`WheelSessionId`] here is the durable counterpart: a caller-provided
//! string bound to a branch in SQLite. The kernel never mints session ids;
//! duplicates fail closed with [`SessionError::AlreadyExists`].
//!
//! ## Namespace rules
//!
//! [`WheelSessionId::parse`] mirrors the store ref-name strictness
//! (non-empty after trim, bounded length, restricted charset) but owns its
//! own namespace: `/` is refused so a session id can never be confused with a
//! ref path, and the exact name `HEAD` is refused so a session id can never
//! be confused with the HEAD pointer. Branch names (`heads/...`) always
//! contain `/`, so the two namespaces are disjoint by construction and a
//! `session-or-branch` lookup is unambiguous. Typical ids look like
//! `sess-01k4...` but no prefix is required.
//!
//! ## Epoch fencing
//!
//! Each row stores a caller-supplied `epoch`. [`bump_session_epoch`] (and the
//! kernel resume path built on it) admits a resume only when `claim_epoch >
//! stored epoch`, otherwise refusing with [`SessionError::StaleEpoch`]
//! (same `claim_epoch`/`current_epoch` field shape as the adoption-rule
//! `StaleEpoch` in `bitty-ai-runtime`). The first bind mints epoch `1` at
//! the kernel layer. Epochs are never incremented by the store: the admitted
//! claim is written verbatim. SQLite `INTEGER` is `i64`, so any
//! caller-supplied `generation`, `epoch` (including `claim_epoch`), or
//! `updated_at_ms` above `i64::MAX` is refused at write time with
//! [`SessionError::Storage`] and zero row writes -- `u64::MAX` is never
//! stored, so one out-of-range claim cannot poison later reads or listings.
//!
//! ## Generation snapshot vs live value
//!
//! The row's `generation` column is a snapshot written at bind/resume time,
//! not a live fence: the authoritative task generation always lives in the
//! task engine. The kernel resume report carries the generation re-read from
//! the task engine at resume time (`0` when the checkpoint's `task_id` names
//! no task row, for example `task-unassigned`); the stored snapshot is then
//! updated to match. Readers that need fencing must use the task-engine or
//! merge-path generation fences, never this column.
//!
//! ## Scope
//!
//! The session list is global to the database file: there is no directory or
//! project-root field (owner decision, AI-0197). The durable pending
//! tool-call log lives in [`crate::pending`]: resume reports name its open
//! entries with an explicit present flag at the kernel layer.
//!
//! Errors never echo caller-supplied text: every [`SessionError`] display
//! string is a static literal except [`SessionError::StaleEpoch`], which
//! carries only the two epoch numbers (same precedent as the merge-path
//! `StaleGeneration` numbers).
//!
//! Time/Space: every function is O(1) single-row SQLite work (bounded
//! string/hex validation) except [`list_sessions`], which is O(N) rows for N
//! bound sessions. All clocks are caller-supplied (`updated_at_ms`); no wall
//! clock is read.

use std::fmt;

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::content_hash::ContentHash;
use crate::content_store::{ContentStore, ContentStoreError};
use crate::session_refs::{BranchName, RefError};

/// Maximum byte length for a session id (128 bytes).
///
/// Mirrors [`crate::content_store::MAX_TASK_ID_BYTES`] rather than the longer
/// ref-name bound: session ids are caller-provided opaque strings, and the
/// tighter bound keeps the primary-key index small.
pub const MAX_SESSION_ID_BYTES: usize = 128;

/// A validated durable session id (caller-provided string).
///
/// Distinct from the runtime process-local `SessionId(u64)` handle, which
/// must never be persisted. See the module docs for the namespace rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct WheelSessionId(String);

impl WheelSessionId {
    /// Parse and validate a caller-provided session id.
    ///
    /// Refuses empty/whitespace-only input, input longer than
    /// [`MAX_SESSION_ID_BYTES`] bytes, bytes outside `[A-Za-z0-9._-]`
    /// (`/` is excluded to keep the session namespace disjoint from ref
    /// paths), and the exact name `HEAD`. All failures are
    /// [`SessionError::InvalidId`]; stored-data failures surface as
    /// [`SessionError::Corrupt`] at read time instead.
    pub fn parse(raw: &str) -> Result<Self, SessionError> {
        if raw.trim().is_empty() {
            return Err(SessionError::InvalidId);
        }
        if raw.len() > MAX_SESSION_ID_BYTES {
            return Err(SessionError::InvalidId);
        }
        if raw == "HEAD" {
            return Err(SessionError::InvalidId);
        }
        for b in raw.bytes() {
            let valid = matches!(
                b,
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.'
            );
            if !valid {
                return Err(SessionError::InvalidId);
            }
        }
        Ok(Self(raw.to_owned()))
    }

    /// Borrow the validated session id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WheelSessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for WheelSessionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// One durable session binding row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionBinding {
    /// Bound session id (primary key).
    pub session_id: WheelSessionId,
    /// Branch this session follows, as validated `heads/...` text.
    pub branch: String,
    /// Last-known branch-tip checkpoint (live tip at bind/resume time).
    pub head: ContentHash,
    /// Task-generation snapshot written at bind/resume time (see module docs:
    /// informational only, never a fence).
    pub generation: u64,
    /// Admitted fencing epoch (monotonic per session, caller-supplied).
    pub epoch: u64,
    /// Caller-supplied timestamp of the last bind/resume, ms since unix epoch.
    pub updated_at_ms: u64,
}

/// Typed failures for session bind/resolve/bump/list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Caller-supplied session id violates its shape bound.
    InvalidId,
    /// Session id is already bound (the kernel never mints or reuses ids).
    AlreadyExists,
    /// Named session (or its branch) does not exist.
    NotFound,
    /// Resume claim epoch is not newer than the stored epoch.
    ///
    /// Field shape mirrors the adoption-rule `StaleEpoch`
    /// (`bitty-ai-runtime`): the claim versus the stored fence.
    StaleEpoch {
        /// Epoch carried by the resume claim.
        claim_epoch: u64,
        /// Epoch stored on the session row (the fence that refused).
        current_epoch: u64,
    },
    /// Stored data failed integrity validation (bad hex, bad shape, bad
    /// stored branch name, negative integer) or the schema is partial. Fail
    /// closed, no partial state.
    Corrupt,
    /// Underlying storage failure (static text only).
    Storage(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId => write!(f, "invalid session id"),
            Self::AlreadyExists => write!(f, "session id already bound"),
            Self::NotFound => write!(f, "session or branch not found"),
            Self::StaleEpoch {
                claim_epoch,
                current_epoch,
            } => write!(
                f,
                "stale session epoch: claim {claim_epoch} is not newer than stored {current_epoch}"
            ),
            Self::Corrupt => write!(f, "session store corrupt or incompatible"),
            Self::Storage(detail) => write!(f, "session storage error: {detail}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// Static storage refusal for a contended single-writer lock.
const WRITER_BUSY_MSG: &str = "writer busy: another writer holds the single-writer lock";

/// Static storage refusal for caller-supplied integers above `i64::MAX`.
///
/// SQLite `INTEGER` is `i64`: storing a larger `u64` via `as i64` would wrap
/// to a negative value, and the next read would fail closed as `Corrupt`
/// (poisoning `list_sessions` for every session). Write paths refuse such
/// values before any row write instead.
const INTEGER_RANGE_MSG: &str = "session integer out of range: value exceeds i64::MAX";

/// Idempotent `session_bindings` table creation.
///
/// Additive migration without touching the frozen content-schema admission
/// list: pre-AI-0197 database files gain the table on the first session call
/// instead of failing admission. Every session function calls this first, so
/// a fresh database lists zero sessions rather than erroring.
const SESSION_BINDINGS_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS session_bindings (
    session_id TEXT PRIMARY KEY,
    branch TEXT NOT NULL,
    head_hex TEXT NOT NULL,
    generation INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);";

/// Map a raw SQLite error to a typed session error (static text only).
fn map_sqlite(err: rusqlite::Error) -> SessionError {
    if let rusqlite::Error::SqliteFailure(failure, _) = &err {
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        ) {
            return SessionError::Storage(WRITER_BUSY_MSG.to_owned());
        }
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return SessionError::Corrupt;
        }
    }
    SessionError::Storage(err.to_string())
}

/// Map a store error to a typed session error, dropping untrusted payloads.
fn map_store_err(err: ContentStoreError) -> SessionError {
    match err {
        ContentStoreError::Corrupt { .. }
        | ContentStoreError::CorruptData { .. }
        | ContentStoreError::CorruptCheckpoint { .. }
        | ContentStoreError::InvalidHash(_)
        | ContentStoreError::MissingParent(_)
        | ContentStoreError::MissingTree(_)
        | ContentStoreError::MissingTarget(_) => SessionError::Corrupt,
        ContentStoreError::InvalidRefName(_)
        | ContentStoreError::EmptyField(_)
        | ContentStoreError::OversizedField { .. } => SessionError::InvalidId,
        ContentStoreError::WriterBusy => SessionError::Storage(WRITER_BUSY_MSG.to_owned()),
        ContentStoreError::Sqlite(err) => map_sqlite(err),
        ContentStoreError::Json(err) => SessionError::Storage(err.to_string()),
    }
}

/// Map a branch-verb error to a session error for branch arguments.
///
/// `AlreadyExists`/`InvalidName` keep their meaning across planes; a missing
/// branch is [`SessionError::NotFound`] on this plane.
fn map_ref_err(err: RefError) -> SessionError {
    match err {
        RefError::InvalidName | RefError::ProtectedHead | RefError::ReservedNamespace => {
            SessionError::InvalidId
        }
        RefError::AlreadyExists => SessionError::AlreadyExists,
        RefError::NotFound => SessionError::NotFound,
        RefError::MissingTarget(_) | RefError::Corrupt => SessionError::Corrupt,
        RefError::Storage(detail) => SessionError::Storage(detail),
    }
}

/// Ensure the `session_bindings` table exists (idempotent).
fn ensure_table(store: &ContentStore) -> Result<(), SessionError> {
    let guard = store.lock_conn().map_err(map_store_err)?;
    guard
        .execute_batch(SESSION_BINDINGS_SCHEMA)
        .map_err(map_sqlite)?;
    Ok(())
}

/// Read one binding row by session id inside a held connection scope.
///
/// Stored branch names re-validate (a row naming a non-branch is
/// [`SessionError::Corrupt`]); negative integers and bad head hex are
/// [`SessionError::Corrupt`]. Missing rows return `None`.
fn read_binding(
    conn: &rusqlite::Connection,
    session_id: &WheelSessionId,
) -> Result<Option<SessionBinding>, SessionError> {
    let row: Option<(String, String, i64, i64, i64)> = conn
        .query_row(
            "SELECT branch, head_hex, generation, epoch, updated_at_ms
             FROM session_bindings WHERE session_id = ?1",
            params![session_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )
        .optional()
        .map_err(map_sqlite)?;
    match row {
        None => Ok(None),
        Some((branch, head_hex, generation, epoch, updated_at_ms)) => {
            if BranchName::parse(&branch).is_err() {
                return Err(SessionError::Corrupt);
            }
            let head = ContentHash::from_hex(&head_hex).map_err(|_| SessionError::Corrupt)?;
            if generation < 0 || epoch < 0 || updated_at_ms < 0 {
                return Err(SessionError::Corrupt);
            }
            Ok(Some(SessionBinding {
                session_id: session_id.clone(),
                branch,
                head,
                generation: generation as u64,
                epoch: epoch as u64,
                updated_at_ms: updated_at_ms as u64,
            }))
        }
    }
}

/// Bind a session id to a branch tip (no force).
///
/// The branch name shape-validates via the refs plane; existence and tip
/// checks belong to the caller (the kernel resolves the live tip first).
/// A bound id reports [`SessionError::AlreadyExists`]; the check and the
/// insert run in one transaction on the single-writer connection. `epoch`
/// should be `1` on first bind (kernel convention); `generation` is the
/// snapshot documented above.
pub fn bind_session(
    store: &ContentStore,
    session_id: &WheelSessionId,
    branch: &BranchName,
    head: &ContentHash,
    generation: u64,
    epoch: u64,
    now_ms: u64,
) -> Result<SessionBinding, SessionError> {
    if generation > i64::MAX as u64 || epoch > i64::MAX as u64 || now_ms > i64::MAX as u64 {
        return Err(SessionError::Storage(INTEGER_RANGE_MSG.to_owned()));
    }
    ensure_table(store)?;
    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    if read_binding(&tx, session_id)?.is_some() {
        return Err(SessionError::AlreadyExists);
    }
    tx.execute(
        "INSERT INTO session_bindings
         (session_id, branch, head_hex, generation, epoch, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            session_id.as_str(),
            branch.as_str(),
            head.to_hex(),
            generation as i64,
            epoch as i64,
            now_ms as i64
        ],
    )
    .map_err(map_sqlite)?;
    tx.commit().map_err(map_sqlite)?;
    Ok(SessionBinding {
        session_id: session_id.clone(),
        branch: branch.as_str().to_owned(),
        head: *head,
        generation,
        epoch,
        updated_at_ms: now_ms,
    })
}

/// Resolve a session binding. Missing ids return `None`; corrupt stored rows
/// fail closed with [`SessionError::Corrupt`].
pub fn resolve_session(
    store: &ContentStore,
    session_id: &WheelSessionId,
) -> Result<Option<SessionBinding>, SessionError> {
    ensure_table(store)?;
    let guard = store.lock_conn().map_err(map_store_err)?;
    read_binding(&guard, session_id)
}

/// Admit a fenced resume: requires `claim_epoch > stored epoch`.
///
/// On success the row advances to (`head`, `generation`, `claim_epoch`,
/// `now_ms`) in the same transaction as the fence check and the updated row
/// is returned. A stale claim (`<=` stored) refuses with
/// [`SessionError::StaleEpoch`] with zero writes; a missing id is
/// [`SessionError::NotFound`].
pub fn bump_session_epoch(
    store: &ContentStore,
    session_id: &WheelSessionId,
    claim_epoch: u64,
    head: &ContentHash,
    generation: u64,
    now_ms: u64,
) -> Result<SessionBinding, SessionError> {
    if claim_epoch > i64::MAX as u64 || generation > i64::MAX as u64 || now_ms > i64::MAX as u64 {
        return Err(SessionError::Storage(INTEGER_RANGE_MSG.to_owned()));
    }
    ensure_table(store)?;
    let mut guard = store.lock_conn().map_err(map_store_err)?;
    let tx = guard.transaction().map_err(map_sqlite)?;
    let Some(current) = read_binding(&tx, session_id)? else {
        return Err(SessionError::NotFound);
    };
    if claim_epoch <= current.epoch {
        return Err(SessionError::StaleEpoch {
            claim_epoch,
            current_epoch: current.epoch,
        });
    }
    let affected = tx
        .execute(
            "UPDATE session_bindings
             SET head_hex = ?1, generation = ?2, epoch = ?3, updated_at_ms = ?4
             WHERE session_id = ?5",
            params![
                head.to_hex(),
                generation as i64,
                claim_epoch as i64,
                now_ms as i64,
                session_id.as_str()
            ],
        )
        .map_err(map_sqlite)?;
    if affected != 1 {
        return Err(SessionError::NotFound);
    }
    tx.commit().map_err(map_sqlite)?;
    Ok(SessionBinding {
        session_id: session_id.clone(),
        branch: current.branch,
        head: *head,
        generation,
        epoch: claim_epoch,
        updated_at_ms: now_ms,
    })
}

/// List all session bindings, ordered by session id ascending.
///
/// Global to the database file: no directory or project-root scoping (owner
/// decision, AI-0197). A fresh database lists zero sessions.
pub fn list_sessions(store: &ContentStore) -> Result<Vec<SessionBinding>, SessionError> {
    ensure_table(store)?;
    let pairs: Vec<(String, String, String, i64, i64, i64)> = {
        let guard = store.lock_conn().map_err(map_store_err)?;
        let mut stmt = guard
            .prepare(
                "SELECT session_id, branch, head_hex, generation, epoch, updated_at_ms
                 FROM session_bindings ORDER BY session_id ASC",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })
            .map_err(map_sqlite)?;
        let mut out = Vec::new();
        for item in rows {
            out.push(item.map_err(map_sqlite)?);
        }
        out
    };
    let mut result = Vec::with_capacity(pairs.len());
    for (id_raw, branch, head_hex, generation, epoch, updated_at_ms) in pairs {
        let session_id = WheelSessionId::parse(&id_raw).map_err(|_| SessionError::Corrupt)?;
        if BranchName::parse(&branch).is_err() {
            return Err(SessionError::Corrupt);
        }
        let head = ContentHash::from_hex(&head_hex).map_err(|_| SessionError::Corrupt)?;
        if generation < 0 || epoch < 0 || updated_at_ms < 0 {
            return Err(SessionError::Corrupt);
        }
        result.push(SessionBinding {
            session_id,
            branch,
            head,
            generation: generation as u64,
            epoch: epoch as u64,
            updated_at_ms: updated_at_ms as u64,
        });
    }
    Ok(result)
}

/// Re-export the refs-plane error mapping for kernel call sites that accept
/// branch-name arguments alongside session ids.
pub fn map_branch_err(err: RefError) -> SessionError {
    map_ref_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content_store::{CheckpointDraft, Rationale};

    fn test_store() -> ContentStore {
        let store = ContentStore::open_in_memory().expect("open");
        ensure_table(&store).expect("ensure");
        store
    }

    fn test_tip(store: &mut ContentStore) -> ContentHash {
        store
            .commit_checkpoint(CheckpointDraft {
                parents: Vec::new(),
                task_id: "AI-0197".to_string(),
                agent_id: "ctx-0197-impl".to_string(),
                rationale: Rationale::new("Test why", "Test what"),
                tree_hash: None,
                summary: "tip".to_string(),
                timestamp_ms: 1000,
            })
            .expect("commit test checkpoint")
            .id
    }

    fn test_branch() -> BranchName {
        BranchName::parse("heads/main").expect("valid test branch")
    }

    #[test]
    fn session_id_accepts_sess_prefix_and_rejects_ref_shapes() {
        assert!(WheelSessionId::parse("sess-01k4abc").is_ok());
        assert!(WheelSessionId::parse("main").is_ok());
        assert!(WheelSessionId::parse("a").is_ok());
        // Ref-namespace shapes are refused: slash paths and HEAD.
        assert_eq!(
            WheelSessionId::parse("heads/main"),
            Err(SessionError::InvalidId)
        );
        assert_eq!(WheelSessionId::parse("HEAD"), Err(SessionError::InvalidId));
        assert_eq!(WheelSessionId::parse("a/b"), Err(SessionError::InvalidId));
        // Empty, blank, oversize, and off-charset inputs fail closed.
        assert_eq!(WheelSessionId::parse(""), Err(SessionError::InvalidId));
        assert_eq!(WheelSessionId::parse("   "), Err(SessionError::InvalidId));
        assert_eq!(
            WheelSessionId::parse(&"s".repeat(MAX_SESSION_ID_BYTES + 1)),
            Err(SessionError::InvalidId)
        );
        assert_eq!(
            WheelSessionId::parse("sess one"),
            Err(SessionError::InvalidId)
        );
        assert_eq!(
            WheelSessionId::parse("sess:1"),
            Err(SessionError::InvalidId)
        );
    }

    #[test]
    fn bind_duplicate_is_already_exists_and_keeps_first_row() {
        let store = test_store();
        let mut owned = ContentStore::open_in_memory().expect("open");
        let tip = test_tip(&mut owned);
        let id = WheelSessionId::parse("sess-dup").expect("valid id");
        let first = bind_session(&store, &id, &test_branch(), &tip, 0, 1, 1000).expect("bind");
        assert_eq!(first.epoch, 1);
        assert_eq!(
            bind_session(&store, &id, &test_branch(), &tip, 0, 1, 2000),
            Err(SessionError::AlreadyExists)
        );
        let kept = resolve_session(&store, &id)
            .expect("resolve")
            .expect("present");
        assert_eq!(
            kept.updated_at_ms, 1000,
            "failed bind must not move the row"
        );
    }

    #[test]
    fn resolve_missing_returns_none_and_bump_missing_is_not_found() {
        let store = test_store();
        let id = WheelSessionId::parse("sess-absent").expect("valid id");
        assert_eq!(resolve_session(&store, &id).expect("resolve"), None);
        let mut owned = ContentStore::open_in_memory().expect("open");
        let tip = test_tip(&mut owned);
        assert_eq!(
            bump_session_epoch(&store, &id, 2, &tip, 0, 1000),
            Err(SessionError::NotFound)
        );
    }

    #[test]
    fn bump_requires_strictly_greater_epoch() {
        let store = test_store();
        let mut owned = ContentStore::open_in_memory().expect("open");
        let tip = test_tip(&mut owned);
        let id = WheelSessionId::parse("sess-fence").expect("valid id");
        bind_session(&store, &id, &test_branch(), &tip, 0, 1, 1000).expect("bind");
        // Equal and lower claims refuse with zero writes.
        assert_eq!(
            bump_session_epoch(&store, &id, 1, &tip, 0, 2000),
            Err(SessionError::StaleEpoch {
                claim_epoch: 1,
                current_epoch: 1
            })
        );
        assert_eq!(
            bump_session_epoch(&store, &id, 0, &tip, 0, 2000),
            Err(SessionError::StaleEpoch {
                claim_epoch: 0,
                current_epoch: 1
            })
        );
        let kept = resolve_session(&store, &id)
            .expect("resolve")
            .expect("present");
        assert_eq!(kept.epoch, 1);
        assert_eq!(kept.updated_at_ms, 1000);
        // A greater claim advances head, generation, epoch, and timestamp.
        let advanced = bump_session_epoch(&store, &id, 3, &tip, 7, 3000).expect("bump");
        assert_eq!(advanced.epoch, 3);
        assert_eq!(advanced.generation, 7);
        assert_eq!(advanced.updated_at_ms, 3000);
    }

    #[test]
    fn corrupt_stored_head_hex_fails_closed() {
        let store = test_store();
        let mut owned = ContentStore::open_in_memory().expect("open");
        let tip = test_tip(&mut owned);
        let id = WheelSessionId::parse("sess-rot").expect("valid id");
        bind_session(&store, &id, &test_branch(), &tip, 0, 1, 1000).expect("bind");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE session_bindings SET head_hex = 'not-hex' WHERE session_id = 'sess-rot'",
                    [],
                )
                .expect("plant corrupt hex");
        }
        assert_eq!(resolve_session(&store, &id), Err(SessionError::Corrupt));
        assert!(matches!(list_sessions(&store), Err(SessionError::Corrupt)));
    }

    #[test]
    fn list_is_global_and_ordered_by_session_id() {
        let store = test_store();
        let mut owned = ContentStore::open_in_memory().expect("open");
        let tip = test_tip(&mut owned);
        assert!(list_sessions(&store).expect("list").is_empty());
        for name in ["sess-b", "sess-a"] {
            let id = WheelSessionId::parse(name).expect("valid id");
            bind_session(&store, &id, &test_branch(), &tip, 0, 1, 1000).expect("bind");
        }
        let listed = list_sessions(&store).expect("list");
        let names: Vec<&str> = listed.iter().map(|b| b.session_id.as_str()).collect();
        assert_eq!(names, vec!["sess-a", "sess-b"]);
    }

    #[test]
    fn out_of_range_integers_refused_with_zero_row_writes() {
        let store = test_store();
        let mut owned = ContentStore::open_in_memory().expect("open");
        let tip = test_tip(&mut owned);
        let range_err = SessionError::Storage(INTEGER_RANGE_MSG.to_owned());
        // Bind-path refusals leave no row behind and the list stays empty.
        for (generation, epoch, now_ms) in
            [(u64::MAX, 1, 1000), (0, u64::MAX, 1000), (0, 1, u64::MAX)]
        {
            let id = WheelSessionId::parse("sess-max").expect("valid id");
            assert_eq!(
                bind_session(&store, &id, &test_branch(), &tip, generation, epoch, now_ms),
                Err(range_err.clone()),
                "out-of-range bind must refuse"
            );
            assert_eq!(resolve_session(&store, &id).expect("resolve"), None);
            assert!(list_sessions(&store).expect("list").is_empty());
        }
        // A valid row binds, then out-of-range bumps move nothing.
        let id = WheelSessionId::parse("sess-max").expect("valid id");
        bind_session(&store, &id, &test_branch(), &tip, 0, 1, 1000).expect("bind");
        for (claim_epoch, generation, now_ms) in
            [(u64::MAX, 0, 2000), (2, u64::MAX, 2000), (2, 0, u64::MAX)]
        {
            assert_eq!(
                bump_session_epoch(&store, &id, claim_epoch, &tip, generation, now_ms),
                Err(range_err.clone()),
                "out-of-range bump must refuse"
            );
        }
        let kept = resolve_session(&store, &id)
            .expect("resolve")
            .expect("present");
        assert_eq!(kept.epoch, 1);
        assert_eq!(kept.generation, 0);
        assert_eq!(kept.updated_at_ms, 1000);
        let listed = list_sessions(&store).expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].epoch, 1);
    }
}

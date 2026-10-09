//! Durable pending-effect log (AI-0199, session slice v1).
//!
//! This module is the in-flight tool-call record: one row per host-contacted
//! tool call, opened before the effect dispatches and closed with a terminal
//! disposition after it returns. A crash between the two leaves an `open`
//! row behind, so a reopened kernel can name the call as a pending Unknown
//! in its resume report instead of silently dropping it.
//!
//! Every row lives in the `pending_effects` table:
//! `(call_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, task_id TEXT, tool
//! TEXT NOT NULL, args_digest TEXT NOT NULL, status TEXT NOT NULL DEFAULT
//! 'open', disposition TEXT, generation INTEGER NOT NULL, epoch INTEGER NOT
//! NULL, begun_at_ms INTEGER NOT NULL, resolved_at_ms INTEGER)`, with an
//! index on `(session_id, status)` serving the resume-path query.
//!
//! ## Identity
//!
//! The durable key is an opaque hex `call_id` minted by the host at begin
//! (see `bitty-ai-slice` `pending_host` for the minting rule), never the
//! runtime [`bitty_ai_runtime` `ExecutionId`]: execution ids are documented
//! process-local handles that must never be persisted as stable external
//! ids (`session.rs`), so they appear here at most as informational
//! attribution and never as keys.
//!
//! ## What opens and what closes
//!
//! `begin` records one contacted call: the executor was reached, so an
//! effect may be in flight. Admission refusals (`Refused`: the bus precheck
//! refused before any executor contact, so nothing was attempted) never open
//! a row -- there is nothing in flight to reconcile. `resolve` closes an
//! open row with a terminal [`PendingDisposition`]; an
//! [`ToolError::EffectUnknown`][bitty_ai_runtime_tool] return never resolves
//! -- the row stays open as the persisted Unknown record until the host or
//! the user reconciles it by inspection or direction (no auto-reconcile, no
//! replay; the log never grants retry eligibility).
//!
//! ## Fencing
//!
//! Each row snapshots the caller-supplied `generation` and `epoch` at begin.
//! `resolve` admits only when `claim_epoch >= stored epoch` (continuity
//! under the existing fence: the normal executor claims the same epoch it
//! began with) and when `expected_generation == stored generation` (the
//! task-generation token, mirroring the task-engine complete/fail fence).
//! A stale writer -- an epoch older than the row's, or a generation that
//! moved on (retry bump) -- refuses with [`PendingError::StaleEpoch`] or
//! [`PendingError::StaleGeneration`] with zero writes; the entry stays open.
//!
//! Note the deliberate asymmetry with
//! [`bump_session_epoch`](crate::sessions::bump_session_epoch): a bump
//! admits a *new* fence and therefore requires a strictly greater claim,
//! while a resolve continues *under* the existing fence and therefore admits
//! equality. The stored snapshot is informational beside the authoritative
//! live values (session-epoch row, task-engine generation); a takeover that
//! advances those must reconcile open entries explicitly -- the store never
//! rewrites another owner's fence.
//!
//! ## Double-resolve
//!
//! Resolve is idempotent under disposition match via a bounded recent cache:
//! resolving an already resolved call id with the *same* disposition returns
//! the cached snapshot with no write (a crash-ack-loss retry is
//! indistinguishable from a duplicate, and erroring would strand callers
//! that cannot tell the two apart). A re-resolve with a *different*
//! disposition refuses with [`PendingError::Mismatch`] and zero writes.
//! A re-resolve of an id evicted from the cache (or never seen) reports
//! [`PendingError::NotFound`]: eviction trades perfect recall for a hard
//! bound, so callers must treat `NotFound` after eviction as unknown, never
//! as proof the effect did not happen.
//!
//! [`PendingError::Mismatch`] is a dedicated variant rather than a
//! `Storage(String)` text match: the idempotency contract is caller-facing,
//! and matching on error text is weaker than matching on a variant. The two
//! dispositions ride along as closed-vocabulary enum values (never
//! caller-supplied text), so the error surface stays static.
//!
//! Retention is bounded by delete-on-resolve: `resolve` deletes the open row
//! from `pending_effects` and inserts a full snapshot into
//! `pending_resolved_recent`, which keeps only the newest
//! [`MAX_RESOLVED_RECENT`] rows (oldest-evict on insert, ordered by
//! `resolved_at_ms` then `call_id`). The `(call_id, disposition)` pair in
//! that cache is the idempotency key. The live table therefore holds opens
//! only, and both resume scans filter `status = 'open'` in SQL (via the
//! `(session_id, status)` index for the session query), so cached growth
//! never widens the resume set or its scan cost: `list_open` reads opens of
//! one session only, `open_blob_roots` reads opens file-wide only. The
//! earlier claim that unbounded retention "never widens the resume set"
//! confused the returned set (opens only) with scan cost (which did grow
//! with every retained row); the cache plus the status filter fix the cost.
//!
//! ## Explicit HEAD association (AI-0200)
//!
//! Direct-`HEAD` resumes report only sessions carrying a stored HEAD marker
//! in the `pending_head_scope` table (`session_id TEXT PRIMARY KEY,
//! bound_at_ms INTEGER NOT NULL`). The marker is explicit: the kernel writes
//! it via [`PendingStore::bind_head`] (caller-supplied `bound_at_ms`, no wall
//! clock) and removes it via [`PendingStore::unbind_head`]; the resume path
//! lists it via [`PendingStore::list_head_sessions`] (ascending session-id
//! order) and unions each marked session oldest-first. Branches that happen
//! to share one checkpoint tip never leak into each other: the `heads/...`
//! path filters on stored branch-name equality, and the `HEAD` path reads
//! only the marker table, never tip equality. A session may carry both a
//! branch binding and a HEAD marker (the marker is additive); an unmarked
//! session is invisible to direct-`HEAD` resumes even when its branch tip
//! equals `HEAD`. Malformed marker rows (bad session-id shape, negative
//! timestamp) fail the whole read as [`PendingError::Corrupt`] with rows
//! preserved, so a poisoned marker refuses the resume with zero writes like
//! any other poisoned pending row.
//!
//! ## Bounds and failure posture
//!
//! At most [`MAX_OPEN_PENDING_PER_SESSION`] open rows per session: an
//! over-cap `begin` refuses with [`PendingError::TooManyOpen`] before any
//! write. Tool names are bounded to [`MAX_PENDING_TOOL_NAME`] bytes,
//! call ids to [`MAX_PENDING_CALL_ID`] bytes of ASCII hex, and the args
//! digest is exactly the 64 lowercase hex characters of the SHA-256 over the
//! raw args (raw args are never stored: they can exceed the tool-argument
//! bound). SQLite `INTEGER` is `i64`, so any caller-supplied `generation`,
//! `epoch`, or timestamp above `i64::MAX` is refused at write time with
//! [`PendingError::Storage`] and zero row writes (AI-0197 lesson).
//!
//! Reads fail closed: any malformed row (bad id shape, bad tool, bad digest
//! hex, unknown status or disposition, negative integer, open row carrying
//! disposition state or vice versa) fails the whole read as
//! [`PendingError::Corrupt`] with rows preserved -- nothing is auto-dropped.
//!
//! Errors never echo caller-supplied text: every [`PendingError`] display
//! string is a static literal except the numeric fence fields (epochs,
//! generations, limits) and the closed-vocabulary disposition pair on
//! `Mismatch`.
//!
//! Time/Space: `begin`/`resolve`/`get` are O(1) single-row SQLite work
//! (plus a bounded cache-eviction step on resolve, at most
//! [`MAX_RESOLVED_RECENT`] cached rows); `list_open` is O(N) rows for N open
//! entries of one session (SQL filters `status = 'open'`, so cached and
//! legacy resolved rows are never read for validation); `open_blob_roots`
//! is O(N) rows for N open entries file-wide (same filter). `bind_head` and
//! `unbind_head` are O(1) single-row marker writes (plus a bounded
//! marker-count check on bind, at most [`MAX_HEAD_MARKERS`] rows);
//! `list_head_sessions` reads at most [`MAX_HEAD_MARKERS`] rows (SQL
//! `LIMIT`, so the direct-`HEAD` resume fan-out stays bounded). All clocks
//! are caller-supplied (`begun_at_ms` / `resolved_at_ms` / `bound_at_ms`);
//! no wall clock, no threads.
//!
//! [bitty_ai_runtime_tool]: https://github.com/bitty-terminal/bitty-ai

use std::fmt;

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::content_hash::ContentHash;
use crate::content_store::{ContentStore, ContentStoreError, MAX_TASK_ID_BYTES};
use crate::sessions::WheelSessionId;

/// Maximum open pending-effect rows per session (32).
///
/// Mirrors the runtime `MAX_EXECUTIONS_PER_AGENT` capacity story: the live
/// set stays small enough to list on every resume.
pub const MAX_OPEN_PENDING_PER_SESSION: usize = 32;

/// Maximum recently resolved snapshots kept for idempotent re-resolve (32).
///
/// The cache is file-global (not per session): every `resolve` deletes its
/// open row and inserts one snapshot here, then evicts oldest beyond this
/// bound (ordered by `resolved_at_ms`, then `call_id`). A re-resolve hitting
/// the cache returns `Ok` (same disposition) or `Mismatch` (conflicting
/// disposition) with no write; an evicted or never-seen id reports
/// `NotFound`. The bound matches the open cap so the worst-case pending
/// footprint stays `32 + 32` rows.
pub const MAX_RESOLVED_RECENT: usize = 32;

/// Maximum byte length for a pending-effect tool name (64).
///
/// Mirrors the runtime `MAX_TOOL_NAME_LEN`: the log records the same names
/// the bus dispatches, never longer ones.
pub const MAX_PENDING_TOOL_NAME: usize = 64;

/// Maximum byte length for a pending-effect call id (128).
///
/// Mirrors the journal-prototype `MAX_ID_BYTES`: opaque hex ids stay index
/// sized.
pub const MAX_PENDING_CALL_ID: usize = 128;

/// Maximum distinct HEAD-marker rows admitted in `pending_head_scope` (32).
///
/// Mirrors the [`MAX_OPEN_PENDING_PER_SESSION`] and [`MAX_RESOLVED_RECENT`]
/// capacity story: the marked set stays small enough to list on every
/// direct-`HEAD` resume. A resume unions each marked session oldest-first,
/// and each session holds at most [`MAX_OPEN_PENDING_PER_SESSION`] open
/// rows, so the worst-case resume fan-out stays `32 x 32` entries -- the
/// same order as the documented worst-case pending footprint. Admission
/// refuses past this bound with zero writes; the read path carries the same
/// bound as SQL `LIMIT`, so even rows planted outside admission (raw SQL)
/// cannot make a resume load more markers.
pub const MAX_HEAD_MARKERS: usize = 32;

/// Lifecycle status of one pending-effect row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PendingStatus {
    /// The call was contacted and has no terminal disposition yet.
    Open,
    /// The call closed with a terminal disposition (the live row is deleted
    /// on resolve; a bounded snapshot lives in `pending_resolved_recent`
    /// for idempotent re-resolve, see the module docs).
    Resolved,
}

impl PendingStatus {
    /// Render the status as its stored text form.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Resolved => "resolved",
        }
    }

    /// Parse stored text back into a status.
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "open" => Some(Self::Open),
            "resolved" => Some(Self::Resolved),
            _ => None,
        }
    }
}

impl fmt::Display for PendingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Terminal disposition closing one pending-effect row.
///
/// Closed vocabulary: only these values are ever stored or accepted, so
/// stored dispositions validate exactly and error output never carries
/// caller-supplied text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PendingDisposition {
    /// Host executed and acknowledged.
    Success,
    /// Host executed and reported failure.
    Failed,
    /// Host or policy refused after executor contact.
    Denied,
    /// Refused before executor contact (recorded only when a resolve races
    /// an admission path; `begin` itself never opens admission refusals).
    Refused,
    /// Uncertain effect reconciled by inspection or user direction:
    /// the effect is confirmed.
    UnknownReconciled,
    /// Uncertain effect reconciled by inspection or user direction:
    /// escalated without confirmation.
    UnknownEscalated,
    /// Abandoned without execution confirmation (owner-directed).
    Abandoned,
}

impl PendingDisposition {
    /// Render the disposition as its stored text form.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Denied => "denied",
            Self::Refused => "refused",
            Self::UnknownReconciled => "unknown-reconciled",
            Self::UnknownEscalated => "unknown-escalated",
            Self::Abandoned => "abandoned",
        }
    }

    /// Parse stored or caller-supplied text back into a disposition.
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "success" => Some(Self::Success),
            "failed" => Some(Self::Failed),
            "denied" => Some(Self::Denied),
            "refused" => Some(Self::Refused),
            "unknown-reconciled" => Some(Self::UnknownReconciled),
            "unknown-escalated" => Some(Self::UnknownEscalated),
            "abandoned" => Some(Self::Abandoned),
            _ => None,
        }
    }
}

impl fmt::Display for PendingDisposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One durable pending-effect row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingEntry {
    /// Opaque host-minted hex id (primary key).
    pub call_id: String,
    /// Session that owns the call.
    pub session_id: WheelSessionId,
    /// Task the call ran under, when task-assigned.
    pub task_id: Option<String>,
    /// Tool that was contacted.
    pub tool: String,
    /// SHA-256 hex over the raw args (raw args are never stored).
    pub args_digest: ContentHash,
    /// Lifecycle status.
    pub status: PendingStatus,
    /// Terminal disposition (`None` while open).
    pub disposition: Option<PendingDisposition>,
    /// Generation snapshot written at begin (fence, not live value).
    pub generation: u64,
    /// Epoch snapshot written at begin (fence, not live value).
    pub epoch: u64,
    /// Caller-supplied begin timestamp, ms since unix epoch.
    pub begun_at_ms: u64,
    /// Caller-supplied resolve timestamp (`None` while open).
    pub resolved_at_ms: Option<u64>,
}

/// Typed failures for pending begin/resolve/list/get.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingError {
    /// Caller-supplied id, tool, task, or digest violates its shape bound.
    InvalidId,
    /// Call id is already recorded (the host never mints or reuses ids).
    AlreadyExists,
    /// Named call id does not exist.
    NotFound,
    /// Resolve claim epoch predates the row's fence.
    ///
    /// Field shape mirrors the session-plane `StaleEpoch`: the claim versus
    /// the fence that refused.
    StaleEpoch {
        /// Epoch carried by the resolve claim.
        claim_epoch: u64,
        /// Epoch stored on the pending row (the fence that refused).
        current_epoch: u64,
    },
    /// Resolve generation token differs from the row's fence.
    ///
    /// Field shape mirrors the merge-path `StaleGeneration` (fence first,
    /// supplied second).
    StaleGeneration {
        /// Generation stored on the pending row (the fence that refused).
        expected: u64,
        /// Caller-supplied generation that failed the fence.
        found: u64,
    },
    /// Admission bound refused.
    ///
    /// The session already holds the maximum open rows
    /// ([`MAX_OPEN_PENDING_PER_SESSION`]), or the store already holds the
    /// maximum distinct HEAD markers ([`MAX_HEAD_MARKERS`]). The bound that
    /// refused travels in `limit`; no new variant is minted for the marker
    /// path because this is the established admission-bound refusal shape.
    TooManyOpen {
        /// Bound that refused.
        limit: usize,
    },
    /// Re-resolve with a different disposition than the stored one.
    ///
    /// Carries only closed-vocabulary dispositions, never caller text (see
    /// the module docs for why this is a variant rather than text).
    Mismatch {
        /// Disposition stored on the resolved row.
        stored: PendingDisposition,
        /// Caller-supplied disposition that failed the guard.
        supplied: PendingDisposition,
    },
    /// Stored data failed integrity validation or the schema is partial.
    /// Fail closed, no partial state, rows preserved.
    Corrupt,
    /// Underlying storage failure (static text only).
    Storage(String),
}

impl fmt::Display for PendingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId => write!(f, "invalid pending id, tool, task, or digest"),
            Self::AlreadyExists => write!(f, "pending call id already recorded"),
            Self::NotFound => write!(f, "pending call id not found"),
            Self::StaleEpoch {
                claim_epoch,
                current_epoch,
            } => write!(
                f,
                "stale pending epoch: claim {claim_epoch} predates stored {current_epoch}"
            ),
            Self::StaleGeneration { expected, found } => {
                write!(
                    f,
                    "stale pending generation: expected {expected}, found {found}"
                )
            }
            Self::TooManyOpen { limit } => {
                write!(f, "too many open pending effects: limit {limit}")
            }
            Self::Mismatch { stored, supplied } => write!(
                f,
                "pending call already resolved as {stored}; refusing conflicting {supplied}"
            ),
            Self::Corrupt => write!(f, "pending store corrupt or incompatible"),
            Self::Storage(detail) => write!(f, "pending storage error: {detail}"),
        }
    }
}

impl std::error::Error for PendingError {}

/// Static storage refusal for a contended single-writer lock.
const WRITER_BUSY_MSG: &str = "writer busy: another writer holds the single-writer lock";

/// Static storage refusal for caller-supplied integers above `i64::MAX`.
///
/// SQLite `INTEGER` is `i64`: storing a larger `u64` via `as i64` would wrap
/// to a negative value, and the next read would fail closed as `Corrupt`
/// (poisoning `list_open` for the whole session). Write paths refuse such
/// values before any row write instead.
const INTEGER_RANGE_MSG: &str = "pending integer out of range: value exceeds i64::MAX";

/// Idempotent pending tables creation plus the resume-path index.
///
/// Additive migration without touching the frozen content-schema admission
/// list: pre-AI-0199 database files gain the tables on the first pending call
/// instead of failing admission. Every pending function calls this first, so
/// a fresh database lists zero open entries rather than erroring. Owned
/// here so `sessions.rs` needs zero churn. The live table holds opens only
/// (resolved rows are deleted on resolve); the recent table holds the newest
/// [`MAX_RESOLVED_RECENT`] resolved snapshots for idempotent re-resolve. The
/// head-scope table holds the explicit HEAD markers for direct-`HEAD`
/// resumes (AI-0200): one row per marked session, never inferred from tips.
const PENDING_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS pending_effects (
    call_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT,
    tool TEXT NOT NULL,
    args_digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    disposition TEXT,
    generation INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    begun_at_ms INTEGER NOT NULL,
    resolved_at_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pending_effects_session_status
    ON pending_effects (session_id, status);
CREATE TABLE IF NOT EXISTS pending_resolved_recent (
    call_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT,
    tool TEXT NOT NULL,
    args_digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'resolved',
    disposition TEXT NOT NULL,
    generation INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    begun_at_ms INTEGER NOT NULL,
    resolved_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_head_scope (
    session_id TEXT PRIMARY KEY,
    bound_at_ms INTEGER NOT NULL
);";

/// Shared column list for every pending row read.
///
/// One spelling so single-row and scan reads validate identically: no read
/// path can observe a subset of columns and silently accept what another
/// path would refuse.
const SELECT_COLUMNS: &str = "call_id, session_id, task_id, tool, args_digest, status,
                    disposition, generation, epoch, begun_at_ms, resolved_at_ms";

/// Begin arguments for [`PendingStore::begin`].
///
/// Raw `args` bytes are hashed inside `begin`; callers never precompute the
/// digest, so the stored digest is always exactly the SHA-256 hex over the
/// dispatched bytes.
pub struct PendingBegin<'a> {
    /// Opaque host-minted hex id (non-empty, ASCII hex, bounded).
    pub call_id: &'a str,
    /// Session owning the call.
    pub session_id: &'a WheelSessionId,
    /// Task the call runs under, when task-assigned.
    pub task_id: Option<&'a str>,
    /// Tool contacted.
    pub tool: &'a str,
    /// Raw dispatched args bytes (hashed, never stored).
    pub args: &'a [u8],
    /// Caller-supplied task-generation snapshot at begin.
    pub generation: u64,
    /// Caller-supplied fencing epoch at begin.
    pub epoch: u64,
    /// Caller-supplied begin timestamp, ms since unix epoch.
    pub now_ms: u64,
}

/// Resolve arguments for [`PendingStore::resolve`].
pub struct PendingResolve<'a> {
    /// Call id to close.
    pub call_id: &'a str,
    /// Terminal disposition to record.
    pub disposition: PendingDisposition,
    /// Epoch carried by the resolver (must not predate the row's fence).
    pub claim_epoch: u64,
    /// Task-generation token (must equal the row's fence).
    pub expected_generation: u64,
    /// Caller-supplied resolve timestamp, ms since unix epoch.
    pub now_ms: u64,
}

/// Durable pending-effect log over the shared single-writer connection.
///
/// Unit struct: all state lives in SQLite behind `store`; every method is an
/// associated function taking `&ContentStore`, mirroring the sessions-plane
/// free-function shape while keeping the AI-0199 surface namespaced.
pub struct PendingStore;

impl PendingStore {
    /// Record one contacted tool call as open.
    ///
    /// Validates shapes (call id hex and bounded, tool bounded, task id
    /// bounded when present), refuses out-of-range integers and over-cap
    /// sessions before any write, then checks id novelty in both the live
    /// set and the recent cache and inserts in one transaction on the
    /// single-writer connection. A taken call id reports
    /// [`PendingError::AlreadyExists`] with existing rows untouched (the
    /// cache check keeps a restarted mint from reusing a recently resolved
    /// id; an evicted id may be reused as a new call, see the module docs).
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::InvalidId`] for malformed shapes,
    /// [`PendingError::Storage`] for out-of-range integers or store
    /// failures, [`PendingError::AlreadyExists`] for a taken call id, and
    /// [`PendingError::TooManyOpen`] past
    /// [`MAX_OPEN_PENDING_PER_SESSION`] open rows.
    pub fn begin(
        store: &ContentStore,
        req: PendingBegin<'_>,
    ) -> Result<PendingEntry, PendingError> {
        if !valid_call_id(req.call_id) || !valid_tool(req.tool) || !valid_task_id(req.task_id) {
            return Err(PendingError::InvalidId);
        }
        if req.generation > i64::MAX as u64
            || req.epoch > i64::MAX as u64
            || req.now_ms > i64::MAX as u64
        {
            return Err(PendingError::Storage(INTEGER_RANGE_MSG.to_owned()));
        }
        ensure_table(store)?;
        let digest = ContentHash::compute(req.args);
        let mut guard = store.lock_conn().map_err(map_store_err)?;
        let tx = guard.transaction().map_err(map_sqlite)?;
        let open_count: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pending_effects WHERE session_id = ?1 AND status = 'open'",
                params![req.session_id.as_str()],
                |row| row.get(0),
            )
            .map_err(map_sqlite)?;
        if open_count >= MAX_OPEN_PENDING_PER_SESSION as i64 {
            return Err(PendingError::TooManyOpen {
                limit: MAX_OPEN_PENDING_PER_SESSION,
            });
        }
        let taken: bool = tx
            .query_row(
                "SELECT 1 FROM pending_effects WHERE call_id = ?1",
                params![req.call_id],
                |_| Ok(true),
            )
            .optional()
            .map_err(map_sqlite)?
            .unwrap_or(false);
        let taken_cache: bool = tx
            .query_row(
                "SELECT 1 FROM pending_resolved_recent WHERE call_id = ?1",
                params![req.call_id],
                |_| Ok(true),
            )
            .optional()
            .map_err(map_sqlite)?
            .unwrap_or(false);
        if taken || taken_cache {
            return Err(PendingError::AlreadyExists);
        }
        let task_value: Option<&str> = req.task_id;
        tx.execute(
            "INSERT INTO pending_effects
             (call_id, session_id, task_id, tool, args_digest, status,
              disposition, generation, epoch, begun_at_ms, resolved_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, 'open', NULL, ?6, ?7, ?8, NULL)",
            params![
                req.call_id,
                req.session_id.as_str(),
                task_value,
                req.tool,
                digest.to_hex(),
                req.generation as i64,
                req.epoch as i64,
                req.now_ms as i64
            ],
        )
        .map_err(map_sqlite)?;
        tx.commit().map_err(map_sqlite)?;
        Ok(PendingEntry {
            call_id: req.call_id.to_owned(),
            session_id: req.session_id.clone(),
            task_id: req.task_id.map(str::to_owned),
            tool: req.tool.to_owned(),
            args_digest: digest,
            status: PendingStatus::Open,
            disposition: None,
            generation: req.generation,
            epoch: req.epoch,
            begun_at_ms: req.now_ms,
            resolved_at_ms: None,
        })
    }

    /// Close an open call with a terminal disposition.
    ///
    /// Fences first (stale claim or generation refuses with zero writes, the
    /// entry stays open), then deletes the open row and inserts a snapshot
    /// into the bounded recent cache in the same transaction, evicting
    /// oldest beyond [`MAX_RESOLVED_RECENT`]. Resolving an already resolved
    /// id with the same disposition returns the cached snapshot with no
    /// write; a conflicting disposition refuses with
    /// [`PendingError::Mismatch`]; an evicted or never-seen id reports
    /// [`PendingError::NotFound`]. A legacy `resolved` row still sitting in
    /// the live table (pre-fix database) follows the same idempotency
    /// without a write and stays inert for scans.
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::InvalidId`] for a malformed call id,
    /// [`PendingError::NotFound`] for an unknown (or cache-evicted) one,
    /// [`PendingError::StaleEpoch`] / [`PendingError::StaleGeneration`] for
    /// stale fences, [`PendingError::Mismatch`] for a conflicting
    /// re-resolve, [`PendingError::Corrupt`] for a malformed stored row, and
    /// [`PendingError::Storage`] for out-of-range timestamps or store
    /// failures.
    pub fn resolve(
        store: &ContentStore,
        req: PendingResolve<'_>,
    ) -> Result<PendingEntry, PendingError> {
        if !valid_call_id(req.call_id) {
            return Err(PendingError::InvalidId);
        }
        if req.now_ms > i64::MAX as u64 {
            return Err(PendingError::Storage(INTEGER_RANGE_MSG.to_owned()));
        }
        ensure_table(store)?;
        let mut guard = store.lock_conn().map_err(map_store_err)?;
        let tx = guard.transaction().map_err(map_sqlite)?;
        if let Some(entry) = read_entry(&tx, req.call_id)? {
            if entry.status == PendingStatus::Resolved {
                let stored = entry.disposition.unwrap_or(PendingDisposition::Abandoned);
                if stored == req.disposition {
                    return Ok(entry);
                }
                return Err(PendingError::Mismatch {
                    stored,
                    supplied: req.disposition,
                });
            }
            if req.claim_epoch < entry.epoch {
                return Err(PendingError::StaleEpoch {
                    claim_epoch: req.claim_epoch,
                    current_epoch: entry.epoch,
                });
            }
            if req.expected_generation != entry.generation {
                return Err(PendingError::StaleGeneration {
                    expected: entry.generation,
                    found: req.expected_generation,
                });
            }
            let resolved = PendingEntry {
                status: PendingStatus::Resolved,
                disposition: Some(req.disposition),
                resolved_at_ms: Some(req.now_ms),
                ..entry
            };
            tx.execute(
                "DELETE FROM pending_effects WHERE call_id = ?1 AND status = 'open'",
                params![req.call_id],
            )
            .map_err(map_sqlite)?;
            insert_resolved_cache(&tx, &resolved)?;
            tx.commit().map_err(map_sqlite)?;
            return Ok(resolved);
        }
        if let Some(cached) = read_cached(&tx, req.call_id)? {
            let stored = cached.disposition.unwrap_or(PendingDisposition::Abandoned);
            if stored == req.disposition {
                return Ok(cached);
            }
            return Err(PendingError::Mismatch {
                stored,
                supplied: req.disposition,
            });
        }
        Err(PendingError::NotFound)
    }

    /// List the open entries of one session, oldest-first.
    ///
    /// Deterministic order (`begun_at_ms`, then `call_id`): same-ms begins
    /// cannot reorder across reads. SQL filters `status = 'open'` (via the
    /// `(session_id, status)` index), so cached snapshots and legacy
    /// resolved rows are never read for validation and never widen scan
    /// cost. A row carrying neither `open` nor `resolved` status fails the
    /// whole read as [`PendingError::Corrupt`] with rows preserved; legacy
    /// `resolved` rows in the live table are inert and skipped. Open-row
    /// validation is fail-closed: any malformed open row fails the whole
    /// read, nothing is skipped, nothing auto-dropped.
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::Corrupt`] for any malformed open row or any
    /// bogus-status row of the session, and [`PendingError::Storage`] for
    /// store failures.
    pub fn list_open(
        store: &ContentStore,
        session_id: &WheelSessionId,
    ) -> Result<Vec<PendingEntry>, PendingError> {
        ensure_table(store)?;
        let (rows, bogus_count): (Vec<StoredRow>, i64) = {
            let guard = store.lock_conn().map_err(map_store_err)?;
            let mut stmt = guard
                .prepare(&format!(
                    "SELECT {SELECT_COLUMNS}
                     FROM pending_effects
                     WHERE session_id = ?1 AND status = 'open'
                     ORDER BY begun_at_ms ASC, call_id ASC"
                ))
                .map_err(map_sqlite)?;
            let mapped = stmt
                .query_map(params![session_id.as_str()], StoredRow::from_row)
                .map_err(map_sqlite)?;
            let mut out = Vec::new();
            for item in mapped {
                out.push(item.map_err(map_sqlite)?);
            }
            let bogus: i64 = guard
                .query_row(
                    "SELECT COUNT(*) FROM pending_effects
                     WHERE session_id = ?1 AND status NOT IN ('open', 'resolved')",
                    params![session_id.as_str()],
                    |row| row.get(0),
                )
                .map_err(map_sqlite)?;
            (out, bogus)
        };
        let mut open = Vec::new();
        for entry in rows.into_iter().map(StoredRow::into_entry) {
            open.push(entry?);
        }
        if bogus_count > 0 {
            return Err(PendingError::Corrupt);
        }
        Ok(open)
    }

    /// Read one entry by call id. Live opens first, then the recent cache.
    /// Evicted or never-seen ids return `None`; malformed stored rows fail
    /// closed with [`PendingError::Corrupt`].
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::InvalidId`] for a malformed call id,
    /// [`PendingError::Corrupt`] for a malformed stored row, and
    /// [`PendingError::Storage`] for store failures.
    pub fn get(store: &ContentStore, call_id: &str) -> Result<Option<PendingEntry>, PendingError> {
        if !valid_call_id(call_id) {
            return Err(PendingError::InvalidId);
        }
        ensure_table(store)?;
        let guard = store.lock_conn().map_err(map_store_err)?;
        if let Some(entry) = read_entry(&guard, call_id)? {
            return Ok(Some(entry));
        }
        read_cached(&guard, call_id)
    }

    /// Collect the blob hashes pinned by open entries file-wide.
    ///
    /// Each open row's `args_digest` is a content hash: when the host stored
    /// the dispatched args (or any payload under that hash) as a blob, the
    /// open entry keeps it alive. Digests naming no blob row are harmless
    /// (the GC set difference drops them). The all-zero hash is skipped as a
    /// root, reusing the tombstone convention (it has no payload). SQL
    /// filters `status = 'open'`, so cached snapshots and legacy resolved
    /// rows are never read for validation and never widen scan cost; only
    /// malformed open rows fail the whole read as [`PendingError::Corrupt`]
    /// (fail closed, rows preserved). A bogus-status COUNT guard matches
    /// [`PendingStore::list_open`]: any row carrying neither `open` nor
    /// `resolved` fails the whole read as [`PendingError::Corrupt`].
    /// Sorted and deduplicated for deterministic GC plans.
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::Corrupt`] for any malformed open row or any
    /// bogus-status row, and [`PendingError::Storage`] for store failures.
    pub fn open_blob_roots(store: &ContentStore) -> Result<Vec<ContentHash>, PendingError> {
        ensure_table(store)?;
        let (rows, bogus_count): (Vec<StoredRow>, i64) = {
            let guard = store.lock_conn().map_err(map_store_err)?;
            let mut stmt = guard
                .prepare(&format!(
                    "SELECT {SELECT_COLUMNS} FROM pending_effects WHERE status = 'open'"
                ))
                .map_err(map_sqlite)?;
            let mapped = stmt
                .query_map([], StoredRow::from_row)
                .map_err(map_sqlite)?;
            let mut out = Vec::new();
            for item in mapped {
                out.push(item.map_err(map_sqlite)?);
            }
            let bogus: i64 = guard
                .query_row(
                    "SELECT COUNT(*) FROM pending_effects
                     WHERE status NOT IN ('open', 'resolved')",
                    [],
                    |row| row.get(0),
                )
                .map_err(map_sqlite)?;
            (out, bogus)
        };
        let zero = ContentHash::from_bytes([0u8; 32]);
        let mut roots = Vec::new();
        for entry in rows.into_iter().map(StoredRow::into_entry) {
            let entry = entry?;
            if entry.status == PendingStatus::Open && entry.args_digest != zero {
                roots.push(entry.args_digest);
            }
        }
        if bogus_count > 0 {
            return Err(PendingError::Corrupt);
        }
        roots.sort();
        roots.dedup();
        Ok(roots)
    }

    /// Bind one session to the explicit HEAD resume scope (AI-0200).
    ///
    /// The marker is stored, never inferred: only sessions listed by
    /// [`PendingStore::list_head_sessions`] resolve on a direct-`HEAD`
    /// resume. The marker is additive to the session branch binding (a
    /// session may be visible on both its branch and `HEAD`); it never moves
    /// the session row and never follows tip equality. `now_ms` is the
    /// caller-supplied bind timestamp (caller clocks only). A duplicate bind
    /// refuses with [`PendingError::AlreadyExists`] and zero writes, even at
    /// the marker bound. Past [`MAX_HEAD_MARKERS`] distinct markers a new
    /// session refuses with [`PendingError::TooManyOpen`] (the admission
    /// bound travels in `limit`) and zero writes, so a long-lived store can
    /// never make a direct-`HEAD` resume fan out without bound.
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::Storage`] for an out-of-range timestamp or a
    /// store failure, [`PendingError::AlreadyExists`] for an already marked
    /// session, [`PendingError::TooManyOpen`] past [`MAX_HEAD_MARKERS`]
    /// distinct markers, and [`PendingError::Corrupt`] for a corrupt
    /// database.
    pub fn bind_head(
        store: &ContentStore,
        session_id: &WheelSessionId,
        now_ms: u64,
    ) -> Result<(), PendingError> {
        if now_ms > i64::MAX as u64 {
            return Err(PendingError::Storage(INTEGER_RANGE_MSG.to_owned()));
        }
        ensure_table(store)?;
        let guard = store.lock_conn().map_err(map_store_err)?;
        let marked: bool = guard
            .query_row(
                "SELECT 1 FROM pending_head_scope WHERE session_id = ?1",
                params![session_id.as_str()],
                |_| Ok(true),
            )
            .optional()
            .map_err(map_sqlite)?
            .unwrap_or(false);
        if marked {
            return Err(PendingError::AlreadyExists);
        }
        let marker_count: i64 = guard
            .query_row("SELECT COUNT(*) FROM pending_head_scope", [], |row| {
                row.get(0)
            })
            .map_err(map_sqlite)?;
        if marker_count >= MAX_HEAD_MARKERS as i64 {
            return Err(PendingError::TooManyOpen {
                limit: MAX_HEAD_MARKERS,
            });
        }
        guard
            .execute(
                "INSERT INTO pending_head_scope (session_id, bound_at_ms) VALUES (?1, ?2)",
                params![session_id.as_str(), now_ms as i64],
            )
            .map_err(map_sqlite)?;
        Ok(())
    }

    /// Remove one session from the explicit HEAD resume scope.
    ///
    /// Deleting the marker makes the session invisible to direct-`HEAD`
    /// resumes again; branch resumes keyed on the session branch binding are
    /// unaffected. A missing marker reports [`PendingError::NotFound`] with
    /// zero writes.
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::NotFound`] for an unmarked session,
    /// [`PendingError::Corrupt`] for a corrupt database, and
    /// [`PendingError::Storage`] for store failures.
    pub fn unbind_head(
        store: &ContentStore,
        session_id: &WheelSessionId,
    ) -> Result<(), PendingError> {
        ensure_table(store)?;
        let guard = store.lock_conn().map_err(map_store_err)?;
        let affected = guard
            .execute(
                "DELETE FROM pending_head_scope WHERE session_id = ?1",
                params![session_id.as_str()],
            )
            .map_err(map_sqlite)?;
        if affected == 0 {
            return Err(PendingError::NotFound);
        }
        Ok(())
    }

    /// List the sessions carrying the explicit HEAD marker, ascending.
    ///
    /// Deterministic order (`session_id ASC`): direct-`HEAD` resumes union
    /// each marked session oldest-first in this order. At most
    /// [`MAX_HEAD_MARKERS`] rows are read (SQL `LIMIT`, matching the
    /// admission bound, so the resume fan-out stays bounded even if rows
    /// were planted outside admission). Fail-closed: any malformed marker
    /// row (bad session-id shape, negative or non-integer timestamp) fails
    /// the whole read as [`PendingError::Corrupt`] with rows preserved. A
    /// fresh database lists zero markers rather than erroring.
    ///
    /// # Errors
    ///
    /// Returns [`PendingError::Corrupt`] for any malformed marker row and
    /// [`PendingError::Storage`] for store failures.
    pub fn list_head_sessions(store: &ContentStore) -> Result<Vec<WheelSessionId>, PendingError> {
        ensure_table(store)?;
        let pairs: Vec<(String, i64)> = {
            let guard = store.lock_conn().map_err(map_store_err)?;
            let mut stmt = guard
                .prepare(
                    "SELECT session_id, bound_at_ms FROM pending_head_scope
                     ORDER BY session_id ASC LIMIT ?1",
                )
                .map_err(map_sqlite)?;
            let mapped = stmt
                .query_map(params![MAX_HEAD_MARKERS as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(map_sqlite)?;
            let mut out = Vec::new();
            for item in mapped {
                // Row-decode failures mean a malformed marker row (e.g. a
                // non-integer timestamp planted via raw SQL), which the
                // contract reports as `Corrupt`; query-execution failures
                // stay `Storage` via `map_sqlite`.
                out.push(item.map_err(|err| match err {
                    rusqlite::Error::FromSqlConversionFailure(..)
                    | rusqlite::Error::IntegralValueOutOfRange(..)
                    | rusqlite::Error::Utf8Error(..)
                    | rusqlite::Error::NulError(_)
                    | rusqlite::Error::InvalidColumnType(..) => PendingError::Corrupt,
                    _ => map_sqlite(err),
                })?);
            }
            out
        };
        let mut result = Vec::with_capacity(pairs.len());
        for (id_raw, bound_at_ms) in pairs {
            let session_id = WheelSessionId::parse(&id_raw).map_err(|_| PendingError::Corrupt)?;
            if bound_at_ms < 0 {
                return Err(PendingError::Corrupt);
            }
            result.push(session_id);
        }
        Ok(result)
    }
}

/// Ensure the pending tables and the resume-path index exist (idempotent).
fn ensure_table(store: &ContentStore) -> Result<(), PendingError> {
    let guard = store.lock_conn().map_err(map_store_err)?;
    guard.execute_batch(PENDING_SCHEMA).map_err(map_sqlite)?;
    Ok(())
}

/// Whether a call id meets the store shape (non-empty ASCII hex, bounded).
fn valid_call_id(raw: &str) -> bool {
    !raw.is_empty()
        && raw.len() <= MAX_PENDING_CALL_ID
        && raw.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether a tool name meets the store shape (non-empty, bounded).
///
/// Charset enforcement stays at the runtime boundary (`validate_tool_name`);
/// the log records whatever contacted name the host supplies, bounded.
fn valid_tool(raw: &str) -> bool {
    !raw.is_empty() && raw.len() <= MAX_PENDING_TOOL_NAME
}

/// Whether an optional task id meets the store shape when present.
fn valid_task_id(raw: Option<&str>) -> bool {
    match raw {
        None => true,
        Some(id) => !id.is_empty() && id.len() <= MAX_TASK_ID_BYTES,
    }
}

/// One raw table row awaiting validation.
struct StoredRow {
    call_id: String,
    session_id: String,
    task_id: Option<String>,
    tool: String,
    args_digest: String,
    status: String,
    disposition: Option<String>,
    generation: i64,
    epoch: i64,
    begun_at_ms: i64,
    resolved_at_ms: Option<i64>,
}

impl StoredRow {
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            call_id: row.get(0)?,
            session_id: row.get(1)?,
            task_id: row.get(2)?,
            tool: row.get(3)?,
            args_digest: row.get(4)?,
            status: row.get(5)?,
            disposition: row.get(6)?,
            generation: row.get(7)?,
            epoch: row.get(8)?,
            begun_at_ms: row.get(9)?,
            resolved_at_ms: row.get(10)?,
        })
    }

    /// Validate a raw row into a typed entry, failing closed as `Corrupt`.
    ///
    /// Open rows must carry no disposition state; resolved rows must carry
    /// both a known disposition and a timestamp. Any violation fails the
    /// whole read -- poison is never auto-dropped.
    fn into_entry(self) -> Result<PendingEntry, PendingError> {
        if !valid_call_id(&self.call_id) {
            return Err(PendingError::Corrupt);
        }
        let session_id =
            WheelSessionId::parse(&self.session_id).map_err(|_| PendingError::Corrupt)?;
        if !valid_tool(&self.tool) || !valid_task_id(self.task_id.as_deref()) {
            return Err(PendingError::Corrupt);
        }
        let args_digest =
            ContentHash::from_hex(&self.args_digest).map_err(|_| PendingError::Corrupt)?;
        let status = PendingStatus::parse(&self.status).ok_or(PendingError::Corrupt)?;
        let disposition = match self.disposition.as_deref() {
            None => None,
            Some(raw) => Some(PendingDisposition::parse(raw).ok_or(PendingError::Corrupt)?),
        };
        match (status, disposition, self.resolved_at_ms) {
            (PendingStatus::Open, None, None) => {}
            (PendingStatus::Resolved, Some(_), Some(_)) => {}
            _ => return Err(PendingError::Corrupt),
        }
        if self.generation < 0 || self.epoch < 0 || self.begun_at_ms < 0 {
            return Err(PendingError::Corrupt);
        }
        if self.resolved_at_ms.is_some_and(|at| at < 0) {
            return Err(PendingError::Corrupt);
        }
        Ok(PendingEntry {
            call_id: self.call_id,
            session_id,
            task_id: self.task_id,
            tool: self.tool,
            args_digest,
            status,
            disposition,
            generation: self.generation as u64,
            epoch: self.epoch as u64,
            begun_at_ms: self.begun_at_ms as u64,
            resolved_at_ms: self.resolved_at_ms.map(|at| at as u64),
        })
    }
}

/// Read one entry by call id inside a held connection scope.
///
/// Takes `&Connection` so both plain guards and in-flight transactions
/// coerce through deref (same shape as the sessions-plane `read_binding`).
fn read_entry(
    conn: &rusqlite::Connection,
    call_id: &str,
) -> Result<Option<PendingEntry>, PendingError> {
    let row: Option<StoredRow> = conn
        .query_row(
            &format!(
                "SELECT {SELECT_COLUMNS}
             FROM pending_effects WHERE call_id = ?1"
            ),
            params![call_id],
            StoredRow::from_row,
        )
        .optional()
        .map_err(map_sqlite)?;
    row.map(StoredRow::into_entry).transpose()
}

/// Read one cached resolved snapshot by call id inside a held scope.
///
/// Same validation as the live read: a malformed cached row fails closed as
/// `Corrupt` with rows preserved. Missing ids return `None`.
fn read_cached(
    conn: &rusqlite::Connection,
    call_id: &str,
) -> Result<Option<PendingEntry>, PendingError> {
    let row: Option<StoredRow> = conn
        .query_row(
            &format!(
                "SELECT {SELECT_COLUMNS}
             FROM pending_resolved_recent WHERE call_id = ?1"
            ),
            params![call_id],
            StoredRow::from_row,
        )
        .optional()
        .map_err(map_sqlite)?;
    row.map(StoredRow::into_entry).transpose()
}

/// Insert a resolved snapshot into the bounded cache and evict oldest.
///
/// Runs inside the caller's resolve transaction: the snapshot carries the
/// full entry (so a cache hit returns the identical `PendingEntry`), then
/// every row beyond the newest [`MAX_RESOLVED_RECENT`] (ordered by
/// `resolved_at_ms`, then `call_id`) is deleted. The bound is inlined as a
/// constant, never caller text.
fn insert_resolved_cache(
    tx: &rusqlite::Transaction<'_>,
    resolved: &PendingEntry,
) -> Result<(), PendingError> {
    let disposition = resolved
        .disposition
        .unwrap_or(PendingDisposition::Abandoned);
    let resolved_at = resolved.resolved_at_ms.unwrap_or(resolved.begun_at_ms);
    tx.execute(
        "INSERT INTO pending_resolved_recent
         (call_id, session_id, task_id, tool, args_digest, status,
          disposition, generation, epoch, begun_at_ms, resolved_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, 'resolved', ?6, ?7, ?8, ?9, ?10)",
        params![
            resolved.call_id,
            resolved.session_id.as_str(),
            resolved.task_id.as_deref(),
            resolved.tool,
            resolved.args_digest.to_hex(),
            disposition.as_str(),
            resolved.generation as i64,
            resolved.epoch as i64,
            resolved.begun_at_ms as i64,
            resolved_at as i64,
        ],
    )
    .map_err(map_sqlite)?;
    tx.execute(
        &format!(
            "DELETE FROM pending_resolved_recent WHERE call_id NOT IN
             (SELECT call_id FROM pending_resolved_recent
              ORDER BY resolved_at_ms DESC, call_id DESC LIMIT {MAX_RESOLVED_RECENT})"
        ),
        [],
    )
    .map_err(map_sqlite)?;
    Ok(())
}

/// Map a raw SQLite error to a typed pending error (static text only).
fn map_sqlite(err: rusqlite::Error) -> PendingError {
    if let rusqlite::Error::SqliteFailure(failure, _) = &err {
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        ) {
            return PendingError::Storage(WRITER_BUSY_MSG.to_owned());
        }
        if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return PendingError::Corrupt;
        }
        if matches!(failure.code, rusqlite::ErrorCode::ConstraintViolation) {
            return PendingError::AlreadyExists;
        }
    }
    PendingError::Storage(err.to_string())
}

/// Map a store error to a typed pending error, dropping untrusted payloads.
fn map_store_err(err: ContentStoreError) -> PendingError {
    match err {
        ContentStoreError::Corrupt { .. }
        | ContentStoreError::CorruptData { .. }
        | ContentStoreError::CorruptCheckpoint { .. }
        | ContentStoreError::InvalidHash(_)
        | ContentStoreError::MissingParent(_)
        | ContentStoreError::MissingTree(_)
        | ContentStoreError::MissingTarget(_) => PendingError::Corrupt,
        ContentStoreError::InvalidRefName(_)
        | ContentStoreError::EmptyField(_)
        | ContentStoreError::OversizedField { .. } => PendingError::InvalidId,
        ContentStoreError::WriterBusy => PendingError::Storage(WRITER_BUSY_MSG.to_owned()),
        ContentStoreError::Sqlite(err) => map_sqlite(err),
        ContentStoreError::Json(err) => PendingError::Storage(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> ContentStore {
        ContentStore::open_in_memory().expect("open")
    }

    fn test_session() -> WheelSessionId {
        WheelSessionId::parse("sess-pending-1").expect("valid test session")
    }

    fn begin_req<'a>(
        call_id: &'a str,
        session_id: &'a WheelSessionId,
        tool: &'a str,
        args: &'a [u8],
    ) -> PendingBegin<'a> {
        PendingBegin {
            call_id,
            session_id,
            task_id: None,
            tool,
            args,
            generation: 0,
            epoch: 1,
            now_ms: 1000,
        }
    }

    fn resolve_req<'a>(call_id: &'a str, disposition: PendingDisposition) -> PendingResolve<'a> {
        PendingResolve {
            call_id,
            disposition,
            claim_epoch: 1,
            expected_generation: 0,
            now_ms: 2000,
        }
    }

    #[test]
    fn begin_then_get_roundtrip_is_open_without_disposition() {
        let store = test_store();
        let session = test_session();
        let entry = PendingStore::begin(&store, begin_req("ab12", &session, "read_file", b"{}"))
            .expect("begin");
        assert_eq!(entry.status, PendingStatus::Open);
        assert_eq!(entry.disposition, None);
        assert_eq!(entry.resolved_at_ms, None);
        assert_eq!(entry.args_digest, ContentHash::compute(b"{}"));
        let read = PendingStore::get(&store, "ab12")
            .expect("get")
            .expect("present");
        assert_eq!(read, entry);
        let open = PendingStore::list_open(&store, &session).expect("list");
        assert_eq!(open, vec![entry]);
    }

    #[test]
    fn duplicate_call_id_is_already_exists_and_keeps_first_row() {
        let store = test_store();
        let session = test_session();
        PendingStore::begin(&store, begin_req("cd34", &session, "read_file", b"a"))
            .expect("first begin");
        assert_eq!(
            PendingStore::begin(&store, begin_req("cd34", &session, "write_file", b"b")),
            Err(PendingError::AlreadyExists)
        );
        let kept = PendingStore::get(&store, "cd34")
            .expect("get")
            .expect("present");
        assert_eq!(kept.tool, "read_file");
        assert_eq!(kept.args_digest, ContentHash::compute(b"a"));
    }

    #[test]
    fn malformed_shapes_refuse_as_invalid_id() {
        let store = test_store();
        let session = test_session();
        // Bad call ids: empty, non-hex, oversize.
        for bad in [
            "",
            "not-hex!!",
            "sess:1",
            &"a".repeat(MAX_PENDING_CALL_ID + 1),
        ] {
            assert_eq!(
                PendingStore::begin(&store, begin_req(bad, &session, "read_file", b"{}")),
                Err(PendingError::InvalidId),
                "call id {bad:?} must refuse"
            );
        }
        // Bad tools: empty, oversize.
        for bad in ["", &"t".repeat(MAX_PENDING_TOOL_NAME + 1)] {
            assert_eq!(
                PendingStore::begin(&store, begin_req("ab12", &session, bad, b"{}")),
                Err(PendingError::InvalidId),
                "tool {bad:?} must refuse"
            );
        }
        // Bad task ids: empty, oversize.
        let mut bad_task = begin_req("ab12", &session, "read_file", b"{}");
        bad_task.task_id = Some("");
        assert_eq!(
            PendingStore::begin(&store, bad_task),
            Err(PendingError::InvalidId)
        );
        let mut bad_task = begin_req("ab12", &session, "read_file", b"{}");
        let oversize = "t".repeat(MAX_TASK_ID_BYTES + 1);
        bad_task.task_id = Some(&oversize);
        assert_eq!(
            PendingStore::begin(&store, bad_task),
            Err(PendingError::InvalidId)
        );
        // Nothing was written by any refusal.
        assert!(
            PendingStore::list_open(&store, &session)
                .expect("list")
                .is_empty()
        );
        assert_eq!(PendingStore::get(&store, "ab12").expect("get"), None);
        // Malformed get ids refuse the same way.
        assert_eq!(
            PendingStore::get(&store, "no hex!!"),
            Err(PendingError::InvalidId)
        );
        assert_eq!(
            PendingStore::resolve(&store, resolve_req("no hex!!", PendingDisposition::Success)),
            Err(PendingError::InvalidId)
        );
    }

    #[test]
    fn out_of_range_integers_refused_with_zero_row_writes() {
        let store = test_store();
        let session = test_session();
        let range_err = PendingError::Storage(INTEGER_RANGE_MSG.to_owned());
        for (generation, epoch, now_ms) in
            [(u64::MAX, 1, 1000), (0, u64::MAX, 1000), (0, 1, u64::MAX)]
        {
            let mut req = begin_req("ab12", &session, "read_file", b"{}");
            req.generation = generation;
            req.epoch = epoch;
            req.now_ms = now_ms;
            assert_eq!(
                PendingStore::begin(&store, req),
                Err(range_err.clone()),
                "out-of-range begin must refuse"
            );
            assert_eq!(PendingStore::get(&store, "ab12").expect("get"), None);
        }
        assert!(
            PendingStore::list_open(&store, &session)
                .expect("list")
                .is_empty()
        );
        // A valid row begins, then an out-of-range resolve moves nothing.
        PendingStore::begin(&store, begin_req("ab12", &session, "read_file", b"{}"))
            .expect("begin");
        let mut req = resolve_req("ab12", PendingDisposition::Success);
        req.now_ms = u64::MAX;
        assert_eq!(
            PendingStore::resolve(&store, req),
            Err(range_err),
            "out-of-range resolve must refuse"
        );
        let kept = PendingStore::get(&store, "ab12")
            .expect("get")
            .expect("present");
        assert_eq!(kept.status, PendingStatus::Open);
    }

    #[test]
    fn resolve_marks_resolved_and_double_resolve_same_disposition_is_noop() {
        let store = test_store();
        let session = test_session();
        PendingStore::begin(&store, begin_req("ab12", &session, "read_file", b"{}"))
            .expect("begin");
        let resolved =
            PendingStore::resolve(&store, resolve_req("ab12", PendingDisposition::Success))
                .expect("resolve");
        assert_eq!(resolved.status, PendingStatus::Resolved);
        assert_eq!(resolved.disposition, Some(PendingDisposition::Success));
        assert_eq!(resolved.resolved_at_ms, Some(2000));
        // Resolved rows leave the live set (deleted on resolve) but stay
        // readable via the bounded recent cache.
        assert!(
            PendingStore::list_open(&store, &session)
                .expect("list")
                .is_empty()
        );
        // Same-disposition re-resolve hits the cache as an idempotent no-op:
        // the cached snapshot returns unchanged even with a later timestamp.
        let mut again = resolve_req("ab12", PendingDisposition::Success);
        again.now_ms = 9999;
        let second = PendingStore::resolve(&store, again).expect("idempotent resolve");
        assert_eq!(second, resolved);
        // A conflicting disposition refuses with zero writes.
        assert_eq!(
            PendingStore::resolve(&store, resolve_req("ab12", PendingDisposition::Failed)),
            Err(PendingError::Mismatch {
                stored: PendingDisposition::Success,
                supplied: PendingDisposition::Failed,
            })
        );
        let kept = PendingStore::get(&store, "ab12")
            .expect("get")
            .expect("present");
        assert_eq!(kept, resolved);
        // The live row is gone: only the cache holds it.
        let live_count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row(
                    "SELECT COUNT(*) FROM pending_effects WHERE call_id = 'ab12'",
                    [],
                    |row| row.get(0),
                )
                .expect("count live")
        };
        assert_eq!(live_count, 0);
        let cache_count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row(
                    "SELECT COUNT(*) FROM pending_resolved_recent WHERE call_id = 'ab12'",
                    [],
                    |row| row.get(0),
                )
                .expect("count cache")
        };
        assert_eq!(cache_count, 1);
    }

    #[test]
    fn resolved_cache_evicts_oldest_and_reports_not_found() {
        let store = test_store();
        let session = test_session();
        // Fill the cache one past its bound with strictly increasing resolve
        // timestamps so eviction order is deterministic (oldest first).
        let total = MAX_RESOLVED_RECENT + 1;
        for seq in 0..total {
            let call_id = format!("{seq:04x}");
            let mut begun = begin_req(&call_id, &session, "read_file", b"{}");
            begun.now_ms = 1000 + seq as u64;
            PendingStore::begin(&store, begun).expect("begin within cap");
            let mut req = resolve_req(&call_id, PendingDisposition::Success);
            req.now_ms = 2000 + seq as u64;
            PendingStore::resolve(&store, req).expect("resolve into cache");
        }
        // The cache holds exactly its bound; the live set is empty.
        let cache_count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row("SELECT COUNT(*) FROM pending_resolved_recent", [], |row| {
                    row.get(0)
                })
                .expect("count cache")
        };
        assert_eq!(cache_count, MAX_RESOLVED_RECENT as i64);
        assert!(
            PendingStore::list_open(&store, &session)
                .expect("list")
                .is_empty()
        );
        // The oldest id was evicted: same-disposition re-resolve reports
        // NotFound (not Ok), and get reports None. This is the judged
        // trade-off: bounded recall means evicted retries are unknown, never
        // proof the effect did not happen.
        let oldest = format!("{:04x}", 0);
        assert_eq!(
            PendingStore::resolve(&store, resolve_req(&oldest, PendingDisposition::Success)),
            Err(PendingError::NotFound)
        );
        assert_eq!(PendingStore::get(&store, &oldest).expect("get"), None);
        // The newest id is still cached: same-disposition returns Ok with no
        // write, conflicting disposition refuses as Mismatch, get returns it.
        let newest = format!("{:04x}", total - 1);
        let mut again = resolve_req(&newest, PendingDisposition::Success);
        again.now_ms = 9999;
        let cached = PendingStore::get(&store, &newest)
            .expect("get")
            .expect("newest cached");
        let second = PendingStore::resolve(&store, again).expect("cached idempotent");
        assert_eq!(second, cached);
        assert_eq!(
            PendingStore::resolve(&store, resolve_req(&newest, PendingDisposition::Failed)),
            Err(PendingError::Mismatch {
                stored: PendingDisposition::Success,
                supplied: PendingDisposition::Failed,
            })
        );
        // Begin refuses to reuse a cached id (AlreadyExists) so the mint
        // retry advances; an evicted id may be reused as a brand-new call.
        assert_eq!(
            PendingStore::begin(&store, begin_req(&newest, &session, "read_file", b"{}")),
            Err(PendingError::AlreadyExists)
        );
        PendingStore::begin(&store, begin_req(&oldest, &session, "read_file", b"{}"))
            .expect("evicted id reusable as new call");
        assert_eq!(
            PendingStore::list_open(&store, &session)
                .expect("list")
                .len(),
            1
        );
    }

    #[test]
    fn resolve_unknown_id_is_not_found() {
        let store = test_store();
        assert_eq!(
            PendingStore::resolve(&store, resolve_req("ab12", PendingDisposition::Success)),
            Err(PendingError::NotFound)
        );
    }

    #[test]
    fn stale_fence_resolve_refused_zero_write_entry_stays_open() {
        let store = test_store();
        let session = test_session();
        let mut begun = begin_req("ab12", &session, "read_file", b"{}");
        begun.generation = 3;
        begun.epoch = 2;
        PendingStore::begin(&store, begun).expect("begin");
        // An older epoch refuses; the entry stays open.
        let mut stale_epoch = resolve_req("ab12", PendingDisposition::Success);
        stale_epoch.claim_epoch = 1;
        stale_epoch.expected_generation = 3;
        assert_eq!(
            PendingStore::resolve(&store, stale_epoch),
            Err(PendingError::StaleEpoch {
                claim_epoch: 1,
                current_epoch: 2
            })
        );
        // A moved generation refuses; the entry stays open.
        let mut stale_generation = resolve_req("ab12", PendingDisposition::Success);
        stale_generation.claim_epoch = 2;
        stale_generation.expected_generation = 4;
        assert_eq!(
            PendingStore::resolve(&store, stale_generation),
            Err(PendingError::StaleGeneration {
                expected: 3,
                found: 4
            })
        );
        let open = PendingStore::list_open(&store, &session).expect("list");
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].status, PendingStatus::Open);
        // The current fence still admits.
        let mut fresh = resolve_req("ab12", PendingDisposition::Success);
        fresh.claim_epoch = 2;
        fresh.expected_generation = 3;
        let resolved = PendingStore::resolve(&store, fresh).expect("fresh resolve");
        assert_eq!(resolved.status, PendingStatus::Resolved);
    }

    #[test]
    fn thirty_third_open_begin_refuses_and_list_stays_bounded() {
        let store = test_store();
        let session = test_session();
        for seq in 0..MAX_OPEN_PENDING_PER_SESSION {
            let call_id = format!("{seq:04x}");
            PendingStore::begin(&store, begin_req(&call_id, &session, "read_file", b"{}"))
                .expect("begin within cap");
        }
        assert_eq!(
            PendingStore::begin(&store, begin_req("ffff", &session, "read_file", b"{}")),
            Err(PendingError::TooManyOpen {
                limit: MAX_OPEN_PENDING_PER_SESSION
            })
        );
        let open = PendingStore::list_open(&store, &session).expect("list");
        assert_eq!(open.len(), MAX_OPEN_PENDING_PER_SESSION);
        // The cap counts open rows only: resolving one admits the next begin.
        let first = open[0].call_id.clone();
        PendingStore::resolve(&store, resolve_req(&first, PendingDisposition::Success))
            .expect("resolve one");
        PendingStore::begin(&store, begin_req("ffff", &session, "read_file", b"{}"))
            .expect("begin after resolve");
        assert_eq!(
            PendingStore::list_open(&store, &session)
                .expect("list")
                .len(),
            MAX_OPEN_PENDING_PER_SESSION
        );
    }

    #[test]
    fn poisoned_row_fails_whole_read_with_rows_preserved() {
        let store = test_store();
        let session = test_session();
        PendingStore::begin(&store, begin_req("ab12", &session, "read_file", b"{}"))
            .expect("begin");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE pending_effects SET status = 'bogus' WHERE call_id = 'ab12'",
                    [],
                )
                .expect("plant bad status");
        }
        assert_eq!(
            PendingStore::get(&store, "ab12"),
            Err(PendingError::Corrupt)
        );
        assert_eq!(
            PendingStore::list_open(&store, &session),
            Err(PendingError::Corrupt)
        );
        assert_eq!(
            PendingStore::resolve(&store, resolve_req("ab12", PendingDisposition::Success)),
            Err(PendingError::Corrupt)
        );
        // Rows are preserved: the poisoned row still counts exactly one.
        let count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row("SELECT COUNT(*) FROM pending_effects", [], |row| row.get(0))
                .expect("count")
        };
        assert_eq!(count, 1);
        // A negative-epoch poison fails the same way.
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE pending_effects SET status = 'open', epoch = -1 WHERE call_id = 'ab12'",
                    [],
                )
                .expect("plant negative epoch");
        }
        assert_eq!(
            PendingStore::list_open(&store, &session),
            Err(PendingError::Corrupt)
        );
    }

    #[test]
    fn open_blob_roots_pins_digests_skips_zero_and_fails_on_poison() {
        let store = test_store();
        let session = test_session();
        PendingStore::begin(&store, begin_req("aa01", &session, "read_file", b"alpha"))
            .expect("begin one");
        PendingStore::begin(&store, begin_req("aa02", &session, "read_file", b"alpha"))
            .expect("begin two, same digest");
        let mut roots = PendingStore::open_blob_roots(&store).expect("roots");
        roots.sort();
        assert_eq!(roots, vec![ContentHash::compute(b"alpha")]);
        // Resolving one entry keeps the shared digest pinned; resolving both
        // releases it.
        PendingStore::resolve(&store, resolve_req("aa01", PendingDisposition::Success))
            .expect("resolve one");
        assert_eq!(
            PendingStore::open_blob_roots(&store).expect("roots"),
            vec![ContentHash::compute(b"alpha")]
        );
        PendingStore::resolve(&store, resolve_req("aa02", PendingDisposition::Success))
            .expect("resolve two");
        assert!(
            PendingStore::open_blob_roots(&store)
                .expect("roots")
                .is_empty()
        );
        // A poisoned digest fails the whole root read.
        PendingStore::begin(&store, begin_req("aa03", &session, "read_file", b"beta"))
            .expect("begin three");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE pending_effects SET args_digest = 'not-hex' WHERE call_id = 'aa03'",
                    [],
                )
                .expect("plant bad digest");
        }
        assert_eq!(
            PendingStore::open_blob_roots(&store),
            Err(PendingError::Corrupt)
        );
    }

    #[test]
    fn head_scope_bind_list_unbind_is_explicit_and_ordered() {
        let store = test_store();
        assert!(
            PendingStore::list_head_sessions(&store)
                .expect("fresh lists zero")
                .is_empty()
        );
        let first = WheelSessionId::parse("sess-head-b").expect("valid session");
        let second = WheelSessionId::parse("sess-head-a").expect("valid session");
        PendingStore::bind_head(&store, &first, 1000).expect("bind first");
        PendingStore::bind_head(&store, &second, 1010).expect("bind second");
        // Duplicate binds refuse with zero extra rows.
        assert_eq!(
            PendingStore::bind_head(&store, &first, 1020),
            Err(PendingError::AlreadyExists)
        );
        let listed = PendingStore::list_head_sessions(&store).expect("list");
        let names: Vec<&str> = listed.iter().map(|id| id.as_str()).collect();
        assert_eq!(names, vec!["sess-head-a", "sess-head-b"]);
        PendingStore::unbind_head(&store, &first).expect("unbind");
        assert_eq!(
            PendingStore::unbind_head(&store, &first),
            Err(PendingError::NotFound)
        );
        let listed = PendingStore::list_head_sessions(&store).expect("list");
        let names: Vec<&str> = listed.iter().map(|id| id.as_str()).collect();
        assert_eq!(names, vec!["sess-head-a"]);
    }

    #[test]
    fn head_scope_out_of_range_timestamp_refuses_with_zero_writes() {
        let store = test_store();
        let session = test_session();
        assert_eq!(
            PendingStore::bind_head(&store, &session, u64::MAX),
            Err(PendingError::Storage(INTEGER_RANGE_MSG.to_owned()))
        );
        assert!(
            PendingStore::list_head_sessions(&store)
                .expect("list")
                .is_empty()
        );
    }

    #[test]
    fn head_scope_bound_refuses_cap_plus_one_with_zero_writes() {
        let store = test_store();
        let mut marked = Vec::with_capacity(MAX_HEAD_MARKERS);
        for index in 0..MAX_HEAD_MARKERS {
            let session =
                WheelSessionId::parse(&format!("sess-head-{index:02}")).expect("valid session");
            PendingStore::bind_head(&store, &session, 1000 + index as u64).expect("bind");
            marked.push(session);
        }
        // The read path serves the full bound in order: this is exactly the
        // set a direct-HEAD resume consumes, so resume works at the cap.
        let listed = PendingStore::list_head_sessions(&store).expect("list at cap");
        let names: Vec<&str> = listed.iter().map(|id| id.as_str()).collect();
        let mut expected: Vec<String> = marked.iter().map(|id| id.as_str().to_owned()).collect();
        expected.sort();
        assert_eq!(
            names,
            expected.iter().map(String::as_str).collect::<Vec<_>>()
        );
        // One past the bound refuses typed with zero writes.
        let extra = WheelSessionId::parse("sess-head-extra").expect("valid session");
        assert_eq!(
            PendingStore::bind_head(&store, &extra, 2000),
            Err(PendingError::TooManyOpen {
                limit: MAX_HEAD_MARKERS,
            })
        );
        // A duplicate bind still reports AlreadyExists at the bound (the
        // novelty check runs before the admission count), and unbinding one
        // marker re-admits a new session.
        assert_eq!(
            PendingStore::bind_head(&store, &marked[0], 2010),
            Err(PendingError::AlreadyExists)
        );
        let count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row("SELECT COUNT(*) FROM pending_head_scope", [], |row| {
                    row.get(0)
                })
                .expect("count")
        };
        assert_eq!(count, MAX_HEAD_MARKERS as i64);
        PendingStore::unbind_head(&store, &marked[0]).expect("unbind");
        PendingStore::bind_head(&store, &extra, 2020).expect("re-admit after unbind");
        let listed = PendingStore::list_head_sessions(&store).expect("list");
        assert_eq!(listed.len(), MAX_HEAD_MARKERS);
        assert!(listed.iter().any(|id| id.as_str() == "sess-head-extra"));
    }

    #[test]
    fn head_scope_poison_fails_whole_read_with_rows_preserved() {
        let store = test_store();
        let session = test_session();
        PendingStore::bind_head(&store, &session, 1000).expect("bind");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE pending_head_scope SET bound_at_ms = -1 WHERE session_id = 'sess-pending-1'",
                    [],
                )
                .expect("plant negative timestamp");
        }
        assert_eq!(
            PendingStore::list_head_sessions(&store),
            Err(PendingError::Corrupt)
        );
        let count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row("SELECT COUNT(*) FROM pending_head_scope", [], |row| {
                    row.get(0)
                })
                .expect("count")
        };
        assert_eq!(count, 1);
    }

    #[test]
    fn head_scope_non_integer_timestamp_fails_whole_read_as_corrupt() {
        let store = test_store();
        let session = test_session();
        PendingStore::bind_head(&store, &session, 1000).expect("bind");
        {
            let guard = store.lock_conn().expect("lock");
            guard
                .execute(
                    "UPDATE pending_head_scope SET bound_at_ms = 'not-an-integer' WHERE session_id = 'sess-pending-1'",
                    [],
                )
                .expect("plant text timestamp");
        }
        assert_eq!(
            PendingStore::list_head_sessions(&store),
            Err(PendingError::Corrupt)
        );
        let count: i64 = {
            let guard = store.lock_conn().expect("lock");
            guard
                .query_row("SELECT COUNT(*) FROM pending_head_scope", [], |row| {
                    row.get(0)
                })
                .expect("count")
        };
        assert_eq!(count, 1);
    }
}

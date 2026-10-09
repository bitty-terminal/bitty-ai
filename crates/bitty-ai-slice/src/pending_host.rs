//! Host-owned pending-effect executor adapter (AI-0199, slice v1).
//!
//! [`PendingToolExecutor`] is the single host-side choke point between the
//! runtime [`ToolBus`][bitty_ai_runtime_tool] and a [`BittyHost`]-backed
//! inner [`ToolExecutor`][bitty_ai_runtime_tool]: it opens a durable
//! [`PendingStore`] entry before delegating to `execute_with_context` and
//! closes it with a terminal [`PendingDisposition`] after the delegate
//! returns. The runtime crate stays std-only by invariant (it cannot name
//! SQLite); the session crate owns the table; this adapter owns the
//! begin/delegate/resolve write path -- mirroring the `UnknownReconciler`
//! seam pattern (query-only runtime trait, host-owned implementation).
//!
//! ## Outcome mapping
//!
//! - `Ok(success)` resolves [`PendingDisposition::Success`] and returns the
//!   success unchanged.
//! - `Err(EffectUnknown)` resolves *nothing*: the open row persists as the
//!   Unknown record, and the original error returns unchanged. This is the
//!   crash window made durable -- a reopen names the call id in the resume
//!   report for inspection-or-direction reconcile (never auto-reconcile,
//!   never replay; the log grants no retry eligibility).
//! - `Err(Denied)` resolves [`PendingDisposition::Denied`].
//! - Any other `Err` resolves [`PendingDisposition::Failed`].
//!
//! Resolve-after-return is best-effort: when the delegate succeeded but the
//! resolve itself fails (stale fence, contended store), the adapter still
//! returns the success -- the effect did happen, and flipping it to an error
//! would lie. The open row survives for exactly the reconcile path above
//! (crash-ack-loss parity: an unrecorded ack is an Unknown, never a silent
//! drop). When the delegate failed, resolve failures likewise leave the row
//! open (the conservative direction: open errs toward inspection).
//!
//! Admission refusals ([`ToolStatus::Refused`][bitty_ai_runtime_tool]) open
//! no entry: refusal happens at the bus precheck before any executor
//! contact, so this adapter -- which begins at executor contact -- never
//! sees refused calls. There is nothing in flight to reconcile.
//!
//! ## Call-id minting (deterministic, no randomness)
//!
//! The runtime offers no RNG (std-only invariant), so the host mints opaque
//! hex ids deterministically: [`PendingToolExecutor::mint_call_id`] hashes a
//! domain-separated `(session_id, sequence)` pair with SHA-256 and renders
//! the 64 lowercase hex characters. Uniqueness comes from the
//! session-scoped monotonic sequence, not from entropy -- and no randomness
//! is claimed. The counter starts at a caller-supplied `initial_seq` (tests
//! pass `0` for hermetic ids); after a reopen the counter restarts while
//! pre-crash rows survive, so a mint that collides retries with the next
//! sequence, bounded by [`MAX_CALL_ID_MINT_ATTEMPTS`]. Stale-epoch resolve
//! refuses with zero writes and the entry stays open; a same-disposition
//! re-resolve hits the bounded recent cache as an idempotent no-op while a
//! conflicting one refuses as `Mismatch` (an evicted id reports `NotFound`,
//! see the session-plane docs).
//!
//! Store failures at begin fail closed before delegation: the delegate is
//! never contacted, so no effect can outrun its record. `TooManyOpen` maps
//! to the exact [`ToolError::CallLimitExceeded`][bitty_ai_runtime_tool]
//! bound failure; every other begin failure maps to a static-text `Denied`
//! (fail-closed refusal before effect, with no storage detail and no caller
//! text echoed).
//!
//! Caller clocks only (`now_ms` arrives per dispatch inside
//! [`ExecutionContext`][bitty_ai_runtime_tool]); no wall clock, no threads.
//!
//! [bitty_ai_runtime_tool]: https://github.com/bitty-terminal/bitty-ai

use bitty_ai_runtime::session::IdIssuer;
use bitty_ai_runtime::tool::{ExecutionContext, ToolError, ToolExecutor, ToolSuccess};
use bitty_ai_session::content_hash::ContentHash;
use bitty_ai_session::content_store::ContentStore;
use bitty_ai_session::pending::{
    MAX_OPEN_PENDING_PER_SESSION, PendingBegin, PendingDisposition, PendingError, PendingResolve,
    PendingStore,
};
use bitty_ai_session::sessions::WheelSessionId;

/// Maximum mint attempts when a freshly minted call id collides.
///
/// Collisions happen only after a counter restart (reopen) against surviving
/// pre-crash rows of the same session; each retry advances the sequence, so
/// at most `retained rows + 1` attempts are ever needed. The bound keeps a
/// pathological store from spinning the adapter.
pub const MAX_CALL_ID_MINT_ATTEMPTS: u64 = 1024;

/// Host-owned [`ToolExecutor`][bitty_ai_runtime_tool] adapter opening and
/// closing durable pending-effect rows around an inner executor.
///
/// Constructed per session binding (plus optional task binding): `generation`
/// and `epoch` snapshot the owner fences at begin time, and the setters
/// below advance them when the owner takes a new fence. Lifetimes borrow the
/// shared store -- no connection is opened here (the Wheel single-open path
/// owns it).
pub struct PendingToolExecutor<'a, E> {
    inner: E,
    store: &'a ContentStore,
    session_id: WheelSessionId,
    task_id: Option<String>,
    generation: u64,
    epoch: u64,
    next_seq: u64,
}

impl<'a, E> PendingToolExecutor<'a, E> {
    /// Wrap `inner` with pending-effect recording for one session binding.
    ///
    /// `task_id` is the task calls run under, when task-assigned;
    /// `initial_seq` seeds the monotonic mint counter (pass `0` for fresh
    /// hermetic sequences; pass a higher value to skip a known-occupied
    /// prefix instead of paying mint retries).
    #[must_use]
    pub fn new(
        inner: E,
        store: &'a ContentStore,
        session_id: WheelSessionId,
        task_id: Option<String>,
        generation: u64,
        epoch: u64,
        initial_seq: u64,
    ) -> Self {
        Self {
            inner,
            store,
            session_id,
            task_id,
            generation,
            epoch,
            next_seq: initial_seq,
        }
    }

    /// Borrow the wrapped executor.
    #[must_use]
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Mutably borrow the wrapped executor.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    /// Borrow the bound session id.
    #[must_use]
    pub fn session_id(&self) -> &WheelSessionId {
        &self.session_id
    }

    /// Read the current generation fence.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Read the current epoch fence.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Read the next mint sequence (advances on every begin).
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Advance the generation fence (task generation moved on).
    pub fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    /// Advance the epoch fence (session took a new fence).
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Rebind the task calls run under, when task-assigned.
    pub fn set_task_id(&mut self, task_id: Option<String>) {
        self.task_id = task_id;
    }

    /// Mint the opaque hex call id for `(session_id, seq)`.
    ///
    /// Thin associated alias over [`mint_pending_call_id`].
    #[must_use]
    pub fn mint_call_id(session_id: &WheelSessionId, seq: u64) -> String {
        mint_pending_call_id(session_id, seq)
    }

    /// Open the pending row for one dispatch, retrying the mint on collision.
    ///
    /// Returns the admitted call id. Begin failures map to fail-closed
    /// [`ToolError`]s and the delegate is never contacted on this path.
    fn begin_guarded(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<String, ToolError> {
        let mut seq = self.next_seq;
        let mut attempts: u64 = 0;
        loop {
            let call_id = Self::mint_call_id(&self.session_id, seq);
            let outcome = PendingStore::begin(
                self.store,
                PendingBegin {
                    call_id: &call_id,
                    session_id: &self.session_id,
                    task_id: self.task_id.as_deref(),
                    tool,
                    args: arguments,
                    generation: self.generation,
                    epoch: self.epoch,
                    now_ms,
                },
            );
            match outcome {
                Ok(_) => {
                    self.next_seq = seq.wrapping_add(1);
                    return Ok(call_id);
                }
                Err(PendingError::AlreadyExists) => {
                    seq = seq.wrapping_add(1);
                    attempts += 1;
                    if attempts >= MAX_CALL_ID_MINT_ATTEMPTS {
                        return Err(ToolError::Denied {
                            name: tool.to_owned(),
                            reason: "pending log call-id mint exhausted".to_owned(),
                        });
                    }
                }
                Err(other) => return Err(map_begin_err(tool, &other)),
            }
        }
    }

    /// Close the pending row after delegation, best-effort.
    ///
    /// Resolve failures leave the row open for the reconcile path (see the
    /// module docs); they never flip the delegate outcome. `EffectUnknown`
    /// never reaches here -- the caller leaves the row open instead.
    fn resolve_best_effort(&self, call_id: &str, disposition: PendingDisposition, now_ms: u64) {
        let _ = PendingStore::resolve(
            self.store,
            PendingResolve {
                call_id,
                disposition,
                claim_epoch: self.epoch,
                expected_generation: self.generation,
                now_ms,
            },
        );
    }
}

impl<E> ToolExecutor for PendingToolExecutor<'_, E>
where
    E: ToolExecutor,
{
    fn execute_with_context(
        &mut self,
        tool: &str,
        arguments: &[u8],
        context: &ExecutionContext,
    ) -> Result<ToolSuccess, ToolError> {
        // Begin before any delegate contact: a begin failure refuses here
        // with the delegate never contacted, so no effect outruns its row.
        let call_id = self.begin_guarded(tool, arguments, context.now_ms)?;
        let outcome = self.inner.execute_with_context(tool, arguments, context);
        match outcome {
            Ok(success) => {
                self.resolve_best_effort(&call_id, PendingDisposition::Success, context.now_ms);
                Ok(success)
            }
            Err(unknown @ ToolError::EffectUnknown { .. }) => {
                // The crash window, made durable: the open row persists as
                // the Unknown record for inspection-or-direction reconcile.
                Err(unknown)
            }
            Err(denied @ ToolError::Denied { .. }) => {
                self.resolve_best_effort(&call_id, PendingDisposition::Denied, context.now_ms);
                Err(denied)
            }
            Err(other) => {
                self.resolve_best_effort(&call_id, PendingDisposition::Failed, context.now_ms);
                Err(other)
            }
        }
    }

    fn execute(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<ToolSuccess, ToolError> {
        let context = ExecutionContext {
            execution_id: IdIssuer::default().execution(),
            now_ms,
        };
        self.execute_with_context(tool, arguments, &context)
    }
}

/// Mint the opaque hex call id for `(session_id, seq)`.
///
/// Deterministic SHA-256 over a domain-separated input, rendered as 64
/// lowercase hex characters: opaque (no ordering assumptions, no
/// `NEXT_ID` coupling) and unique per session while the sequence stays
/// monotonic. Deterministic, not random -- uniqueness comes from the
/// sequence, never from entropy. Free function so callers need no executor
/// type parameter to predict ids.
#[must_use]
pub fn mint_pending_call_id(session_id: &WheelSessionId, seq: u64) -> String {
    let mut input = Vec::with_capacity(64);
    input.extend_from_slice(b"bitty-pending-v1/");
    input.extend_from_slice(session_id.as_str().as_bytes());
    input.extend_from_slice(b"/");
    input.extend_from_slice(seq.to_string().as_bytes());
    ContentHash::compute(&input).to_hex()
}

/// Map a begin failure to a fail-closed [`ToolError`].
///
/// `TooManyOpen` keeps its exact bound shape; everything else is a static
/// refusal before effect (no storage detail, no caller text echoed). The
/// `name` echoes the bus-validated tool name, matching the runtime's own
/// `Denied` attribution shape.
fn map_begin_err(tool: &str, err: &PendingError) -> ToolError {
    match err {
        PendingError::TooManyOpen { limit } => ToolError::CallLimitExceeded { limit: *limit },
        _ => ToolError::Denied {
            name: tool.to_owned(),
            reason: "pending log unavailable; refusing before effect".to_owned(),
        },
    }
}

/// Re-export the session-plane cap so callers read one bound.
#[must_use]
pub fn max_open_pending_per_session() -> usize {
    MAX_OPEN_PENDING_PER_SESSION
}

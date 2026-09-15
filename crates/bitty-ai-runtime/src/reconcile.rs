//! Unknown-outcome reconcile protocol (`MP-7`).
//!
//! Query path without re-execution, bounded retry with deterministic backoff
//! ceilings, and typed escalation when unresolvable.
//!
//! Rules enforced here:
//!
//! - Reconcile queries status; it never executes an effect. The only query
//!   path is [`UnknownReconciler::reconcile`], keyed by [`ExecutionId`].
//!   Re-dispatch under a tracked id is refused by design; a new id is an
//!   explicit re-execution and is never created by this module.
//! - Retry budget is separate from the tool-call budget: [`ReconcileConfig`]
//!   (mirrored on [`crate::agent::AgentConfig`]) counts reconcile queries,
//!   never tool dispatches. It does not read or modify
//!   [`crate::tool::MAX_TOOL_CALLS_PER_TURN`] or the per-turn counter.
//! - Backoff is deterministic: [`reconcile_delay_ms`] is a pure function of
//!   the attempt index plus caller-supplied bounds, and
//!   [`Agent::reconcile_unknown`](crate::agent::Agent::reconcile_unknown)
//!   derives `next_retry_ms` from caller-supplied `now_ms` with saturating
//!   addition. No wall clock, no thread sleep, no async runtime.
//! - Unresolvable `Unknown` escalates to the typed [`UnknownEscalation`]
//!   report (and [`crate::agent::AgentError::UnknownUnresolved`]), never a
//!   bare string. Escalation fails the session closed; no further retry is
//!   attempted under this protocol.
//!
//! Out of scope: host status inspection itself (lives behind
//! [`UnknownReconciler`] like [`crate::tool::ToolExecutor`] lives host-side),
//! persistence, multi-agent coordination, and wall-clock scheduling.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::session::ExecutionId;
use crate::tool::ToolStatus;

/// Default reconcile query budget: maximum status queries per `Unknown`
/// execution before escalation.
pub const DEFAULT_MAX_UNKNOWN_RETRIES: usize = 3;

/// Default base delay in milliseconds for attempt 0.
pub const DEFAULT_RECONCILE_BASE_DELAY_MS: u64 = 100;

/// Default per-attempt delay ceiling in milliseconds.
pub const DEFAULT_RECONCILE_MAX_DELAY_MS: u64 = 5_000;

/// Hard cap on reconcile queries per execution, regardless of configured
/// budget. Bounds host work even when a caller configures a larger budget.
pub const MAX_RECONCILE_ATTEMPTS: usize = 16;

/// Hard ceiling for any single computed backoff delay in milliseconds.
pub const MAX_RECONCILE_DELAY_MS: u64 = 30_000;

/// Maximum bytes kept for a pending reason surfaced on escalation.
pub const MAX_RECONCILE_REASON_BYTES: usize = 512;

/// Bounded reconcile configuration. Separate from the tool-call budget by
/// construction: it counts status queries, never dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileConfig {
    /// Maximum status queries per `Unknown` execution before escalation.
    pub max_unknown_retries: usize,
    /// Base delay in milliseconds for attempt 0; doubles per attempt.
    pub base_delay_ms: u64,
    /// Per-attempt delay ceiling in milliseconds.
    pub max_delay_ms: u64,
}

impl Default for ReconcileConfig {
    fn default() -> Self {
        Self {
            max_unknown_retries: DEFAULT_MAX_UNKNOWN_RETRIES,
            base_delay_ms: DEFAULT_RECONCILE_BASE_DELAY_MS,
            max_delay_ms: DEFAULT_RECONCILE_MAX_DELAY_MS,
        }
    }
}

impl ReconcileConfig {
    /// Effective query budget bounded by [`MAX_RECONCILE_ATTEMPTS`].
    #[must_use]
    pub fn effective_retries(&self) -> usize {
        self.max_unknown_retries.min(MAX_RECONCILE_ATTEMPTS)
    }

    /// Effective per-attempt ceiling bounded by [`MAX_RECONCILE_DELAY_MS`].
    #[must_use]
    pub fn effective_max_delay_ms(&self) -> u64 {
        self.max_delay_ms.min(MAX_RECONCILE_DELAY_MS)
    }
}

/// One status query answer. `Resolved` carries a terminal status only: a
/// `Resolved(ToolStatus::Unknown)` answer is treated as still pending by the
/// driver (fail closed, never accept `Unknown` as resolved).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileStatus {
    /// The effect reached a known terminal status without re-execution.
    Resolved(ToolStatus),
    /// The effect is still uncertain; the driver may retry within budget.
    Pending {
        /// What is still uncertain (bounded, surfaced on escalation).
        reason: String,
    },
}

/// Status query seam for `Unknown` executions (`MP-7`).
///
/// Implementations inspect stored outcomes (host ledger, idempotent status
/// read) and must never execute or re-dispatch an effect. Takes `&mut self`
/// like [`crate::tool::ToolExecutor::execute`] so scripted test peers can
/// replay FIFO answers and record queries deterministically for a given
/// script plus `now_ms`.
pub trait UnknownReconciler {
    /// Query the stored outcome for `execution_id` without re-executing.
    fn reconcile(&mut self, tool: &str, execution_id: ExecutionId, now_ms: u64) -> ReconcileStatus;
}

/// Typed escalation report for an `Unknown` that stayed unresolvable within
/// the reconcile budget. Fail-closed: the session is marked failed when this
/// is produced by
/// [`Agent::reconcile_unknown`](crate::agent::Agent::reconcile_unknown).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownEscalation {
    /// Tool whose effect stayed uncertain.
    pub tool: String,
    /// Last observed pending reason (bounded to
    /// [`MAX_RECONCILE_REASON_BYTES`] bytes).
    pub reason: String,
    /// Status queries performed (bounded by the effective budget).
    pub attempts: usize,
    /// Dispatched executions recorded on the agent when escalation fired.
    pub dispatched: usize,
    /// Backoff delays in milliseconds computed for each query, in order.
    /// Deterministic from the attempt index and caller bounds; the caller
    /// schedules them against its own clock.
    pub delays_ms: Vec<u64>,
}

impl Display for UnknownEscalation {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "tool {} effect unreconciled after {} queries: {}",
            self.tool, self.attempts, self.reason
        )
    }
}

impl std::error::Error for UnknownEscalation {}

/// Bounded reconcile outcome for one `Unknown` execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileOutcome {
    /// The effect reached a known terminal status without re-execution.
    Resolved {
        /// Terminal status reported by the reconciler.
        status: ToolStatus,
        /// Status queries performed.
        attempts: usize,
        /// Backoff delays computed for each query, in order.
        delays_ms: Vec<u64>,
    },
    /// No execution with this id is recorded as `Unknown`; nothing ran.
    NoUnknown,
    /// The budget is exhausted and the effect stays uncertain. Typed
    /// fail-closed report; the session is marked failed.
    Escalated(UnknownEscalation),
}

/// Deterministic exponential backoff for reconcile query `attempt`
/// (0-based): `base_delay_ms * 2^attempt`, capped at `max_delay_ms`.
///
/// Pure function of its inputs: no clock is read. Saturates instead of
/// overflowing so hostile bounds cannot wrap to zero.
#[must_use]
pub fn reconcile_delay_ms(attempt: usize, base_delay_ms: u64, max_delay_ms: u64) -> u64 {
    let ceiling = max_delay_ms.min(MAX_RECONCILE_DELAY_MS);
    let mut delay = base_delay_ms.min(ceiling);
    let mut index = 0;
    while index < attempt.min(30) {
        delay = delay.saturating_mul(2).min(ceiling);
        if delay >= ceiling {
            break;
        }
        index += 1;
    }
    delay.min(ceiling)
}

/// Truncate a pending reason to [`MAX_RECONCILE_REASON_BYTES`] bytes on a
/// character boundary.
pub(crate) fn bound_reason(reason: &str) -> String {
    if reason.len() <= MAX_RECONCILE_REASON_BYTES {
        return reason.to_owned();
    }
    let mut end = MAX_RECONCILE_REASON_BYTES;
    while end > 0 {
        if reason.is_char_boundary(end) {
            break;
        }
        end -= 1;
    }
    reason[..end].to_owned()
}

/// Deterministic test peer for [`UnknownReconciler`]. Replays scripted
/// answers FIFO (defaulting to `Pending`) and records every query as a
/// `(tool, execution_id)` pair. Never executes anything: it performs no
/// [`crate::tool::ToolExecutor`] call and owns no executor handle, so
/// executor call counts stay flat across retries by construction.
#[derive(Debug, Default)]
pub struct FakeReconciler {
    script: VecDeque<ReconcileStatus>,
    queries: Vec<(String, ExecutionId)>,
}

impl FakeReconciler {
    /// Construct an empty reconciler (defaults to `Pending`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a terminal resolution.
    pub fn push_resolved(&mut self, status: ToolStatus) {
        self.script.push_back(ReconcileStatus::Resolved(status));
    }

    /// Queue a still-uncertain answer.
    pub fn push_pending(&mut self, reason: impl Into<String>) {
        self.script.push_back(ReconcileStatus::Pending {
            reason: reason.into(),
        });
    }

    /// Queries so far as `(tool, execution_id)` pairs, in order.
    #[must_use]
    pub fn queries(&self) -> &[(String, ExecutionId)] {
        &self.queries
    }

    /// Number of status queries performed.
    #[must_use]
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }
}

impl UnknownReconciler for FakeReconciler {
    fn reconcile(
        &mut self,
        tool: &str,
        execution_id: ExecutionId,
        _now_ms: u64,
    ) -> ReconcileStatus {
        self.queries.push((tool.to_owned(), execution_id));
        self.script.pop_front().unwrap_or(ReconcileStatus::Pending {
            reason: "still uncertain".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_holds_at_ceiling() {
        assert_eq!(reconcile_delay_ms(0, 100, 5_000), 100);
        assert_eq!(reconcile_delay_ms(1, 100, 5_000), 200);
        assert_eq!(reconcile_delay_ms(2, 100, 5_000), 400);
        assert_eq!(reconcile_delay_ms(10, 100, 5_000), 5_000);
        assert_eq!(reconcile_delay_ms(0, 0, 5_000), 0);
    }

    #[test]
    fn backoff_clamps_to_hard_ceiling_and_saturates() {
        assert_eq!(
            reconcile_delay_ms(0, u64::MAX, u64::MAX),
            MAX_RECONCILE_DELAY_MS
        );
        assert_eq!(reconcile_delay_ms(100, 100, 5_000), 5_000);
        assert_eq!(reconcile_delay_ms(5, 1_000, 1_500), 1_500);
    }

    #[test]
    fn config_bounds_hostile_budgets() {
        let config = ReconcileConfig {
            max_unknown_retries: usize::MAX,
            base_delay_ms: u64::MAX,
            max_delay_ms: u64::MAX,
        };
        assert_eq!(config.effective_retries(), MAX_RECONCILE_ATTEMPTS);
        assert_eq!(config.effective_max_delay_ms(), MAX_RECONCILE_DELAY_MS);
    }

    #[test]
    fn reason_truncates_on_char_boundary() {
        let long = "e".repeat(MAX_RECONCILE_REASON_BYTES + 10);
        assert_eq!(bound_reason(&long).len(), MAX_RECONCILE_REASON_BYTES);
        let emoji = "e".repeat(MAX_RECONCILE_REASON_BYTES - 1) + "🦀🦀";
        let bounded = bound_reason(&emoji);
        assert!(bounded.len() <= MAX_RECONCILE_REASON_BYTES);
        assert!(bounded.ends_with('e') || bounded.ends_with("🦀"));
    }

    #[test]
    fn fake_reconciler_replays_fifo_and_records_queries() {
        let mut peer = FakeReconciler::new();
        peer.push_pending("first");
        peer.push_resolved(ToolStatus::Success);
        let id = ExecutionId(7);
        assert_eq!(
            peer.reconcile("workspace_read", id, 0),
            ReconcileStatus::Pending {
                reason: "first".to_owned(),
            }
        );
        assert_eq!(
            peer.reconcile("workspace_read", id, 0),
            ReconcileStatus::Resolved(ToolStatus::Success)
        );
        assert_eq!(peer.query_count(), 2);
        assert_eq!(
            peer.queries(),
            &[
                ("workspace_read".to_owned(), id),
                ("workspace_read".to_owned(), id),
            ]
        );
    }
}

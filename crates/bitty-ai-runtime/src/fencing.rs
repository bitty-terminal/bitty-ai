//! Interactive writer-fencing rule with stale-writer invalidation (AI-0092,
//! AIQ-2C narrowing).
//!
//! AIQ-2C (interactive writer fencing) is needs-evidence: the single-threaded
//! `!Send` session contract (AI-0062) documents one in-process fence, but
//! interactive writer fencing across takeover/restart has no mechanism. The
//! runtime is single-process sans-I/O, so cross-process fencing cannot be
//! built here; what this module pins at the seam is the WRITER-FENCING RULE:
//! takeover/restart must invalidate stale writers before new input.
//!
//! Pinned rule: an interactive writer holds a typed [`WriterLease`]
//! (`writer_id`, `session_generation`, `epoch`) and the supervisor-side
//! [`check_writer`] refuses unless all three hold: (a) the session is
//! `Active` (terminal sessions refuse as `AlreadyTerminated`); (b) the lease
//! generation equals the session's current generation (restart invalidates:
//! a generation bump retires old writer handles); (c) the lease epoch equals
//! the current supervisor epoch (same `StaleEpoch` shape as
//! [`crate::adoption`]). Check order is terminal, then generation, then
//! epoch, so a dead session never leaks which fence moved first. Takeover is
//! the explicit generation bump
//! ([`AgentSession::rotate_generation`][crate::session::AgentSession::rotate_generation]):
//! afterwards every pre-takeover lease refuses as stale-generation while a
//! freshly issued lease succeeds.
//!
//! Shape note: sessions already carry a generation counter (starts at 1,
//! `rotate_generation` bumps and resets the tier, no-op on terminal
//! sessions), so no lifecycle change was needed; the supervisor epoch has no
//! session carrier and stays caller-supplied like `adoption.rs`'s
//! `fence_token`.
//!
//! ## What is proven vs stub
//!
//! - Proven (here plus `tests/writer_fencing.rs`): the three refusal verbs
//!   with terminal-first check order, post-takeover blanket invalidation of
//!   the exact outstanding lease set, fresh-lease success after takeover, and
//!   the refusal `Display` surface. Deterministic doubles only; no threads,
//!   no processes, no I/O, no clock.
//! - Stub (deliberately absent): cross-process transport of the lease,
//!   supervisor liveness/epoch election, and persistence of the fence values.
//!   The lease and the epoch are caller-supplied values at this seam; a host
//!   that fabricates them bypasses the rule, and that host-side attestation
//!   stays out of scope (stated, not built).
//!
//! ## Determinism rules
//!
//! Pure function of its inputs: no wall clock, no thread, no executor, no
//! I/O. The check reads session state but never mutates it. All behavior in
//! tests is reproducible from the lease plus the session and epoch.

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::session::{AgentSession, SessionState};

/// Interactive writer lease: the typed fence binding one writer to one
/// session generation and one supervisor epoch. Issued by the supervisor at
/// writer attach time (bound to the session's current generation); checked
/// by [`check_writer`] before the writer's input is admitted. All fields are
/// caller-supplied values at this seam; the host owns attesting them
/// honestly (see the module docs: fabrication bypasses the rule by
/// construction and stays out of scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriterLease {
    /// Writer the lease was issued to. Identity only: never compared by the
    /// check and never rendered in messages.
    pub writer_id: u64,
    /// Session generation the lease is bound to. Must equal the session's
    /// current generation; a takeover/restart bump retires the lease.
    pub session_generation: u64,
    /// Supervisor epoch the lease is bound to. Must equal the checking
    /// supervisor's current epoch.
    pub epoch: u64,
}

/// Typed writer refusal. The check refuses fail-closed and mutates nothing:
/// no input is admitted, no session state changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriterRefusal {
    /// The session is terminal (including `Canceled`: cancel leftovers
    /// belong to the cancel-reconcile path, never to a stale interactive
    /// writer). Carries the observed state.
    AlreadyTerminated {
        /// Observed state.
        state: SessionState,
    },
    /// The lease generation does not equal the session's current generation.
    /// A stale writer cannot write after takeover/restart.
    StaleGeneration {
        /// Generation carried by the lease.
        lease_generation: u64,
        /// Generation of the session at check time.
        current_generation: u64,
    },
    /// The lease epoch does not equal the current supervisor epoch. A stale
    /// supervisor cannot admit writes.
    StaleEpoch {
        /// Epoch carried by the lease.
        lease_epoch: u64,
        /// Epoch of the checking supervisor.
        current_epoch: u64,
    },
}

impl Display for WriterRefusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::AlreadyTerminated { state } => {
                write!(f, "write refused: session already terminated ({state:?})")
            }
            Self::StaleGeneration {
                lease_generation,
                current_generation,
            } => write!(
                f,
                "write refused: stale generation {lease_generation}, current {current_generation}"
            ),
            Self::StaleEpoch {
                lease_epoch,
                current_epoch,
            } => write!(
                f,
                "write refused: stale epoch {lease_epoch}, current {current_epoch}"
            ),
        }
    }
}

impl std::error::Error for WriterRefusal {}

/// Supervisor-side writer check: the three-part rule as a pure function.
///
/// Admits the write (returns `Ok(())`) only when (a) the session is
/// `Active`, (b) the lease generation equals the session's current
/// generation, and (c) the lease epoch equals `current_epoch`, checked in
/// that order. Refuses fail-closed with a typed [`WriterRefusal`]
/// otherwise. The check reads session state but never mutates it, takes no
/// executor, runs no provider round, and reads no clock.
///
/// # Errors
///
/// Returns [`WriterRefusal`] for a terminal session, a stale generation, or
/// a stale epoch.
pub fn check_writer(
    lease: &WriterLease,
    session: &AgentSession,
    current_epoch: u64,
) -> Result<(), WriterRefusal> {
    let state = session.state();
    if state != SessionState::Active {
        return Err(WriterRefusal::AlreadyTerminated { state });
    }
    let current_generation = session.generation();
    if lease.session_generation != current_generation {
        return Err(WriterRefusal::StaleGeneration {
            lease_generation: lease.session_generation,
            current_generation,
        });
    }
    if lease.epoch != current_epoch {
        return Err(WriterRefusal::StaleEpoch {
            lease_epoch: lease.epoch,
            current_epoch,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::IdIssuer;

    fn session() -> AgentSession {
        let mut ids = IdIssuer::default();
        AgentSession::new(ids.agent_instance(), ids.run(), ids.session())
    }

    fn lease(writer_id: u64, generation: u64, epoch: u64) -> WriterLease {
        WriterLease {
            writer_id,
            session_generation: generation,
            epoch,
        }
    }

    #[test]
    fn current_lease_is_admitted() {
        let session = session();
        let generation = session.generation();
        check_writer(&lease(1, generation, 4), &session, 4).expect("current lease is admitted");
    }

    #[test]
    fn check_order_is_terminal_then_generation_then_epoch() {
        // A lease that is stale on every fence refuses as `AlreadyTerminated`
        // first, so a dead session never leaks which fence moved first.
        let dead = session();
        dead.rotate_generation();
        dead.finish(false);
        assert_eq!(dead.state(), SessionState::Completed);
        let refusal =
            check_writer(&lease(1, 1, 3), &dead, 9).expect_err("triple-stale lease must refuse");
        assert!(matches!(
            refusal,
            WriterRefusal::AlreadyTerminated { state } if state == SessionState::Completed
        ));
        // Live but stale on both remaining fences refuses as
        // `StaleGeneration` before the epoch is considered.
        let live = session();
        let generation = live.generation();
        let refusal = check_writer(&lease(2, generation + 5, 3), &live, 9)
            .expect_err("generation- and epoch-stale lease must refuse");
        assert!(matches!(refusal, WriterRefusal::StaleGeneration { .. }));
    }

    #[test]
    fn refusal_display_names_no_untrusted_text() {
        // `Display` carries only the static reason shape plus numeric fields
        // (generations, epochs) and the lifecycle state: writer ids never
        // reach the message surface.
        let refusal = WriterRefusal::AlreadyTerminated {
            state: SessionState::Failed,
        };
        assert_eq!(
            refusal.to_string(),
            "write refused: session already terminated (Failed)"
        );
    }
}

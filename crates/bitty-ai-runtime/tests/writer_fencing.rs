//! Interactive writer-fencing rule with stale-writer invalidation (AI-0092,
//! AIQ-2C narrowing).
//!
//! AIQ-2C (interactive writer fencing) is needs-evidence: the single-threaded
//! `!Send` session contract (AI-0062) documents one in-process fence, but
//! interactive writer fencing across takeover/restart has no mechanism. The
//! runtime is single-process sans-I/O, so cross-process fencing cannot be
//! built here; what is proven at the seam is the WRITER-FENCING RULE:
//! takeover/restart must invalidate stale writers before new input.
//!
//! Pinned rule: an interactive writer holds a typed [`WriterLease`]
//! (`writer_id`, `session_generation`, `epoch`) and the supervisor-side
//! [`check_writer`] refuses unless all three hold: (a) the lease generation
//! equals the session's current generation (restart invalidates: a generation
//! bump retires old writer handles); (b) the lease epoch equals the current
//! supervisor epoch (same `StaleEpoch` shape as `adoption.rs`); (c) the
//! session is `Active` (terminal sessions refuse as `AlreadyTerminated`).
//! Takeover is the explicit generation bump
//! ([`AgentSession::rotate_generation`]): afterwards every pre-takeover lease
//! refuses as stale-generation while a freshly issued lease succeeds.
//! Check order is terminal, then generation, then epoch, so a dead session
//! never leaks which fence moved first.
//!
//! Candidate-shape adaptation (verified against `session.rs`/`adoption.rs`):
//! sessions already carry a generation counter (starts at 1,
//! `rotate_generation` bumps and resets the tier, no-op on terminal
//! sessions), so no lifecycle change was needed; the supervisor epoch has no
//! session carrier and stays caller-supplied like `adoption.rs`'s
//! `fence_token`; takeover reuses `rotate_generation` (named here as the
//! takeover operation) instead of a duplicate bump.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `recovery_adoption.rs`: supervisor crash-recovery adoption including the
//!   claim `StaleEpoch`. Epoch fencing appears here only as the writer-lease
//!   check.
//! - `runtime_fail_closed.rs` / `cache_invalidation.rs`: mid-turn
//!   `rotate_generation` downgrade and denial mechanics. Rotation appears here
//!   only as the takeover operation behind lease invalidation.
//! - `session.rs` unit tests: the `!Send` contract, elevation, rotation.
//!   Session mechanics appear here only as lease-check inputs.
//!
//! Deterministic and offline: plain values only, no turns, no executor, no
//! clock, no threads, no processes, no I/O.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! writer ids, generations, and epochs never appear in message strings.

use bitty_ai_runtime::{
    AgentSession, IdIssuer, SessionState, WriterLease, WriterRefusal, check_writer,
};

/// Deterministic supervisor epoch shared by the lease and the checker in the
/// success paths.
const EPOCH: u64 = 7;

fn session() -> AgentSession {
    let mut issuer = IdIssuer::default();
    AgentSession::new(issuer.agent_instance(), issuer.run(), issuer.session())
}

/// Issue a writer lease bound to the session's current generation, mirroring
/// how a supervisor hands a live writer its fence values at attach time.
fn issue(session: &AgentSession, writer_id: u64, epoch: u64) -> WriterLease {
    WriterLease {
        writer_id,
        session_generation: session.generation(),
        epoch,
    }
}

#[test]
fn stale_generation_write_is_refused() {
    // Rule (a): a restart bumps the session generation and retires old
    // writer handles. The pre-restart lease must refuse even though its
    // epoch is current and the session is live.
    let session = session();
    let lease = issue(&session, 1, EPOCH);
    assert_eq!(session.generation(), 1);
    session.rotate_generation();
    assert_eq!(session.generation(), 2);
    let refusal =
        check_writer(&lease, &session, EPOCH).expect_err("stale generation must refuse the write");
    assert!(
        matches!(
            refusal,
            WriterRefusal::StaleGeneration {
                lease_generation,
                current_generation
            } if lease_generation == 1 && current_generation == 2
        ),
        "stale generation must refuse as StaleGeneration"
    );
    // Refusal decided nothing: the session is untouched and live.
    assert_eq!(session.state(), SessionState::Active);
    assert_eq!(session.generation(), 2);
}

#[test]
fn stale_epoch_write_is_refused() {
    // Rule (b): the lease epoch binds the writer to one supervisor epoch. A
    // lease fenced to a superseded epoch refuses even though its generation
    // is current and the session is live.
    let session = session();
    let lease = issue(&session, 1, EPOCH);
    let current_epoch = EPOCH + 1;
    let refusal = check_writer(&lease, &session, current_epoch)
        .expect_err("stale epoch must refuse the write");
    assert!(
        matches!(
            refusal,
            WriterRefusal::StaleEpoch {
                lease_epoch,
                current_epoch: seen_epoch
            } if lease_epoch == EPOCH && seen_epoch == EPOCH + 1
        ),
        "stale epoch must refuse as StaleEpoch"
    );
    // Refusal decided nothing: the session is untouched and live.
    assert_eq!(session.state(), SessionState::Active);
    assert_eq!(session.generation(), 1);
}

#[test]
fn terminal_session_write_is_refused() {
    // Rule (c): terminal sessions refuse every write, even from a
    // generation- and epoch-current lease. `Canceled` counts as terminal
    // for writers: cancel leftovers belong to the cancel-reconcile path,
    // never to a stale interactive writer.
    for want in [
        SessionState::Completed,
        SessionState::Failed,
        SessionState::Canceled,
    ] {
        let terminal = session();
        let lease = issue(&terminal, 1, EPOCH);
        match want {
            SessionState::Completed => terminal.finish(false),
            SessionState::Failed => terminal.finish(true),
            SessionState::Canceled | SessionState::Active => terminal.cancel(),
        }
        assert_eq!(terminal.state(), want);
        let refusal = check_writer(&lease, &terminal, EPOCH)
            .expect_err("terminal session must refuse the write");
        assert!(
            matches!(
                refusal,
                WriterRefusal::AlreadyTerminated { state } if state == want
            ),
            "terminal session must refuse as AlreadyTerminated"
        );
    }
    // Priority pin: terminal wins over stale generation. A lease that is
    // both stale-generation and terminal refuses as `AlreadyTerminated`, so
    // a dead session never leaks which fence moved first.
    let dead = session();
    let stale = issue(&dead, 9, EPOCH);
    dead.rotate_generation();
    dead.finish(false);
    assert_eq!(dead.state(), SessionState::Completed);
    assert_eq!(dead.generation(), 2);
    let refusal =
        check_writer(&stale, &dead, EPOCH).expect_err("terminal session must refuse the write");
    assert!(
        matches!(
            refusal,
            WriterRefusal::AlreadyTerminated { state } if state == SessionState::Completed
        ),
        "terminal check runs before generation"
    );
}

#[test]
fn post_takeover_old_leases_all_refused_with_exact_set() {
    // Rule (d): takeover is the explicit generation bump. Afterwards every
    // pre-takeover lease refuses as stale-generation: the exact outstanding
    // set, none admitted, none reclassified.
    let session = session();
    let outstanding = vec![
        issue(&session, 1, EPOCH),
        issue(&session, 2, EPOCH),
        issue(&session, 3, EPOCH),
    ];
    assert_eq!(outstanding.len(), 3);
    // Steady state first: live writers are admitted before takeover.
    for lease in &outstanding {
        check_writer(lease, &session, EPOCH).expect("live writer is admitted before takeover");
    }
    session.rotate_generation();
    assert_eq!(session.generation(), 2);
    let mut refused = 0;
    for lease in &outstanding {
        let refusal = check_writer(lease, &session, EPOCH)
            .expect_err("pre-takeover lease must refuse after takeover");
        assert!(
            matches!(
                refusal,
                WriterRefusal::StaleGeneration {
                    lease_generation,
                    current_generation
                } if lease_generation == 1 && current_generation == 2
            ),
            "pre-takeover lease must refuse as StaleGeneration"
        );
        refused += 1;
    }
    assert_eq!(refused, outstanding.len());
    assert_eq!(refused, 3);
}

#[test]
fn fresh_lease_after_takeover_succeeds() {
    // Takeover invalidates handles, not writers: the same writer id
    // re-issued at the new generation is admitted.
    let session = session();
    let retired = issue(&session, 1, EPOCH);
    session.rotate_generation();
    assert!(
        check_writer(&retired, &session, EPOCH).is_err(),
        "retired lease stays refused"
    );
    let fresh = issue(&session, 1, EPOCH);
    assert_eq!(fresh.session_generation, 2);
    check_writer(&fresh, &session, EPOCH).expect("fresh lease is admitted after takeover");
    assert_eq!(session.state(), SessionState::Active);
}

#[test]
fn refusal_display_names_no_untrusted_text() {
    // `Display` carries only the static reason shape plus numeric fields
    // (generations, epochs) and the lifecycle state: writer ids never reach
    // the message surface.
    let refusal = WriterRefusal::StaleGeneration {
        lease_generation: 1,
        current_generation: 2,
    };
    assert_eq!(
        refusal.to_string(),
        "write refused: stale generation 1, current 2"
    );
    let refusal = WriterRefusal::StaleEpoch {
        lease_epoch: 7,
        current_epoch: 8,
    };
    assert_eq!(
        refusal.to_string(),
        "write refused: stale epoch 7, current 8"
    );
    let refusal = WriterRefusal::AlreadyTerminated {
        state: SessionState::Completed,
    };
    assert_eq!(
        refusal.to_string(),
        "write refused: session already terminated (Completed)"
    );
}

//! Cancellation metric and buffered-chunk contract (AI-0140, `MP-7`).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns, scripted
//! [`FakeToolExecutor`] outcomes, caller-supplied `now_ms`. No network, no
//! secrets, no wall clock, no threads.
//!
//! Contract under test:
//! 1. Cancellation counter: [`AgentSession::cancel_count`] counts
//!    first-honored `Active` -> `Canceled` transitions only. A fresh session
//!    reads 0, one honored cancel reads 1, idempotent repeats and cancels on
//!    terminal sessions never inflate it. The session owns cancel state, so
//!    it owns the count; [`Agent::cancel_count`] is a read-only delegate.
//! 2. Buffered-chunk keep contract: cancellation stops emission at chunk
//!    boundaries but already-accepted sink bytes stay (no rollback, no drop);
//!    the cancel is still counted. This pins current behavior, it does not
//!    change the stop-at-boundary semantics.
//! 3. Single-agent shared-cancel scope: canceling through one session clone
//!    (or the agent delegate) is visible through every other handle,
//!    including the count. Two-waiter shared work is out of scope for v0.1
//!    (single agent, no multi-agent routing), noted in the session docs.
//!
//! Non-overlap with neighbors (this file extends only the untouched halves):
//! - `session.rs` unit test `cancel_is_idempotent_and_shared`: state-level
//!   clone visibility and terminal preservation. Nothing here re-asserts the
//!   state machine itself, only the count visibility on top of it.
//! - `cancel_after_final.rs`: cancel-at-final-delivery reconcile shapes and
//!   byte preservation. This file pins the count alongside those shapes and
//!   the mid-batch keep shape, never the reconcile outcomes themselves.
//! - `runtime_fail_closed.rs`: cancel before/after dispatch outcomes.
//!   Nothing here re-asserts those outcomes, only the metric.

use bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES;
use bitty_ai_runtime::{
    Agent, AgentConfig, AgentSession, ExecOutcome, FakeProvider, FakeToolExecutor, FragmentKind,
    IdIssuer, ModelProvider, ProviderTurn, ProviderUsage, SessionState, StreamChunk, StreamError,
    StreamSink, ToolBus, ToolRegistry, VecSink, validate_chunk,
};

const NOW_MS: u64 = 1_700_000_000_000;

fn session() -> AgentSession {
    let mut issuer = IdIssuer::default();
    AgentSession::new(issuer.agent_instance(), issuer.run(), issuer.session())
}

fn final_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

/// Sink wrapper that cancels the bound session when the `cancel_on`-th
/// (1-based) chunk has been accepted, exposing the accepted bytes for the
/// keep-contract assertion.
struct CancelOnNthChunk {
    inner: VecSink,
    session: AgentSession,
    cancel_on: usize,
}

impl CancelOnNthChunk {
    fn accepted_bytes(&self) -> Vec<u8> {
        self.inner.concatenated_bytes()
    }
}

impl StreamSink for CancelOnNthChunk {
    fn emit(&mut self, chunk: StreamChunk) -> Result<(), StreamError> {
        self.inner.emit(chunk)?;
        if self.inner.len() == self.cancel_on {
            self.session.cancel();
        }
        Ok(())
    }

    fn chunks(&self) -> &[StreamChunk] {
        self.inner.chunks()
    }
}

#[test]
fn fresh_session_cancel_count_is_zero() {
    assert_eq!(session().cancel_count(), 0);
}

#[test]
fn cancel_once_counts_one() {
    let sess = session();
    sess.cancel();
    assert_eq!(sess.state(), SessionState::Canceled);
    assert_eq!(sess.cancel_count(), 1);
}

#[test]
fn repeat_cancel_does_not_inflate() {
    let sess = session();
    sess.cancel();
    sess.cancel();
    sess.clone().cancel();
    assert_eq!(sess.state(), SessionState::Canceled);
    assert_eq!(sess.cancel_count(), 1);
}

#[test]
fn cancel_on_terminal_session_does_not_count() {
    // A completed session keeps its outcome: cancel is a no-op and the
    // metric stays 0, so one waiter's late cancel never rewrites another
    // waiter's completed outcome nor the count.
    let sess = session();
    sess.finish(false);
    assert_eq!(sess.state(), SessionState::Completed);
    sess.cancel();
    assert_eq!(sess.state(), SessionState::Completed);
    assert_eq!(sess.cancel_count(), 0);
}

#[test]
fn cancel_count_is_visible_through_clones() {
    let sess = session();
    let clone = sess.clone();
    assert_eq!(clone.cancel_count(), 0);
    sess.cancel();
    assert!(clone.is_cancelled());
    assert_eq!(clone.cancel_count(), 1);
    // Repeat through the other handle: still 1, never inflated.
    clone.cancel();
    assert_eq!(sess.cancel_count(), 1);
    assert_eq!(clone.cancel_count(), 1);
}

#[test]
fn agent_cancel_delegate_counts_once() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("done"));
    let sess = session();
    let agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    assert_eq!(agent.cancel_count(), 0);
    agent.cancel();
    agent.cancel();
    assert_eq!(agent.cancel_count(), 1);
    assert_eq!(sess.cancel_count(), 1);
    assert_eq!(agent.session().cancel_count(), 1);
}

#[test]
fn completed_turn_leaves_count_at_zero() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("done"));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert_eq!(agent.cancel_count(), 0);
    assert_eq!(sess.cancel_count(), 0);
}

#[test]
fn cancel_before_turn_counts_once_without_io() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("never streamed"));
    let sess = session();
    sess.cancel();
    let mut agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    assert!(matches!(
        outcome,
        ExecOutcome::Canceled {
            dispatched: 0,
            unknown: 0
        }
    ));
    assert!(sink.chunks().is_empty());
    assert_eq!(agent.cancel_count(), 1);
    assert_eq!(sess.cancel_count(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 0);
    assert_eq!(executor.calls().len(), 0);
}

#[test]
fn mid_batch_cancel_keeps_accepted_bytes_and_counts_cancel() {
    // Keep contract: the first of several fragments is accepted, then
    // cancellation stops emission. Already-accepted bytes stay in the sink
    // (no rollback, no drop), the outcome reconciles as canceled, and the
    // cancel is counted exactly once even though the turn loop observes the
    // canceled state at several boundaries.
    let text = "é".repeat(MAX_FRAGMENT_BYTES);
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn(&text));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = CancelOnNthChunk {
        inner: VecSink::new(),
        session: sess.clone(),
        cancel_on: 1,
    };

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    assert!(
        matches!(
            outcome,
            ExecOutcome::Canceled {
                dispatched: 0,
                unknown: 0
            }
        ),
        "mid-batch cancel must reconcile as Canceled, got: {outcome:?}"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    // Keep: exactly the first chunk stays, still valid framing, prefix bytes.
    assert_eq!(sink.chunks().len(), 1);
    assert!(!sink.chunks()[0].is_final);
    assert_eq!(sink.chunks()[0].fragment.kind, FragmentKind::Markdown);
    validate_chunk(&sink.chunks()[0]).expect("accepted partial chunk stays valid");
    assert!(text.as_bytes().starts_with(&sink.accepted_bytes()));
    assert_ne!(sink.accepted_bytes(), text.as_bytes());
    // Count: one honored cancel, even across repeated boundary observations.
    assert_eq!(agent.cancel_count(), 1);
    assert_eq!(sess.cancel_count(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(executor.calls().len(), 0);
}

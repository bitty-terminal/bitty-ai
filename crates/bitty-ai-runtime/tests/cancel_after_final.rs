//! Cancel-at-final-delivery regression tests (AI-RUN-005, task AI-0110).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns, caller-supplied
//! `now_ms`, and a benign wrapper around [`VecSink`] that requests
//! cancellation at an exact accepted-chunk boundary. No network, no secrets,
//! no wall clock, no threads, no sleeps.
//!
//! Contract under test: a cancel observed before, during, or after the final
//! text emission must reconcile as a cancellation outcome, never
//! [`ExecOutcome::Completed`], and already-accepted bytes stay in the sink.
//! Before the fix `emit_fragments` only checked cancellation *before* each
//! `sink.emit`, so a cancel landing on the batch-closing delivery still
//! reported the batch complete and the turn declared `Completed` while the
//! session state said `Canceled`.

use bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES;
use bitty_ai_runtime::{
    Agent, AgentConfig, AgentSession, ExecOutcome, FakeProvider, FragmentKind, ModelProvider,
    ProviderTurn, ProviderUsage, SessionState, StreamChunk, StreamError, StreamSink, ToolBus,
    ToolRegistry, VecSink, validate_chunk,
};

const NOW_MS: u64 = 1_700_000_000_000;

fn session() -> AgentSession {
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
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

/// Benign sink wrapper that cancels the bound session when a chunk whose
/// `is_final` flag is set has been accepted: the deterministic analogue of a
/// transport callback cancelling on the batch-closing delivery.
struct CancelOnFinalChunk {
    inner: VecSink,
    session: AgentSession,
}

impl StreamSink for CancelOnFinalChunk {
    fn emit(&mut self, chunk: StreamChunk) -> Result<(), StreamError> {
        let is_final = chunk.is_final;
        self.inner.emit(chunk)?;
        if is_final {
            self.session.cancel();
        }
        Ok(())
    }

    fn chunks(&self) -> &[StreamChunk] {
        self.inner.chunks()
    }
}

/// Benign sink wrapper that cancels the bound session when the `cancel_on`-th
/// (1-based) chunk has been accepted: lets a test land cancellation mid-batch
/// without touching a thread or clock.
struct CancelOnNthChunk {
    inner: VecSink,
    session: AgentSession,
    cancel_on: usize,
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

fn multi_fragment_text() -> String {
    "é".repeat(MAX_FRAGMENT_BYTES)
}

#[test]
fn cancel_on_final_single_fragment_reconciles_as_canceled() {
    // Single-fragment final delivery: the cancel lands while the only chunk
    // is accepted. The outcome and the session state must both say canceled,
    // and the accepted bytes stay.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("final answer"));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = bitty_ai_runtime::FakeToolExecutor::new();
    let mut sink = CancelOnFinalChunk {
        inner: VecSink::new(),
        session: sess.clone(),
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
        "cancel at final delivery must reconcile as Canceled, never Completed"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    assert_eq!(sink.chunks().len(), 1);
    assert_eq!(sink.chunks()[0].fragment.kind, FragmentKind::Markdown);
    assert!(sink.chunks()[0].is_final);
    validate_chunk(&sink.chunks()[0]).expect("accepted final chunk stays valid");
    assert_eq!(
        String::from_utf8(sink.chunks()[0].fragment.bytes.clone()).expect("utf8"),
        "final answer"
    );
    // Exactly one provider round ran; no follow-up work was started.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 0);
    assert_eq!(executor.calls().len(), 0);
}

#[test]
fn cancel_on_final_multi_fragment_reconciles_as_canceled() {
    // Multiple-fragment final delivery: the cancel lands on the last of
    // several chunks, so the whole batch is delivered yet the turn must
    // still report cancellation and preserve every byte.
    let text = multi_fragment_text();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn(&text));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = bitty_ai_runtime::FakeToolExecutor::new();
    let mut sink = CancelOnFinalChunk {
        inner: VecSink::new(),
        session: sess.clone(),
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
        "multi-fragment cancel at final delivery must reconcile as Canceled"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    assert!(sink.chunks().len() > 1);
    for chunk in sink.chunks() {
        validate_chunk(chunk).expect("accepted chunks stay valid");
    }
    assert!(sink.chunks().last().expect("last chunk").is_final);
    assert_eq!(sink.inner.concatenated_bytes(), text.as_bytes());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
}

#[test]
fn cancel_during_final_emission_preserves_accepted_bytes() {
    // Mid-batch cancel: the first of several fragments is accepted, then
    // cancellation stops emission. Already-accepted bytes stay and the
    // outcome is canceled; the un-emitted tail is never delivered.
    let text = multi_fragment_text();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn(&text));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        ToolBus::new(ToolRegistry::new()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = bitty_ai_runtime::FakeToolExecutor::new();
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
        "mid-batch cancel must reconcile as Canceled"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    assert_eq!(sink.chunks().len(), 1);
    assert!(!sink.chunks()[0].is_final);
    validate_chunk(&sink.chunks()[0]).expect("accepted partial chunk stays valid");
    assert!(
        text.as_bytes()
            .starts_with(&sink.inner.concatenated_bytes())
    );
    assert_ne!(sink.inner.concatenated_bytes(), text.as_bytes());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
}

#[test]
fn cancel_before_final_emission_reports_canceled_without_io() {
    // Cancel-before boundary: a session canceled before the turn starts
    // short-circuits with a canceled outcome, emits nothing, and performs no
    // provider I/O.
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
    let mut executor = bitty_ai_runtime::FakeToolExecutor::new();
    let mut sink = CancelOnFinalChunk {
        inner: VecSink::new(),
        session: sess.clone(),
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
        "cancel before emission must reconcile as Canceled"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    assert!(sink.chunks().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 0);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(executor.calls().len(), 0);
}

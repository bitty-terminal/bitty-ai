//! Admission-time provider-timeout ceiling gate (MP-8 S1, AI-0223).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns with explicit
//! [`ProviderUsage`], scripted [`FakeToolExecutor`] outcomes, caller-supplied
//! `now_ms`. No network, no secrets, no wall clock, no threads.
//!
//! Covers:
//! - Hostile `provider_timeout_ms` (`0`, `30_001`, `u64::MAX`) is refused at
//!   `run_turn` admission with [`ToolError::Denied`] carrying the
//!   `TimeoutTooLarge` ceiling text: zero provider `complete` calls, zero
//!   script consumption, zero emission, zero executor contact.
//! - The gate fires before context assembly (an assembly-failing seed does
//!   not shadow it) and after the session lifecycle gates (a terminal
//!   session still reports `AlreadyTerminated`).
//! - Boundary timeouts (`1`, default `5_000`, `30_000`) complete exactly as
//!   before (good path byte-identical).

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ContextPriority, ContextRecord,
    ExecOutcome, FakeProvider, FakeToolExecutor, ModelProvider, ProviderTurn, ProviderUsage,
    RecordBody, SessionError, SessionState, StableId, ToolAuthorizer, ToolBus, ToolError,
    ToolPlacement, ToolRegistry, ToolSpec, VecSink,
};

// Hard ceiling from `provider.rs` (`MAX_REQUEST_TIMEOUT_MS`); the constant is
// crate-internal there, so tests name the value with its provenance pinned
// here rather than drifting from the source.
const MAX_REQUEST_TIMEOUT_MS: u64 = 30_000;

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate admission semantics from
/// authorization. Production wiring installs the host capability-plus-consent
/// hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

fn ids() -> (
    bitty_ai_runtime::AgentInstanceId,
    bitty_ai_runtime::RunId,
    bitty_ai_runtime::SessionId,
) {
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    (issuer.agent_instance(), issuer.run(), issuer.session())
}

fn session() -> bitty_ai_runtime::AgentSession {
    let (agent, run, session) = ids();
    bitty_ai_runtime::AgentSession::new(agent, run, session)
}

fn read_tool_bus() -> ToolBus {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "workspace_read",
                "Read a bounded workspace path",
                br#"{"type":"object"}"#.to_vec(),
                "workspace.read",
                true,
                ToolPlacement::Mcp,
                None,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    ToolBus::new(registry).with_authorizer(AllowAll)
}

fn seed_record(provider: &str, id: &str) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: provider.to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: "seed".to_owned(),
        body: RecordBody::Inline(b"seed".to_vec()),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn final_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

fn config_with_timeout(timeout_ms: u64) -> AgentConfig {
    AgentConfig {
        provider_timeout_ms: timeout_ms,
        ..AgentConfig::default()
    }
}

fn agent_with_timeout(timeout_ms: u64) -> Agent<FakeProvider> {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("must never run"));
    Agent::new(
        provider,
        read_tool_bus(),
        session(),
        config_with_timeout(timeout_ms),
    )
}

/// The exact admission refusal the ceiling gate must return: `Denied`-style
/// (pre-contact refusal), never `Failed`/`EffectUnknown`/`ProtocolRejected`,
/// with the `TimeoutTooLarge` ceiling text as the reason.
fn expected_denial(timeout_ms: u64) -> ExecOutcome {
    ExecOutcome::Failed {
        error: AgentError::Tool(ToolError::Denied {
            name: "provider_timeout_ms".to_owned(),
            reason: format!("timeout {timeout_ms}ms exceeds ceiling {MAX_REQUEST_TIMEOUT_MS}ms"),
        }),
    }
}

fn assert_no_contact(agent: &mut Agent<FakeProvider>, executor: &FakeToolExecutor, sink: &VecSink) {
    assert_eq!(
        agent.provider_mut().complete_calls(),
        0,
        "admission refusal must make zero provider complete calls"
    );
    assert_eq!(
        agent.provider_mut().scripted_turns_remaining(),
        1,
        "admission refusal must not consume the scripted turn"
    );
    assert!(sink.is_empty(), "admission refusal must emit nothing");
    assert!(
        executor.calls().is_empty(),
        "admission refusal must not contact the executor"
    );
    assert!(
        agent.executions().is_empty(),
        "admission refusal attributes no execution"
    );
}

#[test]
fn zero_timeout_is_denied_before_provider_contact() {
    let mut agent = agent_with_timeout(0);
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert_eq!(outcome, expected_denial(0));
    assert!(
        matches!(
            outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::Denied { .. }),
            }
        ),
        "refusal must be Denied-style, got: {outcome:?}"
    );
    assert_no_contact(&mut agent, &executor, &sink);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

#[test]
fn over_ceiling_timeout_is_denied_before_provider_contact() {
    let mut agent = agent_with_timeout(MAX_REQUEST_TIMEOUT_MS + 1);
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert_eq!(outcome, expected_denial(MAX_REQUEST_TIMEOUT_MS + 1));
    assert_no_contact(&mut agent, &executor, &sink);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

#[test]
fn max_u64_timeout_is_denied_before_provider_contact() {
    let mut agent = agent_with_timeout(u64::MAX);
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert_eq!(outcome, expected_denial(u64::MAX));
    assert_no_contact(&mut agent, &executor, &sink);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

#[test]
fn ceiling_gate_fires_before_context_assembly() {
    // An assembly-failing seed (unknown provider) must not shadow the
    // timeout gate: admission order is timeout first, assembly second.
    let mut agent = agent_with_timeout(MAX_REQUEST_TIMEOUT_MS + 1);
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let bad_seeds = vec![seed_record("nope", "bad")];
    let outcome = agent.run_turn(
        &mut executor,
        "fake-chat",
        "hi",
        &bad_seeds,
        &mut sink,
        NOW_MS,
    );
    assert_eq!(outcome, expected_denial(MAX_REQUEST_TIMEOUT_MS + 1));
    assert_no_contact(&mut agent, &executor, &sink);
}

#[test]
fn session_lifecycle_gates_stay_first() {
    // A terminal session still reports `AlreadyTerminated` even with a
    // hostile timeout: the lifecycle gates precede the ceiling gate.
    let sess = session();
    sess.finish(false);
    assert_eq!(sess.state(), SessionState::Completed);
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("must never run"));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess,
        config_with_timeout(MAX_REQUEST_TIMEOUT_MS + 1),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert_eq!(
        outcome,
        ExecOutcome::Failed {
            error: AgentError::Session(SessionError::AlreadyTerminated {
                state: SessionState::Completed,
            }),
        }
    );
    assert_no_contact(&mut agent, &executor, &sink);
}

#[test]
fn boundary_timeouts_complete_unchanged() {
    // Good path byte-identical: lower bound, default, and ceiling all run
    // the scripted turn exactly as before the gate.
    for timeout_ms in [
        1,
        AgentConfig::default().provider_timeout_ms,
        MAX_REQUEST_TIMEOUT_MS,
    ] {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(final_turn("done"));
        let mut agent = Agent::new(
            provider,
            read_tool_bus(),
            session(),
            config_with_timeout(timeout_ms),
        );
        let mut executor = FakeToolExecutor::new();
        let mut sink = VecSink::new();
        let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
        assert_eq!(
            outcome,
            ExecOutcome::Completed {
                text: "done".to_owned(),
            },
            "timeout {timeout_ms} must complete unchanged"
        );
        assert_eq!(agent.provider_mut().complete_calls(), 1);
        assert_eq!(agent.session().state(), SessionState::Completed);
    }
}

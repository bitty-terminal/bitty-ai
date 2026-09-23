//! Per-turn budget resolution against the model context window (AI-0139).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns, scripted
//! [`FakeToolExecutor`] outcomes, caller-supplied `now_ms`. No network, no
//! secrets, no wall clock.
//!
//! The agent holds no registry, so it cannot look up the selected
//! registration at [`Agent::run_turn`] time: the honest in-crate window
//! source is the host-supplied [`AgentConfig::context_window_tokens`]
//! (mirrored from the selected registration at wiring time). The turn's
//! effective budget is `min(configured, window_tokens * 4)`; an unknown
//! window (`None` or `0`, mirroring
//! [`ModelRegistration::context_window_tokens`]) keeps the configured
//! ceiling unchanged. `run_turn` applies the same effective bound to both
//! context assembly (`ContextRequest.max_bytes`) and the provider request
//! (`TurnRequest.budget_bytes`).
//!
//! Freshness beyond `collected_at_ms` stays unenforced (out of scope).

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, FakeProvider,
    FakeToolExecutor, ModelProvider, ProviderError, ProviderTurn, ProviderUsage, ToolAuthorizer,
    ToolBus, ToolCallRequest, ToolExecutor, ToolRegistry, ToolSpec, VecSink,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate budget resolution from
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
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    ToolBus::new(registry).with_authorizer(AllowAll)
}

fn final_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

fn tool_call_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"a"}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

fn run(
    agent: &mut Agent<FakeProvider>,
    executor: &mut dyn ToolExecutor,
    prompt: &str,
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, &[], sink, NOW_MS)
}

#[test]
fn effective_budget_is_unit_min_with_unknown_passthrough() {
    // Pure unit coverage of the min() rule plus the unknown boundary:
    // `None` and `Some(0)` keep the configured ceiling unchanged.
    let configured = AgentConfig {
        context_budget_bytes: 1_024,
        ..AgentConfig::default()
    };
    assert_eq!(configured.effective_budget_bytes(), 1_024);

    let zero = AgentConfig {
        context_budget_bytes: 1_024,
        context_window_tokens: Some(0),
        ..AgentConfig::default()
    };
    assert_eq!(zero.effective_budget_bytes(), 1_024);

    // Window bound (tokens * 4) smaller than the configured ceiling wins.
    let smaller = AgentConfig {
        context_budget_bytes: 1_024,
        context_window_tokens: Some(100),
        ..AgentConfig::default()
    };
    assert_eq!(smaller.effective_budget_bytes(), 400);

    // Boundary: window bound exactly equal to the configured ceiling.
    let equal = AgentConfig {
        context_budget_bytes: 400,
        context_window_tokens: Some(100),
        ..AgentConfig::default()
    };
    assert_eq!(equal.effective_budget_bytes(), 400);

    // Window bound larger than the configured ceiling keeps the ceiling.
    let larger = AgentConfig {
        context_budget_bytes: 400,
        context_window_tokens: Some(1_000),
        ..AgentConfig::default()
    };
    assert_eq!(larger.effective_budget_bytes(), 400);
}

#[test]
fn smaller_window_wins_at_the_provider_boundary() {
    // Configured ceiling 1_024 with a 100-token window: the 400-byte
    // window bound wins, so a 500-byte prompt fails with BudgetExceeded
    // carrying the effective limit (not the configured 1_024).
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("never produced"));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            context_budget_bytes: 1_024,
            context_window_tokens: Some(100),
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, &"p".repeat(500), &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Provider(ProviderError::BudgetExceeded { limit: 400, .. }),
            }
        ),
        "window bound must win, got: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
}

#[test]
fn unknown_window_keeps_current_behavior_exactly() {
    // Both unknown spellings (`None`, `Some(0)`) preserve the pre-AI-0139
    // behavior byte-for-byte: the same 1_024 ceiling, so a 500-byte prompt
    // completes where the windowed run above fails.
    for window in [None, Some(0)] {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(final_turn("done"));
        let mut agent = Agent::new(
            provider,
            read_tool_bus(),
            session(),
            AgentConfig {
                context_budget_bytes: 1_024,
                context_window_tokens: window,
                ..AgentConfig::default()
            },
        );
        let mut executor = FakeToolExecutor::new();
        let mut sink = VecSink::new();

        let outcome = run(&mut agent, &mut executor, &"p".repeat(500), &mut sink);

        assert!(
            matches!(&outcome, ExecOutcome::Completed { .. }),
            "unknown window {window:?} must preserve behavior, got: {outcome:?}"
        );
        assert_eq!(agent.provider_mut().complete_calls(), 1);
    }
}

#[test]
fn boundary_window_equal_to_budget_behaves_like_unknown() {
    // Window 256 tokens derives exactly the 1_024-byte configured ceiling,
    // so the same prompt that completes under unknown passthrough completes
    // here too.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("done"));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            context_budget_bytes: 1_024,
            context_window_tokens: Some(256),
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, &"p".repeat(1_000), &mut sink);
    assert!(
        matches!(&outcome, ExecOutcome::Completed { .. }),
        "equal bound must admit the prompt, got: {outcome:?}"
    );
}

#[test]
fn narrower_window_stops_before_tool_dispatch() {
    // The same effective bound drives context assembly and the provider
    // pre-I/O check as one value: a small prompt under the 400-byte window
    // bound completes the provider round and the tool batch still dispatches
    // normally (the bound gates provider I/O, never the tool bus).
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_call_turn("calling"));
    provider.push_turn(final_turn("done"));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            // Window wins (100 tokens -> 400 bytes) over the 32 KiB default:
            // keep the configured ceiling large so the test pins the window
            // as the deciding bound.
            context_budget_bytes: 32 * 1024,
            context_window_tokens: Some(100),
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "small prompt", &mut sink);
    assert!(
        matches!(&outcome, ExecOutcome::Completed { .. }),
        "in-budget windowed turn must complete, got: {outcome:?}"
    );
    assert_eq!(executor.calls().len(), 1);
}

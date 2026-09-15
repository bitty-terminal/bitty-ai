//! Per-turn cost ceiling and overrun fuse (AI-0046).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns with explicit
//! [`ProviderUsage`], scripted [`FakeToolExecutor`] outcomes, caller-supplied
//! `now_ms`. No network, no secrets, no wall clock.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ContextPriority, ContextRecord,
    ExecOutcome, FakeProvider, FakeToolExecutor, ModelProvider, ProviderTurn, ProviderUsage,
    RecordBody, SessionState, StableId, StreamSink, ToolAuthorizer, ToolBus, ToolCallRequest,
    ToolRegistry, ToolSpec, ToolStatus, VecSink, estimate_cost,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate cost accounting from
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

fn seed_record(id: &str, summary: &str, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![b'x'; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn tool_turn(text: &str, input_tokens: u32, output_tokens: u32) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"a"}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn final_turn(text: &str, input_tokens: u32, output_tokens: u32) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn run(
    agent: &mut Agent<FakeProvider>,
    executor: &mut FakeToolExecutor,
    prompt: &str,
    seeds: &[ContextRecord],
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, seeds, sink, NOW_MS)
}

#[test]
fn no_ceiling_preserves_old_behavior() {
    // max_turn_cost None disables the fuse: a two-round tool turn completes
    // exactly as before AI-0046, while turn_cost still tracks observed usage.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("checking workspace", 7, 3));
    provider.push_turn(final_turn("done", 9, 2));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read 12 bytes", b"hello world!".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(
        &mut agent,
        &mut executor,
        "summarize",
        &[seed_record("s1", "manifest", 10)],
        &mut sink,
    );

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert_eq!(agent.session().state(), SessionState::Completed);
    assert_eq!(executor.calls().len(), 1);
    // Default weights are 1/1: cost is the exact reported token sum.
    assert_eq!(agent.turn_cost(), 7 + 3 + 9 + 2);
    assert_eq!(agent.turn_cost(), estimate_cost(16, 5, 1, 1));
}

#[test]
fn ceiling_trips_before_any_dispatch_and_leaves_session_active() {
    // One over-ceiling round with a tool call: the fuse stops the turn
    // before authorization/dispatch, the session stays Active for
    // reconcile-and-retry, and nothing is dispatched.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("expensive plan", 10, 0));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(5),
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never dispatch", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 5,
                    actual: 10
                }
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(agent.session().state(), SessionState::Active);
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.turn_cost(), 10);
    // The over-budget round streams nothing: no dispatch, no text emission.
    assert!(sink.chunks().is_empty());
}

#[test]
fn exact_accounting_across_rounds_keeps_earlier_effects() {
    // Weights 2/3: round one costs 4*2 = 8 (under the 20 ceiling, so its
    // tool dispatches); round two adds 5*3 = 15 for 23 total and trips the
    // fuse with the exact sum. Earlier recorded effects are kept, the
    // overrun round dispatches nothing further, no rollback happens.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("round one", 4, 0));
    provider.push_turn(tool_turn("round two", 0, 5));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(20),
            input_cost_weight: 2,
            output_cost_weight: 3,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("first ok", b"data".to_vec());
    executor.push_success("must never dispatch", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "go", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 20,
                    actual: 23
                }
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(agent.turn_cost(), 23);
    assert_eq!(agent.session().state(), SessionState::Active);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    // Round one dispatched exactly once; the overrun round added nothing.
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
}

#[test]
fn unreported_usage_falls_back_to_byte_estimate() {
    // Zero usage means unreported, not free: the loop estimates input tokens
    // from request bytes and output tokens from response text, so an
    // uncalibrated provider cannot bypass the ceiling by reporting nothing.
    // Prompt "hi" (2 bytes -> 1 token) + text "hey" (3 bytes -> 1 token)
    // costs exactly 2 at default weights.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "hey".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(1),
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 1,
                    actual: 2
                }
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(agent.turn_cost(), 2);
    assert_eq!(agent.session().state(), SessionState::Active);
}

#[test]
fn zero_configured_weights_count_as_one() {
    // A `0` (unset) configured weight counts as `1` in turn accounting so an
    // uncalibrated host cannot bypass the ceiling with free rounds.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("done", 5, 5));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(9),
            input_cost_weight: 0,
            output_cost_weight: 0,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 9,
                    actual: 10
                }
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(agent.session().state(), SessionState::Active);
}

#[test]
fn cost_fuse_is_deterministic() {
    // Same script plus same now_ms replays to the identical typed outcome.
    fn once() -> (ExecOutcome, u64, SessionState) {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(tool_turn("round one", 4, 0));
        provider.push_turn(tool_turn("round two", 0, 5));
        let mut agent = Agent::new(
            provider,
            read_tool_bus(),
            session(),
            AgentConfig {
                max_turn_cost: Some(20),
                input_cost_weight: 2,
                output_cost_weight: 3,
                ..AgentConfig::default()
            },
        );
        let mut executor = FakeToolExecutor::new();
        executor.push_success("first ok", b"data".to_vec());
        let mut sink = VecSink::new();
        let outcome = run(&mut agent, &mut executor, "go", &[], &mut sink);
        (outcome, agent.turn_cost(), agent.session().state())
    }
    let (first, cost_first, state_first) = once();
    let (second, cost_second, state_second) = once();
    assert_eq!(first, second);
    assert_eq!(cost_first, cost_second);
    assert_eq!(state_first, state_second);
    assert!(matches!(
        &first,
        ExecOutcome::Failed {
            error: AgentError::CostCeilingExceeded { .. }
        }
    ));
}

//! Run-turn lifecycle regression tests (AI-0057, review 07 P1-1/P1-2).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns with explicit
//! [`ProviderUsage`], scripted [`FakeToolExecutor`] outcomes, caller-supplied
//! `now_ms`. No network, no secrets, no wall clock.
//!
//! Covers:
//! - P1-1: `executions` is per-turn (cleared at `run_turn` entry) with
//!   `turn_cost` reset, bounded by [`MAX_EXECUTIONS_PER_AGENT`].
//! - P1-2: `Completed`/`Failed` sessions reject a second `run_turn` with
//!   `Failed(Session(AlreadyTerminated))` and no further I/O.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ContextRecord, ExecOutcome,
    FakeProvider, FakeToolExecutor, MAX_EXECUTIONS_PER_AGENT, ModelProvider, ProviderTurn,
    ProviderUsage, SessionError, SessionState, ToolAuthorizer, ToolBus, ToolCallRequest, ToolError,
    ToolExecutor, ToolRegistry, ToolSpec, ToolStatus, VecSink,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate lifecycle semantics from
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
    executor: &mut dyn ToolExecutor,
    prompt: &str,
    seeds: &[ContextRecord],
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, seeds, sink, NOW_MS)
}

#[test]
fn executions_are_per_turn_and_turn_cost_resets() {
    // P1-1: same Agent, two turns where the first leaves the session Active
    // (Unknown) so the second is allowed. The second turn must see only its
    // own executions and cost, never cumulative history.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("first", 10, 0));
    provider.push_turn(tool_turn("second", 4, 0));
    provider.push_turn(final_turn("done", 3, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "ack lost".to_owned(),
    });
    executor.push_success("second ok", b"data".to_vec());

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &[], &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must leave the session Active as Unknown, got: {first:?}"
    );
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
    assert_eq!(agent.session().state(), SessionState::Active);
    assert_eq!(agent.turn_cost(), 10);

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "two", &[], &mut sink2);
    assert!(
        matches!(second, ExecOutcome::Completed { .. }),
        "second turn must complete, got: {second:?}"
    );
    // Per-turn semantics: only the second turn's dispatch is visible, not
    // the first turn's Unknown plus the second (which would be 2 on base).
    assert_eq!(
        agent.executions().len(),
        1,
        "executions must reset per turn, got: {:?}",
        agent.executions()
    );
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    // Cost resets per turn: 4 + 3 = 7, never 10 + 7 = 17.
    assert_eq!(agent.turn_cost(), 7);
    assert_eq!(agent.tool_records().len(), 1);
    assert_eq!(agent.session().state(), SessionState::Completed);
}

#[test]
fn completed_session_rejects_second_turn() {
    // P1-2: a Completed session must reject re-entry with a typed
    // AlreadyTerminated error and perform no further I/O.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("reading", 1, 0));
    provider.push_turn(final_turn("done", 1, 0));
    // Trailing script that must never run once the session is Completed.
    provider.push_turn(tool_turn("must never run", 1, 0));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let first = run(&mut agent, &mut executor, "hi", &[], &mut sink);
    assert!(
        matches!(first, ExecOutcome::Completed { .. }),
        "first turn must complete, got: {first:?}"
    );
    assert_eq!(sess.state(), SessionState::Completed);
    assert_eq!(agent.executions().len(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    assert_eq!(executor.calls().len(), 1);

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "again", &[], &mut sink2);
    assert!(
        matches!(
            &second,
            ExecOutcome::Failed {
                error: AgentError::Session(SessionError::AlreadyTerminated {
                    state: SessionState::Completed,
                }),
            }
        ),
        "second run_turn on Completed must be rejected, got: {second:?}"
    );
    // Rejection preserves the terminal state and performs no I/O: no new
    // provider round, no dispatch, no history mutation.
    assert_eq!(sess.state(), SessionState::Completed);
    assert_eq!(agent.executions().len(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(executor.calls().len(), 1);
    assert!(sink2.is_empty());
}

#[test]
fn failed_session_rejects_second_turn() {
    // P1-2: a Failed session must reject re-entry the same way; retries take
    // the explicit reconcile path, never an implicit second run_turn.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "calling ghost".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "ghost_tool".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(tool_turn("must never run", 1, 0));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let first = run(&mut agent, &mut executor, "hi", &[], &mut sink);
    assert!(
        matches!(
            &first,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::UnknownTool { .. }),
            }
        ),
        "first turn must fail as UnknownTool, got: {first:?}"
    );
    assert_eq!(sess.state(), SessionState::Failed);
    assert!(agent.executions().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 1);

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "again", &[], &mut sink2);
    assert!(
        matches!(
            &second,
            ExecOutcome::Failed {
                error: AgentError::Session(SessionError::AlreadyTerminated {
                    state: SessionState::Failed,
                }),
            }
        ),
        "second run_turn on Failed must be rejected, got: {second:?}"
    );
    assert_eq!(sess.state(), SessionState::Failed);
    assert!(agent.executions().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert!(executor.calls().is_empty());
    assert!(sink2.is_empty());
}

#[test]
fn executions_cap_bounds_max_bus_turn() {
    // P1-1 cap: the absolute ceiling exists (32 = 4 rounds x 8 calls) and a
    // max-bus turn (8 dispatches, the most ToolBus allows per turn) stays
    // within it. The per-turn clear is re-verified at max load: a second
    // allowed turn (first leaves Active via Unknown on its 8th dispatch)
    // sees only its own 8, never 16.
    assert_eq!(MAX_EXECUTIONS_PER_AGENT, 32);
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    // First turn: 8 calls in one round; the executor makes the 8th Unknown
    // so the session stays Active for a second turn.
    provider.push_turn(ProviderTurn {
        text: "eight".to_owned(),
        tool_calls: (0..8)
            .map(|_| ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            })
            .collect(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    // Second turn: 8 fresh calls then completion.
    provider.push_turn(ProviderTurn {
        text: "eight again".to_owned(),
        tool_calls: (0..8)
            .map(|_| ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            })
            .collect(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(final_turn("done", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    for _ in 0..7 {
        executor.push_success("ok", b"data".to_vec());
    }
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "ack lost on 8th".to_owned(),
    });
    for _ in 0..8 {
        executor.push_success("ok", b"data".to_vec());
    }

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &[], &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first max-bus turn must end Unknown, got: {first:?}"
    );
    assert_eq!(agent.executions().len(), 8);
    assert!(agent.executions().len() <= MAX_EXECUTIONS_PER_AGENT);
    assert_eq!(agent.session().state(), SessionState::Active);

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "two", &[], &mut sink2);
    assert!(
        matches!(second, ExecOutcome::Completed { .. }),
        "second max-bus turn must complete, got: {second:?}"
    );
    assert_eq!(
        agent.executions().len(),
        8,
        "max-bus second turn must see only its own 8, got: {:?}",
        agent.executions()
    );
    assert!(agent.executions().len() <= MAX_EXECUTIONS_PER_AGENT);
    assert_eq!(executor.calls().len(), 16);
    assert_eq!(agent.session().state(), SessionState::Completed);
}

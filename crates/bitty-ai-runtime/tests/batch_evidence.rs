//! Batch-call evidence for the 012 claim surface (AI-0086).
//!
//! Pin what "batch" means in the runtime today, end to end through the public
//! API: one assistant turn's `tool_calls` vector is a **transactional batch**
//! (all-or-nothing precheck), executed as a **sequential loop** with a
//! per-call cancel check before every dispatch, under a fail-closed
//! call-limit enforced at two points (config gate in the agent, constant
//! gate in the bus).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns, scripted
//! [`FakeToolExecutor`] outcomes, caller-supplied `now_ms`. No network, no
//! secrets, no wall clock, no threads.
//!
//! Non-overlap with neighbors (this file extends only the untouched halves):
//! - `tool.rs` unit tests: single-call precheck shape, over-limit burst on
//!   the bus alone, hook re-check at the dispatch boundary. Nothing here
//!   re-asserts single-call shapes.
//! - `runtime_fail_closed.rs`: burst-over-limit and relaxed-config clamp
//!   through `run_turn` (9 calls, both fail with nothing dispatched); cancel
//!   before/after dispatch (0/1 shapes); S-1 remaining-write denial after a
//!   mid-turn rotation. Nothing here re-asserts the 9-call burst, the
//!   clamp-at-8, the 0/1 cancel shapes, or the remaining-write denial.
//! - `cache_invalidation.rs`: remaining-*read* continuation under the
//!   downgraded tier (the S-1 untouched half for tier downgrade). Nothing
//!   here re-asserts grant-invalidation visibility.
//! - `result_schema_disclosure.rs`: cancel-after-two-dispatches counting
//!   (`dispatched: 2`). Nothing here re-asserts the 2-dispatch count shape.
//! - `agent_turn_semantics.rs`: cancel between provider rounds, multi-turn
//!   continuation. Nothing here re-asserts round-boundary or resume shapes.
//!
//! Covered here:
//! 1. All-or-nothing precheck at the batch level: a mid-batch refusal
//!    (second of three calls unknown) fails the turn with nothing dispatched.
//! 2. Order within the batch: the precheck authorizes calls in batch order
//!    (first error wins), and dispatch executes in batch order.
//! 3. Per-call cancel visibility inside the batch loop: cancel landing after
//!    the first of three dispatches stops the batch before the second
//!    dispatch, keeps the dispatched effect, and reports `dispatched: 1`.
//! 4. Call-limit fail-closed at the exact boundary: 8 calls in one batch
//!    dispatch fully (the most the bus allows); the 9th call in the same
//!    batch via the per-call counter path is unreachable through `run_turn`
//!    (the precheck gate fires first), so the per-call counter is pinned
//!    directly on the bus.
//! 5. No parallelism: the batch dispatches strictly sequentially — executor
//!    call order matches batch order and every call is recorded before the
//!    next begins (single-threaded, std-only, no spawn/join in the loop).

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AgentLevel, AuthBase, AuthContext, AuthDecision, ExecOutcome,
    FakeProvider, FakeToolExecutor, IdIssuer, ModelProvider, ProviderTurn, ProviderUsage,
    SessionState, ToolAuthorizer, ToolBus, ToolCall, ToolCallRequest, ToolError, ToolExecutor,
    ToolRegistry, ToolSpec, ToolStatus, VecSink,
};

/// Hard bus ceiling mirrored from the `TB-6` contract (`tool.rs`). Asserted
/// here as a literal (not imported) because the constant is intentionally
/// crate-private to the bus boundary; the burst test below pins the exact
/// value the bus reports.
const BUS_CALL_CAP: usize = 8;

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate batch semantics from
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
    let mut issuer = IdIssuer::default();
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

fn read_call(arguments: &[u8]) -> ToolCallRequest {
    ToolCallRequest {
        name: "workspace_read".to_owned(),
        arguments: arguments.to_vec(),
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

fn base() -> AuthBase {
    let (agent_instance_id, _, session_id) = ids();
    AuthBase {
        agent_instance_id,
        session_id,
        level: AgentLevel::Inspect,
    }
}

#[test]
fn batch_precheck_is_all_or_nothing_mid_batch_refusal_dispatches_nothing() {
    // `ToolBus::precheck` validates every call before any dispatch: a batch
    // of [valid, unknown, valid] must fail as `UnknownTool` with zero
    // dispatches, zero executions, and a failed session.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "three reads".to_owned(),
        tool_calls: vec![
            read_call(br#"{"path":"a"}"#),
            ToolCallRequest {
                name: "ghost_tool".to_owned(),
                arguments: br#"{}"#.to_vec(),
            },
            read_call(br#"{"path":"c"}"#),
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never dispatch", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::UnknownTool { name })
            } if name == "ghost_tool"
        ),
        "mid-batch refusal must fail the turn as UnknownTool, got: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
    assert_eq!(agent.session().state(), SessionState::Failed);
    assert_eq!(agent.provider_mut().complete_calls(), 1);
}

#[test]
fn batch_dispatch_executes_in_call_order() {
    // Three valid calls in one batch must dispatch strictly in batch order
    // with distinct arguments, then complete on the follow-up round.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "three reads".to_owned(),
        tool_calls: vec![
            read_call(br#"{"path":"a"}"#),
            read_call(br#"{"path":"b"}"#),
            read_call(br#"{"path":"c"}"#),
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "done".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("a ok", b"a".to_vec());
    executor.push_success("b ok", b"b".to_vec());
    executor.push_success("c ok", b"c".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &mut sink);

    assert!(
        matches!(outcome, ExecOutcome::Completed { .. }),
        "ordered batch must complete, got: {outcome:?}"
    );
    let arguments: Vec<Vec<u8>> = executor
        .calls()
        .iter()
        .map(|(_, arguments)| arguments.clone())
        .collect();
    assert_eq!(
        arguments,
        vec![
            br#"{"path":"a"}"#.to_vec(),
            br#"{"path":"b"}"#.to_vec(),
            br#"{"path":"c"}"#.to_vec(),
        ]
    );
    assert_eq!(agent.executions().len(), 3);
    assert!(
        agent
            .executions()
            .iter()
            .all(|record| matches!(record.status, ToolStatus::Success))
    );
}

#[test]
fn batch_loop_checks_cancel_before_every_dispatch() {
    // Cancel landing during the first of three dispatches must stop the batch
    // before the second dispatch: one host call, one kept execution, and a
    // reconciled `Canceled { dispatched: 1 }` with no follow-up round.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "three reads".to_owned(),
        tool_calls: vec![
            read_call(br#"{"path":"a"}"#),
            read_call(br#"{"path":"b"}"#),
            read_call(br#"{"path":"c"}"#),
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let sess = session();
    let mut executor = FakeToolExecutor::new().with_cancel_on_call(sess.clone(), 1);
    executor.push_success("first ok", b"a".to_vec());
    executor.push_success("must never dispatch", b"b".to_vec());
    let mut agent = Agent::new(provider, read_tool_bus(), sess, AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &mut sink);

    assert!(
        matches!(
            outcome,
            ExecOutcome::Canceled {
                dispatched: 1,
                unknown: 0
            }
        ),
        "mid-batch cancel must stop after the first dispatch, got: {outcome:?}"
    );
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
}

#[test]
fn batch_at_exact_cap_dispatches_fully() {
    // Eight calls (exactly `MAX_TOOL_CALLS_PER_TURN`) in one batch must pass
    // both the agent config gate and the bus batch gate, dispatch all eight,
    // and complete on the follow-up round.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "eight reads".to_owned(),
        tool_calls: (0..8).map(|_| read_call(br#"{"path":"a"}"#)).collect(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "done".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    for _ in 0..8 {
        executor.push_success("ok", b"data".to_vec());
    }
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &mut sink);

    assert!(
        matches!(outcome, ExecOutcome::Completed { .. }),
        "exact-cap batch must complete, got: {outcome:?}"
    );
    assert_eq!(executor.calls().len(), 8);
    assert_eq!(agent.executions().len(), 8);
}

#[test]
fn bus_per_call_counter_is_fail_closed_past_the_cap() {
    // The `ToolBus::dispatch` per-call counter (`calls_this_turn >=
    // MAX_TOOL_CALLS_PER_TURN`) stops the 9th dispatch even when the batch
    // gate is bypassed by calling `dispatch` directly: through `run_turn`
    // this path is unreachable (precheck fires first), so it is pinned here
    // on the bus. Earlier dispatches stay recorded; the counter is unchanged
    // by the refusal.
    let mut bus = read_tool_bus();
    let mut executor = FakeToolExecutor::new();
    for _ in 0..8 {
        executor.push_success("ok", b"data".to_vec());
    }
    let call = ToolCall {
        name: "workspace_read".to_owned(),
        arguments: br#"{}"#.to_vec(),
    };
    let mut issuer = IdIssuer::default();
    for _ in 0..8 {
        bus.dispatch(&mut executor, &call, &base(), issuer.execution(), NOW_MS)
            .expect("first eight dispatches fit the cap");
    }
    assert_eq!(bus.calls_this_turn(), 8);
    let error = bus
        .dispatch(&mut executor, &call, &base(), issuer.execution(), NOW_MS)
        .expect_err("ninth dispatch must fail closed");
    assert_eq!(
        error,
        ToolError::CallLimitExceeded {
            limit: BUS_CALL_CAP
        }
    );
    assert_eq!(executor.calls().len(), 8);
    assert_eq!(bus.calls_this_turn(), 8);
}

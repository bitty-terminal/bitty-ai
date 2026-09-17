//! Pending unresolved-effect attribution across turns (AI-0098, AI-RUN-001, #186).
//!
//! `Agent::run_turn` admission clears the per-turn `executions` vector, but
//! `Unknown` attribution must survive in the bounded pending set so a later
//! turn cannot silently drop the reconcile lookup. Resolution (terminal
//! answer) or explicit [`Agent::abandon_pending`] removes entries; escalation
//! captures the pending tool into the report without leaking per-turn counts.
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns with explicit
//! [`ProviderUsage`], scripted [`FakeToolExecutor`] outcomes, scripted
//! [`FakeReconciler`] answers, caller-supplied `now_ms`. No network, no
//! secrets, no wall clock, no threads. All assert messages are static
//! literals (no identifier/reason interpolation).
//!
//! Non-overlap: `agent_turn_semantics.rs` pins per-turn reset and the
//! stale-id queryable update; `unknown_reconcile.rs` pins the query schedule,
//! budget separation, and clock-advance contract. This file pins only the
//! admission boundary: survival, resolution-removal, abandonment,
//! escalation attribution, bound refusal, and unrelated-turn queryability.

use bitty_ai_runtime::{
    Agent, AgentConfig, AuthContext, AuthDecision, ExecOutcome, FakeProvider, FakeReconciler,
    FakeToolExecutor, ModelProvider, ProviderTurn, ProviderUsage, ReconcileOutcome, SessionState,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolRegistry, ToolSpec, ToolStatus,
    VecSink,
};

const NOW_MS: u64 = 1_700_000_000_000;

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

fn unknown_error(reason: &str) -> ToolError {
    ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: reason.to_owned(),
    }
}

fn run<P: ModelProvider>(
    agent: &mut Agent<P>,
    executor: &mut FakeToolExecutor,
    prompt: &str,
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, &[], sink, NOW_MS)
}

#[test]
fn pending_survives_admission_and_resolves_without_reexecution() {
    // Turn one ends Unknown; turn two is admitted (unrelated prompt);
    // turn one's id must stay queryable and resolve without re-execution.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("first", 5, 0));
    provider.push_turn(final_turn("second done", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown"
    );
    let stale_id = agent.executions()[0].execution_id;

    // An unrelated second turn is admitted and completes; the pending set
    // must still hold the first turn's attribution.
    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "unrelated", &mut sink2);
    assert!(
        matches!(second, ExecOutcome::Completed { .. }),
        "second turn must complete"
    );
    assert_eq!(agent.pending_unknown().len(), 1);
    assert_eq!(agent.pending_unknown()[0].execution_id, stale_id);

    // Reconciling the stale id queries once and removes the entry, with no
    // executor contact (no re-execution).
    let calls_before = executor.calls().len();
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, stale_id, NOW_MS);
    assert!(
        matches!(
            &resolved,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 1,
                ..
            }
        ),
        "stale id must resolve after admission"
    );
    assert_eq!(reconciler.query_count(), 1);
    assert_eq!(executor.calls().len(), calls_before);
    assert!(agent.pending_unknown().is_empty());
}

#[test]
fn abandon_pending_removes_entry_without_reconcile() {
    // Explicit attributed abandonment stops queryability without any
    // reconciler contact.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("first", 5, 0));
    provider.push_turn(final_turn("second done", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));

    let mut sink = VecSink::new();
    let _ = run(&mut agent, &mut executor, "one", &mut sink);
    let stale_id = agent.executions()[0].execution_id;
    let mut sink2 = VecSink::new();
    let _ = run(&mut agent, &mut executor, "two", &mut sink2);
    assert_eq!(agent.pending_unknown().len(), 1);

    assert!(agent.abandon_pending(stale_id));
    assert!(agent.pending_unknown().is_empty());
    // Abandoning twice reports absence; reconciling finds nothing.
    assert!(!agent.abandon_pending(stale_id));
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let gone = agent.reconcile_unknown(&mut reconciler, stale_id, NOW_MS);
    assert_eq!(gone, ReconcileOutcome::NoUnknown);
    assert_eq!(reconciler.query_count(), 0);
}

#[test]
fn escalation_captures_pending_tool_without_per_turn_counts() {
    // A stale id that exhausts the reconcile budget escalates with its own
    // tool attribution; per-turn `dispatched` counts stay local to the
    // current turn (no cross-turn leak). The session fails without any
    // further dispatch.
    let config = AgentConfig {
        max_unknown_retries: 1,
        ..AgentConfig::default()
    };
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("first", 5, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), config);
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown"
    );
    let stale_id = agent.executions()[0].execution_id;

    // Carry the attribution across an admission boundary first, so the
    // escalation exercises the pending-set path rather than the
    // current-turn vector. The second turn ends Unknown (never Completed:
    // completion is session-terminal), so the session stays Active and the
    // first turn's id moves into the pending set.
    agent.provider_mut().push_turn(tool_turn("second", 1, 0));
    executor.push_error(unknown_error("ack lost again"));
    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "two", &mut sink2);
    assert!(
        matches!(second, ExecOutcome::Unknown { .. }),
        "second turn must end Unknown"
    );
    assert_eq!(agent.session().state(), SessionState::Active);
    assert_eq!(agent.pending_unknown().len(), 1);
    assert_eq!(agent.pending_unknown()[0].execution_id, stale_id);

    let calls_before = executor.calls().len();
    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still uncertain");
    let escalated = agent.reconcile_unknown(&mut reconciler, stale_id, NOW_MS);
    let report = match &escalated {
        ReconcileOutcome::Escalated(report) => report,
        _other => panic!("stale id must escalate on exhausted budget"),
    };
    assert_eq!(report.tool, "workspace_read");
    assert_eq!(report.attempts, 1);
    // The report's dispatched count is the current turn's own vector length
    // (exactly this turn's single dispatch), never an accumulation across
    // turns.
    assert_eq!(report.dispatched, 1);
    assert_eq!(executor.calls().len(), calls_before);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

#[test]
fn pending_preserved_across_many_unknown_turns() {
    // Each Unknown turn keeps the session Active; every turn's id must stay
    // queryable in the pending set across later admissions, and each must
    // resolve independently without re-execution. The executor script holds
    // one error per turn (FIFO); the provider script one tool turn per turn.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    for _ in 0..3 {
        provider.push_turn(tool_turn("maybe wrote", 5, 0));
    }
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    for _ in 0..3 {
        executor.push_error(unknown_error("ack lost"));
    }

    let mut ids = Vec::new();
    for index in 0..3 {
        let prompt = format!("task {index}");
        let mut turn_sink = VecSink::new();
        let outcome = run(&mut agent, &mut executor, &prompt, &mut turn_sink);
        assert!(
            matches!(outcome, ExecOutcome::Unknown { .. }),
            "each turn must end Unknown"
        );
        assert_eq!(agent.session().state(), SessionState::Active);
        // Per-turn vector holds exactly this turn's dispatch (fresh id).
        assert_eq!(agent.executions().len(), 1);
        ids.push(agent.executions()[0].execution_id);
        // Admission carried every earlier Unknown id into the pending set:
        // after turn N the set holds exactly the N earlier ids (turns are
        // 1-based here; the current turn's own record joins at the next
        // admission).
        assert_eq!(agent.pending_unknown().len(), index);
        for (position, id) in ids.iter().take(index).enumerate() {
            assert_eq!(agent.pending_unknown()[position].execution_id, *id);
        }
    }
    // One final admission carries the last turn's id as well.
    agent.provider_mut().push_turn(final_turn("wrap up", 1, 0));
    let mut wrap_sink = VecSink::new();
    let wrapped = run(&mut agent, &mut executor, "wrap", &mut wrap_sink);
    assert!(
        matches!(wrapped, ExecOutcome::Completed { .. }),
        "wrap-up turn must complete"
    );
    assert_eq!(agent.pending_unknown().len(), 3);

    // Resolve out of order; each query touches the reconciler once and the
    // executor never sees another call (no re-execution).
    let calls_before = executor.calls().len();
    let mut reconciler = FakeReconciler::new();
    for id in [ids[2], ids[0], ids[1]] {
        reconciler.push_resolved(ToolStatus::Success);
        let resolved = agent.reconcile_unknown(&mut reconciler, id, NOW_MS);
        assert!(
            matches!(
                &resolved,
                ReconcileOutcome::Resolved {
                    status: ToolStatus::Success,
                    attempts: 1,
                    ..
                }
            ),
            "every carried id must resolve"
        );
    }
    assert_eq!(reconciler.query_count(), 3);
    assert_eq!(executor.calls().len(), calls_before);
    assert!(agent.pending_unknown().is_empty());
}

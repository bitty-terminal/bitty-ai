//! Unknown-outcome reconcile protocol (AI-0047).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns drive the
//! initial `Unknown`, scripted [`FakeToolExecutor`] outcomes produce
//! `EffectUnknown`, and scripted [`FakeReconciler`] answers drive the
//! reconcile queries. Caller-supplied `now_ms` throughout: no wall clock,
//! no sleep, no network, no secrets.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, ExecutionId,
    FakeProvider, FakeReconciler, FakeToolExecutor, ModelProvider, ProviderTurn, ProviderUsage,
    ReconcileOutcome, ReconcileStatus, RecordingExecutor, SessionState, ToolAuthorizer, ToolBus,
    ToolCallRequest, ToolError, ToolRegistry, ToolSpec, ToolStatus, UnknownReconciler, VecSink,
    reconcile_delay_ms,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate reconcile accounting from
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

fn tool_turn(text: &str) -> ProviderTurn {
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

/// Drive one turn to `Unknown` via a scripted `EffectUnknown` executor
/// outcome. Returns the agent with one recorded `Unknown` execution.
fn drive_to_unknown(config: AgentConfig) -> (Agent<FakeProvider>, FakeToolExecutor) {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote"));
    // Unreached while the effect stays unreconciled: the turn stops at
    // the first `Unknown`.
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "host crashed before ack".to_owned(),
    });
    let session = session();
    let mut agent = Agent::new(provider, read_tool_bus(), session, config);
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
    // Session stays usable for reconcile-and-retry (not failed).
    assert_eq!(agent.session().state(), SessionState::Active);
    (agent, executor)
}

#[test]
fn unknown_resolves_without_effect_reexecution() {
    let (mut agent, executor) = drive_to_unknown(AgentConfig::default());
    let execution_id = agent.executions()[0].execution_id;
    assert_eq!(executor.calls().len(), 1);

    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still writing");
    reconciler.push_resolved(ToolStatus::Success);
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

    assert!(
        matches!(
            &outcome,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 2,
                ..
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    if let ReconcileOutcome::Resolved { delays_ms, .. } = &outcome {
        assert_eq!(*delays_ms, vec![100, 200]);
    }
    // Reconcile queried status only: the effect never re-executed.
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(reconciler.query_count(), 2);
    assert_eq!(
        reconciler.queries()[0],
        ("workspace_read".to_owned(), execution_id)
    );
    // The recorded status updated in place; the session stays usable.
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(agent.session().state(), SessionState::Active);
    // Reconcile performed no provider round either.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
}

#[test]
fn unknown_escalates_after_bounded_retries_with_typed_report() {
    let config = AgentConfig {
        max_unknown_retries: 3,
        unknown_reconcile_base_delay_ms: 100,
        unknown_reconcile_max_delay_ms: 250,
        ..AgentConfig::default()
    };
    let (mut agent, executor) = drive_to_unknown(config);
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

    let report = match &outcome {
        ReconcileOutcome::Escalated(report) => report,
        other => panic!("unexpected outcome: {other:?}"),
    };
    // Bounded retry count with deterministic backoff ceilings.
    assert_eq!(report.attempts, 3);
    assert_eq!(report.delays_ms, vec![100, 200, 250]);
    // Typed escalation content (not a bare string).
    assert_eq!(report.tool, "workspace_read");
    assert_eq!(report.reason, "still writing");
    assert_eq!(report.dispatched, 1);
    // Bounded work: exactly the budget ran, and the effect never re-executed.
    assert_eq!(reconciler.query_count(), 3);
    assert_eq!(executor.calls().len(), 1);
    // Fail-closed: the session is marked failed.
    assert_eq!(agent.session().state(), SessionState::Failed);
    // The typed report converts to the typed error with identical fields.
    let error = AgentError::from(report.clone());
    assert_eq!(
        error,
        AgentError::UnknownUnresolved {
            tool: "workspace_read".to_owned(),
            reason: "still writing".to_owned(),
            attempts: 3,
            dispatched: 1,
        }
    );
    assert!(error.to_string().contains("workspace_read"));
}

#[test]
fn retry_budget_is_separate_from_tool_call_budget() {
    // Tool-call budget is exhausted (one dispatch of one allowed), yet the
    // separate reconcile budget still permits five status queries.
    let config = AgentConfig {
        max_tool_calls_per_turn: 1,
        max_unknown_retries: 5,
        ..AgentConfig::default()
    };
    assert_ne!(
        config.max_unknown_retries, config.max_tool_calls_per_turn,
        "budgets must differ in this test to prove separation"
    );
    let (mut agent, executor) = drive_to_unknown(config);
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    for _ in 0..5 {
        reconciler.push_pending("still uncertain");
    }
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

    assert!(matches!(outcome, ReconcileOutcome::Escalated(_)));
    assert_eq!(reconciler.query_count(), 5);
    // The exhausted tool-call budget did not cap queries, and no dispatch
    // happened during reconcile.
    assert_eq!(executor.calls().len(), 1);
}

#[test]
fn backoff_schedule_is_deterministic() {
    fn once() -> ReconcileOutcome {
        let (mut agent, _executor) = drive_to_unknown(AgentConfig::default());
        let execution_id = agent.executions()[0].execution_id;
        let mut reconciler = FakeReconciler::new();
        reconciler.push_pending("still writing");
        reconciler.push_resolved(ToolStatus::Denied {
            reason: "host refused on recheck".to_owned(),
        });
        agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS)
    }
    let first = once();
    let second = once();
    assert_eq!(first, second);
    match &first {
        ReconcileOutcome::Resolved {
            status,
            attempts,
            delays_ms,
        } => {
            assert_eq!(
                *status,
                ToolStatus::Denied {
                    reason: "host refused on recheck".to_owned(),
                }
            );
            assert_eq!(*attempts, 2);
            let expected = vec![
                reconcile_delay_ms(0, 100, 5_000),
                reconcile_delay_ms(1, 100, 5_000),
            ];
            assert_eq!(*delays_ms, expected);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

/// Time-dependent reconciler: reports `Pending` until the caller-supplied
/// `now_ms` reaches `deadline_ms`, then resolves. It records every observed
/// instant so the test can assert exactly when the driver queried it.
struct DeadlineReconciler {
    deadline_ms: u64,
    nows: Vec<u64>,
}

impl DeadlineReconciler {
    fn new(deadline_ms: u64) -> Self {
        Self {
            deadline_ms,
            nows: Vec::new(),
        }
    }
}

impl UnknownReconciler for DeadlineReconciler {
    fn reconcile(
        &mut self,
        _tool: &str,
        _execution_id: ExecutionId,
        now_ms: u64,
    ) -> ReconcileStatus {
        self.nows.push(now_ms);
        if now_ms >= self.deadline_ms {
            ReconcileStatus::Resolved(ToolStatus::Success)
        } else {
            ReconcileStatus::Pending {
                reason: "not durable yet".to_owned(),
            }
        }
    }
}

/// Clock-advance contract: `reconcile_unknown` never advances the clock, so a
/// time-dependent reconciler stays pending for the whole frozen-clock
/// invocation; the caller must advance `now_ms` by the reported `delays_ms`
/// and re-invoke. Deterministic: no wall clock, no sleep.
#[test]
fn caller_clock_advance_by_reported_delays_enables_resolution() {
    let config = AgentConfig {
        max_unknown_retries: 2,
        unknown_reconcile_base_delay_ms: 100,
        unknown_reconcile_max_delay_ms: 1_000,
        ..AgentConfig::default()
    };
    let (mut agent, _executor) = drive_to_unknown(config);
    let execution_id = agent.executions()[0].execution_id;

    // Deadline equals the last scheduled retry instant (NOW_MS + 200): the
    // reconciler cannot finish while the clock is frozen.
    let mut reconciler = DeadlineReconciler::new(NOW_MS + 200);

    let frozen = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    let report = match &frozen {
        ReconcileOutcome::Escalated(report) => report,
        other => panic!("frozen clock must escalate, got: {other:?}"),
    };
    // Exact documented formula: attempt `k` is scheduled at
    // `now_ms.saturating_add(delays_ms[k])`; the driver reports the schedule
    // instead of waiting on it.
    assert_eq!(report.delays_ms, vec![100, 200]);
    assert_eq!(NOW_MS.saturating_add(report.delays_ms[0]), NOW_MS + 100);
    assert_eq!(NOW_MS.saturating_add(report.delays_ms[1]), NOW_MS + 200);
    // Observable sequence: both queries saw the same frozen instant.
    assert_eq!(reconciler.nows, vec![NOW_MS, NOW_MS]);

    // Caller role: advance to the last scheduled retry instant, then
    // re-invoke. The execution stays recorded `Unknown`, so this is defined.
    let advanced_now_ms = NOW_MS.saturating_add(
        report
            .delays_ms
            .last()
            .copied()
            .expect("budget is non-zero"),
    );
    assert_eq!(advanced_now_ms, NOW_MS + 200);

    let advanced = agent.reconcile_unknown(&mut reconciler, execution_id, advanced_now_ms);
    match &advanced {
        ReconcileOutcome::Resolved {
            status,
            attempts,
            delays_ms,
        } => {
            assert_eq!(*status, ToolStatus::Success);
            assert_eq!(*attempts, 1);
            assert_eq!(*delays_ms, vec![100]);
        }
        other => panic!("advanced clock must resolve, got: {other:?}"),
    }
    // Observable sequence: NOW_MS twice under the frozen invocation, then the
    // advanced instant once. Time advanced only because the caller did.
    assert_eq!(reconciler.nows, vec![NOW_MS, NOW_MS, NOW_MS + 200]);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
}

#[test]
fn resolved_unknown_answer_is_treated_as_pending() {
    // A reconciler that answers `Resolved(Unknown)` is fail-closed: the
    // driver keeps waiting instead of accepting `Unknown` as resolved.
    let (mut agent, executor) = drive_to_unknown(AgentConfig::default());
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Unknown {
        reason: "bogus resolution".to_owned(),
    });
    reconciler.push_resolved(ToolStatus::Success);
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

    assert!(
        matches!(
            &outcome,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 2,
                ..
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(reconciler.query_count(), 2);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
}

#[test]
fn reconcile_without_recorded_unknown_runs_no_query() {
    // Happy-path turn: no `Unknown` exists, so reconcile runs no query and
    // touches neither the executor nor the session.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "done".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(matches!(outcome, ExecOutcome::Completed { .. }));

    let mut reconciler = FakeReconciler::new();
    let outcome =
        agent.reconcile_unknown(&mut reconciler, bitty_ai_runtime::ExecutionId(999), NOW_MS);
    assert_eq!(outcome, ReconcileOutcome::NoUnknown);
    assert_eq!(reconciler.query_count(), 0);
    assert!(executor.calls().is_empty());
    assert_eq!(agent.session().state(), SessionState::Completed);
}

#[test]
fn zero_retry_budget_escalates_without_query() {
    let config = AgentConfig {
        max_unknown_retries: 0,
        ..AgentConfig::default()
    };
    let (mut agent, executor) = drive_to_unknown(config);
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

    let report = match &outcome {
        ReconcileOutcome::Escalated(report) => report,
        other => panic!("unexpected outcome: {other:?}"),
    };
    assert_eq!(report.attempts, 0);
    assert!(report.delays_ms.is_empty());
    assert_eq!(reconciler.query_count(), 0);
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

#[test]
fn dispatch_identity_propagation_and_reconciler_inspection() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    // Turn 1: request three tool calls: two benign successes, followed by one EffectUnknown
    provider.push_turn(ProviderTurn {
        text: "three calls".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"first"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"second"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"uncertain"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    // Turn 2 unreached because turn stops at the first Unknown
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });

    let mut executor = RecordingExecutor::new();
    executor.push_success("first ok", b"first result".to_vec());
    executor.push_success("second ok", b"second result".to_vec());
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "network drop before ack".to_owned(),
    });

    let session = session();
    let mut agent = Agent::new(provider, read_tool_bus(), session, AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(
        &mut executor,
        "fake-chat",
        "run tasks",
        &[],
        &mut sink,
        NOW_MS,
    );
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "turn must stop at first unknown: {outcome:?}"
    );

    // Three dispatches occurred before the turn stopped
    assert_eq!(agent.executions().len(), 3);
    assert_eq!(executor.call_count(), 3);

    let id_0 = agent.executions()[0].execution_id;
    let id_1 = agent.executions()[1].execution_id;
    let id_2 = agent.executions()[2].execution_id;

    // All execution IDs are distinct
    assert_ne!(id_0, id_1);
    assert_ne!(id_1, id_2);
    assert_ne!(id_0, id_2);

    // Consistent attribution: RecordingExecutor received the exact ExecutionIds from runtime
    assert_eq!(executor.calls()[0].0, id_0);
    assert_eq!(executor.calls()[0].1, "workspace_read");
    assert_eq!(executor.calls()[0].2, br#"{"path":"first"}"#);

    assert_eq!(executor.calls()[1].0, id_1);
    assert_eq!(executor.calls()[1].1, "workspace_read");
    assert_eq!(executor.calls()[1].2, br#"{"path":"second"}"#);

    assert_eq!(executor.calls()[2].0, id_2);
    assert_eq!(executor.calls()[2].1, "workspace_read");
    assert_eq!(executor.calls()[2].2, br#"{"path":"uncertain"}"#);

    // Reconciler keyed by exact ExecutionId inspects status without re-running the tool
    struct CorrelatingStatusReconciler<'a> {
        recorded: &'a [(ExecutionId, String, Vec<u8>)],
        inspected: Vec<(String, ExecutionId)>,
    }
    impl<'a> UnknownReconciler for CorrelatingStatusReconciler<'a> {
        fn reconcile(
            &mut self,
            tool: &str,
            execution_id: ExecutionId,
            _now_ms: u64,
        ) -> ReconcileStatus {
            self.inspected.push((tool.to_owned(), execution_id));
            if let Some((_id, recorded_tool, _args)) =
                self.recorded.iter().find(|(id, _, _)| *id == execution_id)
            {
                assert_eq!(recorded_tool, tool);
                ReconcileStatus::Resolved(ToolStatus::Success)
            } else {
                ReconcileStatus::Pending {
                    reason: "unknown execution id".to_owned(),
                }
            }
        }
    }

    let mut reconciler = CorrelatingStatusReconciler {
        recorded: executor.calls(),
        inspected: Vec::new(),
    };

    let reconcile_outcome = agent.reconcile_unknown(&mut reconciler, id_2, NOW_MS);
    assert!(
        matches!(
            reconcile_outcome,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 1,
                ..
            }
        ),
        "reconcile resolution failed: {reconcile_outcome:?}"
    );

    // Verification: status inspection performed without tool re-execution
    assert_eq!(
        executor.call_count(),
        3,
        "executor must not be re-invoked during reconcile"
    );
    assert_eq!(reconciler.inspected.len(), 1);
    assert_eq!(reconciler.inspected[0], ("workspace_read".to_owned(), id_2));

    // Agent executions all resolved with correct statuses
    assert_eq!(agent.executions()[0].status, ToolStatus::Success);
    assert_eq!(agent.executions()[1].status, ToolStatus::Success);
    assert_eq!(agent.executions()[2].status, ToolStatus::Success);
    assert_eq!(agent.session().state(), SessionState::Active);
}

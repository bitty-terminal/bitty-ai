//! Provider-reported `Unknown` turn-level reconcile path (AI-0134).
//!
//! Deterministic and offline: a test-only [`ModelProvider`] wrapper fails one
//! `complete` call with [`ProviderError::Unknown`] without consuming the
//! inner script, scripted [`FakeToolExecutor`] outcomes drive dispatches, and
//! scripted [`FakeReconciler`] answers drive the reconcile queries.
//! Caller-supplied `now_ms` throughout: no wall clock, no sleep, no network,
//! no secrets.
//!
//! Covered here (and only here):
//! 1. A provider `Unknown` ends the turn as [`ExecOutcome::Unknown`] with a
//!    synthetic reconcilable attribution keyed by a fresh execution id —
//!    never `Failed`, never a terminal session.
//! 2. The minted id resolves through the standard `reconcile_unknown`
//!    protocol with no provider round and no tool re-execution.
//! 3. Budget exhaustion escalates to the typed `UnknownUnresolved` report and
//!    fails the session; the next turn is rejected without I/O.
//! 4. No blind retry: the failing turn performs exactly one `complete` call,
//!    and an unreconciled lookup survives the next turn's admission in the
//!    bounded pending set instead of being dropped.
//! 5. Hostile provider/reason strings are bounded and scrubbed at the
//!    attribution boundary.
//! 6. `fallback_directive` still maps provider `Unknown` to `Stop`.
//!
//! This file does not duplicate `unknown_reconcile.rs` (tool-`Unknown`
//! resolve/escalate accounting, budget separation, backoff determinism,
//! clock-advance contract): those tests stay unmodified and keep passing.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, FakeProvider,
    FakeReconciler, FakeToolExecutor, FallbackDirective, ModelDescriptor, ModelProvider,
    ProviderError, ProviderTurn, ProviderUsage, ReconcileOutcome, SessionError, SessionState,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolStatus, TurnRequest, VecSink, fallback_directive,
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

/// Test-only provider seam: fails the n-th `complete` invocation with
/// [`ProviderError::Unknown`] without touching the inner script (mirroring
/// the "no partial turn, script not consumed" contract), and delegates every
/// other call to the inner [`FakeProvider`]. Counts every invocation so
/// tests can prove the turn loop never blindly retries a failed call.
struct UnknownOnce {
    inner: FakeProvider,
    fail_at: u64,
    attempts: u64,
    provider: String,
    reason: String,
}

impl UnknownOnce {
    fn failing_first(reason: &str) -> Self {
        Self {
            inner: FakeProvider::new("bitty-fake").expect("valid id"),
            fail_at: 1,
            attempts: 0,
            provider: "bitty-fake".to_owned(),
            reason: reason.to_owned(),
        }
    }

    fn attempts(&self) -> u64 {
        self.attempts
    }
}

impl ModelProvider for UnknownOnce {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    fn list_models(&self) -> Vec<ModelDescriptor> {
        self.inner.list_models()
    }

    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError> {
        self.attempts += 1;
        if self.attempts == self.fail_at {
            return Err(ProviderError::Unknown {
                provider: self.provider.clone(),
                reason: self.reason.clone(),
            });
        }
        self.inner.complete(request)
    }

    fn scripted_turns_remaining(&self) -> usize {
        self.inner.scripted_turns_remaining()
    }

    fn complete_calls(&self) -> u64 {
        self.inner.complete_calls()
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
    let mut registry = bitty_ai_runtime::ToolRegistry::new();
    registry
        .register(
            bitty_ai_runtime::ToolSpec::new(
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

fn final_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

#[test]
fn provider_unknown_returns_unknown_with_reconcilable_attribution() {
    let mut provider = UnknownOnce::failing_first("ack lost after billable start");
    provider.inner.push_turn(final_turn("unreached"));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    let (reason, dispatched) = match &outcome {
        ExecOutcome::Unknown { reason, dispatched } => (reason.clone(), *dispatched),
        other => panic!("provider Unknown must end Unknown, got: {other:?}"),
    };
    assert!(
        reason.contains("reconcile before retry"),
        "outcome must direct reconcile-before-retry, got: {reason:?}"
    );
    // One synthetic attribution keyed by a fresh execution id, owned by the
    // bounded provider id (not a registered tool name).
    assert_eq!(dispatched, 1);
    assert_eq!(agent.executions().len(), 1);
    let record = &agent.executions()[0];
    assert_eq!(record.tool, "bitty-fake");
    assert!(
        matches!(&record.status, ToolStatus::Unknown { reason } if reason == "ack lost after billable start"),
        "unexpected status: {:?}",
        record.status
    );
    // Fail-open is forbidden: the session stays usable for
    // reconcile-and-retry instead of failing terminal.
    assert_eq!(agent.session().state(), SessionState::Active);
    // No blind retry and no side work: exactly one `complete` call, no tool
    // dispatch, and the unreached script turn is unconsumed.
    assert_eq!(agent.provider_mut().attempts(), 1);
    assert!(executor.calls().is_empty());
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
}

#[test]
fn provider_unknown_resolves_without_reexecution() {
    let mut provider = UnknownOnce::failing_first("ack lost after billable start");
    provider.inner.push_turn(final_turn("unreached"));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(
        matches!(
            &resolved,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 1,
                ..
            }
        ),
        "unexpected outcome: {resolved:?}"
    );
    // Reconcile queried status only: no further provider round, no dispatch.
    assert_eq!(agent.provider_mut().attempts(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 0);
    assert!(executor.calls().is_empty());
    assert_eq!(reconciler.query_count(), 1);
    assert_eq!(
        reconciler.queries()[0],
        ("bitty-fake".to_owned(), execution_id)
    );
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(agent.session().state(), SessionState::Active);
}

#[test]
fn provider_unknown_escalates_with_typed_report_and_fails_session() {
    let config = AgentConfig {
        max_unknown_retries: 2,
        unknown_reconcile_base_delay_ms: 100,
        unknown_reconcile_max_delay_ms: 1_000,
        ..AgentConfig::default()
    };
    let mut provider = UnknownOnce::failing_first("still rendering");
    provider.inner.push_turn(final_turn("unreached"));
    let sess = session();
    let mut agent = Agent::new(provider, read_tool_bus(), sess.clone(), config);
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still rendering");
    reconciler.push_pending("still rendering");
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    let report = match &outcome {
        ReconcileOutcome::Escalated(report) => report.clone(),
        other => panic!("budget exhaustion must escalate, got: {other:?}"),
    };
    assert_eq!(report.tool, "bitty-fake");
    assert_eq!(report.reason, "still rendering");
    assert_eq!(report.attempts, 2);
    assert_eq!(report.dispatched, 1);
    assert_eq!(report.delays_ms, vec![100, 200]);
    assert_eq!(reconciler.query_count(), 2);
    assert!(executor.calls().is_empty());
    // Fail-closed: the session is marked failed.
    assert_eq!(sess.state(), SessionState::Failed);
    let error = AgentError::from(report);
    assert_eq!(
        error,
        AgentError::UnknownUnresolved {
            tool: "bitty-fake".to_owned(),
            reason: "still rendering".to_owned(),
            attempts: 2,
            dispatched: 1,
        }
    );

    // The escalated session rejects the next turn without I/O.
    let mut sink2 = VecSink::new();
    let retry = agent.run_turn(&mut executor, "fake-chat", "again", &[], &mut sink2, NOW_MS);
    assert!(
        matches!(
            &retry,
            ExecOutcome::Failed {
                error: AgentError::Session(SessionError::AlreadyTerminated {
                    state: SessionState::Failed,
                }),
            }
        ),
        "escalated session must reject the next turn, got: {retry:?}"
    );
    assert_eq!(agent.provider_mut().attempts(), 1);
    assert!(executor.calls().is_empty());
    assert!(sink2.is_empty());
}

#[test]
fn provider_unknown_after_prior_dispatch_keeps_both() {
    // Round 1 dispatches a successful tool; round 2's provider call reports
    // Unknown. Both attributions are kept and only the synthetic id needs
    // reconciling.
    let mut wrapper = UnknownOnce {
        inner: FakeProvider::new("bitty-fake").expect("valid id"),
        fail_at: 2,
        attempts: 0,
        provider: "bitty-fake".to_owned(),
        reason: "ack lost on round two".to_owned(),
    };
    wrapper.inner.push_turn(tool_turn("maybe wrote"));
    wrapper.inner.push_turn(final_turn("unreached"));
    let mut agent = Agent::new(wrapper, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    let dispatched = match &outcome {
        ExecOutcome::Unknown { dispatched, .. } => *dispatched,
        other => panic!("second-round Unknown must end Unknown, got: {other:?}"),
    };
    assert_eq!(dispatched, 2);
    assert_eq!(agent.executions().len(), 2);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(matches!(
        agent.executions()[1].status,
        ToolStatus::Unknown { .. }
    ));
    assert_eq!(agent.executions()[1].tool, "bitty-fake");
    assert_eq!(agent.session().state(), SessionState::Active);
    assert_eq!(executor.calls().len(), 1);

    let synthetic = agent.executions()[1].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, synthetic, NOW_MS);
    assert!(matches!(
        resolved,
        ReconcileOutcome::Resolved {
            status: ToolStatus::Success,
            ..
        }
    ));
    // The earlier dispatch is untouched and nothing re-executed.
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.provider_mut().attempts(), 2);
}

#[test]
fn unreconciled_provider_unknown_lookup_survives_next_turn() {
    // Without an intervening reconcile, the next turn carries the synthetic
    // lookup into the bounded pending set instead of dropping it silently;
    // the runtime still performed no implicit retry of the failed call.
    let mut provider = UnknownOnce::failing_first("ack lost after billable start");
    provider.inner.push_turn(final_turn("done"));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let first = agent.run_turn(&mut executor, "fake-chat", "one", &[], &mut sink, NOW_MS);
    assert!(matches!(first, ExecOutcome::Unknown { .. }));
    let stale_id = agent.executions()[0].execution_id;
    assert_eq!(agent.provider_mut().attempts(), 1);

    let mut sink2 = VecSink::new();
    let second = agent.run_turn(&mut executor, "fake-chat", "two", &[], &mut sink2, NOW_MS);
    assert!(
        matches!(second, ExecOutcome::Completed { .. }),
        "second turn must proceed on its own script, got: {second:?}"
    );
    // Exactly one further provider call (the new turn's own), never an
    // implicit retry of the failed one: attempts went 1 -> 2 across two
    // explicit turns.
    assert_eq!(agent.provider_mut().attempts(), 2);
    // The unreconciled lookup survived admission.
    assert_eq!(agent.pending_unknown().len(), 1);
    assert_eq!(agent.pending_unknown()[0].execution_id, stale_id);

    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, stale_id, NOW_MS);
    assert!(matches!(
        resolved,
        ReconcileOutcome::Resolved {
            status: ToolStatus::Success,
            ..
        }
    ));
    assert!(agent.pending_unknown().is_empty());
    assert_eq!(agent.session().state(), SessionState::Completed);
}

#[test]
fn provider_unknown_bounds_hostile_strings() {
    let hostile_provider = "p".repeat(bitty_ai_runtime::MAX_RECONCILE_REASON_BYTES * 3);
    let hostile_reason = format!("reset\n\u{1b}[2J{}", "x".repeat(2_000));
    let mut provider = UnknownOnce {
        inner: FakeProvider::new("bitty-fake").expect("valid id"),
        fail_at: 1,
        attempts: 0,
        provider: hostile_provider,
        reason: hostile_reason,
    };
    provider.inner.push_turn(final_turn("unreached"));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    let reason = match &outcome {
        ExecOutcome::Unknown { reason, .. } => reason.clone(),
        other => panic!("provider Unknown must end Unknown, got: {other:?}"),
    };
    let record = &agent.executions()[0];
    assert!(record.tool.len() <= bitty_ai_runtime::MAX_RECONCILE_REASON_BYTES);
    match &record.status {
        ToolStatus::Unknown { reason } => {
            assert!(reason.len() <= bitty_ai_runtime::MAX_RECONCILE_REASON_BYTES);
            assert!(!reason.contains(['\n', '\r']));
        }
        other => panic!("synthetic record must stay Unknown, got: {other:?}"),
    }
    assert!(!reason.contains('\n') && !reason.contains('\r'));
    assert_eq!(agent.session().state(), SessionState::Active);
}

#[test]
fn fallback_directive_stops_on_provider_unknown() {
    let unknown = ProviderError::Unknown {
        provider: "bitty-fake".to_owned(),
        reason: "ack lost after billable start".to_owned(),
    };
    assert_eq!(fallback_directive(&unknown), FallbackDirective::Stop);
    // The surrounding table is untouched: transient stays Advance, caller and
    // auth errors stay Stop.
    assert_eq!(
        fallback_directive(&ProviderError::Transport {
            provider: "bitty-fake".to_owned(),
            reason: "reset".to_owned(),
        }),
        FallbackDirective::Advance
    );
    assert_eq!(
        fallback_directive(&ProviderError::Auth {
            provider: "bitty-fake".to_owned(),
            reason: "refused".to_owned(),
        }),
        FallbackDirective::Stop
    );
}

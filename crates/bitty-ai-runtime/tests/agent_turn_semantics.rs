//! Multi-turn, cancel/resume, and escalated-Unknown turn semantics (AI-0076).
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns with explicit
//! [`ProviderUsage`], scripted [`FakeToolExecutor`] outcomes, scripted
//! [`FakeReconciler`] answers, caller-supplied `now_ms`. No network, no
//! secrets, no wall clock, no threads.
//!
//! This file extends only the untouched halves; it does not duplicate:
//! - `turn_lifecycle.rs`: per-turn `executions`/`turn_cost` reset (P1-1) and
//!   terminal-state rejection (P1-2) for `Completed`/`Failed` sessions.
//! - `unknown_reconcile.rs`: resolve/escalate accounting, budget separation,
//!   backoff determinism, clock-advance contract.
//! - `cost_ceiling.rs`: fuse trip accounting, byte-estimate fallback,
//!   zero-weight rule, determinism.
//! - `runtime_fail_closed.rs`: cancel before/after dispatch, Unknown without
//!   session failure.
//!
//! Covered here:
//! 1. Multi-turn continuation: two `Unknown` turns keep per-turn reconcile
//!    counts while the earlier id stays queryable in the bounded pending set
//!    (AI-RUN-001, #186); a `Completed` turn followed by a fresh session
//!    starts fully clean.
//! 2. Cancel-then-resume: provider-side cancel between rounds reports a
//!    reconciled `Canceled` and stays terminal without I/O; an `Unknown`
//!    turn resolves via reconcile and the next turn on the same agent
//!    dispatches fresh; a canceled entry preserves the `Unknown` record for
//!    reconcile but never resumes.
//! 3. Escalated-Unknown session semantics: escalation fails the session and
//!    the next `run_turn` is rejected as `AlreadyTerminated` without I/O,
//!    with the exact `UnknownUnresolved` fields.
//! 4. Round-limit and cost-fuse interaction: `RoundLimitExceeded` fails the
//!    session; `CostCeilingExceeded` leaves it `Active` so a raised-ceiling
//!    retry on the same session proceeds with reset per-turn state.
//! 5. Stream sequence continuity across resume: `next_seq` restarts every
//!    turn (`S-8` scheme A is per-turn), so two turns emit independent
//!    sequences and transport dedup keys must be turn-scoped.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, FakeProvider,
    FakeReconciler, FakeToolExecutor, ModelDescriptor, ModelProvider, ProviderError, ProviderTurn,
    ProviderUsage, ReconcileOutcome, SessionError, SessionState, StreamSink, ToolAuthorizer,
    ToolBus, ToolCallRequest, ToolError, ToolRegistry, ToolSpec, ToolStatus, TurnRequest, VecSink,
    validate_chunk,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate turn semantics from
/// authorization. Production wiring installs the host capability-plus-consent
/// hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

/// Test-only provider wrapper that cancels a shared session handle when the
/// n-th `complete` call returns. Lets tests land cancellation between
/// provider rounds with no dispatch in flight, deterministically.
struct CancelOnComplete {
    inner: FakeProvider,
    session: bitty_ai_runtime::AgentSession,
    cancel_after_calls: u64,
}

impl ModelProvider for CancelOnComplete {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    fn list_models(&self) -> Vec<ModelDescriptor> {
        self.inner.list_models()
    }

    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError> {
        let turn = self.inner.complete(request)?;
        if self.inner.complete_calls() == self.cancel_after_calls {
            self.session.cancel();
        }
        Ok(turn)
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

fn run<P: ModelProvider>(
    agent: &mut Agent<P>,
    executor: &mut FakeToolExecutor,
    prompt: &str,
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, &[], sink, NOW_MS)
}

fn unknown_error(executor_reason: &str) -> ToolError {
    ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: executor_reason.to_owned(),
    }
}

// Area 1: multi-turn continuation.

#[test]
fn unknown_then_unknown_keeps_per_turn_reconcile_counts() {
    // Two consecutive `Unknown` turns on one agent: the second turn must not
    // leak the first turn's counts into its outcome or reconcile reports.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("first", 10, 0));
    provider.push_turn(tool_turn("second", 4, 0));
    provider.push_turn(final_turn("unreached", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost on first"));
    executor.push_error(unknown_error("ack lost on second"));

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown, got: {first:?}"
    );
    assert_eq!(agent.executions().len(), 1);
    assert_eq!(agent.turn_cost(), 10);
    let stale_id = agent.executions()[0].execution_id;

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "two", &mut sink2);
    let dispatched = match &second {
        ExecOutcome::Unknown { dispatched, .. } => *dispatched,
        other => panic!("second turn must end Unknown, got: {other:?}"),
    };
    // Per-turn counts: only this turn's single dispatch, never 1 + 1.
    assert_eq!(dispatched, 1);
    assert_eq!(agent.executions().len(), 1);
    // Cost resets per turn: 4, never 10 + 4 = 14.
    assert_eq!(agent.turn_cost(), 4);
    assert_eq!(executor.calls().len(), 2);
    let fresh_id = agent.executions()[0].execution_id;
    assert_ne!(stale_id, fresh_id);

    // AI-RUN-001 (#186): the previous turn's id survives admission in the
    // bounded pending set, so it stays queryable even though the per-turn
    // vector was cleared. Resolving it removes it from the pending set.
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let stale = agent.reconcile_unknown(&mut reconciler, stale_id, NOW_MS);
    assert!(
        matches!(
            &stale,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 1,
                ..
            }
        ),
        "stale id must stay queryable across turns, got: {stale:?}"
    );
    assert_eq!(reconciler.query_count(), 1);
    assert!(agent.pending_unknown().is_empty());
    // A second query for the same id now finds nothing.
    let gone = agent.reconcile_unknown(&mut reconciler, stale_id, NOW_MS);
    assert_eq!(gone, ReconcileOutcome::NoUnknown);
    assert_eq!(reconciler.query_count(), 1);
    // The current turn's record still reconciles through the pending set.
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, fresh_id, NOW_MS);
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
    assert_eq!(reconciler.query_count(), 2);
    assert_eq!(agent.session().state(), SessionState::Active);
}

#[test]
fn completed_then_fresh_session_starts_fully_clean() {
    // A `Completed` session is terminal, so continuation takes a fresh
    // session ("host reset"): the new agent must observe none of the old
    // turn's state, and its own later `Unknown` escalates with per-turn
    // counts only.
    let mut first_provider = FakeProvider::new("bitty-fake").expect("valid id");
    first_provider.push_turn(tool_turn("reading", 6, 0));
    first_provider.push_turn(final_turn("done", 2, 0));
    let first_session = session();
    let mut first = Agent::new(
        first_provider,
        read_tool_bus(),
        first_session.clone(),
        AgentConfig::default(),
    );
    let mut first_executor = FakeToolExecutor::new();
    first_executor.push_success("read ok", b"data".to_vec());
    let mut first_sink = VecSink::new();
    let completed = run(&mut first, &mut first_executor, "hi", &mut first_sink);
    assert!(
        matches!(completed, ExecOutcome::Completed { .. }),
        "first turn must complete, got: {completed:?}"
    );
    assert_eq!(first_session.state(), SessionState::Completed);

    let mut second_provider = FakeProvider::new("bitty-fake").expect("valid id");
    second_provider.push_turn(tool_turn("maybe wrote", 3, 0));
    second_provider.push_turn(final_turn("unreached", 1, 0));
    let mut second = Agent::new(
        second_provider,
        read_tool_bus(),
        session(),
        AgentConfig::default(),
    );
    // Clean start: no executions, no tool history, zero cost.
    assert!(second.executions().is_empty());
    assert!(second.tool_records().is_empty());
    assert_eq!(second.turn_cost(), 0);

    let mut second_executor = FakeToolExecutor::new();
    second_executor.push_error(unknown_error("ack lost after reset"));
    let mut second_sink = VecSink::new();
    let unknown = run(&mut second, &mut second_executor, "again", &mut second_sink);
    assert!(
        matches!(unknown, ExecOutcome::Unknown { .. }),
        "second turn must end Unknown, got: {unknown:?}"
    );
    assert_eq!(second.executions().len(), 1);
    assert_eq!(second.turn_cost(), 3);
    assert_eq!(second.session().state(), SessionState::Active);

    // Escalation counts only this agent's turn: 1 dispatch, never 2.
    let execution_id = second.executions()[0].execution_id;
    let mut reconciler = FakeReconciler::new();
    for _ in 0..3 {
        reconciler.push_pending("still writing");
    }
    let outcome = second.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    let report = match &outcome {
        ReconcileOutcome::Escalated(report) => report,
        other => panic!("budget exhaustion must escalate, got: {other:?}"),
    };
    assert_eq!(report.dispatched, 1);
    assert_eq!(report.attempts, 3);
    assert_eq!(report.tool, "workspace_read");
    assert_eq!(second.session().state(), SessionState::Failed);
    // The first agent's records are untouched by the second agent's turn.
    assert_eq!(first.executions().len(), 1);
    assert_eq!(first_session.state(), SessionState::Completed);
}

// Area 2: cancel-then-resume.

#[test]
fn cancel_between_rounds_reports_reconciled_cancel_and_stays_terminal() {
    // Cancellation lands between provider rounds with no dispatch in flight:
    // the turn reports the one reconciled dispatch, and the session stays
    // `Canceled`, so the next turn performs no I/O.
    let mut inner = FakeProvider::new("bitty-fake").expect("valid id");
    inner.push_turn(tool_turn("round one", 1, 0));
    inner.push_turn(tool_turn("round two", 1, 0));
    let sess = session();
    let provider = CancelOnComplete {
        inner,
        session: sess.clone(),
        cancel_after_calls: 2,
    };
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("first ok", b"data".to_vec());
    executor.push_success("must never dispatch", b"data".to_vec());

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
        "cancel between rounds must report one reconciled dispatch, got: {outcome:?}"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 2);

    // Resume on the canceled session performs no I/O and preserves history.
    let mut sink2 = VecSink::new();
    let retry = run(&mut agent, &mut executor, "again", &mut sink2);
    assert!(
        matches!(
            retry,
            ExecOutcome::Canceled {
                dispatched: 0,
                unknown: 0
            }
        ),
        "retry on a canceled session must short-circuit, got: {retry:?}"
    );
    assert_eq!(sess.state(), SessionState::Canceled);
    assert_eq!(agent.executions().len(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    assert_eq!(executor.calls().len(), 1);
    assert!(sink2.is_empty());
}

#[test]
fn unknown_resolves_then_next_turn_dispatches_fresh() {
    // Resume path for a usable session: `Unknown` leaves the session
    // `Active`, reconcile resolves without re-execution, and the next turn
    // on the same agent dispatches fresh with reset per-turn state.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 6, 0));
    provider.push_turn(tool_turn("retry write", 6, 0));
    provider.push_turn(final_turn("done", 2, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));
    executor.push_success("retry ok", b"data".to_vec());

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown, got: {first:?}"
    );
    assert_eq!(agent.session().state(), SessionState::Active);
    assert_eq!(agent.turn_cost(), 6);
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(
        matches!(
            &resolved,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                ..
            }
        ),
        "unexpected outcome: {resolved:?}"
    );
    assert_eq!(executor.calls().len(), 1);

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "two", &mut sink2);
    assert!(
        matches!(second, ExecOutcome::Completed { .. }),
        "resumed turn must complete, got: {second:?}"
    );
    // Fresh per-turn state: one new dispatch (6 + 2 = 8), never cumulative.
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(agent.turn_cost(), 8);
    assert_eq!(agent.tool_records().len(), 1);
    assert_eq!(executor.calls().len(), 2);
    assert_eq!(agent.provider_mut().complete_calls(), 3);
    assert_eq!(agent.session().state(), SessionState::Completed);
}

#[test]
fn canceled_entry_preserves_unknown_for_reconcile_but_never_resumes() {
    // Cancel before a turn short-circuits without clearing history: the
    // recorded `Unknown` stays reconcilable, but even a successful resolve
    // never re-activates the session.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    provider.push_turn(final_turn("unreached", 1, 0));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown, got: {first:?}"
    );
    let execution_id = agent.executions()[0].execution_id;

    agent.cancel();
    let mut sink2 = VecSink::new();
    let canceled = run(&mut agent, &mut executor, "two", &mut sink2);
    assert!(
        matches!(
            canceled,
            ExecOutcome::Canceled {
                dispatched: 0,
                unknown: 0
            }
        ),
        "canceled entry must short-circuit, got: {canceled:?}"
    );
    // No I/O ran, and history is preserved for reconcile.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
    assert!(sink2.is_empty());

    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(
        matches!(
            &resolved,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                ..
            }
        ),
        "unexpected outcome: {resolved:?}"
    );
    // Resolution does not re-activate: the session stays canceled and the
    // next turn still short-circuits without I/O.
    assert_eq!(sess.state(), SessionState::Canceled);
    let mut sink3 = VecSink::new();
    let retry = run(&mut agent, &mut executor, "three", &mut sink3);
    assert!(
        matches!(
            retry,
            ExecOutcome::Canceled {
                dispatched: 0,
                unknown: 0
            }
        ),
        "canceled session must never resume, got: {retry:?}"
    );
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(executor.calls().len(), 1);
    assert!(sink3.is_empty());
}

// Area 3: escalated-Unknown session semantics.

#[test]
fn escalated_unknown_rejects_next_turn_without_io() {
    // Reconcile budget exhaustion fails the session; the next `run_turn`
    // reports `AlreadyTerminated` with no provider, executor, or sink I/O,
    // and the escalation converts to the exact typed error.
    let config = AgentConfig {
        max_unknown_retries: 2,
        unknown_reconcile_base_delay_ms: 100,
        unknown_reconcile_max_delay_ms: 1_000,
        ..AgentConfig::default()
    };
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    provider.push_turn(final_turn("unreached", 1, 0));
    let sess = session();
    let mut agent = Agent::new(provider, read_tool_bus(), sess.clone(), config);
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("host crashed before ack"));

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "hi", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown, got: {first:?}"
    );
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    let report = match &outcome {
        ReconcileOutcome::Escalated(report) => report.clone(),
        other => panic!("budget exhaustion must escalate, got: {other:?}"),
    };
    assert_eq!(report.tool, "workspace_read");
    assert_eq!(report.reason, "still writing");
    assert_eq!(report.attempts, 2);
    assert_eq!(report.dispatched, 1);
    assert_eq!(report.delays_ms, vec![100, 200]);
    assert_eq!(sess.state(), SessionState::Failed);
    let error = AgentError::from(report);
    assert_eq!(
        error,
        AgentError::UnknownUnresolved {
            tool: "workspace_read".to_owned(),
            reason: "still writing".to_owned(),
            attempts: 2,
            dispatched: 1,
        }
    );

    let mut sink2 = VecSink::new();
    let retry = run(&mut agent, &mut executor, "again", &mut sink2);
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
    // Rejection performs no I/O and preserves the recorded Unknown.
    assert_eq!(sess.state(), SessionState::Failed);
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
    assert!(sink2.is_empty());
}

// Area 4: round-limit and cost-fuse interaction.

#[test]
fn round_limit_exceeded_fails_session_and_rejects_retry() {
    // A tool-requesting loop past the round bound fails the turn and the
    // session; the recorded dispatches are kept and the retry is rejected
    // without consuming further script or dispatches.
    let config = AgentConfig {
        max_rounds: 2,
        ..AgentConfig::default()
    };
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("loop one", 1, 0));
    provider.push_turn(tool_turn("loop two", 1, 0));
    provider.push_turn(tool_turn("loop three never runs", 1, 0));
    let sess = session();
    let mut agent = Agent::new(provider, read_tool_bus(), sess.clone(), config);
    let mut executor = FakeToolExecutor::new();
    executor.push_success("one ok", b"data".to_vec());
    executor.push_success("two ok", b"data".to_vec());
    executor.push_success("must never dispatch", b"data".to_vec());

    let mut sink = VecSink::new();
    let outcome = run(&mut agent, &mut executor, "go", &mut sink);
    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::RoundLimitExceeded { limit: 2 },
            }
        ),
        "runaway tool loop must hit the round bound, got: {outcome:?}"
    );
    assert_eq!(sess.state(), SessionState::Failed);
    // Both in-bound dispatches are kept; the over-bound round never ran.
    assert_eq!(agent.executions().len(), 2);
    assert_eq!(executor.calls().len(), 2);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);

    let mut sink2 = VecSink::new();
    let retry = run(&mut agent, &mut executor, "again", &mut sink2);
    assert!(
        matches!(
            &retry,
            ExecOutcome::Failed {
                error: AgentError::Session(SessionError::AlreadyTerminated {
                    state: SessionState::Failed,
                }),
            }
        ),
        "round-limited session must reject the retry, got: {retry:?}"
    );
    assert_eq!(sess.state(), SessionState::Failed);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(executor.calls().len(), 2);
    assert_eq!(agent.executions().len(), 2);
    assert!(sink2.is_empty());
}

#[test]
fn cost_fuse_retry_with_raised_ceiling_proceeds_and_resets() {
    // The cost fuse is the exception to failed-means-terminal: the session
    // stays `Active`, so a raised-ceiling retry on the same session proceeds
    // with reset per-turn state and keeps the fused turn's effects recorded
    // on the fused agent.
    let fused_config = AgentConfig {
        max_turn_cost: Some(5),
        ..AgentConfig::default()
    };
    let mut fused_provider = FakeProvider::new("bitty-fake").expect("valid id");
    fused_provider.push_turn(tool_turn("round one", 4, 0));
    fused_provider.push_turn(tool_turn("round two", 0, 5));
    let sess = session();
    let mut fused = Agent::new(fused_provider, read_tool_bus(), sess.clone(), fused_config);
    let mut fused_executor = FakeToolExecutor::new();
    fused_executor.push_success("first ok", b"data".to_vec());
    fused_executor.push_success("must never dispatch", b"data".to_vec());

    let mut fused_sink = VecSink::new();
    let fused_outcome = run(&mut fused, &mut fused_executor, "go", &mut fused_sink);
    assert!(
        matches!(
            &fused_outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 5,
                    actual: 9
                },
            }
        ),
        "unexpected outcome: {fused_outcome:?}"
    );
    assert_eq!(sess.state(), SessionState::Active);
    assert_eq!(fused.turn_cost(), 9);
    assert_eq!(fused.executions().len(), 1);
    assert_eq!(fused_executor.calls().len(), 1);

    // Raised-ceiling retry shares the still-`Active` session and proceeds.
    let retry_config = AgentConfig {
        max_turn_cost: Some(100),
        ..AgentConfig::default()
    };
    let mut retry_provider = FakeProvider::new("bitty-fake").expect("valid id");
    retry_provider.push_turn(tool_turn("retry write", 1, 0));
    retry_provider.push_turn(final_turn("done", 1, 0));
    let mut retry = Agent::new(retry_provider, read_tool_bus(), sess.clone(), retry_config);
    let mut retry_executor = FakeToolExecutor::new();
    retry_executor.push_success("retry ok", b"data".to_vec());

    let mut retry_sink = VecSink::new();
    let retry_outcome = run(&mut retry, &mut retry_executor, "go again", &mut retry_sink);
    assert!(
        matches!(retry_outcome, ExecOutcome::Completed { .. }),
        "raised-ceiling retry must proceed, got: {retry_outcome:?}"
    );
    // Per-turn reset on the retry agent: only its own dispatch and cost.
    assert_eq!(retry.executions().len(), 1);
    assert!(matches!(retry.executions()[0].status, ToolStatus::Success));
    assert_eq!(retry.turn_cost(), 2);
    assert_eq!(retry.tool_records().len(), 1);
    assert_eq!(retry_executor.calls().len(), 1);
    assert_eq!(retry.provider_mut().complete_calls(), 2);
    assert_eq!(sess.state(), SessionState::Completed);
    // The fused turn's accounting is untouched by the retry agent's turn.
    assert_eq!(fused.turn_cost(), 9);
    assert_eq!(fused.executions().len(), 1);
}

// Area 5: stream sequence continuity across resume.

#[test]
fn stream_sequences_restart_each_turn() {
    // `S-8` scheme A is per-turn: the resumed turn restarts at `seq` 0, so
    // the two turns emit independent sequences and transport dedup keys must
    // be turn-scoped rather than assuming cross-turn uniqueness.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("first", 1, 0));
    provider.push_turn(tool_turn("second", 1, 0));
    provider.push_turn(final_turn("done", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));
    executor.push_success("second ok", b"data".to_vec());

    let mut sink = VecSink::new();
    let first = run(&mut agent, &mut executor, "one", &mut sink);
    assert!(
        matches!(first, ExecOutcome::Unknown { .. }),
        "first turn must end Unknown, got: {first:?}"
    );
    for chunk in sink.chunks() {
        validate_chunk(chunk).expect("first-turn framing must hold");
    }
    let first_framing: Vec<(u32, u32, bool)> = sink
        .chunks()
        .iter()
        .map(|chunk| (chunk.seq, chunk.total, chunk.is_final))
        .collect();
    // Assistant text batch then the tool-card batch, continuous from 0.
    assert_eq!(first_framing, vec![(0, 1, true), (1, 2, true)]);

    let execution_id = agent.executions()[0].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let resolved = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(matches!(resolved, ReconcileOutcome::Resolved { .. }));

    let mut sink2 = VecSink::new();
    let second = run(&mut agent, &mut executor, "two", &mut sink2);
    assert!(
        matches!(second, ExecOutcome::Completed { .. }),
        "resumed turn must complete, got: {second:?}"
    );
    for chunk in sink2.chunks() {
        validate_chunk(chunk).expect("second-turn framing must hold");
    }
    let second_framing: Vec<(u32, u32, bool)> = sink2
        .chunks()
        .iter()
        .map(|chunk| (chunk.seq, chunk.total, chunk.is_final))
        .collect();
    // Restart, not continuation: the resumed turn begins at 0 again.
    assert_eq!(
        second_framing,
        vec![(0, 1, true), (1, 2, true), (2, 3, true)]
    );
    // The sequences overlap (`seq` 0 and 1 appear in both turns), so no
    // cross-turn uniqueness claim holds: independence is the contract.
    assert_eq!(sink2.chunks()[0].seq, sink.chunks()[0].seq);
    assert_eq!(sink2.chunks()[1].seq, sink.chunks()[1].seq);
}

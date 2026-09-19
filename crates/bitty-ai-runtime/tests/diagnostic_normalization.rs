//! Outbound diagnostic normalization across every host/model-to-runtime
//! error/status boundary (`AI-RUN-008`).
//!
//! The runtime owns one bounded, UTF-8-safe, single-line diagnostic policy:
//! every host- or model-supplied string that reaches a caller-visible error,
//! status, or escalation surface is scrubbed to printable ASCII and bounded
//! to [`MAX_RECONCILE_REASON_BYTES`]. This file is a non-operational table
//! over the conversion branches the review named: malformed names, authorizer
//! and executor denials, generic executor failures, terminal reconciliation
//! statuses, and pending-reason escalation. Provider/`Unknown` handling is
//! already pinned elsewhere and is deliberately not re-asserted.
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns, scripted
//! [`FakeToolExecutor`] outcomes, scripted [`FakeReconciler`] answers, and
//! caller-supplied `now_ms`. No threads, no sleep, no network. Assert
//! messages stay static (`AI-0082`).

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, FakeProvider,
    FakeReconciler, FakeToolExecutor, MAX_RECONCILE_REASON_BYTES, ProviderTurn, ProviderUsage,
    ReconcileOutcome, SessionState, ToolAuthorizer, ToolBus, ToolCallRequest, ToolError,
    ToolRegistry, ToolSpec, ToolStatus, VecSink,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so the normalization table isolates the seam from
/// authorization. Production wiring installs the host capability-plus-consent
/// hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

fn session() -> bitty_ai_runtime::AgentSession {
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    bitty_ai_runtime::AgentSession::new(issuer.agent_instance(), issuer.run(), issuer.session())
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

fn tool_turn() -> ProviderTurn {
    ProviderTurn {
        text: "checking".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

/// Hostile fixture: newline + ESC + DEL + multi-byte UTF-8 + over-bound
/// filler, standing in for any host/model-supplied reason string.
fn hostile(prefix: &str) -> String {
    format!(
        "{prefix}\n\u{1b}[2J\u{7f}🦀{}",
        "x".repeat(MAX_RECONCILE_REASON_BYTES * 2)
    )
}

/// Assert one outbound string is length-bounded and printable-ASCII, which
/// makes it single-line and terminal-safe. Static assert messages only.
fn assert_display_safe(value: &str) {
    assert!(value.len() <= MAX_RECONCILE_REASON_BYTES);
    assert!(value.bytes().all(|byte| (0x20..=0x7E).contains(&byte)));
}

/// Assert a composed diagnostic carries no control bytes. Concatenating
/// several already-bounded fields can exceed a single-field bound, so only
/// the single-field values are length-checked.
fn assert_single_line(value: &str) {
    assert!(!value.contains('\n'));
    assert!(!value.contains('\r'));
    assert!(!value.contains('\u{1b}'));
    assert!(!value.contains('\u{7f}'));
}

/// Drive one turn to `Unknown` through a scripted `EffectUnknown` outcome.
fn drive_to_unknown(config: AgentConfig) -> (Agent<FakeProvider>, FakeToolExecutor) {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn());
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
    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
    (agent, executor)
}

#[test]
fn executor_denial_is_bounded_through_the_turn_and_record() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::Denied {
        name: "workspace_read".to_owned(),
        reason: hostile("executor denied "),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let ExecOutcome::Failed {
        error: AgentError::Tool(ToolError::Denied { reason, .. }),
    } = &outcome
    else {
        panic!("executor denial must fail the turn as Denied");
    };
    assert_display_safe(reason);
    assert_single_line(&outcome_error_display(&outcome));
    // The attributed record carries the same normalized reason.
    match &agent.executions()[0].status {
        ToolStatus::Denied { reason } => assert_display_safe(reason),
        ToolStatus::Success
        | ToolStatus::Failed { .. }
        | ToolStatus::Refused { .. }
        | ToolStatus::Unknown { .. } => panic!("expected a Denied record"),
    }
}

/// Render a `Failed` outcome's typed error for display-safety checking.
fn outcome_error_display(outcome: &ExecOutcome) -> String {
    match outcome {
        ExecOutcome::Failed { error } => error.to_string(),
        ExecOutcome::Completed { .. }
        | ExecOutcome::Canceled { .. }
        | ExecOutcome::Unknown { .. } => panic!("expected a Failed outcome"),
    }
}

#[test]
fn generic_executor_error_is_bounded_in_the_failed_record() {
    // A non-Denied/non-Unknown executor error travels through the agent's
    // generic arm; its malformed-name text is normalized before it enters
    // both the attributed `Failed` record and the typed `AgentError`.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::InvalidName {
        name: hostile("ghost "),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let ExecOutcome::Failed {
        error: AgentError::Tool(ToolError::InvalidName { name }),
    } = &outcome
    else {
        panic!("generic executor error must fail the turn as its typed cause");
    };
    assert_display_safe(name);
    match &agent.executions()[0].status {
        ToolStatus::Failed { reason } => assert_display_safe(reason),
        ToolStatus::Success
        | ToolStatus::Denied { .. }
        | ToolStatus::Refused { .. }
        | ToolStatus::Unknown { .. } => panic!("expected a Failed record"),
    }
}

#[test]
fn terminal_reconciliation_statuses_are_bounded_on_resolution() {
    // Host reconcilers answer with terminal statuses; each reason must be
    // normalized on resolution, both on the returned outcome and in the
    // recorded status updated in place.
    let cases = vec![
        ToolStatus::Denied {
            reason: hostile("recheck denied "),
        },
        ToolStatus::Failed {
            reason: hostile("recheck failed "),
        },
    ];
    for status in cases {
        let (mut agent, _executor) = drive_to_unknown(AgentConfig::default());
        let execution_id = agent.executions()[0].execution_id;
        let mut reconciler = FakeReconciler::new();
        reconciler.push_resolved(status.clone());

        let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

        let ReconcileOutcome::Resolved { status, .. } = &outcome else {
            panic!("a terminal answer must resolve the unknown");
        };
        match status {
            ToolStatus::Denied { reason } | ToolStatus::Failed { reason } => {
                assert_display_safe(reason);
            }
            ToolStatus::Success | ToolStatus::Refused { .. } | ToolStatus::Unknown { .. } => {
                panic!("expected a text-carrying terminal status")
            }
        }
        match &agent.executions()[0].status {
            ToolStatus::Denied { reason } | ToolStatus::Failed { reason } => {
                assert_display_safe(reason);
            }
            ToolStatus::Success | ToolStatus::Refused { .. } | ToolStatus::Unknown { .. } => {
                panic!("recorded status must match the resolution")
            }
        }
    }
}

#[test]
fn pending_escalation_reason_is_scrubbed_and_bounded() {
    // Pending reasons previously used a truncate-only helper; the shared
    // scrub policy now applies on the escalation report too.
    let config = AgentConfig {
        max_unknown_retries: 1,
        ..AgentConfig::default()
    };
    let (mut agent, _executor) = drive_to_unknown(config);
    let execution_id = agent.executions()[0].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending(hostile("still writing "));

    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);

    let ReconcileOutcome::Escalated(report) = &outcome else {
        panic!("exhausted budget must escalate");
    };
    assert_display_safe(&report.reason);
    // Escalation fails the session closed.
    assert_eq!(agent.session().state(), SessionState::Failed);
}

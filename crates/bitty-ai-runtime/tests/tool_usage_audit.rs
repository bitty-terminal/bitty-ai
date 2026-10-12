//! Permission-use audit surface evidence (AI-0246 Slice 1, #406).
//!
//! The consent ledger records grants but not per-call attribution
//! (who/when/which-capability/where): until this ring, calls are
//! unattributable. What is proven offline here is the in-memory
//! append-only usage ring on [`ToolBus`](bitty_ai_runtime::ToolBus):
//!
//! - every dispatch returning a [`ToolExecution`](bitty_ai_runtime::ToolExecution)
//!   appends one [`UsageEvent`](bitty_ai_runtime::UsageEvent), admission
//!   refusals before any executor contact as well as contacted calls (this
//!   contrasts with the durable pending-effect log, which is contacted-only:
//!   it opens a row only once the executor is reached, so admission refusals
//!   leave no row there);
//! - the event reuses existing types (`ExecutionId`, tool name,
//!   `required_scope` threaded from the registry, protocol id, [`AuthBase`](bitty_ai_runtime::AuthBase)
//!   triple, `now_ms`, placement-derived site, [`ToolStatus`](bitty_ai_runtime::ToolStatus) /
//!   [`ResultDisposition`](bitty_ai_runtime::ResultDisposition), bounded reason,
//!   FNV-1a-64 args digest — never raw args, never cwd bytes);
//! - the ring caps at [`MAX_USAGE_EVENTS`](bitty_ai_runtime::tool::MAX_USAGE_EVENTS)
//!   (mirroring [`MAX_EXECUTIONS_PER_AGENT`](bitty_ai_runtime::MAX_EXECUTIONS_PER_AGENT)),
//!   oldest-evicts on insert, and reads oldest-first through read-only
//!   [`list_usage`](bitty_ai_runtime::ToolBus::list_usage) (filter plus limit,
//!   cloned snapshots; no delete or update API exists).
//!
//! Deterministic and offline: scripted [`FakeToolExecutor`](bitty_ai_runtime::FakeToolExecutor),
//! caller-supplied `now_ms`, doubles only. No network, no filesystem, no
//! wall clock, no threads, no secrets. This file makes no acceptance claim:
//! it is Slice 1 gate evidence only.
//!
//! CodeQL lesson from AI-0082: assert/panic messages are static only.

use bitty_ai_runtime::{
    AgentLevel, AuthBase, AuthContext, AuthDecision, FakeToolExecutor, IdIssuer, ToolAuthorizer,
    ToolBus, ToolCall, ToolError, ToolPlacement, ToolRegistry, ToolSpec, ToolStatus, UsageEvent,
    UsageFilter,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate attribution from
/// authorization. Production wiring installs the host capability-plus-consent
/// hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

fn caller() -> AuthBase {
    let mut ids = IdIssuer::default();
    AuthBase {
        agent_instance_id: ids.agent_instance(),
        session_id: ids.session(),
        level: AgentLevel::Workspace,
    }
}

fn registry() -> ToolRegistry {
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
    registry
        .register(
            ToolSpec::new(
                "terminal_read_zone",
                "Native zone read",
                br#"{"type":"object"}"#.to_vec(),
                "terminal.inspect",
                true,
                ToolPlacement::Native,
                Some(bitty_ai_runtime::NativePlacement {
                    reason: "host-native terminal read".to_owned(),
                    backend: "terminal.inspect".to_owned(),
                }),
            )
            .expect("valid native spec"),
        )
        .expect("capacity");
    registry
}

/// AI-0227 mechanical fixture: live consent grant for the
/// `workspace_read` / `workspace.read` triple dispatched below.
fn consented_ledger() -> bitty_ai_runtime::FakeConsentLedger {
    let mut ledger = bitty_ai_runtime::FakeConsentLedger::new();
    ledger
        .grant(
            "local.assistant",
            "workspace_read",
            "workspace.read",
            u64::MAX,
        )
        .expect("grant fits");
    ledger
}

fn call(name: &str, args: &[u8]) -> ToolCall {
    ToolCall {
        name: name.to_owned(),
        arguments: args.to_vec(),
    }
}

fn assert_display_safe(value: &str, bound: usize) {
    assert!(value.len() <= bound);
    assert!(value.bytes().all(|byte| (0x20..=0x7E).contains(&byte)));
    assert!(!value.contains('\n'));
    assert!(!value.contains('\r'));
}

#[test]
fn usage_cap_mirrors_per_agent_execution_scale() {
    assert_eq!(
        bitty_ai_runtime::tool::MAX_USAGE_EVENTS,
        bitty_ai_runtime::MAX_EXECUTIONS_PER_AGENT
    );
    assert_eq!(bitty_ai_runtime::tool::MAX_USAGE_EVENTS, 32);
}

#[test]
fn contacted_dispatch_populates_who_when_which_where() {
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    assert!(bus.is_usage_empty());
    assert_eq!(bus.usage_len(), 0);

    let base = caller();
    let mut ids = IdIssuer::default();
    let execution_id = ids.execution();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("ok", b"data".to_vec());

    let execution = bus
        .dispatch(
            &mut executor,
            &call("workspace_read", br#"{"path":"a"}"#),
            &base,
            execution_id,
            NOW_MS,
        )
        .expect("contacted dispatch is recorded");
    assert_eq!(execution.status, ToolStatus::Success);

    // One dispatch, one audit row.
    assert_eq!(bus.usage_len(), 1);
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    let event = &rows[0];

    // Who: caller triple preserved verbatim.
    assert_eq!(event.caller, base);
    // When: caller-supplied timestamp preserved verbatim.
    assert_eq!(event.occurred_at_ms, NOW_MS);
    // Which: execution id, tool, threaded scope, protocol id.
    assert_eq!(event.execution_id, execution_id);
    assert_eq!(event.execution_id, execution.execution_id);
    assert_eq!(event.tool, "workspace_read");
    assert_eq!(event.scope, "workspace.read");
    assert_eq!(event.protocol_id, "local.assistant");
    // Where: registry-derived placement, no invented server/task labels.
    assert_eq!(event.site.placement, Some(ToolPlacement::Mcp));
    assert_eq!(event.site.server_id, None);
    assert_eq!(event.site.task_id, None);
    // Outcome axes mirror the execution; success carries no reason text.
    assert_eq!(event.status, ToolStatus::Success);
    assert_eq!(event.result_disposition, execution.result_disposition);
    assert_eq!(event.reason, "");
    // Privacy: digest of the bytes, never the bytes.
    assert_eq!(
        event.args_digest,
        UsageEvent::args_digest(br#"{"path":"a"}"#)
    );
}

#[test]
fn refusal_before_contact_is_logged_with_zero_executor_contact() {
    // No authorizer and no consent pair: the bus refuses fail-closed before
    // any executor contact. The durable pending log would leave no row here
    // (contacted-only); the audit ring must still attribute the refusal.
    let mut bus = ToolBus::new(registry());
    let base = caller();
    let mut ids = IdIssuer::default();
    let execution_id = ids.execution();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never run", b"nope".to_vec());

    let execution = bus
        .dispatch(
            &mut executor,
            &call("workspace_read", br#"{}"#),
            &base,
            execution_id,
            NOW_MS,
        )
        .expect("admission refusal is recorded");
    assert!(execution.status.is_admission_refusal());
    assert!(executor.calls().is_empty());
    assert_eq!(bus.calls_this_turn(), 0);

    // The refusal is still audited: who/when/which/where populated.
    assert_eq!(bus.usage_len(), 1);
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    let event = &rows[0];
    assert_eq!(event.execution_id, execution_id);
    assert_eq!(event.tool, "workspace_read");
    assert_eq!(event.scope, "workspace.read");
    assert_eq!(event.caller, base);
    assert_eq!(event.occurred_at_ms, NOW_MS);
    assert_eq!(event.site.placement, Some(ToolPlacement::Mcp));
    assert!(event.status.is_admission_refusal());
    assert!(!event.reason.is_empty());
    assert_display_safe(&event.reason, 512);
}

#[test]
fn native_placement_is_threaded_from_the_registry() {
    // Data-only S3 vocabulary: the native-placed declaration dispatches
    // through the same gate, and the row carries `native` (never a
    // caller-supplied claim).
    struct AllowNative;
    impl ToolAuthorizer for AllowNative {
        fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
            AuthDecision::Allow
        }
    }
    let mut ledger = bitty_ai_runtime::FakeConsentLedger::new();
    ledger
        .grant(
            "local.assistant",
            "terminal_read_zone",
            "terminal.inspect",
            u64::MAX,
        )
        .expect("grant fits");
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowNative)
        .with_consent(ledger, "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("ok", Vec::new());
    bus.dispatch(
        &mut executor,
        &call("terminal_read_zone", br#"{}"#),
        &base,
        ids.execution(),
        NOW_MS,
    )
    .expect("native dispatch is recorded");
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].site.placement, Some(ToolPlacement::Native));
    assert_eq!(rows[0].scope, "terminal.inspect");
}

#[test]
fn unregistered_tool_still_logs_with_empty_scope_and_no_placement() {
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    let execution = bus
        .dispatch(
            &mut executor,
            &call("workspace_ghost", br#"{}"#),
            &base,
            ids.execution(),
            NOW_MS,
        )
        .expect("unknown-tool refusal is recorded");
    assert!(matches!(
        execution.status,
        ToolStatus::Refused {
            cause: ToolError::UnknownTool { .. }
        }
    ));
    assert!(executor.calls().is_empty());
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].tool, "workspace_ghost");
    assert_eq!(rows[0].scope, "");
    assert_eq!(rows[0].site.placement, None);
}

#[test]
fn executor_outcomes_preserve_status_and_disposition_axes() {
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::Failed {
        name: "workspace_read".to_owned(),
        reason: "host reported failure".to_owned(),
    });
    executor.push_error(ToolError::Denied {
        name: "workspace_read".to_owned(),
        reason: "host policy refused".to_owned(),
    });
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "ack lost".to_owned(),
    });
    for _ in 0..3 {
        bus.dispatch(
            &mut executor,
            &call("workspace_read", br#"{}"#),
            &base,
            ids.execution(),
            NOW_MS,
        )
        .expect("executor outcome is recorded");
    }
    assert_eq!(bus.usage_len(), 3);
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 3);
    assert!(matches!(rows[0].status, ToolStatus::Failed { .. }));
    assert!(matches!(rows[1].status, ToolStatus::Denied { .. }));
    assert!(matches!(rows[2].status, ToolStatus::Unknown { .. }));
    for event in &rows {
        assert_display_safe(&event.reason, 512);
        assert!(!event.reason.is_empty());
    }
}

#[test]
fn over_bound_payload_rejection_is_logged_as_success_with_rejection() {
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", vec![b'y'; 16 * 1024 + 1]);
    let execution = bus
        .dispatch(
            &mut executor,
            &call("workspace_read", br#"{}"#),
            &base,
            ids.execution(),
            NOW_MS,
        )
        .expect("rejected payload is recorded");
    assert_eq!(execution.status, ToolStatus::Success);
    assert!(execution.result_disposition.is_rejection());
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, ToolStatus::Success);
    assert!(rows[0].result_disposition.is_rejection());
}

#[test]
fn hostile_strings_are_bounded_and_display_safe_on_push() {
    let mut bus = ToolBus::new(registry());
    let mut ids = IdIssuer::default();
    let hostile = format!("bad\n\u{1b}[2J\u{7f}{}", "x".repeat(2048));
    let event = UsageEvent {
        execution_id: ids.execution(),
        tool: hostile.clone(),
        scope: hostile.clone(),
        protocol_id: hostile.clone(),
        caller: caller(),
        occurred_at_ms: NOW_MS,
        site: bitty_ai_runtime::UsageSite {
            placement: Some(ToolPlacement::Mcp),
            server_id: Some(hostile.clone()),
            task_id: Some(hostile.clone()),
        },
        status: ToolStatus::Denied {
            reason: hostile.clone(),
        },
        result_disposition: bitty_ai_runtime::ResultDisposition::Accepted,
        reason: hostile.clone(),
        args_digest: UsageEvent::args_digest(b"args"),
    };
    bus.push_usage(event);
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    let stored = &rows[0];
    assert_display_safe(&stored.tool, 512);
    assert_display_safe(&stored.scope, 512);
    assert_display_safe(&stored.protocol_id, 512);
    assert_display_safe(&stored.reason, 512);
    let server = stored.site.server_id.as_ref().expect("server kept");
    assert_display_safe(server, 512);
    let task = stored.site.task_id.as_ref().expect("task kept");
    assert_display_safe(task, 512);
    match &stored.status {
        ToolStatus::Denied { reason } => assert_display_safe(reason, 512),
        _ => panic!("expected Denied status"),
    }
}

#[test]
fn empty_site_labels_normalize_to_none() {
    let mut bus = ToolBus::new(registry());
    let mut ids = IdIssuer::default();
    bus.push_usage(UsageEvent {
        execution_id: ids.execution(),
        tool: "workspace_read".to_owned(),
        scope: "workspace.read".to_owned(),
        protocol_id: "local.assistant".to_owned(),
        caller: caller(),
        occurred_at_ms: NOW_MS,
        site: bitty_ai_runtime::UsageSite {
            placement: None,
            server_id: Some(String::new()),
            task_id: Some(String::new()),
        },
        status: ToolStatus::Success,
        result_disposition: bitty_ai_runtime::ResultDisposition::Accepted,
        reason: String::new(),
        args_digest: 0,
    });
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].site.server_id, None);
    assert_eq!(rows[0].site.task_id, None);
}

#[test]
fn args_digest_is_deterministic_and_hides_raw_args() {
    // Same bytes give the same digest on every call; any byte change moves
    // it. The raw PII-bearing bytes never enter the ring or its `Debug`.
    let first = UsageEvent::args_digest(br#"{"path":"secret-file"}"#);
    let again = UsageEvent::args_digest(br#"{"path":"secret-file"}"#);
    let moved = UsageEvent::args_digest(br#"{"path":"other-file"}"#);
    assert_eq!(first, again);
    assert_ne!(first, moved);

    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("ok", Vec::new());
    let pii = br#"{"path":"super-secret-pii-12345"}"#;
    bus.dispatch(
        &mut executor,
        &call("workspace_read", pii),
        &base,
        ids.execution(),
        NOW_MS,
    )
    .expect("dispatch is recorded");
    let rows = bus.list_usage(&UsageFilter::match_all(), 8);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].args_digest, UsageEvent::args_digest(pii));
    let rendered = format!("{:?}", rows[0]);
    assert!(!rendered.contains("super-secret-pii-12345"));
    let bus_rendered = format!("{bus:?}");
    assert!(!bus_rendered.contains("super-secret-pii-12345"));
}

#[test]
fn ring_oldest_evicts_past_cap_and_lists_oldest_first() {
    let mut bus = ToolBus::new(registry());
    let mut ids = IdIssuer::default();
    let base = caller();
    let cap = bitty_ai_runtime::tool::MAX_USAGE_EVENTS;
    for index in 0..(cap + 8) {
        bus.push_usage(UsageEvent {
            execution_id: ids.execution(),
            tool: format!("tool_{index:02}"),
            scope: "workspace.read".to_owned(),
            protocol_id: "local.assistant".to_owned(),
            caller: base,
            occurred_at_ms: NOW_MS + index as u64,
            site: bitty_ai_runtime::UsageSite {
                placement: Some(ToolPlacement::Mcp),
                server_id: None,
                task_id: None,
            },
            status: ToolStatus::Success,
            result_disposition: bitty_ai_runtime::ResultDisposition::Accepted,
            reason: String::new(),
            args_digest: index as u64,
        });
    }
    // Capped: the 8 oldest rows evicted, the newest 32 kept.
    assert_eq!(bus.usage_len(), cap);
    let rows = bus.list_usage(&UsageFilter::match_all(), cap + 8);
    assert_eq!(rows.len(), cap);
    assert_eq!(rows[0].tool, "tool_08");
    assert_eq!(rows[cap - 1].tool, format!("tool_{:02}", cap + 7));
    // Oldest-first: timestamps ascend with insertion order.
    for pair in rows.windows(2) {
        assert!(pair[0].occurred_at_ms < pair[1].occurred_at_ms);
    }
}

#[test]
fn list_usage_filters_and_limits_without_mutating_the_ring() {
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    for _ in 0..3 {
        executor.push_success("ok", Vec::new());
    }
    for _ in 0..3 {
        bus.dispatch(
            &mut executor,
            &call("workspace_read", br#"{}"#),
            &base,
            ids.execution(),
            NOW_MS,
        )
        .expect("dispatch is recorded");
    }
    assert_eq!(bus.usage_len(), 3);

    // Limit caps the snapshot oldest-first.
    let capped = bus.list_usage(&UsageFilter::match_all(), 2);
    assert_eq!(capped.len(), 2);
    assert_eq!(
        capped[0].execution_id,
        bus.list_usage(&UsageFilter::match_all(), 8)[0].execution_id
    );

    // Filters select; a miss selects nothing.
    let by_tool = bus.list_usage(
        &UsageFilter {
            tool: Some("workspace_read".to_owned()),
            ..UsageFilter::match_all()
        },
        8,
    );
    assert_eq!(by_tool.len(), 3);
    let miss = bus.list_usage(
        &UsageFilter {
            tool: Some("workspace_ghost".to_owned()),
            ..UsageFilter::match_all()
        },
        8,
    );
    assert!(miss.is_empty());
    let by_scope = bus.list_usage(
        &UsageFilter {
            scope: Some("workspace.read".to_owned()),
            ..UsageFilter::match_all()
        },
        8,
    );
    assert_eq!(by_scope.len(), 3);
    let by_protocol = bus.list_usage(
        &UsageFilter {
            protocol_id: Some("local.assistant".to_owned()),
            ..UsageFilter::match_all()
        },
        8,
    );
    assert_eq!(by_protocol.len(), 3);
    let by_placement = bus.list_usage(
        &UsageFilter {
            placement: Some(ToolPlacement::Mcp),
            ..UsageFilter::match_all()
        },
        8,
    );
    assert_eq!(by_placement.len(), 3);
    let by_native = bus.list_usage(
        &UsageFilter {
            placement: Some(ToolPlacement::Native),
            ..UsageFilter::match_all()
        },
        8,
    );
    assert!(by_native.is_empty());

    // Read-only: mutating the snapshot leaves the ring unchanged.
    let mut snapshot = bus.list_usage(&UsageFilter::match_all(), 8);
    snapshot.clear();
    assert_eq!(bus.usage_len(), 3);
    assert_eq!(bus.list_usage(&UsageFilter::match_all(), 8).len(), 3);

    // Zero limit returns no rows but keeps the ring.
    assert!(bus.list_usage(&UsageFilter::match_all(), 0).is_empty());
    assert_eq!(bus.usage_len(), 3);
}

#[test]
fn usage_ring_survives_begin_turn_while_call_counter_resets() {
    // The per-turn call counter is turn-scoped; the audit ring is bus-scoped
    // and append-only across turns (no clear on `begin_turn`).
    let mut bus = ToolBus::new(registry())
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant");
    let base = caller();
    let mut ids = IdIssuer::default();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("ok", Vec::new());
    bus.dispatch(
        &mut executor,
        &call("workspace_read", br#"{}"#),
        &base,
        ids.execution(),
        NOW_MS,
    )
    .expect("dispatch is recorded");
    assert_eq!(bus.calls_this_turn(), 1);
    assert_eq!(bus.usage_len(), 1);
    bus.begin_turn();
    assert_eq!(bus.calls_this_turn(), 0);
    assert_eq!(bus.usage_len(), 1);
}

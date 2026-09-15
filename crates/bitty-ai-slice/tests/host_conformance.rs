//! Shared `BittyHost` conformance: the same assertions run against
//! [`FakeHost`] and [`LiveBittyHost`] (BII-09 rendezvous, AI-0035).
//!
//! This suite is mapping proof, not live wiring: the live path delegates to
//! the real `bitty-ipc` services (`SnapshotService`, `ToolDispatchService`,
//! `ExecutionService`) with canned `fn` providers as the injectable transport
//! seam. No test connects to a real terminal, process, PTY, socket, or
//! network peer; every timestamp is caller-supplied `now_ms`; there are no
//! secrets. What is asserted is that dispatch prefix order, consent
//! attribution, and `ExecutionResult` / `Unknown` reconcile semantics match
//! the `bitty` shapes through both hosts.
//!
//! Direction basis is BII-01 through BII-05 as draft handoff inputs (not
//! accepted architecture); `execution-ownership-r1` and `tool-transport-r2`
//! are likewise draft inputs, and the accepted IPC/Agent RFC remains the
//! overriding authority. If a live shape ever diverges from the shared
//! expectation, the live host fails closed and the divergence is recorded as
//! gap input for the `bitty` track; there is no bypass path.

use bitty_ai_slice::{BittyHost, FakeHost, LiveBittyHost, SliceError};
use bitty_ipc::error::IpcError;
use bitty_ipc::execution::{
    EXECUTION_SCOPE, EffectState, ExecutionRequest, ExecutionResult, ExecutionStatus,
    RawExecutionOutput,
};
use bitty_ipc::scope::{Scope, ScopeSet, required_scope_for_method};
use bitty_ipc::snapshot::{
    DetailLevel, SNAPSHOT_METHOD, SemanticZone, SnapshotData, SnapshotRequest, ZoneKind,
};
use bitty_ipc::tool_dispatch::{ToolOutput, ToolRequest, ToolSpec};

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;
const EXPIRED_MS: u64 = NOW_MS + TTL_MS;
const FAKE_CLIENT: &str = "agent-fakehost";
const LIVE_CLIENT: &str = "agent-livehost";

// ── canned live providers (injectable seam, deterministic) ─────────────────

fn live_snapshot_ok(request: &SnapshotRequest) -> Result<SnapshotData, IpcError> {
    Ok(SnapshotData {
        terminal_id: request.terminal_id.clone(),
        generation: 7,
        cwd: "/work".to_owned(),
        semantic_zones: vec![SemanticZone {
            kind: ZoneKind::Output,
            line_start: 0,
            line_end: 3,
        }],
        text: "hello".to_owned(),
    })
}

fn live_snapshot_rich(request: &SnapshotRequest) -> Result<SnapshotData, IpcError> {
    Ok(SnapshotData {
        terminal_id: request.terminal_id.clone(),
        generation: 7,
        cwd: "/work".to_owned(),
        semantic_zones: vec![
            SemanticZone {
                kind: ZoneKind::Prompt,
                line_start: 0,
                line_end: 0,
            },
            SemanticZone {
                kind: ZoneKind::Output,
                line_start: 1,
                line_end: 2,
            },
        ],
        text: "ééééé".to_owned(),
    })
}

fn live_snapshot_mismatch(_request: &SnapshotRequest) -> Result<SnapshotData, IpcError> {
    Ok(SnapshotData {
        terminal_id: "t:2".to_owned(),
        generation: 7,
        cwd: "/work".to_owned(),
        semantic_zones: Vec::new(),
        text: "hello".to_owned(),
    })
}

fn live_tool_ok(request: &ToolRequest) -> Result<ToolOutput, IpcError> {
    Ok(ToolOutput {
        target_id: request.target.clone(),
        data: b"zone bytes".to_vec(),
        summary: "ok".to_owned(),
    })
}

fn live_tool_mismatch(_request: &ToolRequest) -> Result<ToolOutput, IpcError> {
    Ok(ToolOutput {
        target_id: Some("t:9".to_owned()),
        data: b"data".to_vec(),
        summary: "ok".to_owned(),
    })
}

fn live_tool_oversized(request: &ToolRequest) -> Result<ToolOutput, IpcError> {
    Ok(ToolOutput {
        target_id: request.target.clone(),
        data: vec![b'x'; 17 * 1024],
        summary: "too big".to_owned(),
    })
}

fn live_exec_completed(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: request.target.clone(),
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    })
}

fn live_exec_unknown(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: request.target.clone(),
        status: ExecutionStatus::Unknown,
        exit_code: None,
        stdout: String::new(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Unknown,
    })
}

fn live_exec_status_mismatch(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: request.target.clone(),
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Unknown,
    })
}

fn live_exec_unknown_with_exit(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: request.target.clone(),
        status: ExecutionStatus::Unknown,
        exit_code: Some(1),
        stdout: String::new(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Unknown,
    })
}

fn live_exec_large(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: request.target.clone(),
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "x".repeat(512),
        stderr: "y".repeat(512),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    })
}

fn live_exec_mismatch(_request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: Some("t:9".to_owned()),
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    })
}

// ── builders ────────────────────────────────────────────────────────────────

fn fake_inspect() -> FakeHost {
    let mut host =
        FakeHost::new(FAKE_CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("fake host");
    host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    host
}

fn live_inspect() -> LiveBittyHost {
    let mut host = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    host
}

fn fake_spawn() -> FakeHost {
    let mut host =
        FakeHost::new(FAKE_CLIENT, ScopeSet::single(Scope::ProcessSpawn)).expect("fake host");
    host.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    host
}

fn live_spawn() -> LiveBittyHost {
    let mut host = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    host.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    host
}

fn read_only_spec() -> ToolSpec {
    ToolSpec::new(
        "terminal_read_zone",
        "read-only zone read",
        br#"{"type":"object"}"#.to_vec(),
        Scope::TerminalInspect,
        true,
    )
    .expect("spec validates")
}

fn snapshot_request() -> SnapshotRequest {
    SnapshotRequest::new("t:1", DetailLevel::Standard)
}

fn snapshot_data(text: &str) -> SnapshotData {
    SnapshotData {
        terminal_id: "t:1".to_owned(),
        generation: 7,
        cwd: "/work".to_owned(),
        semantic_zones: vec![SemanticZone {
            kind: ZoneKind::Output,
            line_start: 0,
            line_end: 3,
        }],
        text: text.to_owned(),
    }
}

fn tool_request() -> ToolRequest {
    ToolRequest::new("terminal_read_zone", br#"{"zone":"output"}"#.to_vec())
}

fn tool_output() -> ToolOutput {
    ToolOutput {
        target_id: None,
        data: b"zone bytes".to_vec(),
        summary: "ok".to_owned(),
    }
}

fn exec_request() -> ExecutionRequest {
    ExecutionRequest::new("git", vec!["diff".to_owned()]).with_allow_effects(true)
}

fn exec_unknown_output() -> RawExecutionOutput {
    RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Unknown,
        exit_code: None,
        stdout: String::new(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Unknown,
    }
}

fn ipc_error(error: &SliceError) -> &IpcError {
    match error {
        SliceError::Ipc(inner) => inner,
        other => panic!("expected SliceError::Ipc, got {other:?}"),
    }
}

// ── snapshot DTO ────────────────────────────────────────────────────────────

fn assert_snapshot_dto<H: BittyHost>(host: &mut H) {
    let snapshot = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect("snapshot serves");
    assert_eq!(snapshot.terminal_id, "t:1");
    assert_eq!(snapshot.generation, 7);
    assert_eq!(snapshot.cwd, "/work");
    assert_eq!(snapshot.semantic_zones.len(), 1);
    assert_eq!(snapshot.text, "hello");
    assert!(!snapshot.truncated);
    assert!(snapshot.is_untrusted_surface);
    snapshot.validate().expect("dto validates");
    assert_eq!(host.client_id(), host.client_id());
}

#[test]
fn snapshot_dto_is_shared() {
    let mut fake = fake_inspect();
    fake.push_snapshot_data(snapshot_data("hello"));
    assert_snapshot_dto(&mut fake);

    let mut live = live_inspect();
    assert_snapshot_dto(&mut live);
}

fn assert_snapshot_unknown_method<H: BittyHost>(host: &mut H) {
    let error = host
        .snapshot("panel.context", &snapshot_request(), NOW_MS)
        .expect_err("unknown method must fail closed");
    assert_eq!(
        error,
        SliceError::UnsupportedHostMethod {
            method: "panel.context".to_owned(),
        }
    );
}

#[test]
fn snapshot_unknown_method_is_shared() {
    let mut fake = fake_inspect();
    fake.push_snapshot_data(snapshot_data("hello"));
    let pending_before = fake.pending_snapshot_scripts();
    assert_snapshot_unknown_method(&mut fake);
    assert_eq!(
        fake.pending_snapshot_scripts(),
        pending_before,
        "fake consumes no script"
    );

    let mut live = live_inspect();
    let methods_before = live.snapshot_method_count();
    assert_snapshot_unknown_method(&mut live);
    assert_eq!(
        live.snapshot_method_count(),
        methods_before,
        "live registers no method"
    );
}

fn assert_snapshot_missing_scope<H: BittyHost>(host: &mut H) {
    let error = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );
}

#[test]
fn snapshot_missing_scope_is_shared() {
    let mut fake = FakeHost::new(FAKE_CLIENT, ScopeSet::new()).expect("fake host");
    fake.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    fake.push_snapshot_data(snapshot_data("hello"));
    assert_snapshot_missing_scope(&mut fake);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::new(),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    live.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    assert_snapshot_missing_scope(&mut live);
}

fn assert_snapshot_missing_consent<H: BittyHost>(host: &mut H) {
    let error = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing consent must fail closed");
    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );
}

#[test]
fn snapshot_missing_consent_is_shared() {
    let mut fake =
        FakeHost::new(FAKE_CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("fake host");
    fake.push_snapshot_data(snapshot_data("hello"));
    assert_snapshot_missing_consent(&mut fake);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    assert_snapshot_missing_consent(&mut live);
}

#[test]
fn snapshot_expired_consent_is_shared() {
    let mut fake = fake_inspect();
    fake.push_snapshot_data(snapshot_data("hello"));
    let error = fake
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), EXPIRED_MS)
        .expect_err("expired consent must fail closed");
    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );

    let mut live = live_inspect();
    let error = live
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), EXPIRED_MS)
        .expect_err("expired consent must fail closed");
    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );
}

#[test]
fn snapshot_missing_handler_is_shared() {
    let mut fake = fake_inspect();
    let error = fake
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing scripted data must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::NotFound { .. }),
        "got {error:?}"
    );

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        None,
        live_exec_completed,
    )
    .expect("live host");
    live.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    let error = live
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing handler must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::NotFound { .. }),
        "got {error:?}"
    );
}

#[test]
fn snapshot_terminal_mismatch_is_shared() {
    let mut fake = fake_inspect();
    fake.push_snapshot_data(SnapshotData {
        terminal_id: "t:2".to_owned(),
        ..snapshot_data("hello")
    });
    let error = fake
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("terminal mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        Some(live_snapshot_mismatch),
        live_exec_completed,
    )
    .expect("live host");
    live.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    let error = live
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("terminal mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

#[test]
fn snapshot_zone_narrowing_is_shared() {
    let request = SnapshotRequest::new("t:1", DetailLevel::Minimal)
        .with_max_bytes(7)
        .with_zone(ZoneKind::Output);

    let mut fake = fake_inspect();
    fake.push_snapshot_data(SnapshotData {
        semantic_zones: vec![
            SemanticZone {
                kind: ZoneKind::Prompt,
                line_start: 0,
                line_end: 0,
            },
            SemanticZone {
                kind: ZoneKind::Output,
                line_start: 1,
                line_end: 2,
            },
        ],
        ..snapshot_data("ééééé")
    });
    let snapshot = fake
        .snapshot(SNAPSHOT_METHOD, &request, NOW_MS)
        .expect("snapshot serves");
    assert!(snapshot.truncated);
    assert!(snapshot.text.len() <= 7);
    assert!("ééééé".starts_with(snapshot.text.as_str()));
    assert_eq!(snapshot.semantic_zones.len(), 1);
    assert_eq!(snapshot.semantic_zones[0].kind, ZoneKind::Output);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        Some(live_snapshot_rich),
        live_exec_completed,
    )
    .expect("live host");
    live.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    let snapshot = live
        .snapshot(SNAPSHOT_METHOD, &request, NOW_MS)
        .expect("snapshot serves");
    assert!(snapshot.truncated);
    assert!(snapshot.text.len() <= 7);
    assert!("ééééé".starts_with(snapshot.text.as_str()));
    assert_eq!(snapshot.semantic_zones.len(), 1);
    assert_eq!(snapshot.semantic_zones[0].kind, ZoneKind::Output);
}

// ── dispatch prefix ─────────────────────────────────────────────────────────

#[test]
fn dispatch_unknown_tool_is_shared() {
    let mut fake = fake_inspect();
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(tool_output());
    let request = ToolRequest::new("terminal_destroy", br#"{}"#.to_vec());
    let error = fake
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("unknown tool must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::NotFound { .. }),
        "got {error:?}"
    );
    assert_eq!(fake.pending_tool_scripts(), 1, "fake consumes no script");

    let mut live = live_inspect();
    live.register_tool(read_only_spec(), live_tool_ok)
        .expect("register");
    let tools_before = live.tool_count();
    let error = live
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("unknown tool must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::NotFound { .. }),
        "got {error:?}"
    );
    assert_eq!(live.tool_count(), tools_before, "live registers nothing");
}

#[test]
fn dispatch_missing_scope_is_shared() {
    let mut fake = FakeHost::new(FAKE_CLIENT, ScopeSet::new()).expect("fake host");
    fake.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(tool_output());
    let error = fake
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::new(),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    live.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent");
    live.register_tool(read_only_spec(), live_tool_ok)
        .expect("register");
    let error = live
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );
}

fn assert_dispatch_missing_consent<H: BittyHost>(host: &mut H) {
    let error = host
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("missing consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired denial, got {other:?}"),
    }
}

#[test]
fn dispatch_missing_consent_is_shared() {
    let mut fake =
        FakeHost::new(FAKE_CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("fake host");
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(tool_output());
    assert_dispatch_missing_consent(&mut fake);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    live.register_tool(read_only_spec(), live_tool_ok)
        .expect("register");
    assert_dispatch_missing_consent(&mut live);
}

#[test]
fn dispatch_expired_consent_is_shared() {
    let mut fake = fake_inspect();
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(tool_output());
    let error = fake
        .dispatch_tool(&tool_request(), EXPIRED_MS, 1)
        .expect_err("expired consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired, got {other:?}"),
    }

    let mut live = live_inspect();
    live.register_tool(read_only_spec(), live_tool_ok)
        .expect("register");
    let error = live
        .dispatch_tool(&tool_request(), EXPIRED_MS, 1)
        .expect_err("expired consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired, got {other:?}"),
    }
}

#[test]
fn dispatch_effect_opt_in_is_shared() {
    fn effect_spec() -> ToolSpec {
        ToolSpec::new(
            "terminal_send",
            "send input (effect)",
            br#"{"type":"object"}"#.to_vec(),
            Scope::TerminalInput,
            false,
        )
        .expect("spec validates")
    }

    let granted = {
        let mut set = ScopeSet::new();
        set.insert(Scope::TerminalInput);
        set
    };
    let mut fake = FakeHost::new(FAKE_CLIENT, granted).expect("fake host");
    fake.grant_consent(Scope::TerminalInput, NOW_MS, TTL_MS)
        .expect("consent");
    fake.register_tool(effect_spec()).expect("register");
    fake.push_tool_output(ToolOutput {
        target_id: None,
        data: b"sent".to_vec(),
        summary: "sent".to_owned(),
    });
    let denied = ToolRequest::new("terminal_send", br#"{}"#.to_vec());
    let error = fake
        .dispatch_tool(&denied, NOW_MS, 1)
        .expect_err("effect without opt-in must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "EffectRequiresExplicitConsent"),
        other => panic!("expected opt-in denial, got {other:?}"),
    }
    let allowed = ToolRequest::new("terminal_send", br#"{}"#.to_vec()).with_allow_effects(true);
    let execution = fake
        .dispatch_tool(&allowed, NOW_MS, 1)
        .expect("effect with opt-in serves");
    assert_eq!(execution.tool, "terminal_send");
    assert!(execution.is_untrusted_surface);

    let granted = {
        let mut set = ScopeSet::new();
        set.insert(Scope::TerminalInput);
        set
    };
    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        granted,
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    live.grant_consent(Scope::TerminalInput, NOW_MS, TTL_MS)
        .expect("consent");
    live.register_tool(effect_spec(), live_tool_ok)
        .expect("register");
    let error = live
        .dispatch_tool(&denied, NOW_MS, 1)
        .expect_err("effect without opt-in must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "EffectRequiresExplicitConsent"),
        other => panic!("expected opt-in denial, got {other:?}"),
    }
    let execution = live
        .dispatch_tool(&allowed, NOW_MS, 1)
        .expect("effect with opt-in serves");
    assert_eq!(execution.tool, "terminal_send");
    assert!(execution.is_untrusted_surface);
}

#[test]
fn dispatch_consent_per_scope_is_shared() {
    let mut fake = fake_inspect();
    for name in ["terminal_read_zone", "terminal_read_zone_b"] {
        fake.register_tool(
            ToolSpec::new(
                name,
                "read-only zone read",
                br#"{"type":"object"}"#.to_vec(),
                Scope::TerminalInspect,
                true,
            )
            .expect("spec"),
        )
        .expect("register");
    }
    fake.push_tool_output(tool_output());
    fake.push_tool_output(tool_output());
    fake.dispatch_tool(
        &ToolRequest::new("terminal_read_zone", br#"{}"#.to_vec()),
        NOW_MS,
        1,
    )
    .expect("shared scope allows first");
    fake.dispatch_tool(
        &ToolRequest::new("terminal_read_zone_b", br#"{}"#.to_vec()),
        NOW_MS,
        2,
    )
    .expect("shared scope allows second");

    let mut live = live_inspect();
    for name in ["terminal_read_zone", "terminal_read_zone_b"] {
        live.register_tool(
            ToolSpec::new(
                name,
                "read-only zone read",
                br#"{"type":"object"}"#.to_vec(),
                Scope::TerminalInspect,
                true,
            )
            .expect("spec"),
            live_tool_ok,
        )
        .expect("register");
    }
    live.dispatch_tool(
        &ToolRequest::new("terminal_read_zone", br#"{}"#.to_vec()),
        NOW_MS,
        1,
    )
    .expect("shared scope allows first");
    live.dispatch_tool(
        &ToolRequest::new("terminal_read_zone_b", br#"{}"#.to_vec()),
        NOW_MS,
        2,
    )
    .expect("shared scope allows second");
}

#[test]
fn dispatch_oversized_arguments_is_shared() {
    let mut fake = fake_inspect();
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(tool_output());
    let request = ToolRequest::new("terminal_read_zone", vec![b'x'; 17 * 1024]);
    let error = fake
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("oversized args must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );

    let mut live = live_inspect();
    live.register_tool(read_only_spec(), live_tool_ok)
        .expect("register");
    let error = live
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("oversized args must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );
}

#[test]
fn dispatch_oversized_result_is_shared() {
    let mut fake = fake_inspect();
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(ToolOutput {
        target_id: None,
        data: vec![b'x'; 17 * 1024],
        summary: "too big".to_owned(),
    });
    let error = fake
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("oversized result must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );

    let mut live = live_inspect();
    live.register_tool(read_only_spec(), live_tool_oversized)
        .expect("register");
    let error = live
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("oversized result must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );
}

#[test]
fn dispatch_target_mismatch_is_shared() {
    let mut fake = fake_inspect();
    fake.register_tool(read_only_spec()).expect("register");
    fake.push_tool_output(ToolOutput {
        target_id: Some("t:9".to_owned()),
        data: b"data".to_vec(),
        summary: "ok".to_owned(),
    });
    let request = tool_request().with_target("t:1");
    let error = fake
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("target mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );

    let mut live = live_inspect();
    live.register_tool(read_only_spec(), live_tool_mismatch)
        .expect("register");
    let error = live
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("target mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

// ── execution + Unknown ─────────────────────────────────────────────────────

fn assert_execution_success<H: BittyHost>(host: &mut H, expected_client: &str, execution_id: u64) {
    let result = host
        .execute(&exec_request(), NOW_MS, execution_id)
        .expect("execution serves");
    assert_eq!(result.execution_id, execution_id);
    assert_eq!(result.client_id, expected_client);
    assert_eq!(result.status, ExecutionStatus::Completed);
    assert!(!result.needs_reconciliation());
    assert!(result.is_untrusted_surface);
    result.validate().expect("result validates");
    assert_eq!(host.reconcile(execution_id).expect("reconcile"), result);
}

#[test]
fn execution_success_is_shared() {
    let mut fake = fake_spawn();
    fake.push_execution_output(RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    assert_execution_success(&mut fake, FAKE_CLIENT, 11);

    let mut live = live_spawn();
    assert_execution_success(&mut live, LIVE_CLIENT, 11);
}

#[test]
fn execution_unknown_agreement_is_shared() {
    // Completed status paired with Unknown effect must fail closed.
    let mut fake = fake_spawn();
    fake.push_execution_output(RawExecutionOutput {
        status: ExecutionStatus::Completed,
        effect_state: EffectState::Unknown,
        target_id: None,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
    });
    let error = fake
        .execute(&exec_request(), NOW_MS, 12)
        .expect_err("mismatched Unknown must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
    assert_eq!(fake.execution_count(), 0, "fake refusal stores nothing");

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_status_mismatch,
    )
    .expect("live host");
    live.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    let error = live
        .execute(&exec_request(), NOW_MS, 12)
        .expect_err("mismatched Unknown must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
    assert_eq!(live.execution_count(), 0, "live refusal stores nothing");

    // Unknown status carrying an exit code must fail closed.
    let mut fake = fake_spawn();
    fake.push_execution_output(RawExecutionOutput {
        status: ExecutionStatus::Unknown,
        exit_code: Some(1),
        effect_state: EffectState::Unknown,
        target_id: None,
        stdout: String::new(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
    });
    let error = fake
        .execute(&exec_request(), NOW_MS, 13)
        .expect_err("Unknown with exit code must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_unknown_with_exit,
    )
    .expect("live host");
    live.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    let error = live
        .execute(&exec_request(), NOW_MS, 13)
        .expect_err("Unknown with exit code must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

fn assert_unknown_roundtrip<H: BittyHost>(host: &mut H, expected_client: &str) {
    let unknown = host
        .execute(&exec_request(), NOW_MS, 21)
        .expect("unknown stores");
    assert!(unknown.needs_reconciliation());
    assert_eq!(host.reconcile(21).expect("reconcile"), unknown);

    let retry_error = host
        .execute(&exec_request(), NOW_MS, 21)
        .expect_err("same id must never blind-retry");
    assert!(
        matches!(ipc_error(&retry_error), IpcError::InvalidRequest { .. }),
        "got {retry_error:?}"
    );

    let terminal = ExecutionResult::new(
        21,
        expected_client.to_owned(),
        None,
        ExecutionStatus::Completed,
        Some(0),
        "done".to_owned(),
        String::new(),
        false,
        Vec::new(),
        EffectState::Completed,
    )
    .expect("terminal result validates");
    host.resolve(21, terminal.clone())
        .expect("resolve closes unknown");
    assert_eq!(host.reconcile(21).expect("reconciled"), terminal);

    let second = ExecutionResult::new(
        21,
        expected_client.to_owned(),
        None,
        ExecutionStatus::Failed,
        Some(1),
        "again".to_owned(),
        String::new(),
        false,
        Vec::new(),
        EffectState::Failed,
    )
    .expect("second terminal validates");
    let error = host
        .resolve(21, second)
        .expect_err("terminal entries never overwrite");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

#[test]
fn execution_unknown_roundtrip_is_shared() {
    let mut fake = fake_spawn();
    fake.push_execution_output(exec_unknown_output());
    assert_unknown_roundtrip(&mut fake, FAKE_CLIENT);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_unknown,
    )
    .expect("live host");
    live.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    assert_unknown_roundtrip(&mut live, LIVE_CLIENT);
}

fn assert_resolve_attribution<H: BittyHost>(host: &mut H, expected_client: &str) {
    let wrong_id = ExecutionResult::new(
        99,
        expected_client.to_owned(),
        None,
        ExecutionStatus::Completed,
        Some(0),
        String::new(),
        String::new(),
        false,
        Vec::new(),
        EffectState::Completed,
    )
    .expect("result validates");
    let error = host
        .resolve(31, wrong_id)
        .expect_err("id mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );

    let rewritten = ExecutionResult::new(
        31,
        "someone-else".to_owned(),
        None,
        ExecutionStatus::Completed,
        Some(0),
        String::new(),
        String::new(),
        false,
        Vec::new(),
        EffectState::Completed,
    )
    .expect("result validates");
    let error = host
        .resolve(31, rewritten)
        .expect_err("attribution rewrite must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );

    let still_unknown = ExecutionResult::new(
        31,
        expected_client.to_owned(),
        None,
        ExecutionStatus::Unknown,
        None,
        String::new(),
        String::new(),
        false,
        Vec::new(),
        EffectState::Unknown,
    )
    .expect("unknown validates");
    let error = host
        .resolve(31, still_unknown)
        .expect_err("resolution must be terminal");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

#[test]
fn execution_resolve_attribution_is_shared() {
    let mut fake = fake_spawn();
    fake.push_execution_output(exec_unknown_output());
    fake.execute(&exec_request(), NOW_MS, 31)
        .expect("unknown stores");
    assert_resolve_attribution(&mut fake, FAKE_CLIENT);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_unknown,
    )
    .expect("live host");
    live.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    live.execute(&exec_request(), NOW_MS, 31)
        .expect("unknown stores");
    assert_resolve_attribution(&mut live, LIVE_CLIENT);
}

#[test]
fn execution_gates_are_shared() {
    let mut scoped = FakeHost::new(FAKE_CLIENT, ScopeSet::new()).expect("fake host");
    scoped
        .grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    scoped.push_execution_output(RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    let error = scoped
        .execute(&exec_request(), NOW_MS, 41)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );

    let mut live_scoped = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::new(),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    live_scoped
        .grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    let error = live_scoped
        .execute(&exec_request(), NOW_MS, 41)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );

    let mut unconsented =
        FakeHost::new(FAKE_CLIENT, ScopeSet::single(Scope::ProcessSpawn)).expect("fake host");
    unconsented.push_execution_output(RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    let error = unconsented
        .execute(&exec_request(), NOW_MS, 42)
        .expect_err("missing consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired, got {other:?}"),
    }

    let mut live_unconsented = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    let error = live_unconsented
        .execute(&exec_request(), NOW_MS, 42)
        .expect_err("missing consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired, got {other:?}"),
    }

    let mut fake = fake_spawn();
    fake.push_execution_output(RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    let request = ExecutionRequest::new("git", vec!["diff".to_owned()]);
    let error = fake
        .execute(&request, NOW_MS, 43)
        .expect_err("execution without opt-in must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "EffectRequiresExplicitConsent"),
        other => panic!("expected opt-in denial, got {other:?}"),
    }

    let mut live = live_spawn();
    let error = live
        .execute(&request, NOW_MS, 43)
        .expect_err("execution without opt-in must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "EffectRequiresExplicitConsent"),
        other => panic!("expected opt-in denial, got {other:?}"),
    }

    assert_eq!(EXECUTION_SCOPE, Scope::ProcessSpawn);
}

#[test]
fn execution_truncation_is_shared() {
    let mut fake = fake_spawn();
    fake.push_execution_output(RawExecutionOutput {
        stdout: "x".repeat(512),
        stderr: "y".repeat(512),
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    let request = ExecutionRequest::new("git", Vec::new())
        .with_allow_effects(true)
        .with_output_budget(64);
    let result = fake
        .execute(&request, NOW_MS, 51)
        .expect("execution serves");
    assert!(result.truncated);
    assert_eq!(result.stdout_summary.len(), 64);
    assert_eq!(result.stderr_summary.len(), 64);

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_large,
    )
    .expect("live host");
    live.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    let result = live
        .execute(&request, NOW_MS, 51)
        .expect("execution serves");
    assert!(result.truncated);
    assert_eq!(result.stdout_summary.len(), 64);
    assert_eq!(result.stderr_summary.len(), 64);
}

#[test]
fn execution_target_mismatch_is_shared() {
    let mut fake = fake_spawn();
    fake.push_execution_output(RawExecutionOutput {
        target_id: Some("t:9".to_owned()),
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    let request = exec_request().with_target("t:1");
    let error = fake
        .execute(&request, NOW_MS, 52)
        .expect_err("target mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::ProcessSpawn),
        Some(live_snapshot_ok),
        live_exec_mismatch,
    )
    .expect("live host");
    live.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent");
    let error = live
        .execute(&request, NOW_MS, 52)
        .expect_err("target mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

// ── consent ledger seam ─────────────────────────────────────────────────────

#[test]
fn consent_ledger_is_shared() {
    let mut fake =
        FakeHost::new(FAKE_CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("fake host");
    assert!(!fake.consent_active(Scope::TerminalInspect, NOW_MS));
    fake.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("grant");
    assert!(fake.consent_active(Scope::TerminalInspect, NOW_MS));
    assert!(!fake.consent_active(Scope::TerminalInspect, EXPIRED_MS));
    assert!(fake.revoke_consent(Scope::TerminalInspect));
    assert!(!fake.consent_active(Scope::TerminalInspect, NOW_MS));

    let mut live = LiveBittyHost::new(
        LIVE_CLIENT,
        ScopeSet::single(Scope::TerminalInspect),
        Some(live_snapshot_ok),
        live_exec_completed,
    )
    .expect("live host");
    assert!(!live.consent_active(Scope::TerminalInspect, NOW_MS));
    live.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("grant");
    assert!(live.consent_active(Scope::TerminalInspect, NOW_MS));
    assert!(!live.consent_active(Scope::TerminalInspect, EXPIRED_MS));
    assert!(live.revoke_consent(Scope::TerminalInspect));
    assert!(!live.consent_active(Scope::TerminalInspect, NOW_MS));
}

// ── trait swap ──────────────────────────────────────────────────────────────

fn use_host<H: BittyHost>(host: &mut H, request: &SnapshotRequest, now_ms: u64) -> bool {
    host.snapshot("terminal.snapshot", request, now_ms).is_ok()
}

#[test]
fn same_trait_serves_both_hosts() {
    let mut fake = fake_inspect();
    fake.push_snapshot_data(snapshot_data("hello"));
    assert!(use_host(&mut fake, &snapshot_request(), NOW_MS));
    assert_eq!(fake.client_id(), FAKE_CLIENT);

    let mut live = live_inspect();
    assert!(use_host(&mut live, &snapshot_request(), NOW_MS));
    assert_eq!(live.client_id(), LIVE_CLIENT);
}

#[test]
fn spawn_scope_mapping_is_shared() {
    assert_eq!(
        required_scope_for_method("process.spawn"),
        Some(Scope::ProcessSpawn)
    );
    assert_eq!(EXECUTION_SCOPE, Scope::ProcessSpawn);
}

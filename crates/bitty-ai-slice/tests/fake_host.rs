//! `FakeHost` deterministic double: dispatch prefix, consent, execution
//! `Unknown`, and snapshot DTO coverage (BII-09 route, AI-0025).
//!
//! Every test is deterministic (`now_ms` caller-supplied, no wall clock,
//! no threads, no network, no filesystem) and single-agent (one `FakeHost`
//! per `client_id`). Shapes mirror `bitty` `main` at `be6e63c` read-only
//! (`2cbb1fb..be6e63c` changes `auth.rs` token-free reasons plus the
//! `devtools` surface; existing DTO shapes are byte-identical, so no mirror
//! change); live wiring is out of scope and no test connects to a
//! real host.

use bitty_ai_slice::{BittyHost, FakeHost, SliceError};
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
const CLIENT: &str = "agent-fakehost";

fn inspect_host() -> FakeHost {
    let mut host =
        FakeHost::new(CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("host construction");
    host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent grant");
    host
}

fn spawn_host() -> FakeHost {
    let mut host =
        FakeHost::new(CLIENT, ScopeSet::single(Scope::ProcessSpawn)).expect("host construction");
    host.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent grant");
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

fn exec_success_output() -> RawExecutionOutput {
    RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    }
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

#[test]
fn snapshot_dto_has_seven_fields_and_trust_label() {
    let mut host = inspect_host();
    host.push_snapshot_data(snapshot_data("hello"));
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
}

#[test]
fn snapshot_unknown_method_fails_closed() {
    let mut host = inspect_host();
    host.push_snapshot_data(snapshot_data("hello"));
    let error = host
        .snapshot("panel.context", &snapshot_request(), NOW_MS)
        .expect_err("unknown method must fail closed");
    assert_eq!(
        error,
        SliceError::UnsupportedHostMethod {
            method: "panel.context".to_owned(),
        }
    );
    assert_eq!(host.pending_snapshot_scripts(), 1, "no script consumed");
}

#[test]
fn snapshot_missing_scope_fails_closed() {
    let mut host = FakeHost::new(CLIENT, ScopeSet::new()).expect("host construction");
    host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent grant");
    host.push_snapshot_data(snapshot_data("hello"));
    let error = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );
    assert_eq!(host.pending_snapshot_scripts(), 1, "no script consumed");
}

#[test]
fn snapshot_missing_consent_fails_closed() {
    let mut host =
        FakeHost::new(CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("host construction");
    host.push_snapshot_data(snapshot_data("hello"));
    let error = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing consent must fail closed");
    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );
    assert_eq!(host.pending_snapshot_scripts(), 1, "no script consumed");
}

#[test]
fn snapshot_expired_consent_fails_closed() {
    let mut host = inspect_host();
    host.push_snapshot_data(snapshot_data("hello"));
    let error = host
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
fn snapshot_missing_handler_fails_closed() {
    let mut host = inspect_host();
    let error = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("missing scripted data must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::NotFound { .. }),
        "got {error:?}"
    );
}

#[test]
fn snapshot_terminal_mismatch_fails_closed() {
    let mut host = inspect_host();
    host.push_snapshot_data(SnapshotData {
        terminal_id: "t:2".to_owned(),
        ..snapshot_data("hello")
    });
    let error = host
        .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
        .expect_err("terminal mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

#[test]
fn snapshot_zone_narrowing_and_truncation_hold() {
    let mut host = inspect_host();
    let request = SnapshotRequest::new("t:1", DetailLevel::Minimal)
        .with_max_bytes(7)
        .with_zone(ZoneKind::Output);
    host.push_snapshot_data(SnapshotData {
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
    let snapshot = host
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
fn dispatch_unknown_tool_fails_closed_before_provider() {
    let mut host = inspect_host();
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(tool_output());
    let request = ToolRequest::new("terminal_destroy", br#"{}"#.to_vec());
    let error = host
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("unknown tool must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::NotFound { .. }),
        "got {error:?}"
    );
    assert_eq!(host.pending_tool_scripts(), 1, "no script consumed");
}

#[test]
fn dispatch_missing_scope_fails_closed() {
    let mut host = FakeHost::new(CLIENT, ScopeSet::new()).expect("host construction");
    host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent grant");
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(tool_output());
    let error = host
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );
    assert_eq!(host.pending_tool_scripts(), 1, "no script consumed");
}

#[test]
fn dispatch_missing_consent_fails_closed() {
    let mut host =
        FakeHost::new(CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("host construction");
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(tool_output());
    let error = host
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("missing consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired denial, got {other:?}"),
    }
    assert_eq!(host.pending_tool_scripts(), 1, "no script consumed");
}

#[test]
fn dispatch_expired_consent_fails_closed() {
    let mut host = inspect_host();
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(tool_output());
    let error = host
        .dispatch_tool(&tool_request(), EXPIRED_MS, 1)
        .expect_err("expired consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired denial, got {other:?}"),
    }
}

#[test]
fn dispatch_effect_requires_explicit_opt_in() {
    let granted = {
        let mut set = ScopeSet::new();
        set.insert(Scope::TerminalInput);
        set
    };
    let mut host = FakeHost::new(CLIENT, granted).expect("host construction");
    host.grant_consent(Scope::TerminalInput, NOW_MS, TTL_MS)
        .expect("consent grant");
    host.register_tool(effect_spec()).expect("register");
    host.push_tool_output(ToolOutput {
        target_id: None,
        data: b"sent".to_vec(),
        summary: "sent".to_owned(),
    });
    let denied = ToolRequest::new("terminal_send", br#"{}"#.to_vec());
    let error = host
        .dispatch_tool(&denied, NOW_MS, 1)
        .expect_err("effect without opt-in must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "EffectRequiresExplicitConsent"),
        other => panic!("expected opt-in denial, got {other:?}"),
    }
    let allowed = ToolRequest::new("terminal_send", br#"{}"#.to_vec()).with_allow_effects(true);
    let execution = host
        .dispatch_tool(&allowed, NOW_MS, 1)
        .expect("effect with opt-in serves");
    assert_eq!(execution.tool, "terminal_send");
    assert!(execution.is_untrusted_surface);
}

#[test]
fn dispatch_refuses_scope_with_only_unrelated_grant_and_live_consent() {
    // AIQ-22/42 negative evidence: the host's server-evaluated grant holds
    // only `terminal.inspect` while the ledger carries a live grant for
    // `terminal.input`. Consent for the tool's scope must not widen the
    // grant, and a refusal must consume no script and store no execution.
    let mut host = inspect_host();
    host.grant_consent(Scope::TerminalInput, NOW_MS, TTL_MS)
        .expect("consent grant");
    host.register_tool(effect_spec()).expect("register");
    host.push_tool_output(ToolOutput {
        target_id: None,
        data: b"sent".to_vec(),
        summary: "sent".to_owned(),
    });
    let request = ToolRequest::new("terminal_send", br#"{}"#.to_vec()).with_allow_effects(true);

    let error = host
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("scope gate must ignore consent for another scope");

    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );
    assert_eq!(host.pending_tool_scripts(), 1, "no script consumed");
    assert_eq!(host.execution_count(), 0, "no execution stored");
}

#[test]
fn dispatch_consent_is_per_scope_not_per_tool() {
    let mut host = inspect_host();
    for name in ["terminal_read_zone", "terminal_read_zone_b"] {
        host.register_tool(
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
    host.push_tool_output(tool_output());
    host.push_tool_output(tool_output());
    let first = ToolRequest::new("terminal_read_zone", br#"{}"#.to_vec());
    let second = ToolRequest::new("terminal_read_zone_b", br#"{}"#.to_vec());
    host.dispatch_tool(&first, NOW_MS, 1)
        .expect("shared scope grant allows first tool");
    host.dispatch_tool(&second, NOW_MS, 2)
        .expect("shared scope grant allows second tool");
}

#[test]
fn dispatch_oversized_arguments_fail_closed() {
    let mut host = inspect_host();
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(tool_output());
    let request = ToolRequest::new("terminal_read_zone", vec![b'x'; 17 * 1024]);
    let error = host
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("oversized args must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );
    assert_eq!(host.pending_tool_scripts(), 1, "no script consumed");
}

#[test]
fn dispatch_oversized_result_fails_closed() {
    let mut host = inspect_host();
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(ToolOutput {
        target_id: None,
        data: vec![b'x'; 17 * 1024],
        summary: "too big".to_owned(),
    });
    let error = host
        .dispatch_tool(&tool_request(), NOW_MS, 1)
        .expect_err("oversized result must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );
}

#[test]
fn dispatch_target_mismatch_fails_closed() {
    let mut host = inspect_host();
    host.register_tool(read_only_spec()).expect("register");
    host.push_tool_output(ToolOutput {
        target_id: Some("t:9".to_owned()),
        data: b"data".to_vec(),
        summary: "ok".to_owned(),
    });
    let request = tool_request().with_target("t:1");
    let error = host
        .dispatch_tool(&request, NOW_MS, 1)
        .expect_err("target mismatch must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
}

#[test]
fn dispatch_registry_cap_is_fail_closed() {
    let mut host = inspect_host();
    for index in 0..32 {
        host.register_tool(
            ToolSpec::new(
                format!("tool_{index:02}"),
                "read-only",
                br#"{"type":"object"}"#.to_vec(),
                Scope::TerminalInspect,
                true,
            )
            .expect("spec"),
        )
        .expect("register within cap");
    }
    let error = host
        .register_tool(
            ToolSpec::new(
                "tool_overflow",
                "read-only",
                br#"{"type":"object"}"#.to_vec(),
                Scope::TerminalInspect,
                true,
            )
            .expect("spec"),
        )
        .expect_err("registry cap must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::LimitExceeded { .. }),
        "got {error:?}"
    );
}

// ── execution + Unknown ─────────────────────────────────────────────────────

#[test]
fn execution_success_stores_and_reconciles() {
    let mut host = spawn_host();
    host.push_execution_output(exec_success_output());
    let result = host
        .execute(&exec_request(), NOW_MS, 11)
        .expect("execution serves");
    assert_eq!(result.execution_id, 11);
    assert_eq!(result.client_id, CLIENT);
    assert_eq!(result.status, ExecutionStatus::Completed);
    assert!(!result.needs_reconciliation());
    assert!(result.is_untrusted_surface);
    result.validate().expect("result validates");
    assert_eq!(host.execution_count(), 1);
    assert_eq!(host.reconcile(11).expect("reconcile"), result);
}

#[test]
fn execution_unknown_agreement_is_enforced() {
    let mut host = spawn_host();
    host.push_execution_output(RawExecutionOutput {
        status: ExecutionStatus::Completed,
        effect_state: EffectState::Unknown,
        ..exec_success_output()
    });
    let error = host
        .execute(&exec_request(), NOW_MS, 12)
        .expect_err("mismatched Unknown must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
    assert_eq!(host.execution_count(), 0, "refusal stores nothing");

    host.push_execution_output(RawExecutionOutput {
        status: ExecutionStatus::Unknown,
        exit_code: Some(1),
        effect_state: EffectState::Unknown,
        ..exec_success_output()
    });
    let error = host
        .execute(&exec_request(), NOW_MS, 13)
        .expect_err("Unknown with exit code must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::InvalidRequest { .. }),
        "got {error:?}"
    );
    assert_eq!(host.execution_count(), 0, "refusal stores nothing");
}

#[test]
fn execution_unknown_reconciles_and_resolves_without_retry() {
    let mut host = spawn_host();
    host.push_execution_output(exec_unknown_output());
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
        CLIENT.to_owned(),
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
        CLIENT.to_owned(),
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
fn execution_resolve_preserves_attribution_and_terminality() {
    let mut host = spawn_host();
    host.push_execution_output(exec_unknown_output());
    host.execute(&exec_request(), NOW_MS, 31)
        .expect("unknown stores");

    let wrong_id = ExecutionResult::new(
        99,
        CLIENT.to_owned(),
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
        CLIENT.to_owned(),
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
fn execution_requires_scope_consent_and_opt_in() {
    let mut scoped = FakeHost::new(CLIENT, ScopeSet::new()).expect("host construction");
    scoped
        .grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent grant");
    scoped.push_execution_output(exec_success_output());
    let error = scoped
        .execute(&exec_request(), NOW_MS, 41)
        .expect_err("missing scope must fail closed");
    assert!(
        matches!(ipc_error(&error), IpcError::ScopeDenied { .. }),
        "got {error:?}"
    );

    let mut unconsented =
        FakeHost::new(CLIENT, ScopeSet::single(Scope::ProcessSpawn)).expect("host construction");
    unconsented.push_execution_output(exec_success_output());
    let error = unconsented
        .execute(&exec_request(), NOW_MS, 42)
        .expect_err("missing consent must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "ConsentRequired"),
        other => panic!("expected ConsentRequired, got {other:?}"),
    }

    let mut host = spawn_host();
    host.push_execution_output(exec_success_output());
    let request = ExecutionRequest::new("git", vec!["diff".to_owned()]);
    let error = host
        .execute(&request, NOW_MS, 43)
        .expect_err("execution without opt-in must fail closed");
    match ipc_error(&error) {
        IpcError::Denied { code, .. } => assert_eq!(code, "EffectRequiresExplicitConsent"),
        other => panic!("expected opt-in denial, got {other:?}"),
    }
    assert_eq!(EXECUTION_SCOPE, Scope::ProcessSpawn);
}

#[test]
fn execution_truncates_streams_at_caller_budget() {
    let mut host = spawn_host();
    host.push_execution_output(RawExecutionOutput {
        stdout: "x".repeat(512),
        stderr: "y".repeat(512),
        ..exec_success_output()
    });
    let request = ExecutionRequest::new("git", Vec::new())
        .with_allow_effects(true)
        .with_output_budget(64);
    let result = host
        .execute(&request, NOW_MS, 51)
        .expect("execution serves");
    assert!(result.truncated);
    assert_eq!(result.stdout_summary.len(), 64);
    assert_eq!(result.stderr_summary.len(), 64);
}

// ── process.spawn mapping (AI-0027) ─────────────────────────────────────────

// bitty `main` at `be6e63c` carries the bounded consent-gated `process.spawn`
// surface (#717) behind the `[tools.*]` fail-closed allowlist (#716) in
// `bitty-runtime` / `bitty-plugin-host` only; `crates/bitty-ipc` is
// byte-identical across `2cbb1fb..be6e63c` except `auth.rs` token-free reasons
// plus the `devtools` surface, so `FakeHost` needs no mirror change. This test pins the mapping the new surface relies on: `SpawnService::dispatch`
// resolves `SpawnRequest(tool + args)` via the host `SpawnAuthorizer`, then
// runs the resolved executable under the real `ExecutionService` — the exact
// prefix `FakeHost::execute` mirrors (scope `process.spawn`, effect opt-in,
// per-spawn consent, stream budget, `Unknown` agreement, attributed store,
// `reconcile`/`resolve` without retry). The allowlist itself stays host-side
// (like the runtime `ToolRegistry` + `ToolAuthorizer` seam) and is correctly
// absent here: `FakeHost` takes the already-resolved `ExecutionRequest`.
#[test]
fn spawn_surface_maps_onto_execute_reconcile_resolve() {
    // One scope guards the whole path: the wire method, the spawn surface,
    // and the execution backend agree.
    assert_eq!(
        required_scope_for_method("process.spawn"),
        Some(Scope::ProcessSpawn)
    );
    assert_eq!(EXECUTION_SCOPE, Scope::ProcessSpawn);

    // Spawn-shaped allowlisted request (`git status` is an accepted Layer-2
    // verb) serves through `execute` with scope + opt-in + consent.
    let mut host = spawn_host();
    host.push_execution_output(RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    });
    let request = ExecutionRequest::new("git", vec!["status".to_owned()]).with_allow_effects(true);
    let result = host
        .execute(&request, NOW_MS, 61)
        .expect("spawn-shaped execution serves");
    assert_eq!(result.status, ExecutionStatus::Completed);
    assert!(result.is_untrusted_surface);
    assert_eq!(host.reconcile(61).expect("reconcile queries"), result);

    // Timeout-shaped outcome (`Unknown`/`Unknown` with no exit code: spawn
    // reports a killed child as `Unknown`, never as failure) reconciles and
    // resolves without retry — the same path the runtime maps as
    // `EffectUnknown` → `ToolStatus::Unknown` → `ExecOutcome::Unknown`.
    host.push_execution_output(exec_unknown_output());
    let unknown = host
        .execute(&exec_request(), NOW_MS, 62)
        .expect("unknown stores");
    assert!(unknown.needs_reconciliation());
    let terminal = ExecutionResult::new(
        62,
        CLIENT.to_owned(),
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
    host.resolve(62, terminal.clone())
        .expect("resolve closes unknown");
    assert_eq!(host.reconcile(62).expect("reconciled"), terminal);
}

// ── consent ledger seam ─────────────────────────────────────────────────────

#[test]
fn consent_grant_revoke_expiry_are_deterministic() {
    let mut host =
        FakeHost::new(CLIENT, ScopeSet::single(Scope::TerminalInspect)).expect("host construction");
    assert!(!host.consent_active(Scope::TerminalInspect, NOW_MS));
    host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("grant");
    assert!(host.consent_active(Scope::TerminalInspect, NOW_MS));
    assert!(!host.consent_active(Scope::TerminalInspect, EXPIRED_MS));
    assert!(host.revoke_consent(Scope::TerminalInspect));
    assert!(!host.consent_active(Scope::TerminalInspect, NOW_MS));
    assert!(!host.revoke_consent(Scope::TerminalInspect));
}

// ── trait swap ──────────────────────────────────────────────────────────────

/// A generic consumer proves the mechanical swap: it only names [`BittyHost`],
/// so `FakeHost` today and `LiveBittyHost` tomorrow satisfy it without
/// call-site changes. No live code is constructed here.
fn use_host<H: BittyHost>(host: &mut H, request: &SnapshotRequest, now_ms: u64) -> bool {
    host.snapshot("terminal.snapshot", request, now_ms).is_ok()
}

#[test]
fn same_trait_serves_fakehost_and_future_live_host() {
    let mut host = inspect_host();
    host.push_snapshot_data(snapshot_data("hello"));
    assert!(use_host(&mut host, &snapshot_request(), NOW_MS));
    assert_eq!(host.client_id(), CLIENT);
}

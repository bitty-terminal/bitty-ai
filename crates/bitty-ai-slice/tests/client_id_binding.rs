//! Protocol identity to wire `client_id` binding rule (`AI-0065`).
//!
//! The host keys its consent ledger and execution store by a wire `client_id`
//! string, while the runtime owns the protocol principal
//! `ProtocolAgentId(owner.name)`. These tests pin the slice-side rule that
//! binds them: `client_id` is exactly the validated `owner.name` principal of
//! the bound [`IdentityBridge`], an unbound identity is refused, a
//! caller-supplied id that disagrees with the bound principal is refused, and
//! a bound principal longer than the upstream wire bound is refused. Every
//! refusal is total: no host is constructed, nothing is dispatched, and no
//! execution entry is stored.
//!
//! Deterministic: `now_ms` is caller-supplied and there is no wall clock,
//! thread, network, filesystem, or secret.

use bitty_ai_runtime::bridge::{IdentityBridge, ProtocolAgentId};
use bitty_ai_runtime::session::AgentInstanceId;
use bitty_ai_slice::bridge::{MAX_WIRE_CLIENT_ID_BYTES, verify_wire_client_id, wire_client_id};
use bitty_ai_slice::{BittyHost, FakeHost, IpcBridge, LiveBittyHost, SliceError};
use bitty_ipc::auth::MAX_SCOPED_ID_BYTES;
use bitty_ipc::error::IpcError;
use bitty_ipc::execution::{
    EXECUTION_SCOPE, EffectState, ExecutionRequest, ExecutionStatus, MAX_EXEC_CLIENT_ID_BYTES,
    RawExecutionOutput,
};
use bitty_ipc::scope::ScopeSet;
use bitty_ipc::tool_dispatch::MAX_TOOL_CLIENT_ID_BYTES;

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;
const PRINCIPAL: &str = "local.assistant";

/// A principal whose `owner.name` is valid for the runtime protocol bound
/// (each segment <= 64 bytes, total <= 128) but is over the host
/// `client_id` bound: `40 + 1 + 40 = 81` bytes.
fn over_long_principal() -> String {
    format!("{}.{}", "a".repeat(40), "b".repeat(40))
}

fn bound_identity(principal: &str) -> IdentityBridge {
    let mut identity = IdentityBridge::new();
    identity
        .bind(
            ProtocolAgentId::new(principal).expect("principal validates"),
            AgentInstanceId(1),
        )
        .expect("first bind succeeds");
    identity
}

fn exec_provider(_request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
    Ok(RawExecutionOutput {
        target_id: None,
        status: ExecutionStatus::Completed,
        exit_code: Some(0),
        stdout: "diff ok".to_owned(),
        stderr: String::new(),
        evidence_refs: Vec::new(),
        effect_state: EffectState::Completed,
    })
}

fn exec_request() -> ExecutionRequest {
    ExecutionRequest::new("git", vec!["diff".to_owned()]).with_allow_effects(true)
}

// ── the rule ────────────────────────────────────────────────────────────────

#[test]
fn bound_principal_derives_the_owner_name_client_id() {
    let identity = bound_identity(PRINCIPAL);

    // The wire id is the validated `owner.name` principal, deterministically.
    assert_eq!(wire_client_id(&identity).expect("bound"), PRINCIPAL);
    assert_eq!(
        wire_client_id(&identity).expect("bound"),
        wire_client_id(&identity).expect("bound"),
        "derivation is deterministic"
    );

    // A caller-supplied id is accepted only when it matches the binding.
    verify_wire_client_id(&identity, PRINCIPAL).expect("matching id accepted");
}

#[test]
fn wire_client_id_bound_reads_the_single_upstream_symbol() {
    // The slice length bound must not mirror a literal: it is the upstream
    // `bitty-ipc` scoped-id ceiling that both host dispatch paths already
    // alias. If upstream forks, these equalities fail and the rule is stale.
    assert_eq!(MAX_WIRE_CLIENT_ID_BYTES, MAX_SCOPED_ID_BYTES);
    assert_eq!(MAX_WIRE_CLIENT_ID_BYTES, MAX_TOOL_CLIENT_ID_BYTES);
    assert_eq!(MAX_WIRE_CLIENT_ID_BYTES, MAX_EXEC_CLIENT_ID_BYTES);
    assert_eq!(
        MAX_SCOPED_ID_BYTES, 64,
        "pinned upstream scoped-id ceiling moved; re-derive the binding rule"
    );
}

// ── negative: unbound ───────────────────────────────────────────────────────

#[test]
fn unbound_identity_is_refused() {
    let identity = IdentityBridge::new();

    let err = wire_client_id(&identity).expect_err("unbound must be refused");
    assert!(
        matches!(err, SliceError::Ipc(IpcError::Unauthenticated { .. })),
        "expected Unauthenticated, got {err:?}"
    );
    assert!(verify_wire_client_id(&identity, PRINCIPAL).is_err());

    // No construction path accepts an arbitrary id for an unbound identity.
    assert!(IpcBridge::from_binding(&identity, ScopeSet::all()).is_err());
    assert!(FakeHost::from_binding(&identity, ScopeSet::all()).is_err());
    assert!(LiveBittyHost::from_binding(&identity, ScopeSet::all(), None, exec_provider).is_err());
}

// ── negative: over-long derived id ──────────────────────────────────────────

#[test]
fn over_long_bound_principal_is_refused() {
    let principal = over_long_principal();
    assert_eq!(principal.len(), MAX_WIRE_CLIENT_ID_BYTES + 17);
    let identity = bound_identity(&principal);

    let err = wire_client_id(&identity).expect_err("over-long must be refused");
    match err {
        SliceError::Ipc(IpcError::LimitExceeded {
            field,
            limit,
            actual,
        }) => {
            assert_eq!(field, "wire client_id");
            assert_eq!(limit, MAX_WIRE_CLIENT_ID_BYTES);
            assert_eq!(actual, principal.len());
        }
        other => panic!("expected LimitExceeded, got {other:?}"),
    }

    assert!(IpcBridge::from_binding(&identity, ScopeSet::all()).is_err());
    assert!(FakeHost::from_binding(&identity, ScopeSet::all()).is_err());
    assert!(LiveBittyHost::from_binding(&identity, ScopeSet::all(), None, exec_provider).is_err());
}

// ── negative: disagreeing caller-supplied id ────────────────────────────────

#[test]
fn disagreeing_supplied_id_is_refused() {
    let identity = bound_identity(PRINCIPAL);

    let err = verify_wire_client_id(&identity, "local.imposter")
        .expect_err("disagreeing id must be refused");
    match err {
        SliceError::Ipc(IpcError::Denied { code, reason }) => {
            assert_eq!(code, "ClientIdMismatch");
            assert!(
                !reason.contains("imposter"),
                "refusal must not echo the supplied id: {reason}"
            );
        }
        other => panic!("expected Denied(ClientIdMismatch), got {other:?}"),
    }
}

// ── zero side effects and the counterfactual ────────────────────────────────

#[test]
fn refused_binding_never_dispatches_or_stores() {
    let identity = bound_identity(PRINCIPAL);

    // Positive control: the derived identity is served and stored exactly once.
    let mut host = FakeHost::from_binding(&identity, ScopeSet::single(EXECUTION_SCOPE))
        .expect("bound identity builds a host");
    assert_eq!(host.host_client_id(), PRINCIPAL);
    host.grant_consent(EXECUTION_SCOPE, NOW_MS, TTL_MS)
        .expect("consent grant");
    host.push_execution_output(exec_provider(&exec_request()).expect("provider"));
    let result = host
        .execute(&exec_request(), NOW_MS, 1)
        .expect("dispatch succeeds");
    assert_eq!(result.client_id, PRINCIPAL);
    assert_eq!(host.execution_count(), 1);
    assert_eq!(host.pending_exec_scripts(), 0);

    // A disagreeing id is refused before the host is reached: no store entry
    // and no consumed script beyond the positive control.
    assert!(verify_wire_client_id(&identity, "local.imposter").is_err());
    assert_eq!(host.execution_count(), 1, "refusal must not store");
    assert_eq!(host.pending_exec_scripts(), 0, "refusal must not consume");

    // Unbound and over-long identities are refused before a host (and
    // therefore any ledger, registry, or store) exists at all.
    assert!(FakeHost::from_binding(&IdentityBridge::new(), ScopeSet::all()).is_err());
    assert!(
        FakeHost::from_binding(&bound_identity(&over_long_principal()), ScopeSet::all()).is_err()
    );
}

#[test]
fn raw_constructor_is_the_unbound_seam_the_rule_replaces() {
    // The raw `new` seam is retained for explicit test/host wiring and still
    // accepts an id that disagrees with the bound principal; this is the
    // pre-change behavior the rule closes. `from_binding` refuses the same
    // identity, so product wiring has a fail-closed path.
    let identity = bound_identity(PRINCIPAL);
    let raw = FakeHost::new("local.imposter", ScopeSet::all()).expect("raw seam accepts");
    assert_eq!(raw.host_client_id(), "local.imposter");

    assert!(verify_wire_client_id(&identity, "local.imposter").is_err());
    assert!(FakeHost::from_binding(&identity, ScopeSet::all()).is_ok());
}

// ── both hosts serve the derived identity ───────────────────────────────────

#[test]
fn from_binding_hosts_serve_the_derived_identity() {
    let identity = bound_identity(PRINCIPAL);

    let fake = FakeHost::from_binding(&identity, ScopeSet::all()).expect("fake builds");
    assert_eq!(fake.host_client_id(), PRINCIPAL);
    assert_eq!(fake.client_id(), PRINCIPAL);

    let live = LiveBittyHost::from_binding(&identity, ScopeSet::all(), None, exec_provider)
        .expect("live builds");
    assert_eq!(live.host_client_id(), PRINCIPAL);
    assert_eq!(live.client_id(), PRINCIPAL);
}

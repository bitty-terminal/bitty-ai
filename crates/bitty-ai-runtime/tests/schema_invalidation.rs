//! Stale-schema denial proofs for the tool registry (AI-0090, AIQ-08 narrowing).
//!
//! AIQ-08 (MCP schema cache invalidation) is needs-evidence: a dispatch
//! authorized against schema version N must not silently execute when the
//! registered schema for that tool has changed to version N+1, and the
//! registry must expose a deterministic schema-identity mechanism so a
//! host-side version source can invalidate.
//!
//! Mechanism verdict against `tool.rs`: the re-registration half already
//! exists — `ToolRegistry::register` refuses every same-name re-registration
//! with `ToolError::DuplicateTool` (no replacement, no shadowing, no state
//! change), so changed bytes can never silently enter the registry. What was
//! missing was the identity half: no digest over `schema_json` existed. This
//! file pins `ToolSpec::schema_digest` (deterministic FNV-1a-64 over exactly
//! the schema bytes, mirroring the `cache_key.rs` precedent; std only, no
//! new deps) plus the end-to-end denial sequences through the public API.
//!
//! Pinned rule (refusal, not replace): same name plus different bytes is
//! refused; dispatches keep binding the originally registered schema. There
//! is no update/remove path on `ToolRegistry`, and `ToolBus` exposes no
//! registry-mutation handle after construction, so the spec resolved at the
//! dispatch boundary is identical to the one resolved at precheck regardless
//! of interleaving.
//!
//! Deterministic and offline: caller-built specs, an allow-hook recorder,
//! scripted [`FakeToolExecutor`] outcomes, caller-supplied `now_ms`. No
//! network, no secrets, no wall clock, no threads, no `HashMap`.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `tool.rs` unit tests: refusal mechanics (exact `DuplicateTool` error,
//!   length/name bookkeeping, full-registry precedence, revalidation of
//!   specs built without `new`). Refusal appears here only as one step
//!   inside the authorize-then-replace-then-dispatch sequence, asserted via
//!   the digest/scope stability halves those tests leave out.
//! - `batch_evidence.rs`: batch all-or-nothing precheck, ordering, cancel,
//!   cap boundary. Nothing here re-asserts batch shapes.
//! - `cache_invalidation.rs`: L1 supersede/rotation/artifact invalidation.
//!   Nothing here re-asserts those surfaces.
//! - `cache_key.rs`: FNV-1a-64 over canonical prompt bytes. The digest here
//!   covers tool schema bytes instead; the hasher is re-implemented in this
//!   file (not imported) so the value is pinned independently.

use std::cell::RefCell;
use std::rc::Rc;

use bitty_ai_runtime::{
    AgentLevel, AuthBase, AuthContext, AuthDecision, FakeToolExecutor, IdIssuer, ToolAuthorizer,
    ToolBus, ToolCall, ToolError, ToolRegistry, ToolSpec, ToolStatus,
};

const NOW_MS: u64 = 1_700_000_000_000;
/// Hard bus ceiling mirrored from the `TB-6` contract (`tool.rs`), asserted
/// here as a literal (not imported) because the constant is intentionally
/// crate-private to the bus boundary.
const BUS_CALL_CAP: usize = 8;
const TOOL_NAME: &str = "workspace_read";
const SCHEMA_N: &[u8] = br#"{"type":"object"}"#;
const SCHEMA_N_PLUS_1: &[u8] = br#"{"type":"object","properties":{"path":{"type":"string"}}}"#;

/// Deterministic FNV-1a-64, test-side re-computation of the schema digest.
/// Duplicated (not imported) on purpose: the test pins the digest value
/// independently of the implementation under test.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn spec_n() -> ToolSpec {
    ToolSpec::new(
        TOOL_NAME,
        "Read a bounded workspace path",
        SCHEMA_N.to_vec(),
        "workspace.read",
        true,
    )
    .expect("test spec N is valid")
}

fn spec_n_plus_1() -> ToolSpec {
    ToolSpec::new(
        TOOL_NAME,
        "Read a bounded workspace path with an extra path filter",
        SCHEMA_N_PLUS_1.to_vec(),
        "workspace.write",
        false,
    )
    .expect("test spec N+1 is valid")
}

/// Test-only allow hook that records the exact `(tool, required_scope,
/// read_only)` presented at each authorization boundary. The recorded
/// sequence is the binding probe: both the precheck pass and the dispatch
/// boundary must observe the originally registered spec, never the refused
/// replacement. Production wiring installs the host capability-plus-consent
/// hook instead.
#[derive(Debug, Clone, Default)]
struct BindingRecorder {
    seen: Rc<RefCell<Vec<(String, String, bool)>>>,
}

impl ToolAuthorizer for BindingRecorder {
    fn authorize(&self, ctx: &AuthContext) -> AuthDecision {
        self.seen.borrow_mut().push((
            ctx.tool.to_owned(),
            ctx.required_scope.to_owned(),
            ctx.read_only,
        ));
        AuthDecision::Allow
    }
}

fn base() -> AuthBase {
    let mut issuer = IdIssuer::default();
    AuthBase {
        agent_instance_id: issuer.agent_instance(),
        session_id: issuer.session(),
        level: AgentLevel::Inspect,
    }
}

fn read_call() -> ToolCall {
    ToolCall {
        name: TOOL_NAME.to_owned(),
        arguments: br#"{"path":"a"}"#.to_vec(),
    }
}

#[test]
fn schema_digest_is_deterministic_and_sensitive() {
    // Determinism: repeated construction over the same bytes compares equal
    // and matches the independently re-computed FNV-1a-64.
    let first = spec_n();
    let second = spec_n();
    assert_eq!(first.schema_digest(), second.schema_digest());
    assert_eq!(first.schema_digest(), fnv1a64(SCHEMA_N));
    // Sensitivity: any schema-byte change moves the digest.
    let changed = spec_n_plus_1();
    assert_ne!(first.schema_digest(), changed.schema_digest());
    assert_eq!(changed.schema_digest(), fnv1a64(SCHEMA_N_PLUS_1));
    // A single trailing-byte change moves the digest (never a silent alias).
    let mut one_byte_off = SCHEMA_N.to_vec();
    let last = one_byte_off.len() - 1;
    one_byte_off[last] ^= 0x01;
    let off_spec = ToolSpec::new(
        TOOL_NAME,
        "Read a bounded workspace path",
        one_byte_off.clone(),
        "workspace.read",
        true,
    )
    .expect("one-byte-off spec is valid");
    assert_ne!(first.schema_digest(), off_spec.schema_digest());
    assert_eq!(off_spec.schema_digest(), fnv1a64(&one_byte_off));
    // The empty/non-empty edge compares unequal with a pinned empty digest.
    let empty = ToolSpec::new(
        TOOL_NAME,
        "Read a bounded workspace path",
        Vec::new(),
        "workspace.read",
        true,
    )
    .expect("empty-schema spec is valid");
    assert_ne!(first.schema_digest(), empty.schema_digest());
    assert_eq!(empty.schema_digest(), fnv1a64(&[]));
}

#[test]
fn authorize_then_replace_then_dispatch_binds_original_schema() {
    // Register schema N and pin its digest before any authorization.
    let mut registry = ToolRegistry::new();
    registry
        .register(spec_n())
        .expect("test registry accepts N");
    let digest_n = registry
        .lookup(TOOL_NAME)
        .expect("N is registered")
        .schema_digest();
    // The host delivers changed schema N+1: same-name re-registration is
    // refused fail-closed, so the changed bytes never enter the registry.
    let error = registry
        .register(spec_n_plus_1())
        .expect_err("same-name different-bytes must fail");
    assert!(
        matches!(error, ToolError::DuplicateTool { .. }),
        "stale schema must be refused as a duplicate"
    );
    // No partial state: the digest, scope, and shape still pin N.
    let kept = registry.lookup(TOOL_NAME).expect("original schema is kept");
    assert_eq!(kept.schema_digest(), digest_n);
    assert_eq!(kept.required_scope, "workspace.read");
    assert!(kept.read_only);
    assert_eq!(kept.schema_json, SCHEMA_N.to_vec());
    // A bus over the refused registry authorizes and dispatches against N:
    // the refused N+1 authorizes nothing and executes nothing.
    let recorder = BindingRecorder::default();
    let mut bus = ToolBus::new(registry).with_authorizer(recorder.clone());
    let call = read_call();
    bus.precheck(std::slice::from_ref(&call), &base(), BUS_CALL_CAP)
        .expect("precheck authorizes against N");
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"ok".to_vec());
    let mut issuer = IdIssuer::default();
    let execution = bus
        .dispatch(&mut executor, &call, &base(), issuer.execution(), NOW_MS)
        .expect("dispatch binds the kept schema N");
    assert!(matches!(execution.status, ToolStatus::Success));
    assert_eq!(execution.tool, TOOL_NAME);
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(executor.calls()[0].0, TOOL_NAME);
    // Both authorization boundaries observed N's scope and shape; N+1 was
    // never presented to the hook, so no stale grant exists.
    assert_eq!(
        *recorder.seen.borrow(),
        vec![
            (TOOL_NAME.to_owned(), "workspace.read".to_owned(), true),
            (TOOL_NAME.to_owned(), "workspace.read".to_owned(), true),
        ]
    );
}

#[test]
fn inflight_call_binds_the_spec_authorized_at_precheck() {
    // In-flight binding: the dispatch boundary re-resolves the registry
    // (PP-6 re-check) and must observe the same spec the precheck pass
    // authorized. The hook record proves both boundaries bound N; the host
    // compares `schema_digest` across snapshots to detect drift outside the
    // bus.
    let mut registry = ToolRegistry::new();
    registry
        .register(spec_n())
        .expect("test registry accepts N");
    let recorder = BindingRecorder::default();
    let mut bus = ToolBus::new(registry).with_authorizer(recorder.clone());
    let call = read_call();
    bus.precheck(std::slice::from_ref(&call), &base(), BUS_CALL_CAP)
        .expect("precheck authorizes the call");
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"ok".to_vec());
    let mut issuer = IdIssuer::default();
    let execution = bus
        .dispatch(&mut executor, &call, &base(), issuer.execution(), NOW_MS)
        .expect("dispatch executes the authorized call");
    assert!(matches!(execution.status, ToolStatus::Success));
    assert!(execution.is_untrusted_surface);
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(
        *recorder.seen.borrow(),
        vec![
            (TOOL_NAME.to_owned(), "workspace.read".to_owned(), true),
            (TOOL_NAME.to_owned(), "workspace.read".to_owned(), true),
        ]
    );
}

#[test]
fn identical_schema_reregistration_is_still_refused() {
    // Digest equality never smuggles a replacement: re-registering the exact
    // same bytes under the same name is refused too, so the digest is an
    // invalidation handle for the host, never a replace trigger.
    let mut registry = ToolRegistry::new();
    registry
        .register(spec_n())
        .expect("test registry accepts N");
    let digest_before = registry
        .lookup(TOOL_NAME)
        .expect("N is registered")
        .schema_digest();
    let error = registry
        .register(spec_n())
        .expect_err("identical re-registration must fail");
    assert!(
        matches!(error, ToolError::DuplicateTool { .. }),
        "identical bytes must still be refused as a duplicate"
    );
    assert_eq!(
        registry
            .lookup(TOOL_NAME)
            .expect("N is kept")
            .schema_digest(),
        digest_before
    );
}

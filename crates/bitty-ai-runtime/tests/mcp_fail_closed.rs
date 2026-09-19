//! Fail-closed MCP path without a version source (AI-0096, AIQ-08 narrowing).
//!
//! SEAM PROOF ONLY. This file builds no MCP transport, client, or protocol
//! code: there is no MCP module, type, or wire shape anywhere in this crate
//! (the only MCP mentions are prose in `lib.rs` and the `schema_invalidation`
//! header). What is pinned here is the bitty-ai-side seam against the
//! hypothetical future importer: when no MCP version source is present, the
//! MCP path must fail closed — deny, never silently use an unversioned
//! schema. The host owns the version source; this crate only proves the
//! entry seam cannot accept a schema the host did not version.
//!
//! Gap verdict (NO-GAP, read against `tool.rs` at this revision):
//! - `ToolRegistry` exposes exactly one mutation: `register`, which refuses
//!   every same-name re-registration (`DuplicateTool`), revalidates specs
//!   built without `ToolSpec::new`, and refuses over-bound specs. There is
//!   no update/remove/replace/clear/extend path.
//! - `ToolSpec` carries `schema_digest` by construction: every spec,
//!   however built, has schema bytes (possibly empty) and therefore a
//!   digest. There is no `Option<digest>` / unversioned-spec shape.
//! - `ToolBus` exposes no registry-mutation handle after construction; the
//!   dispatch boundary re-resolves the spec from the registry and the
//!   executor receives only `(name, arguments)` — it cannot inject a spec.
//! - `authorize_call` (shared by `precheck` and `dispatch`) fails closed on
//!   unknown names (`UnknownTool`) and on malformed names (`InvalidName`)
//!   before any authorization or dispatch.
//!
//! A foreign schema therefore has exactly one way in (`register`), where it
//! is either refused (same name, malformed, over-bound) or becomes a
//! first-class registered spec with a digest that `schema_invalidation.rs`
//! already pins. The MCP-side version source stays host-owned and absent,
//! and nothing here invents it.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `schema_invalidation.rs` (AI-0090): the registry half — deterministic
//!   `schema_digest` values (FNV-1a-64, independently re-computed), the
//!   authorize-then-replace-then-dispatch binding sequence with a hook
//!   recorder, in-flight precheck/dispatch binding agreement, and identical-
//!   bytes re-registration refusal. Digest *values* are never re-asserted
//!   here.
//! - `tool.rs` unit tests: refusal mechanics (exact error shapes, length
//!   bookkeeping, full-registry precedence, `new`-bypass revalidation,
//!   hook re-check at the dispatch boundary). Refusal shapes appear here
//!   only as one step inside the no-version-source sequences below.
//! - `runtime_fail_closed.rs` / `batch_evidence.rs`: `UnknownTool` through
//!   `run_turn` and in mid-batch position. `UnknownTool` appears here only
//!   on the bare bus (`precheck`/`dispatch`), never through the agent loop
//!   or a batch.
//! - `extension_points.rs`: the extension manifest surface has no registry,
//!   no resolution, and no versioning; nothing here re-asserts it.
//!
//! Deterministic and offline: caller-built specs, a deny-by-default hook,
//! an allow hook, scripted [`FakeToolExecutor`] outcomes, caller-supplied
//! `now_ms`. No network, no secrets, no wall clock, no threads, no
//! `HashMap`.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static
//! only; tool names, scopes, digests, and byte lengths never appear in
//! message strings.

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
const FOREIGN_NAME: &str = "mcp_foreign_tool";
const SCHEMA_N: &[u8] = br#"{"type":"object"}"#;
const SCHEMA_DRIFTED: &[u8] = br#"{"type":"object","properties":{"path":{"type":"string"}}}"#;

/// Test-only allow hook. Production wiring installs the host
/// capability-plus-consent hook instead.
struct Allow;
impl ToolAuthorizer for Allow {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
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

fn versioned_spec() -> ToolSpec {
    ToolSpec::new(
        TOOL_NAME,
        "Read a bounded workspace path",
        SCHEMA_N.to_vec(),
        "workspace.read",
        true,
    )
    .expect("test spec is valid")
}

fn foreign_spec() -> ToolSpec {
    // A hypothetical MCP-supplied spec arriving without any version source
    // still enters through `register` (the only mutation), with schema
    // bytes and therefore a digest by construction.
    ToolSpec::new(
        FOREIGN_NAME,
        "Unversioned foreign tool",
        SCHEMA_DRIFTED.to_vec(),
        "workspace.read",
        true,
    )
    .expect("test foreign spec is valid")
}

fn call(name: &str) -> ToolCall {
    ToolCall {
        name: name.to_owned(),
        arguments: br#"{"path":"a"}"#.to_vec(),
    }
}

#[test]
fn externally_supplied_spec_still_refuses_same_name_reregistration() {
    // An externally supplied (MCP-shaped) spec is first-class: registering
    // it succeeds exactly once, and a second, drifted spec under the same
    // name — the no-version-source replay — is refused fail-closed with no
    // replacement, no shadowing, and no state change.
    let mut registry = ToolRegistry::new();
    registry
        .register(foreign_spec())
        .expect("first registration succeeds");
    let digest_before = registry
        .lookup(FOREIGN_NAME)
        .expect("foreign spec is registered")
        .schema_digest();
    let drifted = ToolSpec::new(
        FOREIGN_NAME,
        "Drifted foreign description",
        SCHEMA_N.to_vec(),
        "workspace.write",
        false,
    )
    .expect("test drifted spec is valid");
    let error = registry
        .register(drifted)
        .expect_err("drifted re-registration must fail");
    assert!(
        matches!(error, ToolError::DuplicateTool { .. }),
        "unversioned replay must be refused as a duplicate"
    );
    let kept = registry.lookup(FOREIGN_NAME).expect("original is kept");
    assert_eq!(kept.schema_digest(), digest_before);
    assert_eq!(kept.required_scope, "workspace.read");
    assert!(kept.read_only);
    assert_eq!(kept.schema_json, SCHEMA_DRIFTED.to_vec());
    assert_eq!(registry.len(), 1);
}

#[test]
fn dispatch_binds_the_registered_spec_never_a_call_side_schema() {
    // `ToolCall` carries no schema: name plus arguments only. A call naming
    // a registered tool authorizes and dispatches against the registered
    // spec, and the executor observes exactly the call name — a foreign
    // schema cannot ride along the call.
    let mut registry = ToolRegistry::new();
    registry
        .register(versioned_spec())
        .expect("registry accepts");
    let mut bus = ToolBus::new(registry).with_authorizer(Allow);
    let invocation = call(TOOL_NAME);
    bus.precheck(std::slice::from_ref(&invocation), &base(), BUS_CALL_CAP)
        .expect("precheck authorizes");
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"ok".to_vec());
    let mut issuer = IdIssuer::default();
    let execution = bus
        .dispatch(
            &mut executor,
            &invocation,
            &base(),
            issuer.execution(),
            NOW_MS,
        )
        .expect("dispatch executes");
    assert_eq!(execution.tool, TOOL_NAME);
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(executor.calls()[0].0, TOOL_NAME);
    assert_eq!(executor.calls()[0].1, br#"{"path":"a"}"#.to_vec());
    assert!(execution.is_untrusted_surface);
}

#[test]
fn every_spec_carries_a_digest_by_construction() {
    // There is no unversioned-spec shape: a spec built through `new` and a
    // spec built field-by-field (the way a future importer would assemble
    // one) both carry schema bytes and therefore a digest. The digest moves
    // with any byte change, so drift is always observable host-side.
    let via_new = versioned_spec();
    let via_fields = ToolSpec {
        name: TOOL_NAME.to_owned(),
        description: "Read a bounded workspace path".to_owned(),
        schema_json: SCHEMA_N.to_vec(),
        required_scope: "workspace.read".to_owned(),
        read_only: true,
    };
    assert_eq!(via_new.schema_digest(), via_fields.schema_digest());
    let drifted = ToolSpec {
        name: TOOL_NAME.to_owned(),
        description: "Read a bounded workspace path".to_owned(),
        schema_json: SCHEMA_DRIFTED.to_vec(),
        required_scope: "workspace.read".to_owned(),
        read_only: true,
    };
    assert_ne!(via_new.schema_digest(), drifted.schema_digest());
    // The empty-schema edge still has a digest (never an absent version).
    let empty = ToolSpec::new(
        TOOL_NAME,
        "Read a bounded workspace path",
        Vec::new(),
        "workspace.read",
        true,
    )
    .expect("empty-schema spec is valid");
    assert_ne!(via_new.schema_digest(), empty.schema_digest());
}

#[test]
fn precheck_fails_closed_on_unknown_tool_with_no_dispatch() {
    // Without a registered spec there is no version to consult, so the path
    // fails closed: `UnknownTool`, no dispatch, no counter increment.
    let registry = ToolRegistry::new();
    let bus = ToolBus::new(registry).with_authorizer(Allow);
    let invocation = call(FOREIGN_NAME);
    let error = bus
        .precheck(std::slice::from_ref(&invocation), &base(), BUS_CALL_CAP)
        .expect_err("unknown tool must fail");
    assert!(
        matches!(error, ToolError::UnknownTool { .. }),
        "unregistered name must fail as unknown"
    );
    assert_eq!(bus.calls_this_turn(), 0);
    assert!(bus.tool_names().is_empty());
}

#[test]
fn dispatch_fails_closed_on_unknown_tool_with_no_executor_contact() {
    // The dispatch boundary re-checks the registry before touching the
    // executor: an unregistered (unversioned) name is refused with no
    // executor contact and no per-turn counter increment. The refusal is a
    // pre-dispatch admission decision (AI-RUN-004): `Refused` carrying the
    // typed `UnknownTool` cause, never an executed failure.
    let registry = ToolRegistry::new();
    let mut bus = ToolBus::new(registry).with_authorizer(Allow);
    let invocation = call(FOREIGN_NAME);
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never run", b"nope".to_vec());
    let mut issuer = IdIssuer::default();
    let execution = bus
        .dispatch(
            &mut executor,
            &invocation,
            &base(),
            issuer.execution(),
            NOW_MS,
        )
        .expect("admission refusal is a recorded status, not a bus error");
    assert!(execution.status.is_admission_refusal());
    assert!(
        matches!(
            &execution.status,
            ToolStatus::Refused {
                cause: ToolError::UnknownTool { .. }
            }
        ),
        "unregistered name must be refused as a typed unknown-tool admission"
    );
    assert!(executor.calls().is_empty());
    assert_eq!(bus.calls_this_turn(), 0);
}

#[test]
fn missing_authorizer_denies_even_a_registered_spec() {
    // Belt and braces for the no-version-source posture: even a registered,
    // digest-carrying spec is denied when no authorizer is installed, with
    // no dispatch and no partial state.
    let mut registry = ToolRegistry::new();
    registry
        .register(versioned_spec())
        .expect("registry accepts");
    let bus = ToolBus::new(registry);
    let invocation = call(TOOL_NAME);
    let error = bus
        .precheck(std::slice::from_ref(&invocation), &base(), BUS_CALL_CAP)
        .expect_err("missing authorizer must fail");
    assert!(
        matches!(error, ToolError::Denied { .. }),
        "missing authorizer must deny"
    );
    assert_eq!(bus.calls_this_turn(), 0);
}

//! Report-only vocabulary conformance: `bitty-agent-api` vs runtime twins (AI-0257).
//!
//! Maps field-by-field between the agent vocabulary at the pinned
//! `bitty-agent` green-main rev `a8e4112`
//! (`a8e4112116a7e6c74382ab8294a98096f74d2a14`, push CI + CodeQL green) and
//! the `bitty-ai-runtime` twins. Each mapping asserts CURRENT equivalence;
//! drift fails loudly with a message naming both sides (file:line at the
//! pinned rev vs this repo path). Drift is reported, never silently fixed:
//! no production type in `src/` changes to match, per task scope — with the
//! single owner-directed exception of D3 (AI-0316, CONVERGE), where the
//! runtime tool-name grammar was widened to the agent class and the drift
//! test became a converge proof.
//!
//! Method: `bitty-agent-api` at `a8e4112` exposes only `AgentId` plus
//! `AgentError` (`crates/bitty-agent-api/src/id.rs`, `error.rs`), so only
//! the `AgentId` mapping compiles against both sides live. The remaining
//! vocabulary (`ToolSpec`, `ToolCall`, `ToolResult`, `AgentObservation`,
//! `SideQueue`, `AgentSession`) lives in the `bitty-agent` crate at that
//! rev, which is deliberately NOT a dependency here; those mappings pin the
//! agent-side expected scalars/bounds as constants with exact file:line
//! citations and assert the runtime twin against them. That is a
//! report-only bound check, not vendored types: no agent struct is
//! redefined here.
//!
//! Reachability: `bitty-agent-api` resolved and built (this target
//! compiling is the proof). Had it been unreachable or unbuildable, that
//! would be the finding instead of these mappings.
//!
//! Verdicts (M = match, D = drift, test name in parentheses):
//! - AgentId bounds (`agent_id_len_bounds_match`): M.
//! - AgentId grammar + owner/name split (`agent_id_grammar_parity`): M.
//! - ToolSpec name bound 64 (`tool_name_len_bound_matches`): M.
//! - ToolSpec registry cap 32 (`tool_registry_cap_matches`): M (name differs:
//!   `MAX_TOOLS_PER_SESSION` vs `MAX_TOOLS_PER_AGENT`).
//! - ToolSpec description bound (`toolspec_description_bound_drift`): D
//!   (runtime 512 vs agent 1024).
//! - ToolSpec schema bound (`toolspec_schema_bound_drift`): D (runtime
//!   16 KiB vs agent 8 KiB).
//! - ToolSpec shape: D by construction (runtime adds `required_scope`,
//!   `read_only`, `placement`, `native`, `schema_json: Vec<u8>` vs agent
//!   `input_schema: String`); covered by the bound tests, no silent fix.
//! - Tool name grammar shared core (`tool_name_grammar_shared_core_matches`): M.
//! - Tool name grammar `-`/`.` (`tool_name_grammar_converged`): M
//!   (AI-0316 CONVERGE: runtime widened to the agent class
//!   `^[a-z][a-z0-9_.-]*$`; the former drift test is now the converge
//!   proof and runs un-ignored).
//! - ToolCall args bound 16 KiB (`tool_args_bound_matches`): M.
//! - ToolCall id (`toolcall_id_field_drift`): D (runtime `ToolCall` has no
//!   caller id; agent requires `id`, correlated by `ToolResult.call_id`).
//! - ToolResult bound 16 KiB (`tool_result_bound_matches`): M (cap only).
//! - ToolResult shape (`toolresult_shape_drift`): D (no `ToolResult` twin;
//!   runtime splits into `ToolSuccess` + `ToolExecution` keyed by
//!   bus-issued `ExecutionId`).
//! - Observation untrusted flag (`executed_tool_outcome_carries_untrusted_flag`): M.
//! - Observation type/cap (`observation_type_drift`): D (no
//!   `AgentObservation` twin, no 8 KiB observation cap in runtime).
//! - Queue default capacity 64 (`bounded_queue_capacity_default_matches`): M.
//! - Queue never-blocking shed-oldest policy
//!   (`bounded_queue_never_blocks_sheds_oldest`): M (policy; types differ).
//! - Session bounds/state (`session_bounds_drift`): D (runtime
//!   `AgentSession` holds ids/level/generation/state only: no message
//!   history caps, no side queue, different state machine).
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - Error taxonomy (`AgentError` vs `ToolError`/`BridgeError`/`SessionError`)
//!   is out of scope; both sides are typed fail-closed, mapping it is a
//!   separate task.
//! - Consent/placement/scrubbing extensions on either side are product
//!   surface, not vocabulary equivalence, and are not asserted here.
//! - `provider.rs`/`prompt.rs`/`context.rs` unit tests own bound behavior;
//!   this file only pins cross-crate equivalence.
//!
//! Deterministic and offline: no network, no secrets, no wall clock, no
//! threads. All messages are static literals (CodeQL lesson from AI-0082:
//! no dynamic data in assert/panic strings).
//!
//! Covered here (and only here): the 18 AI-0257 mappings above (12 match
//! plus 6 drift-ledger, the drift tests ignored but live, failing under the
//! ignored-only run; D3 converged per the AI-0316 owner ruling and runs
//! un-ignored).

use bitty_agent_api::{AgentId, MAX_AGENT_ID_LEN, MAX_AGENT_ID_SEGMENT_LEN};
use bitty_ai_runtime::tool::{
    MAX_TOOL_ARGUMENTS_BYTES, MAX_TOOL_CALLS_PER_TURN, MAX_TOOL_DESCRIPTION_LEN, MAX_TOOL_NAME_LEN,
    MAX_TOOL_RESULT_BYTES, MAX_TOOL_SCHEMA_BYTES, MAX_TOOLS_PER_SESSION,
};
use bitty_ai_runtime::{
    AgentInstanceId, AgentLevel, AgentSession, AuthBase, AuthContext, AuthDecision, BoundedSink,
    DEFAULT_BOUNDED_SINK_CAPACITY, FakeConsentLedger, FakeToolExecutor, Fragment, IdIssuer,
    MAX_PROTOCOL_ID_LEN, MAX_PROTOCOL_ID_SEGMENT_LEN, ProtocolAgentId, StreamAttribution,
    StreamChunk, StreamHandle, ToolAuthorizer, ToolBus, ToolCall, ToolPlacement, ToolRegistry,
    ToolSpec, validate_tool_name,
};

// Pinned agent-side expectations at bitty-agent rev `a8e4112`
// (crates/bitty-agent/src/tool.rs:41-62).
const AGENT_MAX_TOOL_NAME_LEN: usize = 64;
const AGENT_MAX_TOOL_DESCRIPTION_LEN: usize = 1024;
const AGENT_MAX_TOOL_SCHEMA_BYTES: usize = 8 * 1024;
const AGENT_MAX_TOOL_ARGS_BYTES: usize = 16 * 1024;
const AGENT_MAX_TOOL_RESULT_BYTES: usize = 16 * 1024;
const AGENT_MAX_TOOL_CALLS_PER_TURN: usize = 8;
const AGENT_MAX_TOOLS_PER_AGENT: usize = 32;

// Pinned agent-side expectations at bitty-agent rev `a8e4112`
// (crates/bitty-agent/src/session.rs:17: DEFAULT_SIDE_QUEUE_CAPACITY).
const AGENT_DEFAULT_SIDE_QUEUE_CAPACITY: usize = 64;

#[test]
fn agent_id_len_bounds_match() {
    // bitty-agent-api id.rs:6 vs runtime bridge.rs:75: total bound.
    assert_eq!(
        MAX_PROTOCOL_ID_LEN, MAX_AGENT_ID_LEN,
        "AgentId total bound drift"
    );
    // bitty-agent-api id.rs:9 vs runtime bridge.rs:80: segment bound.
    assert_eq!(
        MAX_PROTOCOL_ID_SEGMENT_LEN, MAX_AGENT_ID_SEGMENT_LEN,
        "AgentId segment bound drift"
    );
}

#[test]
fn agent_id_grammar_parity() {
    // bitty-agent-api id.rs:68-124 vs runtime bridge.rs:250-308: identical
    // owner.name grammar (non-empty, <= 128 B, no whitespace, exactly one
    // dot, segments non-empty, <= 64 B, start [a-z], rest [a-z0-9_-]).
    for id in ["local.assistant", "bitty.agent-1", "x.y_z-0", "a.b"] {
        assert!(AgentId::new(id).is_ok(), "agent AgentId rejects a valid id");
        assert!(
            ProtocolAgentId::new(id).is_ok(),
            "runtime ProtocolAgentId rejects a valid id"
        );
    }
    for id in [
        "",
        "local",
        "Local.assistant",
        "local.assistant.extra",
        "1local.assistant",
        "local.1assistant",
        "local.assistant ",
        ".assistant",
        "local.",
    ] {
        assert!(
            AgentId::new(id).is_err(),
            "agent AgentId accepts an invalid id"
        );
        assert!(
            ProtocolAgentId::new(id).is_err(),
            "runtime ProtocolAgentId accepts an invalid id"
        );
    }
    // Segment cap (67 B total, 65 B segment): both refuse.
    let long_segment = format!("{}.b", "a".repeat(65));
    assert!(AgentId::new(&long_segment).is_err(), "agent segment cap");
    assert!(
        ProtocolAgentId::new(&long_segment).is_err(),
        "runtime segment cap"
    );
    // Total cap (129 B): both refuse. Boundary (128 B): both accept.
    let too_long = format!("{}.{}", "a".repeat(64), "b".repeat(64));
    assert!(AgentId::new(&too_long).is_err(), "agent total cap");
    assert!(
        ProtocolAgentId::new(&too_long).is_err(),
        "runtime total cap"
    );
    let boundary = format!("{}.{}", "a".repeat(63), "b".repeat(64));
    assert_eq!(boundary.len(), 128);
    assert!(AgentId::new(&boundary).is_ok(), "agent boundary");
    assert!(ProtocolAgentId::new(&boundary).is_ok(), "runtime boundary");
    // Owner/name split agrees on a validated id.
    let agent = AgentId::new("local.assistant").expect("valid id");
    let runtime = ProtocolAgentId::new("local.assistant").expect("valid id");
    assert_eq!(agent.owner(), runtime.owner());
    assert_eq!(agent.name(), runtime.name());
    assert_eq!(agent.as_str(), runtime.as_str());
    assert_eq!(agent.to_string(), runtime.to_string());
}

#[test]
fn tool_name_len_bound_matches() {
    // Runtime tool.rs:24 vs agent tool.rs:41: both 64 B.
    assert_eq!(
        MAX_TOOL_NAME_LEN, AGENT_MAX_TOOL_NAME_LEN,
        "tool name bound drift"
    );
}

#[test]
fn tool_name_grammar_shared_core_matches() {
    // Shared core of runtime tool.rs:299-316 and agent tool.rs:1295-1323:
    // lowercase-start alphanumerics plus underscore agree on both sides.
    for name in ["read_file", "tool1", "my_tool_2"] {
        assert!(
            validate_tool_name(name).is_ok(),
            "runtime rejects a shared-core name"
        );
    }
    for name in ["", "1tool", "Tool", "tool with space"] {
        assert!(
            validate_tool_name(name).is_err(),
            "runtime accepts a name the shared core rejects"
        );
    }
    let too_long = "a".repeat(AGENT_MAX_TOOL_NAME_LEN + 1);
    assert!(
        validate_tool_name(&too_long).is_err(),
        "runtime accepts an over-long name"
    );
}

#[test]
fn tool_registry_cap_matches() {
    // Runtime tool.rs:30 (MAX_TOOLS_PER_SESSION) vs agent tool.rs:59
    // (MAX_TOOLS_PER_AGENT): both 32. Names differ; value matches.
    assert_eq!(
        MAX_TOOLS_PER_SESSION, AGENT_MAX_TOOLS_PER_AGENT,
        "tool registry cap drift"
    );
    let mut registry = ToolRegistry::new();
    for i in 0..MAX_TOOLS_PER_SESSION {
        registry
            .register(
                ToolSpec::new(
                    format!("tool_{i}"),
                    "stub",
                    b"{}".to_vec(),
                    "workspace.read",
                    true,
                    ToolPlacement::Mcp,
                    None,
                )
                .expect("valid spec"),
            )
            .expect("registry admits up to cap");
    }
    assert_eq!(registry.len(), AGENT_MAX_TOOLS_PER_AGENT);
}

#[test]
fn tool_calls_per_turn_matches() {
    // Runtime tool.rs:42 vs agent tool.rs:56: both 8.
    assert_eq!(
        MAX_TOOL_CALLS_PER_TURN, AGENT_MAX_TOOL_CALLS_PER_TURN,
        "tool calls per turn drift"
    );
}

#[test]
fn tool_args_bound_matches() {
    // Runtime tool.rs:44 (MAX_TOOL_ARGUMENTS_BYTES) vs agent tool.rs:50
    // (MAX_TOOL_ARGS_BYTES): both 16 KiB. Names differ; value matches.
    assert_eq!(
        MAX_TOOL_ARGUMENTS_BYTES, AGENT_MAX_TOOL_ARGS_BYTES,
        "tool args bound drift"
    );
}

#[test]
fn tool_result_bound_matches() {
    // Runtime tool.rs:46 vs agent tool.rs:53: both 16 KiB (cap only; the
    // result SHAPE drifts, see toolresult_shape_drift).
    assert_eq!(
        MAX_TOOL_RESULT_BYTES, AGENT_MAX_TOOL_RESULT_BYTES,
        "tool result bound drift"
    );
}

#[test]
fn bounded_queue_capacity_default_matches() {
    // Runtime stream.rs:256 (DEFAULT_BOUNDED_SINK_CAPACITY) vs agent
    // session.rs:17 (DEFAULT_SIDE_QUEUE_CAPACITY): both 64.
    assert_eq!(
        DEFAULT_BOUNDED_SINK_CAPACITY, AGENT_DEFAULT_SIDE_QUEUE_CAPACITY,
        "bounded queue default capacity drift"
    );
}

#[test]
fn bounded_queue_never_blocks_sheds_oldest() {
    // Policy parity with agent queue.rs:70-77 (SideQueue::push): producers
    // never block; at capacity the oldest entry is shed and a counter
    // increments. Runtime twin is BoundedSink (stream.rs:301-315), which is
    // chunk-typed rather than generic; the policy is what this asserts.
    let attribution = StreamAttribution {
        agent: AgentInstanceId(1),
        handle: StreamHandle(1),
        generation: 1,
    };
    let mut sink = BoundedSink::with_capacity(attribution, 2);
    for seq in 0..3 {
        sink.emit_attributed(StreamChunk {
            seq,
            total: 3,
            is_final: seq + 1 == 3,
            fragment: Fragment::markdown(vec![seq as u8]),
        })
        .expect("valid chunk accepted");
    }
    assert_eq!(sink.chunks().len(), 2);
    assert_eq!(sink.shed_count(), 1);
    assert_eq!(sink.accepted_count(), 3);
    let retained: Vec<u32> = sink.chunks().iter().map(|chunk| chunk.seq).collect();
    assert_eq!(retained, vec![1, 2]);
}

struct AllowAuthorizer;

impl ToolAuthorizer for AllowAuthorizer {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

#[test]
fn executed_tool_outcome_carries_untrusted_flag() {
    // Flag parity with agent observation.rs:195-200
    // (AgentObservation::is_untrusted_surface, true for TerminalOutput):
    // runtime marks executed tool outcomes untrusted via
    // ToolExecution::is_untrusted_surface (tool.rs:804-805), set true on
    // the executed-success path (tool.rs:1789).
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "read_file",
                "stub",
                b"{}".to_vec(),
                "workspace.read",
                true,
                ToolPlacement::Mcp,
                None,
            )
            .expect("valid spec"),
        )
        .expect("registry admits spec");
    let mut ledger = FakeConsentLedger::new();
    ledger
        .grant("local.assistant", "read_file", "workspace.read", 60_000)
        .expect("grant records");
    let mut bus = ToolBus::new(registry)
        .with_authorizer(AllowAuthorizer)
        .with_consent(ledger, "local.assistant");
    let mut ids = IdIssuer::default();
    let session = AgentSession::new(ids.agent_instance(), ids.run(), ids.session());
    let base = AuthBase {
        agent_instance_id: session.agent_instance_id(),
        session_id: session.session_id(),
        level: AgentLevel::Inspect,
    };
    let call = ToolCall {
        name: "read_file".to_owned(),
        arguments: b"{}".to_vec(),
    };
    let mut executor = FakeToolExecutor::new();
    executor.push_success("ok", b"data".to_vec());
    let execution = bus
        .dispatch(&mut executor, &call, &base, ids.execution(), 1_000)
        .expect("allowed consented call dispatches");
    assert!(execution.is_untrusted_surface, "executed outcome trusted");
}

// DRIFT-LEDGER (ignored but live): runtime tool.rs MAX_TOOL_DESCRIPTION_LEN=512
// vs bitty-agent@a8e4112 tool.rs MAX_TOOL_DESCRIPTION_LEN=1024.
// ARCHIVED (owner ruling 2026-10-16): Independent local fail-closed bounds
// with no shared declaration path.
#[test]
#[ignore = "drift ledger: runtime MAX_TOOL_DESCRIPTION_LEN=512 vs agent MAX_TOOL_DESCRIPTION_LEN=1024"]
fn toolspec_description_bound_drift() {
    // DRIFT toolspec.description bound: runtime tool.rs:26 caps at 512 B
    // while bitty-agent tool.rs:44 caps at 1024 B. Fails loudly; the
    // runtime twin is NOT widened to match (report-only).
    assert_eq!(
        MAX_TOOL_DESCRIPTION_LEN, AGENT_MAX_TOOL_DESCRIPTION_LEN,
        "DRIFT toolspec.description: runtime tool.rs MAX_TOOL_DESCRIPTION_LEN=512 vs bitty-agent@a8e4112 tool.rs MAX_TOOL_DESCRIPTION_LEN=1024"
    );
}

// DRIFT-LEDGER (ignored but live): runtime tool.rs MAX_TOOL_SCHEMA_BYTES=16KiB
// vs bitty-agent@a8e4112 tool.rs MAX_TOOL_SCHEMA_BYTES=8KiB.
// ARCHIVED (owner ruling 2026-10-16): Opposite-direction looseness, neither
// side normative, alignment without shared wire type is arbitrary churn.
#[test]
#[ignore = "drift ledger: runtime MAX_TOOL_SCHEMA_BYTES=16KiB vs agent MAX_TOOL_SCHEMA_BYTES=8KiB"]
fn toolspec_schema_bound_drift() {
    // DRIFT toolspec.schema bound: runtime tool.rs:28 caps at 16 KiB while
    // bitty-agent tool.rs:47 caps at 8 KiB. Fails loudly; the runtime twin
    // is NOT narrowed to match (report-only).
    assert_eq!(
        MAX_TOOL_SCHEMA_BYTES, AGENT_MAX_TOOL_SCHEMA_BYTES,
        "DRIFT toolspec.schema: runtime tool.rs MAX_TOOL_SCHEMA_BYTES=16KiB vs bitty-agent@a8e4112 tool.rs MAX_TOOL_SCHEMA_BYTES=8KiB"
    );
}

// CONVERGED (owner ruling 2026-10-16, CONVERGE D3): runtime tool.rs now
// accepts the agent class `^[a-z][a-z0-9_.-]*$` (bitty-agent@a8e4112
// tool.rs grammar `start with [a-z], then [a-z0-9_.-]`), so the former
// drift is the converge proof and this test runs un-ignored.
#[test]
fn tool_name_grammar_converged() {
    // Converged tool-name grammar: runtime tool.rs `validate_tool_name`
    // accepts `-` and `.` exactly like bitty-agent tool.rs
    // (`valid_tool_names` accepts "run-command" and "a.b").
    for name in ["run-command", "a.b", "my-tool_2"] {
        assert!(
            validate_tool_name(name).is_ok(),
            "converged tool-name grammar: runtime tool.rs rejects `-`/`.` names that bitty-agent@a8e4112 tool.rs accepts"
        );
    }
    // The widening is exact: the still-rejected classes stay rejected.
    for name in ["", "1tool", "Tool", "tool with space", "tool\0name"] {
        assert!(
            validate_tool_name(name).is_err(),
            "converged tool-name grammar: runtime tool.rs accepts a name both sides reject"
        );
    }
    let too_long = "a".repeat(AGENT_MAX_TOOL_NAME_LEN + 1);
    assert!(
        validate_tool_name(&too_long).is_err(),
        "converged tool-name grammar: runtime accepts an over-long name"
    );
}

// DRIFT-LEDGER (ignored but live): runtime tool.rs ToolCall has no caller id
// vs bitty-agent@a8e4112 tool.rs ToolCall.id with ToolResult.call_id correlation.
// ARCHIVED (owner ruling 2026-10-16): Bus-issued ExecutionId (AI-0106) is the
// deliberate correlation identity.
#[test]
#[ignore = "drift ledger: runtime ToolCall has no caller id vs agent ToolCall.id with ToolResult.call_id"]
fn toolcall_id_field_drift() {
    // DRIFT tool-call id: runtime ToolCall (tool.rs:660) carries only
    // {name, arguments} with no caller-stable id, while bitty-agent ToolCall
    // (tool.rs:919) requires {id, name, arguments} with id bounded by
    // MAX_TOOL_CALL_ID_LEN=128 (tool.rs:62) and correlated by
    // ToolResult.call_id (tool.rs:1055). The runtime correlates via the
    // bus-issued ExecutionId handle instead. No equivalence to assert, so
    // this fails loudly rather than passing silently; the runtime twin
    // gains no id field here (report-only).
    panic!(
        "DRIFT tool-call id: runtime tool.rs ToolCall has no caller id field vs bitty-agent@a8e4112 tool.rs ToolCall.id with ToolResult.call_id correlation"
    );
}

// DRIFT-LEDGER (ignored but live): no runtime ToolResult twin (split into
// ToolSuccess plus ToolExecution) vs bitty-agent@a8e4112 tool.rs ToolResult.
// ARCHIVED (owner ruling 2026-10-16): Split ToolSuccess/ToolExecution
// supersedes flat ToolResult by design.
#[test]
#[ignore = "drift ledger: no runtime ToolResult twin (ToolSuccess plus ToolExecution) vs agent ToolResult"]
fn toolresult_shape_drift() {
    // DRIFT tool-result shape: runtime has no ToolResult twin. The agent
    // ToolResult {call_id, content, is_error} (bitty-agent tool.rs:1053)
    // maps onto two runtime types: ToolSuccess {summary, data}
    // (tool.rs:810) plus ToolExecution {execution_id, tool, status,
    // result_disposition, summary, data, is_untrusted_surface} (tool.rs:789)
    // keyed by bus-issued ExecutionId, with error-ness split across
    // ToolStatus and ResultDisposition. Only the 16 KiB content cap matches
    // (see tool_result_bound_matches). Fails loudly; no shape change here
    // (report-only).
    panic!(
        "DRIFT tool-result shape: no runtime ToolResult twin vs bitty-agent@a8e4112 tool.rs ToolResult; runtime splits into ToolSuccess plus ToolExecution"
    );
}

// DRIFT-LEDGER (ignored but live): no runtime AgentObservation twin or 8KiB
// observation cap vs bitty-agent@a8e4112 observation.rs AgentObservation.
// ARCHIVED (owner ruling 2026-10-16): AgentObservation is terminal-layer
// vocabulary, untrusted-surface already parity-matched.
#[test]
#[ignore = "drift ledger: no runtime AgentObservation twin or 8KiB cap vs agent AgentObservation"]
fn observation_type_drift() {
    // DRIFT observation vocabulary: runtime has no AgentObservation twin.
    // The agent enum (bitty-agent observation.rs:33, ten variants with an
    // 8 KiB payload cap at observation.rs:17 and read-only snapshots plus
    // is_untrusted_surface labeling) has no counterpart enum or 8 KiB
    // observation cap in runtime; only the untrusted flag survives, on
    // ToolExecution/ContextRecord (see
    // executed_tool_outcome_carries_untrusted_flag). Fails loudly; no new
    // observation type is added here (report-only).
    panic!(
        "DRIFT observation vocabulary: no runtime AgentObservation twin or 8KiB observation cap vs bitty-agent@a8e4112 observation.rs AgentObservation"
    );
}

// DRIFT-LEDGER (ignored but live): runtime session.rs AgentSession has no
// history caps or side queue vs bitty-agent@a8e4112 session.rs plus message.rs caps.
// ARCHIVED (owner ruling 2026-10-16): History caps/queue ownership deliberately
// outside AgentSession.
#[test]
#[ignore = "drift ledger: runtime AgentSession has no history caps or side queue vs agent session plus message caps"]
fn session_bounds_drift() {
    // DRIFT session bounds: runtime AgentSession (session.rs:329) holds
    // only ids/level/generation/state with no message history and no side
    // queue, so the agent bounds (message.rs:16 MAX_MESSAGES_PER_SESSION=128,
    // message.rs:19 MAX_SESSION_BYTES=256KiB, session.rs:17
    // DEFAULT_SIDE_QUEUE_CAPACITY=64) have no twin, and the state machines
    // differ (runtime session.rs:210 Active/Completed/Failed/Canceled vs
    // agent session.rs:23 Created/Running/WaitingToolResult/Completed/Failed).
    // Fails loudly; the runtime session gains no history or queue here
    // (report-only).
    panic!(
        "DRIFT session bounds: runtime session.rs AgentSession has no history caps or side queue vs bitty-agent@a8e4112 session.rs AgentSession plus message.rs caps"
    );
}

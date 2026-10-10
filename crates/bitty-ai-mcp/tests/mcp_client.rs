//! Hermetic offline acceptance for `bitty-ai-mcp` (AI-0179).
//!
//! No network, no registry, no wall-clock policy time: process spawns use
//! `python3 -u` (fake MCP server) and `/bin/sleep` + `/bin/sh -c` one-shot
//! helper fixtures (spawn tests are Unix-gated),
//! policy decisions take caller `now_ms`, and wall clock appears only where
//! a deadline is itself under test. Covers: frame cap probe, malformed-line
//! drop, no-tools-capability refusal, sanitize/collision/64-byte cap,
//! allowlist-miss zero-contact denial, oversize argument/result typed
//! bounds, hanging-helper deadline to `Unknown` plus reap, secret redaction
//! over errors, and the fake stdio server end to end.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use bitty_ai_mcp::frame::{FramedLines, check_frame_len};
use bitty_ai_mcp::handshake::parse_initialize_result;
use bitty_ai_mcp::{
    CredentialRef, McpAdapterParams, McpFailure, McpServerConfig, McpToolAdapter, PROTOCOL_VERSION,
    call_tool, handshake, list_tools, sanitize_mcp_name,
};
use bitty_ai_mcp::{McpError, McpTransport};
use bitty_ai_runtime::tool::ToolExecutor;

// ── shared fakes ─────────────────────────────────────────────────────────────

type SharedSent = Rc<RefCell<Vec<String>>>;

struct FakeTransport {
    inbound: VecDeque<String>,
    sent: SharedSent,
}

impl FakeTransport {
    fn fresh(inbound: Vec<String>) -> (Self, SharedSent) {
        let sent: SharedSent = Rc::new(RefCell::new(Vec::new()));
        (
            Self {
                inbound: inbound.into_iter().collect(),
                sent: Rc::clone(&sent),
            },
            sent,
        )
    }
}

impl McpTransport for FakeTransport {
    fn send_line(&mut self, line: &str) -> Result<(), McpError> {
        self.sent.borrow_mut().push(line.to_owned());
        Ok(())
    }

    fn recv_line(&mut self, _timeout_ms: u64) -> Result<Option<String>, McpError> {
        Ok(self.inbound.pop_front())
    }
}

fn allow_authorizer() -> AllowHook {
    AllowHook
}

struct AllowHook;

impl bitty_ai_runtime::tool::ToolAuthorizer for AllowHook {
    fn authorize(
        &self,
        _ctx: &bitty_ai_runtime::tool::AuthContext,
    ) -> bitty_ai_runtime::tool::AuthDecision {
        bitty_ai_runtime::tool::AuthDecision::Allow
    }
}

fn consented_ledger(tools: &[(&str, &str)]) -> bitty_ai_runtime::bridge::FakeConsentLedger {
    let mut ledger = bitty_ai_runtime::bridge::FakeConsentLedger::new();
    for (tool, scope) in tools {
        ledger
            .grant("local.assistant", tool, scope, 9_000)
            .expect("grant fits");
    }
    ledger
}

fn workspace_base(
    level: bitty_ai_runtime::session::AgentLevel,
) -> bitty_ai_runtime::tool::AuthBase {
    let mut issuer = bitty_ai_runtime::session::IdIssuer::default();
    bitty_ai_runtime::tool::AuthBase {
        agent_instance_id: issuer.agent_instance(),
        session_id: issuer.session(),
        level,
    }
}

// ── frame bounds over the public API ─────────────────────────────────────────

#[test]
fn frame_cap_probe_is_cap_plus_one() {
    assert!(check_frame_len(bitty_ai_mcp::MAX_FRAME_BYTES).is_ok());
    let error = check_frame_len(bitty_ai_mcp::MAX_FRAME_BYTES + 1).expect_err("over cap");
    assert_eq!(
        error.failure,
        McpFailure::FrameTooLarge {
            limit: bitty_ai_mcp::MAX_FRAME_BYTES,
            actual: bitty_ai_mcp::MAX_FRAME_BYTES + 1,
        }
    );
}

#[test]
fn malformed_lines_drop_and_count_over_public_reader() {
    use std::io::BufReader;
    let input = "{\"ok\":1}\nnope\n{\"ok\":2}\n";
    let mut frames = FramedLines::new(BufReader::new(input.as_bytes()));
    assert_eq!(
        frames.next_line().expect("line"),
        Some("{\"ok\":1}".to_owned())
    );
    assert_eq!(
        frames.next_line().expect("line"),
        Some("{\"ok\":2}".to_owned())
    );
    assert_eq!(frames.stats().dropped_malformed, 1);
}

// ── handshake refusal without tool capability ────────────────────────────────

#[test]
fn no_tools_capability_refuses_fail_closed() {
    let line = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"roots\":{{}}}}}}}}"
    );
    let error = parse_initialize_result(&line).expect_err("no tools");
    assert_eq!(error.failure, McpFailure::NoToolCapability);
    assert!(!error.retryable);
}

#[test]
fn full_handshake_against_fake_transport() {
    let answer = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"fake\",\"version\":\"0.0.1\"}}}}}}"
    );
    let (mut transport, sent) = FakeTransport::fresh(vec![answer]);
    let negotiated =
        handshake(&mut transport, "/tmp/bitty", 1_000, &mut Vec::new()).expect("handshake");
    assert!(negotiated.tools_supported);
    let sent = sent.borrow();
    assert_eq!(sent.len(), 2);
    assert!(sent[1].contains("notifications/initialized"));
}

// ── AI-0205: never-sample / never-elicit refusals ────────────────────────────

#[test]
fn handshake_refuses_sampling_and_elicitation_inline() {
    use bitty_ai_mcp::handshake::{ELICITATION_REFUSED_MESSAGE, SAMPLING_REFUSED_MESSAGE};
    let answer = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"fake\",\"version\":\"0.0.1\"}}}}}}"
    );
    let (mut transport, sent) = FakeTransport::fresh(vec![
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"sampling/createMessage\",\"params\":{\"messages\":[{\"role\":\"user\",\"content\":{\"text\":\"canary-sampling-params\"}}]}}"
            .to_owned(),
        "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"elicitation/create\",\"params\":{\"message\":\"canary-elicitation-params\"}}"
            .to_owned(),
        answer,
    ]);
    let negotiated =
        handshake(&mut transport, "/tmp/bitty", 1_000, &mut Vec::new()).expect("handshake");
    assert!(negotiated.tools_supported);
    let sent = sent.borrow();
    assert_eq!(sent.len(), 4);
    // Sampling refusal: same -32601 code shape, distinct static message, no
    // request-param echo.
    assert_eq!(
        sent[1],
        format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":9,\"error\":{{\"code\":-32601,\"message\":\"{SAMPLING_REFUSED_MESSAGE}\"}}}}"
        )
    );
    assert!(!sent[1].contains("Method not found"));
    assert!(!sent[1].contains("canary-sampling-params"));
    // Elicitation refusal: same code shape, its own distinct message.
    assert_eq!(
        sent[2],
        format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":10,\"error\":{{\"code\":-32601,\"message\":\"{ELICITATION_REFUSED_MESSAGE}\"}}}}"
        )
    );
    assert!(!sent[2].contains("Method not found"));
    assert!(!sent[2].contains("canary-elicitation-params"));
    // The handshake still completes afterwards.
    assert!(sent[3].contains("notifications/initialized"));
}

#[test]
fn handshake_unknown_method_still_32601_ping_roots_unaffected() {
    use bitty_ai_mcp::handshake::UNKNOWN_METHOD_MESSAGE;
    let answer = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"fake\",\"version\":\"0.0.1\"}}}}}}"
    );
    let (mut transport, sent) = FakeTransport::fresh(vec![
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"roots/list\"}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/frobnicate\"}".to_owned(),
        answer,
    ]);
    handshake(&mut transport, "/tmp/bitty", 1_000, &mut Vec::new()).expect("handshake");
    let sent = sent.borrow();
    assert_eq!(sent.len(), 5);
    assert_eq!(sent[1], "{\"jsonrpc\":\"2.0\",\"id\":9,\"result\":{}}");
    assert!(sent[2].contains("file:///tmp/bitty"));
    assert_eq!(sent[2].matches("file://").count(), 1);
    assert_eq!(
        sent[3],
        format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":11,\"error\":{{\"code\":-32601,\"message\":\"{UNKNOWN_METHOD_MESSAGE}\"}}}}"
        )
    );
    assert!(sent[4].contains("notifications/initialized"));
}

// ── sanitize: shape, collision, 64-byte cap ──────────────────────────────────

#[test]
fn sanitize_collisions_and_cap_are_deterministic() {
    // Dots and dashes fold to the same name: a shadowing pair.
    assert_eq!(
        sanitize_mcp_name("demo", "a.b"),
        sanitize_mcp_name("demo", "a_b")
    );
    assert!(
        bitty_ai_runtime::tool::validate_tool_name(&sanitize_mcp_name("demo", "9lives!")).is_ok()
    );
    let capped = sanitize_mcp_name("demo", &"z".repeat(300));
    assert_eq!(capped.len(), 64);
    assert_eq!(capped, sanitize_mcp_name("demo", &"z".repeat(300)));
    assert!(bitty_ai_runtime::tool::validate_tool_name(&capped).is_ok());
}

// ── adapter policy: zero-contact denials and typed bounds ────────────────────

fn echo_import() -> bitty_ai_mcp::ImportedTool {
    let spec = bitty_ai_runtime::tool::ToolSpec::new(
        "mcp_demo_echo",
        "Echo",
        br#"{"type":"object"}"#.to_vec(),
        "mcp.demo",
        true,
    )
    .expect("valid spec");
    bitty_ai_mcp::ImportedTool {
        digest: spec.schema_digest(),
        server_id: "demo".to_owned(),
        raw_name: "echo".to_owned(),
        spec,
    }
}

fn test_adapter(
    transport: FakeTransport,
    consent: bitty_ai_runtime::bridge::FakeConsentLedger,
) -> McpToolAdapter {
    McpToolAdapter::new(
        "demo",
        vec![echo_import()],
        Box::new(transport),
        Box::new(allow_authorizer()),
        Box::new(consent),
        McpAdapterParams {
            protocol_id: "local.assistant".to_owned(),
            base: workspace_base(bitty_ai_runtime::session::AgentLevel::Workspace),
            cwd: "/tmp/bitty".to_owned(),
            timeout_ms: 1_000,
        },
    )
    .expect("adapter builds")
}

#[test]
fn allowlist_miss_denies_with_zero_transport_contact() {
    let (transport, sent) = FakeTransport::fresh(Vec::new());
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let error = adapter
        .execute("mcp_demo_nope", b"{}", 1_000)
        .expect_err("allowlist miss must deny");
    assert!(matches!(
        error,
        bitty_ai_runtime::tool::ToolError::Denied { .. }
    ));
    assert!(sent.borrow().is_empty());
}

#[test]
fn consent_deny_is_zero_contact_and_names_only() {
    let (transport, sent) = FakeTransport::fresh(Vec::new());
    let mut adapter = test_adapter(
        transport,
        bitty_ai_runtime::bridge::FakeConsentLedger::new(),
    );
    let error = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect_err("consent must deny");
    match error {
        bitty_ai_runtime::tool::ToolError::Denied { reason, .. } => {
            assert!(reason.contains("consent denied"));
        }
        other => panic!("expected Denied, got {other:?}"),
    }
    assert!(sent.borrow().is_empty());
}

#[test]
fn oversize_call_arguments_are_a_typed_bound() {
    let (transport, sent) = FakeTransport::fresh(Vec::new());
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let big = vec![b'x'; 16 * 1024 + 1];
    let error = adapter
        .execute("mcp_demo_echo", &big, 1_000)
        .expect_err("oversize args");
    assert_eq!(
        error,
        bitty_ai_runtime::tool::ToolError::ArgumentsTooLarge {
            limit: 16 * 1024,
            actual: 16 * 1024 + 1,
        }
    );
    assert!(sent.borrow().is_empty());
}

#[test]
fn oversize_call_result_is_rejected_never_truncated() {
    let big_text = "r".repeat(16 * 1024 + 4);
    let line = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{big_text}\"}}]}}}}"
    );
    let (transport, _sent) = FakeTransport::fresh(vec![line]);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let error = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect_err("oversize result");
    assert_eq!(
        error,
        bitty_ai_runtime::tool::ToolError::ResultTooLarge {
            limit: 16 * 1024,
            actual: 16 * 1024 + 4,
        }
    );
}

#[test]
fn pre_execution_protocol_errors_map_to_protocol_rejected() {
    // AI-0204: `-32600`/`-32601`/`-32602` never executed, so the adapter
    // reports the distinct protocol outcome: never `Denied`, never
    // `EffectUnknown`, never `Failed`.
    for code in [-32600, -32601, -32602] {
        let line = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{{\"code\":{code},\"message\":\"bad frame\"}}}}"
        );
        let (transport, _sent) = FakeTransport::fresh(vec![line]);
        let mut adapter = test_adapter(
            transport,
            consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
        );
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("protocol error must reject");
        match &error {
            bitty_ai_runtime::tool::ToolError::ProtocolRejected { reason, .. } => {
                assert!(reason.contains("protocol rejected before execution"));
                assert!(reason.contains(&code.to_string()));
            }
            other => panic!("code {code}: expected ProtocolRejected, got {other:?}"),
        }
        assert!(
            !matches!(
                error,
                bitty_ai_runtime::tool::ToolError::Denied { .. }
                    | bitty_ai_runtime::tool::ToolError::EffectUnknown { .. }
                    | bitty_ai_runtime::tool::ToolError::Failed { .. }
            ),
            "code {code}: wrong attribution, got {error:?}"
        );
    }
}

#[test]
fn non_protocol_rpc_errors_and_is_error_stay_failed() {
    // AI-0204: `-32603`, the server `-32000` range, and `isError: true`
    // answers keep the executed-failure mapping unchanged.
    let frames = vec![
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32603,\"message\":\"boom\"}}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32000,\"message\":\"busy\"}}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"kaput\"}],\"isError\":true}}"
            .to_owned(),
    ];
    for line in frames {
        let (transport, _sent) = FakeTransport::fresh(vec![line]);
        let mut adapter = test_adapter(
            transport,
            consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
        );
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("must report failure");
        match error {
            bitty_ai_runtime::tool::ToolError::Failed { reason, .. } => {
                assert!(reason.contains("tool reported failure"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}

#[test]
fn mcp_invoke_is_param_required_shaped() {
    // S6 contribution pattern: `mcp.invoke` requires a per-tool parameter.
    // The client satisfies it by addressing tools only through exact
    // (sanitized-name, raw-name) pairs — no prefix or family dispatch.
    let imported = echo_import();
    assert!(!imported.raw_name.is_empty());
    assert_eq!(imported.spec.name, "mcp_demo_echo");
    let (transport, _sent) = FakeTransport::fresh(Vec::new());
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    // A family-prefix call is not a tool: refused, never dispatched.
    assert!(adapter.execute("mcp_demo", b"{}", 1_000).is_err());
}

// ── AI-0211: versioned re-list + deny-stale over stdio/FakeTransport ─────────

#[test]
fn stale_deny_relist_allow_with_next_id_continuity() {
    use bitty_ai_mcp::is_tools_list_changed_notification;

    let list_frame = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}]}}";
    let call_frame = "{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}";
    let (transport, sent) =
        FakeTransport::fresh(vec![list_frame.to_owned(), call_frame.to_owned()]);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    assert_eq!(adapter.list_version(), 0);
    assert!(!adapter.is_stale());
    assert_eq!(adapter.next_id(), 2);

    let signal = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}";
    assert!(is_tools_list_changed_notification(signal));
    assert!(adapter.observe_notification(signal));
    assert!(adapter.is_stale());
    assert_eq!(adapter.list_version(), 1);

    let denied = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect_err("stale must deny");
    match &denied {
        bitty_ai_runtime::tool::ToolError::Denied { reason, .. } => {
            assert!(reason.contains("tool list stale; re-list required"));
        }
        other => panic!("expected Denied, got {other:?}"),
    }
    assert!(!matches!(
        denied,
        bitty_ai_runtime::tool::ToolError::EffectUnknown { .. }
            | bitty_ai_runtime::tool::ToolError::ProtocolRejected { .. }
    ));
    assert!(sent.borrow().is_empty());

    let diff = adapter.relist(&["echo".to_owned()]).expect("relist");
    assert!(diff.is_empty());
    assert!(!adapter.is_stale());
    assert_eq!(adapter.list_version(), 2);
    assert_eq!(adapter.next_id(), 3);
    assert_eq!(sent.borrow().len(), 1);
    assert!(sent.borrow()[0].contains("\"id\":2"));

    let success = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect("post-relist allows");
    assert_eq!(success.data, b"hi");
    assert_eq!(sent.borrow().len(), 2);
    assert!(sent.borrow()[1].contains("\"id\":3"));
}

#[test]
fn stale_negatives_never_mark_over_stdio() {
    let (transport, _sent) = FakeTransport::fresh(Vec::new());
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    for line in [
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}",
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\"}",
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\"}",
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"notifications/tools/list_changed\"}",
        "not json",
        "",
    ] {
        assert!(
            !adapter.observe_notification(line),
            "must not stale: {line:?}"
        );
    }
    assert!(!adapter.is_stale());
    assert_eq!(adapter.list_version(), 0);
}

#[test]
fn failed_relist_keeps_snapshot_and_stays_stale_over_stdio() {
    let error_frame =
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32603,\"message\":\"boom\"}}";
    let (transport, sent) = FakeTransport::fresh(vec![error_frame.to_owned()]);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let before = adapter.snapshot();
    adapter.mark_stale();
    let version = adapter.list_version();
    adapter.relist(&["echo".to_owned()]).expect_err("must fail");
    assert!(adapter.is_stale());
    assert_eq!(adapter.list_version(), version);
    assert_eq!(adapter.snapshot(), before);
    let denied = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect_err("still stale");
    assert!(matches!(
        denied,
        bitty_ai_runtime::tool::ToolError::Denied { .. }
    ));
    assert_eq!(sent.borrow().len(), 1);
}

// ── AI-0212: mid-round-trip notifications surface; stale preserved ──────────
// `wait_for_response` stashes skipped notification frames into the caller's
// buffer instead of dropping them; `relist`/`execute` route them through
// `observe_notification`. No auto-relist, no retry, no registry mutation.

#[test]
fn wait_surfaces_notifications_to_caller_over_stdio() {
    // AI-0212 (b): a notification skipped mid-wait reaches the caller; the
    // import itself is unchanged.
    let signal = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}".to_owned();
    let page = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}]}}"
        .to_owned();
    let (mut transport, _sent) = FakeTransport::fresh(vec![signal.clone(), page]);
    let mut next_id = 2;
    let mut surfaced = Vec::new();
    let imported = list_tools(
        &mut transport,
        "demo",
        "/tmp/bitty",
        &["echo".to_owned()],
        &mut next_id,
        1_000,
        &mut surfaced,
    )
    .expect("import");
    assert_eq!(imported.len(), 1);
    assert_eq!(imported[0].spec.name, "mcp_demo_echo");
    assert_eq!(next_id, 3);
    assert_eq!(surfaced, vec![signal]);

    // The call wait surfaces too, without changing the outcome.
    let ping = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}".to_owned();
    let answer = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}"
        .to_owned();
    let (mut transport, _sent) = FakeTransport::fresh(vec![ping.clone(), answer]);
    let mut surfaced = Vec::new();
    let success = call_tool(
        &mut transport,
        2,
        "echo",
        "mcp_demo_echo",
        b"{}",
        "/tmp/bitty",
        1_000,
        &mut surfaced,
    )
    .expect("call");
    assert_eq!(success.data, b"hi");
    assert_eq!(surfaced, vec![ping]);
}

#[test]
fn wait_without_notifications_surfaces_nothing() {
    // AI-0212 (d): the clean path leaves the buffer empty — non-notification
    // behavior is byte-identical to before.
    let page = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}]}}"
        .to_owned();
    let (mut transport, _sent) = FakeTransport::fresh(vec![page]);
    let mut next_id = 2;
    let mut surfaced = Vec::new();
    list_tools(
        &mut transport,
        "demo",
        "/tmp/bitty",
        &["echo".to_owned()],
        &mut next_id,
        1_000,
        &mut surfaced,
    )
    .expect("import");
    assert!(surfaced.is_empty());

    let answer = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}"
        .to_owned();
    let (mut transport, _sent) = FakeTransport::fresh(vec![answer]);
    let mut surfaced = Vec::new();
    call_tool(
        &mut transport,
        2,
        "echo",
        "mcp_demo_echo",
        b"{}",
        "/tmp/bitty",
        1_000,
        &mut surfaced,
    )
    .expect("call");
    assert!(surfaced.is_empty());
}

#[test]
fn relist_preserves_stale_on_mid_round_trip_list_changed() {
    // AI-0212 (a): a `list_changed` arriving during the relist round-trip
    // keeps the adapter stale. The completed refresh still bumps the version
    // and returns its diff normally.
    let signal = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}".to_owned();
    let list_frame = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}]}}"
        .to_owned();
    let (transport, sent) = FakeTransport::fresh(vec![signal, list_frame]);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    assert_eq!(adapter.list_version(), 0);
    assert!(!adapter.is_stale());

    let diff = adapter.relist(&["echo".to_owned()]).expect("relist");
    assert!(diff.is_empty());
    assert!(
        adapter.is_stale(),
        "mid-round-trip signal must preserve stale"
    );
    // Refresh bump (0 -> 1) plus the routed signal's mark bump (1 -> 2).
    assert_eq!(adapter.list_version(), 2);
    assert_eq!(adapter.next_id(), 3);
    assert_eq!(sent.borrow().len(), 1);

    // The preserved stale denies the next call with zero contact, as usual.
    let denied = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect_err("stale must deny");
    assert!(matches!(
        denied,
        bitty_ai_runtime::tool::ToolError::Denied { .. }
    ));
    assert_eq!(sent.borrow().len(), 1);
}

#[test]
fn call_wait_notification_marks_stale_without_changing_result() {
    // AI-0212 (c): a `list_changed` seen during a call wait marks the adapter
    // stale for the *next* call; this call's result is unchanged. No
    // auto-relist, no retry: exactly one call frame went out.
    let signal = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}".to_owned();
    let call_frame =
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}"
            .to_owned();
    let (transport, sent) = FakeTransport::fresh(vec![signal, call_frame]);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let success = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect("call succeeds");
    assert_eq!(success.data, b"hi");
    assert_eq!(sent.borrow().len(), 1);
    assert!(adapter.is_stale());
    assert_eq!(adapter.list_version(), 1);
}

#[test]
fn mid_wait_negatives_never_stale_over_stdio() {
    // AI-0212 (e): non-list notifications seen mid-wait never stale — on both
    // the call and the relist paths. The `id`-carrying echo is answered
    // inline like any other server request, never routed as a notification.
    let negatives = vec![
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\"}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\"}".to_owned(),
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"notifications/tools/list_changed\"}".to_owned(),
    ];
    let call_frame =
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}"
            .to_owned();
    let mut inbound = negatives.clone();
    inbound.push(call_frame);
    let (transport, _sent) = FakeTransport::fresh(inbound);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let success = adapter
        .execute("mcp_demo_echo", b"{}", 1_000)
        .expect("call succeeds");
    assert_eq!(success.data, b"hi");
    assert!(!adapter.is_stale());
    assert_eq!(adapter.list_version(), 0);

    let list_frame = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}]}}"
        .to_owned();
    let mut inbound = negatives.clone();
    inbound.push(list_frame);
    let (transport, _sent) = FakeTransport::fresh(inbound);
    let mut adapter = test_adapter(
        transport,
        consented_ledger(&[("mcp_demo_echo", "mcp.demo")]),
    );
    let diff = adapter.relist(&["echo".to_owned()]).expect("relist");
    assert!(diff.is_empty());
    assert!(!adapter.is_stale());
    assert_eq!(adapter.list_version(), 1);
}

// ── Unix live-spawn tests (bounded fixtures) ──────────────────────────────────

#[cfg(unix)]
mod unix {
    use super::*;
    use bitty_ai_mcp::supervise::{ServerCooldown, spawn_server};

    fn fixture(name: &str) -> String {
        let mut dir = std::env::current_dir().expect("cwd");
        dir.push("tests");
        dir.push("fixtures");
        dir.push(name);
        dir.to_string_lossy().into_owned()
    }

    fn fixture_config(allowlist: Vec<String>) -> McpServerConfig {
        McpServerConfig {
            id: "fixture".to_owned(),
            // python3 -u: unbuffered stdio, so fixture replies flush
            // immediately on every host (POSIX sh printf may block-buffer
            // pipes, e.g. dash on CI, stalling the handshake).
            command: "python3".to_owned(),
            args: vec!["-u".to_owned(), fixture("fake_mcp_server.py")],
            env_refs: Vec::new(),
            cwd: std::env::temp_dir().to_string_lossy().into_owned(),
            timeout_ms: 10_000,
            tool_allowlist: allowlist,
        }
    }

    fn no_env(_var: &str) -> Option<String> {
        None
    }

    #[test]
    fn fake_server_end_to_end_handshake_list_call() {
        let config = fixture_config(vec!["echo".to_owned(), "fail_now".to_owned()]);
        let mut server = spawn_server(&config, &no_env).expect("spawn fixture");
        assert!(server.is_alive());

        let negotiated =
            handshake(&mut server, &config.cwd, 5_000, &mut Vec::new()).expect("handshake");
        assert!(negotiated.tools_supported);

        let mut next_id = 2;
        let imported = list_tools(
            &mut server,
            "fixture",
            &config.cwd,
            &config.tool_allowlist,
            &mut next_id,
            5_000,
            &mut Vec::new(),
        )
        .expect("import");
        assert_eq!(imported.len(), 2);
        let names: Vec<&str> = imported
            .iter()
            .map(|entry| entry.spec.name.as_str())
            .collect();
        assert!(names.contains(&"mcp_fixture_echo"));
        assert!(names.contains(&"mcp_fixture_fail_now"));
        for entry in &imported {
            assert_eq!(entry.digest, entry.spec.schema_digest());
        }

        let ledger = {
            let mut ledger = bitty_ai_runtime::bridge::FakeConsentLedger::new();
            for entry in &imported {
                ledger
                    .grant(
                        "local.assistant",
                        &entry.spec.name,
                        &entry.spec.required_scope,
                        90_000,
                    )
                    .expect("grant");
            }
            ledger
        };
        // The live server moves into the adapter: real dispatch over the
        // supervised stdio transport (handshake and import already done).
        let mut adapter = McpToolAdapter::new(
            "fixture",
            imported,
            Box::new(server),
            Box::new(AllowHook),
            Box::new(ledger),
            McpAdapterParams {
                protocol_id: "local.assistant".to_owned(),
                base: workspace_base(bitty_ai_runtime::session::AgentLevel::Workspace),
                cwd: config.cwd.clone(),
                timeout_ms: 5_000,
            },
        )
        .expect("adapter");
        let success = adapter
            .execute("mcp_fixture_echo", b"{}", 1_000)
            .expect("live echo dispatches");
        assert_eq!(success.data, b"fake-echo-ok");
        let error = adapter
            .execute("mcp_fixture_fail_now", b"{}", 1_000)
            .expect_err("live failure reports");
        match error {
            bitty_ai_runtime::tool::ToolError::Failed { reason, .. } => {
                assert!(reason.contains("tool reported failure"));
                assert!(reason.contains("kaput"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn hanging_helper_deadline_is_unknown_and_reaped() {
        use std::time::Instant;
        let config = McpServerConfig {
            id: "sleeper".to_owned(),
            command: "/bin/sleep".to_owned(),
            args: vec!["30".to_owned()],
            env_refs: Vec::new(),
            cwd: std::env::temp_dir().to_string_lossy().into_owned(),
            timeout_ms: 10_000,
            tool_allowlist: vec!["echo".to_owned()],
        };
        let mut server = spawn_server(&config, &no_env).expect("spawn sleeper");
        assert!(server.is_alive());
        let started = Instant::now();
        // `/bin/sleep` never answers: the whole-call deadline must resolve
        // to `Unknown` promptly, never after the helper's own lifetime.
        let outcome = call_tool(
            &mut server,
            2,
            "echo",
            "mcp_sleeper_echo",
            b"{}",
            &config.cwd,
            300,
            &mut Vec::new(),
        );
        let elapsed = started.elapsed();
        match outcome {
            Err(bitty_ai_mcp::McpCallError::Unknown { .. }) => {}
            other => panic!("expected Unknown, got {other:?}"),
        }
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "deadline not honored: {elapsed:?}"
        );
        // Kill plus reap: shutdown synchronously reaps, so the child is
        // gone afterwards and further contact fails closed.
        server.shutdown();
        assert!(!server.is_alive());
        assert!(server.send_line("{\"jsonrpc\":\"2.0\"}").is_err());
    }

    #[test]
    fn credential_errors_name_references_never_values() {
        use bitty_ai_mcp::supervise::resolve_env_refs;
        // Missing variable: the name appears, nothing else can leak.
        let refs = vec![CredentialRef::Env {
            var: "BITTY_MCP_TEST_MISSING".to_owned(),
        }];
        let error = resolve_env_refs(&refs, &no_env).expect_err("missing must fail");
        let text = error.to_string();
        assert!(text.contains("env:BITTY_MCP_TEST_MISSING"));

        // Failing helper: oversize stdout full of a marker denies with the
        // program name only; the marker never enters the diagnostic.
        let marker = "MCP-REDACT-PROBE-7f3a9c-marker";
        let refs = vec![CredentialRef::Cmd {
            name: "PROBE_TOKEN".to_owned(),
            program: "/bin/sh".to_owned(),
            args: vec!["-c".to_owned(), format!("yes {marker} | head -c 8192")],
        }];
        let error = resolve_env_refs(&refs, &no_env).expect_err("oversize must fail");
        let text = error.to_string();
        assert!(text.contains("cmd:/bin/sh"), "{text}");
        assert!(!text.contains(marker), "secret marker leaked: {text}");
        assert!(!text.contains("PROBE_TOKEN ="), "{text}");
    }

    #[test]
    fn cooldown_is_geometric_and_capped() {
        let mut cooldown = ServerCooldown::default();
        assert!(cooldown.may_connect(0));
        cooldown.record_failure(0);
        assert_eq!(cooldown.wait_ms(0), 1_000);
        cooldown.record_failure(1_000);
        assert_eq!(cooldown.wait_ms(1_000), 2_000);
        for _ in 0..10 {
            cooldown.record_failure(5_000);
        }
        assert_eq!(cooldown.wait_ms(5_000), 60_000);
        cooldown.record_success();
        assert!(cooldown.may_connect(5_000));
    }
}

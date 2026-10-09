//! Hermetic acceptance for the remote (Streamable HTTP/SSE) transport.
//!
//! No external network: a recording fake [`NetworkService`] scripts
//! single-JSON and SSE bodies, session headers, and HTTP statuses
//! (401/404/202/405) plus oversize, unknown-id, and malformed shapes. One
//! loopback `TcpListener` fixture pins the wire shape (POST, `Accept`,
//! `Content-Type`, `Mcp-Session-Id`) through a minimal std-only HTTP client
//! used only in that test. Allowlist misses stay zero-contact; secrets never
//! enter errors or `Debug`.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use bitty_ai_mcp::McpTransport;
use bitty_ai_mcp::{
    HttpLineTransport, McpAdapterParams, McpFailure, McpToolAdapter, PROTOCOL_VERSION,
    RemoteServerConfig, call_tool, handshake, list_tools,
};
use bitty_ai_runtime::tool::ToolExecutor;
use bitty_network_api::{NetworkError, NetworkService, Request, Response};

// ── recording fake service ─────────────────────────────────────────────────

struct FakeService {
    requests: Mutex<Vec<Request>>,
    script: Mutex<VecDeque<Result<Response, NetworkError>>>,
}

impl FakeService {
    fn new() -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            script: Mutex::new(VecDeque::new()),
        }
    }

    fn queue_response(&self, response: Response) {
        if let Ok(mut script) = self.script.lock() {
            script.push_back(Ok(response));
        }
    }

    fn queue_error(&self, error: NetworkError) {
        if let Ok(mut script) = self.script.lock() {
            script.push_back(Err(error));
        }
    }

    fn recorded(&self) -> Vec<Request> {
        self.requests
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn recorded_count(&self) -> usize {
        self.requests.lock().map(|guard| guard.len()).unwrap_or(0)
    }
}

impl NetworkService for FakeService {
    type Socket = ();

    fn request(&self, request: &Request) -> Result<Response, NetworkError> {
        if let Ok(mut guard) = self.requests.lock() {
            guard.push(request.clone());
        }
        if let Ok(mut script) = self.script.lock() {
            if let Some(next) = script.pop_front() {
                return next;
            }
        }
        Err(NetworkError::Offline)
    }

    fn websocket(
        &self,
        _request: &bitty_network_api::WebSocketRequest,
    ) -> Result<Self::Socket, NetworkError> {
        Err(NetworkError::Offline)
    }
}

fn remote_config(url: &str) -> RemoteServerConfig {
    RemoteServerConfig {
        id: "demo".to_owned(),
        url: url.to_owned(),
        timeout_ms: 1_000,
        tool_allowlist: vec!["echo".to_owned()],
        headers: Vec::new(),
        capability: bitty_network_api::NetworkCapability::offline().with_domain("mcp.example.com"),
    }
}

fn loopback_config(url: &str, host: &str) -> RemoteServerConfig {
    RemoteServerConfig {
        id: "demo".to_owned(),
        url: url.to_owned(),
        timeout_ms: 2_000,
        tool_allowlist: vec!["echo".to_owned()],
        headers: Vec::new(),
        capability: bitty_network_api::NetworkCapability::offline().with_domain(host),
    }
}

fn json_ok(status: u16, body: &str) -> Response {
    Response {
        status,
        headers: vec![("content-type".to_owned(), "application/json".to_owned())],
        body: body.as_bytes().to_vec(),
    }
}

fn json_ok_session(status: u16, body: &str, session: &str) -> Response {
    Response {
        status,
        headers: vec![
            ("content-type".to_owned(), "application/json".to_owned()),
            ("mcp-session-id".to_owned(), session.to_owned()),
        ],
        body: body.as_bytes().to_vec(),
    }
}

fn initialize_answer(id: u64) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"fake\",\"version\":\"0.0.1\"}}}}}}"
    )
}

fn list_answer(id: u64) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"tools\":[{{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{{\"type\":\"object\"}}}}]}}}}"
    )
}

fn call_answer(id: u64, text: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{text}\"}}]}}}}"
    )
}

// ── config shapes ──────────────────────────────────────────────────────────

#[test]
fn remote_url_scheme_and_userinfo_are_fail_closed() {
    let mut config = remote_config("https://mcp.example.com/rpc");
    config.validate().expect("valid remote");
    for bad in [
        "http://user@mcp.example.com/rpc",
        "https://user:pass@mcp.example.com/rpc",
        "ws://mcp.example.com/rpc",
        "gopher://mcp.example.com/rpc",
        "mcp.example.com/rpc",
    ] {
        config.url = bad.to_owned();
        assert!(config.validate().is_err(), "escaped: {bad}");
    }
    config.url = "http://mcp.example.com/rpc".to_owned();
    assert!(config.validate().is_ok());
}

#[test]
fn remote_timeout_allowlist_headers_are_bounded() {
    let mut config = remote_config("https://mcp.example.com/rpc");
    config.timeout_ms = 0;
    assert!(config.validate().is_err());
    config.timeout_ms = 30_001;
    assert!(config.validate().is_err());
    config.timeout_ms = 1_000;
    config.tool_allowlist.clear();
    assert!(config.validate().is_err());
    config.tool_allowlist = vec!["bad name!".to_owned()];
    assert!(config.validate().is_err());
    config.tool_allowlist = vec!["echo".to_owned()];
    config.headers = vec![("accept".to_owned(), "x".to_owned())];
    assert!(config.validate().is_err());
    config.headers = Vec::new();
    assert!(config.validate().is_ok());
}

// ── capability deny: zero service calls ────────────────────────────────────

#[test]
fn capability_deny_is_zero_contact_transport_closed() {
    let service = FakeService::new();
    let mut config = remote_config("https://mcp.example.com/rpc");
    config.capability = bitty_network_api::NetworkCapability::offline();
    let mut transport = HttpLineTransport::new(config, service).expect("transport");
    let error = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
        .expect_err("offline denies");
    assert_eq!(error.failure, McpFailure::TransportClosed);
    assert_eq!(transport.service().recorded_count(), 0);
}

// ── single-JSON session flow reuse ─────────────────────────────────────────

#[test]
fn single_json_handshake_list_call_reuses_transport() {
    let service = FakeService::new();
    service.queue_response(json_ok_session(200, &initialize_answer(1), "sess-abc"));
    service.queue_response(Response {
        status: 202,
        headers: Vec::new(),
        body: Vec::new(),
    });
    service.queue_response(json_ok(200, &list_answer(2)));
    service.queue_response(json_ok(200, &call_answer(3, "hi")));
    let config = remote_config("https://mcp.example.com/rpc");
    let mut transport = HttpLineTransport::new(config, service).expect("transport");

    let negotiated = handshake(&mut transport, "/tmp/bitty", 1_000).expect("handshake");
    assert!(negotiated.tools_supported);
    assert_eq!(transport.session_id(), Some("sess-abc"));

    let mut next_id = 2;
    let imported = list_tools(
        &mut transport,
        "demo",
        "/tmp/bitty",
        &["echo".to_owned()],
        &mut next_id,
        1_000,
    )
    .expect("import");
    assert_eq!(imported.len(), 1);
    assert_eq!(imported[0].spec.name, "mcp_demo_echo");

    let success = call_tool(
        &mut transport,
        next_id,
        "echo",
        "mcp_demo_echo",
        b"{}",
        "/tmp/bitty",
        1_000,
    )
    .expect("call");
    assert_eq!(success.data, b"hi");

    // Session header flows on requests after capture; whole-op deadline and
    // frame cap map onto the outgoing request.
    let recorded = transport.service().recorded();
    assert_eq!(recorded.len(), 4);
    assert!(recorded[1..].iter().all(|request| {
        request
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("mcp-session-id") && value == "sess-abc")
    }));
    for request in &recorded {
        assert_eq!(request.timeout, Some(Duration::from_millis(1_000)));
        assert_eq!(
            request.max_body_bytes,
            Some(u64::try_from(bitty_ai_mcp::MAX_FRAME_BYTES).unwrap_or(u64::MAX))
        );
        assert!(
            request
                .headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("accept")
                    && value.contains("text/event-stream"))
        );
    }
}

// ── SSE framing ────────────────────────────────────────────────────────────

#[test]
fn sse_data_frames_queue_as_lines() {
    let service = FakeService::new();
    let body = format!(
        "event: message\ndata: {}\n\ndata: {}\n\n",
        initialize_answer(1),
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}"
    );
    service.queue_response(Response {
        status: 200,
        headers: vec![(
            "content-type".to_owned(),
            "text/event-stream; charset=utf-8".to_owned(),
        )],
        body: body.into_bytes(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}")
        .expect("sse posts");
    assert_eq!(transport.queued_len(), 2);
    let first = transport.recv_line(10).expect("recv").expect("line");
    assert!(first.contains("\"id\":1"));
    assert!(transport.recv_line(10).expect("recv").is_some());
    assert_eq!(transport.recv_line(10).expect("drained"), None);
}

#[test]
fn sse_cut_before_match_resolves_to_unknown_on_call() {
    let service = FakeService::new();
    // SSE carries only an unrelated id: the waiter finds no match.
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: b"data: {\"jsonrpc\":\"2.0\",\"id\":999,\"result\":{}}\n\n".to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let outcome = call_tool(
        &mut transport,
        2,
        "echo",
        "mcp_demo_echo",
        b"{}",
        "/tmp/bitty",
        200,
    );
    match outcome {
        Err(bitty_ai_mcp::McpCallError::Unknown { .. }) => {}
        other => panic!("expected Unknown, got {other:?}"),
    }
}

// ── AI-0195: SSE multi-data accumulation ───────────────────────────────────

#[test]
fn sse_multi_data_fields_join_into_one_message() {
    let service = FakeService::new();
    // One SSE event splits a single JSON-RPC object across three data:
    // fields; comment and event fields are ignored while the payload
    // accumulates, and the newline-joined payload queues as one message.
    let body = ": comment ignored\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\ndata: \"id\":1,\ndata: \"result\":{}}\n\n";
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: body.as_bytes().to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}")
        .expect("sse posts");
    assert_eq!(transport.queued_len(), 1);
    let line = transport.recv_line(10).expect("recv").expect("joined");
    assert!(line.contains("\"id\":1"));
    // The joined payload carries the SSE newline separator: three data:
    // lines became one queued message, not three.
    assert!(line.contains('\n'));
    assert_eq!(transport.recv_line(10).expect("drained"), None);
}

#[test]
fn sse_trailing_event_without_blank_separator_still_flushes() {
    let service = FakeService::new();
    // No trailing blank line: the pending event still flushes at end of body,
    // preserving the previous per-line behavior for single-data replies.
    let body = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}";
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: body.as_bytes().to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}")
        .expect("sse posts");
    assert_eq!(transport.queued_len(), 1);
    let line = transport.recv_line(10).expect("recv").expect("flushed");
    assert!(line.contains("\"id\":1"));
}

#[test]
fn sse_fan_out_counts_events_not_lines() {
    let service = FakeService::new();
    // Seventeen data: lines in ONE event (single blank dispatch) join into
    // one payload: fan-out counts the event, so exactly one line queues.
    let mut single_event = String::new();
    for _ in 0..17 {
        single_event.push_str("data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n");
    }
    single_event.push('\n');
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: single_event.into_bytes(),
    });
    // Seventeen blank-dispatched events exceed the 16-frame fan-out bound.
    let mut many_events = String::new();
    for _ in 0..17 {
        many_events.push_str("data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n");
    }
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: many_events.into_bytes(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
        .expect("single event joins to one");
    assert_eq!(transport.queued_len(), 1);
    assert!(transport.recv_line(10).expect("recv").is_some());
    assert_eq!(transport.recv_line(10).expect("drained"), None);
    let error = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}")
        .expect_err("17 events exceed fan-out");
    assert!(matches!(error.failure, McpFailure::FrameTooLarge { .. }));
}

#[test]
fn sse_oversize_joined_payload_is_frame_reject() {
    let service = FakeService::new();
    // Two fragments each under the frame cap join past it in one event.
    let half = "x".repeat(bitty_ai_mcp::MAX_FRAME_BYTES / 2 + 256);
    let body = format!(
        "data: {{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"pad\":\"{half}\ndata: {half}\"}}}}\n\n"
    );
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: body.into_bytes(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let error = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
        .expect_err("joined oversize");
    assert!(matches!(error.failure, McpFailure::FrameTooLarge { .. }));
}

// ── AI-0196: bounded wait in recv_line ─────────────────────────────────────
//
// `HttpLineTransport` is synchronous (`&mut self`, no background thread), so
// no line can arrive concurrently mid-`recv`: the queue grows only via
// `send_line` before `recv` runs. These tests prove the two observable
// halves: a pre-queued line returns promptly without waiting the full caller
// budget, and an empty queue waits a bounded slice (not instant, never past
// the caller deadline) instead of busy-spinning.

#[test]
fn recv_returns_queued_line_without_waiting_full_timeout() {
    let service = FakeService::new();
    service.queue_response(json_ok(200, &call_answer(1, "hi")));
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
        .expect("post");
    let start = Instant::now();
    let line = transport.recv_line(1_000).expect("recv").expect("queued");
    let elapsed = start.elapsed();
    assert!(line.contains("\"id\":1"));
    // Far below the 1s caller budget: the fast path never sleeps.
    assert!(
        elapsed < Duration::from_millis(500),
        "waited full budget: {elapsed:?}"
    );
}

#[test]
fn recv_empty_queue_waits_bounded_then_reports_none() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 202,
        headers: Vec::new(),
        body: Vec::new(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")
        .expect("202 accepts");
    let start = Instant::now();
    let outcome = transport.recv_line(50).expect("recv");
    let elapsed = start.elapsed();
    assert_eq!(outcome, None);
    // Bounded wait: not an instant return, never past the caller budget plus
    // generous scheduling slack (small bounds, no wall-clock flakiness).
    assert!(
        elapsed >= Duration::from_millis(30),
        "returned instantly: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(1_000),
        "extended past budget: {elapsed:?}"
    );
}

#[test]
fn recv_closed_returns_err_immediately() {
    let service = FakeService::new();
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport.shutdown();
    assert!(transport.is_closed());
    let start = Instant::now();
    let error = transport.recv_line(200).expect_err("closed");
    let elapsed = start.elapsed();
    assert_eq!(error.failure, McpFailure::TransportClosed);
    assert!(
        elapsed < Duration::from_millis(500),
        "closed waited: {elapsed:?}"
    );
}

#[test]
fn unrelated_sse_id_still_resolves_to_unknown_within_deadline() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: b"data: {\"jsonrpc\":\"2.0\",\"id\":999,\"result\":{}}\n\n".to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let start = Instant::now();
    let outcome = call_tool(
        &mut transport,
        2,
        "echo",
        "mcp_demo_echo",
        b"{}",
        "/tmp/bitty",
        200,
    );
    let elapsed = start.elapsed();
    match outcome {
        Err(bitty_ai_mcp::McpCallError::Unknown { .. }) => {}
        other => panic!("expected Unknown, got {other:?}"),
    }
    // The bounded wait composes with the whole-op deadline: the 200ms budget
    // resolves near the deadline, never far past it.
    assert!(
        elapsed < Duration::from_millis(2_000),
        "extended past deadline: {elapsed:?}"
    );
}

// ── statuses: 401/404 reject, 202 accepts, 405 shutdown ignored ─────────────

#[test]
fn non_2xx_rejects_handshake_and_fails_call() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 401,
        headers: Vec::new(),
        body: b"unauthorized".to_vec(),
    });
    service.queue_response(Response {
        status: 404,
        headers: Vec::new(),
        body: b"not here".to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let handshake_error = handshake(&mut transport, "/tmp/bitty", 1_000).expect_err("401 rejects");
    assert!(matches!(
        handshake_error.failure,
        McpFailure::HandshakeRejected { .. }
    ));
    assert!(!handshake_error.to_string().contains("unauthorized"));

    let call_error = call_tool(
        &mut transport,
        2,
        "echo",
        "mcp_demo_echo",
        b"{}",
        "/tmp/bitty",
        1_000,
    )
    .expect_err("404 fails call");
    assert!(matches!(
        call_error,
        bitty_ai_mcp::McpCallError::Failed { .. }
    ));
}

#[test]
fn accepted_202_with_empty_body_queues_nothing() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 202,
        headers: Vec::new(),
        body: Vec::new(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")
        .expect("202 accepts");
    assert_eq!(transport.recv_line(10).expect("recv"), None);
}

#[test]
fn shutdown_deletes_session_and_ignores_405() {
    let service = FakeService::new();
    service.queue_response(json_ok_session(200, &initialize_answer(1), "sess-bye"));
    service.queue_response(Response {
        status: 405,
        headers: Vec::new(),
        body: Vec::new(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
        .expect("post");
    assert_eq!(transport.session_id(), Some("sess-bye"));
    transport.shutdown();
    assert!(transport.is_closed());
    let recorded = transport.service().recorded();
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[1].method, bitty_network_api::HttpMethod::Delete);
    assert!(
        recorded[1]
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("mcp-session-id")
                && value == "sess-bye")
    );
    let closed = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}")
        .expect_err("closed");
    assert_eq!(closed.failure, McpFailure::TransportClosed);
}

// ── bounds: oversize, bad content type, malformed ──────────────────────────

#[test]
fn oversize_bodies_are_frame_rejects() {
    let service = FakeService::new();
    let big = "x".repeat(bitty_ai_mcp::MAX_FRAME_BYTES + 16);
    let body = format!("{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"pad\":\"{big}\"}}}}");
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "application/json".to_owned())],
        body: body.into_bytes(),
    });
    let oversize_sse =
        format!("data: {{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{{\"pad\":\"{big}\"}}}}\n\n");
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: oversize_sse.into_bytes(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let first = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
        .expect_err("oversize json");
    assert!(matches!(first.failure, McpFailure::FrameTooLarge { .. }));
    let second = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}")
        .expect_err("oversize sse");
    assert!(matches!(second.failure, McpFailure::FrameTooLarge { .. }));
}

#[test]
fn bad_content_type_and_malformed_reject_without_payload() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/plain".to_owned())],
        body: b"plain, not json".to_vec(),
    });
    service.queue_response(json_ok(200, "not a json object"));
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}")
        .expect("bad ct queues synthetic");
    let line = transport.recv_line(10).expect("recv").expect("queued");
    assert!(line.contains("\"id\":1"));
    assert!(!line.contains("plain"));
    transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\"}")
        .expect("malformed queues synthetic");
    let line = transport.recv_line(10).expect("recv").expect("queued");
    assert!(line.contains("\"id\":2"));
}

#[test]
fn network_timeout_and_tls_map_without_details() {
    let service = FakeService::new();
    service.queue_error(NetworkError::Timeout {
        after: Duration::from_millis(7),
    });
    service.queue_error(NetworkError::Tls {
        reason: bitty_network_api::TlsFailure::CaSourceInvalid,
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let timeout = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
        .expect_err("timeout");
    assert!(matches!(timeout.failure, McpFailure::Timeout { .. }));
    let tls = transport
        .send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}")
        .expect_err("tls");
    assert!(matches!(tls.failure, McpFailure::Io { .. }));
    assert!(!tls.to_string().contains("CaSource"));
}

// ── loopback wire shape ────────────────────────────────────────────────────

/// Minimal std-only HTTP client for the loopback fixture (test-only,
/// `http://127.0.0.1:<port>` only; parses just enough to prove the wire
/// shape without adding dependencies).
struct LoopbackService;

impl LoopbackService {
    fn round_trip(request: &Request) -> Result<Response, NetworkError> {
        let url = request.url.clone();
        let rest = url.strip_prefix("http://").ok_or(NetworkError::Offline)?;
        let (authority, path) = match rest.find('/') {
            Some(index) => (&rest[..index], &rest[index..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) => {
                let port = port.parse::<u16>().map_err(|_| NetworkError::Offline)?;
                (host, port)
            }
            None => (authority, 80),
        };
        if host != "127.0.0.1" && host != "localhost" {
            return Err(NetworkError::Offline);
        }
        let mut stream =
            std::net::TcpStream::connect((host, port)).map_err(|_| NetworkError::Offline)?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|_| NetworkError::Offline)?;
        let method = match request.method {
            bitty_network_api::HttpMethod::Post => "POST",
            bitty_network_api::HttpMethod::Delete => "DELETE",
            _ => "GET",
        };
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nhost: {authority}\r\ncontent-length: {}\r\nconnection: close\r\n",
            request.body.len()
        );
        for (name, value) in &request.headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        {
            use std::io::Write as _;
            stream
                .write_all(head.as_bytes())
                .map_err(|_| NetworkError::Offline)?;
            stream
                .write_all(&request.body)
                .map_err(|_| NetworkError::Offline)?;
            stream.flush().map_err(|_| NetworkError::Offline)?;
        }
        let mut raw = Vec::new();
        {
            use std::io::Read as _;
            stream
                .read_to_end(&mut raw)
                .map_err(|_| NetworkError::Offline)?;
        }
        let text = String::from_utf8_lossy(&raw).into_owned();
        let status = text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .ok_or(NetworkError::Offline)?;
        let mut headers = Vec::new();
        let mut body_start = 0;
        if let Some(index) = text.find("\r\n\r\n") {
            body_start = index + 4;
            for line in text[..index].lines().skip(1) {
                if let Some(colon) = line.find(':') {
                    headers.push((
                        line[..colon].trim().to_owned(),
                        line[colon + 1..].trim().to_owned(),
                    ));
                }
            }
        }
        Ok(Response {
            status,
            headers,
            body: raw[body_start.min(raw.len())..].to_vec(),
        })
    }
}

impl NetworkService for LoopbackService {
    type Socket = ();

    fn request(&self, request: &Request) -> Result<Response, NetworkError> {
        Self::round_trip(request)
    }

    fn websocket(
        &self,
        _request: &bitty_network_api::WebSocketRequest,
    ) -> Result<Self::Socket, NetworkError> {
        Err(NetworkError::Offline)
    }
}

#[test]
fn loopback_wire_shape_posts_accept_session_and_json() {
    use std::io::{Read as _, Write as _};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let addr = listener.local_addr().expect("addr");
    let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
    let seen_server = std::sync::Arc::clone(&seen);
    let server = std::thread::spawn(move || {
        // The handshake POSTs twice: `initialize` (id 1, expects a JSON
        // answer with a session) then `notifications/initialized` (no id,
        // `202` with no body). Serve both connections.
        for which in 0..2 {
            let (mut stream, _) = listener.accept().expect("accept");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            let mut buf = vec![0u8; 16 * 1024];
            let mut filled = 0;
            // Read until the framed JSON body arrives (short fixture read).
            for _ in 0..50 {
                match stream.read(&mut buf[filled..]) {
                    Ok(0) => break,
                    Ok(count) => {
                        filled += count;
                        let text = String::from_utf8_lossy(&buf[..filled]).into_owned();
                        if text.contains("\"method\":\"initialize\"")
                            || text.contains("notifications/initialized")
                        {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            let text = String::from_utf8_lossy(&buf[..filled]).into_owned();
            if let Ok(mut guard) = seen_server.lock() {
                guard.push(text.clone());
            }
            if which == 0 {
                let body = initialize_answer(1);
                let reply = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nmcp-session-id: loop-1\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            } else {
                let reply =
                    "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
                let _ = stream.write_all(reply.as_bytes());
            }
            let _ = stream.flush();
        }
    });

    let url = format!("http://127.0.0.1:{}/rpc", addr.port());
    let host = "127.0.0.1";
    let config = loopback_config(&url, host);
    let mut transport = HttpLineTransport::new(config, LoopbackService).expect("transport");
    let negotiated = handshake(&mut transport, "/tmp/bitty", 5_000).expect("handshake");
    assert!(negotiated.tools_supported);
    assert_eq!(transport.session_id(), Some("loop-1"));
    server.join().expect("server thread");

    let guard = seen.lock().expect("seen");
    let wire = guard.first().expect("wired request").clone();
    assert!(wire.starts_with("POST /rpc HTTP/1.1"));
    let lower = wire.to_lowercase();
    assert!(lower.contains("accept: application/json, text/event-stream"));
    assert!(lower.contains("content-type: application/json"));
    assert!(wire.contains("\"method\":\"initialize\""));
}

// ── bridge allowlist miss stays zero-contact ───────────────────────────────

#[test]
fn bridge_allowlist_miss_is_zero_contact_over_http() {
    use bitty_ai_runtime::tool::{ToolAuthorizer, ToolSpec};

    struct Allow;
    impl ToolAuthorizer for Allow {
        fn authorize(
            &self,
            _ctx: &bitty_ai_runtime::tool::AuthContext,
        ) -> bitty_ai_runtime::tool::AuthDecision {
            bitty_ai_runtime::tool::AuthDecision::Allow
        }
    }

    let spec = ToolSpec::new(
        "mcp_demo_echo",
        "Echo",
        br#"{"type":"object"}"#.to_vec(),
        "mcp.demo",
        true,
    )
    .expect("spec");
    let imported = bitty_ai_mcp::ImportedTool {
        digest: spec.schema_digest(),
        server_id: "demo".to_owned(),
        raw_name: "echo".to_owned(),
        spec,
    };
    let service = FakeService::new();
    let transport = HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
        .expect("transport");
    let mut ledger = bitty_ai_runtime::bridge::FakeConsentLedger::new();
    ledger
        .grant("local.assistant", "mcp_demo_echo", "mcp.demo", 9_000)
        .expect("grant");
    let mut issuer = bitty_ai_runtime::session::IdIssuer::default();
    let mut adapter = McpToolAdapter::new(
        "demo",
        vec![imported],
        Box::new(transport),
        Box::new(Allow),
        Box::new(ledger),
        McpAdapterParams {
            protocol_id: "local.assistant".to_owned(),
            base: bitty_ai_runtime::tool::AuthBase {
                agent_instance_id: issuer.agent_instance(),
                session_id: issuer.session(),
                level: bitty_ai_runtime::session::AgentLevel::Workspace,
            },
            cwd: "/tmp/bitty".to_owned(),
            timeout_ms: 1_000,
        },
    )
    .expect("adapter");
    let error = adapter
        .execute("mcp_demo_ghost", b"{}", 1_000)
        .expect_err("allowlist miss denies");
    assert!(matches!(
        error,
        bitty_ai_runtime::tool::ToolError::Denied { .. }
    ));
}

// ── AI-0206: host-driven poll_server_messages ───────────────────────────────
//
// `HttpLineTransport::poll_server_messages` GETs the same URL with
// `Accept: text/event-stream` plus the session header, parses SSE `data:`
// frames into the line queue (POST-path bounds), answers server requests
// inline via the handshake helper, and queues notifications for the host to
// route. `405`/empty-body/timeout stickily disables (later polls are
// `Ok(0)` with zero I/O; re-enable needs a new transport).

#[test]
fn poll_queues_sse_notifications_and_counts() {
    let service = FakeService::new();
    let body = "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}\n\n";
    service.queue_response(Response {
        status: 200,
        headers: vec![
            ("content-type".to_owned(), "text/event-stream".to_owned()),
            ("mcp-session-id".to_owned(), "sess-poll-1".to_owned()),
        ],
        body: body.as_bytes().to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let queued = transport.poll_server_messages("/tmp/bitty").expect("poll");
    assert_eq!(queued, 2);
    assert_eq!(transport.queued_len(), 2);
    assert_eq!(transport.session_id(), Some("sess-poll-1"));
    assert!(!transport.is_poll_disabled());
    // Queued lines are retrievable via `recv_line` in order.
    let first = transport.recv_line(10).expect("recv").expect("line");
    assert!(first.contains("notifications/tools/list_changed"));
    let second = transport.recv_line(10).expect("recv").expect("line");
    assert!(second.contains("notifications/ping"));
    assert_eq!(transport.recv_line(10).expect("drained"), None);
    // Wire shape: single GET, SSE-only Accept, config timeout + frame cap.
    let recorded = transport.service().recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].method, bitty_network_api::HttpMethod::Get);
    assert!(
        recorded[0].headers.iter().any(
            |(name, value)| name.eq_ignore_ascii_case("accept") && value == "text/event-stream"
        )
    );
    assert_eq!(recorded[0].timeout, Some(Duration::from_millis(1_000)));
    assert_eq!(
        recorded[0].max_body_bytes,
        Some(u64::try_from(bitty_ai_mcp::MAX_FRAME_BYTES).unwrap_or(u64::MAX))
    );
}

#[test]
fn poll_405_disables_and_second_poll_is_zero_io() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 405,
        headers: Vec::new(),
        body: b"nope".to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let first = transport
        .poll_server_messages("/tmp/bitty")
        .expect("405 disables");
    assert_eq!(first, 0);
    assert!(transport.is_poll_disabled());
    assert_eq!(transport.service().recorded_count(), 1);
    let second = transport
        .poll_server_messages("/tmp/bitty")
        .expect("disabled no-op");
    assert_eq!(second, 0);
    assert_eq!(
        transport.service().recorded_count(),
        1,
        "second poll must be zero I/O"
    );
}

#[test]
fn poll_empty_body_disables() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: Vec::new(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let first = transport
        .poll_server_messages("/tmp/bitty")
        .expect("empty disables");
    assert_eq!(first, 0);
    assert!(transport.is_poll_disabled());
    assert_eq!(transport.service().recorded_count(), 1);
    let second = transport
        .poll_server_messages("/tmp/bitty")
        .expect("disabled no-op");
    assert_eq!(second, 0);
    assert_eq!(transport.service().recorded_count(), 1);
}

#[test]
fn poll_non2xx_empty_is_error_without_disable() {
    // Regression for review thread 4231298715: a failed poll with an empty
    // body (for example a transient `503`) must report `HandshakeRejected`
    // without stickily disabling polling; the next poll still performs I/O.
    let service = FakeService::new();
    service.queue_response(Response {
        status: 503,
        headers: Vec::new(),
        body: Vec::new(),
    });
    let retry_body =
        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n";
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: retry_body.as_bytes().to_vec(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let error = transport
        .poll_server_messages("/tmp/bitty")
        .expect_err("503 empty must error");
    assert!(
        matches!(error.failure, McpFailure::HandshakeRejected { .. }),
        "unexpected failure: {:?}",
        error.failure
    );
    assert!(!transport.is_poll_disabled());
    assert_eq!(transport.service().recorded_count(), 1);
    let queued = transport
        .poll_server_messages("/tmp/bitty")
        .expect("retry still performs I/O");
    assert_eq!(queued, 1);
    assert!(!transport.is_poll_disabled());
    assert_eq!(transport.service().recorded_count(), 2);
}

#[test]
fn poll_timeout_disables() {
    let service = FakeService::new();
    service.queue_error(NetworkError::Timeout {
        after: Duration::from_millis(7),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let error = transport
        .poll_server_messages("/tmp/bitty")
        .expect_err("timeout");
    assert!(matches!(error.failure, McpFailure::Timeout { .. }));
    assert!(transport.is_poll_disabled());
    assert_eq!(transport.service().recorded_count(), 1);
    let second = transport
        .poll_server_messages("/tmp/bitty")
        .expect("disabled no-op");
    assert_eq!(second, 0);
    assert_eq!(transport.service().recorded_count(), 1);
}

#[test]
fn poll_answers_server_ping_inline() {
    let service = FakeService::new();
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: b"data: {\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}\n\n".to_vec(),
    });
    service.queue_response(Response {
        status: 202,
        headers: Vec::new(),
        body: Vec::new(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let queued = transport.poll_server_messages("/tmp/bitty").expect("poll");
    // Ping is answered, not queued; the `202` answer POST queues nothing.
    assert_eq!(queued, 0);
    assert_eq!(transport.queued_len(), 0);
    assert!(!transport.is_poll_disabled());
    let recorded = transport.service().recorded();
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[0].method, bitty_network_api::HttpMethod::Get);
    assert_eq!(recorded[1].method, bitty_network_api::HttpMethod::Post);
    let pong = String::from_utf8_lossy(&recorded[1].body).into_owned();
    assert!(pong.contains("\"id\":9"));
    assert!(pong.contains("\"result\":{}"));
    assert!(!pong.contains("error"));
    assert_eq!(transport.recv_line(10).expect("recv"), None);
}

#[test]
fn poll_refuses_sampling_with_distinct_message() {
    let canary = "canary-sampling-params-9f3a";
    let service = FakeService::new();
    let body = format!(
        "data: {{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"sampling/createMessage\",\"params\":{{\"messages\":[{{\"role\":\"user\",\"content\":{{\"text\":\"{canary}\"}}}}]}}}}\n\n"
    );
    service.queue_response(Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: body.into_bytes(),
    });
    service.queue_response(Response {
        status: 202,
        headers: Vec::new(),
        body: Vec::new(),
    });
    let mut transport =
        HttpLineTransport::new(remote_config("https://mcp.example.com/rpc"), service)
            .expect("transport");
    let queued = transport.poll_server_messages("/tmp/bitty").expect("poll");
    assert_eq!(queued, 0);
    assert!(!transport.is_poll_disabled());
    let recorded = transport.service().recorded();
    assert_eq!(recorded.len(), 2);
    let reply = String::from_utf8_lossy(&recorded[1].body).into_owned();
    assert!(reply.contains("\"id\":11"));
    assert!(reply.contains("-32601"));
    assert!(reply.contains(bitty_ai_mcp::handshake::SAMPLING_REFUSED_MESSAGE));
    assert!(!reply.contains("Method not found"));
    assert!(
        !reply.contains(canary),
        "refusal must not echo params: {reply}"
    );
    assert_eq!(transport.recv_line(10).expect("recv"), None);
}

// ── secrets never in errors ────────────────────────────────────────────────

#[test]
fn secrets_never_enter_errors_or_debug() {
    let canary = "sk-live-canary-7f3a9c-secret";
    let service = FakeService::new();
    service.queue_response(Response {
        status: 401,
        headers: Vec::new(),
        body: b"nope".to_vec(),
    });
    let mut config = remote_config("https://mcp.example.com/rpc");
    config.headers = vec![("Authorization".to_owned(), format!("Bearer {canary}"))];
    let rendered = format!("{config:?}");
    assert!(
        !rendered.contains(canary),
        "config debug leaked: {rendered}"
    );
    let mut transport = HttpLineTransport::new(config, service).expect("transport");
    let error = handshake(&mut transport, "/tmp/bitty", 1_000).expect_err("401");
    let text = error.to_string();
    let debug = format!("{error:?}");
    assert!(!text.contains(canary), "error leaked: {text}");
    assert!(!debug.contains(canary), "debug leaked: {debug}");
    for request in transport.service().recorded() {
        let request_debug = format!("{request:?}");
        assert!(
            !request_debug.contains(canary),
            "request debug leaked: {request_debug}"
        );
    }
}

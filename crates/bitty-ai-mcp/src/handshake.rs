//! MCP `initialize` handshake over a line transport.
//!
//! The client sends `initialize` with the pinned [`crate::PROTOCOL_VERSION`],
//! a `roots` capability, and bounded client info, then requires a result that
//! carries the same protocol version and a `tools` capability — anything
//! else fails closed ([`McpFailure::VersionMismatch`],
//! [`McpFailure::NoToolCapability`]). The client then sends
//! `notifications/initialized` (no reply expected).
//!
//! While waiting for the answer, server requests are answered inline:
//! `ping` gets an empty result, `roots/list` gets exactly the pinned `cwd`
//! as a `file://` URI (never the wider filesystem), `sampling/*` and
//! `elicitation/*` are refused with their own fixed messages (the client
//! never samples and never elicits; see [`SAMPLING_REFUSED_MESSAGE`] and
//! [`ELICITATION_REFUSED_MESSAGE`]), and any other method gets JSON-RPC
//! `-32601` (Method not found). Server notifications (no `id`) are never
//! answered inline; every method frame [`handle_server_request`] declines
//! is stashed into the caller's `surfaced` buffer by
//! [`wait_for_response`] instead of being dropped (AI-0212) —
//! `notifications/tools/list_changed` in particular reaches the host, which
//! routes it through
//! [`crate::tools::is_tools_list_changed_notification`] and marks that
//! server's adapter stale (AI-0211).

use std::time::Instant;

use crate::error::{McpError, McpFailure, McpStage, bound_error_text};
use crate::json::{
    escaped, find_bool_field, find_object_field, find_raw_field, find_string_field, has_method,
};
use crate::{McpTransport, PROTOCOL_VERSION};

/// Client name reported in `clientInfo`.
pub const MCP_CLIENT_NAME: &str = "bitty-ai-mcp";

/// Client version reported in `clientInfo` (crate version, single place).
pub const MCP_CLIENT_VERSION: &str = "0.0.1";

/// Fixed refusal message for `sampling/*` server requests.
///
/// Never-sample policy: this client never asks a server to perform model
/// sampling on its behalf, so every server-to-client `sampling/*` request
/// (canonically `sampling/createMessage`) is refused with this static text.
/// The refusal carries the same `-32601` error-code shape as an unknown
/// method, but the distinct message keeps the taxonomy explicit.
pub const SAMPLING_REFUSED_MESSAGE: &str = "Sampling not supported";

/// Fixed refusal message for `elicitation/*` server requests.
///
/// Never-elicit policy: this client never solicits user input through a
/// server, so every server-to-client `elicitation/*` request (canonically
/// `elicitation/create`) is refused with this static text. Same `-32601`
/// error-code shape as an unknown method, distinct message for taxonomy.
pub const ELICITATION_REFUSED_MESSAGE: &str = "Elicitation not supported";

/// Fixed message for any other unknown server-request method.
pub const UNKNOWN_METHOD_MESSAGE: &str = "Method not found";

/// Parsed `initialize` result: the facts the client requires before use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializeResult {
    /// Server protocol version (always the pin; anything else is refused).
    pub protocol_version: String,
    /// Whether the server offers the `tools` capability (always true here;
    /// absence is refused before this returns).
    pub tools_supported: bool,
}

/// Build the `initialize` request frame for `id`.
///
/// Declares `capabilities.roots` (list-changed notifications) and bounded
/// client info. The `roots` themselves are only ever revealed through
/// [`handle_server_request`], which answers with exactly `cwd`.
#[must_use]
pub fn initialize_request(id: u64, _cwd: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"roots\":{{\"listChanged\":true}}}},\"clientInfo\":{{\"name\":\"{MCP_CLIENT_NAME}\",\"version\":\"{MCP_CLIENT_VERSION}\"}}}}}}"
    )
}

/// Build the `notifications/initialized` frame (no `id`, no reply).
#[must_use]
pub fn initialized_notification() -> String {
    "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}".to_owned()
}

/// Parse an `initialize` response frame.
///
/// Requires `result.protocolVersion` equal to the pin and a `tools`
/// capability inside `result.capabilities` (either `"tools":{...}` or
/// `"tools":true`). A JSON-RPC `error` answer, a version drift, or a
/// missing tools capability all fail closed.
///
/// # Errors
///
/// Returns [`McpFailure::HandshakeRejected`], [`McpFailure::VersionMismatch`],
/// or [`McpFailure::NoToolCapability`].
pub fn parse_initialize_result(line: &str) -> Result<InitializeResult, McpError> {
    if let Some(error) = find_object_field(line, "error") {
        let code = find_raw_field(error, "code").unwrap_or("?");
        let message = find_string_field(error, "message").unwrap_or_default();
        return Err(McpError::new(
            McpStage::Handshake,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    &format!("server answered error {code}: {message}"),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        ));
    }
    let result = find_object_field(line, "result").ok_or_else(|| {
        McpError::new(
            McpStage::Handshake,
            McpFailure::HandshakeRejected {
                detail: "initialize answer carries no result".to_owned(),
            },
        )
    })?;
    let version = find_string_field(result, "protocolVersion").ok_or_else(|| {
        McpError::new(
            McpStage::Handshake,
            McpFailure::HandshakeRejected {
                detail: "initialize result carries no protocolVersion".to_owned(),
            },
        )
    })?;
    if version != PROTOCOL_VERSION {
        return Err(McpError::new(
            McpStage::Handshake,
            McpFailure::VersionMismatch {
                expected: PROTOCOL_VERSION.to_owned(),
                got: bound_error_text(&version, crate::error::MAX_ERROR_NAME_BYTES),
            },
        ));
    }
    let capabilities = find_object_field(result, "capabilities")
        .ok_or_else(|| McpError::new(McpStage::Handshake, McpFailure::NoToolCapability))?;
    let tools_supported = find_object_field(capabilities, "tools").is_some()
        || find_bool_field(capabilities, "tools") == Some(true);
    if !tools_supported {
        return Err(McpError::new(
            McpStage::Handshake,
            McpFailure::NoToolCapability,
        ));
    }
    Ok(InitializeResult {
        protocol_version: version,
        tools_supported,
    })
}

/// Answer one server request frame, if `line` is one.
///
/// Returns `Some(response_frame)` for requests carrying an `id` (`ping`,
/// `roots/list`, the `sampling/*` / `elicitation/*` refusals, or `-32601`
/// for anything else) and `None` for responses, notifications, and
/// malformed lines (never reply to those: answering a response would corrupt
/// the id multiplexer, and answering a notification violates JSON-RPC).
///
/// Refusals never echo request params: every error frame carries only a
/// fixed static message ([`SAMPLING_REFUSED_MESSAGE`],
/// [`ELICITATION_REFUSED_MESSAGE`], or [`UNKNOWN_METHOD_MESSAGE`]), so
/// untrusted server payloads cannot reflect through the reply.
#[must_use]
pub fn handle_server_request(line: &str, cwd: &str) -> Option<String> {
    let method = find_string_field(line, "method")?;
    let id = find_raw_field(line, "id")?;
    let id_token = render_id_token(id)?;
    match method.as_str() {
        "ping" => Some(format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{id_token},\"result\":{{}}}}"
        )),
        "roots/list" => {
            let uri = format!("file://{cwd}");
            Some(format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id_token},\"result\":{{\"roots\":[{{\"uri\":\"{}\"}}]}}}}",
                escaped(&uri)
            ))
        }
        "sampling/createMessage" => Some(refusal_frame(&id_token, SAMPLING_REFUSED_MESSAGE)),
        "elicitation/create" => Some(refusal_frame(&id_token, ELICITATION_REFUSED_MESSAGE)),
        other if other.starts_with("sampling/") => {
            Some(refusal_frame(&id_token, SAMPLING_REFUSED_MESSAGE))
        }
        other if other.starts_with("elicitation/") => {
            Some(refusal_frame(&id_token, ELICITATION_REFUSED_MESSAGE))
        }
        _ => Some(refusal_frame(&id_token, UNKNOWN_METHOD_MESSAGE)),
    }
}

/// Build a `-32601` refusal frame carrying only a fixed static message.
///
/// `message` is always one of the module constants; request params are
/// never interpolated, so the reply cannot reflect untrusted server input.
fn refusal_frame(id_token: &str, message: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id_token},\"error\":{{\"code\":-32601,\"message\":\"{message}\"}}}}"
    )
}

/// Render an extracted `id` token back into a frame.
///
/// Numeric and boolean/null literals pass through; string ids (returned
/// without quotes by [`find_raw_field`]) are re-quoted only when they look
/// like plain tokens, otherwise the line is treated as unanswerable (`None`)
/// rather than risking an injection through the echo.
fn render_id_token(token: &str) -> Option<String> {
    if token.is_empty() {
        return None;
    }
    if token.bytes().all(|byte| byte.is_ascii_digit()) {
        return Some(token.to_owned());
    }
    if matches!(token, "true" | "false" | "null") {
        return Some(token.to_owned());
    }
    if token.len() > 128 {
        return None;
    }
    if token
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.')
    {
        return Some(format!("\"{token}\""));
    }
    None
}

/// Wait for the response frame carrying `want_id`, answering interleaved
/// server requests inline (ping, roots/list, sampling/elicitation refusals,
/// `-32601`).
///
/// `deadline` bounds the whole wait; each transport poll uses the remaining
/// time. Responses for other ids are ignored. Every skipped method frame
/// [`handle_server_request`] declines (server notifications, which carry no
/// `id`) is pushed onto `surfaced` in arrival order and returned to the
/// caller alongside the response — synchronously, no threads (AI-0212).
/// Filtering is caller duty: the host routes each surfaced line through
/// [`crate::tools::is_tools_list_changed_notification`] (via
/// [`crate::bridge::McpToolAdapter::observe_notification`]). `surfaced`
/// keeps whatever arrived even when the wait itself fails, so a signal seen
/// before a timeout is still routable. Malformed lines never arrive (the
/// transport drops and counts them).
///
/// Recovery limit (PX-0913): a notification discarded before AI-0212 — or on
/// a failed wait whose caller drops `surfaced` — is unrecoverable unless the
/// server re-sends it. A later poll cannot resurrect it, and the list
/// version cannot expose the missed transition. No server-state
/// reconciliation (digest compare, periodic re-list) is attempted here:
/// only frames observed during this wait are surfaced.
///
/// # Errors
///
/// Returns [`McpFailure::Timeout`] when the deadline passes with no answer,
/// and propagates transport errors (broken pipe, closed child) unchanged.
pub fn wait_for_response(
    transport: &mut dyn McpTransport,
    want_id: u64,
    cwd: &str,
    deadline: Instant,
    budget_ms: u64,
    surfaced: &mut Vec<String>,
) -> Result<String, McpError> {
    let want = want_id.to_string();
    loop {
        let remaining_ms = remaining_ms(deadline);
        match transport.recv_line(remaining_ms)? {
            None => {
                if Instant::now() >= deadline {
                    return Err(McpError::new(
                        McpStage::Handshake,
                        McpFailure::Timeout {
                            timeout_ms: budget_ms,
                        },
                    ));
                }
                continue;
            }
            Some(line) => {
                if has_method(&line) {
                    if let Some(reply) = handle_server_request(&line, cwd) {
                        transport.send_line(&reply)?;
                    } else {
                        // No answerable `id`: a server notification. Stash
                        // the exact frame for the caller instead of
                        // dropping it; non-notification behavior below is
                        // unchanged.
                        surfaced.push(line);
                    }
                    if Instant::now() >= deadline {
                        return Err(McpError::new(
                            McpStage::Handshake,
                            McpFailure::Timeout {
                                timeout_ms: budget_ms,
                            },
                        ));
                    }
                    continue;
                }
                if response_id_matches(&line, &want) {
                    return Ok(line);
                }
                // A response for another id: ignore and keep waiting within
                // the deadline (unchanged; responses are never surfaced).
                if Instant::now() >= deadline {
                    return Err(McpError::new(
                        McpStage::Handshake,
                        McpFailure::Timeout {
                            timeout_ms: budget_ms,
                        },
                    ));
                }
            }
        }
    }
}

/// Milliseconds left until `deadline`, saturating at zero.
fn remaining_ms(deadline: Instant) -> u64 {
    let now = Instant::now();
    if now >= deadline {
        0
    } else {
        deadline
            .duration_since(now)
            .as_millis()
            .min(u128::from(u32::MAX)) as u64
    }
}

/// Whether `line` is a JSON-RPC response for id `want`.
fn response_id_matches(line: &str, want: &str) -> bool {
    match find_raw_field(line, "id") {
        Some(token) => token == want,
        None => false,
    }
}

/// Run the full handshake: send `initialize`, wait for and validate the
/// answer (answering server requests inline), then send `notifications/initialized`.
///
/// `timeout_ms` bounds the whole handshake (`1..=MAX_TIMEOUT_MS` by
/// construction of validated configs; unvalidated callers are clamped to at
/// least 1ms so the wait cannot block without bound).
///
/// Notifications observed while waiting for the `initialize` answer are
/// pushed onto `surfaced` (see [`wait_for_response`]) for the caller to
/// route. No adapter — and therefore no tool snapshot — exists yet, so the
/// handshake itself takes no staleness action; the follow-up `tools/list`
/// import is already a fresh read of post-signal state.
///
/// # Errors
///
/// Returns handshake, version, capability, timeout, or transport errors.
pub fn handshake(
    transport: &mut dyn McpTransport,
    cwd: &str,
    timeout_ms: u64,
    surfaced: &mut Vec<String>,
) -> Result<InitializeResult, McpError> {
    let bound = timeout_ms.max(1);
    let deadline = Instant::now() + std::time::Duration::from_millis(bound);
    transport.send_line(&initialize_request(1, cwd))?;
    let answer = wait_for_response(transport, 1, cwd, deadline, bound, surfaced)?;
    let negotiated = parse_initialize_result(&answer)?;
    transport.send_line(&initialized_notification())?;
    Ok(negotiated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::McpStage;
    use std::collections::VecDeque;

    struct FakeTransport {
        inbound: VecDeque<String>,
        sent: Vec<String>,
    }

    impl FakeTransport {
        fn new(lines: Vec<String>) -> Self {
            Self {
                inbound: lines.into_iter().collect(),
                sent: Vec::new(),
            }
        }
    }

    impl McpTransport for FakeTransport {
        fn send_line(&mut self, line: &str) -> Result<(), McpError> {
            self.sent.push(line.to_owned());
            Ok(())
        }

        fn recv_line(&mut self, _timeout_ms: u64) -> Result<Option<String>, McpError> {
            Ok(self.inbound.pop_front())
        }
    }

    fn ok_result(tools: &str) -> String {
        format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"tools\":{tools}}},\"serverInfo\":{{\"name\":\"fake\",\"version\":\"0.0.1\"}}}}}}"
        )
    }

    #[test]
    fn initialize_request_pins_version_and_client_info() {
        let frame = initialize_request(7, "/tmp/bitty");
        assert!(frame.contains("\"method\":\"initialize\""));
        assert!(frame.contains(&format!("\"protocolVersion\":\"{PROTOCOL_VERSION}\"")));
        assert!(frame.contains(MCP_CLIENT_NAME));
        assert!(frame.contains("\"id\":7"));
    }

    #[test]
    fn parse_accepts_object_and_bool_tool_caps() {
        for tools in ["{}", "{\"listChanged\":true}", "true"] {
            let parsed = parse_initialize_result(&ok_result(tools)).expect("accept");
            assert_eq!(parsed.protocol_version, PROTOCOL_VERSION);
            assert!(parsed.tools_supported);
        }
    }

    #[test]
    fn parse_refuses_missing_tools_capability() {
        let line = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{PROTOCOL_VERSION}\",\"capabilities\":{{\"roots\":{{}}}}}}}}"
        );
        let error = parse_initialize_result(&line).expect_err("no tools");
        assert_eq!(error.stage, McpStage::Handshake);
        assert_eq!(error.failure, McpFailure::NoToolCapability);
        assert!(!error.retryable);
    }

    #[test]
    fn parse_refuses_version_drift() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"protocolVersion\":\"1999-01-01\",\"capabilities\":{\"tools\":{}}}}";
        let error = parse_initialize_result(line).expect_err("drift");
        assert!(matches!(error.failure, McpFailure::VersionMismatch { .. }));
    }

    #[test]
    fn parse_refuses_error_answers() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32600,\"message\":\"bad\"}}";
        let error = parse_initialize_result(line).expect_err("error answer");
        assert!(matches!(
            error.failure,
            McpFailure::HandshakeRejected { .. }
        ));
    }

    #[test]
    fn server_requests_are_answered_cwd_only() {
        let ping = "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"}";
        let pong = handle_server_request(ping, "/tmp/bitty").expect("pong");
        assert!(pong.contains("\"id\":3"));
        assert!(pong.contains("\"result\":{}"));

        let roots = "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"roots/list\"}";
        let answer = handle_server_request(roots, "/tmp/bitty").expect("roots");
        assert!(answer.contains("file:///tmp/bitty"));
        // Exactly one root: the pinned cwd, never the wider filesystem.
        assert_eq!(answer.matches("file://").count(), 1);

        let weird = "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"singular/plural\"}";
        let answer = handle_server_request(weird, "/tmp/bitty").expect("method-not-found");
        assert!(answer.contains("-32601"));

        // Responses and notifications get no reply.
        assert_eq!(
            handle_server_request("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}", "/tmp/bitty"),
            None
        );
        assert_eq!(
            handle_server_request(
                "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}",
                "/tmp/bitty"
            ),
            None
        );
    }

    #[test]
    fn sampling_and_elicitation_are_refused_with_distinct_messages() {
        // Canonical methods carry their own static message under the same
        // -32601 code shape as unknown methods.
        let sampling = "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"sampling/createMessage\",\"params\":{\"messages\":[{\"role\":\"user\",\"content\":{\"text\":\"canary-sampling-params\"}}]}}";
        let answer = handle_server_request(sampling, "/tmp/bitty").expect("sampling refused");
        assert_eq!(
            answer,
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":11,\"error\":{{\"code\":-32601,\"message\":\"{SAMPLING_REFUSED_MESSAGE}\"}}}}"
            )
        );
        assert!(!answer.contains("Method not found"));
        // No request-param echo: the untrusted payload cannot reflect.
        assert!(!answer.contains("canary-sampling-params"));

        let elicitation = "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"elicitation/create\",\"params\":{\"message\":\"canary-elicitation-params\"}}";
        let answer = handle_server_request(elicitation, "/tmp/bitty").expect("elicitation refused");
        assert_eq!(
            answer,
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":12,\"error\":{{\"code\":-32601,\"message\":\"{ELICITATION_REFUSED_MESSAGE}\"}}}}"
            )
        );
        assert!(!answer.contains("Method not found"));
        assert!(!answer.contains("canary-elicitation-params"));

        // Namespace catch-alls: future variants cannot fall through to the
        // generic message, and string ids keep their quoted shape.
        let future_sampling =
            "{\"jsonrpc\":\"2.0\",\"id\":\"abc-1\",\"method\":\"sampling/futureVariant\"}";
        let answer = handle_server_request(future_sampling, "/tmp/bitty").expect("sampling prefix");
        assert_eq!(
            answer,
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":\"abc-1\",\"error\":{{\"code\":-32601,\"message\":\"{SAMPLING_REFUSED_MESSAGE}\"}}}}"
            )
        );

        let future_elicitation =
            "{\"jsonrpc\":\"2.0\",\"id\":13,\"method\":\"elicitation/futureVariant\"}";
        let answer =
            handle_server_request(future_elicitation, "/tmp/bitty").expect("elicitation prefix");
        assert_eq!(
            answer,
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":13,\"error\":{{\"code\":-32601,\"message\":\"{ELICITATION_REFUSED_MESSAGE}\"}}}}"
            )
        );

        // Unknown methods keep the generic message; ping and roots/list are
        // unaffected by the new arms.
        let unknown = "{\"jsonrpc\":\"2.0\",\"id\":14,\"method\":\"tools/frobnicate\"}";
        let answer = handle_server_request(unknown, "/tmp/bitty").expect("unknown refused");
        assert_eq!(
            answer,
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":14,\"error\":{{\"code\":-32601,\"message\":\"{UNKNOWN_METHOD_MESSAGE}\"}}}}"
            )
        );
        assert!(!answer.contains(SAMPLING_REFUSED_MESSAGE));
        assert!(!answer.contains(ELICITATION_REFUSED_MESSAGE));

        let ping = "{\"jsonrpc\":\"2.0\",\"id\":15,\"method\":\"ping\"}";
        let pong = handle_server_request(ping, "/tmp/bitty").expect("pong");
        assert!(pong.contains("\"result\":{}"));
        assert!(!pong.contains("error"));

        let roots = "{\"jsonrpc\":\"2.0\",\"id\":16,\"method\":\"roots/list\"}";
        let answer = handle_server_request(roots, "/tmp/bitty").expect("roots");
        assert!(answer.contains("file:///tmp/bitty"));
        assert!(!answer.contains("error"));
    }

    #[test]
    fn handshake_sequence_sends_initialized() {
        let mut transport = FakeTransport::new(vec![ok_result("{}")]);
        let mut surfaced = Vec::new();
        let negotiated =
            handshake(&mut transport, "/tmp/bitty", 1_000, &mut surfaced).expect("handshake");
        assert!(negotiated.tools_supported);
        assert_eq!(transport.sent.len(), 2);
        assert!(transport.sent[0].contains("\"method\":\"initialize\""));
        assert!(transport.sent[1].contains("notifications/initialized"));
    }

    #[test]
    fn handshake_answers_ping_while_waiting() {
        let mut transport = FakeTransport::new(vec![
            "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}".to_owned(),
            ok_result("{}"),
        ]);
        let mut surfaced = Vec::new();
        handshake(&mut transport, "/tmp/bitty", 1_000, &mut surfaced).expect("handshake");
        assert_eq!(transport.sent.len(), 3);
        assert!(transport.sent[1].contains("\"id\":9"));
        assert!(transport.sent[1].contains("\"result\":{}"));
    }
}

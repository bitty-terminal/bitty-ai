//! Streamable HTTP/SSE line transport over [`bitty_network_api`].
//!
//! [`HttpLineTransport`] implements [`crate::McpTransport`] (`send_line` /
//! `recv_line`) on top of a sync [`NetworkService`]: each `send_line` POSTs
//! one JSON-RPC frame with `Accept: application/json, text/event-stream` and
//! `Content-Type: application/json` (plus `Mcp-Session-Id` once known),
//! captures the session id from the response, and queues the decoded answer
//! lines; `recv_line` pops the queue else reports `Ok(None)`. `shutdown`
//! best-effort DELETEs the session. There is deliberately no background GET
//! listener in v1 (poll-per-request only) and no auto-reconnect: failures
//! surface as [`McpFailure::TransportClosed`], [`McpFailure::Timeout`],
//! [`McpFailure::FrameTooLarge`], [`McpFailure::HandshakeRejected`], or
//! [`McpFailure::Io`] (operation name only), and the [`crate::supervise`]
//! cooldown owns retry.
//!
//! Response shapes:
//!
//! - `application/json`: one JSON-RPC frame queued verbatim (frame-cap
//!   bounded).
//! - `text/event-stream`: each `data:` payload holding a JSON object is one
//!   queued line (frame-cap bounded per event, fan-out bounded by
//!   [`MAX_SSE_FRAMES_PER_RESPONSE`]); other SSE fields are ignored.
//! - Empty bodies (for example `202 Accepted` on a notification) queue
//!   nothing: `recv_line` then reports `Ok(None)`.
//!
//! Error mapping (names-only, never payloads, URLs, or header values):
//!
//! - Capability deny pre-contact: [`McpFailure::TransportClosed`] with zero
//!   service calls.
//! - [`NetworkError::Timeout`]: [`McpFailure::Timeout`] (handshake/list
//!   surface it; `tools/call` folds it to `Unknown` through the existing
//!   wrapper: reconcile before retry, never blind retry).
//! - [`NetworkError::Budget`] / [`NetworkError::CountBudget`]:
//!   [`McpFailure::FrameTooLarge`] (frame-reject for `tools/list`; `Unknown`
//!   for `tools/call` through the existing wrapper).
//! - [`NetworkError::Tls`]: [`McpFailure::Io`] with context `"http post"`
//!   (operation name only; the TLS category never enters the diagnostic).
//! - Non-2xx or bad/missing `Content-Type` or malformed bodies: a synthetic
//!   JSON-RPC error frame carrying the outgoing `id` is queued, so the
//!   existing parsers map it to [`McpFailure::HandshakeRejected`]
//!   (handshake/list) or `Failed` (call). Notifications (no `id`) return
//!   [`McpFailure::HandshakeRejected`] directly.
//! - Oversize single-JSON or SSE events: [`McpFailure::FrameTooLarge`]
//!   directly (frame-reject for list, `Unknown` for call).
//! - An SSE body cut before any matching `id` simply queues no match: the
//!   existing waiter times out and `tools/call` reports `Unknown`.
//!
//! [`NetworkError`]: bitty_network_api::NetworkError
//! [`NetworkService`]: bitty_network_api::NetworkService
//! [`McpFailure::TransportClosed`]: crate::error::McpFailure::TransportClosed
//! [`McpFailure::Timeout`]: crate::error::McpFailure::Timeout
//! [`McpFailure::FrameTooLarge`]: crate::error::McpFailure::FrameTooLarge
//! [`McpFailure::HandshakeRejected`]: crate::error::McpFailure::HandshakeRejected
//! [`McpFailure::Io`]: crate::error::McpFailure::Io

use std::collections::VecDeque;
use std::time::Duration;

use bitty_network_api::{HttpMethod, NetworkError, NetworkService, Request, Response};

use crate::McpTransport;
use crate::error::{McpError, McpFailure, McpStage, bound_error_text};
use crate::frame::MAX_FRAME_BYTES;
use crate::remote::RemoteServerConfig;

/// `Accept` advertised on every POST (single JSON or SSE per reply).
pub const HTTP_ACCEPT: &str = "application/json, text/event-stream";
/// `Content-Type` sent on every POST (frames are JSON).
pub const HTTP_CONTENT_TYPE: &str = "application/json";
/// Session header read on responses and sent once known.
pub const SESSION_HEADER: &str = "mcp-session-id";
/// Maximum SSE `data:` frames queued per response (fan-out bound).
pub const MAX_SSE_FRAMES_PER_RESPONSE: usize = 16;
/// Maximum session id length in bytes (opaque token bound).
pub const MAX_SESSION_ID_LEN: usize = 256;
/// Synthetic JSON-RPC error code for transport-level HTTP rejections.
pub const TRANSPORT_ERROR_CODE: i64 = -32000;

fn timeout_error(timeout_ms: u64) -> McpError {
    McpError::new(McpStage::Frame, McpFailure::Timeout { timeout_ms })
}

fn frame_too_large() -> McpError {
    McpError::new(
        McpStage::Frame,
        McpFailure::FrameTooLarge {
            limit: MAX_FRAME_BYTES,
            actual: MAX_FRAME_BYTES + 1,
        },
    )
}

fn transport_closed() -> McpError {
    McpError::new(McpStage::Supervise, McpFailure::TransportClosed)
}

fn io_post() -> McpError {
    McpError::new(
        McpStage::Frame,
        McpFailure::Io {
            context: bound_error_text("http post", crate::error::MAX_ERROR_TEXT_BYTES),
        },
    )
}

fn handshake_rejected(detail: &str) -> McpError {
    McpError::new(
        McpStage::Frame,
        McpFailure::HandshakeRejected {
            detail: bound_error_text(detail, crate::error::MAX_ERROR_TEXT_BYTES),
        },
    )
}

fn map_network_error(error: NetworkError, timeout_ms: u64) -> McpError {
    match error {
        NetworkError::Offline => transport_closed(),
        NetworkError::Denied { .. } => transport_closed(),
        NetworkError::Timeout { .. } => timeout_error(timeout_ms),
        NetworkError::Budget { .. } | NetworkError::CountBudget { .. } => frame_too_large(),
        NetworkError::Tls { .. } => io_post(),
    }
}

fn is_json_media_type(value: &str) -> bool {
    value
        .split(';')
        .next()
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
}

fn is_sse_media_type(value: &str) -> bool {
    value
        .split(';')
        .next()
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("text/event-stream"))
}

fn is_json_object(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with('{') && trimmed.ends_with('}')
}

/// Outgoing JSON-RPC `id` token (`None` for notifications).
fn outgoing_id(line: &str) -> Option<String> {
    let token = crate::json::find_raw_field(line, "id")?;
    if token.is_empty() {
        return None;
    }
    // Numeric and literal ids pass through; string ids (returned without
    // quotes by `find_raw_field`) re-quote only for safe tokens.
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

/// Synthetic JSON-RPC error frame for `id` (transport-level HTTP rejection).
///
/// Carries only the numeric status or a fixed shape phrase: never bodies,
/// URLs, or header values.
fn synthetic_error(id_token: &str, message: &str) -> String {
    let safe = bound_error_text(message, crate::error::MAX_ERROR_TEXT_BYTES);
    let mut escaped = String::with_capacity(safe.len());
    crate::json::escape_into(&mut escaped, &safe);
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id_token},\"error\":{{\"code\":{},\"message\":\"{escaped}\"}}}}",
        TRANSPORT_ERROR_CODE
    )
}

fn valid_session_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_SESSION_ID_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() || byte == b' ')
        && !value.bytes().any(|byte| byte == b'\0')
}

/// [`McpTransport`] over Streamable HTTP/SSE, backed by a sync service.
///
/// Owns the [`RemoteServerConfig`], the service, a line queue, the optional
/// session id, and a closed flag. Synchronous only (`&mut self`, no threads,
/// no async). Whole-operation deadlines come from the config: each POST sets
/// `Request::timeout` to `timeout_ms` and `max_body_bytes` to the frame cap.
pub struct HttpLineTransport<S> {
    config: RemoteServerConfig,
    service: S,
    queue: VecDeque<String>,
    session_id: Option<String>,
    closed: bool,
}

impl<S> HttpLineTransport<S> {
    /// Build a transport over `service` after validating `config` fail-closed.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::InvalidConfig`] when the config shape is refused.
    pub fn new(config: RemoteServerConfig, service: S) -> Result<Self, McpError> {
        config.validate()?;
        Ok(Self {
            config,
            service,
            queue: VecDeque::new(),
            session_id: None,
            closed: false,
        })
    }

    /// Remote configuration backing this transport.
    #[must_use]
    pub fn config(&self) -> &RemoteServerConfig {
        &self.config
    }

    /// Underlying network service (for recording fakes in tests).
    #[must_use]
    pub fn service(&self) -> &S {
        &self.service
    }

    /// Mutable service access.
    #[must_use]
    pub fn service_mut(&mut self) -> &mut S {
        &mut self.service
    }

    /// Current session id, when the server has issued one.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Whether [`HttpLineTransport::shutdown`] has closed this transport.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Queued line count (unread responses).
    #[must_use]
    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    /// Best-effort session close: DELETEs the session, then marks closed.
    ///
    /// Failures (including `405 Method Not Allowed` for servers without
    /// session management) are ignored; after this call `send_line` and
    /// `recv_line` fail with [`McpFailure::TransportClosed`]. No
    /// auto-reconnect is attempted: the [`crate::supervise::McpHost`]
    /// cooldown owns retry.
    pub fn shutdown(&mut self)
    where
        S: NetworkService,
    {
        if self.closed {
            return;
        }
        self.closed = true;
        self.queue.clear();
        let session = self.session_id.clone();
        self.session_id = None;
        let Some(session) = session else {
            return;
        };
        let mut request = Request {
            method: HttpMethod::Delete,
            url: self.config.url.clone(),
            headers: Vec::new(),
            body: Vec::new(),
            timeout: Some(Duration::from_millis(self.config.effective_timeout_ms())),
            max_body_bytes: Some(MAX_FRAME_BYTES as u64),
        };
        for (name, value) in &self.config.headers {
            request.headers.push((name.clone(), value.clone()));
        }
        request.headers.push((SESSION_HEADER.to_owned(), session));
        if self.config.capability.check_request(&request).is_err() {
            return;
        }
        let _ = self.service.request(&request);
    }

    fn post_request(&self, body: &[u8]) -> Request {
        let mut request = Request::post(self.config.url.clone(), body.to_vec())
            .with_timeout(Duration::from_millis(self.config.effective_timeout_ms()))
            .with_max_body_bytes(MAX_FRAME_BYTES as u64)
            .with_header("accept", HTTP_ACCEPT)
            .with_header("content-type", HTTP_CONTENT_TYPE);
        for (name, value) in &self.config.headers {
            request = request.with_header(name, value);
        }
        if let Some(session) = self.session_id.as_ref() {
            request = request.with_header(SESSION_HEADER, session);
        }
        request
    }

    fn note_session(&mut self, response: &Response) {
        if let Some(value) = response.header(SESSION_HEADER) {
            if valid_session_id(value) {
                self.session_id = Some(value.to_owned());
            }
        }
    }

    fn enqueue_or_reject(&mut self, id: Option<&str>, message: &str) -> Result<(), McpError> {
        match id {
            Some(token) => {
                if self.queue.len() >= MAX_SSE_FRAMES_PER_RESPONSE {
                    return Err(frame_too_large());
                }
                self.queue.push_back(synthetic_error(token, message));
                Ok(())
            }
            None => Err(handshake_rejected(message)),
        }
    }

    fn handle_json_body(&mut self, id: Option<&str>, body: &[u8]) -> Result<(), McpError> {
        if body.is_empty() {
            return Ok(());
        }
        if body.len() > MAX_FRAME_BYTES {
            return Err(frame_too_large());
        }
        let text = match std::str::from_utf8(body) {
            Ok(text) => text,
            Err(_) => {
                return self.enqueue_or_reject(id, "malformed response body");
            }
        };
        if !is_json_object(text) {
            return self.enqueue_or_reject(id, "malformed response body");
        }
        if self.queue.len() >= MAX_SSE_FRAMES_PER_RESPONSE {
            return Err(frame_too_large());
        }
        self.queue.push_back(text.to_owned());
        Ok(())
    }

    fn handle_sse_body(&mut self, id: Option<&str>, body: &[u8]) -> Result<(), McpError> {
        if body.is_empty() {
            return Ok(());
        }
        let text = match std::str::from_utf8(body) {
            Ok(text) => text,
            Err(_) => {
                return self.enqueue_or_reject(id, "malformed event-stream body");
            }
        };
        for raw_line in text.split('\n') {
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            let trimmed = line.trim_start();
            if !trimmed.starts_with("data:") {
                continue;
            }
            let mut payload = &trimmed["data:".len()..];
            if let Some(stripped) = payload.strip_prefix(' ') {
                payload = stripped;
            }
            let payload = payload.trim();
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            if payload.len() > MAX_FRAME_BYTES {
                return Err(frame_too_large());
            }
            if !is_json_object(payload) {
                continue;
            }
            if self.queue.len() >= MAX_SSE_FRAMES_PER_RESPONSE {
                return Err(frame_too_large());
            }
            self.queue.push_back(payload.to_owned());
        }
        let _ = id;
        Ok(())
    }

    fn handle_response(
        &mut self,
        response: &Response,
        id: Option<&str>,
        status: u16,
    ) -> Result<(), McpError> {
        if !(200..300).contains(&status) {
            let message = format!("http status {status}");
            return self.enqueue_or_reject(id, &message);
        }
        // Empty bodies (for example `202 Accepted` on a notification) queue
        // nothing regardless of `Content-Type`: there is no frame to decode.
        if response.body.is_empty() {
            return Ok(());
        }
        match response.header("content-type") {
            None => self.enqueue_or_reject(id, "unsupported content type"),
            Some(content_type) => {
                if is_json_media_type(content_type) {
                    self.handle_json_body(id, &response.body)
                } else if is_sse_media_type(content_type) {
                    self.handle_sse_body(id, &response.body)
                } else {
                    self.enqueue_or_reject(id, "unsupported content type")
                }
            }
        }
    }
}

impl<S: NetworkService> McpTransport for HttpLineTransport<S> {
    fn send_line(&mut self, line: &str) -> Result<(), McpError> {
        if self.closed {
            return Err(transport_closed());
        }
        if line.len() > MAX_FRAME_BYTES {
            return Err(McpError::new(
                McpStage::Frame,
                McpFailure::FrameTooLarge {
                    limit: MAX_FRAME_BYTES,
                    actual: line.len(),
                },
            ));
        }
        let id = outgoing_id(line);
        let request = self.post_request(line.as_bytes());
        // Capability first: zero service calls on deny.
        if self.config.capability.check_request(&request).is_err() {
            return Err(transport_closed());
        }
        let timeout_ms = self.config.effective_timeout_ms();
        match self.service.request(&request) {
            Err(error) => Err(map_network_error(error, timeout_ms)),
            Ok(response) => {
                let status = response.status;
                self.note_session(&response);
                self.handle_response(&response, id.as_deref(), status)
            }
        }
    }

    fn recv_line(&mut self, _timeout_ms: u64) -> Result<Option<String>, McpError> {
        if self.closed {
            return Err(transport_closed());
        }
        Ok(self.queue.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

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

        fn queue(&self, response: Response) {
            if let Ok(mut script) = self.script.lock() {
                script.push_back(Ok(response));
            }
        }

        fn recorded(&self) -> Vec<Request> {
            self.requests
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_default()
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

    fn config_for(url: &str) -> RemoteServerConfig {
        RemoteServerConfig {
            id: "demo".to_owned(),
            url: url.to_owned(),
            timeout_ms: 1_000,
            tool_allowlist: vec!["echo".to_owned()],
            headers: Vec::new(),
            capability: bitty_network_api::NetworkCapability::offline()
                .with_domain("mcp.example.com"),
        }
    }

    #[test]
    fn posts_json_with_session_capture() {
        let service = FakeService::new();
        service.queue(Response {
            status: 200,
            headers: vec![
                ("content-type".to_owned(), "application/json".to_owned()),
                ("mcp-session-id".to_owned(), "sess-1".to_owned()),
            ],
            body: b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}".to_vec(),
        });
        let mut transport =
            HttpLineTransport::new(config_for("https://mcp.example.com/rpc"), service)
                .expect("transport");
        transport
            .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
            .expect("post");
        assert_eq!(transport.session_id(), Some("sess-1"));
        assert_eq!(
            transport.recv_line(10).expect("recv"),
            Some("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}".to_owned())
        );
        let recorded = transport.service().recorded();
        assert_eq!(recorded.len(), 1);
        let headers = &recorded[0].headers;
        assert!(
            headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("accept")
                    && value.contains("text/event-stream"))
        );
        assert!(
            headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("content-type")
                    && value == "application/json")
        );
    }

    #[test]
    fn sse_data_lines_queue_individually() {
        let service = FakeService::new();
        service.queue(Response {
            status: 200,
            headers: vec![(
                "content-type".to_owned(),
                "text/event-stream".to_owned(),
            )],
            body: b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}\n\n".to_vec(),
        });
        let mut transport =
            HttpLineTransport::new(config_for("https://mcp.example.com/rpc"), service)
                .expect("transport");
        transport
            .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}")
            .expect("post");
        assert_eq!(transport.queued_len(), 2);
        assert!(
            transport
                .recv_line(10)
                .expect("recv")
                .is_some_and(|line| line.contains("\"id\":1"))
        );
    }

    #[test]
    fn capability_deny_is_zero_contact() {
        let service = FakeService::new();
        let mut config = config_for("https://mcp.example.com/rpc");
        config.capability = bitty_network_api::NetworkCapability::offline();
        let mut transport = HttpLineTransport::new(config, service).expect("transport");
        let error = transport
            .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
            .expect_err("offline must deny");
        assert_eq!(error.failure, McpFailure::TransportClosed);
        assert_eq!(transport.service().recorded().len(), 0);
    }

    #[test]
    fn non_2xx_queues_synthetic_error() {
        let service = FakeService::new();
        service.queue(Response {
            status: 401,
            headers: Vec::new(),
            body: b"unauthorized".to_vec(),
        });
        let mut transport =
            HttpLineTransport::new(config_for("https://mcp.example.com/rpc"), service)
                .expect("transport");
        transport
            .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}")
            .expect("synthetic queues as ok");
        let line = transport.recv_line(10).expect("recv").expect("queued");
        assert!(line.contains("\"id\":1"));
        assert!(line.contains("http status 401"));
        assert!(!line.contains("unauthorized"));
    }

    #[test]
    fn timeout_maps_to_timeout_budget_to_frame() {
        let service = FakeService::new();
        {
            if let Ok(mut script) = service.script.lock() {
                script.push_back(Err(NetworkError::Timeout {
                    after: Duration::from_millis(5),
                }));
                script.push_back(Err(NetworkError::Budget { limit_bytes: 8 }));
            }
        }
        let mut transport =
            HttpLineTransport::new(config_for("https://mcp.example.com/rpc"), service)
                .expect("transport");
        let timeout = transport
            .send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}")
            .expect_err("timeout");
        assert!(matches!(timeout.failure, McpFailure::Timeout { .. }));
        let budget = transport
            .send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}")
            .expect_err("budget");
        assert!(matches!(budget.failure, McpFailure::FrameTooLarge { .. }));
    }
}

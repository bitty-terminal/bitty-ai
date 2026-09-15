//! Experimental localhost-only model provider (AI-0042, slice-side only).
//!
//! This module proves the runtime [`ModelProvider`] seam can drive a real
//! HTTP round-trip to a local Ollama/OpenAI-compatible endpoint. It lives in
//! the pressure-test crate only; `bitty-ai-runtime/src` is untouched.
//!
//! ## Security posture
//!
//! - Localhost-only: only loopback hosts (`127.0.0.0/8`, `::1`,
//!   `localhost`) are accepted. Any other host fails closed with
//!   [`ProviderError::Transport`] (documented choice: `Transport` rather
//!   than `InvalidProviderId` because the refusal is a network-policy
//!   refusal, and the [`fallback_directive`] `Advance` lets selection try
//!   the next local candidate; no bytes ever leave the host).
//! - Plain HTTP only: there is no TLS in `std`, so `https` is never
//!   attempted. The host field must be bare (no `://` scheme smuggling).
//! - Mandatory timeouts: both connect and read timeouts are required
//!   (`1..=MAX_REQUEST_TIMEOUT_MS`); construction fails closed otherwise.
//! - No secrets in code: a local Ollama endpoint needs no key. Any
//!   key-bearing OpenAI-compatible path takes a caller-supplied value only,
//!   never hardcoded, never logged (see [`LocalEndpoint`] `Debug`
//!   redaction), and never carried in any [`ProviderError`] reason.
//!
//! ## Bounds (reused crate bounds, no new unbounded paths)
//!
//! - Response body ceiling: [`MAX_FRAGMENT_BYTES`] (64 KiB, single-fragment
//!   ceiling from `bitty-ai-runtime::stream`). Larger bodies fail closed.
//! - Request body ceiling: [`MAX_CANONICAL_BYTES`] (96 KiB, canonical prompt
//!   ceiling from `bitty-ai-runtime::prompt`). Larger requests fail closed.
//! - Header ceiling: 8 KiB policy bound (intrinsic HTTP header bound for
//!   this minimal client; headers larger than this fail closed).
//! - Path ceiling: [`MAX_CONSENT_SCOPE_LEN`] (128 bytes, consent-scope bound
//!   reused for the short request-target string).
//! - Model-name shape: [`validate_model_name`] (`MAX_MODEL_NAME_LEN`,
//!   covers `llama3.1:8b`, `org/model`).
//! - Provider-id shape: [`validate_provider_id`] (`MP-2`).
//! - Timeout ceiling: [`MAX_REQUEST_TIMEOUT_MS`] (`MP-8`).
//!
//! ## Protocol (minimal, fail-closed)
//!
//! Requests are OpenAI-compatible chat completions
//! (`POST {path}` with `{"model","messages","stream":false}`), which a local
//! Ollama server also serves at `/v1/chat/completions`. Tool observations
//! (`Role::Tool`) are folded into `user` messages with a `[tool] ` prefix so
//! no OpenAI tool protocol is required. Responses extract the first
//! `"content"` string (OpenAI chat / Ollama chat) with fallback to
//! `"response"` (Ollama `/api/generate`); anything else (malformed JSON,
//! missing field, over-large, non-2xx) fails closed. Chunked
//! `Transfer-Encoding` is rejected (close-delimited or `Content-Length`
//! only).
//!
//! ## Error mapping (existing variants only, `MP-7` parity)
//!
//! - `Transport`: non-loopback refusal, connect failure, malformed envelope,
//!   missing content field, over-large response, unsupported chunked
//!   encoding, non-2xx other than below (no secrets carried).
//! - `Timeout`: connect/read timeout (socket timeout is
//!   `min(endpoint, request)`; the reported `latency_ms` is the effective
//!   timeout plus one so `latency > timeout` holds deterministically).
//! - `Unknown`: truncated response (EOF before headers complete, or body
//!   shorter than declared `Content-Length`, or I/O error mid-body). The
//!   effect is uncertain from the client view, so the caller reconciles
//!   before retry and never falls back blindly.
//! - `Auth` (HTTP 401/403), `RateLimited` (HTTP 429, with `Retry-After`
//!   seconds converted to ms when present), `ModelUnavailable` (HTTP 404):
//!   precise status mapping reusing existing variants.
//! - `UnknownModel` / `BudgetExceeded` / `TimeoutTooLarge`: pre-I/O checks
//!   mirroring [`FakeProvider`] (model mismatch, context budget, timeout
//!   ceiling); the script is never consumed on failure (there is no script;
//!   `complete_calls` only increments on success).
//!
//! [`ModelProvider`]: bitty_ai_runtime::provider::ModelProvider
//! [`FakeProvider`]: bitty_ai_runtime::provider::FakeProvider
//! [`fallback_directive`]: bitty_ai_runtime::selection::fallback_directive

use std::fmt;
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::time::Duration;

use bitty_ai_runtime::bridge::MAX_CONSENT_SCOPE_LEN;
use bitty_ai_runtime::prompt::MAX_CANONICAL_BYTES;
use bitty_ai_runtime::provider::{
    MAX_REQUEST_TIMEOUT_MS, ModelCapability, ModelDescriptor, ModelProvider, ProviderError,
    ProviderTurn, Role, TurnRequest, validate_provider_id,
};
use bitty_ai_runtime::selection::validate_model_name;
use bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES;

/// Response body ceiling: reuses [`MAX_FRAGMENT_BYTES`] (64 KiB).
pub const MAX_LOCAL_RESPONSE_BYTES: usize = MAX_FRAGMENT_BYTES;
/// Request body ceiling: reuses [`MAX_CANONICAL_BYTES`] (96 KiB).
pub const MAX_LOCAL_REQUEST_BYTES: usize = MAX_CANONICAL_BYTES;
/// Header block ceiling: 8 KiB policy bound for this minimal client.
pub const MAX_LOCAL_HEADERS_BYTES: usize = 8 * 1024;
/// Request-target ceiling: reuses [`MAX_CONSENT_SCOPE_LEN`] (128 bytes).
pub const MAX_LOCAL_PATH_LEN: usize = MAX_CONSENT_SCOPE_LEN;
/// Host string ceiling: 253 bytes (intrinsic DNS name maximum).
pub const MAX_LOCAL_HOST_LEN: usize = 253;
/// Read chunk size for socket draining (intrinsic I/O buffer choice).
const LOCAL_READ_CHUNK: usize = 4 * 1024;
/// Default connect timeout in ms (policy, bounded by `MP-8`).
pub const DEFAULT_LOCAL_CONNECT_TIMEOUT_MS: u64 = 2_000;
/// Default read timeout in ms (policy, equals runtime default request timeout).
pub const DEFAULT_LOCAL_READ_TIMEOUT_MS: u64 = 5_000;
/// Default request target (Ollama also serves OpenAI-compat chat here).
pub const DEFAULT_LOCAL_PATH: &str = "/v1/chat/completions";
/// Default provider id (`MP-2` shape).
pub const DEFAULT_LOCAL_PROVIDER_ID: &str = "local-ollama";

/// Localhost-only endpoint configuration.
///
/// The optional `api_key` is caller-supplied only (for key-bearing
/// OpenAI-compatible endpoints; local Ollama needs none). It is sent as
/// `Authorization: Bearer` when present, and is never logged, never included
/// in `Debug`/`Display`, and never carried in any error reason.
#[derive(Clone)]
pub struct LocalEndpoint {
    host: String,
    port: u16,
    model: String,
    path: String,
    provider_id: String,
    connect_timeout_ms: u64,
    read_timeout_ms: u64,
    api_key: Option<String>,
}

impl LocalEndpoint {
    /// Build a localhost-only endpoint with defaults
    /// (`path=/v1/chat/completions`, provider id `local-ollama`,
    /// connect 2s / read 5s, no key).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::InvalidProviderId`] for a malformed default
    /// provider id (unreachable) and [`ProviderError::Transport`] for a
    /// non-loopback host, bad port, bad model name, or (via builders) bad
    /// path/timeouts/key. [`ProviderError::TimeoutTooLarge`] when a timeout
    /// exceeds `MAX_REQUEST_TIMEOUT_MS`.
    pub fn new(
        host: impl Into<String>,
        port: u16,
        model: impl Into<String>,
    ) -> Result<Self, ProviderError> {
        let endpoint = Self {
            host: host.into(),
            port,
            model: model.into(),
            path: DEFAULT_LOCAL_PATH.to_owned(),
            provider_id: DEFAULT_LOCAL_PROVIDER_ID.to_owned(),
            connect_timeout_ms: DEFAULT_LOCAL_CONNECT_TIMEOUT_MS,
            read_timeout_ms: DEFAULT_LOCAL_READ_TIMEOUT_MS,
            api_key: None,
        };
        endpoint.validate()?;
        Ok(endpoint)
    }

    /// Override the request target (must start with `/`, no CRLF).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::Transport`] for a malformed path.
    pub fn with_path(mut self, path: impl Into<String>) -> Result<Self, ProviderError> {
        self.path = path.into();
        self.validate()?;
        Ok(self)
    }

    /// Override the provider id (`MP-2` shape).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::InvalidProviderId`] for a malformed id.
    pub fn with_provider_id(mut self, id: impl Into<String>) -> Result<Self, ProviderError> {
        self.provider_id = id.into();
        self.validate()?;
        Ok(self)
    }

    /// Override connect/read timeouts (both `1..=MAX_REQUEST_TIMEOUT_MS`).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::Transport`] for a zero timeout and
    /// [`ProviderError::TimeoutTooLarge`] above the `MP-8` ceiling.
    pub fn with_timeouts(
        mut self,
        connect_timeout_ms: u64,
        read_timeout_ms: u64,
    ) -> Result<Self, ProviderError> {
        self.connect_timeout_ms = connect_timeout_ms;
        self.read_timeout_ms = read_timeout_ms;
        self.validate()?;
        Ok(self)
    }

    /// Attach a caller-supplied API key (never logged, never in errors).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::Transport`] when the key is empty,
    /// over-large (>4 KiB), or carries CR/LF (header-injection refusal).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Result<Self, ProviderError> {
        let key = key.into();
        if key.is_empty() || key.len() > 4 * 1024 {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "api key length out of bounds".to_owned(),
            });
        }
        if key.bytes().any(|b| b == b'\r' || b == b'\n') {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "api key carries CR/LF".to_owned(),
            });
        }
        self.api_key = Some(key);
        Ok(self)
    }

    /// Endpoint host (bare, localhost-only).
    #[must_use]
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Endpoint port.
    #[must_use]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Configured model name.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Request target path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Provider id for the [`ModelProvider`] seam.
    #[must_use]
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// Whether a caller-supplied key is attached (never exposes the key).
    #[must_use]
    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    fn validate(&self) -> Result<(), ProviderError> {
        validate_provider_id(&self.provider_id)?;
        if self.host.is_empty() || self.host.len() > MAX_LOCAL_HOST_LEN {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "host length out of bounds".to_owned(),
            });
        }
        if self.host.bytes().any(|b| b == b'\r' || b == b'\n') {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "host carries CR/LF".to_owned(),
            });
        }
        if self.host.contains("://") {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "host must be bare (no scheme); plain HTTP only".to_owned(),
            });
        }
        if !is_loopback_host(&self.host) {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "non-loopback host refused (localhost-only)".to_owned(),
            });
        }
        if self.port == 0 {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "port 0 refused".to_owned(),
            });
        }
        if validate_model_name(&self.model).is_err() {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "invalid model name".to_owned(),
            });
        }
        if self.model.bytes().any(|b| b == b'\r' || b == b'\n') {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "model carries CR/LF".to_owned(),
            });
        }
        if self.path.is_empty()
            || self.path.len() > MAX_LOCAL_PATH_LEN
            || !self.path.starts_with('/')
            || self.path.bytes().any(|b| {
                b == b'\r'
                    || b == b'\n'
                    || b == b' '
                    || b == b'"'
                    || b == b'<'
                    || b == b'>'
                    || b == b'\\'
                    || b == b'^'
                    || b == b'{'
                    || b == b'|'
                    || b == b'}'
            })
        {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "invalid request path".to_owned(),
            });
        }
        for (label, value) in [
            ("connect", self.connect_timeout_ms),
            ("read", self.read_timeout_ms),
        ] {
            if value == 0 {
                return Err(ProviderError::Transport {
                    provider: self.provider_id.clone(),
                    reason: format!("{label} timeout must be non-zero"),
                });
            }
            if value > MAX_REQUEST_TIMEOUT_MS {
                return Err(ProviderError::TimeoutTooLarge {
                    max: MAX_REQUEST_TIMEOUT_MS,
                    actual: value,
                });
            }
        }
        Ok(())
    }

    fn socket_addr(&self) -> Result<SocketAddr, ProviderError> {
        let ip: IpAddr = if self.host.eq_ignore_ascii_case("localhost") {
            "127.0.0.1".parse().map_err(|_| ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "localhost resolution failed".to_owned(),
            })?
        } else {
            let bare = self
                .host
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'));
            let literal = bare.unwrap_or(&self.host);
            literal.parse().map_err(|_| ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "non-loopback host refused (localhost-only)".to_owned(),
            })?
        };
        if !ip.is_loopback() {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "non-loopback host refused (localhost-only)".to_owned(),
            });
        }
        Ok(SocketAddr::new(ip, self.port))
    }
}

impl fmt::Debug for LocalEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalEndpoint")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("model", &self.model)
            .field("path", &self.path)
            .field("provider_id", &self.provider_id)
            .field("connect_timeout_ms", &self.connect_timeout_ms)
            .field("read_timeout_ms", &self.read_timeout_ms)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .finish()
    }
}

/// Whether `host` is an accepted loopback literal.
///
/// Accepts `localhost` (any ASCII case), `127.0.0.0/8`, and `::1` (with or
/// without brackets). Everything else — including other DNS names,
/// `0.0.0.0`, and non-loopback literals — is refused.
fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let bare = host
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    false
}

/// Experimental localhost-only [`ModelProvider`] over raw HTTP/1.1.
///
/// Std-only (`TcpStream`); zero new dependencies. See the module docs for
/// the localhost, timeout, bound, and error-mapping contract.
#[derive(Debug, Clone)]
pub struct LocalProvider {
    endpoint: LocalEndpoint,
    models: Vec<ModelDescriptor>,
    complete_calls: u64,
}

impl LocalProvider {
    /// Build a provider from a validated [`LocalEndpoint`].
    #[must_use]
    pub fn new(endpoint: LocalEndpoint) -> Self {
        let models = vec![ModelDescriptor {
            name: endpoint.model.clone(),
            capabilities: vec![ModelCapability::Text],
        }];
        Self {
            endpoint,
            models,
            complete_calls: 0,
        }
    }

    /// Borrow the endpoint configuration.
    #[must_use]
    pub fn endpoint(&self) -> &LocalEndpoint {
        &self.endpoint
    }
}

impl ModelProvider for LocalProvider {
    fn provider_id(&self) -> &str {
        self.endpoint.provider_id()
    }

    fn list_models(&self) -> Vec<ModelDescriptor> {
        self.models.clone()
    }

    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError> {
        if request.timeout_ms > MAX_REQUEST_TIMEOUT_MS {
            return Err(ProviderError::TimeoutTooLarge {
                max: MAX_REQUEST_TIMEOUT_MS,
                actual: request.timeout_ms,
            });
        }
        if request.model != self.endpoint.model {
            return Err(ProviderError::UnknownModel {
                name: request.model.clone(),
            });
        }
        let actual = request.total_message_bytes();
        if actual > request.budget_bytes {
            return Err(ProviderError::BudgetExceeded {
                limit: request.budget_bytes,
                actual,
            });
        }
        let body = build_chat_body(&self.endpoint.model, &request.messages);
        if body.len() > MAX_LOCAL_REQUEST_BYTES {
            return Err(ProviderError::Transport {
                provider: self.provider_id().to_owned(),
                reason: format!("request over-large (max {MAX_LOCAL_REQUEST_BYTES})"),
            });
        }
        let text = http_round_trip(&self.endpoint, &body, request.timeout_ms)?;
        self.complete_calls += 1;
        Ok(ProviderTurn {
            text,
            tool_calls: Vec::new(),
            latency_ms: 0,
        })
    }

    fn scripted_turns_remaining(&self) -> usize {
        0
    }

    fn complete_calls(&self) -> u64 {
        self.complete_calls
    }
}

/// Escape a string as a JSON string body (without surrounding quotes).
fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    for c in input.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "user",
    }
}

/// Build the minimal OpenAI-compatible chat body.
///
/// `Role::Tool` observations are folded into `user` messages with a `[tool] `
/// prefix so no OpenAI tool protocol is required; the untrusted surface stays
/// labeled in the text itself.
fn build_chat_body(model: &str, messages: &[bitty_ai_runtime::provider::Message]) -> Vec<u8> {
    let mut out = String::from("{\"model\":\"");
    out.push_str(&json_escape(model));
    out.push_str("\",\"messages\":[");
    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"role\":\"");
        out.push_str(role_name(&message.role));
        out.push_str("\",\"content\":\"");
        if matches!(message.role, Role::Tool) {
            out.push_str("[tool] ");
        }
        out.push_str(&json_escape(&message.content));
        out.push_str("\"}");
    }
    out.push_str("],\"stream\":false}");
    out.into_bytes()
}

/// One blocking HTTP/1.1 round-trip with mandatory timeouts and bounded reads.
fn http_round_trip(
    endpoint: &LocalEndpoint,
    body: &[u8],
    request_timeout_ms: u64,
) -> Result<String, ProviderError> {
    let provider = endpoint.provider_id().to_owned();
    let addr = endpoint.socket_addr()?;
    // Effective timeouts respect both the mandatory endpoint policy and the
    // caller deadline: the smaller of the two fires first.
    let connect_ms = endpoint
        .connect_timeout_ms
        .min(request_timeout_ms.max(1))
        .max(1);
    let read_ms = endpoint
        .read_timeout_ms
        .min(request_timeout_ms.max(1))
        .max(1);
    let mut stream =
        TcpStream::connect_timeout(&addr, Duration::from_millis(connect_ms)).map_err(|error| {
            if error.kind() == std::io::ErrorKind::TimedOut {
                ProviderError::Timeout {
                    timeout_ms: connect_ms,
                    latency_ms: connect_ms.saturating_add(1),
                }
            } else {
                ProviderError::Transport {
                    provider: provider.clone(),
                    reason: format!("connect failed: {}", short_io_kind(&error)),
                }
            }
        })?;
    stream
        .set_read_timeout(Some(Duration::from_millis(read_ms)))
        .map_err(|error| ProviderError::Transport {
            provider: provider.clone(),
            reason: format!("read timeout arm failed: {}", short_io_kind(&error)),
        })?;
    // `set_write_timeout` is best-effort hardening; a missing write timeout
    // still leaves the read timeout as the binding deadline.
    let _ = stream.set_write_timeout(Some(Duration::from_millis(connect_ms)));

    let mut head = format!(
        "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        endpoint.path,
        endpoint.host,
        endpoint.port,
        body.len()
    );
    if let Some(key) = endpoint.api_key.as_ref() {
        head.push_str("Authorization: Bearer ");
        head.push_str(key);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream
        .write_all(head.as_bytes())
        .map_err(|error| map_write_error(&provider, &error, connect_ms, body.len(), false))?;
    stream
        .write_all(body)
        .map_err(|error| map_write_error(&provider, &error, connect_ms, body.len(), true))?;

    let raw = read_bounded(&mut stream, &provider, read_ms)?;
    let (status, response_body) = split_response(&raw, &provider)?;
    check_status(endpoint, status, &raw, &provider)?;
    if response_body.len() > MAX_LOCAL_RESPONSE_BYTES {
        return Err(ProviderError::Transport {
            provider,
            reason: format!("response over-large (max {MAX_LOCAL_RESPONSE_BYTES})"),
        });
    }
    extract_assistant_text(response_body, &provider)
}

fn short_io_kind(error: &std::io::Error) -> String {
    let kind = match error.kind() {
        std::io::ErrorKind::ConnectionRefused => "connection refused",
        std::io::ErrorKind::TimedOut => "timed out",
        std::io::ErrorKind::WouldBlock => "would block",
        std::io::ErrorKind::NotFound => "not found",
        std::io::ErrorKind::PermissionDenied => "permission denied",
        std::io::ErrorKind::ConnectionReset => "connection reset",
        std::io::ErrorKind::ConnectionAborted => "connection aborted",
        std::io::ErrorKind::UnexpectedEof => "unexpected EOF",
        _ => "I/O error",
    };
    kind.to_owned()
}

fn is_timeout_kind(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    )
}

/// Map a write failure: timeouts become `Timeout`; a failure after partial
/// body bytes is `Unknown` (the server may have the request); anything else
/// is `Transport`. The key and body bytes never enter the reason.
fn map_write_error(
    provider: &str,
    error: &std::io::Error,
    connect_ms: u64,
    body_len: usize,
    body_started: bool,
) -> ProviderError {
    if is_timeout_kind(error) {
        return ProviderError::Timeout {
            timeout_ms: connect_ms,
            latency_ms: connect_ms.saturating_add(1),
        };
    }
    if body_started && body_len > 0 && is_uncertain_kind(error) {
        return ProviderError::Unknown {
            provider: provider.to_owned(),
            reason: format!("request write failed after send: {}", short_io_kind(error)),
        };
    }
    ProviderError::Transport {
        provider: provider.to_owned(),
        reason: format!("request write failed: {}", short_io_kind(error)),
    }
}

fn is_uncertain_kind(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::UnexpectedEof
    )
}

/// Read until EOF (server closes, we sent `Connection: close`) bounded by
/// headers + body ceilings. Timeouts become `Timeout`; mid-body I/O
/// failures become `Unknown`; over-large becomes `Transport`.
fn read_bounded(
    stream: &mut TcpStream,
    provider: &str,
    read_ms: u64,
) -> Result<Vec<u8>, ProviderError> {
    let cap = MAX_LOCAL_HEADERS_BYTES + MAX_LOCAL_RESPONSE_BYTES + 512;
    let mut raw = Vec::new();
    let mut chunk = [0_u8; LOCAL_READ_CHUNK];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                if raw.len() + n > cap {
                    return Err(ProviderError::Transport {
                        provider: provider.to_owned(),
                        reason: format!("response over-large (max {MAX_LOCAL_RESPONSE_BYTES})"),
                    });
                }
                raw.extend_from_slice(&chunk[..n]);
            }
            Err(error) if is_timeout_kind(&error) => {
                return Err(ProviderError::Timeout {
                    timeout_ms: read_ms,
                    latency_ms: read_ms.saturating_add(1),
                });
            }
            Err(error) => {
                if raw.is_empty() {
                    return Err(ProviderError::Transport {
                        provider: provider.to_owned(),
                        reason: format!("response read failed: {}", short_io_kind(&error)),
                    });
                }
                return Err(ProviderError::Unknown {
                    provider: provider.to_owned(),
                    reason: format!("response truncated: {}", short_io_kind(&error)),
                });
            }
        }
    }
    if raw.is_empty() {
        return Err(ProviderError::Transport {
            provider: provider.to_owned(),
            reason: "empty response".to_owned(),
        });
    }
    Ok(raw)
}

struct HttpStatus {
    code: u16,
}

fn split_response<'a>(
    raw: &'a [u8],
    provider: &str,
) -> Result<(HttpStatus, &'a [u8]), ProviderError> {
    let text = std::str::from_utf8(raw).map_err(|_| ProviderError::Transport {
        provider: provider.to_owned(),
        reason: "response is not UTF-8".to_owned(),
    })?;
    let split = text.find("\r\n\r\n").ok_or_else(|| {
        // Headers never completed: the server closed early or the block hit
        // the header ceiling without a terminator.
        if raw.len() >= MAX_LOCAL_HEADERS_BYTES {
            ProviderError::Transport {
                provider: provider.to_owned(),
                reason: "response headers over-large".to_owned(),
            }
        } else {
            ProviderError::Unknown {
                provider: provider.to_owned(),
                reason: "response truncated before headers".to_owned(),
            }
        }
    })?;
    let header_block = &text[..split];
    // Byte offset of the body equals the string offset here because the
    // separator is ASCII and `text` borrows `raw`.
    let body = &raw[split + 4..];
    let status_line = header_block
        .lines()
        .next()
        .ok_or_else(|| ProviderError::Transport {
            provider: provider.to_owned(),
            reason: "missing status line".to_owned(),
        })?;
    let code = parse_status_code(status_line).ok_or_else(|| ProviderError::Transport {
        provider: provider.to_owned(),
        reason: "malformed status line".to_owned(),
    })?;
    validate_headers(header_block, body, provider)?;
    Ok((HttpStatus { code }, body))
}

fn parse_status_code(status_line: &str) -> Option<u16> {
    let mut parts = status_line.split(' ');
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    parts.next()?.parse().ok()
}

/// Validate headers: reject chunked encoding, enforce `Content-Length`
/// agreement and the body ceiling. Length mismatch with a short body is
/// `Unknown` (truncated); anything oversized is `Transport`.
fn validate_headers(header_block: &str, body: &[u8], provider: &str) -> Result<(), ProviderError> {
    let mut content_length: Option<usize> = None;
    for line in header_block.lines().skip(1) {
        if let Some(value) = header_value(line, "content-length") {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|_| ProviderError::Transport {
                    provider: provider.to_owned(),
                    reason: "malformed content-length".to_owned(),
                })?;
            content_length = Some(parsed);
        }
        if let Some(value) = header_value(line, "transfer-encoding") {
            if value.to_ascii_lowercase().contains("chunked") {
                return Err(ProviderError::Transport {
                    provider: provider.to_owned(),
                    reason: "chunked encoding unsupported".to_owned(),
                });
            }
        }
    }
    if let Some(declared) = content_length {
        if declared > MAX_LOCAL_RESPONSE_BYTES {
            return Err(ProviderError::Transport {
                provider: provider.to_owned(),
                reason: format!("response over-large (max {MAX_LOCAL_RESPONSE_BYTES})"),
            });
        }
        if body.len() < declared {
            return Err(ProviderError::Unknown {
                provider: provider.to_owned(),
                reason: "response truncated (short body)".to_owned(),
            });
        }
        if body.len() > declared {
            return Err(ProviderError::Transport {
                provider: provider.to_owned(),
                reason: "response body longer than declared".to_owned(),
            });
        }
    }
    Ok(())
}

fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (key, value) = line.split_once(':')?;
    if key.trim().eq_ignore_ascii_case(name) {
        Some(value.trim())
    } else {
        None
    }
}

/// Map the HTTP status onto existing provider variants (no new kinds).
fn check_status(
    endpoint: &LocalEndpoint,
    status: HttpStatus,
    raw: &[u8],
    provider: &str,
) -> Result<(), ProviderError> {
    match status.code {
        200..=299 => Ok(()),
        401 | 403 => Err(ProviderError::Auth {
            provider: provider.to_owned(),
            reason: format!("authorization refused: HTTP {}", status.code),
        }),
        429 => Err(ProviderError::RateLimited {
            provider: provider.to_owned(),
            retry_after_ms: retry_after_ms(raw),
        }),
        404 => Err(ProviderError::ModelUnavailable {
            provider: provider.to_owned(),
            model: endpoint.model.clone(),
        }),
        code => Err(ProviderError::Transport {
            provider: provider.to_owned(),
            reason: format!("HTTP {code}"),
        }),
    }
}

/// Extract `Retry-After` (seconds, per HTTP) converted to ms when present.
fn retry_after_ms(raw: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(raw).ok()?;
    let end = text.find("\r\n\r\n")?;
    for line in text[..end].lines().skip(1) {
        if let Some(value) = header_value(line, "retry-after") {
            let seconds: u64 = value.trim().parse().ok()?;
            return Some(seconds.saturating_mul(1_000));
        }
    }
    None
}

/// Extract the assistant text: first `"content"` (OpenAI chat / Ollama
/// chat), else `"response"` (Ollama generate). Strictly fail-closed.
fn extract_assistant_text(body: &[u8], provider: &str) -> Result<String, ProviderError> {
    let text = std::str::from_utf8(body).map_err(|_| ProviderError::Transport {
        provider: provider.to_owned(),
        reason: "response body is not UTF-8".to_owned(),
    })?;
    if let Some(value) = find_key_string(text, "content") {
        return Ok(value);
    }
    if let Some(value) = find_key_string(text, "response") {
        return Ok(value);
    }
    Err(ProviderError::Transport {
        provider: provider.to_owned(),
        reason: "response missing content/response".to_owned(),
    })
}

/// Find `"key": "string"` and parse the JSON string value.
fn find_key_string(text: &str, key: &str) -> Option<String> {
    let quoted = format!("\"{key}\"");
    let mut search_from = 0;
    while let Some(hit) = text[search_from..].find(&quoted) {
        let mut cursor = search_from + hit + quoted.len();
        cursor = skip_ws(text, cursor)?;
        if text.as_bytes().get(cursor) != Some(&b':') {
            search_from = cursor.min(text.len());
            continue;
        }
        cursor += 1;
        cursor = skip_ws(text, cursor)?;
        if text.as_bytes().get(cursor) != Some(&b'"') {
            search_from = cursor.min(text.len());
            continue;
        }
        // First well-formed occurrence wins; malformed JSON fails closed at
        // the call site (the caller maps `None` to `Transport`).
        let (value, _) = parse_json_string(text, cursor)?;
        return Some(value);
    }
    None
}

fn skip_ws(text: &str, mut cursor: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    while cursor < bytes.len() && matches!(bytes[cursor], b' ' | b'\t' | b'\n' | b'\r') {
        cursor += 1;
    }
    if cursor <= bytes.len() {
        Some(cursor)
    } else {
        None
    }
}

/// Parse a JSON string starting at the opening quote.
///
/// Returns the unescaped value and the byte index just past the closing
/// quote. Fails closed (`None`) on unterminated strings, bad escapes,
/// invalid `\u`, or raw control characters.
fn parse_json_string(text: &str, open: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut index = open + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return Some((out, index + 1)),
            b'\\' => {
                index += 1;
                let escaped = *bytes.get(index)?;
                match escaped {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if index + 4 >= bytes.len() {
                            return None;
                        }
                        let hex = &text[index + 1..index + 5];
                        let unit = u32::from_str_radix(hex, 16).ok()?;
                        // Reject surrogates without pair handling: fail
                        // closed rather than emit replacement characters.
                        let ch = char::from_u32(unit)?;
                        out.push(ch);
                        index += 4;
                    }
                    _ => return None,
                }
                index += 1;
            }
            byte if byte < 0x20 => return None,
            _ => {
                let ch = text[index..].chars().next()?;
                out.push(ch);
                index += ch.len_utf8();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use bitty_ai_runtime::provider::Message;
    use bitty_ai_runtime::selection::{
        ModelRegistration, ProviderRegistry, SelectRequest, SelectedModel,
    };
    use bitty_ai_runtime::{
        Agent, AgentConfig, AgentSession, FakeToolExecutor, IdIssuer, ModelCapability, ToolBus,
        VecSink,
    };

    use crate::harness::{AllowReadOnly, test_tool_registry};

    fn turn_request(model: &str, text: &str) -> TurnRequest {
        TurnRequest {
            model: model.to_owned(),
            messages: vec![Message::user(text)],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: 32 * 1024,
            timeout_ms: 5_000,
            now_ms: 1_000,
        }
    }

    fn endpoint_for(port: u16) -> LocalEndpoint {
        LocalEndpoint::new("127.0.0.1", port, "llama3.1:8b").expect("endpoint")
    }

    /// Spawn a one-shot stub: accept one connection, optionally wait, then
    /// write the scripted bytes and close. Returns the bound port.
    fn stub_once(response: Vec<u8>, delay_before_write_ms: u64) -> (u16, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let port = listener.local_addr().expect("port").port();
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let handle = thread::spawn(move || {
            let _ = ready_tx.send(());
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_millis(2_000)));
            // Drain the request (bounded): read until end of headers plus
            // any declared body so the client write never blocks.
            let mut buf = [0_u8; 4096];
            let mut seen = Vec::new();
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        seen.extend_from_slice(&buf[..n]);
                        if let Some(end) = find_headers_end(&seen) {
                            let body_len =
                                content_length_of(&seen[..end]).unwrap_or(0).min(96 * 1024);
                            if seen.len() >= end + body_len {
                                break;
                            }
                        }
                        if seen.len() > 128 * 1024 {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            if delay_before_write_ms > 0 {
                thread::sleep(Duration::from_millis(delay_before_write_ms));
            }
            let _ = stream.write_all(&response);
            let _ = stream.flush();
        });
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("stub ready");
        (port, handle)
    }

    fn find_headers_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|pos| pos + 4)
    }

    fn content_length_of(headers: &[u8]) -> Option<usize> {
        let text = std::str::from_utf8(headers).ok()?;
        for line in text.lines().skip(1) {
            if let Some(value) = header_value(line, "content-length") {
                return value.trim().parse().ok();
            }
        }
        None
    }

    fn http_ok(json: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
            json.len()
        )
        .into_bytes()
    }

    fn http_status(status: &str, json: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
            json.len()
        )
        .into_bytes()
    }

    /// Spawn a one-shot stub that captures the raw request bytes for header
    /// assertions, then writes the scripted response and closes.
    ///
    /// Returns the bound port, the join handle, and the captured request
    /// receiver. Read-only test helper; no product path uses it.
    fn stub_capture_once(
        response: Vec<u8>,
    ) -> (u16, thread::JoinHandle<()>, mpsc::Receiver<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let port = listener.local_addr().expect("port").port();
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let (req_tx, req_rx) = mpsc::channel::<Vec<u8>>();
        let handle = thread::spawn(move || {
            let _ = ready_tx.send(());
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_millis(2_000)));
            // Drain the request (bounded): read until end of headers plus
            // any declared body so the client write never blocks.
            let mut buf = [0_u8; 4096];
            let mut seen = Vec::new();
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        seen.extend_from_slice(&buf[..n]);
                        if let Some(end) = find_headers_end(&seen) {
                            let body_len =
                                content_length_of(&seen[..end]).unwrap_or(0).min(96 * 1024);
                            if seen.len() >= end + body_len {
                                break;
                            }
                        }
                        if seen.len() > 128 * 1024 {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = req_tx.send(seen);
            let _ = stream.write_all(&response);
            let _ = stream.flush();
        });
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("stub ready");
        (port, handle, req_rx)
    }

    #[test]
    fn localhost_variants_accepted_elsewhere_refused_at_construction() {
        for host in [
            "127.0.0.1",
            "127.0.0.2",
            "127.1.2.3",
            "::1",
            "[::1]",
            "localhost",
        ] {
            assert!(
                LocalEndpoint::new(host, 11434, "llama3.1:8b").is_ok(),
                "loopback host refused: {host}"
            );
        }
        for host in [
            "8.8.8.8",
            "0.0.0.0",
            "example.com",
            "10.0.0.1",
            "192.168.1.1",
            "http://127.0.0.1",
            "[::2]",
            "fe80::1",
            "",
        ] {
            let err = LocalEndpoint::new(host, 11434, "llama3.1:8b").expect_err("must refuse");
            assert!(
                matches!(err, ProviderError::Transport { .. }),
                "non-loopback must be Transport, got: {err}"
            );
        }
    }

    #[test]
    fn invalid_provider_id_is_typed() {
        let endpoint = LocalEndpoint::new("127.0.0.1", 11434, "llama3.1:8b")
            .expect("endpoint")
            .with_provider_id("BAD.UPPER");
        assert!(matches!(
            endpoint,
            Err(ProviderError::InvalidProviderId { .. })
        ));
    }

    #[test]
    fn debug_redacts_api_key() {
        let endpoint = LocalEndpoint::new("127.0.0.1", 11434, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key("test-key-probe-value")
            .expect("key");
        let rendered = format!("{endpoint:?}");
        assert!(
            !rendered.contains("test-key-probe-value"),
            "key leaked in Debug"
        );
        assert!(rendered.contains("[redacted]"));
        assert!(endpoint.has_api_key());
    }

    #[test]
    fn api_key_emits_exact_bearer_header_on_wire() {
        // AI-0043 (1): a caller-supplied key must appear byte-exact as
        // `Authorization: Bearer <key>` on the wire; the turn still completes.
        let key = "Ak3y-Wire-Probe-9z8y7x6w5v-0043a";
        let json = r#"{"choices":[{"message":{"content":"keyed stub"}}]}"#;
        let (port, handle, req_rx) = stub_capture_once(http_ok(json));
        let endpoint = LocalEndpoint::new("127.0.0.1", port, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key(key)
            .expect("key");
        assert!(endpoint.has_api_key());
        let mut provider = LocalProvider::new(endpoint);
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "keyed stub");
        assert_eq!(provider.complete_calls(), 1);
        let raw = req_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("request bytes");
        let text = String::from_utf8_lossy(&raw);
        let expected = format!("Authorization: Bearer {key}\r\n");
        assert!(
            text.contains(&expected),
            "wire bytes must carry exact Bearer line"
        );
        handle.join().expect("stub");
    }

    #[test]
    fn no_api_key_sends_no_authorization_header() {
        // AI-0043 (2): without a key, no Authorization header may appear
        // anywhere in the request bytes; the turn still completes.
        let json = r#"{"choices":[{"message":{"content":"open stub"}}]}"#;
        let (port, handle, req_rx) = stub_capture_once(http_ok(json));
        let endpoint = endpoint_for(port);
        assert!(!endpoint.has_api_key());
        let mut provider = LocalProvider::new(endpoint);
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "open stub");
        assert_eq!(provider.complete_calls(), 1);
        let raw = req_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("request bytes");
        let text = String::from_utf8_lossy(&raw);
        assert!(
            !text.to_ascii_lowercase().contains("authorization"),
            "wire bytes must not carry Authorization without a key"
        );
        handle.join().expect("stub");
    }

    #[test]
    fn empty_api_key_refused_without_io() {
        // AI-0043 (3a): an empty key fails closed at construction with a
        // typed Transport error. No stub is bound in this test, so zero
        // sockets can open; a fresh provider that never ran stays at zero
        // calls.
        let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key("")
            .expect_err("empty key must fail");
        assert!(
            matches!(err, ProviderError::Transport { .. }),
            "empty key must be Transport, got: {err}"
        );
        let probe = LocalProvider::new(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
        );
        assert_eq!(probe.complete_calls(), 0);
    }

    #[test]
    fn over_large_api_key_refused_without_io() {
        // AI-0043 (3b): keys longer than 4 KiB fail closed at construction.
        // The 4 KiB boundary itself still holds. No stub is bound, so zero
        // sockets open.
        let boundary = "Q".repeat(4 * 1024);
        assert!(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
                .expect("endpoint")
                .with_api_key(boundary)
                .is_ok(),
            "4 KiB key must be accepted"
        );
        let big = "Q".repeat(4 * 1024 + 1);
        let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key(big.clone())
            .expect_err("over-large key must fail");
        assert!(
            matches!(err, ProviderError::Transport { .. }),
            "over-large key must be Transport, got: {err}"
        );
        // Redaction holds on this path too (full value and 8-byte prefix).
        let rendered = err.to_string();
        assert!(!rendered.contains(&big));
        assert!(!rendered.contains(&big[..8]));
        let probe = LocalProvider::new(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
        );
        assert_eq!(probe.complete_calls(), 0);
    }

    #[test]
    fn crlf_api_keys_refused_without_io() {
        // AI-0043 (3c): CR- and LF-bearing keys fail closed at construction
        // (header-injection refusal). No stub is bound, so zero sockets open.
        for bad in [
            "Ak3y-CR-Probe-0043\rx",
            "Ak3y-LF-Probe-0043\nx",
            "Ak3y-CRLF-Probe-0043\r\nx",
        ] {
            let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
                .expect("endpoint")
                .with_api_key(bad)
                .expect_err("CRLF key must fail");
            assert!(
                matches!(err, ProviderError::Transport { .. }),
                "CRLF key must be Transport, got: {err}"
            );
            // The refusal reason must not echo the key material.
            let rendered = err.to_string();
            assert!(!rendered.contains(bad));
        }
        let probe = LocalProvider::new(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
        );
        assert_eq!(probe.complete_calls(), 0);
    }

    #[test]
    fn api_key_absent_from_debug_and_refusal_reasons_including_prefix() {
        // AI-0043 (4): Debug and every new-path error reason carry neither
        // the key nor a non-trivial prefix (first 8 / first 4 bytes) to catch
        // truncation leaks. Single-char prefixes are not asserted: they occur
        // naturally in unrelated fields.
        let key = "Zk9q-Redact-Probe-7m6n5b4v-0043r";
        let endpoint = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key(key)
            .expect("key");
        let rendered = format!("{endpoint:?}");
        assert!(!rendered.contains(key));
        assert!(!rendered.contains(&key[..8]));
        assert!(!rendered.contains(&key[..4]));
        assert!(rendered.contains("[redacted]"));

        // Refusal reasons from the new paths redact the same way.
        let over = "Zk9q-".repeat(820);
        let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key(over.clone())
            .expect_err("over-large must fail");
        let text = err.to_string();
        assert!(!text.contains(&over));
        assert!(!text.contains(&over[..8]));
        assert!(!text.contains(&over[..4]));

        for bad in ["Zk9q-CR-Probe-0043\rx", "Zk9q-LF-Probe-0043\nx"] {
            let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
                .expect("endpoint")
                .with_api_key(bad)
                .expect_err("CRLF must fail");
            let text = err.to_string();
            assert!(!text.contains(bad));
            assert!(!text.contains(&bad[..8]));
            assert!(!text.contains(&bad[..4]));
        }

        // The empty-key refusal carries only a static reason.
        let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key("")
            .expect_err("empty must fail");
        let text = err.to_string();
        assert!(matches!(err, ProviderError::Transport { .. }));
        assert!(text.contains("api key length out of bounds"));
    }

    #[test]
    fn openai_chat_content_round_trip() {
        let json = r#"{"id":"chatcmpl-x","choices":[{"message":{"role":"assistant","content":"hello from stub"}}]}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        assert_eq!(provider.provider_id(), "local-ollama");
        assert_eq!(provider.scripted_turns_remaining(), 0);
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "hello from stub");
        assert!(turn.tool_calls.is_empty());
        assert_eq!(provider.complete_calls(), 1);
        handle.join().expect("stub");
    }

    #[test]
    fn ollama_generate_response_fallback() {
        let json = r#"{"model":"llama3.1:8b","response":"ollama text","done":true}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "ollama text");
        handle.join().expect("stub");
    }

    #[test]
    fn escaped_content_unescapes() {
        let json = r#"{"choices":[{"message":{"content":"a \"quoted\"\nline"}}]}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "a \"quoted\"\nline");
        handle.join().expect("stub");
    }

    #[test]
    fn malformed_json_fails_closed_as_transport() {
        let (port, handle) = stub_once(http_ok("{not json"), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("must fail");
        assert!(matches!(err, ProviderError::Transport { .. }), "got: {err}");
        assert_eq!(provider.complete_calls(), 0);
        handle.join().expect("stub");
    }

    #[test]
    fn missing_content_fails_closed() {
        let (port, handle) = stub_once(http_ok(r#"{"choices":[]}"#), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("must fail");
        assert!(matches!(err, ProviderError::Transport { .. }), "got: {err}");
        handle.join().expect("stub");
    }

    #[test]
    fn over_large_response_fails_closed() {
        let big = "x".repeat(MAX_LOCAL_RESPONSE_BYTES + 1);
        let json = format!("{{\"choices\":[{{\"message\":{{\"content\":\"{big}\"}}}}]}}");
        let (port, handle) = stub_once(http_ok(&json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("must fail");
        assert!(matches!(err, ProviderError::Transport { .. }), "got: {err}");
        handle.join().expect("stub");
    }

    #[test]
    fn http_status_mapping() {
        for (status, expect) in [
            ("500 Internal Server Error", "transport"),
            ("401 Unauthorized", "auth"),
            ("403 Forbidden", "auth"),
            ("429 Too Many Requests", "rate"),
            ("404 Not Found", "unavailable"),
        ] {
            let body = r#"{"error":"nope"}"#;
            let raw = if status.starts_with("429") {
                format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: 2\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_bytes()
            } else {
                http_status(status, body)
            };
            let (port, handle) = stub_once(raw, 0);
            let mut provider = LocalProvider::new(endpoint_for(port));
            let err = provider
                .complete(&turn_request("llama3.1:8b", "hi"))
                .expect_err("must fail");
            match expect {
                "transport" => assert!(
                    matches!(err, ProviderError::Transport { .. }),
                    "status {status}: got {err}"
                ),
                "auth" => {
                    assert!(
                        matches!(err, ProviderError::Auth { .. }),
                        "status {status}: got {err}"
                    );
                    assert!(!err.to_string().contains("test-key-probe"));
                }
                "rate" => match err {
                    ProviderError::RateLimited { retry_after_ms, .. } => {
                        assert_eq!(retry_after_ms, Some(2_000));
                    }
                    other => panic!("status {status}: got {other}"),
                },
                "unavailable" => assert!(
                    matches!(err, ProviderError::ModelUnavailable { .. }),
                    "status {status}: got {err}"
                ),
                _ => unreachable!(),
            }
            handle.join().expect("stub");
        }
    }

    #[test]
    fn truncated_body_is_unknown() {
        let body = r#"{"choices":[{"message":{"content":"hi"}}]}"#;
        let raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len() + 100
        )
        .into_bytes();
        let (port, handle) = stub_once(raw, 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("must fail");
        assert!(matches!(err, ProviderError::Unknown { .. }), "got: {err}");
        handle.join().expect("stub");
    }

    #[test]
    fn read_timeout_maps_to_timeout() {
        let json = r#"{"choices":[{"message":{"content":"late"}}]}"#;
        let (port, handle) = stub_once(http_ok(json), 600);
        let endpoint = LocalEndpoint::new("127.0.0.1", port, "llama3.1:8b")
            .expect("endpoint")
            .with_timeouts(500, 150)
            .expect("timeouts");
        let mut provider = LocalProvider::new(endpoint);
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("must time out");
        assert!(matches!(err, ProviderError::Timeout { .. }), "got: {err}");
        handle.join().expect("stub");
    }

    #[test]
    fn pre_io_checks_mirror_fake_provider() {
        // Pre-I/O checks fail before any socket is opened, so no stub is
        // needed (and none is created, avoiding an accept that never fires).
        let mut provider = LocalProvider::new(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
        );
        let unknown = TurnRequest {
            model: "no-such-model".to_owned(),
            ..turn_request("llama3.1:8b", "hi")
        };
        assert!(matches!(
            provider.complete(&unknown),
            Err(ProviderError::UnknownModel { .. })
        ));
        let over = TurnRequest {
            budget_bytes: 1,
            ..turn_request("llama3.1:8b", "hi")
        };
        assert!(matches!(
            provider.complete(&over),
            Err(ProviderError::BudgetExceeded { .. })
        ));
        let too_big = TurnRequest {
            timeout_ms: 60_000,
            ..turn_request("llama3.1:8b", "hi")
        };
        assert!(matches!(
            provider.complete(&too_big),
            Err(ProviderError::TimeoutTooLarge { .. })
        ));
        assert_eq!(provider.complete_calls(), 0);
    }

    #[test]
    fn zero_timeout_is_refused_at_construction() {
        // Zero timeouts are refused by `with_timeouts` before any socket is
        // opened, so no stub server is needed here.
        let zero_connect = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_timeouts(0, 100);
        assert!(matches!(zero_connect, Err(ProviderError::Transport { .. })));
        let zero_read = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
            .expect("endpoint")
            .with_timeouts(100, 0);
        assert!(matches!(zero_read, Err(ProviderError::Transport { .. })));
    }

    #[test]
    fn selection_to_provider_to_turn_wiring() {
        // Selection (AI-0030) picks the local candidate; the pick drives a
        // real `Agent::run_turn` through the local provider against the stub.
        let json = r#"{"choices":[{"message":{"content":"stub answer"}}]}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let endpoint = endpoint_for(port);
        let provider = LocalProvider::new(endpoint.clone());
        let descriptor = provider.list_models().pop().expect("descriptor");
        let mut registry = ProviderRegistry::new();
        registry
            .register(
                ModelRegistration::snapshot_from(endpoint.provider_id(), &descriptor, 4_096, 1, 1)
                    .expect("registration"),
            )
            .expect("register");
        let request = SelectRequest::capabilities(vec![ModelCapability::Text]);
        let chain = registry.select(&request).expect("select");
        assert_eq!(chain.len(), 1);
        let selected: &SelectedModel = &chain[0];
        assert_eq!(selected.provider_id, endpoint.provider_id());
        assert_eq!(selected.name, endpoint.model());

        let tools =
            ToolBus::new(test_tool_registry().expect("registry")).with_authorizer(AllowReadOnly);
        let session = AgentSession::new(
            IdIssuer::default().agent_instance(),
            IdIssuer::default().run(),
            IdIssuer::default().session(),
        );
        let config = AgentConfig {
            context_budget_bytes: 32 * 1024,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(provider, tools, session, config);
        let mut executor = FakeToolExecutor::new();
        let mut sink = VecSink::new();
        let outcome = agent.run_turn(
            &mut executor,
            &selected.name,
            "what did the stub say?",
            &[],
            &mut sink,
            1_000,
        );
        match outcome {
            bitty_ai_runtime::ExecOutcome::Completed { text } => {
                assert_eq!(text, "stub answer");
            }
            other => panic!("expected Completed, got: {other:?}"),
        }
        assert_eq!(agent.provider_mut().complete_calls(), 1);
        handle.join().expect("stub");
    }

    /// Live Ollama probe (manual only): requires `ollama serve` on
    /// 127.0.0.1:11434 with the named model pulled. Never runs in CI.
    #[test]
    #[ignore]
    fn live_ollama_probe() {
        let endpoint = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint");
        let mut provider = LocalProvider::new(endpoint);
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "say ok"))
            .expect("live turn");
        assert!(!turn.text.is_empty());
    }

    /// Live key-protected probe (manual only, AI-0043): requires a local
    /// key-bearing OpenAI-compatible server on loopback. Example server flags
    /// (llama.cpp): `llama-server --host 127.0.0.1 --port 8080 --model <gguf>
    /// --api-key test-live-key-0043`. The endpoint must use the same key via
    /// `.with_api_key("test-live-key-0043")`. Without the key the server
    /// answers 401 (mapped to `Auth`). Never runs in CI.
    #[test]
    #[ignore]
    fn live_key_protected_probe() {
        let endpoint = LocalEndpoint::new("127.0.0.1", 8080, "llama3.1:8b")
            .expect("endpoint")
            .with_api_key("test-live-key-0043")
            .expect("key");
        let mut provider = LocalProvider::new(endpoint);
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "say ok"))
            .expect("live turn");
        assert!(!turn.text.is_empty());
    }
}

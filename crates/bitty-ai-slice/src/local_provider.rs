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
//!   Read/write-timeout arming is mandatory too: a failed arm refuses rather
//!   than silently dropping the declared duration.
//! - One request deadline: the adapter owns a monotonic clock and arms a
//!   single deadline at adapter-now + `timeout_ms` for the whole request.
//!   Connect, write, and read each derive their remaining allowance at the
//!   moment they start (bounded by the endpoint policy), so no operation is
//!   granted time past the caller-declared deadline. A zero-duration request,
//!   or an operation starting with no remaining time, refuses before I/O.
//!   Socket timeouts bind the per-call allowance, not preemption: a
//!   drip-feeding peer can still hold a call to its granted allowance. The
//!   caller keeps its own logical time (`now_ms`), which never anchors the
//!   deadline; the adapter reads no wall clock and makes no latency or
//!   outage claim (the success turn keeps `latency_ms: 0`, meaning
//!   unreported).
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
//!   this minimal client; headers larger than this fail closed, including a
//!   completed header block).
//! - Response-parser ceilings: the bounded JSON walk enforces an explicit
//!   nesting depth ([`MAX_LOCAL_JSON_DEPTH`]) and an explicit work budget
//!   ([`MAX_LOCAL_PARSER_STEPS`]) in addition to the body-byte ceiling, with
//!   checked byte access and saturating offset arithmetic throughout, so
//!   malformed or hostile text yields a typed refusal, never a panic.
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
//! (`POST {path}` with `{"model","messages","stream":false}` plus any
//! declared, mapped sampling fields), which a local Ollama server also
//! serves at `/v1/chat/completions`. Tool observations
//! (`Role::Tool`) are folded into `user` messages with a `[tool] ` prefix so
//! no OpenAI tool protocol is required. Responses look up
//! `choices[0].message.content` by path (OpenAI chat / Ollama chat); a missing
//! or malformed path fails closed with [`ProviderError::Transport`] and never
//! falls back to a global substring search (so an `error` field carrying a
//! `"content"` substring cannot masquerade as model output). The `"response"`
//! fallback (Ollama `/api/generate`) applies only when the top-level `choices`
//! key is absent and the response `Content-Type` is JSON
//! (`application/json`, optionally with parameters); anything else (malformed
//! JSON, missing field, over-large, non-2xx) fails closed. Chunked
//! `Transfer-Encoding` is rejected (close-delimited or `Content-Length`
//! only). Request targets use an allowlist: leading `/`, an explicit
//! `/v1/` or `/api/` prefix, and only `A-Za-z0-9/_-.~` bytes (rejecting
//! `tab`, C0 controls, `?#;:,@&%`, and non-ASCII). The `Host` header emits
//! bracketed IPv6 (`[::1]:port`) and bracketed `localhost` is refused at
//! construction (use bare `localhost`).
//!
//! ## Error mapping (existing variants only, `MP-7` parity)
//!
//! - `Transport`: non-loopback refusal, connect failure, malformed envelope,
//!   missing content field, over-large response, unsupported chunked
//!   encoding, non-2xx other than below (no secrets carried).
//! - `Timeout`: connect/write/read timeout. Per-operation allowance is
//!   `min(endpoint, remaining-from-request-deadline)`; an operation that
//!   starts with no remaining time refuses immediately. The reported
//!   `latency_ms` is the effective allowance plus one so `latency > timeout`
//!   holds deterministically.
//! - `Transport` (timeout-arm): a mandatory write/read timeout that fails
//!   to arm refuses instead of being silently ignored, because an unarmed
//!   operation would run outside the request deadline. Before the request is
//!   sent the refusal is `Transport`; a read-timeout arm failure after the
//!   request was sent is `Unknown` (the effect may have happened).
//! - `Unknown`: truncated response (EOF before headers complete, or body
//!   shorter than declared `Content-Length`, or I/O error mid-body). The
//!   effect is uncertain from the client view, so the caller reconciles
//!   before retry and never falls back blindly.
//! - `Auth` (HTTP 401/403), `RateLimited` (HTTP 429, with `Retry-After`
//!   seconds converted to ms when present), `ModelUnavailable` (HTTP 404):
//!   precise status mapping reusing existing variants.
//! - `TimeoutTooLarge`: pre-I/O checks mirroring [`FakeProvider`] (model
//!   mismatch, context budget, timeout ceiling); the script is never consumed
//!   on failure (there is no script; `complete_calls` only increments on
//!   success). A zero `timeout_ms` refuses with `Transport` before I/O, and
//!   an operation that starts with no remaining deadline time refuses with
//!   `Timeout`.
//! - `InvalidSampling` / `UnsupportedSampling`: pre-I/O sampling refusal
//!   (below).
//!
//! ## Sampling mapping (AI-CTX-004)
//!
//! A declared [`SamplingParams`] contract is validated fail-closed before any
//! I/O, then mapped to the OpenAI-compatible body with an explicit
//! backend-field mapping: `temperature` -> `temperature`, `top_p` -> `top_p`,
//! `frequency_penalty` -> `frequency_penalty`, `presence_penalty` ->
//! `presence_penalty`, `seed` -> `seed`, `max_tokens` -> `max_tokens`,
//! `stop` -> `stop`. Absent fields emit nothing (undeclared is not a default)
//! and every emitted value carries its declared form. Fields this minimal
//! backend does not carry reject before I/O with
//! [`ProviderError::UnsupportedSampling`]: `top_k`, `repetition_penalty`,
//! `min_p`, `response_format`, and `reasoning`. The completion `max_tokens`
//! value is serialized for the backend to interpret; it is a declared request
//! field, never a client-side enforcement, and the response-byte ceiling
//! remains an unrelated transport bound.
//!
//! [`ModelProvider`]: bitty_ai_runtime::provider::ModelProvider
//! [`FakeProvider`]: bitty_ai_runtime::provider::FakeProvider
//! [`fallback_directive`]: bitty_ai_runtime::selection::fallback_directive

use std::fmt;
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bitty_ai_runtime::bridge::MAX_CONSENT_SCOPE_LEN;
use bitty_ai_runtime::prompt::MAX_CANONICAL_BYTES;
use bitty_ai_runtime::provider::{
    MAX_REQUEST_TIMEOUT_MS, ModelCapability, ModelDescriptor, ModelProvider, ProviderError,
    ProviderTurn, ProviderUsage, Role, SamplingParams, TurnRequest, validate_provider_id,
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
/// Response-parser nesting ceiling (AI-CTX-006): the accepted body may not
/// nest objects/arrays deeper than this, so recursion is explicitly bounded.
pub const MAX_LOCAL_JSON_DEPTH: usize = 64;
/// Response-parser work ceiling (AI-CTX-006): a hard upper bound on the number
/// of parser steps for one body, independent of the body-byte ceiling.
///
/// The accepted `choices[0].message.content` shape is walked by at most five
/// bounded scans of the same bytes: the root envelope, the `choices` lookup,
/// the `choices[0]` shape check, the `message` lookup, and the `content`
/// lookup. Each scan charges at most two steps per value token, and every value
/// token spans at least two bytes once its separator is counted, so the true
/// worst case is about five steps per input byte. This ceiling keeps an
/// explicit safety margin above that worst case, so no valid body at or under
/// [`MAX_LOCAL_RESPONSE_BYTES`] can be refused by the work budget, while a
/// hostile body that provokes more work still exhausts the budget and is
/// refused typed.
pub const MAX_LOCAL_PARSER_STEPS: usize = 8 * MAX_LOCAL_RESPONSE_BYTES;
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

/// Adapter-owned monotonic clock seam for one request's deadline.
///
/// The adapter measures monotonic elapsed time; the caller only supplies the
/// logical budget (`TurnRequest::timeout_ms`). Tests inject a deterministic
/// fixed clock, so no test reads a real clock, sleeps, or opens a socket.
pub trait MonotonicClock {
    /// Monotonic instant in milliseconds; callers make no wall-clock claim.
    fn now_ms(&self) -> u64;
}

/// Production clock: `Instant` elapsed since the adapter was built.
#[derive(Debug, Clone, Copy)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl SystemMonotonicClock {
    /// Anchor the clock at the current monotonic instant.
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now_ms(&self) -> u64 {
        self.origin
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

/// One request's monotonic deadline: `start_ms + timeout_ms`, saturating.
///
/// Every mandatory operation (connect, write, read) derives its remaining
/// allowance from this single deadline, so the caller-declared whole-request
/// duration is upheld instead of granting each operation an independent
/// allowance.
#[derive(Debug, Clone, Copy)]
struct RequestDeadline {
    end_ms: u64,
}

impl RequestDeadline {
    /// Arm a deadline `timeout_ms` after `start_ms` (saturating).
    fn armed_at(start_ms: u64, timeout_ms: u64) -> Self {
        Self {
            end_ms: start_ms.saturating_add(timeout_ms),
        }
    }

    /// Milliseconds left before the deadline; 0 once it has passed.
    fn remaining_ms(&self, now_ms: u64) -> u64 {
        self.end_ms.saturating_sub(now_ms)
    }
}

/// Effective per-operation allowance: the endpoint policy and the request's
/// remaining time, whichever is smaller, floored at 1 ms so zero never
/// disables a socket timeout.
fn effective_allowance_ms(endpoint_ms: u64, remaining_ms: u64) -> u64 {
    endpoint_ms.min(remaining_ms).max(1)
}

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
        // Bracketed `localhost` is never a valid IP literal: reject it at
        // construction instead of failing later at connect time. Bare
        // `localhost` (any ASCII case) stays accepted; `[::1]` stays accepted.
        if self.host.len() >= 2
            && self.host.starts_with('[')
            && self.host.ends_with(']')
            && self.host[1..self.host.len() - 1].eq_ignore_ascii_case("localhost")
        {
            return Err(ProviderError::Transport {
                provider: self.provider_id.clone(),
                reason: "bracketed localhost refused (use bare localhost)".to_owned(),
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
        if !is_valid_local_path(&self.path) {
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
/// `0.0.0.0`, and non-loopback literals — is refused. Bracketed
/// `[localhost]` is rejected by [`LocalEndpoint::validate`] before this check
/// (it would otherwise fail open at construction and only fail at connect).
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

/// Whether `path` is an allowed localhost request target (AI-0054, P2-8).
///
/// Allowlist: non-empty, `<= MAX_LOCAL_PATH_LEN` bytes, leading `/`, an
/// explicit `/v1/` or `/api/` family prefix (`/v1`, `/api`, `/v1/*`,
/// `/api/*`), and only `A-Za-z0-9/_-.~` bytes. The charset implicitly rejects
/// `tab`, C0 controls (`0x00-0x1F`), `DEL` (`0x7F`), non-ASCII, space, and
/// `?#;:,@&%\"<>\\^_{|}`` — the old blocklist allowed
/// `tab/?/#/;/:/@&/%/C0` through. The two families cover the localhost
/// surface this client speaks: OpenAI-compatible chat (`/v1/*`, default
/// `/v1/chat/completions`) and Ollama native (`/api/*`, e.g.
/// `/api/generate` for the `response` fallback shape).
fn is_valid_local_path(path: &str) -> bool {
    if path.is_empty() || path.len() > MAX_LOCAL_PATH_LEN {
        return false;
    }
    if !path.starts_with('/') {
        return false;
    }
    let prefixed =
        path == "/v1" || path == "/api" || path.starts_with("/v1/") || path.starts_with("/api/");
    if !prefixed {
        return false;
    }
    for byte in path.bytes() {
        let allowed = matches!(
            byte,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'_' | b'-' | b'.' | b'~'
        );
        if !allowed {
            return false;
        }
    }
    true
}

/// Render `host` for the HTTP `Host` header (AI-0054, P2-8).
///
/// Unbracketed IPv6 literals (`::1`, which contain `:`) are emitted bracketed
/// (`[::1]`); already-bracketed (`[::1]`) and bare names/IPv4 (`localhost`,
/// `127.0.0.1`) pass through unchanged. Validation guarantees the input is a
/// loopback literal, so this is pure formatting (no validation, no secrets).
fn host_header_value(host: &str) -> String {
    if host.starts_with('[') && host.ends_with(']') {
        return host.to_owned();
    }
    if host.contains(':') {
        return format!("[{host}]");
    }
    host.to_owned()
}

/// Experimental localhost-only [`ModelProvider`] over raw HTTP/1.1.
///
/// Std-only (`TcpStream`); zero new dependencies. See the module docs for
/// the localhost, timeout, bound, and error-mapping contract.
#[derive(Clone)]
pub struct LocalProvider {
    endpoint: LocalEndpoint,
    models: Vec<ModelDescriptor>,
    complete_calls: u64,
    clock: Arc<dyn MonotonicClock + Send + Sync>,
}

impl LocalProvider {
    /// Build a provider from a validated [`LocalEndpoint`].
    #[must_use]
    pub fn new(endpoint: LocalEndpoint) -> Self {
        Self::with_monotonic_clock(endpoint, Arc::new(SystemMonotonicClock::new()))
    }

    /// Build a provider with an injected monotonic clock (deterministic
    /// tests). The endpoint's loopback policy is unchanged.
    #[must_use]
    pub fn with_monotonic_clock(
        endpoint: LocalEndpoint,
        clock: Arc<dyn MonotonicClock + Send + Sync>,
    ) -> Self {
        let models = vec![ModelDescriptor {
            name: endpoint.model.clone(),
            capabilities: vec![ModelCapability::Text],
        }];
        Self {
            endpoint,
            models,
            complete_calls: 0,
            clock,
        }
    }

    /// Borrow the endpoint configuration.
    #[must_use]
    pub fn endpoint(&self) -> &LocalEndpoint {
        &self.endpoint
    }
}

impl fmt::Debug for LocalProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalProvider")
            .field("endpoint", &self.endpoint)
            .field("models", &self.models)
            .field("complete_calls", &self.complete_calls)
            .field("clock", &"monotonic")
            .finish()
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
        if request.timeout_ms == 0 {
            return Err(ProviderError::Transport {
                provider: self.provider_id().to_owned(),
                reason: "request timeout must be non-zero".to_owned(),
            });
        }
        if request.timeout_ms > MAX_REQUEST_TIMEOUT_MS {
            return Err(ProviderError::TimeoutTooLarge {
                max: MAX_REQUEST_TIMEOUT_MS,
                actual: request.timeout_ms,
            });
        }
        if let Some(params) = &request.sampling {
            bitty_ai_runtime::validate_sampling(params)?;
            check_supported_sampling(params)?;
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
        let body = build_chat_body(
            &self.endpoint.model,
            &request.messages,
            request.sampling.as_ref(),
        );
        if body.len() > MAX_LOCAL_REQUEST_BYTES {
            return Err(ProviderError::Transport {
                provider: self.provider_id().to_owned(),
                reason: format!("request over-large (max {MAX_LOCAL_REQUEST_BYTES})"),
            });
        }
        // One monotonic deadline owns the whole request: connect, write, and
        // read each derive their remaining allowance from it, so the declared
        // whole-request duration is upheld instead of granting each operation
        // an independent allowance.
        let deadline = RequestDeadline::armed_at(self.clock.now_ms(), request.timeout_ms);
        let text = http_round_trip(&self.endpoint, &body, &deadline, self.clock.as_ref())?;
        self.complete_calls += 1;
        Ok(ProviderTurn {
            text,
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
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

/// Reject declared sampling fields this minimal backend does not carry.
///
/// Validation support does not imply backend support: a valid declaration
/// still refuses before any I/O when the backend has no explicit mapping for
/// it. The label is a static field name, never caller input.
fn check_supported_sampling(params: &SamplingParams) -> Result<(), ProviderError> {
    if params.top_k.is_some() {
        return Err(ProviderError::UnsupportedSampling { field: "top_k" });
    }
    if params.repetition_penalty.is_some() {
        return Err(ProviderError::UnsupportedSampling {
            field: "repetition_penalty",
        });
    }
    if params.min_p.is_some() {
        return Err(ProviderError::UnsupportedSampling { field: "min_p" });
    }
    if params.response_format.is_some() {
        return Err(ProviderError::UnsupportedSampling {
            field: "response_format",
        });
    }
    if params.reasoning.is_some() {
        return Err(ProviderError::UnsupportedSampling { field: "reasoning" });
    }
    Ok(())
}

/// Append one explicitly declared field name so values stay caller-ordered.
fn push_field_name(out: &mut String, name: &'static str) {
    out.push_str(",\"");
    out.push_str(name);
    out.push_str("\":");
}

/// Append the declared sampling fields this backend maps.
///
/// Fixed order and shortest decimal form make the body byte-deterministic.
/// Only `Some` fields emit: absent means undeclared and stays absent, never
/// defaulted. `max_tokens` is the declared completion-token field for the
/// backend to interpret; it is not client-side enforcement, and the separate
/// response-byte ceiling is an unrelated transport bound.
fn append_sampling_fields(out: &mut String, params: &SamplingParams) {
    if let Some(value) = params.temperature {
        push_field_name(out, "temperature");
        out.push_str(&value.to_string());
    }
    if let Some(value) = params.top_p {
        push_field_name(out, "top_p");
        out.push_str(&value.to_string());
    }
    if let Some(value) = params.frequency_penalty {
        push_field_name(out, "frequency_penalty");
        out.push_str(&value.to_string());
    }
    if let Some(value) = params.presence_penalty {
        push_field_name(out, "presence_penalty");
        out.push_str(&value.to_string());
    }
    if let Some(value) = params.seed {
        push_field_name(out, "seed");
        out.push_str(&value.to_string());
    }
    if let Some(value) = params.max_tokens {
        push_field_name(out, "max_tokens");
        out.push_str(&value.to_string());
    }
    if let Some(stops) = &params.stop {
        push_field_name(out, "stop");
        out.push('[');
        for (index, stop) in stops.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(&json_escape(stop));
            out.push('"');
        }
        out.push(']');
    }
}

/// Build the minimal OpenAI-compatible chat body plus mapped sampling.
///
/// `Role::Tool` observations are folded into `user` messages with a `[tool] `
/// prefix so no OpenAI tool protocol is required; the untrusted surface stays
/// labeled in the text itself. Declared sampling fields emit only when
/// present; undeclared fields never become defaults.
fn build_chat_body(
    model: &str,
    messages: &[bitty_ai_runtime::provider::Message],
    sampling: Option<&SamplingParams>,
) -> Vec<u8> {
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
    out.push_str("],\"stream\":false");
    if let Some(params) = sampling {
        append_sampling_fields(&mut out, params);
    }
    out.push('}');
    out.into_bytes()
}

/// One blocking HTTP/1.1 round-trip under one monotonic request deadline.
///
/// Every operation derives its remaining allowance from `deadline` at the
/// moment it starts (never an independent per-operation budget), and both
/// timeout arms are mandatory: a failed arm refuses instead of silently
/// dropping the declared whole-request duration.
fn http_round_trip(
    endpoint: &LocalEndpoint,
    body: &[u8],
    deadline: &RequestDeadline,
    clock: &dyn MonotonicClock,
) -> Result<String, ProviderError> {
    let provider = endpoint.provider_id().to_owned();
    let addr = endpoint.socket_addr()?;
    let connect_budget = operation_allowance_ms(endpoint.connect_timeout_ms, deadline, clock)?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(connect_budget))
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::TimedOut {
                remaining_timeout(connect_budget)
            } else {
                ProviderError::Transport {
                    provider: provider.clone(),
                    reason: format!("connect failed: {}", short_io_kind(&error)),
                }
            }
        })?;
    // Mandatory, not best-effort: ignoring a failed write arm would silently
    // leave the write side unbounded and break the declared whole-request
    // duration. The refusal is pre-send, so no request bytes are known to
    // have been applied and `Transport` (advance) is the safe mapping.
    let write_budget = operation_allowance_ms(endpoint.connect_timeout_ms, deadline, clock)?;
    stream
        .set_write_timeout(Some(Duration::from_millis(write_budget)))
        .map_err(|error| map_arm_error(&provider, TimeoutArm::Write, false, &error))?;

    let mut head = format!(
        "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        endpoint.path,
        host_header_value(&endpoint.host),
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
        .map_err(|error| map_write_error(&provider, &error, write_budget, body.len(), false))?;
    stream
        .write_all(body)
        .map_err(|error| map_write_error(&provider, &error, write_budget, body.len(), true))?;

    // Derived at read start, after the connect/write consumption has already
    // been deducted from the same deadline: this is the remaining allowance,
    // never a second full budget.
    let read_budget = operation_allowance_ms(endpoint.read_timeout_ms, deadline, clock)?;
    stream
        .set_read_timeout(Some(Duration::from_millis(read_budget)))
        .map_err(|error| map_arm_error(&provider, TimeoutArm::Read, true, &error))?;
    let raw = read_bounded(&mut stream, &provider, read_budget)?;
    let (status, response_body) = split_response(&raw, &provider)?;
    check_status(endpoint, status, &raw, &provider)?;
    if response_body.len() > MAX_LOCAL_RESPONSE_BYTES {
        return Err(ProviderError::Transport {
            provider,
            reason: format!("response over-large (max {MAX_LOCAL_RESPONSE_BYTES})"),
        });
    }
    let content_type = response_content_type(&raw);
    extract_assistant_text(response_body, content_type.as_deref(), &provider)
}

/// Which mandatory socket timeout failed to arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeoutArm {
    Read,
    Write,
}

/// Map a failed mandatory timeout-arm call to a fail-closed refusal.
///
/// Timeout arming cannot be ignored: an unarmed operation would run outside
/// the request deadline. A failure before any request bytes are sent is
/// [`ProviderError::Transport`] (safe to advance); after the request was
/// sent it is [`ProviderError::Unknown`], because the server may already
/// have applied the request and the acknowledgement is unobtainable under
/// policy. The reason carries only a static arm label plus the short I/O
/// kind, never a caller value.
fn map_arm_error(
    provider: &str,
    arm: TimeoutArm,
    request_sent: bool,
    error: &std::io::Error,
) -> ProviderError {
    let label = match arm {
        TimeoutArm::Read => "read",
        TimeoutArm::Write => "write",
    };
    let reason = format!("{label} timeout arm failed: {}", short_io_kind(error));
    if request_sent {
        ProviderError::Unknown {
            provider: provider.to_owned(),
            reason,
        }
    } else {
        ProviderError::Transport {
            provider: provider.to_owned(),
            reason,
        }
    }
}

/// Derive one operation's allowance from the request deadline, refusing with
/// [`ProviderError::Timeout`] when no time remains.
fn operation_allowance_ms(
    endpoint_ms: u64,
    deadline: &RequestDeadline,
    clock: &dyn MonotonicClock,
) -> Result<u64, ProviderError> {
    let remaining = deadline.remaining_ms(clock.now_ms());
    if remaining == 0 {
        return Err(remaining_timeout(0));
    }
    Ok(effective_allowance_ms(endpoint_ms, remaining))
}

/// Timeout variant for an allowance; `latency_ms > timeout_ms` always holds so
/// the runtime invariant is deterministic.
fn remaining_timeout(allowance_ms: u64) -> ProviderError {
    ProviderError::Timeout {
        timeout_ms: allowance_ms,
        latency_ms: allowance_ms.saturating_add(1),
    }
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
    write_ms: u64,
    body_len: usize,
    body_started: bool,
) -> ProviderError {
    if is_timeout_kind(error) {
        return ProviderError::Timeout {
            timeout_ms: write_ms,
            latency_ms: write_ms.saturating_add(1),
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

#[derive(Debug)]
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
    let header_block = text.get(..split).ok_or_else(|| ProviderError::Transport {
        provider: provider.to_owned(),
        reason: "malformed response offset".to_owned(),
    })?;
    // The header ceiling is independent of the aggregate receive cap: a
    // completed header block that exceeds it is refused even though a
    // terminator was found (AI-CTX-006).
    if header_block.len() > MAX_LOCAL_HEADERS_BYTES {
        return Err(ProviderError::Transport {
            provider: provider.to_owned(),
            reason: "response headers over-large".to_owned(),
        });
    }
    // Byte offset of the body equals the string offset here because the
    // separator is ASCII and `text` borrows `raw`. The offset is checked so
    // the slice can never panic.
    let body_start = split
        .checked_add(4)
        .ok_or_else(|| ProviderError::Transport {
            provider: provider.to_owned(),
            reason: "malformed response offset".to_owned(),
        })?;
    let body = raw
        .get(body_start..)
        .ok_or_else(|| ProviderError::Transport {
            provider: provider.to_owned(),
            reason: "malformed response offset".to_owned(),
        })?;
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
    for line in text.get(..end)?.lines().skip(1) {
        if let Some(value) = header_value(line, "retry-after") {
            let seconds: u64 = value.trim().parse().ok()?;
            return Some(seconds.saturating_mul(1_000));
        }
    }
    None
}

/// Extract the assistant text by path (AI-0054, P1-4 SEC).
///
/// - `choices` present: look up `choices[0].message.content` and return it;
///   any absence or shape mismatch is [`ProviderError::Transport`] with no
///   fallback (an `error` field carrying a `"content"` substring must never
///   masquerade as model output).
/// - `choices` absent: allow the Ollama `/api/generate` `"response"` string
///   fallback only when the response `Content-Type` is JSON
///   (`application/json`, optionally with parameters); otherwise
///   `Transport`. A `choices`-present body never falls back to `response`,
///   even when `response` is present.
///
/// Strictly fail-closed; `std`-only with no new dependencies.
fn extract_assistant_text(
    body: &[u8],
    content_type: Option<&str>,
    provider: &str,
) -> Result<String, ProviderError> {
    let text = std::str::from_utf8(body).map_err(|_| ProviderError::Transport {
        provider: provider.to_owned(),
        reason: "response body is not UTF-8".to_owned(),
    })?;
    // One bounded parser owns every walk over the untrusted body: checked byte
    // access, saturating offset arithmetic, an explicit work budget, and an
    // explicit nesting budget make the whole parse panic-free and bounded
    // (AI-CTX-006).
    let mut parser = JsonParser::new(text);
    parser
        .extract_root(content_type)
        .ok_or_else(|| parser.fail(provider))
}

/// A malformed or over-budget JSON response that cannot be mapped to a more
/// precise variant from the parser itself.
fn parse_refusal(provider: &str, reason: &str) -> ProviderError {
    ProviderError::Transport {
        provider: provider.to_owned(),
        reason: reason.to_owned(),
    }
}

/// Bounded, checked, single-pass JSON parser over an untrusted response body.
///
/// Every byte access is checked, every offset advance saturates or is proven
/// in range, and two explicit budgets bound the work: [`MAX_LOCAL_PARSER_STEPS`]
/// charges each structural action, and [`MAX_LOCAL_JSON_DEPTH`] bounds object
/// and array nesting, so recursion cannot exhaust the stack no matter how the
/// input nests. The parser never slices with an unchecked range and never
/// unwraps untrusted input.
struct JsonParser<'a> {
    bytes: &'a [u8],
    steps: usize,
    depth: usize,
    over_budget: bool,
    too_deep: bool,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            steps: 0,
            depth: 0,
            over_budget: false,
            too_deep: false,
        }
    }

    /// Typed refusal for the first reason encountered, as a static policy
    /// label (no runtime value is formatted into the reason).
    fn fail(&self, provider: &str) -> ProviderError {
        if self.over_budget {
            parse_refusal(provider, "response parser over work budget")
        } else if self.too_deep {
            parse_refusal(provider, "response nesting over budget")
        } else {
            parse_refusal(provider, "malformed response envelope")
        }
    }

    /// Charge one unit of parser work; returns `false` once the explicit work
    /// budget is exhausted.
    fn spend(&mut self) -> bool {
        if self.steps >= MAX_LOCAL_PARSER_STEPS {
            self.over_budget = true;
            return false;
        }
        self.steps += 1;
        true
    }

    /// Enter one nesting level under the explicit depth budget.
    fn enter(&mut self) -> bool {
        if self.depth >= MAX_LOCAL_JSON_DEPTH {
            self.too_deep = true;
            return false;
        }
        self.depth += 1;
        true
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn byte(&self, index: usize) -> Option<u8> {
        self.bytes.get(index).copied()
    }

    fn skip_ws(&self, from: usize) -> usize {
        let mut index = from.min(self.bytes.len());
        while let Some(byte) = self.byte(index) {
            if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
                index = index.saturating_add(1);
            } else {
                break;
            }
        }
        index
    }

    fn extract_root(&mut self, content_type: Option<&str>) -> Option<String> {
        let obj_start = self.skip_ws(0);
        if self.byte(obj_start) != Some(b'{') {
            return None;
        }
        let obj_end = self.skip_object(obj_start)?;
        if self.skip_ws(obj_end) != self.bytes.len() {
            return None;
        }
        if let Some((choices_start, choices_end)) = self.find_field(obj_start, "choices") {
            return self.extract_choices_content(choices_start, choices_end);
        }
        if !is_json_content_type(content_type) {
            return None;
        }
        let (resp_start, resp_end) = self.find_field(obj_start, "response")?;
        if self.byte(resp_start) != Some(b'"') {
            return None;
        }
        match self.parse_string(resp_start) {
            Some((value, end)) if end == resp_end => Some(value),
            _ => None,
        }
    }

    /// `choices[0].message.content`, or `None` for any shape mismatch.
    fn extract_choices_content(
        &mut self,
        choices_start: usize,
        choices_end: usize,
    ) -> Option<String> {
        if self.byte(choices_start) != Some(b'[') {
            return None;
        }
        if choices_end <= choices_start || choices_end > self.bytes.len() {
            return None;
        }
        let cursor = self.skip_ws(choices_start.saturating_add(1));
        if self.byte(cursor) == Some(b']') {
            return None;
        }
        let first_start = cursor;
        if self.byte(first_start) != Some(b'{') {
            return None;
        }
        let first_end = self.skip_object(first_start)?;
        if first_end > choices_end {
            return None;
        }
        let (msg_start, _) = self.find_field(first_start, "message")?;
        if self.byte(msg_start) != Some(b'{') {
            return None;
        }
        let (content_start, content_end) = self.find_field(msg_start, "content")?;
        if self.byte(content_start) != Some(b'"') {
            return None;
        }
        let (value, end) = self.parse_string(content_start)?;
        if end == content_end {
            Some(value)
        } else {
            None
        }
    }

    /// First matching field's value range inside the object at `obj_start`.
    fn find_field(&mut self, obj_start: usize, want: &str) -> Option<(usize, usize)> {
        if self.byte(obj_start) != Some(b'{') {
            return None;
        }
        let mut cursor = self.skip_ws(obj_start.saturating_add(1));
        if self.byte(cursor) == Some(b'}') {
            return None;
        }
        loop {
            if !self.spend() || self.byte(cursor) != Some(b'"') {
                return None;
            }
            let (key, key_end) = self.parse_string(cursor)?;
            cursor = self.skip_ws(key_end);
            if self.byte(cursor) != Some(b':') {
                return None;
            }
            cursor = self.skip_ws(cursor.saturating_add(1));
            let value_start = cursor;
            let value_end = self.skip_value(cursor)?;
            if key == want {
                return Some((value_start, value_end));
            }
            cursor = self.skip_ws(value_end);
            match self.byte(cursor) {
                Some(b',') => {
                    cursor = self.skip_ws(cursor.saturating_add(1));
                    continue;
                }
                Some(b'}') => return None,
                _ => return None,
            }
        }
    }

    fn skip_value(&mut self, cursor: usize) -> Option<usize> {
        if !self.spend() {
            return None;
        }
        let start = self.skip_ws(cursor);
        match self.byte(start) {
            Some(b'"') => self.parse_string(start).map(|(_, end)| end),
            Some(b'{') => self.skip_object(start),
            Some(b'[') => self.skip_array(start),
            Some(b't') => self.skip_literal(start, "true", 4),
            Some(b'f') => self.skip_literal(start, "false", 5),
            Some(b'n') => self.skip_literal(start, "null", 4),
            Some(b'-' | b'0'..=b'9') => self.skip_number(start),
            _ => None,
        }
    }

    fn skip_literal(&mut self, start: usize, literal: &str, len: usize) -> Option<usize> {
        let end = start.checked_add(len)?;
        if end > self.bytes.len() {
            return None;
        }
        if self.bytes.get(start..end) == Some(literal.as_bytes()) {
            Some(end)
        } else {
            None
        }
    }

    fn skip_object(&mut self, cursor: usize) -> Option<usize> {
        if !self.spend() || !self.enter() {
            return None;
        }
        if self.byte(cursor) != Some(b'{') {
            self.leave();
            return None;
        }
        let mut index = self.skip_ws(cursor.saturating_add(1));
        if self.byte(index) == Some(b'}') {
            self.leave();
            return Some(index.saturating_add(1));
        }
        loop {
            if !self.spend() || self.byte(index) != Some(b'"') {
                self.leave();
                return None;
            }
            let (_, key_end) = self.parse_string(index)?;
            index = self.skip_ws(key_end);
            if self.byte(index) != Some(b':') {
                self.leave();
                return None;
            }
            index = self.skip_ws(index.saturating_add(1));
            index = self.skip_value(index)?;
            index = self.skip_ws(index);
            match self.byte(index) {
                Some(b',') => {
                    index = self.skip_ws(index.saturating_add(1));
                    continue;
                }
                Some(b'}') => {
                    self.leave();
                    return Some(index.saturating_add(1));
                }
                _ => {
                    self.leave();
                    return None;
                }
            }
        }
    }

    fn skip_array(&mut self, cursor: usize) -> Option<usize> {
        if !self.spend() || !self.enter() {
            return None;
        }
        if self.byte(cursor) != Some(b'[') {
            self.leave();
            return None;
        }
        let mut index = self.skip_ws(cursor.saturating_add(1));
        if self.byte(index) == Some(b']') {
            self.leave();
            return Some(index.saturating_add(1));
        }
        loop {
            index = self.skip_value(index)?;
            index = self.skip_ws(index);
            match self.byte(index) {
                Some(b',') => {
                    index = self.skip_ws(index.saturating_add(1));
                    continue;
                }
                Some(b']') => {
                    self.leave();
                    return Some(index.saturating_add(1));
                }
                _ => {
                    self.leave();
                    return None;
                }
            }
        }
    }

    fn skip_number(&mut self, cursor: usize) -> Option<usize> {
        if !self.spend() {
            return None;
        }
        let mut index = cursor;
        if self.byte(index) == Some(b'-') {
            index = index.saturating_add(1);
        }
        match self.byte(index) {
            Some(b'0') => index = index.saturating_add(1),
            Some(b'1'..=b'9') => {
                index = index.saturating_add(1);
                while matches!(self.byte(index), Some(b'0'..=b'9')) {
                    index = index.saturating_add(1);
                }
            }
            _ => return None,
        }
        if self.byte(index) == Some(b'.') {
            index = index.saturating_add(1);
            if !matches!(self.byte(index), Some(b'0'..=b'9')) {
                return None;
            }
            while matches!(self.byte(index), Some(b'0'..=b'9')) {
                index = index.saturating_add(1);
            }
        }
        if matches!(self.byte(index), Some(b'e' | b'E')) {
            index = index.saturating_add(1);
            if matches!(self.byte(index), Some(b'+' | b'-')) {
                index = index.saturating_add(1);
            }
            if !matches!(self.byte(index), Some(b'0'..=b'9')) {
                return None;
            }
            while matches!(self.byte(index), Some(b'0'..=b'9')) {
                index = index.saturating_add(1);
            }
        }
        Some(index)
    }

    /// Parse a JSON string at the opening quote.
    ///
    /// Returns the unescaped value and the byte index just past the closing
    /// quote. Fails closed (`None`) on unterminated strings, bad escapes,
    /// invalid `\u`, or raw control characters. Digits and boundaries are
    /// inspected as checked bytes and the value is assembled from bytes, so
    /// no string slice can panic on a byte that is not a character boundary.
    fn parse_string(&mut self, open: usize) -> Option<(String, usize)> {
        if !self.spend() || self.byte(open) != Some(b'"') {
            return None;
        }
        let mut out: Vec<u8> = Vec::new();
        let mut index = open.saturating_add(1);
        while index < self.bytes.len() {
            match self.byte(index)? {
                b'"' => {
                    let value = String::from_utf8(out).ok()?;
                    return Some((value, index.saturating_add(1)));
                }
                b'\\' => {
                    index = index.saturating_add(1);
                    match self.byte(index)? {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            index = index.saturating_add(1);
                            let mut unit: u32 = 0;
                            for _ in 0..4 {
                                let digit = match self.byte(index)? {
                                    byte @ b'0'..=b'9' => u32::from(byte - b'0'),
                                    byte @ b'a'..=b'f' => u32::from(byte - b'a') + 10,
                                    byte @ b'A'..=b'F' => u32::from(byte - b'A') + 10,
                                    _ => return None,
                                };
                                unit = unit.checked_mul(16)?.checked_add(digit)?;
                                index = index.saturating_add(1);
                            }
                            // Reject surrogates and out-of-range units: fail
                            // closed rather than emit replacement characters.
                            let ch = char::from_u32(unit)?;
                            let mut buf = [0_u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            continue;
                        }
                        _ => return None,
                    }
                    index = index.saturating_add(1);
                }
                byte if byte < 0x20 => return None,
                byte => {
                    out.push(byte);
                    index = index.saturating_add(1);
                }
            }
        }
        None
    }
}

/// Whether `content_type` authorizes the Ollama `response` fallback.
///
/// Accepts `application/json` (case-insensitive, optionally with `; ...`
/// parameters, e.g. `application/json; charset=utf-8`), which is what Ollama
/// `/api/generate` returns. Missing or non-JSON content types deny the
/// fallback (fail-closed `Transport`).
fn is_json_content_type(content_type: Option<&str>) -> bool {
    match content_type {
        Some(value) => value.to_ascii_lowercase().contains("application/json"),
        None => false,
    }
}

/// Extract the `Content-Type` header value from a raw HTTP response.
///
/// Scans the header block (up to `\r\n\r\n`), case-insensitive name match.
/// Returns the trimmed value (`None` when absent or non-UTF-8).
fn response_content_type(raw: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(raw).ok()?;
    let end = text.find("\r\n\r\n")?;
    for line in text.get(..end)?.lines().skip(1) {
        if let Some(value) = header_value(line, "content-type") {
            return Some(value.trim().to_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use bitty_ai_runtime::provider::{
        Message, ReasoningConfig, ReasoningEffort, ResponseFormat, SamplingParams,
    };
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
            sampling: None,
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

    fn http_ok_with_content_type(json: &str, content_type: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
            json.len()
        )
        .into_bytes()
    }

    #[test]
    fn error_field_first_empty_choices_fails_closed_ai0054_p1_4() {
        // AI-0054 P1-4 SEC: an `error` object carrying a `"content"` substring
        // first must not masquerade as model output. The old global
        // `find_key_string` returned the error text; the path lookup must fail
        // closed with `Transport` when `choices` is empty.
        let json = r#"{"error":{"content":"evil-in-error"},"choices":[]}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("error-field text must not extract");
        assert!(
            matches!(err, ProviderError::Transport { .. }),
            "empty choices with error content must be Transport, got: {err}"
        );
        assert!(!err.to_string().contains("evil-in-error"));
        assert_eq!(provider.complete_calls(), 0);
        handle.join().expect("stub");
    }

    #[test]
    fn error_field_first_with_valid_choices_returns_choices_ai0054_p1_4() {
        // AI-0054 P1-4 SEC: when a valid `choices[0].message.content` exists
        // after an error field, the path lookup returns the choices text, not
        // the earlier error text.
        let json = r#"{"error":{"content":"evil-in-error"},"choices":[{"message":{"role":"assistant","content":"good-from-choices"}}]}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("valid choices must win over error field");
        assert_eq!(turn.text, "good-from-choices");
        assert_eq!(provider.complete_calls(), 1);
        handle.join().expect("stub");
    }

    #[test]
    fn choices_present_without_content_never_falls_back_to_response_ai0054_p1_4() {
        // AI-0054 P1-4: `response` fallback applies only when `choices` is
        // absent. A `choices`-present body with a missing `content` field must
        // fail closed even when a `response` string is present.
        let json =
            r#"{"choices":[{"message":{"role":"assistant"}}],"response":"fallback-must-not-win"}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("choices-present without content must not fall back");
        assert!(matches!(err, ProviderError::Transport { .. }), "got: {err}");
        assert!(!err.to_string().contains("fallback-must-not-win"));
        assert_eq!(provider.complete_calls(), 0);
        handle.join().expect("stub");
    }

    #[test]
    fn choices_and_response_prefers_choices_ai0054_p1_4() {
        // When both shapes are present and `choices` is valid, the choices
        // path wins; `response` is ignored.
        let json =
            r#"{"choices":[{"message":{"content":"from-choices"}}],"response":"from-response"}"#;
        let (port, handle) = stub_once(http_ok(json), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "from-choices");
        handle.join().expect("stub");
    }

    #[test]
    fn response_fallback_requires_json_content_type_ai0054_p1_4() {
        // AI-0054 P1-4: the Ollama `response` fallback applies only when
        // `choices` is absent AND `Content-Type` is JSON. A `text/plain`
        // envelope with a `response` string must fail closed.
        let json = r#"{"model":"llama3.1:8b","response":"ollama text","done":true}"#;
        let (port, handle) = stub_once(http_ok_with_content_type(json, "text/plain"), 0);
        let mut provider = LocalProvider::new(endpoint_for(port));
        let err = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect_err("non-JSON content-type must deny response fallback");
        assert!(matches!(err, ProviderError::Transport { .. }), "got: {err}");
        handle.join().expect("stub");

        // The same body with a JSON content-type (including parameters)
        // succeeds through the fallback.
        let (port, handle) = stub_once(
            http_ok_with_content_type(json, "application/json; charset=utf-8"),
            0,
        );
        let mut provider = LocalProvider::new(endpoint_for(port));
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("JSON content-type must allow response fallback");
        assert_eq!(turn.text, "ollama text");
        handle.join().expect("stub");
    }

    #[test]
    fn malformed_paths_rejected_at_construction_ai0054_p2_8() {
        // AI-0054 P2-8: allowlist rejects `tab/?/#/;/:/@&/%/C0`, missing
        // leading `/`, wrong families, over-long, and non-ASCII. No stub is
        // bound, so zero sockets can open.
        let mut bad: Vec<String> = vec![
            "v1/chat".to_owned(),
            "/v1/chat completions".to_owned(),
            "/v1/chat\tcompletions".to_owned(),
            "/v1/chat?x=1".to_owned(),
            "/v1/chat#frag".to_owned(),
            "/v1/chat;param".to_owned(),
            "/v1/chat:8080".to_owned(),
            "/v1/chat@host".to_owned(),
            "/v1/chat&x".to_owned(),
            "/v1/chat%20x".to_owned(),
            "/v1/chat\n".to_owned(),
            "/v1/chat\r".to_owned(),
            "/v1/chat\"x".to_owned(),
            "/v1/chat<x>".to_owned(),
            "/v1/chat\\x".to_owned(),
            "/v1/chat^x".to_owned(),
            "/v1/chat{x}".to_owned(),
            "/v1/chat|x}".to_owned(),
            "/evil".to_owned(),
            "/".to_owned(),
            "/v1chat".to_owned(),
            "/api".to_owned().replace("api", "ap?"),
            String::from("/v1/chat\x00"),
            String::from("/v1/chat\x01"),
            String::from("/v1/chat\x1f"),
            String::from("/v1/chat\x7f"),
            "/v1/caf\u{e9}".to_owned(),
        ];
        bad.push(format!("/v1/{}", "a".repeat(125)));
        assert!(bad.last().expect("over-long").len() > MAX_LOCAL_PATH_LEN);
        for path in bad {
            let err = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
                .expect("endpoint")
                .with_path(path.clone())
                .expect_err("malformed path must fail");
            assert!(
                matches!(err, ProviderError::Transport { .. }),
                "path {path:?} must be Transport, got: {err}"
            );
        }
        let probe = LocalProvider::new(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
        );
        assert_eq!(probe.complete_calls(), 0);
    }

    #[test]
    fn valid_paths_accepted_ai0054_p2_8() {
        // AI-0054 P2-8: the explicit `/v1/` + `/api/` families with charset
        // `A-Za-z0-9/_-.~` are accepted, including the 128-byte boundary.
        let boundary = format!("/v1/{}", "a".repeat(124));
        assert_eq!(boundary.len(), MAX_LOCAL_PATH_LEN);
        for path in [
            "/v1/chat/completions".to_owned(),
            "/v1/completions".to_owned(),
            "/v1/embeddings".to_owned(),
            "/v1".to_owned(),
            "/api/generate".to_owned(),
            "/api/chat".to_owned(),
            "/api".to_owned(),
            "/v1/a-b_c.d~e/f".to_owned(),
            "/v1/AZaz09".to_owned(),
            boundary,
        ] {
            assert!(
                LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b")
                    .expect("endpoint")
                    .with_path(path.clone())
                    .is_ok(),
                "valid path rejected: {path:?}"
            );
        }
    }

    #[test]
    fn bracketed_localhost_refused_at_construction_ai0054_p2_8() {
        // AI-0054 P2-8: `[localhost]` must fail at construction (explicit
        // refusal) rather than failing later at connect time. Bare
        // `localhost` in any ASCII case stays accepted.
        for bad in ["[localhost]", "[LOCALHOST]", "[LocalHost]"] {
            let err = LocalEndpoint::new(bad, 11_434, "llama3.1:8b").expect_err("must refuse");
            assert!(
                matches!(err, ProviderError::Transport { .. }),
                "bracketed localhost must be Transport, got: {err}"
            );
            assert!(
                err.to_string().contains("bracketed localhost"),
                "explicit reason required, got: {err}"
            );
        }
        for good in ["localhost", "LOCALHOST", "LocalHost"] {
            assert!(
                LocalEndpoint::new(good, 11_434, "llama3.1:8b").is_ok(),
                "bare localhost refused: {good}"
            );
        }
    }

    #[test]
    fn host_header_bracketing_ai0054_p2_8() {
        // AI-0054 P2-8: unbracketed IPv6 normalizes to bracketed form for the
        // `Host` header; bare names/IPv4 pass through; bracketed input is
        // preserved. Construction + `socket_addr` accept both IPv6 forms.
        assert_eq!(host_header_value("::1"), "[::1]");
        assert_eq!(host_header_value("[::1]"), "[::1]");
        assert_eq!(host_header_value("127.0.0.1"), "127.0.0.1");
        assert_eq!(host_header_value("localhost"), "localhost");
        assert!(LocalEndpoint::new("::1", 11_434, "llama3.1:8b").is_ok());
        assert!(LocalEndpoint::new("[::1]", 11_434, "llama3.1:8b").is_ok());
        let v6 = LocalEndpoint::new("::1", 11_434, "llama3.1:8b").expect("endpoint");
        let addr = v6.socket_addr().expect("socket addr");
        assert_eq!(addr.ip().to_string(), "::1");
        let v6b = LocalEndpoint::new("[::1]", 11_434, "llama3.1:8b").expect("endpoint");
        assert_eq!(
            v6b.socket_addr().expect("socket addr").ip().to_string(),
            "::1"
        );
    }

    #[test]
    fn host_header_on_wire_ai0054_p2_8() {
        // AI-0054 P2-8: the wire `Host` header carries the normalized form.
        // `127.0.0.1` stays bare; `localhost` resolves to loopback and stays
        // bare (no `::1:port` malformation).
        let json = r#"{"choices":[{"message":{"content":"host probe"}}]}"#;
        let (port, handle, req_rx) = stub_capture_once(http_ok(json));
        let mut provider = LocalProvider::new(endpoint_for(port));
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "host probe");
        let raw = req_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("request bytes");
        let text = String::from_utf8_lossy(&raw);
        assert!(
            text.contains(&format!("Host: 127.0.0.1:{port}\r\n")),
            "wire Host must be bare IPv4, got: {text}"
        );
        assert!(!text.contains("::1:"), "must not emit unbracketed IPv6");
        handle.join().expect("stub");

        let (port, handle, req_rx) = stub_capture_once(http_ok(json));
        let endpoint = LocalEndpoint::new("localhost", port, "llama3.1:8b").expect("endpoint");
        let mut provider = LocalProvider::new(endpoint);
        let turn = provider
            .complete(&turn_request("llama3.1:8b", "hi"))
            .expect("turn");
        assert_eq!(turn.text, "host probe");
        let raw = req_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("request bytes");
        let text = String::from_utf8_lossy(&raw);
        assert!(
            text.contains(&format!("Host: localhost:{port}\r\n")),
            "wire Host must be bare localhost, got: {text}"
        );
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

    // ── AI-CTX-004: declared sampling reaches the backend ───────────────────

    /// A sampling contract with every field undeclared.
    fn blank_sampling() -> SamplingParams {
        SamplingParams {
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            min_p: None,
            seed: None,
            max_tokens: None,
            stop: None,
            response_format: None,
            reasoning: None,
        }
    }

    fn body_with(sampling: Option<&SamplingParams>) -> String {
        let body = build_chat_body("llama3.1:8b", &[Message::user("hi")], sampling);
        String::from_utf8(body).expect("utf-8 body")
    }

    fn unbound_provider() -> LocalProvider {
        LocalProvider::new(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
        )
    }

    #[test]
    fn supported_sampling_fields_serialize_with_backend_names_ai_ctx_004() {
        let mut params = blank_sampling();
        params.temperature = Some(0.7);
        params.top_p = Some(0.9);
        params.frequency_penalty = Some(-0.5);
        params.presence_penalty = Some(0.25);
        params.seed = Some(7);
        params.max_tokens = Some(128);
        let text = body_with(Some(&params));
        assert!(text.contains("\"temperature\":0.7"));
        assert!(text.contains("\"top_p\":0.9"));
        assert!(text.contains("\"frequency_penalty\":-0.5"));
        assert!(text.contains("\"presence_penalty\":0.25"));
        assert!(text.contains("\"seed\":7"));
        assert!(text.contains("\"max_tokens\":128"));
        assert!(text.contains("\"stream\":false"));
        // Undeclared fields stay absent: never defaulted.
        assert!(!text.contains("\"top_k\":"));
        assert!(!text.contains("\"stop\":"));
        assert!(!text.contains("\"reasoning\":"));
        assert!(!text.contains("\"response_format\":"));
    }

    #[test]
    fn declared_completion_token_limit_serializes_when_supported_ai_ctx_004() {
        // The declared completion-token limit is a backend request field for
        // the server to interpret. The separate response-byte ceiling is a
        // transport bound and never presented as token-budget enforcement.
        let mut declared = blank_sampling();
        declared.max_tokens = Some(256);
        let text = body_with(Some(&declared));
        assert!(text.contains("\"max_tokens\":256"));

        let mut absent = blank_sampling();
        absent.max_tokens = None;
        let text = body_with(Some(&absent));
        assert!(!text.contains("\"max_tokens\":"));
    }

    #[test]
    fn declared_stop_sequences_serialize_escaped_ai_ctx_004() {
        let mut params = blank_sampling();
        params.stop = Some(vec!["END".to_owned(), "\"quoted\"\n".to_owned()]);
        let text = body_with(Some(&params));
        assert!(text.contains("\"stop\":[\"END\",\"\\\"quoted\\\"\\n\"]"));
        assert!(!text.contains("\"top_p\":"));
    }

    #[test]
    fn absent_versus_explicit_sampling_distinction_is_preserved_ai_ctx_004() {
        // `None` and an all-undeclared contract serialize identically: absent
        // stays absent. An explicit value (even the lower range bound `0`)
        // emits its field, so undeclared and declared are distinguishable.
        let absent = body_with(None);
        let blank = body_with(Some(&blank_sampling()));
        assert_eq!(absent, blank);

        let mut explicit = blank_sampling();
        explicit.temperature = Some(0.0);
        let declared = body_with(Some(&explicit));
        assert_ne!(blank, declared);
        assert!(declared.contains("\"temperature\":0"));
    }

    #[test]
    fn unsupported_sampling_is_refused_before_io_ai_ctx_004() {
        // Validation support is not backend support: fields this minimal
        // OpenAI-compatible body does not carry refuse typed before any I/O.
        // No stub is bound, so zero sockets can open.
        let mut top_k = blank_sampling();
        top_k.top_k = Some(40);
        let mut repetition = blank_sampling();
        repetition.repetition_penalty = Some(1.1);
        let mut min_p = blank_sampling();
        min_p.min_p = Some(0.05);
        let mut response_format = blank_sampling();
        response_format.response_format = Some(ResponseFormat::JsonObject);
        let mut reasoning = blank_sampling();
        reasoning.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::Low),
            max_tokens: None,
            exclude: false,
        });
        let cases = [
            (top_k, "top_k"),
            (repetition, "repetition_penalty"),
            (min_p, "min_p"),
            (response_format, "response_format"),
            (reasoning, "reasoning"),
        ];
        for (params, field) in cases {
            let mut provider = unbound_provider();
            let request = TurnRequest {
                sampling: Some(params),
                ..turn_request("llama3.1:8b", "hi")
            };
            let err = provider
                .complete(&request)
                .expect_err("unsupported declaration must refuse");
            assert_eq!(err, ProviderError::UnsupportedSampling { field });
            assert_eq!(provider.complete_calls(), 0);
        }
    }

    #[test]
    fn unsupported_sampling_display_is_static_ai_ctx_004() {
        let err = ProviderError::UnsupportedSampling { field: "top_k" };
        assert_eq!(
            err.to_string(),
            "sampling option unsupported by backend: top_k"
        );
    }

    #[test]
    fn invalid_sampling_takes_precedence_over_unsupported_ai_ctx_004() {
        // Range validation runs before backend support: an out-of-range
        // declaration is `InvalidSampling` even when another field would be
        // unsupported. No stub is bound, so zero sockets open.
        let mut provider = unbound_provider();
        let mut params = blank_sampling();
        params.top_k = Some(-1);
        params.min_p = Some(0.5);
        let request = TurnRequest {
            sampling: Some(params),
            ..turn_request("llama3.1:8b", "hi")
        };
        let err = provider
            .complete(&request)
            .expect_err("invalid declaration must refuse");
        assert_eq!(err, ProviderError::InvalidSampling);
        assert_eq!(provider.complete_calls(), 0);
    }

    #[test]
    fn supported_sampling_declaration_still_serializes_every_declared_field_ai_ctx_004() {
        // The mapped fields together form one deterministic body; every
        // declared value appears exactly once with its backend field name.
        let mut params = blank_sampling();
        params.temperature = Some(1.5);
        params.top_p = Some(0.8);
        params.frequency_penalty = Some(0.0);
        params.presence_penalty = Some(-1.0);
        params.seed = Some(-3);
        params.max_tokens = Some(4_096);
        params.stop = Some(vec!["done".to_owned()]);
        let text = body_with(Some(&params));
        assert_eq!(text.matches("\"temperature\":").count(), 1);
        assert_eq!(text.matches("\"top_p\":").count(), 1);
        assert_eq!(text.matches("\"frequency_penalty\":").count(), 1);
        assert_eq!(text.matches("\"presence_penalty\":").count(), 1);
        assert_eq!(text.matches("\"seed\":").count(), 1);
        assert_eq!(text.matches("\"max_tokens\":").count(), 1);
        assert_eq!(text.matches("\"stop\":").count(), 1);
        assert_eq!(text.matches("\"top_k\":").count(), 0);
    }

    /// Fixed deterministic clock: tests set the instant explicitly; no
    /// thread, sleep, wall clock, or socket is consulted.
    struct FixedClock {
        instant_ms: AtomicU64,
    }

    impl FixedClock {
        fn new(instant_ms: u64) -> Self {
            Self {
                instant_ms: AtomicU64::new(instant_ms),
            }
        }

        fn set(&self, instant_ms: u64) {
            self.instant_ms.store(instant_ms, Ordering::SeqCst);
        }
    }

    impl MonotonicClock for FixedClock {
        fn now_ms(&self) -> u64 {
            self.instant_ms.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn request_deadline_remaining_derives_and_saturates_ai_ctx_005() {
        let deadline = RequestDeadline::armed_at(1_000, 500);
        assert_eq!(deadline.remaining_ms(1_000), 500);
        assert_eq!(deadline.remaining_ms(1_400), 100);
        assert_eq!(deadline.remaining_ms(1_500), 0);
        assert_eq!(deadline.remaining_ms(u64::MAX), 0);
        // Arming saturates instead of wrapping past `u64::MAX`.
        let saturated = RequestDeadline::armed_at(u64::MAX - 50, 100);
        assert_eq!(saturated.remaining_ms(u64::MAX - 50), 50);
    }

    #[test]
    fn effective_allowance_is_the_smaller_bound_ai_ctx_005() {
        assert_eq!(effective_allowance_ms(5_000, 1_000), 1_000);
        assert_eq!(effective_allowance_ms(200, 5_000), 200);
        assert_eq!(effective_allowance_ms(200, 1), 1);
    }

    #[test]
    fn operation_allowance_derives_from_remaining_time_ai_ctx_005() {
        // The injected clock drives every derivation: the endpoint policy
        // bounds the first operation, then the same deadline bounds the rest.
        let clock = FixedClock::new(0);
        let deadline = RequestDeadline::armed_at(0, 1_000);
        let first = operation_allowance_ms(400, &deadline, &clock).expect("remaining time");
        assert_eq!(first, 400, "the endpoint policy is the smaller bound");
        clock.set(800);
        let second = operation_allowance_ms(400, &deadline, &clock).expect("remaining time");
        assert_eq!(
            second, 200,
            "the remaining request time is the smaller bound"
        );
        clock.set(999);
        let last = operation_allowance_ms(5_000, &deadline, &clock).expect("remaining time");
        assert_eq!(last, 1, "the last millisecond stays armed at one");
        clock.set(1_000);
        let err = operation_allowance_ms(5_000, &deadline, &clock).expect_err("deadline passed");
        assert!(
            matches!(
                err,
                ProviderError::Timeout {
                    timeout_ms: 0,
                    latency_ms: 1
                }
            ),
            "an operation starting at the deadline must refuse with Timeout"
        );
    }

    #[test]
    fn operation_allowances_never_extend_past_the_deadline_ai_ctx_005() {
        // Each operation derives its allowance from the same deadline at its
        // own start, so even an operation that consumes its full allowance
        // cannot run past the caller-declared end: start + allowance never
        // exceeds the deadline.
        let clock = FixedClock::new(0);
        let deadline = RequestDeadline::armed_at(0, 300);
        for (now_ms, expected) in [(0_u64, 300_u64), (100, 200), (250, 50), (299, 1)] {
            clock.set(now_ms);
            let granted =
                operation_allowance_ms(10_000, &deadline, &clock).expect("remaining time");
            assert_eq!(granted, expected, "allowance is the remaining time");
            assert!(
                now_ms + granted <= 300,
                "an operation started now cannot be allowed past the deadline"
            );
        }
        clock.set(300);
        assert!(
            operation_allowance_ms(10_000, &deadline, &clock).is_err(),
            "no operation may start at or after the deadline"
        );
    }

    #[test]
    fn expired_deadline_refuses_before_connect_ai_ctx_005() {
        // Connect is the first operation observed: an already-expired
        // deadline refuses with no socket opened (port 11434 has no stub in
        // this test), so the refusal cannot be a connect failure.
        let clock = FixedClock::new(10);
        let endpoint = LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint");
        let deadline = RequestDeadline::armed_at(10, 5);
        clock.set(15);
        let err = http_round_trip(&endpoint, b"{}", &deadline, &clock).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ProviderError::Timeout {
                    timeout_ms: 0,
                    latency_ms: 1
                }
            ),
            "an expired deadline must refuse with Timeout before connect"
        );
    }

    #[test]
    fn zero_request_timeout_refuses_before_io_ai_ctx_005() {
        // A zero-duration request refuses before any socket is opened; the
        // fixed clock proves the adapter consults only its own clock seam.
        let clock = Arc::new(FixedClock::new(4_242));
        let mut provider = LocalProvider::with_monotonic_clock(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
            clock,
        );
        let request = TurnRequest {
            timeout_ms: 0,
            ..turn_request("llama3.1:8b", "hi")
        };
        let err = provider
            .complete(&request)
            .expect_err("zero duration must refuse");
        assert!(
            matches!(&err, ProviderError::Transport { reason, .. }
                if reason.contains("non-zero")),
            "a zero-duration request must refuse before I/O"
        );
        assert!(
            provider.complete_calls() == 0,
            "no turn completes on refusal"
        );
    }

    /// Deterministic stepping clock: every read advances by a fixed step, so
    /// a request can be driven exactly onto its deadline without sleeping.
    struct StepClock {
        step_ms: u64,
        now_ms: AtomicU64,
    }

    impl StepClock {
        fn new(step_ms: u64) -> Self {
            Self {
                step_ms,
                now_ms: AtomicU64::new(0),
            }
        }
    }

    impl MonotonicClock for StepClock {
        fn now_ms(&self) -> u64 {
            self.now_ms.fetch_add(self.step_ms, Ordering::SeqCst)
        }
    }

    #[test]
    fn provider_complete_refuses_when_deadline_expired_at_first_operation_ai_ctx_005() {
        // The injected clock is threaded through construction: arming
        // consumes one instant and the first operation's derivation consumes
        // the next, landing exactly on the deadline. The refusal is before
        // connect, so no socket is opened (the endpoint port has no stub).
        let clock = Arc::new(StepClock::new(50));
        let mut provider = LocalProvider::with_monotonic_clock(
            LocalEndpoint::new("127.0.0.1", 11_434, "llama3.1:8b").expect("endpoint"),
            clock,
        );
        let request = TurnRequest {
            timeout_ms: 50,
            ..turn_request("llama3.1:8b", "hi")
        };
        let err = provider
            .complete(&request)
            .expect_err("expired deadline must refuse");
        assert!(
            matches!(
                err,
                ProviderError::Timeout {
                    timeout_ms: 0,
                    latency_ms: 1
                }
            ),
            "the first operation at the deadline must refuse with Timeout"
        );
        assert_eq!(provider.complete_calls(), 0);
    }

    #[test]
    fn write_arm_failure_before_send_is_transport_ai_ctx_005() {
        let error = std::io::Error::other("probe");
        let mapped = map_arm_error("local-ollama", TimeoutArm::Write, false, &error);
        assert!(
            matches!(&mapped, ProviderError::Transport { provider, reason }
                if provider == "local-ollama" && reason.contains("write timeout arm failed")),
            "a pre-send write arm failure must refuse as Transport"
        );
    }

    #[test]
    fn read_arm_failure_after_send_is_unknown_ai_ctx_005() {
        // The read timeout arms only after the request bytes were sent, so a
        // failed arm leaves the model effect uncertain: `Unknown`, which the
        // runtime reconciles instead of blind-advancing.
        let error = std::io::Error::other("probe");
        let mapped = map_arm_error("local-ollama", TimeoutArm::Read, true, &error);
        assert!(
            matches!(&mapped, ProviderError::Unknown { provider, reason }
                if provider == "local-ollama" && reason.contains("read timeout arm failed")),
            "a post-send read arm failure must refuse as Unknown"
        );
    }

    #[test]
    fn arm_error_reason_carries_no_runtime_detail_ai_ctx_005() {
        // The reason maps the I/O kind to a static label; the underlying
        // error text never reaches the caller-facing refusal.
        let error = std::io::Error::other("probe-value-must-not-leak");
        let mapped = map_arm_error("local-ollama", TimeoutArm::Read, false, &error);
        let rendered = mapped.to_string();
        assert!(
            !rendered.contains("probe-value-must-not-leak"),
            "the arm refusal must not echo runtime detail"
        );
    }

    // ── AI-CTX-006: bounded, panic-free response parsing ────────────────────

    /// Parse one body with no content type (the choices path only).
    fn parse_body(body: &str) -> Result<String, ProviderError> {
        extract_assistant_text(body.as_bytes(), None, "local-ollama")
    }

    #[test]
    fn hostile_json_corpus_refuses_with_typed_errors_ai_ctx_006() {
        // Deterministic hostile-input corpus: every case must return a typed
        // `Transport` refusal without panicking. Assert messages are static
        // (AI-0082): no runtime value is formatted into a panic path.
        let corpus = [
            // Truncated document, unterminated object, string, and escape.
            r#"{"choices":[{"message":{"content":"hi"}"#,
            r#"{"choices":[{"message":{"content":"unterminated"#,
            r#"{"choices":[{"message":{"content":"\#,
            // Malformed structure and stray bytes.
            r#"{"choices":[{"message":{"content":"hi"}}]}"#,
            r#"{"choices":not-json}"#,
            r#"{"choices":[{"message":{"content":123}}]}"#,
            r#"{} trailing"#,
            r#"[1,2,3]"#,
            // Invalid/partial Unicode escapes: surrogate, bad hex, short run,
            // escape truncated at end, and a multibyte byte where a hex digit
            // is required (the old unchecked slice boundary).
            r#"{"choices":[{"message":{"content":"\uD800"}}]}"#,
            r#"{"choices":[{"message":{"content":"\uZZZZ"}}]}"#,
            r#"{"choices":[{"message":{"content":"\u12"}}]}"#,
            r#"{"choices":[{"message":{"content":"\u00"}}]}"#,
            "{\"choices\":[{\"message\":{\"content\":\"\\u\u{e9}0\"}}]}",
            "{\"choices\":[{\"message\":{\"content\":\"\\u00\u{e9}\"}}]}",
            // Missing content and wrong shape.
            r#"{"choices":[]}"#,
            r#"{"choices":[{"message":{}}]}"#,
            r#"{"choices":[{"message":{"content":[]}}]}"#,
            // Raw control byte inside a string.
            "{\"choices\":[{\"message\":{\"content\":\"a\u{1}b\"}}]}",
        ];
        for body in corpus {
            match parse_body(body) {
                Err(ProviderError::Transport { .. }) => {}
                _ => panic!("hostile body must refuse with a typed Transport error"),
            }
        }
    }

    #[test]
    fn valid_json_and_multibyte_escapes_still_parse_ai_ctx_006() {
        // The bounded parser preserves every valid path, including raw
        // multibyte text and valid `\u` escapes.
        let raw = parse_body(r#"{"choices":[{"message":{"content":"café"}}]}"#)
            .expect("raw multibyte content");
        assert_eq!(raw, "café");
        let escaped = parse_body(r#"{"choices":[{"message":{"content":"caf\u00e9 \u0041"}}]}"#)
            .expect("escaped content");
        assert_eq!(escaped, "caf\u{e9} A");
        let nested =
            parse_body(r#"{"choices":[{"message":{"content":"hi"},"extra":{"a":[1,true,null]}}]}"#)
                .expect("nested valid content");
        assert_eq!(nested, "hi");
    }

    #[test]
    fn parser_nesting_budget_is_explicit_ai_ctx_006() {
        // At the boundary the parse succeeds; one level deeper it refuses
        // typed instead of recursing without a bound.
        let mut allowed = "[".repeat(MAX_LOCAL_JSON_DEPTH);
        allowed.push_str(&"]".repeat(MAX_LOCAL_JSON_DEPTH));
        let mut parser = JsonParser::new(&allowed);
        assert!(
            parser.skip_value(0).is_some(),
            "nesting at the budget must parse"
        );
        assert!(!parser.too_deep, "nesting at the budget is not over budget");

        let mut over = "[".repeat(MAX_LOCAL_JSON_DEPTH + 1);
        over.push_str(&"]".repeat(MAX_LOCAL_JSON_DEPTH + 1));
        let mut parser = JsonParser::new(&over);
        assert!(
            parser.skip_value(0).is_none(),
            "nesting over budget must refuse"
        );
        assert!(parser.too_deep, "nesting over budget must be recorded");
        let err = parser.fail("local-ollama");
        assert!(
            matches!(&err, ProviderError::Transport { provider, reason }
                if provider == "local-ollama" && reason.contains("nesting over budget")),
            "over-deep nesting must map to a static typed refusal"
        );
    }

    #[test]
    fn parser_work_budget_is_explicit_and_typed_ai_ctx_006() {
        // The work budget is an explicit finite bound: once exhausted the
        // parser refuses instead of continuing to walk the input.
        let mut parser = JsonParser::new("[0]");
        parser.steps = MAX_LOCAL_PARSER_STEPS;
        assert!(
            parser.skip_value(0).is_none(),
            "exhausted work budget must refuse"
        );
        assert!(parser.over_budget, "the work budget exhaustion is recorded");
        let err = parser.fail("local-ollama");
        assert!(
            matches!(&err, ProviderError::Transport { provider, reason }
                if provider == "local-ollama" && reason.contains("work budget")),
            "work budget exhaustion must map to a static typed refusal"
        );
    }

    /// Build a `choices[0].message.content` body no larger than `filler_cap`
    /// bytes whose accepted path re-walks a flat filler array placed before
    /// `content`, i.e. the shape that maximizes accepted parser steps per byte.
    fn worst_case_choices_body(filler_cap: usize) -> String {
        let mut body = String::from("{\"choices\":[{\"message\":{\"filler\":[");
        let head = body.len();
        let tail = "],\"content\":\"hi\"}}]}";
        // Each element after the first occupies two bytes (`0,`).
        let elements = filler_cap.saturating_sub(head + tail.len() + 1) / 2;
        for index in 0..elements {
            if index > 0 {
                body.push(',');
            }
            body.push('0');
        }
        body.push_str(tail);
        body
    }

    #[test]
    fn max_size_valid_choices_body_parses_within_work_budget_ai_ctx_006() {
        // Regression (AI-0115 F1/PX-0483): a valid body at the response-byte
        // cap, whose accepted path re-walks a large filler array five times,
        // must still parse and return its content instead of being refused by
        // the work budget.
        let body = worst_case_choices_body(MAX_LOCAL_RESPONSE_BYTES);
        assert!(
            body.len() <= MAX_LOCAL_RESPONSE_BYTES,
            "the regression body must sit at or under the response cap"
        );
        assert!(
            body.len().saturating_add(256) >= MAX_LOCAL_RESPONSE_BYTES,
            "the regression body must be near the response cap"
        );
        let parsed = parse_body(&body).expect("max-size valid body must parse");
        assert_eq!(parsed, "hi", "the accepted content must be returned");

        // The accepted path stays strictly inside the explicit work ceiling, so
        // the ceiling is a real margin above the worst-case accepted cost.
        let mut parser = JsonParser::new(&body);
        assert!(
            parser.extract_root(None).is_some(),
            "the near-cap accepted body must parse"
        );
        assert!(
            parser.steps < MAX_LOCAL_PARSER_STEPS,
            "accepted work must remain under the work ceiling"
        );
        assert!(!parser.over_budget, "an accepted body is not over budget");
    }

    #[test]
    fn over_cap_pathological_body_still_exhausts_work_budget_ai_ctx_006() {
        // The work ceiling remains a genuine guard: input forcing more than the
        // worst-case accepted charge per byte (here, a body well beyond the
        // response cap) exhausts the budget and is refused typed, so the parser
        // always terminates on hostile input.
        let body = worst_case_choices_body(4 * MAX_LOCAL_RESPONSE_BYTES);
        let mut parser = JsonParser::new(&body);
        assert!(
            parser.extract_root(None).is_none(),
            "a body over the work budget must refuse"
        );
        assert!(parser.over_budget, "the work budget exhaustion is recorded");
        let err = parser.fail("local-ollama");
        assert!(
            matches!(&err, ProviderError::Transport { provider, reason }
                if provider == "local-ollama" && reason.contains("work budget")),
            "over-budget input must map to a static typed refusal"
        );
    }

    #[test]
    fn completed_header_block_over_ceiling_is_refused_ai_ctx_006() {
        // The header ceiling applies to a completed block, not only to a
        // missing terminator.
        let mut raw = String::from("HTTP/1.1 200 OK\r\n");
        raw.push_str("X-Pad: ");
        raw.push_str(&"a".repeat(MAX_LOCAL_HEADERS_BYTES));
        raw.push_str("\r\n\r\n{}");
        let err = split_response(raw.as_bytes(), "local-ollama").expect_err("over-large headers");
        assert!(
            matches!(&err, ProviderError::Transport { reason, .. }
                if reason.contains("headers over-large")),
            "completed headers over the ceiling must be refused"
        );

        // A small completed block still parses.
        let ok = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert!(
            split_response(ok, "local-ollama").is_ok(),
            "small headers must parse"
        );
    }

    #[test]
    fn malformed_and_huge_declared_length_refuse_typed_ai_ctx_006() {
        // A huge but parseable declared length is refused as over-large; a
        // non-numeric one is refused as malformed. Neither panics or
        // allocates.
        let huge = b"HTTP/1.1 200 OK\r\nContent-Length: 18446744073709551615\r\n\r\n{}";
        let err = split_response(huge, "local-ollama").expect_err("huge declared length");
        assert!(
            matches!(&err, ProviderError::Transport { reason, .. }
                if reason.contains("over-large")),
            "huge declared length must be refused as over-large"
        );

        let malformed = b"HTTP/1.1 200 OK\r\nContent-Length: 12x34\r\n\r\n{}";
        let err = split_response(malformed, "local-ollama").expect_err("malformed length");
        assert!(
            matches!(&err, ProviderError::Transport { reason, .. }
                if reason.contains("malformed content-length")),
            "non-numeric declared length must be refused as malformed"
        );

        let truncated = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n{}";
        let err = split_response(truncated, "local-ollama").expect_err("short body");
        assert!(
            matches!(err, ProviderError::Unknown { .. }),
            "a short body against a declared length must be Unknown"
        );

        let overlong = b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n{}";
        let err = split_response(overlong, "local-ollama").expect_err("long body");
        assert!(
            matches!(&err, ProviderError::Transport { reason, .. }
                if reason.contains("longer than declared")),
            "a body longer than declared must be refused"
        );
    }

    #[test]
    fn parser_never_panics_on_single_byte_prefixes_ai_ctx_006() {
        // Exhaustive byte-prefix sweep of a valid body (and of a body that
        // contains a multibyte character) proves checked access: any prefix
        // either parses or refuses, never panics.
        for full in [
            r#"{"choices":[{"message":{"content":"hi"}}]}"#,
            r#"{"choices":[{"message":{"content":"café"}}]}"#,
            r#"{"choices":[{"message":{"content":"\u00e9"}}]}"#,
        ] {
            for end in 0..=full.len() {
                if full.is_char_boundary(end) {
                    let _ = parse_body(&full[..end]);
                }
            }
        }
    }
}

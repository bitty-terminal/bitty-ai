//! Consumer transport adapter prototype for `bitty-network-api` (AI-0156).
//!
//! Implements the consumer side of the provider transport adapter contract
//! defined in `providers/transport-adapter-contract.md`:
//! - Binds an underlying [`NetworkService`] to drive outgoing model calls.
//! - Maps [`TurnRequest`] into [`bitty_network_api::Request`]:
//!   - `TurnRequest.budget_bytes` maps directly to `Request::max_body_bytes`.
//!   - `TurnRequest.timeout_ms` maps directly to `Request::timeout`.
//!   - `SecretField::expose_for_adapter()` injects bearer authorization strictly
//!     at the host adapter edge, without retaining secret material in
//!     configurations, errors, traces, or debug outputs.
//! - Enforces pre-I/O context-budget, model, timeout, sampling, and capability
//!   gates before any network call. The capability gate defaults to
//!   deny-all (offline-first): `capability.check_request(&net_req)` runs
//!   after request construction and before any socket work, so a denied or
//!   offline capability fails closed with zero service calls.
//! - Maps typed [`bitty_network_api::NetworkError`] outcomes to typed [`ProviderError`]s.

use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use bitty_ai_runtime::prompt::MAX_CANONICAL_BYTES;
use bitty_ai_runtime::provider::{
    MAX_REQUEST_TIMEOUT_MS, ModelDescriptor, ModelProvider, ProviderError, ProviderTurn,
    ProviderUsage, Role, SamplingParams, ToolCallRequest, TurnRequest, validate_provider_id,
    validate_sampling,
};
use bitty_ai_runtime::secret::SecretField;
use bitty_ai_runtime::selection::validate_model_name;
use bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES;
use bitty_network_api::{
    NetworkCapability, NetworkError, NetworkService, Request, Response, WebSocketRequest,
};

/// Adapter configuration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    /// Provider ID does not satisfy the `MP-2` shape.
    InvalidProviderId(String),
    /// Model name does not satisfy the naming rules.
    InvalidModelName(String),
    /// Endpoint URL is empty.
    EmptyEndpointUrl,
    /// Endpoint URL does not start with `http://` or `https://`.
    MalformedEndpointUrl(String),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProviderId(id) => write!(f, "invalid provider id: {id}"),
            Self::InvalidModelName(name) => write!(f, "invalid model name: {name}"),
            Self::EmptyEndpointUrl => write!(f, "endpoint url must not be empty"),
            Self::MalformedEndpointUrl(url) => write!(f, "malformed endpoint url: {url}"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Configuration for the [`NetworkConsumerAdapter`].
pub struct NetworkConsumerAdapterConfig {
    /// Canonical provider identity (`MP-2`).
    pub provider_id: String,
    /// Target model name.
    pub model: String,
    /// Full endpoint URL (e.g. `https://api.openai.com/v1/chat/completions`).
    pub endpoint_url: String,
    /// Optional credential container; secret bytes are exposed only at the transport edge.
    pub api_key: Option<SecretField>,
    /// Additional static HTTP headers.
    pub extra_headers: Vec<(String, String)>,
    /// Model descriptors declared by this adapter.
    pub models: Vec<ModelDescriptor>,
    /// Network capability allowlist enforced pre-I/O via
    /// `capability.check_request(&net_req)` (offline-first: deny-all default).
    pub capability: NetworkCapability,
}

impl NetworkConsumerAdapterConfig {
    /// Create a new adapter configuration with minimal required parameters.
    pub fn new(
        provider_id: impl Into<String>,
        model: impl Into<String>,
        endpoint_url: impl Into<String>,
    ) -> Result<Self, AdapterError> {
        let provider_id = provider_id.into();
        let model = model.into();
        let endpoint_url = endpoint_url.into();

        validate_provider_id(&provider_id)
            .map_err(|_| AdapterError::InvalidProviderId(provider_id.clone()))?;
        validate_model_name(&model).map_err(|_| AdapterError::InvalidModelName(model.clone()))?;

        if endpoint_url.is_empty() {
            return Err(AdapterError::EmptyEndpointUrl);
        }
        if !endpoint_url.starts_with("http://") && !endpoint_url.starts_with("https://") {
            return Err(AdapterError::MalformedEndpointUrl(endpoint_url));
        }

        let default_descriptor = ModelDescriptor {
            name: model.clone(),
            capabilities: Vec::new(),
        };

        Ok(Self {
            provider_id,
            model,
            endpoint_url,
            api_key: None,
            extra_headers: Vec::new(),
            models: vec![default_descriptor],
            capability: NetworkCapability::offline(),
        })
    }

    /// Set an optional API key / bearer token via typed [`SecretField`].
    #[must_use]
    pub fn with_api_key(mut self, api_key: SecretField) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Add an extra HTTP header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Override the declared model descriptors.
    #[must_use]
    pub fn with_models(mut self, models: Vec<ModelDescriptor>) -> Self {
        self.models = models;
        self
    }

    /// Grant the network capability allowlist enforced pre-I/O.
    ///
    /// The default is deny-all ([`NetworkCapability::offline`]): every
    /// endpoint stays unreachable until the caller grants its domain (e.g.
    /// `NetworkCapability::offline().with_domain("api.example.com")`).
    #[must_use]
    pub fn with_capability(mut self, capability: NetworkCapability) -> Self {
        self.capability = capability;
        self
    }
}

impl fmt::Debug for NetworkConsumerAdapterConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetworkConsumerAdapterConfig")
            .field("provider_id", &self.provider_id)
            .field("model", &self.model)
            .field("endpoint_url", &self.endpoint_url)
            .field("api_key_configured", &self.api_key.is_some())
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("extra_headers_count", &self.extra_headers.len())
            .field("models", &self.models)
            .field("capability", &self.capability)
            .finish()
    }
}

/// A consumer transport adapter adapting a [`NetworkService`] to [`ModelProvider`].
pub struct NetworkConsumerAdapter<S> {
    config: NetworkConsumerAdapterConfig,
    service: S,
    complete_calls: u64,
}

impl<S> NetworkConsumerAdapter<S> {
    /// Construct a new adapter with configuration and network service.
    pub fn new(config: NetworkConsumerAdapterConfig, service: S) -> Self {
        Self {
            config,
            service,
            complete_calls: 0,
        }
    }

    /// Reference to the underlying configuration.
    pub fn config(&self) -> &NetworkConsumerAdapterConfig {
        &self.config
    }

    /// Reference to the underlying network service.
    pub fn service(&self) -> &S {
        &self.service
    }

    /// Mutable reference to the underlying network service.
    pub fn service_mut(&mut self) -> &mut S {
        &mut self.service
    }
}

impl<S> fmt::Debug for NetworkConsumerAdapter<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetworkConsumerAdapter")
            .field("config", &self.config)
            .field("complete_calls", &self.complete_calls)
            .finish()
    }
}

impl<S: NetworkService> ModelProvider for NetworkConsumerAdapter<S> {
    fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    fn list_models(&self) -> Vec<ModelDescriptor> {
        self.config.models.clone()
    }

    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError> {
        // Pre-I/O Gate 1: Non-zero caller timeout
        if request.timeout_ms == 0 {
            return Err(ProviderError::Transport {
                provider: self.config.provider_id.clone(),
                reason: "request timeout must be non-zero".to_owned(),
            });
        }

        // Pre-I/O Gate 2: Maximum request timeout ceiling
        if request.timeout_ms > MAX_REQUEST_TIMEOUT_MS {
            return Err(ProviderError::TimeoutTooLarge {
                max: MAX_REQUEST_TIMEOUT_MS,
                actual: request.timeout_ms,
            });
        }

        // Pre-I/O Gate 3: Sampling validation (range) plus backend support.
        // Validation support does not imply backend support: a valid
        // declaration still refuses before any I/O when this adapter has no
        // explicit mapping for it (mirrors `local_provider`).
        if let Some(ref params) = request.sampling {
            validate_sampling(params)?;
            check_supported_sampling(params)?;
        }

        // Pre-I/O Gate 4: Unknown model rejection
        if request.model != self.config.model {
            return Err(ProviderError::UnknownModel {
                name: request.model.clone(),
            });
        }

        // Pre-I/O Gate 5: Context-budget check before initiating provider I/O
        let actual = request.total_message_bytes();
        if actual > request.budget_bytes {
            return Err(ProviderError::BudgetExceeded {
                limit: request.budget_bytes,
                actual,
            });
        }

        // Build serialized payload
        let body = build_request_body(&self.config.model, request)?;
        if body.len() > MAX_CANONICAL_BYTES {
            return Err(ProviderError::Transport {
                provider: self.config.provider_id.clone(),
                reason: format!(
                    "request payload exceeds canonical ceiling (max {MAX_CANONICAL_BYTES})"
                ),
            });
        }

        // Map TurnRequest into bitty_network_api::Request:
        // - budget_bytes maps directly to max_body_bytes
        // - timeout_ms maps directly to timeout
        let mut net_req = Request::post(&self.config.endpoint_url, body)
            .with_max_body_bytes(request.budget_bytes as u64)
            .with_timeout(Duration::from_millis(request.timeout_ms))
            .with_header("content-type", "application/json")
            .with_header("accept", "application/json");

        for (k, v) in &self.config.extra_headers {
            net_req = net_req.with_header(k, v);
        }

        // Host adapter edge: secret injection via expose_for_adapter()
        if let Some(ref secret) = self.config.api_key {
            let token_bytes = secret.expose_for_adapter();
            let token_str = std::str::from_utf8(token_bytes).map_err(|_| ProviderError::Auth {
                provider: self.config.provider_id.clone(),
                reason: "api_key contains invalid utf-8".to_owned(),
            })?;
            net_req = net_req.with_header("authorization", format!("Bearer {token_str}"));
        }

        // Pre-I/O Gate 6: Capability allowlist (host, port, method) enforced
        // after request construction and before any socket work. A denied or
        // offline capability fails closed here with zero service calls; the
        // denial reuses the typed transport-error mapping below.
        if let Err(denial) = self.config.capability.check_request(&net_req) {
            return Err(map_network_error(
                &self.config.provider_id,
                denial,
                request.timeout_ms,
                request.budget_bytes,
            ));
        }

        // Execute request through NetworkService
        self.complete_calls += 1;
        let response = match self.service.request(&net_req) {
            Ok(res) => res,
            Err(net_err) => {
                return Err(map_network_error(
                    &self.config.provider_id,
                    net_err,
                    request.timeout_ms,
                    request.budget_bytes,
                ));
            }
        };

        // Handle HTTP response statuses.
        //
        // Status mapping table (pinned by tests):
        // | status              | ProviderError   |
        // |---------------------|-----------------|
        // | 401, 403            | Auth            |
        // | 429                 | RateLimited     |
        // | 404, 503            | ModelUnavailable|
        // | other non-2xx       | Transport       |
        //
        // `Retry-After` (seconds, per HTTP) converts to ms when present on a
        // 429; an absent or unparseable value yields `retry_after_ms: None`.
        if response.status == 401 || response.status == 403 {
            return Err(ProviderError::Auth {
                provider: self.config.provider_id.clone(),
                reason: "authentication refused by provider".to_owned(),
            });
        }
        if response.status == 429 {
            let retry_after_ms = response
                .header("retry-after")
                .and_then(|h| h.trim().parse::<u64>().ok())
                .map(|sec| sec.saturating_mul(1000));
            return Err(ProviderError::RateLimited {
                provider: self.config.provider_id.clone(),
                retry_after_ms,
            });
        }
        if response.status == 404 || response.status == 503 {
            return Err(ProviderError::ModelUnavailable {
                provider: self.config.provider_id.clone(),
                model: request.model.clone(),
            });
        }
        if !response.is_success() {
            return Err(ProviderError::Transport {
                provider: self.config.provider_id.clone(),
                reason: format!("bad http status: {}", response.status),
            });
        }

        // The 2xx parse path expects a JSON body: the response
        // `Content-Type` must be `application/json` (case-insensitive,
        // parameters such as `; charset=utf-8` allowed, mirroring
        // `local_provider`). Anything else fails closed instead of parsing
        // an untrusted non-JSON body as model output.
        if !is_json_content_type(response.header("content-type")) {
            return Err(ProviderError::Transport {
                provider: self.config.provider_id.clone(),
                reason: "response content-type is not application/json".to_owned(),
            });
        }

        // Response body ceiling check (single-fragment ceiling from stream module)
        if response.body.len() > MAX_FRAGMENT_BYTES {
            return Err(ProviderError::Transport {
                provider: self.config.provider_id.clone(),
                reason: format!(
                    "response body exceeds fragment ceiling (max {MAX_FRAGMENT_BYTES})"
                ),
            });
        }

        // Parse JSON response body
        parse_response_body(&self.config.provider_id, &response)
    }

    fn scripted_turns_remaining(&self) -> usize {
        0
    }

    fn complete_calls(&self) -> u64 {
        self.complete_calls
    }
}

/// Build the OpenAI-compatible chat body plus mapped sampling.
///
/// `Role::Tool` observations are folded into `user` messages with a `[tool] `
/// prefix (aligned with `local_provider`), so no tool protocol is required
/// and the untrusted surface stays labeled in the text itself.
///
/// Sampling mapped-field table (mirrors `local_provider`; only `Some`
/// fields emit, absent stays absent and is never defaulted):
/// | `SamplingParams` field | body field          | carried |
/// |------------------------|---------------------|---------|
/// | `temperature`          | `"temperature"`     | yes     |
/// | `top_p`                | `"top_p"`           | yes     |
/// | `frequency_penalty`    | `"frequency_penalty"`| yes    |
/// | `presence_penalty`     | `"presence_penalty"`| yes     |
/// | `seed`                 | `"seed"`            | yes     |
/// | `max_tokens`           | `"max_tokens"`      | yes     |
/// | `stop`                 | `"stop"`            | yes     |
/// | `top_k`                | —                   | no (rejected: `UnsupportedSampling`) |
/// | `repetition_penalty`   | —                   | no (rejected: `UnsupportedSampling`) |
/// | `min_p`                | —                   | no (rejected: `UnsupportedSampling`) |
/// | `response_format`      | —                   | no (rejected: `UnsupportedSampling`) |
/// | `reasoning`            | —                   | no (rejected: `UnsupportedSampling`) |
///
/// `max_tokens` is the declared completion-token field for the backend to
/// interpret; it is not client-side enforcement, and the separate
/// response-byte ceiling is an unrelated transport bound.
fn build_request_body(model: &str, request: &TurnRequest) -> Result<Vec<u8>, ProviderError> {
    let mut message_list = Vec::with_capacity(request.messages.len());
    for msg in &request.messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user",
        };
        // `[tool] ` prefix keeps tool observations labeled in the text
        // itself (aligned with `local_provider`); see `role_name` there.
        let content = if matches!(msg.role, Role::Tool) {
            format!("[tool] {}", msg.content)
        } else {
            msg.content.clone()
        };
        message_list.push(serde_json::json!({
            "role": role_str,
            "content": content,
        }));
    }

    let mut payload = serde_json::Map::new();
    payload.insert(
        "model".to_owned(),
        serde_json::Value::String(model.to_owned()),
    );
    payload.insert(
        "messages".to_owned(),
        serde_json::Value::Array(message_list),
    );

    if let Some(ref sampling) = request.sampling {
        if let Some(temp) = sampling.temperature {
            payload.insert("temperature".to_owned(), serde_json::json!(temp));
        }
        if let Some(top_p) = sampling.top_p {
            payload.insert("top_p".to_owned(), serde_json::json!(top_p));
        }
        if let Some(frequency_penalty) = sampling.frequency_penalty {
            payload.insert(
                "frequency_penalty".to_owned(),
                serde_json::json!(frequency_penalty),
            );
        }
        if let Some(presence_penalty) = sampling.presence_penalty {
            payload.insert(
                "presence_penalty".to_owned(),
                serde_json::json!(presence_penalty),
            );
        }
        if let Some(seed) = sampling.seed {
            payload.insert("seed".to_owned(), serde_json::json!(seed));
        }
        if let Some(max_tokens) = sampling.max_tokens {
            payload.insert("max_tokens".to_owned(), serde_json::json!(max_tokens));
        }
        if let Some(ref stop) = sampling.stop {
            payload.insert("stop".to_owned(), serde_json::json!(stop));
        }
    }

    serde_json::to_vec(&serde_json::Value::Object(payload)).map_err(|e| ProviderError::Transport {
        provider: "json".to_owned(),
        reason: format!("payload serialization error: {e}"),
    })
}

/// Reject declared sampling fields this adapter does not carry.
///
/// Mirrors `local_provider::check_supported_sampling`: validation support
/// does not imply backend support, so a valid declaration still refuses
/// before any I/O when the backend has no explicit mapping for it. The
/// label is a static field name, never caller input.
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

/// Whether `content_type` authorizes the 2xx JSON parse path.
///
/// Accepts `application/json` (case-insensitive, optionally with `; ...`
/// parameters, e.g. `application/json; charset=utf-8`), mirroring
/// `local_provider::is_json_content_type`.
fn is_json_content_type(content_type: Option<&str>) -> bool {
    match content_type {
        Some(value) => value.to_ascii_lowercase().contains("application/json"),
        None => false,
    }
}

fn parse_response_body(
    provider_id: &str,
    response: &Response,
) -> Result<ProviderTurn, ProviderError> {
    let value: serde_json::Value =
        serde_json::from_slice(&response.body).map_err(|e| ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: format!("failed to parse response JSON: {e}"),
        })?;

    let mut tool_calls = Vec::new();
    if let Some(calls) = value
        .pointer("/choices/0/message/tool_calls")
        .and_then(|c| c.as_array())
    {
        for call in calls {
            if let Some(name) = call.pointer("/function/name").and_then(|n| n.as_str()) {
                let arguments = if let Some(arg_str) =
                    call.pointer("/function/arguments").and_then(|a| a.as_str())
                {
                    arg_str.as_bytes().to_vec()
                } else if let Some(arg_obj) = call.pointer("/function/arguments") {
                    serde_json::to_vec(arg_obj).unwrap_or_default()
                } else {
                    Vec::new()
                };
                tool_calls.push(ToolCallRequest {
                    name: name.to_owned(),
                    arguments,
                });
            }
        }
    }

    let text = if let Some(content) = value
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
    {
        content.to_owned()
    } else if let Some(content) = value.pointer("/message/content").and_then(|c| c.as_str()) {
        content.to_owned()
    } else if let Some(resp) = value.get("response").and_then(|r| r.as_str()) {
        resp.to_owned()
    } else if !tool_calls.is_empty() {
        String::new()
    } else {
        return Err(ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: "response JSON has no recognizable assistant content".to_owned(),
        });
    };

    let input_tokens = value
        .pointer("/usage/prompt_tokens")
        .or_else(|| value.pointer("/usage/input_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0) as u32;

    let output_tokens = value
        .pointer("/usage/completion_tokens")
        .or_else(|| value.pointer("/usage/output_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0) as u32;

    Ok(ProviderTurn {
        text,
        tool_calls,
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    })
}

/// Map one [`NetworkError`] onto the provider taxonomy (no new kinds).
///
/// Mapping table:
/// | `NetworkError`      | `ProviderError`                          |
/// |---------------------|--------------------------------------------|
/// | `Offline`           | `Transport` ("network offline")            |
/// | `Denied { domain }` | `Transport` ("network access denied ...")|
/// | `Timeout { after }` | `Timeout` (caller timeout + observed ms) |
/// | `Budget`            | `Unknown` (transfer budget, MP-7)          |
/// | `CountBudget`       | `Unknown` (transfer budget, MP-7)          |
/// | `Tls { reason }`    | `Transport` ("network tls refused: ...")   |
///
/// Transfer budgets (`Budget`, `CountBudget`) never map to
/// [`ProviderError::BudgetExceeded`]: that variant stays context-only
/// (`CP-5`, the pre-I/O caller-budget gate in `complete`). A transfer
/// budget fires post-send — the request may have been applied while the
/// acknowledgement was withheld — so both map to [`ProviderError::Unknown`]
/// (`MP-7`, reconcile before retry, never blind fallback). `Unknown` is
/// reserved for exactly this post-send truncation shape.
fn map_network_error(
    provider_id: &str,
    err: NetworkError,
    requested_timeout_ms: u64,
    _budget_bytes: usize,
) -> ProviderError {
    match err {
        NetworkError::Offline => ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: "network offline".to_owned(),
        },
        NetworkError::Denied { domain } => ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: format!("network access denied for {domain}"),
        },
        NetworkError::Timeout { after } => ProviderError::Timeout {
            timeout_ms: requested_timeout_ms,
            latency_ms: after.as_millis() as u64,
        },
        NetworkError::Budget { limit_bytes } => ProviderError::Unknown {
            provider: provider_id.to_owned(),
            reason: format!("network body budget exceeded: limit was {limit_bytes} bytes"),
        },
        NetworkError::CountBudget { limit_items } => ProviderError::Unknown {
            provider: provider_id.to_owned(),
            reason: format!("network count budget exceeded: limit was {limit_items} items"),
        },
        NetworkError::Tls { reason } => ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: format!("network tls refused: {reason}"),
        },
    }
}

/// An in-memory [`NetworkService`] recording outgoing requests and mocking responses.
#[derive(Default)]
pub struct RecordingNetworkService {
    requests: Mutex<Vec<Request>>,
    responses: Mutex<Vec<Result<Response, NetworkError>>>,
}

impl RecordingNetworkService {
    /// Create a new recording network service with empty queues.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue an expected HTTP response.
    pub fn queue_response(&self, response: Response) {
        if let Ok(mut lock) = self.responses.lock() {
            lock.push(Ok(response));
        }
    }

    /// Enqueue an expected network error.
    pub fn queue_error(&self, error: NetworkError) {
        if let Ok(mut lock) = self.responses.lock() {
            lock.push(Err(error));
        }
    }

    /// Return all recorded requests.
    pub fn recorded_requests(&self) -> Vec<Request> {
        self.requests.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// Return the last recorded request, if any.
    pub fn last_request(&self) -> Option<Request> {
        self.requests.lock().ok().and_then(|l| l.last().cloned())
    }

    /// Return count of recorded requests.
    pub fn recorded_count(&self) -> usize {
        self.requests.lock().map(|l| l.len()).unwrap_or(0)
    }
}

impl fmt::Debug for RecordingNetworkService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.recorded_count();
        f.debug_struct("RecordingNetworkService")
            .field("recorded_count", &count)
            .finish()
    }
}

impl NetworkService for RecordingNetworkService {
    type Socket = ();

    fn request(&self, request: &Request) -> Result<Response, NetworkError> {
        if let Ok(mut lock) = self.requests.lock() {
            lock.push(request.clone());
        }
        if let Ok(mut lock) = self.responses.lock() {
            if lock.is_empty() {
                Err(NetworkError::Offline)
            } else {
                lock.remove(0)
            }
        } else {
            Err(NetworkError::Offline)
        }
    }

    fn websocket(&self, _request: &WebSocketRequest) -> Result<Self::Socket, NetworkError> {
        Err(NetworkError::Offline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_ai_runtime::provider::Message;
    use bitty_ai_runtime::{ReasoningConfig, ResponseFormat};

    fn sample_turn_request(model: &str, budget: usize, timeout_ms: u64) -> TurnRequest {
        TurnRequest {
            model: model.to_owned(),
            messages: vec![Message::user("Hello, world!")],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: budget,
            timeout_ms,
            now_ms: 1000,
            sampling: None,
        }
    }

    fn sample_openai_response(content: &str) -> Response {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8
            }
        });
        Response {
            status: 200,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body: serde_json::to_vec(&body).unwrap(),
        }
    }

    #[test]
    fn maps_budget_and_timeout_into_request() {
        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("I am ready."));

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let turn_req = sample_turn_request("test-model", 4096, 7500);

        let res = adapter.complete(&turn_req).expect("successful turn");
        assert_eq!(res.text, "I am ready.");
        assert_eq!(res.usage.input_tokens, 12);
        assert_eq!(res.usage.output_tokens, 8);

        let recorded = adapter.service().last_request().expect("recorded request");
        assert_eq!(recorded.max_body_bytes, Some(4096));
        assert_eq!(recorded.timeout, Some(Duration::from_millis(7500)));
        assert_eq!(recorded.method, bitty_network_api::HttpMethod::Post);
        assert_eq!(recorded.url, "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn pre_io_budget_exceeded_fails_closed_without_network_call() {
        let service = RecordingNetworkService::new();
        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        // Turn request with message exceeding budget
        let turn_req = sample_turn_request("test-model", 5, 5000);

        let err = adapter.complete(&turn_req).unwrap_err();
        match err {
            ProviderError::BudgetExceeded { limit, actual } => {
                assert_eq!(limit, 5);
                assert!(actual > limit);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }

        assert_eq!(adapter.service().recorded_count(), 0);
    }

    #[test]
    fn pre_io_timeout_validation_fails_closed() {
        let service = RecordingNetworkService::new();
        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);

        // Zero timeout fails
        let mut zero_req = sample_turn_request("test-model", 4096, 0);
        let err0 = adapter.complete(&zero_req).unwrap_err();
        assert!(matches!(err0, ProviderError::Transport { .. }));

        // Timeout too large fails
        zero_req.timeout_ms = MAX_REQUEST_TIMEOUT_MS + 1;
        let err_large = adapter.complete(&zero_req).unwrap_err();
        assert!(matches!(err_large, ProviderError::TimeoutTooLarge { .. }));

        assert_eq!(adapter.service().recorded_count(), 0);
    }

    #[test]
    fn secret_token_injected_at_adapter_edge_with_strict_redaction() {
        let canary_secret = b"sk-live-super-canary-secret-value-xyz987";
        let secret = SecretField::new(canary_secret.to_vec()).unwrap();

        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("Authorized response"));

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_api_key(secret)
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        // Adapter Debug check: secret value MUST NOT leak
        let adapter_debug = format!("{config:?}");
        assert!(!adapter_debug.contains("canary"));
        assert!(adapter_debug.contains("[redacted]"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let full_adapter_debug = format!("{adapter:?}");
        assert!(!full_adapter_debug.contains("canary"));

        let turn_req = sample_turn_request("test-model", 4096, 5000);
        let turn = adapter.complete(&turn_req).expect("successful turn");
        assert_eq!(turn.text, "Authorized response");

        // Inspect recorded network request
        let recorded = adapter.service().last_request().expect("recorded request");

        // The actual header holds the bearer token
        let auth_header = recorded
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .expect("authorization header present");
        assert_eq!(
            auth_header.1,
            "Bearer sk-live-super-canary-secret-value-xyz987"
        );

        // Crucial Redaction Check: Request's Debug impl MUST redact headers
        let req_debug = format!("{recorded:?}");
        assert!(!req_debug.contains("canary"));
        assert!(req_debug.contains("[redacted]"));

        // Error-path redaction: the response queue is now drained, so the
        // next turn fails with a transport error whose Display/Debug must
        // also carry no secret material.
        let err = adapter.complete(&turn_req).unwrap_err();
        let err_display = format!("{err}");
        let err_debug = format!("{err:?}");
        assert!(!err_display.contains("canary"));
        assert!(!err_debug.contains("canary"));
    }

    #[test]
    fn maps_network_errors_to_typed_provider_errors() {
        let service = RecordingNetworkService::new();
        // 1. Offline
        service.queue_error(NetworkError::Offline);
        // 2. Denied
        service.queue_error(NetworkError::Denied {
            domain: "untrusted.org".to_owned(),
        });
        // 3. Timeout
        service.queue_error(NetworkError::Timeout {
            after: Duration::from_millis(1500),
        });
        // 4. Budget
        service.queue_error(NetworkError::Budget { limit_bytes: 4096 });
        // 5. CountBudget
        service.queue_error(NetworkError::CountBudget { limit_items: 8 });

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let req = sample_turn_request("test-model", 4096, 5000);

        // 1. Offline
        let err1 = adapter.complete(&req).unwrap_err();
        match err1 {
            ProviderError::Transport { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert_eq!(reason, "network offline");
            }
            other => panic!("expected Transport, got {other:?}"),
        }

        // 2. Denied
        let err2 = adapter.complete(&req).unwrap_err();
        match err2 {
            ProviderError::Transport { reason, .. } => {
                assert!(reason.contains("untrusted.org"));
            }
            other => panic!("expected Transport, got {other:?}"),
        }

        // 3. Timeout
        let err3 = adapter.complete(&req).unwrap_err();
        match err3 {
            ProviderError::Timeout {
                timeout_ms,
                latency_ms,
            } => {
                assert_eq!(timeout_ms, 5000);
                assert_eq!(latency_ms, 1500);
            }
            other => panic!("expected Timeout, got {other:?}"),
        }

        // 4. Budget
        let err4 = adapter.complete(&req).unwrap_err();
        match &err4 {
            ProviderError::Unknown { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert!(reason.contains("network body budget exceeded"));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
        assert_eq!(
            bitty_ai_runtime::selection::fallback_directive(&err4),
            bitty_ai_runtime::selection::FallbackDirective::Stop
        );

        // 5. CountBudget: same post-send truncation shape as Budget, so the
        // same Unknown mapping (never BudgetExceeded, which stays
        // context-only per CP-5).
        let err5 = adapter.complete(&req).unwrap_err();
        match &err5 {
            ProviderError::Unknown { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert!(reason.contains("network count budget exceeded"));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
        assert!(!matches!(err5, ProviderError::BudgetExceeded { .. }));
        assert_eq!(
            bitty_ai_runtime::selection::fallback_directive(&err5),
            bitty_ai_runtime::selection::FallbackDirective::Stop
        );
    }

    #[test]
    fn maps_http_error_statuses_to_typed_provider_errors() {
        let service = RecordingNetworkService::new();
        // 401 Unauthorized
        service.queue_response(Response {
            status: 401,
            headers: Vec::new(),
            body: b"{\"error\":\"invalid_key\"}".to_vec(),
        });
        // 429 RateLimited with retry-after
        service.queue_response(Response {
            status: 429,
            headers: vec![("retry-after".to_owned(), "10".to_owned())],
            body: b"{\"error\":\"rate_limited\"}".to_vec(),
        });
        // 503 ModelUnavailable
        service.queue_response(Response {
            status: 503,
            headers: Vec::new(),
            body: b"{\"error\":\"service_unavailable\"}".to_vec(),
        });

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let req = sample_turn_request("test-model", 4096, 5000);

        // 401
        let err_auth = adapter.complete(&req).unwrap_err();
        assert!(matches!(err_auth, ProviderError::Auth { .. }));

        // 429
        let err_rate = adapter.complete(&req).unwrap_err();
        match err_rate {
            ProviderError::RateLimited {
                provider,
                retry_after_ms,
            } => {
                assert_eq!(provider, "test-provider");
                assert_eq!(retry_after_ms, Some(10_000));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }

        // 503
        let err_unavail = adapter.complete(&req).unwrap_err();
        assert!(matches!(
            err_unavail,
            ProviderError::ModelUnavailable { .. }
        ));
    }

    #[test]
    fn parses_tool_calls_from_response() {
        let response_body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"location\":\"Paris\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20
            }
        });

        let service = RecordingNetworkService::new();
        service.queue_response(Response {
            status: 200,
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body: serde_json::to_vec(&response_body).unwrap(),
        });

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let req = sample_turn_request("test-model", 4096, 5000);

        let turn = adapter.complete(&req).expect("successful tool turn");
        assert_eq!(turn.text, "");
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "lookup_weather");
        assert_eq!(
            std::str::from_utf8(&turn.tool_calls[0].arguments).unwrap(),
            "{\"location\":\"Paris\"}"
        );
    }

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

    #[test]
    fn capability_gate_defaults_offline_and_fails_closed_with_zero_calls() {
        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("Must never send."));

        // No capability grant: the deny-all default refuses before any I/O.
        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap();

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let req = sample_turn_request("test-model", 4096, 5000);
        let err = adapter.complete(&req).unwrap_err();
        match err {
            ProviderError::Transport { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert_eq!(reason, "network offline");
            }
            other => panic!("expected offline Transport, got {other:?}"),
        }
        assert_eq!(adapter.service().recorded_count(), 0);
        assert_eq!(adapter.complete_calls(), 0);
    }

    #[test]
    fn capability_gate_denied_domain_fails_closed_with_zero_calls() {
        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("Must never send."));

        // Grant covers another domain only, so this endpoint is denied.
        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("other.example"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let req = sample_turn_request("test-model", 4096, 5000);
        let err = adapter.complete(&req).unwrap_err();
        match err {
            ProviderError::Transport { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert!(reason.contains("api.example.com"));
            }
            other => panic!("expected denied Transport, got {other:?}"),
        }
        assert_eq!(adapter.service().recorded_count(), 0);
        assert_eq!(adapter.complete_calls(), 0);
    }

    #[test]
    fn unsupported_sampling_fields_fail_closed_pre_io() {
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
            effort: None,
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

        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("Must never send."));

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        for (sampling, field) in cases {
            let mut req = sample_turn_request("test-model", 4096, 5000);
            req.sampling = Some(sampling);
            let err = adapter.complete(&req).unwrap_err();
            assert_eq!(err, ProviderError::UnsupportedSampling { field });
            assert_eq!(
                bitty_ai_runtime::selection::fallback_directive(&err),
                bitty_ai_runtime::selection::FallbackDirective::Stop
            );
        }
        assert_eq!(adapter.service().recorded_count(), 0);
        assert_eq!(adapter.complete_calls(), 0);
    }

    #[test]
    fn supported_sampling_fields_map_into_body() {
        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("Mapped."));

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let mut req = sample_turn_request("test-model", 4096, 5000);
        let mut sampling = blank_sampling();
        sampling.temperature = Some(0.7);
        sampling.top_p = Some(0.9);
        sampling.frequency_penalty = Some(0.1);
        sampling.presence_penalty = Some(0.2);
        sampling.seed = Some(42);
        sampling.max_tokens = Some(128);
        sampling.stop = Some(vec!["END".to_owned()]);
        req.sampling = Some(sampling);

        let turn = adapter.complete(&req).expect("mapped sampling succeeds");
        assert_eq!(turn.text, "Mapped.");

        let recorded = adapter.service().last_request().expect("recorded request");
        let body: serde_json::Value =
            serde_json::from_slice(&recorded.body).expect("request body is JSON");
        assert_eq!(body["temperature"], serde_json::json!(0.7));
        assert_eq!(body["top_p"], serde_json::json!(0.9));
        assert_eq!(body["frequency_penalty"], serde_json::json!(0.1));
        assert_eq!(body["presence_penalty"], serde_json::json!(0.2));
        assert_eq!(body["seed"], serde_json::json!(42));
        assert_eq!(body["max_tokens"], serde_json::json!(128));
        assert_eq!(body["stop"], serde_json::json!(["END"]));
    }

    #[test]
    fn tool_role_folds_into_user_with_prefix() {
        let service = RecordingNetworkService::new();
        service.queue_response(sample_openai_response("Observed."));

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let mut req = sample_turn_request("test-model", 4096, 5000);
        req.messages.push(Message::tool("file contents here"));

        let turn = adapter.complete(&req).expect("tool message succeeds");
        assert_eq!(turn.text, "Observed.");

        let recorded = adapter.service().last_request().expect("recorded request");
        let body: serde_json::Value =
            serde_json::from_slice(&recorded.body).expect("request body is JSON");
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], serde_json::json!("user"));
        assert_eq!(
            messages[1]["content"],
            serde_json::json!("[tool] file contents here")
        );
    }

    #[test]
    fn non_json_content_type_fails_closed_on_2xx() {
        let service = RecordingNetworkService::new();
        // 200 with a non-JSON content type must not parse.
        service.queue_response(Response {
            status: 200,
            headers: vec![("content-type".to_owned(), "text/plain".to_owned())],
            body: b"plain text, not model output".to_vec(),
        });
        // 200 with no content type at all must not parse either.
        service.queue_response(Response {
            status: 200,
            headers: Vec::new(),
            body: b"{}".to_vec(),
        });
        // 200 with JSON parameters on the content type still parses.
        let ok_body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "Parametric."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        });
        service.queue_response(Response {
            status: 200,
            headers: vec![(
                "content-type".to_owned(),
                "application/json; charset=utf-8".to_owned(),
            )],
            body: serde_json::to_vec(&ok_body).unwrap(),
        });

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap()
        .with_capability(NetworkCapability::offline().with_domain("api.example.com"));

        let mut adapter = NetworkConsumerAdapter::new(config, service);
        let req = sample_turn_request("test-model", 4096, 5000);

        let err_plain = adapter.complete(&req).unwrap_err();
        match err_plain {
            ProviderError::Transport { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert_eq!(reason, "response content-type is not application/json");
            }
            other => panic!("expected content-type Transport, got {other:?}"),
        }

        let err_missing = adapter.complete(&req).unwrap_err();
        assert!(matches!(err_missing, ProviderError::Transport { .. }));

        let turn = adapter.complete(&req).expect("JSON with parameters parses");
        assert_eq!(turn.text, "Parametric.");
    }
}

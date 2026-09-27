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
//! - Enforces pre-I/O context-budget, model, and timeout gates before any network call.
//! - Maps typed [`bitty_network_api::NetworkError`] outcomes to typed [`ProviderError`]s.

use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use bitty_ai_runtime::prompt::MAX_CANONICAL_BYTES;
use bitty_ai_runtime::provider::{
    MAX_REQUEST_TIMEOUT_MS, ModelDescriptor, ModelProvider, ProviderError, ProviderTurn,
    ProviderUsage, Role, ToolCallRequest, TurnRequest, validate_provider_id, validate_sampling,
};
use bitty_ai_runtime::secret::SecretField;
use bitty_ai_runtime::selection::validate_model_name;
use bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES;
use bitty_network_api::{NetworkError, NetworkService, Request, Response, WebSocketRequest};

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

        // Pre-I/O Gate 3: Sampling validation
        if let Some(ref params) = request.sampling {
            validate_sampling(params)?;
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

        // Handle HTTP response statuses
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
        parse_response_body(&self.config.provider_id, &response.body)
    }

    fn scripted_turns_remaining(&self) -> usize {
        0
    }

    fn complete_calls(&self) -> u64 {
        self.complete_calls
    }
}

fn build_request_body(model: &str, request: &TurnRequest) -> Result<Vec<u8>, ProviderError> {
    let mut message_list = Vec::with_capacity(request.messages.len());
    for msg in &request.messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user",
        };
        message_list.push(serde_json::json!({
            "role": role_str,
            "content": msg.content,
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
        if let Some(max_tokens) = sampling.max_tokens {
            payload.insert("max_tokens".to_owned(), serde_json::json!(max_tokens));
        }
    }

    serde_json::to_vec(&serde_json::Value::Object(payload)).map_err(|e| ProviderError::Transport {
        provider: "json".to_owned(),
        reason: format!("payload serialization error: {e}"),
    })
}

fn parse_response_body(provider_id: &str, body: &[u8]) -> Result<ProviderTurn, ProviderError> {
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| ProviderError::Transport {
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
        NetworkError::Budget { limit_bytes } => ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: format!("network body budget exceeded: limit was {limit_bytes} bytes"),
        },
        NetworkError::CountBudget { limit_items } => ProviderError::Transport {
            provider: provider_id.to_owned(),
            reason: format!("network count budget exceeded: {limit_items}"),
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
        .unwrap();

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
        .unwrap();

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
        .unwrap();

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
        .with_api_key(secret);

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

        let config = NetworkConsumerAdapterConfig::new(
            "test-provider",
            "test-model",
            "https://api.example.com/v1/chat/completions",
        )
        .unwrap();

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
        match err4 {
            ProviderError::Transport { provider, reason } => {
                assert_eq!(provider, "test-provider");
                assert!(reason.contains("network body budget exceeded"));
            }
            other => panic!("expected Transport, got {other:?}"),
        }
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
        .unwrap();

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
        .unwrap();

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
}

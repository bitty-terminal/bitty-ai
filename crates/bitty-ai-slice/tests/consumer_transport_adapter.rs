//! Integration test for the `bitty-network-api` consumer transport adapter (AI-0156).
//!
//! Validates the end-to-end integration between `bitty_ai_runtime::Agent` and
//! `bitty_ai_slice::NetworkConsumerAdapter`, verifying:
//! 1. `TurnRequest.budget_bytes` correctly maps to `Request::max_body_bytes`.
//! 2. `TurnRequest.timeout_ms` correctly maps to `Request::timeout`.
//! 3. `SecretField` authentication is injected strictly at the host adapter edge
//!    without leaking into `Debug`, `Display`, or error descriptions.
//! 4. Typed `NetworkError` variants map into typed `ProviderError` variants.
//! 5. Agent run loop drives complete turns through the consumer transport adapter.

use std::time::Duration;

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, FakeToolExecutor,
    ModelProvider, ProviderError, SecretField, ToolAuthorizer, ToolBus, ToolRegistry, TurnRequest,
    VecSink, provider::Message,
};
use bitty_ai_slice::{
    NetworkConsumerAdapter, NetworkConsumerAdapterConfig, RecordingNetworkService,
};
use bitty_network_api::{HttpMethod, NetworkCapability, NetworkError, Response, TlsFailure};

const NOW_MS: u64 = 1_700_000_000_000;

struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

fn test_session() -> bitty_ai_runtime::AgentSession {
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    bitty_ai_runtime::AgentSession::new(issuer.agent_instance(), issuer.run(), issuer.session())
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
            "prompt_tokens": 25,
            "completion_tokens": 15
        }
    });
    Response {
        status: 200,
        headers: vec![("content-type".to_owned(), "application/json".to_owned())],
        body: serde_json::to_vec(&body).unwrap(),
    }
}

#[test]
fn agent_drives_consumer_transport_adapter_end_to_end() {
    let canary_key = b"sk-live-canary-secret-credential-value-99999";
    let secret = SecretField::new(canary_key.to_vec()).unwrap();

    let service = RecordingNetworkService::new();
    service.queue_response(sample_openai_response("Task completed successfully."));

    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_api_key(secret)
    .with_header("x-client-name", "bitty-ai-slice")
    .with_capability(NetworkCapability::offline().with_domain("api.openai.com"));

    let adapter = NetworkConsumerAdapter::new(config, service);

    let agent_config = AgentConfig {
        context_budget_bytes: 8192,
        provider_timeout_ms: 12_000,
        ..Default::default()
    };

    let session = test_session();
    let tool_bus = ToolBus::new(ToolRegistry::new()).with_authorizer(AllowAll);
    let mut agent = Agent::new(adapter, tool_bus, session, agent_config);

    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::default();

    let outcome = agent.run_turn(
        &mut executor,
        "gpt-4o",
        "Calculate the sum of 40 and 2.",
        &[],
        &mut sink,
        NOW_MS,
    );
    match outcome {
        ExecOutcome::Completed { text } => {
            assert_eq!(text, "Task completed successfully.");
        }
        other => panic!("expected Completed, got {other:?}"),
    }

    assert_eq!(agent.provider_mut().complete_calls(), 1);

    // Verify emitted text
    assert_eq!(
        std::str::from_utf8(&sink.concatenated_bytes()).unwrap(),
        "Task completed successfully."
    );

    // Inspect the recorded network request
    let recorded = agent
        .provider_mut()
        .service()
        .last_request()
        .expect("request recorded");
    assert_eq!(recorded.method, HttpMethod::Post);
    assert_eq!(recorded.url, "https://api.openai.com/v1/chat/completions");
    assert_eq!(recorded.timeout, Some(Duration::from_millis(12_000)));
    assert_eq!(recorded.max_body_bytes, Some(8192));

    // Check authorization header was injected at the adapter edge
    let auth = recorded
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
        .expect("authorization header injected");
    assert_eq!(
        auth.1,
        "Bearer sk-live-canary-secret-credential-value-99999"
    );

    // Verify secret redaction: neither Debug of request nor Debug of adapter leaks canary
    let req_dbg = format!("{recorded:?}");
    assert!(!req_dbg.contains("canary"));
    assert!(req_dbg.contains("[redacted]"));

    let adapter_dbg = format!("{:?}", agent.provider_mut());
    assert!(!adapter_dbg.contains("canary"));
    assert!(adapter_dbg.contains("[redacted]"));
}

#[test]
fn agent_turn_fails_closed_on_network_offline() {
    let service = RecordingNetworkService::new();
    service.queue_error(NetworkError::Offline);

    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_capability(NetworkCapability::offline().with_domain("api.openai.com"));

    let adapter = NetworkConsumerAdapter::new(config, service);
    let agent_config = AgentConfig::default();

    let session = test_session();
    let tool_bus = ToolBus::new(ToolRegistry::new()).with_authorizer(AllowAll);
    let mut agent = Agent::new(adapter, tool_bus, session, agent_config);

    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::default();

    let outcome = agent.run_turn(&mut executor, "gpt-4o", "test", &[], &mut sink, NOW_MS);
    match outcome {
        ExecOutcome::Failed {
            error: AgentError::Provider(ProviderError::Transport { provider, reason }),
        } => {
            assert_eq!(provider, "mock-openai");
            assert_eq!(reason, "network offline");
        }
        other => panic!("expected Transport error on offline, got {other:?}"),
    }
}

#[test]
fn agent_turn_fails_closed_on_network_denied() {
    let service = RecordingNetworkService::new();
    service.queue_error(NetworkError::Denied {
        domain: "unauthorized.api".to_owned(),
    });

    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_capability(NetworkCapability::offline().with_domain("api.openai.com"));

    let adapter = NetworkConsumerAdapter::new(config, service);
    let agent_config = AgentConfig::default();

    let session = test_session();
    let tool_bus = ToolBus::new(ToolRegistry::new()).with_authorizer(AllowAll);
    let mut agent = Agent::new(adapter, tool_bus, session, agent_config);

    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::default();

    let outcome = agent.run_turn(&mut executor, "gpt-4o", "test", &[], &mut sink, NOW_MS);
    match outcome {
        ExecOutcome::Failed {
            error: AgentError::Provider(ProviderError::Transport { reason, .. }),
        } => {
            assert!(reason.contains("unauthorized.api"));
        }
        other => panic!("expected Transport error on denied, got {other:?}"),
    }
}

#[test]
fn agent_turn_maps_network_timeout() {
    let service = RecordingNetworkService::new();
    service.queue_error(NetworkError::Timeout {
        after: Duration::from_millis(8500),
    });

    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_capability(NetworkCapability::offline().with_domain("api.openai.com"));

    let adapter = NetworkConsumerAdapter::new(config, service);
    let agent_config = AgentConfig {
        provider_timeout_ms: 10_000,
        ..Default::default()
    };

    let session = test_session();
    let tool_bus = ToolBus::new(ToolRegistry::new()).with_authorizer(AllowAll);
    let mut agent = Agent::new(adapter, tool_bus, session, agent_config);

    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::default();

    let outcome = agent.run_turn(&mut executor, "gpt-4o", "test", &[], &mut sink, NOW_MS);
    match outcome {
        ExecOutcome::Failed {
            error:
                AgentError::Provider(ProviderError::Timeout {
                    timeout_ms,
                    latency_ms,
                }),
        } => {
            assert_eq!(timeout_ms, 10_000);
            assert_eq!(latency_ms, 8500);
        }
        other => panic!("expected Timeout error, got {other:?}"),
    }
}

#[test]
fn agent_turn_maps_tls_failure() {
    let service = RecordingNetworkService::new();
    service.queue_error(NetworkError::Tls {
        reason: TlsFailure::CaRootRejected,
    });

    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_capability(NetworkCapability::offline().with_domain("api.openai.com"));

    let adapter = NetworkConsumerAdapter::new(config, service);
    let agent_config = AgentConfig::default();

    let session = test_session();
    let tool_bus = ToolBus::new(ToolRegistry::new()).with_authorizer(AllowAll);
    let mut agent = Agent::new(adapter, tool_bus, session, agent_config);

    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::default();

    let outcome = agent.run_turn(&mut executor, "gpt-4o", "test", &[], &mut sink, NOW_MS);
    match outcome {
        ExecOutcome::Failed {
            error: AgentError::Provider(ProviderError::Transport { reason, .. }),
        } => {
            assert!(reason.contains("ca root rejected"));
        }
        other => panic!("expected TLS failure transport error, got {other:?}"),
    }
}

fn granted_config() -> NetworkConsumerAdapterConfig {
    NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_capability(NetworkCapability::offline().with_domain("api.openai.com"))
}

fn run_one_turn(agent: &mut Agent<NetworkConsumerAdapter<RecordingNetworkService>>) -> ExecOutcome {
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::default();
    agent.run_turn(&mut executor, "gpt-4o", "test", &[], &mut sink, NOW_MS)
}

fn agent_with(
    adapter: NetworkConsumerAdapter<RecordingNetworkService>,
) -> Agent<NetworkConsumerAdapter<RecordingNetworkService>> {
    let agent_config = AgentConfig::default();
    let session = test_session();
    let tool_bus = ToolBus::new(ToolRegistry::new()).with_authorizer(AllowAll);
    Agent::new(adapter, tool_bus, session, agent_config)
}

#[test]
fn agent_turn_fails_closed_on_default_offline_capability() {
    let service = RecordingNetworkService::new();
    service.queue_response(sample_openai_response("Must never send."));

    // No capability grant: the deny-all default refuses before any I/O.
    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap();
    let adapter = NetworkConsumerAdapter::new(config, service);
    let mut agent = agent_with(adapter);

    let outcome = run_one_turn(&mut agent);
    match outcome {
        ExecOutcome::Failed {
            error: AgentError::Provider(ProviderError::Transport { provider, reason }),
        } => {
            assert_eq!(provider, "mock-openai");
            assert_eq!(reason, "network offline");
        }
        other => panic!("expected offline Transport, got {other:?}"),
    }
    assert_eq!(agent.provider_mut().service().recorded_count(), 0);
    assert_eq!(agent.provider_mut().complete_calls(), 0);
}

#[test]
fn agent_turn_fails_closed_on_capability_denied_pre_io() {
    let service = RecordingNetworkService::new();
    service.queue_response(sample_openai_response("Must never send."));

    // Grant covers another domain only, so this endpoint is denied.
    let config = NetworkConsumerAdapterConfig::new(
        "mock-openai",
        "gpt-4o",
        "https://api.openai.com/v1/chat/completions",
    )
    .unwrap()
    .with_capability(NetworkCapability::offline().with_domain("other.example"));
    let adapter = NetworkConsumerAdapter::new(config, service);
    let mut agent = agent_with(adapter);

    let outcome = run_one_turn(&mut agent);
    match outcome {
        ExecOutcome::Failed {
            error: AgentError::Provider(ProviderError::Transport { reason, .. }),
        } => {
            assert!(reason.contains("api.openai.com"));
        }
        other => panic!("expected denied Transport, got {other:?}"),
    }
    assert_eq!(agent.provider_mut().service().recorded_count(), 0);
    assert_eq!(agent.provider_mut().complete_calls(), 0);
}

fn direct_turn_request() -> TurnRequest {
    TurnRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![Message::user("test")],
        context_refs: Vec::new(),
        tools: Vec::new(),
        budget_bytes: 8192,
        timeout_ms: 5000,
        now_ms: NOW_MS,
        sampling: None,
    }
}

#[test]
fn adapter_maps_http_status_table() {
    fn error_response(status: u16, retry_after: Option<&str>) -> Response {
        let mut headers = Vec::new();
        if let Some(value) = retry_after {
            headers.push(("retry-after".to_owned(), value.to_owned()));
        }
        Response {
            status,
            headers,
            body: format!("{{\"error\":\"http {status}\"}}").into_bytes(),
        }
    }

    let service = RecordingNetworkService::new();
    service.queue_response(error_response(401, None));
    service.queue_response(error_response(403, None));
    service.queue_response(error_response(429, Some("10")));
    service.queue_response(error_response(429, None));
    service.queue_response(error_response(404, None));
    service.queue_response(error_response(503, None));
    service.queue_response(error_response(500, None));
    let mut adapter = NetworkConsumerAdapter::new(granted_config(), service);
    // One agent turn ends its session on failure, so the table drives the
    // adapter directly with one queued response per status.
    let req = direct_turn_request();

    // 401 -> Auth.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::Auth { provider, .. } => assert_eq!(provider, "mock-openai"),
        other => panic!("expected Auth for 401, got {other:?}"),
    }

    // 403 -> Auth.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::Auth { .. } => {}
        other => panic!("expected Auth for 403, got {other:?}"),
    }

    // 429 + Retry-After -> RateLimited with converted ms.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::RateLimited {
            provider,
            retry_after_ms,
        } => {
            assert_eq!(provider, "mock-openai");
            assert_eq!(retry_after_ms, Some(10_000));
        }
        other => panic!("expected RateLimited for 429, got {other:?}"),
    }

    // 429 without Retry-After -> RateLimited with no hint.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::RateLimited { retry_after_ms, .. } => assert_eq!(retry_after_ms, None),
        other => panic!("expected RateLimited for bare 429, got {other:?}"),
    }

    // 404 -> ModelUnavailable.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::ModelUnavailable { provider, model } => {
            assert_eq!(provider, "mock-openai");
            assert_eq!(model, "gpt-4o");
        }
        other => panic!("expected ModelUnavailable for 404, got {other:?}"),
    }

    // 503 -> ModelUnavailable.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::ModelUnavailable { .. } => {}
        other => panic!("expected ModelUnavailable for 503, got {other:?}"),
    }

    // 500 -> Transport.
    match adapter.complete(&req).unwrap_err() {
        ProviderError::Transport { reason, .. } => assert!(reason.contains("500")),
        other => panic!("expected Transport for 500, got {other:?}"),
    }
    assert_eq!(adapter.service().recorded_count(), 7);
}

#[test]
fn agent_turn_pins_transfer_budgets_as_unknown() {
    let service = RecordingNetworkService::new();
    service.queue_error(NetworkError::Budget { limit_bytes: 4096 });
    service.queue_error(NetworkError::CountBudget { limit_items: 8 });

    let mut adapter = NetworkConsumerAdapter::new(granted_config(), service);
    let req = direct_turn_request();

    // Transfer byte budget: post-send truncation, so Unknown (never
    // BudgetExceeded, which stays context-only per CP-5).
    match adapter.complete(&req).unwrap_err() {
        ProviderError::Unknown { provider, reason } => {
            assert_eq!(provider, "mock-openai");
            assert!(reason.contains("network body budget exceeded"));
        }
        other => panic!("expected Unknown for Budget, got {other:?}"),
    }

    // Transfer count budget: same Unknown mapping (no arbitrary split).
    match adapter.complete(&req).unwrap_err() {
        ProviderError::Unknown { provider, reason } => {
            assert_eq!(provider, "mock-openai");
            assert!(reason.contains("network count budget exceeded"));
        }
        other => panic!("expected Unknown for CountBudget, got {other:?}"),
    }
    // Both fired post-send: the service observed each request exactly once.
    assert_eq!(adapter.service().recorded_count(), 2);
    assert_eq!(adapter.complete_calls(), 2);

    // End to end, the agent surfaces provider Unknown on its MP-7 reconcile
    // path (never a blind provider failure): a fresh agent drives one turn
    // against a transfer-budget error.
    let service = RecordingNetworkService::new();
    service.queue_error(NetworkError::Budget { limit_bytes: 4096 });
    let mut agent = agent_with(NetworkConsumerAdapter::new(granted_config(), service));
    let outcome = run_one_turn(&mut agent);
    match outcome {
        ExecOutcome::Unknown { reason, .. } => {
            assert!(reason.contains("reconcile before retry"));
        }
        other => panic!("expected agent Unknown for Budget, got {other:?}"),
    }
}

#[test]
fn adapter_tool_role_prefix_and_error_display_redaction() {
    let canary_key = b"sk-live-integration-canary-secret-00042";
    let secret = SecretField::new(canary_key.to_vec()).unwrap();

    let service = RecordingNetworkService::new();
    service.queue_response(sample_openai_response("Observed."));

    let config = granted_config().with_api_key(secret);
    let mut adapter = NetworkConsumerAdapter::new(config, service);

    let req = TurnRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![
            Message::user("Summarize the observation."),
            Message::tool("observation payload"),
        ],
        context_refs: Vec::new(),
        tools: Vec::new(),
        budget_bytes: 8192,
        timeout_ms: 5000,
        now_ms: NOW_MS,
        sampling: None,
    };
    let turn = adapter.complete(&req).expect("tool turn succeeds");
    assert_eq!(turn.text, "Observed.");

    // Tool observations fold into `user` messages with a `[tool] ` prefix.
    let recorded = adapter.service().last_request().expect("recorded request");
    let body: serde_json::Value =
        serde_json::from_slice(&recorded.body).expect("request body is JSON");
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], serde_json::json!("user"));
    assert_eq!(
        messages[1]["content"],
        serde_json::json!("[tool] observation payload")
    );

    // Recorded-request redaction holds at the integration edge too.
    let req_debug = format!("{recorded:?}");
    assert!(!req_debug.contains("canary"));
    assert!(req_debug.contains("[redacted]"));

    // Error Display/Debug carry no secret material either: the queue is now
    // drained, so the next turn fails closed at the service boundary.
    let err = adapter.complete(&req).unwrap_err();
    let err_display = format!("{err}");
    let err_debug = format!("{err:?}");
    assert!(!err_display.contains("canary"));
    assert!(!err_debug.contains("canary"));
}

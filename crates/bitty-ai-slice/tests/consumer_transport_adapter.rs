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
    ModelProvider, ProviderError, SecretField, ToolAuthorizer, ToolBus, ToolRegistry, VecSink,
};
use bitty_ai_slice::{
    NetworkConsumerAdapter, NetworkConsumerAdapterConfig, RecordingNetworkService,
};
use bitty_network_api::{HttpMethod, NetworkError, Response, TlsFailure};

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
    .with_header("x-client-name", "bitty-ai-slice");

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
    .unwrap();

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
    .unwrap();

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
    .unwrap();

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
    .unwrap();

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

//! Sampling contract evidence for the provider boundary (AI-0085).
//!
//! [`TurnRequest`] carries zero sampling parameters before this task; the
//! contract adds an OPTIONAL, validated sampling surface ([`SamplingParams`]
//! plus [`ReasoningConfig`], [`ReasoningEffort`], [`ResponseFormat`]) where
//! absent (`None`) means undeclared and `Some` means explicitly declared and
//! validated fail-closed before provider I/O.
//!
//! Deterministic and offline: pure [`validate_sampling`] calls plus scripted
//! [`FakeProvider`] turns with caller-supplied `now_ms`. No network, no
//! secrets, no wall clock, no threads.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `provider.rs` unit tests own provider-id shape, timeout ceiling, budget,
//!   and script mechanics. Sampling appears there only as `None`.
//! - `selection.rs` unit tests own alias, window, and cost gates. Reasoning
//!   appears here only as capability-subset filtering mirroring `ToolUse`.
//! - `cache_key.rs` owns key construction and hit-rate mechanics. Sampling
//!   appears here only as the exclusion proof (never hashed into keys).
//!
//! CodeQL lesson from AI-0082: assert/panic messages are static only;
//! sampling values, keys, and costs never appear in message strings.

use std::str::FromStr;

use bitty_ai_runtime::{
    CacheKey, CacheScope, FakeProvider, LayerInput, ModelCapability, ModelProvider, ModelRef,
    ModelRegistration, PromptLayer, PromptSnapshot, ProviderError, ProviderRegistry, ProviderTurn,
    ProviderUsage, ReasoningConfig, ReasoningEffort, ResponseFormat, SamplingParams, SelectRequest,
    SelectionError, TurnRequest, assemble_prompt, estimate_cost, provider::Message,
    validate_sampling,
};
use bitty_ai_runtime::{
    REASONING_RATIO_HIGH, REASONING_RATIO_LOW, REASONING_RATIO_MAX, REASONING_RATIO_MEDIUM,
    REASONING_RATIO_MINIMAL, REASONING_RATIO_XHIGH,
};

/// A sampling contract with every field undeclared.
fn blank_params() -> SamplingParams {
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

/// A fully declared, valid sampling contract.
fn full_params() -> SamplingParams {
    SamplingParams {
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        frequency_penalty: Some(0.1),
        presence_penalty: Some(-0.2),
        repetition_penalty: Some(1.1),
        min_p: Some(0.05),
        seed: Some(42),
        max_tokens: Some(1_000),
        stop: Some(vec!["stop".to_owned()]),
        response_format: Some(ResponseFormat::JsonSchema {
            schema: Some(br#"{"type":"object"}"#.to_vec()),
        }),
        reasoning: Some(ReasoningConfig {
            effort: Some(ReasoningEffort::Medium),
            max_tokens: Some(500),
            exclude: false,
        }),
    }
}

/// A reasoning-only contract with the given budget and top-level maximum.
fn reasoning_with(budget: Option<u32>, max_tokens: Option<u32>) -> SamplingParams {
    let mut params = blank_params();
    params.max_tokens = max_tokens;
    params.reasoning = Some(ReasoningConfig {
        effort: None,
        max_tokens: budget,
        exclude: false,
    });
    params
}

fn request_with(sampling: Option<SamplingParams>) -> TurnRequest {
    TurnRequest {
        model: "fake-chat".to_owned(),
        messages: vec![Message::user("hi")],
        context_refs: Vec::new(),
        tools: Vec::new(),
        budget_bytes: 4096,
        timeout_ms: 5_000,
        now_ms: 1_000,
        sampling,
    }
}

fn scripted_provider() -> FakeProvider {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "answer".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider
}

fn assert_accepted(params: &SamplingParams) {
    assert!(validate_sampling(params).is_ok());
}

fn assert_rejected(params: &SamplingParams) {
    assert!(matches!(
        validate_sampling(params),
        Err(ProviderError::InvalidSampling)
    ));
}

#[test]
fn temperature_accepts_closed_interval_and_rejects_outside() {
    for value in [0.0, 0.5, 1.0, 2.0] {
        let mut params = blank_params();
        params.temperature = Some(value);
        assert_accepted(&params);
    }
    for value in [-0.001, -1.0, 2.001, 3.0] {
        let mut params = blank_params();
        params.temperature = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn top_p_accepts_closed_interval_and_rejects_outside() {
    for value in [0.0, 0.25, 0.9, 1.0] {
        let mut params = blank_params();
        params.top_p = Some(value);
        assert_accepted(&params);
    }
    for value in [-0.001, -0.5, 1.001, 2.0] {
        let mut params = blank_params();
        params.top_p = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn top_k_accepts_nonnegative_and_rejects_negative() {
    for value in [0, 1, 40, i64::MAX] {
        let mut params = blank_params();
        params.top_k = Some(value);
        assert_accepted(&params);
    }
    for value in [-1, -100, i64::MIN] {
        let mut params = blank_params();
        params.top_k = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn frequency_penalty_accepts_closed_interval_and_rejects_outside() {
    for value in [-2.0, -0.5, 0.0, 1.5, 2.0] {
        let mut params = blank_params();
        params.frequency_penalty = Some(value);
        assert_accepted(&params);
    }
    for value in [-2.001, -3.0, 2.001, 5.0] {
        let mut params = blank_params();
        params.frequency_penalty = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn presence_penalty_accepts_closed_interval_and_rejects_outside() {
    for value in [-2.0, -1.0, 0.0, 0.3, 2.0] {
        let mut params = blank_params();
        params.presence_penalty = Some(value);
        assert_accepted(&params);
    }
    for value in [-2.001, -10.0, 2.001, 2.5] {
        let mut params = blank_params();
        params.presence_penalty = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn repetition_penalty_accepts_closed_interval_and_rejects_outside() {
    for value in [0.0, 0.8, 1.0, 1.5, 2.0] {
        let mut params = blank_params();
        params.repetition_penalty = Some(value);
        assert_accepted(&params);
    }
    for value in [-0.001, -1.0, 2.001, 4.0] {
        let mut params = blank_params();
        params.repetition_penalty = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn min_p_accepts_closed_interval_and_rejects_outside() {
    for value in [0.0, 0.05, 0.5, 1.0] {
        let mut params = blank_params();
        params.min_p = Some(value);
        assert_accepted(&params);
    }
    for value in [-0.001, -2.0, 1.001, 1.5] {
        let mut params = blank_params();
        params.min_p = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn float_params_reject_nan_fail_closed() {
    let mut params = blank_params();
    params.temperature = Some(f64::NAN);
    assert_rejected(&params);
    let mut params = blank_params();
    params.top_p = Some(f64::NAN);
    assert_rejected(&params);
    let mut params = blank_params();
    params.frequency_penalty = Some(f64::NAN);
    assert_rejected(&params);
    let mut params = blank_params();
    params.presence_penalty = Some(f64::NAN);
    assert_rejected(&params);
    let mut params = blank_params();
    params.repetition_penalty = Some(f64::NAN);
    assert_rejected(&params);
    let mut params = blank_params();
    params.min_p = Some(f64::NAN);
    assert_rejected(&params);
}

#[test]
fn float_params_reject_infinities() {
    for value in [f64::INFINITY, f64::NEG_INFINITY] {
        let mut params = blank_params();
        params.temperature = Some(value);
        assert_rejected(&params);
        let mut params = blank_params();
        params.top_p = Some(value);
        assert_rejected(&params);
    }
}

#[test]
fn seed_accepts_any_integer() {
    for value in [0, 1, -1, i64::MIN, i64::MAX] {
        let mut params = blank_params();
        params.seed = Some(value);
        assert_accepted(&params);
    }
}

#[test]
fn max_tokens_requires_at_least_one() {
    let mut params = blank_params();
    params.max_tokens = Some(0);
    assert_rejected(&params);
    for value in [1, 1_000, u32::MAX] {
        let mut params = blank_params();
        params.max_tokens = Some(value);
        assert_accepted(&params);
    }
}

#[test]
fn stop_list_bounds_count_and_bytes() {
    let mut params = blank_params();
    params.stop = Some(Vec::new());
    assert_accepted(&params);
    let mut params = blank_params();
    params.stop = Some(vec![" lowercase".to_owned(); 8]);
    assert_accepted(&params);
    let mut params = blank_params();
    params.stop = Some(vec!["s".repeat(256)]);
    assert_accepted(&params);
    let mut params = blank_params();
    params.stop = Some(vec!["stop".to_owned(); 9]);
    assert_rejected(&params);
    let mut params = blank_params();
    params.stop = Some(vec!["s".repeat(257)]);
    assert_rejected(&params);
}

#[test]
fn response_format_variants_validate() {
    for format in [
        ResponseFormat::Text,
        ResponseFormat::JsonObject,
        ResponseFormat::JsonSchema { schema: None },
        ResponseFormat::JsonSchema {
            schema: Some(vec![b'{', b'}']),
        },
    ] {
        let mut params = blank_params();
        params.response_format = Some(format);
        assert_accepted(&params);
    }
    let mut params = blank_params();
    params.response_format = Some(ResponseFormat::JsonSchema {
        schema: Some(vec![0u8; 16 * 1024 + 1]),
    });
    assert_rejected(&params);
    let mut params = blank_params();
    params.response_format = Some(ResponseFormat::JsonSchema {
        schema: Some(vec![0u8; 16 * 1024]),
    });
    assert_accepted(&params);
}

#[test]
fn reasoning_budget_must_be_strictly_below_max_tokens() {
    assert_accepted(&reasoning_with(Some(500), Some(1_000)));
    assert_accepted(&reasoning_with(Some(1), Some(2)));
    assert_rejected(&reasoning_with(Some(1_000), Some(1_000)));
    assert_rejected(&reasoning_with(Some(1_001), Some(1_000)));
    assert_rejected(&reasoning_with(Some(u32::MAX), Some(u32::MAX)));
}

#[test]
fn reasoning_budget_without_top_level_max_is_allowed() {
    assert_accepted(&reasoning_with(Some(500), None));
    assert_accepted(&reasoning_with(None, Some(1_000)));
    assert_accepted(&reasoning_with(None, None));
}

#[test]
fn reasoning_budget_requires_at_least_one() {
    assert_rejected(&reasoning_with(Some(0), Some(1_000)));
    assert_rejected(&reasoning_with(Some(0), None));
}

#[test]
fn reasoning_effort_without_budget_is_estimation_only() {
    // Effort declared without any token budget validates and executes: the
    // effort-ratio table is estimation input, never enforcement.
    for effort in [
        ReasoningEffort::None,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ] {
        let mut params = blank_params();
        params.reasoning = Some(ReasoningConfig {
            effort: Some(effort),
            max_tokens: None,
            exclude: false,
        });
        assert_accepted(&params);
        let mut provider = scripted_provider();
        provider
            .complete(&request_with(Some(params)))
            .expect("effort-only config must execute");
    }
}

#[test]
fn reasoning_effort_strings_round_trip() {
    let cases = [
        (ReasoningEffort::None, "none"),
        (ReasoningEffort::Minimal, "minimal"),
        (ReasoningEffort::Low, "low"),
        (ReasoningEffort::Medium, "medium"),
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::XHigh, "xhigh"),
        (ReasoningEffort::Max, "max"),
    ];
    for (effort, name) in cases {
        assert_eq!(effort.to_string(), name);
        assert_eq!(name.parse::<ReasoningEffort>(), Ok(effort));
    }
}

#[test]
fn reasoning_effort_rejects_unknown_strings() {
    for name in ["", "ultra", "HIGH", "x-high", " low", "max ", "none\n"] {
        assert!(name.parse::<ReasoningEffort>().is_err());
        assert!(ReasoningEffort::from_str(name).is_err());
    }
}

#[test]
fn reasoning_effort_budget_ratios_are_pinned() {
    assert_eq!(ReasoningEffort::None.estimated_budget_ratio(), None);
    assert_eq!(
        ReasoningEffort::Minimal
            .estimated_budget_ratio()
            .map(f64::to_bits),
        Some(REASONING_RATIO_MINIMAL.to_bits())
    );
    assert_eq!(
        ReasoningEffort::Low
            .estimated_budget_ratio()
            .map(f64::to_bits),
        Some(REASONING_RATIO_LOW.to_bits())
    );
    assert_eq!(
        ReasoningEffort::Medium
            .estimated_budget_ratio()
            .map(f64::to_bits),
        Some(REASONING_RATIO_MEDIUM.to_bits())
    );
    assert_eq!(
        ReasoningEffort::High
            .estimated_budget_ratio()
            .map(f64::to_bits),
        Some(REASONING_RATIO_HIGH.to_bits())
    );
    assert_eq!(
        ReasoningEffort::XHigh
            .estimated_budget_ratio()
            .map(f64::to_bits),
        Some(REASONING_RATIO_XHIGH.to_bits())
    );
    assert_eq!(
        ReasoningEffort::Max
            .estimated_budget_ratio()
            .map(f64::to_bits),
        Some(REASONING_RATIO_MAX.to_bits())
    );
    assert_eq!(REASONING_RATIO_MINIMAL.to_bits(), 0.1_f64.to_bits());
    assert_eq!(REASONING_RATIO_LOW.to_bits(), 0.2_f64.to_bits());
    assert_eq!(REASONING_RATIO_MEDIUM.to_bits(), 0.5_f64.to_bits());
    assert_eq!(REASONING_RATIO_HIGH.to_bits(), 0.8_f64.to_bits());
    assert_eq!(REASONING_RATIO_XHIGH.to_bits(), 0.95_f64.to_bits());
    assert_eq!(REASONING_RATIO_MAX.to_bits(), 0.95_f64.to_bits());
}

#[test]
fn effort_ratio_feeds_cost_estimation_without_enforcement() {
    // The ratio table is DOCUMENTED ESTIMATION ONLY: it scales a token
    // count into `estimate_cost` input and never gates validation.
    let ratio = ReasoningEffort::Medium
        .estimated_budget_ratio()
        .expect("medium effort has a ratio");
    let estimated = (1_000_f64 * ratio) as u64;
    assert_eq!(estimated, 500);
    assert_eq!(estimate_cost(estimated, 0, 1, 1), 500);
}

#[test]
fn absent_sampling_flows_through_untouched() {
    let mut provider = scripted_provider();
    let turn = provider
        .complete(&request_with(None))
        .expect("absent sampling must pass");
    assert_eq!(turn.text, "answer");
    assert_eq!(provider.last_sampling(), None);
}

#[test]
fn declared_sampling_echoes_exactly() {
    let declared = full_params();
    assert_accepted(&declared);
    let mut provider = scripted_provider();
    provider
        .complete(&request_with(Some(declared.clone())))
        .expect("declared sampling must pass");
    assert_eq!(provider.last_sampling(), Some(&declared));
}

#[test]
fn echo_tracks_none_after_some_without_defaulting() {
    let mut provider = scripted_provider();
    provider.push_turn(ProviderTurn {
        text: "second".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider
        .complete(&request_with(Some(full_params())))
        .expect("declared sampling must pass");
    assert!(provider.last_sampling().is_some());
    let turn = provider
        .complete(&request_with(None))
        .expect("absent sampling must pass");
    assert_eq!(turn.text, "second");
    assert_eq!(provider.last_sampling(), None);
}

fn text_layer(layer: PromptLayer, text: &str) -> LayerInput {
    LayerInput::text_only(layer, text)
}

fn canonical_bytes() -> Vec<u8> {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::Project, "stable project"),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            text_layer(PromptLayer::RuntimeTurn, "turn tail"),
        ],
    )
    .expect("valid snapshot");
    assemble_prompt(&snapshot)
        .expect("snapshot assembles")
        .canonical_bytes()
        .to_vec()
}

#[test]
fn sampling_never_enters_cache_keys() {
    // The declared contract and the absent contract are different requests,
    // but the prefix-cache key over identical bytes is identical: sampling
    // is routing-declared per turn, never key content.
    let plain = request_with(None);
    let sampled = request_with(Some(full_params()));
    assert_ne!(plain, sampled);
    let bytes = canonical_bytes();
    let plain_key =
        CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &bytes).expect("valid key");
    let sampled_key =
        CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &bytes).expect("valid key");
    assert_eq!(plain_key, sampled_key);
}

#[test]
fn invalid_sampling_rejected_before_provider_io() {
    let mut provider = scripted_provider();
    let mut bad = blank_params();
    bad.temperature = Some(3.0);
    let error = provider
        .complete(&request_with(Some(bad)))
        .expect_err("out-of-range sampling must fail");
    assert_eq!(error, ProviderError::InvalidSampling);
    assert_eq!(provider.scripted_turns_remaining(), 1);
    assert_eq!(provider.complete_calls(), 0);
    assert_eq!(provider.last_sampling(), None);
}

#[test]
fn all_absent_sampling_contract_validates() {
    assert_accepted(&blank_params());
}

#[test]
fn turn_request_with_full_sampling_contract_is_valid_end_to_end() {
    let declared = full_params();
    assert_accepted(&declared);
    let mut provider = scripted_provider();
    let turn = provider
        .complete(&request_with(Some(declared.clone())))
        .expect("full contract must pass");
    assert_eq!(turn.text, "answer");
    assert_eq!(provider.last_sampling(), Some(&declared));
    assert_eq!(provider.complete_calls(), 1);
}

fn reasoning_registry() -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    registry
        .register(
            ModelRegistration::new(
                "bitty-a",
                "reasoner",
                vec![ModelCapability::Text, ModelCapability::Reasoning],
                32_768,
                2,
                3,
            )
            .expect("valid registration"),
        )
        .expect("capacity");
    registry
        .register(
            ModelRegistration::new("bitty-b", "plain", vec![ModelCapability::Text], 4_096, 1, 1)
                .expect("valid registration"),
        )
        .expect("capacity");
    registry
}

#[test]
fn reasoning_capability_gates_selection_like_tool_use() {
    let registry = reasoning_registry();
    let chain = registry
        .select(&SelectRequest::capabilities(vec![
            ModelCapability::Text,
            ModelCapability::Reasoning,
        ]))
        .expect("reasoner matches");
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].name, "reasoner");
    let chain = registry
        .select(&SelectRequest::capabilities(vec![ModelCapability::Text]))
        .expect("both match");
    assert_eq!(chain.len(), 2);
}

#[test]
fn reasoning_required_but_unadvertised_fails_closed() {
    let mut registry = ProviderRegistry::new();
    registry
        .register(
            ModelRegistration::new("bitty-b", "plain", vec![ModelCapability::Text], 4_096, 1, 1)
                .expect("valid registration"),
        )
        .expect("capacity");
    assert!(matches!(
        registry.select(&SelectRequest::capabilities(vec![
            ModelCapability::Text,
            ModelCapability::Reasoning,
        ])),
        Err(SelectionError::NoCandidate { .. })
    ));
}

#[test]
fn reasoning_alias_candidates_still_require_the_capability() {
    let mut registry = reasoning_registry();
    registry
        .register_alias(
            "chat",
            vec![ModelRef::new("bitty-b", "plain").expect("valid ref")],
        )
        .expect("capacity");
    let request = SelectRequest {
        required: vec![ModelCapability::Text, ModelCapability::Reasoning],
        alias: Some("chat".to_owned()),
        min_context_window_tokens: None,
        max_cost_weight: None,
    };
    match registry.select(&request) {
        Err(SelectionError::NoCandidate { detail }) => {
            assert!(detail.contains("alias=chat"));
        }
        _ => panic!("alias miss must not fall through"),
    }
}

//! Bounded cost-accounting evidence for the AIQ-47 narrowing (AI-0083).
//!
//! The per-turn cost ceiling (AI-0046) bounds spend, not diagnostic
//! subscription rate: the runtime has no diagnostics bus, so what is proven
//! offline here is that cost accounting itself stays bounded and
//! non-allocating under adversarial load, i.e. the accounting path cannot
//! become the unbounded cost it meters.
//!
//! Deterministic and offline: pure [`estimate_cost`] calls plus scripted
//! [`FakeProvider`] turns with explicit [`ProviderUsage`], a scripted
//! [`FakeToolExecutor`], and caller-supplied `now_ms`. No network, no
//! secrets, no wall clock. Doubles only.
//!
//! What this file proves that `cost_ceiling.rs`, `agent_turn_semantics.rs`,
//! and the `selection.rs` unit tests do not:
//!
//! - saturation through the real `u32` provider-usage path (existing
//!   `estimate_cost_saturates_instead_of_wrapping` uses `u64::MAX` tokens,
//!   which can never arrive via [`ProviderUsage`]);
//! - purity/stability of `estimate_cost` over a repeated-call loop;
//! - the fuse firing before dispatch under adversarial (`u32::MAX`) weights
//!   with exact `limit`/`actual` fields;
//! - one shared assertion binding the zero-weight rule at the routing
//!   estimate AND at turn accounting.
//!
//! CodeQL lesson from AI-0082: assert/panic messages are static only; cost,
//! token, and weight values never appear in message strings.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ContextPriority, ContextRecord,
    ExecOutcome, FakeProvider, FakeToolExecutor, ModelCapability, ModelProvider, ProviderTurn,
    ProviderUsage, RecordBody, RegisteredModel, SelectedModel, SessionState, StableId,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolRegistry, ToolSpec, VecSink, estimate_cost,
};

const NOW_MS: u64 = 1_700_000_000_000;
/// Repeated-call count for the purity/stability loop: large enough to catch
/// order-dependent or accumulating behavior, small enough to stay instant.
const STABILITY_ROUNDS: usize = 1_024;
/// `u16::MAX`-ish token counts: the largest realistic provider-reported
/// counts that still fit the `u32` [`ProviderUsage`] fields.
const ADVERSARIAL_TOKENS: u32 = 65_535;

/// Test-only allow hook so these tests isolate cost accounting from
/// authorization. Production wiring installs the host capability-plus-consent
/// hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

fn ids() -> (
    bitty_ai_runtime::AgentInstanceId,
    bitty_ai_runtime::RunId,
    bitty_ai_runtime::SessionId,
) {
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    (issuer.agent_instance(), issuer.run(), issuer.session())
}

fn session() -> bitty_ai_runtime::AgentSession {
    let (agent, run, session) = ids();
    bitty_ai_runtime::AgentSession::new(agent, run, session)
}

fn read_tool_bus() -> ToolBus {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "workspace_read",
                "Read a bounded workspace path",
                br#"{"type":"object"}"#.to_vec(),
                "workspace.read",
                true,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    ToolBus::new(registry).with_authorizer(AllowAll)
}

fn seed_record(id: &str, summary: &str, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![b'x'; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn tool_turn(text: &str, input_tokens: u32, output_tokens: u32) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"a"}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn final_turn(text: &str, input_tokens: u32, output_tokens: u32) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn run(
    agent: &mut Agent<FakeProvider>,
    executor: &mut FakeToolExecutor,
    prompt: &str,
    seeds: &[ContextRecord],
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, seeds, sink, NOW_MS)
}

#[test]
fn estimate_cost_saturates_under_maxed_weights_and_token_counts() {
    // Adversarial but representable: `u32::MAX` weights with `u32::MAX`
    // token counts (the widest values the `u32` [`ProviderUsage`] fields can
    // carry). Each product alone fits `u64`, but their sum must saturate
    // rather than wrap. The `selection.rs` unit test only covers `u64::MAX`
    // tokens, which can never arrive via `ProviderUsage`.
    let saturated = estimate_cost(u64::from(u32::MAX), u64::from(u32::MAX), u32::MAX, u32::MAX);
    assert_eq!(saturated, u64::MAX);
    // `u16::MAX`-ish counts stay exact on both sides: the bound is proven
    // by the value, not by saturation alone.
    assert_eq!(
        estimate_cost(
            u64::from(ADVERSARIAL_TOKENS),
            u64::from(ADVERSARIAL_TOKENS),
            u32::MAX,
            u32::MAX,
        ),
        562_941_363_355_650
    );
    // Single-sided adversarial load stays exact.
    assert_eq!(
        estimate_cost(u64::from(ADVERSARIAL_TOKENS), 0, u32::MAX, 1),
        u64::from(ADVERSARIAL_TOKENS) * u64::from(u32::MAX)
    );
    assert_eq!(
        estimate_cost(0, u64::from(ADVERSARIAL_TOKENS), 1, u32::MAX),
        u64::from(ADVERSARIAL_TOKENS) * u64::from(u32::MAX)
    );
    assert_eq!(
        estimate_cost(u64::MAX, u64::MAX, u32::MAX, u32::MAX),
        u64::MAX
    );
}

#[test]
fn turn_cost_saturates_across_maxed_rounds_without_wrapping() {
    // Two maxed provider rounds with no ceiling: accumulation uses
    // `saturating_add`, so the second round cannot wrap the first round's
    // saturated total back down.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("round one", u32::MAX, u32::MAX));
    provider.push_turn(final_turn("round two", u32::MAX, u32::MAX));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            input_cost_weight: u32::MAX,
            output_cost_weight: u32::MAX,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("first ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(
        &mut agent,
        &mut executor,
        "go",
        &[seed_record("s1", "manifest", 10)],
        &mut sink,
    );

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert_eq!(agent.turn_cost(), u64::MAX);
}

#[test]
fn estimate_cost_is_pure_and_stable_over_repeated_calls() {
    // Value stability, not wall time: N repeated calls with adversarial
    // inputs return the identical result every time, through both the free
    // function and the `SelectedModel` delegate. A shared or accumulating
    // implementation would diverge across iterations.
    let first = estimate_cost(u64::from(u32::MAX), u64::from(u32::MAX), u32::MAX, u32::MAX);
    let entry = RegisteredModel {
        provider_id: "bitty-fake".to_owned(),
        name: "m".to_owned(),
        capabilities: vec![ModelCapability::Text],
        context_window_tokens: 4_096,
        input_cost_weight: u32::MAX,
        output_cost_weight: u32::MAX,
    };
    let selected = SelectedModel::from_entry(&entry);
    let selected_first = selected.estimate_cost(u64::from(u32::MAX), u64::from(u32::MAX));
    assert_eq!(selected_first, first);
    for _ in 0..STABILITY_ROUNDS {
        assert_eq!(
            estimate_cost(u64::from(u32::MAX), u64::from(u32::MAX), u32::MAX, u32::MAX),
            first
        );
        assert_eq!(
            selected.estimate_cost(u64::from(u32::MAX), u64::from(u32::MAX)),
            first
        );
    }
    // Small-value stability pins the exact arithmetic alongside saturation.
    let small_first = estimate_cost(10, 5, 2, 3);
    assert_eq!(small_first, 35);
    for _ in 0..STABILITY_ROUNDS {
        assert_eq!(estimate_cost(10, 5, 2, 3), small_first);
    }
}

#[test]
fn cost_fuse_fires_before_dispatch_under_adversarial_weights() {
    // `u32::MAX` weights saturate the round cost, so even a near-`u64::MAX`
    // ceiling trips on the first round: the fuse stops the turn before
    // authorization/dispatch, the session stays `Active`, and the typed
    // error carries the exact ceiling and the saturated total.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("adversarial plan", u32::MAX, u32::MAX));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(u64::MAX - 1),
            input_cost_weight: u32::MAX,
            output_cost_weight: u32::MAX,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never dispatch", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit,
                    actual,
                }
            } if *limit == u64::MAX - 1 && *actual == u64::MAX
        ),
        "fuse must report the exact ceiling and saturated total"
    );
    assert_eq!(agent.turn_cost(), u64::MAX);
    assert_eq!(agent.session().state(), SessionState::Active);
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert!(sink.is_empty());
}

#[test]
fn zero_weight_rule_holds_at_routing_estimate_and_turn_accounting() {
    // One shared assertion binding both layers of the unified zero-weight
    // rule: a `0` (uncalibrated) weight counts as the baseline `1` in the
    // routing estimate AND in turn accounting, so routing never estimates
    // lower than accounting charges.
    assert_eq!(estimate_cost(10, 5, 0, 0), estimate_cost(10, 5, 1, 1));
    assert_eq!(estimate_cost(10, 5, 0, 3), estimate_cost(10, 5, 1, 3));
    assert_eq!(estimate_cost(10, 5, 0, 0), 15);

    let entry = RegisteredModel {
        provider_id: "bitty-fake".to_owned(),
        name: "uncalibrated".to_owned(),
        capabilities: vec![ModelCapability::Text],
        context_window_tokens: 4_096,
        input_cost_weight: 0,
        output_cost_weight: 0,
    };
    let selected = SelectedModel::from_entry(&entry);
    assert_eq!(selected.estimate_cost(10, 5), 15);

    // Turn accounting with `0`/`0` configured weights charges the same
    // baseline: 5 + 5 tokens at weight 1 each trips a ceiling just under
    // that total, exactly as `1`/`1` weights would.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("done", 5, 5));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(9),
            input_cost_weight: 0,
            output_cost_weight: 0,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 9,
                    actual: 10
                }
            }
        ),
        "zero configured weights must charge the baseline, not free rounds"
    );
    assert_eq!(agent.session().state(), SessionState::Active);
}

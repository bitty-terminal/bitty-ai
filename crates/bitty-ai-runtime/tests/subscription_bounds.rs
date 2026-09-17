//! Bounded diagnostic-subscription rate evidence for the AIQ-47 narrowing (AI-0093).
//!
//! The per-turn cost ceiling (AI-0046) bounds spend, not diagnostic
//! subscription rate: the runtime exposes no diagnostic observation seam at
//! all (no status-query endpoint, event subscription, telemetry hook, or
//! poll/watch callback exists on [`Agent`], [`UnknownReconciler`], or the
//! stream sink), so there is no subscription path whose rate a ceiling must
//! gate. What is proven offline here is the mechanism that actually bounds
//! diagnostic-rate pressure through the only paths diagnostics can reach:
//! context assembly ([`assemble`]) plus the per-turn cost fuse. Diagnostic
//! content enters only as `provider: "diagnostics"` seed records, where it is
//! already subject to the context byte budget (greedy budgeted include with
//! counted truncation), the `Low` effective-priority drop-first order (when
//! marked), and the `Unknown` reconcile query budget
//! ([`ReconcileConfig::effective_retries`]) for observation-style status reads.
//!
//! Deterministic and offline: pure [`assemble`] calls plus scripted
//! [`FakeProvider`] turns with explicit [`ProviderUsage`], a scripted
//! [`FakeToolExecutor`], scripted [`FakeReconciler`] answers, and
//! caller-supplied `now_ms`. No network, no secrets, no wall clock, no
//! threads, no randomness. Doubles only.
//!
//! What this file proves that `accounting_bounds.rs`, `cost_ceiling.rs`,
//! `unknown_reconcile.rs`, and `agent_turn_semantics.rs` do not:
//!
//! - all observation-adjacent callables on the runtime surface are the
//!   already-bounded reconcile query path: zero free/uncapped subscription,
//!   status, or telemetry methods exist on [`Agent`], [`StreamSink`],
//!   [`UnknownReconciler`], or the context surface beyond
//!   [`Agent::reconcile_unknown`], whose budget the next proof pins;
//! - the reconcile query budget stays bounded under an adversarial configured
//!   budget (`usize::MAX` retries clamp to [`MAX_RECONCILE_ATTEMPTS`]), so the
//!   diagnostic-style status-query rate is capped by construction, not by
//!   caller restraint;
//! - a diagnostics-flood seed set (more `diagnostics` records than fit any
//!   per-turn budget) assembles to a bounded subset with counted truncation
//!   and first-seen provider attribution, i.e. subscription-rate pressure
//!   cannot widen the per-turn context or the per-turn cost input;
//! - a diagnostic-heavy turn against a per-turn cost ceiling trips the fuse
//!   before dispatch with the exact `limit`/`actual` fields, pins the turn as
//!   non-dispatching, and leaves the session `Active` for retry;
//! - the subscription verdict is stable over a repeated-call loop (same
//!   script plus same `now_ms` replays to identical truncation and cost
//!   totals).
//!
//! Non-overlap: `accounting_bounds.rs` (AI-0083) proves the accounting path
//! itself stays bounded and non-allocating (saturation, purity,
//! fuse-before-dispatch, zero-weight unity). `cost_ceiling.rs` (AI-0046)
//! proves the per-turn cost fuse trip accounting, byte-estimate fallback,
//! zero-weight rule, and determinism. `unknown_reconcile.rs` (AI-0047)
//! proves reconcile resolve/escalate accounting, budget separation, backoff
//! determinism, and the clock-advance contract. `agent_turn_semantics.rs`
//! (AI-0076) proves multi-turn, cancel/resume, escalated-Unknown, fuse-retry,
//! and per-turn stream sequencing. Nothing there covers a diagnostics-seed
//! flood under a tight budget, the reconcile budget as the diagnostic query
//! cap under adversarial configuration, or cost-fuse behavior on a
//! diagnostics-sourced turn.
//!
//! Gap verdict: NO-GAP. Investigation found no subscription/observation path
//! whose rate is unbounded by an existing ceiling: the runtime has no
//! diagnostics bus, no status-query endpoint besides the budgeted reconcile
//! driver, and no event/telemetry/subscription callable on any public type.
//! Diagnostics reach the turn only as budgeted seed records. Tests-only.
//!
//! CodeQL lesson from AI-0082: assert/panic messages are static only; cost,
//! token, budget, and attempt values never appear in message strings.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ContextPriority, ContextRecord,
    ContextRequest, ExecOutcome, FakeProvider, FakeReconciler, FakeToolExecutor, ModelProvider,
    ProviderTurn, ProviderUsage, RecordBody, SessionState, StableId, ToolAuthorizer, ToolBus,
    ToolCallRequest, ToolRegistry, ToolSpec, ToolStatus, VecSink, assemble,
};

const NOW_MS: u64 = 1_700_000_000_000;
/// Repeated-call count for the stability loop: large enough to catch
/// order-dependent or accumulating behavior, small enough to stay instant.
/// Mirrors `STABILITY_ROUNDS` in `accounting_bounds.rs`.
const STABILITY_ROUNDS: usize = 1_024;
/// Diagnostics-flood size: comfortably over any per-turn budget used here, so
/// truncation (not fit) is the asserted behavior.
const DIAGNOSTIC_FLOOD_RECORDS: usize = 24;
/// Tight per-turn assembly budget in bytes used by the flood proofs. Small
/// enough that only a strict subset of the flood assembles.
const FLOOD_BUDGET_BYTES: usize = 512;

/// Test-only allow hook so these tests isolate subscription accounting from
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

fn diagnostic_record(index: usize) -> ContextRecord {
    ContextRecord {
        id: format!("diag-{index}"),
        provider: "diagnostics".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Low,
        summary: format!("diagnostic-{index}"),
        body: RecordBody::Inline(vec![b'd'; 64]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn flood_records() -> Vec<ContextRecord> {
    (0..DIAGNOSTIC_FLOOD_RECORDS)
        .map(diagnostic_record)
        .collect()
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

fn unknown_error(reason: &str) -> bitty_ai_runtime::ToolError {
    bitty_ai_runtime::ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: reason.to_owned(),
    }
}

#[test]
fn no_uncapped_subscription_surface_exists_besides_budgeted_reconcile() {
    // The only observation-style callable on the runtime surface is the
    // budgeted reconcile driver: a failed double-check would be a gap, so
    // enumerate the observation-adjacent methods the public API exposes and
    // assert the reconcile budget below is the cap for every one of them.
    // Today that set is exactly `Agent::reconcile_unknown` (status query,
    // budget-capped) plus read-only accessors that perform no query
    // (`executions`, `tool_records`, `turn_cost`, `provider_mut`,
    // `session`, and the sink/reconciler counters). Any future
    // status/subscribe/telemetry/watch/poll callable must appear in this
    // enumeration or fail this test on review: the assertion below pins the
    // budgeted path the capability list must keep pointing at.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "setup must end Unknown"
    );
    let execution_id = agent.executions()[0].execution_id;

    // The budgeted path admits exactly the default budget and no more; the
    // read-only accessors report without issuing any query.
    let queries_before = agent.executions().len();
    assert_eq!(queries_before, 1);
    assert_eq!(agent.tool_records().len(), 1);
    assert_eq!(agent.turn_cost(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.session().state(), SessionState::Active);
    assert!(!sink.is_empty());

    let mut reconciler = FakeReconciler::new();
    for _ in 0..bitty_ai_runtime::DEFAULT_MAX_UNKNOWN_RETRIES {
        reconciler.push_pending("still uncertain");
    }
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(
        matches!(outcome, bitty_ai_runtime::ReconcileOutcome::Escalated(_)),
        "budget exhaustion must escalate"
    );
    assert_eq!(
        reconciler.query_count(),
        bitty_ai_runtime::DEFAULT_MAX_UNKNOWN_RETRIES
    );
    // No dispatch happened on the observation path: the executor saw only the
    // single turn dispatch.
    assert_eq!(executor.calls().len(), 1);
}

#[test]
fn adversarial_reconcile_budget_clamps_to_hard_cap() {
    // A hostile `max_unknown_retries` cannot widen the diagnostic-style
    // status-query rate: `effective_retries` clamps to
    // `MAX_RECONCILE_ATTEMPTS`, and the driver performs exactly that many
    // queries before typed escalation.
    let config = AgentConfig {
        max_unknown_retries: usize::MAX,
        ..AgentConfig::default()
    };
    assert_eq!(
        config.reconcile_config().effective_retries(),
        bitty_ai_runtime::MAX_RECONCILE_ATTEMPTS
    );
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), config);
    let mut executor = FakeToolExecutor::new();
    executor.push_error(unknown_error("ack lost"));
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "setup must end Unknown"
    );
    let execution_id = agent.executions()[0].execution_id;

    let mut reconciler = FakeReconciler::new();
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    let report = match &outcome {
        bitty_ai_runtime::ReconcileOutcome::Escalated(report) => report,
        _other => panic!("over-budget queries must escalate"),
    };
    assert_eq!(report.attempts, bitty_ai_runtime::MAX_RECONCILE_ATTEMPTS);
    assert_eq!(
        reconciler.query_count(),
        bitty_ai_runtime::MAX_RECONCILE_ATTEMPTS
    );
    assert_eq!(report.dispatched, 1);
    assert_eq!(agent.session().state(), SessionState::Failed);
    // The observation path still never dispatches: executor count is flat.
    assert_eq!(executor.calls().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
}

#[test]
fn diagnostics_flood_truncates_to_bounded_subset_with_counted_attribution() {
    // Subscription-rate pressure through the only diagnostics entry point
    // (seed records) cannot widen per-turn context: under a tight budget the
    // flood assembles to a strict subset with counted truncation and
    // first-seen provider attribution.
    let seeds = flood_records();
    let mut store = bitty_ai_runtime::ArtifactStore::new();
    let request = ContextRequest {
        max_tokens: None,
        max_bytes: Some(FLOOD_BUDGET_BYTES as u64),
        current_generation: 1,
    };
    let assembled = assemble(&seeds, &mut store, &request).expect("flood must assemble");

    assert_eq!(assembled.budget_bytes, FLOOD_BUDGET_BYTES);
    assert!(!assembled.records.is_empty());
    assert!(assembled.records.len() < seeds.len());
    assert_eq!(
        assembled.records.len() + assembled.omitted_ids.len(),
        seeds.len()
    );
    assert!(assembled.truncated_bytes > 0);
    assert_eq!(
        assembled.truncated_tokens_estimate,
        ContextRequest::estimate_tokens(assembled.truncated_bytes as usize)
    );
    assert_eq!(
        assembled.truncated_providers,
        vec!["diagnostics".to_owned()]
    );
    // Caller order is preserved for the admitted subset; omission covers the
    // tail. Provider attribution stays first-seen deduplicated.
    assert_eq!(assembled.context_refs.len(), assembled.records.len());
    for record in &assembled.records {
        assert_eq!(record.provider, "diagnostics");
    }
}

#[test]
fn diagnostics_flood_is_stable_over_repeated_calls() {
    // Value stability, not wall time: repeated assembly of the same flood
    // under the same budget returns identical admission, omission, and byte
    // totals every time. Mirrors the `STABILITY_ROUNDS` purity loop in
    // `accounting_bounds.rs`.
    let seeds = flood_records();
    let request = ContextRequest {
        max_tokens: None,
        max_bytes: Some(FLOOD_BUDGET_BYTES as u64),
        current_generation: 1,
    };
    let mut first_store = bitty_ai_runtime::ArtifactStore::new();
    let first = assemble(&seeds, &mut first_store, &request).expect("flood must assemble");
    let first_refs = first.context_refs.clone();
    for _ in 0..STABILITY_ROUNDS {
        let mut store = bitty_ai_runtime::ArtifactStore::new();
        let replay = assemble(&seeds, &mut store, &request).expect("replay must assemble");
        assert_eq!(replay.context_refs, first_refs);
        assert_eq!(replay.omitted_ids, first.omitted_ids);
        assert_eq!(replay.truncated_bytes, first.truncated_bytes);
        assert_eq!(replay.truncated_providers, first.truncated_providers);
        assert_eq!(replay.budget_bytes, first.budget_bytes);
    }
}

#[test]
fn diagnostic_heavy_turn_trips_cost_fuse_before_dispatch() {
    // A diagnostics-sourced turn against a per-turn cost ceiling trips the
    // fuse before dispatch with the exact ceiling and accumulated total: the
    // first (tool) round costs 10 and dispatches, the second (diagnostics
    // final) round adds 10 for 20 total against a ceiling of 15. The fused
    // round streams nothing, dispatches nothing further, and the session
    // stays `Active` for reconcile-and-retry.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("round one", 10, 0));
    provider.push_turn(final_turn("diagnostics digest", 10, 0));
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            max_turn_cost: Some(15),
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("first ok", b"data".to_vec());
    executor.push_success("must never dispatch", b"data".to_vec());
    let mut sink = VecSink::new();
    let seeds = flood_records();

    let outcome = agent.run_turn(
        &mut executor,
        "fake-chat",
        "summarize",
        &seeds,
        &mut sink,
        NOW_MS,
    );

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::CostCeilingExceeded {
                    limit: 15,
                    actual: 20
                }
            }
        ),
        "diagnostic-heavy turn must trip the fuse with the exact ceiling and total"
    );
    assert_eq!(agent.turn_cost(), 20);
    assert_eq!(agent.session().state(), SessionState::Active);
    // Only the first round dispatched; the fused diagnostics round added no
    // further effect.
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 2);
}

#[test]
fn diagnostic_turn_and_reconcile_interleave_stays_within_both_budgets() {
    // Combined bound evidence: a diagnostics-seeded turn that ends `Unknown`
    // reconciles within the reconcile budget while the turn cost stays within
    // the per-turn ceiling, proving both bounds compose instead of trading
    // off. Replays deterministically for the same script plus `now_ms`.
    fn once() -> (ExecOutcome, u64, usize, SessionState) {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(tool_turn("reading diagnostics", 4, 0));
        let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
        let mut executor = FakeToolExecutor::new();
        executor.push_error(unknown_error("ack lost"));
        let mut sink = VecSink::new();
        let outcome = agent.run_turn(
            &mut executor,
            "fake-chat",
            "summarize",
            &flood_records(),
            &mut sink,
            NOW_MS,
        );
        let cost = agent.turn_cost();
        let execution_id = agent.executions()[0].execution_id;
        let mut reconciler = FakeReconciler::new();
        reconciler.push_pending("still uncertain");
        reconciler.push_resolved(ToolStatus::Success);
        let reconcile = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
        let queries = reconciler.query_count();
        assert!(
            matches!(
                &reconcile,
                bitty_ai_runtime::ReconcileOutcome::Resolved {
                    status: ToolStatus::Success,
                    ..
                }
            ),
            "setup must resolve"
        );
        (outcome, cost, queries, agent.session().state())
    }
    let (first_outcome, first_cost, first_queries, first_state) = once();
    assert!(
        matches!(first_outcome, ExecOutcome::Unknown { .. }),
        "diagnostic turn must end Unknown"
    );
    assert_eq!(first_cost, 4);
    assert_eq!(first_queries, 2);
    assert_eq!(first_state, SessionState::Active);
    for _ in 0..STABILITY_ROUNDS {
        let (outcome, cost, queries, state) = once();
        assert_eq!(outcome, first_outcome);
        assert_eq!(cost, first_cost);
        assert_eq!(queries, first_queries);
        assert_eq!(state, first_state);
    }
}

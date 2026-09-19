//! Supervisor crash-recovery adoption rule with survivor fencing (AI-0091,
//! AIQ-2B narrowing).
//!
//! AIQ-2B (supervisor crash recovery/adoption) is stay-open-narrowed:
//! single-agent `Unknown` handling is bounded (`unknown_reconcile.rs`: resolve
//! / escalate accounting, budget separation, backoff determinism,
//! clock-advance contract) and multi-turn/cancel/resume/escalation semantics
//! are pinned (`agent_turn_semantics.rs`), but supervisor crash adoption
//! across processes has no mechanism. The runtime is single-process sans-I/O,
//! so cross-process adoption cannot be built here; what is proven at the seam
//! is the ADOPTION RULE: a supervisor must never adopt arbitrary survivors or
//! repeat `Unknown` effects.
//!
//! Pinned rule: adoption requires an explicit, typed [`AdoptionClaim`] and
//! the supervisor-side [`check_adoption`] refuses unless all four hold:
//! (a) the prior session is in terminal `Failed` state (live sessions are
//! never adopted; `Completed`/`Canceled` have no adoptable leftovers and a
//! different owner);
//! (b) every carried still-`Unknown` effect is named `Escalated` (claimed
//! `Reconciled` over still-uncertain evidence, omitted `Unknown`s, and any
//! claimed disposition over never-`Unknown` evidence all refuse; a
//! `Reconciled` disposition is verifiable only for evidence the pre-crash
//! protocol actually resolved, covered by the `Unknown`-free success path;
//! escalated ones adopt as quarantined declarations);
//! (c) nothing replays: the check takes no executor, returns declarative
//! data only, and adopted ids are rejected by a fresh agent's
//! `reconcile_unknown` as `NoUnknown` (adopted history can never re-enter
//! the dispatch/reconcile path);
//! (d) the claim's fence token equals the current supervisor epoch
//! (stale-epoch claims refused, mirroring the AIQ-2C writer-fencing
//! direction without building it).
//!
//! Candidate-shape adaptation (verified against `session.rs`/`agent.rs`): the
//! candidate's bare `unknown_effect_ids` cannot carry dispositions, so the
//! claim names `unknown_effects: Vec<ClaimedUnknownEffect>` pairing each id
//! with its [`UnknownDisposition`]. Survivor evidence reuses
//! [`ExecutionRecord`] (the exact attributed-record shape the crashed turn
//! recorded) instead of a duplicate type. No `Agent`/`AgentSession` change
//! was needed: lifecycle states, `reconcile_unknown`, and the per-turn
//! clearing that bounds survivors to the crashed turn already exist.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `unknown_reconcile.rs`: resolve/escalate mechanics, budget separation,
//!   backoff, clock-advance contract. Reconcile appears here only as scenario
//!   setup (the reconcile-first ordering the rule requires).
//! - `agent_turn_semantics.rs`: multi-turn continuation, cancel/resume,
//!   escalated-session `AlreadyTerminated` rejection, fuse interaction.
//!   Terminal-state behavior appears here only as claim priors.
//! - `runtime_fail_closed.rs`: cancel and `Unknown` session semantics. Cancel
//!   appears here only as the `Canceled`-prior refusal.
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns, scripted
//! [`FakeToolExecutor`] outcomes, scripted [`FakeReconciler`] answers,
//! caller-supplied `now_ms`. No network, no secrets, no wall clock, no
//! threads, no processes, no I/O. Doubles only.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! ids, epochs, counts, and tool names never appear in message strings.

use bitty_ai_runtime::{
    AdoptedHistory, AdoptionClaim, AdoptionRefusal, Agent, AgentConfig, AuthContext, AuthDecision,
    ClaimedUnknownEffect, ExecOutcome, ExecutionId, ExecutionRecord, FakeProvider, FakeReconciler,
    FakeToolExecutor, IdIssuer, MAX_ADOPTION_SURVIVORS, ProviderTurn, ProviderUsage,
    ReconcileOutcome, SessionState, ToolAuthorizer, ToolBus, ToolCallRequest, ToolError,
    ToolRegistry, ToolSpec, ToolStatus, UnknownDisposition, VecSink, check_adoption,
};

const NOW_MS: u64 = 1_700_000_000_000;
/// Deterministic supervisor epoch shared by the claim fence token and the
/// adopting supervisor in the success paths.
const EPOCH: u64 = 7;

/// Test-only allow hook so these tests isolate the adoption rule from
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
    let mut issuer = IdIssuer::default();
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

fn final_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

/// Drive one turn to `Unknown` with a terminal sibling: the batch dispatches
/// a success first, then the second effect loses its acknowledgement. The
/// turn stops at the first `Unknown`, so the crashed turn records exactly
/// `[Success, Unknown]` with the session left `Active` for reconcile.
fn drive_unknown(config: AgentConfig) -> (Agent<FakeProvider>, FakeToolExecutor) {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "two effects".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"b"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut executor = FakeToolExecutor::new();
    executor.push_success("a ok", b"a".to_vec());
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "host crashed before ack".to_owned(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), config);
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "effect uncertainty must end the turn as Unknown"
    );
    assert_eq!(agent.executions().len(), 2);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(matches!(
        agent.executions()[1].status,
        ToolStatus::Unknown { .. }
    ));
    assert_eq!(agent.session().state(), SessionState::Active);
    (agent, executor)
}

/// Fresh supervisor-side agent: the adopting session. Adopted execution ids
/// belong to the crashed turn and must be foreign here.
fn fresh_agent() -> Agent<FakeProvider> {
    let provider = FakeProvider::new("bitty-fake").expect("valid id");
    Agent::new(provider, read_tool_bus(), session(), AgentConfig::default())
}

/// The no-re-dispatch structural proof, shared by the success paths: adopted
/// ids are declarative data, so the fresh agent's `reconcile_unknown` must
/// report `NoUnknown`, run no query, and leave its session untouched.
fn assert_ids_are_declarative(agent: &mut Agent<FakeProvider>, history: &AdoptedHistory) {
    let mut reconciler = FakeReconciler::new();
    for effect in &history.effects {
        let outcome = agent.reconcile_unknown(&mut reconciler, effect.execution_id, NOW_MS);
        assert_eq!(outcome, ReconcileOutcome::NoUnknown);
    }
    assert_eq!(reconciler.query_count(), 0);
    assert_eq!(agent.session().state(), SessionState::Active);
}

/// Crash attestation for tests that need a `Failed` prior without a
/// protocol failure: the supervisor is dead, so its session can never
/// complete; the adopting supervisor records it as failed.
fn attest_crash(agent: &Agent<FakeProvider>) {
    agent.session().finish(true);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

#[test]
fn live_session_adoption_is_refused() {
    // Rule (a): never adopt a live session survivor. The prior is still
    // `Active` (reconcile pending), so even a fully-described claim with a
    // matching epoch must refuse before any disposition is considered.
    let (agent, executor) = drive_unknown(AgentConfig::default());
    let calls_before = executor.calls().len();
    let records = agent.executions().to_vec();
    let unknown_id = records[1].execution_id;
    let claim = AdoptionClaim {
        prior_session_state: agent.session().state(),
        survivor_ids: vec![records[0].execution_id, unknown_id],
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: unknown_id,
            disposition: UnknownDisposition::Escalated,
        }],
        fence_token: EPOCH,
    };
    assert_eq!(claim.prior_session_state, SessionState::Active);
    let refusal =
        check_adoption(&claim, &records, EPOCH).expect_err("live session must refuse adoption");
    assert!(
        matches!(refusal, AdoptionRefusal::LiveSession),
        "live prior must refuse as LiveSession"
    );
    // Refusal decided nothing and dispatched nothing.
    assert_eq!(executor.calls().len(), calls_before);
}

#[test]
fn non_failed_terminal_priors_are_refused() {
    // Rule (a), remainder: only `Failed` sessions are adoptable. `Completed`
    // sessions have no unreconciled leftovers and `Canceled` leftovers belong
    // to the cancel-reconcile path, so both refuse without touching evidence.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(final_turn("done"));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert_eq!(agent.session().state(), SessionState::Completed);
    let completed_claim = AdoptionClaim {
        prior_session_state: SessionState::Completed,
        survivor_ids: Vec::new(),
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal = check_adoption(&completed_claim, &[], EPOCH)
        .expect_err("completed session must refuse adoption");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::NotFailed { state } if state == SessionState::Completed
        ),
        "completed prior must refuse as NotFailed"
    );

    let canceled = session();
    canceled.cancel();
    assert_eq!(canceled.state(), SessionState::Canceled);
    let canceled_claim = AdoptionClaim {
        prior_session_state: SessionState::Canceled,
        survivor_ids: Vec::new(),
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal = check_adoption(&canceled_claim, &[], EPOCH)
        .expect_err("canceled session must refuse adoption");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::NotFailed { state } if state == SessionState::Canceled
        ),
        "canceled prior must refuse as NotFailed"
    );
    assert!(executor.calls().is_empty());
}

#[test]
fn unreconciled_unknown_adoption_is_refused() {
    // Rule (b): claims whose `Unknown`s are not all reconciled-or-escalated
    // refuse. Both the omitted unknown and the false `Reconciled` disposition
    // over still-`Unknown` evidence refuse identically, and neither dispatches.
    let (agent, executor) = drive_unknown(AgentConfig::default());
    attest_crash(&agent);
    let calls_before = executor.calls().len();
    assert_eq!(calls_before, 2);
    let records = agent.executions().to_vec();
    let survivor_ids: Vec<ExecutionId> = records.iter().map(|record| record.execution_id).collect();
    let unknown_id = records[1].execution_id;

    let omitted = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: survivor_ids.clone(),
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal =
        check_adoption(&omitted, &records, EPOCH).expect_err("omitted unknown must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::UnreconciledUnknown { execution_id } if execution_id == unknown_id
        ),
        "omitted unknown must refuse as UnreconciledUnknown"
    );

    let mislabeled = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids,
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: unknown_id,
            disposition: UnknownDisposition::Reconciled,
        }],
        fence_token: EPOCH,
    };
    let refusal =
        check_adoption(&mislabeled, &records, EPOCH).expect_err("false reconcile must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::UnreconciledUnknown { execution_id } if execution_id == unknown_id
        ),
        "reconciled claim over uncertain evidence must refuse"
    );
    assert_eq!(executor.calls().len(), calls_before);
}

#[test]
fn stale_epoch_claim_is_refused() {
    // Rule (d): the fence token binds the claim to one supervisor epoch. A
    // claim fenced to a superseded epoch refuses even when every other check
    // would pass, so a stale supervisor cannot decide adoptions.
    let (mut agent, executor) = drive_unknown(AgentConfig::default());
    let records = agent.executions().to_vec();
    let unknown_id = records[1].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    let outcome = agent.reconcile_unknown(&mut reconciler, unknown_id, NOW_MS);
    assert!(matches!(outcome, ReconcileOutcome::Escalated(_)));
    assert_eq!(agent.session().state(), SessionState::Failed);
    let survivor_ids: Vec<ExecutionId> = records.iter().map(|record| record.execution_id).collect();
    let claim = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids,
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: unknown_id,
            disposition: UnknownDisposition::Escalated,
        }],
        fence_token: EPOCH,
    };
    let calls_before = executor.calls().len();
    let current_epoch = EPOCH + 1;
    let refusal = check_adoption(&claim, agent.executions(), current_epoch)
        .expect_err("stale epoch must refuse adoption");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::StaleEpoch {
                claim_epoch,
                current_epoch: seen_epoch
            } if claim_epoch == EPOCH && seen_epoch == EPOCH + 1
        ),
        "stale fence token must refuse as StaleEpoch"
    );
    assert_eq!(executor.calls().len(), calls_before);
}

#[test]
fn reconciled_adoption_succeeds_with_exact_survivor_set() {
    // Rules (b)+(c) success half one: the pre-crash protocol ran first (the
    // `Unknown` resolved via `reconcile_unknown`), a later turn failed the
    // session, and the failed turn's terminal survivors adopt as declared
    // history with nothing quarantined and nothing re-dispatched.
    let config = AgentConfig {
        max_rounds: 1,
        ..AgentConfig::default()
    };
    let (mut agent, mut executor) = drive_unknown(config);
    let unknown_id = agent.executions()[1].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Success);
    let outcome = agent.reconcile_unknown(&mut reconciler, unknown_id, NOW_MS);
    assert!(
        matches!(
            outcome,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                ..
            }
        ),
        "pre-crash reconcile must resolve the unknown"
    );
    assert_eq!(reconciler.query_count(), 1);
    assert!(matches!(agent.executions()[1].status, ToolStatus::Success));
    // A later turn fails the session (round bound) after two terminal
    // dispatches. Per-turn clearing bounds the survivors to this crashed
    // turn; the resolved unknown from the earlier turn is already history.
    agent.provider_mut().push_turn(ProviderTurn {
        text: "two more reads".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"c"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"d"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    executor.push_success("c ok", b"c".to_vec());
    executor.push_success("d ok", b"d".to_vec());
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "again", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Failed { .. }),
        "second turn must fail on the round bound"
    );
    assert_eq!(agent.session().state(), SessionState::Failed);
    assert_eq!(agent.executions().len(), 2);

    let records = agent.executions().to_vec();
    let survivor_ids: Vec<ExecutionId> = records.iter().map(|record| record.execution_id).collect();
    let claim = AdoptionClaim {
        prior_session_state: agent.session().state(),
        survivor_ids: survivor_ids.clone(),
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let calls_before = executor.calls().len();
    assert_eq!(calls_before, 4);
    let history: AdoptedHistory =
        check_adoption(&claim, &records, EPOCH).expect("reconciled adoption must succeed");
    // Exact survivor set, in claim order, all terminal and none quarantined.
    assert_eq!(history.len(), 2);
    assert!(!history.is_empty());
    let mut adopted_ids: Vec<ExecutionId> = history
        .effects
        .iter()
        .map(|effect| effect.execution_id)
        .collect();
    adopted_ids.sort_by_key(|id| id.0);
    let mut expected_ids = survivor_ids.clone();
    expected_ids.sort_by_key(|id| id.0);
    assert_eq!(adopted_ids, expected_ids);
    for effect in &history.effects {
        assert!(matches!(effect.status, ToolStatus::Success));
        assert!(!effect.quarantined);
        assert_eq!(effect.tool, "workspace_read");
    }
    assert!(history.quarantined_ids().is_empty());
    // Rule (c): adoption performed no effect; the only executor calls are the
    // four pre-crash dispatches.
    assert_eq!(executor.calls().len(), calls_before);

    // The adopted ids are declarative: a fresh agent cannot reconcile them
    // and its own dispatches mint fresh ids only.
    let mut fresh = fresh_agent();
    assert_ids_are_declarative(&mut fresh, &history);
    fresh.provider_mut().push_turn(final_turn("fresh done"));
    let mut fresh_executor = FakeToolExecutor::new();
    let mut fresh_sink = VecSink::new();
    let outcome = fresh.run_turn(
        &mut fresh_executor,
        "fake-chat",
        "hello",
        &[],
        &mut fresh_sink,
        NOW_MS,
    );
    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert!(fresh_executor.calls().is_empty());
    for record in fresh.executions() {
        assert!(!expected_ids.contains(&record.execution_id));
    }
}

#[test]
fn escalated_adoption_succeeds_with_quarantined_unknown() {
    // Rules (b)+(c) success half two: the `Unknown` exhausted its reconcile
    // budget and escalated (session `Failed`), so it adopts as a quarantined
    // declaration alongside the terminal sibling. Exact set, no replay.
    let (mut agent, executor) = drive_unknown(AgentConfig::default());
    let records_before = agent.executions().to_vec();
    let terminal_id = records_before[0].execution_id;
    let unknown_id = records_before[1].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    reconciler.push_pending("still writing");
    let outcome = agent.reconcile_unknown(&mut reconciler, unknown_id, NOW_MS);
    assert!(matches!(outcome, ReconcileOutcome::Escalated(_)));
    assert_eq!(reconciler.query_count(), 3);
    assert_eq!(agent.session().state(), SessionState::Failed);
    // Escalation fails the session but never rewrites the recorded status.
    assert!(matches!(
        agent.executions()[1].status,
        ToolStatus::Unknown { .. }
    ));

    let records = agent.executions().to_vec();
    let claim = AdoptionClaim {
        prior_session_state: agent.session().state(),
        survivor_ids: vec![terminal_id, unknown_id],
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: unknown_id,
            disposition: UnknownDisposition::Escalated,
        }],
        fence_token: EPOCH,
    };
    let calls_before = executor.calls().len();
    assert_eq!(calls_before, 2);
    let history: AdoptedHistory =
        check_adoption(&claim, &records, EPOCH).expect("escalated adoption must succeed");
    assert_eq!(history.len(), 2);
    assert_eq!(history.effects[0].execution_id, terminal_id);
    assert!(matches!(history.effects[0].status, ToolStatus::Success));
    assert!(!history.effects[0].quarantined);
    assert_eq!(history.effects[1].execution_id, unknown_id);
    assert!(matches!(
        history.effects[1].status,
        ToolStatus::Unknown { .. }
    ));
    assert!(history.effects[1].quarantined);
    assert_eq!(history.quarantined_ids(), vec![unknown_id]);
    // Rule (c): reconcile ran status queries only, adoption ran nothing; the
    // executor still shows exactly the two pre-crash dispatches.
    assert_eq!(executor.calls().len(), calls_before);

    let mut fresh = fresh_agent();
    assert_ids_are_declarative(&mut fresh, &history);
}

#[test]
fn adoption_coverage_and_bounds_refuse_fail_closed() {
    // The check is total over the claim/evidence relation: stowaway evidence,
    // missing evidence, duplicates, over-bound sets, and dispositions that
    // contradict the evidence all refuse with a typed reason. The empty
    // crash (no survivors) vacuously succeeds.
    let (agent, executor) = drive_unknown(AgentConfig::default());
    attest_crash(&agent);
    let records = agent.executions().to_vec();
    let terminal_id = records[0].execution_id;
    let unknown_id = records[1].execution_id;
    let mut issuer = IdIssuer::default();
    let ghost_id = issuer.execution();
    assert!(!records.iter().any(|record| record.execution_id == ghost_id));

    // Stowaway evidence: the claim must name every survivor it adopts.
    let stowaway = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id],
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal =
        check_adoption(&stowaway, &records, EPOCH).expect_err("stowaway evidence must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::UnclaimedSurvivor { execution_id } if execution_id == unknown_id
        ),
        "unnamed evidence must refuse as UnclaimedSurvivor"
    );

    // Missing evidence: every claimed survivor must be carried as evidence.
    let missing = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id, ghost_id],
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal =
        check_adoption(&missing, &records, EPOCH).expect_err("missing evidence must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::MissingEvidence { execution_id } if execution_id == ghost_id
        ),
        "unevidenced claim must refuse as MissingEvidence"
    );

    // Duplicate claim ids break the exact-set contract.
    let duplicated = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id, terminal_id],
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal =
        check_adoption(&duplicated, &records, EPOCH).expect_err("duplicate claim must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::DuplicateEffect { execution_id } if execution_id == terminal_id
        ),
        "duplicate claim ids must refuse as DuplicateEffect"
    );

    // Duplicate evidence ids are equally unadoptable.
    let mut duplicated_evidence = records.clone();
    duplicated_evidence.push(records[0].clone());
    let dup_evidence_claim = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id, unknown_id],
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: unknown_id,
            disposition: UnknownDisposition::Escalated,
        }],
        fence_token: EPOCH,
    };
    let refusal = check_adoption(&dup_evidence_claim, &duplicated_evidence, EPOCH)
        .expect_err("duplicate evidence must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::DuplicateEffect { execution_id } if execution_id == terminal_id
        ),
        "duplicate evidence ids must refuse as DuplicateEffect"
    );

    // An escalation claim over an already-terminal effect contradicts the
    // evidence (escalation never rewrites a resolved status).
    let terminal_record = records[0].clone();
    let mismatch = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id],
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: terminal_id,
            disposition: UnknownDisposition::Escalated,
        }],
        fence_token: EPOCH,
    };
    let refusal = check_adoption(&mismatch, std::slice::from_ref(&terminal_record), EPOCH)
        .expect_err("contradicted disposition must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::DispositionMismatch { execution_id } if execution_id == terminal_id
        ),
        "escalated claim over terminal evidence must refuse"
    );

    // Unknowns must be adopted as survivors, not merely named: this id has
    // no survivor coverage at all.
    let orphan = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id],
        unknown_effects: vec![ClaimedUnknownEffect {
            execution_id: ghost_id,
            disposition: UnknownDisposition::Reconciled,
        }],
        fence_token: EPOCH,
    };
    let terminal_evidence = vec![terminal_record];
    let refusal =
        check_adoption(&orphan, &terminal_evidence, EPOCH).expect_err("orphan unknown must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::MissingEvidence { execution_id } if execution_id == ghost_id
        ),
        "unknown without survivor coverage must refuse"
    );

    // Over-bound sets refuse before any content is considered.
    let mut wide_issuer = IdIssuer::default();
    let mut wide_records = Vec::new();
    let mut wide_ids = Vec::new();
    for _ in 0..=MAX_ADOPTION_SURVIVORS {
        let id = wide_issuer.execution();
        wide_ids.push(id);
        wide_records.push(ExecutionRecord {
            execution_id: id,
            tool: "workspace_read".to_owned(),
            status: ToolStatus::Success,
            result_disposition: bitty_ai_runtime::ResultDisposition::Accepted,
        });
    }
    assert_eq!(wide_ids.len(), MAX_ADOPTION_SURVIVORS + 1);
    let wide = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: wide_ids,
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let refusal =
        check_adoption(&wide, &wide_records, EPOCH).expect_err("over-bound set must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::TooManySurvivors { limit, actual }
                if limit == MAX_ADOPTION_SURVIVORS && actual == MAX_ADOPTION_SURVIVORS + 1
        ),
        "over-bound survivor set must refuse as TooManySurvivors"
    );

    // The auxiliary `Unknown` disposition list is bounded too: it cannot
    // legitimately exceed the survivor cap, and an over-bound list refuses
    // as a typed size refusal before any allocation or content inspection,
    // even though none of its entries could ever verify against evidence.
    let mut disposition_issuer = IdIssuer::default();
    let wide_unknowns: Vec<ClaimedUnknownEffect> = (0..=MAX_ADOPTION_SURVIVORS)
        .map(|_| ClaimedUnknownEffect {
            execution_id: disposition_issuer.execution(),
            disposition: UnknownDisposition::Escalated,
        })
        .collect();
    assert_eq!(wide_unknowns.len(), MAX_ADOPTION_SURVIVORS + 1);
    let wide_dispositions = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: vec![terminal_id],
        unknown_effects: wide_unknowns,
        fence_token: EPOCH,
    };
    let refusal = check_adoption(&wide_dispositions, &terminal_evidence, EPOCH)
        .expect_err("over-bound disposition list must refuse");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::TooManySurvivors { limit, actual }
                if limit == MAX_ADOPTION_SURVIVORS && actual == MAX_ADOPTION_SURVIVORS + 1
        ),
        "over-bound disposition list must refuse as TooManySurvivors"
    );

    // The disposition count alone drives the refusal: a live-session claim
    // with a stale epoch and an over-bound disposition list still refuses as
    // a size refusal, proving the count gate runs before content and state
    // checks.
    let size_first = AdoptionClaim {
        prior_session_state: SessionState::Active,
        survivor_ids: vec![terminal_id],
        unknown_effects: (0..=MAX_ADOPTION_SURVIVORS)
            .map(|_| ClaimedUnknownEffect {
                execution_id: disposition_issuer.execution(),
                disposition: UnknownDisposition::Escalated,
            })
            .collect(),
        fence_token: EPOCH + 1,
    };
    let refusal = check_adoption(&size_first, &terminal_evidence, EPOCH)
        .expect_err("size gate must outrank state and epoch checks");
    assert!(
        matches!(
            refusal,
            AdoptionRefusal::TooManySurvivors { limit, actual }
                if limit == MAX_ADOPTION_SURVIVORS && actual == MAX_ADOPTION_SURVIVORS + 1
        ),
        "size refusal must precede state and epoch checks"
    );

    // The empty crash adopts vacuously: nothing to recover is a success with
    // the empty set, not a refusal.
    let empty_session = session();
    empty_session.finish(true);
    let empty = AdoptionClaim {
        prior_session_state: SessionState::Failed,
        survivor_ids: Vec::new(),
        unknown_effects: Vec::new(),
        fence_token: EPOCH,
    };
    let history: AdoptedHistory =
        check_adoption(&empty, &[], EPOCH).expect("empty crash must adopt vacuously");
    assert!(history.effects.is_empty());
    assert!(history.quarantined_ids().is_empty());

    // No check above dispatched anything: the only executor calls are the
    // two pre-crash dispatches from setup.
    assert_eq!(executor.calls().len(), 2);
}

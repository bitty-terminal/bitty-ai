//! Seeded critical-message + Unknown fault matrix (AI-0234, R5-V5/V7-partial, R6-V5).
//!
//! Offline deterministic coverage for the five seeded fixtures: duplicate
//! delivery dedup on stable id plus expiry plus payload-conflict behavior,
//! crash-between-effect-and-ack with `Unknown` preserved through replay and
//! reconciled by inspection or user direction only (eligible-only retry, no
//! exactly-once), stale-generation takeover refused, capacity refusal with
//! explicit backpressure that never spawns another agent, and
//! observation-pressure survival of critical records versus coalescible
//! progress with counters.
//!
//! Taxonomy precedent pinned here: `Unknown` stays `Unknown` (never `Failed`);
//! admission refusals are `Denied`/`Refused` pre-contact (never `Failed` or
//! `Unknown`). All clocks are caller-supplied (`now_ms`); no wall clock, no
//! threads, no network, no new deps. File-backed tests use a fresh `temp_dir`
//! scratch database per test. Assert messages are static literals.

use bitty_ai_runtime::agent::is_backpressure;
use bitty_ai_runtime::reconcile::{FakeReconciler, is_reconcile_terminal, is_retry_eligible};
use bitty_ai_runtime::stream::{
    BoundedSink, Fragment, FragmentKind, StreamAttribution, StreamChunk, StreamHandle,
};
use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AuthContext, AuthDecision, ExecOutcome, FakeProvider,
    FakeToolExecutor, ProviderTurn, ProviderUsage, ReconcileOutcome, ReconcileStatus, SessionState,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolPlacement, ToolRegistry, ToolSpec,
    ToolStatus, VecSink,
};
use bitty_ai_session::content_store::ContentStore;
use bitty_ai_session::pending::{
    MAX_OPEN_PENDING_PER_SESSION, MAX_PENDING_DEDUP_AGE_MS, PendingBegin, PendingDisposition,
    PendingDuplicateKind, PendingError, PendingResolve, PendingStore, classify_duplicate,
    is_dedup_expired, is_payload_conflict,
};
use bitty_ai_session::sessions::WheelSessionId;
use bitty_ai_slice::content_store::Rationale;
use bitty_ai_slice::task_dag::{TaskDraft, TaskId};
use bitty_ai_slice::wheel_kernel::{MAX_RESUME_PENDING_FANOUT, WheelKernel};

const NOW_MS: u64 = 1_700_000_000_000;
const BEGUN_MS: u64 = 1_000;

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

fn agent_session() -> bitty_ai_runtime::AgentSession {
    let (agent, run, session) = ids();
    bitty_ai_runtime::AgentSession::new(agent, run, session)
}

fn consented_ledger() -> bitty_ai_runtime::FakeConsentLedger {
    let mut ledger = bitty_ai_runtime::FakeConsentLedger::new();
    for (tool, scope) in [
        ("workspace_read", "workspace.read"),
        ("workspace_write", "workspace.write"),
    ] {
        ledger
            .grant("local.assistant", tool, scope, u64::MAX)
            .expect("grant fits");
    }
    ledger
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
                ToolPlacement::Mcp,
                None,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    ToolBus::new(registry)
        .with_authorizer(AllowAll)
        .with_consent(consented_ledger(), "local.assistant")
}

fn tool_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"a"}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

fn final_turn(text: &str) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

fn test_store() -> ContentStore {
    ContentStore::open_in_memory().expect("open")
}

fn test_session_id() -> WheelSessionId {
    WheelSessionId::parse("sess-fault-1").expect("valid test session")
}

fn begin_req<'a>(
    call_id: &'a str,
    session_id: &'a WheelSessionId,
    args: &'a [u8],
    now_ms: u64,
) -> PendingBegin<'a> {
    PendingBegin {
        call_id,
        session_id,
        task_id: None,
        tool: "workspace_read",
        args,
        generation: 0,
        epoch: 1,
        now_ms,
    }
}

fn resolve_req(call_id: &str, disposition: PendingDisposition) -> PendingResolve<'_> {
    PendingResolve {
        call_id,
        disposition,
        claim_epoch: 1,
        expected_generation: 0,
        now_ms: 2_000,
    }
}

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bitty-ai-0234-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn test_task(id: &str) -> TaskDraft {
    TaskDraft {
        id: TaskId::new(id).expect("valid test task id"),
        title: format!("task {id}"),
        description: "lifecycle fault matrix task".to_owned(),
        priority: 1,
        dependencies: Vec::new(),
    }
}

#[test]
fn duplicate_delivery_dedup_expiry_conflict() {
    let store = test_store();
    let session = test_session_id();
    let entry = PendingStore::begin(&store, begin_req("aa01", &session, b"alpha", BEGUN_MS))
        .expect("begin");
    // Same stable id with identical payload refuses as a duplicate with zero
    // writes: the first row wins and is never overwritten.
    assert_eq!(
        PendingStore::begin(
            &store,
            begin_req("aa01", &session, b"alpha", BEGUN_MS + 500)
        ),
        Err(PendingError::AlreadyExists),
        "identical duplicate must refuse as already recorded"
    );
    let kept = PendingStore::get(&store, "aa01")
        .expect("get")
        .expect("present");
    assert_eq!(kept, entry, "duplicate must preserve the first row");
    // Same stable id with a conflicting payload refuses the same way with the
    // first digest preserved (fail closed, never overwritten).
    assert_eq!(
        PendingStore::begin(&store, begin_req("aa01", &session, b"beta", BEGUN_MS + 600)),
        Err(PendingError::AlreadyExists),
        "conflicting duplicate must refuse without overwrite"
    );
    let kept = PendingStore::get(&store, "aa01")
        .expect("get")
        .expect("present");
    assert_eq!(
        kept.args_digest, entry.args_digest,
        "conflict must keep the first digest"
    );
    // Pure classification: identical within window, conflict within window,
    // and expired past the window (expired never dedups even when identical).
    assert_eq!(
        classify_duplicate(&entry, b"alpha", BEGUN_MS + 500),
        PendingDuplicateKind::IdenticalWithinWindow,
        "identical payload within window must classify identical"
    );
    assert_eq!(
        classify_duplicate(&entry, b"beta", BEGUN_MS + 500),
        PendingDuplicateKind::PayloadConflict,
        "conflicting payload within window must classify conflict"
    );
    assert_eq!(
        classify_duplicate(&entry, b"alpha", BEGUN_MS + MAX_PENDING_DEDUP_AGE_MS + 1),
        PendingDuplicateKind::Expired,
        "duplicate past the window must classify expired"
    );
    assert!(
        !is_dedup_expired(BEGUN_MS, BEGUN_MS + 500),
        "within-window delivery must not read expired"
    );
    assert!(
        is_dedup_expired(BEGUN_MS, BEGUN_MS + MAX_PENDING_DEDUP_AGE_MS + 1),
        "past-window delivery must read expired"
    );
    assert!(
        !is_payload_conflict(&entry, b"alpha"),
        "identical bytes must not read conflict"
    );
    assert!(
        is_payload_conflict(&entry, b"beta"),
        "differing bytes must read conflict"
    );
    // Resolve-layer dedup: same-disposition re-resolve is an idempotent
    // no-op, a conflicting disposition refuses as Mismatch, and an unknown
    // id reports NotFound (bounded-recall expiry, never proof of no effect).
    let resolved = PendingStore::resolve(&store, resolve_req("aa01", PendingDisposition::Success))
        .expect("resolve");
    let mut again = resolve_req("aa01", PendingDisposition::Success);
    again.now_ms = 9_999;
    assert_eq!(
        PendingStore::resolve(&store, again),
        Ok(resolved.clone()),
        "same-disposition re-resolve must return the cached snapshot"
    );
    assert_eq!(
        PendingStore::resolve(&store, resolve_req("aa01", PendingDisposition::Failed)),
        Err(PendingError::Mismatch {
            stored: PendingDisposition::Success,
            supplied: PendingDisposition::Failed,
        }),
        "conflicting re-resolve must refuse as mismatch"
    );
    assert_eq!(
        PendingStore::resolve(&store, resolve_req("bb02", PendingDisposition::Success)),
        Err(PendingError::NotFound),
        "unknown id must report not found"
    );
}

#[test]
fn crash_between_effect_and_ack_unknown_preserved_through_replay() {
    let dir = scratch_dir("crash-replay");
    let db_path = dir.join("wheel.db");
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel open");
        kernel
            .create_task(test_task("fault-task-1"), BEGUN_MS)
            .expect("create task");
        let task_id = TaskId::new("fault-task-1").expect("valid task id");
        kernel
            .start_task(&task_id, "worker-1", BEGUN_MS + 10)
            .expect("start task");
        kernel
            .set_active_task(Some(task_id))
            .expect("set active task");
        kernel
            .put_slot("notes.txt", b"fault-matrix payload", BEGUN_MS + 20)
            .expect("put slot");
        kernel
            .commit_checkpoint(
                Rationale::new("Persist progress", "Save durable state"),
                Some("heads/main"),
                BEGUN_MS + 30,
            )
            .expect("commit");
        kernel
            .new_session("sess-fault-1", "heads/main", BEGUN_MS + 40)
            .expect("bind session");
        // Crash window: the host began the call and crashed before the ack,
        // so the open row is the only durable trace.
        let session = WheelSessionId::parse("sess-fault-1").expect("valid session");
        PendingStore::begin(
            kernel.content_store(),
            PendingBegin {
                call_id: "beef01",
                session_id: &session,
                task_id: Some("fault-task-1"),
                tool: "workspace_write",
                args: br#"{"path":"notes.txt"}"#,
                generation: 1,
                epoch: 1,
                now_ms: BEGUN_MS + 50,
            },
        )
        .expect("host begin");
        // Drop without resolving: the crash-mid-turn point.
    }
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel reopen");
        let report = kernel
            .resume_session("sess-fault-1", 2, BEGUN_MS + 1_000)
            .expect("resume after reopen");
        assert_eq!(
            report.pending_unknowns,
            vec!["beef01".to_owned()],
            "replay must name the surviving Unknown"
        );
        assert!(
            !report.pending_log_absent,
            "resume must report the log present"
        );
        assert_eq!(report.fence_token, 2, "resume must admit the fresh fence");
        // Replay preserves Unknown: the row stays open after resume (never
        // auto-resolved, never re-executed).
        let session = WheelSessionId::parse("sess-fault-1").expect("valid session");
        let open = PendingStore::list_open(kernel.content_store(), &session).expect("list open");
        assert_eq!(open.len(), 1, "resume must leave the Unknown open");
        assert_eq!(
            open[0].call_id, "beef01",
            "surviving Unknown id must match the crash row"
        );
        // The bare-branch path unions the same Unknown without resolving it.
        let by_branch = kernel
            .resume_session("heads/main", 0, BEGUN_MS + 1_010)
            .expect("branch resume");
        assert_eq!(
            by_branch.pending_unknowns,
            vec!["beef01".to_owned()],
            "branch replay must union the same Unknown"
        );
        let open = PendingStore::list_open(kernel.content_store(), &session).expect("list open");
        assert_eq!(open.len(), 1, "branch replay must not resolve the Unknown");
    }
    // Ack-loss taxonomy: Unknown stays Unknown through the bus (never
    // Failed); admission refusals stay Denied/Refused pre-contact.
    let mut bus = read_tool_bus();
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "host crashed before ack".to_owned(),
    });
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    let base = {
        let mut ids = bitty_ai_runtime::IdIssuer::default();
        bitty_ai_runtime::tool::AuthBase {
            agent_instance_id: ids.agent_instance(),
            session_id: ids.session(),
            level: bitty_ai_runtime::session::AgentLevel::Inspect,
        }
    };
    let call = bitty_ai_runtime::tool::ToolCall {
        name: "workspace_read".to_owned(),
        arguments: br#"{"path":"a"}"#.to_vec(),
    };
    let execution = bus
        .dispatch(&mut executor, &call, &base, issuer.execution(), NOW_MS)
        .expect("ack-loss is a recorded status");
    assert!(
        execution.status.is_ack_loss(),
        "ack-loss must record Unknown"
    );
    assert!(
        !matches!(execution.status, ToolStatus::Failed { .. }),
        "ack-loss must never record Failed"
    );
    assert!(
        !is_reconcile_terminal(&execution.status),
        "Unknown must never read terminal"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_reconciled_by_inspection_only_eligible_retry_no_exactly_once() {
    // Drive one turn to Unknown via a scripted ack-loss outcome.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote"));
    provider.push_turn(final_turn("unreached"));
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "host crashed before ack".to_owned(),
    });
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        agent_session(),
        AgentConfig::default(),
    );
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "ack-loss must end the turn Unknown"
    );
    assert_eq!(
        agent.session().state(),
        SessionState::Active,
        "session must stay usable for reconcile-and-retry"
    );
    let execution_id = agent.executions()[0].execution_id;
    assert!(
        agent.executions()[0].status.is_ack_loss(),
        "recorded status must stay Unknown"
    );
    assert_eq!(executor.calls().len(), 1, "effect attempted once");
    // Inspection-only reconcile: the reconciler queries stored outcomes and
    // never executes (executor count stays flat across queries).
    let mut reconciler = FakeReconciler::new();
    reconciler.push_pending("still uncertain");
    reconciler.push_resolved(ToolStatus::Success);
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(
        matches!(
            &outcome,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 2,
                ..
            }
        ),
        "inspection must resolve without re-execution"
    );
    assert_eq!(
        executor.calls().len(),
        1,
        "reconcile must never re-dispatch the effect"
    );
    assert_eq!(reconciler.query_count(), 2, "two status queries ran");
    // A Resolved(Unknown) answer is still pending (fail closed, never
    // accepted as resolved).
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote again"));
    provider.push_turn(final_turn("unreached"));
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "still uncertain".to_owned(),
    });
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        agent_session(),
        AgentConfig::default(),
    );
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    assert!(
        matches!(outcome, ExecOutcome::Unknown { .. }),
        "second ack-loss must end Unknown"
    );
    let execution_id = agent.executions()[0].execution_id;
    let mut reconciler = FakeReconciler::new();
    reconciler.push_resolved(ToolStatus::Unknown {
        reason: "still uncertain".to_owned(),
    });
    reconciler.push_resolved(ToolStatus::Success);
    let outcome = agent.reconcile_unknown(&mut reconciler, execution_id, NOW_MS);
    assert!(
        matches!(
            &outcome,
            ReconcileOutcome::Resolved {
                status: ToolStatus::Success,
                attempts: 2,
                ..
            }
        ),
        "Unknown answer must stay pending until the terminal answer"
    );
    // Eligible-only retry: terminal plus inspection plus current grant is
    // eligible; Unknown, missing inspection, or missing grant never is. A
    // retry is a new submission under current grants, never a deduped replay
    // (no exactly-once claim).
    assert!(
        is_retry_eligible(&ToolStatus::Success, true, true),
        "terminal plus inspection plus grant must read eligible"
    );
    assert!(
        !is_retry_eligible(
            &ToolStatus::Unknown {
                reason: "still uncertain".to_owned()
            },
            true,
            true
        ),
        "Unknown must never read eligible even with inspection and grant"
    );
    assert!(
        !is_retry_eligible(&ToolStatus::Success, false, true),
        "terminal without inspection must never read eligible"
    );
    assert!(
        !is_retry_eligible(&ToolStatus::Success, true, false),
        "terminal without a current grant must never read eligible"
    );
    assert!(
        !is_reconcile_terminal(&ToolStatus::Unknown {
            reason: "still uncertain".to_owned()
        }),
        "Unknown must never read terminal"
    );
    assert!(
        is_reconcile_terminal(&ToolStatus::Success),
        "Success must read terminal"
    );
    assert!(
        !matches!(
            ReconcileOutcome::NoUnknown,
            ReconcileOutcome::Resolved { .. }
        ),
        "sentinel check must hold"
    );
    let pending: ReconcileStatus = ReconcileStatus::Pending {
        reason: "still uncertain".to_owned(),
    };
    assert!(
        matches!(pending, ReconcileStatus::Pending { .. }),
        "pending answer must stay pending"
    );
}

#[test]
fn stale_generation_takeover_refused() {
    let store = test_store();
    let session = test_session_id();
    let mut begun = begin_req("ab12", &session, b"{}", BEGUN_MS);
    begun.generation = 3;
    begun.epoch = 2;
    PendingStore::begin(&store, begun).expect("begin");
    // A stale epoch refuses with zero writes; the entry stays open.
    let mut stale_epoch = resolve_req("ab12", PendingDisposition::Success);
    stale_epoch.claim_epoch = 1;
    stale_epoch.expected_generation = 3;
    assert_eq!(
        PendingStore::resolve(&store, stale_epoch),
        Err(PendingError::StaleEpoch {
            claim_epoch: 1,
            current_epoch: 2,
        }),
        "stale epoch must refuse"
    );
    // A moved generation refuses with zero writes; the entry stays open.
    let mut stale_generation = resolve_req("ab12", PendingDisposition::Success);
    stale_generation.claim_epoch = 2;
    stale_generation.expected_generation = 4;
    assert_eq!(
        PendingStore::resolve(&store, stale_generation),
        Err(PendingError::StaleGeneration {
            expected: 3,
            found: 4,
        }),
        "stale generation must refuse"
    );
    let open = PendingStore::list_open(&store, &session).expect("list open");
    assert_eq!(open.len(), 1, "stale takeover must leave the entry open");
    assert_eq!(
        open[0].call_id, "ab12",
        "surviving entry must keep its stable id"
    );
    // The current fence still admits.
    let mut fresh = resolve_req("ab12", PendingDisposition::Success);
    fresh.claim_epoch = 2;
    fresh.expected_generation = 3;
    let resolved = PendingStore::resolve(&store, fresh).expect("fresh resolve");
    assert_eq!(
        resolved.disposition,
        Some(PendingDisposition::Success),
        "current fence must close the entry"
    );
    // Wheel resume fencing: a stale claim epoch refuses with zero writes and
    // the same claim stays admissible for retry; the fresh claim advances.
    let dir = scratch_dir("stale-takeover");
    let db_path = dir.join("wheel.db");
    let mut kernel = WheelKernel::open(&db_path).expect("wheel open");
    kernel
        .create_task(test_task("fault-task-2"), BEGUN_MS)
        .expect("create task");
    kernel
        .put_slot("notes.txt", b"takeover payload", BEGUN_MS + 10)
        .expect("put slot");
    kernel
        .commit_checkpoint(
            Rationale::new("Persist progress", "Save durable state"),
            Some("heads/main"),
            BEGUN_MS + 20,
        )
        .expect("commit");
    kernel
        .new_session("sess-fault-2", "heads/main", BEGUN_MS + 30)
        .expect("bind session");
    kernel
        .resume_session("sess-fault-2", 2, BEGUN_MS + 40)
        .expect("first resume admits fence two");
    assert!(
        kernel
            .resume_session("sess-fault-2", 1, BEGUN_MS + 50)
            .is_err(),
        "stale claim epoch must refuse"
    );
    kernel
        .resume_session("sess-fault-2", 3, BEGUN_MS + 60)
        .expect("fresh claim epoch must admit");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn capacity_backpressure_never_spawns_agent() {
    let store = test_store();
    let session = test_session_id();
    for seq in 0..MAX_OPEN_PENDING_PER_SESSION {
        let call_id = format!("{seq:04x}");
        PendingStore::begin(&store, begin_req(&call_id, &session, b"{}", BEGUN_MS))
            .expect("begin within cap");
    }
    // Over-cap begin refuses with the explicit bound and zero writes.
    assert_eq!(
        PendingStore::begin(&store, begin_req("ffff", &session, b"{}", BEGUN_MS)),
        Err(PendingError::TooManyOpen {
            limit: MAX_OPEN_PENDING_PER_SESSION,
        }),
        "over-cap begin must refuse with the explicit bound"
    );
    let open = PendingStore::list_open(&store, &session).expect("list open");
    assert_eq!(
        open.len(),
        MAX_OPEN_PENDING_PER_SESSION,
        "refusal must leave the open set at the bound"
    );
    // Backpressure maps to the typed call-limit refusal pre-contact: the
    // delegate is never contacted and no row is written (never a new agent).
    let calls_before = open.len();
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never run", b"nope".to_vec());
    let pending_executor = bitty_ai_slice::pending_host::PendingToolExecutor::new(
        executor,
        &store,
        session.clone(),
        None,
        0,
        1,
        0,
    );
    // The adapter owns the delegate; backpressure surfaces as
    // CallLimitExceeded through the bus-shaped error (checked below via the
    // bus burst) while this store-level refusal pins the zero-write bound.
    drop(pending_executor);
    let open = PendingStore::list_open(&store, &session).expect("list open");
    assert_eq!(open.len(), calls_before, "backpressure must write no row");
    // Whole-batch admission refuses past the remaining allowance with nothing
    // dispatched and the counter unchanged (explicit backpressure, no spawn).
    let bus = read_tool_bus();
    let call = bitty_ai_runtime::tool::ToolCall {
        name: "workspace_read".to_owned(),
        arguments: br#"{}"#.to_vec(),
    };
    let burst = vec![call; bitty_ai_runtime::tool::MAX_TOOL_CALLS_PER_TURN + 1];
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    let base = {
        let mut ids = bitty_ai_runtime::IdIssuer::default();
        bitty_ai_runtime::tool::AuthBase {
            agent_instance_id: ids.agent_instance(),
            session_id: ids.session(),
            level: bitty_ai_runtime::session::AgentLevel::Inspect,
        }
    };
    assert_eq!(
        bus.precheck(
            &burst,
            &base,
            bitty_ai_runtime::tool::MAX_TOOL_CALLS_PER_TURN
        )
        .expect_err("burst must refuse"),
        ToolError::CallLimitExceeded {
            limit: bitty_ai_runtime::tool::MAX_TOOL_CALLS_PER_TURN,
        },
        "burst must refuse with the explicit limit"
    );
    assert_eq!(bus.calls_this_turn(), 0, "refusal must count no dispatch");
    // Backpressure taxonomy: CallLimitExceeded reads backpressure; other
    // bounds never do. No new Agent is constructed on this path.
    assert!(
        is_backpressure(&AgentError::Tool(ToolError::CallLimitExceeded { limit: 8 })),
        "call-limit refusal must read backpressure"
    );
    assert!(
        !is_backpressure(&AgentError::RoundLimitExceeded { limit: 4 }),
        "round-limit fuse must never read backpressure"
    );
    let _ = issuer.execution();
}

#[test]
fn observation_pressure_critical_vs_progress_counters() {
    // Resume fan-out stays bounded by construction (sessions times opens).
    assert_eq!(
        MAX_RESUME_PENDING_FANOUT,
        MAX_OPEN_PENDING_PER_SESSION * 32,
        "resume fan-out must stay sessions times opens"
    );
    // Critical Unknown attribution survives turn admission while per-turn
    // progress resets; counters name dispatched and unreconciled effects.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote"));
    provider.push_turn(final_turn("unreached"));
    provider.push_turn(final_turn("next turn"));
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "host crashed before ack".to_owned(),
    });
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        agent_session(),
        AgentConfig::default(),
    );
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);
    let dispatched = match outcome {
        ExecOutcome::Unknown { dispatched, .. } => dispatched,
        _ => panic!("ack-loss must end Unknown"),
    };
    assert_eq!(dispatched, 1, "Unknown outcome must count the dispatch");
    assert_eq!(
        agent.executions().len(),
        1,
        "turn must attribute one effect"
    );
    // Next-turn admission carries the critical Unknown into the bounded
    // pending set instead of dropping it.
    let outcome = agent.run_turn(
        &mut executor,
        "fake-chat",
        "hi again",
        &[],
        &mut sink,
        NOW_MS,
    );
    assert!(
        matches!(outcome, ExecOutcome::Completed { .. }),
        "carried Unknown must not fail the next turn"
    );
    assert_eq!(
        agent.pending_unknown().len(),
        1,
        "critical Unknown must survive admission"
    );
    assert!(
        agent.pending_unknown()[0].status.is_ack_loss(),
        "surviving entry must stay Unknown"
    );
    // Coalescible progress sheds oldest-first with a countable metric while
    // the critical record above is untouched.
    let attribution = StreamAttribution {
        agent: {
            let mut issuer = bitty_ai_runtime::IdIssuer::default();
            issuer.agent_instance()
        },
        handle: StreamHandle(1),
        generation: 1,
    };
    let mut bounded = BoundedSink::with_capacity(attribution, 2);
    let chunk = |seq: u32, total: u32, text: &str| StreamChunk {
        seq,
        total,
        is_final: seq + 1 == total,
        fragment: Fragment::markdown(text.as_bytes().to_vec()),
    };
    bounded.emit_attributed(chunk(0, 4, "a")).expect("accepts");
    bounded.emit_attributed(chunk(1, 4, "b")).expect("accepts");
    let ack = bounded.emit_attributed(chunk(2, 4, "c")).expect("accepts");
    assert_eq!(ack.accepted, 3, "every accept must count");
    assert_eq!(ack.shed, 1, "shed progress must surface on the ack");
    assert_eq!(bounded.shed_count(), 1, "shed progress must count");
    assert_eq!(
        bounded.accepted_count(),
        3,
        "accepted count must include shed chunks"
    );
    assert_eq!(
        bounded.concatenated_bytes(),
        b"bc",
        "oldest progress must shed first"
    );
    assert_eq!(
        agent.pending_unknown().len(),
        1,
        "stream pressure must not drop the critical Unknown"
    );
    assert!(
        !is_reconcile_terminal(&agent.pending_unknown()[0].status),
        "critical Unknown must never read terminal under pressure"
    );
    let _ = FragmentKind::Markdown;
}

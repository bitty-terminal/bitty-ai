//! Cache-invalidation granularity denial proofs (AI-0078, AIQ-44 narrowing).
//!
//! AIQ-44 is needs-evidence: stale inputs must not produce a falsely current
//! PASS. The runtime has no cache layer, but context assembly reuses records
//! across rounds within a turn (`tool_history`, assembled records, artifact
//! references); this file pins the invalidation-relevant surface end to end
//! through the public API:
//!
//! 1. Superseded L1 records never resurface in later-round assembly,
//!    regardless of caller order or `tool_history` growth.
//! 2. A mid-turn `rotate_generation` downgrade is visible as a per-dispatch
//!    tier change at the authorization boundary (grant-invalidation
//!    visibility): remaining reads proceed under the downgraded tier while
//!    the hook observes the exact tier sequence.
//! 3. Artifact references are store-scoped: foreign/evicted references fail
//!    closed with typed absence, retained bytes resolve exactly, and the
//!    count cap fails closed with store identity unchanged.
//! 4. Re-assembly after `tool_history` growth keeps stable-prefix bytes for
//!    unchanged heads (prefix-stability, no false-current claim).
//!
//! Deterministic and offline: [`assemble`] over caller-built records,
//! scripted [`FakeProvider`] turns, deterministic executor/authorizer
//! doubles, caller-supplied `now_ms`. No network, no secrets, no wall clock,
//! no threads.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `context.rs` unit tests: supersede/dedupe mechanics, externalize
//!   atomicity, byte-cap exhaustion, duplicate-id and generation gates.
//! - `runtime_fail_closed.rs`: S-1 denial (remaining writes refused after a
//!   mid-turn rotation). Rotation appears here only for the visibility half
//!   that file leaves out (hook-observed tier sequence, reads proceeding
//!   under the downgraded tier).
//! - `result_schema_disclosure.rs`: disclosure rendering of denials,
//!   Unknown, S-2, and truncation. Denial/artifact surfaces appear here
//!   only as invalidation granularity, never as rendered schema.
//! - `turn_lifecycle.rs` / `agent_turn_semantics.rs`: per-turn reset and
//!   multi-turn continuation. Nothing here re-asserts those shapes.

use std::cell::RefCell;
use std::rc::Rc;

use bitty_ai_runtime::context::{EXTERNALIZE_THRESHOLD_BYTES, MAX_ARTIFACTS};
use bitty_ai_runtime::{
    Agent, AgentConfig, AgentLevel, AgentSession, ArtifactStore, AuthContext, AuthDecision,
    ContextError, ContextPriority, ContextRecord, ContextRequest, ElevationGrant, ExecOutcome,
    FakeProvider, IdIssuer, ProviderTurn, ProviderUsage, RecordBody, StableId, ToolAuthorizer,
    ToolBus, ToolCallRequest, ToolError, ToolExecutor, ToolRegistry, ToolSpec, ToolStatus,
    ToolSuccess, VecSink, assemble,
};

const NOW_MS: u64 = 1_700_000_000_000;
const GENERATION: u64 = 1;

fn ids() -> (
    bitty_ai_runtime::AgentInstanceId,
    bitty_ai_runtime::RunId,
    bitty_ai_runtime::SessionId,
) {
    let mut issuer = IdIssuer::default();
    (issuer.agent_instance(), issuer.run(), issuer.session())
}

fn session() -> AgentSession {
    let (agent, run, session) = ids();
    AgentSession::new(agent, run, session)
}

fn record(id: &str, summary: &str, body_byte: u8, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: GENERATION,
        collected_at_ms: 100,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![body_byte; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

/// Production-shaped tool-result entry, mirroring what
/// `Agent::record_execution` appends to `tool_history`: untrusted, workspace
/// provider, session generation, turn-scoped `exec-<n>` id.
fn tool_record(id: &str, summary: &str, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("agent-turn").expect("valid stable id"),
        generation: GENERATION,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![b't'; body_len]),
        supersedes: None,
        is_untrusted_surface: true,
    }
}

fn request_for(generation: u64) -> ContextRequest {
    ContextRequest {
        max_tokens: None,
        max_bytes: Some(32_768),
        current_generation: generation,
    }
}

// Proof 1: superseded L1 records never resurface in later-round assembly.

#[test]
fn superseded_records_never_resurface_in_later_round_assembly() {
    // Supersede chain v1 <- v2 <- v3. Distinct summaries/bodies so dedupe
    // cannot collapse them: only the trusted supersede links may prune.
    let v1 = record("read-v1", "outline revision one", b'a', 10);
    let mut v2 = record("read-v2", "outline revision two", b'b', 12);
    v2.supersedes = Some("read-v1".to_owned());
    let mut v3 = record("read-v3", "outline revision three", b'c', 14);
    v3.supersedes = Some("read-v2".to_owned());

    // Round one: v1 is superseded and stays out of the assembled refs.
    let mut store = ArtifactStore::new();
    let first = assemble(
        &[v1.clone(), v2.clone()],
        &mut store,
        &request_for(GENERATION),
    )
    .expect("round one assembles");
    assert_eq!(first.context_refs, vec!["read-v2".to_owned()]);
    assert_eq!(first.pruned_ids, vec!["read-v1".to_owned()]);

    // Later round: tool_history grew, and the superseded records are still
    // present in the input (retained history). In both caller orders the
    // superseded ids must never reappear in the assembled refs.
    let tool = tool_record("exec-7", "tool=workspace_read read ok", 8);
    for ordered in [
        vec![v1.clone(), v2.clone(), v3.clone(), tool.clone()],
        vec![v3.clone(), v2.clone(), v1.clone(), tool.clone()],
    ] {
        let mut later_store = ArtifactStore::new();
        let later = assemble(&ordered, &mut later_store, &request_for(GENERATION))
            .expect("later round assembles");
        assert!(
            !later.context_refs.contains(&"read-v1".to_owned()),
            "superseded v1 resurfaced: {:?}",
            later.context_refs
        );
        assert!(
            !later.context_refs.contains(&"read-v2".to_owned()),
            "superseded v2 resurfaced: {:?}",
            later.context_refs
        );
        assert!(later.context_refs.contains(&"read-v3".to_owned()));
        assert!(later.context_refs.contains(&tool.id));
        assert!(later.pruned_ids.contains(&"read-v1".to_owned()));
        assert!(later.pruned_ids.contains(&"read-v2".to_owned()));
        assert!(later.omitted_ids.is_empty());
    }
}

// Proof 2: mid-turn rotation invalidates remaining dispatches with
// grant-invalidation visibility (the untouched half beyond S-1 denial).

/// Test-only authorizer that allows everything but records the exact
/// `(tool, tier)` presented at each authorization boundary. The recorded
/// sequence is the grant-invalidation visibility probe: a mid-turn
/// downgrade must be observable here, not just as a terminal denial.
#[derive(Debug, Clone, Default)]
struct TierRecorder {
    seen: Rc<RefCell<Vec<(String, AgentLevel)>>>,
}

impl ToolAuthorizer for TierRecorder {
    fn authorize(&self, ctx: &AuthContext) -> AuthDecision {
        self.seen
            .borrow_mut()
            .push((ctx.tool.to_owned(), ctx.base.level));
        AuthDecision::Allow
    }
}

struct AllowElevation;
impl ElevationGrant for AllowElevation {
    fn elevation_granted(&self, _current: AgentLevel, _requested: AgentLevel) -> bool {
        true
    }
}

/// Test-only executor that rotates the shared session generation during the
/// first dispatch, downgrading the remaining dispatches mid-turn.
struct RotatingExecutor {
    session: AgentSession,
    calls: Vec<String>,
}

impl ToolExecutor for RotatingExecutor {
    fn execute(
        &mut self,
        tool: &str,
        _arguments: &[u8],
        _now_ms: u64,
    ) -> Result<ToolSuccess, ToolError> {
        self.calls.push(tool.to_owned());
        if self.calls.len() == 1 {
            self.session.rotate_generation();
        }
        Ok(ToolSuccess::new(format!("{tool} ok"), b"ok".to_vec()).expect("bounded success"))
    }
}

#[test]
fn mid_turn_rotation_is_visible_as_tier_downgrade_at_each_dispatch() {
    // Untouched half beyond the S-1 denial proof in `runtime_fail_closed.rs`
    // (which pins remaining *writes* refused): remaining *reads* proceed,
    // but observably under the downgraded tier. The hook-recorded sequence
    // proves the prior elevated grant no longer authorizes anything after
    // the rotation point.
    let sess = session();
    sess.elevate(AgentLevel::Workspace, &AllowElevation)
        .expect("grant allows elevation");
    let recorder = TierRecorder::default();
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "workspace_write",
                "Write a bounded workspace path",
                br#"{"type":"object"}"#.to_vec(),
                "workspace.write",
                false,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
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
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "write then read".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_write".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "done".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(
        provider,
        ToolBus::new(registry).with_authorizer(recorder.clone()),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = RotatingExecutor {
        session: sess.clone(),
        calls: Vec::new(),
    };
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    assert!(
        matches!(outcome, ExecOutcome::Completed { .. }),
        "remaining read under the downgraded tier must complete, got: {outcome:?}"
    );
    // Rotation happened mid-turn through the shared handle: the grant is
    // visibly revoked, not silently kept.
    assert_eq!(sess.level(), AgentLevel::Inspect);
    assert_eq!(sess.generation(), GENERATION + 1);
    // Both dispatches reached the host; invalidation downgrades authority,
    // it never freezes the turn.
    assert_eq!(executor.calls, vec!["workspace_write", "workspace_read"]);
    assert_eq!(agent.executions().len(), 2);
    assert!(
        agent
            .executions()
            .iter()
            .all(|record| matches!(record.status, ToolStatus::Success)),
        "unexpected executions: {:?}",
        agent.executions()
    );
    // Grant-invalidation visibility: the hook observed the elevated tier up
    // to the rotation point (precheck x2, first dispatch) and exactly the
    // downgraded tier after it. No dispatch after rotation ever presented
    // the stale elevated grant.
    assert_eq!(
        *recorder.seen.borrow(),
        vec![
            ("workspace_write".to_owned(), AgentLevel::Workspace),
            ("workspace_read".to_owned(), AgentLevel::Workspace),
            ("workspace_write".to_owned(), AgentLevel::Workspace),
            ("workspace_read".to_owned(), AgentLevel::Inspect),
        ]
    );
}

// Proof 3: artifact store full/evicted references fail closed (no stale bytes).

#[test]
fn foreign_artifact_references_fail_closed_without_stale_bytes() {
    // An assembled reference is store-scoped, not a stable content address:
    // resolving it against a store that never retained those bytes (the
    // evicted/foreign case) must fail with typed absence, never stale bytes.
    let mut store = ArtifactStore::new();
    let big = record("big", "zone dump", b'z', EXTERNALIZE_THRESHOLD_BYTES + 1024);
    let assembled = assemble(&[big], &mut store, &request_for(GENERATION)).expect("assemble");
    assert_eq!(assembled.externalized, 1);
    let reference = match &assembled.records[0].content {
        bitty_ai_runtime::AssembledContent::Reference(reference) => reference.clone(),
        other => panic!("large body must externalize, got: {other:?}"),
    };

    // The owning store resolves the exact retained bytes (no stale mix).
    assert_eq!(
        store
            .resolve(&reference)
            .expect("owning store resolves its reference"),
        vec![b'z'; EXTERNALIZE_THRESHOLD_BYTES + 1024].as_slice()
    );

    // A foreign store (evicted scope) reports typed absence carrying the
    // exact dangling reference: attributable, never substituted.
    let foreign = ArtifactStore::new();
    let err = foreign
        .resolve(&reference)
        .expect_err("foreign reference must fail");
    assert!(
        matches!(
            &err,
            ContextError::ArtifactUnavailable { reference: name } if name == reference.as_str()
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn retained_artifacts_resolve_to_exact_bytes_without_cross_talk() {
    // Two retained artifacts resolve to exactly their own bytes: no
    // cross-talk, no stale prefix from an earlier entry.
    let mut store = ArtifactStore::new();
    let first = assemble(
        &[record(
            "first",
            "zone dump one",
            b'a',
            EXTERNALIZE_THRESHOLD_BYTES + 16,
        )],
        &mut store,
        &request_for(GENERATION),
    )
    .expect("first assembles");
    let second = assemble(
        &[record(
            "second",
            "zone dump two",
            b'b',
            EXTERNALIZE_THRESHOLD_BYTES + 32,
        )],
        &mut store,
        &request_for(GENERATION),
    )
    .expect("second assembles");
    let ref_of =
        |assembled: &bitty_ai_runtime::AssembledContext| match &assembled.records[0].content {
            bitty_ai_runtime::AssembledContent::Reference(reference) => reference.clone(),
            other => panic!("large body must externalize, got: {other:?}"),
        };
    let first_ref = ref_of(&first);
    let second_ref = ref_of(&second);
    assert_ne!(first_ref, second_ref);
    assert_eq!(
        store.resolve(&first_ref).expect("first resolves"),
        vec![b'a'; EXTERNALIZE_THRESHOLD_BYTES + 16].as_slice()
    );
    assert_eq!(
        store.resolve(&second_ref).expect("second resolves"),
        vec![b'b'; EXTERNALIZE_THRESHOLD_BYTES + 32].as_slice()
    );
}

#[test]
fn store_count_cap_fails_closed_with_identity_unchanged() {
    // Direct complements to the byte-cap exhaustion pinned in
    // `context.rs`: filling the count cap fails the next store with the
    // typed reason and leaves count, bytes, and id sequence unchanged.
    let mut store = ArtifactStore::new();
    for _ in 0..MAX_ARTIFACTS {
        store.store(vec![b'a'; 8]).expect("fill fits");
    }
    assert_eq!(store.len(), MAX_ARTIFACTS);
    let before_bytes = store.total_bytes();
    let before_next = store.next_id();
    let err = store
        .store(vec![b'b'; 8])
        .expect_err("count-full store must fail");
    assert!(
        matches!(err, ContextError::ArtifactStoreFull { .. }),
        "unexpected error: {err:?}"
    );
    assert_eq!(store.len(), MAX_ARTIFACTS);
    assert_eq!(store.total_bytes(), before_bytes);
    assert_eq!(store.next_id(), before_next);
}

// Proof 4: re-assembly after tool_history growth keeps stable-prefix bytes.

#[test]
fn reassembly_after_tool_history_growth_keeps_stable_prefix_bytes() {
    // Prefix-stability without a false-current claim: after tool_history
    // growth, re-assembly keeps the unchanged head byte-identical (same
    // ids, same order, same payload bytes) and only appends the new tail.
    let heads = vec![
        record("seed-one", "manifest outline", b'm', 64),
        record("seed-two", "git stat outline", b'g', 64),
        record("seed-three", "diagnostics notes", b'd', 64),
    ];
    let mut store = ArtifactStore::new();
    let first = assemble(&heads, &mut store, &request_for(GENERATION)).expect("first assembles");
    assert!(first.omitted_ids.is_empty());
    assert!(first.pruned_ids.is_empty());

    let grown: Vec<ContextRecord> = heads
        .iter()
        .cloned()
        .chain([
            tool_record("exec-11", "tool=workspace_read read ok", 16),
            tool_record("exec-12", "tool=workspace_read read ok again", 16),
        ])
        .collect();
    let second = assemble(&grown, &mut store, &request_for(GENERATION)).expect("second assembles");
    assert!(second.omitted_ids.is_empty());
    assert!(second.pruned_ids.is_empty());

    fn head_bytes(
        assembled: &bitty_ai_runtime::AssembledContext,
    ) -> Vec<(String, String, String, Vec<u8>)> {
        assembled
            .records
            .iter()
            .map(|record| {
                let payload = match &record.content {
                    bitty_ai_runtime::AssembledContent::Inline(bytes) => bytes.clone(),
                    bitty_ai_runtime::AssembledContent::Reference(reference) => {
                        reference.as_str().as_bytes().to_vec()
                    }
                };
                (
                    record.id.clone(),
                    record.provider.clone(),
                    record.summary.clone(),
                    payload,
                )
            })
            .collect()
    }
    let before = head_bytes(&first);
    let after = head_bytes(&second);
    // Stable prefix: every head byte identical, growth purely additive.
    assert_eq!(&after[..before.len()], before.as_slice());
    assert_eq!(after.len(), before.len() + 2);
    assert_eq!(
        second.context_refs,
        vec![
            "seed-one".to_owned(),
            "seed-two".to_owned(),
            "seed-three".to_owned(),
            "exec-11".to_owned(),
            "exec-12".to_owned(),
        ]
    );
}

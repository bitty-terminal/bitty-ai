//! Invalidation-granularity stale-PASS denial proofs (AI-0094, AIQ-44 narrowing).
//!
//! AIQ-44 is needs-evidence: when a cached entry's invalidation granularity
//! (which generation, scope, or artifact-set it covers) does not cover the
//! current request, the request must be denied (stale-PASS), never served
//! from the stale entry. The runtime has no cache layer to consult, but the
//! granularity-to-denial mechanism exists at four seams, and this file pins
//! each one end to end through the public API:
//!
//! 1. Generation granularity: a record cached at generation N never serves a
//!    request at generation N+1. `assemble` rejects the whole batch with
//!    `StaleGeneration` (fail closed, store untouched); re-assembly under the
//!    new generation denies rather than resurfacing the entry. The seed for a
//!    `run_turn` after `rotate_generation` is therefore denied before any
//!    provider I/O, session marked failed.
//! 2. Scope granularity: a [`CacheKey`] at scope Turn never equals a
//!    [`CacheKey`] at scope Session or Round over the same bytes, so a
//!    provider-scoped prefix entry cached at one scope can never serve a
//!    lookup at another scope (miss, never stale hit). Hit/miss accounting
//!    confirms: a same-scope repeat hits, a scope-crossing repeat misses.
//! 3. Artifact-set granularity: an assembled reference resolves only against
//!    the store that retained its bytes. A prior artifact-set entry presented
//!    to a store at a different scope (evicted/foreign) resolves as typed
//!    absence (`ArtifactUnavailable` carrying the exact dangling reference),
//!    never as substituted bytes; the owning store still resolves exactly.
//! 4. Grant granularity: a mid-turn grant tightening (rotation downgrading
//!    the session tier) denies remaining mutating dispatches at the fresh
//!    tier boundary while the executor call count proves the denied call
//!    never reached the host. Already-dispatched effects are kept, never
//!    rolled back; the denied remainder fails the turn.
//!
//! Gap verdict: NO-GAP (tests-only, no production change). Every granularity
//! mismatch already denies through the mechanism above: the generation gate
//! in `assemble` (unit-proven in `context.rs`), the scope field in `CacheKey`
//! equality (unit-proven in `cache_key.rs`), store-scoped `resolve`, and the
//! per-call JIT `AuthBase` re-read in `Agent::run_turn`. This file pins the
//! end-to-end denial half — exact errors, exact keys, untouched stores, zero
//! host dispatches — that those files leave out.
//!
//! Deterministic and offline: caller-built records, canonical bytes from
//! [`assemble_prompt`] over caller-built snapshots, scripted [`FakeProvider`]
//! turns, deterministic executor/authorizer doubles, caller-supplied `now_ms`.
//! No network, no secrets, no wall clock, no threads, no `HashMap`.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `cache_invalidation.rs`: supersede pruning, rotation visibility (reads
//!   proceeding under the downgraded tier with the hook-observed tier
//!   sequence), foreign-artifact/count-cap absence, reassembly
//!   prefix-stability. Generation gating appears here only as the
//!   cross-generation stale-PASS denial (exact `StaleGeneration` fields,
//!   store identity, turn-level `Failed` with zero provider I/O); rotation
//!   appears here only as the grant-tightening denial half (writes refused,
//!   host untouched); artifacts appear here only as the scope-crossing
//!   resolution denial (deny, never serve), never as store identity/caps.
//! - `cache_key.rs`: key-scope rule (inequality matrix), determinism, stable
//!   vs trailing changes, hit-rate harness, constructor failures, embedded
//!   markers. Scope equality appears here only as the lookup-granularity
//!   denial (same-scope hit vs cross-scope miss accounting through a
//!   scripted seen-list lookup), never as key construction.
//! - `context.rs` unit tests: supersede/dedupe mechanics, externalize
//!   atomicity, duplicate-id and generation gates at the unit level. The
//!   generation gate appears here only as the end-to-end (cross-generation
//!   entry, turn-level denial) half those tests leave out.
//! - `runtime_fail_closed.rs`: S-1 denial (remaining writes refused after a
//!   mid-turn rotation). The rotation denial appears here only as the
//!   granularity-framed tightening sequence (precheck-then-rotate-then-deny
//!   with hook-observed tier drop and zero host dispatch), asserted via the
//!   grant fields that file leaves out.
//! - `writer_fencing.rs` / `schema_invalidation.rs`: lease-generation and
//!   schema-digest invalidation. Nothing here re-asserts those surfaces.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! ids, generations, digests, and references never appear in message strings.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AgentLevel, AgentSession, ArtifactStore, AssembledPrompt,
    AuthContext, AuthDecision, CacheKey, CacheScope, ContextError, ContextPriority, ContextRecord,
    ContextRequest, ElevationGrant, ExecOutcome, FakeProvider, FakeToolExecutor, IdIssuer,
    LayerInput, ModelProvider, PromptLayer, PromptSnapshot, ProviderTurn, ProviderUsage,
    RecordBody, StableId, ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolExecutor,
    ToolRegistry, ToolSpec, ToolStatus, ToolSuccess, VecSink, assemble, assemble_prompt,
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

fn record(id: &str, summary: &str, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: GENERATION,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![b'x'; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn big_record(id: &str, summary: &str, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: GENERATION,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![b'z'; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn request_for(generation: u64) -> ContextRequest {
    ContextRequest {
        max_tokens: None,
        max_bytes: Some(32_768),
        current_generation: generation,
    }
}

fn text_layer(layer: PromptLayer, text: &str) -> LayerInput {
    LayerInput::text_only(layer, text)
}

fn canonical_bytes(turn_text: &str) -> Vec<u8> {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::Project, "stable project"),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            text_layer(PromptLayer::RuntimeTurn, turn_text),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot)
        .expect("test snapshot assembles")
        .canonical_bytes()
        .to_vec()
}

fn scoped_key(scope: CacheScope, bytes: &[u8]) -> CacheKey {
    CacheKey::new("bitty-fake", "fake-chat", scope, bytes).expect("valid test key inputs")
}

fn read_write_tool_bus(authorizer: AllowWrites) -> ToolBus {
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
    ToolBus::new(registry).with_authorizer(authorizer)
}

/// Test-only authorizer that allows everything. This exists so denial tests
/// isolate the granularity mechanism (generation gate, tier floor) from
/// hook policy; production wiring installs the host
/// capability-plus-consent hook instead.
#[derive(Debug, Clone, Copy, Default)]
struct AllowWrites;

impl ToolAuthorizer for AllowWrites {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

struct AllowElevation;
impl ElevationGrant for AllowElevation {
    fn elevation_granted(&self, _current: AgentLevel, _requested: AgentLevel) -> bool {
        true
    }
}

// Proof 1: cross-generation cached entries never serve post-rotation.

#[test]
fn cross_generation_entry_denies_with_exact_stale_generation() {
    // A record cached at generation N presented to a request at generation
    // N+1 must deny with the exact stale entry identity, never assemble.
    let mut cached = record("cached-head", "manifest outline", 64);
    assert_eq!(cached.generation, GENERATION);
    let mut store = ArtifactStore::new();
    let err = assemble(&[cached.clone()], &mut store, &request_for(GENERATION + 1))
        .expect_err("stale-generation entry must deny");
    assert!(
        matches!(
            err,
            ContextError::StaleGeneration {
                ref id,
                actual: GENERATION,
                current: expect_current
            } if id == "cached-head" && expect_current == GENERATION + 1
        ),
        "cross-generation entry must deny as StaleGeneration"
    );
    // Denial decided nothing: the store is untouched.
    assert!(store.is_empty());
    assert_eq!(store.total_bytes(), 0);
    assert_eq!(store.next_id(), 0);
    // Control: the same entry assembles under its own generation.
    cached.generation = GENERATION + 1;
    let mut control_store = ArtifactStore::new();
    let assembled = assemble(&[cached], &mut control_store, &request_for(GENERATION + 1))
        .expect("same-generation entry assembles");
    assert_eq!(assembled.context_refs, vec!["cached-head".to_owned()]);
}

#[test]
fn post_rotation_turn_seed_denies_before_provider_io() {
    // End-to-end stale-PASS denial: after rotation the session is at
    // generation 2, but the seed still carries the cached generation 1
    // entry. The turn must fail as `Context(StaleGeneration)` with zero
    // provider rounds consumed and the session marked failed: the stale
    // entry never serves the new generation.
    let sess = session();
    assert_eq!(sess.generation(), GENERATION);
    sess.rotate_generation();
    assert_eq!(sess.generation(), GENERATION + 1);
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowWrites),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(
        &mut executor,
        "fake-chat",
        "hi",
        &[record("cached-head", "manifest outline", 16)],
        &mut sink,
        NOW_MS,
    );

    let error = match &outcome {
        ExecOutcome::Failed { error } => error.clone(),
        ExecOutcome::Completed { .. }
        | ExecOutcome::Canceled { .. }
        | ExecOutcome::Unknown { .. } => panic!("stale seed must fail the turn"),
    };
    assert!(
        matches!(
            error,
            AgentError::Context(ContextError::StaleGeneration { .. })
        ),
        "post-rotation seed must deny as stale generation"
    );
    // No provider I/O happened: the script and counters are untouched, no
    // executions were attributed, and the sink stayed empty.
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 0);
    assert!(agent.executions().is_empty());
    assert!(executor.calls().is_empty());
    assert!(sink.is_empty());
    assert_eq!(sess.state(), bitty_ai_runtime::SessionState::Failed);
}

// Proof 2: scope-mismatched keys never serve (miss, never stale hit).

/// Test-only seen-list lookup: linear scan, no `HashMap`, no `RandomState`.
/// Returns true on a repeat (hit), false on a first-seen key (miss).
fn lookup(seen: &mut Vec<CacheKey>, key: CacheKey) -> bool {
    if seen.contains(&key) {
        true
    } else {
        seen.push(key);
        false
    }
}

#[test]
fn scope_mismatched_key_never_serves_cached_entry() {
    // The same bytes keyed at Turn scope populate the seen-list (a cached
    // entry). The identical bytes at Session and Round scope must MISS:
    // scope is part of key equality, so a coarser or finer entry can never
    // serve a lookup at another scope.
    let bytes = canonical_bytes("turn one");
    let turn_key = scoped_key(CacheScope::Turn, &bytes);
    let session_key = scoped_key(CacheScope::Session, &bytes);
    let round_key = scoped_key(CacheScope::Round, &bytes);
    assert_ne!(turn_key, session_key);
    assert_ne!(turn_key, round_key);
    assert_ne!(session_key, round_key);
    let mut seen: Vec<CacheKey> = Vec::new();
    assert!(!lookup(&mut seen, turn_key.clone()));
    assert_eq!(seen.len(), 1);
    // Cross-scope repeats miss: the Turn entry never serves Session/Round.
    assert!(!lookup(&mut seen, session_key.clone()));
    assert!(!lookup(&mut seen, round_key.clone()));
    assert_eq!(seen.len(), 3);
    // Control: same-scope repeats hit. The hit proves the lookup admits
    // entries, so the misses above are scope denials, not a dead cache.
    assert!(lookup(&mut seen, turn_key));
    assert!(lookup(&mut seen, session_key));
    assert!(lookup(&mut seen, round_key));
    assert_eq!(seen.len(), 3);
}

#[test]
fn stable_prefix_change_denies_reuse_while_tail_change_serves() {
    // Granularity inside the hashed region: a stable-region change must deny
    // reuse (miss), while a trailing-only change keeps serving (hit). The
    // pinned per-lookup flags make stale-PASS vs fresh-HIT exact.
    let mut seen: Vec<CacheKey> = Vec::new();
    let base = canonical_bytes("turn one");
    assert!(!lookup(&mut seen, scoped_key(CacheScope::Session, &base)));
    // Trailing-only change: same key, hit (the entry covers this request).
    let tail_moved = canonical_bytes("turn two");
    assert_ne!(base, tail_moved);
    assert!(lookup(
        &mut seen,
        scoped_key(CacheScope::Session, &tail_moved)
    ));
    assert_eq!(seen.len(), 1);
    // Stable-region change: different key, miss (the entry does not cover
    // this request, so the request denies the cached entry). Flip one byte
    // inside the hashed stable prefix at a fixed offset (the User text sits
    // strictly inside the prefix for these controlled snapshots): the layout
    // stays valid, so construction succeeds with a moved digest.
    let mut stable_moved = base.clone();
    let prefix_len = scoped_key(CacheScope::Session, &base).prefix_len;
    assert!(prefix_len > 64);
    let stable_offset = prefix_len - 8;
    if stable_moved[stable_offset] == b'a' {
        stable_moved[stable_offset] = b'b';
    } else {
        stable_moved[stable_offset] = b'a';
    }
    let stable_key = scoped_key(CacheScope::Session, &stable_moved);
    assert_ne!(scoped_key(CacheScope::Session, &base), stable_key);
    assert!(!lookup(&mut seen, stable_key));
    assert_eq!(seen.len(), 2);
}

// Proof 3: artifact-set change invalidates the prior resolution.

#[test]
fn artifact_set_change_denies_prior_resolution_with_typed_absence() {
    // An entry externalized into one artifact set (store) presented to a
    // store at a different scope (evicted/foreign set) must deny with typed
    // absence carrying the exact dangling reference: attributable, never
    // substituted with stale bytes from the foreign set.
    let mut owner = ArtifactStore::new();
    let big = big_record(
        "zone-dump",
        "zone dump",
        bitty_ai_runtime::context::EXTERNALIZE_THRESHOLD_BYTES + 64,
    );
    let assembled = assemble(
        std::slice::from_ref(&big),
        &mut owner,
        &request_for(GENERATION),
    )
    .expect("assemble");
    assert_eq!(assembled.externalized, 1);
    let reference = match &assembled.records[0].content {
        bitty_ai_runtime::AssembledContent::Reference(reference) => reference.clone(),
        bitty_ai_runtime::AssembledContent::Inline(_) => {
            panic!("large body must externalize")
        }
    };
    // Owning set still serves exactly: denial is scoped, not destructive.
    assert_eq!(
        owner
            .resolve(&reference)
            .expect("owning store resolves its reference")
            .len(),
        bitty_ai_runtime::context::EXTERNALIZE_THRESHOLD_BYTES + 64
    );
    // A store holding unrelated retained inline bytes (a live distinct
    // set) must still deny the foreign reference rather than serve its own
    // stale bytes in its place. The foreign bytes stay Inline (under the
    // externalize threshold), so the only externalized reference in play is
    // the dangling one under test.
    let mut foreign = ArtifactStore::new();
    let filler = record("filler", "filler summary", 128);
    assemble(&[filler], &mut foreign, &request_for(GENERATION)).expect("foreign set assembles");
    assert_eq!(foreign.len(), 0);
    assert_eq!(foreign.total_bytes(), 0);
    let err = foreign
        .resolve(&reference)
        .expect_err("foreign reference must deny");
    assert!(
        matches!(
            &err,
            ContextError::ArtifactUnavailable { reference: name }
                if name == reference.as_str()
        ),
        "artifact-set change must deny with typed absence"
    );
    // Denial decided nothing on the foreign set: no artifact retained.
    assert_eq!(foreign.len(), 0);
    assert_eq!(foreign.total_bytes(), 0);
    assert_eq!(foreign.next_id(), 0);
}

// Proof 4: stale entry under a tightened grant denies rather than serves.

/// Test-only executor that tightens the shared session grant during the
/// first dispatch (rotation resets the tier to Inspect), then records the
/// exact calls that reached the host.
struct TighteningExecutor {
    session: AgentSession,
    calls: Vec<String>,
}

impl ToolExecutor for TighteningExecutor {
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
fn stale_entry_under_tightened_grant_denies_without_host_dispatch() {
    // Grant granularity: the turn starts elevated (Workspace) and the first
    // dispatch tightens the grant mid-turn (rotation drops the tier to
    // Inspect). The remaining mutating call was authorized under the stale
    // grant; it must now deny at the fresh tier boundary. The executor call
    // count proves the denied call never reached the host: deny, never
    // serve from the stale grant.
    let sess = session();
    sess.elevate(AgentLevel::Workspace, &AllowElevation)
        .expect("grant allows elevation");
    assert_eq!(sess.level(), AgentLevel::Workspace);
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "two writes".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_write".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_write".to_owned(),
                arguments: br#"{"path":"b"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowWrites),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = TighteningExecutor {
        session: sess.clone(),
        calls: Vec::new(),
    };
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let error = match &outcome {
        ExecOutcome::Failed { error } => error.clone(),
        ExecOutcome::Completed { .. }
        | ExecOutcome::Canceled { .. }
        | ExecOutcome::Unknown { .. } => panic!("tightened grant must fail the turn"),
    };
    assert!(
        matches!(error, AgentError::Tool(ToolError::Denied { .. })),
        "stale grant must deny the remaining write"
    );
    // The grant visibly tightened through the shared handle.
    assert_eq!(sess.level(), AgentLevel::Inspect);
    assert_eq!(sess.generation(), GENERATION + 1);
    // The denied call never reached the host executor.
    assert_eq!(executor.calls, vec!["workspace_write".to_owned()]);
    // The already-dispatched effect is kept, never rolled back; the denied
    // remainder is a pre-dispatch admission refusal (`Refused`, AI-RUN-004),
    // never an executed-failure attribution: the host never saw it.
    assert_eq!(agent.executions().len(), 2);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(agent.executions()[1].status.is_admission_refusal());
    assert!(matches!(
        agent.executions()[1].status,
        ToolStatus::Refused { .. }
    ));
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(sess.state(), bitty_ai_runtime::SessionState::Failed);
}

#[test]
fn prompt_grant_narrowing_denies_rather_than_serves() {
    // Grant narrowing at the prompt seam: [`AssembledPrompt`] denies a tool
    // the dispatcher would grant once a lower layer narrows the allow-set
    // past it. The stale (pre-narrowing) entry must never serve: the check
    // denies with the exact tool identity.
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            LayerInput::text_only(PromptLayer::CoreContract, "core"),
            LayerInput::text_only(PromptLayer::User, "user"),
            LayerInput::text_only(PromptLayer::Project, "project"),
            LayerInput::text_only(PromptLayer::SkillsProfile, "stable skills"),
            LayerInput {
                layer: PromptLayer::RuntimeTurn,
                text: "turn".to_owned(),
                allowed_tools: Some(vec!["workspace_read".to_owned()]),
                denied_tools: Vec::new(),
                budget_ceiling_bytes: None,
                allowed_scopes: None,
                directives: Vec::new(),
            },
        ],
    )
    .expect("test snapshot is valid");
    let prompt: AssembledPrompt = assemble_prompt(&snapshot).expect("test prompt assembles");
    // Dispatcher would serve; the narrowed prompt denies instead.
    assert!(bitty_ai_runtime::is_dispatch_allowed(
        &prompt,
        "workspace_read",
        true
    ));
    let err = bitty_ai_runtime::check_dispatch(&prompt, "workspace_write", true)
        .expect_err("narrowed tool must deny");
    assert!(
        matches!(
            err,
            bitty_ai_runtime::PromptError::PromptNotAllowed { ref tool }
                if tool == "workspace_write"
        ),
        "narrowed grant must deny rather than serve"
    );
    // The denial is tool-scoped: the allowed tool still dispatches.
    assert!(bitty_ai_runtime::is_dispatch_allowed(
        &prompt,
        "workspace_read",
        true
    ));
}

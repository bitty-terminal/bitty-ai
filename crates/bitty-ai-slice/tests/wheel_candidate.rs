//! Candidate Wheel falsifiability harness (AI-0236, beyond-v0.1).
//!
//! Candidate wording only: every test below pins a Wheel verification item
//! (W-V1 demotion survival, W-V3 temperature orthogonality, W-V4 per-claim
//! bytes-at-revision, W-V5 rationale-alone reconstruction, W-V8 lossy
//! projection with reversible provenance, W-V9 retention-then-propagation,
//! W-V10 terminal-change / AI-contract-no-change matrix) as a falsifiable,
//! reviewable test. Beyond the v0.1 scope: NOTHING here is promotable to
//! accepted architecture or to the v0.1 contract, and no comment makes an
//! accepted-status claim. No schema change, no package split, no new
//! dependency/thread/network; all clocks are caller-supplied `*_ms` values,
//! all stores are in-memory, and every assertion runs offline and
//! deterministically.
//!
//! Temperature note: W-V2 stays owner-pending and uncalibrated, so no test
//! invents a default, policy, or threshold. W-V3 labels are caller-supplied
//! ([`CandidateTemperature`]); demotion reuses the existing Tier-1
//! `scratch/*` + `temp/*` rule and the existing pinned-surface/cache-affinity
//! hints only.
//!
//! What each test pins:
//!
//! - `w_v1_*`: finding+ref answers the same question after raw evidence is
//!   demoted from the compiled view, while the raw blob stays resolvable by
//!   hash (plus a negative: unknown hashes are not resolvable).
//! - `w_v3_*`: Hot evidence and Cold knowledge coexist in one tree; a pinned
//!   Hot slot survives Tier-1 while a stale scratch slot demotes; Cold slots
//!   outside `scratch/*` + `temp/*` are untouched (plus a negative: absent
//!   slots break coexistence).
//! - `w_v4_*`: each claim resolves independently to its bytes at its revision
//!   (plus negatives: rotated hash, unknown slot, unknown revision).
//! - `w_v5_*`: a later agent holding only the checkpoint row reconstructs the
//!   same "why" string from the [`Rationale`] alone with no raw-thinking input
//!   (plus a negative: empty `why` is refused at validation).
//! - `w_v8_*`: the lossy projection is strictly shorter than the original yet
//!   reverses to the exact original bytes via stored provenance (plus
//!   negatives: unknown provenance and length mismatch resolve to absence).
//! - `w_v9_*`: an unreferenced fork stays retrievable before GC (retention has
//!   not ended), preview byte-matches the destructive batch, and after GC the
//!   fork row is gone while live state still dereferences (negatives: fork
//!   blobs pruned, live blobs kept).
//! - `w_v10_*`: the 2x2 terminal-change / contract-change matrix holds behind
//!   both the inherent and the facade compile paths (facade/host-conformance
//!   style: shared assertions across two serving paths), and the pure matrix
//!   rule is exactly "stable iff contract unchanged".

use bitty_ai_slice::content_store::{
    Checkpoint, CheckpointDraft, ContentHash, ContentStore, Rationale,
};
use bitty_ai_slice::context_compiler::{
    CandidateClaimRef, CandidateProjection, CandidateTempSlot, CandidateTemperature,
    CompiledContext, CompilerBudgetConfig, ContextCompiler, ContextTree, EntryKind, TreeEntry,
    candidate_answer_with_finding, candidate_finding_present_after_demotion,
    candidate_hot_cold_coexist, candidate_pinned_survives_tier1, candidate_project_lossy,
    candidate_raw_still_resolvable, candidate_reconstruct_why, candidate_resolve_claim,
    candidate_reverse_projection, candidate_zone1_moves_with_contract_change,
    candidate_zone1_stable_after_terminal_change,
};
use bitty_ai_slice::facade::AiEngine;
use bitty_ai_slice::merge::{
    CandidateBoundaryCell, GcOptions, candidate_blob_retrievable,
    candidate_boundary_expects_stable_prefix, candidate_checkpoint_retrievable,
    candidate_deletion_propagated_to_checkpoint, collect_garbage, gc_preview,
};

fn test_rationale() -> Rationale {
    Rationale::new("candidate why", "candidate what")
}

fn commit_tree(
    store: &mut ContentStore,
    parents: Vec<ContentHash>,
    tree: &ContextTree,
    summary: &str,
    at_ms: u64,
) -> Checkpoint {
    let bytes = tree.canonical_bytes();
    let draft = CheckpointDraft {
        parents,
        task_id: "AI-0236".to_string(),
        agent_id: "ai-0236-impl".to_string(),
        rationale: test_rationale(),
        tree_hash: None,
        summary: summary.to_string(),
        timestamp_ms: at_ms,
    };
    store
        .commit_checkpoint_atomic(&bytes, at_ms, draft, &[], at_ms)
        .expect("setup commit")
}

// ── W-V1 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v1_finding_answers_after_raw_demoted_and_raw_stays_resolvable() {
    let mut store = ContentStore::open_in_memory().expect("store");
    let raw: &[u8] = b"raw terminal evidence: 400-line log";
    let finding_text = "finding: the log shows a missing match arm in crates/core";
    let raw_hash = store.put_blob(raw, 1000).expect("raw blob");
    let finding_hash = store
        .put_blob(finding_text.as_bytes(), 1001)
        .expect("finding blob");

    let mut tree = ContextTree::new();
    tree.insert(
        TreeEntry::new(
            "finding/log-analysis",
            finding_hash,
            EntryKind::Blob,
            finding_text.len(),
        )
        .expect("finding slot"),
    );
    tree.insert(
        TreeEntry::new("scratch/raw-evidence", raw_hash, EntryKind::Blob, raw.len())
            .expect("raw slot"),
    );

    let question = "what does the log show?";
    let answer_before = candidate_answer_with_finding(question, finding_text);

    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "sys".to_string();
    compiler.context_tree = tree;
    let compiled_before = compiler
        .compile(&CompilerBudgetConfig::default())
        .expect("generous compiles");
    assert!(candidate_finding_present_after_demotion(
        &compiled_before,
        "finding/log-analysis"
    ));
    assert!(
        compiled_before.zone2_state.contains("scratch/raw-evidence"),
        "generous view still names the raw slot"
    );

    // Tight Zone 2 budget triggers Tier-1 demotion of the scratch raw slot
    // (existing rule; two slot lines exceed 200 bytes, one does not).
    let tight = CompilerBudgetConfig {
        max_total_bytes: 64 * 1024,
        max_zone1_bytes: 16 * 1024,
        max_zone2_bytes: 200,
        max_zone3_bytes: 16 * 1024,
    };
    let compiled_after = compiler.compile(&tight).expect("tight compiles");
    assert!(
        compiled_after
            .pruned_slots
            .contains(&"scratch/raw-evidence".to_string()),
        "Tier-1 must demote the scratch raw slot, pruned={:?}",
        compiled_after.pruned_slots
    );
    assert!(candidate_finding_present_after_demotion(
        &compiled_after,
        "finding/log-analysis"
    ));
    let answer_after = candidate_answer_with_finding(question, finding_text);
    assert_eq!(
        answer_before, answer_after,
        "finding+ref answers the same question after raw demotion"
    );
    assert!(candidate_raw_still_resolvable(&store, &raw_hash));
    assert_eq!(
        store.get_blob(&raw_hash).expect("raw read").as_deref(),
        Some(raw),
        "demoted raw evidence stays resolvable by hash"
    );
    assert!(
        !candidate_raw_still_resolvable(&store, &ContentHash::compute(b"never stored")),
        "unknown hashes are not resolvable"
    );
}

// ── W-V3 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v3_temperature_orthogonality_pinned_hot_stale_demotes() {
    // Caller-supplied labels only; no thresholds/defaults/policies (W-V2
    // owner-pending and uncalibrated).
    let hot = CandidateTempSlot {
        slot: "evidence/hot-obs".to_string(),
        temperature: CandidateTemperature::Hot,
        pinned: false,
    };
    let cold = CandidateTempSlot {
        slot: "knowledge/cold-doc".to_string(),
        temperature: CandidateTemperature::Cold,
        pinned: false,
    };
    let pinned_hot = CandidateTempSlot {
        slot: "pinned/contract".to_string(),
        temperature: CandidateTemperature::Hot,
        pinned: true,
    };
    assert_ne!(
        hot.temperature, cold.temperature,
        "labels must distinguish Hot from Cold"
    );
    assert!(pinned_hot.pinned);

    let mut store = ContentStore::open_in_memory().expect("store");
    let hot_bytes: &[u8] = b"hot observation bytes";
    let cold_bytes: &[u8] = b"cold background knowledge bytes";
    let pinned_bytes: &[u8] = b"pinned contract text";
    let stale_bytes: &[u8] = b"stale scratch note";
    let hot_hash = store.put_blob(hot_bytes, 1000).expect("hot blob");
    let cold_hash = store.put_blob(cold_bytes, 1001).expect("cold blob");
    let pinned_hash = store.put_blob(pinned_bytes, 1002).expect("pinned blob");
    let stale_hash = store.put_blob(stale_bytes, 1003).expect("stale blob");

    let mut tree = ContextTree::new();
    tree.insert(
        TreeEntry::new(
            hot.slot.as_str(),
            hot_hash,
            EntryKind::Blob,
            hot_bytes.len(),
        )
        .expect("hot slot"),
    );
    tree.insert(
        TreeEntry::new(
            cold.slot.as_str(),
            cold_hash,
            EntryKind::Blob,
            cold_bytes.len(),
        )
        .expect("cold slot"),
    );
    tree.insert(
        TreeEntry::new(
            pinned_hot.slot.as_str(),
            pinned_hash,
            EntryKind::Blob,
            pinned_bytes.len(),
        )
        .expect("pinned slot"),
    );
    tree.insert(
        TreeEntry::new(
            "scratch/stale-note",
            stale_hash,
            EntryKind::Blob,
            stale_bytes.len(),
        )
        .expect("stale slot"),
    );

    assert!(candidate_hot_cold_coexist(
        &tree,
        hot.slot.as_str(),
        cold.slot.as_str()
    ));
    assert!(
        !candidate_hot_cold_coexist(&tree, hot.slot.as_str(), "knowledge/absent"),
        "absent slots break coexistence"
    );

    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "sys".to_string();
    compiler.context_tree = tree;
    // Four slot lines exceed 400 bytes; three do not — Tier-1 demotes the
    // scratch slot only (existing rule, no new policy).
    let tight = CompilerBudgetConfig {
        max_total_bytes: 64 * 1024,
        max_zone1_bytes: 16 * 1024,
        max_zone2_bytes: 400,
        max_zone3_bytes: 16 * 1024,
    };
    let compiled = compiler.compile(&tight).expect("tight compiles");
    assert!(
        compiled
            .pruned_slots
            .contains(&"scratch/stale-note".to_string()),
        "Tier-1 must demote the stale scratch slot, pruned={:?}",
        compiled.pruned_slots
    );
    assert!(candidate_pinned_survives_tier1(
        &compiled,
        pinned_hot.slot.as_str(),
        "scratch/stale-note"
    ));
    assert!(
        compiled.zone2_state.contains(cold.slot.as_str()),
        "Cold knowledge coexists through Tier-1 untouched"
    );
    assert!(
        compiled.zone2_state.contains(hot.slot.as_str()),
        "Hot evidence coexists through Tier-1 untouched"
    );
}

// ── W-V4 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v4_per_claim_reference_resolves_bytes_at_revision() {
    let mut store = ContentStore::open_in_memory().expect("store");
    let alpha: &[u8] = b"alpha revision bytes";
    let beta: &[u8] = b"beta revision bytes";
    let alpha_hash = store.put_blob(alpha, 1000).expect("alpha blob");
    let beta_hash = store.put_blob(beta, 1001).expect("beta blob");
    let mut tree = ContextTree::new();
    tree.insert(
        TreeEntry::new("claim/alpha", alpha_hash, EntryKind::Blob, alpha.len())
            .expect("alpha slot"),
    );
    tree.insert(
        TreeEntry::new("claim/beta", beta_hash, EntryKind::Blob, beta.len()).expect("beta slot"),
    );
    let revision = commit_tree(&mut store, Vec::new(), &tree, "rev1", 1002);

    let claim_a = CandidateClaimRef {
        claim_id: "claim-a".to_string(),
        slot: "claim/alpha".to_string(),
        hash: alpha_hash,
        revision: revision.id,
    };
    let claim_b = CandidateClaimRef {
        claim_id: "claim-b".to_string(),
        slot: "claim/beta".to_string(),
        hash: beta_hash,
        revision: revision.id,
    };
    assert_eq!(
        candidate_resolve_claim(&store, &claim_a)
            .expect("resolve a")
            .as_deref(),
        Some(alpha),
        "claim A resolves to its own bytes at the revision"
    );
    assert_eq!(
        candidate_resolve_claim(&store, &claim_b)
            .expect("resolve b")
            .as_deref(),
        Some(beta),
        "claim B resolves independently at the same revision"
    );
    let rotated = CandidateClaimRef {
        claim_id: "claim-a".to_string(),
        slot: "claim/alpha".to_string(),
        hash: beta_hash,
        revision: revision.id,
    };
    assert_eq!(
        candidate_resolve_claim(&store, &rotated).expect("rotated"),
        None,
        "a rotated hash under the same slot must not resolve"
    );
    let missing = CandidateClaimRef {
        claim_id: "claim-x".to_string(),
        slot: "claim/absent".to_string(),
        hash: alpha_hash,
        revision: revision.id,
    };
    assert_eq!(
        candidate_resolve_claim(&store, &missing).expect("missing"),
        None,
        "an unknown slot resolves to absence"
    );
    let ghost = CandidateClaimRef {
        claim_id: "claim-a".to_string(),
        slot: "claim/alpha".to_string(),
        hash: alpha_hash,
        revision: ContentHash::compute(b"no such checkpoint"),
    };
    assert_eq!(
        candidate_resolve_claim(&store, &ghost).expect("ghost"),
        None,
        "an unknown revision resolves to absence, not an error"
    );
}

// ── W-V5 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v5_later_agent_reconstructs_why_from_rationale_alone() {
    let mut store = ContentStore::open_in_memory().expect("store");
    let rationale = Rationale::new("investigate the missing match arm", "inspect the error log")
        .with_where("crates/core")
        .with_how("grep")
        .with_expected("find the error line")
        .with_observed("found the missing arm at line 42");
    let tree = ContextTree::new();
    let draft = CheckpointDraft {
        parents: Vec::new(),
        task_id: "AI-0236".to_string(),
        agent_id: "ai-0236-impl".to_string(),
        rationale: rationale.clone(),
        tree_hash: None,
        summary: "candidate checkpoint".to_string(),
        timestamp_ms: 1000,
    };
    let committed = store
        .commit_checkpoint_atomic(&tree.canonical_bytes(), 1000, draft, &[], 1000)
        .expect("commit");
    // A later agent holds only the checkpoint row: no raw thinking travels
    // with it, only the six structured rationale fields.
    let later = store
        .get_checkpoint(&committed.id)
        .expect("read back")
        .expect("checkpoint row");
    assert_eq!(
        candidate_reconstruct_why(&rationale),
        candidate_reconstruct_why(&later.rationale),
        "reconstruction from Rationale alone is stable across the round-trip"
    );
    let reconstructed = candidate_reconstruct_why(&later.rationale);
    assert!(
        reconstructed.contains("investigate the missing match arm"),
        "reconstructed why carries the intent: {reconstructed}"
    );
    assert!(
        reconstructed.contains("inspect the error log"),
        "reconstructed why carries the action: {reconstructed}"
    );
    assert!(
        reconstructed.contains("line 42"),
        "reconstructed why carries the observation: {reconstructed}"
    );
    assert!(
        Rationale::new("", "what").validate().is_err(),
        "empty why is refused at validation: no vacuous reconstruction"
    );
}

// ── W-V8 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v8_lossy_projection_keeps_reversible_provenance() {
    let mut store = ContentStore::open_in_memory().expect("store");
    let original: &[u8] = b"full observation bytes that exceed the projection window by far";
    let original_hash = store.put_blob(original, 1000).expect("original blob");
    // Caller-supplied window per call; no default or threshold.
    let projection = candidate_project_lossy(original, 16);
    assert!(
        projection.projected.len() < original.len(),
        "the projection must be lossy"
    );
    assert!(
        projection.projected.contains("LOSSY PROJECTION TRUNCATED"),
        "truncation must be marked: {}",
        projection.projected
    );
    assert_eq!(projection.provenance, original_hash);
    assert_eq!(projection.original_len, original.len());
    assert_eq!(
        candidate_reverse_projection(&store, &projection)
            .expect("reverse")
            .as_deref(),
        Some(original),
        "provenance reverses to the exact original bytes"
    );
    let ghost = CandidateProjection {
        projected: "x".to_string(),
        provenance: ContentHash::compute(b"ghost bytes"),
        original_len: 11,
    };
    assert_eq!(
        candidate_reverse_projection(&store, &ghost).expect("ghost"),
        None,
        "unknown provenance resolves to absence"
    );
    let tampered = CandidateProjection {
        projected: projection.projected.clone(),
        provenance: projection.provenance,
        original_len: projection.original_len + 1,
    };
    assert_eq!(
        candidate_reverse_projection(&store, &tampered).expect("tampered"),
        None,
        "length mismatch must not return wrong bytes"
    );
}

// ── W-V9 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v9_evicted_retrievable_until_retention_ends_and_deletion_propagates() {
    let mut store = ContentStore::open_in_memory().expect("store");
    let live_payload: &[u8] = b"live payload";
    let live_payload_hash = store.put_blob(live_payload, 1000).expect("live payload");
    let mut live_tree = ContextTree::new();
    live_tree.insert(
        TreeEntry::new(
            "doc",
            live_payload_hash,
            EntryKind::Blob,
            live_payload.len(),
        )
        .expect("live slot"),
    );
    let live = commit_tree(&mut store, Vec::new(), &live_tree, "live", 1000);
    let live_tree_hash = ContentHash::compute(&live_tree.canonical_bytes());
    store
        .update_refs_atomic(&[("HEAD", &live.id), ("heads/main", &live.id)], 1000)
        .expect("live refs");

    // Evicted fork: no refs and no parents, so it is unreachable once the
    // retention window ends.
    let fork_payload: &[u8] = b"fork payload";
    let fork_payload_hash = store.put_blob(fork_payload, 1500).expect("fork payload");
    let mut fork_tree = ContextTree::new();
    fork_tree.insert(
        TreeEntry::new(
            "doc",
            fork_payload_hash,
            EntryKind::Blob,
            fork_payload.len(),
        )
        .expect("fork slot"),
    );
    let fork = commit_tree(&mut store, Vec::new(), &fork_tree, "fork", 1500);
    let fork_tree_hash = ContentHash::compute(&fork_tree.canonical_bytes());

    // Before GC: evicted rows stay retrievable (retention has not ended).
    assert!(candidate_checkpoint_retrievable(&store, &fork.id));
    assert!(candidate_blob_retrievable(&store, &fork_tree_hash));
    assert!(candidate_blob_retrievable(&store, &fork_payload_hash));

    let options = GcOptions {
        now_ms: 100_000,
        reflog_grace_ms: 1000,
        max_deletes_per_call: 1000,
        dry_run: false,
    };
    let preview = gc_preview(&store, &options).expect("preview");
    assert_eq!(
        preview.deleted_checkpoints, 1,
        "exactly the fork checkpoint is garbage: {preview:?}"
    );
    assert_eq!(
        preview.deleted_blobs, 2,
        "the fork tree blob and its payload blob are garbage: {preview:?}"
    );
    let report = collect_garbage(&mut store, &options).expect("collect");
    assert_eq!(
        report, preview,
        "preview must byte-match the destructive batch"
    );

    // Deletion/expiry propagated to references: the fork row reads as absence.
    assert!(candidate_deletion_propagated_to_checkpoint(
        &store, &fork.id
    ));
    assert!(
        !candidate_blob_retrievable(&store, &fork_tree_hash),
        "fork tree blob pruned"
    );
    assert!(
        !candidate_blob_retrievable(&store, &fork_payload_hash),
        "fork payload blob pruned"
    );
    assert!(candidate_checkpoint_retrievable(&store, &live.id));
    assert!(candidate_blob_retrievable(&store, &live_tree_hash));
    assert_eq!(
        store
            .get_blob(&live_payload_hash)
            .expect("live payload")
            .as_deref(),
        Some(live_payload),
        "live slot payload stays dereferenceable"
    );
}

// ── W-V10 ───────────────────────────────────────────────────────────────────

#[test]
fn w_v10_terminal_change_ai_contract_matrix() {
    fn compile_with(system_prompt: &str, turn_prompt: &str, observation: &str) -> CompiledContext {
        let mut compiler = ContextCompiler::new();
        compiler.system_prompt = system_prompt.to_string();
        compiler.project_rules = "rules".to_string();
        compiler.turn_prompt = turn_prompt.to_string();
        compiler.uncollapsed_observations = vec![observation.to_string()];
        compiler
            .compile(&CompilerBudgetConfig::default())
            .expect("compiles")
    }

    let base = compile_with("system contract v1", "turn one", "observation one");

    // Facade/host-conformance style: the same prefix assertion runs behind
    // both the inherent and the facade serving paths.
    let mut facade_compiler = ContextCompiler::new();
    facade_compiler.system_prompt = "system contract v1".to_string();
    facade_compiler.project_rules = "rules".to_string();
    facade_compiler.turn_prompt = "turn one".to_string();
    facade_compiler.uncollapsed_observations = vec!["observation one".to_string()];
    let via_facade = AiEngine::compile_context(&facade_compiler, &CompilerBudgetConfig::default())
        .expect("facade compiles");
    assert_eq!(
        via_facade.prefix_hash, base.prefix_hash,
        "facade and inherent paths agree on the stable prefix"
    );
    assert!(candidate_zone1_stable_after_terminal_change(
        &base,
        &via_facade
    ));

    let cells = [
        (
            CandidateBoundaryCell {
                terminal_changed: false,
                contract_changed: false,
            },
            compile_with("system contract v1", "turn one", "observation one"),
        ),
        (
            CandidateBoundaryCell {
                terminal_changed: true,
                contract_changed: false,
            },
            compile_with("system contract v1", "turn two", "observation two"),
        ),
        (
            CandidateBoundaryCell {
                terminal_changed: false,
                contract_changed: true,
            },
            compile_with("system contract v2", "turn one", "observation one"),
        ),
        (
            CandidateBoundaryCell {
                terminal_changed: true,
                contract_changed: true,
            },
            compile_with("system contract v2", "turn two", "observation two"),
        ),
    ];
    for (cell, compiled) in &cells {
        if candidate_boundary_expects_stable_prefix(*cell) {
            assert!(
                candidate_zone1_stable_after_terminal_change(&base, compiled),
                "cell {cell:?}: contract unchanged so the prefix must stay stable"
            );
        } else {
            assert!(
                candidate_zone1_moves_with_contract_change(&base, compiled),
                "cell {cell:?}: contract changed so the prefix must move"
            );
        }
    }
    for terminal_changed in [false, true] {
        for contract_changed in [false, true] {
            assert_eq!(
                candidate_boundary_expects_stable_prefix(CandidateBoundaryCell {
                    terminal_changed,
                    contract_changed,
                }),
                !contract_changed,
                "the matrix rule must ignore terminal change"
            );
        }
    }
}

// ── W-V6 ────────────────────────────────────────────────────────────────────

#[test]
fn w_v6_provider_continuation_from_canonical_dag_alone() {
    // Candidate wording only (beyond-v0.1, falsifiable, reviewable): provider
    // continuity holds from the canonical DAG bytes alone. Two FakeProviders
    // carry different ids; A commits the canonical DAG + checkpoint, A is
    // dropped entirely (script queue, last-sampling echo, call counter), and
    // B rehydrates from `canonical_bytes` alone before continuing with a
    // byte-equal turn. Offline and in-memory only: injected `*_ms` clocks, no
    // sockets, no live tests, no real-model calls. No continuation contract
    // is adopted (only `canonical_bytes` / `from_canonical_bytes` /
    // `complete`), and no W-V2 economics are pinned (`sampling: None`).
    use bitty_ai_runtime::{
        FakeProvider, ModelProvider, ProviderTurn, ProviderUsage, TurnRequest, provider::Message,
    };

    const NOW_MS: u64 = 1_000;
    const CHILD_MS: u64 = 1_001;
    const BUDGET_BYTES: usize = 32 * 1024;
    const TIMEOUT_MS: u64 = 5_000;
    const CONTINUATION_TEXT: &str = "continuation answer bytes from canonical dag";

    fn turn_request() -> TurnRequest {
        TurnRequest {
            model: "fake-chat".to_owned(),
            messages: vec![Message::user("continue from the canonical dag")],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: BUDGET_BYTES,
            timeout_ms: TIMEOUT_MS,
            now_ms: NOW_MS,
            sampling: None,
        }
    }

    fn scripted_continuation() -> ProviderTurn {
        ProviderTurn {
            text: CONTINUATION_TEXT.to_owned(),
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
        }
    }

    // Provider A commits the canonical DAG + checkpoint (a two-node checkpoint
    // chain so the DAG link itself is pinned, not just a lone row).
    let mut store = ContentStore::open_in_memory().expect("store");
    let seed: &[u8] = b"seed context bytes";
    let seed_hash = store.put_blob(seed, NOW_MS).expect("seed blob");
    let mut root_tree = ContextTree::new();
    root_tree.insert(
        TreeEntry::new("doc/seed", seed_hash, EntryKind::Blob, seed.len()).expect("seed slot"),
    );
    let root = commit_tree(&mut store, Vec::new(), &root_tree, "w-v6 root", NOW_MS);

    let answer_bytes = CONTINUATION_TEXT.as_bytes();
    let answer_hash = store.put_blob(answer_bytes, CHILD_MS).expect("answer blob");
    let mut child_tree = root_tree.clone();
    child_tree.insert(
        TreeEntry::new(
            "turn/answer",
            answer_hash,
            EntryKind::Blob,
            answer_bytes.len(),
        )
        .expect("answer slot"),
    );
    let canonical_bytes = child_tree.canonical_bytes();
    let child = commit_tree(
        &mut store,
        vec![root.id],
        &child_tree,
        "w-v6 child",
        CHILD_MS,
    );
    assert_eq!(child.parents, vec![root.id], "checkpoint DAG link pinned");
    assert!(
        store
            .get_checkpoint(&child.id)
            .expect("read back")
            .is_some(),
        "child checkpoint committed"
    );

    let mut provider_a = FakeProvider::new("provider-a").expect("valid id");
    provider_a.push_turn(scripted_continuation());
    assert_eq!(provider_a.scripted_turns_remaining(), 1);
    assert_eq!(provider_a.complete_calls(), 0);
    assert!(provider_a.last_sampling().is_none());
    let turn_a = provider_a
        .complete(&turn_request())
        .expect("provider A turn");
    assert_eq!(turn_a.text.as_bytes(), answer_bytes);
    assert_eq!(provider_a.scripted_turns_remaining(), 0);
    assert_eq!(provider_a.complete_calls(), 1);
    assert!(provider_a.last_sampling().is_none());
    let a_id = provider_a.provider_id().to_owned();
    // Drop A entirely: the script queue, the last-sampling echo, and the call
    // counter go with it. Only `canonical_bytes` (plus the expected literal)
    // survive for B.
    drop(provider_a);

    // Provider B rehydrates from `canonical_bytes` alone: no store handle, no
    // provider-A state, no continuation contract.
    let rehydrated =
        ContextTree::from_canonical_bytes(&canonical_bytes).expect("rehydrate from bytes alone");
    assert_eq!(
        rehydrated.canonical_bytes(),
        canonical_bytes,
        "rehydrated DAG byte-matches the committed canonical bytes"
    );

    let mut provider_b = FakeProvider::new("provider-b").expect("valid id");
    assert_ne!(
        a_id,
        provider_b.provider_id(),
        "providers carry different ids"
    );
    assert_eq!(provider_b.complete_calls(), 0);
    assert!(provider_b.last_sampling().is_none());
    provider_b.push_turn(scripted_continuation());
    let turn_b = provider_b
        .complete(&turn_request())
        .expect("provider B turn");
    assert_eq!(
        turn_b.text.as_bytes(),
        turn_a.text.as_bytes(),
        "continuation turn is byte-equal across the provider switch"
    );
    assert_eq!(
        turn_b, turn_a,
        "full scripted turn matches across providers"
    );
    assert_eq!(provider_b.complete_calls(), 1);
}

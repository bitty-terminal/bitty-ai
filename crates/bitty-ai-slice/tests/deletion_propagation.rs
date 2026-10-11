//! Seeded deletion-propagation proof (AI-0233, S2).
//!
//! Offline, deterministic, single-threaded: every timestamp is caller-supplied,
//! every payload is a fixed seed, no wall clock, no threads, no network, no
//! subprocess. The seeds below are the whole corpus: fixed record ids, fixed
//! marker payloads, fixed branch/session/pending ids, fixed `at_ms` values.
//!
//! What this file pins, end to end through the public API only:
//!
//! 1. Cross-store propagation: one seeded delete per store (journal
//!    tombstone, [`ArtifactStore::invalidate`](bitty_ai_runtime::context::ArtifactStore::invalidate),
//!    branch [`delete_branch`](bitty_ai_slice::session_refs::delete_branch))
//!    is visible as absence on every read path (journal `read_all`, artifact
//!    `resolve`, derived assembly re-resolution, `get_branch`/`list_branches`,
//!    reflog tombstone, pending `list_open`, session `resolve_session` +
//!    [`branch_absence`](bitty_ai_session::session_refs::branch_absence)).
//!    Tombstones hold only permitted absence metadata: reference strings,
//!    tip hashes, bounded reason/actor strings, timestamps. No payload bytes
//!    survive in any tombstone; every tombstone string is asserted
//!    marker-free.
//! 2. No resurrection: after deletion, GC
//!    ([`collect_garbage`](bitty_ai_slice::merge::collect_garbage)),
//!    compaction reload
//!    ([`CompactionStore::load_view`](bitty_ai_slice::compaction_store::CompactionStore::load_view)),
//!    store reopen, re-resolution, and replay all keep reporting absence.
//!    The simulated backup/export (read-only list APIs over every store)
//!    excludes every tombstoned id, hash, and reference.
//! 3. Uniform typed-unavailable: never-recorded fails as
//!    [`ContextError::ArtifactUnavailable`](bitty_ai_runtime::context::ContextError),
//!    wrong-generation as `ArtifactExpired`, explicitly deleted as the new
//!    `ArtifactDeleted`; the four-way helper
//!    ([`absence_reason`](bitty_ai_runtime::context::ArtifactStore::absence_reason))
//!    keeps never-recorded vs missing distinct. Session-plane absence is
//!    typed too (`None` for never-recorded reads, `NotFound` for unknown
//!    deletes, `Deleted` vs `NeverExisted` via `branch_absence`).
//!    Resolve-after-delete stays failing (no silent recovery) across GC and
//!    compaction.
//!
//! Non-overlap: `bridge_merge_gc.rs` pins merge/GC dispatch parity and dirty
//! gates; `session_refs.rs` pins branch-verb mechanics and reflog pruning;
//! `durability_slice.rs` pins compaction reopen and expiry tombstones;
//! `session_wiring.rs` pins snapshot ingest plus invalidation fail-closed.
//! Nothing here re-asserts those mechanics: every assertion below is a
//! deletion-propagation, no-resurrection, or typed-absence claim those files
//! leave out.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! ids, generations, digests, references, and payload markers never appear in
//! message strings.

use bitty_ai_runtime::compression::{
    CompressedSpan, CompressedView, RetentionClass, RetentionTags,
};
use bitty_ai_runtime::context::{
    ArtifactAbsence, ArtifactStore, AssembledContent, ContextError, ContextPriority, ContextRecord,
    ContextRequest, RecordBody, StableId, assemble,
};
use bitty_ai_session::pending::{PendingBegin, PendingDisposition, PendingResolve, PendingStore};
use bitty_ai_session::session_refs::{BranchAbsence, branch_absence};
use bitty_ai_session::sessions::{WheelSessionId, bind_session, resolve_session};
use bitty_ai_slice::compaction_store::CompactionStore;
use bitty_ai_slice::content_store::{
    Checkpoint, CheckpointDraft, ContentHash, ContentStore, Rationale,
};
use bitty_ai_slice::context_compiler::{ContextTree, EntryKind, TreeEntry};
use bitty_ai_slice::journal_prototype::Journal;
use bitty_ai_slice::merge::{GcOptions, collect_garbage, gc_preview};
use bitty_ai_slice::session_refs::{
    create_branch, delete_branch, get_branch, list_branches, read_reflog,
};

/// Seeded payload marker for the deleted corpus. Every tombstone and every
/// absence path below is asserted free of this string.
const ALPHA_MARKER: &str = "ALPHA-PAYLOAD-SEED-0233";
/// Seeded payload marker for the surviving corpus. Proves deletion is
/// selective: the survivor stays fully readable after the delete.
const BETA_MARKER: &str = "BETA-PAYLOAD-SEED-0233";

const GENERATION: u64 = 1;
/// Far-future caller clock for GC grace-window expiry. No wall clock is read.
const GC_NOW_MS: u64 = 100_000;
const GC_GRACE_MS: u64 = 1_000;

fn test_rationale() -> Rationale {
    Rationale::new("seed why", "seed what")
}

fn runtime_record(id: &str, summary: &str, body: Vec<u8>) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: GENERATION,
        collected_at_ms: 1_000,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(body),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn request() -> ContextRequest {
    ContextRequest {
        max_tokens: None,
        max_bytes: Some(32_768),
        current_generation: GENERATION,
    }
}

fn slot_tree(pairs: &[(&str, &str)]) -> ContextTree {
    let mut tree = ContextTree::new();
    for (name, content) in pairs {
        let hash = ContentHash::compute(content.as_bytes());
        tree.insert(
            TreeEntry::new(*name, hash, EntryKind::Blob, content.len()).expect("valid test slot"),
        );
    }
    tree
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
        task_id: "AI-0233".to_string(),
        agent_id: "ai-0233-impl".to_string(),
        rationale: test_rationale(),
        tree_hash: None,
        summary: summary.to_string(),
        timestamp_ms: at_ms,
    };
    store
        .commit_checkpoint_atomic(&bytes, at_ms, draft, &[], at_ms)
        .expect("setup commit")
}

fn gc_options() -> GcOptions {
    GcOptions {
        now_ms: GC_NOW_MS,
        reflog_grace_ms: GC_GRACE_MS,
        max_deletes_per_call: 1_000,
        dry_run: false,
    }
}

/// Deterministic caller-owned scratch directory for file-backed stores.
/// Removed on drop so test runs never litter the shared temp dir.
struct ScratchDir {
    path: std::path::PathBuf,
}

impl ScratchDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("bitty-ai-delprop-{name}-{}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("clear previous scratch dir");
        }
        std::fs::create_dir_all(&path).expect("create scratch dir");
        Self { path }
    }

    fn file(&self, name: &str) -> std::path::PathBuf {
        self.path.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn cross_store_delete_propagates_and_tombstones_are_metadata_only() {
    let scratch = ScratchDir::new("propagate");
    // Journal plane: two seeded records, one survives.
    let journal = Journal::open(&scratch.file("journal.sqlite3")).expect("open journal");
    journal
        .append("seed-alpha", "note", ALPHA_MARKER, 1_000)
        .expect("append alpha");
    journal
        .append("seed-beta", "note", BETA_MARKER, 1_000)
        .expect("append beta");

    // Artifact plane: one externalized record (derived reference holder) and
    // one inline survivor in a single deterministic assembly.
    let mut artifacts = ArtifactStore::new();
    let alpha_body = ALPHA_MARKER.repeat(400);
    assert!(
        alpha_body.len() > bitty_ai_runtime::context::EXTERNALIZE_THRESHOLD_BYTES,
        "seed body must externalize"
    );
    let assembled = assemble(
        &[
            runtime_record("rec-alpha", "alpha summary", alpha_body.into_bytes()),
            runtime_record("rec-beta", "beta summary", BETA_MARKER.as_bytes().to_vec()),
        ],
        &mut artifacts,
        &request(),
    )
    .expect("seed assembly externalizes alpha");
    assert_eq!(assembled.externalized, 1);
    let alpha_ref = match &assembled.records[0].content {
        AssembledContent::Reference(reference) => reference.clone(),
        AssembledContent::Inline(_) => panic!("seed alpha must externalize"),
    };
    assert!(
        artifacts.resolve(&alpha_ref, GENERATION).is_ok(),
        "seed alpha resolves before delete"
    );

    // Durable plane: two checkpoint tips on two branches.
    let mut store = ContentStore::open_in_memory().expect("open store");
    let tip_alpha = commit_tree(
        &mut store,
        Vec::new(),
        &slot_tree(&[("slot-a", ALPHA_MARKER)]),
        "alpha tip",
        1_000,
    );
    let tip_beta = commit_tree(
        &mut store,
        Vec::new(),
        &slot_tree(&[("slot-b", BETA_MARKER)]),
        "beta tip",
        1_100,
    );
    create_branch(
        &mut store,
        "heads/alpha",
        &tip_alpha.id,
        "seed",
        "tester",
        2_000,
    )
    .expect("seed alpha branch");
    create_branch(
        &mut store,
        "heads/beta",
        &tip_beta.id,
        "seed",
        "tester",
        2_100,
    )
    .expect("seed beta branch");

    // Pending + session planes: one open entry and one binding on the doomed
    // branch. Raw args are hashed at begin, never stored.
    let session_id = WheelSessionId::parse("sess-del-01").expect("valid session id");
    let branch_alpha =
        bitty_ai_session::session_refs::BranchName::parse("heads/alpha").expect("valid branch");
    bind_session(
        &store,
        &session_id,
        &branch_alpha,
        &tip_alpha.id,
        1,
        1,
        2_000,
    )
    .expect("bind");
    let args: &[u8] = b"alpha-args";
    let entry = PendingStore::begin(
        &store,
        PendingBegin {
            call_id: "a1b2c3d4",
            session_id: &session_id,
            task_id: None,
            tool: "test-tool",
            args,
            generation: 1,
            epoch: 1,
            now_ms: 2_000,
        },
    )
    .expect("begin pending");
    assert_eq!(entry.args_digest, ContentHash::compute(args));

    // The three deletes, each host-authorized with fixed stamps.
    journal
        .tombstone("seed-alpha", 3_000)
        .expect("tombstone alpha");
    assert!(artifacts.invalidate(&alpha_ref), "seed delete must retire");
    let deleted_tip =
        delete_branch(&mut store, "heads/alpha", "retire", "tester", 3_000).expect("delete branch");
    assert_eq!(deleted_tip, tip_alpha.id);

    // Journal propagation: tombstoned id reads back absent, survivor intact,
    // history rows preserved (tombstone hides, never rewrites).
    let surviving = journal.read_all().expect("read surviving");
    assert_eq!(surviving.len(), 1);
    assert_eq!(surviving[0].id, "seed-beta");
    assert_eq!(surviving[0].payload, BETA_MARKER);
    assert!(
        !surviving[0].payload.contains(ALPHA_MARKER),
        "survivor must not carry deleted bytes"
    );

    // Artifact propagation: deleted reference fails with the distinct
    // Deleted variant; budget and count reflect the removal; the tombstone
    // holds the reference only.
    assert!(
        matches!(
            artifacts.resolve(&alpha_ref, GENERATION),
            Err(ContextError::ArtifactDeleted { .. })
        ),
        "deleted artifact must fail as Deleted"
    );
    assert_eq!(
        artifacts.absence_reason(&alpha_ref, GENERATION),
        ArtifactAbsence::Deleted
    );
    assert!(artifacts.is_deleted(&alpha_ref));
    assert_eq!(artifacts.deleted_count(), 1);
    assert_eq!(artifacts.len(), 0);
    assert_eq!(artifacts.total_bytes(), 0);
    // Derived summaries are inline inert text: the host-held assembled value
    // keeps its summary, which carries no payload bytes.
    assert_eq!(assembled.records[0].summary, "alpha summary");
    assert!(
        !assembled.records[0].summary.contains(ALPHA_MARKER),
        "derived summary must not carry payload bytes"
    );
    assert!(
        matches!(assembled.records[1].content, AssembledContent::Inline(_)),
        "survivor stays inline"
    );

    // Refs propagation: live row gone, tombstone row present with absence
    // metadata only, survivor branch untouched.
    assert_eq!(
        get_branch(&store, "heads/alpha").expect("get deleted"),
        None
    );
    let branches = list_branches(&store).expect("list branches");
    assert_eq!(branches.len(), 1);
    assert_eq!(branches[0].0.as_str(), "heads/beta");
    assert_eq!(branches[0].1, tip_beta.id);
    match branch_absence(&store, "heads/alpha").expect("classify deleted") {
        BranchAbsence::Deleted {
            tip,
            reason,
            actor,
            at_ms,
        } => {
            assert_eq!(tip, tip_alpha.id);
            assert_eq!(reason, "retire");
            assert_eq!(actor, "tester");
            assert_eq!(at_ms, 3_000);
            assert!(
                !reason.contains(ALPHA_MARKER),
                "tombstone reason is metadata"
            );
            assert!(!actor.contains(ALPHA_MARKER), "tombstone actor is metadata");
        }
        _ => panic!("deleted branch must classify as Deleted"),
    }
    assert_eq!(
        branch_absence(&store, "heads/beta").expect("classify live"),
        BranchAbsence::Live(tip_beta.id)
    );
    assert_eq!(
        branch_absence(&store, "heads/never").expect("classify absent"),
        BranchAbsence::NeverExisted
    );
    let history = read_reflog(&store, "heads/alpha", 10).expect("read tombstone history");
    assert!(!history.is_empty(), "tombstone history must survive");
    let tombstone = &history[0];
    assert_eq!(tombstone.old_hash, Some(tip_alpha.id));
    assert_eq!(tombstone.new_hash, ContentHash::from_bytes([0u8; 32]));
    assert_eq!(tombstone.reason, "retire");
    assert_eq!(tombstone.actor, "tester");
    assert!(
        !tombstone.reason.contains(ALPHA_MARKER),
        "reflog holds no payload"
    );
    assert!(
        !tombstone.actor.contains(ALPHA_MARKER),
        "reflog holds no payload"
    );

    // Pending + session isolation: branch deletion moves neither plane. The
    // open entry keeps only the args digest; the binding keeps its tip, now
    // observably detached from any live branch.
    let open = PendingStore::list_open(&store, &session_id).expect("list open");
    assert_eq!(open.len(), 1);
    assert_eq!(open[0].args_digest, ContentHash::compute(args));
    let binding = resolve_session(&store, &session_id)
        .expect("resolve binding")
        .expect("binding survives branch delete");
    assert_eq!(binding.branch, "heads/alpha");
    assert_eq!(binding.head, tip_alpha.id);
}

#[test]
fn deleted_bytes_never_resurrect_through_gc_compaction_replay_or_export() {
    let scratch = ScratchDir::new("noresurrect");
    // Durable plane: live chain pinned, fork abandoned and tombstoned long
    // before the GC clock, plus one standalone alpha blob (orphan payload).
    let mut store = ContentStore::open_in_memory().expect("open store");
    let base = commit_tree(
        &mut store,
        Vec::new(),
        &slot_tree(&[("slot", "base")]),
        "base",
        1_000,
    );
    let live = commit_tree(
        &mut store,
        vec![base.id],
        &slot_tree(&[("slot", "base"), ("live", BETA_MARKER)]),
        "live",
        2_000,
    );
    store
        .update_refs_atomic(&[("HEAD", &live.id), ("heads/main", &live.id)], 2_000)
        .expect("pin live refs");
    let fork = commit_tree(
        &mut store,
        vec![base.id],
        &slot_tree(&[("slot", "base"), ("fork", ALPHA_MARKER)]),
        "fork",
        1_500,
    );
    create_branch(&mut store, "heads/gone", &fork.id, "seed", "tester", 1_600)
        .expect("seed gone branch");
    delete_branch(&mut store, "heads/gone", "retire", "tester", 1_700).expect("delete gone");
    let orphan_hash = store
        .put_blob(ALPHA_MARKER.as_bytes(), 1_200)
        .expect("seed orphan blob");
    assert!(
        store
            .has_blob(&orphan_hash)
            .expect("orphan present before GC"),
        "orphan blob must exist before GC"
    );

    // Artifact plane: one deleted reference that must never resolve again.
    let mut artifacts = ArtifactStore::new();
    let doomed = artifacts
        .store(ALPHA_MARKER.repeat(300).into_bytes(), GENERATION)
        .expect("store doomed");
    assert!(artifacts.invalidate(&doomed), "seed artifact delete");

    // Journal + compaction planes with the same deleted id.
    let journal = Journal::open(&scratch.file("journal.sqlite3")).expect("open journal");
    journal
        .append("gone-alpha", "note", ALPHA_MARKER, 1_000)
        .expect("append gone");
    journal
        .append("kept-beta", "note", BETA_MARKER, 1_000)
        .expect("append kept");
    journal
        .tombstone("gone-alpha", 1_700)
        .expect("tombstone gone");
    let mut compaction =
        CompactionStore::open(&scratch.file("compaction.sqlite3")).expect("open compaction");
    let view = CompressedView {
        records: vec![
            runtime_record(
                "gone-alpha",
                "gone summary",
                ALPHA_MARKER.as_bytes().to_vec(),
            ),
            runtime_record("kept-beta", "kept summary", BETA_MARKER.as_bytes().to_vec()),
        ],
        spans: vec![CompressedSpan {
            span_id: "span-derived".to_owned(),
            source_ids: vec!["gone-alpha".to_owned()],
            summary: "derived-from-gone".to_owned(),
            is_untrusted_surface: false,
            retention: RetentionClass::Normal,
            source_deadline_ms: 5_000,
            created_at_ms: 2_000,
        }],
        tombstones: vec!["gone-alpha".to_owned()],
        tags: RetentionTags::new(),
    };
    compaction.record_view(&view).expect("record seeded view");

    // GC: preview byte-matches the destructive batch; the tombstoned fork
    // and the orphan blob are pruned while live state survives.
    let preview = gc_preview(&store, &gc_options()).expect("preview GC");
    let report = collect_garbage(&mut store, &gc_options()).expect("collect GC");
    assert_eq!(report, preview);
    assert!(
        !store.has_checkpoint(&fork.id).expect("fork pruned"),
        "tombstoned fork must be pruned"
    );
    assert!(
        store.has_checkpoint(&live.id).expect("live survives"),
        "live tip must survive GC"
    );
    assert!(
        !store.has_blob(&orphan_hash).expect("orphan pruned"),
        "standalone deleted bytes must be pruned"
    );
    assert!(
        store.get_blob(&orphan_hash).expect("read pruned").is_none(),
        "pruned bytes must read back absent"
    );
    assert!(
        store
            .get_checkpoint(&fork.id)
            .expect("read pruned")
            .is_none(),
        "pruned checkpoint must read back absent"
    );
    // Reflog tombstones are never pruned by GC: the deletion proof survives
    // collection, and the branch still classifies as Deleted.
    assert_eq!(
        branch_absence(&store, "heads/gone").expect("classify after GC"),
        BranchAbsence::Deleted {
            tip: fork.id,
            reason: "retire".to_owned(),
            actor: "tester".to_owned(),
            at_ms: 1_700,
        }
    );
    let zero = ContentHash::from_bytes([0u8; 32]);
    assert!(
        store.get_blob(&zero).expect("zero blob").is_none(),
        "tombstone payload stays absent"
    );

    // No re-expansion: every deleted handle keeps failing after GC.
    assert!(
        matches!(
            artifacts.resolve(&doomed, GENERATION),
            Err(ContextError::ArtifactDeleted { .. })
        ),
        "deleted artifact must stay Deleted after GC"
    );
    assert!(
        matches!(
            artifacts.resolve(&doomed, GENERATION + 1),
            Err(ContextError::ArtifactDeleted { .. })
        ),
        "tombstone must win over expiry after GC"
    );

    // No replay resurrection: compaction reload drops the tombstoned record
    // and its derived span, keeps the survivor, and survives reopen.
    let reloaded = compaction.load_view().expect("reload view");
    assert!(
        reloaded
            .records
            .iter()
            .all(|record| record.id != "gone-alpha"),
        "tombstoned record must not reload"
    );
    assert!(
        reloaded.spans.is_empty(),
        "span derived from deleted source must not reload"
    );
    assert!(reloaded.tombstones.contains(&"gone-alpha".to_owned()));
    assert_eq!(reloaded.records.len(), 1);
    assert_eq!(reloaded.records[0].id, "kept-beta");
    drop(compaction);
    let reopened =
        CompactionStore::open(&scratch.file("compaction.sqlite3")).expect("reopen compaction");
    let again = reopened.load_view().expect("reload after reopen");
    assert!(
        again.records.iter().all(|record| record.id != "gone-alpha"),
        "deleted record must not resurrect across reopen"
    );
    assert!(again.spans.is_empty(), "derived span must not resurrect");
    drop(reopened);
    drop(journal);
    let journal_again = Journal::open(&scratch.file("journal.sqlite3")).expect("reopen journal");
    let replayed = journal_again.read_all().expect("replay journal");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, "kept-beta");

    // Simulated backup/export over read-only list APIs: every tombstoned id,
    // hash, and reference is absent from the exported set.
    let exported_checkpoints: Vec<String> = store
        .all_checkpoint_hashes()
        .expect("export checkpoints")
        .iter()
        .map(ContentHash::to_hex)
        .collect();
    assert!(
        !exported_checkpoints.contains(&fork.id.to_hex()),
        "export must exclude tombstoned checkpoint"
    );
    assert!(
        exported_checkpoints.contains(&live.id.to_hex()),
        "export must keep live checkpoint"
    );
    let exported_blobs: Vec<String> = store
        .all_blob_hashes()
        .expect("export blobs")
        .iter()
        .map(ContentHash::to_hex)
        .collect();
    assert!(
        !exported_blobs.contains(&orphan_hash.to_hex()),
        "export must exclude deleted blob"
    );
    let exported_refs: Vec<String> = store
        .list_refs()
        .expect("export refs")
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    assert!(
        !exported_refs.contains(&"heads/gone".to_owned()),
        "export must exclude deleted branch"
    );
    let exported_journal: Vec<String> = replayed.iter().map(|record| record.id.clone()).collect();
    assert!(
        !exported_journal.contains(&"gone-alpha".to_owned()),
        "export must exclude tombstoned journal id"
    );
    let exported_compaction: Vec<String> = again
        .records
        .iter()
        .map(|record| record.id.clone())
        .collect();
    assert!(
        !exported_compaction.contains(&"gone-alpha".to_owned()),
        "export must exclude tombstoned compaction id"
    );
}

#[test]
fn typed_absence_reasons_are_distinct_and_sticky_after_delete() {
    // Artifact plane: Available vs Expired vs Deleted vs NeverRecorded.
    let mut artifacts = ArtifactStore::new();
    let live = artifacts
        .store(vec![b'x'; 32], GENERATION)
        .expect("store live");
    let doomed = artifacts
        .store(vec![b'y'; 32], GENERATION)
        .expect("store doomed");
    assert!(artifacts.invalidate(&doomed), "seed delete");
    assert_eq!(
        artifacts.absence_reason(&live, GENERATION),
        ArtifactAbsence::Available
    );
    assert!(artifacts.resolve(&live, GENERATION).is_ok());
    assert_eq!(
        artifacts.absence_reason(&live, GENERATION + 1),
        ArtifactAbsence::Expired
    );
    assert!(
        matches!(
            artifacts.resolve(&live, GENERATION + 1),
            Err(ContextError::ArtifactExpired { .. })
        ),
        "wrong generation must fail as Expired"
    );
    assert_eq!(
        artifacts.absence_reason(&doomed, GENERATION),
        ArtifactAbsence::Deleted
    );
    assert!(
        matches!(
            artifacts.resolve(&doomed, GENERATION),
            Err(ContextError::ArtifactDeleted { .. })
        ),
        "deleted reference must fail as Deleted"
    );
    // Deleted stays Deleted across generations: no silent recovery through a
    // generation change, and the error carries the reference only.
    assert_eq!(
        artifacts.absence_reason(&doomed, GENERATION + 1),
        ArtifactAbsence::Deleted
    );
    assert!(
        matches!(
            artifacts.resolve(&doomed, GENERATION + 1),
            Err(ContextError::ArtifactDeleted { .. })
        ),
        "deleted reference must stay Deleted across generations"
    );
    let err = artifacts
        .resolve(&doomed, GENERATION)
        .expect_err("deleted must fail");
    assert_eq!(
        format!("{err}"),
        format!("artifact deleted: {}", doomed.as_str())
    );
    // Never-recorded: a foreign reference in a store that never assigned it.
    let foreign = ArtifactStore::new();
    assert_eq!(
        foreign.absence_reason(&live, GENERATION),
        ArtifactAbsence::NeverRecorded
    );
    assert!(
        matches!(
            foreign.resolve(&live, GENERATION),
            Err(ContextError::ArtifactUnavailable { .. })
        ),
        "foreign reference must fail as Unavailable"
    );
    // The four helper reasons are pairwise distinct.
    let reasons = [
        artifacts.absence_reason(&live, GENERATION),
        artifacts.absence_reason(&live, GENERATION + 1),
        artifacts.absence_reason(&doomed, GENERATION),
        foreign.absence_reason(&live, GENERATION),
    ];
    assert_eq!(
        reasons,
        [
            ArtifactAbsence::Available,
            ArtifactAbsence::Expired,
            ArtifactAbsence::Deleted,
            ArtifactAbsence::NeverRecorded,
        ]
    );

    // Session plane: None for never-recorded reads, NotFound for unknown
    // deletes, Deleted vs NeverExisted for branches.
    let mut store = ContentStore::open_in_memory().expect("open store");
    let unknown_hash = ContentHash::compute(b"never-recorded-payload");
    assert!(
        store
            .get_blob(&unknown_hash)
            .expect("read unknown")
            .is_none(),
        "unknown blob must read back absent"
    );
    assert!(
        store
            .get_checkpoint(&unknown_hash)
            .expect("read unknown")
            .is_none(),
        "unknown checkpoint must read back absent"
    );
    assert!(
        store
            .get_ref("heads/never")
            .expect("read unknown")
            .is_none(),
        "unknown ref must read back absent"
    );
    assert_eq!(
        branch_absence(&store, "heads/never").expect("classify unknown"),
        BranchAbsence::NeverExisted
    );
    assert!(
        PendingStore::get(&store, "beef01")
            .expect("read unknown")
            .is_none(),
        "unknown pending id must read back absent"
    );
    let unknown_session = WheelSessionId::parse("sess-never").expect("valid session id");
    assert_eq!(
        resolve_session(&store, &unknown_session).expect("resolve unknown"),
        None
    );
    // Unknown deletes fail with typed NotFound, never silent success.
    assert!(
        delete_branch(&mut store, "heads/never", "retire", "tester", 1_000).is_err(),
        "unknown branch delete must fail"
    );

    // Sticky after delete: resolve-after-delete stays failing across
    // repeated resolves and across GC.
    let tip = commit_tree(
        &mut store,
        Vec::new(),
        &slot_tree(&[("slot", BETA_MARKER)]),
        "tip",
        1_000,
    );
    create_branch(&mut store, "heads/sticky", &tip.id, "seed", "tester", 2_000)
        .expect("seed sticky branch");
    store
        .update_refs_atomic(&[("HEAD", &tip.id)], 2_000)
        .expect("pin HEAD");
    delete_branch(&mut store, "heads/sticky", "retire", "tester", 3_000).expect("delete sticky");
    for _ in 0..3 {
        assert_eq!(
            get_branch(&store, "heads/sticky").expect("reread deleted"),
            None
        );
        assert!(
            matches!(
                branch_absence(&store, "heads/sticky").expect("reclassify"),
                BranchAbsence::Deleted { .. }
            ),
            "deleted branch must stay Deleted"
        );
    }
    let report = collect_garbage(&mut store, &gc_options()).expect("GC after delete");
    assert!(!report.truncated, "seeded GC must converge");
    assert_eq!(
        get_branch(&store, "heads/sticky").expect("read after GC"),
        None
    );
    assert!(
        matches!(
            branch_absence(&store, "heads/sticky").expect("classify after GC"),
            BranchAbsence::Deleted { .. }
        ),
        "deleted branch must stay Deleted across GC"
    );
}

#[test]
fn pending_entries_and_session_bindings_observe_branch_delete_without_loss() {
    // An open pending entry pins its args blob across GC; resolving it
    // releases the pin so a second GC reclaims the blob. Branch deletion in
    // the middle moves neither the pending row nor the session binding.
    let mut store = ContentStore::open_in_memory().expect("open store");
    let tip = commit_tree(
        &mut store,
        Vec::new(),
        &slot_tree(&[("slot", BETA_MARKER)]),
        "tip",
        1_000,
    );
    create_branch(&mut store, "heads/work", &tip.id, "seed", "tester", 2_000)
        .expect("seed work branch");
    store
        .update_refs_atomic(&[("HEAD", &tip.id)], 2_000)
        .expect("pin HEAD");
    let session_id = WheelSessionId::parse("sess-pin-01").expect("valid session id");
    let branch =
        bitty_ai_session::session_refs::BranchName::parse("heads/work").expect("valid branch");
    bind_session(&store, &session_id, &branch, &tip.id, 1, 1, 2_000).expect("bind session");
    let args: &[u8] = b"pinned-call-args";
    PendingStore::begin(
        &store,
        PendingBegin {
            call_id: "c0ffee11",
            session_id: &session_id,
            task_id: None,
            tool: "test-tool",
            args,
            generation: 1,
            epoch: 1,
            now_ms: 2_000,
        },
    )
    .expect("begin pending");
    let args_hash = ContentHash::compute(args);
    store.put_blob(args, 2_000).expect("store args blob");

    // Open pin holds across GC: the args blob survives while the entry is open.
    let first = collect_garbage(&mut store, &gc_options()).expect("first GC");
    assert!(!first.truncated, "seeded GC must converge");
    assert!(
        store.has_blob(&args_hash).expect("pinned blob survives"),
        "open pending entry must pin its args blob"
    );

    // Branch deletion moves neither plane: the pending row stays open and
    // the binding keeps its tip, now detached from any live branch.
    delete_branch(&mut store, "heads/work", "retire", "tester", 3_000).expect("delete work");
    let open = PendingStore::list_open(&store, &session_id).expect("list open after delete");
    assert_eq!(open.len(), 1);
    assert_eq!(open[0].args_digest, args_hash);
    assert!(
        !open[0].tool.contains(ALPHA_MARKER),
        "pending row must not carry deleted bytes"
    );
    let binding = resolve_session(&store, &session_id)
        .expect("resolve after delete")
        .expect("binding survives branch delete");
    assert_eq!(binding.head, tip.id);
    assert!(
        matches!(
            branch_absence(&store, "heads/work").expect("classify"),
            BranchAbsence::Deleted { .. }
        ),
        "deleted work branch must classify as Deleted"
    );

    // Resolving the entry releases the pin: the next GC reclaims the args
    // blob while the live tip survives. No resurrection either way.
    PendingStore::resolve(
        &store,
        PendingResolve {
            call_id: "c0ffee11",
            disposition: PendingDisposition::Success,
            claim_epoch: 1,
            expected_generation: 1,
            now_ms: 4_000,
        },
    )
    .expect("resolve pending");
    assert!(
        PendingStore::list_open(&store, &session_id)
            .expect("list after resolve")
            .is_empty(),
        "resolved entry must leave the open set"
    );
    // Re-pin HEAD to the live tip (the work branch is gone) so only the
    // unpinned args blob is collectable.
    store
        .update_refs_atomic(&[("HEAD", &tip.id), ("heads/main", &tip.id)], 4_000)
        .expect("repin live refs");
    let second = collect_garbage(&mut store, &gc_options()).expect("second GC");
    assert!(!second.truncated, "seeded GC must converge");
    assert!(
        !store.has_blob(&args_hash).expect("unpinned blob pruned"),
        "resolved entry must release its args blob"
    );
    assert!(
        store.has_checkpoint(&tip.id).expect("live tip survives"),
        "live tip must survive both collections"
    );
}

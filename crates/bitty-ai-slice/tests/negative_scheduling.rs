//! R6-V6 negative-scheduling absence probe (AI-0239, RFC AIQ-5C).
//!
//! Offline deterministic, in-memory only: idle [`ContentStore`], [`WheelKernel`],
//! and [`TaskEngine`] opened via `open_in_memory`, caller-supplied clocks only,
//! then a large caller-clock jump with read-only observations. Asserts zero
//! autonomous mutation: checkpoint/blob counts, `list_refs`, and reflog length
//! are stable, and task scheduling does not advance on its own.
//!
//! The stability window issues no explicit `collect_garbage`, `merge_commit`,
//! or `commit_checkpoint_atomic`: only read-only queries plus read-only
//! `gc_preview` run between the before/after snapshots. `gc_preview` with
//! `max_deletes_per_call = 0` deletes nothing (zero-bound preview reports
//! `deleted_checkpoints == 0` and `deleted_blobs == 0`, with `truncated` set
//! exactly when garbage remains).
//!
//! Out of scope (must not fail this probe): the `ai_contrib.rs`
//! `thread::spawn` credential helper and the `local_provider` test-only
//! `thread::spawn` plus sleep. Neither surface is imported or exercised here.
//!
//! No new dependencies, threads, network, filesystem, wall-clock, or live
//! providers. All timestamps are caller-supplied `now_ms` constants.

use bitty_ai_slice::content_store::{CheckpointDraft, ContentHash, ContentStore, Rationale};
use bitty_ai_slice::context_compiler::{ContextTree, EntryKind, TreeEntry};
use bitty_ai_slice::merge::{GcOptions, gc_preview};
use bitty_ai_slice::session_refs::{create_branch, dump_reflog_all};
use bitty_ai_slice::task_dag::{TaskDraft, TaskEngine, TaskId};
use bitty_ai_slice::wheel_kernel::WheelKernel;

/// Baseline caller clock for seeded writes.
const T0: u64 = 1_000;
/// Jumped caller clock for read-only observations (about one year later).
const TJUMP: u64 = 31_536_000_000;

fn test_rationale() -> Rationale {
    Rationale::new("r6-v6 why", "r6-v6 what")
}

fn checkpoint_draft(summary: &str, at_ms: u64) -> CheckpointDraft {
    CheckpointDraft {
        parents: Vec::new(),
        task_id: "AI-0239".to_string(),
        agent_id: "ai-0239-impl".to_string(),
        rationale: test_rationale(),
        tree_hash: None,
        summary: summary.to_string(),
        timestamp_ms: at_ms,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct StoreSnapshot {
    checkpoints: Vec<ContentHash>,
    blobs: Vec<ContentHash>,
    refs: Vec<(String, ContentHash)>,
    reflog_len: usize,
}

fn store_snapshot(store: &ContentStore) -> StoreSnapshot {
    StoreSnapshot {
        checkpoints: store.all_checkpoint_hashes().expect("checkpoint hashes"),
        blobs: store.all_blob_hashes().expect("blob hashes"),
        refs: store.list_refs().expect("list refs"),
        reflog_len: dump_reflog_all(store).expect("reflog dump").len(),
    }
}

fn zero_gc_options(now_ms: u64) -> GcOptions {
    GcOptions {
        now_ms,
        reflog_grace_ms: 0,
        max_deletes_per_call: 0,
        dry_run: false,
    }
}

#[test]
fn idle_empty_triple_clock_jump_writes_nothing() {
    let store = ContentStore::open_in_memory().expect("content store opens");
    let kernel = WheelKernel::open_in_memory().expect("kernel opens");
    let engine = TaskEngine::open_in_memory().expect("task engine opens");

    let before = store_snapshot(&store);
    assert!(
        before.checkpoints.is_empty(),
        "idle store starts with no checkpoints"
    );
    assert!(before.blobs.is_empty(), "idle store starts with no blobs");
    assert!(before.refs.is_empty(), "idle store starts with no refs");
    assert_eq!(before.reflog_len, 0, "idle store starts with empty reflog");
    let kbefore = store_snapshot(kernel.content_store());
    assert!(kbefore.checkpoints.is_empty());
    assert!(kbefore.blobs.is_empty());
    assert!(kbefore.refs.is_empty());
    assert_eq!(kbefore.reflog_len, 0);
    assert!(engine.list_tasks().expect("list tasks").is_empty());
    assert!(kernel.list_tasks().expect("kernel tasks").is_empty());

    // Jumped-clock read-only observations only. No collect_garbage,
    // merge_commit, or commit_checkpoint_atomic in this window.
    let _ = store.get_ref("HEAD").expect("get ref");
    let _ = store.list_refs().expect("list refs");
    let _ = store.all_checkpoint_hashes().expect("checkpoint hashes");
    let _ = store.all_blob_hashes().expect("blob hashes");
    let _ = dump_reflog_all(&store).expect("reflog dump");
    let _ = store.reflog_window_hashes(TJUMP, 0).expect("reflog window");
    let _ = engine.list_tasks().expect("engine tasks at jump");
    let _ = engine.ready_tasks().expect("ready tasks at jump");
    let _ = engine.next_ready_task().expect("next ready at jump");
    let _ = engine.topological_sort().expect("topo at jump");
    let _ = kernel.list_tasks().expect("kernel tasks at jump");
    let _ = kernel.list_branches().expect("branches at jump");
    let _ = kernel.read_reflog("HEAD", 10).expect("kernel reflog");
    let _ = kernel.log(8).expect("kernel log");
    let store_preview = gc_preview(&store, &zero_gc_options(TJUMP)).expect("store preview");
    let kernel_preview = kernel
        .gc_preview(&zero_gc_options(TJUMP))
        .expect("kernel preview");

    assert_eq!(store_preview.deleted_checkpoints, 0);
    assert_eq!(store_preview.deleted_blobs, 0);
    assert!(
        !store_preview.truncated,
        "empty store has no garbage to truncate"
    );
    assert_eq!(kernel_preview.deleted_checkpoints, 0);
    assert_eq!(kernel_preview.deleted_blobs, 0);
    assert!(
        !kernel_preview.truncated,
        "empty kernel has no garbage to truncate"
    );

    let after = store_snapshot(&store);
    assert_eq!(
        before.checkpoints, after.checkpoints,
        "checkpoint set stable across clock jump"
    );
    assert_eq!(
        before.blobs, after.blobs,
        "blob set stable across clock jump"
    );
    assert_eq!(before.refs, after.refs, "refs stable across clock jump");
    assert_eq!(
        before.reflog_len, after.reflog_len,
        "reflog stable across clock jump"
    );
    let kafter = store_snapshot(kernel.content_store());
    assert_eq!(kbefore, kafter);
    assert!(engine.list_tasks().expect("tasks after").is_empty());
    assert!(kernel.list_tasks().expect("kernel tasks after").is_empty());
}

#[test]
fn content_store_seeded_clock_jump_writes_nothing() {
    let mut store = ContentStore::open_in_memory().expect("store opens");

    let payload = b"r6-v6 seeded payload";
    let payload_hash = store.put_blob(payload, T0).expect("put payload");
    let mut tree = ContextTree::new();
    tree.insert(
        TreeEntry::new("seed.txt", payload_hash, EntryKind::Blob, payload.len())
            .expect("valid test slot"),
    );
    let tree_bytes = tree.canonical_bytes();
    let tree_hash = store.put_blob(&tree_bytes, T0).expect("put tree");
    let mut root_draft = checkpoint_draft("r6-v6 root", T0);
    root_draft.tree_hash = Some(tree_hash);
    let root = store.commit_checkpoint(root_draft).expect("commit root");
    create_branch(&mut store, "heads/main", &root.id, "seed", "tester", T0).expect("seed branch");
    let orphan_blob = store.put_blob(b"r6-v6 orphan", T0).expect("orphan blob");
    let orphan = store
        .commit_checkpoint(checkpoint_draft("r6-v6 orphan", T0 + 1))
        .expect("commit orphan");

    let before = store_snapshot(&store);
    assert_eq!(
        before.checkpoints.len(),
        2,
        "seeded store holds two checkpoints"
    );
    assert_eq!(before.blobs.len(), 3, "seeded store holds three blobs");
    assert_eq!(before.refs.len(), 1, "seeded store holds one live ref");
    assert_eq!(before.reflog_len, 1, "seeded store holds one reflog row");
    assert!(store.has_blob(&orphan_blob).expect("has orphan blob"));
    assert!(store.has_checkpoint(&orphan.id).expect("has orphan"));

    // Full preview proves garbage exists, so the zero bound is meaningful.
    let full = gc_preview(
        &store,
        &GcOptions {
            now_ms: TJUMP,
            reflog_grace_ms: 0,
            max_deletes_per_call: 1_000,
            dry_run: false,
        },
    )
    .expect("full preview");
    assert!(
        full.deleted_checkpoints >= 1,
        "orphan checkpoint must be collectable, got {full:?}"
    );

    // Jumped-clock read-only window. No collect_garbage, merge_commit, or
    // commit_checkpoint_atomic here.
    let _ = store.get_checkpoint(&root.id).expect("get checkpoint");
    let _ = store.get_blob(&tree_hash).expect("get blob");
    let _ = store.get_ref("heads/main").expect("get ref");
    let _ = store.list_refs().expect("list refs");
    let _ = store.all_checkpoint_hashes().expect("checkpoint hashes");
    let _ = store.all_blob_hashes().expect("blob hashes");
    let _ = dump_reflog_all(&store).expect("reflog dump");
    let _ = store.log(&root.id, 10).expect("log");
    let _ = store
        .merge_base(&root.id, &root.id)
        .expect("merge base self");
    let _ = store
        .reflog_window_hashes(TJUMP, 0)
        .expect("reflog window at jump");
    let zero = gc_preview(&store, &zero_gc_options(TJUMP)).expect("zero preview");
    assert_eq!(
        zero.deleted_checkpoints, 0,
        "zero bound deletes no checkpoints"
    );
    assert_eq!(zero.deleted_blobs, 0, "zero bound deletes no blobs");
    assert!(
        zero.truncated,
        "zero bound must report truncated while garbage remains"
    );

    let after = store_snapshot(&store);
    assert_eq!(before, after, "store stable across clock jump");
    assert!(store.has_blob(&orphan_blob).expect("orphan survives"));
    assert!(store.has_checkpoint(&orphan.id).expect("orphan survives"));
}

#[test]
fn task_engine_seeded_no_autonomous_scheduling() {
    let mut engine = TaskEngine::open_in_memory().expect("engine opens");

    let t1 = TaskId::new("r6v6-root").expect("valid id");
    let t2 = TaskId::new("r6v6-child").expect("valid id");
    engine
        .create_task(
            TaskDraft {
                id: t1.clone(),
                title: "R6-V6 root".to_string(),
                description: String::new(),
                priority: 10,
                dependencies: Vec::new(),
            },
            T0,
        )
        .expect("create root");
    engine
        .create_task(
            TaskDraft {
                id: t2.clone(),
                title: "R6-V6 child".to_string(),
                description: String::new(),
                priority: 5,
                dependencies: vec![t1.clone()],
            },
            T0 + 1,
        )
        .expect("create child");

    let tasks_before = engine.list_tasks().expect("list tasks");
    assert_eq!(tasks_before.len(), 2);
    assert_eq!(tasks_before[0].id, t1);
    assert_eq!(tasks_before[1].id, t2);
    assert_eq!(
        tasks_before[0].status,
        bitty_ai_slice::task_dag::TaskStatus::Ready
    );
    assert_eq!(
        tasks_before[1].status,
        bitty_ai_slice::task_dag::TaskStatus::Pending
    );
    let ready_before = engine.ready_tasks().expect("ready tasks");
    assert_eq!(ready_before.len(), 1);
    assert_eq!(ready_before[0].id, t1);
    let next_before = engine
        .next_ready_task()
        .expect("next ready")
        .expect("one ready task");
    assert_eq!(next_before.id, t1);
    let topo_before = engine.topological_sort().expect("topo sort");
    assert_eq!(topo_before, vec![t1.clone(), t2.clone()]);

    // Caller clock jumps to TJUMP, but no assign, complete, fail, cancel,
    // retry, or dependency mutation is issued with it. Scheduling advances
    // only through those explicit calls, so read-only queries must observe
    // identical state.
    let _ = TJUMP;
    let tasks_mid = engine.list_tasks().expect("tasks at jump");
    let ready_mid = engine.ready_tasks().expect("ready at jump");
    let next_mid = engine
        .next_ready_task()
        .expect("next at jump")
        .expect("still one ready");
    let topo_mid = engine.topological_sort().expect("topo at jump");
    let root_mid = engine.get_task(&t1).expect("get root at jump");
    let child_mid = engine.get_task(&t2).expect("get child at jump");
    assert_eq!(
        tasks_mid, tasks_before,
        "task rows stable without explicit transition"
    );
    assert_eq!(
        ready_mid, ready_before,
        "ready queue stable without dispatch"
    );
    assert_eq!(next_mid, next_before, "next-ready stable without dispatch");
    assert_eq!(topo_mid, topo_before, "topological order stable");
    assert_eq!(root_mid.status, tasks_before[0].status);
    assert_eq!(root_mid.generation, tasks_before[0].generation);
    assert_eq!(child_mid.status, tasks_before[1].status);
    assert_eq!(child_mid.generation, tasks_before[1].generation);
    assert_eq!(
        engine.get_dependencies(&t2).expect("deps"),
        vec![t1.clone()]
    );
    assert_eq!(
        engine.get_dependents(&t1).expect("dependents"),
        vec![t2.clone()]
    );

    let tasks_after = engine.list_tasks().expect("tasks after");
    assert_eq!(
        tasks_after, tasks_before,
        "no autonomous scheduling across clock jump"
    );
}

#[test]
fn wheel_kernel_seeded_clock_jump_writes_nothing() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    kernel
        .put_slot("r6v6/seed.txt", b"r6-v6 seed", T0)
        .expect("put slot");
    let cp = kernel
        .commit_checkpoint(test_rationale(), Some("heads/main"), T0 + 1)
        .expect("commit checkpoint");
    assert_eq!(kernel.head_checkpoint(), Some(&cp.id));

    let before = store_snapshot(kernel.content_store());
    assert_eq!(before.checkpoints.len(), 1, "kernel holds one checkpoint");
    assert_eq!(before.blobs.len(), 2, "kernel holds payload plus tree blob");
    assert_eq!(before.refs.len(), 2, "kernel holds HEAD plus heads/main");
    assert_eq!(before.reflog_len, 1, "kernel holds one branch reflog row");
    let tasks_before = kernel.list_tasks().expect("kernel tasks");
    assert!(tasks_before.is_empty());
    let branches_before = kernel.list_branches().expect("branches");
    assert_eq!(branches_before.len(), 1);
    let log_before = kernel.log(8).expect("log");
    assert_eq!(log_before.len(), 1);
    assert_eq!(log_before[0].id, cp.id);

    // Jumped-clock read-only window. No collect_garbage, merge_commit, or
    // commit_checkpoint_atomic here; the working tree stays clean so the
    // preview gate passes without writes.
    let _ = kernel.get_slot("r6v6/seed.txt").expect("get slot");
    let _ = kernel.list_slots();
    let _ = kernel.get_checkpoint(&cp.id).expect("get checkpoint");
    let _ = kernel.list_tasks().expect("tasks at jump");
    let _ = kernel.list_branches().expect("branches at jump");
    let _ = kernel
        .read_reflog("heads/main", 10)
        .expect("reflog at jump");
    let _ = kernel.log(8).expect("log at jump");
    let _ = kernel
        .content_store()
        .reflog_window_hashes(TJUMP, 0)
        .expect("reflog window at jump");
    let zero = kernel
        .gc_preview(&zero_gc_options(TJUMP))
        .expect("zero preview");
    assert_eq!(
        zero.deleted_checkpoints, 0,
        "zero bound deletes no checkpoints"
    );
    assert_eq!(zero.deleted_blobs, 0, "zero bound deletes no blobs");
    assert!(
        !zero.truncated,
        "clean kernel has no garbage, so zero preview is not truncated"
    );

    let after = store_snapshot(kernel.content_store());
    assert_eq!(before, after, "kernel store stable across clock jump");
    assert_eq!(
        kernel.list_tasks().expect("tasks after"),
        tasks_before,
        "tasks stable across clock jump"
    );
    assert_eq!(
        kernel.head_checkpoint(),
        Some(&cp.id),
        "HEAD stable without explicit commit"
    );
    assert_eq!(
        kernel
            .get_slot("r6v6/seed.txt")
            .expect("slot after")
            .expect("seed survives"),
        b"r6-v6 seed",
        "committed payload survives the jump"
    );
}

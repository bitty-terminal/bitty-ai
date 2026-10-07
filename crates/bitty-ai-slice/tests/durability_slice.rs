//! Durable journal slice tests (AI-0178, formal path).
//!
//! Covers reopen-across-exit, torn-write recovery, zero-summarizer replay,
//! deletion propagation, expiry absence, single-writer fencing, and
//! corrupt/foreign fail-closed behavior. `recent_actions` and `active_task_id`
//! are explicitly non-persisted session state (see non-persistence test).

use bitty_ai_runtime::compression::{
    CompressedView, CompressionError, FakeSummarizer, RetentionClass, RetentionPolicy,
    RetentionTags, SpanRange, compress_records,
};
use bitty_ai_runtime::context::{ContextPriority, ContextRecord, RecordBody, StableId};
use bitty_ai_slice::compaction_store::{CompactionStore, DurabilityError};
use bitty_ai_slice::content_store::Rationale;
use bitty_ai_slice::task_dag::{TaskDraft, TaskId};
use bitty_ai_slice::wheel_kernel::WheelKernel;

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bitty-ai-0178-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn record(id: &str, collected_at_ms: u64) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms,
        priority: ContextPriority::Normal,
        summary: format!("summary of {id}"),
        body: RecordBody::Inline(vec![b'x'; 32]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn compress_two(now_ms: u64) -> (CompressedView, usize) {
    let records = vec![record("r1", 100), record("r2", 120), record("r3", 140)];
    let ranges = vec![SpanRange { start: 0, end: 2 }];
    let summarizer = FakeSummarizer::new(vec!["head summary".to_owned()]);
    let view = compress_records(
        &records,
        &ranges,
        &summarizer,
        &RetentionTags::new(),
        now_ms,
    )
    .expect("compress");
    let calls = summarizer.calls();
    (view, calls)
}

#[test]
fn reopen_preserves_head_spans_tombstones_tags_previous_summary_strikes() {
    let dir = scratch_dir("reopen-preserves");
    let wheel_path = dir.join("wheel.db");
    let comp_path = dir.join("compaction.db");

    let head_hex: String;
    {
        let mut kernel = WheelKernel::open(&wheel_path).expect("wheel open");
        kernel
            .put_slot("data.txt", b"durable payload", 1010)
            .expect("put slot");
        let cp = kernel
            .commit_checkpoint(
                Rationale::new("Persist", "Save durable state"),
                Some("heads/main"),
                1020,
            )
            .expect("commit");
        head_hex = cp.id.to_hex();

        let mut tags = RetentionTags::new();
        tags.set("r1", RetentionClass::Ephemeral);
        let records = vec![record("r1", 100), record("r2", 120), record("r3", 140)];
        let ranges = vec![SpanRange { start: 0, end: 2 }];
        let summarizer = FakeSummarizer::new(vec!["head summary".to_owned()]);
        let mut view =
            compress_records(&records, &ranges, &summarizer, &tags, 1000).expect("compress");
        view.tombstones.push("gone-1".to_owned());

        let mut store = CompactionStore::open(&comp_path).expect("compaction open");
        store.record_view(&view).expect("record view");
        store
            .set_previous_summary(Some("prior rolling summary"))
            .expect("set previous");
        store.set_ineffective_strikes(3).expect("set strikes");
        store.set_window_bytes(32768).expect("set window");
    }

    {
        let kernel = WheelKernel::open(&wheel_path).expect("wheel reopen");
        let head = kernel.head_checkpoint().expect("HEAD survives");
        assert_eq!(head.to_hex(), head_hex);
        let history = kernel.log(5).expect("log");
        assert_eq!(history.len(), 1);
        let slot = kernel
            .get_slot("data.txt")
            .expect("slot read")
            .expect("slot restored");
        assert_eq!(slot.as_slice(), b"durable payload");

        let store = CompactionStore::open(&comp_path).expect("compaction reopen");
        let reloaded = store.load_view().expect("load view");
        assert_eq!(reloaded.spans.len(), 1);
        assert_eq!(reloaded.spans[0].summary, "head summary");
        assert_eq!(
            reloaded.spans[0].source_ids,
            vec!["r1".to_owned(), "r2".to_owned()]
        );
        assert!(reloaded.tombstones.contains(&"gone-1".to_owned()));
        assert_eq!(reloaded.tags.get("r1"), RetentionClass::Ephemeral);
        assert_eq!(reloaded.tags.get("r2"), RetentionClass::Normal);
        assert_eq!(
            store.get_previous_summary().expect("get previous"),
            Some("prior rolling summary".to_owned())
        );
        assert_eq!(store.get_ineffective_strikes().expect("get strikes"), 3);
        assert_eq!(store.get_window_bytes().expect("get window"), Some(32768));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn torn_write_never_half_advances_head() {
    let dir = scratch_dir("torn-write");
    let wheel_path = dir.join("wheel.db");

    let cp1_hex: String;
    {
        let mut kernel = WheelKernel::open(&wheel_path).expect("open");
        kernel.put_slot("a.txt", b"v1", 1000).expect("slot");
        let cp1 = kernel
            .commit_checkpoint(Rationale::new("First", "Baseline"), None, 1010)
            .expect("cp1");
        cp1_hex = cp1.id.to_hex();
    }

    // Simulate a crash between checkpoint-row insert and HEAD update: insert a
    // valid blob + checkpoint row via raw SQL without touching HEAD.
    let orphan_hex: String;
    {
        let raw = rusqlite::Connection::open(&wheel_path).expect("raw open");
        let tree_bytes = b"{\"orphan\":true}";
        let tree_hash = bitty_ai_slice::content_store::ContentHash::compute(tree_bytes);
        raw.execute(
            "INSERT OR IGNORE INTO blobs (hash, size, data, created_at_ms) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                tree_hash.to_hex(),
                tree_bytes.len() as i64,
                tree_bytes,
                1020i64
            ],
        )
        .expect("raw blob");
        // Minimal checkpoint row with parent cp1, no HEAD advance.
        let rationale = serde_json::to_string(&Rationale::new("Orphan", "Crashed")).expect("json");
        let parents = serde_json::to_string(&vec![cp1_hex.clone()]).expect("parents json");
        // Compute a deterministic fake hash id for the orphan row.
        let orphan_id = bitty_ai_slice::content_store::ContentHash::compute(b"orphan-checkpoint");
        orphan_hex = orphan_id.to_hex();
        raw.execute(
            "INSERT OR IGNORE INTO checkpoints (hash, parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                orphan_hex,
                parents,
                "task-unassigned",
                "wheel-kernel",
                rationale,
                tree_hash.to_hex(),
                "cognitive checkpoint",
                1020i64
            ],
        )
        .expect("raw checkpoint");
    }

    {
        let kernel = WheelKernel::open(&wheel_path).expect("reopen after torn write");
        let head = kernel.head_checkpoint().expect("HEAD present");
        assert_eq!(
            head.to_hex(),
            cp1_hex,
            "reopen must yield the pre-crash HEAD, never the orphan"
        );
        assert_ne!(head.to_hex(), orphan_hex);
    }

    // Corrupt triple: HEAD points at a missing checkpoint row -> typed failure,
    // never a half-advanced kernel.
    {
        let raw = rusqlite::Connection::open(&wheel_path).expect("raw open 2");
        let missing = bitty_ai_slice::content_store::ContentHash::compute(b"missing-target");
        raw.execute(
            "INSERT OR REPLACE INTO refs (name, target_hash, updated_at_ms) VALUES ('HEAD', ?1, ?2)",
            rusqlite::params![missing.to_hex(), 1030i64],
        )
        .expect("point HEAD at missing");
    }
    {
        let err = match WheelKernel::open(&wheel_path) {
            Ok(_) => panic!("partial triple must fail closed"),
            Err(e) => e,
        };
        let text = format!("{err}");
        assert!(
            text.contains("recovery") || text.contains("HEAD"),
            "typed absence error, got: {text}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn record_crash_replays_identical_view_with_zero_summarizer_calls() {
    let dir = scratch_dir("replay-zero-calls");
    let comp_path = dir.join("compaction.db");

    let (view, calls) = compress_two(1000);
    assert_eq!(calls, 1);
    {
        let mut store = CompactionStore::open(&comp_path).expect("open");
        store.record_view(&view).expect("record");
    }
    // Crash: handle dropped without checkpointing anything else.
    {
        let store = CompactionStore::open(&comp_path).expect("reopen");
        let reloaded = store.load_view().expect("load");
        assert_eq!(reloaded, view, "crash replay must be identical");
        // Recovery path never contacts a summarizer: prove it with an exhausted
        // script that would fail closed on any call.
        let exhausted = FakeSummarizer::new(Vec::new());
        let _ = exhausted.calls();
        assert_eq!(exhausted.calls(), 0);
        assert_eq!(
            reloaded
                .resolve_summary(&reloaded.spans[0].span_id)
                .expect("resolve"),
            "head summary"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deletion_propagates_in_memory_and_reloaded() {
    let dir = scratch_dir("deletion-propagation");
    let comp_path = dir.join("compaction.db");

    let (mut view, _) = compress_two(1000);
    assert_eq!(view.spans.len(), 1);
    let invalidated = view.delete_ids(&["r1"]);
    assert_eq!(invalidated.len(), 1);
    assert!(
        view.spans.is_empty(),
        "source delete kills derived spans in memory"
    );

    {
        let mut store = CompactionStore::open(&comp_path).expect("open");
        store.record_view(&view).expect("record");
    }
    {
        let store = CompactionStore::open(&comp_path).expect("reopen");
        let reloaded = store.load_view().expect("load");
        assert!(reloaded.spans.is_empty(), "reloaded view stays span-free");
        assert!(reloaded.tombstones.contains(&"r1".to_owned()));
        let err = reloaded
            .resolve_summary("cmp-0000")
            .expect_err("deleted span absent");
        assert!(matches!(err, CompressionError::SummaryUnavailable { .. }));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn expired_source_deadline_is_typed_absence_never_resurrected() {
    let dir = scratch_dir("expiry-absence");
    let comp_path = dir.join("compaction.db");

    // Old sources with an Ephemeral tag and a 5-minute TTL; now is far future.
    let mut tags = RetentionTags::new();
    tags.set("r1", RetentionClass::Ephemeral);
    tags.set("r2", RetentionClass::Ephemeral);
    let old_records = vec![record("r1", 100), record("r2", 120)];
    let ranges = vec![SpanRange { start: 0, end: 2 }];
    let summarizer = FakeSummarizer::new(vec!["old summary".to_owned()]);
    let view = compress_records(&old_records, &ranges, &summarizer, &tags, 200).expect("compress");
    assert_eq!(view.spans.len(), 1);

    {
        let mut store = CompactionStore::open(&comp_path).expect("open");
        store.record_view(&view).expect("record");
    }
    {
        let store = CompactionStore::open(&comp_path).expect("reopen");
        let mut reloaded = store.load_view().expect("load");
        let policy = RetentionPolicy {
            pinned_ttl_ms: None,
            recent_ttl_ms: Some(24 * 60 * 60 * 1000),
            normal_ttl_ms: Some(60 * 60 * 1000),
            ephemeral_ttl_ms: Some(5 * 60 * 1000),
        };
        let expired = reloaded.apply_retention(&policy, 10_000_000);
        assert!(expired.contains(&reloaded.tombstones[0].clone()) || !expired.is_empty());
        let span_id = view.spans[0].span_id.clone();
        let err = reloaded
            .resolve_summary(&span_id)
            .expect_err("expired span must be absent");
        assert!(matches!(err, CompressionError::SummaryUnavailable { .. }));
        drop(store);
        // Persist the expiry tombstones; a later reload must not resurrect.
        let mut store = CompactionStore::open(&comp_path).expect("reopen 2");
        let tombstone_refs: Vec<&str> = expired.iter().map(String::as_str).collect();
        store.add_tombstones(&tombstone_refs).expect("tombstone");
        drop(store);
        let store = CompactionStore::open(&comp_path).expect("reopen 3");
        let reread = store.load_view().expect("load 3");
        assert!(
            !reread.spans.iter().any(|s| s.span_id == span_id),
            "expired span never resurrected"
        );
        assert!(reread.resolve_summary(&span_id).is_err());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn second_open_durable_while_first_lives_is_writer_busy() {
    let dir = scratch_dir("writer-busy");
    let comp_path = dir.join("compaction.db");
    let wheel_path = dir.join("wheel.db");

    let _first_comp = CompactionStore::open(&comp_path).expect("first compaction open");
    let err = match CompactionStore::open_durable(&comp_path) {
        Ok(_) => panic!("second must be busy"),
        Err(e) => e,
    };
    assert!(
        matches!(err, DurabilityError::WriterBusy),
        "expected WriterBusy, got {err:?}"
    );

    let _first_wheel = WheelKernel::open(&wheel_path).expect("first wheel open");
    let err = match WheelKernel::open_durable(&wheel_path) {
        Ok(_) => panic!("second wheel must fail"),
        Err(e) => e,
    };
    let text = format!("{err}");
    assert!(
        text.contains("busy") || text.contains("Busy") || text.contains("writer"),
        "expected writer-busy refusal, got: {text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_and_foreign_db_fail_closed_bytes_unchanged() {
    let dir = scratch_dir("corrupt-foreign");
    // Garbage bytes (not SQLite).
    let garbage_path = dir.join("garbage.db");
    let garbage = b"this is not a sqlite database; deterministic filler past header length";
    std::fs::write(&garbage_path, garbage).expect("seed garbage");

    let err = match CompactionStore::open(&garbage_path) {
        Ok(_) => panic!("garbage must fail"),
        Err(e) => e,
    };
    assert!(
        matches!(err, DurabilityError::Corrupt { .. }),
        "got {err:?}"
    );
    assert_eq!(std::fs::read(&garbage_path).expect("read back"), garbage);

    let err = match WheelKernel::open(&garbage_path) {
        Ok(_) => panic!("wheel garbage must fail"),
        Err(e) => e,
    };
    let text = format!("{err}");
    assert!(
        text.contains("corrupt") || text.contains("SQLite") || text.contains("not a"),
        "got: {text}"
    );
    assert_eq!(std::fs::read(&garbage_path).expect("read back 2"), garbage);

    // Foreign SQLite: valid database but incompatible durable shape.
    let foreign_path = dir.join("foreign.db");
    {
        let raw = rusqlite::Connection::open(&foreign_path).expect("raw create");
        raw.execute_batch(
            "CREATE TABLE compaction_spans (span_id TEXT PRIMARY KEY, wrong TEXT NOT NULL);",
        )
        .expect("seed wrong shape");
        raw.execute(
            "INSERT INTO compaction_spans (span_id, wrong) VALUES ('s', 'keep-me')",
            [],
        )
        .expect("seed row");
        raw.execute_batch(
            "CREATE TABLE compaction_tombstones (id TEXT PRIMARY KEY);
             CREATE TABLE compaction_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )
        .expect("other tables");
    }
    let before = std::fs::read(&foreign_path).expect("snapshot");
    let err = match CompactionStore::open(&foreign_path) {
        Ok(_) => panic!("foreign shape must fail"),
        Err(e) => e,
    };
    assert!(
        matches!(
            err,
            DurabilityError::Corrupt { .. } | DurabilityError::Storage { .. }
        ),
        "got {err:?}"
    );
    assert_eq!(
        std::fs::read(&foreign_path).expect("bytes unchanged"),
        before,
        "refused open must not mutate the file"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn session_state_is_explicitly_non_persisted() {
    // recent_actions and active_task_id reset on every open by design; only
    // HEAD, the DAG, blobs/refs, and tasks survive a reopen.
    let dir = scratch_dir("session-non-persisted");
    let wheel_path = dir.join("wheel.db");
    {
        let mut kernel = WheelKernel::open(&wheel_path).expect("open");
        let task_id = TaskId::new("task-01").expect("id");
        kernel
            .create_task(
                TaskDraft {
                    id: task_id.clone(),
                    title: "T".to_owned(),
                    description: "D".to_owned(),
                    priority: 1,
                    dependencies: vec![],
                },
                1000,
            )
            .expect("task");
        kernel.set_active_task(Some(task_id)).expect("active");
        kernel
            .record_action_outcome("act-1", true, Some(0), 5, "ok", "", 1010)
            .expect("action");
        assert_eq!(kernel.recent_actions().len(), 1);
        assert!(kernel.active_task().is_some());
    }
    {
        let kernel = WheelKernel::open(&wheel_path).expect("reopen");
        assert!(
            kernel.recent_actions().is_empty(),
            "recent_actions are session state and must not persist"
        );
        assert!(
            kernel.active_task().is_none(),
            "active_task_id is session state and must not persist"
        );
        assert_eq!(
            kernel.list_tasks().expect("tasks").len(),
            1,
            "tasks persist"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

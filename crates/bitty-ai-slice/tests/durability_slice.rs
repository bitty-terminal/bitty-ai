//! Durable journal slice tests (AI-0178, formal path; v0.1 gate AI-0238).
//!
//! The Wheel halves below refuse under the v0.1 gate (path opens fail closed
//! with the gate error and create no file); the `CompactionStore` halves are
//! a separate prototype surface outside the gate and keep their file-backed
//! reopen coverage byte-identical. `recent_actions` and `active_task_id`
//! are explicitly non-persisted session state (see non-persistence test).

use bitty_ai_runtime::compression::{
    CompressedView, CompressionError, FakeSummarizer, RetentionClass, RetentionPolicy,
    RetentionTags, SpanRange, compress_records,
};
use bitty_ai_runtime::context::{ContextPriority, ContextRecord, RecordBody, StableId};
use bitty_ai_slice::compaction_store::{CompactionStore, DurabilityError};
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

/// v0.1 gate (AI-0238): the Wheel path opens refuse, so wheel reopen
/// preservation is out of the v0.1 profile. The compaction reopen half below
/// is a separate prototype surface and keeps its asserts byte-identical.
#[test]
fn reopen_preserves_head_spans_tombstones_tags_previous_summary_strikes() {
    let dir = scratch_dir("reopen-preserves");
    let wheel_path = dir.join("wheel.db");
    let comp_path = dir.join("compaction.db");

    // Wheel side: both spellings refuse with the gate error, no file.
    for api in ["open", "open_durable"] {
        let err = if api == "open" {
            match WheelKernel::open(&wheel_path) {
                Ok(_) => panic!("durable wheel open must refuse under the v0.1 gate"),
                Err(err) => err,
            }
        } else {
            match WheelKernel::open_durable(&wheel_path) {
                Ok(_) => panic!("durable wheel open must refuse under the v0.1 gate"),
                Err(err) => err,
            }
        };
        let text = err.to_string();
        assert!(
            text.contains("gated out of v0.1"),
            "{api} refusal must carry the v0.1 gate message, got: {text}"
        );
    }
    assert!(
        !wheel_path.exists(),
        "refused wheel opens must not create a file: {}",
        wheel_path.display()
    );

    {
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

/// v0.1 gate (AI-0238): the Wheel path open refuses before any filesystem
/// access, so torn-write recovery across reopen is out of the v0.1 profile.
/// The refusal itself is fail-closed with zero filesystem effects.
#[test]
fn torn_write_never_half_advances_head() {
    let dir = scratch_dir("torn-write");
    let wheel_path = dir.join("wheel.db");

    let err = match WheelKernel::open(&wheel_path) {
        Ok(_) => panic!("durable wheel open must refuse under the v0.1 gate"),
        Err(err) => err,
    };
    let text = err.to_string();
    assert!(
        text.contains("gated out of v0.1"),
        "refusal must carry the v0.1 gate message, got: {text}"
    );
    assert!(
        !wheel_path.exists(),
        "refused open must not create a file: {}",
        wheel_path.display()
    );
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
        // Exact expired id: the span whose source deadline is past the
        // Ephemeral TTL. (Both sources sit inside the span range, so the view
        // holds no passthrough records for them: the only record is the
        // span's synthetic summary, dropped with its span.)
        let span_id = view.spans[0].span_id.clone();
        assert_eq!(
            expired,
            vec![span_id.clone()],
            "only the expired span id may expire"
        );
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
        // Tombstoned bodies stay deleted: no expired id may reload with a
        // payload, even though its bytes were on disk before tombstoning.
        // Here that means the stripped synthetic summary record is gone.
        for id in &expired {
            assert!(
                !reread.records.iter().any(|record| &record.id == id),
                "expired id {id} resurrected with payload bytes"
            );
        }
        assert!(
            reread.records.is_empty(),
            "the expired synthetic record must be stripped from the image"
        );
        drop(store);
        // The bytes must be gone from the disk image itself, not merely
        // filtered at load: the persisted records_json carries no expired id.
        {
            let raw = rusqlite::Connection::open(&comp_path).expect("raw probe");
            let image: String = raw
                .query_row(
                    "SELECT value FROM compaction_meta WHERE key = 'records_json'",
                    [],
                    |row| row.get(0),
                )
                .expect("records_json present");
            for id in &expired {
                assert!(
                    !image.contains(id.as_str()),
                    "expired id {id} still persisted on disk"
                );
            }
            // Tombstoned spans are purged from the spans table itself, not
            // merely filtered at load: no span row may reference an expired
            // source, and the expired span row must be gone.
            let span_rows: Vec<(String, String)> = {
                let mut stmt = raw
                    .prepare("SELECT span_id, source_ids_json FROM compaction_spans")
                    .expect("spans probe");
                stmt.query_map([], |row| {
                    let span_id: String = row.get(0)?;
                    let source_json: String = row.get(1)?;
                    Ok((span_id, source_json))
                })
                .expect("span rows")
                .map(|row| row.expect("span row"))
                .collect()
            };
            assert!(
                !span_rows.iter().any(|(sid, _)| sid == &span_id),
                "expired span row still persisted on disk"
            );
            for id in &expired {
                assert!(
                    !span_rows
                        .iter()
                        .any(|(_, sources)| sources.contains(id.as_str())),
                    "expired id {id} still referenced by a span row on disk"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// v0.1 gate (AI-0238): the Wheel path opens refuse, so the wheel half of
/// writer fencing is out of the v0.1 profile. The compaction half below is a
/// separate prototype surface and keeps its asserts byte-identical.
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

    // Wheel side: both spellings refuse with the gate error, no file.
    for api in ["open", "open_durable"] {
        let err = if api == "open" {
            match WheelKernel::open(&wheel_path) {
                Ok(_) => panic!("durable wheel open must refuse under the v0.1 gate"),
                Err(err) => err,
            }
        } else {
            match WheelKernel::open_durable(&wheel_path) {
                Ok(_) => panic!("durable wheel open must refuse under the v0.1 gate"),
                Err(err) => err,
            }
        };
        let text = format!("{err}");
        assert!(
            text.contains("gated out of v0.1"),
            "{api} refusal must carry the v0.1 gate message, got: {text}"
        );
    }
    assert!(
        !wheel_path.exists(),
        "refused wheel opens must not create a file: {}",
        wheel_path.display()
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
        text.contains("gated out of v0.1"),
        "v0.1 gate (AI-0238): wheel path open must refuse gated, got: {text}"
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

/// v0.1 gate (AI-0238): the Wheel path open refuses, so session-state
/// non-persistence across reopen is out of the v0.1 profile. The same
/// session-only property on a live handle stays covered by
/// `test_kernel_session_state_is_non_persisted_documented` in
/// `tests/wheel_kernel.rs`.
#[test]
fn session_state_is_explicitly_non_persisted() {
    // recent_actions and active_task_id reset on every open by design; only
    // HEAD, the DAG, blobs/refs, and tasks survive a reopen.
    let dir = scratch_dir("session-non-persisted");
    let wheel_path = dir.join("wheel.db");
    let err = match WheelKernel::open(&wheel_path) {
        Ok(_) => panic!("durable wheel open must refuse under the v0.1 gate"),
        Err(err) => err,
    };
    let text = format!("{err}");
    assert!(
        text.contains("gated out of v0.1"),
        "refusal must carry the v0.1 gate message, got: {text}"
    );
    assert!(
        !wheel_path.exists(),
        "refused open must not create a file: {}",
        wheel_path.display()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

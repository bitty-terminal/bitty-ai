//! Integration tests for the unified `AiEngine` facade (AI-0161).

use bitty_ai_runtime::prompt::{LayerInput, PromptLayer, PromptSnapshot};
use bitty_ai_slice::{
    AiEngine, RefreshAuthorization, SNAPSHOT_CANONICALIZATION_VERSION, SNAPSHOT_SCHEMA_VERSION,
    SnapshotIngestRequest,
};

#[test]
fn stream_session_drives_incremental_chunks_and_pipes_to_sink() {
    let mut session = AiEngine::new_stream_session();

    let stream_bytes = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"from \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"Facade!\"}}]}\n\ndata: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n";

    // Feed in small 7-byte slices to test fragmentation handling
    let mut all_deltas = Vec::new();
    for chunk in stream_bytes.chunks(7) {
        let deltas = session.feed_chunk(chunk).expect("feed chunk success");
        all_deltas.extend(deltas);
    }

    assert_eq!(session.chunks().len(), 3);
    assert_eq!(session.seq(), 3);
    assert_eq!(session.concatenated_text(), "Hello from Facade!");

    let turn = session.finish().expect("finish stream");
    assert_eq!(turn.text, "Hello from Facade!");
    assert_eq!(turn.usage.input_tokens, 10);
    assert_eq!(turn.usage.output_tokens, 5);
}

#[test]
fn stream_session_pipes_eof_flushed_content_to_sink_without_trailing_newline() {
    let mut session = AiEngine::new_stream_session();

    // Stream ends abruptly without the second newline delimiter
    let stream_bytes =
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Incomplete delimiter\"}}]}\n";

    let deltas = session
        .feed_chunk(stream_bytes)
        .expect("feed chunk success");
    assert!(deltas.is_empty(), "event is buffered in SSE parser");

    let (turn, sink) = session.finish_with_sink().expect("finish stream");
    assert_eq!(turn.text, "Incomplete delimiter");
    assert_eq!(sink.chunks().len(), 1);
    assert_eq!(
        String::from_utf8_lossy(&sink.chunks()[0].fragment.bytes),
        "Incomplete delimiter"
    );
}

#[test]
fn engine_assembles_canonical_prompt_layers() {
    let snapshot = PromptSnapshot {
        core_version: "bitty-core-prompt@1".to_owned(),
        layers: vec![
            LayerInput::text_only(PromptLayer::CoreContract, "You are Bitty AI."),
            LayerInput::text_only(PromptLayer::User, "Be concise."),
        ],
    };

    let assembled = AiEngine::assemble_prompt(&snapshot).expect("assemble prompt");
    let bytes = assembled.canonical_bytes();
    let text = std::str::from_utf8(bytes).expect("valid utf-8");

    assert!(text.contains("You are Bitty AI."));
    assert!(text.contains("Be concise."));
}

#[test]
fn snapshot_engine_verifies_and_ingests() {
    let mut engine = AiEngine::new_snapshot_engine();

    let json_bytes = format!(
        r#"{{"schema":{{"name":"ProjectSnapshot","version":{},"canonicalization_version":{}}},"label":"test-project","revision":"rev-1","units":[],"entrypoints":[],"dependencies":[]}}"#,
        SNAPSHOT_SCHEMA_VERSION, SNAPSHOT_CANONICALIZATION_VERSION
    )
    .into_bytes();

    let digest = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&json_bytes);
        hex::encode(hasher.finalize())
    };

    let req = SnapshotIngestRequest {
        canonical_bytes: &json_bytes,
        expected_digest: &digest,
        record_id: "test-snap-1",
        owner: "term-1",
        collected_at_ms: 1_000,
        priority: bitty_ai_runtime::context::ContextPriority::High,
        refresh: RefreshAuthorization::authorize(1),
    };

    let record = engine.ingest(&req).expect("ingest success");
    assert_eq!(record.provider.as_str(), "project");
    assert!(record.is_untrusted_surface);
}

#[test]
fn engine_opens_transactional_journal() {
    let temp_dir = std::env::temp_dir().join(format!("bitty_test_journal_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let db_path = temp_dir.join("test_journal.sqlite");
    let _ = std::fs::remove_file(&db_path);

    let journal = AiEngine::open_journal(&db_path).expect("open journal");
    let rec = journal
        .append("evt-1", "task.created", "payload-1", 1_000)
        .expect("append event");
    assert_eq!(rec.seq, 1);

    let records = journal.read_all().expect("read records");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "evt-1");
    assert_eq!(records[0].kind, "task.created");
    assert_eq!(records[0].payload, "payload-1");

    // Clean up
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir_all(&temp_dir);
}

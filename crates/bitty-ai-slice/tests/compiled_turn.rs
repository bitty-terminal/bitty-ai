//! Compiled-turn ingestion tests (AI-0141): real compiler-produced layer
//! texts flow into project + delta records through
//! [`ingest_compiled_turn`](crate::snapshot_ingest::ingest_compiled_turn).
//!
//! Fixtures are recorded compiler outputs (PROJECT + DELTA texts, generated
//! read-only via the context-compiler binary over synthetic inputs); no
//! subprocess, no filesystem, no network in tests.

use bitty_ai_runtime::{ArtifactStore, ContextPriority, ContextRequest, assemble};
use bitty_ai_slice::{
    COMPILED_DELTA_MARKER, COMPILED_DELTA_PROVIDER, CompiledTurnIngestError,
    CompiledTurnIngestRequest, RefreshAuthorization, SNAPSHOT_LAYER_MARKER, SNAPSHOT_PROVIDER,
    ingest_compiled_turn,
};

const PROJECT_TEXT: &str = include_str!("fixtures/compiled_turn/project.txt");
const PROJECT_DIGEST: &str = include_str!("fixtures/compiled_turn/project.digest.txt");
const DELTA_1: &str = include_str!("fixtures/compiled_turn/delta1.txt");
const DELTA_2: &str = include_str!("fixtures/compiled_turn/delta2.txt");

fn request<'a>(
    project_text: &'a str,
    project_digest: &'a str,
    deltas: &'a [&'a str],
    generation: u64,
) -> CompiledTurnIngestRequest<'a> {
    CompiledTurnIngestRequest {
        project_text,
        project_digest,
        delta_texts: deltas,
        generation,
        project_record_id: "compiler-turn-1",
        delta_record_id_stem: "compiler-delta-1",
        owner: "term-1",
        collected_at_ms: 100,
        project_priority: ContextPriority::High,
        delta_priority: ContextPriority::Normal,
        refresh: RefreshAuthorization::authorize(generation),
    }
}

fn digest_of(text: &str) -> String {
    // Fixture files may carry one trailing newline from the editor; the
    // compiler emits texts without trailing newlines, so strip exactly one.
    text.strip_suffix('\n').unwrap_or(text).to_owned()
}

#[test]
fn recorded_fixture_markers_are_exact() {
    let project = digest_of(PROJECT_TEXT);
    assert!(project.starts_with(SNAPSHOT_LAYER_MARKER));
    assert_eq!(project.matches(SNAPSHOT_LAYER_MARKER).count(), 1);
    assert!(project.contains("full-digest"));
    for delta in [DELTA_1, DELTA_2] {
        let delta = digest_of(delta);
        assert!(delta.starts_with(COMPILED_DELTA_MARKER));
    }
}

#[test]
fn compiled_turn_ingests_project_plus_deltas() {
    let project = digest_of(PROJECT_TEXT);
    let digest = digest_of(PROJECT_DIGEST);
    let d1 = digest_of(DELTA_1);
    let d2 = digest_of(DELTA_2);
    let deltas = [d1.as_str(), d2.as_str()];
    let mut store = ArtifactStore::new();
    let records = ingest_compiled_turn(&request(&project, &digest, &deltas, 1), &mut store)
        .expect("compiled turn ingests");
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].provider, SNAPSHOT_PROVIDER);
    assert!(records[0].is_untrusted_surface);
    assert_eq!(records[0].id, "compiler-turn-1");
    assert_eq!(records[0].summary, project);
    for (index, record) in records[1..].iter().enumerate() {
        assert_eq!(record.provider, COMPILED_DELTA_PROVIDER);
        assert!(!record.is_untrusted_surface);
        assert_eq!(record.id, format!("compiler-delta-1-{index}"));
        assert_eq!(record.generation, 1);
    }
    // Assembles together under one generation.
    let assembled = assemble(
        &records,
        &mut store,
        &ContextRequest {
            max_tokens: None,
            max_bytes: Some(64 * 1024),
            current_generation: 1,
        },
    )
    .expect("compiled records assemble");
    assert_eq!(assembled.records.len(), 3);
}

#[test]
fn stale_delta_fails_the_whole_turn_closed() {
    let project = digest_of(PROJECT_TEXT);
    let digest = digest_of(PROJECT_DIGEST);
    let stale = DELTA_1.trim_end_matches('\n').replacen("gen 1", "gen 0", 1);
    let deltas = [stale.as_str()];
    let mut store = ArtifactStore::new();
    let before = (store.len(), store.total_bytes());
    let error = ingest_compiled_turn(&request(&project, &digest, &deltas, 1), &mut store)
        .expect_err("stale delta must fail closed");
    assert!(matches!(
        error,
        CompiledTurnIngestError::StaleLayer {
            index: 0,
            actual: 0,
            current: 1
        }
    ));
    assert_eq!((store.len(), store.total_bytes()), before);
}

#[test]
fn bad_markers_fail_closed() {
    let project = digest_of(PROJECT_TEXT);
    let digest = digest_of(PROJECT_DIGEST);
    let mut store = ArtifactStore::new();
    // Hand-written PROJECT text without the marker.
    let error = ingest_compiled_turn(
        &request("hand-written project text", &digest, &[], 1),
        &mut store,
    )
    .expect_err("marker-less project must fail");
    assert_eq!(error, CompiledTurnIngestError::BadProjectMarker);
    // Marker-less delta.
    let deltas = ["hand-written delta"];
    let error = ingest_compiled_turn(&request(&project, &digest, &deltas, 1), &mut store)
        .expect_err("marker-less delta must fail");
    assert_eq!(error, CompiledTurnIngestError::BadDelta { index: 0 });
    // Wrong digest.
    let error = ingest_compiled_turn(
        &request(&project, "0".repeat(64).as_str(), &[], 1),
        &mut store,
    )
    .expect_err("foreign digest must fail");
    assert_eq!(error, CompiledTurnIngestError::ProjectDigestMismatch);
}

#[test]
fn truncation_marker_tail_compiles_as_inert_delta() {
    let project = digest_of(PROJECT_TEXT);
    let digest = digest_of(PROJECT_DIGEST);
    let tail = "delta/1 trunc gen 1 authority host [truncated 2 deltas]";
    let deltas = [tail];
    let mut store = ArtifactStore::new();
    let records = ingest_compiled_turn(&request(&project, &digest, &deltas, 1), &mut store)
        .expect("truncation tail compiles");
    assert_eq!(records.len(), 2);
    assert!(records[1].summary.contains("[truncated 2 deltas]"));
}

#[test]
fn compiled_ingest_is_deterministic() {
    let project = digest_of(PROJECT_TEXT);
    let digest = digest_of(PROJECT_DIGEST);
    let d1 = digest_of(DELTA_1);
    let deltas = [d1.as_str()];
    let mut first = ArtifactStore::new();
    let mut second = ArtifactStore::new();
    let a = ingest_compiled_turn(&request(&project, &digest, &deltas, 1), &mut first)
        .expect("first ingest");
    let b = ingest_compiled_turn(&request(&project, &digest, &deltas, 1), &mut second)
        .expect("second ingest");
    assert_eq!(a, b);
}

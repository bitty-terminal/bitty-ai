//! Snapshot-backed PROJECT layer builder (AI-0128).
//!
//! Follows AI-0127: [`project_layer_text`] renders the text; this builder
//! fills the PROJECT layer of a validated [`PromptSnapshot`] from an
//! ingested record, failing closed on non-project or marker-less records.
//! Slice-only adapter tests; no runtime changes.

use bitty_ai_runtime::{ArtifactStore, ContextPriority, PromptLayer, assemble_prompt};
use bitty_ai_slice::{
    ProjectLayerError, RefreshAuthorization, SNAPSHOT_PROVIDER, SnapshotIngestRequest,
    ingest_snapshot, prompt_snapshot_with_project, snapshot_digest_hex,
};

fn snapshot_bytes(revision: &str) -> Vec<u8> {
    format!(
        "{{\"schema\":{{\"name\":\"ProjectSnapshot\",\"version\":1,\"canonicalization_version\":1}},\"source\":{{\"label\":\"demo\",\"revision\":\"{revision}\"}},\"project_units\":[{{\"id\":1}}],\"entrypoints\":[],\"dependencies\":[]}}"
    )
    .into_bytes()
}

fn ingest(revision: &str) -> (bitty_ai_runtime::ContextRecord, String) {
    let bytes = snapshot_bytes(revision);
    let digest = snapshot_digest_hex(&bytes);
    let mut store = ArtifactStore::new();
    let request = SnapshotIngestRequest {
        canonical_bytes: &bytes,
        expected_digest: &digest,
        record_id: "builder-1",
        owner: "term-1",
        collected_at_ms: 100,
        priority: ContextPriority::High,
        refresh: RefreshAuthorization::authorize(1),
    };
    let record = ingest_snapshot(&request, &mut store).expect("ingests");
    (record, digest)
}

fn fixed_layers() -> (&'static str, &'static str, &'static str, &'static str) {
    (
        "core contract text",
        "user text",
        "skills text",
        "turn text",
    )
}

#[test]
fn builder_fills_project_layer_from_ingested_record() {
    let (mut record, digest) = ingest("abc123");
    // The builder gates on the marker: the L0 summary alone lacks it, so
    // wrap it the way the PROJECT text path does (summary is one consumer,
    // the marker-prefixed rendering is the other — see AI-0127).
    record.summary = format!("project-snapshot/1 {}", record.summary);
    let (core, user, skills, turn) = fixed_layers();
    let snapshot = prompt_snapshot_with_project(
        &record,
        &digest,
        core,
        user,
        skills,
        turn,
        "bitty-core-prompt@1",
    )
    .expect("builds");
    let assembled = assemble_prompt(&snapshot).expect("assembles");
    let project = assembled
        .sections
        .iter()
        .find(|section| section.layer == PromptLayer::Project)
        .expect("project section present");
    assert!(
        project.text.starts_with("project-snapshot/1 "),
        "{}",
        project.text
    );
    assert!(
        project.text.contains(&digest),
        "digest pinned: {}",
        project.text
    );
    assert_eq!(record.provider, SNAPSHOT_PROVIDER);
}

#[test]
fn builder_rejects_non_project_provider() {
    let (mut record, digest) = ingest("abc123");
    record.provider = "diagnostics".to_owned();
    record.summary = format!("project-snapshot/1 {}", record.summary);
    let (core, user, skills, turn) = fixed_layers();
    let err = prompt_snapshot_with_project(
        &record,
        &digest,
        core,
        user,
        skills,
        turn,
        "bitty-core-prompt@1",
    )
    .expect_err("wrong provider must fail");
    assert_eq!(
        err,
        ProjectLayerError::NotProjectRecord {
            provider: "diagnostics".to_owned()
        }
    );
}

#[test]
fn builder_rejects_marker_less_summary() {
    // A project-provider record whose summary was not produced by the
    // snapshot rendering path (no marker) must not fill the layer.
    let (record, digest) = ingest("abc123");
    assert!(
        !record.summary.contains("project-snapshot/1"),
        "L0 summary itself carries no marker"
    );
    let (core, user, skills, turn) = fixed_layers();
    let err = prompt_snapshot_with_project(
        &record,
        &digest,
        core,
        user,
        skills,
        turn,
        "bitty-core-prompt@1",
    )
    .expect_err("marker-less summary must fail");
    assert_eq!(err, ProjectLayerError::MissingMarker);
}

#[test]
fn builder_propagates_layer_bound_violations() {
    let (mut record, digest) = ingest("abc123");
    record.summary = format!("project-snapshot/1 {}", record.summary);
    let (_, user, skills, turn) = fixed_layers();
    let oversized = "x".repeat(bitty_ai_runtime::MAX_LAYER_TEXT_BYTES + 1);
    let err = prompt_snapshot_with_project(
        &record,
        &digest,
        &oversized,
        user,
        skills,
        turn,
        "bitty-core-prompt@1",
    )
    .expect_err("oversized fixed layer must fail");
    assert!(
        matches!(err, ProjectLayerError::InvalidLayer { .. }),
        "typed bound failure: {err:?}"
    );
}

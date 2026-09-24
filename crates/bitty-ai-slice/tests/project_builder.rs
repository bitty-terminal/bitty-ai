//! Snapshot-backed PROJECT layer builder (AI-0128).
//!
//! Follows AI-0127: [`project_layer_text`] renders the text; this builder
//! fills the PROJECT layer of a validated [`PromptSnapshot`] from a raw
//! ingested record plus its verified digest, failing closed on non-project
//! records and digest mismatches. Slice-only adapter tests; no runtime
//! changes.

use bitty_ai_runtime::{ArtifactStore, ContextPriority, PromptLayer, assemble_prompt};
use bitty_ai_slice::{
    ProjectLayerError, RefreshAuthorization, SNAPSHOT_LAYER_MARKER, SNAPSHOT_PROVIDER,
    SnapshotIngestRequest, ingest_snapshot, prompt_snapshot_with_project, snapshot_digest_hex,
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
fn builder_fills_project_layer_from_raw_ingested_record() {
    // Raw ingest record in, no caller-side wrapping: the builder renders the
    // marker itself exactly once.
    let (record, digest) = ingest("abc123");
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
    assert_eq!(
        project.text.matches(SNAPSHOT_LAYER_MARKER).count(),
        1,
        "marker rendered exactly once: {}",
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
fn builder_rejects_digest_mismatch() {
    // A digest that does not belong to the record fails closed instead of
    // silently poisoning the PROJECT layer and the cache-affinity claim.
    let (record, _) = ingest("abc123");
    let (other, other_digest) = ingest("xyz999");
    let _ = other;
    let (core, user, skills, turn) = fixed_layers();
    let err = prompt_snapshot_with_project(
        &record,
        &other_digest,
        core,
        user,
        skills,
        turn,
        "bitty-core-prompt@1",
    )
    .expect_err("foreign digest must fail");
    assert_eq!(err, ProjectLayerError::DigestMismatch);
    // Empty digest fails the same way (no vacuous binding).
    let err =
        prompt_snapshot_with_project(&record, "", core, user, skills, turn, "bitty-core-prompt@1")
            .expect_err("empty digest must fail");
    assert_eq!(err, ProjectLayerError::DigestMismatch);
}

#[test]
fn builder_propagates_layer_bound_violations() {
    let (record, digest) = ingest("abc123");
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

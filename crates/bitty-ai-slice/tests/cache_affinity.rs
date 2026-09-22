//! Snapshot digest as prefix-cache stable input (AI-0127).
//!
//! Affinity rule: the PROJECT layer of the prompt stable prefix carries the
//! [`project_layer_text`] rendering of the ingested snapshot, so the
//! [`CacheKey`] warms exactly when the snapshot digest is unchanged and
//! misses exactly when it changes. Slice-side adapter test only: the prompt
//! assembly and keying machinery is runtime-owned and unchanged here.
//!
//! Deterministic, no subprocess, no filesystem, no network. Snapshot bytes
//! are synthetic (AI-0121 style); prompt assembly runs over caller-built
//! snapshots; keys run over a `Vec` seen-list.

use bitty_ai_runtime::{ArtifactStore, ContextPriority};
use bitty_ai_runtime::{
    CacheKey, CacheScope, LayerInput, PromptLayer, PromptSnapshot, assemble_prompt,
};
use bitty_ai_slice::{
    RefreshAuthorization, SnapshotIngestRequest, ingest_snapshot, project_layer_text,
    snapshot_digest_hex,
};

/// Minimal synthetic ProjectSnapshot v1 JSON with a variable revision.
fn snapshot_bytes(revision: &str) -> Vec<u8> {
    format!(
        "{{\"schema\":{{\"name\":\"ProjectSnapshot\",\"version\":1,\"canonicalization_version\":1}},\"source\":{{\"label\":\"demo\",\"revision\":\"{revision}\"}},\"project_units\":[{{\"id\":1}}],\"entrypoints\":[],\"dependencies\":[]}}"
    )
    .into_bytes()
}

/// Ingest `bytes` at `generation` and return the PROJECT-layer text plus the
/// full digest the layer pins.
fn ingest_project_text(bytes: &[u8], digest: &str, generation: u64) -> String {
    let mut store = ArtifactStore::new();
    let request = SnapshotIngestRequest {
        canonical_bytes: bytes,
        expected_digest: digest,
        record_id: "affinity-1",
        owner: "term-1",
        collected_at_ms: 100,
        priority: ContextPriority::High,
        refresh: RefreshAuthorization::authorize(generation),
    };
    let record = ingest_snapshot(&request, &mut store).expect("ingests");
    project_layer_text(&record.summary, digest)
}

/// Assemble a prompt with `project_text` at the PROJECT layer (all other
/// layers fixed) and key it for the session scope.
fn key_for_project_text(project_text: &str) -> CacheKey {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            LayerInput::text_only(PromptLayer::CoreContract, "core contract text"),
            LayerInput::text_only(PromptLayer::User, "user text"),
            LayerInput::text_only(PromptLayer::Project, project_text),
            LayerInput::text_only(PromptLayer::SkillsProfile, "skills text"),
            LayerInput::text_only(PromptLayer::RuntimeTurn, "turn text"),
        ],
    )
    .expect("valid test snapshot");
    let assembled = assemble_prompt(&snapshot).expect("assembles");
    CacheKey::new(
        "bitty-fake",
        "fake-chat",
        CacheScope::Session,
        assembled.canonical_bytes(),
    )
    .expect("valid test key inputs")
}

#[test]
fn same_digest_warms_same_key() {
    // Same snapshot bytes ingested twice (two generations, same content):
    // identical PROJECT text, identical session key. The key warms.
    let bytes = snapshot_bytes("abc123");
    let digest = snapshot_digest_hex(&bytes);
    let first = ingest_project_text(&bytes, &digest, 1);
    let second = ingest_project_text(&bytes, &digest, 2);
    assert_eq!(first, second, "same digest renders same project text");
    assert_eq!(
        key_for_project_text(&first),
        key_for_project_text(&second),
        "same project text keys identically"
    );
}

#[test]
fn changed_digest_misses_key() {
    // Refreshed snapshot (revision changes, digest changes): PROJECT text
    // differs byte-for-byte, so the session key misses. The full digest is
    // embedded precisely to make any snapshot change a key change.
    let before = snapshot_bytes("abc123");
    let digest_before = snapshot_digest_hex(&before);
    let after = snapshot_bytes("abc124");
    let digest_after = snapshot_digest_hex(&after);
    assert_ne!(digest_before, digest_after, "revisions digest differently");
    let text_before = ingest_project_text(&before, &digest_before, 1);
    let text_after = ingest_project_text(&after, &digest_after, 2);
    assert_ne!(
        text_before, text_after,
        "changed digest renders different text"
    );
    assert_ne!(
        key_for_project_text(&text_before),
        key_for_project_text(&text_after),
        "changed project text misses the cached key"
    );
}

#[test]
fn project_text_is_stable_prefixed_and_bounded() {
    // The rendering is self-describing, carries the full digest, and stays
    // far below layer-text bounds for real snapshots.
    let bytes = snapshot_bytes("abc123");
    let digest = snapshot_digest_hex(&bytes);
    let text = ingest_project_text(&bytes, &digest, 1);
    assert!(text.starts_with("project-snapshot/1 "), "{text}");
    assert!(text.contains(&digest), "full digest embedded: {text}");
    assert!(
        text.len() < bitty_ai_runtime::MAX_LAYER_TEXT_BYTES,
        "bounded layer text"
    );
}

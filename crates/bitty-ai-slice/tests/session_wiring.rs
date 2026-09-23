//! Snapshot stream session wiring end-to-end (AI-0130).
//!
//! One deterministic session script composing every piece built since
//! AI-0121: [`RefreshLedger`] issuance, [`ingest_snapshot`] verify, builder,
//! prompt assembly, [`CacheKey`], generation-scoped expiry, and
//! invalidation. Slice-only; runtime unchanged. No subprocess, no
//! filesystem, no network.

use bitty_ai_runtime::{ArtifactStore, CacheKey, CacheScope, ContextPriority, assemble_prompt};
use bitty_ai_slice::{
    RefreshAuthorization, RefreshLedger, SnapshotIngestRequest, ingest_snapshot,
    prompt_snapshot_with_project, snapshot_digest_hex,
};

fn snapshot_bytes(revision: &str) -> Vec<u8> {
    format!(
        "{{\"schema\":{{\"name\":\"ProjectSnapshot\",\"version\":1,\"canonicalization_version\":1}},\"source\":{{\"label\":\"demo\",\"revision\":\"{revision}\"}},\"project_units\":[{{\"id\":1}}],\"entrypoints\":[],\"dependencies\":[]}}"
    )
    .into_bytes()
}

fn session_key(record: &bitty_ai_runtime::ContextRecord, digest: &str, turn: &str) -> CacheKey {
    let snapshot = prompt_snapshot_with_project(
        record,
        digest,
        "core contract text",
        "user text",
        "skills text",
        turn,
        "bitty-core-prompt@1",
    )
    .expect("builds");
    let assembled = assemble_prompt(&snapshot).expect("assembles");
    CacheKey::new(
        "bitty-fake",
        "fake-chat",
        CacheScope::Session,
        assembled.canonical_bytes(),
    )
    .expect("valid key inputs")
}

#[test]
fn full_session_lifecycle_composes() {
    let mut ledger = RefreshLedger::new();
    let mut store = ArtifactStore::new();

    // Turn 1: issue gen 1, ingest, build, key. The pin is warm.
    let bytes_v1 = snapshot_bytes("abc123");
    let digest_v1 = snapshot_digest_hex(&bytes_v1);
    let token_v1 = ledger.issue(1).expect("issues gen 1");
    assert_eq!(token_v1, RefreshAuthorization::authorize(1));
    let record_v1 = ingest_snapshot(
        &SnapshotIngestRequest {
            canonical_bytes: &bytes_v1,
            expected_digest: &digest_v1,
            record_id: "sess-1",
            owner: "term-1",
            collected_at_ms: 100,
            priority: ContextPriority::High,
            refresh: token_v1,
        },
        &mut store,
    )
    .expect("ingests v1");
    let key_v1 = session_key(&record_v1, &digest_v1, "user: q1");
    // Tail-only follow-up hits the same key.
    let key_v1b = session_key(&record_v1, &digest_v1, "user: q2");
    assert_eq!(key_v1b, key_v1, "stable head keeps warming");

    // Turn 2: refresh to gen 2 with new bytes. Old key misses, new warms.
    let bytes_v2 = snapshot_bytes("abc124");
    let digest_v2 = snapshot_digest_hex(&bytes_v2);
    assert_ne!(digest_v2, digest_v1);
    let token_v2 = ledger.issue(2).expect("issues gen 2");
    assert_eq!(ledger.retired(), &[1], "gen 1 retired for audit");
    let record_v2 = ingest_snapshot(
        &SnapshotIngestRequest {
            canonical_bytes: &bytes_v2,
            expected_digest: &digest_v2,
            record_id: "sess-2",
            owner: "term-1",
            collected_at_ms: 200,
            priority: ContextPriority::High,
            refresh: token_v2,
        },
        &mut store,
    )
    .expect("ingests v2");
    let key_v2 = session_key(&record_v2, &digest_v2, "user: q1");
    assert_ne!(key_v2, key_v1, "refresh misses the retired key");

    // Replay of gen 1 fails closed at the ledger: no token, no ingestion.
    assert!(
        ledger.issue(1).is_err(),
        "retired generation cannot re-issue"
    );

    // Invalidate the v1 backing artifact (if externalized it retires; inline
    // bodies need no store action — assert whichever applies fail-closed).
    // v1 here is small (inline), so demonstrate invalidation on the store
    // with a large body instead: ingest a large snapshot, externalize,
    // invalidate, and prove re-resolution fails.
    let big_value = serde_json::json!({
        "schema": {"name": "ProjectSnapshot", "version": 1, "canonicalization_version": 1},
        "source": {"label": "demo", "revision": "big001"},
        "project_units": [{"id": 1}],
        "entrypoints": [],
        "dependencies": [],
        "padding": "x".repeat(bitty_ai_runtime::context::MAX_RECORD_BODY_BYTES),
    });
    let big_bytes = serde_json::to_vec(&big_value).expect("serializes");
    let big_digest = snapshot_digest_hex(&big_bytes);
    let token_v3 = ledger.issue(3).expect("issues gen 3");
    let big_record = ingest_snapshot(
        &SnapshotIngestRequest {
            canonical_bytes: &big_bytes,
            expected_digest: &big_digest,
            record_id: "sess-3",
            owner: "term-1",
            collected_at_ms: 300,
            priority: ContextPriority::High,
            refresh: token_v3,
        },
        &mut store,
    )
    .expect("ingests big");
    let big_ref = match &big_record.body {
        bitty_ai_runtime::RecordBody::Artifact(reference) => reference.clone(),
        bitty_ai_runtime::RecordBody::Inline(_) => panic!("big body must externalize"),
    };
    assert_eq!(
        store.resolve(&big_ref, 3).expect("resolves at pin").len(),
        big_bytes.len()
    );
    assert!(store.invalidate(&big_ref), "host invalidates");
    assert!(
        store.resolve(&big_ref, 3).is_err(),
        "derived reference fails closed after invalidation"
    );
}

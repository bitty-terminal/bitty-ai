//! Multi-turn prefix-cache hit-rate evidence with snapshot pin (AI-0129).
//!
//! AIQ-13 narrowing input: scripted turn sequence proving the stable head
//! (snapshot-pinned PROJECT layer) keeps the session [`CacheKey`] warm while
//! only the RuntimeTurn tail varies, and that a refreshed snapshot misses.
//! Slice-only adapter test; runtime machinery unchanged. Deterministic, no
//! subprocess, no filesystem, no network.

use bitty_ai_runtime::{ArtifactStore, CacheKey, CacheScope, ContextPriority, assemble_prompt};
use bitty_ai_slice::{
    RefreshAuthorization, SnapshotIngestRequest, ingest_snapshot, prompt_snapshot_with_project,
    snapshot_digest_hex,
};

fn snapshot_bytes(revision: &str) -> Vec<u8> {
    format!(
        "{{\"schema\":{{\"name\":\"ProjectSnapshot\",\"version\":1,\"canonicalization_version\":1}},\"source\":{{\"label\":\"demo\",\"revision\":\"{revision}\"}},\"project_units\":[{{\"id\":1}}],\"entrypoints\":[],\"dependencies\":[]}}"
    )
    .into_bytes()
}

fn ingest(revision: &str, generation: u64) -> (bitty_ai_runtime::ContextRecord, String) {
    let bytes = snapshot_bytes(revision);
    let digest = snapshot_digest_hex(&bytes);
    let mut store = ArtifactStore::new();
    let request = SnapshotIngestRequest {
        canonical_bytes: &bytes,
        expected_digest: &digest,
        record_id: "hit-rate-1",
        owner: "term-1",
        collected_at_ms: 100,
        priority: ContextPriority::High,
        refresh: RefreshAuthorization::authorize(generation),
    };
    let record = ingest_snapshot(&request, &mut store).expect("ingests");
    (record, digest)
}

/// Session key for one turn: snapshot-backed PROJECT layer fixed, only the
/// RuntimeTurn tail varies (simulated user follow-ups).
fn session_key_for_turn(
    record: &bitty_ai_runtime::ContextRecord,
    digest: &str,
    turn_text: &str,
) -> CacheKey {
    let snapshot = prompt_snapshot_with_project(
        record,
        digest,
        "core contract text",
        "user text",
        "skills text",
        turn_text,
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
    .expect("valid test key inputs")
}

#[test]
fn stable_head_with_varying_tail_hits_consecutively() {
    // One ingest pins the head; three follow-up turns change only the tail.
    // All four session keys are identical: 3 hits over 4 turns.
    let (record, digest) = ingest("abc123", 1);
    let turns = [
        "user: explain foo",
        "user: now explain bar",
        "user: and what about baz?",
    ];
    let first = session_key_for_turn(&record, &digest, "user: initial question");
    let mut hits = 0;
    for turn in turns {
        let key = session_key_for_turn(&record, &digest, turn);
        assert_eq!(key, first, "tail-only change keeps the key");
        hits += 1;
    }
    assert_eq!(hits, 3, "three consecutive hits after the pinning turn");
}

#[test]
fn refreshed_snapshot_misses_then_rewarms() {
    // Refresh changes the PROJECT text: the next key misses. Repeating the
    // refreshed head warms the new key.
    let (before, digest_before) = ingest("abc123", 1);
    let (after, digest_after) = ingest("abc124", 2);
    assert_ne!(digest_before, digest_after);
    let key_before = session_key_for_turn(&before, &digest_before, "user: same question");
    let key_after = session_key_for_turn(&after, &digest_after, "user: same question");
    assert_ne!(key_before, key_after, "refreshed head misses the old key");
    let key_after_again = session_key_for_turn(&after, &digest_after, "user: follow-up");
    assert_ne!(key_after_again, key_before, "still misses the retired key");
    let key_after_repeat = session_key_for_turn(&after, &digest_after, "user: same question");
    assert_eq!(
        key_after_repeat, key_after,
        "repeated refreshed head warms the new key"
    );
}

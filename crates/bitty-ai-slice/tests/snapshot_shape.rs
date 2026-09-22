//! Ingestion regression over recorded real-shape `psnap` bytes (AI-0122).
//!
//! Follow-up to AI-0121, whose `snapshot_ingest` unit tests use synthetic JSON
//! only. The fixtures here are dead bytes: exact `psnap` stdout captured once
//! from the project-analysis experiment (main `2491019`, EXP-0018) and checked
//! in under `fixtures/`. No test spawns a subprocess, touches the filesystem,
//! or uses the network; `include_bytes!` keeps the bytes deterministic.
//!
//! Provenance (record once, never re-run in tests):
//!
//! ```sh
//! ./target/release/psnap --label fx-shape-small ./fixtures/minimal-rust \
//!     > crates/bitty-ai-slice/tests/fixtures/fx-shape-small.psnap.json
//! ./target/release/psnap --label fx-shape-large <bitty-ai checkout> \
//!     > crates/bitty-ai-slice/tests/fixtures/fx-shape-large.psnap.json
//! ```
//!
//! Only stdout was redirected: the stderr `digest:` line is excluded from the
//! fixture, and its hex must equal `sha256sum` of the fixture file (verified
//! at record time; also asserted below via the hardcoded digests). The small
//! fixture stays inline; the large fixture crosses `MAX_RECORD_BODY_BYTES`
//! and covers the externalize path. The newline-mismatch check pins the
//! EXP-0018 exact-bytes contract: the recorded digest verifies with no
//! stripping, so even one appended byte fails closed.
//!
//! If `psnap` output ever changes shape, these tests fail loudly on the
//! hardcoded digests: re-record deliberately, never silently.

use bitty_ai_runtime::context::{MAX_ARTIFACT_BYTES, MAX_RECORD_BODY_BYTES, MAX_SUMMARY_BYTES};
use bitty_ai_runtime::{
    ArtifactStore, AssembledContent, ContextPriority, ContextRequest, RecordBody, assemble,
};
use bitty_ai_slice::{
    RefreshAuthorization, SNAPSHOT_DIGEST_PREFIX_LEN, SNAPSHOT_PROVIDER, SnapshotIngestRequest,
    ingest_snapshot, snapshot_digest_hex,
};

/// Recorded `psnap --label fx-shape-small` stdout over `minimal-rust`
/// (2,286 bytes, inline path). Digest recorded alongside the fixture.
const SMALL_BYTES: &[u8] = include_bytes!("fixtures/fx-shape-small.psnap.json");
const SMALL_DIGEST: &str = "d152a97bb0b8599c145e03229321fc7d451871a29abc36d65021837c6d41b7e6";

/// Recorded `psnap --label fx-shape-large` stdout over the `bitty-ai`
/// checkout (22,748 bytes, externalize path). Digest recorded alongside.
const LARGE_BYTES: &[u8] = include_bytes!("fixtures/fx-shape-large.psnap.json");
const LARGE_DIGEST: &str = "4a3b44cacb1af8ac6b964fb1fdb241af301954d557a7c0e3846c4ccc798d7c48";

const COLLECTED_AT_MS: u64 = 1_000;

fn ingest_request<'a>(
    bytes: &'a [u8],
    digest: &'a str,
    record_id: &'a str,
    generation: u64,
) -> SnapshotIngestRequest<'a> {
    SnapshotIngestRequest {
        canonical_bytes: bytes,
        expected_digest: digest,
        record_id,
        owner: "term-1",
        collected_at_ms: COLLECTED_AT_MS,
        priority: ContextPriority::High,
        refresh: RefreshAuthorization::authorize(generation),
    }
}

fn budget(bytes: usize, generation: u64) -> ContextRequest {
    ContextRequest {
        max_tokens: None,
        max_bytes: Some(bytes as u64),
        current_generation: generation,
    }
}

/// Assert the fixture carries the real `psnap` envelope shape (schema,
/// source, units, entrypoints, dependencies, conventions, diagnostics).
fn assert_real_shape(bytes: &[u8], label: &str, units: usize, entrypoints: usize, deps: usize) {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).expect("recorded fixture parses as JSON");
    assert_eq!(
        value
            .get("schema")
            .and_then(|schema| schema.get("name"))
            .and_then(serde_json::Value::as_str),
        Some("ProjectSnapshot"),
        "real-shape schema name"
    );
    for field in ["version", "canonicalization_version"] {
        assert_eq!(
            value
                .get("schema")
                .and_then(|schema| schema.get(field))
                .and_then(serde_json::Value::as_u64),
            Some(1),
            "real-shape schema.{field}"
        );
    }
    assert_eq!(
        value
            .get("source")
            .and_then(|source| source.get("label"))
            .and_then(serde_json::Value::as_str),
        Some(label),
        "real-shape source label"
    );
    for (field, expected) in [
        ("project_units", units),
        ("entrypoints", entrypoints),
        ("dependencies", deps),
    ] {
        assert_eq!(
            value
                .get(field)
                .and_then(serde_json::Value::as_array)
                .map(Vec::len),
            Some(expected),
            "real-shape {field} count"
        );
    }
    for field in ["conventions", "diagnostics"] {
        assert!(
            value
                .get(field)
                .and_then(serde_json::Value::as_array)
                .is_some(),
            "real-shape carries a {field} array"
        );
    }
}

#[test]
fn recorded_digests_pin_exact_fixture_bytes() {
    // The hardcoded hex is the digest recorded alongside each fixture at
    // capture time (stderr `digest:` line, equal to `sha256sum` of the
    // file). Any silent fixture drift fails here first, loudly.
    assert_eq!(snapshot_digest_hex(SMALL_BYTES), SMALL_DIGEST);
    assert_eq!(snapshot_digest_hex(LARGE_BYTES), LARGE_DIGEST);

    assert_eq!(SMALL_BYTES.len(), 2_286);
    assert_eq!(LARGE_BYTES.len(), 22_748);
    assert!(SMALL_BYTES.len() <= MAX_RECORD_BODY_BYTES);
    assert!(LARGE_BYTES.len() > MAX_RECORD_BODY_BYTES);
    assert!(LARGE_BYTES.len() <= MAX_ARTIFACT_BYTES);

    // Exact bytes, no trailing newline: the recorded digest pins the shape
    // byte-for-byte.
    for bytes in [SMALL_BYTES, LARGE_BYTES] {
        assert!(!bytes.is_empty());
        assert!(
            !bytes.ends_with(b"\n"),
            "recorded stdout carries no trailing newline"
        );
    }
}

#[test]
fn small_real_shape_ingest_verifies_without_stripping() {
    assert_real_shape(SMALL_BYTES, "fx-shape-small", 1, 2, 1);

    let mut store = ArtifactStore::new();
    let record = ingest_snapshot(
        &ingest_request(SMALL_BYTES, SMALL_DIGEST, "fx-shape-small-1", 7),
        &mut store,
    )
    .expect("recorded small bytes ingest with the recorded digest");
    assert_eq!(record.provider, SNAPSHOT_PROVIDER);
    assert_eq!(record.provider, "project");
    assert!(record.is_untrusted_surface);
    assert_eq!(record.generation, 7);
    assert_eq!(record.owner.as_str(), "term-1");
    assert_eq!(record.priority, ContextPriority::High);
    assert_eq!(record.supersedes, None);
    // Real label plus a null revision (carried as `unknown`), real counts,
    // and the digest prefix of the recorded digest.
    assert!(
        record.summary.contains("fx-shape-small"),
        "{}",
        record.summary
    );
    assert!(record.summary.contains("rev unknown"), "{}", record.summary);
    assert!(record.summary.contains("units 1"), "{}", record.summary);
    assert!(
        record.summary.contains("entrypoints 2"),
        "{}",
        record.summary
    );
    assert!(record.summary.contains("deps 1"), "{}", record.summary);
    assert!(
        record
            .summary
            .contains(&SMALL_DIGEST[..SNAPSHOT_DIGEST_PREFIX_LEN]),
        "{}",
        record.summary
    );
    assert!(record.summary.len() <= MAX_SUMMARY_BYTES);
    match &record.body {
        RecordBody::Inline(inline) => assert_eq!(inline, SMALL_BYTES),
        RecordBody::Artifact(_) => panic!("small recorded bytes must stay inline"),
    }
    record.validate().expect("record validates");
    assert!(store.is_empty());

    // The record assembles through the existing L0/L1 builder with no
    // omissions and its inline content intact.
    let assembled = assemble(&[record], &mut store, &budget(32_768, 7)).expect("assembles");
    assert_eq!(assembled.context_refs, vec!["fx-shape-small-1".to_owned()]);
    assert_eq!(assembled.records.len(), 1);
    assert!(assembled.records[0].is_untrusted_surface);
    assert!(assembled.omitted_ids.is_empty());
    assert!(assembled.pruned_ids.is_empty());
    match &assembled.records[0].content {
        AssembledContent::Inline(inline) => assert_eq!(inline, SMALL_BYTES),
        AssembledContent::Reference(_) => panic!("small content must assemble inline"),
    }

    // EXP-0018 exact-bytes contract: the helper verifies with no stripping,
    // so the same bytes plus one newline byte fail closed against the
    // recorded digest.
    let mut with_newline = SMALL_BYTES.to_vec();
    with_newline.push(b'\n');
    assert_ne!(snapshot_digest_hex(&with_newline), SMALL_DIGEST);
    let mut store = ArtifactStore::new();
    let err = ingest_snapshot(
        &ingest_request(&with_newline, SMALL_DIGEST, "fx-shape-small-1", 7),
        &mut store,
    )
    .expect_err("newline-suffixed bytes must fail the recorded digest");
    assert!(
        matches!(
            err,
            bitty_ai_slice::SnapshotIngestError::DigestMismatch { .. }
        ),
        "typed fail-closed error, got {err:?}"
    );
    assert!(store.is_empty());
}

#[test]
fn large_real_shape_ingest_externalizes_and_round_trips() {
    assert_real_shape(LARGE_BYTES, "fx-shape-large", 3, 2, 8);

    let mut store = ArtifactStore::new();
    let record = ingest_snapshot(
        &ingest_request(LARGE_BYTES, LARGE_DIGEST, "fx-shape-large-1", 9),
        &mut store,
    )
    .expect("recorded large bytes ingest with the recorded digest");
    assert_eq!(record.provider, SNAPSHOT_PROVIDER);
    assert!(record.is_untrusted_surface);
    assert_eq!(record.generation, 9);
    assert_eq!(record.supersedes, None);
    // Real label, real full-length git revision, real counts, digest prefix.
    assert!(
        record.summary.contains("fx-shape-large"),
        "{}",
        record.summary
    );
    assert!(
        record
            .summary
            .contains("71ce680a93fc26421ec341e464c4e83089425fb4"),
        "{}",
        record.summary
    );
    assert!(record.summary.contains("units 3"), "{}", record.summary);
    assert!(
        record.summary.contains("entrypoints 2"),
        "{}",
        record.summary
    );
    assert!(record.summary.contains("deps 8"), "{}", record.summary);
    assert!(
        record
            .summary
            .contains(&LARGE_DIGEST[..SNAPSHOT_DIGEST_PREFIX_LEN]),
        "{}",
        record.summary
    );
    assert!(record.summary.len() <= MAX_SUMMARY_BYTES);
    match &record.body {
        RecordBody::Artifact(reference) => {
            assert_eq!(store.len(), 1);
            assert_eq!(
                store.resolve(reference, 9).expect("artifact resolves"),
                LARGE_BYTES
            );
        }
        RecordBody::Inline(_) => panic!("large recorded bytes must externalize"),
    }
    record.validate().expect("record validates");

    // Assembly keeps the externalized body as a reference that resolves to
    // the exact recorded bytes, with nothing omitted.
    let assembled = assemble(&[record], &mut store, &budget(32_768, 9)).expect("assembles");
    assert_eq!(assembled.context_refs, vec!["fx-shape-large-1".to_owned()]);
    assert_eq!(assembled.records.len(), 1);
    assert!(assembled.omitted_ids.is_empty());
    match &assembled.records[0].content {
        AssembledContent::Reference(reference) => {
            assert_eq!(
                store.resolve(reference, 9).expect("reference resolves"),
                LARGE_BYTES,
                "artifact resolves to the exact recorded bytes"
            );
        }
        AssembledContent::Inline(_) => panic!("large content must assemble as a reference"),
    }
    assert_eq!(store.len(), 1);
}

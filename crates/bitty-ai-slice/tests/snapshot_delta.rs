//! Session-pinned snapshot plus per-turn delta assembly demo (AI-0123).
//!
//! Demonstrates the prefix-cache design's session-pinned snapshot cycle using
//! only existing machinery: [`ingest_snapshot`](bitty_ai_slice::ingest_snapshot)
//! from AI-0121 plus the runtime's [`assemble`](bitty_ai_runtime::assemble)
//! with its generation pin ([`StaleGeneration`](bitty_ai_runtime::ContextError::StaleGeneration))
//! gate (`AG-2`). Slice-side demo/test code only; no runtime changes.
//!
//! Deterministic multi-turn walk, synthetic bytes only (AI-0121 synthetic
//! style, dead bytes, no subprocess, no filesystem, no network):
//!
//! - Turn 1 (gen 1): ingest snapshot bytes with
//!   [`RefreshAuthorization::authorize(1)`](bitty_ai_slice::RefreshAuthorization)
//!   into record `R1` (provider `"project"`, untrusted). Assemble `[R1]` at
//!   generation 1: ok, refs `[R1]`.
//! - Turn 2 (gen 1, delta): the host adds a plain trusted [`ContextRecord`]
//!   (provider `"diagnostics"`, constructed directly, never via ingest)
//!   describing `src/foo.rs` changing. Assemble `[R1, delta]` at generation 1:
//!   both present, the snapshot stays pinned, the delta sits alongside.
//! - Turn 3 (refresh, gen 2): re-ingest refreshed bytes with `authorize(2)`
//!   into record `R2` (generation 2). Assemble `[R2, delta-gen2]` at
//!   generation 2: ok. Assemble `[R1, ..]` at generation 2: `StaleGeneration`
//!   fail-closed. This is the money assertion: refresh rotates the pin, and a
//!   stale snapshot never leaks (`AG-2`).
//!
//! Generations pin every record, so the per-turn delta is re-collected at the
//! current generation each turn (`delta-gen1` vs `delta-gen2`): a gen-1 delta
//! assembled at gen 2 is stale exactly like a gen-1 snapshot. No `supersedes`
//! links are used anywhere here; the demo covers pin plus delta plus refresh
//! plus stale only.
//!
//! On `supersedes`: ingested snapshot records carry `supersedes: None`, and
//! the runtime ignores `supersedes` links from untrusted-surface records
//! anyway (deny by default, AIQ-11; see `context.rs` assembly docs and the
//! `untrusted_supersede_link_is_ignored` runtime test). The `None` assertions
//! below pin the ingest half of that rule.

use bitty_ai_runtime::context::KNOWN_PROVIDERS;
use bitty_ai_runtime::{
    ArtifactStore, ContextError, ContextPriority, ContextRecord, ContextRequest, RecordBody,
    StableId, assemble,
};
use bitty_ai_slice::{
    RefreshAuthorization, SNAPSHOT_PROVIDER, SnapshotIngestRequest, ingest_snapshot,
    snapshot_digest_hex,
};

/// Minimal synthetic ProjectSnapshot v1 JSON (inline fixture, never produced
/// by running `psnap`: no process spawning in tests). `revision` varies per
/// refresh so the gen-2 bytes differ from the gen-1 bytes.
fn snapshot_bytes(revision: &str) -> Vec<u8> {
    format!(
        "{{\"schema\":{{\"name\":\"ProjectSnapshot\",\"version\":1,\"canonicalization_version\":1}},\"source\":{{\"label\":\"demo\",\"revision\":\"{revision}\"}},\"project_units\":[{{\"id\":1}}],\"entrypoints\":[{{\"id\":1}},{{\"id\":2}}],\"dependencies\":[{{\"id\":1}},{{\"id\":2}},{{\"id\":3}}]}}"
    )
    .into_bytes()
}

fn ingest_request<'a>(
    bytes: &'a [u8],
    digest: &'a str,
    record_id: &'a str,
    generation: u64,
    collected_at_ms: u64,
) -> SnapshotIngestRequest<'a> {
    SnapshotIngestRequest {
        canonical_bytes: bytes,
        expected_digest: digest,
        record_id,
        owner: "term-1",
        collected_at_ms,
        priority: ContextPriority::High,
        refresh: RefreshAuthorization::authorize(generation),
    }
}

/// One host-collected per-turn delta: a plain trusted record built directly
/// (never via ingest) describing `src/foo.rs` changing. Each turn re-collects
/// it at the turn's generation, since assembly pins every record generation.
fn delta_record(id: &str, generation: u64, collected_at_ms: u64) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "diagnostics".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation,
        collected_at_ms,
        priority: ContextPriority::Normal,
        summary: "diagnostics: src/foo.rs changed (unused import removed)".to_owned(),
        body: RecordBody::Inline(b"src/foo.rs changed: removed unused import".to_vec()),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn budget(generation: u64) -> ContextRequest {
    ContextRequest {
        max_tokens: None,
        max_bytes: Some(32_768),
        current_generation: generation,
    }
}

#[test]
fn session_pinned_snapshot_with_per_turn_delta() {
    assert!(KNOWN_PROVIDERS.contains(&SNAPSHOT_PROVIDER));
    assert!(KNOWN_PROVIDERS.contains(&"diagnostics"));

    // One session-scoped store across all turns: inline bodies never mutate
    // it, so it stays empty throughout this demo.
    let mut store = ArtifactStore::new();

    // Turn 1 (gen 1): ingest pins the snapshot at generation 1.
    let bytes_gen1 = snapshot_bytes("abc123");
    let digest_gen1 = snapshot_digest_hex(&bytes_gen1);
    let r1 = ingest_snapshot(
        &ingest_request(&bytes_gen1, &digest_gen1, "psnap-delta-1", 1, 1_000),
        &mut store,
    )
    .expect("gen-1 snapshot ingests");
    assert_eq!(r1.provider, SNAPSHOT_PROVIDER);
    assert_eq!(r1.provider, "project");
    assert!(r1.is_untrusted_surface);
    assert_eq!(r1.generation, 1);
    // Ingest half of the deny-by-default rule: untrusted observations must
    // not name eviction victims (the runtime ignores such links anyway).
    assert_eq!(r1.supersedes, None);
    r1.validate().expect("ingested record validates");

    let assembled_t1 =
        assemble(std::slice::from_ref(&r1), &mut store, &budget(1)).expect("turn 1 assembles");
    assert_eq!(assembled_t1.context_refs, vec!["psnap-delta-1".to_owned()]);
    assert_eq!(assembled_t1.records.len(), 1);
    assert!(assembled_t1.records[0].is_untrusted_surface);
    assert!(assembled_t1.omitted_ids.is_empty());
    assert!(assembled_t1.pruned_ids.is_empty());

    // Turn 2 (gen 1, delta): the host adds a trusted delta alongside the
    // pinned snapshot; both assemble at the pinned generation.
    let delta_gen1 = delta_record("delta-foo-rs-1", 1, 1_100);
    delta_gen1.validate().expect("delta record validates");
    assert!(!delta_gen1.is_untrusted_surface);

    let assembled_t2 = assemble(&[r1.clone(), delta_gen1.clone()], &mut store, &budget(1))
        .expect("turn 2 assembles");
    assert_eq!(
        assembled_t2.context_refs,
        vec!["psnap-delta-1".to_owned(), "delta-foo-rs-1".to_owned()],
        "snapshot pinned first, delta alongside second (caller order)"
    );
    assert_eq!(assembled_t2.records.len(), 2);
    assert_eq!(assembled_t2.records[0].id, r1.id);
    assert!(assembled_t2.records[0].is_untrusted_surface);
    assert_eq!(assembled_t2.records[1].provider, "diagnostics");
    assert!(!assembled_t2.records[1].is_untrusted_surface);
    assert!(assembled_t2.omitted_ids.is_empty());
    assert!(assembled_t2.pruned_ids.is_empty());

    // Turn 3 (refresh, gen 2): re-ingest refreshed bytes under a new
    // authorization; the pin rotates to generation 2.
    let bytes_gen2 = snapshot_bytes("abc124");
    assert_ne!(bytes_gen2, bytes_gen1, "refresh carries new content");
    let digest_gen2 = snapshot_digest_hex(&bytes_gen2);
    let r2 = ingest_snapshot(
        &ingest_request(&bytes_gen2, &digest_gen2, "psnap-delta-2", 2, 2_000),
        &mut store,
    )
    .expect("gen-2 refresh ingests");
    assert_eq!(r2.provider, "project");
    assert!(r2.is_untrusted_surface);
    assert_eq!(r2.generation, 2);
    assert_eq!(r2.supersedes, None);
    assert_ne!(r1, r2, "refresh rotates the pin to a new record");

    // The delta is re-collected at the current generation each turn.
    let delta_gen2 = delta_record("delta-foo-rs-2", 2, 2_100);
    let assembled_t3 = assemble(&[r2.clone(), delta_gen2.clone()], &mut store, &budget(2))
        .expect("turn 3 assembles");
    assert_eq!(
        assembled_t3.context_refs,
        vec!["psnap-delta-2".to_owned(), "delta-foo-rs-2".to_owned()]
    );
    assert_eq!(assembled_t3.records.len(), 2);
    assert!(assembled_t3.omitted_ids.is_empty());
    assert!(assembled_t3.pruned_ids.is_empty());

    // Money assertion: the gen-1 pin is invalidated by the refresh. A stale
    // snapshot never leaks into a gen-2 turn, alone or beside current data.
    let stale_alone = assemble(std::slice::from_ref(&r1), &mut store, &budget(2))
        .expect_err("stale snapshot alone must fail closed");
    assert_eq!(
        stale_alone,
        ContextError::StaleGeneration {
            id: "psnap-delta-1".to_owned(),
            actual: 1,
            current: 2,
        }
    );
    let stale_beside_current = assemble(&[r1.clone(), delta_gen2.clone()], &mut store, &budget(2))
        .expect_err("stale snapshot beside a current delta must fail closed");
    assert_eq!(
        stale_beside_current,
        ContextError::StaleGeneration {
            id: "psnap-delta-1".to_owned(),
            actual: 1,
            current: 2,
        }
    );
    // Symmetric discipline for the delta: a gen-1 delta is stale at gen 2.
    let stale_delta = assemble(&[r2.clone(), delta_gen1.clone()], &mut store, &budget(2))
        .expect_err("stale delta must fail closed");
    assert_eq!(
        stale_delta,
        ContextError::StaleGeneration {
            id: "delta-foo-rs-1".to_owned(),
            actual: 1,
            current: 2,
        }
    );

    // All bodies stayed inline (small synthetic bytes), so the session store
    // is untouched end to end.
    assert!(store.is_empty());
}

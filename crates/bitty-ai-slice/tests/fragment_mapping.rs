//! BII-07 fragment mapping conformance: runtime stream fragments onto the
//! `bitty-ipc` `rich_fragment` text+zone contract.
//!
//! This suite proves what fits and documents what does not, without inventing
//! a wire method or claiming live/render behavior:
//!
//! - Direction basis is the draft handoff input
//!   `bitty-side-integration-input.md` BII-07 (bounded rich scene-fragment
//!   transport for Markdown/diff/tool-card fragments) plus the read-only
//!   `bitty` contract `crates/bitty-ipc/src/rich_fragment.rs`
//!   (`FragmentData`/`RichFragment`/`FragmentIngestService`, 16 KiB text bound,
//!   64-deep queue, always-`true` untrusted label) and `scope.rs` (no
//!   `rich.*` wire method registered).
//! - Runtime side is `bitty-ai-runtime/src/stream.rs` (untouched by this
//!   task): `FragmentKind::{Markdown,Diff,ToolCard}`, `Fragment{kind,bytes}`,
//!   `StreamChunk{seq: u32,total,is_final,fragment}`, `validate_chunk`,
//!   `fragment_text`, `emit_fragments`, `VecSink`. The runtime bound is
//!   `MAX_FRAGMENT_BYTES` (64 KiB); the transport bound is
//!   `MAX_FRAGMENT_TEXT_BYTES` (16 KiB).
//! - Mapping under test is test-only: validated runtime bytes that are valid
//!   UTF-8 become `FragmentData{text,zone}` with the caller-supplied
//!   `(terminal_id,generation,seq)` key and a reused snapshot `ZoneKind`
//!   (`None` leaves zoning to projection). `FragmentIngestService::ingest` is
//!   the end-to-end proof that the real service accepts our fragments.
//! - Headless mapping proof only: no socket, process, PTY, network, wall
//!   clock, secret, or render/projection step. Drained `RichFragment`s are
//!   asserted as DTOs; no test claims pixels, `RichBlock`s, or live display.
//! - `bitty-ipc` is the pinned Git revision `2cbb1fb` (AI-0039 verified
//!   `64e1709..2cbb1fb` additive: `rich_fragment.rs` byte-identical;
//!   `scope.rs` contains no `rich` method in either). The pin bump is
//!   mechanics-only in this task.
//! - Typed gaps (kind erasure, render wiring, wire method, consent and
//!   provider-echo) are asserted as absence where checkable and otherwise
//!   recorded as gap input for the `bitty` track; there is no bypass path.

use bitty_ai_runtime::{
    Fragment, FragmentKind, StreamChunk, StreamSink, VecSink, emit_fragments, fragment_text,
    validate_chunk,
};
use bitty_ipc::error::IpcError;
use bitty_ipc::rich_fragment::{
    FragmentData, FragmentIngestService, MAX_FRAGMENT_TEXT_BYTES, MAX_PENDING_FRAGMENTS,
};
use bitty_ipc::scope::required_scope_for_method;
use bitty_ipc::snapshot::ZoneKind;

const TERMINAL_ID: &str = "t:1";
const GENERATION: u64 = 7;

/// Map one validated runtime chunk onto the transport pre-bound input.
///
/// The runtime carries no zone: the caller attaches a reused snapshot
/// `ZoneKind` (or `None`). Non-UTF-8 bytes cannot become `text` without loss
/// and fail mapping closed; NUL is left for the service to reject as
/// `InvalidRequest` so the authoritative fail-closed path is exercised.
fn map_chunk_to_fragment_data(
    chunk: &StreamChunk,
    terminal_id: &str,
    generation: u64,
    zone: Option<ZoneKind>,
) -> Result<FragmentData, String> {
    validate_chunk(chunk).map_err(|err| err.to_string())?;
    let text = String::from_utf8(chunk.fragment.bytes.clone()).map_err(|_| {
        "fragment bytes are not valid UTF-8: cannot map to rich text without loss".to_owned()
    })?;
    Ok(FragmentData {
        terminal_id: terminal_id.to_owned(),
        generation,
        seq: u64::from(chunk.seq),
        zone,
        text,
    })
}

fn chunk(seq: u32, total: u32, fragment: Fragment) -> StreamChunk {
    StreamChunk {
        seq,
        total,
        is_final: seq + 1 == total,
        fragment,
    }
}

fn markdown_chunk(seq: u32, total: u32, text: &str) -> StreamChunk {
    chunk(seq, total, Fragment::markdown(text.as_bytes().to_vec()))
}

// ── text content ─────────────────────────────────────────────────────────────

#[test]
fn markdown_text_fits_verbatim_when_in_budget() {
    let chunk = markdown_chunk(0, 1, "hello");
    let data = map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
        .expect("in-budget UTF-8 maps");
    let mut service = FragmentIngestService::new();
    let stored = service
        .ingest(data)
        .expect("real service accepts mapped fragment");
    assert_eq!(stored.text, "hello");
    assert!(!stored.truncated);
    assert_eq!(stored.zone, Some(ZoneKind::Output));
    assert_eq!(stored.seq, 0);
    assert_eq!(stored.terminal_id, TERMINAL_ID);
    assert_eq!(stored.generation, GENERATION);
    assert!(stored.is_untrusted_surface);
    stored.validate().expect("stored DTO validates");
    assert_eq!(service.len(), 1);
}

#[test]
fn diff_and_toolcard_text_fit_the_same_text_contract() {
    // The transport is text-only: three runtime kinds with identical text map
    // to identical DTO shapes. Kind preservation is a typed-fragment gap for
    // the `bitty` track (see gap notes); text carriage itself fits.
    let kinds = [
        FragmentKind::Markdown,
        FragmentKind::Diff,
        FragmentKind::ToolCard,
    ];
    let mut service = FragmentIngestService::new();
    let mut stored_texts = Vec::new();
    for (index, kind) in kinds.iter().enumerate() {
        let fragment = Fragment {
            kind: *kind,
            bytes: b"same text".to_vec(),
        };
        let chunk = chunk(index as u32, kinds.len() as u32, fragment);
        let data =
            map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
                .expect("each kind maps when bytes are in-budget UTF-8");
        let stored = service
            .ingest(data)
            .expect("service accepts each kind as text");
        assert_eq!(stored.text, "same text");
        assert!(!stored.truncated);
        stored_texts.push(stored.text.clone());
    }
    assert_eq!(stored_texts, vec!["same text", "same text", "same text"]);
    assert_eq!(service.len(), 3);
    // Kind erasure proof: DTOs differ only by key, never by kind.
    let drained = service.drain_bounded(8);
    assert_eq!(drained.len(), 3);
    for fragment in &drained {
        assert_eq!(fragment.text, "same text");
    }
}

// ── zone reuse ───────────────────────────────────────────────────────────────

#[test]
fn zone_reuse_round_trips_output_and_none() {
    let zoned = map_chunk_to_fragment_data(
        &markdown_chunk(0, 2, "zoned"),
        TERMINAL_ID,
        GENERATION,
        Some(ZoneKind::Output),
    )
    .expect("map zoned");
    let unzoned = map_chunk_to_fragment_data(
        &markdown_chunk(1, 2, "unzoned"),
        TERMINAL_ID,
        GENERATION,
        None,
    )
    .expect("map unzoned");
    let mut service = FragmentIngestService::new();
    let stored_zoned = service.ingest(zoned).expect("zoned ingests");
    let stored_unzoned = service.ingest(unzoned).expect("unzoned ingests");
    // `ZoneKind` reuses the snapshot vocabulary; no second zone enum exists.
    assert_eq!(stored_zoned.zone, Some(ZoneKind::Output));
    assert_eq!(stored_unzoned.zone, None);
    stored_zoned.validate().expect("zoned validates");
    stored_unzoned.validate().expect("unzoned validates");
}

// ── bound / truncation ───────────────────────────────────────────────────────

#[test]
fn over_budget_text_truncates_at_char_boundary_with_flag() {
    let text = "é".repeat(20 * 1024);
    assert!(text.len() > MAX_FRAGMENT_TEXT_BYTES);
    let chunk = StreamChunk {
        seq: 0,
        total: 1,
        is_final: true,
        fragment: Fragment::markdown(text.as_bytes().to_vec()),
    };
    // Runtime validation still passes here only when the producer text fits
    // the runtime 64 KiB fragment bound; the transport 16 KiB ceiling is
    // enforced by the service with truncation, never rejection.
    assert!(text.len() <= bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES);
    let data = map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
        .expect("over-transport text still maps; service bounds it");
    let mut service = FragmentIngestService::new();
    let stored = service
        .ingest(data)
        .expect("over-budget truncates, never rejects");
    assert!(stored.text.len() <= MAX_FRAGMENT_TEXT_BYTES);
    assert!(stored.truncated);
    assert!(text.starts_with(stored.text.as_str()));
    assert!(stored.is_untrusted_surface);
    stored.validate().expect("truncated DTO validates");
}

#[test]
fn runtime_64kib_fragment_does_not_fit_verbatim() {
    // Bound mismatch input for the `bitty` track: a full-size runtime
    // fragment (64 KiB) truncates to the 16 KiB transport ceiling with
    // `truncated = true`. Verbatim carriage requires re-splitting runtime
    // output to the transport ceiling before ingest; truncation alone loses
    // bytes by construction.
    let bytes = vec![b'x'; bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES];
    assert!(bytes.len() > MAX_FRAGMENT_TEXT_BYTES);
    let chunk = chunk(0, 1, Fragment::markdown(bytes));
    let data = map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
        .expect("maps; service enforces transport ceiling");
    let mut service = FragmentIngestService::new();
    let stored = service.ingest(data).expect("truncates");
    assert_eq!(stored.text.len(), MAX_FRAGMENT_TEXT_BYTES);
    assert!(stored.truncated);
}

// ── untrusted marking ────────────────────────────────────────────────────────

#[test]
fn ingested_fragments_are_always_untrusted() {
    let data = map_chunk_to_fragment_data(
        &markdown_chunk(0, 1, "hi"),
        TERMINAL_ID,
        GENERATION,
        Some(ZoneKind::Output),
    )
    .expect("map");
    let mut service = FragmentIngestService::new();
    let stored = service.ingest(data).expect("ingest");
    assert!(stored.is_untrusted_surface);
    assert!(stored.is_untrusted_surface());
    // The label cannot be cleared: a DTO with the flag off fails validation.
    let mut cleared = stored;
    cleared.is_untrusted_surface = false;
    assert!(
        matches!(cleared.validate(), Err(IpcError::InvalidRequest { .. })),
        "cleared trust label must fail closed"
    );
}

// ── seq / total ──────────────────────────────────────────────────────────────

#[test]
fn seq_total_framing_survives_mapping_in_fifo_order() {
    let mut service = FragmentIngestService::new();
    let total = 3_u32;
    for seq in 0..total {
        let text = format!("chunk-{seq}");
        let chunk = markdown_chunk(seq, total, &text);
        validate_chunk(&chunk).expect("runtime framing holds");
        let data =
            map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
                .expect("map");
        let stored = service.ingest(data).expect("ingest in order");
        assert_eq!(stored.seq, u64::from(seq));
        assert_eq!(stored.text, text);
    }
    assert_eq!(service.len(), total as usize);
    assert!(service.contains(TERMINAL_ID, GENERATION, 0));
    assert!(service.contains(TERMINAL_ID, GENERATION, 2));
    assert!(!service.contains(TERMINAL_ID, GENERATION, 3));
    let drained = service.drain_bounded(8);
    assert_eq!(drained.len(), total as usize);
    for (index, fragment) in drained.iter().enumerate() {
        assert_eq!(fragment.seq, index as u64);
        assert_eq!(fragment.text, format!("chunk-{index}"));
    }
    assert!(service.is_empty());
}

#[test]
fn duplicate_seq_is_rejected_before_capacity() {
    let mut service = FragmentIngestService::new();
    let first = map_chunk_to_fragment_data(
        &markdown_chunk(0, 1, "first"),
        TERMINAL_ID,
        GENERATION,
        Some(ZoneKind::Output),
    )
    .expect("map first");
    service.ingest(first).expect("first ingests");
    let replay = map_chunk_to_fragment_data(
        &markdown_chunk(0, 1, "replay"),
        TERMINAL_ID,
        GENERATION,
        Some(ZoneKind::Output),
    )
    .expect("replay maps; service rejects the duplicate key");
    let err = service
        .ingest(replay)
        .expect_err("duplicate key must fail closed");
    assert!(
        matches!(err, IpcError::InvalidRequest { .. }),
        "got {err:?}"
    );
    assert_eq!(service.len(), 1);
    assert_eq!(MAX_PENDING_FRAGMENTS, 64);
}

#[test]
fn bad_terminal_grammar_is_rejected_fail_closed() {
    let chunk = markdown_chunk(0, 1, "hi");
    let mut service = FragmentIngestService::new();
    for bad in ["x:1", "t:01"] {
        let data = map_chunk_to_fragment_data(&chunk, bad, GENERATION, Some(ZoneKind::Output))
            .expect("mapping keeps the id verbatim for the service to judge");
        assert!(service.ingest(data).is_err(), "id {bad} must be rejected");
    }
    assert!(service.is_empty(), "refused ingest stores nothing");
}

// ── end-to-end through the real service ──────────────────────────────────────

#[test]
fn streamed_turn_ingests_end_to_end_through_real_service() {
    let answer = "The last command printed `hello` with a zero exit status.";
    let fragments = fragment_text(FragmentKind::Markdown, answer);
    assert!(!fragments.is_empty());
    let mut sink = VecSink::new();
    let done =
        emit_fragments(&mut sink, &fragments, &|| false).expect("in-budget turn emits cleanly");
    assert!(done);
    assert_eq!(sink.len(), fragments.len());

    let mut service = FragmentIngestService::new();
    for chunk in sink.chunks() {
        validate_chunk(chunk).expect("emitted chunk validates");
        let data =
            map_chunk_to_fragment_data(chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
                .expect("emitted chunk maps");
        service
            .ingest(data)
            .expect("real service accepts mapped chunk");
    }
    assert_eq!(service.len(), sink.len());
    let drained = service.drain_bounded(64);
    assert_eq!(drained.len(), sink.len());
    let joined: String = drained
        .iter()
        .map(|fragment| fragment.text.clone())
        .collect();
    assert_eq!(joined, answer);
    for fragment in &drained {
        assert!(!fragment.truncated);
        fragment.validate().expect("each DTO validates");
    }
}

// ── fail-closed mapping edges ────────────────────────────────────────────────

#[test]
fn nul_bytes_are_rejected_by_the_service_fail_closed() {
    let chunk = chunk(0, 1, Fragment::markdown(b"ab\0cd".to_vec()));
    // The runtime stream contract does not forbid NUL; the transport does.
    validate_chunk(&chunk).expect("runtime framing passes; NUL is a transport refusal");
    let data = map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
        .expect("mapping preserves bytes for the service to judge");
    let mut service = FragmentIngestService::new();
    let err = service.ingest(data).expect_err("NUL must fail closed");
    assert!(
        matches!(err, IpcError::InvalidRequest { .. }),
        "got {err:?}"
    );
    assert!(service.is_empty(), "refused ingest stores nothing");
}

#[test]
fn non_utf8_bytes_cannot_map_without_loss() {
    let chunk = chunk(0, 1, Fragment::markdown(vec![0xff, 0xfe, b'a']));
    validate_chunk(&chunk).expect("runtime framing passes for opaque bytes");
    let err = map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
        .expect_err("non-UTF-8 cannot become transport text");
    assert!(err.contains("not valid UTF-8"), "got {err}");
}

// ── absence proofs (no invented wire method) ─────────────────────────────────

#[test]
fn no_rich_wire_method_is_registered() {
    // `rich_fragment.rs` states no wire method is registered (`scope.rs`
    // unchanged). These candidate names must stay unknown so no test can
    // accidentally route fragments through a fabricated method.
    for method in [
        "rich.publish",
        "rich.ingest",
        "rich_fragment.ingest",
        "terminal.rich_fragment",
        "rich.get",
    ] {
        assert_eq!(
            required_scope_for_method(method),
            None,
            "method {method} must stay unregistered"
        );
    }
}

#[test]
fn ingest_takes_no_consent_or_scope_proof_by_construction() {
    // `FragmentIngestService::ingest` takes only `FragmentData`: there is no
    // `granted: &ScopeSet`, no `ConsentLedger`, and no provider-echo
    // parameter. Ingest therefore succeeds with no consent object in scope;
    // authorization, consent, and provider echo are sequel work for the
    // `bitty` track, not something this suite works around.
    let data = map_chunk_to_fragment_data(
        &markdown_chunk(0, 1, "no consent objects in scope"),
        TERMINAL_ID,
        GENERATION,
        Some(ZoneKind::Output),
    )
    .expect("map");
    let mut service = FragmentIngestService::new();
    let stored = service
        .ingest(data)
        .expect("headless ingest needs no consent");
    assert!(stored.is_untrusted_surface);
}

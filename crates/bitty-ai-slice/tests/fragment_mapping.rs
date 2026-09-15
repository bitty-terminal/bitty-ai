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
//! - Runtime side is `bitty-ai-runtime/src/stream.rs` (AI-0058): `FragmentKind`
//!   `{Markdown,Diff,ToolCard}`, `Fragment{kind,bytes}`,
//!   `StreamChunk{seq: u32,total,is_final,fragment}`, `validate_chunk`,
//!   `fragment_text`, `emit_fragments`, `VecSink`. The runtime numbers `seq`
//!   continuously across the emission batches of one logical turn (`S-8`
//!   scheme A, `P1-5`), so the transport `seq` is the runtime `seq` projected
//!   directly and needs no renumbering; `total` is the running water mark and
//!   `is_final` closes each emission batch. The runtime bound is
//!   `MAX_FRAGMENT_BYTES` (64 KiB); the transport bound is
//!   `MAX_FRAGMENT_TEXT_BYTES` (16 KiB).
//! - Mapping under test: validated runtime bytes that are valid UTF-8 become
//!   `FragmentData{text,zone}` with the caller-supplied
//!   `(terminal_id,generation,seq)` key and a reused snapshot `ZoneKind`
//!   (`None` leaves zoning to projection). `FragmentIngestService::ingest` is
//!   the end-to-end proof that the real service accepts our fragments.
//!   Since AI-0066 the pre-split rule lives in production slice code
//!   (`bitty_ai_slice::fragment_transport`): a fragment larger than the 16 KiB
//!   transport ceiling is cut at code-point boundaries into parts with a
//!   continuation marker and dense `seq`, so a full 64 KiB block reassembles
//!   byte-identically instead of truncating. The counterfactual direct
//!   projection (no pre-split) still loses bytes and is asserted as such.
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

use bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES;
use bitty_ai_runtime::{
    Fragment, FragmentKind, StreamChunk, StreamSink, VecSink, emit_fragments, fragment_text,
    validate_chunk,
};
use bitty_ai_slice::fragment_transport::{
    FragmentIdentity, FragmentTransportCursor, FragmentTransportError, TransportPart,
    pre_split_chunk, pre_split_fragment, reassemble, reassemble_expected,
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
    // Bound mismatch input: a full-size runtime fragment (64 KiB) truncates to
    // the 16 KiB transport ceiling with `truncated = true` when projected
    // verbatim. AI-0066 removes the loss with the `fragment_transport`
    // pre-split rule (proven below); this test keeps the unsplit counterfactual
    // so the byte loss stays visible.
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
        emit_fragments(&mut sink, &fragments, 0, &|| false).expect("in-budget turn emits cleanly");
    assert!(done.is_some());
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

#[test]
fn continuous_turn_seq_yields_distinct_transport_keys() {
    // P1-5 / S-8 scheme A: three emission batches of one logical turn
    // (draft text, tool card, final text) number continuously, so the direct
    // projection keeps the transport dedup key unique without renumbering.
    // The pre-fix runtime restarted at `seq 0` per batch and the second
    // ingest failed as a duplicate key.
    let mut sink = VecSink::new();
    let draft = fragment_text(FragmentKind::Markdown, "checking");
    let after_draft = emit_fragments(&mut sink, &draft, 0, &|| false)
        .expect("draft batch emits")
        .expect("draft batch completes");
    let card = fragment_text(FragmentKind::ToolCard, "tool=terminal_read_zone status=ok");
    let after_card = emit_fragments(&mut sink, &card, after_draft, &|| false)
        .expect("card batch emits")
        .expect("card batch completes");
    let answer = fragment_text(FragmentKind::Markdown, "the output was hello");
    let after_answer = emit_fragments(&mut sink, &answer, after_card, &|| false)
        .expect("answer batch emits")
        .expect("answer batch completes");
    assert_eq!((after_draft, after_card, after_answer), (1, 2, 3));
    assert_eq!(sink.len(), 3);
    // Single-fragment batches: seq/total advance as a running water mark and
    // every batch-closing chunk carries `is_final`.
    let framing: Vec<(u32, u32, bool)> = sink
        .chunks()
        .iter()
        .map(|chunk| (chunk.seq, chunk.total, chunk.is_final))
        .collect();
    assert_eq!(framing, vec![(0, 1, true), (1, 2, true), (2, 3, true)]);

    let mut service = FragmentIngestService::new();
    for chunk in sink.chunks() {
        let data =
            map_chunk_to_fragment_data(chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
                .expect("continuous chunk maps");
        service
            .ingest(data)
            .expect("unique transport key across emission batches");
    }
    assert_eq!(service.len(), 3);
    assert!(service.contains(TERMINAL_ID, GENERATION, 0));
    assert!(service.contains(TERMINAL_ID, GENERATION, 1));
    assert!(service.contains(TERMINAL_ID, GENERATION, 2));
    let drained = service.drain_bounded(8);
    assert_eq!(
        drained
            .iter()
            .map(|fragment| fragment.seq)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(drained[2].text, "the output was hello");
    assert!(drained.iter().all(|fragment| !fragment.truncated));
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

// ── AI-0066 pre-split: no byte loss across the 16 KiB ceiling ────────────────

/// A multi-byte block just under the 64 KiB runtime fragment bound: 21845
/// three-byte code points = 65535 bytes. The 16 KiB cut falls mid-code-point
/// (16384 is not a multiple of 3), so a byte-wise split would corrupt UTF-8;
/// the pre-split rule must back off to a code-point boundary.
fn euro_block() -> String {
    "€".repeat(21845)
}

#[test]
fn pre_split_64kib_multibyte_reassembles_byte_identical() {
    let text = euro_block();
    let bytes = text.as_bytes();
    assert!(bytes.len() <= bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES);
    assert!(bytes.len() > MAX_FRAGMENT_TEXT_BYTES);
    let chunk = chunk(0, 1, Fragment::markdown(bytes.to_vec()));
    validate_chunk(&chunk).expect("the runtime accepts a full 64 KiB-scale fragment");

    let mut cursor = FragmentTransportCursor::new(TERMINAL_ID, GENERATION, 0);
    let parts = cursor
        .map_chunk(&chunk, Some(ZoneKind::Output))
        .expect("pre-split maps UTF-8 without loss");
    assert!(parts.len() > 1, "a 64 KiB fragment must split");
    assert_eq!(
        cursor.next_seq(),
        parts.len() as u64,
        "the cursor advances by the part count"
    );

    let mut offset = 0_usize;
    for (index, part) in parts.iter().enumerate() {
        assert!(
            part.data.text.len() <= MAX_FRAGMENT_TEXT_BYTES,
            "every part fits the 16 KiB transport ceiling"
        );
        offset += part.data.text.len();
        assert!(
            text.is_char_boundary(offset),
            "part {index} must not split a code point"
        );
        assert_eq!(part.part_index, index as u32);
        assert_eq!(part.part_count, parts.len() as u32);
        assert_eq!(part.is_continuation, index > 0);
        assert_eq!(part.data.seq, index as u64);
        assert_eq!(part.source_seq, 0);
    }
    assert_eq!(offset, text.len(), "the parts cover every source byte");

    // The real ingest service accepts every part without truncating, and the
    // ordered concatenation is byte-identical to the source.
    let mut service = FragmentIngestService::new();
    let mut stored_texts = Vec::new();
    for part in &parts {
        let stored = service
            .ingest(part.data.clone())
            .expect("a pre-split part ingests");
        assert!(!stored.truncated, "a pre-split part must never truncate");
        stored.validate().expect("stored DTO validates");
        stored_texts.push(stored.text);
    }
    assert_eq!(stored_texts.concat(), text);
    let reassembled = reassemble(&parts).expect("well-formed parts reassemble");
    assert_eq!(reassembled, text);
    assert_eq!(reassembled.as_bytes(), bytes);
}

#[test]
fn direct_projection_of_a_full_fragment_loses_bytes() {
    // Counterfactual: with no pre-split, the same 64 KiB multi-byte block
    // truncates at the 16 KiB ceiling and every trailing byte is lost. This is
    // the byte loss `pre_split_64kib_multibyte_reassembles_byte_identical`
    // removes.
    let text = euro_block();
    let chunk = chunk(0, 1, Fragment::markdown(text.as_bytes().to_vec()));
    let data = map_chunk_to_fragment_data(&chunk, TERMINAL_ID, GENERATION, Some(ZoneKind::Output))
        .expect("maps; the service bounds it");
    let mut service = FragmentIngestService::new();
    let stored = service.ingest(data).expect("over-budget text truncates");
    assert!(stored.truncated);
    assert!(stored.text.len() <= MAX_FRAGMENT_TEXT_BYTES);
    assert!(stored.text.len() < text.len());
    assert!(text.starts_with(&stored.text));
    assert_ne!(stored.text, text, "direct projection loses trailing bytes");
}

#[test]
fn cursor_assigns_dense_seq_across_a_turn() {
    // A split fragment followed by another fragment must not collide under the
    // transport dedup key: the cursor advances by the part count, and a new
    // source fragment is not a continuation.
    let text = euro_block();
    let fragments = vec![
        Fragment::markdown(text.as_bytes().to_vec()),
        Fragment::markdown(b"tail".to_vec()),
    ];
    let mut sink = VecSink::new();
    let done = emit_fragments(&mut sink, &fragments, 0, &|| false).expect("turn emits cleanly");
    assert!(done.is_some());
    assert_eq!(sink.len(), 2);

    let mut cursor = FragmentTransportCursor::new(TERMINAL_ID, GENERATION, 0);
    let first = cursor
        .map_chunk(&sink.chunks()[0], Some(ZoneKind::Output))
        .expect("first chunk maps");
    let second = cursor
        .map_chunk(&sink.chunks()[1], Some(ZoneKind::Output))
        .expect("second chunk maps");
    assert!(first.len() > 1);
    let seqs: Vec<u64> = first
        .iter()
        .chain(second.iter())
        .map(|part| part.data.seq)
        .collect();
    let expected: Vec<u64> = (0..seqs.len() as u64).collect();
    assert_eq!(seqs, expected, "transport seq stays dense across the turn");
    assert!(
        !second[0].is_continuation,
        "a new source fragment starts fresh"
    );
    assert_eq!(second[0].part_count, 1);

    let mut service = FragmentIngestService::new();
    for part in first.iter().chain(second.iter()) {
        service
            .ingest(part.data.clone())
            .expect("dense keys avoid the duplicate-key refusal");
    }
    assert_eq!(service.len(), seqs.len());
}

#[test]
fn reassembly_rejects_misordered_parts() {
    let chunk = markdown_chunk(0, 1, &"é".repeat(20 * 1024));
    let parts = pre_split_chunk(&chunk, TERMINAL_ID, GENERATION, None, 0).expect("split");
    assert!(parts.len() > 1);
    let mut swapped = parts.clone();
    swapped.swap(0, 1);
    assert!(matches!(
        reassemble(&swapped),
        Err(FragmentTransportError::PartOrder { .. })
    ));
}

#[test]
fn pre_split_does_not_bypass_the_runtime_fragment_bound() {
    // The pre-split runs after `validate_chunk`: an over-64 KiB runtime
    // fragment is rejected, never silently accepted as many small parts.
    let oversized = chunk(
        0,
        1,
        Fragment::markdown(vec![b'x'; bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES + 1]),
    );
    assert!(matches!(
        pre_split_chunk(&oversized, TERMINAL_ID, GENERATION, None, 0),
        Err(FragmentTransportError::Runtime(_))
    ));
}

#[test]
fn pre_split_fails_closed_on_non_utf8() {
    let chunk = chunk(0, 1, Fragment::markdown(vec![0xff, 0xfe, b'a']));
    assert!(matches!(
        pre_split_chunk(&chunk, TERMINAL_ID, GENERATION, None, 0),
        Err(FragmentTransportError::NonUtf8)
    ));
}

// ── AI-0069 reassembly identity + fragment-level split bound ─────────────────

/// A multi-part split used by the identity negatives below: the 16 KiB cut
/// lands on a code-point boundary and yields more than one part, so mutating
/// one part is a real cross-part disagreement.
fn split_parts() -> Vec<TransportPart> {
    let chunk = markdown_chunk(0, 1, &"é".repeat(20 * 1024));
    let parts = pre_split_chunk(&chunk, TERMINAL_ID, GENERATION, None, 0).expect("split");
    assert!(parts.len() > 1, "the fixture must yield several parts");
    parts
}

#[test]
fn reassemble_refuses_a_foreign_terminal_part() {
    // A part from a different terminal with a matching index/count must not be
    // silently reassembled into this fragment's bytes.
    let mut parts = split_parts();
    parts[1].data.terminal_id = "t:2".to_owned();
    assert!(matches!(
        reassemble(&parts),
        Err(FragmentTransportError::TerminalMismatch { index: 1, .. })
    ));
}

#[test]
fn reassemble_refuses_a_foreign_generation_part() {
    let mut parts = split_parts();
    parts[1].data.generation = GENERATION + 1;
    assert!(matches!(
        reassemble(&parts),
        Err(FragmentTransportError::GenerationMismatch { index: 1, .. })
    ));
}

#[test]
fn reassemble_refuses_a_foreign_source_seq_part() {
    // `source_seq` is the split-time identity of the source fragment: a part
    // that disagrees cannot belong to this reassembly even if its index/count
    // and transport key line up.
    let mut parts = split_parts();
    parts[1].source_seq += 1;
    assert!(matches!(
        reassemble(&parts),
        Err(FragmentTransportError::SourceSeqMismatch { index: 1, .. })
    ));
}

#[test]
fn reassemble_refuses_a_noncontiguous_part_seq() {
    // `data.seq` must be `first_seq + part_index`; a gapped transport seq is a
    // reordered or truncated part, not a contiguous split.
    let mut parts = split_parts();
    parts[1].data.seq += 100;
    assert!(matches!(
        reassemble(&parts),
        Err(FragmentTransportError::SequenceGap { index: 1, .. })
    ));
}

#[test]
fn reassemble_refuses_a_mismatched_part_count_on_a_later_part() {
    // The first part's `part_count` is checked against the input length; a later
    // part disagreeing is a different failure from that length check and must
    // fail closed with the dedicated variant.
    let mut parts = split_parts();
    let first_count = parts[0].part_count;
    let mutated_count = first_count + 1;
    parts[1].part_count = mutated_count;
    assert_eq!(
        reassemble(&parts).expect_err("a later part with a foreign part_count is refused"),
        FragmentTransportError::InconsistentPartCount {
            index: 1,
            expected: first_count,
            found: mutated_count,
        }
    );
}

// ── AI-0070 absolute identity binding + precise part-count diagnostics ───────

/// A second, internally consistent split built from a foreign source identity:
/// the same text split with a different `(terminal_id, generation, source_seq)`.
/// Every part agrees with the first, so the relative `reassemble` cannot tell it
/// apart from the original; only a caller-supplied expectation can.
fn foreign_split_parts() -> (Vec<TransportPart>, FragmentIdentity) {
    let chunk = markdown_chunk(1, 2, &"é".repeat(20 * 1024));
    let identity = FragmentIdentity::new("t:2", GENERATION + 1, 1);
    let parts = pre_split_chunk(&chunk, &identity.terminal_id, identity.generation, None, 0)
        .expect("the foreign fragment splits");
    assert!(parts.len() > 1, "the fixture must yield several parts");
    (parts, identity)
}

#[test]
fn reassemble_expected_rejects_a_foreign_but_consistent_part_set() {
    let (foreign, foreign_identity) = foreign_split_parts();
    let foreign_text: String = foreign.iter().map(|part| part.data.text.as_str()).collect();

    // The relative entry point accepts the internally consistent set, because
    // every part agrees with the first: this is the silent gap AI-0070 closes.
    assert_eq!(
        reassemble(&foreign).expect("relative reassembly accepts a consistent set"),
        foreign_text
    );

    // The absolute entry point refuses it: the set belongs to another source
    // fragment than the caller asked for.
    let expected = FragmentIdentity::new(TERMINAL_ID, GENERATION, 0);
    assert_eq!(
        reassemble_expected(&foreign, &expected)
            .expect_err("a foreign but consistent part set must be refused"),
        FragmentTransportError::IdentityMismatch {
            expected: expected.clone(),
            found: foreign_identity.clone(),
        }
    );
    assert!(
        reassemble_expected(&foreign, &expected)
            .expect_err("still refused")
            .to_string()
            .contains(&foreign_identity.terminal_id)
    );

    // The caller can still accept the set it actually asked for.
    assert_eq!(
        reassemble_expected(&foreign, &foreign_identity)
            .expect("the matching identity reassembles"),
        foreign_text
    );
}

#[test]
fn part_count_failure_modes_are_distinguishable() {
    let parts = split_parts();

    // Whole-input disagreement: more parts are supplied than the split recorded.
    let mut too_many = parts.clone();
    too_many.push(parts[0].clone());
    let length_error = reassemble(&too_many).expect_err("supplied length must match part_count");
    assert_eq!(
        length_error,
        FragmentTransportError::PartCountMismatch {
            expected: parts[0].part_count,
            actual: parts.len() + 1,
        }
    );

    // Later-part disagreement: one part records a different `part_count`.
    let mut inconsistent = parts.clone();
    let recorded = inconsistent[0].part_count;
    inconsistent[1].part_count = recorded + 1;
    let part_error = reassemble(&inconsistent).expect_err("later part_count must agree");
    assert_eq!(
        part_error,
        FragmentTransportError::InconsistentPartCount {
            index: 1,
            expected: recorded,
            found: recorded + 1,
        }
    );

    // The two typed errors, and therefore their messages, are unambiguous.
    assert_ne!(length_error, part_error);
    assert_ne!(length_error.to_string(), part_error.to_string());
}

#[test]
fn pre_split_fragment_refuses_an_oversized_raw_fragment() {
    // The fragment-level entry point must enforce the 64 KiB runtime bound
    // itself so a raw `Fragment` caller cannot split an oversized fragment.
    let fragment = Fragment::markdown(vec![b'x'; MAX_FRAGMENT_BYTES + 1]);
    let err = pre_split_fragment(&fragment, 0, TERMINAL_ID, GENERATION, None, 0)
        .expect_err("an over-64 KiB fragment must be refused, not split");
    assert_eq!(
        err,
        FragmentTransportError::OversizedFragment {
            actual: MAX_FRAGMENT_BYTES + 1,
            limit: MAX_FRAGMENT_BYTES,
        }
    );
}

#[test]
fn pre_split_fragment_accepts_exactly_the_runtime_bound() {
    // The bound is a ceiling, not a strict inequality: exactly one runtime
    // fragment (64 KiB) still splits and reassembles byte-identically.
    let fragment = Fragment::markdown(vec![b'x'; MAX_FRAGMENT_BYTES]);
    let parts = pre_split_fragment(&fragment, 0, TERMINAL_ID, GENERATION, None, 0)
        .expect("the exact runtime bound is admitted");
    assert!(parts.len() > 1, "64 KiB exceeds the 16 KiB part ceiling");
    assert_eq!(
        reassemble(&parts).expect("the split parts reassemble"),
        "x".repeat(MAX_FRAGMENT_BYTES)
    );
}

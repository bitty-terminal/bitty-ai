//! Minimal-envelope syntax fallback (AI-0097, AIQ-46 narrowing).
//!
//! DEC-0007: when structured disclosure fails because the reader cannot
//! parse the schema, fall back to one permanently-readable minimal envelope
//! (`id` + `kind` + `text`) — machine-parseable, consistent with the
//! existing typed-error style. This file pins the [`FallbackEnvelope`]
//! primitive (exactly three bounded `String` fields, byte-cap validation
//! fail-closed at construction), the total [`fallback_for`] constructor
//! (any unreadable input yields an envelope or a typed [`FallbackError`],
//! never a panic), and the deterministic byte-encoding round-trip.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `input_fingerprint.rs` (AI-0095) pins complete-input fingerprints with
//!   disable-on-unknown reuse. Unknown-input refusal appears here only as
//!   the contrast: the fallback never refuses unknown bytes, it envelopes
//!   them (or reports a bound violation).
//! - `cache_key.rs` (AI-0082) pins stable-prefix keying over canonical
//!   bytes. Length-aware framing appears here only as the envelope encoding
//!   rule (magic plus length-prefixed fields, never a marker scan).
//! - `result_schema_disclosure.rs` (AI-0077) pins the structured disclosure
//!   schema (typed `Display`, card words, bounded Unknown reasons). Typed
//!   errors and the 512-byte reason precedent appear here only as the style
//!   the fallback stays consistent with when structured parsing fails.
//!
//! Deterministic and offline: fixed byte vectors only, no turns, no clock,
//! no threads, no I/O, no `HashMap`, no `RandomState`.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! envelope content, input bytes, and digests never appear in message strings.

use bitty_ai_runtime::{
    FALLBACK_ID, FALLBACK_KIND, FallbackEnvelope, FallbackError, MAX_FALLBACK_ID_BYTES,
    MAX_FALLBACK_KIND_BYTES, MAX_FALLBACK_TEXT_BYTES, fallback_for,
};

/// Build a valid envelope for round-trip probes (test inputs always satisfy
/// the caps; over-cap inputs go through the bound tests below).
fn valid_envelope(id: String, kind: String, text: String) -> FallbackEnvelope {
    FallbackEnvelope::new(id, kind, text).expect("test envelope satisfies the byte caps")
}

#[test]
fn round_trip_preserves_all_three_fields_byte_exact() {
    let envelope = valid_envelope(
        "disclosure-7".to_owned(),
        "syntax-fallback".to_owned(),
        "héllo wörld ✓ plain".to_owned(),
    );
    let bytes = envelope.to_bytes();
    let decoded = FallbackEnvelope::from_bytes(&bytes).expect("own encoding must decode");
    assert!(
        decoded.id() == "disclosure-7",
        "round-trip must preserve id"
    );
    assert!(
        decoded.kind() == "syntax-fallback",
        "round-trip must preserve kind"
    );
    assert!(
        decoded.text() == "héllo wörld ✓ plain",
        "round-trip must preserve text byte-exact"
    );
    assert!(
        decoded == envelope,
        "round-trip must preserve the whole envelope"
    );
}

#[test]
fn boundary_lengths_pass_and_round_trip() {
    let envelope = valid_envelope(
        "i".repeat(MAX_FALLBACK_ID_BYTES),
        "k".repeat(MAX_FALLBACK_KIND_BYTES),
        "t".repeat(MAX_FALLBACK_TEXT_BYTES),
    );
    assert!(
        envelope.id().len() == MAX_FALLBACK_ID_BYTES,
        "id at exactly the cap must pass"
    );
    assert!(
        envelope.kind().len() == MAX_FALLBACK_KIND_BYTES,
        "kind at exactly the cap must pass"
    );
    assert!(
        envelope.text().len() == MAX_FALLBACK_TEXT_BYTES,
        "text at exactly the cap must pass"
    );
    let decoded =
        FallbackEnvelope::from_bytes(&envelope.to_bytes()).expect("boundary envelope must decode");
    assert!(
        decoded == envelope,
        "boundary envelope must round-trip whole"
    );
}

#[test]
fn over_cap_id_refuses_with_typed_error() {
    let big = "i".repeat(MAX_FALLBACK_ID_BYTES + 1);
    let Err(FallbackError::IdTooLarge { limit, actual }) =
        FallbackEnvelope::new(big, "kind".to_owned(), "text".to_owned())
    else {
        panic!("over-cap id must refuse");
    };
    assert!(
        limit == MAX_FALLBACK_ID_BYTES,
        "id bound must report the documented cap"
    );
    assert!(
        actual == MAX_FALLBACK_ID_BYTES + 1,
        "id bound must report observed bytes"
    );
}

#[test]
fn over_cap_kind_refuses_with_typed_error() {
    let big = "k".repeat(MAX_FALLBACK_KIND_BYTES + 1);
    let Err(FallbackError::KindTooLarge { limit, actual }) =
        FallbackEnvelope::new("id".to_owned(), big, "text".to_owned())
    else {
        panic!("over-cap kind must refuse");
    };
    assert!(
        limit == MAX_FALLBACK_KIND_BYTES,
        "kind bound must report the documented cap"
    );
    assert!(
        actual == MAX_FALLBACK_KIND_BYTES + 1,
        "kind bound must report observed bytes"
    );
}

#[test]
fn over_cap_text_refuses_with_typed_error() {
    let big = "t".repeat(MAX_FALLBACK_TEXT_BYTES + 1);
    let Err(FallbackError::TextTooLarge { limit, actual }) =
        FallbackEnvelope::new("id".to_owned(), "kind".to_owned(), big)
    else {
        panic!("over-cap text must refuse");
    };
    assert!(
        limit == MAX_FALLBACK_TEXT_BYTES,
        "text bound must report the documented cap"
    );
    assert!(
        actual == MAX_FALLBACK_TEXT_BYTES + 1,
        "text bound must report observed bytes"
    );
}

#[test]
fn typed_error_display_is_single_line() {
    let id_limit = MAX_FALLBACK_ID_BYTES;
    let kind_limit = MAX_FALLBACK_KIND_BYTES;
    let text_limit = MAX_FALLBACK_TEXT_BYTES;
    let cases: Vec<(FallbackError, String)> = vec![
        (
            FallbackError::IdTooLarge {
                limit: id_limit,
                actual: id_limit + 1,
            },
            format!(
                "fallback id of {} bytes exceeds {id_limit} byte limit",
                id_limit + 1
            ),
        ),
        (
            FallbackError::KindTooLarge {
                limit: kind_limit,
                actual: kind_limit + 1,
            },
            format!(
                "fallback kind of {} bytes exceeds {kind_limit} byte limit",
                kind_limit + 1
            ),
        ),
        (
            FallbackError::TextTooLarge {
                limit: text_limit,
                actual: text_limit + 1,
            },
            format!(
                "fallback text of {} bytes exceeds {text_limit} byte limit",
                text_limit + 1
            ),
        ),
        (
            FallbackError::Malformed,
            "malformed fallback envelope".to_owned(),
        ),
    ];
    assert!(cases.len() == 4, "every error variant must be pinned");
    for (error, expected) in &cases {
        let display = error.to_string();
        assert!(
            display == *expected,
            "display must pin the typed-error schema"
        );
        assert!(
            !display.contains(['\n', '\r']),
            "typed display must stay single-line"
        );
    }
}

#[test]
fn fallback_error_is_a_std_error() {
    fn assert_std_error(_: &dyn std::error::Error) {}
    let error = FallbackError::Malformed;
    assert_std_error(&error);
}

#[test]
fn fallback_for_is_total_over_adversarial_inputs() {
    let scrub_boundary = vec![0xFF; MAX_FALLBACK_TEXT_BYTES];
    let just_over = vec![b'x'; MAX_FALLBACK_TEXT_BYTES + 1];
    let massive = vec![0xFF; 1024 * 1024];
    let inputs: Vec<&[u8]> = vec![
        b"".as_slice(),
        b"plain readable disclosure".as_slice(),
        b"tab\tand newline\nand carriage\rreturn".as_slice(),
        b"\x1b[2Jescape sequence".as_slice(),
        [0xFF, 0xFE, 0x80, 0xC3].as_slice(),
        "héllo wörld ✓".as_bytes(),
        scrub_boundary.as_slice(),
        just_over.as_slice(),
        massive.as_slice(),
    ];
    assert!(inputs.len() == 9, "every adversarial shape must run");
    for input in inputs {
        match fallback_for(input) {
            Ok(envelope) => {
                assert!(
                    envelope.id() == FALLBACK_ID,
                    "fallback identity must be fixed"
                );
                assert!(
                    envelope.kind() == FALLBACK_KIND,
                    "fallback kind must be fixed"
                );
                assert!(
                    envelope.id().len() <= MAX_FALLBACK_ID_BYTES,
                    "emitted id must satisfy the cap"
                );
                assert!(
                    envelope.kind().len() <= MAX_FALLBACK_KIND_BYTES,
                    "emitted kind must satisfy the cap"
                );
                assert!(
                    envelope.text().len() <= MAX_FALLBACK_TEXT_BYTES,
                    "emitted text must satisfy the cap"
                );
                assert!(
                    envelope
                        .text()
                        .chars()
                        .all(|c| c.is_ascii_graphic() || c == ' '),
                    "emitted text must stay readable ASCII"
                );
            }
            Err(FallbackError::TextTooLarge { .. }) => {}
            Err(_) => panic!("fallback_for may only fail with the text bound"),
        }
    }
}

#[test]
fn fallback_for_maps_empty_input_to_empty_text_envelope() {
    let envelope = fallback_for(b"").expect("empty input must still envelope");
    assert!(
        envelope.id() == FALLBACK_ID,
        "empty input keeps the fixed identity"
    );
    assert!(
        envelope.kind() == FALLBACK_KIND,
        "empty input keeps the fixed kind"
    );
    assert!(
        envelope.text().is_empty(),
        "empty input carries no payload text"
    );
}

#[test]
fn fallback_for_scrubs_without_silent_truncation() {
    let envelope = fallback_for(b"hello world").expect("short readable input must envelope");
    assert!(
        envelope.text() == "hello world",
        "readable input must pass through unchanged"
    );
    let scrubbed = fallback_for(b"a\x00b\xff\n").expect("scrubbable input must envelope");
    assert!(
        scrubbed.text() == "a?b??",
        "unreadable bytes must scrub to placeholders"
    );
}

#[test]
fn fallback_for_refuses_over_cap_text_instead_of_truncating() {
    let big = "x".repeat(MAX_FALLBACK_TEXT_BYTES + 1);
    let Err(FallbackError::TextTooLarge { limit, actual }) = fallback_for(big.as_bytes()) else {
        panic!("over-cap payload must refuse, never truncate");
    };
    assert!(
        limit == MAX_FALLBACK_TEXT_BYTES,
        "payload bound must report the documented cap"
    );
    assert!(
        actual == MAX_FALLBACK_TEXT_BYTES + 1,
        "payload bound must report observed bytes"
    );
    let edge = "y".repeat(MAX_FALLBACK_TEXT_BYTES);
    let envelope = fallback_for(edge.as_bytes()).expect("boundary payload must envelope whole");
    assert!(
        envelope.text().len() == MAX_FALLBACK_TEXT_BYTES,
        "boundary payload must survive whole"
    );
}

/// Reference scrub for independent expected values: lossy-decode the input
/// and map every character outside printable ASCII to `?`. Small test inputs
/// only; the point is pinning exact scrub length semantics.
fn reference_scrub(unreadable: &[u8]) -> String {
    String::from_utf8_lossy(unreadable)
        .chars()
        .map(|character| {
            if character.is_ascii_graphic() || character == ' ' {
                character
            } else {
                '?'
            }
        })
        .collect()
}

#[test]
fn over_cap_refusal_reports_exact_scrubbed_length() {
    let unit = b"a\xC3\xA9\xE2\x9C\x93\x00\xFF\xF0";
    let input = unit.repeat(90);
    let expected = reference_scrub(&input);
    assert!(
        expected.len() > MAX_FALLBACK_TEXT_BYTES,
        "mixed-cap payload must exceed the scrubbed cap"
    );
    assert!(
        expected.len() < input.len(),
        "scrubbed length must differ from raw bytes"
    );
    let Err(FallbackError::TextTooLarge { limit, actual }) = fallback_for(&input) else {
        panic!("over-cap scrubbed payload must refuse");
    };
    assert!(
        limit == MAX_FALLBACK_TEXT_BYTES,
        "refusal must report the documented cap"
    );
    assert!(
        actual == expected.len(),
        "refusal must report the exact scrubbed length"
    );
    assert!(
        actual != input.len(),
        "refusal must not report the raw byte length"
    );
}

#[test]
fn scrubbed_to_cap_output_matches_the_reference_scrub() {
    let unit = b"a\xC3\xA9\xE2\x9C\x93\x00\xFF\xF0";
    let input = unit.repeat(50);
    let expected = reference_scrub(&input);
    assert!(
        expected.len() <= MAX_FALLBACK_TEXT_BYTES,
        "fixture must stay inside the scrubbed cap"
    );
    let envelope = fallback_for(&input).expect("in-cap scrubbed payload must envelope");
    assert!(
        envelope.text() == expected,
        "in-cap scrubbed text must match the reference scrub"
    );
}

#[test]
fn raw_bytes_over_cap_still_envelope_when_scrubbed_fits() {
    let text = "é".repeat(MAX_FALLBACK_TEXT_BYTES / 2 + 1);
    assert!(
        text.len() > MAX_FALLBACK_TEXT_BYTES,
        "fixture must exceed the cap in raw bytes"
    );
    let envelope =
        fallback_for(text.as_bytes()).expect("scrubbed payload inside the cap envelopes");
    assert!(
        envelope.text() == reference_scrub(text.as_bytes()),
        "multi-byte input must scrub to the reference text"
    );
    assert!(
        envelope.text().len() <= MAX_FALLBACK_TEXT_BYTES,
        "enveloped text must satisfy the scrubbed cap"
    );
}

#[test]
fn same_input_yields_same_envelope_and_bytes() {
    let input = b"deterministic disclosure \xff bytes";
    let first = fallback_for(input).expect("deterministic input must envelope");
    let second = fallback_for(input).expect("deterministic input must envelope");
    assert!(first == second, "same input must yield the same envelope");
    assert!(
        first.to_bytes() == second.to_bytes(),
        "same input must yield the same bytes"
    );
}

#[test]
fn from_bytes_rejects_garbage_fail_closed() {
    let valid = valid_envelope("id".to_owned(), "kind".to_owned(), "text".to_owned()).to_bytes();
    let mut truncated = valid.clone();
    truncated.pop();
    let mut bad_magic = valid.clone();
    bad_magic[0] ^= 0x01;
    let mut trailing = valid.clone();
    trailing.push(b'!');
    let mut overrun = Vec::new();
    overrun.extend_from_slice(b"fallback/1\n");
    overrun.extend_from_slice(&u64::MAX.to_le_bytes());
    let mut non_utf8 = Vec::new();
    non_utf8.extend_from_slice(b"fallback/1\n");
    non_utf8.extend_from_slice(&1u64.to_le_bytes());
    non_utf8.push(0xFF);
    non_utf8.extend_from_slice(&4u64.to_le_bytes());
    non_utf8.extend_from_slice(b"kind");
    non_utf8.extend_from_slice(&4u64.to_le_bytes());
    non_utf8.extend_from_slice(b"text");
    let cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"short".to_vec(),
        b"fallback/1\n".to_vec(),
        truncated,
        bad_magic,
        trailing,
        overrun,
        non_utf8,
    ];
    assert!(cases.len() == 8, "every garbage shape must run");
    for case in &cases {
        assert!(
            FallbackEnvelope::from_bytes(case).is_err(),
            "garbage must refuse, never panic"
        );
    }
}

#[test]
fn from_bytes_declared_over_cap_reports_the_field_bound() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"fallback/1\n");
    bytes.extend_from_slice(&(MAX_FALLBACK_ID_BYTES as u64 + 1).to_le_bytes());
    bytes.extend_from_slice(&[b'i'; MAX_FALLBACK_ID_BYTES + 1][..MAX_FALLBACK_ID_BYTES + 1]);
    let Err(FallbackError::IdTooLarge { limit, actual }) = FallbackEnvelope::from_bytes(&bytes)
    else {
        panic!("declared over-cap id must report the id bound");
    };
    assert!(
        limit == MAX_FALLBACK_ID_BYTES,
        "declared bound must report the documented cap"
    );
    assert!(
        actual == MAX_FALLBACK_ID_BYTES + 1,
        "declared bound must report declared bytes"
    );
}

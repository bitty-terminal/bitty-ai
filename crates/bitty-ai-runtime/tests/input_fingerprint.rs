//! Complete input fingerprints with disable-on-unknown reuse (AI-0095,
//! AIQ-43 narrowing).
//!
//! AIQ-43 is needs-evidence: R4 requires complete input fingerprints and
//! disables generic reuse when equivalence is unknown, but no fingerprint
//! mechanism is implemented. This file pins the [`InputFingerprint`]
//! primitive (FNV-1a-64 over the COMPLETE input, no truncation) and the
//! disable-on-unknown rule (unknown input refuses fail-closed with a typed
//! [`FingerprintError`]; no digest is emitted, so no reuse decision can
//! serve from cache).
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `cache_key.rs` (AI-0082) pins stable-prefix keying plus the hit-rate
//!   harness. Prefix addressing appears here only as the contrast: the
//!   fingerprint hashes the whole input, never a stable prefix.
//! - `cache_invalidation.rs` (AI-0078) pins stale-byte denial through
//!   re-assembly and artifact absence. Store-scoped absence appears here
//!   only as the unknown-candidate reuse refusal.
//! - `runtime_fail_closed.rs` pins fail-closed denials at the runtime
//!   boundary. Denial mechanics appear here only as the constructor-level
//!   unknown-input refusal shape.
//!
//! Deterministic and offline: fixed byte vectors only, no turns, no clock,
//! no threads, no I/O, no `HashMap`, no `RandomState`.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! input bytes and digests never appear in message strings.

use bitty_ai_runtime::{FingerprintError, InputFingerprint};

/// Deterministic FNV-1a-64, test-side re-computation of the fingerprint
/// digest. Duplicated (not imported) on purpose: the test pins the digest
/// value independently of the implementation under test.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Independent re-computation of the ordered-component framing: each
/// component contributes its `u64` little-endian length then its bytes.
fn framed_fnv1a64(components: &[&[u8]]) -> u64 {
    let mut framed: Vec<u8> = Vec::new();
    for component in components {
        framed.extend_from_slice(&(component.len() as u64).to_le_bytes());
        framed.extend_from_slice(component);
    }
    fnv1a64(&framed)
}

/// Build the fingerprint for a complete known input (test inputs are always
/// known and non-empty; unknown inputs go through `of_components`).
fn complete_input(bytes: &[u8]) -> InputFingerprint {
    InputFingerprint::new(bytes).expect("test input is known and non-empty")
}

#[test]
fn same_bytes_give_same_digest_pinned_against_independent_recomputation() {
    let bytes = b"complete prompt bytes for reuse check";
    let first = complete_input(bytes);
    let second = complete_input(bytes);
    let third = complete_input(bytes);
    assert_eq!(first, second);
    assert_eq!(second, third);
    assert_eq!(first.input_len, bytes.len());
    assert_eq!(
        first.digest,
        fnv1a64(bytes),
        "digest must be FNV-1a-64 over the complete input"
    );
    assert!(first.allows_reuse(&second));
    assert!(second.allows_reuse(&first));
}

#[test]
fn one_byte_flip_changes_digest_and_refuses_reuse() {
    let base = b"complete prompt bytes for reuse check";
    let base_print = complete_input(base);
    let mut flipped = base.to_vec();
    flipped[7] ^= 0x01;
    let flipped_print = complete_input(&flipped);
    assert_ne!(
        base_print.digest, flipped_print.digest,
        "one flipped byte must move the digest"
    );
    assert_eq!(base_print.input_len, flipped_print.input_len);
    assert_ne!(base_print, flipped_print);
    assert!(!base_print.allows_reuse(&flipped_print));
}

#[test]
fn every_input_byte_is_covered_by_the_digest() {
    // Completeness: flipping any single byte of the input must change the
    // digest, so no byte is ignored or truncated away.
    let base = b"every byte counts: 0123456789";
    let base_print = complete_input(base);
    let mut seen: Vec<u64> = Vec::with_capacity(base.len());
    for index in 0..base.len() {
        let mut flipped = base.to_vec();
        flipped[index] ^= 0x01;
        let digest = complete_input(&flipped).digest;
        assert_ne!(
            digest, base_print.digest,
            "a flip at any position must move the digest"
        );
        assert!(
            !seen.contains(&digest),
            "flips at distinct positions must not alias"
        );
        seen.push(digest);
    }
    assert_eq!(seen.len(), base.len());
}

#[test]
fn last_byte_only_difference_changes_digest() {
    // The trailing byte is covered: inputs differing only there must not
    // share a fingerprint (a stable-prefix-only shortcut would alias them).
    let base = b"complete prompt bytes for reuse check";
    let mut right = base.to_vec();
    let last = right.len() - 1;
    right[last] ^= 0x01;
    let left_print = complete_input(base);
    let right_print = complete_input(&right);
    assert_ne!(
        left_print.digest, right_print.digest,
        "a last-byte difference must move the digest"
    );
    assert!(!left_print.allows_reuse(&right_print));
}

#[test]
fn truncated_input_digest_never_equals_full_input_digest() {
    // Completeness against truncation: the digest over a strict prefix of
    // the input must not equal the digest over the full input.
    let full = b"complete prompt bytes for reuse check";
    let full_print = complete_input(full);
    for end in [1usize, full.len() / 2, full.len() - 1] {
        let prefix_print = complete_input(&full[..end]);
        assert_ne!(
            prefix_print.digest, full_print.digest,
            "a truncated input must not share the full-input digest"
        );
        assert_ne!(prefix_print, full_print);
    }
}

#[test]
fn component_boundaries_and_order_participate_in_the_digest() {
    // Ordered components are length-framed, so mere concatenation never
    // aliases: ["ab", "c"] and ["a", "bc"] are different inputs with
    // different fingerprints, as are opposite orders.
    let ab_c = InputFingerprint::of_components(&[Some(b"ab".as_slice()), Some(b"c".as_slice())])
        .expect("known components build a fingerprint");
    let ab_c_again =
        InputFingerprint::of_components(&[Some(b"ab".as_slice()), Some(b"c".as_slice())])
            .expect("known components build a fingerprint");
    assert_eq!(ab_c, ab_c_again);
    assert_eq!(ab_c.input_len, 3);
    assert_eq!(
        ab_c.digest,
        framed_fnv1a64(&[b"ab".as_slice(), b"c".as_slice()]),
        "component digest must frame each length before its bytes"
    );
    let a_bc = InputFingerprint::of_components(&[Some(b"a".as_slice()), Some(b"bc".as_slice())])
        .expect("known components build a fingerprint");
    let abc = InputFingerprint::of_components(&[Some(b"abc".as_slice())])
        .expect("known components build a fingerprint");
    assert_ne!(
        ab_c.digest, a_bc.digest,
        "regrouped components must not share a digest"
    );
    assert_ne!(
        ab_c.digest, abc.digest,
        "split components must not share the joined digest"
    );
    assert_ne!(
        a_bc.digest, abc.digest,
        "split components must not share the joined digest"
    );
    let ordered =
        InputFingerprint::of_components(&[Some(b"head".as_slice()), Some(b"tail".as_slice())])
            .expect("known components build a fingerprint");
    let reversed =
        InputFingerprint::of_components(&[Some(b"tail".as_slice()), Some(b"head".as_slice())])
            .expect("known components build a fingerprint");
    assert_ne!(
        ordered.digest, reversed.digest,
        "component order must move the digest"
    );
    assert!(!ordered.allows_reuse(&reversed));
}

#[test]
fn empty_input_is_refused_fail_closed() {
    assert_eq!(
        InputFingerprint::new(&[]).expect_err("empty input must fail"),
        FingerprintError::MissingInput
    );
    assert_eq!(
        InputFingerprint::of_components(&[]).expect_err("no components must fail"),
        FingerprintError::MissingInput
    );
    assert_eq!(
        InputFingerprint::of_components(&[Some(b"".as_slice())])
            .expect_err("zero known bytes must fail"),
        FingerprintError::MissingInput
    );
    assert_eq!(
        FingerprintError::MissingInput.to_string(),
        "missing fingerprint input"
    );
}

#[test]
fn unknown_component_is_refused_with_index_and_emits_no_digest() {
    // Disable-on-unknown: the first unknown component refuses with its
    // index; no fingerprint value exists, so no cache decision can proceed.
    let refused = InputFingerprint::of_components(&[
        Some(b"known-head".as_slice()),
        None,
        Some(b"known-tail".as_slice()),
    ])
    .expect_err("unknown component must fail");
    assert_eq!(refused, FingerprintError::UnknownComponent { index: 1 });
    // Unknowns are reported in index order even when several are unknown.
    let first =
        InputFingerprint::of_components(&[None, None]).expect_err("unknown components must fail");
    assert_eq!(first, FingerprintError::UnknownComponent { index: 0 });
    // Unknown wins over empty: a lone unknown refuses as unknown.
    let lone = InputFingerprint::of_components(&[None]).expect_err("unknown component must fail");
    assert_eq!(lone, FingerprintError::UnknownComponent { index: 0 });
    assert_eq!(
        FingerprintError::UnknownComponent { index: 1 }.to_string(),
        "unknown fingerprint component at index 1"
    );
}

/// Test-only reuse gate: a stored fingerprint serves a candidate only on an
/// exact match. `Err` candidates (unknown input) and a missing store both
/// refuse fail-closed and serve nothing.
struct ReuseHarness {
    stored: Option<InputFingerprint>,
    served: usize,
    refused: usize,
}

impl ReuseHarness {
    fn new(stored: Option<InputFingerprint>) -> Self {
        Self {
            stored,
            served: 0,
            refused: 0,
        }
    }

    /// Attempt one reuse. Returns true only when served from cache.
    fn attempt(&mut self, candidate: Result<InputFingerprint, FingerprintError>) -> bool {
        match (&self.stored, candidate) {
            (Some(stored), Ok(print)) if stored.allows_reuse(&print) => {
                self.served += 1;
                true
            }
            _ => {
                self.refused += 1;
                false
            }
        }
    }
}

#[test]
fn reuse_with_unknown_candidate_refuses_fail_closed() {
    let bytes = b"complete prompt bytes for reuse check";
    let mut harness = ReuseHarness::new(Some(complete_input(bytes)));
    // Exact match serves.
    assert!(harness.attempt(InputFingerprint::new(bytes)));
    // Unknown candidate refuses: nothing is served from cache.
    assert!(!harness.attempt(InputFingerprint::of_components(&[
        Some(bytes.as_slice()),
        None,
    ])));
    // Known-but-different candidate refuses.
    assert!(!harness.attempt(InputFingerprint::new(b"changed prompt bytes")));
    assert_eq!(harness.served, 1);
    assert_eq!(harness.refused, 2);
    // An empty store refuses everything, even an exact candidate.
    let mut empty = ReuseHarness::new(None);
    assert!(!empty.attempt(InputFingerprint::new(bytes)));
    assert_eq!(empty.served, 0);
    assert_eq!(empty.refused, 1);
}

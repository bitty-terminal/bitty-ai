//! Provider-scoped prefix-cache keys plus measured hit-rate evidence (AI-0082).
//!
//! AIQ-13 (prefix-cache effectiveness) narrowing input: deterministic encoding
//! (AIQ-12) is settled via [`assemble_prompt`] canonical bytes and the
//! stable-before-dynamic layout; this file pins the provider-scoped key
//! mechanism ([`CacheKey`]) and measures its reuse behavior with numbers.
//!
//! # Pinned key-scope rule
//!
//! A key is `(provider_id, model_id, scope, stable_prefix_hash, prefix_len)`
//! where the stable prefix is the leading canonical bytes before the
//! Runtime/Turn section header and the hash is deterministic FNV-1a-64 over
//! exactly those bytes:
//!
//! - Same bytes plus a different provider, model, or scope compare unequal
//!   (no cross-talk: a cached prefix never serves another route).
//! - Same triple plus same bytes compare equal (repeated construction is
//!   stable: no clock or hash-seed input).
//! - The hash covers the FULL stable prefix: any stable-region change changes
//!   the key (miss), while a trailing-only (Runtime/Turn) change keeps the
//!   key (hit) even though the full canonical bytes differ. The key addresses
//!   the reusable prefix, not the whole prompt.
//!
//! # Hit-rate harness (mechanism demonstration, not a performance claim)
//!
//! A test-only scripted turn sequence with repeated stable heads plus varying
//! tails counts key hits over N simulated rounds and asserts the exact pinned
//! outcome (per-round hit flags, total hits, total rounds, distinct keys).
//! No I/O, no clock, no threads: every byte comes from [`assemble_prompt`]
//! over caller-built snapshots and every lookup runs over a `Vec` seen-list
//! (no `HashMap`, no `RandomState`).
//!
//! Non-overlap: `prompt.rs` unit tests own canonical determinism and the
//! trailing-change prefix property; `cache_invalidation.rs` owns re-assembly
//! prefix-stability. Nothing here re-asserts those shapes beyond consuming
//! them as the keying input.

use std::hash::{Hash, Hasher};

use bitty_ai_runtime::{
    AssembledPrompt, CacheKey, CacheKeyError, CacheScope, LayerInput, MAX_CANONICAL_BYTES,
    PromptError, PromptLayer, PromptSnapshot, SUPPORTED_SKILL_VERSIONS, assemble_prompt,
    common_prefix_len,
};

/// Deterministic FNV-1a-64, test-side re-computation of the keying digest.
/// Duplicated (not imported) on purpose: the test pins the digest value
/// independently of the implementation under test.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn text_layer(layer: PromptLayer, text: &str) -> LayerInput {
    LayerInput::text_only(layer, text)
}

/// Assemble canonical bytes with a fixed stable head (every layer but
/// Runtime/Turn) plus one varying tail.
fn canonical_bytes(stable_user_text: &str, turn_text: &str) -> Vec<u8> {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, stable_user_text),
            text_layer(PromptLayer::Project, "stable project"),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            text_layer(PromptLayer::RuntimeTurn, turn_text),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot)
        .expect("test snapshot assembles")
        .canonical_bytes()
        .to_vec()
}

fn session_key(provider: &str, model: &str, bytes: &[u8]) -> CacheKey {
    CacheKey::new(provider, model, CacheScope::Session, bytes).expect("valid test key inputs")
}

/// Assemble canonical bytes with a fixed core/skills head, caller-chosen
/// user and project texts (which may legally carry section-like literals),
/// and one tail.
fn canonical_bytes_with_texts(user_text: &str, project_text: &str, turn_text: &str) -> Vec<u8> {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, user_text),
            text_layer(PromptLayer::Project, project_text),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            text_layer(PromptLayer::RuntimeTurn, turn_text),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot)
        .expect("test snapshot assembles")
        .canonical_bytes()
        .to_vec()
}

fn assemble_skills(source: &str) -> AssembledPrompt {
    let layer = LayerInput::skills_from_str(source).expect("valid skill registry");
    let snapshot =
        PromptSnapshot::new("bitty-core-prompt@1", vec![layer]).expect("valid skills snapshot");
    assemble_prompt(&snapshot).expect("skills snapshot assembles")
}

fn single_skill_source(fragment: &str, fields: &str) -> String {
    format!("version = 1\n---\nname = alpha\nversion = 1\n{fields}text:\n{fragment}\n")
}

/// Test-only deterministic hasher: proves equal keys hash equal without
/// touching `RandomState`.
struct FnvHasher(u64);

impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

fn hash_of<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = FnvHasher(0xcbf29ce484222325);
    value.hash(&mut hasher);
    hasher.finish()
}

/// Test-only hit/miss counter over a scripted round sequence. The seen-list
/// is a `Vec` with linear scan: deterministic iteration, no `HashMap`, no
/// `RandomState`, no clock.
struct PrefixHarness {
    seen: Vec<CacheKey>,
    hits: usize,
    total: usize,
}

impl PrefixHarness {
    fn new() -> Self {
        Self {
            seen: Vec::new(),
            hits: 0,
            total: 0,
        }
    }

    /// Observe one round key. Returns true on a repeat (hit).
    fn observe(&mut self, key: CacheKey) -> bool {
        self.total += 1;
        if self.seen.contains(&key) {
            self.hits += 1;
            true
        } else {
            self.seen.push(key);
            false
        }
    }

    fn distinct(&self) -> usize {
        self.seen.len()
    }
}

#[test]
fn key_scope_inequality_matrix() {
    let bytes = canonical_bytes("stable user", "turn one");
    let base = session_key("bitty-fake", "fake-chat", &bytes);
    // Single-axis variations: each one must break equality.
    let provider = session_key("bitty-other", "fake-chat", &bytes);
    let model = session_key("bitty-fake", "other-chat", &bytes);
    let turn = CacheKey::new("bitty-fake", "fake-chat", CacheScope::Turn, &bytes)
        .expect("valid test key inputs");
    let round = CacheKey::new("bitty-fake", "fake-chat", CacheScope::Round, &bytes)
        .expect("valid test key inputs");
    let stable_changed = session_key(
        "bitty-fake",
        "fake-chat",
        &canonical_bytes("changed user", "turn one"),
    );
    assert_ne!(base, provider, "provider variation must change the key");
    assert_ne!(base, model, "model variation must change the key");
    assert_ne!(base, turn, "scope variation (turn) must change the key");
    assert_ne!(base, round, "scope variation (round) must change the key");
    assert_ne!(
        base, stable_changed,
        "stable-region byte variation must change the key"
    );
    // Control: an identical rebuild is equal.
    assert_eq!(base, session_key("bitty-fake", "fake-chat", &bytes));
}

#[test]
fn repeated_construction_is_deterministic_with_pinned_digest() {
    let bytes = canonical_bytes("stable user", "turn one");
    let first = session_key("bitty-fake", "fake-chat", &bytes);
    let second = session_key("bitty-fake", "fake-chat", &bytes);
    let third = session_key("bitty-fake", "fake-chat", &bytes);
    assert_eq!(first, second);
    assert_eq!(second, third);
    // The hashed region is exactly the leading bytes up to the Runtime/Turn
    // section header (including the trailing LF of the skills section, which
    // is part of the stable encoding). Recompute that offset independently
    // here; the marker occurs exactly once in these controlled texts.
    let marker: &[u8] = b"[layer:runtime-turn len=";
    let positions: Vec<usize> = bytes
        .windows(marker.len())
        .enumerate()
        .filter_map(|(index, window)| (window == marker).then_some(index))
        .collect();
    assert_eq!(positions.len(), 1, "controlled bytes carry one header");
    let expected_len = positions[0];
    assert_eq!(first.prefix_len, expected_len);
    assert_eq!(
        first.stable_prefix_hash,
        fnv1a64(&bytes[..expected_len]),
        "digest must be FNV-1a-64 over exactly the stable prefix bytes"
    );
    // Equal keys hash equal under a deterministic hasher (map-key contract).
    assert_eq!(hash_of(&first), hash_of(&second));
    assert_eq!(hash_of(&first), hash_of(&third));
}

#[test]
fn scopes_are_separated() {
    let bytes = canonical_bytes("stable user", "turn one");
    let session = CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &bytes)
        .expect("valid test key inputs");
    let turn = CacheKey::new("bitty-fake", "fake-chat", CacheScope::Turn, &bytes)
        .expect("valid test key inputs");
    let round = CacheKey::new("bitty-fake", "fake-chat", CacheScope::Round, &bytes)
        .expect("valid test key inputs");
    assert_ne!(session, turn, "session key must not equal turn key");
    assert_ne!(session, round, "session key must not equal round key");
    assert_ne!(turn, round, "turn key must not equal round key");
}

#[test]
fn trailing_only_change_keeps_key_but_moves_canonical_bytes() {
    let left = canonical_bytes("stable user", "turn one");
    let right = canonical_bytes("stable user", "turn two");
    assert_ne!(left, right, "tails differ, so canonical bytes differ");
    let left_key = session_key("bitty-fake", "fake-chat", &left);
    let right_key = session_key("bitty-fake", "fake-chat", &right);
    assert_eq!(left_key.prefix_len, right_key.prefix_len);
    let common = common_prefix_len(&left, &right);
    assert!(
        common >= left_key.prefix_len,
        "trailing-only change keeps a measurable common prefix covering the whole hashed region"
    );
    assert_eq!(
        left_key, right_key,
        "pinned rule: a change outside the hashed stable prefix must NOT change the key"
    );
}

#[test]
fn stable_region_change_breaks_key_inside_hashed_region() {
    let base = canonical_bytes("stable user", "turn one");
    let changed = canonical_bytes("changed user", "turn one");
    let base_key = session_key("bitty-fake", "fake-chat", &base);
    let changed_key = session_key("bitty-fake", "fake-chat", &changed);
    assert_ne!(
        base_key, changed_key,
        "pinned rule: any stable-region change changes the key"
    );
    let common = common_prefix_len(&base, &changed);
    assert!(
        common < base_key.prefix_len,
        "divergence sits inside the hashed stable prefix"
    );
    assert_ne!(
        base_key.stable_prefix_hash, changed_key.stable_prefix_hash,
        "stable-region change must move the digest"
    );
}

#[test]
fn skill_version_only_change_has_no_valid_renderable_pair() {
    assert_eq!(SUPPORTED_SKILL_VERSIONS, &["1"]);
    let valid = assemble_skills(&single_skill_source("same fragment", ""));
    let valid_key = session_key("bitty-fake", "fake-chat", valid.canonical_bytes());
    assert_eq!(
        valid.section_text(PromptLayer::SkillsProfile),
        "same fragment"
    );
    assert!(valid_key.prefix_len > 0);

    let bumped = single_skill_source("same fragment", "")
        .replace("name = alpha\nversion = 1", "name = alpha\nversion = 2");
    assert!(matches!(
        LayerInput::skills_from_str(&bumped),
        Err(PromptError::UnsupportedSkillVersion { version }) if version == "2"
    ));
}

#[test]
fn skill_fragment_change_breaks_stable_prefix_and_cache_key() {
    let left = assemble_skills(&single_skill_source("fragment alpha", ""));
    let right = assemble_skills(&single_skill_source("fragment bravo", ""));
    assert_ne!(
        left.section_text(PromptLayer::SkillsProfile).as_bytes(),
        right.section_text(PromptLayer::SkillsProfile).as_bytes()
    );
    assert_ne!(left.canonical_bytes(), right.canonical_bytes());

    let left_key = session_key("bitty-fake", "fake-chat", left.canonical_bytes());
    let right_key = session_key("bitty-fake", "fake-chat", right.canonical_bytes());
    assert_eq!(left_key.prefix_len, right_key.prefix_len);
    assert_ne!(left_key.stable_prefix_hash, right_key.stable_prefix_hash);
    assert_ne!(left_key, right_key);
}

#[test]
fn skill_policy_fields_render_after_stable_prefix_and_conflicts_fail_closed() {
    let left = assemble_skills(&single_skill_source(
        "shared fragment",
        "allow_tool = tool_a\ndeny_tool = tool_b\nscope = workspace.read\ndirective.tone = terse\n",
    ));
    let right = assemble_skills(&single_skill_source(
        "shared fragment",
        "allow_tool = tool_c\ndeny_tool = tool_d\nscope = terminal.read\ndirective.tone = casual\n",
    ));
    assert_eq!(
        left.section_text(PromptLayer::SkillsProfile),
        right.section_text(PromptLayer::SkillsProfile)
    );
    assert_ne!(left.canonical_bytes(), right.canonical_bytes());

    let left_key = session_key("bitty-fake", "fake-chat", left.canonical_bytes());
    let right_key = session_key("bitty-fake", "fake-chat", right.canonical_bytes());
    assert_eq!(left_key.prefix_len, right_key.prefix_len);
    assert_eq!(left_key.stable_prefix_hash, right_key.stable_prefix_hash);
    assert_eq!(left_key, right_key);

    let conflict = concat!(
        "version = 1\n",
        "---\n",
        "name = alpha\n",
        "version = 1\n",
        "directive.tone = concise\n",
        "---\n",
        "name = beta\n",
        "version = 1\n",
        "directive.tone = casual\n",
    );
    assert!(matches!(
        LayerInput::skills_from_str(conflict),
        Err(PromptError::UnresolvableConflict { key, .. }) if key == "tone"
    ));
}

#[test]
fn hit_rate_harness_repeated_heads_hit_changed_heads_miss() {
    // Scripted 10-round sequence: stable heads repeat with a fresh tail every
    // round, so every hit proves tail-independence and every miss proves
    // head-sensitivity. The per-round outcome is pinned exactly.
    let heads = [
        "head-alpha",
        "head-alpha",
        "head-alpha",
        "head-beta",
        "head-alpha",
        "head-alpha",
        "head-beta",
        "head-alpha",
        "head-gamma",
        "head-alpha",
    ];
    let expected_hits = [
        false, true, true, false, true, true, true, true, false, true,
    ];
    let mut harness = PrefixHarness::new();
    let mut observed = Vec::with_capacity(heads.len());
    for (round, head) in heads.into_iter().enumerate() {
        let bytes = canonical_bytes(head, &format!("turn-{round}"));
        observed.push(harness.observe(session_key("bitty-fake", "fake-chat", &bytes)));
    }
    assert_eq!(observed, Vec::from(expected_hits));
    assert_eq!(harness.total, 10);
    assert_eq!(harness.hits, 7);
    assert_eq!(harness.distinct(), 3, "three stable heads circulated");
    // 7 hits / 10 rounds = 70%: every repeat-head round hit, every
    // changed-head round missed. Mechanism demonstration, not a
    // performance claim.
    assert_eq!(harness.hits * 100 / harness.total, 70);
}

#[test]
fn constructor_rejects_malformed_inputs_fail_closed() {
    let bytes = canonical_bytes("stable user", "turn one");
    assert_eq!(
        CacheKey::new("BITTY-FAKE", "fake-chat", CacheScope::Session, &bytes)
            .expect_err("uppercase provider id must fail"),
        CacheKeyError::InvalidProviderId {
            id: "BITTY-FAKE".to_owned(),
        }
    );
    assert_eq!(
        CacheKey::new("bitty-fake", "Fake Chat", CacheScope::Session, &bytes)
            .expect_err("malformed model id must fail"),
        CacheKeyError::InvalidModelId {
            id: "Fake Chat".to_owned(),
        }
    );
    assert_eq!(
        CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &[])
            .expect_err("empty canonical bytes must fail"),
        CacheKeyError::EmptyCanonical
    );
    assert_eq!(
        CacheKey::new(
            "bitty-fake",
            "fake-chat",
            CacheScope::Session,
            b"not a canonical form",
        )
        .expect_err("marker-free bytes must fail"),
        CacheKeyError::MissingStableMarker
    );
    let oversized = vec![0u8; MAX_CANONICAL_BYTES + 1];
    assert!(
        matches!(
            CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &oversized)
                .expect_err("over-bound canonical bytes must fail"),
            CacheKeyError::CanonicalTooLarge { .. }
        ),
        "over-bound canonical bytes must fail closed"
    );
}

#[test]
fn embedded_marker_in_stable_text_must_not_alias_keys() {
    // AI-0084: stable-layer text may legally carry the
    // `[layer:runtime-turn len=` literal (text validation rejects only
    // CR/NUL), so a first-occurrence byte scan truncates the stable prefix
    // early. Two prompts whose stable bytes differ only after such an
    // embedded literal must NOT share a key.
    let head = "payload [layer:runtime-turn len=0]\n";
    let user_a = format!("{head}stable-variant-alpha-0001");
    let user_b = format!("{head}stable-variant-beta--0001");
    assert_eq!(
        user_a.len(),
        user_b.len(),
        "test controls carry equal-length user texts"
    );
    let bytes_a = canonical_bytes_with_texts(&user_a, "stable project", "turn one");
    let bytes_b = canonical_bytes_with_texts(&user_b, "stable project", "turn one");
    assert_ne!(bytes_a, bytes_b, "stable-region texts differ");
    let key_a = session_key("bitty-fake", "fake-chat", &bytes_a);
    let key_b = session_key("bitty-fake", "fake-chat", &bytes_b);
    assert_eq!(
        key_a.prefix_len, key_b.prefix_len,
        "equal-length stable sections share the true boundary"
    );
    assert_ne!(
        key_a.stable_prefix_hash, key_b.stable_prefix_hash,
        "digest must cover the full stable prefix past any embedded literal"
    );
    assert_ne!(
        key_a, key_b,
        "differing stable prefixes must never alias to one key"
    );
}

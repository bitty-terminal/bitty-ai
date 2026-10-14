//! Host-side pure-type/key mirror of the cache-disclosure seam (AI-0276).
//!
//! **Status: Phase 2 S1 pure-type/key mirror per AI-0276 (survey AI-0268
//! PX-1167..PX-1170).** This module mirrors the *shapes* of the runtime
//! cache-key contract plus its disclosure adjuncts as dependency-free
//! pure types: [`CacheScope`], [`CacheKey`], [`CacheKeyError`] with the
//! verbatim [`CacheKey::new`] pure logic, the length-aware stable-prefix
//! walker, inline FNV-1a-64, [`common_prefix_len`], [`InputFingerprint`] /
//! [`FingerprintError`], [`WriterLease`] / [`WriterRefusal`] /
//! [`SessionState`] with the terminal-first check order, [`FallbackEnvelope`]
//! / [`FallbackError`] with [`FALLBACK_ID`] / [`FALLBACK_KIND`] plus the
//! length-prefixed wire form and [`fallback_for`], [`SecretField`] /
//! [`SecretError`] with [`SECRET_REDACTED`] plus the pre-write
//! scrub/absence helpers, [`ProviderUsage`] with the cost-park rule, the
//! read-only export/gc-preview discipline ([`export_preview_copy`],
//! [`ref_export_lines`], [`GcPreview`], [`bound_gc_preview`]), and
//! [`ensure_0600`]. All helpers are pure `&str`/`&[u8]`-in/value-out (plus
//! the caller-supplied temp-only path for [`ensure_0600`]) with no clock,
//! threads, network, IPC, or ambient authority: `std` only, like the rest of
//! this crate.
//!
//! Explicitly **not** mirrored here (each needs its own scoped task):
//!
//! - Cache store, admission/reuse policy, hit-rate promotion, cross-provider
//!   reuse (same bytes under a different route stay unequal by construction).
//! - IPC transport, discovery, durable paths, journals, retention policies,
//!   ledgers, credential material, currency/cost accounting, grants.
//! - Filesystem access beyond the caller-supplied temp-only [`ensure_0600`]
//!   probe path (no store file is ever opened here).
//! - Wire encoding beyond the envelope bytes: byte encoding for any IPC
//!   channel stays with the scoped IPC task (S-2), which must also take on
//!   the IPC review trigger noted in this crate's `Cargo.toml`.
//!
//! S-1/S-2 posture (AIQ-38 still open): this mirror is terminal-side
//! metadata only — shapes a host uses to *talk about* cached disclosure,
//! never a channel carrying it. It grants nothing: constructing these values
//! performs no I/O and bestows no capability. Key equality never authorizes
//! reuse on its own; admission stays deny-all until a scoped policy task
//! designs it (S-3).
//!
//! Value sources (read-only, verified against the AI-0268 survey):
//! `bitty-ai-runtime` `cache_key.rs` (scope rule, key shape, `new` logic,
//! length-aware walker, FNV-1a-64), `prompt.rs` ([`MAX_CANONICAL_BYTES`],
//! [`common_prefix_len`]), `provider.rs` (`MP-2` id shape,
//! [`ProviderUsage`]), `selection.rs` (model-name shape), `fingerprint.rs`
//! (complete-input rule, disable-on-unknown), `fencing.rs` (writer-fencing
//! rule, terminal-first order), `fallback.rs` (DEC-0007 envelope, caps,
//! fixed identity, scrub rule), `secret.rs` (typed container, unconditional
//! redaction, pre-write helpers), `session.rs` (lifecycle states for the
//! fencing check), `bitty-ai-slice` `merge.rs` (export/gc-preview read-only
//! discipline) and `bitty-ai-session` `fs_perms.rs` (`0600` enforcement).
//!
//! ## Mirror discipline
//!
//! Every bound, variant, grammar, refusal order, gate order, and display
//! string here matches the runtime source exactly so host and runtime agree
//! on the shape without the host crate depending on the runtime crate:
//!
//! - Bounds: [`MAX_CANONICAL_BYTES`], [`MAX_PROVIDER_ID_LEN`],
//!   [`MAX_MODEL_NAME_LEN`], [`MAX_FALLBACK_ID_BYTES`],
//!   [`MAX_FALLBACK_KIND_BYTES`], [`MAX_FALLBACK_TEXT_BYTES`],
//!   [`MAX_SECRET_LEN`].
//! - Key: [`CacheScope`] (3 variants, never-cross-scopes) plus [`CacheKey`]
//!   (5 fields, `Eq`+`Hash` cover all = scope rule as code) plus
//!   [`CacheKeyError`] (5 variants, all fail-closed) with the identical
//!   [`Display`](std::fmt::Display) strings.
//! - Grammar: provider-id `^[a-z][a-z0-9_-]*$` (mirrors `provider.rs`
//!   `MP-2`) and model-name (mirrors `selection.rs`) checked by private
//!   predicates with the same shape; failures map to
//!   [`CacheKeyError::InvalidProviderId`] /
//!   [`CacheKeyError::InvalidModelId`] so this seam owns no
//!   `ProviderError`/`SelectionError` (those stay with AI-0271/AI-0272).
//! - Walker: [`STABLE_LAYERS`](STABLE_LAYERS) order, length-aware section
//!   skip (never a marker scan), trailing runtime + `[effective]` tail
//!   validation, inline FNV-1a-64 (never `DefaultHasher`/`RandomState`,
//!   never a `HashMap`).
//! - Adjuncts: fingerprint exact-only reuse; lease fencing terminal-first;
//!   fallback fixed identity plus length-prefixed framing plus scrub rule;
//!   secret unconditional redaction plus pre-write helpers; usage
//!   observed-not-billed with zero-means-unreported; export/gc previews
//!   read-only and byte-matching; `0600` owner-only on the caller path.
//!
//! Deny-all posture holds by construction: nothing here grants execution
//! authority — keys are data (equality never admits a read), fingerprints
//! refuse on unknown input, leases refuse on stale fences, envelopes refuse
//! on over-cap input, secrets never render, previews write nothing, and
//! every refusal fails closed with no partial state.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::path::Path;

// ---------------------------------------------------------------------------
// Bounds (mirror of `prompt.rs` + `provider.rs` + `selection.rs`).
// ---------------------------------------------------------------------------

/// Maximum canonical byte form length in bytes (mirrors `prompt.rs`).
pub const MAX_CANONICAL_BYTES: usize = 96 * 1024;

/// Maximum provider id length in bytes (mirrors the `MP-2` bound in
/// `provider.rs`).
pub const MAX_PROVIDER_ID_LEN: usize = 64;

/// Maximum model name length in bytes (mirrors `selection.rs`).
pub const MAX_MODEL_NAME_LEN: usize = 128;

/// Maximum envelope identity length in bytes (mirrors `fallback.rs`).
pub const MAX_FALLBACK_ID_BYTES: usize = 64;

/// Maximum envelope kind tag length in bytes (mirrors `fallback.rs`).
pub const MAX_FALLBACK_KIND_BYTES: usize = 64;

/// Maximum envelope text payload length in bytes (mirrors `fallback.rs`).
pub const MAX_FALLBACK_TEXT_BYTES: usize = 512;

/// Maximum secret value length in bytes (mirrors `secret.rs`).
pub const MAX_SECRET_LEN: usize = 4 * 1024;

/// Redaction placeholder emitted by [`SecretField`] formatting impls.
///
/// Fixed and value-independent: it leaks neither length nor prefix.
pub const SECRET_REDACTED: &str = "[redacted secret]";

/// Fixed envelope identity emitted by [`fallback_for`]: the fallback source
/// is always the same reader-side mechanism, never caller-chosen (mirrors
/// `fallback.rs`, DEC-0007).
pub const FALLBACK_ID: &str = "fallback";

/// Fixed envelope kind emitted by [`fallback_for`]: unreadable structured
/// input that fell back to the minimal envelope (mirrors `fallback.rs`,
/// DEC-0007).
pub const FALLBACK_KIND: &str = "syntax-fallback";

// ---------------------------------------------------------------------------
// Walker constants (mirror of `cache_key.rs`; private like the source).
// ---------------------------------------------------------------------------

/// Stable section names in canonical order (`prompt::render_canonical`).
/// The stable prefix ends right after the last one.
const STABLE_LAYERS: [&str; 4] = ["core-contract", "user", "project", "skills-profile"];
/// Dynamic section name that must open immediately after the boundary.
const RUNTIME_LAYER: &str = "runtime-turn";
/// Fixed first line of the canonical form (`prompt::render_canonical`).
const CANONICAL_MAGIC: &[u8] = b"prompt/1\n";
/// Fixed prefix of the pinned-version line (the version itself varies).
const CORE_VERSION_PREFIX: &[u8] = b"core-version:";
/// Fixed bytes opening a section header (`[layer:`).
const SECTION_OPEN: &[u8] = b"[layer:";
/// Fixed bytes between a section name and its declared length (` len=`).
const SECTION_LEN_SEP: &[u8] = b" len=";
/// Fixed bytes closing a section header (`]\n`).
const SECTION_CLOSE: &[u8] = b"]\n";
/// Fixed bytes opening the trailing policy block (`prompt::render_canonical`
/// always emits it after the Runtime/Turn text section).
const EFFECTIVE_OPEN: &[u8] = b"[effective]\n";

/// FNV-1a-64 offset basis (deterministic across processes and platforms).
/// Shared by the key digest and the fingerprint digest (both sources use
/// the same constants; mirrors `cache_key.rs` and `fingerprint.rs`).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (deterministic across processes and platforms).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Fixed magic opening the envelope wire form (`to_bytes` /
/// `from_bytes`). Versioned (`/1`) so a future envelope revision decodes
/// fail-closed as [`FallbackError::Malformed`] instead of aliasing (mirrors
/// `fallback.rs`).
const ENVELOPE_MAGIC: &[u8] = b"fallback/1\n";
/// Wire width of one length-prefixed field length (mirrors `fallback.rs`).
const LEN_WIDTH: usize = 8;

// ---------------------------------------------------------------------------
// Cache key (mirror of `cache_key.rs`).
// ---------------------------------------------------------------------------

/// Routing scope a cache key is valid for. Keys never cross scopes: a
/// [`CacheScope::Session`] key never equals a [`CacheScope::Turn`] key over
/// the same bytes (mirrors `cache_key.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheScope {
    /// Reusable across turns of one session (stable head unchanged).
    Session,
    /// Reusable within one turn only.
    Turn,
    /// Reusable within one provider round only.
    Round,
}

impl Display for CacheScope {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Session => write!(f, "session"),
            Self::Turn => write!(f, "turn"),
            Self::Round => write!(f, "round"),
        }
    }
}

/// Cache-key construction errors. Every variant fails closed with no key
/// (mirrors `cache_key.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheKeyError {
    /// Provider id violates the `MP-2` shape.
    InvalidProviderId {
        /// Rejected id.
        id: String,
    },
    /// Model id violates the registry name shape.
    InvalidModelId {
        /// Rejected id.
        id: String,
    },
    /// No canonical bytes were supplied (no stable prefix exists).
    EmptyCanonical,
    /// The canonical bytes do not follow the length-prefixed section layout,
    /// so the stable prefix boundary is undefined.
    MissingStableMarker,
    /// The canonical bytes exceed [`MAX_CANONICAL_BYTES`].
    CanonicalTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
}

impl Display for CacheKeyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidProviderId { id } => write!(f, "invalid provider id: {id}"),
            Self::InvalidModelId { id } => write!(f, "invalid model id: {id}"),
            Self::EmptyCanonical => write!(f, "empty canonical bytes"),
            Self::MissingStableMarker => {
                write!(f, "canonical bytes carry no runtime-turn section")
            }
            Self::CanonicalTooLarge { limit, actual } => write!(
                f,
                "canonical bytes of {actual} bytes exceed {limit} byte limit"
            ),
        }
    }
}

impl std::error::Error for CacheKeyError {}

/// One provider-scoped prefix-cache key.
///
/// Equality and hashing cover every field, which is the key-scope rule
/// stated as code: route fields (`provider_id`, `model_id`, `scope`) plus
/// content fields (`stable_prefix_hash`, `prefix_len`) all participate, so
/// same bytes under a different route compare unequal and the same
/// route-plus-bytes compares equal (mirrors `cache_key.rs`; no
/// cross-provider reuse).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Owning provider id (`MP-2` shape).
    pub provider_id: String,
    /// Registry-known model name (registry name shape).
    pub model_id: String,
    /// Routing scope this key is valid for.
    pub scope: CacheScope,
    /// FNV-1a-64 over exactly the leading `prefix_len` canonical bytes.
    pub stable_prefix_hash: u64,
    /// Length of the hashed stable prefix in bytes.
    pub prefix_len: usize,
}

impl CacheKey {
    /// Build a key over `canonical` (a `prompt::render_canonical` canonical
    /// byte form) for (`provider_id`, `model_id`, `scope`).
    ///
    /// Fail-closed: malformed ids, empty or over-bound bytes, or bytes with
    /// no Runtime/Turn section header yield a typed [`CacheKeyError`] and no
    /// key (mirrors `cache_key.rs` verbatim; the id grammars match
    /// `provider.rs` `MP-2` and `selection.rs` exactly, mapped here to
    /// [`CacheKeyError`] so this seam owns no `ProviderError`/`SelectionError`).
    ///
    /// # Errors
    ///
    /// Returns [`CacheKeyError`] as described above.
    pub fn new(
        provider_id: impl Into<String>,
        model_id: impl Into<String>,
        scope: CacheScope,
        canonical: &[u8],
    ) -> Result<Self, CacheKeyError> {
        let provider_id = provider_id.into();
        let model_id = model_id.into();
        if !provider_id_valid(&provider_id) {
            return Err(CacheKeyError::InvalidProviderId {
                id: provider_id.clone(),
            });
        }
        if !model_name_valid(&model_id) {
            return Err(CacheKeyError::InvalidModelId {
                id: model_id.clone(),
            });
        }
        if canonical.is_empty() {
            return Err(CacheKeyError::EmptyCanonical);
        }
        if canonical.len() > MAX_CANONICAL_BYTES {
            return Err(CacheKeyError::CanonicalTooLarge {
                limit: MAX_CANONICAL_BYTES,
                actual: canonical.len(),
            });
        }
        let prefix_len = stable_prefix_len(canonical).ok_or(CacheKeyError::MissingStableMarker)?;
        let stable_prefix_hash = fnv1a64(&canonical[..prefix_len]);
        Ok(Self {
            provider_id,
            model_id,
            scope,
            stable_prefix_hash,
            prefix_len,
        })
    }
}

/// Provider-id shape (`MP-2`, mirrors `provider.rs`
/// [`validate_provider_id`](https://github.com/bitty-terminal/bitty-ai)):
/// non-empty, at most [`MAX_PROVIDER_ID_LEN`] bytes,
/// `^[a-z][a-z0-9_-]*$`. Private: this seam maps the refusal to
/// [`CacheKeyError::InvalidProviderId`] instead of owning `ProviderError`.
fn provider_id_valid(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_PROVIDER_ID_LEN
        && id.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Model-name shape (mirrors `selection.rs` `validate_model_name`):
/// non-empty, at most [`MAX_MODEL_NAME_LEN`] bytes, starting with lowercase
/// alphanumeric, continuing with lowercase alphanumeric or `_`, `-`, `.`,
/// `:`, `/`. Private: this seam maps the refusal to
/// [`CacheKeyError::InvalidModelId`] instead of owning `SelectionError`.
fn model_name_valid(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_MODEL_NAME_LEN
        && name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && name.bytes().all(|b| {
            b.is_ascii_lowercase()
                || b.is_ascii_digit()
                || b == b'_'
                || b == b'-'
                || b == b'.'
                || b == b':'
                || b == b'/'
        })
}

/// Length of the stable prefix: the offset right after the terminating LF
/// of the last stable text section, walked with the length-prefixed section
/// headers of `prompt::render_canonical`. Returns `None` when the bytes do
/// not follow that layout (so `CacheKey::new` reports
/// [`CacheKeyError::MissingStableMarker`]).
///
/// Length-aware walking (never a naive marker scan): stable-layer text may
/// legally carry the literal `[layer:runtime-turn len=` (text validation
/// rejects only CR/NUL), so a first-occurrence search truncates the prefix
/// early and aliases differing stable prefixes to one key. Skipping each
/// section with its declared `len=` hops over any embedded header-shaped
/// bytes; anything malformed fails closed (mirrors `cache_key.rs` verbatim).
#[must_use]
fn stable_prefix_len(canonical: &[u8]) -> Option<usize> {
    let mut cursor: &[u8] = canonical.strip_prefix(CANONICAL_MAGIC)?;
    cursor = cursor.strip_prefix(CORE_VERSION_PREFIX)?;
    let version_end = cursor.iter().position(|byte| *byte == b'\n')?;
    cursor = &cursor[version_end + 1..];
    for layer in STABLE_LAYERS {
        cursor = strip_section(cursor, layer)?;
    }
    // The stable prefix ends where the Runtime/Turn section opens: record
    // that offset, then still validate the trailing runtime section and the
    // `[effective]` block so malformed tails fail closed instead of keying a
    // truncated layout.
    let boundary = canonical.len() - cursor.len();
    cursor = strip_section(cursor, RUNTIME_LAYER)?;
    if !cursor.starts_with(EFFECTIVE_OPEN) {
        return None;
    }
    Some(boundary)
}

/// Strip one `[layer:<name> len=<n>]\n<text>\n` section from the front of
/// `bytes`, returning the remainder. The declared `len=` drives the skip:
/// `<text>` hops over any header-shaped bytes it carries, so an embedded
/// literal can never shift the boundary. A declared length that overruns
/// the bytes or misses its terminating LF fails closed (mirrors
/// `cache_key.rs` verbatim).
fn strip_section<'bytes>(bytes: &'bytes [u8], layer: &str) -> Option<&'bytes [u8]> {
    let (text_len, rest) = section_header_len(bytes, layer)?;
    rest.get(text_len..)?.strip_prefix(b"\n")
}

/// Read a `[layer:<layer> len=<n>]\n` header from the front of `bytes`,
/// returning the declared text length and the remainder starting at the
/// section text. Rejects a missing shape, a non-decimal or empty length,
/// or a header naming a different layer.
///
/// The header prefix is matched byte-wise (never a substring search) so one
/// layer's header never parses as another's: a text literal like
/// `[layer:runtime-turn len=...` sits inside a skipped `<text>` span and is
/// never re-scanned as a header. The `len=`/`]\n` framing is required, not
/// just the layer name, so a bare `[layer:runtime-turn` line inside text
/// still fails closed instead of silently shifting the boundary (mirrors
/// `cache_key.rs` verbatim).
fn section_header_len<'bytes>(bytes: &'bytes [u8], layer: &str) -> Option<(usize, &'bytes [u8])> {
    let rest = bytes.strip_prefix(SECTION_OPEN)?;
    let rest = rest.strip_prefix(layer.as_bytes())?;
    let rest = rest.strip_prefix(SECTION_LEN_SEP)?;
    let digits_end = rest.iter().position(|byte| !byte.is_ascii_digit())?;
    if digits_end == 0 {
        return None;
    }
    let digits = core::str::from_utf8(rest.get(..digits_end)?).ok()?;
    let text_len = digits.parse::<usize>().ok()?;
    let rest = rest[digits_end..].strip_prefix(SECTION_CLOSE)?;
    Some((text_len, rest))
}

/// Deterministic FNV-1a-64 over `bytes`. Inline small hasher: no
/// `DefaultHasher`, no `RandomState`, no seed input, identical output on
/// every platform and process (mirrors `cache_key.rs` and `fingerprint.rs`;
/// both sources share these constants).
#[must_use]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Length of the common byte prefix of `left` and `right`.
///
/// Deterministic helper for prefix-stability evidence: changing only the
/// trailing Runtime/Turn layer must preserve the leading prefix bytes
/// (mirrors `prompt.rs` verbatim).
#[must_use]
pub fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let mut len = 0usize;
    for (a, b) in left.iter().zip(right.iter()) {
        if a != b {
            break;
        }
        len += 1;
    }
    len
}

// ---------------------------------------------------------------------------
// Fingerprint (types-only mirror of `fingerprint.rs`).
// ---------------------------------------------------------------------------

/// Fingerprint construction errors. Every variant fails closed with no
/// fingerprint value, so no reuse decision can proceed (mirrors
/// `fingerprint.rs` verbatim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerprintError {
    /// No input bytes were supplied (empty input or no known bytes).
    MissingInput,
    /// One component's bytes are unavailable, so equivalence is unknown.
    /// Carries the index of the first unknown component.
    UnknownComponent {
        /// Index of the first unknown component.
        index: usize,
    },
}

impl Display for FingerprintError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::MissingInput => write!(f, "missing fingerprint input"),
            Self::UnknownComponent { index } => {
                write!(f, "unknown fingerprint component at index {index}")
            }
        }
    }
}

impl std::error::Error for FingerprintError {}

/// One complete-input fingerprint.
///
/// Equality covers every field, which is the reuse rule stated as code:
/// the same complete input compares equal (repeated construction is
/// stable) and any input change compares unequal, so only an exact match
/// may reuse. `input_len` travels with the digest so a truncated input can
/// never compare equal to the full input it came from (mirrors
/// `fingerprint.rs`; types-only: no store, no admission policy — equality
/// never authorizes reuse on its own).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputFingerprint {
    /// FNV-1a-64 over the complete input (length-framed per component for
    /// multi-component inputs).
    pub digest: u64,
    /// Total fingerprinted input bytes (framing lengths excluded).
    pub input_len: usize,
}

impl InputFingerprint {
    /// Fingerprint the complete input `bytes`.
    ///
    /// The digest covers every byte: no truncation, no prefix shortcut.
    /// Fail-closed on empty input (mirrors `fingerprint.rs` verbatim).
    ///
    /// # Errors
    ///
    /// Returns [`FingerprintError::MissingInput`] when `bytes` is empty.
    pub fn new(bytes: &[u8]) -> Result<Self, FingerprintError> {
        if bytes.is_empty() {
            return Err(FingerprintError::MissingInput);
        }
        Ok(Self {
            digest: fnv1a64(bytes),
            input_len: bytes.len(),
        })
    }

    /// Fingerprint an ordered multi-component input. Each `Some` component
    /// contributes its `u64` little-endian byte length followed by its
    /// bytes, so regrouping (`["ab", "c"]` vs `["a", "bc"]` vs `["abc"]`)
    /// or reordering components never aliases one fingerprint.
    ///
    /// Fail-closed: the first `None` component refuses as
    /// [`FingerprintError::UnknownComponent`] (disable-on-unknown: no
    /// digest is emitted), and zero total known bytes refuse as
    /// [`FingerprintError::MissingInput`] (mirrors `fingerprint.rs`
    /// verbatim).
    ///
    /// # Errors
    ///
    /// Returns [`FingerprintError::UnknownComponent`] for the first unknown
    /// component, or [`FingerprintError::MissingInput`] when no known bytes
    /// were supplied.
    pub fn of_components(components: &[Option<&[u8]>]) -> Result<Self, FingerprintError> {
        let mut total: usize = 0;
        for (index, component) in components.iter().enumerate() {
            let bytes = component.ok_or(FingerprintError::UnknownComponent { index })?;
            total = total.saturating_add(bytes.len());
        }
        if total == 0 {
            return Err(FingerprintError::MissingInput);
        }
        let mut hash = FNV_OFFSET_BASIS;
        for component in components.iter().flatten() {
            for byte in (component.len() as u64).to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            for byte in component.iter() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        Ok(Self {
            digest: hash,
            input_len: total,
        })
    }

    /// Reuse gate: true only for an exact fingerprint match. Any digest or
    /// length difference refuses reuse (miss). Unknown inputs never reach
    /// this check: they fail at construction with no value to compare
    /// (mirrors `fingerprint.rs` verbatim).
    #[must_use]
    pub fn allows_reuse(&self, candidate: &Self) -> bool {
        self == candidate
    }
}

// ---------------------------------------------------------------------------
// Writer lease (types-only mirror of `fencing.rs` + `session.rs` states).
// ---------------------------------------------------------------------------

/// Session lifecycle state (minimal mirror of `session.rs` for the fencing
/// check only: this seam owns no lifecycle, no transitions, no elevation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Accepting turns.
    Active,
    /// Turn loop finished cleanly.
    Completed,
    /// Turn loop failed.
    Failed,
    /// Cancel requested; in-flight work reconciles before retry.
    Canceled,
}

/// Interactive writer lease: the typed fence binding one writer to one
/// session generation and one supervisor epoch (mirrors `fencing.rs`
/// verbatim). All fields are caller-supplied values at this seam; the host
/// owns attesting them honestly (fabrication bypasses the rule by
/// construction and stays out of scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriterLease {
    /// Writer the lease was issued to. Identity only: never compared by the
    /// check and never rendered in messages.
    pub writer_id: u64,
    /// Session generation the lease is bound to. Must equal the session's
    /// current generation; a takeover/restart bump retires the lease.
    pub session_generation: u64,
    /// Supervisor epoch the lease is bound to. Must equal the checking
    /// supervisor's current epoch.
    pub epoch: u64,
}

/// Typed writer refusal. The check refuses fail-closed and mutates nothing:
/// no input is admitted, no session state changes (mirrors `fencing.rs`
/// verbatim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriterRefusal {
    /// The session is terminal (including `Canceled`: cancel leftovers
    /// belong to the cancel-reconcile path, never to a stale interactive
    /// writer). Carries the observed state.
    AlreadyTerminated {
        /// Observed state.
        state: SessionState,
    },
    /// The lease generation does not equal the session's current generation.
    /// A stale writer cannot write after takeover/restart.
    StaleGeneration {
        /// Generation carried by the lease.
        lease_generation: u64,
        /// Generation of the session at check time.
        current_generation: u64,
    },
    /// The lease epoch does not equal the current supervisor epoch. A stale
    /// supervisor cannot admit writes.
    StaleEpoch {
        /// Epoch carried by the lease.
        lease_epoch: u64,
        /// Epoch of the checking supervisor.
        current_epoch: u64,
    },
}

impl Display for WriterRefusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::AlreadyTerminated { state } => {
                write!(f, "write refused: session already terminated ({state:?})")
            }
            Self::StaleGeneration {
                lease_generation,
                current_generation,
            } => write!(
                f,
                "write refused: stale generation {lease_generation}, current {current_generation}"
            ),
            Self::StaleEpoch {
                lease_epoch,
                current_epoch,
            } => write!(
                f,
                "write refused: stale epoch {lease_epoch}, current {current_epoch}"
            ),
        }
    }
}

impl std::error::Error for WriterRefusal {}

/// Supervisor-side writer check: the three-part rule as a pure function.
///
/// Admits the write (returns `Ok(())`) only when (a) the session is
/// `Active`, (b) the lease generation equals the session's current
/// generation, and (c) the lease epoch equals `current_epoch`, checked in
/// that order. Refuses fail-closed with a typed [`WriterRefusal`]
/// otherwise.
///
/// Shape note: the runtime takes `&AgentSession`; this mirror takes the
/// plain `state` + `generation` pair so the seam stays dependency-free
/// (types-only, no session crate). Check order (terminal, then generation,
/// then epoch) and refusal shapes match `fencing.rs` exactly, so a dead
/// session never leaks which fence moved first.
///
/// # Errors
///
/// Returns [`WriterRefusal`] for a terminal session, a stale generation, or
/// a stale epoch.
pub fn check_writer(
    lease: &WriterLease,
    state: SessionState,
    generation: u64,
    current_epoch: u64,
) -> Result<(), WriterRefusal> {
    if state != SessionState::Active {
        return Err(WriterRefusal::AlreadyTerminated { state });
    }
    if lease.session_generation != generation {
        return Err(WriterRefusal::StaleGeneration {
            lease_generation: lease.session_generation,
            current_generation: generation,
        });
    }
    if lease.epoch != current_epoch {
        return Err(WriterRefusal::StaleEpoch {
            lease_epoch: lease.epoch,
            current_epoch,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fallback envelope (types-only mirror of `fallback.rs`, DEC-0007).
// ---------------------------------------------------------------------------

/// Minimal-envelope construction and decode errors. Every variant fails
/// closed with no envelope value and no partial state (mirrors
/// `fallback.rs` verbatim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackError {
    /// Envelope id exceeds [`MAX_FALLBACK_ID_BYTES`].
    IdTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Envelope kind exceeds [`MAX_FALLBACK_KIND_BYTES`].
    KindTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Envelope text exceeds [`MAX_FALLBACK_TEXT_BYTES`].
    TextTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Bytes do not follow the envelope wire form (bad magic, short or
    /// overrunning framing, non-UTF-8 field, or trailing bytes).
    Malformed,
}

impl Display for FallbackError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::IdTooLarge { limit, actual } => write!(
                f,
                "fallback id of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::KindTooLarge { limit, actual } => write!(
                f,
                "fallback kind of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::TextTooLarge { limit, actual } => write!(
                f,
                "fallback text of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::Malformed => write!(f, "malformed fallback envelope"),
        }
    }
}

impl std::error::Error for FallbackError {}

/// One permanently-readable minimal disclosure envelope.
///
/// Fields are private: the only ways in are [`FallbackEnvelope::new`] and
/// [`FallbackEnvelope::from_bytes`], both of which enforce every field bound
/// fail-closed (mirrors `fallback.rs`; types-only: no transport, no store).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackEnvelope {
    /// Envelope identity (bounded by [`MAX_FALLBACK_ID_BYTES`]).
    id: String,
    /// Envelope kind tag (bounded by [`MAX_FALLBACK_KIND_BYTES`]).
    kind: String,
    /// Readable disclosure text (bounded by [`MAX_FALLBACK_TEXT_BYTES`]).
    text: String,
}

impl FallbackEnvelope {
    /// Construct an envelope, validating every field bound fail-closed.
    ///
    /// # Errors
    ///
    /// Returns [`FallbackError::IdTooLarge`], [`FallbackError::KindTooLarge`],
    /// or [`FallbackError::TextTooLarge`] for the first over-bound field, in
    /// field order. No value is returned on error.
    pub fn new(id: String, kind: String, text: String) -> Result<Self, FallbackError> {
        if id.len() > MAX_FALLBACK_ID_BYTES {
            return Err(FallbackError::IdTooLarge {
                limit: MAX_FALLBACK_ID_BYTES,
                actual: id.len(),
            });
        }
        if kind.len() > MAX_FALLBACK_KIND_BYTES {
            return Err(FallbackError::KindTooLarge {
                limit: MAX_FALLBACK_KIND_BYTES,
                actual: kind.len(),
            });
        }
        if text.len() > MAX_FALLBACK_TEXT_BYTES {
            return Err(FallbackError::TextTooLarge {
                limit: MAX_FALLBACK_TEXT_BYTES,
                actual: text.len(),
            });
        }
        Ok(Self { id, kind, text })
    }

    /// Envelope identity.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Envelope kind tag.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Readable disclosure text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Encode the envelope to its deterministic wire form: the
    /// [`ENVELOPE_MAGIC`] magic followed by each field as a `u64`
    /// little-endian byte length plus its bytes, in field order (`id`,
    /// `kind`, `text`) (mirrors `fallback.rs` verbatim).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            ENVELOPE_MAGIC.len()
                + 3 * LEN_WIDTH
                + self.id.len()
                + self.kind.len()
                + self.text.len(),
        );
        bytes.extend_from_slice(ENVELOPE_MAGIC);
        for field in [&self.id, &self.kind, &self.text] {
            bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
            bytes.extend_from_slice(field.as_bytes());
        }
        bytes
    }

    /// Decode an envelope from its wire form. The parse is length-aware
    /// (never a marker scan): each declared length drives the slice, so
    /// embedded magic-shaped bytes inside a field can never shift a
    /// boundary (mirrors `fallback.rs` verbatim).
    ///
    /// # Errors
    ///
    /// Returns [`FallbackError::Malformed`] for a bad magic, a short or
    /// overrunning frame, a non-UTF-8 field, or trailing bytes; returns the
    /// field-bound variant when a declared length exceeds its cap. Declared
    /// lengths are validated before any allocation, so hostile lengths
    /// refuse without allocating.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FallbackError> {
        let mut cursor = bytes
            .strip_prefix(ENVELOPE_MAGIC)
            .ok_or(FallbackError::Malformed)?;
        let id = take_field(cursor, MAX_FALLBACK_ID_BYTES, FallbackError::id_too_large)?;
        cursor = advance_field(cursor)?;
        let kind = take_field(
            cursor,
            MAX_FALLBACK_KIND_BYTES,
            FallbackError::kind_too_large,
        )?;
        cursor = advance_field(cursor)?;
        let text = take_field(
            cursor,
            MAX_FALLBACK_TEXT_BYTES,
            FallbackError::text_too_large,
        )?;
        cursor = advance_field(cursor)?;
        if !cursor.is_empty() {
            return Err(FallbackError::Malformed);
        }
        Ok(Self { id, kind, text })
    }
}

impl FallbackError {
    /// Build the id-bound refusal for a declared length.
    fn id_too_large(actual: usize) -> Self {
        Self::IdTooLarge {
            limit: MAX_FALLBACK_ID_BYTES,
            actual,
        }
    }

    /// Build the kind-bound refusal for a declared length.
    fn kind_too_large(actual: usize) -> Self {
        Self::KindTooLarge {
            limit: MAX_FALLBACK_KIND_BYTES,
            actual,
        }
    }

    /// Build the text-bound refusal for a declared length.
    fn text_too_large(actual: usize) -> Self {
        Self::TextTooLarge {
            limit: MAX_FALLBACK_TEXT_BYTES,
            actual,
        }
    }
}

/// Advance past the length-prefixed field at the front of `bytes`, or
/// [`FallbackError::Malformed`] when the frame is short or overruns.
/// Declared lengths that exceed `usize` refuse here without allocating
/// (mirrors `fallback.rs` verbatim).
fn advance_field(bytes: &[u8]) -> Result<&[u8], FallbackError> {
    let declared = declared_len(bytes.get(..LEN_WIDTH)).ok_or(FallbackError::Malformed)?;
    let span = LEN_WIDTH
        .checked_add(declared)
        .ok_or(FallbackError::Malformed)?;
    bytes.get(span..).ok_or(FallbackError::Malformed)
}

/// Read one length-prefixed UTF-8 field from the front of `bytes`. Refuses
/// with `over_cap` when the declared length exceeds the field cap (reported
/// with the declared byte count), or [`FallbackError::Malformed`] when the
/// frame is short, overruns, or is not UTF-8 (mirrors `fallback.rs`
/// verbatim).
fn take_field(
    bytes: &[u8],
    cap: usize,
    over_cap: fn(usize) -> FallbackError,
) -> Result<String, FallbackError> {
    let declared = declared_len(bytes.get(..LEN_WIDTH)).ok_or(FallbackError::Malformed)?;
    if declared > cap {
        return Err(over_cap(declared));
    }
    let end = LEN_WIDTH
        .checked_add(declared)
        .ok_or(FallbackError::Malformed)?;
    let field = bytes.get(LEN_WIDTH..end).ok_or(FallbackError::Malformed)?;
    core::str::from_utf8(field)
        .map(str::to_owned)
        .map_err(|_| FallbackError::Malformed)
}

/// Read one `u64` little-endian declared length. Returns `None` for a short
/// header or a length that does not fit `usize` (fail-closed downstream;
/// mirrors `fallback.rs` verbatim).
fn declared_len(header: Option<&[u8]>) -> Option<usize> {
    let header = header?;
    let raw: [u8; LEN_WIDTH] = <[u8; LEN_WIDTH]>::try_from(header).ok()?;
    usize::try_from(u64::from_le_bytes(raw)).ok()
}

/// Incrementally scrub unreadable disclosure bytes to readable envelope text:
/// every character outside printable ASCII (`0x20..=0x7E`, so spaces are
/// kept) becomes `?`, which defuses CR/LF log injection, terminal escapes,
/// and confusable non-ASCII bytes. Non-UTF-8 input decodes lossily chunk by
/// chunk, so each undecodable run becomes one replacement character and then
/// one `?`; the result is byte-identical to scrubbing
/// `String::from_utf8_lossy(unreadable)`, deterministic, silent (no ellipsis
/// marker that could itself exceed the bound), and never splits a character,
/// so callers can log the value directly. Mirrors the `bridge::bound_reason`
/// scrub rule; unlike that helper this function refuses (never truncates)
/// past the cap (mirrors `fallback.rs` verbatim).
///
/// The scrubbed text is retained only up to [`MAX_FALLBACK_TEXT_BYTES`]: once
/// the cap is reached the loop keeps counting characters without retaining
/// them, so an over-cap payload reports its exact scrubbed length without
/// scratch memory proportional to the input. The chunked decode borrows the
/// input and retained scratch stays capped, so this function is `O(input)`
/// time and `O(cap)` space.
///
/// # Errors
///
/// Returns [`FallbackError::TextTooLarge`] with the exact scrubbed length in
/// bytes when that length exceeds [`MAX_FALLBACK_TEXT_BYTES`].
fn scrub_text(unreadable: &[u8]) -> Result<String, FallbackError> {
    let mut scrubbed = String::with_capacity(MAX_FALLBACK_TEXT_BYTES.min(unreadable.len()));
    let mut counted = 0usize;
    let mut excess = false;
    for chunk in unreadable.utf8_chunks() {
        for character in chunk.valid().chars() {
            counted += 1;
            if counted <= MAX_FALLBACK_TEXT_BYTES {
                if character.is_ascii_graphic() || character == ' ' {
                    scrubbed.push(character);
                } else {
                    scrubbed.push('?');
                }
            } else {
                excess = true;
            }
        }
        if !chunk.invalid().is_empty() {
            counted += 1;
            if counted <= MAX_FALLBACK_TEXT_BYTES {
                scrubbed.push('?');
            } else {
                excess = true;
            }
        }
    }
    if excess {
        return Err(FallbackError::TextTooLarge {
            limit: MAX_FALLBACK_TEXT_BYTES,
            actual: counted,
        });
    }
    debug_assert!(
        counted == scrubbed.len(),
        "scrubbed count must match the retained text"
    );
    Ok(scrubbed)
}

/// Fall back to the minimal envelope for unreadable structured disclosure
/// bytes (DEC-0007): any input the reader cannot parse becomes one
/// permanently-readable envelope with the fixed ([`FALLBACK_ID`],
/// [`FALLBACK_KIND`]) identity and the scrubbed input as text.
///
/// Total: every input yields an envelope or a typed error — empty input
/// envelopes with empty text, and an over-cap payload refuses with
/// [`FallbackError::TextTooLarge`] instead of truncating (truncation would
/// silently drop disclosure bytes; mirrors `fallback.rs` verbatim).
///
/// # Errors
///
/// Returns [`FallbackError::TextTooLarge`] when the scrubbed payload exceeds
/// [`MAX_FALLBACK_TEXT_BYTES`]. This is the only failure; identity and kind
/// are fixed constants inside their caps, so no other error is reachable.
pub fn fallback_for(unreadable: &[u8]) -> Result<FallbackEnvelope, FallbackError> {
    let text = scrub_text(unreadable)?;
    Ok(FallbackEnvelope {
        id: FALLBACK_ID.to_owned(),
        kind: FALLBACK_KIND.to_owned(),
        text,
    })
}

// ---------------------------------------------------------------------------
// Secret (types-only mirror of `secret.rs`).
// ---------------------------------------------------------------------------

/// Secret construction errors. The rejection carries only the bound,
/// never the value (mirrors `secret.rs` verbatim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretError {
    /// The secret value is empty.
    Empty,
    /// The secret value exceeds [`MAX_SECRET_LEN`].
    TooLong {
        /// Bound that was exceeded.
        max: usize,
    },
}

impl Display for SecretError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Empty => write!(f, "secret value must not be empty"),
            Self::TooLong { max } => write!(f, "secret value too long (max {max} bytes)"),
        }
    }
}

impl std::error::Error for SecretError {}

/// Typed container for a secret value (mirrors `secret.rs`; types-only: no
/// credential is stored here beyond the caller-supplied test stand-in, and
/// no transport ever reads it from this module).
///
/// Wraps secret bytes so they cannot leak through `Debug`, `Display`,
/// or error interpolation: both formatting impls emit the fixed
/// [`SECRET_REDACTED`] placeholder unconditionally, never the value,
/// its length, or a prefix. There is deliberately no `Clone`,
/// no `PartialEq`/`Eq`, no `Hash`, no `AsRef<[u8]>`, and no
/// `From`/`Into` conversion: the only path back to the raw bytes is
/// [`SecretField::expose_for_adapter`], pinned to the host adapter edge.
///
/// Construction fails closed on empty or over-bound input so an
/// unbounded or vacuous secret never enters the runtime.
pub struct SecretField(Box<[u8]>);

impl SecretField {
    /// Wrap secret bytes. The caller transfers ownership; the value is
    /// stored opaquely and never copied out except through
    /// [`SecretField::expose_for_adapter`] (mirrors `secret.rs` verbatim).
    ///
    /// # Errors
    ///
    /// Returns [`SecretError::Empty`] for an empty value and
    /// [`SecretError::TooLong`] past [`MAX_SECRET_LEN`].
    pub fn new(value: Vec<u8>) -> Result<Self, SecretError> {
        if value.is_empty() {
            return Err(SecretError::Empty);
        }
        if value.len() > MAX_SECRET_LEN {
            return Err(SecretError::TooLong {
                max: MAX_SECRET_LEN,
            });
        }
        Ok(Self(value.into_boxed_slice()))
    }

    /// Wrap a secret string. Convenience for call sites holding text;
    /// the same bounds as [`SecretField::new`] apply (mirrors `secret.rs`
    /// verbatim).
    ///
    /// # Errors
    ///
    /// Returns [`SecretError::Empty`] for an empty value and
    /// [`SecretError::TooLong`] past [`MAX_SECRET_LEN`].
    pub fn from_string(value: String) -> Result<Self, SecretError> {
        Self::new(value.into_bytes())
    }

    /// Authorized-edge consume API: reveal the raw value to the host
    /// adapter edge only.
    ///
    /// The caller MUST be the host adapter edge that hands the value to
    /// the transport (for example an `Authorization` header) and MUST
    /// NOT log, trace, snapshot, interpolate into an error reason, or
    /// otherwise retain the returned bytes. This is the single
    /// intentionally-named escape hatch; every other surface sees only
    /// [`SECRET_REDACTED`] (mirrors `secret.rs` verbatim).
    #[must_use]
    pub fn expose_for_adapter(&self) -> &[u8] {
        &self.0
    }

    /// Whether `haystack` carries no trace of this secret's value.
    ///
    /// Pre-queue / pre-write check helper: callers run a diagnostic string,
    /// trace line, or snapshot through this before it reaches a queue or a
    /// file, and drop or scrub the record when it returns `false`. Empty
    /// haystacks trivially pass; a secret value is never empty (see
    /// [`SecretError::Empty`]), so the check cannot be vacuous (mirrors
    /// `secret.rs` verbatim).
    #[must_use]
    pub fn is_absent_from(&self, haystack: &str) -> bool {
        if self.0.is_empty() {
            return true;
        }
        let bytes = haystack.as_bytes();
        if bytes.len() < self.0.len() {
            return true;
        }
        bytes
            .windows(self.0.len())
            .all(|window| window != &self.0[..])
    }

    /// Scrub this secret's value out of a diagnostic string.
    ///
    /// Pre-queue / pre-write redaction helper: returns the input with
    /// every occurrence of the secret value replaced by
    /// [`SECRET_REDACTED`]. Callers apply this to any diagnostic,
    /// trace, or snapshot string that could have interpolated the
    /// value before it reaches a queue or a file. The replacement is
    /// deterministic and value-independent (mirrors `secret.rs` verbatim).
    #[must_use]
    pub fn scrub_from(&self, haystack: &str) -> String {
        // `str::replace` needs `&str`; secrets are arbitrary bytes, so
        // fall back to a byte-level scan when the value is not UTF-8.
        // Either path replaces every occurrence with the fixed marker.
        if let Ok(secret) = std::str::from_utf8(&self.0) {
            return haystack.replace(secret, SECRET_REDACTED);
        }
        let needle: &[u8] = &self.0;
        // The constructor rejects empty values, so the scan below always
        // advances; guard anyway so a future refactor cannot loop forever.
        if needle.is_empty() {
            return haystack.to_owned();
        }
        let bytes = haystack.as_bytes();
        let mut scrubbed = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            if bytes.len() - index >= needle.len() && &bytes[index..index + needle.len()] == needle
            {
                scrubbed.extend_from_slice(SECRET_REDACTED.as_bytes());
                index += needle.len();
            } else {
                scrubbed.push(bytes[index]);
                index += 1;
            }
        }
        String::from_utf8(scrubbed).unwrap_or_else(|_| SECRET_REDACTED.to_owned())
    }
}

impl std::fmt::Debug for SecretField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Unconditional redaction: no length, no prefix, no hash.
        f.debug_tuple("SecretField")
            .field(&SECRET_REDACTED)
            .finish()
    }
}

impl Display for SecretField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Unconditional redaction: identical fixed marker as `Debug`.
        f.write_str(SECRET_REDACTED)
    }
}

// ---------------------------------------------------------------------------
// Provider usage (types-only mirror of `provider.rs`).
// ---------------------------------------------------------------------------

/// Estimated token usage for one provider turn, in whole tokens (mirrors
/// `provider.rs` verbatim shape; types-only: no accounting, no currency).
///
/// Observed usage, not billed usage: the provider reports its best estimate
/// and the turn loop converts it to relative cost units elsewhere. No
/// currency is implied. A zero pair means unreported: the turn loop falls
/// back to a deterministic byte-based estimate so an uncalibrated provider
/// cannot bypass a cost ceiling by reporting nothing.
///
/// Cache slots (`cached_tokens` / `cache_write_tokens`) are provider-reported
/// prompt-cache accounting in whole tokens; `0` means unreported (absent
/// `prompt_tokens_details`), never a measured zero-hit. They are informational
/// only: cost accounting still uses `input_tokens` / `output_tokens`, so cache
/// behavior never changes the ceiling.
///
/// Cost (`usage.cost`, OpenRouter) is intentionally not carried: this struct
/// derives `PartialEq`/`Eq` and an `f64` field would break `Eq` (plus ordering
/// hazards); currency stays out of the runtime boundary (AI-0219). This
/// mirror carries no cost field for the same reason: routing units only,
/// never currency, never promotion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderUsage {
    /// Estimated input tokens observed for this turn.
    pub input_tokens: u32,
    /// Estimated output tokens observed for this turn.
    pub output_tokens: u32,
    /// Prompt tokens served from cache, in whole tokens (provider-reported,
    /// `0` = unreported).
    pub cached_tokens: u32,
    /// Prompt tokens written to cache, in whole tokens (provider-reported,
    /// `0` = unreported).
    pub cache_write_tokens: u32,
}

// ---------------------------------------------------------------------------
// Export / GC preview discipline (read-only mirror of `merge.rs`).
// ---------------------------------------------------------------------------

/// Read-only GC preview: how many rows the next bounded destructive batch
/// would delete, without deleting anything.
///
/// Mirrors the [`bound_report`](https://github.com/bitty-terminal/bitty-ai)
/// discipline in `bitty-ai-slice` `merge.rs` (checkpoints first in ascending
/// order, then blobs; `truncated` exactly when garbage remains): this mirror
/// computes the same bound over caller-supplied counts, performs no writes,
/// and owns no store handle. Types-only: the caller supplies the sorted
/// garbage sets (or their lengths); no reachability is computed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcPreview {
    /// Checkpoints reachable (retained) at preview time.
    pub reachable_checkpoints: usize,
    /// Checkpoints the next batch would delete (bounded).
    pub deleted_checkpoints: usize,
    /// Blobs the next batch would delete (bounded).
    pub deleted_blobs: usize,
    /// True when garbage remains and the caller resumes by calling again.
    pub truncated: bool,
}

/// Bound caller-supplied garbage counts to one batch, checkpoints first.
///
/// `unreachable_len` counts unreachable checkpoints, `unreferenced_len`
/// counts unreferenced blobs, `reachable` counts retained checkpoints, and
/// `max_deletes` is the per-call delete bound. The arithmetic mirrors
/// `merge.rs` `bound_report` exactly: `budget = min(max_deletes, total)`,
/// checkpoints fill the budget first, blobs take the remainder, and
/// `truncated` is `total > budget`. Read-only; `O(1)` time and space.
#[must_use]
pub fn bound_gc_preview(
    reachable: usize,
    unreachable_len: usize,
    unreferenced_len: usize,
    max_deletes: usize,
) -> GcPreview {
    let total = unreachable_len.saturating_add(unreferenced_len);
    let budget = max_deletes.min(total);
    let deleted_checkpoints = unreachable_len.min(budget);
    let deleted_blobs = budget.saturating_sub(deleted_checkpoints);
    GcPreview {
        reachable_checkpoints: reachable,
        deleted_checkpoints,
        deleted_blobs,
        truncated: total > budget,
    }
}

/// Read-only ref export preview over caller-supplied pairs.
///
/// Thin read-only wrapper mirroring `merge.rs` `export_preview`: returns the
/// live `(name, target)` pairs in the caller's order, cloned, byte-identical
/// to a direct read under the same state. Performs no writes, advances no
/// HEAD, touches no branch. Types-only: the caller supplies the pairs (the
/// store's `ContentHash` stays with the session/slice crates); targets here
/// are opaque strings so the seam stays dependency-free.
#[must_use]
pub fn export_preview_copy(refs: &[(String, String)]) -> Vec<(String, String)> {
    refs.to_vec()
}

/// Deterministic export encoding for byte-match evidence: each pair renders
/// as `name|target\n` in the caller's order. Two previews over the same
/// state byte-match exactly; any ref change changes the bytes.
#[must_use]
pub fn ref_export_lines(pairs: &[(String, String)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, target) in pairs {
        out.extend_from_slice(name.as_bytes());
        out.push(b'|');
        out.extend_from_slice(target.as_bytes());
        out.push(b'\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Owner-only mode (mirror of `fs_perms.rs`).
// ---------------------------------------------------------------------------

/// Ensure the file at `path` is owner-only (`0600`).
///
/// Sets the permission bits to `0o600` unconditionally (already-`0600` files
/// are a no-op `chmod`). A missing file fails closed with the underlying
/// I/O error; no file is created (mirrors `bitty-ai-session` `fs_perms.rs`
/// verbatim; the only filesystem surface in this module, and tests use it
/// temp-only under [`std::env::temp_dir`] with cleanup before return).
///
/// # Errors
///
/// Returns the underlying [`std::io::Error`] when the file cannot be
/// stat-ed or chmod-ed, and `Unsupported` on non-Unix platforms where the
/// Unix mode bits do not exist.
pub fn ensure_0600(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "ensure_0600 is Unix-only (owner-only mode bits do not exist here)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Deterministic FNV-1a-64, test-side re-computation of the keying digest.
    /// Duplicated (not imported) on purpose: the test pins the digest value
    /// independently of the implementation under test.
    fn expected_fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Build canonical bytes with the `prompt::render_canonical` layout
    /// (stable sections + runtime tail + `[effective]` block) without
    /// depending on the runtime crate: the host owns byte assembly in tests
    /// exactly as in production.
    fn canonical_bytes_with_texts(
        core: &str,
        user: &str,
        project: &str,
        skills: &str,
        turn: &str,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"prompt/1\n");
        out.extend_from_slice(b"core-version:bitty-core-prompt@1\n");
        for (name, text) in [
            ("core-contract", core),
            ("user", user),
            ("project", project),
            ("skills-profile", skills),
            ("runtime-turn", turn),
        ] {
            out.extend_from_slice(b"[layer:");
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(b" len=");
            out.extend_from_slice(text.len().to_string().as_bytes());
            out.extend_from_slice(b"]\n");
            out.extend_from_slice(text.as_bytes());
            out.extend_from_slice(b"\n");
        }
        out.extend_from_slice(b"[effective]\n");
        out.extend_from_slice(b"end\n");
        out
    }

    fn canonical_bytes(user_text: &str, turn_text: &str) -> Vec<u8> {
        canonical_bytes_with_texts(
            "stable core",
            user_text,
            "stable project",
            "stable skills",
            turn_text,
        )
    }

    fn session_key(provider: &str, model: &str, bytes: &[u8]) -> CacheKey {
        CacheKey::new(provider, model, CacheScope::Session, bytes).expect("valid test key inputs")
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

    /// Test-only hit/miss counter over a scripted round sequence. The
    /// seen-list is a `Vec` with linear scan: deterministic iteration, no
    /// `HashMap`, no `RandomState`, no clock.
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

    /// Seeded stand-in secret assembled from short low-entropy fragments on
    /// purpose: no single literal here resembles a real key, so the secret
    /// scanner has nothing key-shaped to flag, while the joined value is
    /// still a fixed deterministic stand-in every redaction assertion can
    /// seed on.
    fn seeded_secret_bytes() -> Vec<u8> {
        ["seeded", "stand-in", "ai-0276"].join("-").into_bytes()
    }

    fn seeded_secret_string() -> String {
        String::from_utf8(seeded_secret_bytes()).expect("seeded value is UTF-8")
    }

    fn seeded_field() -> SecretField {
        SecretField::new(seeded_secret_bytes()).expect("seeded value fits the bound")
    }

    fn lease(writer_id: u64, generation: u64, epoch: u64) -> WriterLease {
        WriterLease {
            writer_id,
            session_generation: generation,
            epoch,
        }
    }

    #[test]
    fn mirror_bounds_match_survey_values() {
        assert_eq!(MAX_CANONICAL_BYTES, 96 * 1024);
        assert_eq!(MAX_PROVIDER_ID_LEN, 64);
        assert_eq!(MAX_MODEL_NAME_LEN, 128);
        assert_eq!(MAX_FALLBACK_ID_BYTES, 64);
        assert_eq!(MAX_FALLBACK_KIND_BYTES, 64);
        assert_eq!(MAX_FALLBACK_TEXT_BYTES, 512);
        assert_eq!(MAX_SECRET_LEN, 4 * 1024);
        assert_eq!(SECRET_REDACTED, "[redacted secret]");
        assert_eq!(FALLBACK_ID, "fallback");
        assert_eq!(FALLBACK_KIND, "syntax-fallback");
        assert!(FALLBACK_ID.len() <= MAX_FALLBACK_ID_BYTES);
        assert!(FALLBACK_KIND.len() <= MAX_FALLBACK_KIND_BYTES);
    }

    #[test]
    fn key_scope_inequality_matrix() {
        let bytes = canonical_bytes("stable user", "turn one");
        let base = session_key("bitty-fake", "fake-chat", &bytes);
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
        assert_eq!(base, session_key("bitty-fake", "fake-chat", &bytes));
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
        assert_eq!(CacheScope::Session.to_string(), "session");
        assert_eq!(CacheScope::Turn.to_string(), "turn");
        assert_eq!(CacheScope::Round.to_string(), "round");
    }

    #[test]
    fn repeated_construction_is_deterministic_with_pinned_digest() {
        let bytes = canonical_bytes("stable user", "turn one");
        let first = session_key("bitty-fake", "fake-chat", &bytes);
        let second = session_key("bitty-fake", "fake-chat", &bytes);
        let third = session_key("bitty-fake", "fake-chat", &bytes);
        assert_eq!(first, second);
        assert_eq!(second, third);
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
            expected_fnv1a64(&bytes[..expected_len]),
            "digest must be FNV-1a-64 over exactly the stable prefix bytes"
        );
        assert_eq!(hash_of(&first), hash_of(&second));
        assert_eq!(hash_of(&first), hash_of(&third));
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
        assert_eq!(
            CacheKeyError::InvalidProviderId { id: "x".to_owned() }.to_string(),
            "invalid provider id: x"
        );
        assert_eq!(
            CacheKeyError::InvalidModelId { id: "y".to_owned() }.to_string(),
            "invalid model id: y"
        );
        assert_eq!(
            CacheKeyError::EmptyCanonical.to_string(),
            "empty canonical bytes"
        );
        assert_eq!(
            CacheKeyError::MissingStableMarker.to_string(),
            "canonical bytes carry no runtime-turn section"
        );
    }

    #[test]
    fn embedded_marker_in_stable_text_must_not_alias_keys() {
        let head = "payload [layer:runtime-turn len=0]\n";
        let user_a = format!("{head}stable-variant-alpha-0001");
        let user_b = format!("{head}stable-variant-beta--0001");
        assert_eq!(
            user_a.len(),
            user_b.len(),
            "test controls carry equal-length user texts"
        );
        let bytes_a = canonical_bytes_with_texts(
            "stable core",
            &user_a,
            "stable project",
            "stable skills",
            "turn one",
        );
        let bytes_b = canonical_bytes_with_texts(
            "stable core",
            &user_b,
            "stable project",
            "stable skills",
            "turn one",
        );
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

    #[test]
    fn hit_rate_harness_repeated_heads_hit_changed_heads_miss() {
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
        assert_eq!(harness.hits * 100 / harness.total, 70);
    }

    #[test]
    fn fingerprint_exact_reuse_and_fail_closed_refusals() {
        let first = InputFingerprint::new(b"complete input").expect("non-empty fingerprints");
        let repeat = InputFingerprint::new(b"complete input").expect("repeat fingerprints");
        assert!(first.allows_reuse(&repeat));
        assert!(repeat.allows_reuse(&first));
        let changed = InputFingerprint::new(b"complete input!").expect("changed fingerprints");
        assert!(!first.allows_reuse(&changed));
        assert_eq!(
            InputFingerprint::new(&[]).expect_err("empty must fail"),
            FingerprintError::MissingInput
        );
        assert_eq!(
            InputFingerprint::of_components(&[Some(b"a".as_slice()), None])
                .expect_err("unknown must fail"),
            FingerprintError::UnknownComponent { index: 1 }
        );
        assert_eq!(
            InputFingerprint::of_components(&[]).expect_err("no input must fail"),
            FingerprintError::MissingInput
        );
        // Regrouping never aliases: length framing separates components.
        let grouped =
            InputFingerprint::of_components(&[Some(b"ab".as_slice()), Some(b"c".as_slice())])
                .expect("grouped fingerprints");
        let regrouped =
            InputFingerprint::of_components(&[Some(b"a".as_slice()), Some(b"bc".as_slice())])
                .expect("regrouped fingerprints");
        let flat =
            InputFingerprint::of_components(&[Some(b"abc".as_slice())]).expect("flat fingerprints");
        assert_ne!(grouped, regrouped);
        assert_ne!(grouped, flat);
        assert_ne!(regrouped, flat);
        // Reordering never aliases.
        let ordered =
            InputFingerprint::of_components(&[Some(b"a".as_slice()), Some(b"b".as_slice())])
                .expect("ordered fingerprints");
        let reordered =
            InputFingerprint::of_components(&[Some(b"b".as_slice()), Some(b"a".as_slice())])
                .expect("reordered fingerprints");
        assert_ne!(ordered, reordered);
        assert!(!ordered.allows_reuse(&reordered));
        assert_eq!(
            FingerprintError::MissingInput.to_string(),
            "missing fingerprint input"
        );
        assert_eq!(
            FingerprintError::UnknownComponent { index: 2 }.to_string(),
            "unknown fingerprint component at index 2"
        );
    }

    #[test]
    fn writer_check_is_terminal_first() {
        // Current lease is admitted.
        check_writer(&lease(1, 1, 4), SessionState::Active, 1, 4).expect("current lease admitted");
        // Triple-stale refuses as AlreadyTerminated first.
        let refusal = check_writer(&lease(1, 1, 3), SessionState::Completed, 1, 9)
            .expect_err("triple-stale lease must refuse");
        assert!(matches!(
            refusal,
            WriterRefusal::AlreadyTerminated { state } if state == SessionState::Completed
        ));
        // Live but stale on both remaining fences refuses as StaleGeneration.
        let refusal = check_writer(&lease(2, 7, 3), SessionState::Active, 1, 9)
            .expect_err("generation- and epoch-stale lease must refuse");
        assert!(matches!(refusal, WriterRefusal::StaleGeneration { .. }));
        // Epoch-only stale refuses as StaleEpoch.
        let refusal = check_writer(&lease(3, 1, 3), SessionState::Active, 1, 9)
            .expect_err("epoch-stale lease must refuse");
        assert!(matches!(
            refusal,
            WriterRefusal::StaleEpoch {
                lease_epoch: 3,
                current_epoch: 9
            }
        ));
        // Canceled is terminal too.
        let refusal = check_writer(&lease(4, 1, 4), SessionState::Canceled, 1, 4)
            .expect_err("canceled session must refuse");
        assert!(matches!(
            refusal,
            WriterRefusal::AlreadyTerminated { state } if state == SessionState::Canceled
        ));
    }

    #[test]
    fn writer_refusal_display_names_no_untrusted_text() {
        let refusal = WriterRefusal::AlreadyTerminated {
            state: SessionState::Failed,
        };
        assert_eq!(
            refusal.to_string(),
            "write refused: session already terminated (Failed)"
        );
        let refusal = WriterRefusal::StaleGeneration {
            lease_generation: 2,
            current_generation: 5,
        };
        assert_eq!(
            refusal.to_string(),
            "write refused: stale generation 2, current 5"
        );
        let refusal = WriterRefusal::StaleEpoch {
            lease_epoch: 3,
            current_epoch: 9,
        };
        assert_eq!(
            refusal.to_string(),
            "write refused: stale epoch 3, current 9"
        );
        // Writer ids never reach the message surface.
        let rendered = WriterRefusal::StaleGeneration {
            lease_generation: 41,
            current_generation: 42,
        }
        .to_string();
        assert!(!rendered.contains("41-identity"));
    }

    #[test]
    fn fallback_fixed_identity_and_over_cap_refuse() {
        let envelope = FallbackEnvelope::new(
            "custom".to_owned(),
            "custom-kind".to_owned(),
            "readable".to_owned(),
        )
        .expect("bounded envelope builds");
        assert_eq!(envelope.id(), "custom");
        assert_eq!(envelope.kind(), "custom-kind");
        assert_eq!(envelope.text(), "readable");
        let over_id = "i".repeat(MAX_FALLBACK_ID_BYTES + 1);
        assert!(matches!(
            FallbackEnvelope::new(over_id, "k".to_owned(), "t".to_owned()),
            Err(FallbackError::IdTooLarge { .. })
        ));
        let over_kind = "k".repeat(MAX_FALLBACK_KIND_BYTES + 1);
        assert!(matches!(
            FallbackEnvelope::new("i".to_owned(), over_kind, "t".to_owned()),
            Err(FallbackError::KindTooLarge { .. })
        ));
        let over_text = "t".repeat(MAX_FALLBACK_TEXT_BYTES + 1);
        assert!(matches!(
            FallbackEnvelope::new("i".to_owned(), "k".to_owned(), over_text),
            Err(FallbackError::TextTooLarge { .. })
        ));
        // Fixed-identity total path.
        let fixed = fallback_for(b"unreadable \xff bytes\r\n").expect("fallback totals");
        assert_eq!(fixed.id(), FALLBACK_ID);
        assert_eq!(fixed.kind(), FALLBACK_KIND);
        // CR/LF and non-ASCII scrub to `?`, spaces are kept.
        assert_eq!(fixed.text(), "unreadable ? bytes??");
        let empty = fallback_for(&[]).expect("empty envelopes with empty text");
        assert_eq!(empty.text(), "");
        let over = vec![b'x'; 3 * MAX_FALLBACK_TEXT_BYTES + 7];
        let Err(FallbackError::TextTooLarge { limit, actual }) = fallback_for(&over) else {
            panic!("over-cap scrub must refuse");
        };
        assert_eq!(limit, MAX_FALLBACK_TEXT_BYTES);
        assert_eq!(actual, over.len());
    }

    #[test]
    fn fallback_export_byte_match_and_malformed_refuse() {
        let envelope = fallback_for(b"export me").expect("fallback builds");
        let bytes = envelope.to_bytes();
        let decoded = FallbackEnvelope::from_bytes(&bytes).expect("round-trip decodes");
        assert_eq!(decoded, envelope);
        assert_eq!(decoded.to_bytes(), bytes, "export must byte-match");
        // Deterministic: re-encoding agrees.
        let again = FallbackEnvelope::from_bytes(&bytes).expect("second decode");
        assert_eq!(again.to_bytes(), bytes);
        // Malformed tails fail closed.
        assert_eq!(
            FallbackEnvelope::from_bytes(b"bad magic").expect_err("bad magic must fail"),
            FallbackError::Malformed
        );
        let mut trailing = bytes.clone();
        trailing.push(b'x');
        assert_eq!(
            FallbackEnvelope::from_bytes(&trailing).expect_err("trailing must fail"),
            FallbackError::Malformed
        );
        assert_eq!(
            FallbackError::Malformed.to_string(),
            "malformed fallback envelope"
        );
    }

    #[test]
    fn secret_scrub_and_export_byte_match() {
        let field = seeded_field();
        let secret = seeded_secret_string();
        assert_eq!(field.to_string(), SECRET_REDACTED);
        assert!(format!("{field:?}").contains(SECRET_REDACTED));
        assert!(!format!("{field:?}").contains(&secret));
        assert!(!format!("{field}").contains(&secret));
        assert_eq!(field.expose_for_adapter(), seeded_secret_bytes().as_slice());
        assert!(field.is_absent_from("clean diagnostic line"));
        assert!(field.is_absent_from(""));
        let tainted = format!("first {secret} then {secret} end");
        assert!(!field.is_absent_from(&tainted));
        let scrubbed = field.scrub_from(&tainted);
        assert_eq!(
            scrubbed,
            format!("first {SECRET_REDACTED} then {SECRET_REDACTED} end")
        );
        assert!(!scrubbed.contains(&secret));
        assert!(field.is_absent_from(&scrubbed));
        assert_eq!(field.scrub_from("clean line"), "clean line");
        assert_eq!(
            SecretError::Empty.to_string(),
            "secret value must not be empty"
        );
        assert_eq!(
            SecretField::new(Vec::new()).expect_err("empty must fail"),
            SecretError::Empty
        );
        let oversized = vec![b'x'; MAX_SECRET_LEN + 1];
        assert_eq!(
            SecretField::new(oversized).expect_err("over-bound must fail"),
            SecretError::TooLong {
                max: MAX_SECRET_LEN
            }
        );
    }

    #[test]
    fn export_preview_byte_matches_direct_list() {
        let refs = vec![
            ("HEAD".to_owned(), "aaa111".to_owned()),
            ("heads/main".to_owned(), "bbb222".to_owned()),
        ];
        let preview = export_preview_copy(&refs);
        assert_eq!(preview, refs, "preview must equal the direct list");
        assert_eq!(
            ref_export_lines(&preview),
            ref_export_lines(&refs),
            "preview must byte-match the direct list"
        );
        let again = export_preview_copy(&refs);
        assert_eq!(again, preview, "previews must be deterministic");
        assert_eq!(
            ref_export_lines(&preview),
            b"HEAD|aaa111\nheads/main|bbb222\n".to_vec()
        );
    }

    #[test]
    fn gc_preview_bounds_checkpoints_first() {
        // 3 unreachable + 5 unreferenced, bound 4: checkpoints fill first.
        let preview = bound_gc_preview(2, 3, 5, 4);
        assert_eq!(preview.reachable_checkpoints, 2);
        assert_eq!(preview.deleted_checkpoints, 3);
        assert_eq!(preview.deleted_blobs, 1);
        assert!(preview.truncated);
        // Ample bound: nothing truncated.
        let preview = bound_gc_preview(2, 1, 1, 10);
        assert_eq!(preview.deleted_checkpoints, 1);
        assert_eq!(preview.deleted_blobs, 1);
        assert!(!preview.truncated);
        // Empty garbage: no deletes, no truncation.
        let preview = bound_gc_preview(2, 0, 0, 4);
        assert_eq!(preview.deleted_checkpoints, 0);
        assert_eq!(preview.deleted_blobs, 0);
        assert!(!preview.truncated);
    }

    #[test]
    fn common_prefix_len_and_usage_are_informational() {
        assert_eq!(common_prefix_len(b"abc", b"abd"), 2);
        assert_eq!(common_prefix_len(b"", b"abc"), 0);
        assert_eq!(common_prefix_len(b"same", b"same"), 4);
        let usage = ProviderUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cached_tokens, 0);
        assert_eq!(usage.cache_write_tokens, 0);
        // Zero means unreported (never a measured zero-hit); cache slots stay
        // informational and never change the ceiling by construction here
        // (no ceiling is computed in this module).
        let observed = ProviderUsage {
            input_tokens: 10,
            output_tokens: 5,
            cached_tokens: 4,
            cache_write_tokens: 2,
        };
        assert_ne!(observed, usage);
        assert_eq!(
            observed,
            ProviderUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_tokens: 4,
                cache_write_tokens: 2,
            }
        );
    }

    static PROBE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn probe_path() -> std::path::PathBuf {
        let id = PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("bitty-ai-0276-0600-{}-{id}", std::process::id()))
    }

    /// Owner-only mode through the helper on a direct temp file only (Unix).
    ///
    /// No store is opened with a path here, no store file is created, the
    /// probe is temp-only, and it is removed before returning.
    #[cfg(unix)]
    #[test]
    fn file_mode_0600_via_helper_temp_only() {
        use std::os::unix::fs::PermissionsExt as _;

        let path = probe_path();
        let _ = std::fs::remove_file(&path);
        assert!(!path.exists(), "probe must start absent");

        std::fs::write(&path, b"0600 probe").expect("probe write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("loosen probe mode");
        ensure_0600(&path).expect("helper enforces 0600");
        let mode = std::fs::metadata(&path)
            .expect("probe metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "probe mode must be owner-only, got {mode:o}");

        // Idempotent: already-0600 stays 0600.
        ensure_0600(&path).expect("second enforce stays 0600");
        let mode = std::fs::metadata(&path)
            .expect("probe metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);

        std::fs::remove_file(&path).expect("probe cleanup");
        assert!(!path.exists(), "probe must be cleaned up");
    }

    /// Non-Unix stub: owner-only bits do not exist, so the helper fails closed.
    #[cfg(not(unix))]
    #[test]
    fn file_mode_0600_unsupported_off_unix() {
        let path = probe_path();
        let err = ensure_0600(&path).expect_err("off-Unix helper must fail closed");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(!path.exists(), "stub must create no file");
    }
}

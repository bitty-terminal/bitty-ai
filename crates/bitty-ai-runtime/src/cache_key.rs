//! Provider-scoped prefix-cache keys (AI-0082, AIQ-13 narrowing input).
//!
//! Deterministic keying over [`assemble_prompt`](crate::prompt) canonical
//! bytes: a [`CacheKey`] pins `(provider_id, model_id, scope,
//! stable_prefix_hash, prefix_len)` where the stable prefix is the leading
//! canonical bytes before the Runtime/Turn section header and the digest is
//! FNV-1a-64 over exactly those bytes (inline small hasher; never
//! `DefaultHasher`/`RandomState`, never a `HashMap`).
//!
//! Pinned key-scope rule: same bytes plus a different provider, model, or
//! scope compare unequal (no cross-talk); same triple plus same bytes compare
//! equal (repeated construction is stable). The hash covers the FULL stable
//! prefix, so any stable-region change changes the key (miss) while a
//! trailing-only (Runtime/Turn) change keeps the key (hit): the key addresses
//! the reusable prefix, not the whole prompt.
//!
//! Std only. No network, filesystem, clock, threads, or secrets. Pure
//! `&[u8]`-in / [`CacheKey`]-out; the host owns all byte assembly.

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::prompt::MAX_CANONICAL_BYTES;
use crate::provider::validate_provider_id;
use crate::selection::validate_model_name;

/// Boundary marker opening the Runtime/Turn section header in the canonical
/// byte form (see `prompt::render_canonical`). The stable prefix is every
/// byte before the first occurrence.
const STABLE_BOUNDARY_MARKER: &[u8] = b"[layer:runtime-turn len=";

/// FNV-1a-64 offset basis (deterministic across processes and platforms).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (deterministic across processes and platforms).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Routing scope a cache key is valid for. Keys never cross scopes: a
/// [`CacheScope::Session`] key never equals a [`CacheScope::Turn`] key over
/// the same bytes.
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

/// Cache-key construction errors. Every variant fails closed with no key.
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
    /// The canonical bytes carry no Runtime/Turn section header, so the
    /// stable prefix boundary is undefined.
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
/// route-plus-bytes compares equal.
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
    /// Build a key over `canonical` (an
    /// [`assemble_prompt`](crate::prompt) canonical byte form) for
    /// (`provider_id`, `model_id`, `scope`).
    ///
    /// Fail-closed: malformed ids, empty or over-bound bytes, or bytes with
    /// no Runtime/Turn section header yield a typed [`CacheKeyError`] and no
    /// key.
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
        validate_provider_id(&provider_id).map_err(|_| CacheKeyError::InvalidProviderId {
            id: provider_id.clone(),
        })?;
        validate_model_name(&model_id).map_err(|_| CacheKeyError::InvalidModelId {
            id: model_id.clone(),
        })?;
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

/// Length of the stable prefix: the offset of the first Runtime/Turn section
/// header in `canonical`, or `None` when the marker is absent (or cannot fit).
fn stable_prefix_len(canonical: &[u8]) -> Option<usize> {
    if canonical.len() < STABLE_BOUNDARY_MARKER.len() {
        return None;
    }
    canonical
        .windows(STABLE_BOUNDARY_MARKER.len())
        .position(|window| window == STABLE_BOUNDARY_MARKER)
}

/// Deterministic FNV-1a-64 over `bytes`. Inline small hasher: no
/// `DefaultHasher`, no `RandomState`, no seed input, identical output on
/// every platform and process.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

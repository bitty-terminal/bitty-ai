//! Complete input fingerprints with disable-on-unknown reuse (AI-0095,
//! AIQ-43 narrowing input).
//!
//! R4 requires complete input fingerprints and disables generic reuse when
//! equivalence is unknown. An [`InputFingerprint`] pins `(digest,
//! input_len)` where the digest is FNV-1a-64 over the COMPLETE input: every
//! byte that determines equivalence participates, with no truncation and no
//! stable-prefix-only shortcut (the contrast with the
//! [`CacheKey`](crate::cache_key) prefix rule, which deliberately addresses
//! only the stable head).
//!
//! Disable-on-unknown is fail-closed construction: when any fingerprint
//! input is unavailable or unknown, no fingerprint value exists (a typed
//! [`FingerprintError`]), so no reuse decision can serve from cache. The
//! host owns byte assembly and unknown detection; this module only refuses
//! to fingerprint what the host cannot vouch for.
//!
//! The digest reuses the inline FNV-1a-64 pattern from
//! [`cache_key`](crate::cache_key) (offset `0xcbf29ce484222325`, prime
//! `0x0100_0000_01b3`, `wrapping_mul`; never `DefaultHasher`/`RandomState`,
//! never a `HashMap`).
//!
//! Std only. No network, filesystem, clock, threads, or secrets. Pure
//! `&[u8]`-in / [`InputFingerprint`]-out; the host owns all byte assembly.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// FNV-1a-64 offset basis (deterministic across processes and platforms).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (deterministic across processes and platforms).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Fingerprint construction errors. Every variant fails closed with no
/// fingerprint value, so no reuse decision can proceed.
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
/// never compare equal to the full input it came from.
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
    /// Fail-closed on empty input.
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
    /// [`FingerprintError::MissingInput`].
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
    /// this check: they fail at construction with no value to compare.
    #[must_use]
    pub fn allows_reuse(&self, candidate: &Self) -> bool {
        self == candidate
    }
}

/// Deterministic FNV-1a-64 over `bytes`. Inline small hasher: no
/// `DefaultHasher`, no `RandomState`, no seed input, identical output on
/// every platform and process.
#[must_use]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_refuses() {
        assert_eq!(
            InputFingerprint::new(&[]),
            Err(FingerprintError::MissingInput)
        );
    }

    #[test]
    fn unknown_component_reports_first_index() {
        assert_eq!(
            InputFingerprint::of_components(&[Some(b"a".as_slice()), None]),
            Err(FingerprintError::UnknownComponent { index: 1 })
        );
    }
}

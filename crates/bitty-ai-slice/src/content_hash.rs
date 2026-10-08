//! Type-safe SHA-256 content address leaf (AI-0184, DEC-0008 step 3).
//!
//! This module is the canonical home of [`ContentHash`]: the 32-byte digest
//! value plus its strict lowercase-hex validation, rendering, and serde
//! mapping. It depends only on external crates (`sha2`, `hex`, `serde`) and
//! the standard library — on no other `bitty-ai-slice` module — so the hash
//! value can later travel with the session plane without dragging the store.
//!
//! ## Why a slice-internal leaf instead of `bitty-ai-runtime`
//!
//! `bitty-ai-runtime` is std-only by invariant (zero dependencies; see its
//! `Cargo.toml` and the AI-0121 boundary note in the slice manifest).
//! [`ContentHash`] needs `sha2` (digest), `hex` (rendering/parsing), and
//! `serde` (content-addressed rows serialize hashes as hex strings), and its
//! previous parse error lived on the storage error type. Moving the type to
//! the runtime would force the runtime to gain those three dependencies (plus
//! a storage-coupled error), breaking the invariant for no behavioral gain.
//! The slice-internal leaf keeps every existing dependency exactly where it
//! already is: no manifest changes.
//!
//! ## Error carrier
//!
//! [`ContentHashError`] carries the single parse failure
//! (`InvalidHash`) with the byte-identical message the store error used
//! (`"invalid SHA-256 hex string: {input:?}"`), so string-observed behavior
//! (bridge `to_string` mapping, boxed task-DAG conversions, serde custom
//! errors) is unchanged. [`crate::content_store::ContentStoreError`] keeps
//! its own `InvalidHash` variant and converts from this error via
//! [`From`], so `?` at store call sites behaves as before.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Parse failure for [`ContentHash::from_hex`].
///
/// Carries the rejected input string. Display matches the historical store
/// error text exactly: `invalid SHA-256 hex string: {input:?}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentHashError {
    /// Input is not a valid 64-character lowercase hex SHA-256 digest.
    InvalidHash(String),
}

impl fmt::Display for ContentHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHash(hex) => write!(f, "invalid SHA-256 hex string: {hex:?}"),
        }
    }
}

impl std::error::Error for ContentHashError {}

/// Type-safe SHA-256 content address (32 bytes).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Compute the SHA-256 digest of the given byte slice.
    #[must_use]
    pub fn compute(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        Self(bytes)
    }

    /// Construct a `ContentHash` from raw 32 bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Access the underlying 32-byte digest array.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Parse a 64-character lowercase hex string into a `ContentHash`.
    pub fn from_hex(s: &str) -> Result<Self, ContentHashError> {
        if s.len() != 64 {
            return Err(ContentHashError::InvalidHash(s.to_string()));
        }

        // Validate strictly lowercase hex digits
        for b in s.bytes() {
            if !matches!(b, b'0'..=b'9' | b'a'..=b'f') {
                return Err(ContentHashError::InvalidHash(s.to_string()));
            }
        }

        let mut bytes = [0u8; 32];
        hex::decode_to_slice(s, &mut bytes)
            .map_err(|_| ContentHashError::InvalidHash(s.to_string()))?;

        Ok(Self(bytes))
    }

    /// Render the content hash as a 64-character lowercase hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({})", self.to_hex())
    }
}

impl FromStr for ContentHash {
    type Err = ContentHashError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_hex(s)
    }
}

impl Serialize for ContentHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

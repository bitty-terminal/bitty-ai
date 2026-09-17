//! Minimal-envelope syntax fallback (AI-0097, AIQ-46 narrowing input).
//!
//! DEC-0007: when structured disclosure fails because the reader cannot
//! parse the schema, fall back to one permanently-readable minimal envelope
//! (`id` + `kind` + `text`) — machine-parseable, consistent with the
//! existing typed-error style.
//!
//! A [`FallbackEnvelope`] carries exactly three fields. The fields are
//! `String` (never raw bytes) so the envelope stays readable text end to
//! end; every bound is a byte bound enforced fail-closed at construction
//! (over-cap input refuses with a typed [`FallbackError` — never truncated
//! silently, which would hide disclosure bytes, and never split at a UTF-8
//! boundary). Byte bounds match the codebase convention: `.len()` counts
//! bytes, exactly like every `*_BYTES`/`*_LEN` bound in `tool.rs`,
//! `prompt.rs`, and `reconcile.rs`.
//!
//! The wire form is length-prefixed framing (`fallback/1` magic plus one
//! `u64` little-endian length per field, mirroring the length-aware
//! `prompt::render_canonical` / `cache_key` section layout, never a marker
//! scan), so embedded header-shaped bytes can never shift a boundary and
//! any malformed tail fails closed as [`FallbackError::Malformed`].
//!
//! [`fallback_for`] is the total fallback path: any unreadable input yields
//! an envelope or a typed error — no panics, no truncation, no clock,
//! threads, network, randomness, or `HashMap`. Std only.
//!
//! Std only. No network, filesystem, clock, threads, or secrets. Pure
//! `&[u8]`-in / envelope-or-error-out; the host owns all byte assembly.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Maximum envelope identity length in bytes.
///
/// Precedent: `tool::MAX_TOOL_NAME_LEN` (64) and `context::MAX_STABLE_ID_LEN`
/// (64) — identity-shaped strings share the 64-byte class.
pub const MAX_FALLBACK_ID_BYTES: usize = 64;
/// Maximum envelope kind tag length in bytes.
///
/// Precedent: `tool::MAX_TOOL_NAME_LEN` (64) — short machine-readable tags
/// share the 64-byte class.
pub const MAX_FALLBACK_KIND_BYTES: usize = 64;
/// Maximum envelope text payload length in bytes.
///
/// Precedent: `tool::MAX_TOOL_DESCRIPTION_LEN` (512),
/// `reconcile::MAX_RECONCILE_REASON_BYTES` (512), and the runtime-wide
/// `MAX_REASON_BYTES` (512) — human-readable disclosure payloads share the
/// 512-byte class.
pub const MAX_FALLBACK_TEXT_BYTES: usize = 512;

/// Fixed envelope identity emitted by [`fallback_for`]: the fallback source
/// is always the same reader-side mechanism, never caller-chosen.
pub const FALLBACK_ID: &str = "fallback";
/// Fixed envelope kind emitted by [`fallback_for`]: unreadable structured
/// input that fell back to the minimal envelope.
pub const FALLBACK_KIND: &str = "syntax-fallback";

/// Fixed magic opening the envelope wire form (`to_bytes` /
/// `from_bytes`). Versioned (`/1`) so a future envelope revision decodes
/// fail-closed as [`FallbackError::Malformed`] instead of aliasing.
const ENVELOPE_MAGIC: &[u8] = b"fallback/1\n";
/// Wire width of one length-prefixed field length.
const LEN_WIDTH: usize = 8;

/// Minimal-envelope construction and decode errors. Every variant fails
/// closed with no envelope value and no partial state.
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
/// Equality covers every field: the same unreadable input rebuilds the same
/// envelope (see [`fallback_for`]), and any field change compares unequal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackEnvelope {
    /// Envelope identity (bounded by [`MAX_FALLBACK_ID_BYTES`]).
    pub id: String,
    /// Envelope kind tag (bounded by [`MAX_FALLBACK_KIND_BYTES`]).
    pub kind: String,
    /// Readable disclosure text (bounded by [`MAX_FALLBACK_TEXT_BYTES`]).
    pub text: String,
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
    /// `kind`, `text`).
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
    /// boundary.
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
/// Declared lengths that exceed `usize` refuse here without allocating.
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
/// frame is short, overruns, or is not UTF-8.
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
/// header or a length that does not fit `usize` (fail-closed downstream).
fn declared_len(header: Option<&[u8]>) -> Option<usize> {
    let header = header?;
    let raw: [u8; LEN_WIDTH] = <[u8; LEN_WIDTH]>::try_from(header).ok()?;
    usize::try_from(u64::from_le_bytes(raw)).ok()
}

/// Scrub unreadable disclosure bytes to readable envelope text: every
/// character outside printable ASCII (`0x20..=0x7E`, so spaces are kept)
/// becomes `?`, which defuses CR/LF log injection, terminal escapes, and
/// confusable non-ASCII bytes. Non-UTF-8 input decodes lossy first, so each
/// undecodable run becomes one replacement character and then one `?`; the
/// result is deterministic, silent (no ellipsis marker that could itself
/// exceed the bound), and never splits a character, so callers can log the
/// value directly. Mirrors the `bridge::bound_reason` scrub rule; unlike
/// that helper this function refuses (never truncates) past the cap.
fn scrub_text(unreadable: &[u8]) -> String {
    let decoded = String::from_utf8_lossy(unreadable);
    let mut scrubbed = String::with_capacity(decoded.len().min(MAX_FALLBACK_TEXT_BYTES));
    for character in decoded.chars() {
        if character.is_ascii_graphic() || character == ' ' {
            scrubbed.push(character);
        } else {
            scrubbed.push('?');
        }
    }
    scrubbed
}

/// Fall back to the minimal envelope for unreadable structured disclosure
/// bytes (DEC-0007): any input the reader cannot parse becomes one
/// permanently-readable envelope with the fixed ([`FALLBACK_ID`],
/// [`FALLBACK_KIND`]) identity and the scrubbed input as text.
///
/// Total: every input yields an envelope or a typed error — empty input
/// envelopes with empty text, and an over-cap payload refuses with
/// [`FallbackError::TextTooLarge`] instead of truncating (truncation would
/// silently drop disclosure bytes).
///
/// # Errors
///
/// Returns [`FallbackError::TextTooLarge`] when the scrubbed payload exceeds
/// [`MAX_FALLBACK_TEXT_BYTES`]. This is the only failure; identity and kind
/// are fixed constants inside their caps, so no other error is reachable.
pub fn fallback_for(unreadable: &[u8]) -> Result<FallbackEnvelope, FallbackError> {
    let text = scrub_text(unreadable);
    if text.len() > MAX_FALLBACK_TEXT_BYTES {
        return Err(FallbackError::TextTooLarge {
            limit: MAX_FALLBACK_TEXT_BYTES,
            actual: text.len(),
        });
    }
    Ok(FallbackEnvelope {
        id: FALLBACK_ID.to_owned(),
        kind: FALLBACK_KIND.to_owned(),
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_identity_satisfies_its_caps() {
        assert!(FALLBACK_ID.len() <= MAX_FALLBACK_ID_BYTES);
        assert!(FALLBACK_KIND.len() <= MAX_FALLBACK_KIND_BYTES);
    }
}

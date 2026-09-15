//! Runtime-to-transport fragment pre-split (AI-0066, review 07 / PX-0270).
//!
//! Three accepted ceilings disagree across the runtime/transport boundary:
//!
//! | Layer                                              | Bound   | Behavior                 |
//! | -------------------------------------------------- | ------- | ------------------------ |
//! | `bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES`     | 64 KiB  | reject                   |
//! | `bitty_ipc::wire::CHUNK_CEILING`                   | 256 KiB | reject                   |
//! | `bitty_ipc::rich_fragment::MAX_FRAGMENT_TEXT_BYTES`| 16 KiB  | truncate (`truncated`)   |
//!
//! A full 64 KiB runtime fragment projected verbatim into the 16 KiB ingest
//! service therefore truncates and silently loses bytes. This module owns the
//! deterministic pre-split rule that removes that loss: a fragment is cut into
//! parts of at most [`MAX_FRAGMENT_TEXT_BYTES`] (16 KiB) at UTF-8 code-point
//! boundaries, emitted in order with a dense transport `seq`, and tagged with a
//! continuation marker. Concatenating the parts in order reproduces the source
//! fragment bytes exactly.
//!
//! # The rule (normative for the future production mapper)
//!
//! 1. The source is one validated runtime [`StreamChunk`] (`validate_chunk`)
//!    whose fragment bytes are valid UTF-8. Non-UTF-8 bytes fail closed: they
//!    cannot become transport `text` without loss.
//! 2. Parts are cut greedily left to right. Each part is the longest prefix of
//!    the remaining bytes that is `<= MAX_FRAGMENT_TEXT_BYTES` and ends on a
//!    UTF-8 code-point boundary. A code point is never split.
//! 3. Parts of one source fragment carry a dense zero-based `part_index`
//!    (`0..part_count`), `part_count >= 1`, and `is_continuation =
//!    part_index > 0`.
//! 4. Transport `seq` values are assigned contiguously from a caller cursor:
//!    part `i` of a fragment starting at `first_seq` gets `first_seq + i`, and
//!    the cursor advances by `part_count`. Ordering is deterministic and
//!    collision-free under the transport dedup key
//!    `(terminal_id, generation, seq)`.
//! 5. Every part fits `MAX_FRAGMENT_TEXT_BYTES`, so the ingest service never
//!    truncates a pre-split part and its `truncated` flag stays `false`.
//!
//! # Why this lives in the slice, not in the runtime
//!
//! `bitty-ai-runtime` is std-only with zero dependencies (a crate rule): it
//! must not link `bitty-ipc`, so it cannot name the transport ceiling, and its
//! validated `Fragment`/`StreamChunk` types carry no continuation concept.
//! Adding one would be a breaking abstraction change to those types, which
//! AI-0066 forbids; `bitty-ipc` is upstream and unchanged. This module lives in
//! the slice crate that already composes both real crates, and the crate docs
//! (`lib.rs`, `README.md`) state the rule above as the contract the future
//! production mapper must follow. No `rich.*` wire method is registered and no
//! live path exists yet: this is the mapping layer, not a shipped transport.

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

use bitty_ai_runtime::{Fragment, StreamChunk, validate_chunk};
use bitty_ipc::rich_fragment::{FragmentData, MAX_FRAGMENT_TEXT_BYTES};
use bitty_ipc::snapshot::ZoneKind;

/// One transport-bound part produced by the pre-split rule.
///
/// `data` is the [`FragmentData`] the ingest service consumes; the remaining
/// fields are the continuation metadata the transport DTO does not carry, kept
/// here so the mapping is self-describing and reassemblable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportPart {
    /// Transport input for this part (bounded text, dense `seq`).
    pub data: FragmentData,
    /// Zero-based index of this part within its source fragment.
    pub part_index: u32,
    /// Number of parts the source fragment was split into (`>= 1`).
    pub part_count: u32,
    /// Runtime `seq` of the source fragment, kept for traceability.
    pub source_seq: u32,
    /// `true` when this part continues a previous part (`part_index > 0`).
    pub is_continuation: bool,
}

/// Fail-closed mapping errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentTransportError {
    /// The runtime refused the chunk (oversized or misframed).
    Runtime(String),
    /// Fragment bytes are not valid UTF-8 and cannot map to transport text.
    NonUtf8,
    /// The fragment would need more parts than a `u32` can index.
    TooManyParts {
        /// Number of parts the split produced.
        parts: usize,
    },
    /// The transport `seq` space overflowed.
    SequenceExhausted,
    /// Reassembly was called with no parts.
    EmptyParts,
    /// Reassembly input disagrees with the recorded `part_count`.
    PartCountMismatch {
        /// `part_count` recorded on the first part.
        expected: u32,
        /// Number of parts actually supplied.
        actual: usize,
    },
    /// Reassembly input is out of order or mislabels its continuation flag.
    PartOrder {
        /// Expected zero-based index at this position.
        expected_index: u32,
        /// Index actually found.
        found: u32,
    },
    /// A reassembled part exceeds the transport ceiling.
    PartTooLarge {
        /// Zero-based index of the offending part.
        index: u32,
        /// Observed bytes.
        actual: usize,
        /// Transport ceiling in bytes.
        limit: usize,
    },
}

impl Display for FragmentTransportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Runtime(reason) => write!(f, "runtime rejected the chunk: {reason}"),
            Self::NonUtf8 => {
                write!(
                    f,
                    "fragment bytes are not valid UTF-8: cannot map without loss"
                )
            }
            Self::TooManyParts { parts } => {
                write!(
                    f,
                    "fragment split into {parts} parts, exceeds the u32 index space"
                )
            }
            Self::SequenceExhausted => write!(f, "transport sequence space exhausted"),
            Self::EmptyParts => write!(f, "reassembly needs at least one part"),
            Self::PartCountMismatch { expected, actual } => {
                write!(f, "expected {expected} parts, got {actual}")
            }
            Self::PartOrder {
                expected_index,
                found,
            } => write!(
                f,
                "expected part index {expected_index} with matching continuation flag, got {found}"
            ),
            Self::PartTooLarge {
                index,
                actual,
                limit,
            } => write!(
                f,
                "part {index} of {actual} bytes exceeds the {limit} byte ceiling"
            ),
        }
    }
}

impl Error for FragmentTransportError {}

/// Pre-split one runtime fragment for the 16 KiB transport ceiling.
///
/// Implements rules 2-4 of the module contract: greedy code-point-safe cut,
/// dense `part_index`/`part_count`, and contiguous transport `seq` starting at
/// `first_seq`. An empty fragment yields exactly one empty part so every source
/// fragment keeps a `part_count >= 1` projection.
///
/// # Errors
///
/// Returns [`FragmentTransportError::NonUtf8`] when the bytes are not valid
/// UTF-8, [`FragmentTransportError::TooManyParts`] when the part count exceeds
/// the `u32` index space, or [`FragmentTransportError::SequenceExhausted`] when
/// the transport `seq` range would overflow.
pub fn pre_split_fragment(
    fragment: &Fragment,
    source_seq: u32,
    terminal_id: &str,
    generation: u64,
    zone: Option<ZoneKind>,
    first_seq: u64,
) -> Result<Vec<TransportPart>, FragmentTransportError> {
    let text = std::str::from_utf8(&fragment.bytes).map_err(|_| FragmentTransportError::NonUtf8)?;
    let ends = part_ends(text);
    let total = ends.len();
    let part_count =
        u32::try_from(total).map_err(|_| FragmentTransportError::TooManyParts { parts: total })?;
    // The split always yields at least one part, so `part_count - 1` is exact.
    first_seq
        .checked_add(u64::from(part_count) - 1)
        .ok_or(FragmentTransportError::SequenceExhausted)?;

    let mut parts = Vec::with_capacity(total);
    let mut start = 0;
    for (index, end) in ends.into_iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| FragmentTransportError::TooManyParts { parts: total })?;
        parts.push(TransportPart {
            data: FragmentData {
                terminal_id: terminal_id.to_owned(),
                generation,
                seq: first_seq + u64::from(index),
                zone,
                text: text[start..end].to_owned(),
            },
            part_index: index,
            part_count,
            source_seq,
            is_continuation: index > 0,
        });
        start = end;
    }
    Ok(parts)
}

/// Validate a runtime chunk then pre-split its fragment.
///
/// # Errors
///
/// Returns [`FragmentTransportError::Runtime`] when `validate_chunk` rejects
/// the chunk, plus every error [`pre_split_fragment`] documents.
pub fn pre_split_chunk(
    chunk: &StreamChunk,
    terminal_id: &str,
    generation: u64,
    zone: Option<ZoneKind>,
    first_seq: u64,
) -> Result<Vec<TransportPart>, FragmentTransportError> {
    validate_chunk(chunk).map_err(|err| FragmentTransportError::Runtime(err.to_string()))?;
    pre_split_fragment(
        &chunk.fragment,
        chunk.seq,
        terminal_id,
        generation,
        zone,
        first_seq,
    )
}

/// Reassemble pre-split parts into the source text.
///
/// Verifies the continuation marker before concatenating: `part_count` must
/// match the supplied length, indices must be dense and in order, each part's
/// `is_continuation` must equal `part_index > 0`, and every part must fit the
/// transport ceiling. Successful output is byte-identical to the source
/// fragment when the parts came from [`pre_split_fragment`].
///
/// # Errors
///
/// Returns a [`FragmentTransportError`] for empty, mismatched, out-of-order,
/// or over-ceiling input.
pub fn reassemble(parts: &[TransportPart]) -> Result<String, FragmentTransportError> {
    let Some(first) = parts.first() else {
        return Err(FragmentTransportError::EmptyParts);
    };
    if usize::try_from(first.part_count).ok() != Some(parts.len()) {
        return Err(FragmentTransportError::PartCountMismatch {
            expected: first.part_count,
            actual: parts.len(),
        });
    }
    let mut out = String::new();
    for (position, part) in parts.iter().enumerate() {
        let expected_index = u32::try_from(position)
            .map_err(|_| FragmentTransportError::TooManyParts { parts: parts.len() })?;
        if part.part_index != expected_index || part.is_continuation != (expected_index > 0) {
            return Err(FragmentTransportError::PartOrder {
                expected_index,
                found: part.part_index,
            });
        }
        if part.data.text.len() > MAX_FRAGMENT_TEXT_BYTES {
            return Err(FragmentTransportError::PartTooLarge {
                index: expected_index,
                actual: part.data.text.len(),
                limit: MAX_FRAGMENT_TEXT_BYTES,
            });
        }
        out.push_str(&part.data.text);
    }
    Ok(out)
}

/// Stateful cursor that maps a whole runtime turn with dense transport `seq`.
///
/// Wraps [`pre_split_chunk`] with a running `next_seq` so consecutive chunks
/// never collide under the transport dedup key
/// `(terminal_id, generation, seq)`, even when an earlier chunk split into more
/// than one part.
#[derive(Debug, Clone)]
pub struct FragmentTransportCursor {
    terminal_id: String,
    generation: u64,
    next_seq: u64,
}

impl FragmentTransportCursor {
    /// Create a cursor for one `(terminal_id, generation)` starting at
    /// `first_seq`.
    #[must_use]
    pub fn new(terminal_id: &str, generation: u64, first_seq: u64) -> Self {
        Self {
            terminal_id: terminal_id.to_owned(),
            generation,
            next_seq: first_seq,
        }
    }

    /// Next unused transport `seq`.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Validate and pre-split one chunk, advancing the cursor.
    ///
    /// # Errors
    ///
    /// Returns every error [`pre_split_chunk`] documents, or
    /// [`FragmentTransportError::SequenceExhausted`] when advancing the cursor
    /// would overflow.
    pub fn map_chunk(
        &mut self,
        chunk: &StreamChunk,
        zone: Option<ZoneKind>,
    ) -> Result<Vec<TransportPart>, FragmentTransportError> {
        let parts = pre_split_chunk(
            chunk,
            &self.terminal_id,
            self.generation,
            zone,
            self.next_seq,
        )?;
        let advance =
            u64::try_from(parts.len()).map_err(|_| FragmentTransportError::SequenceExhausted)?;
        self.next_seq = self
            .next_seq
            .checked_add(advance)
            .ok_or(FragmentTransportError::SequenceExhausted)?;
        Ok(parts)
    }
}

/// End offsets, in order, of the greedily cut parts of `text`.
///
/// Every offset is a UTF-8 code-point boundary and consecutive offsets differ
/// by at most `MAX_FRAGMENT_TEXT_BYTES`. Empty text yields a single `0` offset
/// so a source fragment always projects to at least one part.
fn part_ends(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut ends = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = (start + MAX_FRAGMENT_TEXT_BYTES).min(bytes.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        ends.push(end);
        start = end;
    }
    if ends.is_empty() {
        ends.push(0);
    }
    ends
}

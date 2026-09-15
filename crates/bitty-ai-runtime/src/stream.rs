//! Bounded rich-streaming fragments.
//!
//! Mirrors the `RS-1`..`RS-5` shape: incremental output as
//! `Markdown`/`Diff`/`ToolCard` fragments carried by sequenced chunks with
//! `seq`/`total`/`final`, each chunk bounded by the RC-10 ceiling. Fragments
//! map to presentation-model blocks owned by the host; this crate only
//! produces and validates the bounded byte stream. Cancellation is observed at
//! chunk boundaries by the emitter loop, never inside a chunk.
//!
//! Sequencing follows `S-8` scheme A: `seq` is continuous across every
//! emission batch of one logical turn, so a direct projection onto the
//! transport dedup key `(terminal_id, generation, seq)` cannot false-positive
//! (`P1-5`). `total` is a running water mark and `is_final` closes each
//! emission batch (see [`StreamChunk`]); logical-turn completion stays with
//! the caller's run outcome, not with a chunk flag.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Decoded-byte ceiling per streamed chunk (`RC-10`/`RS-5`, 256 KiB).
///
/// This is the transport-layer aggregate ceiling, mirrored here for the
/// accepted `RC-10` number. It is deliberately not a runtime error condition:
/// a runtime chunk carries exactly one fragment (`RS-3`), so the tighter
/// [`MAX_FRAGMENT_BYTES`] bound always fires first and no per-chunk payload
/// can reach 256 KiB. The aggregate ceiling is enforced where chunks are
/// encoded (`bitty-ipc` wire/frame validation); see [`StreamError`] for the
/// `P2-1` removal rationale.
pub const MAX_STREAM_CHUNK_BYTES: usize = 256 * 1024;
/// Per-fragment bound (`RS-3`: at most one dirty rich block per chunk,
/// 64 KiB). This is the bound that rejects an oversized fragment.
pub const MAX_FRAGMENT_BYTES: usize = 64 * 1024;

/// Rich fragment kind (`RS-1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentKind {
    /// Versioned Markdown block (default).
    Markdown,
    /// Unified diff with bounded per-hunk text.
    Diff,
    /// Typed tool-result card.
    ToolCard,
}

/// One bounded rich fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    /// Fragment kind.
    pub kind: FragmentKind,
    /// Bounded fragment bytes.
    pub bytes: Vec<u8>,
}

impl Fragment {
    /// Build a Markdown fragment.
    #[must_use]
    pub fn markdown(bytes: Vec<u8>) -> Self {
        Self {
            kind: FragmentKind::Markdown,
            bytes,
        }
    }

    /// Build a ToolCard fragment.
    #[must_use]
    pub fn tool_card(bytes: Vec<u8>) -> Self {
        Self {
            kind: FragmentKind::ToolCard,
            bytes,
        }
    }

    /// Fragment byte length.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the fragment carries no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// One streamed chunk carrying `seq`/`total`/`final` (`RS-5`).
///
/// Sequencing is logical-turn scoped (`S-8` scheme A): `seq` is continuous
/// across every emission batch of the turn, while `total` and `is_final`
/// stay batch-scoped — `total` is the running water mark of chunks emitted
/// so far and `is_final` closes the emission batch (`seq + 1 == total`).
/// Logical-turn completion is not a chunk flag here (a failed or cancelled
/// turn and the completing turn both end with a batch-closing chunk); the
/// caller drives the turn and observes its `ExecOutcome`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamChunk {
    /// Zero-based chunk sequence, continuous across the logical turn.
    pub seq: u32,
    /// Running water mark: chunks emitted in the logical turn through this
    /// emission batch (always > `seq` when emitted). Equals the turn's chunk
    /// count once the last batch is emitted.
    pub total: u32,
    /// Whether this is the last chunk of its emission batch
    /// (`seq + 1 == total`).
    pub is_final: bool,
    /// Fragment payload.
    pub fragment: Fragment,
}

/// Streaming validation errors. Oversized or misframed chunks are rejected
/// before they reach the presentation model; nothing is emitted partially.
///
/// `P2-1` / `S-11`: there is deliberately no separate oversized-chunk variant.
/// A chunk carries exactly one fragment (`RS-3`), so [`MAX_FRAGMENT_BYTES`]
/// (64 KiB) always fires before the aggregate [`MAX_STREAM_CHUNK_BYTES`]
/// ceiling (256 KiB) could; a second variant would be unreachable and would
/// misclassify every over-limit block. The aggregate ceiling is enforced at
/// the transport encoding layer instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// Fragment exceeds [`MAX_FRAGMENT_BYTES`].
    OversizedFragment {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// `seq`/`total`/`final` framing violation.
    InvalidFraming {
        /// What was wrong.
        reason: String,
    },
}

impl Display for StreamError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::OversizedFragment { limit, actual } => {
                write!(f, "fragment of {actual} bytes exceeds {limit} byte limit")
            }
            Self::InvalidFraming { reason } => write!(f, "invalid stream framing: {reason}"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Validate one chunk against the fragment bound and the `seq`/`total`/
/// `final` framing (`RS-5`).
///
/// Framing rules under `S-8` scheme A: `total` is the running water mark, so
/// `seq < total` must hold for every chunk, and `is_final` closes the
/// emission batch (`seq + 1 == total`) — more batches may still follow in the
/// same logical turn.
///
/// The RC-10 aggregate ceiling is not checked here: one fragment per chunk
/// (`RS-3`) keeps a chunk payload at or below [`MAX_FRAGMENT_BYTES`], so a
/// distinct chunk-layer refusal is unreachable (`P2-1` / `S-11`); the
/// transport enforces the aggregate bound on the encoded chunk.
///
/// # Errors
///
/// Returns [`StreamError`] for oversized or misframed chunks.
pub fn validate_chunk(chunk: &StreamChunk) -> Result<(), StreamError> {
    if chunk.total == 0 {
        return Err(StreamError::InvalidFraming {
            reason: "total must be > 0".to_owned(),
        });
    }
    if chunk.seq >= chunk.total {
        return Err(StreamError::InvalidFraming {
            reason: "seq must be < total".to_owned(),
        });
    }
    if chunk.is_final != (chunk.seq + 1 == chunk.total) {
        return Err(StreamError::InvalidFraming {
            reason: "final flag does not match seq/total".to_owned(),
        });
    }
    if chunk.fragment.bytes.len() > MAX_FRAGMENT_BYTES {
        return Err(StreamError::OversizedFragment {
            limit: MAX_FRAGMENT_BYTES,
            actual: chunk.fragment.bytes.len(),
        });
    }
    Ok(())
}

/// Sink that consumes validated chunks into the presentation model.
pub trait StreamSink {
    /// Validate and accept one chunk.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError`] when the chunk is oversized or misframed; the
    /// sink state is unchanged on error.
    fn emit(&mut self, chunk: StreamChunk) -> Result<(), StreamError>;

    /// Accepted chunks in emission order.
    fn chunks(&self) -> &[StreamChunk];
}

/// In-memory sink used by tests and headless runs. Models one dirty rich
/// block per accepted chunk (`RS-3`).
#[derive(Debug, Default)]
pub struct VecSink {
    chunks: Vec<StreamChunk>,
}

impl VecSink {
    /// Construct an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Accepted chunk count (equals dirty blocks: one per chunk).
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether no chunk was accepted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Concatenated fragment bytes in emission order.
    #[must_use]
    pub fn concatenated_bytes(&self) -> Vec<u8> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.fragment.bytes.iter().copied())
            .collect()
    }
}

impl StreamSink for VecSink {
    fn emit(&mut self, chunk: StreamChunk) -> Result<(), StreamError> {
        validate_chunk(&chunk)?;
        self.chunks.push(chunk);
        Ok(())
    }

    fn chunks(&self) -> &[StreamChunk] {
        &self.chunks
    }
}

/// Split UTF-8 text into fragments of at most [`MAX_FRAGMENT_BYTES`] bytes
/// without splitting a code point. Returns an empty vector for empty text.
#[must_use]
pub fn fragment_text(kind: FragmentKind, text: &str) -> Vec<Fragment> {
    if text.is_empty() {
        return Vec::new();
    }
    let bytes = text.as_bytes();
    let mut fragments = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = (start + MAX_FRAGMENT_BYTES).min(bytes.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        fragments.push(Fragment {
            kind,
            bytes: bytes[start..end].to_vec(),
        });
        start = end;
    }
    fragments
}

/// Emit one batch of fragments into the logical turn's continuous sequence,
/// checking `is_cancelled` at every chunk boundary (`MP-7`, `PP-6`). Emits
/// nothing for an empty fragment list.
///
/// `start_seq` is the turn's next unused sequence number (the previous
/// batch's return value, or `0` for the first batch). Every chunk in this
/// batch carries the running water mark `total = start_seq + fragments.len()`
/// and the batch-closing chunk is marked `is_final`.
///
/// Returns `Ok(Some(next_seq))` when every fragment was emitted, where
/// `next_seq` is the next unused sequence number for a later batch of the
/// same turn; `Ok(None)` when cancellation stopped emission early
/// (already-emitted chunks stay emitted; the caller reconciles).
///
/// # Errors
///
/// Returns [`StreamError`] when a fragment violates chunk validation or the
/// sequence space is exhausted; chunks emitted before the failure stay
/// emitted.
pub fn emit_fragments(
    sink: &mut dyn StreamSink,
    fragments: &[Fragment],
    start_seq: u32,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<Option<u32>, StreamError> {
    let batch_len = u32::try_from(fragments.len()).map_err(|_| StreamError::InvalidFraming {
        reason: "fragment batch exceeds the sequence space".to_owned(),
    })?;
    let next_seq = start_seq
        .checked_add(batch_len)
        .ok_or_else(|| StreamError::InvalidFraming {
            reason: "stream sequence exhausted".to_owned(),
        })?;
    for (index, fragment) in fragments.iter().enumerate() {
        if is_cancelled() {
            return Ok(None);
        }
        sink.emit(StreamChunk {
            seq: start_seq + index as u32,
            total: next_seq,
            is_final: index + 1 == fragments.len(),
            fragment: fragment.clone(),
        })?;
    }
    Ok(Some(next_seq))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_must_match_seq_total() {
        let mut sink = VecSink::new();
        let bad = StreamChunk {
            seq: 0,
            total: 2,
            is_final: true,
            fragment: Fragment::markdown(b"hi".to_vec()),
        };
        assert!(matches!(
            sink.emit(bad),
            Err(StreamError::InvalidFraming { .. })
        ));
        assert!(sink.is_empty());
        let zero = StreamChunk {
            seq: 0,
            total: 0,
            is_final: false,
            fragment: Fragment::markdown(b"hi".to_vec()),
        };
        assert!(matches!(
            sink.emit(zero),
            Err(StreamError::InvalidFraming { .. })
        ));
        assert!(sink.is_empty());
    }

    #[test]
    fn seq_continues_across_emission_batches() {
        // `S-8` scheme A regression (P1-5): three batches of one fragment
        // each must number 0, 1, 2 (not 0, 0, 0) with a running water mark
        // as `total`, while `is_final` stays the batch-closing flag.
        let mut sink = VecSink::new();
        let draft = fragment_text(FragmentKind::Markdown, "draft");
        let after_draft = emit_fragments(&mut sink, &draft, 0, &|| false)
            .expect("draft batch emits")
            .expect("draft batch completes");
        assert_eq!(after_draft, 1);
        let card = fragment_text(FragmentKind::ToolCard, "tool=terminal_read_zone status=ok");
        let after_card = emit_fragments(&mut sink, &card, after_draft, &|| false)
            .expect("card batch emits")
            .expect("card batch completes");
        assert_eq!(after_card, 2);
        let answer = fragment_text(FragmentKind::Markdown, "answer");
        let after_answer = emit_fragments(&mut sink, &answer, after_card, &|| false)
            .expect("answer batch emits")
            .expect("answer batch completes");
        assert_eq!(after_answer, 3);
        let framing: Vec<(u32, u32, bool)> = sink
            .chunks()
            .iter()
            .map(|chunk| (chunk.seq, chunk.total, chunk.is_final))
            .collect();
        assert_eq!(framing, vec![(0, 1, true), (1, 2, true), (2, 3, true)]);
    }

    #[test]
    fn multi_fragment_batch_shares_one_water_mark() {
        let mut sink = VecSink::new();
        let text = "é".repeat(MAX_FRAGMENT_BYTES);
        let fragments = fragment_text(FragmentKind::Markdown, &text);
        assert!(fragments.len() > 1);
        let total = fragments.len() as u32;
        let next = emit_fragments(&mut sink, &fragments, 0, &|| false)
            .expect("batch emits")
            .expect("batch completes");
        assert_eq!(next, total);
        for (index, chunk) in sink.chunks().iter().enumerate() {
            assert_eq!(chunk.seq, index as u32);
            assert_eq!(chunk.total, total);
            assert_eq!(chunk.is_final, index + 1 == fragments.len());
            validate_chunk(chunk).expect("framing holds");
        }
    }

    #[test]
    fn oversized_fragment_rejected_before_sink_mutation() {
        let mut sink = VecSink::new();
        let big = StreamChunk {
            seq: 0,
            total: 1,
            is_final: true,
            fragment: Fragment::markdown(vec![b'x'; MAX_FRAGMENT_BYTES + 1]),
        };
        assert!(matches!(
            sink.emit(big),
            Err(StreamError::OversizedFragment { .. })
        ));
        assert!(sink.is_empty());
        // `P2-1`: an over-RC-10 block reports the reachable fragment error,
        // never a distinct chunk-layer one; one fragment per chunk (`RS-3`)
        // makes the aggregate ceiling structurally unreachable at this layer.
        let over_rc10 = StreamChunk {
            seq: 0,
            total: 1,
            is_final: true,
            fragment: Fragment::markdown(vec![b'x'; MAX_STREAM_CHUNK_BYTES + 1]),
        };
        assert!(matches!(
            sink.emit(over_rc10),
            Err(StreamError::OversizedFragment { .. })
        ));
        assert!(sink.is_empty());
    }

    #[test]
    fn fragment_text_never_splits_code_point() {
        let text = "é".repeat(MAX_FRAGMENT_BYTES);
        let fragments = fragment_text(FragmentKind::Markdown, &text);
        assert!(fragments.len() > 1);
        for fragment in &fragments {
            assert!(fragment.bytes.len() <= MAX_FRAGMENT_BYTES);
            assert!(std::str::from_utf8(&fragment.bytes).is_ok());
        }
        let joined: Vec<u8> = fragments.iter().flat_map(|f| f.bytes.clone()).collect();
        assert_eq!(joined, text.as_bytes());
    }

    #[test]
    fn emit_stops_at_cancellation_boundary() {
        let mut sink = VecSink::new();
        let fragments = fragment_text(FragmentKind::Markdown, "hello");
        let next = emit_fragments(&mut sink, &fragments, 0, &|| true).expect("no chunk error");
        assert!(next.is_none());
        assert!(sink.is_empty());
    }
}

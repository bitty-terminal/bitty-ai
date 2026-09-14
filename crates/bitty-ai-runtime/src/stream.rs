//! Bounded rich-streaming fragments.
//!
//! Mirrors the `RS-1`..`RS-5` shape: incremental output as
//! `Markdown`/`Diff`/`ToolCard` fragments carried by sequenced chunks with
//! `seq`/`total`/`final`, each chunk bounded by the RC-10 ceiling. Fragments
//! map to presentation-model blocks owned by the host; this crate only
//! produces and validates the bounded byte stream. Cancellation is observed at
//! chunk boundaries by the emitter loop, never inside a chunk.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Decoded-byte ceiling per streamed chunk (`RC-10`/`RS-5`, 256 KiB).
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamChunk {
    /// Zero-based chunk sequence.
    pub seq: u32,
    /// Total chunks in the logical turn (always > 0 when emitted).
    pub total: u32,
    /// Whether this is the final chunk (`seq + 1 == total`).
    pub is_final: bool,
    /// Fragment payload.
    pub fragment: Fragment,
}

/// Streaming validation errors. Oversized or misframed chunks are rejected
/// before they reach the presentation model; nothing is emitted partially.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// Fragment exceeds [`MAX_FRAGMENT_BYTES`].
    OversizedFragment {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Chunk exceeds [`MAX_STREAM_CHUNK_BYTES`].
    OversizedChunk {
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
            Self::OversizedChunk { limit, actual } => {
                write!(f, "chunk of {actual} bytes exceeds {limit} byte limit")
            }
            Self::InvalidFraming { reason } => write!(f, "invalid stream framing: {reason}"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Validate one chunk against the fragment bound, the RC-10 ceiling, and the
/// `seq`/`total`/`final` framing (`RS-5`).
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
    if chunk.fragment.bytes.len() > MAX_STREAM_CHUNK_BYTES {
        return Err(StreamError::OversizedChunk {
            limit: MAX_STREAM_CHUNK_BYTES,
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

/// Emit fragments as one sequenced turn, checking `is_cancelled` at every
/// chunk boundary (`MP-7`, `PP-6`). Emits nothing for an empty fragment list.
///
/// Returns `Ok(true)` when every fragment was emitted, `Ok(false)` when
/// cancellation stopped emission early (already-emitted chunks stay emitted;
/// the caller reconciles).
///
/// # Errors
///
/// Returns [`StreamError`] when a fragment violates chunk validation; chunks
/// emitted before the failure stay emitted.
pub fn emit_fragments(
    sink: &mut dyn StreamSink,
    fragments: &[Fragment],
    is_cancelled: &dyn Fn() -> bool,
) -> Result<bool, StreamError> {
    let total = fragments.len() as u32;
    for (index, fragment) in fragments.iter().enumerate() {
        if is_cancelled() {
            return Ok(false);
        }
        let seq = index as u32;
        sink.emit(StreamChunk {
            seq,
            total,
            is_final: seq + 1 == total,
            fragment: fragment.clone(),
        })?;
    }
    Ok(true)
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
        let done = emit_fragments(&mut sink, &fragments, &|| true).expect("no chunk error");
        assert!(!done);
        assert!(sink.is_empty());
    }
}

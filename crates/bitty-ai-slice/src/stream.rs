//! Rich streaming boundary for the slice.
//!
//! The accepted Rich streaming contract (`RS-1`..`RS-5`) delivers incremental
//! output as `Markdown`/`Diff`/`ToolCard` fragments bounded by RC-10 and
//! composed as scene damage. `bitty-rich`'s `Scene`/`RichBlock` types live in
//! Core and are not consumable out of process today, so this slice models the
//! fragment kinds and validates every emitted chunk with the **real** generic
//! RC-10 primitive `bitty_ipc::wire::validate_chunk`. That absence is recorded
//! as a pressure-test gap; the slice does not pretend to compose Core scenes.

use bitty_ipc::wire::{CHUNK_CEILING, validate_chunk};

use crate::error::SliceError;

/// RC-10 decoded-byte ceiling per streamed chunk.
pub const MAX_STREAM_CHUNK_BYTES: usize = CHUNK_CEILING;

/// Per-fragment bound used by the slice (one rich block per chunk, `RS-3`).
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

/// One streamed chunk carrying `seq`/`total`/`final` (`RS-5`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamChunk {
    /// Zero-based chunk sequence.
    pub seq: u32,
    /// Total chunks in the logical turn.
    pub total: u32,
    /// Whether this is the final chunk.
    pub is_final: bool,
    /// Fragment payload.
    pub fragment: Fragment,
}

/// Sink that consumes validated chunks into the presentation model.
pub trait StreamSink {
    /// Validate and accept one chunk.
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::StreamViolation`] when a chunk violates RC-10 or
    /// the fragment bound.
    fn emit(&mut self, chunk: StreamChunk) -> Result<(), SliceError>;

    /// Accepted chunks in order.
    fn chunks(&self) -> &[StreamChunk];
}

/// In-memory presentation sink modeling one dirty rich block per chunk.
#[derive(Debug, Default)]
pub struct PanelStreamSink {
    chunks: Vec<StreamChunk>,
    dirty_blocks: usize,
}

impl PanelStreamSink {
    /// Construct an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of blocks marked dirty (`RS-3`: one per chunk).
    #[must_use]
    pub fn dirty_blocks(&self) -> usize {
        self.dirty_blocks
    }

    /// Concatenated bytes of all accepted fragments.
    #[must_use]
    pub fn aggregated_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.fragment.bytes.len()).sum()
    }
}

impl StreamSink for PanelStreamSink {
    fn emit(&mut self, chunk: StreamChunk) -> Result<(), SliceError> {
        validate_chunk(chunk.seq, chunk.total, &chunk.fragment.bytes)?;
        if chunk.fragment.bytes.len() > MAX_FRAGMENT_BYTES {
            return Err(SliceError::StreamViolation {
                reason: format!(
                    "fragment of {} bytes exceeds {MAX_FRAGMENT_BYTES}",
                    chunk.fragment.bytes.len()
                ),
            });
        }
        if chunk.total == 0 {
            return Err(SliceError::StreamViolation {
                reason: "total must be > 0".to_owned(),
            });
        }
        if chunk.is_final != (chunk.seq + 1 == chunk.total) {
            return Err(SliceError::StreamViolation {
                reason: "final flag does not match seq/total".to_owned(),
            });
        }
        self.chunks.push(chunk);
        self.dirty_blocks += 1;
        Ok(())
    }

    fn chunks(&self) -> &[StreamChunk] {
        &self.chunks
    }
}

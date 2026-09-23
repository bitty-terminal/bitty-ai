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

/// Default capacity of a [`BoundedSink`] in chunks (`MP-6`).
///
/// 64 chunks at the 64 KiB fragment bound is at most 4 MiB of retained
/// fragment bytes: large enough for a full logical-turn head (system prompt
/// layers plus project snapshot text) without shedding, small enough to stay
/// a bounded queue under the RC-10 sharing budget.
pub const DEFAULT_BOUNDED_SINK_CAPACITY: usize = 64;

/// Turn-scoped stream handle (`MP-6`, `RS-5`).
///
/// Names one emission stream of one logical turn: the `(agent, handle,
/// generation)` triple in [`StreamAttribution`] disambiguates it. The id is
/// a process-local handle minted by the caller (for example via
/// [`crate::session::IdIssuer`]); it carries no authority and is never
/// persisted as a stable external id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHandle(pub u64);

/// Per-emission attribution key: which agent, which stream, which session
/// generation produced the chunk (`MP-6`).
///
/// The triple disambiguates streams across agents sharing a process and
/// across generation rotations of one session: a rotated generation never
/// reuses a stale handle's accounting. Attribution is metadata only; it
/// grants no capability and widens no authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamAttribution {
    /// Agent object that owns the emission.
    pub agent: crate::session::AgentInstanceId,
    /// Stream this chunk was emitted on.
    pub handle: StreamHandle,
    /// Session generation at emission time.
    pub generation: u64,
}

/// Acknowledgement for one accepted chunk (`MP-6`).
///
/// Emission is synchronous in this skeleton: `emit` returning `Ok` means the
/// chunk is accepted, so the ack is the observable record of that fact. It
/// carries the accepted sequence position plus the running totals, making
/// delivery explicit and testable instead of implied by a silent push.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamAck {
    /// Attribution of the accepted chunk.
    pub attribution: StreamAttribution,
    /// Accepted chunk count after this accept (1-based position).
    pub accepted: u64,
    /// Shed count at accept time (stays put when nothing shed).
    pub shed: u64,
}

/// Bounded sink with shed-oldest backpressure and a countable shed metric
/// (`MP-6`, `RS-5`).
///
/// At capacity, the oldest retained chunk is shed to admit the new one and
/// the shed counter increments: no silent loss, every shed is observable via
/// [`BoundedSink::shed_count`]. [`VecSink`] stays the unbounded sink for
/// tests and headless runs; this type is the bounded policy for live paths.
#[derive(Debug)]
pub struct BoundedSink {
    attribution: StreamAttribution,
    capacity: usize,
    chunks: Vec<StreamChunk>,
    accepted: u64,
    shed: u64,
}

impl BoundedSink {
    /// Construct a bounded sink for one stream with an explicit capacity.
    ///
    /// A zero capacity would shed every chunk it accepts; it is normalized
    /// to 1 so the sink always retains the newest chunk.
    #[must_use]
    pub fn with_capacity(attribution: StreamAttribution, capacity: usize) -> Self {
        Self {
            attribution,
            capacity: capacity.max(1),
            chunks: Vec::new(),
            accepted: 0,
            shed: 0,
        }
    }

    /// Construct a bounded sink with [`DEFAULT_BOUNDED_SINK_CAPACITY`].
    #[must_use]
    pub fn new(attribution: StreamAttribution) -> Self {
        Self::with_capacity(attribution, DEFAULT_BOUNDED_SINK_CAPACITY)
    }

    /// Attribution key of this stream.
    #[must_use]
    pub fn attribution(&self) -> StreamAttribution {
        self.attribution
    }

    /// Chunk capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Accepted-chunk count (includes shed chunks: every accept is acked).
    #[must_use]
    pub fn accepted_count(&self) -> u64 {
        self.accepted
    }

    /// Shed-chunk count (0 when nothing was shed: no silent loss).
    #[must_use]
    pub fn shed_count(&self) -> u64 {
        self.shed
    }

    /// Accept one chunk, shedding the oldest when at capacity.
    ///
    /// Validation runs before any mutation: a misframed or oversized chunk
    /// is rejected with [`StreamError`] and changes neither the retained
    /// chunks nor either counter.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError`] for oversized or misframed chunks; the sink
    /// state is unchanged on error.
    pub fn emit_attributed(&mut self, chunk: StreamChunk) -> Result<StreamAck, StreamError> {
        validate_chunk(&chunk)?;
        if self.chunks.len() >= self.capacity {
            self.chunks.remove(0);
            self.shed += 1;
        }
        self.chunks.push(chunk);
        self.accepted += 1;
        Ok(StreamAck {
            attribution: self.attribution,
            accepted: self.accepted,
            shed: self.shed,
        })
    }

    /// Retained chunks in emission order (oldest first).
    #[must_use]
    pub fn chunks(&self) -> &[StreamChunk] {
        &self.chunks
    }

    /// Whether no chunk is retained (shed chunks still count as accepted).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Concatenated fragment bytes of retained chunks in emission order.
    #[must_use]
    pub fn concatenated_bytes(&self) -> Vec<u8> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.fragment.bytes.iter().copied())
            .collect()
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
/// Returns `Ok(Some(next_seq))` when every fragment was emitted and no
/// cancellation was observed at any boundary, where `next_seq` is the next
/// unused sequence number for a later batch of the same turn; `Ok(None)`
/// when cancellation was requested before, during, or after delivery
/// (already-emitted chunks stay emitted; the caller reconciles).
///
/// Cancellation is rechecked after the final delivery because a callback
/// (`sink.emit`) can request it while accepting the last fragment: without
/// the post-delivery check a batch whose final callback cancelled would be
/// reported as complete, disagreeing with the session's terminal state.
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
    // AI-RUN-005: a callback can request cancellation while accepting the
    // final fragment, so acceptance is not completion. Recheck after
    // delivery and report `None` (the caller's reconcile path) rather than
    // advancing the water mark; delivered bytes stay in the sink.
    if is_cancelled() {
        return Ok(None);
    }
    Ok(Some(next_seq))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;
    use crate::session::{AgentInstanceId, IdIssuer};

    fn attribution(handle_value: u64, generation: u64) -> StreamAttribution {
        StreamAttribution {
            agent: AgentInstanceId(7),
            handle: StreamHandle(handle_value),
            generation,
        }
    }

    fn chunk(seq: u32, total: u32, text: &str) -> StreamChunk {
        StreamChunk {
            seq,
            total,
            is_final: seq + 1 == total,
            fragment: Fragment::markdown(text.as_bytes().to_vec()),
        }
    }

    #[test]
    fn bounded_sink_accepts_under_capacity_with_zero_shed_and_acks() {
        let key = attribution(1, 1);
        let mut sink = BoundedSink::with_capacity(key, 4);
        let ack = sink
            .emit_attributed(chunk(0, 2, "a"))
            .expect("first chunk accepts");
        assert_eq!(
            ack,
            StreamAck {
                attribution: key,
                accepted: 1,
                shed: 0,
            }
        );
        let ack = sink
            .emit_attributed(chunk(1, 2, "b"))
            .expect("second chunk accepts");
        assert_eq!(ack.accepted, 2);
        assert_eq!(ack.shed, 0);
        assert_eq!(sink.accepted_count(), 2);
        assert_eq!(sink.shed_count(), 0);
        assert_eq!(sink.chunks().len(), 2);
        assert_eq!(sink.concatenated_bytes(), b"ab");
    }

    #[test]
    fn bounded_sink_sheds_oldest_with_countable_metric() {
        let key = attribution(1, 1);
        let mut sink = BoundedSink::with_capacity(key, 2);
        sink.emit_attributed(chunk(0, 4, "a")).expect("accepts");
        sink.emit_attributed(chunk(1, 4, "b")).expect("accepts");
        let ack = sink.emit_attributed(chunk(2, 4, "c")).expect("accepts");
        // No silent loss: the shed is counted on the ack and the counter.
        assert_eq!(ack.accepted, 3);
        assert_eq!(ack.shed, 1);
        assert_eq!(ack.attribution, key);
        assert_eq!(sink.shed_count(), 1);
        assert_eq!(sink.accepted_count(), 3);
        // Oldest-first retention: "a" shed, "b" then "c" retained.
        assert_eq!(sink.concatenated_bytes(), b"bc");
        let ack = sink.emit_attributed(chunk(3, 4, "d")).expect("accepts");
        assert_eq!(ack.shed, 2);
        assert_eq!(sink.shed_count(), 2);
        assert_eq!(sink.concatenated_bytes(), b"cd");
    }

    #[test]
    fn bounded_sink_validation_rejects_without_mutation() {
        let key = attribution(1, 1);
        let mut sink = BoundedSink::with_capacity(key, 2);
        sink.emit_attributed(chunk(0, 1, "a")).expect("accepts");
        let bad = StreamChunk {
            seq: 0,
            total: 2,
            is_final: true,
            fragment: Fragment::markdown(b"hi".to_vec()),
        };
        assert!(matches!(
            sink.emit_attributed(bad),
            Err(StreamError::InvalidFraming { .. })
        ));
        // Rejection changes neither retained chunks nor either counter.
        assert_eq!(sink.accepted_count(), 1);
        assert_eq!(sink.shed_count(), 0);
        assert_eq!(sink.concatenated_bytes(), b"a");
    }

    #[test]
    fn attribution_triple_disambiguates_handle_and_generation() {
        let first = StreamAttribution {
            agent: AgentInstanceId(1),
            handle: StreamHandle(1),
            generation: 1,
        };
        let rotated = StreamAttribution {
            generation: 2,
            ..first
        };
        let other_handle = StreamAttribution {
            handle: StreamHandle(2),
            ..first
        };
        assert_ne!(first, rotated);
        assert_ne!(first, other_handle);
        // Handles mint from the shared id issuer without colliding.
        let mut issuer = IdIssuer::default();
        let first_id = issuer.session();
        let second_id = issuer.session();
        assert_ne!(first_id, second_id);
    }

    #[test]
    fn zero_capacity_normalizes_to_newest_retained() {
        let key = attribution(9, 3);
        let mut sink = BoundedSink::with_capacity(key, 0);
        assert_eq!(sink.capacity(), 1);
        sink.emit_attributed(chunk(0, 2, "a")).expect("accepts");
        sink.emit_attributed(chunk(1, 2, "b")).expect("accepts");
        assert_eq!(sink.shed_count(), 1);
        assert_eq!(sink.concatenated_bytes(), b"b");
    }

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

    /// Sink wrapper that requests cancellation while accepting a chunk whose
    /// `is_final` is true — the deterministic analogue of a transport
    /// callback cancelling on the batch-closing delivery.
    struct CancelOnFinal {
        inner: VecSink,
        cancelled: Rc<Cell<bool>>,
    }

    impl StreamSink for CancelOnFinal {
        fn emit(&mut self, chunk: StreamChunk) -> Result<(), StreamError> {
            let is_final = chunk.is_final;
            self.inner.emit(chunk)?;
            if is_final {
                self.cancelled.set(true);
            }
            Ok(())
        }

        fn chunks(&self) -> &[StreamChunk] {
            self.inner.chunks()
        }
    }

    fn cancel_on_final_sink() -> (CancelOnFinal, Rc<Cell<bool>>) {
        let cancelled = Rc::new(Cell::new(false));
        (
            CancelOnFinal {
                inner: VecSink::new(),
                cancelled: Rc::clone(&cancelled),
            },
            cancelled,
        )
    }

    #[test]
    fn emit_rechecks_cancellation_after_final_delivery() {
        // AI-RUN-005: cancel landing on the final delivery must not report
        // the batch complete. The bytes are already accepted and preserved,
        // but the outcome is the cancellation signal (`None`).
        let (mut sink, cancelled) = cancel_on_final_sink();
        let fragments = fragment_text(FragmentKind::Markdown, "final");
        let next =
            emit_fragments(&mut sink, &fragments, 0, &|| cancelled.get()).expect("no chunk error");
        assert!(next.is_none());
        assert!(cancelled.get());
        assert_eq!(sink.chunks().len(), fragments.len());
        assert!(sink.chunks()[0].is_final);
    }

    #[test]
    fn emit_rechecks_cancellation_after_multi_fragment_final_delivery() {
        // Same boundary with a multi-fragment batch: the cancel lands on the
        // last of several chunks, so the whole batch is delivered, yet the
        // batch still reports cancellation rather than completion.
        let (mut sink, cancelled) = cancel_on_final_sink();
        let text = "é".repeat(MAX_FRAGMENT_BYTES);
        let fragments = fragment_text(FragmentKind::Markdown, &text);
        assert!(fragments.len() > 1);
        let next =
            emit_fragments(&mut sink, &fragments, 0, &|| cancelled.get()).expect("no chunk error");
        assert!(next.is_none());
        assert_eq!(sink.chunks().len(), fragments.len());
        let joined: Vec<u8> = sink
            .chunks()
            .iter()
            .flat_map(|chunk| chunk.fragment.bytes.iter().copied())
            .collect();
        assert_eq!(joined, text.as_bytes());
        assert!(sink.chunks().last().expect("last chunk").is_final);
    }
}

//! Bounded control-record framing over an injected byte stream.
//!
//! One frame is exactly a four-byte big-endian `u32` length followed by that
//! many record bytes. The prefix is read into a stack buffer, validated
//! against [`CONTROL_FRAME_MAX_BYTES`] and [`CONTROL_HEADER_BYTES`], and only
//! then is the body allocated. A zero-length, oversized, short, truncated, or
//! non-record frame fails closed.
//!
//! The writer keeps at most [`CONTROL_QUEUE_MAX_RECORDS`] encoded frames and
//! [`CONTROL_INFLIGHT_MAX_BYTES`] pending bytes. It writes synchronously and
//! clears the queue on the first failure: a partially written record is never
//! retried on a replacement stream. The caller owns every deadline; this
//! module creates no thread, socket, listener, or clock.

use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Read, Write};

use bitty_ipc_api::provider_publication::{
    CONTROL_FRAME_MAX_BYTES, CONTROL_FRAME_PREFIX_BYTES, CONTROL_HEADER_BYTES,
    CONTROL_INFLIGHT_MAX_BYTES, CONTROL_QUEUE_MAX_RECORDS, CONTROL_SMALL_RECORD_MAX_BYTES,
    ControlError, ControlRecord, Kind,
};

/// Fail-closed framed-I/O rejection.
///
/// No variant contains a payload, record identifier, path, or secret. The
/// `Debug` fields carry only local counts for tests; `Display` is fixed text.
#[derive(Debug)]
pub enum FrameError {
    /// The stream closed cleanly between frames.
    Closed,
    /// The stream ended part-way through the four-byte length prefix.
    TruncatedPrefix {
        /// Prefix bytes received before EOF.
        received: usize,
    },
    /// The stream ended part-way through the declared record body.
    TruncatedBody {
        /// Declared body length.
        expected: usize,
        /// Body bytes received before EOF.
        received: usize,
    },
    /// The declared frame length is zero.
    ZeroLength,
    /// The declared frame length exceeds [`CONTROL_FRAME_MAX_BYTES`].
    FrameTooLarge,
    /// The declared frame is shorter than a control record header.
    FrameShorterThanHeader {
        /// Declared length.
        actual: usize,
    },
    /// A non-payload record exceeds [`CONTROL_SMALL_RECORD_MAX_BYTES`].
    SmallRecordTooLarge {
        /// Actual frame length.
        actual: usize,
    },
    /// The complete frame is not a valid control record.
    Record(ControlError),
    /// The bounded body allocation failed.
    AllocationFailed {
        /// Requested body bytes.
        requested: usize,
    },
    /// The injected stream failed.
    Io(io::Error),
    /// A write made no progress.
    WriteZero,
    /// The bounded outbound queue already holds its maximum record count.
    QueueFull {
        /// Configured record limit.
        limit: usize,
    },
    /// The bounded outbound queue would exceed its byte budget.
    ByteBudgetExceeded {
        /// Configured byte limit.
        limit: usize,
    },
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => formatter.write_str("control stream closed between frames"),
            Self::TruncatedPrefix { .. } => {
                formatter.write_str("control stream ended inside a frame length prefix")
            }
            Self::TruncatedBody { .. } => {
                formatter.write_str("control stream ended inside a control record")
            }
            Self::ZeroLength => formatter.write_str("control frame length is zero"),
            Self::FrameTooLarge => formatter.write_str("control frame exceeds the named bound"),
            Self::FrameShorterThanHeader { .. } => {
                formatter.write_str("control frame is shorter than a record header")
            }
            Self::SmallRecordTooLarge { .. } => {
                formatter.write_str("small control record exceeds the named bound")
            }
            Self::Record(_) => formatter.write_str("control record rejected"),
            Self::AllocationFailed { .. } => formatter.write_str("control frame allocation failed"),
            Self::Io(_) => formatter.write_str("control stream I/O failed"),
            Self::WriteZero => formatter.write_str("control stream write made no progress"),
            Self::QueueFull { .. } => formatter.write_str("control outbound queue is full"),
            Self::ByteBudgetExceeded { .. } => {
                formatter.write_str("control outbound byte budget exceeded")
            }
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Record(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Closed
            | Self::TruncatedPrefix { .. }
            | Self::TruncatedBody { .. }
            | Self::ZeroLength
            | Self::FrameTooLarge
            | Self::FrameShorterThanHeader { .. }
            | Self::SmallRecordTooLarge { .. }
            | Self::AllocationFailed { .. }
            | Self::WriteZero
            | Self::QueueFull { .. }
            | Self::ByteBudgetExceeded { .. } => None,
        }
    }
}

/// Bounded single-frame reader.
///
/// The reader holds no unread frame between calls: each [`Self::read_record`]
/// consumes exactly one complete frame or fails closed. That is the one-record
/// inbound queue, well inside [`CONTROL_QUEUE_MAX_RECORDS`] and the
/// [`CONTROL_INFLIGHT_MAX_BYTES`] two-frame budget.
#[derive(Debug, Default)]
pub struct FrameReader {
    _private: (),
}

impl FrameReader {
    /// Construct an empty reader.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }

    /// Read and decode exactly one bounded control record.
    ///
    /// A clean EOF between frames returns [`FrameError::Closed`]. EOF inside a
    /// prefix or body returns the matching truncation error. The declared
    /// length is validated before the body allocation.
    ///
    /// # Errors
    ///
    /// Returns the first framing, allocation, I/O, or record-codec rejection.
    pub fn read_record<R: Read + ?Sized>(
        &mut self,
        reader: &mut R,
    ) -> Result<ControlRecord, FrameError> {
        let mut prefix = [0_u8; CONTROL_FRAME_PREFIX_BYTES];
        read_prefix(reader, &mut prefix)?;

        let length =
            usize::try_from(u32::from_be_bytes(prefix)).map_err(|_| FrameError::FrameTooLarge)?;
        if length == 0 {
            return Err(FrameError::ZeroLength);
        }
        if length > CONTROL_FRAME_MAX_BYTES {
            return Err(FrameError::FrameTooLarge);
        }
        if length < CONTROL_HEADER_BYTES {
            return Err(FrameError::FrameShorterThanHeader { actual: length });
        }

        let mut frame = Vec::new();
        frame
            .try_reserve_exact(length)
            .map_err(|_| FrameError::AllocationFailed { requested: length })?;
        frame.resize(length, 0);
        read_body(reader, &mut frame)?;

        enforce_small_record_bound(&frame)?;
        ControlRecord::decode(&frame).map_err(FrameError::Record)
    }
}

/// Bounded synchronous record writer.
///
/// Encoded frames are queued and flushed in order. The queue is bounded by
/// [`CONTROL_QUEUE_MAX_RECORDS`] and [`CONTROL_INFLIGHT_MAX_BYTES`]; any
/// excess fails before the frame is accepted. A failed flush clears the queue
/// and is never retried.
#[derive(Debug, Default)]
pub struct FrameWriter {
    pending: VecDeque<Vec<u8>>,
    pending_bytes: usize,
}

impl FrameWriter {
    /// Construct an empty writer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            pending_bytes: 0,
        }
    }

    /// Encode and queue exactly one control record.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError::Record`] when the record codec rejects it,
    /// [`FrameError::QueueFull`] at the record bound, or
    /// [`FrameError::ByteBudgetExceeded`] at the byte bound.
    pub fn enqueue(&mut self, record: &ControlRecord) -> Result<(), FrameError> {
        if self.pending.len() >= CONTROL_QUEUE_MAX_RECORDS {
            return Err(FrameError::QueueFull {
                limit: CONTROL_QUEUE_MAX_RECORDS,
            });
        }
        let frame = record.encode().map_err(FrameError::Record)?;
        self.enqueue_frame(frame)
    }

    /// Write every queued frame and flush the stream.
    ///
    /// On the first failure the queue is cleared and the error is returned;
    /// the caller must treat the control stream as unusable.
    ///
    /// # Errors
    ///
    /// Returns an I/O or zero-progress error from the injected stream.
    pub fn flush<W: Write + ?Sized>(&mut self, writer: &mut W) -> Result<(), FrameError> {
        while let Some(frame) = self.pending.pop_front() {
            self.pending_bytes = self
                .pending_bytes
                .saturating_sub(CONTROL_FRAME_PREFIX_BYTES + frame.len());
            if let Err(error) = write_frame(writer, &frame) {
                self.clear();
                return Err(error);
            }
        }
        writer.flush().map_err(FrameError::Io)
    }

    /// Number of frames still queued.
    #[must_use]
    pub fn pending_records(&self) -> usize {
        self.pending.len()
    }

    /// Number of frame bytes plus prefixes still queued.
    #[must_use]
    pub fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }

    /// Drop every queued frame without writing it.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.pending_bytes = 0;
    }

    fn enqueue_frame(&mut self, frame: Vec<u8>) -> Result<(), FrameError> {
        if self.pending.len() >= CONTROL_QUEUE_MAX_RECORDS {
            return Err(FrameError::QueueFull {
                limit: CONTROL_QUEUE_MAX_RECORDS,
            });
        }
        if frame.is_empty() {
            return Err(FrameError::ZeroLength);
        }
        if frame.len() > CONTROL_FRAME_MAX_BYTES {
            return Err(FrameError::FrameTooLarge);
        }
        let frame_bytes = CONTROL_FRAME_PREFIX_BYTES.checked_add(frame.len()).ok_or(
            FrameError::ByteBudgetExceeded {
                limit: CONTROL_INFLIGHT_MAX_BYTES,
            },
        )?;
        let total =
            self.pending_bytes
                .checked_add(frame_bytes)
                .ok_or(FrameError::ByteBudgetExceeded {
                    limit: CONTROL_INFLIGHT_MAX_BYTES,
                })?;
        if total > CONTROL_INFLIGHT_MAX_BYTES {
            return Err(FrameError::ByteBudgetExceeded {
                limit: CONTROL_INFLIGHT_MAX_BYTES,
            });
        }
        self.pending.push_back(frame);
        self.pending_bytes = total;
        Ok(())
    }
}

fn read_prefix<R: Read + ?Sized>(
    reader: &mut R,
    prefix: &mut [u8; CONTROL_FRAME_PREFIX_BYTES],
) -> Result<(), FrameError> {
    let mut filled = 0_usize;
    while filled < prefix.len() {
        match reader.read(&mut prefix[filled..]) {
            Ok(0) if filled == 0 => return Err(FrameError::Closed),
            Ok(0) => {
                return Err(FrameError::TruncatedPrefix { received: filled });
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(FrameError::Io(error)),
        }
    }
    Ok(())
}

fn read_body<R: Read + ?Sized>(reader: &mut R, body: &mut [u8]) -> Result<(), FrameError> {
    let mut filled = 0_usize;
    while filled < body.len() {
        match reader.read(&mut body[filled..]) {
            Ok(0) => {
                return Err(FrameError::TruncatedBody {
                    expected: body.len(),
                    received: filled,
                });
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(FrameError::Io(error)),
        }
    }
    Ok(())
}

fn enforce_small_record_bound(frame: &[u8]) -> Result<(), FrameError> {
    if frame.len() <= CONTROL_SMALL_RECORD_MAX_BYTES {
        return Ok(());
    }
    // The length was checked against the header minimum before allocation, so
    // the kind bytes are present. Projection and Publish are the only records
    // allowed to exceed the small-record bound.
    let kind = u16::from_be_bytes([frame[6], frame[7]]);
    match Kind::from_code(kind) {
        Some(Kind::Projection | Kind::Publish) => Ok(()),
        // Unknown kinds fall through so the record codec reports the closed
        // enum violation rather than a size error.
        None => Ok(()),
        Some(_) => Err(FrameError::SmallRecordTooLarge {
            actual: frame.len(),
        }),
    }
}

fn write_frame<W: Write + ?Sized>(writer: &mut W, frame: &[u8]) -> Result<(), FrameError> {
    if frame.is_empty() {
        return Err(FrameError::ZeroLength);
    }
    if frame.len() > CONTROL_FRAME_MAX_BYTES {
        return Err(FrameError::FrameTooLarge);
    }
    let length = u32::try_from(frame.len()).map_err(|_| FrameError::FrameTooLarge)?;
    write_all(writer, &length.to_be_bytes())?;
    write_all(writer, frame)
}

fn write_all<W: Write + ?Sized>(writer: &mut W, mut bytes: &[u8]) -> Result<(), FrameError> {
    while !bytes.is_empty() {
        match writer.write(bytes) {
            Ok(0) => return Err(FrameError::WriteZero),
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(FrameError::Io(error)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FrameError, FrameWriter};
    use bitty_ipc_api::provider_publication::{
        CONTROL_FRAME_MAX_BYTES, CONTROL_INFLIGHT_MAX_BYTES, CONTROL_QUEUE_MAX_RECORDS,
    };

    #[test]
    fn queue_record_bound_is_enforced() {
        let mut writer = FrameWriter::new();
        assert!(writer.enqueue_frame(vec![0_u8; 64]).is_ok());
        assert!(writer.enqueue_frame(vec![0_u8; 64]).is_ok());
        assert!(matches!(
            writer.enqueue_frame(vec![0_u8; 64]),
            Err(FrameError::QueueFull { limit }) if limit == CONTROL_QUEUE_MAX_RECORDS
        ));
    }

    #[test]
    fn queue_byte_bound_is_enforced() {
        let mut writer = FrameWriter::new();
        let frame = vec![0_u8; CONTROL_FRAME_MAX_BYTES];
        assert!(writer.enqueue_frame(frame.clone()).is_ok());
        assert!(matches!(
            writer.enqueue_frame(frame),
            Err(FrameError::ByteBudgetExceeded { limit }) if limit == CONTROL_INFLIGHT_MAX_BYTES
        ));
    }

    #[test]
    fn empty_frame_is_refused() {
        let mut writer = FrameWriter::new();
        assert!(matches!(
            writer.enqueue_frame(Vec::new()),
            Err(FrameError::ZeroLength)
        ));
    }

    #[test]
    fn clear_releases_the_queue() {
        let mut writer = FrameWriter::new();
        writer.enqueue_frame(vec![0_u8; 32]).expect("frame");
        assert_eq!(writer.pending_records(), 1);
        writer.clear();
        assert_eq!(writer.pending_records(), 0);
        assert_eq!(writer.pending_bytes(), 0);
    }
}

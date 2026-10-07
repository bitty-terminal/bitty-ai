//! Newline-delimited JSON frames with a cap-plus-one probe.
//!
//! The child speaks newline-delimited JSON-RPC on stdout. Every line is
//! bounded by [`MAX_FRAME_BYTES`] (the 256 KiB transport convention shared
//! with the runtime stream chunk ceiling): the reader retains at most
//! cap-plus-one bytes, so an oversize report carries `actual` of
//! cap-plus-one rather than the true stream size, and at most that much is
//! ever buffered. Oversize lines fail with [`McpFailure::FrameTooLarge`];
//! malformed lines (non-UTF-8, empty, or not a JSON object) are dropped and
//! counted in [`FrameStats::dropped_malformed`], never returned.

use std::io::BufRead;

use crate::error::{McpError, McpFailure, McpStage};

/// Maximum decoded frame bytes (256 KiB transport convention).
pub const MAX_FRAME_BYTES: usize = 256 * 1024;

/// Frame decode counters, owned by the reader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Frames accepted and returned.
    pub received: u64,
    /// Malformed lines dropped (non-UTF-8, empty, non-object).
    pub dropped_malformed: u64,
    /// Oversize lines refused.
    pub rejected_oversize: u64,
}

/// Check a frame length against [`MAX_FRAME_BYTES`].
///
/// The caller passes the retained length, which is at most cap-plus-one by
/// construction of [`FramedLines`], so `actual` never measures the true
/// stream size.
///
/// # Errors
///
/// Returns [`McpFailure::FrameTooLarge`] past the cap.
pub fn check_frame_len(len: usize) -> Result<(), McpError> {
    if len > MAX_FRAME_BYTES {
        return Err(McpError::new(
            McpStage::Frame,
            McpFailure::FrameTooLarge {
                limit: MAX_FRAME_BYTES,
                actual: len,
            },
        ));
    }
    Ok(())
}

/// Decode one raw line (newline already stripped).
///
/// Returns `None` and counts one drop when the line is empty, non-UTF-8, or
/// not a JSON object (`{...}` after trimming); otherwise counts one receipt
/// and returns the line. Oversize is checked separately by
/// [`check_frame_len`] before this runs.
#[must_use]
pub fn decode_line(raw: &[u8], stats: &mut FrameStats) -> Option<String> {
    let text = match std::str::from_utf8(raw) {
        Ok(text) => text,
        Err(_) => {
            stats.dropped_malformed += 1;
            return None;
        }
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        stats.dropped_malformed += 1;
        return None;
    }
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        stats.dropped_malformed += 1;
        return None;
    }
    stats.received += 1;
    Some(text.to_owned())
}

/// Newline-delimited frame reader over any `BufRead`.
///
/// [`FramedLines::next_line`] returns the next well-formed frame, `Ok(None)`
/// on clean EOF, and `Err(FrameTooLarge)` on an oversize line (after
/// consuming it through the newline so the stream stays aligned). Malformed
/// lines are dropped, counted, and skipped internally: callers only observe
/// well-formed frames or EOF.
pub struct FramedLines<R> {
    inner: R,
    stats: FrameStats,
}

impl<R: BufRead> FramedLines<R> {
    /// Wrap a buffered byte source.
    #[must_use]
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            stats: FrameStats::default(),
        }
    }

    /// Decode counters so far.
    #[must_use]
    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// Return the next well-formed frame, or `Ok(None)` on clean EOF.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::FrameTooLarge`] when a line exceeds
    /// [`MAX_FRAME_BYTES`] (retaining only cap-plus-one bytes), and
    /// [`McpFailure::Io`] when the underlying read fails.
    pub fn next_line(&mut self) -> Result<Option<String>, McpError> {
        loop {
            let mut raw: Vec<u8> = Vec::new();
            // Cap-plus-one probe: retain at most this much, then keep
            // consuming through the newline so the next frame still aligns.
            let mut retained = 0_usize;
            let mut saw_newline = false;
            let mut eof = false;
            while !saw_newline {
                let available = self.inner.fill_buf().map_err(|_| {
                    McpError::new(
                        McpStage::Frame,
                        McpFailure::Io {
                            context: crate::error::bound_error_text(
                                "stdout read",
                                crate::error::MAX_ERROR_TEXT_BYTES,
                            ),
                        },
                    )
                })?;
                if available.is_empty() {
                    eof = true;
                    break;
                }
                let take = available.iter().position(|byte| *byte == b'\n');
                match take {
                    Some(offset) => {
                        let want = offset + 1;
                        if retained < MAX_FRAME_BYTES + 1 {
                            let room = MAX_FRAME_BYTES + 1 - retained;
                            raw.extend_from_slice(&available[..want.min(room)]);
                        }
                        retained += want;
                        self.inner.consume(want);
                        saw_newline = true;
                    }
                    None => {
                        if retained < MAX_FRAME_BYTES + 1 {
                            let room = MAX_FRAME_BYTES + 1 - retained;
                            raw.extend_from_slice(&available[..available.len().min(room)]);
                        }
                        let consumed = available.len();
                        retained += consumed;
                        self.inner.consume(consumed);
                    }
                }
            }
            if eof && raw.is_empty() {
                return Ok(None);
            }
            // Strip one trailing newline plus an optional carriage return.
            if raw.last() == Some(&b'\n') {
                raw.pop();
                if raw.last() == Some(&b'\r') {
                    raw.pop();
                }
            }
            if retained > MAX_FRAME_BYTES {
                self.stats.rejected_oversize += 1;
                return Err(McpError::new(
                    McpStage::Frame,
                    McpFailure::FrameTooLarge {
                        limit: MAX_FRAME_BYTES,
                        actual: raw.len().min(MAX_FRAME_BYTES + 1),
                    },
                ));
            }
            if eof && raw.is_empty() {
                return Ok(None);
            }
            if let Some(line) = decode_line(&raw, &mut self.stats) {
                return Ok(Some(line));
            }
            if eof {
                return Ok(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn well_formed_lines_decode() {
        let input = "{\"a\":1}\n{\"b\":2}\n";
        let mut frames = FramedLines::new(BufReader::new(input.as_bytes()));
        assert_eq!(
            frames.next_line().expect("line"),
            Some("{\"a\":1}".to_owned())
        );
        assert_eq!(
            frames.next_line().expect("line"),
            Some("{\"b\":2}".to_owned())
        );
        assert_eq!(frames.next_line().expect("eof"), None);
        assert_eq!(
            frames.stats(),
            FrameStats {
                received: 2,
                dropped_malformed: 0,
                rejected_oversize: 0,
            }
        );
    }

    #[test]
    fn malformed_lines_drop_and_count() {
        let input = "{\"good\":1}\nnot json\n\n[1,2]\n{\"good\":2}\n";
        let mut frames = FramedLines::new(BufReader::new(input.as_bytes()));
        assert_eq!(
            frames.next_line().expect("line"),
            Some("{\"good\":1}".to_owned())
        );
        assert_eq!(
            frames.next_line().expect("line"),
            Some("{\"good\":2}".to_owned())
        );
        assert_eq!(frames.next_line().expect("eof"), None);
        let stats = frames.stats();
        assert_eq!(stats.received, 2);
        // "not json", the empty line, and the non-object array all drop.
        assert_eq!(stats.dropped_malformed, 3);
        assert_eq!(stats.rejected_oversize, 0);
    }

    #[test]
    fn non_utf8_lines_drop_and_count() {
        let mut input = b"{\"good\":1}\n".to_vec();
        input.extend_from_slice(&[0xFF, 0xFE, b'\n']);
        input.extend_from_slice(b"{\"good\":2}\n");
        let mut frames = FramedLines::new(BufReader::new(input.as_slice()));
        assert!(frames.next_line().expect("line").is_some());
        assert!(frames.next_line().expect("line").is_some());
        assert_eq!(frames.stats().dropped_malformed, 1);
    }

    #[test]
    fn oversize_line_probes_cap_plus_one_and_stays_aligned() {
        let mut input = String::from("{\"ok\":1}\n");
        input.push('{');
        input.push_str(&"x".repeat(MAX_FRAME_BYTES + 999));
        input.push_str("}\n");
        input.push_str("{\"next\":2}\n");
        let mut frames = FramedLines::new(BufReader::new(input.as_bytes()));
        assert_eq!(
            frames.next_line().expect("line"),
            Some("{\"ok\":1}".to_owned())
        );
        let error = frames.next_line().expect_err("oversize must fail");
        match error.failure {
            McpFailure::FrameTooLarge { limit, actual } => {
                assert_eq!(limit, MAX_FRAME_BYTES);
                assert_eq!(actual, MAX_FRAME_BYTES + 1);
            }
            other => panic!("expected FrameTooLarge, got {other:?}"),
        }
        // The stream stays aligned: the frame after the oversize line reads.
        assert_eq!(
            frames.next_line().expect("line"),
            Some("{\"next\":2}".to_owned())
        );
        assert_eq!(frames.stats().rejected_oversize, 1);
    }

    #[test]
    fn check_frame_len_reports_probe_value() {
        assert!(check_frame_len(MAX_FRAME_BYTES).is_ok());
        let error = check_frame_len(MAX_FRAME_BYTES + 1).expect_err("over cap");
        assert_eq!(
            error.failure,
            McpFailure::FrameTooLarge {
                limit: MAX_FRAME_BYTES,
                actual: MAX_FRAME_BYTES + 1,
            }
        );
    }
}

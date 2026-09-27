//! Incremental Server-Sent Events (SSE) stream parser (AI-0159).
//!
//! Implements a WHATWG-compliant, zero-dependency, fail-closed incremental
//! parser for `text/event-stream` byte streams.
//!
//! Key security and protocol invariants:
//! - Std-only: zero external dependencies, memory-safe, `#![deny(unsafe_code)]`.
//! - Incremental: handles chunks split at arbitrary byte boundaries (including
//!   multibyte UTF-8 codepoints, line breaks, and field prefixes).
//! - Bounded: enforces hard limits on maximum line length and maximum event size
//!   using [`MAX_FRAGMENT_BYTES`] (64 KiB) to prevent memory exhaustion DoS.
//! - Protocol compliance: supports `data`, `event`, `id`, `retry`, `: comment`
//!   lines, and `\r\n`, `\n`, or `\r` line terminators.
//! - LLM stream awareness: provides [`SseEvent::is_done`] helper for `[DONE]`
//!   termination detection.

use std::fmt;

use crate::stream::MAX_FRAGMENT_BYTES;

/// Maximum line length allowed in an SSE stream before fail-closed rejection.
pub const MAX_SSE_LINE_BYTES: usize = MAX_FRAGMENT_BYTES;

/// Maximum event payload length allowed in an SSE stream before fail-closed rejection.
pub const MAX_SSE_EVENT_BYTES: usize = MAX_FRAGMENT_BYTES;

/// Parsed Server-Sent Event (`text/event-stream`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Event type name (defaults to `"message"` if unspecified).
    pub event: String,
    /// Event data payload (multiline `data:` fields joined with `\n`).
    pub data: String,
    /// Event identifier if set by an `id:` field.
    pub id: Option<String>,
    /// Reconnection time in milliseconds if set by a `retry:` field.
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create a new event with standard fields.
    #[must_use]
    pub fn new(event: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            event: event.into(),
            data: data.into(),
            id: None,
            retry: None,
        }
    }

    /// Whether this event is the standard LLM stream termination marker (`[DONE]`).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

/// Errors returned during SSE parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseError {
    /// Line length exceeded security ceiling before finding a line terminator.
    LineTooLong {
        /// Maximum allowed line bytes.
        max: usize,
        /// Actual line length observed.
        actual: usize,
    },
    /// Event data payload exceeded security ceiling.
    EventTooLong {
        /// Maximum allowed event payload bytes.
        max: usize,
        /// Actual event payload bytes observed.
        actual: usize,
    },
    /// Input bytes are not valid UTF-8.
    InvalidUtf8,
    /// Integer parsing failed for `retry:` field value.
    InvalidRetry(String),
}

impl fmt::Display for SseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LineTooLong { max, actual } => {
                write!(f, "sse line length {actual} exceeded maximum {max}")
            }
            Self::EventTooLong { max, actual } => {
                write!(
                    f,
                    "sse event payload length {actual} exceeded maximum {max}"
                )
            }
            Self::InvalidUtf8 => write!(f, "sse stream contains invalid utf-8"),
            Self::InvalidRetry(val) => write!(f, "invalid sse retry value: '{val}'"),
        }
    }
}

impl std::error::Error for SseError {}

/// Incremental Server-Sent Events (SSE) parser.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    current_event: Option<String>,
    current_data: String,
    current_id: Option<String>,
    current_retry: Option<u64>,
    last_event_id: Option<String>,
    has_event_fields: bool,
}

impl SseParser {
    /// Create a new incremental SSE parser with empty buffers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Last observed event ID in the stream.
    #[must_use]
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// Reset internal state, clearing buffered bytes and partial events.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_event = None;
        self.current_data.clear();
        self.current_id = None;
        self.current_retry = None;
        self.last_event_id = None;
        self.has_event_fields = false;
    }

    /// Feed a chunk of raw stream bytes into the parser.
    ///
    /// Returns all complete [`SseEvent`]s dispatched by this chunk.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, SseError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        loop {
            // Check if buffer contains a line break
            let next_line = match find_line_break(&self.buffer) {
                Some((line_end, advance)) => {
                    if line_end > MAX_SSE_LINE_BYTES {
                        return Err(SseError::LineTooLong {
                            max: MAX_SSE_LINE_BYTES,
                            actual: line_end,
                        });
                    }
                    let line_bytes = self.buffer[..line_end].to_vec();
                    self.buffer.drain(..advance);
                    Some(line_bytes)
                }
                None => {
                    // Check if trailing incomplete line exceeds ceiling
                    if self.buffer.len() > MAX_SSE_LINE_BYTES {
                        return Err(SseError::LineTooLong {
                            max: MAX_SSE_LINE_BYTES,
                            actual: self.buffer.len(),
                        });
                    }
                    None
                }
            };

            let Some(line_bytes) = next_line else {
                break;
            };

            let line_str = std::str::from_utf8(&line_bytes).map_err(|_| SseError::InvalidUtf8)?;

            if line_str.is_empty() {
                // Empty line triggers event dispatch if any fields were collected
                if self.has_event_fields {
                    events.push(self.dispatch_current());
                }
            } else if !line_str.starts_with(':') {
                // Not a comment line, process field
                self.process_line(line_str)?;
            }
        }

        Ok(events)
    }

    /// Finalize parsing at end of stream, flushing any pending unterminated event.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, SseError> {
        if !self.buffer.is_empty() {
            let line_bytes = std::mem::take(&mut self.buffer);
            let line_str = std::str::from_utf8(&line_bytes).map_err(|_| SseError::InvalidUtf8)?;
            if !line_str.starts_with(':') && !line_str.is_empty() {
                self.process_line(line_str)?;
            }
        }

        if self.has_event_fields {
            Ok(Some(self.dispatch_current()))
        } else {
            Ok(None)
        }
    }

    fn process_line(&mut self, line: &str) -> Result<(), SseError> {
        let (field, value) = match line.find(':') {
            Some(colon_idx) => {
                let f = &line[..colon_idx];
                let mut v = &line[colon_idx + 1..];
                // Strip single leading space if present
                if let Some(stripped) = v.strip_prefix(' ') {
                    v = stripped;
                }
                (f, v)
            }
            None => (line, ""),
        };

        match field {
            "event" => {
                self.current_event = Some(value.to_owned());
                self.has_event_fields = true;
            }
            "data" => {
                let additional_len = if self.current_data.is_empty() {
                    value.len()
                } else {
                    1 + value.len() // 1 for '\n'
                };

                let new_len = self.current_data.len() + additional_len;
                if new_len > MAX_SSE_EVENT_BYTES {
                    return Err(SseError::EventTooLong {
                        max: MAX_SSE_EVENT_BYTES,
                        actual: new_len,
                    });
                }

                if !self.current_data.is_empty() {
                    self.current_data.push('\n');
                }
                self.current_data.push_str(value);
                self.has_event_fields = true;
            }
            "id" => {
                // If the id field contains a null byte, the field is ignored
                if !value.contains('\0') {
                    self.current_id = Some(value.to_owned());
                    self.has_event_fields = true;
                }
            }
            "retry" => {
                let parsed = value
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| SseError::InvalidRetry(value.to_owned()))?;
                self.current_retry = Some(parsed);
                self.has_event_fields = true;
            }
            _ => {
                // Other fields are ignored per WHATWG spec
            }
        }

        Ok(())
    }

    fn dispatch_current(&mut self) -> SseEvent {
        let event_name = self
            .current_event
            .take()
            .unwrap_or_else(|| "message".to_owned());
        let data = std::mem::take(&mut self.current_data);
        let id = self.current_id.take();
        let retry = self.current_retry.take();

        if let Some(ref eid) = id {
            self.last_event_id = Some(eid.clone());
        }

        self.has_event_fields = false;

        SseEvent {
            event: event_name,
            data,
            id,
            retry,
        }
    }
}

/// Find line break in buffer according to WHATWG SSE rules:
/// `\r\n` (length 2), `\n` (length 1), or standalone `\r` (length 1).
///
/// Returns `Some((line_end_index, bytes_to_advance))` if a complete line break is found,
/// or `None` if more bytes are needed (e.g. trailing `\r` waiting for potential `\n`).
fn find_line_break(buf: &[u8]) -> Option<(usize, usize)> {
    for (i, &b) in buf.iter().enumerate() {
        if b == b'\n' {
            return Some((i, i + 1));
        } else if b == b'\r' {
            if i + 1 < buf.len() {
                if buf[i + 1] == b'\n' {
                    return Some((i, i + 2));
                }
                return Some((i, i + 1));
            }
            // Trailing `\r` at the very end of the buffer; wait for next byte to check for `\n`
            return None;
        }
    }
    None
}

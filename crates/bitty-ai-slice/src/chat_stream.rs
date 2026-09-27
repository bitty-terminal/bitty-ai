//! Chat completion SSE stream parser and delta adapter (AI-0159).
//!
//! Transforms incremental `text/event-stream` chunks from OpenAI / OpenRouter
//! compatible model endpoints into structured [`ChatStreamDelta`] events and
//! final [`ProviderTurn`] records, connecting the raw SSE protocol to
//! [`bitty_ai_runtime::StreamSink`] and [`bitty_ai_runtime::Fragment`].

use std::fmt;

use bitty_ai_runtime::provider::{ProviderTurn, ProviderUsage, ToolCallRequest};
use bitty_ai_runtime::sse::{SseError, SseParser};
use bitty_ai_runtime::stream::{Fragment, StreamChunk, StreamError, StreamSink};

/// Incremental delta emitted during chat completion streaming.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamDelta {
    /// Incremental content text.
    Content(String),
    /// Incremental tool call invocation data.
    ToolCall {
        /// Tool call index.
        index: usize,
        /// Call identifier if provided.
        id: Option<String>,
        /// Function/tool name if provided.
        name: Option<String>,
        /// Incremental argument chunk.
        arguments: String,
    },
    /// Stream finished with a declared finish reason (e.g. `"stop"`, `"tool_calls"`).
    Finished {
        /// Reason string.
        reason: String,
    },
    /// Final token usage statistics if reported in the stream.
    Usage(ProviderUsage),
}

/// Errors returned during chat stream processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatStreamError {
    /// Underlying SSE framing error.
    Sse(SseError),
    /// JSON payload parsing error.
    JsonParse(String),
    /// Sink emission error.
    Stream(StreamError),
}

impl fmt::Display for ChatStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sse(err) => write!(f, "sse error: {err}"),
            Self::JsonParse(msg) => write!(f, "failed to parse chat stream json: {msg}"),
            Self::Stream(err) => write!(f, "stream sink error: {err}"),
        }
    }
}

impl std::error::Error for ChatStreamError {}

impl From<SseError> for ChatStreamError {
    fn from(err: SseError) -> Self {
        Self::Sse(err)
    }
}

impl From<StreamError> for ChatStreamError {
    fn from(err: StreamError) -> Self {
        Self::Stream(err)
    }
}

/// Internal accumulator for streaming tool calls across multiple chunks.
#[derive(Debug, Default, Clone)]
struct ToolCallAccumulator {
    _index: usize,
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Incremental parser for chat completion SSE streams.
#[derive(Debug, Default)]
pub struct ChatCompletionStreamParser {
    sse: SseParser,
    accumulated_text: String,
    accumulated_tool_calls: Vec<ToolCallAccumulator>,
    usage: Option<ProviderUsage>,
    finished: bool,
}

impl ChatCompletionStreamParser {
    /// Create a new stream parser with default SSE parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Complete accumulated text response across all chunks fed so far.
    #[must_use]
    pub fn accumulated_text(&self) -> &str {
        &self.accumulated_text
    }

    /// Whether the stream reached completion (`[DONE]` or non-null `finish_reason`).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Feed a raw byte chunk from the network stream and return any extracted deltas.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ChatStreamDelta>, ChatStreamError> {
        let events = self.sse.feed(chunk)?;
        let mut deltas = Vec::new();

        for event in events {
            if event.is_done() {
                self.finished = true;
                continue;
            }

            let data_str = event.data.trim();
            if data_str.is_empty() {
                continue;
            }

            let json: serde_json::Value = serde_json::from_str(data_str)
                .map_err(|e| ChatStreamError::JsonParse(format!("{e}: '{data_str}'")))?;

            // Extract content text delta
            if let Some(content) = json
                .pointer("/choices/0/delta/content")
                .and_then(|c| c.as_str())
            {
                if !content.is_empty() {
                    self.accumulated_text.push_str(content);
                    deltas.push(ChatStreamDelta::Content(content.to_owned()));
                }
            }

            // Extract tool call deltas
            if let Some(tool_calls) = json
                .pointer("/choices/0/delta/tool_calls")
                .and_then(|t| t.as_array())
            {
                for tc in tool_calls {
                    let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let id = tc.get("id").and_then(|s| s.as_str()).map(str::to_owned);
                    let name = tc
                        .pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .map(str::to_owned);
                    let arguments = tc
                        .pointer("/function/arguments")
                        .and_then(|a| a.as_str())
                        .unwrap_or("")
                        .to_owned();

                    // Ensure accumulator exists for this index
                    while self.accumulated_tool_calls.len() <= index {
                        self.accumulated_tool_calls.push(ToolCallAccumulator {
                            _index: self.accumulated_tool_calls.len(),
                            id: None,
                            name: String::new(),
                            arguments: String::new(),
                        });
                    }

                    let acc = &mut self.accumulated_tool_calls[index];
                    if let Some(ref call_id) = id {
                        acc.id = Some(call_id.clone());
                    }
                    if let Some(ref fn_name) = name {
                        acc.name.push_str(fn_name);
                    }
                    if !arguments.is_empty() {
                        acc.arguments.push_str(&arguments);
                    }

                    deltas.push(ChatStreamDelta::ToolCall {
                        index,
                        id,
                        name,
                        arguments,
                    });
                }
            }

            // Extract finish reason
            if let Some(finish_reason) = json
                .pointer("/choices/0/finish_reason")
                .and_then(|r| r.as_str())
            {
                self.finished = true;
                deltas.push(ChatStreamDelta::Finished {
                    reason: finish_reason.to_owned(),
                });
            }

            // Extract usage report
            if let Some(usage_obj) = json.get("usage") {
                let input_tokens = usage_obj
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
                let output_tokens = usage_obj
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
                let usage = ProviderUsage {
                    input_tokens,
                    output_tokens,
                };
                self.usage = Some(usage);
                deltas.push(ChatStreamDelta::Usage(usage));
            }
        }

        Ok(deltas)
    }

    /// Pipe a single content delta to a [`StreamSink`] as a sequenced Markdown [`Fragment`].
    pub fn pipe_to_sink(
        &self,
        delta: &ChatStreamDelta,
        sink: &mut dyn StreamSink,
        seq: &mut u32,
    ) -> Result<(), ChatStreamError> {
        if let ChatStreamDelta::Content(text) = delta {
            let fragment = Fragment::markdown(text.as_bytes().to_vec());
            let chunk = StreamChunk {
                seq: *seq,
                total: *seq + 1,
                is_final: true,
                fragment,
            };
            *seq += 1;
            sink.emit(chunk)?;
        }
        Ok(())
    }

    /// Finalize stream processing and construct the consolidated [`ProviderTurn`].
    pub fn finish(mut self) -> Result<ProviderTurn, ChatStreamError> {
        // Drain any unterminated event from the SSE parser
        if let Some(event) = self.sse.finish()? {
            if !event.is_done() {
                let data_str = event.data.trim();
                if !data_str.is_empty() {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data_str) {
                        if let Some(content) = json
                            .pointer("/choices/0/delta/content")
                            .and_then(|c| c.as_str())
                        {
                            self.accumulated_text.push_str(content);
                        }
                    }
                }
            }
        }

        let mut tool_calls = Vec::new();
        for acc in self.accumulated_tool_calls {
            if !acc.name.is_empty() {
                tool_calls.push(ToolCallRequest {
                    name: acc.name,
                    arguments: acc.arguments.into_bytes(),
                });
            }
        }

        let usage = self.usage.unwrap_or_default();

        Ok(ProviderTurn {
            text: self.accumulated_text,
            tool_calls,
            latency_ms: 0,
            usage,
        })
    }
}

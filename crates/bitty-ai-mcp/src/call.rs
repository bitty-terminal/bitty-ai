//! `tools/call` dispatch with a whole-call deadline.
//!
//! [`call_tool`] sends one `tools/call` frame and waits up to `timeout_ms`
//! for the matching response, answering interleaved server requests inline.
//! Arguments are opaque JSON bytes checked against the 16 KiB bus bound
//! *before* any byte is written (oversize calls never touch the transport).
//!
//! Outcome mapping:
//!
//! - `isError: true` becomes [`McpCallError::Failed`]: the tool executed and
//!   reported failure.
//! - A JSON-RPC `error` answer with code `-32600` (invalid request),
//!   `-32601` (method not found), or `-32602` (invalid params) becomes
//!   [`McpCallError::ProtocolRejected`]: the server refused the frame before
//!   any effect, so the call never executed. All other `error` answers
//!   (including `-32603` and the server `-32000` range) stay
//!   [`McpCallError::Failed`].
//! - Results concatenate `content[].text` parts; a `structuredContent`-only
//!   result synthesizes its text from the raw structured bytes (bounded).
//! - Results past 16 KiB become [`McpCallError::ResultRejected`]: rejected
//!   with a typed bound, never silently truncated.
//! - Deadline expiry or a mid-call close becomes
//!   [`McpCallError::Unknown`]: the effect may have happened, so the caller
//!   reconciles before retry and never blindly retries.

use std::time::Instant;

use bitty_ai_runtime::tool::ToolSuccess;

use crate::error::{McpError, McpFailure, McpStage, bound_error_text};
use crate::handshake::wait_for_response;
use crate::json::{escaped, find_bool_field, find_object_field, find_raw_field, find_string_field};
use crate::{MAX_TIMEOUT_MS, McpTransport};

/// Maximum tool argument bytes (runtime `TB-3` parity).
pub const MAX_CALL_ARGUMENTS_BYTES: usize = 16 * 1024;
/// Maximum tool result bytes (runtime `TB-6` parity).
pub const MAX_CALL_RESULT_BYTES: usize = 16 * 1024;
/// Maximum summary bytes for the synthesized L0 summary (runtime context
/// parity; the full text stays in `ToolSuccess.data`).
pub const MAX_CALL_SUMMARY_BYTES: usize = 4 * 1024;

/// Whole-call outcome: success, or a typed error the bridge maps onto the
/// [`bitty_ai_runtime::tool::ToolExecutor`] seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpCallError {
    /// Transport failure (broken pipe, closed child, oversize frame).
    /// Mid-call, every transport failure means [`McpCallOutcome::Unknown`]
    /// at the bridge: the effect may have happened.
    Transport(McpError),
    /// Arguments exceed [`MAX_CALL_ARGUMENTS_BYTES`] (checked pre-contact).
    ArgumentsTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Result exceeds [`MAX_CALL_RESULT_BYTES`] (rejected, never truncated).
    ResultRejected {
        /// Bound in bytes.
        limit: usize,
        /// Observed text bytes.
        actual: usize,
    },
    /// The tool executed and reported failure (`isError: true` answers and
    /// JSON-RPC `error` answers outside the pre-execution protocol set).
    Failed {
        /// Tool display name.
        tool: String,
        /// Bounded server-reported reason.
        reason: String,
    },
    /// The server rejected the call at the protocol level before any effect
    /// (JSON-RPC `-32600` invalid request, `-32601` method not found,
    /// `-32602` invalid params). Non-executed: distinct from [`McpCallError::Failed`]
    /// (executed, then failed) and [`McpCallError::Unknown`] (uncertain
    /// effect, reconcile before retry). Retryable only with fixed params as
    /// a new call, never a blind retry of the same bytes.
    ProtocolRejected {
        /// Tool display name.
        tool: String,
        /// JSON-RPC error code (`-32600`, `-32601`, or `-32602`).
        code: i32,
        /// Bounded server-reported message (no payload echo: the error
        /// `data` member is never read).
        message: String,
    },
    /// Acknowledgement was lost (deadline or mid-call close): reconcile
    /// before retry, never blindly retry.
    Unknown {
        /// Tool display name.
        tool: String,
        /// What is uncertain (bounded).
        reason: String,
    },
}

impl std::fmt::Display for McpCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(error) => write!(f, "{error}"),
            Self::ArgumentsTooLarge { limit, actual } => write!(
                f,
                "tool arguments of {actual} bytes exceed {limit} byte limit"
            ),
            Self::ResultRejected { limit, actual } => write!(
                f,
                "tool result of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::Failed { tool, reason } => write!(f, "tool '{tool}' reported failure: {reason}"),
            Self::ProtocolRejected {
                tool,
                code,
                message,
            } => {
                write!(
                    f,
                    "tool '{tool}' protocol rejected (code {code}): {message}"
                )
            }
            Self::Unknown { tool, reason } => write!(f, "tool '{tool}' effect unknown: {reason}"),
        }
    }
}

impl std::error::Error for McpCallError {}

/// Success side of [`call_tool`] (mirrors the bus `ToolSuccess` shape one
/// level down so tests can assert without the runtime seam).
pub type McpCallOutcome = ToolSuccess;

/// Build one `tools/call` request frame.
///
/// `arguments` are opaque JSON bytes embedded verbatim (empty becomes `{}`);
/// the bound is enforced by [`call_tool`] before this runs.
#[must_use]
pub fn call_request(id: u64, raw_tool: &str, arguments: &[u8]) -> String {
    let args_text = if arguments.is_empty() {
        "{}".to_owned()
    } else {
        String::from_utf8_lossy(arguments).into_owned()
    };
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"tools/call\",\"params\":{{\"name\":\"{}\",\"arguments\":{args_text}}}}}",
        escaped(raw_tool)
    )
}

/// Extract the synthesized result text from a `tools/call` response frame.
///
/// Returns the concatenated `content[].text` parts, or — when no text part
/// exists but `structuredContent` does — the raw structured bytes as text.
/// Reports the observed text length alongside so callers can enforce the
/// result bound without retaining more than the frame already holds.
///
/// # Errors
///
/// Returns [`McpCallError::Failed`] for `isError: true` answers and
/// JSON-RPC `error` answers outside the pre-execution protocol set, and
/// [`McpCallError::ProtocolRejected`] for `error` answers with code
/// `-32600`, `-32601`, or `-32602` (rejected before any effect).
pub fn parse_call_result(line: &str, tool: &str) -> Result<(String, usize), McpCallError> {
    if let Some(error) = find_object_field(line, "error") {
        let code_raw = find_raw_field(error, "code").unwrap_or("?");
        let message = find_string_field(error, "message").unwrap_or_default();
        // Standard pre-execution rejections (AI-0204): invalid request,
        // method not found, invalid params. The server refused the frame
        // before any effect, so this is never `Failed` (which claims an
        // executed effect). The error `data` member is deliberately never
        // read: no request or result payload bytes enter the diagnostic.
        if let Ok(code) = code_raw.trim().parse::<i32>() {
            // Owner scope (PX-0897): exactly -32600/-32601/-32602, no wider
            // range, no server -32000 range, no -32603.
            if (-32602..=-32600).contains(&code) {
                return Err(McpCallError::ProtocolRejected {
                    tool: tool.to_owned(),
                    code,
                    message: bound_error_text(&message, crate::error::MAX_ERROR_TEXT_BYTES),
                });
            }
        }
        return Err(McpCallError::Failed {
            tool: tool.to_owned(),
            reason: bound_error_text(
                &format!("server error {code_raw}: {message}"),
                crate::error::MAX_ERROR_TEXT_BYTES,
            ),
        });
    }
    let result = find_object_field(line, "result").ok_or_else(|| McpCallError::Failed {
        tool: tool.to_owned(),
        reason: "call answer carries no result".to_owned(),
    })?;
    if find_bool_field(result, "isError") == Some(true) {
        let reason = collect_text_parts(result);
        let reason = if reason.text.is_empty() {
            "tool reported error".to_owned()
        } else {
            reason.text
        };
        return Err(McpCallError::Failed {
            tool: tool.to_owned(),
            reason: bound_error_text(&reason, crate::error::MAX_ERROR_TEXT_BYTES),
        });
    }
    let collected = collect_text_parts(result);
    // Always report the observed length, even when the retained text is
    // empty (the probe stops retaining past cap-plus-one while still
    // counting): the caller enforces the result bound from the length.
    if !collected.text.is_empty() || collected.len > 0 {
        return Ok((collected.text, collected.len));
    }
    if let Some(structured) = find_raw_field(result, "structuredContent") {
        return Ok((structured.to_owned(), structured.len()));
    }
    Ok((String::new(), 0))
}

/// Concatenated text parts plus their byte length.
struct CollectedText {
    text: String,
    len: usize,
}

/// Concatenate every `{"type":"text","text":"..."}` part of a result
/// `content` array, accumulating at most [`MAX_CALL_RESULT_BYTES`] plus one
/// probe byte so oversize results are detected without retaining more.
fn collect_text_parts(result: &str) -> CollectedText {
    let mut text = String::new();
    let mut len = 0_usize;
    let Some(array) = find_raw_field(result, "content") else {
        return CollectedText { text, len };
    };
    let mut rest = array;
    while let Some(key_at) = rest.find("\"text\"") {
        let after_key = &rest[key_at + 6..];
        let Some(colon) = after_key.find(':') else {
            break;
        };
        let value = after_key[colon + 1..].trim_start();
        if !value.starts_with('"') {
            rest = &value[1..];
            continue;
        }
        let probe = format!("{{\"text\":{value}");
        // Reuse the string-field parser on a synthetic object prefix.
        let part = crate::json::find_string_field(&probe, "text").unwrap_or_default();
        len += part.len();
        if len <= MAX_CALL_RESULT_BYTES + 1 {
            text.push_str(&part);
        }
        // Advance past the parsed value: find its closing quote.
        let bytes = value.as_bytes();
        let mut index = 1_usize;
        while index < bytes.len() {
            match bytes[index] {
                b'\\' => index += 2,
                b'"' => {
                    index += 1;
                    break;
                }
                _ => index += 1,
            }
        }
        rest = &value[index.min(value.len())..];
        if len > MAX_CALL_RESULT_BYTES + 1 {
            break;
        }
    }
    CollectedText { text, len }
}

/// Dispatch one tool call with a whole-call deadline.
///
/// `id` is the JSON-RPC request id (owned by the connection: unique per
/// in-flight call; calls are single-flight, so a counter suffices).
/// `raw_tool` is the server-side name, `tool` the sanitized display name
/// for errors, `arguments` the opaque JSON argument bytes.
///
/// # Errors
///
/// Returns [`McpCallError::ArgumentsTooLarge`] before any transport contact,
/// [`McpCallError::Failed`] for server-reported failures,
/// [`McpCallError::ProtocolRejected`] for pre-execution protocol rejections,
/// [`McpCallError::ResultRejected`] past 16 KiB, [`McpCallError::Unknown`]
/// on deadline or mid-call close, and [`McpCallError::Transport`] for
/// frame-level faults.
pub fn call_tool(
    transport: &mut dyn McpTransport,
    id: u64,
    raw_tool: &str,
    tool: &str,
    arguments: &[u8],
    cwd: &str,
    timeout_ms: u64,
) -> Result<ToolSuccess, McpCallError> {
    if arguments.len() > MAX_CALL_ARGUMENTS_BYTES {
        return Err(McpCallError::ArgumentsTooLarge {
            limit: MAX_CALL_ARGUMENTS_BYTES,
            actual: arguments.len(),
        });
    }
    let bound = timeout_ms.clamp(1, MAX_TIMEOUT_MS);
    let deadline = Instant::now() + std::time::Duration::from_millis(bound);
    transport
        .send_line(&call_request(id, raw_tool, arguments))
        .map_err(McpCallError::Transport)?;
    let answer = wait_for_response(transport, id, cwd, deadline, bound).map_err(|error| {
        if matches!(error.failure, McpFailure::Timeout { .. }) {
            McpCallError::Unknown {
                tool: tool.to_owned(),
                reason: bound_error_text(
                    &format!("no answer within {bound}ms"),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            }
        } else {
            McpCallError::Unknown {
                tool: tool.to_owned(),
                reason: bound_error_text(
                    &format!("transport closed mid-call: {error}"),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            }
        }
    })?;
    let (text, observed) = parse_call_result(&answer, tool).map_err(|error| match error {
        McpCallError::Transport(inner) => McpCallError::Unknown {
            tool: tool.to_owned(),
            reason: bound_error_text(
                &format!("transport closed mid-call: {inner}"),
                crate::error::MAX_ERROR_TEXT_BYTES,
            ),
        },
        other => other,
    })?;
    if observed > MAX_CALL_RESULT_BYTES {
        return Err(McpCallError::ResultRejected {
            limit: MAX_CALL_RESULT_BYTES,
            actual: observed,
        });
    }
    let summary = truncate_to(&text, MAX_CALL_SUMMARY_BYTES);
    ToolSuccess::new(summary, text.into_bytes()).map_err(|error| match error {
        bitty_ai_runtime::tool::ToolError::SummaryTooLarge { .. }
        | bitty_ai_runtime::tool::ToolError::ResultTooLarge { .. } => {
            McpCallError::ResultRejected {
                limit: MAX_CALL_RESULT_BYTES,
                actual: observed,
            }
        }
        _ => McpCallError::Failed {
            tool: tool.to_owned(),
            reason: bound_error_text(
                &format!("result rejected: {error}"),
                crate::error::MAX_ERROR_TEXT_BYTES,
            ),
        },
    })
}

/// Truncate `value` to `max_bytes` on a UTF-8 boundary.
fn truncate_to(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

/// Map a call error onto the stage failures for bridge reporting.
#[must_use]
pub fn call_stage(error: &McpCallError) -> McpError {
    match error {
        McpCallError::Transport(inner) => inner.clone(),
        McpCallError::ArgumentsTooLarge { limit, actual } => McpError::new(
            McpStage::CallTool,
            McpFailure::ArgumentsTooLarge {
                limit: *limit,
                actual: *actual,
            },
        ),
        McpCallError::ResultRejected { limit, actual } => McpError::new(
            McpStage::CallTool,
            McpFailure::ResultTooLarge {
                limit: *limit,
                actual: *actual,
            },
        ),
        McpCallError::Failed { tool, reason } => McpError::new(
            McpStage::CallTool,
            McpFailure::ToolFailed {
                tool: tool.clone(),
                reason: reason.clone(),
            },
        ),
        // Bridge reporting keeps the `CallTool`/`ToolFailed` shape (a
        // JSON-RPC error answer, fatal: never blindly retried); the distinct
        // non-executed outcome travels on the `McpCallError` carrier into
        // `ToolError::ProtocolRejected`, where the message marks the code.
        McpCallError::ProtocolRejected {
            tool,
            code,
            message,
        } => McpError::new(
            McpStage::CallTool,
            McpFailure::ToolFailed {
                tool: tool.clone(),
                reason: bound_error_text(
                    &format!("protocol rejected (code {code}): {message}"),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        ),
        McpCallError::Unknown { tool, reason } => McpError::new(
            McpStage::CallTool,
            McpFailure::EffectUnknown {
                tool: tool.clone(),
                reason: reason.clone(),
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeTransport {
        inbound: VecDeque<String>,
        sent: Vec<String>,
    }

    impl McpTransport for FakeTransport {
        fn send_line(&mut self, line: &str) -> Result<(), McpError> {
            self.sent.push(line.to_owned());
            Ok(())
        }

        fn recv_line(&mut self, _timeout_ms: u64) -> Result<Option<String>, McpError> {
            Ok(self.inbound.pop_front())
        }
    }

    fn answer(id: u64, result: &str) -> String {
        format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{result}}}")
    }

    #[test]
    fn text_parts_concatenate() {
        let line = answer(
            1,
            "{\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\"b\"}]}",
        );
        let (text, len) = parse_call_result(&line, "mcp_demo_echo").expect("text");
        assert_eq!(text, "ab");
        assert_eq!(len, 2);
    }

    #[test]
    fn structured_only_synthesizes_text() {
        let line = answer(1, "{\"structuredContent\":{\"n\":42}}");
        let (text, _) = parse_call_result(&line, "mcp_demo_echo").expect("synth");
        assert_eq!(text, "{\"n\":42}");
    }

    #[test]
    fn is_error_maps_to_failed() {
        let line = answer(
            1,
            "{\"content\":[{\"type\":\"text\",\"text\":\"boom\"}],\"isError\":true}",
        );
        let error = parse_call_result(&line, "mcp_demo_fail").expect_err("failed");
        assert!(matches!(error, McpCallError::Failed { .. }));
        assert!(error.to_string().contains("boom"));
    }

    #[test]
    fn rpc_error_maps_to_failed() {
        // `-32603` (internal error) and the server `-32000` range are not
        // pre-execution rejections: they keep the executed-failure mapping.
        for (code, message) in [("-32603", "boom"), ("-32000", "transport busy")] {
            let line = format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{{\"code\":{code},\"message\":\"{message}\"}}}}"
            );
            let error = parse_call_result(&line, "mcp_demo_echo").expect_err("failed");
            assert!(
                matches!(error, McpCallError::Failed { .. }),
                "code {code} must stay Failed, got {error:?}"
            );
        }
    }

    #[test]
    fn pre_execution_codes_map_to_protocol_rejected() {
        // `-32600`/`-32601`/`-32602` never executed: distinct non-executed
        // outcome, never `Failed` (executed-then-failed).
        for code in [-32600, -32601, -32602] {
            let line = format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{{\"code\":{code},\"message\":\"bad frame\"}}}}"
            );
            let error = parse_call_result(&line, "mcp_demo_echo").expect_err("rejected");
            match &error {
                McpCallError::ProtocolRejected {
                    tool,
                    code: seen,
                    message,
                } => {
                    assert_eq!(tool, "mcp_demo_echo");
                    assert_eq!(*seen, code);
                    assert!(message.contains("bad frame"));
                }
                other => panic!("code {code} must reject, got {other:?}"),
            }
            assert!(!matches!(error, McpCallError::Failed { .. }));
        }
    }

    #[test]
    fn protocol_rejection_carries_no_payload_echo() {
        // The error `data` member (request/result payload bytes) is never
        // read into the diagnostic: only code plus message travel.
        let line = "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32602,\"message\":\"bad params\",\"data\":{\"secret\":\"canary-payload-bytes\"}}}";
        let error = parse_call_result(line, "mcp_demo_echo").expect_err("rejected");
        match &error {
            McpCallError::ProtocolRejected { message, .. } => {
                assert!(!message.contains("canary-payload-bytes"));
            }
            other => panic!("expected ProtocolRejected, got {other:?}"),
        }
        let render = error.to_string();
        assert!(render.contains("-32602"));
        assert!(!render.contains("canary-payload-bytes"));
        assert!(!render.contains('\n'));
    }

    #[test]
    fn oversize_args_refuse_before_contact() {
        let mut transport = FakeTransport {
            inbound: VecDeque::new(),
            sent: Vec::new(),
        };
        let big = vec![b'x'; MAX_CALL_ARGUMENTS_BYTES + 1];
        let error = call_tool(
            &mut transport,
            1,
            "echo",
            "mcp_demo_echo",
            &big,
            "/tmp/bitty",
            1_000,
        )
        .expect_err("oversize args");
        assert!(matches!(error, McpCallError::ArgumentsTooLarge { .. }));
        assert!(transport.sent.is_empty());
    }

    #[test]
    fn oversize_result_rejects_never_truncates() {
        let big_text = "r".repeat(MAX_CALL_RESULT_BYTES + 10);
        let line = answer(
            1,
            &format!("{{\"content\":[{{\"type\":\"text\",\"text\":\"{big_text}\"}}]}}"),
        );
        let mut transport = FakeTransport {
            inbound: vec![line].into_iter().collect(),
            sent: Vec::new(),
        };
        let error = call_tool(
            &mut transport,
            1,
            "echo",
            "mcp_demo_echo",
            b"{}",
            "/tmp/bitty",
            1_000,
        )
        .expect_err("oversize result");
        assert!(matches!(error, McpCallError::ResultRejected { .. }));
    }

    #[test]
    fn empty_result_is_empty_success() {
        let line = answer(1, "{}");
        let mut transport = FakeTransport {
            inbound: vec![line].into_iter().collect(),
            sent: Vec::new(),
        };
        let success = call_tool(
            &mut transport,
            1,
            "echo",
            "mcp_demo_echo",
            b"{}",
            "/tmp/bitty",
            1_000,
        )
        .expect("empty success");
        assert!(success.data.is_empty());
    }
}

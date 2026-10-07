//! Bounded names-only MCP errors.
//!
//! [`McpError`] carries exactly three fields: the pipeline [`McpStage`], the
//! typed [`McpFailure`], and whether the failure is retryable. `Display`
//! quotes only names (server ids, program names, tool names), bounds, and
//! counts — never secret values, payload bytes, or helper output. Every
//! string field is scrubbed to printable ASCII and truncated at construction
//! through [`bound_error_text`], mirroring the runtime `bound_reason`
//! precedent (which stays `pub(crate)` there, so the bound is duplicated
//! here and pinned by tests).

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Maximum bytes kept for any string carried by [`McpError`].
///
/// Names (server ids, programs, tools) are short by construction; reasons
/// share the runtime 512-byte diagnostic bound so denial text stays usable
/// without becoming a log-fill vector.
pub const MAX_ERROR_TEXT_BYTES: usize = 512;

/// Maximum bytes kept for a name echoed into [`McpError`].
pub const MAX_ERROR_NAME_BYTES: usize = 128;

/// Scrub `value` to printable ASCII (`0x20..=0x7E`) and truncate to
/// `max_bytes`, mirroring the runtime `bound_reason` precedent.
///
/// Deterministic and silent (no ellipsis marker that could itself exceed the
/// bound). Every [`McpError`] string field passes through this at
/// construction, so `Display` output is safe to log directly.
#[must_use]
pub fn bound_error_text(value: &str, max_bytes: usize) -> String {
    let mut bounded = String::with_capacity(value.len().min(max_bytes));
    for character in value.chars() {
        if bounded.len() >= max_bytes {
            break;
        }
        if character.is_ascii_graphic() || character == ' ' {
            bounded.push(character);
        } else {
            bounded.push('?');
        }
    }
    bounded
}

/// Bound a name echoed into [`McpError`] ([`MAX_ERROR_NAME_BYTES`]).
#[must_use]
pub fn bound_error_name(value: &str) -> String {
    bound_error_text(value, MAX_ERROR_NAME_BYTES)
}

/// Pipeline stage that produced the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpStage {
    /// Server configuration validation.
    Config,
    /// Child process spawn and supervision.
    Spawn,
    /// Credential reference resolution into the child environment.
    Credential,
    /// `initialize` handshake.
    Handshake,
    /// `tools/list` import.
    ListTools,
    /// `tools/call` dispatch.
    CallTool,
    /// Newline-delimited frame decoding.
    Frame,
    /// Connection supervision (cooldown, liveness, shutdown).
    Supervise,
    /// [`crate::bridge::McpToolAdapter`] policy gates.
    Bridge,
}

impl Display for McpStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let label = match self {
            Self::Config => "config",
            Self::Spawn => "spawn",
            Self::Credential => "credential",
            Self::Handshake => "handshake",
            Self::ListTools => "list-tools",
            Self::CallTool => "call-tool",
            Self::Frame => "frame",
            Self::Supervise => "supervise",
            Self::Bridge => "bridge",
        };
        f.write_str(label)
    }
}

/// Typed MCP failure. Every string field holds a bounded, scrubbed name or a
/// caller-supplied reason that itself must be names-only (callers never
/// interpolate values, payloads, or helper output).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpFailure {
    /// Server configuration violates its fail-closed shape.
    InvalidConfig {
        /// What was rejected (names and bounds only).
        reason: String,
    },
    /// The child process failed to spawn.
    SpawnFailed {
        /// Program name only.
        program: String,
    },
    /// The child exited or its pipes closed unexpectedly.
    ChildExited {
        /// Bounded exit description (code text only).
        detail: String,
    },
    /// The transport is dead; only an explicit reconnect may revive it (no
    /// auto-restart).
    TransportClosed,
    /// A credential reference is missing or empty.
    CredentialMissing {
        /// Reference name only (`env:VAR` / `cmd:program`).
        name: String,
    },
    /// A credential helper failed (spawn, non-zero exit, timeout, empty,
    /// oversize, or non-UTF-8 output).
    CredentialFailed {
        /// Reference name only.
        name: String,
    },
    /// Handshake response was not acceptable.
    HandshakeRejected {
        /// Bounded detail (protocol facts only).
        detail: String,
    },
    /// Server protocol version differs from the [`crate::PROTOCOL_VERSION`]
    /// pin.
    VersionMismatch {
        /// Expected pin.
        expected: String,
        /// Server-reported version (bounded).
        got: String,
    },
    /// Server offers no tool capability.
    NoToolCapability,
    /// A frame exceeds [`crate::frame::MAX_FRAME_BYTES`].
    FrameTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes (cap-plus-one probe value, never the stream size).
        actual: usize,
    },
    /// `tools/list` pagination exceeded [`crate::tools::MAX_TOOL_PAGES`].
    TooManyPages {
        /// Bound.
        limit: usize,
    },
    /// A pagination cursor repeated (looping server).
    DuplicateCursor {
        /// Bounded cursor echo (opaque token, truncated).
        cursor: String,
    },
    /// Two imported tools sanitize to the same registered name.
    DuplicateTool {
        /// Rejected sanitized name.
        name: String,
    },
    /// No imported tool matches the requested name.
    UnknownTool {
        /// Requested name.
        name: String,
    },
    /// The tool is not allowlisted: refused with zero transport contact.
    AllowlistDeny {
        /// Requested name.
        name: String,
    },
    /// The runtime authorizer hook refused the call.
    AuthorizerDeny {
        /// Bounded hook reason.
        reason: String,
    },
    /// The consent ledger refused the call.
    ConsentDeny {
        /// Bounded ledger reason.
        reason: String,
    },
    /// The `inspect` tier may not reach a mutating tool.
    TierDeny {
        /// Bounded detail (tool name only).
        reason: String,
    },
    /// Call arguments exceed the 16 KiB bus bound.
    ArgumentsTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Tool result exceeds the 16 KiB bus bound (rejected, never truncated).
    ResultTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// The tool executed and reported failure (`isError: true` or a
    /// JSON-RPC error answer).
    ToolFailed {
        /// Tool display name.
        tool: String,
        /// Bounded server-reported reason.
        reason: String,
    },
    /// The effect may have happened but acknowledgement was lost (timeout or
    /// mid-call close): reconcile before retry, never blindly retry.
    EffectUnknown {
        /// Tool display name.
        tool: String,
        /// What is uncertain (bounded).
        reason: String,
    },
    /// A bounded wait elapsed with no answer.
    Timeout {
        /// Wait bound in milliseconds.
        timeout_ms: u64,
    },
    /// Supervision misuse (double connect, use while disconnected).
    Supervision {
        /// Bounded detail (server id only).
        detail: String,
    },
    /// Underlying I/O failed (error-kind name only, never payload bytes).
    Io {
        /// Bounded context (operation name only).
        context: String,
    },
}

impl McpFailure {
    /// Whether the failure may resolve on retry (after cooldown/reconcile).
    ///
    /// Timeouts, child exits, closed transports, spawn failures, and unknown
    /// effects are retryable; validation, authorization, consent, bounds,
    /// duplicates, and tool-reported failures are not.
    #[must_use]
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::SpawnFailed { .. }
                | Self::ChildExited { .. }
                | Self::TransportClosed
                | Self::EffectUnknown { .. }
                | Self::Timeout { .. }
                | Self::Io { .. }
        )
    }
}

impl Display for McpFailure {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid config: {reason}"),
            Self::SpawnFailed { program } => write!(f, "failed to spawn '{program}'"),
            Self::ChildExited { detail } => write!(f, "child exited: {detail}"),
            Self::TransportClosed => write!(f, "transport closed (explicit reconnect required)"),
            Self::CredentialMissing { name } => {
                write!(f, "credential '{name}' is missing or empty")
            }
            Self::CredentialFailed { name } => write!(f, "credential '{name}' helper failed"),
            Self::HandshakeRejected { detail } => write!(f, "handshake rejected: {detail}"),
            Self::VersionMismatch { expected, got } => {
                write!(
                    f,
                    "protocol version mismatch: expected '{expected}', got '{got}'"
                )
            }
            Self::NoToolCapability => write!(f, "server offers no tool capability"),
            Self::FrameTooLarge { limit, actual } => {
                write!(f, "frame of {actual} bytes exceeds {limit} byte limit")
            }
            Self::TooManyPages { limit } => {
                write!(f, "tool listing exceeds {limit} pages")
            }
            Self::DuplicateCursor { cursor } => {
                write!(f, "tool listing repeated cursor '{cursor}'")
            }
            Self::DuplicateTool { name } => write!(f, "duplicate tool: {name}"),
            Self::UnknownTool { name } => write!(f, "unknown tool: {name}"),
            Self::AllowlistDeny { name } => write!(f, "tool '{name}' is not allowlisted"),
            Self::AuthorizerDeny { reason } => write!(f, "authorizer denied: {reason}"),
            Self::ConsentDeny { reason } => write!(f, "consent denied: {reason}"),
            Self::TierDeny { reason } => write!(f, "tier denied: {reason}"),
            Self::ArgumentsTooLarge { limit, actual } => write!(
                f,
                "tool arguments of {actual} bytes exceed {limit} byte limit"
            ),
            Self::ResultTooLarge { limit, actual } => write!(
                f,
                "tool result of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::ToolFailed { tool, reason } => {
                write!(f, "tool '{tool}' reported failure: {reason}")
            }
            Self::EffectUnknown { tool, reason } => {
                write!(f, "tool '{tool}' effect unknown: {reason}")
            }
            Self::Timeout { timeout_ms } => write!(f, "timed out after {timeout_ms}ms"),
            Self::Supervision { detail } => write!(f, "supervision: {detail}"),
            Self::Io { context } => write!(f, "i/o failure: {context}"),
        }
    }
}

/// Bounded names-only MCP error: stage plus typed failure plus retryability.
///
/// Construct through [`McpError::new`] or the stage helpers so string fields
/// are scrubbed and bounded at the boundary; `Display` renders
/// `mcp <stage>: <failure>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpError {
    /// Pipeline stage that produced the error.
    pub stage: McpStage,
    /// Typed failure.
    pub failure: McpFailure,
    /// Whether the failure may resolve on retry.
    pub retryable: bool,
}

impl McpError {
    /// Build an error, deriving `retryable` from the failure.
    #[must_use]
    pub fn new(stage: McpStage, failure: McpFailure) -> Self {
        let retryable = failure.retryable();
        Self {
            stage,
            failure,
            retryable,
        }
    }

    /// Fail-closed configuration rejection with a names-only reason.
    #[must_use]
    pub fn invalid_config(reason: &str) -> Self {
        Self::new(
            McpStage::Config,
            McpFailure::InvalidConfig {
                reason: bound_error_text(reason, MAX_ERROR_TEXT_BYTES),
            },
        )
    }
}

impl Display for McpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "mcp {}: {}", self.stage, self.failure)
    }
}

impl std::error::Error for McpError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_display_safe(value: &str) {
        assert!(
            value.len() <= MAX_ERROR_TEXT_BYTES + 64,
            "error text too long: {value:?}"
        );
        assert!(
            value.bytes().all(|byte| (0x20..=0x7E).contains(&byte)),
            "error text is not printable ASCII: {value:?}"
        );
        assert!(!value.contains('\n'), "newline leaked: {value:?}");
    }

    #[test]
    fn display_quotes_names_never_values() {
        let cases = vec![
            McpError::new(
                McpStage::CallTool,
                McpFailure::ToolFailed {
                    tool: bound_error_name("mcp_demo_echo"),
                    reason: bound_error_text("boom", MAX_ERROR_TEXT_BYTES),
                },
            ),
            McpError::new(
                McpStage::Frame,
                McpFailure::FrameTooLarge {
                    limit: 262_144,
                    actual: 262_145,
                },
            ),
            McpError::new(McpStage::Handshake, McpFailure::NoToolCapability),
            McpError::new(
                McpStage::Config,
                McpFailure::InvalidConfig {
                    reason: bound_error_text("command is required", MAX_ERROR_TEXT_BYTES),
                },
            ),
        ];
        for error in cases {
            assert_display_safe(&error.to_string());
            assert!(error.to_string().starts_with("mcp "));
        }
    }

    #[test]
    fn hostile_strings_are_scrubbed_and_bounded() {
        let hostile = format!("bad\nname\r\u{1b}[2J\u{7f}{}", "x".repeat(2048));
        let scrubbed = bound_error_name(&hostile);
        assert!(scrubbed.len() <= MAX_ERROR_NAME_BYTES);
        assert!(scrubbed.starts_with("bad?name?"));
        assert_display_safe(&scrubbed);
        let long_reason = bound_error_text(&hostile, MAX_ERROR_TEXT_BYTES);
        assert_eq!(long_reason.len(), MAX_ERROR_TEXT_BYTES);
    }

    #[test]
    fn retryable_flags_are_fail_closed() {
        let retryable = [
            McpFailure::Timeout { timeout_ms: 10 },
            McpFailure::TransportClosed,
            McpFailure::SpawnFailed {
                program: "helper".to_owned(),
            },
            McpFailure::EffectUnknown {
                tool: "t".to_owned(),
                reason: "lost".to_owned(),
            },
        ];
        for failure in retryable {
            assert!(McpError::new(McpStage::CallTool, failure).retryable);
        }
        let fatal = [
            McpFailure::NoToolCapability,
            McpFailure::AllowlistDeny {
                name: "t".to_owned(),
            },
            McpFailure::DuplicateTool {
                name: "t".to_owned(),
            },
            McpFailure::ToolFailed {
                tool: "t".to_owned(),
                reason: "no".to_owned(),
            },
            McpFailure::FrameTooLarge {
                limit: 1,
                actual: 2,
            },
        ];
        for failure in fatal {
            assert!(!McpError::new(McpStage::CallTool, failure).retryable);
        }
    }
}

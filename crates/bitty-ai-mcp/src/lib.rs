//! Hand-rolled synchronous stdio MCP client (`bitty-ai-mcp`).
//!
//! **Status: draft scope (AI-0179).** This crate implements the owner decision
//! `DEC-0006`: a minimal Model Context Protocol client over newline-delimited
//! JSON-RPC on a supervised child process, written with `std` only. The `rmcp`
//! SDK stays deferred; nothing here claims conformance to it.
//!
//! ## Placement and trust
//!
//! [`PLACEMENT`] is `"mcp"`: this client satisfies the S6 `mcp.invoke`
//! contribution pattern (param-required head) from the host side. Every byte
//! read from the child is untrusted server output: frames are capped at
//! [`frame::MAX_FRAME_BYTES`], malformed lines are dropped and counted, and
//! tool results stay `is_untrusted_surface` observations once they cross the
//! [`bitty_ai_runtime::tool::ToolBus`] boundary (this crate only produces
//! bounded [`bitty_ai_runtime::tool::ToolSuccess`] values; the bus labels
//! them).
//!
//! ## Determinism and scope rules
//!
//! - Synchronous only. Every flow takes `&mut` transport access, never shares
//!   it across threads, and never spawns threads beyond the single
//!   stdout-drain worker owned by [`supervise::SupervisedServer`] (the same
//!   shape as the credential-helper drain precedent).
//! - Policy decisions take caller-supplied `now_ms` (cooldown, consent,
//!   authorization). Live I/O deadlines use a monotonic clock from just
//!   before the wait; they are I/O bounds, not policy time.
//! - No wall clock, no async runtime, no network, no ambient environment:
//!   children spawn with a closed environment plus resolved credential refs
//!   only.
//! - MSRV 1.85: no post-1.85 standard APIs, no let-chains.
//! - English only. `#![deny(unsafe_code)]`.

#![deny(unsafe_code)]

pub mod bridge;
pub mod call;
pub mod config;
pub mod error;
pub mod frame;
pub mod handshake;
pub mod http_transport;
pub mod remote;
pub mod supervise;
pub mod tools;

pub(crate) mod json;

pub use bridge::{McpAdapterParams, McpToolAdapter};
pub use call::{McpCallError, McpCallOutcome, call_tool};
pub use config::{CredentialRef, McpServerConfig};
pub use error::{McpError, McpFailure, McpStage};
pub use frame::{FrameStats, MAX_FRAME_BYTES};
pub use handshake::{InitializeResult, handshake, initialize_request, initialized_notification};
pub use http_transport::{
    HTTP_ACCEPT, HTTP_CONTENT_TYPE, HttpLineTransport, MAX_SESSION_ID_LEN,
    MAX_SSE_FRAMES_PER_RESPONSE, SESSION_HEADER, TRANSPORT_ERROR_CODE,
};
pub use remote::{
    MAX_REMOTE_HEADER_NAME_LEN, MAX_REMOTE_HEADER_VALUE_LEN, MAX_REMOTE_HEADERS,
    MAX_REMOTE_HOST_LEN, MAX_REMOTE_URL_LEN, RemoteServerConfig,
};
pub use supervise::{McpHost, SupervisedServer};
pub use tools::{ImportedTool, MAX_TOOL_PAGES, list_tools, sanitize_mcp_name};

/// Capability placement label for this client (`mcp.invoke` host side).
pub const PLACEMENT: &str = "mcp";

/// Pinned MCP protocol version negotiated at `initialize`.
///
/// Strict pin: [`handshake::parse_initialize_result`] refuses any other
/// version fail-closed with [`McpFailure::VersionMismatch`].
pub const PROTOCOL_VERSION: &str = "2025-11-25";

/// Default whole-operation timeout in milliseconds (MCP ceiling parity with
/// the runtime `provider.rs` tool-stream default).
pub const DEFAULT_MCP_TIMEOUT_MS: u64 = 10_000;

/// Hard timeout ceiling in milliseconds (runtime `MP-8` parity: larger
/// values fail closed at [`McpServerConfig::validate`]).
pub const MAX_TIMEOUT_MS: u64 = 30_000;

/// Sync line transport over newline-delimited JSON-RPC.
///
/// Implementations move whole frames: [`McpTransport::send_line`] writes one
/// `\n`-terminated frame, [`McpTransport::recv_line`] returns the next
/// decoded frame, `Ok(None)` when `timeout_ms` elapses with no line, and
/// `Err` when the transport is broken or closed. Malformed lines are dropped
/// and counted inside the implementation, never returned.
pub trait McpTransport {
    /// Write one newline-terminated frame.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] when the transport is dead or the write fails.
    fn send_line(&mut self, line: &str) -> Result<(), McpError>;

    /// Return the next decoded frame, waiting at most `timeout_ms`.
    ///
    /// # Errors
    ///
    /// Returns `Ok(None)` on timeout with no line, [`McpError`] when the
    /// transport is broken or closed.
    fn recv_line(&mut self, timeout_ms: u64) -> Result<Option<String>, McpError>;
}

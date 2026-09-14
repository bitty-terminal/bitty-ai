//! Tool Bus boundary for the slice.
//!
//! Mirrors the accepted `TB-2`/`TB-3`/`TB-4`/`TB-6`/`TB-7` shape: a bounded
//! `ToolSpec`-like registry, validation before dispatch (unknown tool fails
//! closed), a read-only-by-default permission profile, a per-turn call cap, and
//! the headless MCP client stub as the adapter transport (`TB-1`). The slice
//! never executes a tool itself: `ToolHost` is the host/runtime seam, and the
//! in-repo implementation is a deterministic test peer.

use bitty_ipc::mcp::{McpClientConfig, McpClientStub, McpResponse};

use crate::error::SliceError;
use crate::provider::MAX_TOOL_ARGUMENTS_BYTES;

/// Maximum tool name length (`TB-2`).
pub const MAX_TOOL_NAME_LEN: usize = 64;
/// Maximum tool result bytes (`TB-6`).
pub const MAX_TOOL_RESULT_BYTES: usize = 16 * 1024;
/// Maximum registered tools per session (`TB-2`).
pub const MAX_TOOLS_PER_AGENT: usize = 32;
/// Maximum tool calls per assistant turn (`TB-6`).
pub const MAX_TOOL_CALLS_PER_TURN: usize = 8;

/// A bounded tool declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDecl {
    /// Registered tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Whether the tool is read-only; the default profile allows only these.
    pub read_only: bool,
}

impl ToolDecl {
    /// Construct a declaration.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, read_only: bool) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            read_only,
        }
    }
}

/// A model-requested tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolInvocation {
    /// Registered tool name.
    pub name: String,
    /// Opaque, bounded JSON arguments.
    pub arguments: String,
}

/// A bounded tool observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutcome {
    /// Tool that produced the result.
    pub name: String,
    /// Bounded result bytes.
    pub result: Vec<u8>,
    /// Tool results carry terminal/tool content, so they are untrusted.
    pub is_untrusted_surface: bool,
}

/// Host/runtime seam that executes a tool (`TB-7`: the agent layer never does).
pub trait ToolHost {
    /// Execute `tool` with bounded `arguments`.
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::ToolDenied`] when the host refuses the call.
    fn call_tool(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<Vec<u8>, SliceError>;
}

/// Tool Bus dispatch surface.
pub trait ToolBus {
    /// Validate and dispatch one tool invocation.
    ///
    /// # Errors
    ///
    /// Fails closed for unknown tools, denied tools, over-bound arguments, a
    /// per-turn call overflow, or an over-bound result.
    fn dispatch(
        &mut self,
        host: &mut dyn ToolHost,
        call: &ToolInvocation,
        now_ms: u64,
    ) -> Result<ToolOutcome, SliceError>;
}

/// `ToolBus` implementation over the generic MCP adapter stub.
pub struct HostToolBus {
    registry: Vec<ToolDecl>,
    mcp: McpClientStub,
    read_only_only: bool,
    calls_this_turn: usize,
}

impl HostToolBus {
    /// Construct a bus with `registry` and the read-only-by-default profile.
    ///
    /// # Errors
    ///
    /// Fails when the registry exceeds [`MAX_TOOLS_PER_AGENT`], a name is
    /// malformed, or the MCP client stub cannot be created.
    pub fn new(registry: Vec<ToolDecl>) -> Result<Self, SliceError> {
        if registry.len() > MAX_TOOLS_PER_AGENT {
            return Err(SliceError::ToolCallLimitExceeded {
                limit: MAX_TOOLS_PER_AGENT,
            });
        }
        for decl in &registry {
            validate_tool_name(&decl.name)?;
        }
        Ok(Self {
            registry,
            mcp: McpClientStub::new(McpClientConfig::default())?,
            read_only_only: true,
            calls_this_turn: 0,
        })
    }

    /// The default read-only registry used by the slice.
    #[must_use]
    pub fn read_only_registry() -> Vec<ToolDecl> {
        vec![ToolDecl::new(
            "terminal.read_zone",
            "Read a bounded terminal semantic zone (read-only)",
            true,
        )]
    }

    /// Reset the per-turn call counter (called at the start of each turn).
    pub fn begin_turn(&mut self) {
        self.calls_this_turn = 0;
    }

    /// Number of calls dispatched in the current turn.
    #[must_use]
    pub fn calls_this_turn(&self) -> usize {
        self.calls_this_turn
    }
}

impl ToolBus for HostToolBus {
    fn dispatch(
        &mut self,
        host: &mut dyn ToolHost,
        call: &ToolInvocation,
        now_ms: u64,
    ) -> Result<ToolOutcome, SliceError> {
        validate_tool_name(&call.name)?;
        let decl = self
            .registry
            .iter()
            .find(|decl| decl.name == call.name)
            .ok_or_else(|| SliceError::ToolNotRegistered {
                name: call.name.clone(),
            })?;
        if call.arguments.len() > MAX_TOOL_ARGUMENTS_BYTES {
            return Err(SliceError::ToolArgumentsTooLarge {
                limit: MAX_TOOL_ARGUMENTS_BYTES,
                actual: call.arguments.len(),
            });
        }
        if self.read_only_only && !decl.read_only {
            return Err(SliceError::ToolDenied {
                name: call.name.clone(),
            });
        }
        if self.calls_this_turn >= MAX_TOOL_CALLS_PER_TURN {
            return Err(SliceError::ToolCallLimitExceeded {
                limit: MAX_TOOL_CALLS_PER_TURN,
            });
        }

        let params = format!(
            r#"{{"name":"{}","arguments":{}}}"#,
            call.name, call.arguments
        )
        .into_bytes();
        let id = self
            .mcp
            .send_request("tools/call".to_owned(), params, now_ms)?;
        let result = host.call_tool(&call.name, call.arguments.as_bytes(), now_ms)?;
        if result.len() > MAX_TOOL_RESULT_BYTES {
            return Err(SliceError::ToolResultTooLarge {
                limit: MAX_TOOL_RESULT_BYTES,
                actual: result.len(),
            });
        }
        self.mcp
            .inject_response(McpResponse::success(id, result.clone())?)?;
        let responses = self.mcp.poll_responses();
        let correlated = responses.iter().any(|response| response.id == id);
        if !correlated {
            return Err(SliceError::ContextUnavailable {
                reason: "MCP response was not correlated with the tool request".to_owned(),
            });
        }
        self.calls_this_turn += 1;
        Ok(ToolOutcome {
            name: call.name.clone(),
            result,
            is_untrusted_surface: true,
        })
    }
}

fn validate_tool_name(name: &str) -> Result<(), SliceError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LEN
        && name.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.');
    if valid {
        Ok(())
    } else {
        Err(SliceError::ToolNotRegistered {
            name: name.to_owned(),
        })
    }
}

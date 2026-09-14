//! Integration-harness fixtures: the real `bitty-ai-runtime` driven through
//! the real `bitty-ipc` bridge.
//!
//! This module owns no AI mechanism. It adapts the generic host boundary
//! ([`IpcBridge`]) to runtime inputs: bounded terminal snapshots become
//! runtime [`ContextRecord`]s, and the scripted [`FakeProvider`] plus the
//! read-only tool registry give tests a deterministic single-agent loop.
//!
//! Vocabulary notes (all renames forced by the runtime's validated shapes,
//! not by preference):
//!
//! - Tool `terminal_read_zone`: the runtime `TB-2` name shape
//!   (`^[a-z][a-z0-9_]*$`) rejects the old slice `terminal.read_zone`.
//!   Mapping this to the generic `bitty-agent` vocabulary is later P1 work.
//! - Provider id `local-deterministic`: the runtime `MP-2` id shape rejects
//!   the old slice `local.deterministic`.
//! - Record owner `term-1`: a runtime [`StableId`] is one hierarchy level
//!   (`^[a-z0-9_-]+$`), so the old slice `inst-1/term-1` path is kept in the
//!   request params but not in the owner field.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentSession, AuthContext, AuthDecision, ContextError, ContextPriority,
    ContextRecord, FakeProvider, IdIssuer, ProviderError, ProviderTurn, RecordBody, StableId,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolRegistry, ToolSpec,
};

use crate::bridge::{HostPeer, IpcBridge};
use crate::error::SliceError;

/// Read-only terminal-zone tool served by the harness executor.
pub const HARNESS_TOOL: &str = "terminal_read_zone";

/// Scripted model name served by the runtime [`FakeProvider`].
pub const HARNESS_MODEL: &str = "fake-chat";

/// Harness provider identity (runtime `MP-2` shape: no dots).
pub const HARNESS_PROVIDER_ID: &str = "local-deterministic";

/// Bounded terminal-snapshot request: wire params for the generic
/// `terminal.snapshot` method plus the runtime [`StableId`] owner head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotRequest {
    /// Instance the terminal belongs to (wire params only).
    pub instance: String,
    /// Terminal to read (wire params only).
    pub terminal: String,
    /// Semantic zone name, e.g. `"output"` (wire params only).
    pub zone: String,
    /// Runtime [`StableId`] owner head, e.g. `"term-1"`.
    pub owner: String,
    /// Caller's byte ceiling for the snapshot.
    pub max_bytes: usize,
}

impl SnapshotRequest {
    /// Build a request for one bounded zone read.
    pub fn new(
        instance: impl Into<String>,
        terminal: impl Into<String>,
        zone: impl Into<String>,
        owner: impl Into<String>,
        max_bytes: usize,
    ) -> Self {
        Self {
            instance: instance.into(),
            terminal: terminal.into(),
            zone: zone.into(),
            owner: owner.into(),
            max_bytes,
        }
    }

    /// Serialize the bounded request params for `terminal.snapshot`.
    #[must_use]
    pub fn params_json(&self) -> Vec<u8> {
        format!(
            r#"{{"instance":"{}","terminal":"{}","zone":"{}","max_bytes":{}}}"#,
            self.instance, self.terminal, self.zone, self.max_bytes
        )
        .into_bytes()
    }
}

/// Build a runtime terminal [`ContextRecord`] from snapshot bytes.
///
/// The record is always an untrusted observation surface: terminal output is
/// attacker-controlled data, never instructions.
pub fn terminal_record(
    owner: &str,
    generation: u64,
    now_ms: u64,
    bytes: Vec<u8>,
) -> Result<ContextRecord, ContextError> {
    Ok(ContextRecord {
        id: format!("terminal-{generation}"),
        provider: "terminal".to_owned(),
        owner: StableId::new(owner)?,
        generation,
        collected_at_ms: now_ms,
        priority: ContextPriority::Normal,
        summary: "bounded terminal snapshot".to_owned(),
        body: RecordBody::Inline(bytes),
        supersedes: None,
        is_untrusted_surface: true,
    })
}

/// Collect one bounded terminal snapshot through the real [`IpcBridge`] and
/// adapt it to a runtime [`ContextRecord`].
///
/// # Errors
///
/// - [`SliceError`] from the bridge (unknown method, missing scope/consent,
///   host refusal) or when the snapshot exceeds `request.max_bytes`.
/// - [`SliceError::ContextUnavailable`] when the bytes cannot form a valid
///   runtime record (e.g. a malformed owner [`StableId`]).
pub fn collect_terminal_context(
    bridge: &mut IpcBridge,
    peer: &mut dyn HostPeer,
    request: &SnapshotRequest,
    generation: u64,
    now_ms: u64,
) -> Result<ContextRecord, SliceError> {
    let bytes = bridge.call("terminal.snapshot", &request.params_json(), now_ms, peer)?;
    if bytes.len() > request.max_bytes {
        return Err(SliceError::ContextBudgetExceeded {
            limit: request.max_bytes,
            actual: bytes.len(),
        });
    }
    terminal_record(&request.owner, generation, now_ms, bytes).map_err(|error| {
        SliceError::ContextUnavailable {
            reason: error.to_string(),
        }
    })
}

/// The harness read-only registry: exactly one tool, [`HARNESS_TOOL`].
///
/// # Errors
///
/// Returns [`ToolError`] when the registry or spec bounds reject the
/// declaration (unreachable for these constants; kept fallible so the
/// harness never panics on a bound).
pub fn test_tool_registry() -> Result<ToolRegistry, ToolError> {
    let mut registry = ToolRegistry::new();
    registry.register(ToolSpec::new(
        HARNESS_TOOL,
        "Read a bounded terminal semantic zone (read-only)",
        br#"{"type":"object"}"#.to_vec(),
        "terminal.inspect",
        true,
    )?)?;
    Ok(registry)
}

/// Authorizer that allows read-only tools and denies mutating ones.
///
/// This is a test-harness stand-in for the host capability/consent check,
/// not an accepted security mechanism: the real grant lives behind the
/// runtime [`ToolAuthorizer`] seam on the host side.
#[derive(Debug, Default)]
pub struct AllowReadOnly;

impl ToolAuthorizer for AllowReadOnly {
    fn authorize(&self, ctx: &AuthContext) -> AuthDecision {
        if ctx.read_only {
            AuthDecision::Allow
        } else {
            AuthDecision::Deny {
                reason: format!("harness denies mutating tool {}", ctx.tool),
            }
        }
    }
}

/// Build a scripted [`FakeProvider`] replaying one deterministic turn.
///
/// # Errors
///
/// Returns [`ProviderError::InvalidProviderId`] when the harness provider id
/// violates the `MP-2` shape (unreachable for the constant id).
pub fn scripted_provider(
    answer: &str,
    tool_arguments: Option<Vec<u8>>,
) -> Result<FakeProvider, ProviderError> {
    let mut provider = FakeProvider::new(HARNESS_PROVIDER_ID)?;
    let tool_calls = match tool_arguments {
        Some(arguments) => vec![ToolCallRequest {
            name: HARNESS_TOOL.to_owned(),
            arguments,
        }],
        None => Vec::new(),
    };
    provider.push_turn(ProviderTurn {
        text: answer.to_owned(),
        tool_calls,
        latency_ms: 0,
    });
    Ok(provider)
}

/// Fresh deterministic session (inspect tier, `Active`, generation 1).
#[must_use]
pub fn test_session() -> AgentSession {
    let mut ids = IdIssuer::default();
    AgentSession::new(ids.agent_instance(), ids.run(), ids.session())
}

/// Wire a deterministic single-agent runtime: `provider` plus the harness
/// read-only tool bus at `context_budget_bytes`.
///
/// # Errors
///
/// Returns [`ToolError`] when the harness registry violates a tool bound
/// (unreachable for these constants).
pub fn harness_agent(
    provider: FakeProvider,
    context_budget_bytes: usize,
) -> Result<Agent<FakeProvider>, ToolError> {
    let tools = ToolBus::new(test_tool_registry()?).with_authorizer(AllowReadOnly);
    let config = AgentConfig {
        context_budget_bytes,
        ..AgentConfig::default()
    };
    Ok(Agent::new(provider, tools, test_session(), config))
}

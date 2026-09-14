//! ContextProvider boundary for the slice.
//!
//! The slice reads bounded terminal context through the **generic**
//! `terminal.snapshot` wire method (scope `terminal.inspect`). It does not
//! invent a `panel.context` or `ai.context` host method: real terminal state is
//! requested through the same primitive any other read-only client would use,
//! and the result is labeled `is_untrusted_surface` per `CP-10` because terminal
//! output is attacker-controlled observation data.
//!
//! If the host does not implement `terminal.snapshot`, or the method is absent
//! from the generic registry, collection fails closed. That outcome is a
//! pressure-test finding, not a reason to add an AI-specific Core API.

use crate::bridge::{HostPeer, IpcBridge};
use crate::error::SliceError;

/// The accepted Context Budget (`CP-5`): 32 KiB combined per turn.
pub const CONTEXT_BUDGET_BYTES: usize = 32 * 1024;

/// Semantic-zone selector derived from OSC 7/133 boundaries (`CP-8`/`CP-9`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticZone {
    /// Shell prompt region.
    Prompt,
    /// User input region.
    Input,
    /// Command line region.
    Command,
    /// Command output region.
    Output,
}

impl SemanticZone {
    /// Wire-stable zone name used in the request payload.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Input => "input",
            Self::Command => "command",
            Self::Output => "output",
        }
    }
}

/// Explicit, non-ambient context selection (`CP-2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextRequest {
    /// Instance Stable Id the terminal belongs to.
    pub instance_id: String,
    /// Terminal Stable Id to read.
    pub terminal_id: String,
    /// Zone to read; never an implicit full-scrollback default (`CP-9`).
    pub zone: SemanticZone,
    /// Caller's byte ceiling for this record.
    pub max_bytes: usize,
}

impl ContextRequest {
    /// Serialize the bounded request params.
    #[must_use]
    pub fn params_json(&self) -> Vec<u8> {
        format!(
            r#"{{"instance":"{}","terminal":"{}","zone":"{}","max_bytes":{}}}"#,
            self.instance_id,
            self.terminal_id,
            self.zone.as_str(),
            self.max_bytes
        )
        .into_bytes()
    }
}

/// One bounded, attributed context record (`CP-3`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextRecord {
    /// Provider name (`terminal`).
    pub provider: &'static str,
    /// Stable Id path that contributed the bytes.
    pub owner: String,
    /// Generation the record was collected against.
    pub generation: u64,
    /// Deterministic collection time.
    pub collected_at_ms: u64,
    /// Terminal content is always an untrusted observation surface (`CP-10`).
    pub is_untrusted_surface: bool,
    /// Bounded snapshot bytes.
    pub bytes: Vec<u8>,
}

/// Context source the agent turn consumes.
pub trait ContextProvider {
    /// Collect one bounded record for `request`.
    ///
    /// # Errors
    ///
    /// Fails closed when the host does not implement the generic read method,
    /// consent is absent, or the returned bytes exceed the caller's ceiling.
    fn collect(
        &mut self,
        bridge: &mut IpcBridge,
        peer: &mut dyn HostPeer,
        request: &ContextRequest,
        generation: u64,
        now_ms: u64,
    ) -> Result<ContextRecord, SliceError>;
}

/// Reads bounded terminal zones through the generic `terminal.snapshot` method.
#[derive(Debug, Clone, Copy, Default)]
pub struct IpcTerminalContext;

impl ContextProvider for IpcTerminalContext {
    fn collect(
        &mut self,
        bridge: &mut IpcBridge,
        peer: &mut dyn HostPeer,
        request: &ContextRequest,
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
        Ok(ContextRecord {
            provider: "terminal",
            owner: format!("{}/{}", request.instance_id, request.terminal_id),
            generation,
            collected_at_ms: now_ms,
            is_untrusted_surface: true,
            bytes,
        })
    }
}

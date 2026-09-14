//! ModelProvider boundary for the slice.
//!
//! Mirrors the accepted `MP-1`/`MP-5`/`MP-7` shape at the smallest useful size:
//! a host-owned provider identity, a bounded `complete` turn, and a typed tool
//! request. The only implementation here is deterministic and local; it never
//! performs network I/O and holds no credential, matching `MP-3` (local-first
//! default) and the "no network in CI, no secrets" constraint.

use crate::error::SliceError;

/// Combined bound for `messages`, mirroring `MP-5` (`<= 32 KiB` combined).
pub const MAX_MESSAGE_BYTES: usize = 32 * 1024;

/// Bound for a single tool-call argument payload, mirroring `TB-3` (`16 KiB`).
pub const MAX_TOOL_ARGUMENTS_BYTES: usize = 16 * 1024;

/// Conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System instruction channel.
    System,
    /// User input channel.
    User,
    /// Assistant output channel.
    Assistant,
    /// Tool observation channel.
    Tool,
}

/// One bounded conversation message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Channel the content belongs to.
    pub role: Role,
    /// Bounded text content.
    pub content: String,
}

impl Message {
    /// Construct a message.
    #[must_use]
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Byte length of the content.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.content.len()
    }
}

/// A model-requested tool call, bounded by [`MAX_TOOL_ARGUMENTS_BYTES`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Registered tool name.
    pub name: String,
    /// Opaque, bounded JSON argument text.
    pub arguments: String,
}

/// One completed provider turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderTurn {
    /// Assistant text for the turn.
    pub text: String,
    /// Optional single tool request (the slice exercises at most one).
    pub tool_call: Option<ToolCall>,
}

/// Provider abstraction the agent runtime consumes.
pub trait ModelProvider {
    /// Stable provider identity (`owner.name` shape at full size).
    fn provider_id(&self) -> &str;

    /// Execute one synchronous turn over bounded messages.
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::ProviderBoundExceeded`] when the combined message
    /// bytes exceed [`MAX_MESSAGE_BYTES`] (`MP-5` fails at the boundary before
    /// provider I/O).
    fn complete(&self, messages: &[Message]) -> Result<ProviderTurn, SliceError>;
}

/// Deterministic local provider that replays a scripted turn.
///
/// There is deliberately no network, model file, or credential: the slice must
/// be reproducible in CI, and a provider that pretends to be live would violate
/// the "no fabricated live data" rule.
#[derive(Debug, Clone)]
pub struct DeterministicLocalProvider {
    provider_id: String,
    scripted: ProviderTurn,
}

impl DeterministicLocalProvider {
    /// Construct a provider replaying exactly `scripted`.
    #[must_use]
    pub fn new(provider_id: impl Into<String>, scripted: ProviderTurn) -> Self {
        Self {
            provider_id: provider_id.into(),
            scripted,
        }
    }

    /// The canonical slice provider: a local, read-only terminal explainer that
    /// asks for one bounded output-zone read.
    #[must_use]
    pub fn default_slice() -> Self {
        Self::new(
            "local.deterministic",
            ProviderTurn {
                text: "The last command printed `hello` with a zero exit status.".to_owned(),
                tool_call: Some(ToolCall {
                    name: "terminal.read_zone".to_owned(),
                    arguments: r#"{"zone":"output"}"#.to_owned(),
                }),
            },
        )
    }
}

impl ModelProvider for DeterministicLocalProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn complete(&self, messages: &[Message]) -> Result<ProviderTurn, SliceError> {
        let total: usize = messages.iter().map(Message::byte_len).sum();
        if total > MAX_MESSAGE_BYTES {
            return Err(SliceError::ProviderBoundExceeded {
                field: "messages",
                limit: MAX_MESSAGE_BYTES,
                actual: total,
            });
        }
        Ok(self.scripted.clone())
    }
}

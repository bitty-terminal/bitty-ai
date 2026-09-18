//! Capability-checked tool dispatch surface.
//!
//! Mirrors the `TB-2`..`TB-7` shape: a bounded [`ToolRegistry`], validation
//! before dispatch (unknown tools fail closed), per-tool authorization hooks
//! that deny by default, a per-turn call cap, and bounded arguments/results.
//! The runtime never executes a tool (`TB-7`): [`ToolExecutor`] is the
//! host/runtime seam, and [`FakeToolExecutor`] is the deterministic test peer
//! (the analogue of `ToolRegistry::stub_invoke`).
//!
//! Authorization hooks are fail-closed (`FS-AI7`): a missing authorizer, a
//! denied grant, or an `inspect`-tier call to a mutating tool all refuse with
//! no dispatch and no partial state. These hooks are **not** the consent
//! ledger or capability system; the host wires those behind
//! [`ToolAuthorizer`] and this crate claims no accepted security mechanism.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::context::MAX_SUMMARY_BYTES;
use crate::session::{AgentInstanceId, AgentLevel, AgentSession, SessionId};

/// Maximum tool name length in bytes (`TB-2`).
pub const MAX_TOOL_NAME_LEN: usize = 64;
/// Maximum tool description length in bytes (`TB-2`).
pub const MAX_TOOL_DESCRIPTION_LEN: usize = 512;
/// Maximum tool JSON Schema length in bytes (`TB-2`).
pub const MAX_TOOL_SCHEMA_BYTES: usize = 16 * 1024;
/// Maximum registered tools per session (`TB-2`).
pub const MAX_TOOLS_PER_SESSION: usize = 32;
/// Maximum tool calls per assistant turn (`TB-6`).
///
/// Hard bus ceiling: the effective per-turn cap is
/// `min(crate::agent::AgentConfig::max_tool_calls_per_turn,
/// MAX_TOOL_CALLS_PER_TURN)`. Raising the config above this constant does
/// not relax the bus: both [`ToolBus::precheck`] (batch gate) and
/// [`ToolBus::dispatch`] (per-call counter) still clamp at this constant,
/// so a config of 16 still fails past 8. Only tighter, never looser
/// (fail-closed).
pub const MAX_TOOL_CALLS_PER_TURN: usize = 8;
/// Maximum tool argument bytes (`TB-3`).
pub const MAX_TOOL_ARGUMENTS_BYTES: usize = 16 * 1024;
/// Maximum tool result bytes (`TB-6`).
pub const MAX_TOOL_RESULT_BYTES: usize = 16 * 1024;

/// FNV-1a-64 offset basis (deterministic across processes and platforms;
/// mirrors the `cache_key.rs` precedent).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (deterministic across processes and platforms; mirrors
/// the `cache_key.rs` precedent).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Tool Bus errors. Validation and authorization failures leave no partial
/// state: no queue entry, no dispatch, no counter increment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolError {
    /// Tool name violates the `TB-2` shape.
    InvalidName {
        /// Rejected name.
        name: String,
    },
    /// Tool is not registered (unknown tool fails closed, `TB-3`).
    UnknownTool {
        /// Requested name.
        name: String,
    },
    /// Tool name is already registered. Re-registration is refused
    /// fail-closed (`TB-2`): no replacement, no shadowing, no state change.
    DuplicateTool {
        /// Rejected name.
        name: String,
    },
    /// Arguments exceed [`MAX_TOOL_ARGUMENTS_BYTES`] (`TB-3`).
    ArgumentsTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Result exceeds [`MAX_TOOL_RESULT_BYTES`] (`TB-6`).
    ResultTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Result summary exceeds [`MAX_SUMMARY_BYTES`].
    SummaryTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Per-turn call cap exceeded (`TB-6`).
    CallLimitExceeded {
        /// Bound.
        limit: usize,
    },
    /// Registry exceeds [`MAX_TOOLS_PER_SESSION`] (`TB-2`).
    RegistryFull {
        /// Bound.
        limit: usize,
    },
    /// Description exceeds [`MAX_TOOL_DESCRIPTION_LEN`].
    DescriptionTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Schema exceeds [`MAX_TOOL_SCHEMA_BYTES`].
    SchemaTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Authorization hook refused the call (`TB-4`).
    Denied {
        /// Tool name.
        name: String,
        /// Hook reason.
        reason: String,
    },
    /// The effect may have happened but acknowledgement was lost (crash after
    /// execution, before ack). Returned by executors; the bus records it as
    /// [`ToolStatus::Unknown`] and the agent reconciles before retry.
    EffectUnknown {
        /// Tool name.
        name: String,
        /// What is uncertain.
        reason: String,
    },
}

impl Display for ToolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidName { name } => write!(f, "invalid tool name: {name}"),
            Self::UnknownTool { name } => write!(f, "unknown tool: {name}"),
            Self::DuplicateTool { name } => write!(f, "duplicate tool: {name}"),
            Self::ArgumentsTooLarge { limit, actual } => write!(
                f,
                "tool arguments of {actual} bytes exceed {limit} byte limit"
            ),
            Self::ResultTooLarge { limit, actual } => write!(
                f,
                "tool result of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::SummaryTooLarge { limit, actual } => write!(
                f,
                "tool summary of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::CallLimitExceeded { limit } => {
                write!(f, "tool call limit of {limit} per turn exceeded")
            }
            Self::RegistryFull { limit } => {
                write!(f, "tool registry full at {limit} tools")
            }
            Self::DescriptionTooLarge { limit, actual } => write!(
                f,
                "tool description of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::SchemaTooLarge { limit, actual } => write!(
                f,
                "tool schema of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::Denied { name, reason } => write!(f, "tool {name} denied: {reason}"),
            Self::EffectUnknown { name, reason } => {
                write!(f, "tool {name} effect unknown: {reason}")
            }
        }
    }
}

impl std::error::Error for ToolError {}

/// Validate a tool name (`TB-2`): non-empty, at most 64 bytes,
/// `^[a-z][a-z0-9_]*$` within the owner namespace.
///
/// # Errors
///
/// Returns [`ToolError::InvalidName`] when the shape is violated.
pub fn validate_tool_name(name: &str) -> Result<(), ToolError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LEN
        && name.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if valid {
        Ok(())
    } else {
        Err(ToolError::InvalidName {
            name: name.to_owned(),
        })
    }
}

/// A bounded tool declaration (`TB-2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    /// Registered tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for arguments (opaque bytes, bounded).
    pub schema_json: Vec<u8>,
    /// Required capability scope (for example `workspace.read`).
    pub required_scope: String,
    /// Whether the tool is side-effect free. The default profile allows only
    /// these at the `inspect` tier (read-only by default).
    pub read_only: bool,
}

impl ToolSpec {
    /// Construct and validate a declaration.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] for malformed names or over-bound description
    /// and schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema_json: Vec<u8>,
        required_scope: impl Into<String>,
        read_only: bool,
    ) -> Result<Self, ToolError> {
        let name = name.into();
        let description = description.into();
        validate_tool_name(&name)?;
        if description.len() > MAX_TOOL_DESCRIPTION_LEN {
            return Err(ToolError::DescriptionTooLarge {
                limit: MAX_TOOL_DESCRIPTION_LEN,
                actual: description.len(),
            });
        }
        if schema_json.len() > MAX_TOOL_SCHEMA_BYTES {
            return Err(ToolError::SchemaTooLarge {
                limit: MAX_TOOL_SCHEMA_BYTES,
                actual: schema_json.len(),
            });
        }
        Ok(Self {
            name,
            description,
            schema_json,
            required_scope: required_scope.into(),
            read_only,
        })
    }

    /// Deterministic identity of [`ToolSpec::schema_json`] (AI-0090, AIQ-08
    /// narrowing input): FNV-1a-64 over exactly the schema bytes (inline
    /// small hasher mirroring the `cache_key.rs` precedent; never
    /// `DefaultHasher`/`RandomState`). Same bytes give the same digest on
    /// every platform and process; any byte change moves it. The digest is
    /// the host-side invalidation handle: the registry keeps refusing
    /// same-name re-registration ([`ToolError::DuplicateTool`]), so the
    /// host compares digests across its schema snapshots to detect drift
    /// and never relies on the digest to smuggle a replacement.
    #[must_use]
    pub fn schema_digest(&self) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for byte in &self.schema_json {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

/// Bounded per-session tool registry (`TB-2`).
#[derive(Debug, Default)]
pub struct ToolRegistry {
    specs: Vec<ToolSpec>,
}

impl ToolRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one validated spec.
    ///
    /// Fail-closed: a name that is already registered is refused with
    /// [`ToolError::DuplicateTool`] (no replacement, no shadowing); a
    /// malformed or over-bound spec is refused with its typed error even
    /// when built without [`ToolSpec::new`]; a full registry refuses with
    /// [`ToolError::RegistryFull`]. Every refusal leaves the registry
    /// unchanged: no push, no replacement, no counter increment.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::DuplicateTool`] for a re-registered name,
    /// [`ToolError::InvalidName`], [`ToolError::DescriptionTooLarge`], or
    /// [`ToolError::SchemaTooLarge`] for a malformed spec, and
    /// [`ToolError::RegistryFull`] past [`MAX_TOOLS_PER_SESSION`].
    pub fn register(&mut self, spec: ToolSpec) -> Result<(), ToolError> {
        if self.specs.iter().any(|kept| kept.name == spec.name) {
            return Err(ToolError::DuplicateTool { name: spec.name });
        }
        validate_tool_name(&spec.name)?;
        if spec.description.len() > MAX_TOOL_DESCRIPTION_LEN {
            return Err(ToolError::DescriptionTooLarge {
                limit: MAX_TOOL_DESCRIPTION_LEN,
                actual: spec.description.len(),
            });
        }
        if spec.schema_json.len() > MAX_TOOL_SCHEMA_BYTES {
            return Err(ToolError::SchemaTooLarge {
                limit: MAX_TOOL_SCHEMA_BYTES,
                actual: spec.schema_json.len(),
            });
        }
        if self.specs.len() >= MAX_TOOLS_PER_SESSION {
            return Err(ToolError::RegistryFull {
                limit: MAX_TOOLS_PER_SESSION,
            });
        }
        self.specs.push(spec);
        Ok(())
    }

    /// Registered spec count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether the registry holds no spec.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Look up a spec by name.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|spec| spec.name == name)
    }

    /// Registered tool names in registration order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }
}

/// Caller identity presented to the authorization hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthBase {
    /// Calling runtime-local agent instance.
    pub agent_instance_id: AgentInstanceId,
    /// Calling session.
    pub session_id: SessionId,
    /// Server-side authority tier of the session.
    pub level: AgentLevel,
}

/// Per-call context presented to the authorization hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthContext<'a> {
    /// Caller identity and tier.
    pub base: AuthBase,
    /// Requested tool name.
    pub tool: &'a str,
    /// Capability scope the tool declaration requires.
    pub required_scope: &'a str,
    /// Whether the tool declares itself side-effect free.
    pub read_only: bool,
}

/// Authorization verdict. There is no allow-all grant in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthDecision {
    /// Proceed to host execution.
    Allow,
    /// Refuse with no dispatch and no partial state.
    Deny {
        /// Hook reason (attributed in enforcement records).
        reason: String,
    },
}

/// Per-tool authorization hook (`TB-4` seam). The host implements capability
/// plus consent checks behind it; the skeleton default denies.
pub trait ToolAuthorizer {
    /// Decide one tool call. Called at validation time and again at each
    /// dispatch boundary, so revocation takes effect for the next dispatch
    /// (`PP-6`).
    fn authorize(&self, ctx: &AuthContext) -> AuthDecision;
}

/// Authorizer that denies every call (default posture).
#[derive(Debug, Default)]
pub struct DenyAllAuthorizer;

impl ToolAuthorizer for DenyAllAuthorizer {
    fn authorize(&self, ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Deny {
            reason: format!("default-deny: no grant for {}", ctx.tool),
        }
    }
}

/// One validated tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Registered tool name.
    pub name: String,
    /// Opaque bounded JSON arguments.
    pub arguments: Vec<u8>,
}

/// Structured execution status, including `Unknown` for effects that may
/// have happened without acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    /// Host executed and acknowledged.
    Success,
    /// Host executed and reported failure.
    Failed {
        /// Host reason.
        reason: String,
    },
    /// Refused before host execution (no effect).
    Denied {
        /// Hook or host reason.
        reason: String,
    },
    /// Effect uncertain: reconcile (status inspection or user direction)
    /// before retry; never blindly retry (`MP-7`).
    Unknown {
        /// What is uncertain.
        reason: String,
    },
}

/// One recorded tool execution (L0 structured result shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecution {
    /// Attribution handle for this dispatch.
    pub execution_id: crate::session::ExecutionId,
    /// Tool that ran (or was refused).
    pub tool: String,
    /// Structured status.
    pub status: ToolStatus,
    /// Bounded L0 summary (always inline).
    pub summary: String,
    /// Bounded result bytes (empty for denials).
    pub data: Vec<u8>,
    /// Tool results carry tool content and stay untrusted observations.
    pub is_untrusted_surface: bool,
}

/// Successful host execution returned to the bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSuccess {
    /// Bounded L0 summary.
    pub summary: String,
    /// Bounded result bytes.
    pub data: Vec<u8>,
}

impl ToolSuccess {
    /// Construct and validate a success payload.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::SummaryTooLarge`] or
    /// [`ToolError::ResultTooLarge`] for over-bound payloads.
    pub fn new(summary: String, data: Vec<u8>) -> Result<Self, ToolError> {
        if summary.len() > MAX_SUMMARY_BYTES {
            return Err(ToolError::SummaryTooLarge {
                limit: MAX_SUMMARY_BYTES,
                actual: summary.len(),
            });
        }
        if data.len() > MAX_TOOL_RESULT_BYTES {
            return Err(ToolError::ResultTooLarge {
                limit: MAX_TOOL_RESULT_BYTES,
                actual: data.len(),
            });
        }
        Ok(Self { summary, data })
    }
}

/// Context for a single tool dispatch attempt presented to [`ToolExecutor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionContext {
    /// Attribution handle for this dispatch issued by the runtime.
    pub execution_id: crate::session::ExecutionId,
    /// Caller-supplied dispatch timestamp in milliseconds.
    pub now_ms: u64,
}

/// Host/runtime seam that executes a tool (`TB-7`: the agent layer never
/// does). Implementations must be deterministic for a given script plus
/// `now_ms` when used under test.
pub trait ToolExecutor {
    /// Execute `tool` with bounded `arguments`.
    ///
    /// # Errors
    ///
    /// Return [`ToolError::Denied`] when the host refuses, or
    /// [`ToolError::EffectUnknown`] when the effect may have happened but
    /// acknowledgement was lost. Other errors propagate as bus failures.
    fn execute(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<ToolSuccess, ToolError>;

    /// Execute `tool` with bounded `arguments` and typed dispatch
    /// [`ExecutionContext`].
    ///
    /// By default delegates to [`ToolExecutor::execute`] using `context.now_ms`
    /// so existing adapters compile without breakage. Adapters requiring dispatch
    /// identity propagation (such as correlation with host-side reconciliation ledgers)
    /// should override this method.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`ToolExecutor::execute`].
    fn execute_with_context(
        &mut self,
        tool: &str,
        arguments: &[u8],
        context: &ExecutionContext,
    ) -> Result<ToolSuccess, ToolError> {
        self.execute(tool, arguments, context.now_ms)
    }
}

/// Deterministic test peer for [`ToolExecutor`]. Replays scripted outcomes
/// FIFO (defaulting to an empty success) and records every invocation. An
/// optional cancel hook cancels a session handle on the n-th call, letting
/// tests exercise cancel-after-dispatch deterministically.
#[derive(Debug, Default)]
pub struct FakeToolExecutor {
    script: VecDeque<Result<ToolSuccess, ToolError>>,
    calls: Vec<(String, Vec<u8>)>,
    cancel_hook: Option<(AgentSession, usize)>,
}

impl FakeToolExecutor {
    /// Construct an empty executor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a successful outcome.
    pub fn push_success(&mut self, summary: impl Into<String>, data: Vec<u8>) {
        self.script.push_back(Ok(ToolSuccess {
            summary: summary.into(),
            data,
        }));
    }

    /// Queue a failing outcome.
    pub fn push_error(&mut self, error: ToolError) {
        self.script.push_back(Err(error));
    }

    /// Cancel `session` when the `call_index`-th (1-based) execution starts.
    #[must_use]
    pub fn with_cancel_on_call(mut self, session: AgentSession, call_index: usize) -> Self {
        self.cancel_hook = Some((session, call_index));
        self
    }

    /// Invocations so far as `(tool, arguments)` pairs.
    #[must_use]
    pub fn calls(&self) -> &[(String, Vec<u8>)] {
        &self.calls
    }
}

impl ToolExecutor for FakeToolExecutor {
    fn execute(
        &mut self,
        tool: &str,
        arguments: &[u8],
        _now_ms: u64,
    ) -> Result<ToolSuccess, ToolError> {
        self.calls.push((tool.to_owned(), arguments.to_vec()));
        if let Some((session, index)) = &self.cancel_hook {
            if self.calls.len() == *index {
                session.cancel();
            }
        }
        self.script.pop_front().unwrap_or(Ok(ToolSuccess {
            summary: String::new(),
            data: Vec::new(),
        }))
    }
}

/// Deterministic recording test peer for [`ToolExecutor`].
///
/// Records every invocation as `(execution_id, tool, arguments)` triples,
/// preserving exact [`crate::session::ExecutionId`] attribution handles across
/// dispatches. Replays scripted outcomes FIFO (defaulting to empty success).
#[derive(Debug, Default)]
pub struct RecordingExecutor {
    script: VecDeque<Result<ToolSuccess, ToolError>>,
    calls: Vec<(crate::session::ExecutionId, String, Vec<u8>)>,
}

impl RecordingExecutor {
    /// Construct an empty recording executor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a successful outcome.
    pub fn push_success(&mut self, summary: impl Into<String>, data: Vec<u8>) {
        self.script.push_back(Ok(ToolSuccess {
            summary: summary.into(),
            data,
        }));
    }

    /// Queue a failing outcome.
    pub fn push_error(&mut self, error: ToolError) {
        self.script.push_back(Err(error));
    }

    /// Invocations so far as `(execution_id, tool, arguments)` triples.
    #[must_use]
    pub fn calls(&self) -> &[(crate::session::ExecutionId, String, Vec<u8>)] {
        &self.calls
    }

    /// Number of invocations recorded so far.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Number of invocations recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Whether no invocations have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

impl ToolExecutor for RecordingExecutor {
    fn execute(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<ToolSuccess, ToolError> {
        self.execute_with_context(
            tool,
            arguments,
            &ExecutionContext {
                execution_id: crate::session::ExecutionId(0),
                now_ms,
            },
        )
    }

    fn execute_with_context(
        &mut self,
        tool: &str,
        arguments: &[u8],
        context: &ExecutionContext,
    ) -> Result<ToolSuccess, ToolError> {
        self.calls
            .push((context.execution_id, tool.to_owned(), arguments.to_vec()));
        self.script.pop_front().unwrap_or(Ok(ToolSuccess {
            summary: String::new(),
            data: Vec::new(),
        }))
    }
}

/// Tool Bus dispatch surface.
#[derive(Default)]
pub struct ToolBus {
    registry: ToolRegistry,
    authorizer: Option<Box<dyn ToolAuthorizer>>,
    calls_this_turn: usize,
}

impl std::fmt::Debug for ToolBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolBus")
            .field("registry_len", &self.registry.len())
            .field("has_authorizer", &self.authorizer.is_some())
            .field("calls_this_turn", &self.calls_this_turn)
            .finish()
    }
}

impl ToolBus {
    /// Construct a bus over `registry` with no authorizer: every call is
    /// denied until the host installs one (fail closed, `FS-AI7`).
    #[must_use]
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            registry,
            authorizer: None,
            calls_this_turn: 0,
        }
    }

    /// Install the host authorization hook.
    #[must_use]
    pub fn with_authorizer(mut self, authorizer: impl ToolAuthorizer + 'static) -> Self {
        self.authorizer = Some(Box::new(authorizer));
        self
    }

    /// Reset the per-turn call counter at the start of each assistant turn.
    pub fn begin_turn(&mut self) {
        self.calls_this_turn = 0;
    }

    /// Calls dispatched in the current turn.
    #[must_use]
    pub fn calls_this_turn(&self) -> usize {
        self.calls_this_turn
    }

    /// Registered tool names in registration order.
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.registry.names()
    }

    /// Authorize one call against the tier floor and the hook. The `inspect`
    /// tier may only reach read-only tools (read-only by default); anything
    /// else, a missing hook, or a hook denial refuses.
    fn authorize_call(&self, base: &AuthBase, call: &ToolCall) -> Result<(), ToolError> {
        validate_tool_name(&call.name)?;
        let spec = self
            .registry
            .lookup(&call.name)
            .ok_or_else(|| ToolError::UnknownTool {
                name: call.name.clone(),
            })?;
        if call.arguments.len() > MAX_TOOL_ARGUMENTS_BYTES {
            return Err(ToolError::ArgumentsTooLarge {
                limit: MAX_TOOL_ARGUMENTS_BYTES,
                actual: call.arguments.len(),
            });
        }
        if base.level == AgentLevel::Inspect && !spec.read_only {
            return Err(ToolError::Denied {
                name: call.name.clone(),
                reason: "inspect tier is read-only; elevation required".to_owned(),
            });
        }
        let Some(authorizer) = &self.authorizer else {
            return Err(ToolError::Denied {
                name: call.name.clone(),
                reason: "no authorizer installed (fail closed)".to_owned(),
            });
        };
        match authorizer.authorize(&AuthContext {
            base: *base,
            tool: &call.name,
            required_scope: &spec.required_scope,
            read_only: spec.read_only,
        }) {
            AuthDecision::Allow => Ok(()),
            AuthDecision::Deny { reason } => Err(ToolError::Denied {
                name: call.name.clone(),
                reason,
            }),
        }
    }

    /// Validate and authorize every call without dispatching any
    /// (`FS-AI1` transactional denial: a refused turn dispatches nothing).
    ///
    /// Enforces the constant side of the effective cap
    /// `min(crate::agent::AgentConfig::max_tool_calls_per_turn,
    /// MAX_TOOL_CALLS_PER_TURN)`: even when the host raises the config, a
    /// batch larger than [`MAX_TOOL_CALLS_PER_TURN`] still fails here with
    /// [`ToolError::CallLimitExceeded`].
    ///
    /// # Errors
    ///
    /// Returns the first [`ToolError`] encountered; no call is dispatched.
    pub fn precheck(&self, calls: &[ToolCall], base: &AuthBase) -> Result<(), ToolError> {
        for call in calls {
            self.authorize_call(base, call)?;
        }
        if calls.len() > MAX_TOOL_CALLS_PER_TURN {
            return Err(ToolError::CallLimitExceeded {
                limit: MAX_TOOL_CALLS_PER_TURN,
            });
        }
        Ok(())
    }

    /// Validate, re-authorize at this dispatch boundary (`PP-6`), and
    /// dispatch one call through `executor`.
    ///
    /// Enforces the constant side of the effective cap
    /// `min(crate::agent::AgentConfig::max_tool_calls_per_turn,
    /// MAX_TOOL_CALLS_PER_TURN)`: the per-turn counter still stops at
    /// [`MAX_TOOL_CALLS_PER_TURN`] even when the host raises the config,
    /// failing with [`ToolError::CallLimitExceeded`].
    ///
    /// # Errors
    ///
    /// Fails closed with [`ToolError`] for validation, authorization, cap, or
    /// over-bound results. [`ToolError::EffectUnknown`] and
    /// [`ToolError::Denied`] from the executor become recorded
    /// [`ToolExecution`] statuses instead of errors.
    pub fn dispatch(
        &mut self,
        executor: &mut dyn ToolExecutor,
        call: &ToolCall,
        base: &AuthBase,
        execution_id: crate::session::ExecutionId,
        now_ms: u64,
    ) -> Result<ToolExecution, ToolError> {
        self.authorize_call(base, call)?;
        if self.calls_this_turn >= MAX_TOOL_CALLS_PER_TURN {
            return Err(ToolError::CallLimitExceeded {
                limit: MAX_TOOL_CALLS_PER_TURN,
            });
        }
        let context = ExecutionContext {
            execution_id,
            now_ms,
        };
        let outcome = executor.execute_with_context(&call.name, &call.arguments, &context);
        self.calls_this_turn += 1;
        match outcome {
            Ok(success) => {
                if success.summary.len() > MAX_SUMMARY_BYTES {
                    return Err(ToolError::SummaryTooLarge {
                        limit: MAX_SUMMARY_BYTES,
                        actual: success.summary.len(),
                    });
                }
                if success.data.len() > MAX_TOOL_RESULT_BYTES {
                    return Err(ToolError::ResultTooLarge {
                        limit: MAX_TOOL_RESULT_BYTES,
                        actual: success.data.len(),
                    });
                }
                Ok(ToolExecution {
                    execution_id,
                    tool: call.name.clone(),
                    status: ToolStatus::Success,
                    summary: success.summary,
                    data: success.data,
                    is_untrusted_surface: true,
                })
            }
            Err(ToolError::EffectUnknown { reason, .. }) => Ok(ToolExecution {
                execution_id,
                tool: call.name.clone(),
                // AI-0077: host-shaped uncertainty reasons are bounded at the
                // bus boundary (`MAX_REASON_BYTES`, scrubbed) so the recorded
                // `Unknown` status never carries unbounded or newline-bearing
                // text into reconcile reports or provider messages.
                status: ToolStatus::Unknown {
                    reason: crate::bridge::bound_reason(&reason),
                },
                summary: "effect uncertain; reconcile before retry".to_owned(),
                data: Vec::new(),
                is_untrusted_surface: true,
            }),
            Err(ToolError::Denied { reason, .. }) => Ok(ToolExecution {
                execution_id,
                tool: call.name.clone(),
                status: ToolStatus::Denied { reason },
                summary: "host denied execution".to_owned(),
                data: Vec::new(),
                is_untrusted_surface: true,
            }),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_only_registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolSpec::new(
                    "workspace_read",
                    "Read a bounded workspace path",
                    br#"{"type":"object"}"#.to_vec(),
                    "workspace.read",
                    true,
                )
                .expect("valid spec"),
            )
            .expect("capacity");
        registry
            .register(
                ToolSpec::new(
                    "workspace_write",
                    "Write a bounded workspace path",
                    br#"{"type":"object"}"#.to_vec(),
                    "workspace.write",
                    false,
                )
                .expect("valid spec"),
            )
            .expect("capacity");
        registry
    }

    fn base() -> AuthBase {
        let mut ids = crate::session::IdIssuer::default();
        AuthBase {
            agent_instance_id: ids.agent_instance(),
            session_id: ids.session(),
            level: AgentLevel::Inspect,
        }
    }

    #[test]
    fn tool_name_shape() {
        assert!(validate_tool_name("workspace_read").is_ok());
        assert!(validate_tool_name("").is_err());
        assert!(validate_tool_name("Read").is_err());
        assert!(validate_tool_name("terminal.read").is_err());
        assert!(validate_tool_name("a".repeat(65).as_str()).is_err());
    }

    #[test]
    fn registry_bound_is_fail_closed() {
        let mut registry = ToolRegistry::new();
        for index in 0..MAX_TOOLS_PER_SESSION {
            registry
                .register(
                    ToolSpec::new(
                        format!("tool_{index}"),
                        "test tool",
                        Vec::new(),
                        "test.scope",
                        true,
                    )
                    .expect("valid spec"),
                )
                .expect("capacity");
        }
        let overflow =
            ToolSpec::new("one_more", "test", Vec::new(), "test.scope", true).expect("valid spec");
        assert!(matches!(
            registry.register(overflow),
            Err(ToolError::RegistryFull { .. })
        ));
    }

    #[test]
    fn duplicate_registration_is_fail_closed() {
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolSpec::new(
                    "workspace_read",
                    "Read a bounded workspace path",
                    br#"{"type":"object"}"#.to_vec(),
                    "workspace.read",
                    true,
                )
                .expect("valid spec"),
            )
            .expect("capacity");
        let retry = ToolSpec::new(
            "workspace_read",
            "Shadow description with wider scope",
            br#"{"type":"object"}"#.to_vec(),
            "workspace.write",
            false,
        )
        .expect("valid spec");
        let error = registry.register(retry).expect_err("duplicate must fail");
        assert_eq!(
            error,
            ToolError::DuplicateTool {
                name: "workspace_read".to_owned(),
            }
        );
        // No partial state: no push, no replacement, lookup keeps the first.
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.names(), vec!["workspace_read".to_owned()]);
        let kept = registry.lookup("workspace_read").expect("original kept");
        assert_eq!(kept.description, "Read a bounded workspace path");
        assert_eq!(kept.required_scope, "workspace.read");
        assert!(kept.read_only);
    }

    #[test]
    fn duplicate_reports_even_when_registry_is_full() {
        let mut registry = ToolRegistry::new();
        for index in 0..MAX_TOOLS_PER_SESSION {
            registry
                .register(
                    ToolSpec::new(
                        format!("tool_{index}"),
                        "test tool",
                        Vec::new(),
                        "test.scope",
                        true,
                    )
                    .expect("valid spec"),
                )
                .expect("capacity");
        }
        let retry = ToolSpec::new("tool_0", "test tool", Vec::new(), "test.scope", true)
            .expect("valid spec");
        let error = registry.register(retry).expect_err("duplicate must fail");
        assert_eq!(
            error,
            ToolError::DuplicateTool {
                name: "tool_0".to_owned(),
            }
        );
        assert_eq!(registry.len(), MAX_TOOLS_PER_SESSION);
    }

    #[test]
    fn register_revalidates_specs_built_without_new() {
        let mut registry = ToolRegistry::new();
        // Legacy dotted vocabulary (`terminal.read_zone`) never enters the
        // registry, even when the spec bypasses `ToolSpec::new`.
        let dotted = ToolSpec {
            name: "terminal.read_zone".to_owned(),
            description: "legacy dotted name".to_owned(),
            schema_json: Vec::new(),
            required_scope: "terminal.inspect".to_owned(),
            read_only: true,
        };
        assert_eq!(
            registry
                .register(dotted)
                .expect_err("dotted name must fail"),
            ToolError::InvalidName {
                name: "terminal.read_zone".to_owned(),
            }
        );
        let oversized = ToolSpec {
            name: "workspace_read".to_owned(),
            description: "x".repeat(MAX_TOOL_DESCRIPTION_LEN + 1),
            schema_json: Vec::new(),
            required_scope: "workspace.read".to_owned(),
            read_only: true,
        };
        assert!(matches!(
            registry.register(oversized),
            Err(ToolError::DescriptionTooLarge { .. })
        ));
        assert!(registry.is_empty());
    }

    #[test]
    fn legacy_dotted_tool_call_fails_as_invalid_name() {
        struct Allow;
        impl ToolAuthorizer for Allow {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                AuthDecision::Allow
            }
        }
        let bus = ToolBus::new(read_only_registry()).with_authorizer(Allow);
        let call = ToolCall {
            name: "terminal.read_zone".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        assert_eq!(
            bus.precheck(std::slice::from_ref(&call), &base())
                .expect_err("dotted name must fail"),
            ToolError::InvalidName {
                name: "terminal.read_zone".to_owned(),
            }
        );
        assert_eq!(bus.calls_this_turn(), 0);
    }

    #[test]
    fn precheck_over_limit_leaves_no_partial_state() {
        struct Allow;
        impl ToolAuthorizer for Allow {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                AuthDecision::Allow
            }
        }
        let bus = ToolBus::new(read_only_registry()).with_authorizer(Allow);
        let call = ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        let burst = vec![call; MAX_TOOL_CALLS_PER_TURN + 1];
        assert_eq!(
            bus.precheck(&burst, &base()).expect_err("burst must fail"),
            ToolError::CallLimitExceeded {
                limit: MAX_TOOL_CALLS_PER_TURN,
            }
        );
        assert_eq!(bus.calls_this_turn(), 0);
    }

    #[test]
    fn missing_authorizer_denies() {
        let bus = ToolBus::new(read_only_registry());
        let call = ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        assert!(matches!(
            bus.precheck(std::slice::from_ref(&call), &base()),
            Err(ToolError::Denied { .. })
        ));
    }

    #[test]
    fn inspect_tier_cannot_reach_mutating_tools() {
        struct Allow;
        impl ToolAuthorizer for Allow {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                AuthDecision::Allow
            }
        }
        let bus = ToolBus::new(read_only_registry()).with_authorizer(Allow);
        let call = ToolCall {
            name: "workspace_write".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        assert!(matches!(
            bus.precheck(std::slice::from_ref(&call), &base()),
            Err(ToolError::Denied { .. })
        ));
    }

    #[test]
    fn dispatch_records_executor_denial_as_status_without_data() {
        struct Allow;
        impl ToolAuthorizer for Allow {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                AuthDecision::Allow
            }
        }
        let mut bus = ToolBus::new(read_only_registry()).with_authorizer(Allow);
        let mut executor = FakeToolExecutor::new();
        executor.push_error(ToolError::Denied {
            name: "workspace_read".to_owned(),
            reason: "host policy refused".to_owned(),
        });
        let call = ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        let mut ids = crate::session::IdIssuer::default();
        let execution = bus
            .dispatch(&mut executor, &call, &base(), ids.execution(), 1_000)
            .expect("host denial is a recorded status, not a bus error");
        // A host-side refusal is attributed as `Denied` with no payload and
        // the dispatch is still counted against the per-turn cap.
        assert!(matches!(execution.status, ToolStatus::Denied { .. }));
        assert!(execution.data.is_empty());
        assert!(execution.is_untrusted_surface);
        assert_eq!(execution.tool, "workspace_read");
        assert_eq!(bus.calls_this_turn(), 1);
    }

    #[test]
    fn dispatch_rechecks_the_hook_at_the_boundary_after_precheck() {
        use std::cell::Cell;

        // Revocation between the transactional gate and the next dispatch
        // boundary must take effect: the hook allows the validation pass and
        // denies every later check (`PP-6`). The call is refused before the
        // executor with no partial state.
        struct RevokeAfterPrecheck {
            checks: Cell<usize>,
        }
        impl ToolAuthorizer for RevokeAfterPrecheck {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                let seen = self.checks.get();
                self.checks.set(seen + 1);
                if seen == 0 {
                    AuthDecision::Allow
                } else {
                    AuthDecision::Deny {
                        reason: "grant revoked".to_owned(),
                    }
                }
            }
        }
        let mut bus = ToolBus::new(read_only_registry()).with_authorizer(RevokeAfterPrecheck {
            checks: Cell::new(0),
        });
        let call = ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        bus.precheck(std::slice::from_ref(&call), &base())
            .expect("validation pass allows");
        let mut executor = FakeToolExecutor::new();
        let mut ids = crate::session::IdIssuer::default();
        let error = bus
            .dispatch(&mut executor, &call, &base(), ids.execution(), 1_000)
            .expect_err("dispatch boundary must re-check the hook");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert!(executor.calls().is_empty(), "revoked call never dispatched");
        assert_eq!(bus.calls_this_turn(), 0);
    }

    #[test]
    fn dispatch_propagates_execution_context_to_recording_executor() {
        struct Allow;
        impl ToolAuthorizer for Allow {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                AuthDecision::Allow
            }
        }
        let mut bus = ToolBus::new(read_only_registry()).with_authorizer(Allow);
        let mut executor = RecordingExecutor::new();
        executor.push_success("ok 1", b"res 1".to_vec());
        executor.push_success("ok 2", b"res 2".to_vec());

        let call1 = ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"file1"}"#.to_vec(),
        };
        let call2 = ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"file2"}"#.to_vec(),
        };

        let mut ids = crate::session::IdIssuer::default();
        let exec_id_1 = ids.execution();
        let exec_id_2 = ids.execution();

        let execution1 = bus
            .dispatch(&mut executor, &call1, &base(), exec_id_1, 1_000)
            .expect("dispatch call 1");
        assert_eq!(execution1.execution_id, exec_id_1);
        assert_eq!(execution1.status, ToolStatus::Success);

        let execution2 = bus
            .dispatch(&mut executor, &call2, &base(), exec_id_2, 2_000)
            .expect("dispatch call 2");
        assert_eq!(execution2.execution_id, exec_id_2);
        assert_eq!(execution2.status, ToolStatus::Success);

        assert_eq!(executor.call_count(), 2);
        assert_eq!(executor.calls()[0].0, exec_id_1);
        assert_eq!(executor.calls()[0].1, "workspace_read");
        assert_eq!(executor.calls()[0].2, br#"{"path":"file1"}"#);

        assert_eq!(executor.calls()[1].0, exec_id_2);
        assert_eq!(executor.calls()[1].1, "workspace_read");
        assert_eq!(executor.calls()[1].2, br#"{"path":"file2"}"#);
    }

    #[test]
    fn default_execute_with_context_delegates_to_execute() {
        struct LegacyExecutor {
            recorded_now_ms: Option<u64>,
        }
        impl ToolExecutor for LegacyExecutor {
            fn execute(
                &mut self,
                _tool: &str,
                _arguments: &[u8],
                now_ms: u64,
            ) -> Result<ToolSuccess, ToolError> {
                self.recorded_now_ms = Some(now_ms);
                Ok(ToolSuccess::new("ok".to_owned(), Vec::new()).expect("valid"))
            }
        }

        let mut legacy = LegacyExecutor {
            recorded_now_ms: None,
        };
        let ctx = ExecutionContext {
            execution_id: crate::session::ExecutionId(42),
            now_ms: 12345,
        };
        let res = legacy.execute_with_context("tool", b"arg", &ctx);
        assert!(res.is_ok());
        assert_eq!(legacy.recorded_now_ms, Some(12345));
    }
}

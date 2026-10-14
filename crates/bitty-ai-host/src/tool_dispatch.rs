//! Phase 2 S1 pure-type and gate mirror of the tool-dispatch seam.
//!
//! Offline, `std`-only, no behavior: this module mirrors the registry,
//! placement, authorization, consent, narrowing, prompt-gate, and usage-ring
//! vocabulary owned by `bitty-ai-runtime` (`tool.rs` plus the consent seam in
//! `bridge.rs`) so the host can re-impose every gate in order when dispatch
//! moves host-side. It grants nothing: the default authorizer and the default
//! ledger both deny, an empty registry admits nothing, and there is no
//! executor, no ledger implementation, no IPC, no routing, no cost fuse, no
//! server-binary digest or SPKI enforcement, no MCP transport, no threads, no
//! network, and no filesystem access.
//!
//! Mirrored sources (read-only, values and wording kept byte-identical):
//!
//! - `tool.rs` gate doc plus bounds: `MAX_TOOL_NAME_LEN` 64,
//!   `MAX_TOOL_DESCRIPTION_LEN` 512, `MAX_TOOL_SCHEMA_BYTES` 16 KiB,
//!   `MAX_TOOLS_PER_SESSION` 32, `MAX_TOOL_CALLS_PER_TURN` 8 (hard bus
//!   ceiling `min(configured, 8)` cumulative across the turn's provider
//!   rounds), `MAX_TOOL_ARGUMENTS_BYTES` 16 KiB, `MAX_TOOL_RESULT_BYTES` 16
//!   KiB, `MAX_NATIVE_PLACEMENT_BYTES` 512 (mirrors the description bound),
//!   `MAX_USAGE_EVENTS` 32 (mirrors the per-agent execution scale).
//! - `ToolError` taxonomy plus `normalized` bound/scrub policy (`AI-RUN-008`):
//!   `Denied` (policy refusal, no effect), `Failed` (executed, then failed),
//!   `ProtocolRejected` (rejected before any effect, non-executed),
//!   `EffectUnknown` (uncertain effect, reconcile before retry), and
//!   `UndeclaredPlacement` as a registration refusal (same family as
//!   `DuplicateTool` and `RegistryFull`): never `Denied`, which is
//!   dispatch-time only.
//! - `validate_tool_name` (`TB-2` shape `^[a-z][a-z0-9_]*$`, non-empty, at
//!   most 64 bytes), `ToolPlacement::parse`/`as_str` over the closed
//!   `mcp`/`native` vocabulary (fail-closed), `validate_placement` (`None`
//!   refuses, `Mcp` carrying native detail refuses as contradictory, `Native`
//!   requires non-empty reason/backend within 512 bytes each), and the
//!   `ToolSpec`/`ToolRegistry` shape with the register refusal order
//!   dup-first, then bounds, then placement, then capacity (no state change
//!   on any refusal).
//! - `AuthDecision`/`ToolAuthorizer`/`DenyAllAuthorizer` plus the
//!   `ConsentQuery`/`ConsentDecision`/`ConsentLedger`-seam/`DenyAllConsent`
//!   pair, `DEFAULT_CONSENT_TTL_MS` (`120_000`, one turn-scoped gesture), the
//!   exact `(protocol, tool, scope)` triple rule (never a wildcard, prefix,
//!   session, persistent, trusted-bypass, or auto-approve grant), and the
//!   exclusive-expiry rule (`now_ms < expires_at_ms` allows,
//!   `now_ms == expires_at_ms` denies).
//! - Gate-order documentation (no enforcement here): native
//!   authorize-then-consent-then-cap-then-executor; the MCP six-gate
//!   `stale -> narrowing -> allowlist-lookup -> tier -> authorizer ->
//!   consent -> dispatch` (deny-stale and narrowing are zero-contact);
//!   the agent `narrow -> prompt -> precheck -> dispatch` composition at the
//!   same `now_ms` with admission `Refused`, result `Rejected`, and ack-loss
//!   `Unknown` kept on separate axes.
//! - Narrowing predicate plus `Denied` wording (byte-identical on the bus and
//!   agent paths, `Denied` only, never `UnknownTool`/`Failed`/
//!   `EffectUnknown`/`ProtocolRejected`), usage types plus FNV-1a-64
//!   `args_digest` (digest only, never raw args), and the `check_dispatch`
//!   composition (dispatcher-first; the prompt never grants).
//!
//! Explicitly not in scope (stays runtime/MCP/host-design owned): executor
//! wiring (`ToolExecutor`, test peers, `ExecutionContext`, dispatch,
//! precheck, cap accounting), ledger implementation (grant tables,
//! revocation, drain, capacity), scoped-IPC envelope types, placement
//! routing, cost-fuse accumulation, capability pins, server-binary digest
//! enforcement, SPKI enforcement, durable pending-`Unknown` log, MCP
//! transport, and any thread, network, or filesystem use. `Cargo.toml` stays
//! `std`-only.
//!
//! Deny-all posture holds by construction: [`ToolRegistry::new`] starts
//! empty, [`DenyAllAuthorizer`] denies every call, [`DenyAllConsent`] denies
//! every query, a bus without installed pairs would deny (documented, not
//! built here), and narrowing/prompt gates narrow only (they never grant).

use std::collections::VecDeque;

/// Maximum tool name length in bytes (`TB-2`).
pub const MAX_TOOL_NAME_LEN: usize = 64;
/// Maximum tool description length in bytes (`TB-2`).
pub const MAX_TOOL_DESCRIPTION_LEN: usize = 512;
/// Maximum tool JSON Schema length in bytes (`TB-2`).
pub const MAX_TOOL_SCHEMA_BYTES: usize = 16 * 1024;
/// Maximum registered tools per session (`TB-2`).
pub const MAX_TOOLS_PER_SESSION: usize = 32;
/// Maximum tool calls per logical assistant turn (`TB-6`).
///
/// Hard bus ceiling mirror: the effective per-turn cap stays
/// `min(configured, 8)`, accounted cumulatively across the turn's provider
/// rounds. Raising a config above this constant never relaxes the bus.
/// Only tighter, never looser (fail-closed).
pub const MAX_TOOL_CALLS_PER_TURN: usize = 8;
/// Maximum tool argument bytes (`TB-3`).
pub const MAX_TOOL_ARGUMENTS_BYTES: usize = 16 * 1024;
/// Maximum tool result bytes (`TB-6`).
pub const MAX_TOOL_RESULT_BYTES: usize = 16 * 1024;
/// Maximum native placement reason/backend length in bytes (AIQ-33 S3).
///
/// Mirrors [`MAX_TOOL_DESCRIPTION_LEN`]: short owner-supplied opaque strings
/// share the 512-byte inclusive bound (`len() > LIMIT` refuses).
pub const MAX_NATIVE_PLACEMENT_BYTES: usize = 512;
/// Maximum permission-use audit events retained in the ring.
///
/// Mirrors the per-agent attributed-execution scale (32): the audit ring
/// covers the same execution population as the turn attribution it
/// complements. Oldest-evict on insert; read oldest-first. In-memory only.
pub const MAX_USAGE_EVENTS: usize = 32;
/// Maximum bytes kept for any reason string crossing this mirror boundary.
///
/// Mirrors the runtime reason bound (512): every externally supplied name or
/// reason is scrubbed to printable ASCII and truncated here so `Display`,
/// logs, and attribution records never carry unbounded or newline-bearing
/// text.
pub const MAX_REASON_BYTES: usize = 512;
/// Default consent grant time-to-live in milliseconds (AI-0255).
///
/// One turn-scoped gesture: `120_000` (120s). Used when the granter passes no
/// explicit TTL; an explicit TTL overrides per call. Added to
/// caller-supplied `now_ms` with saturating addition; no wall clock is read
/// anywhere on this path. Never persistent: every grant carries an absolute
/// expiry and lapses without refresh.
pub const DEFAULT_CONSENT_TTL_MS: u64 = 120_000;

/// FNV-1a-64 offset basis (deterministic across processes and platforms).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (deterministic across processes and platforms).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Bound an outbound reason string.
///
/// Every character outside printable ASCII (`0x20..=0x7E`, spaces kept)
/// becomes `?` (defuses CR/LF log injection, terminal escapes, and
/// confusable non-ASCII bytes); the result is truncated to
/// [`MAX_REASON_BYTES`] bytes. Deterministic, silent (no ellipsis marker),
/// and never splits a character, so callers can log the value directly.
#[must_use]
pub fn bound_reason(reason: &str) -> String {
    let mut bounded = String::with_capacity(reason.len().min(MAX_REASON_BYTES));
    for character in reason.chars() {
        if bounded.len() >= MAX_REASON_BYTES {
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

/// Tool-bus errors. Validation and authorization failures leave no partial
/// state: no queue entry, no dispatch, no counter increment.
///
/// Taxonomy mirror: `Denied` is a policy refusal with no effect;
/// `Failed` is executed-then-failed; `ProtocolRejected` is rejected at the
/// protocol level before any effect (non-executed); `EffectUnknown` is an
/// uncertain effect (reconcile before retry). `UndeclaredPlacement` is a
/// registration refusal (same family as `DuplicateTool`/`RegistryFull`):
/// never `Denied`, which is dispatch-time only.
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
    /// fail-closed: no replacement, no shadowing, no state change.
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
    /// Result summary exceeds the context summary bound.
    ///
    /// Mirror-only variant: the summary bound itself lives context-side and
    /// is not re-declared here. Carried so admission and
    /// result-acceptance causes keep the full taxonomy.
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
    /// Placement declaration missing, outside the owner-approved vocabulary,
    /// or native without reason/backend (AIQ-33 S3, refuse-undeclared).
    /// Registration refusal (like [`ToolError::DuplicateTool`] and
    /// [`ToolError::RegistryFull`]): never [`ToolError::Denied`], which is
    /// dispatch-time.
    UndeclaredPlacement {
        /// Rejected tool name (registration boundary) or rejected placement
        /// string (vocabulary parse boundary).
        name: String,
    },
    /// Authorization hook refused the call (`TB-4`).
    Denied {
        /// Tool name.
        name: String,
        /// Hook reason.
        reason: String,
    },
    /// The host executed the tool and reported failure (executed, then
    /// failed). Distinct from [`ToolError::Denied`] (a policy refusal with
    /// no effect) and [`ToolError::EffectUnknown`] (uncertain effect).
    Failed {
        /// Tool name.
        name: String,
        /// Host-reported failure reason.
        reason: String,
    },
    /// The server rejected the call at the protocol level before any effect
    /// (JSON-RPC `-32600` invalid request, `-32601` method not found,
    /// `-32602` invalid params). Non-executed: distinct from
    /// [`ToolError::Failed`] (executed, then failed), [`ToolError::Denied`]
    /// (a policy refusal), and [`ToolError::EffectUnknown`] (uncertain
    /// effect).
    ProtocolRejected {
        /// Tool name.
        name: String,
        /// Protocol rejection reason (code plus bounded server message).
        reason: String,
    },
    /// The effect may have happened but acknowledgement was lost.
    EffectUnknown {
        /// Tool name.
        name: String,
        /// What is uncertain.
        reason: String,
    },
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            Self::UndeclaredPlacement { name } => {
                write!(f, "undeclared tool placement: {name}")
            }
            Self::Denied { name, reason } => write!(f, "tool {name} denied: {reason}"),
            Self::Failed { name, reason } => write!(f, "tool {name} failed: {reason}"),
            Self::ProtocolRejected { name, reason } => {
                write!(f, "tool {name} protocol rejected: {reason}")
            }
            Self::EffectUnknown { name, reason } => {
                write!(f, "tool {name} effect unknown: {reason}")
            }
        }
    }
}

impl std::error::Error for ToolError {}

impl ToolError {
    /// Apply the one bounded diagnostic policy to every host-supplied string
    /// carried by this error, returning the error unchanged when it already
    /// fits.
    ///
    /// The runtime-owned numeric/typed fields (bounds, limits, counts) are
    /// preserved verbatim; only externally sourced text (tool names and
    /// authorizer/executor denial, failure, and uncertainty reasons) is
    /// scrubbed to printable ASCII and bounded to [`MAX_REASON_BYTES`].
    #[must_use]
    pub fn normalized(self) -> Self {
        match self {
            Self::InvalidName { name } => Self::InvalidName {
                name: bound_reason(&name),
            },
            Self::UnknownTool { name } => Self::UnknownTool {
                name: bound_reason(&name),
            },
            Self::DuplicateTool { name } => Self::DuplicateTool {
                name: bound_reason(&name),
            },
            Self::UndeclaredPlacement { name } => Self::UndeclaredPlacement {
                name: bound_reason(&name),
            },
            Self::Denied { name, reason } => Self::Denied {
                name: bound_reason(&name),
                reason: bound_reason(&reason),
            },
            Self::Failed { name, reason } => Self::Failed {
                name: bound_reason(&name),
                reason: bound_reason(&reason),
            },
            Self::ProtocolRejected { name, reason } => Self::ProtocolRejected {
                name: bound_reason(&name),
                reason: bound_reason(&reason),
            },
            Self::EffectUnknown { name, reason } => Self::EffectUnknown {
                name: bound_reason(&name),
                reason: bound_reason(&reason),
            },
            Self::ArgumentsTooLarge { .. }
            | Self::ResultTooLarge { .. }
            | Self::SummaryTooLarge { .. }
            | Self::CallLimitExceeded { .. }
            | Self::RegistryFull { .. }
            | Self::DescriptionTooLarge { .. }
            | Self::SchemaTooLarge { .. } => self,
        }
    }
}

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
            name: bound_reason(name),
        })
    }
}

/// Closed tool-placement vocabulary (AIQ-33 S3, owner-approved).
///
/// Data-only declaration: routing is unchanged by placement. `Mcp` marks
/// tools imported over MCP; `Native` marks host-native tools and
/// additionally requires [`NativePlacement`] reason/backend. Closed:
/// unknown strings never synthesize a placement ([`ToolPlacement::parse`]
/// fails closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPlacement {
    /// Tool imported over MCP.
    Mcp,
    /// Host-native tool (requires [`ToolSpec::native`] reason/backend).
    Native,
}

impl ToolPlacement {
    /// Parse the owner-approved vocabulary strings (`mcp`/`native`).
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::UndeclaredPlacement`] carrying the rejected
    /// (bounded, scrubbed) string for anything else.
    pub fn parse(value: &str) -> Result<Self, ToolError> {
        match value {
            "mcp" => Ok(Self::Mcp),
            "native" => Ok(Self::Native),
            _ => Err(ToolError::UndeclaredPlacement {
                name: bound_reason(value),
            }),
        }
    }

    /// Owner-approved vocabulary string for this placement.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mcp => "mcp",
            Self::Native => "native",
        }
    }
}

/// Native-placement reason/backend (AIQ-33 S3).
///
/// Data-only: stored on the declaration, never interpreted. Required
/// exactly when [`ToolSpec::placement`] is [`ToolPlacement::Native`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlacement {
    /// Why this tool is host-native (owner-supplied, opaque).
    pub reason: String,
    /// Which native backend serves it (owner-supplied label, opaque).
    pub backend: String,
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
    /// Whether the tool is side-effect free.
    pub read_only: bool,
    /// Declared placement (AIQ-33 S3, refuse-undeclared): `None` is
    /// undeclared and [`ToolRegistry::register`] refuses it fail-closed
    /// with [`ToolError::UndeclaredPlacement`].
    pub placement: Option<ToolPlacement>,
    /// Native reason/backend; required exactly when `placement` is
    /// [`ToolPlacement::Native`] (refused when missing or empty, and refused
    /// when present under [`ToolPlacement::Mcp`] as a contradictory
    /// declaration).
    pub native: Option<NativePlacement>,
}

impl ToolSpec {
    /// Construct and validate a declaration.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] for malformed names, over-bound description
    /// and schema, undeclared placement, or a native/native-detail mismatch
    /// (native without reason/backend, or MCP carrying native detail).
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema_json: Vec<u8>,
        required_scope: impl Into<String>,
        read_only: bool,
        placement: ToolPlacement,
        native: Option<NativePlacement>,
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
        validate_placement(&name, Some(placement), native.as_ref())?;
        Ok(Self {
            name,
            description,
            schema_json,
            required_scope: required_scope.into(),
            read_only,
            placement: Some(placement),
            native,
        })
    }
}

/// Validate a placement declaration (AIQ-33 S3, refuse-undeclared).
///
/// `None` is undeclared; `Native` requires a non-empty reason/backend within
/// [`MAX_NATIVE_PLACEMENT_BYTES`] bytes each (inclusive bound); `Mcp`
/// carrying native detail is a contradictory declaration. Every violation
/// fails closed with [`ToolError::UndeclaredPlacement`]. The caller
/// shape-checks `name` first (as [`ToolSpec::new`] and
/// [`ToolRegistry::register`] do), so the refusal carries it raw.
///
/// # Errors
///
/// Returns [`ToolError::UndeclaredPlacement`] for a missing, mismatched,
/// contradictory, or over-long declaration.
fn validate_placement(
    name: &str,
    placement: Option<ToolPlacement>,
    native: Option<&NativePlacement>,
) -> Result<(), ToolError> {
    let refused = || ToolError::UndeclaredPlacement {
        name: name.to_owned(),
    };
    let Some(placement) = placement else {
        return Err(refused());
    };
    match (placement, native) {
        (ToolPlacement::Mcp, None) => Ok(()),
        (ToolPlacement::Mcp, Some(_)) => Err(refused()),
        (ToolPlacement::Native, Some(detail))
            if !detail.reason.is_empty()
                && !detail.backend.is_empty()
                && detail.reason.len() <= MAX_NATIVE_PLACEMENT_BYTES
                && detail.backend.len() <= MAX_NATIVE_PLACEMENT_BYTES =>
        {
            Ok(())
        }
        (ToolPlacement::Native, _) => Err(refused()),
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
    /// Fail-closed refusal order (no state change on any refusal):
    /// duplicate name first, then name/description/schema bounds, then
    /// placement, then capacity. An undeclared placement (`None`), a native
    /// spec without reason/backend, or an MCP spec carrying native detail is
    /// refused with [`ToolError::UndeclaredPlacement`] (AIQ-33 S3,
    /// refuse-undeclared, never [`ToolError::Denied`], which is
    /// dispatch-time); a full registry refuses with
    /// [`ToolError::RegistryFull`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::DuplicateTool`] for a re-registered name,
    /// [`ToolError::InvalidName`], [`ToolError::DescriptionTooLarge`],
    /// [`ToolError::SchemaTooLarge`], or [`ToolError::UndeclaredPlacement`]
    /// for a malformed spec, and [`ToolError::RegistryFull`] past
    /// [`MAX_TOOLS_PER_SESSION`].
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
        validate_placement(&spec.name, spec.placement, spec.native.as_ref())?;
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

/// Runtime-local agent instance handle (host-side mirror, `u64` identity).
///
/// Local mirror of the runtime session vocabulary so [`AuthBase`] and usage
/// attribution keep the same shape without depending on the runtime crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AgentInstanceId(pub u64);

/// Session handle (host-side mirror, `u64` identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SessionId(pub u64);

/// One dispatched tool execution (attribution handle, host-side mirror).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExecutionId(pub u64);

/// Server-side authority tier of the session (host-side mirror).
///
/// `Inspect` is read-only: it may only reach tools declaring `read_only`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AgentLevel {
    /// Read-only tier. Default for every fresh session.
    #[default]
    Inspect,
    /// `Inspect` plus ephemeral scratch and model turns.
    Own,
    /// `Own` plus reads/writes within the current workspace.
    Workspace,
    /// `Workspace` plus cross-workspace actions under separate grants.
    All,
}

/// Caller identity presented to the authorization hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthBase {
    /// Calling agent instance.
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

/// Authorization verdict. There is no allow-all grant in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthDecision {
    /// Proceed toward dispatch.
    Allow,
    /// Refuse with no dispatch and no partial state.
    Deny {
        /// Hook reason (attributed in enforcement records).
        reason: String,
    },
}

/// Per-tool authorization hook (`TB-4` seam). The host implements capability
/// plus consent checks behind it; the mirror default denies.
pub trait ToolAuthorizer {
    /// Decide one tool call. Checked at validation time and again at each
    /// dispatch boundary, so revocation takes effect for the next dispatch.
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
///
/// Mirror of the runtime outcome axis so usage attribution keeps the same
/// vocabulary: `Refused` is admission-only (before any executor contact);
/// `Denied` is a host/policy refusal; `Failed` is executed-then-failed;
/// `Unknown` is ack-loss (never `Failed`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    /// Host executed and acknowledged.
    Success,
    /// Host executed and reported failure.
    Failed {
        /// Host reason.
        reason: String,
    },
    /// Refused by host/policy (no effect conflated with admission).
    Denied {
        /// Hook or host reason.
        reason: String,
    },
    /// Refused at the dispatch boundary before any executor contact (no
    /// effect, nothing attempted). Carries the typed admission cause.
    Refused {
        /// Typed admission failure.
        cause: ToolError,
    },
    /// Effect uncertain: reconcile before retry, never blindly retry.
    Unknown {
        /// What is uncertain.
        reason: String,
    },
}

impl ToolStatus {
    /// Whether this status represents a call refused at the dispatch
    /// boundary before any executor contact.
    ///
    /// `Refused` is admission-only; `Denied` is a host/policy refusal after
    /// contact, so it is deliberately not included here.
    #[must_use]
    pub fn is_admission_refusal(&self) -> bool {
        matches!(self, Self::Refused { .. })
    }

    /// Whether this status is ack-loss.
    ///
    /// True only for `Unknown`: a crash between effect and acknowledgement.
    /// Ack-loss stays `Unknown` (never `Failed`). Pure.
    #[must_use]
    pub fn is_ack_loss(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }

    /// Apply the one bounded diagnostic policy to the host-supplied reason
    /// carried by a terminal status.
    #[must_use]
    pub fn normalized(self) -> Self {
        match self {
            Self::Failed { reason } => Self::Failed {
                reason: bound_reason(&reason),
            },
            Self::Denied { reason } => Self::Denied {
                reason: bound_reason(&reason),
            },
            Self::Unknown { reason } => Self::Unknown {
                reason: bound_reason(&reason),
            },
            Self::Refused { cause } => Self::Refused {
                cause: cause.normalized(),
            },
            Self::Success => Self::Success,
        }
    }
}

/// Terminal accepted-or-refused marker kept separate from [`ToolStatus`] so
/// an acknowledged effect is never conflated with its result acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultDisposition {
    /// The result was accepted and recorded as-is.
    Accepted,
    /// The effect happened but the payload could not be accepted (over-bound
    /// summary or result bytes). Carries the typed bound failure.
    Rejected {
        /// Typed acceptance failure.
        cause: ToolError,
    },
}

impl ResultDisposition {
    /// Whether the effect happened but its payload was refused.
    #[must_use]
    pub fn is_rejection(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }
}

/// Where a tool call executed (audit site label).
///
/// Data-only attribution: `placement` is derived from the registry at
/// dispatch time, never caller-supplied; `server_id`/`task_id` are optional
/// host-threaded labels. There is deliberately no `cwd`, argument, summary,
/// or result field: raw arguments never enter the ring (only
/// [`UsageEvent::args_digest`] digests them).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UsageSite {
    /// Declared placement of the dispatched tool, or `None` when the tool
    /// was unregistered (admission refusal before lookup).
    pub placement: Option<ToolPlacement>,
    /// Owning server label when the host threads one.
    pub server_id: Option<String>,
    /// Task label when the host threads one.
    pub task_id: Option<String>,
}

/// One per-call permission-use attribution.
///
/// Reuses the execution id, tool name, registry `required_scope`, wire
/// principal, caller triple, dispatch `now_ms`, site where-label, outcome
/// axes, bounded reason, and FNV-1a-64 `args_digest` (never raw args). All
/// variable-length strings are bounded on ring push (idempotent). The ring
/// carries attribution, never payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageEvent {
    /// Attribution handle for this dispatch.
    pub execution_id: ExecutionId,
    /// Dispatched tool name (bounded on push).
    pub tool: String,
    /// Capability scope the registry declaration requires (empty when the
    /// tool was unregistered).
    pub scope: String,
    /// Wire principal charged for consent (empty when no consent pair is
    /// installed).
    pub protocol_id: String,
    /// Caller identity and tier (who).
    pub caller: AuthBase,
    /// Caller-supplied dispatch timestamp in milliseconds (when).
    pub occurred_at_ms: u64,
    /// Placement plus optional server/task labels (where; never cwd).
    pub site: UsageSite,
    /// Effect axis of the dispatch outcome.
    pub status: ToolStatus,
    /// Result-acceptance axis of the dispatch outcome.
    pub result_disposition: ResultDisposition,
    /// Bounded diagnostic reason derived from `status` (empty for success).
    pub reason: String,
    /// FNV-1a-64 over the raw argument bytes (digest only, never raw args).
    pub args_digest: u64,
}

impl UsageEvent {
    /// Deterministic digest of raw call arguments.
    ///
    /// FNV-1a-64 over exactly the argument bytes (inline small hasher;
    /// never `DefaultHasher`/`RandomState`). Same bytes give the same digest
    /// on every platform and process; any byte change moves it. The digest
    /// is the privacy handle: the ring stores this `u64`, never the raw
    /// bytes.
    #[must_use]
    pub fn args_digest(args: &[u8]) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for byte in args {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

/// Read-only filter for [`UsageRing::list`].
///
/// Every `Some` field must match exactly; `None` constrains nothing. The
/// default (all `None`) matches every event. Pure predicate, no state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UsageFilter {
    /// Match the dispatched tool name exactly.
    pub tool: Option<String>,
    /// Match the threaded required scope exactly.
    pub scope: Option<String>,
    /// Match the wire protocol id exactly.
    pub protocol_id: Option<String>,
    /// Match the caller agent-instance id exactly.
    pub agent_instance_id: Option<AgentInstanceId>,
    /// Match the caller session id exactly.
    pub session_id: Option<SessionId>,
    /// Match the derived placement exactly.
    pub placement: Option<ToolPlacement>,
}

impl UsageFilter {
    /// Filter matching every event (all fields `None`).
    #[must_use]
    pub fn match_all() -> Self {
        Self::default()
    }

    /// Whether `event` satisfies every constrained field.
    #[must_use]
    pub fn matches(&self, event: &UsageEvent) -> bool {
        if let Some(tool) = &self.tool {
            if event.tool != *tool {
                return false;
            }
        }
        if let Some(scope) = &self.scope {
            if event.scope != *scope {
                return false;
            }
        }
        if let Some(protocol_id) = &self.protocol_id {
            if event.protocol_id != *protocol_id {
                return false;
            }
        }
        if let Some(agent_instance_id) = &self.agent_instance_id {
            if event.caller.agent_instance_id != *agent_instance_id {
                return false;
            }
        }
        if let Some(session_id) = &self.session_id {
            if event.caller.session_id != *session_id {
                return false;
            }
        }
        if let Some(placement) = &self.placement {
            if event.site.placement.as_ref() != Some(placement) {
                return false;
            }
        }
        true
    }
}

/// In-memory permission-use audit ring (host-side mirror).
///
/// Pure mirror of the runtime bus ring policy without any dispatch: every
/// push is normalized (tool, scope, protocol, server, task, and reason
/// bounded; status normalized) and oldest-evicts past
/// [`MAX_USAGE_EVENTS`]; reads are oldest-first clones capped at `limit`
/// (`limit == 0` returns no rows). Append-only by construction: no delete
/// or update API.
#[derive(Debug, Default)]
pub struct UsageRing {
    events: VecDeque<UsageEvent>,
}

impl UsageRing {
    /// Empty ring.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one event, oldest-evicting past [`MAX_USAGE_EVENTS`].
    pub fn push(&mut self, mut event: UsageEvent) {
        event.tool = bound_reason(&event.tool);
        event.scope = bound_reason(&event.scope);
        event.protocol_id = bound_reason(&event.protocol_id);
        event.status = event.status.normalized();
        event.reason = bound_reason(&event.reason);
        if let Some(server_id) = event.site.server_id.take() {
            let bounded = bound_reason(&server_id);
            event.site.server_id = if bounded.is_empty() {
                None
            } else {
                Some(bounded)
            };
        }
        if let Some(task_id) = event.site.task_id.take() {
            let bounded = bound_reason(&task_id);
            event.site.task_id = if bounded.is_empty() {
                None
            } else {
                Some(bounded)
            };
        }
        if self.events.len() >= MAX_USAGE_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Retained event count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the ring holds no event.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Read-only snapshot, oldest-first, matching `filter`, capped at
    /// `limit` rows. Pure; mutating the returned vector leaves the ring
    /// unchanged.
    #[must_use]
    pub fn list(&self, filter: &UsageFilter, limit: usize) -> Vec<UsageEvent> {
        self.events
            .iter()
            .filter(|event| filter.matches(event))
            .take(limit)
            .cloned()
            .collect()
    }
}

/// Per-call consent query presented to the ledger seam.
///
/// All strings are wire-level values: the protocol principal, the requested
/// tool name, and the capability scope the tool declaration requires.
/// `now_ms` is caller-supplied; no wall clock is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsentQuery<'a> {
    /// Wire protocol principal making the call.
    pub protocol_id: &'a str,
    /// Agent instance the principal is bound to.
    pub agent_instance: AgentInstanceId,
    /// Requested tool name.
    pub tool: &'a str,
    /// Capability scope the tool declaration requires.
    pub scope: &'a str,
    /// Caller-supplied timestamp; expiry is evaluated against this.
    pub now_ms: u64,
}

/// Consent verdict. There is no allow-all grant in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsentDecision {
    /// Proceed toward dispatch (still subject to bus gates).
    Allow,
    /// Refuse with no dispatch and no partial state.
    Deny {
        /// Hook reason (attributed in enforcement records).
        reason: String,
    },
}

/// Consent-ledger seam (host-owned ledger).
///
/// The host implements per-client consent checks behind this trait; this
/// module defines only the seam plus the deny-by-default posture, and claims
/// no accepted security mechanism.
pub trait ConsentLedger {
    /// Decide one tool call at `query.now_ms`.
    ///
    /// Implementations must be deterministic for a given grant table plus
    /// `now_ms` when used under test. Revocation takes effect on the next
    /// check: there is no caching across calls.
    fn check(&self, query: &ConsentQuery) -> ConsentDecision;
}

/// Ledger seam that denies every call (default posture).
#[derive(Debug, Default)]
pub struct DenyAllConsent;

impl ConsentLedger for DenyAllConsent {
    fn check(&self, query: &ConsentQuery) -> ConsentDecision {
        ConsentDecision::Deny {
            reason: bound_reason(&format!("default-deny: no grant for {}", query.tool)),
        }
    }
}

/// One exact `(protocol, tool, scope)` grant with absolute expiry.
///
/// Data-only mirror of the ledger row (no table, no capacity, no revoke):
/// a check allows only when one entry matches all three strings and
/// `now_ms` precedes its expiry. Refreshing a triple updates its expiry;
/// expiry is evaluated against caller-supplied `now_ms`, never a wall
/// clock. There is deliberately no wildcard, prefix, session-scoped, or
/// persistent grant shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentGrant {
    /// Wire protocol principal.
    pub protocol: String,
    /// Granted tool name.
    pub tool: String,
    /// Granted capability scope.
    pub scope: String,
    /// Absolute expiry in milliseconds.
    pub expires_at_ms: u64,
}

impl ConsentGrant {
    /// Whether this grant covers `query`: exact match on all three strings
    /// and `query.now_ms < expires_at_ms` (exclusive boundary: equality
    /// denies, matching the host ledger rule).
    #[must_use]
    pub fn covers(&self, query: &ConsentQuery) -> bool {
        self.protocol == query.protocol_id
            && self.tool == query.tool
            && self.scope == query.scope
            && query.now_ms < self.expires_at_ms
    }
}

/// Compute the absolute expiry for a grant starting at `now_ms`.
///
/// `ttl_ms.unwrap_or(DEFAULT_CONSENT_TTL_MS)` added with saturating
/// addition; no wall clock is read. Pure.
#[must_use]
pub fn consent_grant_expiry(now_ms: u64, ttl_ms: Option<u64>) -> u64 {
    now_ms.saturating_add(ttl_ms.unwrap_or(DEFAULT_CONSENT_TTL_MS))
}

/// Map a ledger denial to the shared dispatch-path refusal.
///
/// Mirror of the runtime `ensure_consented_for_dispatch` rendering: denial
/// returns [`ToolError::Denied`] (never `Failed`/`Unknown`/`ProtocolRejected`,
/// no new variant) with the established `consent denied: ...` wording, so
/// both the native dispatch path and the MCP adapter execute path report
/// byte-identical reasons pre-contact.
#[must_use]
pub fn consent_denial(tool: &str, reason: &str) -> ToolError {
    ToolError::Denied {
        name: bound_reason(tool),
        reason: bound_reason(&format!("consent denied: {reason}")),
    }
    .normalized()
}

/// Whether `tool` is narrowed away by a per-call allow-set.
///
/// `None` constrains nothing; `Some(list)` admits only the listed names (an
/// explicit empty list narrows everything away). Pure predicate, no state.
/// Mirrors the bus and agent narrowing predicate byte-for-byte.
#[must_use]
pub fn narrowing_omits(allowed: Option<&[String]>, tool: &str) -> bool {
    match allowed {
        None => false,
        Some(list) => !list.iter().any(|entry| entry == tool),
    }
}

/// Typed narrowing refusal for one tool name: [`ToolError::Denied`] naming
/// the tool (never `UnknownTool`/`Failed`/`EffectUnknown`/`ProtocolRejected`,
/// no new variant), normalized because the refused name is model-supplied.
///
/// Wording is byte-identical to the runtime bus/agent denial:
/// `tool '<name>' omitted by per-call narrowing`.
#[must_use]
pub fn narrowing_denial(tool: &str) -> ToolError {
    ToolError::Denied {
        name: tool.to_owned(),
        reason: format!("tool '{tool}' omitted by per-call narrowing"),
    }
    .normalized()
}

/// Prompt-gate refusal vocabulary (host-side mirror).
///
/// Kept minimal: only the three refusals reachable from [`check_dispatch`].
/// Any other prompt failure (assembly-time) never reaches dispatch and is
/// not mirrored here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptError {
    /// The dispatcher (capability registry) does not grant the tool.
    DispatcherDenied {
        /// Refused tool.
        tool: String,
    },
    /// The prompt denies the tool.
    PromptDenied {
        /// Refused tool.
        tool: String,
    },
    /// A constrained allow-set omits the tool.
    PromptNotAllowed {
        /// Refused tool.
        tool: String,
    },
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DispatcherDenied { tool } => {
                write!(f, "dispatcher denied tool '{tool}'")
            }
            Self::PromptDenied { tool } => write!(f, "prompt denied tool '{tool}'"),
            Self::PromptNotAllowed { tool } => {
                write!(f, "prompt allow-set omits tool '{tool}'")
            }
        }
    }
}

impl std::error::Error for PromptError {}

/// Check one dispatch against the prompt allow/deny sets plus the
/// dispatcher grant.
///
/// Fail-closed composition mirror: dispatcher denial first, then prompt
/// denial, then allow-set omission. The prompt never grants:
/// `dispatcher_grants == false` always denies, even when the prompt text or
/// allow-set names the tool. The prompt only narrows.
///
/// # Errors
///
/// Returns [`PromptError::DispatcherDenied`] when the dispatcher does not
/// grant, [`PromptError::PromptDenied`] when the prompt denies, or
/// [`PromptError::PromptNotAllowed`] when a constrained allow-set omits the
/// tool.
pub fn check_dispatch(
    allowed: Option<&[String]>,
    denied: &[String],
    tool: &str,
    dispatcher_grants: bool,
) -> Result<(), PromptError> {
    if !dispatcher_grants {
        return Err(PromptError::DispatcherDenied {
            tool: tool.to_owned(),
        });
    }
    if denied.iter().any(|entry| entry == tool) {
        return Err(PromptError::PromptDenied {
            tool: tool.to_owned(),
        });
    }
    if let Some(list) = allowed {
        if !list.iter().any(|entry| entry == tool) {
            return Err(PromptError::PromptNotAllowed {
                tool: tool.to_owned(),
            });
        }
    }
    Ok(())
}

/// Map a prompt-gate refusal to its pre-contact [`ToolError::Denied`] cause.
///
/// Pure error mapping for the agent-boundary composition (every call of a
/// batch is checked with [`check_dispatch`] before precheck): the refusal
/// keeps its tool identity and carries its display text as the reason.
/// Never [`ToolError::Failed`], [`ToolError::EffectUnknown`], or
/// [`ToolError::ProtocolRejected`]: a pre-contact refusal executed nothing.
#[must_use]
pub fn dispatch_denial_cause(tool: &str, error: &PromptError) -> ToolError {
    // Signature mirrors the runtime helper (which falls back to `tool` for
    // prompt variants unreachable from `check_dispatch`); every variant
    // mirrored here carries its own identity, so the fallback stays unused.
    let _ = tool;
    let name = match error {
        PromptError::DispatcherDenied { tool }
        | PromptError::PromptDenied { tool }
        | PromptError::PromptNotAllowed { tool } => tool.clone(),
    };
    ToolError::Denied {
        name,
        reason: error.to_string(),
    }
    .normalized()
}

/// Gate order the host must re-impose when dispatch moves host-side (no
/// enforcement here, documentation only).
///
/// ```text
/// native: authorize -> consent -> cap -> executor
/// mcp:    stale -> narrowing -> allowlist-lookup -> tier -> authorizer
///         -> consent -> dispatch (deny-stale and narrowing are zero-contact)
/// agent:  narrow -> prompt -> precheck -> dispatch (same now_ms, pre-contact)
/// ```
///
/// Every refusal is admission-only with zero executor/transport contact,
/// reported as [`ToolError::Denied`] carrying the typed cause (narrowing,
/// prompt, authorizer, or consent), never `Failed`/`Unknown`/
/// `ProtocolRejected`. Consent is always the exact `(protocol, tool, scope)`
/// triple at the dispatch `now_ms` (no caching across calls); the prompt
/// never grants (it only narrows what the dispatcher would allow).
pub const GATE_ORDER_DOC: &str = "native: authorize -> consent -> cap -> executor; \
    mcp: stale -> narrowing -> allowlist-lookup -> tier -> authorizer -> consent -> dispatch; \
    agent: narrow -> prompt -> precheck -> dispatch";

#[cfg(test)]
mod tests {
    use super::*;

    fn mcp_spec(name: &str) -> ToolSpec {
        ToolSpec::new(
            name,
            "test tool",
            br#"{"type":"object"}"#.to_vec(),
            "test.scope",
            true,
            ToolPlacement::Mcp,
            None,
        )
        .expect("valid spec")
    }

    fn base() -> AuthBase {
        AuthBase {
            agent_instance_id: AgentInstanceId(1),
            session_id: SessionId(2),
            level: AgentLevel::Inspect,
        }
    }

    fn usage_event(tool: &str, now_ms: u64) -> UsageEvent {
        UsageEvent {
            execution_id: ExecutionId(7),
            tool: tool.to_owned(),
            scope: "test.scope".to_owned(),
            protocol_id: "local.assistant".to_owned(),
            caller: base(),
            occurred_at_ms: now_ms,
            site: UsageSite {
                placement: Some(ToolPlacement::Mcp),
                server_id: None,
                task_id: None,
            },
            status: ToolStatus::Success,
            result_disposition: ResultDisposition::Accepted,
            reason: String::new(),
            args_digest: UsageEvent::args_digest(b"{}"),
        }
    }

    #[test]
    fn bounds_match_runtime_values() {
        assert_eq!(MAX_TOOL_NAME_LEN, 64);
        assert_eq!(MAX_TOOL_DESCRIPTION_LEN, 512);
        assert_eq!(MAX_TOOL_SCHEMA_BYTES, 16 * 1024);
        assert_eq!(MAX_TOOLS_PER_SESSION, 32);
        assert_eq!(MAX_TOOL_CALLS_PER_TURN, 8);
        assert_eq!(MAX_TOOL_ARGUMENTS_BYTES, 16 * 1024);
        assert_eq!(MAX_TOOL_RESULT_BYTES, 16 * 1024);
        assert_eq!(MAX_NATIVE_PLACEMENT_BYTES, 512);
        assert_eq!(MAX_USAGE_EVENTS, 32);
        assert_eq!(MAX_REASON_BYTES, 512);
        assert_eq!(DEFAULT_CONSENT_TTL_MS, 120_000);
    }

    #[test]
    fn tool_name_shape_mirror() {
        assert!(validate_tool_name("workspace_read").is_ok());
        assert!(validate_tool_name("a").is_ok());
        assert!(validate_tool_name("tool_1").is_ok());
        assert!(validate_tool_name("").is_err());
        assert!(validate_tool_name("Read").is_err());
        assert!(validate_tool_name("9start").is_err());
        assert!(validate_tool_name("terminal.read").is_err());
        assert!(validate_tool_name("tool-name").is_err());
        assert!(validate_tool_name(&"a".repeat(65)).is_err());
        assert!(validate_tool_name(&"a".repeat(64)).is_ok());
    }

    #[test]
    fn registry_refuses_duplicate_first() {
        let mut registry = ToolRegistry::new();
        registry.register(mcp_spec("workspace_read")).unwrap();
        let before = registry.len();
        let err = registry.register(mcp_spec("workspace_read")).unwrap_err();
        assert_eq!(
            err,
            ToolError::DuplicateTool {
                name: "workspace_read".to_owned()
            }
        );
        assert_eq!(registry.len(), before);
    }

    #[test]
    fn registry_refusal_order_bounds_before_placement_before_capacity() {
        // Bounds beat placement: over-bound description with an undeclared
        // placement still reports the bound, not placement.
        let mut registry = ToolRegistry::new();
        let spec = ToolSpec {
            name: "ok_name".to_owned(),
            description: "x".repeat(MAX_TOOL_DESCRIPTION_LEN + 1),
            schema_json: Vec::new(),
            required_scope: "test.scope".to_owned(),
            read_only: true,
            placement: None,
            native: None,
        };
        let err = registry.register(spec).unwrap_err();
        assert!(matches!(err, ToolError::DescriptionTooLarge { .. }));

        // Duplicate beats bounds: a duplicate name with an over-bound
        // description still reports the duplicate.
        let mut registry = ToolRegistry::new();
        registry.register(mcp_spec("dup_tool")).unwrap();
        let spec = ToolSpec {
            name: "dup_tool".to_owned(),
            description: "x".repeat(MAX_TOOL_DESCRIPTION_LEN + 1),
            schema_json: Vec::new(),
            required_scope: "test.scope".to_owned(),
            read_only: true,
            placement: Some(ToolPlacement::Mcp),
            native: None,
        };
        let err = registry.register(spec).unwrap_err();
        assert!(matches!(err, ToolError::DuplicateTool { .. }));
        assert_eq!(registry.len(), 1);

        // Placement beats capacity: an undeclared placement on a full
        // registry still reports placement, not capacity.
        let mut registry = ToolRegistry::new();
        for index in 0..MAX_TOOLS_PER_SESSION {
            registry
                .register(mcp_spec(&format!("tool_{index}")))
                .unwrap();
        }
        assert_eq!(registry.len(), MAX_TOOLS_PER_SESSION);
        let spec = ToolSpec {
            name: "placement_probe".to_owned(),
            description: "test".to_owned(),
            schema_json: Vec::new(),
            required_scope: "test.scope".to_owned(),
            read_only: true,
            placement: None,
            native: None,
        };
        let err = registry.register(spec).unwrap_err();
        assert!(matches!(err, ToolError::UndeclaredPlacement { .. }));

        // Capacity last: a well-formed spec on a full registry reports full.
        let err = registry.register(mcp_spec("one_more")).unwrap_err();
        assert_eq!(
            err,
            ToolError::RegistryFull {
                limit: MAX_TOOLS_PER_SESSION
            }
        );
    }

    #[test]
    fn placement_vocabulary_closed_and_refusals() {
        assert_eq!(ToolPlacement::parse("mcp").unwrap(), ToolPlacement::Mcp);
        assert_eq!(
            ToolPlacement::parse("native").unwrap(),
            ToolPlacement::Native
        );
        assert_eq!(ToolPlacement::Mcp.as_str(), "mcp");
        assert_eq!(ToolPlacement::Native.as_str(), "native");
        let err = ToolPlacement::parse("shell").unwrap_err();
        assert!(matches!(err, ToolError::UndeclaredPlacement { .. }));

        // Mcp carrying native detail is contradictory.
        let spec = ToolSpec {
            name: "mcp_tool".to_owned(),
            description: "test".to_owned(),
            schema_json: Vec::new(),
            required_scope: "test.scope".to_owned(),
            read_only: true,
            placement: Some(ToolPlacement::Mcp),
            native: Some(NativePlacement {
                reason: "r".to_owned(),
                backend: "b".to_owned(),
            }),
        };
        let mut registry = ToolRegistry::new();
        let err = registry.register(spec).unwrap_err();
        assert!(matches!(err, ToolError::UndeclaredPlacement { .. }));

        // Native without detail refuses; with detail admits.
        let spec = ToolSpec {
            name: "native_tool".to_owned(),
            description: "test".to_owned(),
            schema_json: Vec::new(),
            required_scope: "test.scope".to_owned(),
            read_only: false,
            placement: Some(ToolPlacement::Native),
            native: None,
        };
        let mut registry = ToolRegistry::new();
        let err = registry.register(spec).unwrap_err();
        assert!(matches!(err, ToolError::UndeclaredPlacement { .. }));

        let spec = ToolSpec::new(
            "native_tool",
            "test",
            Vec::new(),
            "test.scope",
            false,
            ToolPlacement::Native,
            Some(NativePlacement {
                reason: "host-native".to_owned(),
                backend: "backend".to_owned(),
            }),
        )
        .expect("native with detail admits");
        let mut registry = ToolRegistry::new();
        registry.register(spec).expect("capacity");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn denial_taxonomy_never_conflates() {
        let denied = ToolError::Denied {
            name: "tool_a".to_owned(),
            reason: "policy".to_owned(),
        };
        let failed = ToolError::Failed {
            name: "tool_a".to_owned(),
            reason: "host failed".to_owned(),
        };
        let rejected = ToolError::ProtocolRejected {
            name: "tool_a".to_owned(),
            reason: "-32601".to_owned(),
        };
        let unknown = ToolError::EffectUnknown {
            name: "tool_a".to_owned(),
            reason: "ack lost".to_owned(),
        };
        assert_ne!(denied, failed);
        assert_ne!(denied, rejected);
        assert_ne!(denied, unknown);
        assert_ne!(failed, rejected);
        assert!(format!("{denied}").contains("denied"));
        assert!(format!("{failed}").contains("failed"));
        assert!(format!("{rejected}").contains("protocol rejected"));
        assert!(format!("{unknown}").contains("effect unknown"));

        // Registration refusal is never dispatch-Denied.
        let err = ToolPlacement::parse("nope").unwrap_err();
        assert!(matches!(err, ToolError::UndeclaredPlacement { .. }));
        assert!(!matches!(err, ToolError::Denied { .. }));

        // Status axes stay distinct: admission Refused vs host Denied vs
        // ack-loss Unknown.
        let refused = ToolStatus::Refused {
            cause: denied.clone(),
        };
        assert!(refused.is_admission_refusal());
        assert!(
            !ToolStatus::Denied {
                reason: "x".to_owned()
            }
            .is_admission_refusal()
        );
        assert!(
            ToolStatus::Unknown {
                reason: "x".to_owned()
            }
            .is_ack_loss()
        );
        assert!(
            !ToolStatus::Failed {
                reason: "x".to_owned()
            }
            .is_ack_loss()
        );
        assert!(!matches!(
            ResultDisposition::Accepted,
            ResultDisposition::Rejected { .. }
        ));
        assert!(ResultDisposition::Rejected { cause: denied }.is_rejection());
    }

    #[test]
    fn normalized_scrubs_and_bounds() {
        let evil = "ok\nline\x1besc\u{00e9}";
        let err = ToolError::Denied {
            name: "tool_a".to_owned(),
            reason: evil.to_owned(),
        }
        .normalized();
        match err {
            ToolError::Denied { reason, .. } => {
                assert!(!reason.contains('\n'));
                assert!(!reason.contains('\x1b'));
                assert!(reason.contains('?'));
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        let long = "x".repeat(MAX_REASON_BYTES + 100);
        let err = ToolError::Failed {
            name: long.clone(),
            reason: long.clone(),
        }
        .normalized();
        match err {
            ToolError::Failed { name, reason } => {
                assert_eq!(name.len(), MAX_REASON_BYTES);
                assert_eq!(reason.len(), MAX_REASON_BYTES);
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn narrowing_predicate_and_denied_wording() {
        assert!(!narrowing_omits(None, "anything"));
        assert!(!narrowing_omits(Some(&["tool_a".to_owned()]), "tool_a"));
        assert!(narrowing_omits(Some(&["tool_a".to_owned()]), "tool_b"));
        assert!(narrowing_omits(Some(&[]), "tool_a"));

        let denial = narrowing_denial("workspace_read");
        assert_eq!(
            denial,
            ToolError::Denied {
                name: "workspace_read".to_owned(),
                reason: "tool 'workspace_read' omitted by per-call narrowing".to_owned(),
            }
        );
        assert!(!matches!(denial, ToolError::UnknownTool { .. }));
        assert!(!matches!(denial, ToolError::Failed { .. }));
        assert!(!matches!(denial, ToolError::EffectUnknown { .. }));
        assert!(!matches!(denial, ToolError::ProtocolRejected { .. }));
    }

    #[test]
    fn consent_exact_triple_and_exclusive_expiry() {
        let grant = ConsentGrant {
            protocol: "local.assistant".to_owned(),
            tool: "workspace_read".to_owned(),
            scope: "workspace.read".to_owned(),
            expires_at_ms: 9_000,
        };
        fn query<'a>(
            protocol: &'a str,
            tool: &'a str,
            scope: &'a str,
            now_ms: u64,
        ) -> ConsentQuery<'a> {
            ConsentQuery {
                protocol_id: protocol,
                agent_instance: AgentInstanceId(1),
                tool,
                scope,
                now_ms,
            }
        }
        assert!(grant.covers(&query(
            "local.assistant",
            "workspace_read",
            "workspace.read",
            1_000
        )));
        // Triple must match exactly: any field differs denies.
        assert!(!grant.covers(&query(
            "other.assistant",
            "workspace_read",
            "workspace.read",
            1_000
        )));
        assert!(!grant.covers(&query(
            "local.assistant",
            "workspace_write",
            "workspace.read",
            1_000
        )));
        assert!(!grant.covers(&query(
            "local.assistant",
            "workspace_read",
            "workspace.write",
            1_000
        )));
        // Exclusive boundary: before allows, equality and after deny.
        assert!(grant.covers(&query(
            "local.assistant",
            "workspace_read",
            "workspace.read",
            8_999
        )));
        assert!(!grant.covers(&query(
            "local.assistant",
            "workspace_read",
            "workspace.read",
            9_000
        )));
        assert!(!grant.covers(&query(
            "local.assistant",
            "workspace_read",
            "workspace.read",
            9_001
        )));
    }

    #[test]
    fn consent_ttl_default_and_boundary() {
        assert_eq!(consent_grant_expiry(1_000, None), 121_000);
        assert_eq!(consent_grant_expiry(1_000, Some(60_000)), 61_000);
        assert_eq!(consent_grant_expiry(u64::MAX - 10, None), u64::MAX);

        // Deny-all seam denies every query; authorizer denies every call.
        let ledger = DenyAllConsent;
        let query = ConsentQuery {
            protocol_id: "local.assistant",
            agent_instance: AgentInstanceId(1),
            tool: "workspace_read",
            scope: "workspace.read",
            now_ms: 1_000,
        };
        assert!(matches!(ledger.check(&query), ConsentDecision::Deny { .. }));
        let authorizer = DenyAllAuthorizer;
        assert!(matches!(
            authorizer.authorize(&AuthContext {
                base: base(),
                tool: "workspace_read",
                required_scope: "workspace.read",
                read_only: true,
            }),
            AuthDecision::Deny { .. }
        ));

        // Consent denial maps to dispatch Denied, never other taxonomy.
        let denial = consent_denial("workspace_read", "no live consent");
        assert!(matches!(denial, ToolError::Denied { .. }));
        assert!(format!("{denial}").contains("consent denied:"));
    }

    #[test]
    fn usage_ring_evicts_oldest_and_filters() {
        let mut ring = UsageRing::new();
        assert!(ring.is_empty());
        for index in 0..(MAX_USAGE_EVENTS + 1) {
            let mut event = usage_event(&format!("tool_{index}"), index as u64);
            event.execution_id = ExecutionId(index as u64);
            ring.push(event);
        }
        assert_eq!(ring.len(), MAX_USAGE_EVENTS);
        let all = ring.list(&UsageFilter::match_all(), usize::MAX);
        assert_eq!(all.len(), MAX_USAGE_EVENTS);
        // Oldest evicted (`tool_0` gone), oldest-first order kept.
        assert_eq!(all.first().unwrap().tool, "tool_1");
        assert_eq!(
            all.last().unwrap().tool,
            format!("tool_{}", MAX_USAGE_EVENTS)
        );

        // Filter by tool narrows to one row; limit caps rows; zero returns none.
        let filter = UsageFilter {
            tool: Some("tool_1".to_owned()),
            ..UsageFilter::default()
        };
        let rows = ring.list(&filter, usize::MAX);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].tool, "tool_1");
        assert!(ring.list(&UsageFilter::match_all(), 0).is_empty());
        assert_eq!(ring.list(&UsageFilter::match_all(), 2).len(), 2);

        // Placement filter follows the derived site label.
        let filter = UsageFilter {
            placement: Some(ToolPlacement::Native),
            ..UsageFilter::default()
        };
        assert!(ring.list(&filter, usize::MAX).is_empty());
    }

    #[test]
    fn args_digest_is_deterministic_and_private() {
        let first = UsageEvent::args_digest(b"{}");
        assert_eq!(first, UsageEvent::args_digest(b"{}"));
        assert_ne!(first, UsageEvent::args_digest(b"{\"x\":1}"));
        // The ring stores the digest, never the raw bytes.
        let event = usage_event("tool_a", 1_000);
        assert_eq!(event.args_digest, first);
    }

    #[test]
    fn prompt_never_grants_dispatcher_first() {
        let allowed = vec!["tool_a".to_owned()];
        let denied: Vec<String> = Vec::new();
        // Dispatcher denial wins even when the prompt would allow.
        let err = check_dispatch(Some(&allowed), &denied, "tool_a", false).unwrap_err();
        assert!(matches!(err, PromptError::DispatcherDenied { .. }));
        // Prompt denial beats allow-set membership.
        let denied = vec!["tool_a".to_owned()];
        let err = check_dispatch(Some(&allowed), &denied, "tool_a", true).unwrap_err();
        assert!(matches!(err, PromptError::PromptDenied { .. }));
        // Allow-set omission denies even when the dispatcher grants.
        let allowed = vec!["tool_b".to_owned()];
        let err = check_dispatch(Some(&allowed), &[], "tool_a", true).unwrap_err();
        assert!(matches!(err, PromptError::PromptNotAllowed { .. }));
        // Granted, not denied, listed: allows.
        let allowed = vec!["tool_a".to_owned()];
        assert!(check_dispatch(Some(&allowed), &[], "tool_a", true).is_ok());
        // Unconstrained allow-set with no denial allows when granted.
        assert!(check_dispatch(None, &[], "tool_a", true).is_ok());
        assert!(check_dispatch(None, &[], "tool_a", false).is_err());

        // Denial cause keeps tool identity as dispatch Denied, never other axes.
        let cause = dispatch_denial_cause(
            "tool_a",
            &PromptError::DispatcherDenied {
                tool: "tool_a".to_owned(),
            },
        );
        assert!(matches!(cause, ToolError::Denied { .. }));
        assert!(!matches!(cause, ToolError::Failed { .. }));
        assert!(!matches!(cause, ToolError::EffectUnknown { .. }));
        assert!(!matches!(cause, ToolError::ProtocolRejected { .. }));
    }
}

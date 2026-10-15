//! Phase 2 S1 pure-type and pin mirror of the MCP-placement seam.
//!
//! Offline, `std`-only, no behavior: this module mirrors the placement
//! vocabulary, declaration shape, import provenance, snapshot/diff, and
//! protocol-pin vocabulary owned by `bitty-ai-runtime` (`tool.rs`) and
//! `bitty-ai-mcp` (`tools.rs`, `handshake.rs`, `error.rs`, plus the protocol
//! pin in `lib.rs`) so the host can re-impose every placement gate in order
//! when placement moves host-side. It grants nothing: there is no transport,
//! no spawn, no re-list, no execute, no IPC, no routing change, no ledger, no
//! credential or secret material, no SPKI fetch, no binary-hash execution, no
//! resources/prompts consumer, no TOFU or auto-update, no threads, no network,
//! and no filesystem access.
//!
//! Mirrored sources (read-only, values and wording kept byte-identical):
//!
//! - `tool.rs` placement vocabulary plus bounds: [`ToolPlacement`] (`mcp` /
//!   `native`, closed), [`ToolPlacement::parse`] / [`ToolPlacement::as_str`],
//!   [`NativePlacement`] (`reason` / `backend`), [`ToolSpec`] declaration
//!   shape with [`ToolSpec::new`] and [`ToolSpec::schema_digest`],
//!   `validate_placement` (`None` refuses, `Mcp` carrying native detail
//!   refuses as contradictory, `Native` requires non-empty reason/backend
//!   within [`MAX_NATIVE_PLACEMENT_BYTES`] bytes each), [`PlacementError`]
//!   as the registration-refusal subset (`UndeclaredPlacement` is never a
//!   dispatch denial), [`validate_tool_name`] (`TB-2` shape
//!   `^[a-z][a-z0-9_]*$`, non-empty, at most [`MAX_TOOL_NAME_LEN`] bytes),
//!   and the FNV-1a-64 precedent (`offset basis`, `prime`, byte-wise
//!   `schema_digest`).
//! - `tools.rs` import provenance plus snapshot/diff: [`MAX_TOOL_PAGES`] (5,
//!   paging bound; paging itself stays MCP-owned),
//!   [`sanitize_mcp_name`] (`mcp_<server>_<tool>`, lowercase,
//!   non-`[a-z0-9_]` to `_`, 64-byte cap with deterministic FNV-1a-64 hash
//!   suffix past the cap), [`RawMcpTool`] / [`ImportedTool`] shapes
//!   (`digest == spec.schema_digest()` at import time), [`import_tool`]
//!   (description truncates at [`MAX_TOOL_DESCRIPTION_LEN`] on a UTF-8
//!   boundary, schemas past [`MAX_TOOL_SCHEMA_BYTES`] fail the import,
//!   sanitized collisions are detected by the caller across the whole import,
//!   every imported spec carries `read_only = true`, scope `mcp.<server>`,
//!   and placement [`ToolPlacement::Mcp`] by construction),
//!   [`ToolListSnapshot`] (`from_imported` sorted by name, `entries`,
//!   `digest_of`, `len` / `is_empty`) and [`ToolListDiff`] (`added` /
//!   `removed` / `changed`, all sorted by name) plus
//!   [`diff_tool_snapshots`] (pure, no I/O; empty diff means the re-list
//!   observed the same digests under the same names).
//! - `handshake.rs` plus the `lib.rs` protocol pin: [`PROTOCOL_VERSION`]
//!   (`2025-11-25`), [`ToolsCapabilityShape`] (`BoolTrue` vs
//!   `Object { list_changed }`), [`InitializeResult`] (`protocol_version` +
//!   `tools_supported` + `tools_shape`) with [`InitializeResult::matches_pin`]
//!   (version plus full shape; downgrade, upgrade, and version drift report
//!   false; a server-binary swap with identical protocol state still reports
//!   true — binary identity is out of scope, `std`-only pin).
//! - `error.rs` names-only taxonomy subset: [`MAX_ERROR_TEXT_BYTES`] (512),
//!   [`MAX_ERROR_NAME_BYTES`] (128), [`bound_error_text`] /
//!   [`bound_error_name`] (printable-ASCII scrub plus silent truncation,
//!   mirroring the runtime `bound_reason` precedent), [`McpStage`] (placement
//!   subset: `Handshake` / `ListTools` / `Bridge`), [`McpFailure`]
//!   (placement subset: `HandshakeRejected`, `VersionMismatch`,
//!   `NoToolCapability`, `CapabilityDrift`, `IdentityMismatch`,
//!   `DuplicateTool`, `UnknownTool`, `DuplicateCursor`, `TooManyPages`),
//!   [`McpFailure::retryable`] (all mirrored variants are fatal; the
//!   retryable set — timeouts, closed transports, spawn failures, unknown
//!   effects, I/O — stays MCP-owned and is never produced here), and the
//!   `Display` strings (names, bounds, and counts only — never secret values,
//!   payload bytes, path bytes, hash bytes, or helper output).
//!
//! Rank-2 binary-digest note (type comment only, no code): the pre-configured
//! `sha256` pin on `McpServerConfig`, the streamed `hash_binary` verification
//! (64 KiB buffer, 8 MiB cap, never-load-never-truncate), and the pre-spawn
//! `verify_pin` gate stay `supervise.rs` / `config.rs` owned. This module
//! carries the [`McpFailure::IdentityMismatch`] shape (server id only) so the
//! taxonomy documents the Rank-1 / Rank-2 separation, but it never reads a
//! binary, never hashes, and never spawns. The TOCTOU window between hash and
//! exec and the surfaced-loss residual past the notification cap are accepted
//! by design MCP-side and unreconciled host-side (per AI-0269 G5).
//!
//! Explicitly not in scope (stays runtime/MCP/host-design owned): transport
//! (`McpTransport`, `send_line` / `recv_line`, HTTP/SSE polling, shutdown),
//! spawn and supervision (`McpHost` roster, cooldown, `resolve_env_refs`,
//! `run_credential_cmd`, `spawn_server`, drain worker), re-list and staleness
//! (`list_version` / `is_stale` / `mark_stale` / `observe_notification` /
//! `relist` / pin revalidation, allowlist lookup, tier/authorizer/consent
//! gates, bounded dispatch), execute paths (`execute_narrowed` /
//! `execute_inner`, call-tool mapping), scoped-IPC envelope types (S-2,
//! parked on the host `Cargo.toml` IPC review plus CTX-0147 Phase 2
//! (S-1 adopted 2026-10-15, bitty-ai-docs#277)), placement-policy admission (S-3, parked on
//! policy design plus security review and AIQ-36 acceptance), SPKI fetch and
//! peer-key providers (real provider stays slice-gated), credential helpers,
//! resources/prompts consumers (classifiers stay symmetry-only, no version,
//! no stale, no re-list until a consumer exists), TOFU or auto-update, and
//! any thread, network, or filesystem use. `Cargo.toml` stays `std`-only.
//!
//! Deny-all posture holds by construction: placement declarations refuse
//! closed on any missing, mismatched, contradictory, or over-long input with
//! no partial state; imports fail closed on over-bound schemas; snapshots
//! refuse to smuggle replacements (the registry keeps refusing same-name
//! re-registration, so digests are refusal-only drift handles); pin
//! mismatches keep the old snapshot stale and deny via the existing
//! deny-stale gate (documented, not built here); and every diagnostic is
//! names-only and bounded so `Display` output is safe to log directly.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Maximum tool name length in bytes (`TB-2`, mirrors `tool.rs`).
pub const MAX_TOOL_NAME_LEN: usize = 64;
/// Maximum tool description length in bytes (`TB-2`, mirrors `tool.rs`).
///
/// Import truncates past this bound on a UTF-8 boundary (display-only, no
/// validation impact); registration refuses past it.
pub const MAX_TOOL_DESCRIPTION_LEN: usize = 512;
/// Maximum tool JSON Schema length in bytes (`TB-2`, mirrors `tool.rs`).
///
/// Import fails the whole entry past this bound (schemas shape validation,
/// so a partial import would lie); registration refuses past it.
pub const MAX_TOOL_SCHEMA_BYTES: usize = 16 * 1024;
/// Maximum native placement reason/backend length in bytes (AIQ-33 S3,
/// mirrors `tool.rs`).
///
/// Mirrors [`MAX_TOOL_DESCRIPTION_LEN`]: short owner-supplied opaque strings
/// share the 512-byte inclusive bound (`len() > LIMIT` refuses).
pub const MAX_NATIVE_PLACEMENT_BYTES: usize = 512;
/// Maximum bytes kept for any placement reason string (mirrors the runtime
/// `MAX_REASON_BYTES`).
pub const MAX_REASON_BYTES: usize = 512;
/// Maximum bytes kept for any string carried by [`McpError`] (mirrors
/// `error.rs`).
pub const MAX_ERROR_TEXT_BYTES: usize = 512;
/// Maximum bytes kept for a name echoed into [`McpError`] (mirrors
/// `error.rs`).
pub const MAX_ERROR_NAME_BYTES: usize = 128;
/// Maximum `tools/list` pages followed per import (mirrors `tools.rs`).
///
/// Paging bound only: paging itself stays MCP-owned. The host agrees on the
/// bound so a future host-side re-list envelope cannot silently widen it.
pub const MAX_TOOL_PAGES: usize = 5;
/// Pinned MCP protocol version negotiated at `initialize` (mirrors `lib.rs`).
///
/// Strict pin: anything else fails closed with
/// [`McpFailure::VersionMismatch`].
pub const PROTOCOL_VERSION: &str = "2025-11-25";

/// FNV-1a-64 offset basis (deterministic across processes and platforms;
/// mirrors the `tool.rs` / `tools.rs` precedent).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (deterministic across processes and platforms; mirrors
/// the `tool.rs` / `tools.rs` precedent).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Deterministic 64-bit FNV-1a hash (runtime `schema_digest` precedent).
///
/// Never `DefaultHasher` / `RandomState`: same bytes give the same digest on
/// every platform and process; any byte change moves it.
fn fnv_1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Bound an outbound reason string.
///
/// Every character outside printable ASCII (`0x20..=0x7E`, spaces kept)
/// becomes `?` (defuses CR/LF log injection, terminal escapes, and
/// confusable non-ASCII bytes); the result is truncated to
/// [`MAX_REASON_BYTES`] bytes. Deterministic, silent (no ellipsis marker),
/// and never splits a character, so callers can log the value directly.
#[must_use]
pub fn bound_reason(reason: &str) -> String {
    bound_error_text(reason, MAX_REASON_BYTES)
}

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

/// Validate a tool name (`TB-2`, mirrors `tool.rs`): non-empty, at most 64
/// bytes, `^[a-z][a-z0-9_]*$` within the owner namespace.
///
/// # Errors
///
/// Returns [`PlacementError::InvalidName`] when the shape is violated.
pub fn validate_tool_name(name: &str) -> Result<(), PlacementError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LEN
        && name.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if valid {
        Ok(())
    } else {
        // The echoed malformed name is host/model-supplied, so it is bounded
        // and scrubbed at this conversion boundary like every other outbound
        // diagnostic string.
        Err(PlacementError::InvalidName {
            name: bound_reason(name),
        })
    }
}

/// Closed tool-placement vocabulary (AIQ-33 S3, owner-approved, mirrors
/// `tool.rs`).
///
/// Data-only declaration: routing is unchanged — `Mcp` marks tools imported
/// over MCP (the import path carries it by construction); `Native` marks
/// host-native tools and additionally requires [`NativePlacement`]
/// reason/backend. Closed: unknown strings never synthesize a placement
/// ([`ToolPlacement::parse`] fails closed).
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
    /// Returns [`PlacementError::UndeclaredPlacement`] carrying the rejected
    /// (bounded, scrubbed) string for anything else.
    pub fn parse(value: &str) -> Result<Self, PlacementError> {
        match value {
            "mcp" => Ok(Self::Mcp),
            "native" => Ok(Self::Native),
            // The rejected string is externally supplied, so it is bounded
            // and scrubbed at this conversion boundary like every other
            // outbound diagnostic string.
            _ => Err(PlacementError::UndeclaredPlacement {
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

/// Native-placement reason/backend (AIQ-33 S3, mirrors `tool.rs`).
///
/// Data-only: stored on the declaration, never interpreted — dispatch
/// routing is unchanged. Required exactly when [`ToolSpec::placement`] is
/// [`ToolPlacement::Native`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlacement {
    /// Why this tool is host-native (owner-supplied, opaque).
    pub reason: String,
    /// Which native backend serves it (owner-supplied label, opaque).
    pub backend: String,
}

/// Placement declaration errors: the registration-refusal subset of the
/// runtime `ToolError` taxonomy (mirrors `tool.rs`).
///
/// Every variant fails closed with no partial state. `UndeclaredPlacement`
/// is a registration refusal (same family as `DuplicateTool` /
/// `RegistryFull` in the full taxonomy): never a dispatch-time denial, which
/// this subset does not declare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    /// Tool name violates the `TB-2` shape.
    InvalidName {
        /// Rejected name.
        name: String,
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
    /// Registration refusal: never a dispatch-time denial.
    UndeclaredPlacement {
        /// Rejected tool name (registration boundary) or rejected placement
        /// string (vocabulary parse boundary).
        name: String,
    },
}

impl Display for PlacementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidName { name } => write!(f, "invalid tool name: {name}"),
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
        }
    }
}

impl std::error::Error for PlacementError {}

impl PlacementError {
    /// Apply the one bounded diagnostic policy to every host-supplied string
    /// carried by this error, returning the error unchanged when it already
    /// fits (mirrors the runtime `ToolError::normalized` precedent).
    ///
    /// The runtime-owned numeric/typed fields (bounds, limits, counts) are
    /// preserved verbatim; only externally sourced text — tool names and the
    /// rejected placement string — is scrubbed to printable ASCII and
    /// bounded to [`MAX_REASON_BYTES`].
    #[must_use]
    pub fn normalized(self) -> Self {
        match self {
            Self::InvalidName { name } => Self::InvalidName {
                name: bound_reason(&name),
            },
            Self::UndeclaredPlacement { name } => Self::UndeclaredPlacement {
                name: bound_reason(&name),
            },
            Self::DescriptionTooLarge { .. } | Self::SchemaTooLarge { .. } => self,
        }
    }
}

/// A bounded tool declaration (`TB-2`, mirrors `tool.rs`).
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
    /// Whether the tool is side-effect free. The default import profile
    /// allows only these at the `inspect` tier (read-only by default).
    pub read_only: bool,
    /// Declared placement (AIQ-33 S3, refuse-undeclared): `None` is
    /// undeclared and refused fail-closed with
    /// [`PlacementError::UndeclaredPlacement`].
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
    /// Returns [`PlacementError`] for malformed names, over-bound
    /// description and schema, undeclared placement, or a
    /// native/native-detail mismatch (native without reason/backend, or MCP
    /// carrying native detail).
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema_json: Vec<u8>,
        required_scope: impl Into<String>,
        read_only: bool,
        placement: ToolPlacement,
        native: Option<NativePlacement>,
    ) -> Result<Self, PlacementError> {
        let name = name.into();
        let description = description.into();
        validate_tool_name(&name)?;
        if description.len() > MAX_TOOL_DESCRIPTION_LEN {
            return Err(PlacementError::DescriptionTooLarge {
                limit: MAX_TOOL_DESCRIPTION_LEN,
                actual: description.len(),
            });
        }
        if schema_json.len() > MAX_TOOL_SCHEMA_BYTES {
            return Err(PlacementError::SchemaTooLarge {
                limit: MAX_TOOL_SCHEMA_BYTES,
                actual: schema_json.len(),
            });
        }
        // The name is shape-checked above, so the refusal carries it raw
        // (the `DuplicateTool` precedent in the full taxonomy).
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

    /// Deterministic identity of [`ToolSpec::schema_json`] (AI-0090, AIQ-08
    /// narrowing input, mirrors `tool.rs`): FNV-1a-64 over exactly the schema
    /// bytes (inline small hasher; never `DefaultHasher` / `RandomState`).
    /// Same bytes give the same digest on every platform and process; any
    /// byte change moves it. The digest is the host-side invalidation handle:
    /// the registry keeps refusing same-name re-registration, so the host
    /// compares digests across its schema snapshots to detect drift and never
    /// relies on the digest to smuggle a replacement (refusal-only stands).
    #[must_use]
    pub fn schema_digest(&self) -> u64 {
        fnv_1a_64(&self.schema_json)
    }
}

/// Validate a placement declaration (AIQ-33 S3, refuse-undeclared; mirrors
/// `tool.rs`).
///
/// `None` is undeclared; `Native` requires a non-empty reason/backend within
/// [`MAX_NATIVE_PLACEMENT_BYTES`] bytes each (inclusive bound, the
/// [`MAX_TOOL_DESCRIPTION_LEN`] convention); `Mcp` carrying native detail is
/// a contradictory declaration. Every violation fails closed with
/// [`PlacementError::UndeclaredPlacement`]. The caller shape-checks `name`
/// first (as [`ToolSpec::new`] does), so the refusal carries it raw.
///
/// # Errors
///
/// Returns [`PlacementError::UndeclaredPlacement`] for a missing, mismatched,
/// contradictory, or over-long declaration.
fn validate_placement(
    name: &str,
    placement: Option<ToolPlacement>,
    native: Option<&NativePlacement>,
) -> Result<(), PlacementError> {
    let refused = || PlacementError::UndeclaredPlacement {
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

/// Pipeline stage that produced the error: the placement-relevant subset of
/// the `error.rs` taxonomy (mirrors `McpStage`).
///
/// Configuration, spawn, credential, frame, supervision, and call-tool stages
/// stay MCP-owned; only the handshake pin, list import, and adapter-policy
/// stages touch placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpStage {
    /// `initialize` handshake.
    Handshake,
    /// `tools/list` import.
    ListTools,
    /// Adapter policy gates (pin comparison, stale discipline).
    Bridge,
}

impl Display for McpStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let label = match self {
            Self::Handshake => "handshake",
            Self::ListTools => "list-tools",
            Self::Bridge => "bridge",
        };
        f.write_str(label)
    }
}

/// Typed MCP failure: the placement-relevant subset of the `error.rs`
/// taxonomy (mirrors `McpFailure`).
///
/// Every string field holds a bounded, scrubbed name or a caller-supplied
/// reason that itself must be names-only (callers never interpolate values,
/// payloads, or helper output). Transport, spawn, credential, supervision,
/// dispatch, and I/O failures stay MCP-owned and have no variant here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpFailure {
    /// Handshake or import response was not acceptable.
    HandshakeRejected {
        /// Bounded detail (protocol facts only).
        detail: String,
    },
    /// Server protocol version differs from the [`PROTOCOL_VERSION`] pin.
    VersionMismatch {
        /// Expected pin.
        expected: String,
        /// Server-reported version (bounded).
        got: String,
    },
    /// Server offers no tool capability.
    NoToolCapability,
    /// Pinned protocol state drifted (AI-0244 capability-set pin).
    ///
    /// The stored [`InitializeResult`] (negotiated `protocolVersion` plus
    /// the `tools`-capability shape) no longer matches a fresh `initialize`
    /// answer: a capability downgrade/upgrade or a version drift.
    /// Fail-closed, non-retryable. This catches protocol-state drift only,
    /// NOT a server-binary swap (no binary digest, no hash, `std`-only).
    CapabilityDrift {
        /// Bounded detail (protocol facts only, never payloads).
        detail: String,
    },
    /// Server binary digest pin mismatch (AI-0244 Rank-2 pin shape only).
    ///
    /// Carries the server id only — never path bytes, never hash bytes. No
    /// verification runs here; the shape exists so the taxonomy documents
    /// the Rank-1 / Rank-2 separation.
    IdentityMismatch {
        /// Server id only.
        server: String,
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
    /// A pagination cursor repeated (looping server).
    DuplicateCursor {
        /// Bounded cursor echo (opaque token, truncated).
        cursor: String,
    },
    /// `tools/list` pagination exceeded [`MAX_TOOL_PAGES`].
    TooManyPages {
        /// Bound.
        limit: usize,
    },
}

impl McpFailure {
    /// Whether the failure may resolve on retry.
    ///
    /// Every mirrored placement variant is fatal (`false`): validation,
    /// pin, identity, bound, duplicate, and unknown-tool failures never
    /// resolve by retrying the same input. The retryable set (timeouts,
    /// closed transports, spawn failures, unknown effects, I/O) stays
    /// MCP-owned and has no variant here, so it is never produced from this
    /// module.
    #[must_use]
    pub fn retryable(&self) -> bool {
        false
    }
}

impl Display for McpFailure {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::HandshakeRejected { detail } => write!(f, "handshake rejected: {detail}"),
            Self::VersionMismatch { expected, got } => {
                write!(
                    f,
                    "protocol version mismatch: expected '{expected}', got '{got}'"
                )
            }
            Self::NoToolCapability => write!(f, "server offers no tool capability"),
            Self::CapabilityDrift { detail } => write!(f, "capability drift: {detail}"),
            Self::IdentityMismatch { server } => write!(
                f,
                "server identity mismatch: '{server}' binary digest does not match pin"
            ),
            Self::DuplicateCursor { cursor } => {
                write!(f, "tool listing repeated cursor '{cursor}'")
            }
            Self::DuplicateTool { name } => write!(f, "duplicate tool: {name}"),
            Self::UnknownTool { name } => write!(f, "unknown tool: {name}"),
            Self::TooManyPages { limit } => {
                write!(f, "tool listing exceeds {limit} pages")
            }
        }
    }
}

/// Bounded names-only MCP error: stage plus typed failure plus retryability
/// (mirrors `error.rs`).
///
/// Construct through [`McpError::new`] so `retryable` derives from the
/// failure; `Display` renders `mcp <stage>: <failure>`.
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
}

impl Display for McpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "mcp {}: {}", self.stage, self.failure)
    }
}

impl std::error::Error for McpError {}

/// Shape of the negotiated `tools` capability (AI-0244 pin, mirrors
/// `handshake.rs`).
///
/// Distinguishes `"tools": true` from `"tools": { ... }` and records the
/// `listChanged` flag when present. A server that upgrades (`{}` to
/// `{"listChanged":true}`) or downgrades (`{"listChanged":true}` to `{}` or
/// to `true`) changes this shape: the pin catches it even though every shape
/// is handshake-valid on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolsCapabilityShape {
    /// `"tools": true`.
    BoolTrue,
    /// `"tools": { ... }`, with the `listChanged` flag when present.
    Object {
        /// Value of `listChanged` inside the tools object, if any.
        list_changed: Option<bool>,
    },
}

/// Parsed `initialize` result: the facts the client requires before use
/// (mirrors `handshake.rs`).
///
/// AI-0244 capability-set / protocol-state pin: the adapter stores this whole
/// value at construction/handshake and compares it on every re-list success.
/// A mismatch keeps the old snapshot stale and denies via the existing
/// deny-stale gate. The pin catches capability downgrade/upgrade plus
/// version drift, NOT a server-binary swap (no binary digest, no hash,
/// `std`-only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializeResult {
    /// Server protocol version (always the pin; anything else is refused).
    pub protocol_version: String,
    /// Whether the server offers the `tools` capability (always true here;
    /// absence is refused before this returns).
    pub tools_supported: bool,
    /// Negotiated `tools`-capability shape (bool vs object, `listChanged`).
    pub tools_shape: ToolsCapabilityShape,
}

impl InitializeResult {
    /// Whether `other` matches this pin (protocol version plus tools shape).
    ///
    /// Pure, no I/O. True only when the version and the full
    /// [`ToolsCapabilityShape`] agree. Version drift and any
    /// capability-shape move (downgrade or upgrade) report false; a
    /// server-binary swap with identical protocol state still reports true
    /// (binary identity is out of scope, `std`-only pin).
    #[must_use]
    pub fn matches_pin(&self, other: &Self) -> bool {
        self.protocol_version == other.protocol_version
            && self.tools_supported == other.tools_supported
            && self.tools_shape == other.tools_shape
    }
}

/// One raw tool entry from a `tools/list` page (mirrors `tools.rs`).
///
/// Server-reported pre-sanitization shape: name plus description plus raw
/// `inputSchema` object bytes. Pure data; parsing stays MCP-owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawMcpTool {
    /// Server-reported name (pre-sanitization).
    pub name: String,
    /// Server-reported description (may be empty).
    pub description: String,
    /// Raw `inputSchema` object bytes (`{}` when the server sends none).
    pub schema_json: Vec<u8>,
}

/// One imported tool: a validated spec plus provenance (mirrors `tools.rs`).
///
/// `digest` equals `spec.schema_digest()` at import time; the host compares
/// digests across imports to detect server-side drift (the registry keeps
/// refusing same-name re-registration, so drift never smuggles a
/// replacement).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedTool {
    /// Validated spec under the sanitized name.
    pub spec: ToolSpec,
    /// `spec.schema_digest()` at import time.
    pub digest: u64,
    /// Owning server id.
    pub server_id: String,
    /// Server-reported raw name.
    pub raw_name: String,
}

/// Sanitize an MCP tool name into the `TB-2` registered shape (mirrors
/// `tools.rs`).
///
/// Builds `mcp_<server>_<tool>`, lowercases it, and maps every byte outside
/// `[a-z0-9_]` to `_`. Results longer than 64 bytes keep the first 55 bytes
/// plus `_` plus 8 lowercase hex digits of FNV-1a-64 over the full
/// candidate, so the cap is deterministic and collision-resistant. The
/// `mcp_` prefix guarantees the leading-lowercase rule.
#[must_use]
pub fn sanitize_mcp_name(server_id: &str, raw_tool: &str) -> String {
    fn clean(value: &str) -> String {
        value
            .bytes()
            .map(|byte| {
                let lower = byte.to_ascii_lowercase();
                if lower.is_ascii_lowercase() || lower.is_ascii_digit() || lower == b'_' {
                    lower as char
                } else {
                    '_'
                }
            })
            .collect()
    }
    let candidate = format!("mcp_{}_{}", clean(server_id), clean(raw_tool));
    if candidate.len() <= MAX_TOOL_NAME_LEN && validate_tool_name(&candidate).is_ok() {
        return candidate;
    }
    let digest = fnv_1a_64(candidate.as_bytes());
    let keep = MAX_TOOL_NAME_LEN - 1 - 8;
    let prefix = candidate.get(..keep).unwrap_or(&candidate);
    // Low 32 bits render as exactly 8 hex digits (`:08x` pads short
    // values but never truncates long ones, so the full 64-bit digest
    // would overflow the cap).
    let suffix = (digest & 0xffff_ffff) as u32;
    format!("{prefix}_{suffix:08x}")
}

/// Truncate `value` to `max_bytes` on a UTF-8 boundary (mirrors `tools.rs`).
fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

/// Convert one raw entry into an [`ImportedTool`], enforcing [`ToolSpec`]
/// bounds (mirrors `tools.rs`).
///
/// Descriptions past 512 bytes truncate on a UTF-8 boundary; schemas past
/// 16 KiB fail; sanitized collisions are detected by the caller across the
/// whole import (this function only guarantees the spec itself validates).
/// Imported specs default to `read_only = true` (least privilege: MCP access
/// is read-only by default); the host may loosen individual specs before
/// registry insert under its own consent and tier policy.
///
/// # Errors
///
/// Returns [`McpFailure::HandshakeRejected`]-shaped list errors for bound
/// and validation failures (the closest typed failure in this subset: the
/// entry is not a usable import).
pub fn import_tool(server_id: &str, raw: &RawMcpTool) -> Result<ImportedTool, McpError> {
    if raw.schema_json.len() > MAX_TOOL_SCHEMA_BYTES {
        return Err(McpError::new(
            McpStage::ListTools,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    &format!("tool '{}' schema exceeds 16384 bytes", raw.name),
                    MAX_ERROR_TEXT_BYTES,
                ),
            },
        ));
    }
    let description = truncate_utf8(&raw.description, MAX_TOOL_DESCRIPTION_LEN);
    let sanitized = sanitize_mcp_name(server_id, &raw.name);
    let spec = ToolSpec::new(
        sanitized.clone(),
        description,
        raw.schema_json.clone(),
        format!("mcp.{server_id}"),
        true,
        ToolPlacement::Mcp,
        None,
    )
    .map_err(|error| {
        McpError::new(
            McpStage::ListTools,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    &format!("tool '{}' rejected: {error}", raw.name),
                    MAX_ERROR_TEXT_BYTES,
                ),
            },
        )
    })?;
    let digest = spec.schema_digest();
    Ok(ImportedTool {
        spec,
        digest,
        server_id: server_id.to_owned(),
        raw_name: raw.name.clone(),
    })
}

/// Immutable host-side digest snapshot of one server's imported tool list
/// (mirrors `tools.rs`, AI-0211 diff handle).
///
/// Holds `(sanitized name, schema digest)` pairs sorted by name, built from
/// [`ImportedTool`] slices via [`ToolListSnapshot::from_imported`]. The
/// digest is [`ToolSpec::schema_digest`] at import time: same bytes give the
/// same digest, any byte moves it. The host compares snapshots across an
/// explicit re-list to detect drift; the registry keeps refusing same-name
/// re-registration, so the digest never smuggles a replacement
/// (refusal-only stands).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolListSnapshot {
    entries: Vec<(String, u64)>,
}

impl ToolListSnapshot {
    /// Build a snapshot from imported tools, sorted by sanitized name for
    /// deterministic diffs.
    #[must_use]
    pub fn from_imported(tools: &[ImportedTool]) -> Self {
        let mut entries: Vec<(String, u64)> = tools
            .iter()
            .map(|entry| (entry.spec.name.clone(), entry.digest))
            .collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        Self { entries }
    }

    /// Snapshot entries as `(sanitized name, digest)` pairs in name order.
    #[must_use]
    pub fn entries(&self) -> &[(String, u64)] {
        &self.entries
    }

    /// Digest for `name`, when present.
    #[must_use]
    pub fn digest_of(&self, name: &str) -> Option<u64> {
        self.entries
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, digest)| *digest)
    }

    /// Entry count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the snapshot holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Pure diff between two [`ToolListSnapshot`] values (mirrors `tools.rs`,
/// AI-0211 re-list handle).
///
/// `added` holds names only in `new`, `removed` only in `old`, `changed`
/// holds names in both with different digests. All three lists sort by name.
/// Empty diff means the re-list observed the same digests under the same
/// names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolListDiff {
    /// Names only in the new snapshot.
    pub added: Vec<String>,
    /// Names only in the old snapshot.
    pub removed: Vec<String>,
    /// Names in both with different digests.
    pub changed: Vec<String>,
}

impl ToolListDiff {
    /// Whether the diff reports no change.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Diff `old` against `new` (pure, no I/O; mirrors `tools.rs`).
#[must_use]
pub fn diff_tool_snapshots(old: &ToolListSnapshot, new: &ToolListSnapshot) -> ToolListDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    for (name, old_digest) in old.entries() {
        match new.digest_of(name) {
            None => removed.push(name.clone()),
            Some(new_digest) if new_digest != *old_digest => changed.push(name.clone()),
            Some(_) => {}
        }
    }
    for (name, _) in new.entries() {
        if old.digest_of(name).is_none() {
            added.push(name.clone());
        }
    }
    added.sort();
    removed.sort();
    changed.sort();
    ToolListDiff {
        added,
        removed,
        changed,
    }
}

// ── Phase 2 S2 scoped-IPC envelope types (no transport, AI-0284) ──
//
// Scoped-IPC pin/snapshot shapes for the MCP-placement seam so a future
// host-side caller can cross the boundary without touching the registry,
// adapter, or transport directly. Types-only, `std`-only, no behavior: no
// socket, no transport, no thread, no clock, no network, no filesystem, no
// dependency change (`Cargo.toml` stays `std`-only), no new byte bounds (the
// S1 [`MAX_TOOL_NAME_LEN`], [`MAX_TOOL_DESCRIPTION_LEN`],
// [`MAX_TOOL_SCHEMA_BYTES`], [`MAX_NATIVE_PLACEMENT_BYTES`],
// [`MAX_REASON_BYTES`], [`MAX_ERROR_TEXT_BYTES`], [`MAX_ERROR_NAME_BYTES`],
// [`MAX_TOOL_PAGES`], and [`PROTOCOL_VERSION`] values apply unchanged), no
// version field on any envelope, and no placement execution. Every envelope
// reuses the S1 mirror above ([`InitializeResult`],
// [`ToolsCapabilityShape`], [`ToolListSnapshot`], [`ToolListDiff`],
// [`ImportedTool`], [`PlacementError`], [`McpError`], [`McpStage`],
// [`McpFailure`], [`bound_error_text`], [`diff_tool_snapshots`]) and
// introduces no re-mirrored vocabulary.
//
// Gate and mapping summary (re-stated, not re-implemented): the pin envelope
// echoes the negotiated [`InitializeResult`] (protocol version plus full
// tools-capability shape); shape validation checks the version pin plus the
// tools-capability presence while any capability-shape move (downgrade or
// upgrade) is caught by pin comparison only; the snapshot envelope echoes
// the [`ToolListSnapshot`] digest pairs plus the `list_version` / `list_stale`
// flags as data with no enforcement (matching, staleness discipline, and
// re-list stay MCP-owned); every [`PlacementError`] maps to the typed
// [`McpError`] envelope form via [`placement_to_mcp_error`] (total over the
// four declaration-refusal variants, fatal-only, names-only bounded detail).
//
// No placement decision or routing is modeled here (the S3 deny-all stub in
// [`PlacementStub`] denies without routing; see the S3 section below):
// envelopes carry pins and snapshots only — never an admit/deny verdict,
// never a dispatch route, never a registry mutation. Rank-2 binary identity
// stays out of scope: the envelope carries no path bytes and no hash bytes
// (the [`McpFailure::IdentityMismatch`] server-id-only shape documents the
// separation, same as S1). SPKI fetch, peer-key providers, spawn,
// supervision, credential helpers, resources/prompts consumers, TOFU, and
// auto-update stay out of scope, same as S1.
//
// Shared S2 convention: types-only, `std`-only, no socket/transport/dep
// change, NO version field on any envelope (envelope versioning is a
// host-design concern, never a silent field here), no behavior beyond pure
// shape checks and pure error mapping, no thread/clock/net/fs.

/// Scoped-IPC pin envelope: echoes the S1 [`InitializeResult`] pin.
///
/// Data-only carrier for the negotiated protocol facts (protocol version
/// plus tools-capability shape). Construction performs no I/O and grants
/// nothing; comparison and shape validation are pure. No version field, no
/// transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinEnvelope {
    /// Negotiated pin echoed across the boundary (S1 shape, reused).
    pub pin: InitializeResult,
}

impl PinEnvelope {
    /// Build a pin envelope echoing `pin`. No validation runs here: shape
    /// checks stay in [`PinEnvelope::validate`] and drift checks in
    /// [`PinEnvelope::matches`].
    #[must_use]
    pub fn new(pin: InitializeResult) -> Self {
        Self { pin }
    }

    /// Borrow the echoed pin.
    #[must_use]
    pub fn pin(&self) -> &InitializeResult {
        &self.pin
    }

    /// Whether `fresh` matches the echoed pin (protocol version plus full
    /// [`ToolsCapabilityShape`).
    ///
    /// Pure delegate to [`InitializeResult::matches_pin`]: version drift and
    /// any capability-shape move (downgrade or upgrade) report `false`; a
    /// server-binary swap with identical protocol state still reports `true`
    /// (binary identity is out of scope, `std`-only pin). No enforcement
    /// runs here; the caller decides staleness.
    #[must_use]
    pub fn matches(&self, fresh: &InitializeResult) -> bool {
        self.pin.matches_pin(fresh)
    }

    /// Fail-closed pin shape validation: the echoed pin must carry the
    /// [`PROTOCOL_VERSION`] pin and must offer the tools capability.
    ///
    /// Any [`ToolsCapabilityShape`] is shape-valid on its own (bool vs
    /// object, any `listChanged` flag); shape moves are drift, caught by
    /// [`PinEnvelope::matches`], not here. Pure, deterministic, offline.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::VersionMismatch`] when the echoed protocol
    /// version differs from the pin, and [`McpFailure::NoToolCapability`]
    /// when the echoed pin offers no tools capability.
    pub fn validate(&self) -> Result<(), McpError> {
        if self.pin.protocol_version != PROTOCOL_VERSION {
            return Err(McpError::new(
                McpStage::Handshake,
                McpFailure::VersionMismatch {
                    expected: PROTOCOL_VERSION.to_owned(),
                    got: bound_error_text(&self.pin.protocol_version, MAX_ERROR_NAME_BYTES),
                },
            ));
        }
        if !self.pin.tools_supported {
            return Err(McpError::new(
                McpStage::Handshake,
                McpFailure::NoToolCapability,
            ));
        }
        Ok(())
    }
}

/// Scoped-IPC snapshot envelope: echoes the S1 [`ToolListSnapshot`] digest
/// pairs plus the `list_version` / `list_stale` flags as data.
///
/// Data-only carrier: `snapshot` holds the `(sanitized name, schema digest)`
/// pairs in name order, `list_version` echoes the monotonic MCP list version
/// (starts at `0`, saturating bumps MCP-side), and `list_stale` echoes the
/// deny-stale flag. Both flags travel verbatim with no enforcement here:
/// a stale envelope is still well-formed data, a version mismatch is still
/// well-formed data, and re-list stays the sole MCP-owned un-staling path.
/// No version field on the envelope itself, no transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotEnvelope {
    /// Digest snapshot echoed across the boundary (S1 shape, reused).
    pub snapshot: ToolListSnapshot,
    /// Echoed MCP list version (data only, no monotonic check here).
    pub list_version: u64,
    /// Echoed deny-stale flag (data only, no deny here).
    pub list_stale: bool,
}

impl SnapshotEnvelope {
    /// Build a snapshot envelope echoing `snapshot` plus the version/stale
    /// flags. No enforcement runs here: any flag combination is well-formed
    /// data.
    #[must_use]
    pub fn new(snapshot: ToolListSnapshot, list_version: u64, list_stale: bool) -> Self {
        Self {
            snapshot,
            list_version,
            list_stale,
        }
    }

    /// Build a snapshot envelope from imported tools, echoing the given
    /// version/stale flags. Pure delegate to
    /// [`ToolListSnapshot::from_imported`]; sorting and digest rules apply
    /// unchanged.
    #[must_use]
    pub fn from_imported(tools: &[ImportedTool], list_version: u64, list_stale: bool) -> Self {
        Self {
            snapshot: ToolListSnapshot::from_imported(tools),
            list_version,
            list_stale,
        }
    }

    /// Borrow the echoed snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &ToolListSnapshot {
        &self.snapshot
    }

    /// Echoed list version (data only).
    #[must_use]
    pub fn list_version(&self) -> u64 {
        self.list_version
    }

    /// Echoed stale flag (data only, never a deny here).
    #[must_use]
    pub fn is_stale(&self) -> bool {
        self.list_stale
    }
}

/// Map a [`PlacementError`] declaration refusal to the typed [`McpError`]
/// envelope form.
///
/// Uniform single-path mapping over all S1 refusal variants via [`Display`]:
/// coverage is test-enforced (not compiler-enforced), so classify a new
/// refusal text before any envelope carries it. Every mapping reports [`McpStage::ListTools`]
/// with [`McpFailure::HandshakeRejected`] carrying the bounded, scrubbed
/// refusal text (mirroring the S1 [`import_tool`] precedent, which maps
/// [`ToolSpec`] bound and validation failures to the same typed failure):
/// the entry is not a usable import. Counts and bounds cross verbatim;
/// names and rejected strings cross via the one bounded diagnostic policy
/// ([`bound_error_text`] to [`MAX_ERROR_TEXT_BYTES`], printable ASCII only),
/// so the mapped error is safe to log directly. Every mapped error is fatal
/// (`retryable` is `false`); transport, spawn, credential, supervision,
/// dispatch, and I/O failures have no variant here and are never produced
/// from this mapping.
#[must_use]
pub fn placement_to_mcp_error(error: &PlacementError) -> McpError {
    let detail = bound_error_text(&error.to_string(), MAX_ERROR_TEXT_BYTES);
    McpError::new(
        McpStage::ListTools,
        McpFailure::HandshakeRejected { detail },
    )
}

// ── Phase 2 S3 placement deny-all stub (no routing, AI-0296) ──
//
// Deny-all placement-policy stub evaluated host-side at dispatch, per the
// CTX-0154 candidate (`providers/provider-plugin-boundary.md`, MCP placement
// policy section): every placement denies with a typed refusal until its
// backend is reviewed and listed. No routing, no model selection, no I/O.
//
// Denial covers every placement input: [`ToolPlacement::Mcp`] and
// [`ToolPlacement::Native`] alike, declared or not (`None` placement,
// contradictory `Mcp`-with-detail, `Native`-without-detail, and valid
// declarations the S1 [`validate_placement`] rule would admit all deny here
// with the same typed refusal). There is no admit path, no fallback to the
// other placement path, and no partial state: the single `Err` below is the
// only outcome.
//
// Transport kind never doubles as placement: the provider transport kinds
// (`HttpApi`, `LocalEndpoint`, `CliHarness`, `ManagedAccount`, `Router`) name
// adapter classes and grant no capability by themselves; tool placement is
// only the `mcp` / `native` declaration above plus, for `native`, the
// reviewed backend name and the recorded reason. Provider adapters stay
// behind the same backend, never a second authorization path.
//
// AIQ-36 (native versus MCP tool transport and bridge placement) stays open:
// this stub closes nothing, grants nothing, reviews no backend, and lists no
// backend. Acceptance belongs to reviewers through the canonical admission
// rule, not to this task.
//
// Reuse only: this section reuses the S1/S2 types above ([`ToolPlacement`],
// [`NativePlacement`], [`McpError`], [`McpStage`], [`McpFailure`],
// [`bound_error_text`], [`MAX_ERROR_TEXT_BYTES`]) and introduces no
// re-mirrored vocabulary, no new byte bounds, no version field, and no
// envelope. `std`-only, no threads, no clock, no network, no filesystem, no
// SPKI fetch, no spawn, no credential or secret material. `Cargo.toml` stays
// `std`-only.

/// Deny-all placement-policy stub: every placement denies until its backend
/// is reviewed and listed.
///
/// Unit struct, no state: construction performs no I/O and grants nothing.
/// Evaluation is a pure `&str`/`&[u8]`-in/value-out refusal — never a route,
/// never a model selection, never a dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlacementStub;

impl PlacementStub {
    /// Build the stub. No state, no I/O, grants nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Deny every placement with the typed refusal.
    ///
    /// Both inputs are observed only as denied: `placement` covers `mcp`,
    /// `native`, and undeclared (`None`); `native` covers absent, valid, and
    /// contradictory detail. Contents are never inspected for routing, never
    /// select a model, and never move provider or tool I/O — every
    /// combination returns the same [`McpStage::Bridge`] +
    /// [`McpFailure::HandshakeRejected`] refusal. Pure, deterministic,
    /// offline.
    ///
    /// # Errors
    ///
    /// Always returns `Err`: the placement is denied until its backend is
    /// reviewed and listed.
    pub fn check(
        &self,
        placement: Option<ToolPlacement>,
        native: Option<&NativePlacement>,
    ) -> Result<(), McpError> {
        // Observed only as denied: no branch on placement or detail, so no
        // routing, selection, or fallback observable can leak through.
        let _ = (placement, native);
        Err(McpError::new(
            McpStage::Bridge,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    "placement denied: backend not reviewed and listed",
                    MAX_ERROR_TEXT_BYTES,
                ),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(name: &str, schema: &[u8]) -> RawMcpTool {
        RawMcpTool {
            name: name.to_owned(),
            description: "test".to_owned(),
            schema_json: schema.to_vec(),
        }
    }

    fn imported(name: &str, schema: &[u8]) -> ImportedTool {
        import_tool("demo", &raw(name, schema)).expect("valid import")
    }

    fn pin() -> InitializeResult {
        InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            tools_supported: true,
            tools_shape: ToolsCapabilityShape::Object {
                list_changed: Some(true),
            },
        }
    }

    #[test]
    fn mirror_bounds_match_survey_values() {
        assert_eq!(MAX_TOOL_NAME_LEN, 64);
        assert_eq!(MAX_TOOL_DESCRIPTION_LEN, 512);
        assert_eq!(MAX_TOOL_SCHEMA_BYTES, 16 * 1024);
        assert_eq!(MAX_NATIVE_PLACEMENT_BYTES, 512);
        assert_eq!(MAX_REASON_BYTES, 512);
        assert_eq!(MAX_ERROR_TEXT_BYTES, 512);
        assert_eq!(MAX_ERROR_NAME_BYTES, 128);
        assert_eq!(MAX_TOOL_PAGES, 5);
        assert_eq!(PROTOCOL_VERSION, "2025-11-25");
        // FNV-1a-64 is byte-identical with the runtime/mcp precedent: empty
        // input hashes to the offset basis, and the schema digest of empty
        // bytes agrees.
        assert_eq!(fnv_1a_64(b""), 0xcbf29ce484222325);
        let spec = ToolSpec::new(
            "mcp_demo_echo",
            "test",
            Vec::new(),
            "mcp.demo",
            true,
            ToolPlacement::Mcp,
            None,
        )
        .expect("valid spec");
        assert_eq!(spec.schema_digest(), 0xcbf29ce484222325);
    }

    #[test]
    fn placement_vocabulary_is_closed() {
        assert_eq!(ToolPlacement::parse("mcp"), Ok(ToolPlacement::Mcp));
        assert_eq!(ToolPlacement::parse("native"), Ok(ToolPlacement::Native));
        assert_eq!(ToolPlacement::Mcp.as_str(), "mcp");
        assert_eq!(ToolPlacement::Native.as_str(), "native");
        // Round-trip through the vocabulary string.
        for placement in [ToolPlacement::Mcp, ToolPlacement::Native] {
            assert_eq!(ToolPlacement::parse(placement.as_str()), Ok(placement));
        }
        // Unknown strings fail closed with the rejected string carried.
        for rejected in [
            "",
            "MCP",
            "mcp ",
            " native",
            "remote",
            "shell",
            "mcp/native",
        ] {
            let error = ToolPlacement::parse(rejected).expect_err("unknown must fail");
            assert!(
                matches!(error, PlacementError::UndeclaredPlacement { .. }),
                "unexpected: {error:?}"
            );
            assert_eq!(
                error.to_string(),
                format!("undeclared tool placement: {rejected}")
            );
        }
    }

    #[test]
    fn placement_contradiction_and_missing_refuse() {
        // Missing declaration refuses.
        let missing = validate_placement("probe_tool", None, None).expect_err("None must fail");
        assert_eq!(
            missing,
            PlacementError::UndeclaredPlacement {
                name: "probe_tool".to_owned(),
            }
        );
        // Mcp carrying native detail is contradictory.
        let contradictory = validate_placement(
            "mcp_tool",
            Some(ToolPlacement::Mcp),
            Some(&NativePlacement {
                reason: "r".to_owned(),
                backend: "b".to_owned(),
            }),
        )
        .expect_err("Mcp with detail must fail");
        assert!(matches!(
            contradictory,
            PlacementError::UndeclaredPlacement { .. }
        ));
        // Native without detail refuses; empty halves refuse.
        for native in [
            None,
            Some(NativePlacement {
                reason: String::new(),
                backend: "b".to_owned(),
            }),
            Some(NativePlacement {
                reason: "r".to_owned(),
                backend: String::new(),
            }),
        ] {
            let error =
                validate_placement("native_tool", Some(ToolPlacement::Native), native.as_ref())
                    .expect_err("native without full detail must fail");
            assert!(matches!(error, PlacementError::UndeclaredPlacement { .. }));
        }
        // Native halves past 512 bytes refuse (inclusive bound).
        let overlong = "x".repeat(MAX_NATIVE_PLACEMENT_BYTES + 1);
        let error = validate_placement(
            "native_tool",
            Some(ToolPlacement::Native),
            Some(&NativePlacement {
                reason: overlong.clone(),
                backend: "b".to_owned(),
            }),
        )
        .expect_err("overlong reason must fail");
        assert!(matches!(error, PlacementError::UndeclaredPlacement { .. }));
        let error = validate_placement(
            "native_tool",
            Some(ToolPlacement::Native),
            Some(&NativePlacement {
                reason: "r".to_owned(),
                backend: overlong,
            }),
        )
        .expect_err("overlong backend must fail");
        assert!(matches!(error, PlacementError::UndeclaredPlacement { .. }));
        // At-cap admits; Mcp without detail admits.
        assert!(
            validate_placement(
                "native_tool",
                Some(ToolPlacement::Native),
                Some(&NativePlacement {
                    reason: "x".repeat(MAX_NATIVE_PLACEMENT_BYTES),
                    backend: "y".repeat(MAX_NATIVE_PLACEMENT_BYTES),
                }),
            )
            .is_ok()
        );
        assert!(validate_placement("mcp_tool", Some(ToolPlacement::Mcp), None).is_ok());
        // The constructor enforces the same rule: Mcp with detail refuses,
        // Native without detail refuses, Native with detail admits.
        assert!(
            ToolSpec::new(
                "mcp_tool",
                "test",
                Vec::new(),
                "mcp.demo",
                true,
                ToolPlacement::Mcp,
                Some(NativePlacement {
                    reason: "r".to_owned(),
                    backend: "b".to_owned(),
                }),
            )
            .is_err()
        );
        assert!(
            ToolSpec::new(
                "native_tool",
                "test",
                Vec::new(),
                "test.scope",
                false,
                ToolPlacement::Native,
                None,
            )
            .is_err()
        );
        ToolSpec::new(
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
    }

    #[test]
    fn undeclared_is_registration_never_dispatch() {
        // Registration refusal carries its own Display and never shares the
        // dispatch-denial shape (this subset declares no dispatch denial).
        let error = PlacementError::UndeclaredPlacement {
            name: "probe".to_owned(),
        };
        assert_eq!(error.to_string(), "undeclared tool placement: probe");
        assert!(error.to_string().contains("undeclared"));
        assert!(!error.to_string().contains("denied"));
        let parsed = ToolPlacement::parse("nope").expect_err("unknown must fail");
        assert!(matches!(parsed, PlacementError::UndeclaredPlacement { .. }));
        assert!(!matches!(parsed, PlacementError::InvalidName { .. }));
    }

    #[test]
    fn import_carries_mcp_placement_by_construction() {
        let imported = import_tool(
            "demo",
            &RawMcpTool {
                name: "echo".to_owned(),
                description: "test".to_owned(),
                schema_json: br#"{"type":"object"}"#.to_vec(),
            },
        )
        .expect("import");
        assert_eq!(imported.spec.placement, Some(ToolPlacement::Mcp));
        assert_eq!(imported.spec.native, None);
        assert!(imported.spec.read_only);
        assert_eq!(imported.spec.required_scope, "mcp.demo");
        assert_eq!(imported.digest, imported.spec.schema_digest());
        assert_eq!(imported.server_id, "demo");
        assert_eq!(imported.raw_name, "echo");
        assert_eq!(imported.spec.name, "mcp_demo_echo");
    }

    #[test]
    fn relist_preserves_mcp_placement_across_snapshot_diff() {
        let first = imported("echo", br#"{"type":"object"}"#);
        let second = imported("echo", br#"{"type":"object"}"#);
        assert_eq!(first.spec.placement, Some(ToolPlacement::Mcp));
        assert_eq!(second.spec.placement, Some(ToolPlacement::Mcp));
        let old = ToolListSnapshot::from_imported(std::slice::from_ref(&first));
        let new = ToolListSnapshot::from_imported(std::slice::from_ref(&second));
        assert!(diff_tool_snapshots(&old, &new).is_empty());
    }

    #[test]
    fn snapshot_diff_detects_added_removed_changed() {
        let old = ToolListSnapshot::from_imported(&[
            imported("alpha", b"{}"),
            imported("beta", b"{}"),
            imported("gamma", b"{}"),
        ]);
        let new = ToolListSnapshot::from_imported(&[
            imported("beta", b"{}"),
            imported("gamma", b"{\"type\":\"object\"}"),
            imported("delta", b"{}"),
        ]);
        // Entries sort by sanitized name for deterministic diffs.
        let names: Vec<&str> = old
            .entries()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        let diff = diff_tool_snapshots(&old, &new);
        assert_eq!(diff.added, vec!["mcp_demo_delta".to_owned()]);
        assert_eq!(diff.removed, vec!["mcp_demo_alpha".to_owned()]);
        assert_eq!(diff.changed, vec!["mcp_demo_gamma".to_owned()]);
        assert!(!diff.is_empty());
        assert!(diff_tool_snapshots(&old, &old).is_empty());
        assert_eq!(old.len(), 3);
        assert!(!old.is_empty());
        assert!(ToolListSnapshot::from_imported(&[]).is_empty());
        assert_eq!(
            old.digest_of("mcp_demo_beta"),
            new.digest_of("mcp_demo_beta")
        );
        assert!(old.digest_of("mcp_demo_missing").is_none());
    }

    #[test]
    fn pin_matches_catches_drift_not_swap() {
        let listed = pin();
        let again = pin();
        assert!(listed.matches_pin(&again));
        assert!(again.matches_pin(&listed));
        // Version drift refuses.
        let drifted = InitializeResult {
            protocol_version: "1999-01-01".to_owned(),
            ..listed.clone()
        };
        assert!(!listed.matches_pin(&drifted));
        assert!(!drifted.matches_pin(&listed));
        // Capability downgrade refuses (object with listChanged to bare
        // object, to bool, and bool to object all move the shape).
        for shape in [
            ToolsCapabilityShape::Object { list_changed: None },
            ToolsCapabilityShape::Object {
                list_changed: Some(false),
            },
            ToolsCapabilityShape::BoolTrue,
        ] {
            let moved = InitializeResult {
                tools_shape: shape,
                ..listed.clone()
            };
            assert!(!listed.matches_pin(&moved), "shape move must mismatch");
        }
        // Capability upgrade refuses too (bool to listChanged object).
        let boolean = InitializeResult {
            tools_shape: ToolsCapabilityShape::BoolTrue,
            ..listed.clone()
        };
        assert!(!listed.matches_pin(&boolean));
        assert!(!boolean.matches_pin(&listed));
        // Identical protocol state still matches even for a swapped binary:
        // binary identity is out of scope for this pin (Rank-2 documents the
        // separation via IdentityMismatch, never via this comparison).
        let swapped = pin();
        assert!(listed.matches_pin(&swapped));
    }

    #[test]
    fn sanitize_shapes_names_and_caps_at_64() {
        assert_eq!(sanitize_mcp_name("demo", "echo"), "mcp_demo_echo");
        assert_eq!(
            sanitize_mcp_name("demo", "List.Files-2"),
            "mcp_demo_list_files_2"
        );
        let folded = sanitize_mcp_name("demo", "9 lives!");
        assert!(validate_tool_name(&folded).is_ok());
        let long = sanitize_mcp_name("demo", &"x".repeat(200));
        assert_eq!(long.len(), 64);
        assert!(validate_tool_name(&long).is_ok());
        assert_eq!(long, sanitize_mcp_name("demo", &"x".repeat(200)));
        assert_ne!(long, sanitize_mcp_name("demo", &"y".repeat(200)));
        assert_eq!(long.as_bytes()[55], b'_');
    }

    #[test]
    fn names_only_bound_scrubs_and_truncates() {
        // Newlines, carriage returns, escapes, and non-ASCII become `?`.
        let hostile = format!("bad\nname\r\u{1b}[2J\u{7f}{}", "x".repeat(2048));
        let scrubbed = bound_error_name(&hostile);
        assert!(scrubbed.len() <= MAX_ERROR_NAME_BYTES);
        assert_eq!(scrubbed.len(), MAX_ERROR_NAME_BYTES);
        assert!(scrubbed.starts_with("bad?name?"));
        assert!(scrubbed.bytes().all(|b| (0x20..=0x7E).contains(&b)));
        assert!(!scrubbed.contains('\n'));
        let long_reason = bound_error_text(&hostile, MAX_ERROR_TEXT_BYTES);
        assert_eq!(long_reason.len(), MAX_ERROR_TEXT_BYTES);
        // The vocabulary parse boundary scrubs too.
        let error = ToolPlacement::parse("bad\nname").expect_err("must fail");
        assert!(error.to_string().contains("bad?name"));
        assert!(!error.to_string().contains('\n'));
        // Identity mismatch names the server only: no path bytes, no hash.
        let mismatch = McpError::new(
            McpStage::Handshake,
            McpFailure::IdentityMismatch {
                server: bound_error_name("demo"),
            },
        );
        let text = mismatch.to_string();
        assert!(text.contains("'demo'"));
        assert!(!text.contains('/'));
        assert!(!text.contains("e3b0c442"));
        assert!(text.starts_with("mcp "));
        // Display quotes names never values: every mirrored Display stays
        // printable ASCII and bounded.
        for error in [
            McpError::new(
                McpStage::ListTools,
                McpFailure::DuplicateTool {
                    name: bound_error_name("mcp_demo_echo"),
                },
            ),
            McpError::new(McpStage::Handshake, McpFailure::NoToolCapability),
            McpError::new(
                McpStage::Bridge,
                McpFailure::CapabilityDrift {
                    detail: bound_error_text("protocol pin mismatch", MAX_ERROR_TEXT_BYTES),
                },
            ),
            McpError::new(
                McpStage::Handshake,
                McpFailure::VersionMismatch {
                    expected: PROTOCOL_VERSION.to_owned(),
                    got: bound_error_text("1999-01-01", MAX_ERROR_NAME_BYTES),
                },
            ),
        ] {
            let text = error.to_string();
            assert!(text.starts_with("mcp "));
            assert!(text.bytes().all(|b| (0x20..=0x7E).contains(&b)));
            assert!(!error.retryable, "placement subset is fatal-only");
        }
        // Placement refusal normalizes hostile names at the boundary.
        let hostile_error = PlacementError::InvalidName {
            name: hostile.clone(),
        }
        .normalized();
        assert!(hostile_error.to_string().len() <= MAX_ERROR_TEXT_BYTES + 64);
    }

    #[test]
    fn stage_and_failure_taxonomy_stays_subset() {
        assert_eq!(McpStage::Handshake.to_string(), "handshake");
        assert_eq!(McpStage::ListTools.to_string(), "list-tools");
        assert_eq!(McpStage::Bridge.to_string(), "bridge");
        assert_eq!(
            McpFailure::NoToolCapability.to_string(),
            "server offers no tool capability"
        );
        assert_eq!(
            McpFailure::TooManyPages { limit: 5 }.to_string(),
            "tool listing exceeds 5 pages"
        );
        assert_eq!(
            McpFailure::DuplicateCursor {
                cursor: "abc".to_owned(),
            }
            .to_string(),
            "tool listing repeated cursor 'abc'"
        );
    }
}

// ---------------------------------------------------------------------------
// S2 offline tests: pin shape, snapshot echo, stale-flag data-only, error
// axes. Deterministic, no network, no filesystem, no clock.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod placement_envelope_tests {
    use super::*;

    fn pin() -> InitializeResult {
        InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            tools_supported: true,
            tools_shape: ToolsCapabilityShape::Object {
                list_changed: Some(true),
            },
        }
    }

    fn raw(name: &str, schema: &[u8]) -> RawMcpTool {
        RawMcpTool {
            name: name.to_owned(),
            description: "test".to_owned(),
            schema_json: schema.to_vec(),
        }
    }

    fn imported(name: &str, schema: &[u8]) -> ImportedTool {
        import_tool("demo", &raw(name, schema)).expect("valid import")
    }

    #[test]
    fn pin_envelope_echoes_and_validates_shape() {
        let envelope = PinEnvelope::new(pin());
        assert_eq!(envelope.pin(), &pin());
        assert_eq!(envelope, PinEnvelope::new(pin()));
        // Shape validation admits the pinned handshake facts.
        assert!(envelope.validate().is_ok());
        // Every capability shape is handshake-valid on its own: drift is
        // caught by matches, never by validate.
        for shape in [
            ToolsCapabilityShape::BoolTrue,
            ToolsCapabilityShape::Object { list_changed: None },
            ToolsCapabilityShape::Object {
                list_changed: Some(false),
            },
            ToolsCapabilityShape::Object {
                list_changed: Some(true),
            },
        ] {
            let shaped = PinEnvelope::new(InitializeResult {
                protocol_version: PROTOCOL_VERSION.to_owned(),
                tools_supported: true,
                tools_shape: shape,
            });
            assert!(shaped.validate().is_ok(), "any shape validates");
        }
        // Version drift refuses with the typed pin mismatch.
        let drifted = PinEnvelope::new(InitializeResult {
            protocol_version: "1999-01-01".to_owned(),
            ..pin()
        });
        let error = drifted.validate().expect_err("version drift must fail");
        assert_eq!(error.stage, McpStage::Handshake);
        assert!(matches!(error.failure, McpFailure::VersionMismatch { .. }));
        assert!(!error.retryable);
        assert!(error.to_string().contains("protocol version mismatch"));
        // Missing tools capability refuses with the typed absence refusal.
        let unsupported = PinEnvelope::new(InitializeResult {
            tools_supported: false,
            ..pin()
        });
        let error = unsupported
            .validate()
            .expect_err("missing capability must fail");
        assert_eq!(
            error,
            McpError::new(McpStage::Handshake, McpFailure::NoToolCapability)
        );
        assert!(!error.retryable);
        // Pin comparison delegates to the S1 matches_pin rule.
        assert!(envelope.matches(&pin()));
        assert!(pin_matches(&envelope, &pin()));
        let moved = InitializeResult {
            tools_shape: ToolsCapabilityShape::BoolTrue,
            ..pin()
        };
        assert!(!envelope.matches(&moved));
        // Envelopes carry no version field by construction: exact-field
        // construction round-trips without loss.
        let rebuilt = PinEnvelope::new(envelope.pin().clone());
        assert_eq!(rebuilt, envelope);
    }

    fn pin_matches(envelope: &PinEnvelope, fresh: &InitializeResult) -> bool {
        envelope.matches(fresh)
    }

    #[test]
    fn snapshot_envelope_echoes_snapshot_and_flags() {
        let tools = vec![
            imported("beta", b"{}"),
            imported("alpha", b"{}"),
            imported("gamma", b"{\"type\":\"object\"}"),
        ];
        let expected = ToolListSnapshot::from_imported(&tools);
        let envelope = SnapshotEnvelope::from_imported(&tools, 7, false);
        assert_eq!(envelope.snapshot(), &expected);
        assert_eq!(envelope.snapshot().entries(), expected.entries());
        assert_eq!(envelope.list_version(), 7);
        assert!(!envelope.is_stale());
        assert_eq!(envelope, SnapshotEnvelope::new(expected.clone(), 7, false));
        // Entries sort by sanitized name for deterministic diffs, echoed
        // verbatim through the envelope.
        let names: Vec<&str> = envelope
            .snapshot()
            .entries()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert_eq!(envelope.snapshot().len(), 3);
        assert!(!envelope.snapshot().is_empty());
        assert_eq!(
            envelope.snapshot().digest_of("mcp_demo_beta"),
            expected.digest_of("mcp_demo_beta")
        );
        assert!(envelope.snapshot().digest_of("mcp_demo_missing").is_none());
        // Diff reuses the S1 pure helper over the echoed snapshots.
        let newer = SnapshotEnvelope::from_imported(
            &[imported("beta", b"{}"), imported("delta", b"{}")],
            8,
            false,
        );
        let diff = diff_tool_snapshots(envelope.snapshot(), newer.snapshot());
        assert!(diff.added.contains(&"mcp_demo_delta".to_owned()));
        assert!(!diff.is_empty());
        assert!(diff_tool_snapshots(envelope.snapshot(), envelope.snapshot()).is_empty());
        // Empty import echoes an empty snapshot without staleness.
        let empty = SnapshotEnvelope::from_imported(&[], 0, false);
        assert!(empty.snapshot().is_empty());
        assert_eq!(empty.snapshot().len(), 0);
        assert_eq!(empty.list_version(), 0);
        assert!(!empty.is_stale());
    }

    #[test]
    fn stale_flag_is_data_only_without_enforcement() {
        let tools = vec![imported("echo", br#"{"type":"object"}"#)];
        // A stale envelope is still well-formed data: construction admits
        // every flag combination and the flags echo verbatim.
        let stale = SnapshotEnvelope::from_imported(&tools, 3, true);
        assert!(stale.is_stale());
        assert_eq!(stale.list_version(), 3);
        assert_eq!(stale.snapshot().len(), 1);
        let fresh = SnapshotEnvelope::from_imported(&tools, 3, false);
        assert!(!fresh.is_stale());
        // Same snapshot bytes under both flags: staleness never rewrites
        // the digest pairs.
        assert_eq!(stale.snapshot(), fresh.snapshot());
        // A version/stale mismatch is still well-formed data here; matching
        // and deny-stale discipline stay caller/MCP-owned, never envelope
        // enforcement.
        let mismatched = SnapshotEnvelope::new(stale.snapshot().clone(), u64::MAX, true);
        assert!(mismatched.is_stale());
        assert_eq!(mismatched.list_version(), u64::MAX);
        assert_eq!(mismatched.snapshot(), stale.snapshot());
        let cleared = SnapshotEnvelope::new(stale.snapshot().clone(), u64::MAX, false);
        assert!(!cleared.is_stale());
        assert_eq!(cleared.snapshot(), stale.snapshot());
    }

    #[test]
    fn placement_error_mapping_covers_all_axes() {
        let cases = [
            PlacementError::InvalidName {
                name: "BAD".to_owned(),
            },
            PlacementError::DescriptionTooLarge {
                limit: MAX_TOOL_DESCRIPTION_LEN,
                actual: MAX_TOOL_DESCRIPTION_LEN + 1,
            },
            PlacementError::SchemaTooLarge {
                limit: MAX_TOOL_SCHEMA_BYTES,
                actual: MAX_TOOL_SCHEMA_BYTES + 1,
            },
            PlacementError::UndeclaredPlacement {
                name: "probe_tool".to_owned(),
            },
        ];
        for error in &cases {
            let mapped = placement_to_mcp_error(error);
            // Every declaration refusal maps to the import-stage typed
            // failure (the S1 import precedent: the entry is not usable).
            assert_eq!(mapped.stage, McpStage::ListTools);
            assert!(matches!(
                mapped.failure,
                McpFailure::HandshakeRejected { .. }
            ));
            assert!(!mapped.retryable, "placement mapping is fatal-only");
            let text = mapped.to_string();
            assert!(text.starts_with("mcp "));
            assert!(text.contains("handshake rejected"));
            assert!(text.bytes().all(|b| (0x20..=0x7E).contains(&b)));
        }
        // Bounds cross verbatim; names cross bounded and scrubbed.
        let over = placement_to_mcp_error(&PlacementError::SchemaTooLarge {
            limit: MAX_TOOL_SCHEMA_BYTES,
            actual: MAX_TOOL_SCHEMA_BYTES + 1,
        });
        assert!(
            over.to_string()
                .contains(&format!("{MAX_TOOL_SCHEMA_BYTES}"))
        );
        let hostile = placement_to_mcp_error(&PlacementError::InvalidName {
            name: "bad\nname".to_owned(),
        });
        assert!(!hostile.to_string().contains('\n'));
        assert!(hostile.to_string().contains("bad?name"));
        // Long refusal text truncates to the names-only bound.
        let long = placement_to_mcp_error(&PlacementError::UndeclaredPlacement {
            name: "x".repeat(2048),
        });
        assert!(
            long.to_string().len()
                <= "mcp list-tools: handshake rejected: ".len() + MAX_ERROR_TEXT_BYTES
        );
        // Total mapping: no PlacementError variant falls through, and the
        // mapped Display never leaks payload bytes.
        for error in &cases {
            let text = placement_to_mcp_error(error).to_string();
            assert!(!text.contains('{'));
            assert!(!text.contains('}'));
        }
    }

    #[test]
    fn envelope_bounds_match_s1_and_carry_no_version() {
        // Byte bounds are the S1 values, unchanged for the envelope layer.
        assert_eq!(MAX_TOOL_NAME_LEN, 64);
        assert_eq!(MAX_TOOL_DESCRIPTION_LEN, 512);
        assert_eq!(MAX_TOOL_SCHEMA_BYTES, 16 * 1024);
        assert_eq!(MAX_NATIVE_PLACEMENT_BYTES, 512);
        assert_eq!(MAX_REASON_BYTES, 512);
        assert_eq!(MAX_ERROR_TEXT_BYTES, 512);
        assert_eq!(MAX_ERROR_NAME_BYTES, 128);
        assert_eq!(MAX_TOOL_PAGES, 5);
        assert_eq!(PROTOCOL_VERSION, "2025-11-25");
    }
}

// ---------------------------------------------------------------------------
// S3 offline tests: deny-all placements, declared-still-denies, typed
// refusal, no routing observables. Deterministic, no network, no filesystem,
// no clock.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod placement_stub_tests {
    use super::*;

    fn native_detail() -> NativePlacement {
        NativePlacement {
            reason: "host-native".to_owned(),
            backend: "backend".to_owned(),
        }
    }

    fn denied_error(
        placement: Option<ToolPlacement>,
        native: Option<&NativePlacement>,
    ) -> McpError {
        PlacementStub::new()
            .check(placement, native)
            .expect_err("deny-all stub never admits")
    }

    #[test]
    fn deny_all_placements_mcp_native_and_undeclared() {
        let stub = PlacementStub::new();
        // Undeclared denies.
        assert!(stub.check(None, None).is_err());
        // MCP without detail denies.
        assert!(stub.check(Some(ToolPlacement::Mcp), None).is_err());
        // Native with valid detail denies.
        let detail = native_detail();
        assert!(
            stub.check(Some(ToolPlacement::Native), Some(&detail))
                .is_err()
        );
        // Contradictory and incomplete declarations deny with no fallback to
        // the other path.
        assert!(stub.check(Some(ToolPlacement::Mcp), Some(&detail)).is_err());
        assert!(stub.check(Some(ToolPlacement::Native), None).is_err());
        assert!(
            stub.check(
                Some(ToolPlacement::Native),
                Some(&NativePlacement {
                    reason: String::new(),
                    backend: "b".to_owned(),
                }),
            )
            .is_err()
        );
    }

    #[test]
    fn declared_still_denies() {
        // S1 admits this MCP declaration, the stub still denies it.
        let mcp_spec = ToolSpec::new(
            "mcp_tool",
            "test",
            Vec::new(),
            "mcp.demo",
            true,
            ToolPlacement::Mcp,
            None,
        )
        .expect("S1 admits valid MCP declaration");
        assert!(
            PlacementStub::new()
                .check(mcp_spec.placement, mcp_spec.native.as_ref())
                .is_err()
        );
        // S1 admits this native declaration, the stub still denies it.
        let native_spec = ToolSpec::new(
            "native_tool",
            "test",
            Vec::new(),
            "test.scope",
            false,
            ToolPlacement::Native,
            Some(native_detail()),
        )
        .expect("S1 admits valid native declaration");
        assert!(
            PlacementStub::new()
                .check(native_spec.placement, native_spec.native.as_ref())
                .is_err()
        );
        // Imported MCP tools carry Mcp placement by construction and still
        // deny.
        let imported = import_tool(
            "demo",
            &RawMcpTool {
                name: "echo".to_owned(),
                description: "test".to_owned(),
                schema_json: br#"{"type":"object"}"#.to_vec(),
            },
        )
        .expect("valid import");
        assert_eq!(imported.spec.placement, Some(ToolPlacement::Mcp));
        assert!(
            PlacementStub::new()
                .check(imported.spec.placement, imported.spec.native.as_ref())
                .is_err()
        );
    }

    #[test]
    fn typed_refusal_is_bridge_handshake_rejected_fatal() {
        let error = denied_error(Some(ToolPlacement::Mcp), None);
        assert_eq!(error.stage, McpStage::Bridge);
        assert!(
            matches!(error.failure, McpFailure::HandshakeRejected { .. }),
            "unexpected: {error:?}"
        );
        assert!(!error.retryable, "placement denial is fatal-only");
        let text = error.to_string();
        assert!(text.starts_with("mcp bridge: handshake rejected: "));
        assert!(text.contains("placement denied: backend not reviewed and listed"));
        // Names-only bounded diagnostic: printable ASCII, within bound, no
        // payload bytes.
        assert!(text.bytes().all(|b| (0x20..=0x7E).contains(&b)));
        assert!(text.len() <= "mcp bridge: handshake rejected: ".len() + MAX_ERROR_TEXT_BYTES);
        assert!(!text.contains('{'));
        assert!(!text.contains('}'));
    }

    #[test]
    fn no_routing_observables() {
        // Every placement input maps to the identical typed refusal: no
        // branch, no route, no model selection, no fallback observable.
        let detail = native_detail();
        let cases = [
            denied_error(None, None),
            denied_error(Some(ToolPlacement::Mcp), None),
            denied_error(Some(ToolPlacement::Native), Some(&detail)),
            denied_error(Some(ToolPlacement::Mcp), Some(&detail)),
            denied_error(Some(ToolPlacement::Native), None),
        ];
        for error in &cases[1..] {
            assert_eq!(*error, cases[0]);
        }
        // Deterministic across calls and instances; the stub carries no
        // state that could select or route.
        assert_eq!(PlacementStub::new(), PlacementStub);
        assert_eq!(std::mem::size_of::<PlacementStub>(), 0);
        assert_eq!(
            denied_error(Some(ToolPlacement::Native), Some(&detail)),
            denied_error(Some(ToolPlacement::Native), Some(&detail))
        );
    }
}

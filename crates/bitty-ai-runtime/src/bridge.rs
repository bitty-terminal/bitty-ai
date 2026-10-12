//! P1 bridge: protocol `owner.name` identity mapping plus consent-ledger seam.
//!
//! This module is the `bitty-ai` side of the bridge whose host side lives in
//! `bitty` (`bitty-ipc`). The real consent ledger, capability enforcement,
//! and execution backend stay host-owned; this crate defines only the seam
//! the host wires behind. There is deliberately no dependency on the generic
//! `bitty-agent` protocol crate: protocol identity travels as a validated
//! wire string (`owner.name`), never as an imported type.
//!
//! # Wire shapes mirrored here (read-only, never modified here)
//!
//! - Generic bridge client boundary (`bitty` #709): routing, authorization,
//!   consent pre-check, bounded params (`16 KiB`), envelope, attribution by
//!   bounded `client_id` (`<= 64` bytes), and correlated outcome. The client
//!   ledger there is a local pre-check mirror; enforcement stays server-side.
//! - Host-side tool dispatch with per-tool consent (`bitty` #705):
//!   routing, server-evaluated scope, per-`(client_id, scope)` consent,
//!   captured target, budget, attribution (`client_id` + `tool` +
//!   `execution_id`), and bounded outcome labeled `is_untrusted_surface`.
//!   Read-only by default: effect tools additionally require an explicit
//!   per-call opt-in and must not launder through pure-inspect scopes.
//! - Generic `ExecutionContext` backend plus structured `ExecutionResult`
//!   (`bitty` #707): `ExecutionRequest` (executable, args, cwd, closed
//!   `EnvPolicy`, target, timeout, output budget) dispatches through the
//!   `process.spawn` scope plus consent plus explicit effect opt-in into a
//!   bounded result with closed `effect_state`. `Unknown` reconciles via a
//!   query path; there is no blind-retry primitive.
//! - Protocol identity `owner.name` (generic `bitty-agent` wire principal,
//!   e.g. `"local.assistant"`): exactly `owner.name` (one dot), each segment
//!   `^[a-z][a-z0-9_-]*$`, total `<= 128` bytes, each segment `<= 64` bytes.
//!   [`ProtocolAgentId`] re-validates that exact shape from the wire string
//!   so this crate never imports the protocol type.
//!
//! # What this module owns
//!
//! - [`ProtocolAgentId`]: validated wire-level `owner.name` principal.
//! - [`IdentityBridge`]: single-agent binding of one protocol principal to
//!   one runtime-local [`crate::session::AgentInstanceId`]. `v0.1` runs a
//!   single agent, so at most one binding exists; a second distinct binding
//!   fails closed.
//! - [`ConsentLedger`] seam plus [`DenyAllConsent`] (deny-by-default) and
//!   [`FakeConsentLedger`] (bounded deterministic test double). The host
//!   implements the real ledger behind [`ConsentLedger`]; this crate claims
//!   no accepted security mechanism.
//!
//! # Gate order (R2 draft disposition, host composes)
//!
//! ```text
//! wire protocol id -> IdentityBridge -> ConsentLedger -> ToolBus dispatch
//! ```
//!
//! The bridge never grants authority by itself: unknown protocol ids,
//! unmapped instances, missing or expired consent, and missing host wiring
//! all deny with no partial state. This mirrors the `R1` primary-execution
//! disposition (authorized `ExecutionContext` with captured target and
//! generation) and the `R2` unified-backend gate order (principal, ToolSpec,
//! caller/target authorization, consent, budget, placed dispatch, redaction,
//! attributed outcome).
//!
//! # Determinism rules
//!
//! Every operation takes caller-supplied `now_ms` where time matters. There
//! is no wall clock, thread, async runtime, network, filesystem, or secret.
//! All behavior in tests is reproducible from the bound pair plus the fake
//! grant table plus `now_ms`. The module is `std`-only and keeps no ambient
//! authority.

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::session::AgentInstanceId;

/// Maximum protocol `owner.name` length in bytes.
///
/// Mirrors the generic `bitty-agent` wire bound: total `<= 128` bytes.
pub const MAX_PROTOCOL_ID_LEN: usize = 128;

/// Maximum bytes per `owner` / `name` segment.
///
/// Mirrors the generic `bitty-agent` wire bound: each segment `<= 64` bytes.
pub const MAX_PROTOCOL_ID_SEGMENT_LEN: usize = 64;

/// Maximum grants tracked by [`FakeConsentLedger`].
///
/// Mirrors the accepted host ledger scale (`bitty-ipc` consent ledger
/// `MAX_GRANTS`): the fake is explicit, never auto-evicting, and fails
/// closed at capacity.
pub const MAX_CONSENT_GRANTS: usize = 64;

/// Maximum consent scope string length in bytes (skeleton default).
///
/// The runtime treats `required_scope` as an opaque wire string (e.g.
/// `"workspace.read"`); the typed scope enum lives host-side. This bound
/// keeps the fake grant table bounded: `64` grants times bounded strings.
/// Real scope validation stays host-side.
pub const MAX_CONSENT_SCOPE_LEN: usize = 128;

/// Default consent grant time-to-live in milliseconds (AI-0255).
///
/// One turn-scoped gesture: `120_000` (`120`s). The accepted solicitation
/// window is `60s..=300s`: below `60s` a default four-round turn
/// (`DEFAULT_MAX_ROUNDS` plus provider timeouts plus the human gesture)
/// lapses mid-turn and re-prompts every round (more gestures, not fewer);
/// above `300s` the least-privilege window outlives the turn it was granted
/// for and widens replay. `120s` is the midpoint that covers a full turn
/// with margin while keeping the window short. [`grant_turn_set`] uses this
/// when the caller passes `ttl_ms: None`; an explicit `Some(ttl)` overrides
/// per call (no range enforcement, only documentation). The value is a
/// duration added to caller-supplied `now_ms` with saturating addition;
/// there is no wall-clock read anywhere on this path. Never persistent:
/// every grant carries an absolute `expires_at_ms` and lapses without
/// refresh, and there is no infinite or session-scoped grant shape.
pub const DEFAULT_CONSENT_TTL_MS: u64 = 120_000;

/// Maximum bytes kept for any reason string emitted across a runtime
/// boundary (`AI-0059`).
///
/// Mirrors [`crate::reconcile::MAX_RECONCILE_REASON_BYTES`] so every
/// runtime-owned reason surface shares one bound. Reasons are interpolated
/// with wire or host-supplied strings (tool names, provider details), so
/// they are truncated and scrubbed before they reach a caller-visible
/// `Display` or log line.
pub(crate) const MAX_REASON_BYTES: usize = 512;

/// Bound an outbound reason string (`AI-0059`).
///
/// Every character outside printable ASCII (`0x20..=0x7E`, so spaces are
/// kept) becomes `?`, which defuses CR/LF log injection, terminal escapes,
/// and confusable non-ASCII bytes; the result is then truncated to
/// [`MAX_REASON_BYTES`] bytes. The transformation is deterministic, silent
/// (no ellipsis marker that could itself exceed the bound), and never
/// splits a character, so callers can log the value directly.
///
/// [`crate::reconcile::bound_reason`] stays a separate truncate-only helper
/// for retry bookkeeping; this function is the scrub-aware bound for values
/// leaving the runtime through an error or decision surface.
pub(crate) fn bound_reason(reason: &str) -> String {
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

/// Allowed bytes in a [`FakeConsentLedger`] scope (`AI-0059`).
///
/// Scopes are lowercase dotted-underscore wire vocabulary today
/// (`workspace.read`, `terminal.inspect`); the allowlist keeps the fake
/// table free of whitespace, control bytes, quotes, and case-confusable
/// variants. Real host ledgers still own their own scope validation.
fn is_scope_byte(byte: u8) -> bool {
    byte.is_ascii_lowercase()
        || byte.is_ascii_digit()
        || byte == b'_'
        || byte == b'.'
        || byte == b'-'
}

/// Bridge identity and consent-seam errors. Every variant fails closed with
/// no partial state: no binding is stored, no grant is recorded, no dispatch
/// is implied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// Wire protocol id violates the `owner.name` shape.
    InvalidProtocolId {
        /// Rejected wire string.
        id: String,
        /// Why it was rejected.
        reason: String,
    },
    /// A second distinct binding was attempted on a single-agent bridge.
    AlreadyBound {
        /// Currently bound protocol principal.
        existing: String,
        /// Rejected protocol principal.
        rejected: String,
    },
    /// No binding exists for this protocol principal.
    UnknownProtocol {
        /// Requested wire string.
        protocol: String,
    },
    /// No binding exists for this runtime instance.
    UnknownInstance {
        /// Requested instance handle.
        instance: u64,
    },
    /// A consent grant request was malformed.
    InvalidGrant {
        /// Why the grant was refused.
        reason: String,
    },
    /// The fake grant table is at capacity.
    LedgerFull {
        /// Bound.
        limit: usize,
    },
    /// Consent was checked and denied (no grant, expired, or revoked).
    ConsentDenied {
        /// Tool the check was for.
        tool: String,
        /// Hook reason.
        reason: String,
    },
}

impl Display for BridgeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidProtocolId { id, reason } => {
                write!(f, "invalid protocol id '{id}': {reason}")
            }
            Self::AlreadyBound { existing, rejected } => write!(
                f,
                "bridge already bound to '{existing}'; rejected '{rejected}' (single-agent)"
            ),
            Self::UnknownProtocol { protocol } => {
                write!(f, "unknown protocol principal '{protocol}'")
            }
            Self::UnknownInstance { instance } => {
                write!(f, "unknown agent instance '{instance}'")
            }
            Self::InvalidGrant { reason } => write!(f, "invalid consent grant: {reason}"),
            Self::LedgerFull { limit } => write!(f, "consent ledger full at {limit} grants"),
            Self::ConsentDenied { tool, reason } => {
                write!(f, "consent denied for '{tool}': {reason}")
            }
        }
    }
}

impl std::error::Error for BridgeError {}

/// Validate a wire-level protocol principal (`owner.name`).
///
/// The shape mirrors the generic `bitty-agent` wire grammar exactly so this
/// crate never imports that type: non-empty, `<= 128` bytes, no whitespace,
/// exactly one `.`, each segment non-empty, `<= 64` bytes, starting with
/// `[a-z]`, remainder `[a-z0-9_-]`.
///
/// # Errors
///
/// Returns [`BridgeError::InvalidProtocolId`] when the shape is violated.
pub fn validate_protocol_id(raw: &str) -> Result<(), BridgeError> {
    if raw.is_empty() {
        return Err(BridgeError::InvalidProtocolId {
            id: bound_reason(raw),
            reason: "protocol id must not be empty".to_owned(),
        });
    }
    if raw.len() > MAX_PROTOCOL_ID_LEN {
        return Err(BridgeError::InvalidProtocolId {
            id: bound_reason(raw),
            reason: format!("protocol id too long (max {MAX_PROTOCOL_ID_LEN})"),
        });
    }
    if raw.chars().any(|c| c.is_whitespace()) {
        return Err(BridgeError::InvalidProtocolId {
            id: bound_reason(raw),
            reason: "protocol id must not contain whitespace".to_owned(),
        });
    }
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() != 2 {
        return Err(BridgeError::InvalidProtocolId {
            id: bound_reason(raw),
            reason: "protocol id must be exactly owner.name (one dot)".to_owned(),
        });
    }
    for segment in &parts {
        if segment.is_empty() {
            return Err(BridgeError::InvalidProtocolId {
                id: bound_reason(raw),
                reason: "protocol id segment must not be empty".to_owned(),
            });
        }
        if segment.len() > MAX_PROTOCOL_ID_SEGMENT_LEN {
            return Err(BridgeError::InvalidProtocolId {
                id: bound_reason(raw),
                reason: format!("segment too long (max {MAX_PROTOCOL_ID_SEGMENT_LEN})"),
            });
        }
        let first = segment.as_bytes()[0];
        if !first.is_ascii_lowercase() {
            return Err(BridgeError::InvalidProtocolId {
                id: bound_reason(raw),
                reason: "segment must start with lowercase letter".to_owned(),
            });
        }
        for byte in segment.bytes() {
            let ok =
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_';
            if !ok {
                return Err(BridgeError::InvalidProtocolId {
                    id: bound_reason(raw),
                    reason: "segment must be [a-z0-9_-]".to_owned(),
                });
            }
        }
    }
    Ok(())
}

/// Validated wire-level protocol principal (`owner.name`, e.g.
/// `"local.assistant"`).
///
/// This is the external principal on the generic wire protocol, held as a
/// validated string. It is intentionally distinct from the runtime-local
/// [`AgentInstanceId`] (`u64` handle owned by this crate); the mapping
/// between them lives in [`IdentityBridge`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProtocolAgentId(String);

impl ProtocolAgentId {
    /// Parse and validate a wire principal.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::InvalidProtocolId`] when the shape is violated.
    pub fn new(raw: &str) -> Result<Self, BridgeError> {
        validate_protocol_id(raw)?;
        Ok(Self(raw.to_owned()))
    }

    /// Raw wire string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Owner segment (before the dot).
    #[must_use]
    pub fn owner(&self) -> &str {
        self.0.split_once('.').map(|(a, _)| a).unwrap_or(&self.0)
    }

    /// Name segment (after the dot).
    #[must_use]
    pub fn name(&self) -> &str {
        self.0.split_once('.').map(|(_, b)| b).unwrap_or("")
    }
}

impl Display for ProtocolAgentId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(&self.0)
    }
}

/// Single-agent binding of one wire principal to one runtime instance.
///
/// `v0.1` runs a single agent, so the bridge holds at most one pair. Binding
/// is explicit and fail-closed: a second distinct pair is refused with
/// [`BridgeError::AlreadyBound`], and lookups for unbound principals or
/// instances fail with typed errors. Re-binding the identical pair is an
/// idempotent success.
///
/// # When the single-agent exception ends (`AI-0059`)
///
/// The exception is "one bridge instance serves exactly one agent", not
/// "instance ids may repeat". [`crate::session::IdIssuer`] now mints
/// process-unique ids, so two co-located agents no longer collide; what
/// remains single-agent is this type's routing shape, because one
/// `(protocol, instance)` pair cannot describe two principals. The window
/// therefore ends the moment a second [`crate::agent::Agent`] is constructed
/// in the same process: the host must then own one `IdentityBridge` per
/// agent instance, or the bridge must move to the composite
/// `(ProtocolAgentId, AgentInstanceId, SessionId)` key sketched in review
/// `S-9`. Until that migration, a second distinct pair is still refused
/// rather than silently re-routed, and
/// [`crate::session::SessionId`] adds no routing here; it is only available
/// for a future composite key.
#[derive(Debug, Default)]
pub struct IdentityBridge {
    binding: Option<(ProtocolAgentId, AgentInstanceId)>,
}

impl IdentityBridge {
    /// Empty bridge: every lookup fails closed until [`Self::bind`] stores
    /// the single-agent pair.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind one protocol principal to one runtime instance.
    ///
    /// Idempotent for the identical pair; refuses any second distinct pair
    /// so the single-agent invariant cannot silently widen.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::AlreadyBound`] when a different pair is
    /// already bound.
    pub fn bind(
        &mut self,
        protocol: ProtocolAgentId,
        instance: AgentInstanceId,
    ) -> Result<(), BridgeError> {
        if let Some((existing_protocol, existing_instance)) = &self.binding {
            if *existing_protocol == protocol && *existing_instance == instance {
                return Ok(());
            }
            return Err(BridgeError::AlreadyBound {
                existing: existing_protocol.as_str().to_owned(),
                rejected: protocol.as_str().to_owned(),
            });
        }
        self.binding = Some((protocol, instance));
        Ok(())
    }

    /// Whether a pair is bound.
    #[must_use]
    pub fn is_bound(&self) -> bool {
        self.binding.is_some()
    }

    /// Bound protocol principal, if any.
    #[must_use]
    pub fn protocol(&self) -> Option<&ProtocolAgentId> {
        self.binding.as_ref().map(|(protocol, _)| protocol)
    }

    /// Bound runtime instance, if any.
    #[must_use]
    pub fn instance(&self) -> Option<AgentInstanceId> {
        self.binding.as_ref().map(|(_, instance)| *instance)
    }

    /// Resolve a wire principal to its bound instance.
    ///
    /// Validates the wire shape first so malformed principals fail as
    /// [`BridgeError::InvalidProtocolId`] before any mapping lookup.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::InvalidProtocolId`] for a malformed principal
    /// or [`BridgeError::UnknownProtocol`] when nothing is bound for it.
    pub fn resolve_instance(&self, protocol: &str) -> Result<AgentInstanceId, BridgeError> {
        validate_protocol_id(protocol)?;
        match &self.binding {
            Some((bound_protocol, instance)) if bound_protocol.as_str() == protocol => {
                Ok(*instance)
            }
            _ => Err(BridgeError::UnknownProtocol {
                protocol: protocol.to_owned(),
            }),
        }
    }

    /// Resolve a runtime instance to its bound wire principal.
    ///
    /// # Errors
    ///
    /// Returns [`BridgeError::UnknownInstance`] when nothing is bound for it.
    pub fn resolve_protocol(
        &self,
        instance: AgentInstanceId,
    ) -> Result<&ProtocolAgentId, BridgeError> {
        match &self.binding {
            Some((protocol, bound)) if *bound == instance => Ok(protocol),
            _ => Err(BridgeError::UnknownInstance {
                instance: instance.0,
            }),
        }
    }
}

/// Per-call consent query presented to the ledger seam.
///
/// All strings are wire-level values: the protocol principal (`owner.name`),
/// the requested tool name, and the capability scope the tool declaration
/// requires. `now_ms` is caller-supplied; no wall clock is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsentQuery<'a> {
    /// Wire protocol principal making the call.
    pub protocol_id: &'a str,
    /// Runtime-local agent instance the principal is bound to.
    pub agent_instance: AgentInstanceId,
    /// Requested tool name.
    pub tool: &'a str,
    /// Capability scope the tool declaration requires.
    pub scope: &'a str,
    /// Caller-supplied timestamp; expiry is evaluated against this.
    pub now_ms: u64,
}

/// Consent verdict. There is no allow-all grant in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsentDecision {
    /// Proceed toward host dispatch (still subject to ToolBus gates).
    Allow,
    /// Refuse with no dispatch and no partial state.
    Deny {
        /// Hook reason (attributed in enforcement records).
        reason: String,
    },
}

/// Consent-ledger seam (`TB-4` / `PP-3` shape, host-owned ledger).
///
/// The host implements per-client consent checks behind this trait; the real
/// ledger lives `bitty`-side with identity, agent, tool name, grant time,
/// expiry, and granter. This crate defines only the seam plus the
/// deny-by-default posture, and claims no accepted security mechanism.
pub trait ConsentLedger {
    /// Decide one tool call at `query.now_ms`.
    ///
    /// Implementations must be deterministic for a given grant table plus
    /// `now_ms` when used under test. Revocation takes effect on the next
    /// check: there is no caching across calls.
    fn check(&self, query: &ConsentQuery) -> ConsentDecision;
}

/// Ledger seam that denies every call (default posture).
///
/// Install this until the host wires the real ledger. It never inspects the
/// query beyond attributing the denial to the requested tool.
#[derive(Debug, Default)]
pub struct DenyAllConsent;

impl ConsentLedger for DenyAllConsent {
    fn check(&self, query: &ConsentQuery) -> ConsentDecision {
        ConsentDecision::Deny {
            reason: bound_reason(&format!("default-deny: no grant for {}", query.tool)),
        }
    }
}

/// One fake grant entry: an exact `(protocol, tool, scope)` triple valid
/// while `now_ms < expires_at_ms`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FakeGrant {
    protocol: String,
    tool: String,
    scope: String,
    expires_at_ms: u64,
}

/// Deterministic test double for [`ConsentLedger`].
///
/// Grants are exact-match triples with absolute expiry. A check allows only
/// when one entry matches all three strings and `now_ms` precedes its
/// expiry; anything else denies. Refreshing an existing triple updates its
/// expiry; a new triple past [`MAX_CONSENT_GRANTS`] fails closed. Revocation
/// removes the triple immediately. Expiry is evaluated against the
/// caller-supplied `now_ms`, never a wall clock, so `FakeProvider`-driven
/// turns stay reproducible.
///
/// This double is test-only: production wiring implements [`ConsentLedger`]
/// over the host ledger instead.
#[derive(Debug, Default)]
pub struct FakeConsentLedger {
    grants: Vec<FakeGrant>,
}

impl FakeConsentLedger {
    /// Empty ledger: every check denies until [`Self::grant`] records one.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an exact-match grant valid until `expires_at_ms`.
    ///
    /// Refreshing an existing `(protocol, tool, scope)` triple updates its
    /// expiry without growing the table.
    ///
    /// # Errors
    ///
    /// - [`BridgeError::InvalidProtocolId`] when `protocol` violates the
    ///   `owner.name` shape.
    /// - [`BridgeError::InvalidGrant`] when `tool` or `scope` is empty,
    ///   over-bound, or carries an interior NUL, when `scope` leaves the
    ///   lowercase `[a-z0-9_.-]` allowlist, or when `tool` violates the
    ///   runtime `TB-2` name shape.
    /// - [`BridgeError::LedgerFull`] when a new triple would exceed
    ///   [`MAX_CONSENT_GRANTS`].
    pub fn grant(
        &mut self,
        protocol: &str,
        tool: &str,
        scope: &str,
        expires_at_ms: u64,
    ) -> Result<(), BridgeError> {
        validate_protocol_id(protocol)?;
        if tool.is_empty() || scope.is_empty() {
            return Err(BridgeError::InvalidGrant {
                reason: "tool and scope must not be empty".to_owned(),
            });
        }
        if tool.contains('\0') || scope.contains('\0') {
            return Err(BridgeError::InvalidGrant {
                reason: "tool and scope must not contain NUL".to_owned(),
            });
        }
        if let Err(error) = crate::tool::validate_tool_name(tool) {
            return Err(BridgeError::InvalidGrant {
                reason: bound_reason(&error.to_string()),
            });
        }
        if scope.len() > MAX_CONSENT_SCOPE_LEN {
            return Err(BridgeError::InvalidGrant {
                reason: format!("scope too long (max {MAX_CONSENT_SCOPE_LEN})"),
            });
        }
        if !scope.bytes().all(is_scope_byte) {
            return Err(BridgeError::InvalidGrant {
                reason: "scope must be [a-z0-9_.-]".to_owned(),
            });
        }
        if let Some(existing) = self
            .grants
            .iter_mut()
            .find(|grant| grant.protocol == protocol && grant.tool == tool && grant.scope == scope)
        {
            existing.expires_at_ms = expires_at_ms;
            return Ok(());
        }
        if self.grants.len() >= MAX_CONSENT_GRANTS {
            return Err(BridgeError::LedgerFull {
                limit: MAX_CONSENT_GRANTS,
            });
        }
        self.grants.push(FakeGrant {
            protocol: protocol.to_owned(),
            tool: tool.to_owned(),
            scope: scope.to_owned(),
            expires_at_ms,
        });
        Ok(())
    }

    /// Revoke one exact triple immediately. Returns `true` when an entry was
    /// present.
    pub fn revoke(&mut self, protocol: &str, tool: &str, scope: &str) -> bool {
        let before = self.grants.len();
        self.grants.retain(|grant| {
            !(grant.protocol == protocol && grant.tool == tool && grant.scope == scope)
        });
        self.grants.len() != before
    }

    /// Stored grant count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Whether no grant is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }

    /// Drop entries expired at `now_ms`, returning their triples.
    pub fn drain_expired(&mut self, now_ms: u64) -> Vec<(String, String, String)> {
        let mut expired = Vec::new();
        self.grants.retain(|grant| {
            if now_ms >= grant.expires_at_ms {
                expired.push((
                    grant.protocol.clone(),
                    grant.tool.clone(),
                    grant.scope.clone(),
                ));
                false
            } else {
                true
            }
        });
        expired
    }
}

/// Grant one turn-scoped batch of exact `(tool, scope)` triples (AI-0255).
///
/// One user gesture grants `N` pairs as `N` exact ledger triples, never a
/// wildcard: the loop calls [`FakeConsentLedger::grant`] once per pair with
/// `expires_at_ms = now_ms.saturating_add(ttl)`, where `ttl` is
/// `ttl_ms.unwrap_or(DEFAULT_CONSENT_TTL_MS)`. The ledger stores `N` exact
/// `(protocol, tool, scope)` entries; a grant for `tool_a` never admits
/// `tool_b` and a grant for `scope.a` never admits `scope.b`. Refreshing an
/// existing triple updates its expiry without growing the table; a new
/// triple past [`MAX_CONSENT_GRANTS`] fails closed with
/// [`BridgeError::LedgerFull`].
///
/// No wall-clock read: expiry derives from caller-supplied `now_ms` only,
/// so `FakeProvider`-driven turns stay reproducible. The TTL boundary is
/// exclusive (`now_ms < expires_at_ms` allows, `now_ms == expires_at_ms`
/// denies), matching the host ledger rule pinned in
/// [`FakeConsentLedger::check`]. Revocation stays immediate through
/// [`FakeConsentLedger::revoke`]: the next [`check`](ConsentLedger::check)
/// denies.
///
/// The return value is the display list for the solicitation surface: one
/// `(tool, scope, expires_at_ms)` tuple per granted pair, in caller order,
/// so the host renders every triple with its tool, scope, and expiry. There
/// is no hidden batch: a caller that does not render every entry violates
/// the consent UX contract.
///
/// Declined (do not extend this helper): persistent or session-scoped
/// grants, wildcard or prefix grants, trusted-tool bypass, auto-approve, and
/// any silent grant that bypasses the per-triple render.
///
/// # Errors
///
/// - [`BridgeError::InvalidProtocolId`] when `protocol` violates the
///   `owner.name` shape.
/// - [`BridgeError::InvalidGrant`] when any `tool` or `scope` is malformed
///   (empty, NUL, scope outside `[a-z0-9_.-]`, tool outside the `TB-2`
///   shape, or scope over [`MAX_CONSENT_SCOPE_LEN`]).
/// - [`BridgeError::LedgerFull`] when a new triple would exceed
///   [`MAX_CONSENT_GRANTS`]. Earlier pairs in this batch stay granted
///   (exact triples, no rollback, no wildcard); the caller decides whether
///   to revoke them or keep the partial batch.
pub fn grant_turn_set(
    ledger: &mut FakeConsentLedger,
    protocol: &str,
    grants: &[(&str, &str)],
    now_ms: u64,
    ttl_ms: Option<u64>,
) -> Result<Vec<(String, String, u64)>, BridgeError> {
    let ttl = ttl_ms.unwrap_or(DEFAULT_CONSENT_TTL_MS);
    let expires_at = now_ms.saturating_add(ttl);
    let mut rendered = Vec::with_capacity(grants.len());
    for (tool, scope) in grants {
        ledger.grant(protocol, tool, scope, expires_at)?;
        rendered.push(((*tool).to_owned(), (*scope).to_owned(), expires_at));
    }
    Ok(rendered)
}

impl ConsentLedger for FakeConsentLedger {
    fn check(&self, query: &ConsentQuery) -> ConsentDecision {
        let allowed = self.grants.iter().any(|grant| {
            grant.protocol == query.protocol_id
                && grant.tool == query.tool
                && grant.scope == query.scope
                && query.now_ms < grant.expires_at_ms
        });
        if allowed {
            ConsentDecision::Allow
        } else {
            ConsentDecision::Deny {
                reason: bound_reason(&format!(
                    "no live consent for '{}' on '{}' (tool {})",
                    query.protocol_id, query.scope, query.tool
                )),
            }
        }
    }
}

/// Require consent for `query`, converting a denial into a typed error.
///
/// This is the fail-closed helper hosts call after [`IdentityBridge`]
/// resolution and before [`crate::tool::ToolBus`] dispatch: allowed checks
/// return `Ok(())`, denied checks return [`BridgeError::ConsentDenied`] with
/// no partial state. The denial is a runtime boundary (`AI-0059`): both the
/// queried tool name and a host ledger's reason are truncated to the
/// runtime reason bound and scrubbed to printable ASCII before they reach
/// the error.
///
/// # Errors
///
/// Returns [`BridgeError::ConsentDenied`] when the ledger denies.
pub fn ensure_consented(
    ledger: &dyn ConsentLedger,
    query: &ConsentQuery,
) -> Result<(), BridgeError> {
    match ledger.check(query) {
        ConsentDecision::Allow => Ok(()),
        ConsentDecision::Deny { reason } => Err(BridgeError::ConsentDenied {
            tool: bound_reason(query.tool),
            reason: bound_reason(&reason),
        }),
    }
}

/// Enforce the consent gate for a dispatch path and report denial as the
/// shared dispatch-path refusal (AI-0227, AIQ-33 S2).
///
/// ONE helper for both dispatch paths: after the authorizer allows, the
/// native [`crate::tool::ToolBus`] dispatch path and the MCP adapter execute
/// path both call this with the exact `(protocol, tool, scope)` triple at
/// `now_ms`. Allow returns `Ok(())`; denial returns
/// [`crate::tool::ToolError::Denied`] (never `Failed`/`Unknown`/
/// `ProtocolRejected`, no new variant) with a byte-identical reason on both
/// paths, before any executor or transport contact. The reason keeps the
/// established `consent denied: ...` wording (the same mapping the MCP
/// adapter applied inline before this slice), so existing `Denied`-reason
/// pins keep matching on both paths.
///
/// No new types: this reuses [`ConsentLedger::check`], [`ConsentQuery`], and
/// the [`BridgeError::ConsentDenied`] mapping from [`ensure_consented`],
/// adding only the dispatch-path [`crate::tool::ToolError::Denied`]
/// rendering both paths share.
///
/// # Errors
///
/// Returns [`crate::tool::ToolError::Denied`] when the ledger denies.
pub fn ensure_consented_for_dispatch(
    ledger: &dyn ConsentLedger,
    query: &ConsentQuery,
) -> Result<(), crate::tool::ToolError> {
    match ensure_consented(ledger, query) {
        Ok(()) => Ok(()),
        Err(error) => Err(crate::tool::ToolError::Denied {
            name: bound_reason(query.tool),
            reason: bound_reason(&format!("consent denied: {error}")),
        }
        .normalized()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance(value: u64) -> AgentInstanceId {
        AgentInstanceId(value)
    }

    fn query<'a>(
        protocol: &'a str,
        tool: &'a str,
        scope: &'a str,
        now_ms: u64,
    ) -> ConsentQuery<'a> {
        ConsentQuery {
            protocol_id: protocol,
            agent_instance: instance(1),
            tool,
            scope,
            now_ms,
        }
    }

    #[test]
    fn protocol_id_accepts_wire_shape() {
        for id in ["local.assistant", "bitty.agent-1", "x.y_z-0", "a.b"] {
            assert!(validate_protocol_id(id).is_ok(), "must accept {id}");
            let parsed = ProtocolAgentId::new(id).expect("valid wire id");
            assert_eq!(parsed.as_str(), id);
        }
        let parsed = ProtocolAgentId::new("local.assistant").expect("valid");
        assert_eq!(parsed.owner(), "local");
        assert_eq!(parsed.name(), "assistant");
        assert_eq!(parsed.to_string(), "local.assistant");
    }

    #[test]
    fn protocol_id_rejects_malformed_wire_strings() {
        for id in [
            "",
            "local",
            "local.assistant.extra",
            "Local.assistant",
            "local.Assistant",
            "1local.assistant",
            "local.1assistant",
            ".assistant",
            "local.",
            "local.assistant ",
            "local .assistant",
        ] {
            assert!(
                matches!(
                    validate_protocol_id(id),
                    Err(BridgeError::InvalidProtocolId { .. })
                ),
                "must reject {id:?}"
            );
            assert!(ProtocolAgentId::new(id).is_err(), "must reject {id:?}");
        }
    }

    #[test]
    fn protocol_id_rejects_over_bound_wire_strings() {
        let long = format!("{}.{}", "a".repeat(64), "b".repeat(64));
        assert_eq!(long.len(), 129);
        assert!(validate_protocol_id(&long).is_err());
        let segment = format!("{}.b", "a".repeat(65));
        assert!(validate_protocol_id(&segment).is_err());
        let ok = format!("{}.{}", "a".repeat(63), "b".repeat(64));
        assert_eq!(ok.len(), 128);
        assert!(validate_protocol_id(&ok).is_ok());
    }

    #[test]
    fn protocol_id_rejects_disallowed_bytes() {
        for id in [
            "local.assistant!",
            "local.assistant\n",
            "local.ass\0istant",
            "local.assistant\t",
            "lo/cal.assistant",
            "local.assistant?",
        ] {
            assert!(validate_protocol_id(id).is_err(), "must reject {id:?}");
        }
    }

    #[test]
    fn bridge_binds_single_pair_and_resolves_both_directions() {
        let mut bridge = IdentityBridge::new();
        assert!(!bridge.is_bound());
        assert!(bridge.resolve_instance("local.assistant").is_err());
        let protocol = ProtocolAgentId::new("local.assistant").expect("valid");
        bridge
            .bind(protocol.clone(), instance(7))
            .expect("first bind serves");
        assert!(bridge.is_bound());
        assert_eq!(bridge.resolve_instance("local.assistant"), Ok(instance(7)));
        assert_eq!(
            bridge
                .resolve_protocol(instance(7))
                .expect("reverse resolves"),
            &protocol
        );
        assert_eq!(bridge.protocol().expect("protocol accessor"), &protocol);
        assert_eq!(bridge.instance(), Some(instance(7)));
    }

    #[test]
    fn bridge_rebind_of_identical_pair_is_idempotent() {
        let mut bridge = IdentityBridge::new();
        let protocol = ProtocolAgentId::new("local.assistant").expect("valid");
        bridge
            .bind(protocol.clone(), instance(1))
            .expect("first bind");
        bridge
            .bind(protocol, instance(1))
            .expect("identical rebind is idempotent");
        assert_eq!(bridge.resolve_instance("local.assistant"), Ok(instance(1)));
    }

    #[test]
    fn bridge_rejects_second_distinct_binding() {
        let mut bridge = IdentityBridge::new();
        bridge
            .bind(
                ProtocolAgentId::new("local.assistant").expect("valid"),
                instance(1),
            )
            .expect("first bind");
        let error = bridge
            .bind(
                ProtocolAgentId::new("other.agent").expect("valid"),
                instance(2),
            )
            .expect_err("second distinct protocol must fail");
        assert!(matches!(error, BridgeError::AlreadyBound { .. }));
        // No partial state: the first binding still resolves.
        assert_eq!(bridge.resolve_instance("local.assistant"), Ok(instance(1)));
        assert!(bridge.resolve_instance("other.agent").is_err());
        // Same protocol but different instance is also a distinct pair.
        let error = bridge
            .bind(
                ProtocolAgentId::new("local.assistant").expect("valid"),
                instance(99),
            )
            .expect_err("same protocol new instance must fail");
        assert!(matches!(error, BridgeError::AlreadyBound { .. }));
    }

    #[test]
    fn bridge_lookups_are_fail_closed() {
        let mut bridge = IdentityBridge::new();
        bridge
            .bind(
                ProtocolAgentId::new("local.assistant").expect("valid"),
                instance(3),
            )
            .expect("bind");
        assert_eq!(
            bridge.resolve_instance("other.agent").expect_err("unknown"),
            BridgeError::UnknownProtocol {
                protocol: "other.agent".to_owned(),
            }
        );
        assert_eq!(
            bridge
                .resolve_protocol(instance(99))
                .expect_err("unknown instance"),
            BridgeError::UnknownInstance { instance: 99 }
        );
        // Malformed wire strings fail as invalid before mapping lookup.
        assert!(matches!(
            bridge.resolve_instance("not-a-principal"),
            Err(BridgeError::InvalidProtocolId { .. })
        ));
        assert!(matches!(
            bridge.resolve_instance(""),
            Err(BridgeError::InvalidProtocolId { .. })
        ));
    }

    #[test]
    fn deny_all_consent_denies_every_tool() {
        let ledger = DenyAllConsent;
        for tool in ["workspace_read", "workspace_write", "terminal_read_zone"] {
            let decision = ledger.check(&query("local.assistant", tool, "workspace.read", 1_000));
            assert_eq!(
                decision,
                ConsentDecision::Deny {
                    reason: format!("default-deny: no grant for {tool}"),
                }
            );
            assert!(matches!(
                ensure_consented(
                    &ledger,
                    &query("local.assistant", tool, "workspace.read", 1_000)
                ),
                Err(BridgeError::ConsentDenied { .. })
            ));
        }
    }

    /// Assert a reason is a bounded, printable-ASCII string (no CR/LF, no
    /// escape bytes, no non-ASCII confusables).
    fn assert_bounded_reason(value: &str) {
        assert!(
            value.len() <= MAX_REASON_BYTES,
            "reason exceeds {MAX_REASON_BYTES} bytes: {}",
            value.len()
        );
        assert!(
            value.bytes().all(|byte| (0x20..=0x7E).contains(&byte)),
            "reason is not printable ASCII: {value:?}"
        );
    }

    #[test]
    fn bound_reason_truncates_on_the_byte_bound_and_scrubs_controls() {
        let long = "e".repeat(MAX_REASON_BYTES + 64);
        let bounded = bound_reason(&long);
        assert_eq!(bounded.len(), MAX_REASON_BYTES);
        assert!(bounded.bytes().all(|byte| byte == b'e'));

        // CR/LF, tab, ESC, and DEL all become `?`; printable text survives.
        let hostile = "line1\r\nline2\ttab\u{1b}[31mred\u{7f}";
        assert_eq!(bound_reason(hostile), "line1??line2?tab?[31mred?");

        // Each multi-byte character collapses to one ASCII byte, so the byte
        // bound holds for dense UTF-8 input too.
        let unicode = "🦀".repeat(MAX_REASON_BYTES);
        assert_eq!(bound_reason(&unicode).len(), MAX_REASON_BYTES);
    }

    #[test]
    fn deny_all_consent_bounds_hostile_tool_reason() {
        let ledger = DenyAllConsent;
        let tool = format!("bad\n{}\u{7f}", "t".repeat(MAX_REASON_BYTES * 2));
        match ledger.check(&query("local.assistant", &tool, "workspace.read", 1_000)) {
            ConsentDecision::Deny { reason } => {
                assert_bounded_reason(&reason);
                assert!(reason.starts_with("default-deny: no grant for bad?"));
            }
            ConsentDecision::Allow => panic!("deny-all must deny"),
        }
    }

    #[test]
    fn fake_consent_denial_bounds_hostile_query_strings() {
        let ledger = FakeConsentLedger::new();
        let tool = format!("tool\n{}", "\r".repeat(64));
        let scope = format!("scope\r{}", "\n".repeat(64));
        match ledger.check(&query("local.assistant", &tool, &scope, 1_000)) {
            ConsentDecision::Deny { reason } => assert_bounded_reason(&reason),
            ConsentDecision::Allow => panic!("empty ledger must deny"),
        }
    }

    #[test]
    fn ensure_consented_bounds_host_ledger_reason() {
        struct HostileLedger;
        impl ConsentLedger for HostileLedger {
            fn check(&self, _query: &ConsentQuery) -> ConsentDecision {
                ConsentDecision::Deny {
                    reason: format!("boom\n\u{1b}[2J{}", "x".repeat(MAX_REASON_BYTES * 2)),
                }
            }
        }

        let error = ensure_consented(
            &HostileLedger,
            &query("local.assistant", "workspace_read", "workspace.read", 1_000),
        )
        .expect_err("denying ledger must fail closed");
        match error {
            BridgeError::ConsentDenied { tool, reason } => {
                assert_eq!(tool, "workspace_read");
                assert_bounded_reason(&reason);
                assert!(reason.starts_with("boom?"));
            }
            other => panic!("expected ConsentDenied, got {other:?}"),
        }
    }

    #[test]
    fn ensure_consented_bounds_hostile_tool_echo() {
        let hostile = format!("tool\n{}", "x".repeat(MAX_REASON_BYTES * 2));
        let error = ensure_consented(
            &DenyAllConsent,
            &query("local.assistant", &hostile, "workspace.read", 1_000),
        )
        .expect_err("deny-all must fail closed");
        match error {
            BridgeError::ConsentDenied { tool, reason } => {
                assert_bounded_reason(&tool);
                assert_bounded_reason(&reason);
                assert!(tool.starts_with("tool?"));
            }
            other => panic!("expected ConsentDenied, got {other:?}"),
        }
    }

    #[test]
    fn invalid_protocol_id_echo_is_bounded_and_scrubbed() {
        let hostile = format!("bad\n{}\u{7f}", "X".repeat(MAX_REASON_BYTES * 2));
        match validate_protocol_id(&hostile) {
            Err(BridgeError::InvalidProtocolId { id, .. }) => {
                assert_bounded_reason(&id);
                assert!(id.starts_with("bad?"));
            }
            other => panic!("expected InvalidProtocolId, got {other:?}"),
        }
    }

    #[test]
    fn fake_grant_scope_charset_rejects_hostile_bytes() {
        let mut ledger = FakeConsentLedger::new();
        for scope in [
            "workspace read",
            "workspace\tread",
            "workspace\nread",
            "workspace\rread",
            "workspace\0read",
            "Workspace.Read",
            "workspace.read!",
            "workspace.read\u{7f}",
            "workspace.réad",
            "workspace/read",
            "workspace.read;drop",
            "workspace.read\u{1b}[31m",
        ] {
            let error = ledger
                .grant("local.assistant", "workspace_read", scope, 9)
                .expect_err("hostile scope must be rejected");
            assert!(
                matches!(error, BridgeError::InvalidGrant { .. }),
                "scope {scope:?} must fail as InvalidGrant"
            );
        }
        assert!(ledger.is_empty());
        // The documented lowercase dotted vocabulary still enters the table.
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 9)
            .expect("lowercase dotted scope");
        ledger
            .grant("local.assistant", "workspace_read", "scope.a_b-c9", 9)
            .expect("allowlisted punctuation");
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn fake_grant_allows_exact_triple_before_expiry() {
        let mut ledger = FakeConsentLedger::new();
        assert!(ledger.is_empty());
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 2_000)
            .expect("grant fits");
        assert_eq!(ledger.len(), 1);
        assert_eq!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                1_000
            )),
            ConsentDecision::Allow
        );
        ensure_consented(
            &ledger,
            &query("local.assistant", "workspace_read", "workspace.read", 1_000),
        )
        .expect("live grant allows");
        // Boundary: expiry is exclusive, matching the host ledger rule.
        assert!(matches!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                2_000
            )),
            ConsentDecision::Deny { .. }
        ));
    }

    #[test]
    fn fake_consent_is_exact_match_per_field() {
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                10_000,
            )
            .expect("grant");
        // Each field mismatch denies: protocol, tool, and scope are
        // independent deny reasons (no cross-tool or cross-scope sharing).
        for (protocol, tool, scope) in [
            ("other.agent", "workspace_read", "workspace.read"),
            ("local.assistant", "workspace_write", "workspace.read"),
            ("local.assistant", "workspace_read", "workspace.write"),
        ] {
            assert!(
                matches!(
                    ledger.check(&query(protocol, tool, scope, 1_000)),
                    ConsentDecision::Deny { .. }
                ),
                "must deny {protocol}/{tool}/{scope}"
            );
        }
    }

    #[test]
    fn fake_grant_refresh_updates_expiry_without_growth() {
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 2_000)
            .expect("grant");
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 9_000)
            .expect("refresh");
        assert_eq!(ledger.len(), 1);
        assert_eq!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                5_000
            )),
            ConsentDecision::Allow
        );
    }

    #[test]
    fn fake_revoke_and_expiry_deny_and_drain() {
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 2_000)
            .expect("grant");
        assert!(ledger.revoke("local.assistant", "workspace_read", "workspace.read"));
        assert!(ledger.is_empty());
        assert!(matches!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                1_000
            )),
            ConsentDecision::Deny { .. }
        ));
        assert!(!ledger.revoke("local.assistant", "workspace_read", "workspace.read"));

        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 2_000)
            .expect("re-grant");
        let expired = ledger.drain_expired(2_000);
        assert_eq!(
            expired,
            vec![(
                "local.assistant".to_owned(),
                "workspace_read".to_owned(),
                "workspace.read".to_owned()
            )]
        );
        assert!(ledger.is_empty());
    }

    #[test]
    fn fake_grant_rejects_malformed_inputs() {
        let mut ledger = FakeConsentLedger::new();
        assert!(
            ledger
                .grant("not-a-principal", "workspace_read", "s", 9)
                .is_err()
        );
        assert!(
            ledger
                .grant("local.assistant", "", "workspace.read", 9)
                .is_err()
        );
        assert!(
            ledger
                .grant("local.assistant", "workspace_read", "", 9)
                .is_err()
        );
        // Runtime TB-2 tool shape still applies at the seam: dotted legacy
        // vocabulary never enters the fake table.
        assert!(
            ledger
                .grant("local.assistant", "terminal.read_zone", "s", 9)
                .is_err()
        );
        assert!(ledger.is_empty());
    }

    #[test]
    fn fake_grant_tool_error_reason_is_bounded_and_scrubbed() {
        let mut ledger = FakeConsentLedger::new();
        // The TB-2 rejection interpolates the raw tool string, so a hostile
        // name must not reach the public `InvalidGrant.reason` unbounded.
        let tool = format!("bad\n{}\u{7f}\u{1b}[2J", "N".repeat(MAX_REASON_BYTES * 2));
        match ledger.grant("local.assistant", &tool, "workspace.read", 9) {
            Err(BridgeError::InvalidGrant { reason }) => {
                assert_bounded_reason(&reason);
                assert!(reason.starts_with("invalid tool name: bad?"));
            }
            other => panic!("expected InvalidGrant, got {other:?}"),
        }
        assert!(ledger.is_empty());
    }

    #[test]
    fn fake_ledger_is_bounded_fail_closed() {
        let mut ledger = FakeConsentLedger::new();
        for index in 0..MAX_CONSENT_GRANTS {
            ledger
                .grant(
                    "local.assistant",
                    &format!("tool_{index}"),
                    &format!("scope.{index}"),
                    99_000,
                )
                .expect("capacity");
        }
        assert_eq!(ledger.len(), MAX_CONSENT_GRANTS);
        let error = ledger
            .grant("local.assistant", "one_more", "scope.extra", 99_000)
            .expect_err("overflow must fail");
        assert_eq!(
            error,
            BridgeError::LedgerFull {
                limit: MAX_CONSENT_GRANTS
            }
        );
        assert_eq!(ledger.len(), MAX_CONSENT_GRANTS);
    }

    #[test]
    fn bridge_plus_consent_compose_before_fake_provider_turn() {
        // Seam composition smoke: resolve the wire principal, require live
        // consent, then run one deterministic FakeProvider turn. The bridge
        // never executes a tool; it only gates the turn that does.
        use crate::provider::{FakeProvider, ModelProvider, ProviderTurn, ProviderUsage};
        use crate::tool::{AuthBase, ToolBus, ToolPlacement, ToolRegistry, ToolSpec};

        let mut ids = crate::session::IdIssuer::default();
        let bound_instance = ids.agent_instance();
        let mut bridge = IdentityBridge::new();
        bridge
            .bind(
                ProtocolAgentId::new("local.assistant").expect("valid"),
                bound_instance,
            )
            .expect("bind");
        let resolved = bridge
            .resolve_instance("local.assistant")
            .expect("resolves");
        assert_eq!(resolved, bound_instance);

        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 9_000)
            .expect("grant");
        let auth_base = AuthBase {
            agent_instance_id: resolved,
            session_id: ids.session(),
            level: crate::session::AgentLevel::Workspace,
        };
        let _ = auth_base;
        ensure_consented(
            &ledger,
            &ConsentQuery {
                protocol_id: "local.assistant",
                agent_instance: resolved,
                tool: "workspace_read",
                scope: "workspace.read",
                now_ms: 1_000,
            },
        )
        .expect("consented turn may proceed");

        // FakeProvider stays the deterministic model peer behind the gate.
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(ProviderTurn {
            text: "gated answer".to_owned(),
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
        });
        let request = crate::provider::TurnRequest {
            model: "fake-chat".to_owned(),
            messages: vec![crate::provider::Message::user("hi")],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: 4096,
            timeout_ms: 5_000,
            now_ms: 1_000,
            sampling: None,
        };
        let turn = provider.complete(&request).expect("scripted turn");
        assert_eq!(turn.text, "gated answer");

        // Denied consent blocks before any provider or tool I/O.
        let denied = ledger.check(&query(
            "local.assistant",
            "workspace_write",
            "workspace.read",
            1_000,
        ));
        assert!(matches!(denied, ConsentDecision::Deny { .. }));

        // Registry shape referenced so the composition stays honest about
        // the ToolBus vocabulary it gates (no dispatch happens here).
        let registry = ToolRegistry::new();
        let _bus = ToolBus::new(registry);
        let _spec = ToolSpec::new(
            "workspace_read",
            "Read a bounded workspace path",
            br#"{"type":"object"}"#.to_vec(),
            "workspace.read",
            true,
            ToolPlacement::Mcp,
            None,
        )
        .expect("valid spec");
    }

    #[test]
    fn default_consent_ttl_is_within_documented_window() {
        assert!(
            (60_000..=300_000).contains(&DEFAULT_CONSENT_TTL_MS),
            "default must stay in the 60-300s solicitation window"
        );
        assert_eq!(DEFAULT_CONSENT_TTL_MS, 120_000);
    }

    #[test]
    fn grant_turn_set_stores_exact_triples_not_a_wildcard() {
        let mut ledger = FakeConsentLedger::new();
        let rendered = grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("workspace_read", "workspace.read"), ("tool_a", "scope.a")],
            1_000,
            Some(5_000),
        )
        .expect("batch fits");
        assert_eq!(ledger.len(), 2);
        // Display list renders every triple with tool, scope, and expiry.
        assert_eq!(
            rendered,
            vec![
                (
                    "workspace_read".to_owned(),
                    "workspace.read".to_owned(),
                    6_000
                ),
                ("tool_a".to_owned(), "scope.a".to_owned(), 6_000),
            ]
        );
        // Each exact triple allows at its own scope.
        assert_eq!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                2_000
            )),
            ConsentDecision::Allow
        );
        assert_eq!(
            ledger.check(&query("local.assistant", "tool_a", "scope.a", 2_000)),
            ConsentDecision::Allow
        );
        // No wildcard: cross-tool and cross-scope combinations still deny.
        for (tool, scope) in [
            ("workspace_read", "scope.a"),
            ("tool_a", "workspace.read"),
            ("workspace_write", "workspace.read"),
            ("workspace_read", "workspace.write"),
        ] {
            assert!(
                matches!(
                    ledger.check(&query("local.assistant", tool, scope, 2_000)),
                    ConsentDecision::Deny { .. }
                ),
                "must deny {tool}/{scope}"
            );
        }
    }

    #[test]
    fn grant_turn_set_defaults_ttl_without_wall_clock() {
        let mut ledger = FakeConsentLedger::new();
        let rendered = grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("workspace_read", "workspace.read")],
            10_000,
            None,
        )
        .expect("default ttl");
        let expected_expiry = 10_000 + DEFAULT_CONSENT_TTL_MS;
        assert_eq!(
            rendered,
            vec![(
                "workspace_read".to_owned(),
                "workspace.read".to_owned(),
                expected_expiry
            )]
        );
        // Deterministic from caller now_ms only: same inputs give same expiry.
        let mut second = FakeConsentLedger::new();
        let rerendered = grant_turn_set(
            &mut second,
            "local.assistant",
            &[("workspace_read", "workspace.read")],
            10_000,
            None,
        )
        .expect("reproducible");
        assert_eq!(rendered, rerendered);
        assert_eq!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                expected_expiry - 1
            )),
            ConsentDecision::Allow
        );
    }

    #[test]
    fn grant_turn_set_ttl_boundary_is_exclusive() {
        let mut ledger = FakeConsentLedger::new();
        grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("workspace_read", "workspace.read")],
            1_000,
            Some(1_000),
        )
        .expect("batch");
        // expires_at = 2_000: live just before, denied exactly at expiry.
        assert_eq!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                1_999
            )),
            ConsentDecision::Allow
        );
        assert!(matches!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                2_000
            )),
            ConsentDecision::Deny { .. }
        ));
    }

    #[test]
    fn grant_turn_set_refresh_updates_expiry_without_growth() {
        let mut ledger = FakeConsentLedger::new();
        grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("workspace_read", "workspace.read")],
            1_000,
            Some(1_000),
        )
        .expect("first batch");
        assert_eq!(ledger.len(), 1);
        grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("workspace_read", "workspace.read")],
            1_500,
            Some(9_000),
        )
        .expect("refresh batch");
        assert_eq!(ledger.len(), 1);
        // Old expiry would have denied at 2_000; refreshed expiry allows.
        assert_eq!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                5_000
            )),
            ConsentDecision::Allow
        );
    }

    #[test]
    fn grant_turn_set_revoke_is_immediate() {
        let mut ledger = FakeConsentLedger::new();
        grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("workspace_read", "workspace.read")],
            1_000,
            Some(9_000),
        )
        .expect("batch");
        assert!(ledger.revoke("local.assistant", "workspace_read", "workspace.read"));
        assert!(matches!(
            ledger.check(&query(
                "local.assistant",
                "workspace_read",
                "workspace.read",
                1_500
            )),
            ConsentDecision::Deny { .. }
        ));
    }

    #[test]
    fn grant_turn_set_cap_fails_closed() {
        let mut ledger = FakeConsentLedger::new();
        for index in 0..MAX_CONSENT_GRANTS {
            ledger
                .grant(
                    "local.assistant",
                    &format!("tool_{index}"),
                    &format!("scope.{index}"),
                    99_000,
                )
                .expect("capacity");
        }
        let error = grant_turn_set(
            &mut ledger,
            "local.assistant",
            &[("one_more", "scope.extra")],
            1_000,
            Some(5_000),
        )
        .expect_err("overflow must fail");
        assert_eq!(
            error,
            BridgeError::LedgerFull {
                limit: MAX_CONSENT_GRANTS
            }
        );
        assert_eq!(ledger.len(), MAX_CONSENT_GRANTS);
    }

    #[test]
    fn dispatch_consent_denial_reason_is_byte_identical_across_paths() {
        // Both the native `ToolBus` dispatch path and the MCP adapter
        // execute path deny through the ONE shared helper
        // `ensure_consented_for_dispatch`, so their `Denied` reasons are
        // byte-identical by construction. This pins the helper wording the
        // cross-path test in `bitty-ai-mcp` asserts against.
        use crate::tool::{
            AuthBase, AuthContext, AuthDecision, ToolAuthorizer, ToolBus, ToolPlacement,
            ToolRegistry, ToolSpec,
        };

        struct Allow;
        impl ToolAuthorizer for Allow {
            fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
                AuthDecision::Allow
            }
        }

        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolSpec::new(
                    "workspace_read",
                    "Read a bounded workspace path",
                    br#"{"type":"object"}"#.to_vec(),
                    "workspace.read",
                    true,
                    ToolPlacement::Mcp,
                    None,
                )
                .expect("valid spec"),
            )
            .expect("capacity");
        let empty = FakeConsentLedger::new();
        let mut bus = ToolBus::new(registry)
            .with_authorizer(Allow)
            .with_consent(FakeConsentLedger::new(), "local.assistant");
        let mut ids = crate::session::IdIssuer::default();
        let base = AuthBase {
            agent_instance_id: ids.agent_instance(),
            session_id: ids.session(),
            level: crate::session::AgentLevel::Workspace,
        };
        let call = crate::tool::ToolCall {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        };
        let mut executor = crate::tool::FakeToolExecutor::new();
        executor.push_success("must never run", b"nope".to_vec());
        let execution = bus
            .dispatch(&mut executor, &call, &base, ids.execution(), 1_000)
            .expect("admission refusal is a recorded status");
        let bus_reason = match &execution.status {
            crate::tool::ToolStatus::Refused {
                cause: crate::tool::ToolError::Denied { reason, .. },
            } => reason.clone(),
            other => panic!("expected Refused{{Denied}}, got {other:?}"),
        };
        let helper_reason = match ensure_consented_for_dispatch(
            &empty,
            &ConsentQuery {
                protocol_id: "local.assistant",
                agent_instance: base.agent_instance_id,
                tool: "workspace_read",
                scope: "workspace.read",
                now_ms: 1_000,
            },
        ) {
            Err(crate::tool::ToolError::Denied { reason, .. }) => reason,
            other => panic!("helper must deny, got {other:?}"),
        };
        assert_eq!(bus_reason, helper_reason);
        assert!(bus_reason.contains("consent denied"));
    }
}

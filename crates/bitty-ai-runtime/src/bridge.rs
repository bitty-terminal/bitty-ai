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
            id: raw.to_owned(),
            reason: "protocol id must not be empty".to_owned(),
        });
    }
    if raw.len() > MAX_PROTOCOL_ID_LEN {
        return Err(BridgeError::InvalidProtocolId {
            id: raw.to_owned(),
            reason: format!("protocol id too long (max {MAX_PROTOCOL_ID_LEN})"),
        });
    }
    if raw.chars().any(|c| c.is_whitespace()) {
        return Err(BridgeError::InvalidProtocolId {
            id: raw.to_owned(),
            reason: "protocol id must not contain whitespace".to_owned(),
        });
    }
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() != 2 {
        return Err(BridgeError::InvalidProtocolId {
            id: raw.to_owned(),
            reason: "protocol id must be exactly owner.name (one dot)".to_owned(),
        });
    }
    for segment in &parts {
        if segment.is_empty() {
            return Err(BridgeError::InvalidProtocolId {
                id: raw.to_owned(),
                reason: "protocol id segment must not be empty".to_owned(),
            });
        }
        if segment.len() > MAX_PROTOCOL_ID_SEGMENT_LEN {
            return Err(BridgeError::InvalidProtocolId {
                id: raw.to_owned(),
                reason: format!("segment too long (max {MAX_PROTOCOL_ID_SEGMENT_LEN})"),
            });
        }
        let first = segment.as_bytes()[0];
        if !first.is_ascii_lowercase() {
            return Err(BridgeError::InvalidProtocolId {
                id: raw.to_owned(),
                reason: "segment must start with lowercase letter".to_owned(),
            });
        }
        for byte in segment.bytes() {
            let ok =
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_';
            if !ok {
                return Err(BridgeError::InvalidProtocolId {
                    id: raw.to_owned(),
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
            reason: format!("default-deny: no grant for {}", query.tool),
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
    ///   over-bound, or carries an interior NUL, or when `tool` violates the
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
                reason: error.to_string(),
            });
        }
        if scope.len() > MAX_CONSENT_SCOPE_LEN {
            return Err(BridgeError::InvalidGrant {
                reason: format!("scope too long (max {MAX_CONSENT_SCOPE_LEN})"),
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
                reason: format!(
                    "no live consent for '{}' on '{}' (tool {})",
                    query.protocol_id, query.scope, query.tool
                ),
            }
        }
    }
}

/// Require consent for `query`, converting a denial into a typed error.
///
/// This is the fail-closed helper hosts call after [`IdentityBridge`]
/// resolution and before [`crate::tool::ToolBus`] dispatch: allowed checks
/// return `Ok(())`, denied checks return [`BridgeError::ConsentDenied`] with
/// no partial state.
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
            tool: query.tool.to_owned(),
            reason,
        }),
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
        use crate::provider::{FakeProvider, ModelProvider, ProviderTurn};
        use crate::tool::{AuthBase, ToolBus, ToolRegistry, ToolSpec};

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
        });
        let request = crate::provider::TurnRequest {
            model: "fake-chat".to_owned(),
            messages: vec![crate::provider::Message::user("hi")],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: 4096,
            timeout_ms: 5_000,
            now_ms: 1_000,
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
        )
        .expect("valid spec");
    }
}

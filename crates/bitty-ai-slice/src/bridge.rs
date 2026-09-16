//! The generic host boundary the slice is allowed to use.
//!
//! `IpcBridge` is the only place that talks to the host. It composes real
//! `bitty-ipc` primitives:
//!
//! - [`validate_method_name`] and [`required_scope_for_method`] from the
//!   generic method registry,
//! - [`authorize_method`] for server-side-equivalent scope checks,
//! - [`ConsentLedger`] for per-client, per-scope, time-bounded grants,
//! - [`IpcEndpoint`] for the bounded request/response channel and correlation,
//! - [`validate_request_envelope`] / [`validate_response_envelope`] for the
//!   bounded wire-envelope checks.
//!
//! When the registry does not know a method, the bridge fails closed with
//! [`SliceError::UnsupportedHostMethod`] instead of inventing a new Core API.
//!
//! # Protocol identity to wire `client_id` binding (`AI-0065`)
//!
//! The host keys its consent ledger and execution store by a wire
//! `client_id` string, while the runtime owns the protocol principal
//! `ProtocolAgentId(owner.name)`. This module owns the one deterministic rule
//! that binds them:
//!
//! ```text
//! IdentityBridge::protocol() -> ProtocolAgentId(owner.name) -> client_id
//! ```
//!
//! [`wire_client_id`] derives the wire `client_id` as exactly the validated
//! `owner.name` principal ([`bitty_ai_runtime::bridge::ProtocolAgentId::as_str`]).
//! That matches the upstream `ConsentGrant.client_id` contract, which states
//! the id is a "UID string or `owner.name` pair". The derivation is total or
//! fail-closed:
//!
//! - an **unbound** [`IdentityBridge`] is refused, so a caller cannot supply
//!   an arbitrary id instead of presenting a bound identity, and
//! - a bound principal whose `owner.name` exceeds the host `client_id` bound
//!   ([`MAX_WIRE_CLIENT_ID_BYTES`]) is refused, because the runtime protocol
//!   bound (128 bytes) is wider than the host wire bound (64 bytes).
//!
//! Callers that already hold a candidate id must present it to
//! [`verify_wire_client_id`], which refuses any id that disagrees with the
//! bound principal. A refusal is total: nothing is constructed, dispatched,
//! or stored.
//!
//! [`IpcBridge::from_binding`], [`crate::fake_host::FakeHost::from_binding`],
//! and [`crate::live_host::LiveBittyHost::from_binding`] are the sanctioned
//! construction paths; the raw `new(client_id, ..)` constructors remain the
//! explicit test/host seam and are documented as such.

use bitty_ai_runtime::bridge::IdentityBridge;
use bitty_ipc::channel::{DEFAULT_REQUEST_TIMEOUT_MS, IpcEndpoint, IpcRequest, IpcResponse};
use bitty_ipc::error::IpcError;
use bitty_ipc::scope::{
    ConsentLedger, Scope, ScopeSet, authorize_method, required_scope_for_method,
    validate_method_name,
};
use bitty_ipc::wire::{WIRE_VERSION, validate_request_envelope, validate_response_envelope};

use crate::error::SliceError;

/// Maximum wire `client_id` bytes, re-derived from the pinned upstream source
/// of truth instead of mirrored as a literal.
///
/// Upstream: `bitty-ipc` `crates/bitty-ipc/src/auth.rs` `MAX_SCOPED_ID_BYTES`
/// (value `64`), which the tool-dispatch and execution services each alias
/// (`MAX_TOOL_CLIENT_ID_BYTES` and `MAX_EXEC_CLIENT_ID_BYTES`). Pinned
/// revision: `be6e63c55a18cb0a4bae1a528527b97251375bff`.
pub const MAX_WIRE_CLIENT_ID_BYTES: usize = bitty_ipc::auth::MAX_SCOPED_ID_BYTES;

/// Derive the wire `client_id` for a bound protocol principal (`AI-0065`).
///
/// The rule is deterministic and identity-bound: the `client_id` is exactly
/// the runtime-bound principal's validated `owner.name` string
/// ([`bitty_ai_runtime::bridge::ProtocolAgentId::as_str`]). It is never
/// caller-supplied, so two callers cannot pick the same attribution.
///
/// # Errors
///
/// Returns [`SliceError::Ipc`] with:
///
/// - `Unauthenticated` when `identity` is unbound, and
/// - `LimitExceeded` when the bound `owner.name` exceeds
///   [`MAX_WIRE_CLIENT_ID_BYTES`].
///
/// Both refusals happen before any host state exists: no dispatch, no store
/// entry, no consent grant.
pub fn wire_client_id(identity: &IdentityBridge) -> Result<String, SliceError> {
    let protocol = identity.protocol().ok_or_else(|| {
        SliceError::Ipc(IpcError::Unauthenticated {
            reason: "no protocol identity is bound; refusing to fabricate a client_id".to_owned(),
        })
    })?;
    let derived = protocol.as_str();
    if derived.len() > MAX_WIRE_CLIENT_ID_BYTES {
        return Err(SliceError::Ipc(IpcError::LimitExceeded {
            field: "wire client_id".to_owned(),
            limit: MAX_WIRE_CLIENT_ID_BYTES,
            actual: derived.len(),
        }));
    }
    Ok(derived.to_owned())
}

/// Verify a caller-supplied wire `client_id` against the bound principal
/// (`AI-0065`).
///
/// This is the fail-closed seam for callers that already hold an id (for
/// example one read from a wire request): the id is accepted only when it is
/// byte-identical to [`wire_client_id`] for the same [`IdentityBridge`].
/// Unbound bridges and over-long bound principals are refused by
/// [`wire_client_id`] first, so a mismatched id can never be laundered into
/// an attribution.
///
/// The refusal reason deliberately never echoes `supplied`; it reports only
/// the bound and supplied lengths, so an untrusted id cannot inject bytes
/// into a log or error surface.
///
/// # Errors
///
/// Returns [`SliceError::Ipc`] with `Unauthenticated` (unbound),
/// `LimitExceeded` (over-long bound principal), or `Denied` with code
/// `ClientIdMismatch` when `supplied` disagrees with the bound principal.
pub fn verify_wire_client_id(identity: &IdentityBridge, supplied: &str) -> Result<(), SliceError> {
    let derived = wire_client_id(identity)?;
    if derived == supplied {
        return Ok(());
    }
    Err(SliceError::Ipc(IpcError::Denied {
        code: "ClientIdMismatch".to_owned(),
        reason: format!(
            "caller-supplied client_id does not match the bound protocol identity (bound {} bytes, supplied {} bytes)",
            derived.len(),
            supplied.len()
        ),
    }))
}

/// Host-side peer that serves bounded IPC requests.
///
/// In a real deployment this is the Bitty host over the scoped IPC
/// transport. In tests it is a deterministic loopback peer; the slice never
/// treats a missing handler as success.
pub trait HostPeer {
    /// Serve one request, returning a bounded response.
    ///
    /// # Errors
    ///
    /// Implementations may return [`SliceError`] for transport-level faults;
    /// application-level refusal should instead be returned as an error
    /// [`IpcResponse`] so correlation and fail-closed typing are preserved.
    fn serve(&mut self, request: &IpcRequest) -> Result<IpcResponse, SliceError>;
}

/// Client-side generic IPC bridge with a consent ledger.
pub struct IpcBridge {
    endpoint: IpcEndpoint,
    consent: ConsentLedger,
    client_id: String,
    granted: ScopeSet,
}

impl IpcBridge {
    /// Construct a bridge for `client_id` whose server-evaluated scope set is
    /// `granted`.
    ///
    /// This is the raw seam: `client_id` is taken verbatim and only the host
    /// dispatch path bounds it. Product callers must use
    /// [`Self::from_binding`] so the id is derived from a bound protocol
    /// principal (`AI-0065`) instead of supplied by the caller.
    #[must_use]
    pub fn new(client_id: impl Into<String>, granted: ScopeSet) -> Self {
        Self {
            endpoint: IpcEndpoint::new(),
            consent: ConsentLedger::new(),
            client_id: client_id.into(),
            granted,
        }
    }

    /// Construct a bridge whose client identity is derived from the bound
    /// protocol principal of `identity` (`AI-0065`).
    ///
    /// # Errors
    ///
    /// Returns the failures of [`wire_client_id`]: an unbound identity or an
    /// over-long derived id is refused before any bridge is built.
    pub fn from_binding(identity: &IdentityBridge, granted: ScopeSet) -> Result<Self, SliceError> {
        Ok(Self::new(wire_client_id(identity)?, granted))
    }

    /// Authenticated client identity.
    #[must_use]
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// Record a per-client consent grant.
    ///
    /// # Errors
    ///
    /// Returns an IPC error when the ledger cap is reached or the grant is
    /// malformed.
    pub fn grant_consent(
        &mut self,
        scope: Scope,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<(), SliceError> {
        self.consent.grant(
            self.client_id.clone(),
            scope,
            now_ms,
            ttl_ms,
            "bitty-ai-slice".to_owned(),
        )?;
        Ok(())
    }

    /// Whether `scope` is currently granted to this client at `now_ms`.
    #[must_use]
    pub fn consent_active(&self, scope: Scope, now_ms: u64) -> bool {
        self.consent.is_granted(&self.client_id, scope, now_ms)
    }

    /// Number of requests still awaiting a correlated response.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.endpoint.pending_count()
    }

    /// Validate, authorize, send, and correlate one bounded request.
    ///
    /// # Errors
    ///
    /// - [`SliceError::Ipc`] when the wire/method/scope primitives reject.
    /// - [`SliceError::UnsupportedHostMethod`] when the generic registry does
    ///   not know `method`.
    /// - [`SliceError::ConsentRequired`] when no consent grant exists.
    /// - [`SliceError::ContextUnavailable`] when the host refuses or the
    ///   response cannot be correlated.
    pub fn call(
        &mut self,
        method: &str,
        params: &[u8],
        now_ms: u64,
        peer: &mut dyn HostPeer,
    ) -> Result<Vec<u8>, SliceError> {
        validate_method_name(method)?;
        if required_scope_for_method(method).is_none() {
            return Err(SliceError::UnsupportedHostMethod {
                method: method.to_owned(),
            });
        }
        let required = authorize_method(method, &self.granted)?;
        if !self.consent.is_granted(&self.client_id, required, now_ms) {
            return Err(SliceError::ConsentRequired {
                scope: required.as_str(),
            });
        }
        let id = self.endpoint.next_request_id();
        validate_request_envelope(WIRE_VERSION, &id.0.to_string(), method, params)?;
        let request = IpcRequest::new(
            id,
            method.to_owned(),
            params.to_vec(),
            now_ms,
            DEFAULT_REQUEST_TIMEOUT_MS,
        )?;
        self.endpoint.send_request(request.clone())?;
        let request =
            self.endpoint
                .recv_request()
                .ok_or_else(|| SliceError::ContextUnavailable {
                    reason: "endpoint lost the enqueued request".to_owned(),
                })?;
        let response = peer.serve(&request)?;
        if response.id != id {
            return Err(SliceError::ContextUnavailable {
                reason: "response id does not correlate with the request".to_owned(),
            });
        }
        self.endpoint.send_response(response)?;
        let received =
            self.endpoint
                .recv_response()
                .ok_or_else(|| SliceError::ContextUnavailable {
                    reason: "endpoint lost the response".to_owned(),
                })?;
        if !self.endpoint.complete(received.id) {
            return Err(SliceError::ContextUnavailable {
                reason: "response was not correlated with a pending request".to_owned(),
            });
        }
        validate_response_envelope(WIRE_VERSION, &received.id.0.to_string(), &received.payload)?;
        if received.is_error {
            return Err(SliceError::ContextUnavailable {
                reason: String::from_utf8_lossy(&received.payload).into_owned(),
            });
        }
        Ok(received.payload)
    }
}

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
//!   accepted wire bounds.
//!
//! When the registry does not know a method, the bridge fails closed with
//! [`SliceError::UnsupportedHostMethod`] instead of inventing a new Core API.

use bitty_ipc::channel::{DEFAULT_REQUEST_TIMEOUT_MS, IpcEndpoint, IpcRequest, IpcResponse};
use bitty_ipc::scope::{
    ConsentLedger, Scope, ScopeSet, authorize_method, required_scope_for_method,
    validate_method_name,
};
use bitty_ipc::wire::{WIRE_VERSION, validate_request_envelope, validate_response_envelope};

use crate::error::SliceError;

/// Host-side peer that serves bounded IPC requests.
///
/// In a real deployment this is the Bitty host over the accepted scoped IPC
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
    #[must_use]
    pub fn new(client_id: impl Into<String>, granted: ScopeSet) -> Self {
        Self {
            endpoint: IpcEndpoint::new(),
            consent: ConsentLedger::new(),
            client_id: client_id.into(),
            granted,
        }
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

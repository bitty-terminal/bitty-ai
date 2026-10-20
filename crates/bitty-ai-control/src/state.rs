//! Helper-side provider-publication application state machine.
//!
//! This module sits above the authenticated TLS session and below no provider
//! handler. It owns exactly one launch-scoped publication lifetime at a time,
//! validates the authenticated `Establish` echo, announces the initial
//! registry projection, and drives `Publish -> Ack -> Bind -> Bound` plus the
//! helper/core invalidation and rotation sequences from
//! `architecture/authenticated-provider-publication.md`.
//!
//! The wire header, bodies, closed enums, canonical payload, constants, and
//! codec come from `bitty-ipc-api`'s pure `provider_publication` module; this
//! module adds receiver state, retention, equality under the registry read
//! gate, and fail-closed local revocation.
//!
//! Strict provider publication stays disabled in this slice:
//!
//! - no `Admit`/`Admitted` reservation or data-channel admission is
//!   implemented; an incoming `Admit` revokes;
//! - no `ai.provider_list` handler, provider client, credential, or tool is
//!   contacted;
//! - no listener, socket, thread, or clock is created. The caller injects the
//!   stream, drives every step, and owns the establishment, transaction, and
//!   absolute-authority deadlines.
//!
//! A [`HelperControl`] value is one-shot. Closing or revoking authority is
//! terminal: no method reconnects, replays a record on a replacement stream,
//! or revives a retired lifetime.

use std::fmt;
use std::io::{Read, Write};

use bitty_ipc_api::provider_publication::{
    AckBody, AckId, BoundBody, ControlError, ControlRecord, ErrorCode, EstablishedBody,
    InvalidateBody, Kind, LIFETIME_SECRET_BYTES, LifetimeBody, LifetimeId, ProjectionBody,
    ProviderRoutingProjection, PublishBody, RANDOM_ID_BYTES, Reason, ReceiptId, RecordBody, Sender,
    SessionId, Tuple, next_expected_sequence, validate_incoming_sequence, validate_session,
};
use rustls::crypto::aws_lc_rs;
use zeroize::Zeroizing;

use crate::framing::{FrameError, FrameReader, FrameWriter};
use crate::manifest::{EstablishmentExpectations, LaunchManifest, ManifestError};
use crate::server::ControlSession;

/// The registry seam used by the helper state machine.
///
/// Implementations must hold their read/registry gate for the whole call:
/// the returned snapshot is atomic, and [`Self::retained_matches`] compares
/// the current canonical projection with retained publication bytes without a
/// mutation slipping between the revision check and the comparison.
///
/// The trait exposes no write method: a registry mutation is performed by the
/// caller's closure through [`HelperControl::registry_mutated`], after the
/// helper has emitted its predecessor invalidation.
pub trait RegistryView {
    /// Read the current feed revision and canonical provider-routing
    /// projection under the registry gate.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the registry cannot be read
    /// consistently.
    fn snapshot(&self) -> Result<RegistrySnapshot, RegistryError>;

    /// Compare retained publication bytes with the current canonical
    /// projection under the registry gate.
    ///
    /// The implementation must return `false` for a revision mismatch as well
    /// as a byte mismatch. A mutation between `Ack` and `Bind` therefore
    /// refuses instead of silently rebinding.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the comparison cannot be completed.
    fn retained_matches(&self, feed_revision: u64, retained: &[u8]) -> Result<bool, RegistryError>;
}

/// Why a registry snapshot or comparison failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    /// The registry is unavailable under its gate.
    Unavailable,
    /// The feed revision is zero or inconsistent with the mutation contract.
    InvalidRevision,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => formatter.write_str("registry view is unavailable"),
            Self::InvalidRevision => formatter.write_str("registry feed revision is invalid"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// One atomic registry read: feed revision plus canonical projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrySnapshot {
    feed_revision: u64,
    projection: ProviderRoutingProjection,
}

impl RegistrySnapshot {
    /// Build a snapshot, rejecting the zero feed revision.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::InvalidRevision`] for revision `0`.
    pub fn new(
        feed_revision: u64,
        projection: ProviderRoutingProjection,
    ) -> Result<Self, RegistryError> {
        if feed_revision == 0 {
            return Err(RegistryError::InvalidRevision);
        }
        Ok(Self {
            feed_revision,
            projection,
        })
    }

    /// Feed revision announced with this projection.
    #[must_use]
    pub const fn feed_revision(&self) -> u64 {
        self.feed_revision
    }

    /// Canonical provider-routing projection.
    #[must_use]
    pub fn projection(&self) -> &ProviderRoutingProjection {
        &self.projection
    }
}

/// Observable proof state for the current lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofState {
    /// No lifetime has been established.
    Unestablished,
    /// The current lifetime has no acknowledged publication.
    Unbound,
    /// A publication has been acknowledged and is awaiting `Bind`.
    Acknowledged,
    /// A receipt is active for the current tuple.
    Bound {
        /// Fresh helper proof receipt.
        receipt: ReceiptId,
    },
    /// Provider-publication authority has been revoked.
    Revoked,
}

/// Why the helper revoked provider-publication authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevocationReason {
    /// The control stream closed or failed.
    ControlTransport,
    /// A peer record violated the control contract.
    ProtocolViolation,
    /// The establishment echo did not match the launch manifest.
    EstablishmentMismatch,
    /// The registry view failed or was inconsistent.
    RegistryUnavailable,
    /// Retained publication bytes no longer equal the registry projection.
    RetainedBytesMismatch,
    /// A bounded queue, frame, or byte budget was exhausted.
    ResourceLimit,
    /// Fresh secure-random generation failed or repeated a live identifier.
    RandomUnavailable,
    /// A sequence or version space was exhausted.
    VersionExhausted,
    /// The peer reported a terminal control error.
    PeerError,
    /// The Core withdrew authority with grant loss or shutdown.
    PeerWithdrewAuthority,
    /// The caller closed authority locally.
    LocalClose,
}

impl RevocationReason {
    /// Best-effort wire error code for this local closure, when one applies.
    ///
    /// A transport failure or a peer-originated closure returns `None`
    /// because another record is neither useful nor guaranteed deliverable.
    #[must_use]
    pub const fn error_code(self) -> Option<ErrorCode> {
        match self {
            Self::ProtocolViolation | Self::EstablishmentMismatch | Self::RetainedBytesMismatch => {
                Some(ErrorCode::InvalidControl)
            }
            Self::RegistryUnavailable | Self::RandomUnavailable => Some(ErrorCode::Unavailable),
            Self::ResourceLimit | Self::VersionExhausted => Some(ErrorCode::ResourceLimit),
            Self::ControlTransport
            | Self::PeerError
            | Self::PeerWithdrewAuthority
            | Self::LocalClose => None,
        }
    }
}

impl fmt::Display for RevocationReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ControlTransport => formatter.write_str("control transport closed"),
            Self::ProtocolViolation => formatter.write_str("control protocol violation"),
            Self::EstablishmentMismatch => {
                formatter.write_str("establishment echo did not match the launch")
            }
            Self::RegistryUnavailable => formatter.write_str("registry view unavailable"),
            Self::RetainedBytesMismatch => {
                formatter.write_str("retained publication bytes differ from the registry")
            }
            Self::ResourceLimit => formatter.write_str("control resource limit exceeded"),
            Self::RandomUnavailable => formatter.write_str("secure random generation failed"),
            Self::VersionExhausted => formatter.write_str("control version space exhausted"),
            Self::PeerError => formatter.write_str("peer reported a terminal control error"),
            Self::PeerWithdrewAuthority => {
                formatter.write_str("peer withdrew provider-publication authority")
            }
            Self::LocalClose => formatter.write_str("local closure"),
        }
    }
}

/// One successful driver step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// A publication was retained and acknowledged.
    Published {
        /// Exact retained tuple.
        tuple: Tuple,
    },
    /// Retained bytes matched the registry and a receipt was issued.
    Bound {
        /// Exact bound tuple.
        tuple: Tuple,
        /// Fresh helper proof receipt.
        receipt: ReceiptId,
    },
    /// A lifetime was revoked and replaced.
    LifetimeRotated {
        /// Immediately retired lifetime.
        previous: LifetimeId,
        /// New current lifetime.
        current: LifetimeId,
        /// Feed revision announced with the new lifetime.
        feed_revision: u64,
        /// Closed rotation reason.
        reason: Reason,
    },
    /// A crossed Core invalidation matched the retired predecessor once.
    PredecessorCorrelated,
    /// The Core withdrew authority; control is closed with strict disabled.
    AuthorityClosed {
        /// Closed withdrawal reason.
        reason: Reason,
    },
    /// The peer sent a terminal control error.
    PeerError {
        /// Closed peer error code.
        code: ErrorCode,
    },
}

/// Fail-closed helper state-machine rejection.
///
/// Every variant is a local structured cause. `Display` uses fixed text and
/// never includes a payload, identifier, path, model string, secret, or
/// parser detail; [`std::error::Error::source`] exposes the structured
/// framing/codec cause for local diagnostics.
#[derive(Debug)]
pub enum HelperControlError {
    /// The bounded frame layer rejected a frame.
    Frame(FrameError),
    /// The pure control-record codec rejected a record.
    Record(ControlError),
    /// The launch manifest was structurally invalid.
    Manifest(ManifestError),
    /// The establishment echo differed from the launch manifest.
    EstablishmentMismatch {
        /// Fixed field label.
        field: &'static str,
    },
    /// The record session identifier did not match the launch.
    SessionMismatch,
    /// The per-direction sequence was not the exact next value.
    SequenceMismatch,
    /// The record kind is not valid in the current state.
    UnexpectedRecord {
        /// Received record kind.
        kind: Kind,
    },
    /// The record sender is not the authenticated Core role.
    UnexpectedSender {
        /// Received sender role.
        sender: Sender,
    },
    /// The Core sent a registry mutation, which only the helper owns.
    CoreRegistryMutation,
    /// The lifetime or feed revision is not current.
    StaleAuthority,
    /// A publication is already pending acknowledgement.
    PublicationAlreadyPending,
    /// The publication identity is not bound to this launch.
    PublicationIdentityMismatch,
    /// The publication version did not strictly increase.
    PublicationVersionNotIncreasing,
    /// No acknowledged publication is pending a bind.
    NoPendingBind,
    /// The bind did not match the exact pending acknowledgement.
    BindMismatch,
    /// Retained bytes differ from the current registry projection.
    RetainedBytesMismatch,
    /// The registry seam failed.
    Registry(RegistryError),
    /// Fresh secure-random generation failed.
    RandomUnavailable,
    /// A sequence or version space is exhausted.
    VersionExhausted,
    /// A bounded queue, frame, or byte budget was exhausted.
    ResourceLimit,
    /// The peer sent a terminal control error.
    PeerError {
        /// Peer error code.
        code: ErrorCode,
    },
    /// The launch manifest and authenticated session do not describe the same
    /// launch.
    LaunchMismatch,
    /// The control stream closed before establishment completed.
    ClosedByPeer,
    /// Establishment has not completed.
    NotEstablished,
    /// Establishment already completed.
    AlreadyEstablished,
    /// Provider-publication authority is already revoked or closed.
    Closed,
}

impl fmt::Display for HelperControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Frame(_) => formatter.write_str("control framing rejected"),
            Self::Record(_) => formatter.write_str("control record rejected"),
            Self::Manifest(_) => formatter.write_str("launch manifest rejected"),
            Self::EstablishmentMismatch { field } => {
                write!(
                    formatter,
                    "establishment echo differs from the launch manifest at {field}"
                )
            }
            Self::SessionMismatch => {
                formatter.write_str("control session differs from the launch nonce")
            }
            Self::SequenceMismatch => {
                formatter.write_str("control sequence is not the exact next value")
            }
            Self::UnexpectedRecord { .. } => {
                formatter.write_str("control record is not valid in the current state")
            }
            Self::UnexpectedSender { .. } => {
                formatter.write_str("control record sender is not the authenticated Core")
            }
            Self::CoreRegistryMutation => {
                formatter.write_str("Core may not mutate the helper registry")
            }
            Self::StaleAuthority => {
                formatter.write_str("control lifetime or feed revision is not current")
            }
            Self::PublicationAlreadyPending => {
                formatter.write_str("a publication is already pending acknowledgement")
            }
            Self::PublicationIdentityMismatch => {
                formatter.write_str("publication identity is not bound to this launch")
            }
            Self::PublicationVersionNotIncreasing => {
                formatter.write_str("publication version did not strictly increase")
            }
            Self::NoPendingBind => {
                formatter.write_str("no acknowledged publication is pending a bind")
            }
            Self::BindMismatch => {
                formatter.write_str("bind does not match the pending acknowledgement")
            }
            Self::RetainedBytesMismatch => {
                formatter.write_str("retained publication bytes differ from the registry")
            }
            Self::Registry(_) => formatter.write_str("registry view rejected the read"),
            Self::RandomUnavailable => formatter.write_str("secure random generation failed"),
            Self::VersionExhausted => formatter.write_str("control version space exhausted"),
            Self::ResourceLimit => formatter.write_str("control resource limit exceeded"),
            Self::PeerError { code } => code.diagnostic().fmt(formatter),
            Self::LaunchMismatch => {
                formatter.write_str("launch manifest and authenticated session differ")
            }
            Self::ClosedByPeer => formatter.write_str("control stream closed by the peer"),
            Self::NotEstablished => formatter.write_str("control establishment has not completed"),
            Self::AlreadyEstablished => {
                formatter.write_str("control establishment already completed")
            }
            Self::Closed => formatter.write_str("provider-publication authority is closed"),
        }
    }
}

impl std::error::Error for HelperControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Frame(error) => Some(error),
            Self::Record(error) => Some(error),
            Self::Manifest(error) => Some(error),
            Self::Registry(error) => Some(error),
            Self::EstablishmentMismatch { .. }
            | Self::SessionMismatch
            | Self::SequenceMismatch
            | Self::UnexpectedRecord { .. }
            | Self::UnexpectedSender { .. }
            | Self::CoreRegistryMutation
            | Self::StaleAuthority
            | Self::PublicationAlreadyPending
            | Self::PublicationIdentityMismatch
            | Self::PublicationVersionNotIncreasing
            | Self::NoPendingBind
            | Self::BindMismatch
            | Self::RetainedBytesMismatch
            | Self::RandomUnavailable
            | Self::VersionExhausted
            | Self::ResourceLimit
            | Self::PeerError { .. }
            | Self::LaunchMismatch
            | Self::ClosedByPeer
            | Self::NotEstablished
            | Self::AlreadyEstablished
            | Self::Closed => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    AwaitingEstablish,
    Ready,
    AwaitingBind,
    Revoked,
}

struct Lifetime {
    id: LifetimeId,
    feed_revision: u64,
    // The private capability is never serialized or exposed. This slice has
    // no proof object yet; retaining it here keeps ownership in the single
    // authority owner for later receipt binding.
    _secret: Zeroizing<[u8; LIFETIME_SECRET_BYTES]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RetiredLifetime {
    id: LifetimeId,
    feed_revision: u64,
}

struct PendingPublication {
    lifetime_id: LifetimeId,
    feed_revision: u64,
    tuple: Tuple,
    retained: Vec<u8>,
    ack_id: AckId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveReceipt {
    tuple: Tuple,
    receipt: ReceiptId,
}

struct Fault {
    error: HelperControlError,
    reason: RevocationReason,
    code: Option<ErrorCode>,
}

impl Fault {
    fn protocol(error: HelperControlError) -> Self {
        Self {
            error,
            reason: RevocationReason::ProtocolViolation,
            code: Some(ErrorCode::InvalidControl),
        }
    }

    fn establishment(field: &'static str) -> Self {
        Self {
            error: HelperControlError::EstablishmentMismatch { field },
            reason: RevocationReason::EstablishmentMismatch,
            code: Some(ErrorCode::InvalidControl),
        }
    }

    fn registry(error: RegistryError) -> Self {
        Self {
            error: HelperControlError::Registry(error),
            reason: RevocationReason::RegistryUnavailable,
            code: Some(ErrorCode::Unavailable),
        }
    }

    fn retained() -> Self {
        Self {
            error: HelperControlError::RetainedBytesMismatch,
            reason: RevocationReason::RetainedBytesMismatch,
            code: Some(ErrorCode::InvalidControl),
        }
    }

    fn version() -> Self {
        Self {
            error: HelperControlError::VersionExhausted,
            reason: RevocationReason::VersionExhausted,
            code: Some(ErrorCode::ResourceLimit),
        }
    }

    fn resource(error: HelperControlError) -> Self {
        Self {
            error,
            reason: RevocationReason::ResourceLimit,
            code: Some(ErrorCode::ResourceLimit),
        }
    }

    fn transport(error: HelperControlError) -> Self {
        Self {
            error,
            reason: RevocationReason::ControlTransport,
            code: None,
        }
    }

    fn peer(code: ErrorCode) -> Self {
        Self {
            error: HelperControlError::PeerError { code },
            reason: RevocationReason::PeerError,
            code: None,
        }
    }
}

/// Last retired public identifiers retained for RNG-collision checks.
///
/// The predecessor window is a wire/correlation concept and is cleared after
/// one crossed invalidation. Collision checking must keep the last retired
/// lifetime and receipt independently so a repeated random value is never
/// accepted just because correlation consumed the predecessor.
#[derive(Debug, Default)]
struct RetiredIdentifiers {
    lifetime: Option<LifetimeId>,
    receipt: Option<ReceiptId>,
}

impl RetiredIdentifiers {
    fn note_lifetime(&mut self, lifetime: LifetimeId) {
        self.lifetime = Some(lifetime);
    }

    fn note_receipt(&mut self, receipt: ReceiptId) {
        self.receipt = Some(receipt);
    }

    fn lifetime_conflicts(&self, candidate: LifetimeId) -> bool {
        self.lifetime == Some(candidate)
    }

    fn receipt_bytes_conflict(&self, candidate: &[u8; RANDOM_ID_BYTES]) -> bool {
        self.receipt
            .is_some_and(|receipt| receipt.as_bytes() == candidate)
    }
}

/// Helper-side provider-publication state machine.
///
/// `T` is the authenticated control stream and `R` is the injected registry
/// seam. The production constructors are [`HelperControl::from_session`] (the
/// server already captured the launch expectations) and
/// [`HelperControl::new`] (re-derive and cross-check against a retained
/// manifest); [`HelperControl::from_stream`] is the transport-generic form for
/// a future adapter and for deterministic tests. None creates a thread,
/// socket, listener, or clock.
pub struct HelperControl<T, R> {
    stream: T,
    reader: FrameReader,
    writer: FrameWriter,
    registry: R,
    expectations: EstablishmentExpectations,
    session_id: SessionId,
    phase: Phase,
    in_sequence: u64,
    out_sequence: u64,
    lifetime: Option<Lifetime>,
    predecessor: Option<RetiredLifetime>,
    retired: RetiredIdentifiers,
    pending: Option<PendingPublication>,
    receipt: Option<ActiveReceipt>,
    last_core_publication_version: u64,
    revocation: Option<RevocationReason>,
}

impl<S: Read + Write, R: RegistryView> HelperControl<ControlSession<S>, R> {
    /// Build the helper state machine from an authenticated control session.
    ///
    /// This is the production path when the launch manifest has already been
    /// consumed by [`crate::ControlServer`]: the server captured the
    /// non-secret expectations before dropping the private key, and this
    /// constructor reuses them. No manifest, key material, or second
    /// authentication step is needed.
    #[must_use]
    pub fn from_session(session: ControlSession<S>, registry: R) -> Self {
        let expectations = session.establishment_expectations().clone();
        Self::with_expectations(session, registry, expectations)
    }

    /// Build the helper state machine from a launch manifest and an
    /// authenticated control session.
    ///
    /// The non-secret expectations are re-derived from `manifest` and must
    /// equal the expectations captured when the TLS server was built. A
    /// session authenticated with different launch material is refused, so
    /// the state machine never mixes two launches.
    ///
    /// # Errors
    ///
    /// Returns [`HelperControlError::LaunchMismatch`] when the manifest and
    /// session differ, or [`HelperControlError::Manifest`] when the manifest
    /// is structurally invalid.
    pub fn new(
        manifest: LaunchManifest,
        session: ControlSession<S>,
        registry: R,
    ) -> Result<Self, HelperControlError> {
        let expected = EstablishmentExpectations::from_manifest(&manifest);
        if &expected != session.establishment_expectations() {
            return Err(HelperControlError::LaunchMismatch);
        }
        Self::from_stream(manifest, session, registry)
    }
}

impl<T: Read + Write, R: RegistryView> HelperControl<T, R> {
    /// Build the state machine over an already-authenticated stream.
    ///
    /// **Test/adapter-only.** This constructor performs no authentication
    /// itself. Production callers must use [`HelperControl::from_session`] or
    /// [`HelperControl::new`], which require a pin-checked
    /// [`ControlSession`]. This form exists for deterministic tests and for a
    /// future transport adapter that has already authenticated and pinned the
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns [`HelperControlError::Manifest`] when the manifest is
    /// structurally invalid.
    #[doc(hidden)]
    pub fn from_stream(
        manifest: LaunchManifest,
        stream: T,
        registry: R,
    ) -> Result<Self, HelperControlError> {
        manifest
            .validate_non_key()
            .map_err(HelperControlError::Manifest)?;
        let expectations = EstablishmentExpectations::from_manifest(&manifest);
        Ok(Self::with_expectations(stream, registry, expectations))
    }

    fn with_expectations(stream: T, registry: R, expectations: EstablishmentExpectations) -> Self {
        let session_id = expectations.launch_nonce();
        Self {
            stream,
            reader: FrameReader::new(),
            writer: FrameWriter::new(),
            registry,
            expectations,
            session_id,
            phase: Phase::AwaitingEstablish,
            in_sequence: 0,
            out_sequence: 0,
            lifetime: None,
            predecessor: None,
            retired: RetiredIdentifiers::default(),
            pending: None,
            receipt: None,
            last_core_publication_version: 0,
            revocation: None,
        }
    }

    /// Read exactly one `Establish`, verify every non-secret field, and send
    /// `Established` followed immediately by the initial `Projection`.
    ///
    /// The registry snapshot is taken before `Established` is written, so a
    /// registry failure never mints a lifetime. The helper is the sole
    /// lifetime creator and starts the feed at the registry's current
    /// non-zero revision.
    ///
    /// # Errors
    ///
    /// Returns the first framing, ordering, establishment, registry, random,
    /// or write rejection. Every failure revokes authority.
    pub fn establish(&mut self) -> Result<(), HelperControlError> {
        if self.phase == Phase::Revoked {
            return Err(HelperControlError::Closed);
        }
        if self.phase != Phase::AwaitingEstablish {
            return Err(self.fail(Fault::protocol(HelperControlError::AlreadyEstablished)));
        }

        let record = self.read_incoming()?;
        self.validate_envelope(&record)?;
        let echo = match record.body {
            RecordBody::Establish(echo) => echo,
            body => {
                return Err(
                    self.fail(Fault::protocol(HelperControlError::UnexpectedRecord {
                        kind: body.kind(),
                    })),
                );
            }
        };
        if let Err(field) = self.expectations.verify(&echo) {
            return Err(self.fail(Fault::establishment(field)));
        }

        // The registry owns the helper-global feed revision. It starts at `1`
        // on a fresh helper and may already be greater when population or
        // mutation happened before this control session; announce the current
        // non-zero revision rather than fabricating a stale `1`.
        let snapshot = self.snapshot_or_fail()?;
        let lifetime = match self.fresh_lifetime(snapshot.feed_revision()) {
            Ok(lifetime) => lifetime,
            Err(error) => return Err(self.fail(Fault::random_for(error))),
        };
        let lifetime_id = lifetime.id;
        let feed_revision = snapshot.feed_revision();
        let projection = snapshot.projection().clone();

        self.lifetime = Some(lifetime);
        self.phase = Phase::Ready;
        self.emit(RecordBody::Established(EstablishedBody {
            echo,
            lifetime_id,
        }))?;
        self.emit(RecordBody::Projection(ProjectionBody {
            lifetime_id,
            feed_revision,
            payload: projection,
        }))?;
        Ok(())
    }

    /// Read and process exactly one Core record after establishment.
    ///
    /// `Publish`, `Bind`, `Invalidate`, and `Error` are the only legal
    /// incoming kinds. `Admit` and data-channel admission are intentionally
    /// unsupported in this slice and revoke.
    ///
    /// # Errors
    ///
    /// Returns the first framing, order, state, registry, random, or write
    /// rejection. Every failure revokes authority; subsequent calls return
    /// [`HelperControlError::Closed`].
    pub fn step(&mut self) -> Result<Step, HelperControlError> {
        self.require_active()?;
        let record = self.read_incoming()?;
        self.validate_envelope(&record)?;
        match record.body {
            RecordBody::Publish(body) => self.handle_publish(body),
            RecordBody::Bind(body) => self.handle_bind(body),
            RecordBody::Invalidate(body) => self.handle_invalidate(body),
            RecordBody::Error(code) => {
                self.fail(Fault::peer(code));
                Ok(Step::PeerError { code })
            }
            body => Err(
                self.fail(Fault::protocol(HelperControlError::UnexpectedRecord {
                    kind: body.kind(),
                })),
            ),
        }
    }

    /// Invalidate the current lifetime, mutate the registry through `mutate`,
    /// and announce `Lifetime -> Projection` at the next feed revision.
    ///
    /// `Invalidate(L1,R1)` is written before `mutate` runs, so Core cannot
    /// keep using the predecessor while the registry changes. The closure must
    /// advance the registry's feed revision by exactly one; `R2 = R1 + 1` is
    /// computed with checked arithmetic before mutation. A failed mutation, a
    /// missing/extra revision step, or an unreadable projection closes
    /// authority.
    ///
    /// # Errors
    ///
    /// Returns the first ordering, registry, revision, random, or write
    /// rejection. Every failure revokes authority.
    pub fn registry_mutated<F>(&mut self, mutate: F) -> Result<Step, HelperControlError>
    where
        F: FnOnce(&mut R) -> Result<(), RegistryError>,
    {
        self.require_active()?;
        let (previous, feed_revision) = self.current_identity()?;
        let new_revision = match feed_revision.checked_add(1) {
            Some(revision) => revision,
            None => return Err(self.fail(Fault::version())),
        };

        let before = self.snapshot_or_fail()?;
        if before.feed_revision() != feed_revision {
            return Err(self.fail(Fault::protocol(HelperControlError::StaleAuthority)));
        }

        self.pending = None;
        self.clear_receipt();
        self.emit(RecordBody::Invalidate(InvalidateBody {
            lifetime_id: previous,
            feed_revision,
            reason: Reason::RegistryMutation,
        }))?;

        if let Err(error) = mutate(&mut self.registry) {
            return Err(self.fail(Fault::registry(error)));
        }

        let after = self.snapshot_or_fail()?;
        if after.feed_revision() != new_revision {
            return Err(self.fail(Fault::registry(RegistryError::InvalidRevision)));
        }

        let lifetime = match self.fresh_lifetime(new_revision) {
            Ok(lifetime) => lifetime,
            Err(error) => return Err(self.fail(Fault::random_for(error))),
        };
        let current = lifetime.id;
        self.retire_current();
        self.lifetime = Some(lifetime);
        self.phase = Phase::Ready;
        self.emit(RecordBody::Lifetime(LifetimeBody {
            lifetime_id: current,
            feed_revision: new_revision,
        }))?;
        self.emit(RecordBody::Projection(ProjectionBody {
            lifetime_id: current,
            feed_revision: new_revision,
            payload: after.projection().clone(),
        }))?;
        Ok(Step::LifetimeRotated {
            previous,
            current,
            feed_revision: new_revision,
            reason: Reason::RegistryMutation,
        })
    }

    /// Perform a helper-origin reset/reconstruction rotation.
    ///
    /// Emits `Invalidate(L1,R1,Reset) -> Lifetime(L2,R1) -> Projection(L2,R1)`.
    /// The registry bytes and revision do not change; any pending
    /// acknowledgement or receipt is revoked first. The retired predecessor is
    /// retained for one crossed Core invalidation.
    ///
    /// # Errors
    ///
    /// Returns the first ordering, registry, random, or write rejection. Every
    /// failure revokes authority.
    pub fn reset(&mut self) -> Result<Step, HelperControlError> {
        self.require_active()?;
        let (previous, feed_revision) = self.current_identity()?;

        self.pending = None;
        self.clear_receipt();
        self.emit(RecordBody::Invalidate(InvalidateBody {
            lifetime_id: previous,
            feed_revision,
            reason: Reason::Reset,
        }))?;

        let snapshot = self.snapshot_or_fail()?;
        if snapshot.feed_revision() != feed_revision {
            return Err(self.fail(Fault::protocol(HelperControlError::StaleAuthority)));
        }

        let lifetime = match self.fresh_lifetime(feed_revision) {
            Ok(lifetime) => lifetime,
            Err(error) => return Err(self.fail(Fault::random_for(error))),
        };
        let current = lifetime.id;
        self.retire_current();
        self.lifetime = Some(lifetime);
        self.phase = Phase::Ready;
        self.emit(RecordBody::Lifetime(LifetimeBody {
            lifetime_id: current,
            feed_revision,
        }))?;
        self.emit(RecordBody::Projection(ProjectionBody {
            lifetime_id: current,
            feed_revision,
            payload: snapshot.projection().clone(),
        }))?;
        Ok(Step::LifetimeRotated {
            previous,
            current,
            feed_revision,
            reason: Reason::Reset,
        })
    }

    /// Revoke authority and close locally without sending another record.
    ///
    /// A caller uses this for its own deadline, shutdown, or replacement
    /// policy. Revocation is terminal and clears the current lifetime,
    /// predecessor window, retired identifier checks, pending publication, and
    /// receipt. No old record is retried on a replacement stream.
    ///
    /// This method does not write an `Invalidate`, TLS `close_notify`, or EOF.
    /// The caller owns the transport: dropping this value (and its
    /// [`ControlSession`]) closes the injected stream. A wire-visible
    /// shutdown record, if required, belongs to the caller's shutdown policy.
    pub fn close(&mut self) {
        if self.phase == Phase::Revoked {
            return;
        }
        self.phase = Phase::Revoked;
        self.revocation = Some(RevocationReason::LocalClose);
        self.clear_authority();
    }

    /// Current public lifetime identifier, if one is live.
    #[must_use]
    pub fn lifetime_id(&self) -> Option<LifetimeId> {
        self.lifetime.as_ref().map(|lifetime| lifetime.id)
    }

    /// Current feed revision, if a lifetime is live.
    #[must_use]
    pub fn feed_revision(&self) -> Option<u64> {
        self.lifetime
            .as_ref()
            .map(|lifetime| lifetime.feed_revision)
    }

    /// Current proof state.
    #[must_use]
    pub fn proof_state(&self) -> ProofState {
        if self.phase == Phase::Revoked {
            return ProofState::Revoked;
        }
        if self.lifetime.is_none() {
            return ProofState::Unestablished;
        }
        if self.pending.is_some() {
            return ProofState::Acknowledged;
        }
        match self.receipt {
            Some(receipt) => ProofState::Bound {
                receipt: receipt.receipt,
            },
            None => ProofState::Unbound,
        }
    }

    /// Highest accepted Core publication version on this control session.
    #[must_use]
    pub const fn publication_version(&self) -> u64 {
        self.last_core_publication_version
    }

    /// Why authority was revoked or closed, if it was.
    #[must_use]
    pub const fn revocation_reason(&self) -> Option<RevocationReason> {
        self.revocation
    }

    /// Session/launch identifier bound into every record header.
    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Whether authority is terminal.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        matches!(self.phase, Phase::Revoked)
    }

    fn handle_publish(&mut self, body: PublishBody) -> Result<Step, HelperControlError> {
        let (lifetime_id, feed_revision) = self.current_identity()?;
        if body.lifetime_id != lifetime_id || body.feed_revision != feed_revision {
            return Err(self.fail(Fault::protocol(HelperControlError::StaleAuthority)));
        }
        if self.pending.is_some() {
            return Err(self.fail(Fault::protocol(
                HelperControlError::PublicationAlreadyPending,
            )));
        }
        if body.tuple.publication_id.launch_nonce() != self.session_id {
            return Err(self.fail(Fault::protocol(
                HelperControlError::PublicationIdentityMismatch,
            )));
        }
        let version = body.tuple.publication_id.version();
        if version <= self.last_core_publication_version {
            return Err(self.fail(Fault::protocol(
                HelperControlError::PublicationVersionNotIncreasing,
            )));
        }

        // The codec already validated the canonical bytes and their digest.
        let retained = body.payload.canonical_bytes();
        let ack_id = match random_id::<RANDOM_ID_BYTES>() {
            Ok(bytes) => {
                let ack_id = AckId::new(bytes);
                if self
                    .receipt
                    .is_some_and(|receipt| receipt.receipt.as_bytes() == ack_id.as_bytes())
                    || self.retired.receipt_bytes_conflict(ack_id.as_bytes())
                {
                    return Err(self.fail(Fault::random_for(HelperControlError::RandomUnavailable)));
                }
                ack_id
            }
            Err(error) => return Err(self.fail(Fault::random_for(error))),
        };

        // One current tuple: accepting a new publication revokes any prior
        // receipt before the new proof exists.
        self.clear_receipt();
        self.pending = Some(PendingPublication {
            lifetime_id,
            feed_revision,
            tuple: body.tuple,
            retained,
            ack_id,
        });
        self.last_core_publication_version = version;
        self.phase = Phase::AwaitingBind;
        self.emit(RecordBody::Ack(AckBody {
            lifetime_id,
            feed_revision,
            tuple: body.tuple,
            ack_id,
        }))?;
        Ok(Step::Published { tuple: body.tuple })
    }

    fn handle_bind(&mut self, body: AckBody) -> Result<Step, HelperControlError> {
        let pending = match self.pending.take() {
            Some(pending) => pending,
            None => return Err(self.fail(Fault::protocol(HelperControlError::NoPendingBind))),
        };
        if body.lifetime_id != pending.lifetime_id
            || body.feed_revision != pending.feed_revision
            || body.tuple != pending.tuple
            || body.ack_id != pending.ack_id
        {
            return Err(self.fail(Fault::protocol(HelperControlError::BindMismatch)));
        }

        let comparison = self
            .registry
            .retained_matches(pending.feed_revision, &pending.retained);
        match comparison {
            Ok(true) => {}
            Ok(false) => return Err(self.fail(Fault::retained())),
            Err(error) => return Err(self.fail(Fault::registry(error))),
        }

        let receipt = match random_id::<RANDOM_ID_BYTES>() {
            Ok(bytes) => {
                let receipt = ReceiptId::new(bytes);
                if receipt.as_bytes() == pending.ack_id.as_bytes()
                    || self.retired.receipt_bytes_conflict(receipt.as_bytes())
                {
                    return Err(self.fail(Fault::random_for(HelperControlError::RandomUnavailable)));
                }
                receipt
            }
            Err(error) => return Err(self.fail(Fault::random_for(error))),
        };
        self.receipt = Some(ActiveReceipt {
            tuple: pending.tuple,
            receipt,
        });
        self.phase = Phase::Ready;
        self.emit(RecordBody::Bound(BoundBody {
            lifetime_id: pending.lifetime_id,
            feed_revision: pending.feed_revision,
            tuple: pending.tuple,
            receipt,
        }))?;
        Ok(Step::Bound {
            tuple: pending.tuple,
            receipt,
        })
    }

    /// Handle a Core-origin invalidation.
    ///
    /// Accepted reasons: `SourceMutation`, `Reset`, and `Supersession` rotate
    /// the current lifetime with the same feed revision; `GrantLoss` and
    /// `Shutdown` close authority without a replacement lifetime.
    /// `RegistryMutation` is refused because only the helper owns the registry
    /// and emits that reason on its own mutation path.
    fn handle_invalidate(&mut self, body: InvalidateBody) -> Result<Step, HelperControlError> {
        if body.reason == Reason::RegistryMutation {
            return Err(self.fail(Fault::protocol(HelperControlError::CoreRegistryMutation)));
        }
        let current = self
            .lifetime
            .as_ref()
            .map(|lifetime| (lifetime.id, lifetime.feed_revision));
        if current == Some((body.lifetime_id, body.feed_revision)) {
            self.pending = None;
            self.clear_receipt();
            if matches!(body.reason, Reason::GrantLoss | Reason::Shutdown) {
                self.phase = Phase::Revoked;
                self.revocation = Some(RevocationReason::PeerWithdrewAuthority);
                self.clear_authority();
                return Ok(Step::AuthorityClosed {
                    reason: body.reason,
                });
            }

            let snapshot = self.snapshot_or_fail()?;
            if snapshot.feed_revision() != body.feed_revision {
                return Err(self.fail(Fault::protocol(HelperControlError::StaleAuthority)));
            }
            let lifetime = match self.fresh_lifetime(body.feed_revision) {
                Ok(lifetime) => lifetime,
                Err(error) => return Err(self.fail(Fault::random_for(error))),
            };
            let current_id = lifetime.id;
            self.retire_current();
            self.lifetime = Some(lifetime);
            self.phase = Phase::Ready;
            self.emit(RecordBody::Lifetime(LifetimeBody {
                lifetime_id: current_id,
                feed_revision: body.feed_revision,
            }))?;
            self.emit(RecordBody::Projection(ProjectionBody {
                lifetime_id: current_id,
                feed_revision: body.feed_revision,
                payload: snapshot.projection().clone(),
            }))?;
            return Ok(Step::LifetimeRotated {
                previous: body.lifetime_id,
                current: current_id,
                feed_revision: body.feed_revision,
                reason: body.reason,
            });
        }

        if self.predecessor
            == Some(RetiredLifetime {
                id: body.lifetime_id,
                feed_revision: body.feed_revision,
            })
        {
            self.predecessor = None;
            if matches!(body.reason, Reason::GrantLoss | Reason::Shutdown) {
                self.phase = Phase::Revoked;
                self.revocation = Some(RevocationReason::PeerWithdrewAuthority);
                self.clear_authority();
                return Ok(Step::AuthorityClosed {
                    reason: body.reason,
                });
            }
            return Ok(Step::PredecessorCorrelated);
        }

        Err(self.fail(Fault::protocol(HelperControlError::StaleAuthority)))
    }

    fn require_active(&self) -> Result<(), HelperControlError> {
        match self.phase {
            Phase::Revoked => Err(HelperControlError::Closed),
            Phase::AwaitingEstablish => Err(HelperControlError::NotEstablished),
            Phase::Ready | Phase::AwaitingBind => Ok(()),
        }
    }

    fn current_identity(&self) -> Result<(LifetimeId, u64), HelperControlError> {
        match self.lifetime.as_ref() {
            Some(lifetime) => Ok((lifetime.id, lifetime.feed_revision)),
            None => Err(HelperControlError::NotEstablished),
        }
    }

    fn read_incoming(&mut self) -> Result<ControlRecord, HelperControlError> {
        let result = self.reader.read_record(&mut self.stream);
        match result {
            Ok(record) => Ok(record),
            Err(error) => {
                let fault = if matches!(
                    &error,
                    FrameError::Closed
                        | FrameError::TruncatedPrefix { .. }
                        | FrameError::TruncatedBody { .. }
                        | FrameError::Io(_)
                        | FrameError::WriteZero
                ) {
                    Fault::transport(frame_error(error))
                } else if matches!(
                    &error,
                    FrameError::AllocationFailed { .. }
                        | FrameError::QueueFull { .. }
                        | FrameError::ByteBudgetExceeded { .. }
                ) {
                    Fault::resource(frame_error(error))
                } else {
                    Fault::protocol(frame_error(error))
                };
                Err(self.fail(fault))
            }
        }
    }

    fn validate_envelope(&mut self, record: &ControlRecord) -> Result<(), HelperControlError> {
        if record.sender != Sender::Core {
            return Err(
                self.fail(Fault::protocol(HelperControlError::UnexpectedSender {
                    sender: record.sender,
                })),
            );
        }
        if validate_session(self.session_id, record.session_id).is_err() {
            return Err(self.fail(Fault::protocol(HelperControlError::SessionMismatch)));
        }
        match validate_incoming_sequence(self.in_sequence, record.sequence) {
            Ok(()) => {
                self.in_sequence = record.sequence;
                Ok(())
            }
            Err(_) => Err(self.fail(Fault::protocol(HelperControlError::SequenceMismatch))),
        }
    }

    fn snapshot_or_fail(&mut self) -> Result<RegistrySnapshot, HelperControlError> {
        let result = self.registry.snapshot();
        match result {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => Err(self.fail(Fault::registry(error))),
        }
    }

    fn fresh_lifetime(&self, feed_revision: u64) -> Result<Lifetime, HelperControlError> {
        let mut id = [0_u8; RANDOM_ID_BYTES];
        let mut secret = Zeroizing::new([0_u8; LIFETIME_SECRET_BYTES]);
        let provider = aws_lc_rs::default_provider();
        provider
            .secure_random
            .fill(&mut id)
            .map_err(|_| HelperControlError::RandomUnavailable)?;
        provider
            .secure_random
            .fill(&mut secret[..])
            .map_err(|_| HelperControlError::RandomUnavailable)?;
        let id = LifetimeId::new(id);
        if self
            .lifetime
            .as_ref()
            .is_some_and(|lifetime| lifetime.id == id)
            || self
                .predecessor
                .is_some_and(|predecessor| predecessor.id == id)
            || self.retired.lifetime_conflicts(id)
        {
            // A repeated fresh identifier is a terminal RNG failure. Never
            // retry with a timestamp, counter, fixed seed, or guessed value.
            return Err(HelperControlError::RandomUnavailable);
        }
        Ok(Lifetime {
            id,
            feed_revision,
            _secret: secret,
        })
    }

    fn retire_current(&mut self) {
        if let Some(previous) = self.lifetime.take() {
            self.retired.note_lifetime(previous.id);
            self.predecessor = Some(RetiredLifetime {
                id: previous.id,
                feed_revision: previous.feed_revision,
            });
        }
    }

    fn clear_receipt(&mut self) {
        if let Some(receipt) = self.receipt.take() {
            self.retired.note_receipt(receipt.receipt);
        }
    }

    fn emit(&mut self, body: RecordBody) -> Result<(), HelperControlError> {
        match self.raw_send(body) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.fail(send_fault(error))),
        }
    }

    fn raw_send(&mut self, body: RecordBody) -> Result<(), HelperControlError> {
        let sequence = next_expected_sequence(self.out_sequence)
            .map_err(|_| HelperControlError::VersionExhausted)?;
        let record = ControlRecord {
            sender: Sender::Helper,
            sequence,
            session_id: self.session_id,
            body,
        };
        self.writer.enqueue(&record).map_err(frame_error)?;
        self.writer.flush(&mut self.stream).map_err(frame_error)?;
        self.out_sequence = sequence;
        Ok(())
    }

    fn send_error_best_effort(&mut self, code: ErrorCode) {
        let _ = self.raw_send(RecordBody::Error(code));
    }

    fn fail(&mut self, fault: Fault) -> HelperControlError {
        let Fault {
            error,
            reason,
            code,
        } = fault;
        if self.phase == Phase::Revoked {
            return error;
        }
        self.phase = Phase::Revoked;
        self.revocation = Some(reason);
        // Local revocation is not conditional on delivering a diagnostic.
        // Clear the lifetime, pending publication, receipt, retired IDs, and
        // outbound queue first; only then make one best-effort Error attempt.
        self.clear_authority();
        if let Some(code) = code {
            self.send_error_best_effort(code);
        }
        error
    }

    fn clear_authority(&mut self) {
        self.lifetime = None;
        self.predecessor = None;
        self.retired = RetiredIdentifiers::default();
        self.pending = None;
        self.receipt = None;
        self.writer.clear();
    }
}

impl<T: Read + Write, R: RegistryView> fmt::Debug for HelperControl<T, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safe-diagnostics rule: never print session, lifetime, ack, or
        // receipt identifiers from the state owner.
        formatter
            .debug_struct("HelperControl")
            .field("phase", &self.phase)
            .field("established", &self.lifetime.is_some())
            .field("pending_publication", &self.pending.is_some())
            .field("active_receipt", &self.receipt.is_some())
            .field("closed", &self.is_closed())
            .field("revocation", &self.revocation)
            .finish_non_exhaustive()
    }
}

impl Fault {
    fn random_for(error: HelperControlError) -> Self {
        Self {
            error,
            reason: RevocationReason::RandomUnavailable,
            code: Some(ErrorCode::Unavailable),
        }
    }
}

fn frame_error(error: FrameError) -> HelperControlError {
    match error {
        FrameError::Record(error) => HelperControlError::Record(error),
        other => HelperControlError::Frame(other),
    }
}

fn send_fault(error: HelperControlError) -> Fault {
    match &error {
        HelperControlError::VersionExhausted => Fault {
            error,
            reason: RevocationReason::VersionExhausted,
            code: None,
        },
        HelperControlError::ResourceLimit
        | HelperControlError::Frame(FrameError::AllocationFailed { .. })
        | HelperControlError::Frame(FrameError::QueueFull { .. })
        | HelperControlError::Frame(FrameError::ByteBudgetExceeded { .. }) => Fault {
            error,
            reason: RevocationReason::ResourceLimit,
            code: Some(ErrorCode::ResourceLimit),
        },
        _ => Fault {
            error,
            reason: RevocationReason::ControlTransport,
            code: None,
        },
    }
}

fn random_id<const N: usize>() -> Result<[u8; N], HelperControlError> {
    let mut bytes = [0_u8; N];
    aws_lc_rs::default_provider()
        .secure_random
        .fill(&mut bytes)
        .map_err(|_| HelperControlError::RandomUnavailable)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{LifetimeId, ReceiptId, RetiredIdentifiers};

    #[test]
    fn retired_lifetime_is_checked_until_the_next_rotation() {
        let first = LifetimeId::new([0x11; 32]);
        let second = LifetimeId::new([0x22; 32]);
        let third = LifetimeId::new([0x33; 32]);
        let mut retired = RetiredIdentifiers::default();
        retired.note_lifetime(first);
        // Crossing an invalidation clears the wire predecessor window, but
        // not this collision set.
        assert!(retired.lifetime_conflicts(first));
        retired.note_lifetime(second);
        assert!(!retired.lifetime_conflicts(first));
        assert!(retired.lifetime_conflicts(second));
        assert!(!retired.lifetime_conflicts(third));
    }

    #[test]
    fn retired_receipt_is_checked_after_replacement() {
        let first = ReceiptId::new([0x44; 32]);
        let second = ReceiptId::new([0x55; 32]);
        let mut retired = RetiredIdentifiers::default();
        retired.note_receipt(first);
        assert!(retired.receipt_bytes_conflict(first.as_bytes()));
        retired.note_receipt(second);
        assert!(!retired.receipt_bytes_conflict(first.as_bytes()));
        assert!(retired.receipt_bytes_conflict(second.as_bytes()));
    }
}

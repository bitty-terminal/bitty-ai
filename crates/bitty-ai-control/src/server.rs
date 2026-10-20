//! TLS 1.3 server with a launch-scoped certificate pin.
//!
//! The server accepts exactly one client leaf whose SHA-256 DER digest equals
//! the Core pin from the launch manifest. Only TLS 1.3 is enabled, ALPN must
//! select the control profile, and resumption/tickets/early data/secret
//! extraction/key logging are disabled. Signature checks are delegated to the
//! rustls provider helpers; this module contains no handshake cryptography.

use std::fmt;
use std::io::{self, Read, Write};
use std::sync::Arc;

use rustls::crypto::CryptoProvider;
use rustls::crypto::aws_lc_rs;
use rustls::pki_types::{CertificateDer, UnixTime};
use rustls::server::NoServerSessionStorage;
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::sign::SingleCertAndKey;
use rustls::{
    CertificateError, DigitallySignedStruct, DistinguishedName, Error as RustlsError, NoKeyLog,
    ProtocolVersion, ServerConfig, ServerConnection, SignatureScheme,
};

use crate::manifest::{
    CertificatePin, EstablishmentExpectations, LaunchManifest, ManifestError, validated_helper_key,
};
use crate::{CONTROL_ALPN, TLS_PREAUTH_STEPS_MAX, TLS_PREAUTH_WIRE_MAX_BYTES};

/// Helper-side control server configuration.
///
/// Construction consumes the [`LaunchManifest`] so the owned private-key
/// buffer is dropped (and zeroized) as soon as rustls has loaded the key.
/// The value is one-shot: [`Self::handshake`] and [`Self::accept`] consume it,
/// so one server serves exactly one handshake. Older launch material must
/// never authenticate a replacement launch.
pub struct ControlServer {
    config: Arc<ServerConfig>,
    peer_certificate_pin: CertificatePin,
    establishment_expectations: EstablishmentExpectations,
}

impl ControlServer {
    /// Build the TLS server configuration from a parsed launch manifest.
    ///
    /// The manifest is consumed. The helper key pair is validated again here
    /// (including ECDSA P-256 selection) so a directly constructed manifest
    /// cannot bypass the ingestion check.
    ///
    /// # Errors
    ///
    /// Returns [`ControlError::Manifest`] for a structural or key-material
    /// rejection and [`ControlError::Rustls`] if rustls rejects the
    /// configuration or certificate/key combination.
    #[allow(clippy::new_without_default)] // A default server would have no pin.
    pub fn new(manifest: LaunchManifest) -> Result<Self, ControlError> {
        manifest
            .validate_non_key()
            .map_err(ControlError::Manifest)?;

        let provider = Arc::new(aws_lc_rs::default_provider());
        let certified_key =
            validated_helper_key(&manifest, &provider).map_err(ControlError::Manifest)?;
        let peer_certificate_pin = manifest.core_certificate_pin();
        let establishment_expectations = EstablishmentExpectations::from_manifest(&manifest);

        let verifier = Arc::new(PinnedClientCertVerifier {
            pin: peer_certificate_pin,
            provider: provider.clone(),
        });

        let mut config = ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(ControlError::Rustls)?
            .with_client_cert_verifier(verifier)
            .with_cert_resolver(Arc::new(SingleCertAndKey::from(certified_key)));

        config.alpn_protocols = vec![CONTROL_ALPN.to_vec()];
        // No resumption, tickets, half-RTT/0-RTT data, secret extraction, key
        // logging, or certificate compression on this launch-scoped profile.
        config.session_storage = Arc::new(NoServerSessionStorage {});
        config.send_tls13_tickets = 0;
        config.max_tls13_tickets = 0;
        config.max_early_data_size = 0;
        config.send_half_rtt_data = false;
        config.enable_secret_extraction = false;
        config.key_log = Arc::new(NoKeyLog {});
        config.cert_compressors.clear();
        config.cert_decompressors.clear();

        Ok(Self {
            config: Arc::new(config),
            peer_certificate_pin,
            establishment_expectations,
        })
    }

    /// Core leaf pin derived from the received certificate DER.
    #[must_use]
    pub fn peer_certificate_pin(&self) -> CertificatePin {
        self.peer_certificate_pin
    }

    /// Consume the launch-scoped configuration into one handshake state
    /// machine.
    ///
    /// The value is consumed deliberately: launch material authenticates one
    /// launch and one control session, never a reconnect or a replacement
    /// launch. A failed handshake is a revoked launch, not a retry.
    ///
    /// # Errors
    ///
    /// Returns [`ControlError::Rustls`] if rustls cannot instantiate the
    /// connection (for example, an invalid configured bound).
    pub fn handshake(self) -> Result<ControlServerHandshake, ControlError> {
        let Self {
            config,
            peer_certificate_pin,
            establishment_expectations,
        } = self;
        ControlServerHandshake::new(&config, peer_certificate_pin, establishment_expectations)
    }

    /// Accept one mutual TLS session over an injected duplex stream.
    ///
    /// This is the blocking convenience adapter. It enforces the named
    /// pre-authentication wire and step bounds but has no clock: the caller
    /// must wrap the inherited stream in the five-second establishment
    /// deadline. It returns only after the handshake completes, the peer
    /// certificate pin matches, and ALPN selected the control profile.
    ///
    /// # Errors
    ///
    /// Returns a launch-failing [`ControlError`] on EOF, I/O failure, TLS
    /// rejection, ALPN mismatch, missing/unexpected peer certificate, or an
    /// exhausted wire/step budget.
    pub fn accept<S: Read + Write>(self, mut stream: S) -> Result<ControlSession<S>, ControlError> {
        let Self {
            config,
            peer_certificate_pin,
            establishment_expectations,
        } = self;
        let mut handshake =
            ControlServerHandshake::new(&config, peer_certificate_pin, establishment_expectations)?;
        let mut received_wire = 0_usize;
        let mut sent_wire = 0_usize;

        for _ in 0..TLS_PREAUTH_STEPS_MAX {
            if handshake.wants_read() {
                let remaining = TLS_PREAUTH_WIRE_MAX_BYTES
                    .checked_sub(received_wire)
                    .ok_or(ControlError::HandshakeWireBudgetExceeded)?;
                if remaining == 0 {
                    return Err(ControlError::HandshakeWireBudgetExceeded);
                }
                let read = {
                    let mut limited = (&mut stream).take(remaining as u64);
                    handshake.read_tls(&mut limited)?
                };
                if read == 0 {
                    return Err(ControlError::UnexpectedEof);
                }
                received_wire = received_wire
                    .checked_add(read)
                    .ok_or(ControlError::HandshakeWireBudgetExceeded)?;
            }

            handshake.process_new_packets()?;

            while handshake.wants_write() {
                let remaining = TLS_PREAUTH_WIRE_MAX_BYTES
                    .checked_sub(sent_wire)
                    .ok_or(ControlError::HandshakeWireBudgetExceeded)?;
                if remaining == 0 {
                    return Err(ControlError::HandshakeWireBudgetExceeded);
                }
                let mut limited = LimitedWriter {
                    inner: &mut stream,
                    remaining,
                    exhausted: false,
                };
                let written = match handshake.write_tls(&mut limited) {
                    Ok(written) => written,
                    Err(_) if limited.exhausted => {
                        return Err(ControlError::HandshakeWireBudgetExceeded);
                    }
                    Err(error) => return Err(ControlError::Io(error)),
                };
                if written == 0 {
                    return Err(ControlError::Io(io::Error::from(io::ErrorKind::WriteZero)));
                }
                sent_wire = sent_wire
                    .checked_add(written)
                    .ok_or(ControlError::HandshakeWireBudgetExceeded)?;
            }

            if let Some(peer) = handshake.established_peer()? {
                return Ok(handshake.into_control_session(stream, peer));
            }
        }

        Err(ControlError::HandshakeStepBudgetExceeded)
    }
}

impl fmt::Debug for ControlServer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlServer")
            .field("peer_certificate_pin", &self.peer_certificate_pin)
            .finish_non_exhaustive()
    }
}

/// Deterministic TLS handshake state machine over an injected stream.
///
/// Tests and bounded adapters drive this by alternating [`Self::read_tls`],
/// [`Self::process_new_packets`], and [`Self::write_tls`], exactly mirroring
/// the rustls connection API. No thread, socket, or clock is created.
pub struct ControlServerHandshake {
    connection: ServerConnection,
    expected_peer_pin: CertificatePin,
    establishment_expectations: EstablishmentExpectations,
}

impl ControlServerHandshake {
    fn new(
        config: &Arc<ServerConfig>,
        expected_peer_pin: CertificatePin,
        establishment_expectations: EstablishmentExpectations,
    ) -> Result<Self, ControlError> {
        Ok(Self {
            connection: ServerConnection::new(config.clone()).map_err(ControlError::Rustls)?,
            expected_peer_pin,
            establishment_expectations,
        })
    }

    /// Read raw TLS bytes from the control stream.
    ///
    /// # Errors
    ///
    /// Propagates the stream's I/O error.
    pub fn read_tls(&mut self, reader: &mut dyn Read) -> io::Result<usize> {
        self.connection.read_tls(reader)
    }

    /// Write pending raw TLS bytes to the control stream.
    ///
    /// # Errors
    ///
    /// Propagates the stream's I/O error.
    pub fn write_tls(&mut self, writer: &mut dyn Write) -> io::Result<usize> {
        self.connection.write_tls(writer)
    }

    /// Process buffered TLS records.
    ///
    /// # Errors
    ///
    /// Returns [`ControlError::Rustls`] for a fatal TLS or certificate-pin
    /// rejection.
    pub fn process_new_packets(&mut self) -> Result<(), ControlError> {
        self.connection
            .process_new_packets()
            .map(|_state| ())
            .map_err(ControlError::Rustls)
    }

    /// Whether the connection still wants to read handshake records.
    #[must_use]
    pub fn wants_read(&self) -> bool {
        self.connection.wants_read()
    }

    /// Whether the connection has pending bytes to write.
    #[must_use]
    pub fn wants_write(&self) -> bool {
        self.connection.wants_write()
    }

    /// Whether the handshake is still in progress.
    #[must_use]
    pub fn is_handshaking(&self) -> bool {
        self.connection.is_handshaking()
    }

    /// Negotiated ALPN protocol, once selected.
    #[must_use]
    pub fn alpn_protocol(&self) -> Option<&[u8]> {
        self.connection.alpn_protocol()
    }

    /// Return the established peer after validating completion, TLS version,
    /// ALPN, exact single-leaf chain, and the launch pin.
    ///
    /// Returns `Ok(None)` while the handshake is still in progress. A malformed
    /// or mismatched completion is a launch failure.
    ///
    /// # Errors
    ///
    /// Returns [`ControlError::UnexpectedProtocolVersion`],
    /// [`ControlError::AlpnNotNegotiated`],
    /// [`ControlError::MissingPeerCertificate`],
    /// [`ControlError::UnexpectedPeerCertificateChain`], or
    /// [`ControlError::PeerCertificatePinMismatch`] when completion does not
    /// match the launch contract.
    pub fn established_peer(&self) -> Result<Option<EstablishedPeer>, ControlError> {
        if self.connection.is_handshaking() {
            return Ok(None);
        }

        if self.connection.protocol_version() != Some(ProtocolVersion::TLSv1_3) {
            return Err(ControlError::UnexpectedProtocolVersion);
        }

        // rustls tolerates a peer that offers no ALPN at all; the control
        // profile requires an exact negotiated match, so re-check here before
        // exposing the session.
        if self.connection.alpn_protocol() != Some(CONTROL_ALPN) {
            return Err(ControlError::AlpnNotNegotiated);
        }

        let certificates = self
            .connection
            .peer_certificates()
            .ok_or(ControlError::MissingPeerCertificate)?;
        if certificates.len() != 1 {
            return Err(ControlError::UnexpectedPeerCertificateChain(
                certificates.len(),
            ));
        }

        let certificate = certificates[0].clone();
        let certificate_pin = CertificatePin::of_der(certificate.as_ref());
        if certificate_pin != self.expected_peer_pin {
            return Err(ControlError::PeerCertificatePinMismatch);
        }

        Ok(Some(EstablishedPeer {
            certificate,
            certificate_pin,
        }))
    }

    /// Non-secret establishment expectations captured by the server.
    ///
    /// A caller that drives the handshake manually can retain these values
    /// across [`Self::into_connection`] and build the application state
    /// machine without re-parsing the launch manifest.
    #[must_use]
    pub fn establishment_expectations(&self) -> &EstablishmentExpectations {
        &self.establishment_expectations
    }

    /// Consume a completed handshake into one control session.
    ///
    /// The caller must already have obtained the completed [`EstablishedPeer`]
    /// from [`Self::established_peer`]. This does not re-check completion; it
    /// only preserves the launch expectations that the blocking
    /// [`ControlServer::accept`] path would otherwise carry.
    #[must_use]
    pub fn into_control_session<S>(self, stream: S, peer: EstablishedPeer) -> ControlSession<S> {
        ControlSession {
            connection: self.connection,
            stream,
            peer,
            establishment_expectations: self.establishment_expectations,
        }
    }

    /// Consume the handshake and return the underlying rustls connection.
    #[must_use]
    pub fn into_connection(self) -> ServerConnection {
        self.connection
    }
}

/// Established mutual-TLS peer identity.
pub struct EstablishedPeer {
    certificate: CertificateDer<'static>,
    certificate_pin: CertificatePin,
}

impl EstablishedPeer {
    /// Complete peer leaf certificate DER (already pin-checked).
    #[must_use]
    pub fn certificate_der(&self) -> &CertificateDer<'static> {
        &self.certificate
    }

    /// SHA-256 DER pin of the peer leaf.
    #[must_use]
    pub fn certificate_pin(&self) -> CertificatePin {
        self.certificate_pin
    }

    /// Negotiated ALPN protocol. The control profile is the only accepted
    /// value, so this is always [`CONTROL_ALPN`].
    #[must_use]
    pub fn alpn_protocol(&self) -> &'static [u8] {
        CONTROL_ALPN
    }
}

impl fmt::Debug for EstablishedPeer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EstablishedPeer")
            .field("certificate_pin", &self.certificate_pin)
            .field("certificate_der_len", &self.certificate.as_ref().len())
            .field("alpn_protocol", &CONTROL_ALPN)
            .finish()
    }
}

/// One established control session over the injected stream.
///
/// This type exposes the pin-checked peer, the non-secret launch
/// expectations captured when the server was built, and the rustls
/// connection plus stream. [`crate::HelperControl`] consumes it for the
/// application record layer.
pub struct ControlSession<S> {
    connection: ServerConnection,
    stream: S,
    peer: EstablishedPeer,
    establishment_expectations: EstablishmentExpectations,
}

impl<S> ControlSession<S> {
    /// Pin-checked peer identity.
    #[must_use]
    pub fn peer(&self) -> &EstablishedPeer {
        &self.peer
    }

    /// Non-secret establishment expectations derived from the launch manifest
    /// that authenticated this session.
    ///
    /// The later application state machine compares the Core `Establish` echo
    /// against these values; no private key or certificate bytes are exposed.
    #[must_use]
    pub fn establishment_expectations(&self) -> &EstablishmentExpectations {
        &self.establishment_expectations
    }

    /// Borrow the rustls server connection.
    #[must_use]
    pub fn connection(&self) -> &ServerConnection {
        &self.connection
    }

    /// Mutably borrow the rustls server connection.
    pub fn connection_mut(&mut self) -> &mut ServerConnection {
        &mut self.connection
    }

    /// Borrow the injected control stream.
    #[must_use]
    pub fn stream(&self) -> &S {
        &self.stream
    }

    /// Mutably borrow the injected control stream.
    pub fn stream_mut(&mut self) -> &mut S {
        &mut self.stream
    }

    /// Consume the session into the rustls connection, stream, and peer.
    ///
    /// The launch expectations are dropped. Use
    /// [`Self::into_parts_with_expectations`] when the parts must be rebuilt
    /// into an application state machine.
    #[must_use]
    pub fn into_parts(self) -> (ServerConnection, S, EstablishedPeer) {
        (self.connection, self.stream, self.peer)
    }

    /// Consume the session into the rustls connection, stream, peer, and
    /// non-secret launch expectations.
    #[must_use]
    pub fn into_parts_with_expectations(
        self,
    ) -> (
        ServerConnection,
        S,
        EstablishedPeer,
        EstablishmentExpectations,
    ) {
        (
            self.connection,
            self.stream,
            self.peer,
            self.establishment_expectations,
        )
    }
}

impl<S: Read + Write> Read for ControlSession<S> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        rustls::Stream::new(&mut self.connection, &mut self.stream).read(buffer)
    }
}

impl<S: Read + Write> Write for ControlSession<S> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        rustls::Stream::new(&mut self.connection, &mut self.stream).write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        rustls::Stream::new(&mut self.connection, &mut self.stream).flush()
    }
}

/// Fail-closed control-transport error.
///
/// No variant contains key material, certificate payload bytes, or paths.
#[derive(Debug)]
pub enum ControlError {
    /// Launch-manifest structure or key material was rejected.
    Manifest(ManifestError),
    /// Fatal rustls error, including client-certificate pin rejection.
    Rustls(RustlsError),
    /// Injected stream I/O error.
    Io(io::Error),
    /// The stream closed before the handshake completed.
    UnexpectedEof,
    /// The handshake completed without the required ALPN selection.
    AlpnNotNegotiated,
    /// The peer sent no certificate.
    MissingPeerCertificate,
    /// The peer sent a chain instead of exactly one end-entity certificate.
    UnexpectedPeerCertificateChain(usize),
    /// The peer leaf digest does not equal the manifest-derived pin.
    PeerCertificatePinMismatch,
    /// The negotiated TLS version is not 1.3.
    UnexpectedProtocolVersion,
    /// The pre-authentication wire budget was exhausted.
    HandshakeWireBudgetExceeded,
    /// The pre-authentication step budget was exhausted.
    HandshakeStepBudgetExceeded,
}

impl fmt::Display for ControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(formatter, "launch material rejected: {error}"),
            Self::Rustls(error) => write!(formatter, "TLS handshake failed: {error}"),
            Self::Io(error) => write!(formatter, "control stream I/O failed: {error}"),
            Self::UnexpectedEof => {
                formatter.write_str("control stream closed before the handshake completed")
            }
            Self::AlpnNotNegotiated => {
                formatter.write_str("control ALPN protocol was not negotiated")
            }
            Self::MissingPeerCertificate => formatter.write_str("peer presented no certificate"),
            Self::UnexpectedPeerCertificateChain(count) => write!(
                formatter,
                "peer presented {count} certificates; exactly one end-entity leaf is required"
            ),
            Self::PeerCertificatePinMismatch => {
                formatter.write_str("peer certificate digest does not match the launch pin")
            }
            Self::UnexpectedProtocolVersion => {
                formatter.write_str("negotiated TLS version is not 1.3")
            }
            Self::HandshakeWireBudgetExceeded => {
                formatter.write_str("pre-authentication TLS wire budget exceeded")
            }
            Self::HandshakeStepBudgetExceeded => {
                formatter.write_str("pre-authentication TLS step budget exceeded")
            }
        }
    }
}

impl std::error::Error for ControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Manifest(error) => Some(error),
            Self::Rustls(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::UnexpectedEof
            | Self::AlpnNotNegotiated
            | Self::MissingPeerCertificate
            | Self::UnexpectedPeerCertificateChain(_)
            | Self::PeerCertificatePinMismatch
            | Self::UnexpectedProtocolVersion
            | Self::HandshakeWireBudgetExceeded
            | Self::HandshakeStepBudgetExceeded => None,
        }
    }
}

impl From<ManifestError> for ControlError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

impl From<RustlsError> for ControlError {
    fn from(error: RustlsError) -> Self {
        Self::Rustls(error)
    }
}

impl From<io::Error> for ControlError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Bounded writer for the blocking convenience path. When the pre-auth wire
/// allowance is exhausted it fails closed instead of letting rustls overshoot
/// the named bound by one record.
struct LimitedWriter<'a, W> {
    inner: &'a mut W,
    remaining: usize,
    exhausted: bool,
}

impl<W: Write> Write for LimitedWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            self.exhausted = true;
            return Err(io::Error::from(io::ErrorKind::WriteZero));
        }
        let allowed = buffer.len().min(self.remaining);
        let written = self.inner.write(&buffer[..allowed])?;
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Debug)]
struct PinnedClientCertVerifier {
    pin: CertificatePin,
    provider: Arc<CryptoProvider>,
}

impl ClientCertVerifier for PinnedClientCertVerifier {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        // No PKI hints: the client has exactly one launch leaf and must send
        // it. CertificateRequest with an empty authority list is valid.
        &[]
    }

    fn client_auth_mandatory(&self) -> bool {
        true
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, RustlsError> {
        if !intermediates.is_empty() {
            return Err(RustlsError::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ));
        }
        if CertificatePin::of_der(end_entity.as_ref()) != self.pin {
            return Err(RustlsError::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, RustlsError> {
        rustls::crypto::verify_tls12_signature(
            message,
            certificate,
            signature,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, RustlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            certificate,
            signature,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

//! Shared test fixtures and deterministic in-memory TLS pumping.
//!
//! No socket, port, thread, or clock is used. The client and server
//! `rustls` connections are driven by alternating `read_tls`/`write_tls`/
//! `process_new_packets` calls over `Vec` buffers.

#![allow(dead_code)] // Each integration test uses a different subset.

use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use bitty_ai_control::{
    CertificatePin, ControlError, ControlServerHandshake, EstablishedPeer, LaunchKeyMaterial,
    LaunchManifest,
};
use bitty_ipc_api::provider_publication::{Digest, EstablishmentEcho, Nonce, SessionId};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, aws_lc_rs};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{
    CertificateError, ClientConfig, ClientConnection, DigitallySignedStruct, Error, SignatureScheme,
};
use zeroize::Zeroizing;

/// Checked-in test-only key material.
pub struct Fixtures {
    /// Core client leaf.
    pub core_certificate: Vec<u8>,
    /// Core client PKCS#8 key.
    pub core_key: Vec<u8>,
    /// Helper server leaf.
    pub helper_certificate: Vec<u8>,
    /// Helper server PKCS#8 key.
    pub helper_key: Vec<u8>,
    /// Unpinned rogue client leaf.
    pub rogue_core_certificate: Vec<u8>,
    /// Unpinned rogue client PKCS#8 key.
    pub rogue_core_key: Vec<u8>,
    /// P-384 leaf used only for curve-rejection coverage.
    pub p384_certificate: Vec<u8>,
    /// Matching P-384 PKCS#8 key.
    pub p384_key: Vec<u8>,
}

/// Load the checked-in fixtures.
pub fn fixtures() -> Fixtures {
    Fixtures {
        core_certificate: fixture("core-cert.der"),
        core_key: fixture("core-key-pkcs8.der"),
        helper_certificate: fixture("helper-cert.der"),
        helper_key: fixture("helper-key-pkcs8.der"),
        rogue_core_certificate: fixture("rogue-core-cert.der"),
        rogue_core_key: fixture("rogue-core-key-pkcs8.der"),
        p384_certificate: fixture("p384-cert.der"),
        p384_key: fixture("p384-key-pkcs8.der"),
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    fs::read(&path).unwrap_or_else(|error| panic!("read fixture {}: {error}", path.display()))
}

/// A valid manifest bound to the fixture certificates.
pub fn valid_manifest(fixtures: &Fixtures) -> LaunchManifest {
    LaunchManifest {
        core_instance_nonce: [0x11; 32],
        launch_nonce: [0x22; 32],
        component_name: "bitty-test-core".to_owned(),
        descriptor_version: "1.0.0-test".to_owned(),
        executable_name: "bitty".to_owned(),
        executable_sha256: [0x33; 32],
        component_protocol_version: 1,
        control_channel_nonce: [0x44; 32],
        data_channel_nonce: [0x55; 32],
        delegation_purpose: "provider-routing-publication".to_owned(),
        control_profile: "provider-publication-control/1".to_owned(),
        key_material: LaunchKeyMaterial {
            profile: 1,
            key_type: 1,
            core_certificate_der: fixtures.core_certificate.clone(),
            helper_certificate_der: fixtures.helper_certificate.clone(),
            helper_private_key_pkcs8_der: Zeroizing::new(fixtures.helper_key.clone()),
        },
    }
}

/// Reference-encoded valid manifest bytes.
pub fn valid_manifest_bytes(fixtures: &Fixtures) -> Zeroizing<Vec<u8>> {
    valid_manifest(fixtures).encode()
}

/// Exact non-secret `Establish` echo for a manifest.
pub fn establishment_echo(manifest: &LaunchManifest) -> EstablishmentEcho {
    EstablishmentEcho {
        bootstrap_format_version: 1,
        control_protocol_version: 1,
        core_instance_nonce: Nonce::new(manifest.core_instance_nonce),
        launch_nonce: SessionId::new(manifest.launch_nonce),
        component_name: manifest.component_name.clone(),
        descriptor_version: manifest.descriptor_version.clone(),
        executable_name: manifest.executable_name.clone(),
        executable_sha256: Digest::new(manifest.executable_sha256),
        component_protocol_version: manifest.component_protocol_version,
        control_channel_nonce: Nonce::new(manifest.control_channel_nonce),
        data_channel_nonce: Nonce::new(manifest.data_channel_nonce),
        delegation_purpose: manifest.delegation_purpose.clone(),
        control_profile: manifest.control_profile.clone(),
        key_material_profile: manifest.key_material.profile,
        key_type: manifest.key_material.key_type,
        core_leaf_digest: Digest::of(&manifest.key_material.core_certificate_der),
        helper_leaf_digest: Digest::of(&manifest.key_material.helper_certificate_der),
    }
}

/// Build a test client that pins the helper leaf and presents the Core leaf.
pub fn client_config(fixtures: &Fixtures, alpn: &[u8], tls12_only: bool) -> Arc<ClientConfig> {
    client_config_with_identity(
        &fixtures.helper_certificate,
        &fixtures.core_certificate,
        &fixtures.core_key,
        alpn,
        tls12_only,
    )
}

/// Build a test client with an explicit identity (rogue-client tests).
pub fn client_config_with_identity(
    pinned_server_certificate: &[u8],
    client_certificate: &[u8],
    client_key: &[u8],
    alpn: &[u8],
    tls12_only: bool,
) -> Arc<ClientConfig> {
    build_client_config(
        pinned_server_certificate,
        vec![CertificateDer::from(client_certificate.to_vec())],
        client_key,
        Some(alpn),
        tls12_only,
    )
}

/// Build a test client that sends an explicit certificate chain.
pub fn client_config_with_chain(
    pinned_server_certificate: &[u8],
    client_chain: &[&[u8]],
    client_key: &[u8],
    alpn: &[u8],
    tls12_only: bool,
) -> Arc<ClientConfig> {
    let chain = client_chain
        .iter()
        .map(|certificate| CertificateDer::from(certificate.to_vec()))
        .collect();
    build_client_config(
        pinned_server_certificate,
        chain,
        client_key,
        Some(alpn),
        tls12_only,
    )
}

/// Build a test client that deliberately presents no client certificate.
pub fn client_config_without_client_auth(fixtures: &Fixtures, alpn: &[u8]) -> Arc<ClientConfig> {
    let provider = Arc::new(aws_lc_rs::default_provider());
    let pin = CertificatePin::of_der(&fixtures.helper_certificate);
    let verifier = Arc::new(PinnedServerVerifier {
        pin,
        provider: provider.clone(),
    });
    let mut config = ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("test client protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    config.alpn_protocols = vec![alpn.to_vec()];
    Arc::new(config)
}

/// Build a test client that offers no ALPN extension at all.
pub fn client_config_without_alpn(fixtures: &Fixtures, tls12_only: bool) -> Arc<ClientConfig> {
    build_client_config(
        &fixtures.helper_certificate,
        vec![CertificateDer::from(fixtures.core_certificate.clone())],
        &fixtures.core_key,
        None,
        tls12_only,
    )
}

fn build_client_config(
    pinned_server_certificate: &[u8],
    client_chain: Vec<CertificateDer<'static>>,
    client_key: &[u8],
    alpn: Option<&[u8]>,
    tls12_only: bool,
) -> Arc<ClientConfig> {
    let provider = Arc::new(aws_lc_rs::default_provider());
    let pin = CertificatePin::of_der(pinned_server_certificate);
    let verifier = Arc::new(PinnedServerVerifier {
        pin,
        provider: provider.clone(),
    });
    let versions: &[&'static rustls::SupportedProtocolVersion] = if tls12_only {
        &[&rustls::version::TLS12]
    } else {
        &[&rustls::version::TLS13]
    };

    let builder = ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(versions)
        .expect("test client protocol versions");
    let mut config = builder
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(
            client_chain,
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(client_key.to_vec())),
        )
        .expect("test client identity");

    if let Some(alpn) = alpn {
        config.alpn_protocols = vec![alpn.to_vec()];
    }
    Arc::new(config)
}

/// Build a client connection with a fixed test server name.
pub fn client_connection(config: Arc<ClientConfig>) -> ClientConnection {
    let server_name = ServerName::try_from("bitty-ai-control.test").expect("static server name");
    ClientConnection::new(config, server_name).expect("test client connection")
}

/// Outcome of one deterministic in-memory handshake pump.
#[derive(Default)]
pub struct HandshakeOutcome {
    /// First server-side failure, if any.
    pub server_error: Option<ControlError>,
    /// First client-side failure, if any (rendered for test diagnostics).
    pub client_error: Option<String>,
    /// Server-observed peer when the handshake completed successfully.
    pub peer: Option<EstablishedPeer>,
    /// Whether the server is still handshaking when the pump stopped.
    pub server_handshaking: bool,
    /// Whether the client is still handshaking when the pump stopped.
    pub client_handshaking: bool,
}

/// Drive both sides without threads or sockets.
pub fn drive_handshake(
    server: &mut ControlServerHandshake,
    client: &mut ClientConnection,
) -> HandshakeOutcome {
    let mut client_to_server = Vec::new();
    let mut server_to_client = Vec::new();
    let mut outcome = HandshakeOutcome::default();

    for _ in 0..32 {
        drain_server_writes(server, &mut server_to_client, &mut outcome);
        drain_client_writes(client, &mut client_to_server, &mut outcome);

        if !client_to_server.is_empty() {
            let mut cursor = Cursor::new(std::mem::take(&mut client_to_server));
            loop {
                match server.read_tls(&mut cursor) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(error) => {
                        outcome.server_error = Some(ControlError::Io(error));
                        break;
                    }
                }
            }
        }

        if !server_to_client.is_empty() {
            let mut cursor = Cursor::new(std::mem::take(&mut server_to_client));
            loop {
                match client.read_tls(&mut cursor) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(error) => {
                        outcome.client_error = Some(format!("read: {error}"));
                        break;
                    }
                }
            }
        }

        if outcome.server_error.is_none() {
            if let Err(error) = server.process_new_packets() {
                outcome.server_error = Some(error);
            }
        }
        if outcome.client_error.is_none() {
            if let Err(error) = client.process_new_packets() {
                outcome.client_error = Some(error.to_string());
            }
        }

        if outcome.server_error.is_none() {
            match server.established_peer() {
                Ok(Some(peer)) => outcome.peer = Some(peer),
                Ok(None) => {}
                Err(error) => outcome.server_error = Some(error),
            }
        }

        if !server.is_handshaking() && !client.is_handshaking() {
            break;
        }
        if outcome.server_error.is_some() && outcome.client_error.is_some() {
            break;
        }
        if !server.wants_write()
            && !client.wants_write()
            && client_to_server.is_empty()
            && server_to_client.is_empty()
        {
            break;
        }
    }

    outcome.server_handshaking = server.is_handshaking();
    outcome.client_handshaking = client.is_handshaking();
    outcome
}

fn drain_server_writes(
    server: &mut ControlServerHandshake,
    output: &mut Vec<u8>,
    outcome: &mut HandshakeOutcome,
) {
    while server.wants_write() {
        match server.write_tls(output) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                outcome.server_error = Some(ControlError::Io(error));
                break;
            }
        }
    }
}

fn drain_client_writes(
    client: &mut ClientConnection,
    output: &mut Vec<u8>,
    outcome: &mut HandshakeOutcome,
) {
    while client.wants_write() {
        match client.write_tls(output) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                outcome.client_error = Some(format!("write: {error}"));
                break;
            }
        }
    }
}

/// Test-only server verifier pinning the helper leaf.
#[derive(Debug)]
struct PinnedServerVerifier {
    pin: CertificatePin,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinnedServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        if !intermediates.is_empty() {
            return Err(Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ));
        }
        if CertificatePin::of_der(end_entity.as_ref()) != self.pin {
            return Err(Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
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
    ) -> Result<HandshakeSignatureValid, Error> {
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

/// Read a manifest from an in-memory reader.
pub fn parse_reader(bytes: &[u8]) -> Result<LaunchManifest, String> {
    let mut cursor = Cursor::new(bytes);
    LaunchManifest::read_from(&mut cursor).map_err(|error| error.to_string())
}

//! Deterministic in-memory mutual-TLS acceptance and refusal tests.

mod common;

use std::io::{self, Read, Write};

use bitty_ai_control::{CONTROL_ALPN, CertificatePin, ControlError, ControlServer, ManifestError};
use common::{
    client_config, client_config_with_chain, client_config_with_identity,
    client_config_without_alpn, client_config_without_client_auth, client_connection,
    drive_handshake, fixtures, valid_manifest,
};
use rustls::{CertificateError, Error};

#[test]
fn mutual_handshake_succeeds() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let expected_pin = CertificatePin::of_der(&fixtures.core_certificate);
    let server = ControlServer::new(manifest).expect("server config");
    assert_eq!(server.peer_certificate_pin(), expected_pin);

    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config(&fixtures, CONTROL_ALPN, false));
    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(
        outcome.server_error.is_none(),
        "server error: {:?}",
        outcome.server_error
    );
    assert!(
        outcome.client_error.is_none(),
        "client error: {:?}",
        outcome.client_error
    );
    assert!(!outcome.server_handshaking);
    assert!(!outcome.client_handshaking);
    assert_eq!(handshake.alpn_protocol(), Some(CONTROL_ALPN));
    assert!(
        !handshake.wants_write(),
        "server queued post-handshake bytes (session tickets)"
    );

    let peer = outcome.peer.expect("established peer");
    assert_eq!(
        peer.certificate_der().as_ref(),
        fixtures.core_certificate.as_slice()
    );
    assert_eq!(peer.certificate_pin(), expected_pin);
    assert_eq!(peer.alpn_protocol(), CONTROL_ALPN);
    assert_eq!(peer.alpn_protocol(), b"provider-publication-control/1");
}

#[test]
fn alpn_mismatch_refused() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config(
        &fixtures,
        b"provider-publication-control/2",
        false,
    ));

    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(outcome.peer.is_none());
    assert!(
        matches!(
            outcome.server_error,
            Some(ControlError::Rustls(Error::NoApplicationProtocol))
        ),
        "unexpected server error: {:?}",
        outcome.server_error
    );
    assert!(
        outcome.client_error.is_some(),
        "client should observe the ALPN alert"
    );
}

#[test]
fn missing_alpn_never_exposed() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config_without_alpn(&fixtures, false));

    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(outcome.peer.is_none());
    assert!(
        matches!(outcome.server_error, Some(ControlError::AlpnNotNegotiated)),
        "unexpected server error: {:?}",
        outcome.server_error
    );
    assert!(outcome.client_error.is_none());
}

#[test]
fn tls12_only_client_refused() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config(&fixtures, CONTROL_ALPN, true));

    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(outcome.peer.is_none());
    assert!(
        outcome.server_error.is_some(),
        "TLS 1.2-only ClientHello was not refused"
    );
    assert!(
        outcome.client_error.is_some(),
        "client should observe the protocol-version refusal"
    );
}

#[test]
fn pin_mismatch_refuses_rogue_core_leaf() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config_with_identity(
        &fixtures.helper_certificate,
        &fixtures.rogue_core_certificate,
        &fixtures.rogue_core_key,
        CONTROL_ALPN,
        false,
    ));

    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(outcome.peer.is_none());
    assert!(
        matches!(
            outcome.server_error,
            Some(ControlError::Rustls(Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure
            )))
        ),
        "unexpected server error: {:?}",
        outcome.server_error
    );
}

#[test]
fn client_certificate_required() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config_without_client_auth(&fixtures, CONTROL_ALPN));

    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(outcome.peer.is_none());
    assert!(
        matches!(
            outcome.server_error,
            Some(ControlError::Rustls(Error::NoCertificatesPresented))
        ),
        "unexpected server error: {:?}",
        outcome.server_error
    );
}

#[test]
fn certificate_chain_refused() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let mut handshake = server.handshake().expect("server handshake");
    let mut client = client_connection(client_config_with_chain(
        &fixtures.helper_certificate,
        &[
            fixtures.core_certificate.as_slice(),
            fixtures.rogue_core_certificate.as_slice(),
        ],
        &fixtures.core_key,
        CONTROL_ALPN,
        false,
    ));

    let outcome = drive_handshake(&mut handshake, &mut client);

    assert!(outcome.peer.is_none());
    assert!(
        matches!(
            outcome.server_error,
            Some(ControlError::Rustls(Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure
            )))
        ),
        "unexpected server error: {:?}",
        outcome.server_error
    );
}

#[test]
fn server_build_rejects_helper_key_mismatch() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.helper_private_key_pkcs8_der =
        zeroize::Zeroizing::new(fixtures.core_key.clone());

    match ControlServer::new(manifest) {
        Err(ControlError::Manifest(ManifestError::HelperKeyMismatch)) => {}
        Err(other) => panic!("unexpected server-build error: {other:?}"),
        Ok(_) => panic!("mismatched helper key was accepted"),
    }
}

#[test]
fn accept_rejects_immediate_eof() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");

    match server.accept(EofStream) {
        Err(ControlError::UnexpectedEof) => {}
        Err(other) => panic!("unexpected accept error: {other:?}"),
        Ok(_) => panic!("empty control stream was accepted"),
    }
}

#[test]
fn server_debug_redacts_private_key() {
    let fixtures = fixtures();
    let server = ControlServer::new(valid_manifest(&fixtures)).expect("server config");
    let rendered = format!("{server:?}");
    let key_hex = hex(&fixtures.helper_key);
    assert!(!rendered.contains(&key_hex), "key leaked: {rendered}");
    assert!(rendered.contains("CertificatePin"));
}

struct EofStream;

impl Read for EofStream {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        Ok(0)
    }
}

impl Write for EofStream {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut rendered = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut rendered, "{byte:02x}").expect("write to String");
    }
    rendered
}

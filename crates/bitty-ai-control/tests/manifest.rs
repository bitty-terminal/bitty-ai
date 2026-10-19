//! Launch-manifest acceptance and rejection vectors.

mod common;

use std::io::{self, Read};

use bitty_ai_control::{
    CONTROL_BOOTSTRAP_MAX_BYTES, CONTROL_STRING_MAX_BYTES, CertificatePin, LAUNCH_CERT_MAX_BYTES,
    LaunchManifest, ManifestError, ManifestReadError,
};
use common::{fixtures, parse_reader, valid_manifest};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

#[test]
fn valid_manifest_round_trips() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let encoded = manifest.encode();

    assert!(encoded.len() < CONTROL_BOOTSTRAP_MAX_BYTES);

    let parsed = LaunchManifest::parse(encoded.as_slice()).expect("valid manifest");
    assert_eq!(parsed.core_instance_nonce, [0x11; 32]);
    assert_eq!(parsed.launch_nonce, [0x22; 32]);
    assert_eq!(parsed.component_name, "bitty-test-core");
    assert_eq!(parsed.descriptor_version, "1.0.0-test");
    assert_eq!(parsed.executable_name, "bitty");
    assert_eq!(parsed.executable_sha256, [0x33; 32]);
    assert_eq!(parsed.component_protocol_version, 1);
    assert_eq!(parsed.control_channel_nonce, [0x44; 32]);
    assert_eq!(parsed.data_channel_nonce, [0x55; 32]);
    assert_eq!(parsed.delegation_purpose, "provider-routing-publication");
    assert_eq!(parsed.control_profile, "provider-publication-control/1");
    assert_eq!(parsed.key_material.profile, 1);
    assert_eq!(parsed.key_material.key_type, 1);
    assert_eq!(
        parsed.key_material.core_certificate_der.as_slice(),
        fixtures.core_certificate.as_slice()
    );
    assert_eq!(
        parsed.key_material.helper_certificate_der.as_slice(),
        fixtures.helper_certificate.as_slice()
    );
    assert_eq!(
        parsed.key_material.helper_private_key_pkcs8_der.as_slice(),
        fixtures.helper_key.as_slice()
    );

    let expected_pin: [u8; 32] = Sha256::digest(&fixtures.core_certificate).into();
    assert_eq!(parsed.core_certificate_pin().as_bytes(), &expected_pin);
    assert_eq!(
        parsed.core_certificate_pin(),
        CertificatePin::of_der(&fixtures.core_certificate)
    );
}

#[test]
fn read_from_accepts_exact_record() {
    let fixtures = fixtures();
    let encoded = valid_manifest(&fixtures).encode();
    let parsed = parse_reader(encoded.as_slice()).expect("exact record");
    assert_eq!(parsed.launch_nonce, [0x22; 32]);
}

#[test]
fn read_from_rejects_non_eof_close() {
    let fixtures = fixtures();
    let encoded = valid_manifest(&fixtures).encode();
    let mut reader = FailingReader {
        data: encoded.as_slice().to_vec(),
        position: 0,
    };

    match LaunchManifest::read_from(&mut reader) {
        Err(ManifestReadError::Io(error)) => {
            assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        }
        other => panic!("expected non-EOF read failure, got {other:?}"),
    }
}

#[test]
fn read_from_rejects_stream_past_total_bound() {
    let mut reader = io::Cursor::new(vec![0_u8; CONTROL_BOOTSTRAP_MAX_BYTES + 1]);
    match LaunchManifest::read_from(&mut reader) {
        Err(ManifestReadError::Manifest(ManifestError::TotalTooLarge)) => {}
        other => panic!("expected total-bound rejection, got {other:?}"),
    }
}

#[test]
fn truncated_record_rejected() {
    let fixtures = fixtures();
    let encoded = valid_manifest(&fixtures).encode();
    let mut bytes = encoded.as_slice().to_vec();
    bytes.pop();

    assert_eq!(
        parse_error(&bytes),
        ManifestError::Truncated("helper_private_key_pkcs8_der")
    );
}

#[test]
fn clean_eof_truncation_rejected() {
    let fixtures = fixtures();
    let encoded = valid_manifest(&fixtures).encode();
    let mut bytes = encoded.as_slice().to_vec();
    bytes.truncate(bytes.len() - 10);

    match parse_reader(&bytes) {
        Err(error) => assert!(error.contains("truncated")),
        Ok(_) => panic!("truncated record accepted"),
    }
}

#[test]
fn oversize_total_rejected() {
    let bytes = vec![0_u8; CONTROL_BOOTSTRAP_MAX_BYTES + 1];
    assert_eq!(parse_error(&bytes), ManifestError::TotalTooLarge);
}

#[test]
fn oversize_certificate_field_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.core_certificate_der = vec![0_u8; LAUNCH_CERT_MAX_BYTES + 1];
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::FieldTooLarge {
            field: "core_certificate_der",
            len: LAUNCH_CERT_MAX_BYTES + 1,
            max: LAUNCH_CERT_MAX_BYTES,
        }
    );
}

#[test]
fn trailing_bytes_rejected() {
    let fixtures = fixtures();
    let encoded = valid_manifest(&fixtures).encode();
    let mut bytes = encoded.as_slice().to_vec();
    bytes.push(0);

    assert_eq!(parse_error(&bytes), ManifestError::TrailingBytes);
}

#[test]
fn unknown_key_material_profile_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.profile = 2;
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::UnsupportedKeyMaterialProfile(2)
    );
}

#[test]
fn unknown_key_type_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.key_type = 2;
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::UnsupportedKeyType(2)
    );
}

#[test]
fn empty_certificate_field_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.core_certificate_der.clear();
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::EmptyField("core_certificate_der")
    );
}

#[test]
fn declared_core_cert_length_past_end_rejected() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let mut bytes = manifest.encode().as_slice().to_vec();
    let offset = core_cert_len_offset(&manifest);
    let declared = u32::try_from(fixtures.core_certificate.len() + 1).expect("length fits");
    bytes[offset..offset + 4].copy_from_slice(&declared.to_be_bytes());

    let error = parse_error(&bytes);
    assert!(
        matches!(
            error,
            ManifestError::Truncated(_)
                | ManifestError::FieldTooLarge { .. }
                | ManifestError::TrailingBytes
                | ManifestError::EmptyField(_)
        ),
        "unexpected wrong-length error: {error:?}"
    );
}

#[test]
fn unknown_bootstrap_version_rejected() {
    let fixtures = fixtures();
    let mut bytes = valid_manifest(&fixtures).encode().as_slice().to_vec();
    bytes[0..2].copy_from_slice(&2_u16.to_be_bytes());

    assert_eq!(
        parse_error(&bytes),
        ManifestError::UnsupportedBootstrapVersion(2)
    );
}

#[test]
fn unknown_control_protocol_version_rejected() {
    let fixtures = fixtures();
    let mut bytes = valid_manifest(&fixtures).encode().as_slice().to_vec();
    bytes[2..4].copy_from_slice(&2_u16.to_be_bytes());

    assert_eq!(
        parse_error(&bytes),
        ManifestError::UnsupportedControlProtocolVersion(2)
    );
}

#[test]
fn wrong_core_role_rejected() {
    let fixtures = fixtures();
    let mut bytes = valid_manifest(&fixtures).encode().as_slice().to_vec();
    bytes[4] = 9;

    assert_eq!(parse_error(&bytes), ManifestError::UnexpectedCoreRole(9));
}

#[test]
fn wrong_helper_role_rejected() {
    let fixtures = fixtures();
    let mut bytes = valid_manifest(&fixtures).encode().as_slice().to_vec();
    bytes[5] = 9;

    assert_eq!(parse_error(&bytes), ManifestError::UnexpectedHelperRole(9));
}

#[test]
fn invalid_utf8_component_name_rejected() {
    let fixtures = fixtures();
    let mut bytes = valid_manifest(&fixtures).encode().as_slice().to_vec();
    // Fixed prefix is 70 bytes; component_name's u32 length starts at 70 and
    // its first byte at 74.
    bytes[74] = 0xff;

    assert_eq!(
        parse_error(&bytes),
        ManifestError::InvalidUtf8("component_name")
    );
}

#[test]
fn empty_component_name_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.component_name.clear();
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::EmptyField("component_name")
    );
}

#[test]
fn oversized_component_name_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.component_name = "a".repeat(CONTROL_STRING_MAX_BYTES + 1);
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::FieldTooLarge {
            field: "component_name",
            len: CONTROL_STRING_MAX_BYTES + 1,
            max: CONTROL_STRING_MAX_BYTES,
        }
    );
}

#[test]
fn wrong_delegation_purpose_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.delegation_purpose = "provider-routing".to_owned();
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::UnsupportedDelegationPurpose
    );
}

#[test]
fn wrong_control_profile_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.control_profile = "provider-publication-control/2".to_owned();
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::UnsupportedControlProfile
    );
}

#[test]
fn non_p256_helper_key_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.helper_certificate_der = fixtures.p384_certificate.clone();
    manifest.key_material.helper_private_key_pkcs8_der = Zeroizing::new(fixtures.p384_key.clone());
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::HelperKeyNotP256
    );
}

#[test]
fn encoding_uses_documented_field_order() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let encoded = manifest.encode();

    assert_eq!(&encoded[0..6], &[0, 1, 0, 1, 1, 2]);
    let component_name_len = u32::from_be_bytes(
        encoded[70..74]
            .try_into()
            .expect("component name length field"),
    );
    assert_eq!(component_name_len as usize, manifest.component_name.len());

    let core_cert_len = core_cert_len_offset(&manifest);
    let declared = u32::from_be_bytes(
        encoded[core_cert_len..core_cert_len + 4]
            .try_into()
            .expect("core certificate length field"),
    );
    assert_eq!(declared as usize, fixtures.core_certificate.len());
}

#[test]
fn helper_key_mismatch_rejected() {
    let fixtures = fixtures();
    let mut manifest = valid_manifest(&fixtures);
    manifest.key_material.helper_private_key_pkcs8_der = Zeroizing::new(fixtures.core_key.clone());
    let encoded = manifest.encode();

    assert_eq!(
        parse_error(encoded.as_slice()),
        ManifestError::HelperKeyMismatch
    );
}

#[test]
fn debug_and_errors_do_not_expose_private_key() {
    let fixtures = fixtures();
    let manifest = valid_manifest(&fixtures);
    let key_hex = hex(&fixtures.helper_key);

    for rendered in [
        format!("{manifest:?}"),
        format!("{:?}", manifest.key_material),
        format!("{}", manifest.core_certificate_pin()),
        format!("{:?}", manifest.core_certificate_pin()),
    ] {
        assert!(!rendered.contains(&key_hex), "key leaked: {rendered}");
        assert!(!rendered.to_lowercase().contains(&key_hex));
    }

    assert!(format!("{manifest:?}").contains("<redacted>"));

    let mut mismatch = valid_manifest(&fixtures);
    mismatch.key_material.helper_private_key_pkcs8_der = Zeroizing::new(fixtures.core_key.clone());
    let error =
        LaunchManifest::parse(mismatch.encode().as_slice()).expect_err("mismatched key accepted");
    let rendered = format!("{error} {error:?}");
    assert!(!rendered.contains(&key_hex));
}

struct FailingReader {
    data: Vec<u8>,
    position: usize,
}

impl Read for FailingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "bootstrap closed before EOF",
            ));
        }
        let remaining = &self.data[self.position..];
        let length = remaining.len().min(buffer.len());
        buffer[..length].copy_from_slice(&remaining[..length]);
        self.position += length;
        Ok(length)
    }
}

fn parse_error(bytes: &[u8]) -> ManifestError {
    LaunchManifest::parse(bytes).expect_err("manifest unexpectedly accepted")
}

fn core_cert_len_offset(manifest: &LaunchManifest) -> usize {
    let mut offset = 2 + 2 + 1 + 1 + 32 + 32;
    offset += 4 + manifest.component_name.len();
    offset += 4 + manifest.descriptor_version.len();
    offset += 4 + manifest.executable_name.len();
    offset += 32;
    offset += 2;
    offset += 32 + 32;
    offset += 4 + manifest.delegation_purpose.len();
    offset += 4 + manifest.control_profile.len();
    offset += 1 + 1;
    offset
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut rendered = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut rendered, "{byte:02x}").expect("write to String");
    }
    rendered
}

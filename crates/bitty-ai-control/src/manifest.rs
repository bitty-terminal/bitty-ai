//! Bounded launch-manifest ingestion and certificate pinning.
//!
//! The concrete record layout is documented in the crate root. This module is
//! intentionally allocation-conservative: the total record is checked first,
//! then every `u32` length is checked against both its own field bound and the
//! bytes remaining before a `Vec`/`String` is allocated.

use std::fmt;
use std::io::{self, Read};

use rustls::crypto::CryptoProvider;
use rustls::crypto::aws_lc_rs;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::sign::CertifiedKey;
use rustls::{Error as RustlsError, InconsistentKeys, SignatureScheme};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::{
    BOOTSTRAP_FORMAT_VERSION, CONTROL_BOOTSTRAP_MAX_BYTES, CONTROL_PROFILE,
    CONTROL_PROTOCOL_VERSION, CONTROL_STRING_MAX_BYTES, CORE_ROLE, DELEGATION_PURPOSE, HELPER_ROLE,
    KEY_MATERIAL_PROFILE_MUTUAL_CERTIFICATES, KEY_TYPE_ECDSA_P256, LAUNCH_CERT_MAX_BYTES,
    LAUNCH_KEY_MAX_BYTES, RANDOM_ID_BYTES,
};

/// SHA-256 over a leaf certificate's DER encoding.
///
/// The pin is public data (a certificate digest), not a secret. It is shown in
/// hexadecimal by `Debug`/`Display` to make mismatch diagnostics usable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CertificatePin([u8; 32]);

impl CertificatePin {
    /// Derive the pin from one DER-encoded leaf certificate.
    #[must_use]
    pub fn of_der(certificate_der: &[u8]) -> Self {
        let digest = Sha256::digest(certificate_der);
        let mut pin = [0_u8; 32];
        pin.copy_from_slice(&digest);
        Self(pin)
    }

    /// Raw 32-byte digest.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hexadecimal rendering for diagnostics.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut rendered = String::with_capacity(64);
        for byte in self.0 {
            rendered.push(char::from(HEX[usize::from(byte >> 4)]));
            rendered.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        rendered
    }
}

impl fmt::Display for CertificatePin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl fmt::Debug for CertificatePin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "CertificatePin({})", self.to_hex())
    }
}

/// Launch key-material block appended to the manifest.
///
/// `Debug` deliberately redacts the private key. Drop erases the owned key
/// bytes through [`Zeroizing`].
pub struct LaunchKeyMaterial {
    /// Key-material block profile; must be `1` for mutual certificates.
    pub profile: u8,
    /// Key type; must be `1` for ECDSA P-256.
    pub key_type: u8,
    /// DER-encoded self-signed `core-supervisor` client leaf.
    pub core_certificate_der: Vec<u8>,
    /// DER-encoded self-signed `ai-helper` server leaf.
    pub helper_certificate_der: Vec<u8>,
    /// PKCS#8 DER helper private key. Redacted from `Debug`.
    pub helper_private_key_pkcs8_der: Zeroizing<Vec<u8>>,
}

impl fmt::Debug for LaunchKeyMaterial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LaunchKeyMaterial")
            .field("profile", &self.profile)
            .field("key_type", &self.key_type)
            .field("core_certificate_der_len", &self.core_certificate_der.len())
            .field(
                "helper_certificate_der_len",
                &self.helper_certificate_der.len(),
            )
            .field("helper_private_key_pkcs8_der", &"<redacted>")
            .finish()
    }
}

/// One parsed launch manifest, including the launch key-material block.
///
/// Constructing this struct directly is intentionally allowed for reference
/// encoding and tests, but the only validating production path is
/// [`LaunchManifest::parse`] (or [`LaunchManifest::read_from`]). TLS
/// configuration re-validates the key material before use, so a manually
/// assembled value cannot silently bypass the helper key check.
///
/// `Debug` prints the non-secret identity fields and redacts the private key.
pub struct LaunchManifest {
    /// Core instance nonce.
    pub core_instance_nonce: [u8; RANDOM_ID_BYTES],
    /// Fresh launch nonce; also the control session identifier later.
    pub launch_nonce: [u8; RANDOM_ID_BYTES],
    /// Component name, bounded UTF-8.
    pub component_name: String,
    /// Exact descriptor version, bounded UTF-8 (opaque to this slice).
    pub descriptor_version: String,
    /// Executable name, bounded UTF-8.
    pub executable_name: String,
    /// Verified executable SHA-256.
    pub executable_sha256: [u8; 32],
    /// Selected component protocol version (value not constrained here).
    pub component_protocol_version: u16,
    /// Control-channel nonce.
    pub control_channel_nonce: [u8; RANDOM_ID_BYTES],
    /// Data-channel nonce.
    pub data_channel_nonce: [u8; RANDOM_ID_BYTES],
    /// Sole delegation purpose; exact match required.
    pub delegation_purpose: String,
    /// Control profile; exact match required.
    pub control_profile: String,
    /// Launch key material (certificates plus the helper private key).
    pub key_material: LaunchKeyMaterial,
}

impl fmt::Debug for LaunchManifest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LaunchManifest")
            .field("core_instance_nonce", &HexSlice(&self.core_instance_nonce))
            .field("launch_nonce", &HexSlice(&self.launch_nonce))
            .field("component_name", &self.component_name)
            .field("descriptor_version", &self.descriptor_version)
            .field("executable_name", &self.executable_name)
            .field("executable_sha256", &HexSlice(&self.executable_sha256))
            .field(
                "component_protocol_version",
                &self.component_protocol_version,
            )
            .field(
                "control_channel_nonce",
                &HexSlice(&self.control_channel_nonce),
            )
            .field("data_channel_nonce", &HexSlice(&self.data_channel_nonce))
            .field("delegation_purpose", &self.delegation_purpose)
            .field("control_profile", &self.control_profile)
            .field("key_material", &self.key_material)
            .finish()
    }
}

impl LaunchManifest {
    /// Parse exactly one bounded launch-manifest record.
    ///
    /// The input must contain the complete record and nothing else. The total
    /// length is checked against [`CONTROL_BOOTSTRAP_MAX_BYTES`] before any
    /// field is read; every length is then checked against its bound and the
    /// remaining bytes before allocation.
    ///
    /// On success the helper certificate/key pair has been validated and the
    /// Core leaf pin can be derived with [`Self::core_certificate_pin`].
    ///
    /// # Errors
    ///
    /// Returns the first fail-closed rejection: oversize total, truncation,
    /// trailing bytes, unknown versions/roles/profile/key type, invalid or
    /// oversized fields, and helper key mismatch/curve mismatch.
    pub fn parse(bytes: &[u8]) -> Result<Self, ManifestError> {
        if bytes.len() > CONTROL_BOOTSTRAP_MAX_BYTES {
            return Err(ManifestError::TotalTooLarge);
        }

        let mut cursor = Cursor::new(bytes);

        let bootstrap_format_version = cursor.u16("bootstrap_format_version")?;
        if bootstrap_format_version != BOOTSTRAP_FORMAT_VERSION {
            return Err(ManifestError::UnsupportedBootstrapVersion(
                bootstrap_format_version,
            ));
        }

        let control_protocol_version = cursor.u16("control_protocol_version")?;
        if control_protocol_version != CONTROL_PROTOCOL_VERSION {
            return Err(ManifestError::UnsupportedControlProtocolVersion(
                control_protocol_version,
            ));
        }

        let core_role = cursor.u8("core_role")?;
        if core_role != CORE_ROLE {
            return Err(ManifestError::UnexpectedCoreRole(core_role));
        }

        let helper_role = cursor.u8("helper_role")?;
        if helper_role != HELPER_ROLE {
            return Err(ManifestError::UnexpectedHelperRole(helper_role));
        }

        let core_instance_nonce = cursor.array::<RANDOM_ID_BYTES>("core_instance_nonce")?;
        let launch_nonce = cursor.array::<RANDOM_ID_BYTES>("launch_nonce")?;
        let component_name = cursor.bounded_string("component_name")?;
        let descriptor_version = cursor.bounded_string("descriptor_version")?;
        let executable_name = cursor.bounded_string("executable_name")?;
        let executable_sha256 = cursor.array::<32>("executable_sha256")?;
        let component_protocol_version = cursor.u16("component_protocol_version")?;
        let control_channel_nonce = cursor.array::<RANDOM_ID_BYTES>("control_channel_nonce")?;
        let data_channel_nonce = cursor.array::<RANDOM_ID_BYTES>("data_channel_nonce")?;
        let delegation_purpose = cursor.bounded_string("delegation_purpose")?;
        let control_profile = cursor.bounded_string("control_profile")?;

        let key_material_profile = cursor.u8("key_material_profile")?;
        if key_material_profile != KEY_MATERIAL_PROFILE_MUTUAL_CERTIFICATES {
            return Err(ManifestError::UnsupportedKeyMaterialProfile(
                key_material_profile,
            ));
        }

        let key_type = cursor.u8("key_type")?;
        if key_type != KEY_TYPE_ECDSA_P256 {
            return Err(ManifestError::UnsupportedKeyType(key_type));
        }

        let core_certificate_der = cursor
            .bounded_bytes("core_certificate_der", LAUNCH_CERT_MAX_BYTES)?
            .to_vec();
        let helper_certificate_der = cursor
            .bounded_bytes("helper_certificate_der", LAUNCH_CERT_MAX_BYTES)?
            .to_vec();
        let helper_private_key_pkcs8_der = Zeroizing::new(
            cursor
                .bounded_bytes("helper_private_key_pkcs8_der", LAUNCH_KEY_MAX_BYTES)?
                .to_vec(),
        );

        if !cursor.is_empty() {
            return Err(ManifestError::TrailingBytes);
        }

        let manifest = Self {
            core_instance_nonce,
            launch_nonce,
            component_name,
            descriptor_version,
            executable_name,
            executable_sha256,
            component_protocol_version,
            control_channel_nonce,
            data_channel_nonce,
            delegation_purpose,
            control_profile,
            key_material: LaunchKeyMaterial {
                profile: key_material_profile,
                key_type,
                core_certificate_der,
                helper_certificate_der,
                helper_private_key_pkcs8_der,
            },
        };

        manifest.validate()?;
        Ok(manifest)
    }

    /// Read one bounded launch-manifest record to EOF and parse it.
    ///
    /// The reader must deliver exactly one manifest and then a clean EOF.
    /// A read error before EOF is a launch failure; a clean EOF mid-record is
    /// a truncation failure, and more than
    /// [`CONTROL_BOOTSTRAP_MAX_BYTES`] fails before parsing. This function
    /// does not impose a deadline: wrap the inherited pipe/handle in the
    /// caller's five-second establishment deadline.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestReadError::Io`] for a non-EOF read error and
    /// [`ManifestReadError::Manifest`] for every record rejection.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, ManifestReadError> {
        let mut buffer = Zeroizing::new(Vec::new());
        let mut chunk = Zeroizing::new([0_u8; 512]);

        loop {
            let read = reader.read(&mut chunk[..]).map_err(ManifestReadError::Io)?;
            if read == 0 {
                break;
            }
            let total = buffer
                .len()
                .checked_add(read)
                .ok_or(ManifestReadError::Manifest(ManifestError::TotalTooLarge))?;
            if total > CONTROL_BOOTSTRAP_MAX_BYTES {
                return Err(ManifestReadError::Manifest(ManifestError::TotalTooLarge));
            }
            buffer.extend_from_slice(&chunk[..read]);
        }

        Self::parse(&buffer).map_err(ManifestReadError::Manifest)
    }

    /// Encode the reference wire form.
    ///
    /// This is the test-vector and Core-alignment encoder, not a helper
    /// runtime operation. The returned buffer contains the private key, so it
    /// is wrapped in [`Zeroizing`] for the caller.
    #[must_use]
    pub fn encode(&self) -> Zeroizing<Vec<u8>> {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&BOOTSTRAP_FORMAT_VERSION.to_be_bytes());
        encoded.extend_from_slice(&CONTROL_PROTOCOL_VERSION.to_be_bytes());
        encoded.push(CORE_ROLE);
        encoded.push(HELPER_ROLE);
        encoded.extend_from_slice(&self.core_instance_nonce);
        encoded.extend_from_slice(&self.launch_nonce);
        write_string(&mut encoded, &self.component_name);
        write_string(&mut encoded, &self.descriptor_version);
        write_string(&mut encoded, &self.executable_name);
        encoded.extend_from_slice(&self.executable_sha256);
        encoded.extend_from_slice(&self.component_protocol_version.to_be_bytes());
        encoded.extend_from_slice(&self.control_channel_nonce);
        encoded.extend_from_slice(&self.data_channel_nonce);
        write_string(&mut encoded, &self.delegation_purpose);
        write_string(&mut encoded, &self.control_profile);
        encoded.push(self.key_material.profile);
        encoded.push(self.key_material.key_type);
        write_bytes(&mut encoded, &self.key_material.core_certificate_der);
        write_bytes(&mut encoded, &self.key_material.helper_certificate_der);
        write_bytes(
            &mut encoded,
            &self.key_material.helper_private_key_pkcs8_der,
        );
        Zeroizing::new(encoded)
    }

    /// Validate all non-secret structure and the helper key pair.
    ///
    /// [`Self::parse`] and the TLS server builder call this; direct struct
    /// construction should call it before use.
    ///
    /// # Errors
    ///
    /// Returns the first structural, exact-value, or key-material rejection.
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_non_key()?;
        let provider = aws_lc_rs::default_provider();
        validated_helper_key(self, &provider)?;
        Ok(())
    }

    /// Derive the Core leaf pin from the received certificate DER.
    #[must_use]
    pub fn core_certificate_pin(&self) -> CertificatePin {
        CertificatePin::of_der(&self.key_material.core_certificate_der)
    }

    pub(crate) fn validate_non_key(&self) -> Result<(), ManifestError> {
        validate_string("component_name", &self.component_name)?;
        validate_string("descriptor_version", &self.descriptor_version)?;
        validate_string("executable_name", &self.executable_name)?;
        validate_string("delegation_purpose", &self.delegation_purpose)?;
        validate_string("control_profile", &self.control_profile)?;

        if self.delegation_purpose != DELEGATION_PURPOSE {
            return Err(ManifestError::UnsupportedDelegationPurpose);
        }
        if self.control_profile != CONTROL_PROFILE {
            return Err(ManifestError::UnsupportedControlProfile);
        }
        if self.key_material.profile != KEY_MATERIAL_PROFILE_MUTUAL_CERTIFICATES {
            return Err(ManifestError::UnsupportedKeyMaterialProfile(
                self.key_material.profile,
            ));
        }
        if self.key_material.key_type != KEY_TYPE_ECDSA_P256 {
            return Err(ManifestError::UnsupportedKeyType(
                self.key_material.key_type,
            ));
        }

        validate_bytes(
            "core_certificate_der",
            &self.key_material.core_certificate_der,
            LAUNCH_CERT_MAX_BYTES,
        )?;
        validate_bytes(
            "helper_certificate_der",
            &self.key_material.helper_certificate_der,
            LAUNCH_CERT_MAX_BYTES,
        )?;
        validate_bytes(
            "helper_private_key_pkcs8_der",
            &self.key_material.helper_private_key_pkcs8_der,
            LAUNCH_KEY_MAX_BYTES,
        )?;
        Ok(())
    }
}

/// Load and validate the helper leaf/key pair and require ECDSA P-256.
///
/// Returns the prepared [`CertifiedKey`] so the server builder does not load
/// the key a second time.
pub(crate) fn validated_helper_key(
    manifest: &LaunchManifest,
    provider: &CryptoProvider,
) -> Result<CertifiedKey, ManifestError> {
    let certificate = CertificateDer::from(manifest.key_material.helper_certificate_der.clone());
    let key_bytes = manifest.key_material.helper_private_key_pkcs8_der.to_vec();
    // `PrivatePkcs8KeyDer::from(Vec<u8>)` takes ownership; rustls' provider
    // wraps the consumed DER in `Zeroizing`, so no separate wipe is needed here.
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_bytes));

    let certified =
        CertifiedKey::from_der(vec![certificate], key, provider).map_err(|error| match error {
            RustlsError::InconsistentKeys(_) => ManifestError::HelperKeyMismatch,
            _ => ManifestError::InvalidHelperKeyMaterial,
        })?;

    // `from_der` treats an unknown consistency result as acceptable. This
    // profile requires a verified match, so fail closed on unknown as well.
    match certified.keys_match() {
        Ok(()) => {}
        Err(RustlsError::InconsistentKeys(InconsistentKeys::KeyMismatch)) => {
            return Err(ManifestError::HelperKeyMismatch);
        }
        Err(_) => return Err(ManifestError::InvalidHelperKeyMaterial),
    }

    if certified
        .key
        .choose_scheme(&[SignatureScheme::ECDSA_NISTP256_SHA256])
        .is_none()
    {
        return Err(ManifestError::HelperKeyNotP256);
    }

    Ok(certified)
}

fn validate_string(field: &'static str, value: &str) -> Result<(), ManifestError> {
    if value.is_empty() {
        return Err(ManifestError::EmptyField(field));
    }
    if value.len() > CONTROL_STRING_MAX_BYTES {
        return Err(ManifestError::FieldTooLarge {
            field,
            len: value.len(),
            max: CONTROL_STRING_MAX_BYTES,
        });
    }
    Ok(())
}

fn validate_bytes(field: &'static str, value: &[u8], max: usize) -> Result<(), ManifestError> {
    if value.is_empty() {
        return Err(ManifestError::EmptyField(field));
    }
    if value.len() > max {
        return Err(ManifestError::FieldTooLarge {
            field,
            len: value.len(),
            max,
        });
    }
    Ok(())
}

fn write_string(encoded: &mut Vec<u8>, value: &str) {
    let length = u32::try_from(value.len()).expect("manifest string length fits in u32");
    encoded.extend_from_slice(&length.to_be_bytes());
    encoded.extend_from_slice(value.as_bytes());
}

fn write_bytes(encoded: &mut Vec<u8>, value: &[u8]) {
    let length = u32::try_from(value.len()).expect("manifest field length fits in u32");
    encoded.extend_from_slice(&length.to_be_bytes());
    encoded.extend_from_slice(value);
}

struct Cursor<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.input.len()
    }

    fn take(&mut self, length: usize, field: &'static str) -> Result<&'a [u8], ManifestError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ManifestError::Truncated(field))?;
        if end > self.input.len() {
            return Err(ManifestError::Truncated(field));
        }
        let bytes = &self.input[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, ManifestError> {
        let bytes = self.take(1, field)?;
        Ok(bytes[0])
    }

    fn u16(&mut self, field: &'static str) -> Result<u16, ManifestError> {
        let bytes = self.take(2, field)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, ManifestError> {
        let bytes = self.take(4, field)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn array<const N: usize>(&mut self, field: &'static str) -> Result<[u8; N], ManifestError> {
        let bytes = self.take(N, field)?;
        let mut value = [0_u8; N];
        value.copy_from_slice(bytes);
        Ok(value)
    }

    fn bounded_bytes(
        &mut self,
        field: &'static str,
        max: usize,
    ) -> Result<&'a [u8], ManifestError> {
        let length = self.u32(field)? as usize;
        if length > max {
            return Err(ManifestError::FieldTooLarge {
                field,
                len: length,
                max,
            });
        }
        self.take(length, field)
    }

    fn bounded_string(&mut self, field: &'static str) -> Result<String, ManifestError> {
        let bytes = self.bounded_bytes(field, CONTROL_STRING_MAX_BYTES)?;
        if bytes.is_empty() {
            return Err(ManifestError::EmptyField(field));
        }
        let value = std::str::from_utf8(bytes).map_err(|_| ManifestError::InvalidUtf8(field))?;
        Ok(value.to_owned())
    }
}

/// Fail-closed launch-manifest rejection.
///
/// No variant contains key bytes, certificate bytes, paths, or other launch
/// payload data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The complete record exceeds [`CONTROL_BOOTSTRAP_MAX_BYTES`].
    TotalTooLarge,
    /// The record ended while the named field was being read.
    Truncated(&'static str),
    /// Bytes remain after the exact record.
    TrailingBytes,
    /// The bootstrap format version is not supported.
    UnsupportedBootstrapVersion(u16),
    /// The control protocol version is not supported.
    UnsupportedControlProtocolVersion(u16),
    /// The Core role byte is not `1`.
    UnexpectedCoreRole(u8),
    /// The helper role byte is not `2`.
    UnexpectedHelperRole(u8),
    /// A field exceeds its named bound.
    FieldTooLarge {
        /// Field name.
        field: &'static str,
        /// Declared or observed length in bytes.
        len: usize,
        /// Named maximum.
        max: usize,
    },
    /// A required field is empty.
    EmptyField(&'static str),
    /// A bounded string is not valid UTF-8.
    InvalidUtf8(&'static str),
    /// The delegation purpose is not `provider-routing-publication`.
    UnsupportedDelegationPurpose,
    /// The control profile is not `provider-publication-control/1`.
    UnsupportedControlProfile,
    /// The key-material profile is not `1` (mutual certificates).
    UnsupportedKeyMaterialProfile(u8),
    /// The key type is not `1` (ECDSA P-256).
    UnsupportedKeyType(u8),
    /// The helper certificate public key does not match the private key.
    HelperKeyMismatch,
    /// The provider could not load the helper certificate or private key.
    InvalidHelperKeyMaterial,
    /// The helper key does not offer ECDSA P-256 as required by `key_type`.
    HelperKeyNotP256,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TotalTooLarge => write!(
                formatter,
                "launch manifest exceeds the {CONTROL_BOOTSTRAP_MAX_BYTES}-byte bootstrap bound"
            ),
            Self::Truncated(field) => {
                write!(formatter, "launch manifest truncated while reading {field}")
            }
            Self::TrailingBytes => {
                formatter.write_str("launch manifest has trailing bytes after the exact record")
            }
            Self::UnsupportedBootstrapVersion(version) => {
                write!(formatter, "unsupported bootstrap format version {version}")
            }
            Self::UnsupportedControlProtocolVersion(version) => {
                write!(formatter, "unsupported control protocol version {version}")
            }
            Self::UnexpectedCoreRole(role) => {
                write!(formatter, "unexpected Core role byte {role}")
            }
            Self::UnexpectedHelperRole(role) => {
                write!(formatter, "unexpected helper role byte {role}")
            }
            Self::FieldTooLarge { field, len, max } => {
                write!(
                    formatter,
                    "{field} is {len} bytes, exceeding the {max}-byte bound"
                )
            }
            Self::EmptyField(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidUtf8(field) => write!(formatter, "{field} is not valid UTF-8"),
            Self::UnsupportedDelegationPurpose => {
                formatter.write_str("delegation purpose is not the fixed control purpose")
            }
            Self::UnsupportedControlProfile => {
                formatter.write_str("control profile is not provider-publication-control/1")
            }
            Self::UnsupportedKeyMaterialProfile(profile) => {
                write!(formatter, "unsupported key-material profile {profile}")
            }
            Self::UnsupportedKeyType(key_type) => {
                write!(formatter, "unsupported key type {key_type}")
            }
            Self::HelperKeyMismatch => formatter
                .write_str("helper certificate public key does not match the helper private key"),
            Self::InvalidHelperKeyMaterial => formatter
                .write_str("helper certificate or private key could not be loaded by the provider"),
            Self::HelperKeyNotP256 => {
                formatter.write_str("helper key is not ECDSA P-256 as required by key_type 1")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

/// Bootstrap-channel read failure.
#[derive(Debug)]
pub enum ManifestReadError {
    /// The underlying reader failed before clean EOF.
    Io(io::Error),
    /// The complete record was rejected.
    Manifest(ManifestError),
}

impl fmt::Display for ManifestReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "bootstrap channel read failed: {error}"),
            Self::Manifest(error) => write!(formatter, "launch manifest rejected: {error}"),
        }
    }
}

impl std::error::Error for ManifestReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Manifest(error) => Some(error),
        }
    }
}

impl From<ManifestError> for ManifestReadError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

impl From<io::Error> for ManifestReadError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

struct HexSlice<'a>(&'a [u8]);

impl fmt::Debug for HexSlice<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CertificatePin;

    #[test]
    fn pin_is_sha256_of_certificate_der() {
        // SHA-256 of the empty byte string; a stable digest sanity check
        // independent of the checked-in fixtures.
        assert_eq!(
            CertificatePin::of_der(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}

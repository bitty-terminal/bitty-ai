//! Helper-side launch-scoped mutual-certificate control transport (slice 1).
//!
//! **Status: AI-0333 slice 1, pre-review implementation.** This crate
//! implements only the helper side of the selected launch-scoped
//! mutual-certificate profile in `bitty-ai-docs`
//! (`architecture/authenticated-provider-publication.md`): bounded
//! launch-manifest ingestion and a mutual-certificate TLS 1.3 server.
//! It does not implement control records, the application establishment
//! exchange, lifetime/publication state, the Core supervisor side, or
//! authenticated publication itself. Those remain later slices and gates.
//!
//! ## Scope and boundaries
//!
//! This is a separately owned composition boundary. The endpoint crate keeps
//! its three-edge purity policy (runtime, host, ipc-api); this crate is the
//! only place where launch material, rustls configuration, and the inherited
//! control byte stream meet. It opens no listener, socket, thread, or clock:
//! the caller injects a `Read + Write` stream and owns every deadline. The
//! design's five-second establishment deadline is a caller policy bound, not
//! implemented here.
//!
//! The crate accepts only these inputs:
//!
//! - one bounded launch-manifest record (exactly one record, then EOF);
//! - one injected duplex stream carrying TLS from its first byte.
//!
//! ## Concrete launch-manifest encoding
//!
//! All integers are unsigned big-endian. A variable byte string is a `u32`
//! byte length followed by exactly that many bytes. The record is the
//! existing field sequence from the design, followed by the launch
//! key-material block; there are no tags, padding, duplicate fields, or
//! extensions.
//!
//! | #  | Field                         | Encoding                          | Constraint                          |
//! | -- | ----------------------------- | --------------------------------- | ----------------------------------- |
//! | 1  | `bootstrap_format_version`    | `u16`                             | exactly `1`                         |
//! | 2  | `control_protocol_version`    | `u16`                             | exactly `1`                         |
//! | 3  | `core_role`                   | `u8`                              | exactly `1` (`core-supervisor`)     |
//! | 4  | `helper_role`                 | `u8`                              | exactly `2` (`ai-helper`)           |
//! | 5  | `core_instance_nonce`         | 32 raw bytes                      | fixed                               |
//! | 6  | `launch_nonce`                | 32 raw bytes                      | fixed                               |
//! | 7  | `component_name`              | bounded UTF-8 string              | 1..=`CONTROL_STRING_MAX_BYTES`      |
//! | 8  | `descriptor_version`          | bounded UTF-8 string              | 1..=`CONTROL_STRING_MAX_BYTES`      |
//! | 9  | `executable_name`             | bounded UTF-8 string              | 1..=`CONTROL_STRING_MAX_BYTES`      |
//! | 10 | `executable_sha256`           | 32 raw bytes                      | fixed                               |
//! | 11 | `component_protocol_version`  | `u16`                             | stored, value not constrained       |
//! | 12 | `control_channel_nonce`       | 32 raw bytes                      | fixed                               |
//! | 13 | `data_channel_nonce`          | 32 raw bytes                      | fixed                               |
//! | 14 | `delegation_purpose`          | bounded UTF-8 string              | exactly `provider-routing-publication` |
//! | 15 | `control_profile`             | bounded UTF-8 string              | exactly `provider-publication-control/1` |
//! | 16 | key-material `profile`        | `u8`                              | exactly `1` (mutual certificates)   |
//! | 17 | key-material `key_type`       | `u8`                              | exactly `1` (ECDSA P-256)           |
//! | 18 | `core_cert_len` + `core_cert` | `u32` + DER bytes                 | 1..=`LAUNCH_CERT_MAX_BYTES`         |
//! | 19 | `helper_cert_len` + `helper_cert` | `u32` + DER bytes             | 1..=`LAUNCH_CERT_MAX_BYTES`         |
//! | 20 | `helper_key_len` + `helper_key` | `u32` + PKCS#8 DER bytes        | 1..=`LAUNCH_KEY_MAX_BYTES`          |
//!
//! The total record is capped at [`CONTROL_BOOTSTRAP_MAX_BYTES`] (4 KiB). The
//! parser checks the total before doing any work, checks every length with
//! checked arithmetic before allocating, requires the exact field order, and
//! requires exact EOF: a trailing byte, a missing byte, or a non-EOF close
//! fails the launch.
//!
//! [`LaunchManifest::encode`] emits the same layout and exists as the
//! reference encoder for test vectors and Core-side adoption alignment. The
//! helper runtime path only parses; it never serializes launch material.
//!
//! ## Pin shape
//!
//! The Core pin is SHA-256 over the received Core leaf certificate DER, not
//! over its SPKI and not over a textual fingerprint. The custom
//! `ClientCertVerifier` receives the end-entity DER directly, so the decision
//! needs no X.509 parsing. Exactly one end-entity certificate is accepted;
//! intermediates, chains, and alternate encodings are rejected. Digests are
//! not transmitted in the manifest.
//!
//! ## Key and certificate checks
//!
//! Ingestion loads the helper PKCS#8 key through the aws-lc-rs rustls
//! provider, verifies that its public key matches the helper leaf (SPKI
//! comparison performed by rustls), and verifies that the key offers the
//! ECDSA P-256 signature scheme selected by `key_type = 1`. The helper leaf's
//! EKU, CA flag, serial, and validity window are generated by the trusted
//! supervisor; this crate intentionally does not add an X.509 policy parser
//! in slice 1. The TLS verifier's digest pin plus TLS Finished is the
//! authentication decision.
//!
//! ## TLS profile
//!
//! - TLS 1.3 only; ALPN must select `provider-publication-control/1`.
//! - Mutual authentication with one ECDSA P-256 leaf per side. The server
//!   presents only the helper leaf and requires the client leaf whose SHA-256
//!   DER digest equals the manifest-derived pin.
//! - Signature verification is delegated to the rustls provider helpers
//!   (`verify_tls12_signature` / `verify_tls13_signature`); no handwritten
//!   handshake, MAC, or certificate parser exists here.
//! - No resumption, session tickets, half-RTT/0-RTT data, secret extraction,
//!   key logging, or certificate compression.
//! - ALPN is checked again after the handshake because rustls tolerates a
//!   client that offers no ALPN at all; such a session is never exposed.
//!
//! ## Zeroization
//!
//! Owned copies of the helper private key (and the raw bootstrap buffer in
//! [`LaunchManifest::read_from`]) are held in `zeroize::Zeroizing` and erased
//! when dropped. This is cooperative cleanup of owned accessible storage
//! only: copies inside rustls internals, the aws-lc-rs provider, allocator
//! remnants, registers, kernel buffers, core dumps, and snapshots are not
//! covered, and destructors do not run after an abrupt kill. Revocation never
//! depends on successful erasure.
//!
//! ## Underspecified fields for Core adoption
//!
//! These are recorded as alignment points, not silently invented contracts:
//!
//! - `component_protocol_version` has no defined expected value in the draft
//!   design. The crate stores it and does not compare it.
//! - `descriptor_version` is called an "exact descriptor version" but its
//!   textual encoding and expected value are not defined. The crate treats it
//!   as an opaque bounded UTF-8 string; only Core's private launch expectation
//!   can compare it.
//! - The component identity fields (`component_name`, `descriptor_version`,
//!   `executable_name`, `executable_sha256`) are parsed and exposed but not
//!   locally verified in slice 1: this crate has no trusted descriptor source
//!   and takes no filesystem/environment input. A later helper-side adoption
//!   must define how the helper compares them (or record that supervisor
//!   generation is the sole trust root), and the establishment exchange must
//!   echo them exactly.
//! - The design does not say whether the Core leaf must use the same
//!   `key_type`/curve as the helper. The manifest has one `key_type` tag; the
//!   crate validates only the helper key against it and pins the Core leaf by
//!   digest. The Core leaf's curve, EKU, serial, CA flag, and validity window
//!   are checked by the supervisor that mints it, not by this helper slice.
//! - The executable SHA-256 is carried verbatim; its comparison with the
//!   verified descriptor belongs to Core's trusted launch path, not this crate.
//! - The design bounds the whole bootstrap record at 4 KiB but does not
//!   specify whether lengths must be minimal or whether empty identity strings
//!   are legal. The crate requires non-empty identity strings and treats the
//!   declared lengths literally.
//! - The five-second establishment deadline and all TLS work/rate budgets are
//!   caller-owned adapter policy; this crate exposes named byte/step bounds
//!   only for its blocking convenience path.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod manifest;
pub mod server;

pub use manifest::{
    CertificatePin, LaunchKeyMaterial, LaunchManifest, ManifestError, ManifestReadError,
};
pub use server::{
    ControlError, ControlServer, ControlServerHandshake, ControlSession, EstablishedPeer,
};

/// Bootstrap/control protocol version accepted by this slice.
pub const BOOTSTRAP_FORMAT_VERSION: u16 = 1;

/// Control protocol version accepted by this slice.
pub const CONTROL_PROTOCOL_VERSION: u16 = 1;

/// Fixed Core-supervisor role byte (TLS client).
pub const CORE_ROLE: u8 = 1;

/// Fixed AI-helper role byte (TLS server).
pub const HELPER_ROLE: u8 = 2;

/// Sole delegation purpose accepted by this slice.
pub const DELEGATION_PURPOSE: &str = "provider-routing-publication";

/// Sole control profile and ALPN token accepted by this slice.
pub const CONTROL_PROFILE: &str = "provider-publication-control/1";

/// ALPN token, the byte form of [`CONTROL_PROFILE`].
pub const CONTROL_ALPN: &[u8] = CONTROL_PROFILE.as_bytes();

/// Key-material block profile for launch-scoped mutual certificates.
pub const KEY_MATERIAL_PROFILE_MUTUAL_CERTIFICATES: u8 = 1;

/// Key-material block key type for ECDSA P-256.
pub const KEY_TYPE_ECDSA_P256: u8 = 1;

/// Bound on the complete one-shot launch manifest record.
pub const CONTROL_BOOTSTRAP_MAX_BYTES: usize = 4 * 1024;

/// Bound on one DER-encoded leaf certificate.
pub const LAUNCH_CERT_MAX_BYTES: usize = 1024;

/// Bound on the helper PKCS#8 private key.
pub const LAUNCH_KEY_MAX_BYTES: usize = 256;

/// Bound on each manifest variable string.
pub const CONTROL_STRING_MAX_BYTES: usize = 128;

/// Length of launch/channel nonces.
pub const RANDOM_ID_BYTES: usize = 32;

/// Named logical handshake ceiling from the design (64 KiB per peer).
///
/// This adapter does not need a second logical-byte counter: exactly one leaf
/// per side plus [`LAUNCH_CERT_MAX_BYTES`] / [`LAUNCH_KEY_MAX_BYTES`] keeps the
/// handshake flight well below this ceiling. [`TLS_PREAUTH_WIRE_MAX_BYTES`] is
/// the enforced raw-wire admission bound.
pub const CONTROL_HANDSHAKE_MAX_BYTES: usize = 64 * 1024;

/// Pre-authentication wire bound per direction used by the blocking helper.
pub const TLS_PREAUTH_WIRE_MAX_BYTES: usize = 128 * 1024;

/// Pre-authentication bounded processing steps used by the blocking helper.
pub const TLS_PREAUTH_STEPS_MAX: usize = 256;

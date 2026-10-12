//! Peer SPKI consumer seam for the remote (HTTP/SSE) transport.
//!
//! [`PeerKeyProvider`] answers one question per request path: what is the
//! endpoint's observed SPKI SHA-256 (32 raw bytes), or `None` when the key is
//! unavailable? [`crate::http_transport::HttpLineTransport`] consults the
//! attached provider BEFORE any request bytes whenever
//! [`crate::remote::RemoteServerConfig::spki_pins`] is non-empty; mismatch or
//! absent key refuses fail-closed as [`crate::error::McpFailure::TransportClosed`]
//! with zero service calls. Diagnostics are names-only: neither pin bytes nor
//! observed key bytes ever enter errors or logs. When no pins are configured
//! (unpinned), the provider is never consulted and behavior is byte-identical
//! to before.
//!
//! Unblock trigger (cross-repo, NOT this task): the real implementation is a
//! `NetworkService`-backed provider that observes the TLS peer key once
//! `bitty-network-api` exposes a server peer-key accessor. Verified absent
//! today: the api surface carries only client-identity config
//! (`ClientIdentity` / `ClientIdentityRule` / `TlsConfig` plus CA roots), so
//! no production provider can be built here. Until that accessor lands,
//! [`FakePeerKeys`] (exact-URL static map, test-only) is the only source, and
//! every test below runs fully offline.
//!
//! No digest computation lives here: the provider returns digests and the
//! transport compares them against the configured pins, so the pure-Rust
//! SHA-256 in [`crate::supervise`] is deliberately NOT reused (that module is
//! untouched; hex decode plus constant-time equality are duplicated here in
//! the established bool-helper style and pinned by tests below).

use crate::config::SHA256_PIN_LEN;

/// Endpoint peer-key source consulted pre-request by pinned transports.
///
/// Returns the observed SPKI SHA-256 for `endpoint_url` as 32 raw bytes, or
/// `None` when the key cannot be observed (no TLS session, unsupported
/// backend, unknown endpoint). The fake keys by exact endpoint URL string; a
/// future `NetworkService`-backed implementation will key by host and port and
/// ignore the path (see the module docs for the unblock trigger).
pub trait PeerKeyProvider {
    /// Observed SPKI SHA-256 for `endpoint_url`, or `None` when unavailable.
    fn peer_spki_sha256(&self, endpoint_url: &str) -> Option<[u8; 32]>;
}

/// Static exact-URL peer-key map for offline tests (the only source until
/// `bitty-network-api` exposes a peer-key accessor).
///
/// Keys by exact endpoint URL string (`FakePeerKeys::insert_hex` records one
/// entry; unlisted URLs report absent). Shape-validated on insert: malformed
/// hex is refused with `false` and records nothing.
#[derive(Debug, Clone, Default)]
pub struct FakePeerKeys {
    entries: Vec<(String, [u8; 32])>,
}

impl FakePeerKeys {
    /// Empty map: every lookup reports absent.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record `digest_hex` (exactly 64 lowercase hex digits) for `url`.
    ///
    /// Returns `true` on success; on shape violation returns `false` and
    /// records nothing (fail-closed insert, no panic, no value echo by the
    /// caller convention).
    pub fn insert_hex(&mut self, url: &str, digest_hex: &str) -> bool {
        let Some(digest) = decode_spki_pin(digest_hex) else {
            return false;
        };
        if let Some(slot) = self.entries.iter_mut().find(|(known, _)| known == url) {
            slot.1 = digest;
        } else {
            self.entries.push((url.to_owned(), digest));
        }
        true
    }

    /// Number of recorded entries (shape probe for tests).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any entry is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl PeerKeyProvider for FakePeerKeys {
    fn peer_spki_sha256(&self, endpoint_url: &str) -> Option<[u8; 32]> {
        self.entries
            .iter()
            .find(|(known, _)| known == endpoint_url)
            .map(|(_, digest)| *digest)
    }
}

/// Decode a 64-character lowercase-hex pin to 32 bytes (`None` on shape
/// violation; callers fail closed without echoing the input).
///
/// Same helper style as the stdio digest pin (duplicated, not shared: that
/// module stays untouched per scope).
pub(crate) fn decode_spki_pin(pin: &str) -> Option<[u8; 32]> {
    if pin.len() != SHA256_PIN_LEN {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in pin.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_val(chunk[0])?;
        let lo = hex_val(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

/// Single lowercase-hex digit value (`None` for anything else, including
/// uppercase: pins are strictly `0-9a-f`).
fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Constant-time 32-byte equality (accumulated difference, no early exit).
pub(crate) fn spki_digest_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (a, b) in left.iter().zip(right.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Whether `observed` matches any well-formed entry of `pins`.
///
/// Malformed pin entries never match (defense in depth: config validation
/// already refused them, but the transport consult must stay fail-closed even
/// if a malformed pin ever reaches it). Never echoes bytes; returns a bool.
pub(crate) fn spki_pin_matches(pins: &[String], observed: &[u8; 32]) -> bool {
    pins.iter()
        .any(|pin| decode_spki_pin(pin).is_some_and(|expected| spki_digest_eq(observed, &expected)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD_HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const OTHER_HEX: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

    #[test]
    fn fake_reports_recorded_absent_and_overwrite() {
        let mut keys = FakePeerKeys::new();
        assert!(keys.is_empty());
        assert_eq!(keys.len(), 0);
        assert_eq!(keys.peer_spki_sha256("https://mcp.example.com/rpc"), None);
        assert!(keys.insert_hex("https://mcp.example.com/rpc", GOOD_HEX));
        assert!(!keys.is_empty());
        assert_eq!(keys.len(), 1);
        let first = keys
            .peer_spki_sha256("https://mcp.example.com/rpc")
            .expect("recorded");
        assert!(spki_pin_matches(&[GOOD_HEX.to_owned()], &first));
        // Unknown URL stays absent; overwrite replaces in place.
        assert_eq!(keys.peer_spki_sha256("https://other.example/x"), None);
        assert!(keys.insert_hex("https://mcp.example.com/rpc", OTHER_HEX));
        assert_eq!(keys.len(), 1);
        let second = keys
            .peer_spki_sha256("https://mcp.example.com/rpc")
            .expect("overwritten");
        assert!(!spki_digest_eq(&first, &second));
        assert!(spki_pin_matches(&[OTHER_HEX.to_owned()], &second));
    }

    #[test]
    fn fake_insert_refuses_malformed_hex() {
        let mut keys = FakePeerKeys::new();
        for bad in [
            "",
            "abc",
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85g",
            "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8",
        ] {
            assert!(!keys.insert_hex("https://mcp.example.com/rpc", bad));
        }
        assert!(keys.is_empty());
    }

    #[test]
    fn decode_and_equality_are_shape_exact() {
        assert!(decode_spki_pin(GOOD_HEX).is_some());
        assert_eq!(decode_spki_pin(""), None);
        assert_eq!(decode_spki_pin(&format!("{GOOD_HEX}00")), None);
        let zero = [0u8; 32];
        let mut one = [0u8; 32];
        one[31] = 1;
        assert!(spki_digest_eq(&zero, &zero));
        assert!(!spki_digest_eq(&zero, &one));
        // Malformed pins never match, even against themselves.
        assert!(!spki_pin_matches(&["not-hex".to_owned()], &zero));
        assert!(!spki_pin_matches(&[], &zero));
        assert!(spki_pin_matches(
            &["not-hex".to_owned(), GOOD_HEX.to_owned()],
            &decode_spki_pin(GOOD_HEX).expect("good")
        ));
    }
}

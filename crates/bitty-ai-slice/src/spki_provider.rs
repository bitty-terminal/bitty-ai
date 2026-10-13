//! Real SPKI provider experiment over the `bitty-network-tls` peer accessor.
//!
//! **Status: slice-only experiment (AI-0259), not a shipped component.** This
//! module lives in `bitty-ai-slice` — the pressure-test harness, never a
//! shipped crate — and exists to prove that the [`bitty_ai_mcp::PeerKeyProvider`]
//! consumer seam (AI-0254) can be backed by a real TLS observation now that
//! the network side exposes one. Nothing here changes `bitty-ai-mcp`: that
//! crate's sources are untouched, and promotion of a real provider into the
//! mcp crate would need an owner decision (see below).
//!
//! ## Unblock chain
//!
//! - AI-0254 defined the consumer seam: [`bitty_ai_mcp::PeerKeyProvider`]
//!   (`peer_spki_sha256(endpoint_url) -> Option<[u8; 32]>`), the
//!   [`bitty_ai_mcp::FakePeerKeys`] exact-URL static map (test-only), and the
//!   fail-closed `check_spki_pin` consult on all three
//!   [`bitty_ai_mcp::HttpLineTransport`] request paths (POST, poll GET,
//!   shutdown DELETE).
//! - CTX-0073 (bitty-network PR #89) exposed the producer accessor:
//!   [`peer_leaf_spki_der`](bitty_network_tls::peer_leaf_spki_der)
//!   opens one direct TLS connection, completes exactly one handshake, and
//!   returns the leaf certificate's `subjectPublicKeyInfo` DER. Hashing stays
//!   consumer-side by network design: the accessor slices bytes, it never
//!   hashes.
//! - This experiment (AI-0259) closes the loop slice-side: parse host/port
//!   from the endpoint URL, fetch the SPKI DER through the accessor, and hash
//!   it to the 32-byte digest the seam compares.
//!
//! ## Hash-reuse verdict (no new registry edge)
//!
//! `bitty-ai-mcp` deliberately keeps its own minimal in-crate SHA-256 for the
//! stdio digest pin (AI-0248) and that hasher is private — reusing it here
//! would mean either exporting it (touching mcp sources, out of scope) or
//! depending on `bitty-ai-session` (which would drag vendored SQLite into the
//! graph for no behavioral gain; the AI-0248 record rejects exactly that).
//! This crate already carries the maintained `sha2 = "=0.11.0"` pin for the
//! snapshot-ingest experiment, so the digest below reuses that exact pin:
//! zero new registry dependencies, zero new transitive crates.
//!
//! ## Keying difference from the fake (intentional)
//!
//! [`bitty_ai_mcp::FakePeerKeys`] keys by exact endpoint URL string; the real
//! provider keys by host and port and ignores the path, query, and fragment
//! (the TLS handshake never sees them). Pin matching compares digests, so the
//! coarser keying is safe: two paths on one origin share one TLS identity.
//!
//! ## What promotion to the mcp crate would require (owner decision)
//!
//! - A dependency review for `bitty-network-tls` (rustls, platform verifier,
//!   PKI types) inside the mcp crate, which is std-only today by owner
//!   decision (AI-0179, AI-0248).
//! - A hash decision there: reuse of a maintained SHA-256 versus extending
//!   the in-crate hasher, with the same supply-chain review.
//! - A trust-policy decision: which roots the provider verifies against and
//!   who configures them (this experiment uses native-only trust).
//! - The offline-first capability discipline: the default provider carries
//!   no grant (every lookup is `None` with zero socket contact); a grant
//!   arrives only through [`NetworkSpkiProvider::with_capability`].
//!
//! [`bitty_ai_mcp::HttpLineTransport`]: https://github.com/bitty-terminal/bitty-ai

use std::fmt;
use std::time::Duration;

use bitty_ai_mcp::PeerKeyProvider;
use bitty_network_api::NetworkCapability;
use bitty_network_tls::{TlsProvider, peer_leaf_spki_der};
use sha2::Digest;

/// Default HTTPS port when the endpoint URL carries no explicit port.
const DEFAULT_HTTPS_PORT: u16 = 443;

/// Longest host this provider will dial (mirrors the mcp
/// `MAX_REMOTE_HOST_LEN` bound so absurd inputs fail closed at parse).
const MAX_HOST_LEN: usize = 253;

/// Real [`PeerKeyProvider`] over the `bitty-network-tls` peer accessor.
///
/// Parses host/port from the endpoint URL, fetches the leaf SPKI DER with the
/// configured base capability plus a native-only [`TlsProvider`], and SHA-256
/// hashes the DER consumer-side. Any failure — missing grant, malformed URL,
/// or unparsable certificate — reports `None` (unavailable key), never a
/// panic and never key bytes in an error.
#[derive(Debug)]
pub struct NetworkSpkiProvider {
    /// Native-only trust provider (no custom roots, no client identity).
    tls: TlsProvider,
    /// Optional restrictive base grant enforced before any socket work. When
    /// `None` (default) every lookup returns `None` with zero socket contact
    /// (fail-closed: no self-grant); when `Some`, the host/port must first
    /// pass this grant or the lookup returns `None` with zero socket contact.
    base_capability: Option<NetworkCapability>,
    /// Whole-fetch bound (DNS, TCP, handshake together); `None` applies the
    /// accessor default (`bitty_network_tls::DEFAULT_PEER_TIMEOUT`).
    timeout: Option<Duration>,
}

impl NetworkSpkiProvider {
    /// Provider with native-only trust, no grant, and the accessor
    /// default timeout. Performs no I/O at construction. Every lookup
    /// returns `None` until a grant arrives via [`with_capability`]
    /// (fail-closed default: no self-grant, zero socket contact).
    #[must_use]
    pub fn new() -> Self {
        Self {
            tls: TlsProvider::native_only(),
            base_capability: None,
            timeout: None,
        }
    }

    /// Enforce `capability` before any socket work (restrictive mode).
    ///
    /// The observed host/port must pass `capability.check_host_port` or the
    /// lookup returns `None` without calling the accessor at all. A deny-all
    /// grant makes every lookup `None` with zero socket contact.
    #[must_use]
    pub fn with_capability(mut self, capability: NetworkCapability) -> Self {
        self.base_capability = Some(capability);
        self
    }

    /// Bound the whole fetch (DNS, TCP, handshake together).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Fetch the SPKI SHA-256 for one `host:port` (path-insensitive).
    ///
    /// Returns `None` on any failure: capability denial (zero socket),
    /// transport failure, or unparsable certificate. Never panics, never
    /// echoes key material.
    pub fn fetch_for_host(&self, host: &str, port: u16) -> Option<[u8; 32]> {
        let base = self.base_capability.as_ref()?;
        base.check_host_port(host, port).ok()?;
        let capability = base.clone();
        let der = peer_leaf_spki_der(host, port, &capability, &self.tls, self.timeout).ok()?;
        Some(spki_sha256_of_der(&der))
    }
}

impl Default for NetworkSpkiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerKeyProvider for NetworkSpkiProvider {
    fn peer_spki_sha256(&self, endpoint_url: &str) -> Option<[u8; 32]> {
        let (host, port) = parse_https_host_port(endpoint_url)?;
        self.fetch_for_host(&host, port)
    }
}

impl fmt::Display for NetworkSpkiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NetworkSpkiProvider(restrictive={}, timeout={})",
            self.base_capability.is_some(),
            self.timeout
                .map_or("default".to_owned(), |t| format!("{}ms", t.as_millis()))
        )
    }
}

/// SHA-256 over SPKI DER bytes (reuses this crate's existing `sha2` pin; see
/// the module-level hash-reuse verdict).
fn spki_sha256_of_der(der: &[u8]) -> [u8; 32] {
    let digest = sha2::Sha256::digest(der);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Parse `(host, port)` from an `https://` endpoint URL, or `None`.
///
/// Accepts only lowercase-`https` URLs with a non-empty authority, no
/// userinfo, and no control/space bytes; the path, query, and fragment are
/// ignored (the handshake never sees them). A bare host defaults to 443; an
/// explicit port must be a non-zero `u16`. IPv6 literals stay bracketed only
/// at parse — the brackets are stripped and the inner address returned, since
/// the accessor canonicalizes the host itself.
///
/// Every refusal is fail-closed: the caller reports unavailable without
/// opening a socket.
fn parse_https_host_port(url: &str) -> Option<(String, u16)> {
    let rest = url.strip_prefix("https://")?;
    if rest.is_empty() {
        return None;
    }
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .filter(|a| !a.is_empty())?;
    if authority.contains('@') {
        return None;
    }
    if authority.bytes().any(|b| b.is_ascii_control() || b == b' ') {
        return None;
    }
    if let Some(bracketed) = authority.strip_prefix('[') {
        let end = bracketed.find(']')?;
        let host = &bracketed[..end];
        if host.is_empty() || host.len() > MAX_HOST_LEN {
            return None;
        }
        let tail = &bracketed[end + 1..];
        let port = if tail.is_empty() {
            DEFAULT_HTTPS_PORT
        } else {
            parse_explicit_port(tail.strip_prefix(':')?)?
        };
        return Some((host.to_owned(), port));
    }
    if let Some((head, tail)) = authority.rsplit_once(':') {
        if tail.is_empty() || !tail.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if head.is_empty() || head.len() > MAX_HOST_LEN || head.contains(':') {
            return None;
        }
        return Some((head.to_owned(), parse_explicit_port(tail)?));
    }
    if authority.len() > MAX_HOST_LEN {
        return None;
    }
    Some((authority.to_owned(), DEFAULT_HTTPS_PORT))
}

/// Parse a non-zero explicit port, or `None` (fail-closed: port 0 can never
/// be dialed, and overflow is unrepresentable).
fn parse_explicit_port(text: &str) -> Option<u16> {
    let port: u16 = text.parse().ok()?;
    if port == 0 { None } else { Some(port) }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::time::Duration;

    use bitty_network_api::NetworkError;

    /// Deny-everything provider: every lookup must return `None` with zero
    /// socket contact (the Nothing-To-Observe shape the offline tests pin).
    fn denied_provider() -> NetworkSpkiProvider {
        NetworkSpkiProvider::new().with_capability(NetworkCapability::offline())
    }

    /// Fail-closed default: a provider with no grant fetches nothing (no
    /// self-grant, zero socket contact) — CodeRabbit Minor on PR #433.
    #[test]
    fn default_provider_without_grant_fetches_nothing() {
        let bare = NetworkSpkiProvider::new();
        assert_eq!(bare.fetch_for_host("example.com", 443), None);
        assert_eq!(bare.peer_spki_sha256("https://example.com/rpc"), None);
        assert_eq!(
            NetworkSpkiProvider::default().fetch_for_host("example.com", 443),
            None
        );
    }

    #[test]
    fn deny_capability_returns_none_with_zero_socket() {
        let denied = denied_provider();
        assert_eq!(denied.peer_spki_sha256("https://example.com/rpc"), None);
        assert_eq!(denied.fetch_for_host("example.com", 443), None);
        // Accessor-level taxonomy pins the pre-socket gate this relies on:
        // deny-all yields `Offline`, and an allowlist miss yields `Denied`
        // (post-capability transport failures never produce `Denied`, so the
        // miss case proves zero-socket refusal, not a failed dial).
        let tls = TlsProvider::native_only();
        let quick = Some(Duration::from_millis(200));
        let offline = NetworkCapability::offline();
        assert!(matches!(
            peer_leaf_spki_der("example.com", 443, &offline, &tls, quick),
            Err(NetworkError::Offline)
        ));
        let miss = NetworkCapability::offline().with_domain("other.example");
        assert!(matches!(
            peer_leaf_spki_der("example.com", 443, &miss, &tls, quick),
            Err(NetworkError::Denied { .. })
        ));
    }

    #[test]
    fn malformed_urls_return_none_without_socket() {
        let denied = denied_provider();
        // Each entry must fail URL parsing (never reach the accessor): the
        // deny-all grant is belt-and-braces so a parser regression still
        // cannot open a socket in these offline tests.
        const BAD: &[&str] = &[
            "",
            "   ",
            "gibberish",
            "http://example.com/rpc",
            "HTTPS://example.com/rpc",
            "ftp://example.com/x",
            "https://",
            "https:///path",
            "https://?q=1",
            "https://#f",
            "https://user@example.com/",
            "https://user:pass@example.com:443/x",
            "https://example.com:99999/x",
            "https://example.com:0/x",
            "https://example.com:/x",
            "https://example.com:443a/x",
            "https://exam ple.com/",
            "https://exam\tple.com/",
            "https://a:b:c/x",
            "https://[::1",
            "https://[]/x",
            "https://[2001:db8::1]extra/x",
        ];
        for url in BAD {
            assert_eq!(parse_https_host_port(url), None, "must refuse {url:?}");
            assert_eq!(
                denied.peer_spki_sha256(url),
                None,
                "must report unavailable for {url:?}"
            );
        }
    }

    #[test]
    fn parse_pins_host_port_shape_and_ignores_path() {
        const GOOD: &[(&str, &str, u16)] = &[
            ("https://example.com/rpc", "example.com", 443),
            ("https://example.com/", "example.com", 443),
            ("https://example.com", "example.com", 443),
            ("https://example.com:8443/a?b#c", "example.com", 8443),
            ("https://192.0.2.1:443/x", "192.0.2.1", 443),
            ("https://[2001:db8::1]/x", "2001:db8::1", 443),
            ("https://[2001:db8::1]:8443/x", "2001:db8::1", 8443),
        ];
        for (url, host, port) in GOOD {
            assert_eq!(
                parse_https_host_port(url),
                Some(((*host).to_owned(), *port)),
                "must parse {url:?}"
            );
        }
        // Same origin, different paths: one TLS identity (the keying
        // difference from FakePeerKeys' exact-URL map, by network design).
        let (first_host, first_port) =
            parse_https_host_port("https://mcp.example.com/rpc").expect("parse");
        let (second_host, second_port) =
            parse_https_host_port("https://mcp.example.com/other/path?q=1").expect("parse");
        assert_eq!((first_host, first_port), (second_host, second_port));
    }

    #[test]
    fn fake_keys_compat_shape() {
        use bitty_ai_mcp::{FakePeerKeys, PeerKeyProvider as _};

        const URL: &str = "https://mcp.example.com/rpc";
        const GOOD_HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        const OTHER_HEX: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        // Unavailable key reads `None` on both sources through the same
        // trait shape (`Option<[u8; 32]>` keyed by endpoint URL string).
        let fake = FakePeerKeys::new();
        assert_eq!(fake.peer_spki_sha256(URL), None);
        assert_eq!(denied_provider().peer_spki_sha256(URL), None);
        // Fake lookup semantics the live test leans on: exact-URL record,
        // overwrite-in-place, malformed insert refused.
        let mut keys = FakePeerKeys::new();
        assert!(keys.insert_hex(URL, GOOD_HEX));
        let first = keys.peer_spki_sha256(URL).expect("recorded");
        assert!(keys.insert_hex(URL, OTHER_HEX));
        assert_eq!(keys.len(), 1);
        let second = keys.peer_spki_sha256(URL).expect("overwritten");
        assert_ne!(first, second);
        assert!(!keys.insert_hex(URL, "not-hex"));
        assert_eq!(keys.peer_spki_sha256("https://other.example/x"), None);
    }

    /// Live handshake-only probe: one public host, TLS handshake only — no
    /// HTTP bytes, no billing, no secrets.
    ///
    /// Fetches the SPKI digest twice (stability + non-degenerate shape),
    /// then drives a pinned [`bitty_ai_mcp::HttpLineTransport`] over a
    /// recording fake service with this real provider attached: the pin
    /// check passes on the live-observed key and the POST reaches the fake
    /// service (exactly one service call), so zero real HTTP bytes leave the
    /// host — only the handshake itself is live.
    #[test]
    #[ignore = "live TLS handshake to a public host; run manually only, no secrets"]
    fn live_fetch_stable_digest_over_handshake_only() {
        use bitty_ai_mcp::{HttpLineTransport, McpTransport, RemoteServerConfig};

        struct RecordingService {
            requests: Mutex<Vec<bitty_network_api::Request>>,
        }

        impl bitty_network_api::NetworkService for RecordingService {
            type Socket = ();

            fn request(
                &self,
                request: &bitty_network_api::Request,
            ) -> Result<bitty_network_api::Response, NetworkError> {
                if let Ok(mut guard) = self.requests.lock() {
                    guard.push(request.clone());
                }
                Ok(bitty_network_api::Response {
                    status: 200,
                    headers: vec![("content-type".to_owned(), "application/json".to_owned())],
                    body: br#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_vec(),
                })
            }

            fn websocket(
                &self,
                _request: &bitty_network_api::WebSocketRequest,
            ) -> Result<Self::Socket, NetworkError> {
                Err(NetworkError::Offline)
            }
        }

        const URL: &str = "https://example.com/rpc";
        let bound = Duration::from_secs(10);
        let grant = NetworkCapability::offline().with_domain("example.com");
        let probe = NetworkSpkiProvider::new()
            .with_capability(grant)
            .with_timeout(bound);
        let first = probe
            .peer_spki_sha256(URL)
            .expect("live handshake must observe a key");
        assert!(
            first.iter().any(|byte| *byte != 0),
            "digest must be non-degenerate"
        );
        let second = probe
            .peer_spki_sha256(URL)
            .expect("second live handshake must observe a key");
        assert_eq!(first, second, "digest must be stable across fetches");

        let config = RemoteServerConfig {
            id: "live".to_owned(),
            url: URL.to_owned(),
            timeout_ms: 1_000,
            tool_allowlist: vec!["echo".to_owned()],
            headers: Vec::new(),
            capability: NetworkCapability::offline().with_domain("example.com"),
            spki_pins: vec![hex::encode(first)],
        };
        let service = RecordingService {
            requests: Mutex::new(Vec::new()),
        };
        let mut transport = HttpLineTransport::new(config, service)
            .expect("pinned config validates")
            .with_peer_keys(
                NetworkSpkiProvider::new()
                    .with_capability(NetworkCapability::offline().with_domain("example.com"))
                    .with_timeout(bound),
            );
        transport
            .send_line(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#)
            .expect("pinned live key must pass the pin check");
        let line = transport.recv_line(100).expect("recv").expect("queued");
        assert!(line.contains("\"id\":1"));
        let recorded = transport
            .service()
            .requests
            .lock()
            .map(|guard| guard.len())
            .unwrap_or_default();
        assert_eq!(
            recorded, 1,
            "pin check must pass pre-request (exactly one service call)"
        );
    }
}

//! Fail-closed remote (HTTP/SSE) MCP server configuration.
//!
//! [`RemoteServerConfig`] is pure data plus [`RemoteServerConfig::validate`]:
//! the endpoint must be `http`/`https` with no userinfo, the host must be
//! parseable, the whole-operation timeout stays `1..=MAX_TIMEOUT_MS`, and the
//! tool allowlist mirrors [`crate::config`] (non-empty, at most 32 entries of
//! `1..=64` bytes over `[a-zA-Z0-9_.-]`). Custom headers are optional,
//! bounded, and never echo values into errors or `Debug`.
//!
//! The network [`bitty_network_api::NetworkCapability`] grant travels with
//! the config but is enforced per request by
//! [`crate::http_transport::HttpLineTransport`] through
//! `capability.check_request` before any socket work, so a denied or offline
//! capability fails closed with zero service calls. Secrets (for example an
//! `Authorization` header value) are carried only as opaque bytes in
//! `headers`; validation and errors quote shapes and bounds, never values.

use crate::MAX_TIMEOUT_MS;
use crate::config::{MAX_ALLOWLIST_ENTRY_LEN, MAX_ALLOWLIST_LEN, MAX_SERVER_ID_LEN};
use crate::error::McpError;

/// Maximum remote URL length in bytes (fail-closed bound; common URL ceiling).
pub const MAX_REMOTE_URL_LEN: usize = 2048;
/// Maximum custom headers per remote server.
pub const MAX_REMOTE_HEADERS: usize = 16;
/// Maximum header name length in bytes.
pub const MAX_REMOTE_HEADER_NAME_LEN: usize = 128;
/// Maximum header value length in bytes (covers Bearer JWT shapes).
pub const MAX_REMOTE_HEADER_VALUE_LEN: usize = 4096;
/// Maximum host length in bytes (DNS ceiling).
pub const MAX_REMOTE_HOST_LEN: usize = 253;

fn is_server_id(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_SERVER_ID_LEN {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_lowercase() {
        return false;
    }
    bytes
        .iter()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
}

fn is_allowlist_entry(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_ALLOWLIST_ENTRY_LEN {
        return false;
    }
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.' || byte == b'-')
}

fn is_header_name(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_REMOTE_HEADER_NAME_LEN {
        return false;
    }
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' || byte == b'.')
}

fn is_transport_owned_header(name: &str) -> bool {
    name.eq_ignore_ascii_case("accept")
        || name.eq_ignore_ascii_case("content-type")
        || name.eq_ignore_ascii_case("mcp-session-id")
}

fn is_header_value(value: &str) -> bool {
    if value.len() > MAX_REMOTE_HEADER_VALUE_LEN {
        return false;
    }
    // Forbid NUL/CR/LF plus other controls (header-injection shape);
    // printable ASCII plus space and tab stay legal.
    !value.bytes().any(|byte| {
        byte == b'\0'
            || byte == b'\r'
            || byte == b'\n'
            || (byte.is_ascii_control() && byte != b'\t')
    })
}

/// Authority slice of `url` (after `://`, before `/`, `?`, `#`, or `\`).
fn authority_of(url: &str) -> Option<&str> {
    let rest = url.split_once("://").map(|(_, rest)| rest)?;
    Some(rest.split(['/', '?', '#', '\\']).next().unwrap_or(""))
}

/// Whether `url` starts with `http://` or `https://` (ASCII case-insensitive).
fn has_http_scheme(url: &str) -> bool {
    (url.len() >= 7 && url[..7].eq_ignore_ascii_case("http://"))
        || (url.len() >= 8 && url[..8].eq_ignore_ascii_case("https://"))
}

/// Host slice of `url` (authority minus userinfo and port, brackets honored).
fn host_of(url: &str) -> &str {
    let authority = authority_of(url).unwrap_or("");
    let hostport = match authority.rsplit_once('@') {
        Some((_, host)) => host,
        None => authority,
    };
    if let Some(stripped) = hostport.strip_prefix('[') {
        match stripped.split_once(']') {
            Some((host, _)) => host,
            None => stripped,
        }
    } else {
        match hostport.split_once(':') {
            Some((host, _)) => host,
            None => hostport,
        }
    }
}

/// Remote (Streamable HTTP/SSE) MCP server configuration plus tool allowlist.
///
/// Fail-closed validation ([`RemoteServerConfig::validate`]): `http`/`https`
/// scheme only, no userinfo, parseable host, timeout `1..=MAX_TIMEOUT_MS`,
/// non-empty allowlist mirroring [`crate::config`], bounded custom headers
/// that never override transport-owned names, and a capability grant enforced
/// pre-contact per request. Every refusal names the violated bound only.
#[derive(Clone, PartialEq, Eq)]
pub struct RemoteServerConfig {
    /// Server id (`^[a-z][a-z0-9_]*$`, 1..=32 bytes): namespaces sanitized
    /// tool names (`mcp_<id>_<tool>`), same shape as stdio servers.
    pub id: String,
    /// Endpoint URL (`http://` or `https://`, no userinfo, 1..=2048 bytes).
    pub url: String,
    /// Whole-operation timeout in milliseconds (`1..=MAX_TIMEOUT_MS`).
    pub timeout_ms: u64,
    /// Raw MCP tool names this server may expose (non-empty, at most 32).
    pub tool_allowlist: Vec<String>,
    /// Extra HTTP headers (bounded; values are opaque secrets, never logged).
    pub headers: Vec<(String, String)>,
    /// Network capability allowlist enforced pre-contact per request
    /// (offline-first deny-all default; zero service calls on deny).
    pub capability: bitty_network_api::NetworkCapability,
}

impl RemoteServerConfig {
    /// Validate fail-closed. Every refusal names the violated bound only;
    /// no URLs, hosts, header names, values, or secrets enter the error.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] with [`McpStage::Config`] for any shape violation.
    pub fn validate(&self) -> Result<(), McpError> {
        if !is_server_id(&self.id) {
            return Err(McpError::invalid_config(
                "server id must match ^[a-z][a-z0-9_]*$ (1..=32 bytes)",
            ));
        }
        if self.url.is_empty() || self.url.len() > MAX_REMOTE_URL_LEN {
            return Err(McpError::invalid_config(
                "remote url is empty or over-bound",
            ));
        }
        if self.url.contains('\0') || self.url.bytes().any(|byte| byte.is_ascii_control()) {
            return Err(McpError::invalid_config(
                "remote url must not carry control bytes",
            ));
        }
        if self.url.contains(' ') {
            return Err(McpError::invalid_config(
                "remote url must not contain spaces",
            ));
        }
        if !has_http_scheme(&self.url) {
            return Err(McpError::invalid_config(
                "remote url must use http or https",
            ));
        }
        if let Some(authority) = authority_of(&self.url) {
            if authority.contains('@') {
                return Err(McpError::invalid_config(
                    "remote url must not carry userinfo",
                ));
            }
        } else {
            return Err(McpError::invalid_config("remote url carries no authority"));
        }
        let host = host_of(&self.url);
        if host.is_empty() || host.len() > MAX_REMOTE_HOST_LEN {
            return Err(McpError::invalid_config(
                "remote url host is empty or over-bound",
            ));
        }
        if host.bytes().any(|byte| byte.is_ascii_control()) {
            return Err(McpError::invalid_config(
                "remote url host carries control bytes",
            ));
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_TIMEOUT_MS {
            return Err(McpError::invalid_config(
                "timeout_ms must be within 1..=30000",
            ));
        }
        if self.tool_allowlist.is_empty() {
            return Err(McpError::invalid_config("tool allowlist must not be empty"));
        }
        if self.tool_allowlist.len() > MAX_ALLOWLIST_LEN {
            return Err(McpError::invalid_config(
                "tool allowlist exceeds 32 entries",
            ));
        }
        for entry in &self.tool_allowlist {
            if !is_allowlist_entry(entry) {
                return Err(McpError::invalid_config(
                    "allowlist entry must be 1..=64 bytes of [a-zA-Z0-9_.-]",
                ));
            }
        }
        if self.headers.len() > MAX_REMOTE_HEADERS {
            return Err(McpError::invalid_config("remote carries too many headers"));
        }
        for (name, value) in &self.headers {
            if !is_header_name(name) {
                return Err(McpError::invalid_config(
                    "remote header name is over-bound or malformed",
                ));
            }
            if is_transport_owned_header(name) {
                return Err(McpError::invalid_config(
                    "remote header overrides a transport header",
                ));
            }
            if !is_header_value(value) {
                return Err(McpError::invalid_config(
                    "remote header value is over-bound or malformed",
                ));
            }
        }
        Ok(())
    }

    /// Effective whole-operation timeout: the configured value, defaulting
    /// to [`crate::DEFAULT_MCP_TIMEOUT_MS`] when zero (callers should still
    /// prefer [`RemoteServerConfig::validate`], which refuses zero).
    #[must_use]
    pub fn effective_timeout_ms(&self) -> u64 {
        if self.timeout_ms == 0 {
            crate::DEFAULT_MCP_TIMEOUT_MS
        } else {
            self.timeout_ms.min(MAX_TIMEOUT_MS)
        }
    }
}

// Manual Debug keeps header values out of logs while still showing shapes.
impl std::fmt::Debug for RemoteServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Rebuild without a derived impl so values can never leak through a
        // future field addition without touching this site.
        let redacted_headers: Vec<(&str, &str)> = self
            .headers
            .iter()
            .map(|(name, _)| (name.as_str(), "[redacted]"))
            .collect();
        f.debug_struct("RemoteServerConfig")
            .field("id", &self.id)
            .field(
                "url",
                &crate::error::bound_error_text(&self.url, MAX_REMOTE_URL_LEN),
            )
            .field("timeout_ms", &self.timeout_ms)
            .field("tool_allowlist", &self.tool_allowlist)
            .field("header_count", &self.headers.len())
            .field("headers", &redacted_headers)
            .field("capability", &self.capability)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> RemoteServerConfig {
        RemoteServerConfig {
            id: "demo".to_owned(),
            url: "https://mcp.example.com/rpc".to_owned(),
            timeout_ms: 10_000,
            tool_allowlist: vec!["echo".to_owned()],
            headers: Vec::new(),
            capability: bitty_network_api::NetworkCapability::offline()
                .with_domain("mcp.example.com"),
        }
    }

    #[test]
    fn valid_remote_config_passes() {
        valid_config().validate().expect("valid remote config");
    }

    #[test]
    fn scheme_is_http_only() {
        let mut config = valid_config();
        for url in [
            String::from(""),
            String::from("ws://mcp.example.com/rpc"),
            String::from("file:///tmp/x"),
            String::from("mcp.example.com/rpc"),
            String::from("ftp://mcp.example.com/rpc"),
            String::from("gopher://mcp.example.com/rpc"),
        ] {
            config.url = url.clone();
            assert!(config.validate().is_err(), "scheme escaped: {url:?}");
        }
        for url in [
            "http://mcp.example.com/rpc",
            "https://mcp.example.com/rpc",
            "HTTP://mcp.example.com/rpc",
            "HTTPS://mcp.example.com/rpc",
        ] {
            config.url = url.to_owned();
            assert!(config.validate().is_ok(), "scheme refused: {url:?}");
        }
    }

    #[test]
    fn userinfo_in_url_is_refused() {
        let mut config = valid_config();
        for url in [
            "https://user@mcp.example.com/rpc",
            "https://user:pass@mcp.example.com/rpc",
            "http://user:pass@mcp.example.com:8080/rpc",
        ] {
            config.url = url.to_owned();
            assert!(config.validate().is_err(), "userinfo escaped: {url:?}");
        }
    }

    #[test]
    fn timeout_and_allowlist_mirror_stdio_bounds() {
        let mut config = valid_config();
        for timeout in [0, 30_001, u64::MAX] {
            config.timeout_ms = timeout;
            assert!(config.validate().is_err());
        }
        config.timeout_ms = 1;
        assert!(config.validate().is_ok());
        config.timeout_ms = 10_000;

        config.tool_allowlist.clear();
        assert!(config.validate().is_err());
        config.tool_allowlist = vec!["t".to_owned(); MAX_ALLOWLIST_LEN + 1];
        assert!(config.validate().is_err());
        config.tool_allowlist = vec!["bad name!".to_owned()];
        assert!(config.validate().is_err());
        config.tool_allowlist = vec!["echo".to_owned(), "list.files-2".to_owned()];
        assert!(config.validate().is_ok());
    }

    #[test]
    fn headers_are_bounded_and_transport_owned() {
        let mut config = valid_config();
        config.headers = vec![("X-Custom".to_owned(), "v".to_owned())];
        assert!(config.validate().is_ok());
        for owned in ["accept", "Content-Type", "MCP-SESSION-ID"] {
            config.headers = vec![(owned.to_owned(), "v".to_owned())];
            assert!(config.validate().is_err(), "owned header escaped: {owned}");
        }
        config.headers = vec![("X-Bad\nName".to_owned(), "v".to_owned())];
        assert!(config.validate().is_err());
        config.headers = vec![("X-Ok".to_owned(), "bad\nvalue".to_owned())];
        assert!(config.validate().is_err());
        // Over-bound header count fails without echoing any value.
        config.headers = (0..MAX_REMOTE_HEADERS + 1)
            .map(|index| (format!("X-H-{index}"), "v".to_owned()))
            .collect();
        assert!(config.validate().is_err());
    }

    #[test]
    fn debug_redacts_header_values() {
        let mut config = valid_config();
        config.headers = vec![("Authorization".to_owned(), "Bearer canary-9f3a".to_owned())];
        let rendered = format!("{config:?}");
        assert!(
            !rendered.contains("canary"),
            "header value leaked: {rendered}"
        );
        assert!(rendered.contains("[redacted]"));
    }
}

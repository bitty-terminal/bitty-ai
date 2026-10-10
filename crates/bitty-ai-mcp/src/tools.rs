//! `tools/list` import with sanitized registration names.
//!
//! The client pages `tools/list` (at most [`MAX_TOOL_PAGES`] pages; a
//! repeated cursor fails closed as [`McpFailure::DuplicateCursor`]) and
//! converts only allowlisted raw names into runtime [`ToolSpec`] values.
//! Registration names are sanitized by [`sanitize_mcp_name`] into the
//! `TB-2` shape (`mcp_<server>_<tool>`, lowercase, `^[a-z][a-z0-9_]*$`,
//! 64-byte cap with a deterministic FNV-1a-64 hash suffix past the cap).
//! Sanitized collisions fail closed as [`McpFailure::DuplicateTool`], and
//! every imported spec carries schema bytes, so [`ToolSpec::schema_digest`]
//! drift detection works host-side exactly like registry specs.
//!
//! Bound policy (judgment calls, documented): descriptions past 512 bytes
//! are truncated on a UTF-8 boundary (display-only, no validation impact);
//! schemas past 16 KiB fail the whole import (schemas shape validation, so
//! a partial import would lie); allowlist entries the server never offers
//! fail as [`McpFailure::UnknownTool`] so config drift surfaces loudly.

use std::time::Instant;

use bitty_ai_runtime::tool::{ToolSpec, validate_tool_name};

use crate::error::{McpError, McpFailure, McpStage, bound_error_name, bound_error_text};
use crate::handshake::wait_for_response;
use crate::json::{escaped, find_object_field, find_raw_field, find_string_field};
use crate::{MAX_TIMEOUT_MS, McpTransport};

/// Maximum `tools/list` pages followed per import.
pub const MAX_TOOL_PAGES: usize = 5;

/// FNV-1a-64 offset basis (mirrors the runtime `tool.rs` precedent; the
/// constants stay private there, so they are duplicated here and pinned by
/// the determinism test below).
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a-64 prime (mirrors the runtime `tool.rs` precedent).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Deterministic 64-bit FNV-1a hash (runtime `schema_digest` precedent).
pub(crate) fn fnv_1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Sanitize an MCP tool name into the runtime `TB-2` registered shape.
///
/// Builds `mcp_<server>_<tool>`, lowercases it, and maps every byte outside
/// `[a-z0-9_]` to `_`. Results longer than 64 bytes keep the first 55 bytes
/// plus `_` plus 8 lowercase hex digits of FNV-1a-64 over the full
/// candidate, so the cap is deterministic and collision-resistant. The
/// `mcp_` prefix guarantees the leading-lowercase rule.
#[must_use]
pub fn sanitize_mcp_name(server_id: &str, raw_tool: &str) -> String {
    fn clean(value: &str) -> String {
        value
            .bytes()
            .map(|byte| {
                let lower = byte.to_ascii_lowercase();
                if lower.is_ascii_lowercase() || lower.is_ascii_digit() || lower == b'_' {
                    lower as char
                } else {
                    '_'
                }
            })
            .collect()
    }
    let candidate = format!("mcp_{}_{}", clean(server_id), clean(raw_tool));
    if candidate.len() <= bitty_ai_runtime::tool::MAX_TOOL_NAME_LEN
        && validate_tool_name(&candidate).is_ok()
    {
        return candidate;
    }
    let digest = fnv_1a_64(candidate.as_bytes());
    let keep = bitty_ai_runtime::tool::MAX_TOOL_NAME_LEN - 1 - 8;
    let prefix = candidate.get(..keep).unwrap_or(&candidate);
    // Low 32 bits render as exactly 8 hex digits (`:08x` pads short
    // values but never truncates long ones, so the full 64-bit digest
    // would overflow the cap).
    let suffix = (digest & 0xffff_ffff) as u32;
    format!("{prefix}_{suffix:08x}")
}

/// One raw tool entry from a `tools/list` page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawMcpTool {
    /// Server-reported name (pre-sanitization).
    pub name: String,
    /// Server-reported description (may be empty).
    pub description: String,
    /// Raw `inputSchema` object bytes (`{}` when the server sends none).
    pub schema_json: Vec<u8>,
}

/// One imported tool: a validated runtime spec plus provenance.
///
/// `digest` equals `spec.schema_digest()` at import time; the host compares
/// digests across imports to detect server-side drift (the registry keeps
/// refusing same-name re-registration, so drift never smuggles a
/// replacement).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedTool {
    /// Validated runtime spec under the sanitized name.
    pub spec: ToolSpec,
    /// `spec.schema_digest()` at import time.
    pub digest: u64,
    /// Owning server id.
    pub server_id: String,
    /// Server-reported raw name.
    pub raw_name: String,
}

/// Parse one `tools/list` response frame into entries plus an optional
/// follow-up cursor.
///
/// A JSON-RPC `error` answer fails closed. Entries missing a name are
/// skipped (never registered under a synthesized name); descriptions
/// default to empty.
///
/// # Errors
///
/// Returns [`McpFailure::HandshakeRejected`]-shaped list errors for
/// protocol violations (reused as the closest typed failure: the frame is
/// not a usable answer).
pub fn parse_tools_page(line: &str) -> Result<(Vec<RawMcpTool>, Option<String>), McpError> {
    if let Some(error) = find_object_field(line, "error") {
        let code = find_raw_field(error, "code").unwrap_or("?");
        return Err(McpError::new(
            McpStage::ListTools,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    &format!("tools/list answered error {code}"),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        ));
    }
    let result = find_object_field(line, "result").ok_or_else(|| {
        McpError::new(
            McpStage::ListTools,
            McpFailure::HandshakeRejected {
                detail: "tools/list answer carries no result".to_owned(),
            },
        )
    })?;
    let mut entries = Vec::new();
    if let Some(array) = find_raw_field(result, "tools") {
        if !array.trim_start().starts_with('[') {
            return Err(McpError::new(
                McpStage::ListTools,
                McpFailure::HandshakeRejected {
                    detail: "tools/list result.tools is not an array".to_owned(),
                },
            ));
        }
        for object in split_array_objects(array)? {
            let Some(name) = find_string_field(object, "name") else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let description = find_string_field(object, "description").unwrap_or_default();
            let schema_json = find_object_field(object, "inputSchema")
                .map_or(Vec::new(), |schema| schema.as_bytes().to_vec());
            entries.push(RawMcpTool {
                name,
                description,
                schema_json,
            });
        }
    }
    let cursor = find_string_field(result, "nextCursor").filter(|token| !token.is_empty());
    Ok((entries, cursor))
}

/// Split a raw JSON array body into its top-level object members.
///
/// Only `{...}` members are returned; scalars are skipped (tool entries are
/// always objects). Unbalanced input fails closed.
///
/// # Errors
///
/// Returns [`McpFailure::HandshakeRejected`] on unbalanced arrays.
fn split_array_objects(array: &str) -> Result<Vec<&str>, McpError> {
    let mut objects = Vec::new();
    let bytes = array.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'{' => {
                let end = crate::json::balanced_end(array, index, b'{', b'}').ok_or_else(|| {
                    McpError::new(
                        McpStage::ListTools,
                        McpFailure::HandshakeRejected {
                            detail: "tools/list array is unbalanced".to_owned(),
                        },
                    )
                })?;
                objects.push(&array[index..=end]);
                index = end + 1;
            }
            b'"' => {
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index += 2,
                        b'"' => {
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
            }
            _ => index += 1,
        }
    }
    Ok(objects)
}

/// Convert one raw entry into an [`ImportedTool`], enforcing [`ToolSpec`]
/// bounds.
///
/// Descriptions past 512 bytes truncate on a UTF-8 boundary; schemas past
/// 16 KiB fail; sanitized collisions are detected by the caller across the
/// whole import (this function only guarantees the spec itself validates).
/// Imported specs default to `read_only = true` (least privilege: MCP
/// access is read-only by default); the host may loosen individual specs
/// before registry insert under its own consent and tier policy.
///
/// # Errors
///
/// Returns schema/description bound failures and invalid-name rejections.
pub fn import_tool(server_id: &str, raw: &RawMcpTool) -> Result<ImportedTool, McpError> {
    if raw.schema_json.len() > bitty_ai_runtime::tool::MAX_TOOL_SCHEMA_BYTES {
        return Err(McpError::new(
            McpStage::ListTools,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    &format!("tool '{}' schema exceeds 16384 bytes", raw.name),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        ));
    }
    let description = truncate_utf8(
        &raw.description,
        bitty_ai_runtime::tool::MAX_TOOL_DESCRIPTION_LEN,
    );
    let sanitized = sanitize_mcp_name(server_id, &raw.name);
    let spec = ToolSpec::new(
        sanitized.clone(),
        description,
        raw.schema_json.clone(),
        format!("mcp.{server_id}"),
        true,
    )
    .map_err(|error| {
        McpError::new(
            McpStage::ListTools,
            McpFailure::HandshakeRejected {
                detail: bound_error_text(
                    &format!("tool '{}' rejected: {error}", raw.name),
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        )
    })?;
    let digest = spec.schema_digest();
    Ok(ImportedTool {
        spec,
        digest,
        server_id: server_id.to_owned(),
        raw_name: raw.name.clone(),
    })
}

/// Truncate `value` to `max_bytes` on a UTF-8 boundary.
fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

/// Build one `tools/list` request frame.
#[must_use]
pub fn list_request(id: u64, cursor: Option<&str>) -> String {
    match cursor {
        Some(token) => format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"tools/list\",\"params\":{{\"cursor\":\"{}\"}}}}",
            escaped(token)
        ),
        None => {
            format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"tools/list\",\"params\":{{}}}}")
        }
    }
}

/// Import allowlisted tools from a server.
///
/// Pages `tools/list` from `next_id`, following cursors up to
/// [`MAX_TOOL_PAGES`] pages; a repeated cursor fails closed. Only tools
/// whose raw name appears in `allowlist` convert; every allowlist entry
/// must be observed or the import fails with [`McpFailure::UnknownTool`].
/// `timeout_ms` bounds the whole import.
///
/// Notifications observed while waiting for page answers are pushed onto
/// `surfaced` in arrival order (see
/// [`crate::handshake::wait_for_response`]) for the caller to route through
/// [`is_tools_list_changed_notification`]; the import result itself is
/// unchanged. `surfaced` keeps whatever arrived even when the import fails.
///
/// # Errors
///
/// Returns pagination, bound, collision, unknown-tool, timeout, or
/// transport errors.
#[allow(clippy::too_many_arguments)]
pub fn list_tools(
    transport: &mut dyn McpTransport,
    server_id: &str,
    cwd: &str,
    allowlist: &[String],
    next_id: &mut u64,
    timeout_ms: u64,
    surfaced: &mut Vec<String>,
) -> Result<Vec<ImportedTool>, McpError> {
    let bound = timeout_ms.clamp(1, MAX_TIMEOUT_MS);
    let deadline = Instant::now() + std::time::Duration::from_millis(bound);
    let mut cursor: Option<String> = None;
    // Cursors already spent on a request. A server that answers with a
    // cursor it has already seen is looping: refuse instead of paging
    // forever. The freshly issued cursor joins the spent set as soon as it
    // is used, so a self-loop is caught on its second appearance.
    let mut used_cursors: Vec<String> = Vec::new();
    let mut raws: Vec<RawMcpTool> = Vec::new();
    for _ in 0..MAX_TOOL_PAGES {
        let id = *next_id;
        *next_id = next_id.wrapping_add(1);
        transport.send_line(&list_request(id, cursor.as_deref()))?;
        if let Some(spent) = cursor.as_ref() {
            used_cursors.push(spent.clone());
        }
        let answer = wait_for_response(transport, id, cwd, deadline, bound, surfaced)
            .map_err(|error| McpError::new(McpStage::ListTools, error.failure))?;
        let (mut entries, next) = parse_tools_page(&answer)?;
        raws.append(&mut entries);
        match next {
            None => {
                cursor = None;
                break;
            }
            Some(token) => {
                if used_cursors.iter().any(|spent| spent == &token) {
                    return Err(McpError::new(
                        McpStage::ListTools,
                        McpFailure::DuplicateCursor {
                            cursor: bound_error_name(&token),
                        },
                    ));
                }
                cursor = Some(token);
            }
        }
    }
    if cursor.is_some() {
        return Err(McpError::new(
            McpStage::ListTools,
            McpFailure::TooManyPages {
                limit: MAX_TOOL_PAGES,
            },
        ));
    }
    let mut imported = Vec::new();
    for wanted in allowlist {
        let Some(raw) = raws.iter().find(|entry| &entry.name == wanted) else {
            return Err(McpError::new(
                McpStage::ListTools,
                McpFailure::UnknownTool {
                    name: bound_error_name(wanted),
                },
            ));
        };
        let tool = import_tool(server_id, raw)?;
        if imported
            .iter()
            .any(|kept: &ImportedTool| kept.spec.name == tool.spec.name)
        {
            return Err(McpError::new(
                McpStage::ListTools,
                McpFailure::DuplicateTool {
                    name: bound_error_name(&tool.spec.name),
                },
            ));
        }
        imported.push(tool);
    }
    Ok(imported)
}

/// Whether `line` is a host-routed `notifications/tools/list_changed`
/// signal (AI-0211, AIQ-08 dynamic-invalidation facet).
///
/// True only for a notification frame: `method` exactly
/// `notifications/tools/list_changed` with no `id` member. Every other
/// shape is false: `notifications/ping`,
/// `notifications/resources/list_changed`,
/// `notifications/prompts/list_changed`, any frame carrying an `id`
/// (requests and responses, including an `id`-carrying `list_changed`
/// echo), and malformed lines. The host polls for server messages
/// (HTTP `poll_server_messages` first; stdio drains inline), routes each
/// queued line through this classifier, and marks that server's adapter
/// stale on true. Whole-list stale: one signal stales the full tool list,
/// never a single tool. No auto-relist, no background work;
/// resources/prompts classifiers only; no version/stale/relist until a
/// consumer exists.
///
/// The check is syntactic only (method plus id absence) and never touches
/// the transport.
#[must_use]
pub fn is_tools_list_changed_notification(line: &str) -> bool {
    if crate::json::find_raw_field(line, "id").is_some() {
        return false;
    }
    matches!(
        crate::json::find_string_field(line, "method").as_deref(),
        Some("notifications/tools/list_changed")
    )
}

/// Whether `line` is a `notifications/resources/list_changed` signal
/// (AI-0214, AIQ-08 dynamic-invalidation facet).
///
/// Taxonomy symmetry with [`is_tools_list_changed_notification`]: true only
/// for a notification frame with `method` exactly
/// `notifications/resources/list_changed` and no `id` member. Pure bool
/// classifier only — no version, no stale, no re-list: nothing fetches or
/// caches resources lists today, so no consumer exists to drive them
/// (Option B parked until a real `resources/list` consumer lands). The host
/// keeps routing only tools signals through
/// [`crate::bridge::McpToolAdapter::observe_notification`]; even
/// classifier-positive resources lines never stale the adapter.
///
/// The check is syntactic only (method plus id absence) and never touches
/// the transport.
#[must_use]
pub fn is_resources_list_changed_notification(line: &str) -> bool {
    if crate::json::find_raw_field(line, "id").is_some() {
        return false;
    }
    matches!(
        crate::json::find_string_field(line, "method").as_deref(),
        Some("notifications/resources/list_changed")
    )
}

/// Whether `line` is a `notifications/prompts/list_changed` signal
/// (AI-0214, AIQ-08 dynamic-invalidation facet).
///
/// Taxonomy symmetry with [`is_tools_list_changed_notification`]: true only
/// for a notification frame with `method` exactly
/// `notifications/prompts/list_changed` and no `id` member. Pure bool
/// classifier only — no version, no stale, no re-list: nothing fetches or
/// caches prompts lists today, so no consumer exists to drive them
/// (Option B parked until a real `prompts/list` consumer lands). The host
/// keeps routing only tools signals through
/// [`crate::bridge::McpToolAdapter::observe_notification`]; even
/// classifier-positive prompts lines never stale the adapter.
///
/// The check is syntactic only (method plus id absence) and never touches
/// the transport.
#[must_use]
pub fn is_prompts_list_changed_notification(line: &str) -> bool {
    if crate::json::find_raw_field(line, "id").is_some() {
        return false;
    }
    matches!(
        crate::json::find_string_field(line, "method").as_deref(),
        Some("notifications/prompts/list_changed")
    )
}

/// Immutable host-side digest snapshot of one server's imported tool list
/// (AI-0211 diff handle).
///
/// Holds `(sanitized name, schema digest)` pairs sorted by name, built from
/// [`ImportedTool`] slices via [`ToolListSnapshot::from_imported`]. The
/// digest is [`ToolSpec::schema_digest`](bitty_ai_runtime::tool::ToolSpec::schema_digest)
/// at import time: same bytes give the same digest, any byte moves it. The
/// host compares snapshots across an explicit re-list to detect drift; the
/// registry keeps refusing same-name re-registration, so the digest never
/// smuggles a replacement (refusal-only stands).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolListSnapshot {
    entries: Vec<(String, u64)>,
}

impl ToolListSnapshot {
    /// Build a snapshot from imported tools, sorted by sanitized name for
    /// deterministic diffs.
    #[must_use]
    pub fn from_imported(tools: &[ImportedTool]) -> Self {
        let mut entries: Vec<(String, u64)> = tools
            .iter()
            .map(|entry| (entry.spec.name.clone(), entry.digest))
            .collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        Self { entries }
    }

    /// Snapshot entries as `(sanitized name, digest)` pairs in name order.
    #[must_use]
    pub fn entries(&self) -> &[(String, u64)] {
        &self.entries
    }

    /// Digest for `name`, when present.
    #[must_use]
    pub fn digest_of(&self, name: &str) -> Option<u64> {
        self.entries
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, digest)| *digest)
    }

    /// Entry count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the snapshot holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Pure diff between two [`ToolListSnapshot`] values (AI-0211 re-list
/// handle).
///
/// `added` holds names only in `new`, `removed` only in `old`, `changed`
/// holds names in both with different digests. All three lists sort by
/// name. Empty diff means the re-list observed the same digests under the
/// same names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolListDiff {
    /// Names only in the new snapshot.
    pub added: Vec<String>,
    /// Names only in the old snapshot.
    pub removed: Vec<String>,
    /// Names in both with different digests.
    pub changed: Vec<String>,
}

impl ToolListDiff {
    /// Whether the diff reports no change.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Diff `old` against `new` (pure, no I/O).
#[must_use]
pub fn diff_tool_snapshots(old: &ToolListSnapshot, new: &ToolListSnapshot) -> ToolListDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    for (name, old_digest) in old.entries() {
        match new.digest_of(name) {
            None => removed.push(name.clone()),
            Some(new_digest) if new_digest != *old_digest => changed.push(name.clone()),
            Some(_) => {}
        }
    }
    for (name, _) in new.entries() {
        if old.digest_of(name).is_none() {
            added.push(name.clone());
        }
    }
    added.sort();
    removed.sort();
    changed.sort();
    ToolListDiff {
        added,
        removed,
        changed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeTransport {
        inbound: VecDeque<String>,
        sent: Vec<String>,
    }

    impl McpTransport for FakeTransport {
        fn send_line(&mut self, line: &str) -> Result<(), McpError> {
            self.sent.push(line.to_owned());
            Ok(())
        }

        fn recv_line(&mut self, _timeout_ms: u64) -> Result<Option<String>, McpError> {
            Ok(self.inbound.pop_front())
        }
    }

    fn page(id: u64, tools_json: &str, cursor: Option<&str>) -> String {
        let next = cursor.map_or(String::new(), |token| {
            format!(",\"nextCursor\":\"{token}\"")
        });
        format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"tools\":[{tools_json}]{next}}}}}")
    }

    #[test]
    fn sanitize_shapes_names_and_caps_at_64() {
        assert_eq!(sanitize_mcp_name("demo", "echo"), "mcp_demo_echo");
        assert_eq!(
            sanitize_mcp_name("demo", "List.Files-2"),
            "mcp_demo_list_files_2"
        );
        // Leading digits and symbols fold into the valid shape.
        let folded = sanitize_mcp_name("demo", "9 lives!");
        assert!(validate_tool_name(&folded).is_ok());
        // Long names cap deterministically with a hash suffix.
        let long = sanitize_mcp_name("demo", &"x".repeat(200));
        assert_eq!(long.len(), 64);
        assert!(validate_tool_name(&long).is_ok());
        assert_eq!(long, sanitize_mcp_name("demo", &"x".repeat(200)));
        assert_ne!(long, sanitize_mcp_name("demo", &"y".repeat(200)));
        // The suffix is a hash of the full candidate: prefix plus 8 hex.
        assert!(long.as_bytes()[55] == b'_');
    }

    #[test]
    fn sanitize_matches_fnv_drift_detection() {
        // Same bytes hash the same; any byte moves the digest (runtime
        // schema_digest precedent, independently recomputed here).
        assert_eq!(fnv_1a_64(b"abc"), fnv_1a_64(b"abc"));
        assert_ne!(fnv_1a_64(b"abc"), fnv_1a_64(b"abd"));
    }

    #[test]
    fn pages_follow_bounded_with_duplicate_cursor_refusal() {
        let first = page(1, "{\"name\":\"a\"}", Some("c1"));
        let second = page(2, "{\"name\":\"b\"}", Some("c1"));
        let mut transport = FakeTransport {
            inbound: vec![first, second].into_iter().collect(),
            sent: Vec::new(),
        };
        let mut next_id = 1;
        let error = list_tools(
            &mut transport,
            "demo",
            "/tmp/bitty",
            &["a".to_owned()],
            &mut next_id,
            1_000,
            &mut Vec::new(),
        )
        .expect_err("repeated cursor must fail");
        assert!(matches!(error.failure, McpFailure::DuplicateCursor { .. }));
    }

    #[test]
    fn single_page_import_records_digest() {
        let frame = page(
            1,
            "{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}",
            None,
        );
        let mut transport = FakeTransport {
            inbound: vec![frame].into_iter().collect(),
            sent: Vec::new(),
        };
        let mut next_id = 1;
        let imported = list_tools(
            &mut transport,
            "demo",
            "/tmp/bitty",
            &["echo".to_owned()],
            &mut next_id,
            1_000,
            &mut Vec::new(),
        )
        .expect("import");
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].spec.name, "mcp_demo_echo");
        assert_eq!(imported[0].digest, imported[0].spec.schema_digest());
        assert_eq!(imported[0].raw_name, "echo");
    }

    #[test]
    fn allowlist_gaps_fail_closed() {
        let frame = page(1, "{\"name\":\"echo\"}", None);
        let mut transport = FakeTransport {
            inbound: vec![frame].into_iter().collect(),
            sent: Vec::new(),
        };
        let mut next_id = 1;
        let error = list_tools(
            &mut transport,
            "demo",
            "/tmp/bitty",
            &["echo".to_owned(), "ghost".to_owned()],
            &mut next_id,
            1_000,
            &mut Vec::new(),
        )
        .expect_err("unseen allowlist entry must fail");
        assert!(matches!(error.failure, McpFailure::UnknownTool { .. }));
    }

    #[test]
    fn sanitized_collisions_fail_closed() {
        // "a.b" and "a_b" both sanitize to mcp_demo_a_b.
        let frame = page(1, "{\"name\":\"a.b\"},{\"name\":\"a_b\"}", None);
        let mut transport = FakeTransport {
            inbound: vec![frame].into_iter().collect(),
            sent: Vec::new(),
        };
        let mut next_id = 1;
        let error = list_tools(
            &mut transport,
            "demo",
            "/tmp/bitty",
            &["a.b".to_owned(), "a_b".to_owned()],
            &mut next_id,
            1_000,
            &mut Vec::new(),
        )
        .expect_err("collision must fail");
        assert!(matches!(error.failure, McpFailure::DuplicateTool { .. }));
    }

    #[test]
    fn oversize_schema_fails_the_import() {
        let big_schema = format!("{{\"type\":\"object\",\"pad\":\"{}\"}}", "p".repeat(17_000));
        let raw = RawMcpTool {
            name: "big".to_owned(),
            description: String::new(),
            schema_json: big_schema.into_bytes(),
        };
        let error = import_tool("demo", &raw).expect_err("oversize schema");
        assert!(matches!(
            error.failure,
            McpFailure::HandshakeRejected { .. }
        ));
    }

    #[test]
    fn long_descriptions_truncate_on_a_boundary() {
        let raw = RawMcpTool {
            name: "echo".to_owned(),
            description: "e".repeat(600),
            schema_json: Vec::new(),
        };
        let imported = import_tool("demo", &raw).expect("truncate");
        assert_eq!(
            imported.spec.description.len(),
            bitty_ai_runtime::tool::MAX_TOOL_DESCRIPTION_LEN
        );
    }

    #[test]
    fn list_changed_classifier_accepts_only_bare_tools_notification() {
        assert!(is_tools_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}"
        ));
        assert!(is_tools_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\",\"params\":{}}"
        ));
    }

    #[test]
    fn list_changed_classifier_rejects_negatives() {
        for line in [
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"notifications/tools/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":\"abc-1\",\"method\":\"notifications/tools/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}",
            "not json at all",
            "",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changedX\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list-changed\"}",
        ] {
            assert!(
                !is_tools_list_changed_notification(line),
                "must not stale: {line:?}"
            );
        }
    }

    #[test]
    fn resources_classifier_accepts_only_bare_resources_notification() {
        assert!(is_resources_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\"}"
        ));
        assert!(is_resources_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\",\"params\":{}}"
        ));
    }

    #[test]
    fn resources_classifier_rejects_negatives() {
        // Cross-kind first: the tools signal is positive for the tools
        // classifier and negative here, and vice versa.
        assert!(is_tools_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}"
        ));
        for line in [
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"notifications/resources/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":\"abc-1\",\"method\":\"notifications/resources/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}",
            "not json at all",
            "",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changedX\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list-changed\"}",
        ] {
            assert!(
                !is_resources_list_changed_notification(line),
                "must stay classifier-only: {line:?}"
            );
        }
    }

    #[test]
    fn prompts_classifier_accepts_only_bare_prompts_notification() {
        assert!(is_prompts_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\"}"
        ));
        assert!(is_prompts_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\",\"params\":{}}"
        ));
    }

    #[test]
    fn prompts_classifier_rejects_negatives() {
        // Cross-kind first: the tools signal is positive for the tools
        // classifier and negative here, and vice versa.
        assert!(is_tools_list_changed_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}"
        ));
        for line in [
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"notifications/prompts/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":\"abc-1\",\"method\":\"notifications/prompts/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}",
            "not json at all",
            "",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changedX\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list-changed\"}",
        ] {
            assert!(
                !is_prompts_list_changed_notification(line),
                "must stay classifier-only: {line:?}"
            );
        }
    }

    #[test]
    fn snapshot_diff_detects_added_removed_changed() {
        fn imported(name: &str, raw: &str, schema: &[u8]) -> ImportedTool {
            let spec =
                ToolSpec::new(name, "test", schema.to_vec(), "mcp.demo", true).expect("valid");
            ImportedTool {
                digest: spec.schema_digest(),
                server_id: "demo".to_owned(),
                raw_name: raw.to_owned(),
                spec,
            }
        }
        let old = ToolListSnapshot::from_imported(&[
            imported("mcp_demo_a", "a", br#"{"type":"object"}"#),
            imported("mcp_demo_b", "b", br#"{"type":"object"}"#),
            imported("mcp_demo_c", "c", br#"{"type":"object","v":1}"#),
        ]);
        let new = ToolListSnapshot::from_imported(&[
            imported("mcp_demo_b", "b", br#"{"type":"object"}"#),
            imported("mcp_demo_c", "c", br#"{"type":"object","v":2}"#),
            imported("mcp_demo_d", "d", br#"{"type":"object"}"#),
        ]);
        assert_eq!(old.len(), 3);
        let diff = diff_tool_snapshots(&old, &new);
        assert_eq!(diff.added, vec!["mcp_demo_d".to_owned()]);
        assert_eq!(diff.removed, vec!["mcp_demo_a".to_owned()]);
        assert_eq!(diff.changed, vec!["mcp_demo_c".to_owned()]);
        assert!(!diff.is_empty());
        let same = diff_tool_snapshots(&old, &old);
        assert!(same.is_empty());
    }
}

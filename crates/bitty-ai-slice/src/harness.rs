//! Integration-harness fixtures: the real `bitty-ai-runtime` driven through
//! the real `bitty-ipc` bridge.
//!
//! This module owns no AI mechanism. It adapts the generic host boundary
//! ([`IpcBridge`]) to runtime inputs: bounded terminal snapshots become
//! runtime [`ContextRecord`]s, and the scripted [`FakeProvider`] plus the
//! read-only tool registry give tests a deterministic single-agent loop.
//!
//! Vocabulary notes (all renames forced by the runtime's validated shapes,
//! not by preference):
//!
//! - Tool `terminal_read_zone`: the runtime `TB-2` name shape
//!   (`^[a-z][a-z0-9_]*$`) rejects the old slice `terminal.read_zone`.
//!   Mapping this to the generic `bitty-agent` vocabulary is later P1 work.
//! - Provider id `local-deterministic`: the runtime `MP-2` id shape rejects
//!   the old slice `local.deterministic`.
//! - Record owner `term-1`: a runtime [`StableId`] is one hierarchy level
//!   (`^[a-z0-9_-]+$`), so the old slice `inst-1/term-1` path is kept in the
//!   request params but not in the owner field.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentSession, AuthContext, AuthDecision, ContextError, ContextPriority,
    ContextRecord, FakeProvider, IdIssuer, ProviderError, ProviderTurn, ProviderUsage, RecordBody,
    StableId, ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolRegistry, ToolSpec,
};

use crate::bridge::{HostPeer, IpcBridge};
use crate::error::SliceError;

/// Read-only terminal-zone tool served by the harness executor.
pub const HARNESS_TOOL: &str = "terminal_read_zone";

/// Scripted model name served by the runtime [`FakeProvider`].
pub const HARNESS_MODEL: &str = "fake-chat";

/// Harness provider identity (runtime `MP-2` shape: no dots).
pub const HARNESS_PROVIDER_ID: &str = "local-deterministic";

/// Bounded terminal-snapshot request: wire params for the generic
/// `terminal.snapshot` method plus the runtime [`StableId`] owner head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotRequest {
    /// Instance the terminal belongs to (wire params only).
    pub instance: String,
    /// Terminal to read (wire params only).
    pub terminal: String,
    /// Semantic zone name, e.g. `"output"` (wire params only).
    pub zone: String,
    /// Runtime [`StableId`] owner head, e.g. `"term-1"`.
    pub owner: String,
    /// Caller's byte ceiling for the snapshot.
    pub max_bytes: usize,
}

/// Escape `value` as JSON string content (without surrounding quotes).
///
/// Mirrors the host-side bounded encoder (`bitty-ipc` `json_escape_into`)
/// and the slice `local_provider::json_escape`: `"` / `\` plus C0 controls
/// (short forms for `\n` / `\r` / `\t` / `\b` / `\f`, remainder as
/// `\u00xx`) and `DEL` (`0x7F`) as `\u007f`. All other chars (including
/// non-ASCII UTF-8) pass through unchanged. Std-only, zero new deps.
///
/// Without this, `format!`-interpolated wire params let a hostile zone name
/// break out of its string and inject extra fields (P1-3 SEC).
fn escape_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) == 0x7F => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

impl SnapshotRequest {
    /// Build a request for one bounded zone read.
    pub fn new(
        instance: impl Into<String>,
        terminal: impl Into<String>,
        zone: impl Into<String>,
        owner: impl Into<String>,
        max_bytes: usize,
    ) -> Self {
        Self {
            instance: instance.into(),
            terminal: terminal.into(),
            zone: zone.into(),
            owner: owner.into(),
            max_bytes,
        }
    }

    /// Serialize the bounded request params for `terminal.snapshot`.
    ///
    /// `instance` / `terminal` / `zone` are JSON-escaped (quote, backslash,
    /// C0 controls, `DEL`); the output never contains a raw C0 byte, so a
    /// zone name cannot break the wire-params structure. Length is bounded
    /// by the caller inputs plus the fixed envelope; the bridge enforces
    /// the wire-envelope ceiling in [`crate::bridge::IpcBridge::call`].
    #[must_use]
    pub fn params_json(&self) -> Vec<u8> {
        let mut out =
            String::with_capacity(self.instance.len() + self.terminal.len() + self.zone.len() + 64);
        out.push_str("{\"instance\":\"");
        out.push_str(&escape_json_string(&self.instance));
        out.push_str("\",\"terminal\":\"");
        out.push_str(&escape_json_string(&self.terminal));
        out.push_str("\",\"zone\":\"");
        out.push_str(&escape_json_string(&self.zone));
        out.push_str("\",\"max_bytes\":");
        out.push_str(&self.max_bytes.to_string());
        out.push('}');
        out.into_bytes()
    }
}

/// Build a runtime terminal [`ContextRecord`] from snapshot bytes.
///
/// The record is always an untrusted observation surface: terminal output is
/// attacker-controlled data, never instructions.
///
/// The turn-scoped id names the source terminal and its generation
/// (`terminal-{owner}-{generation}`): distinct terminals sampled in one
/// generation never share an id, which assembly would reject as
/// [`ContextError::DuplicateRecordId`]. Re-sampling one terminal within the
/// same generation reuses its id, so callers advance the generation per
/// sample instead of assembling two same-id records.
pub fn terminal_record(
    owner: &str,
    generation: u64,
    now_ms: u64,
    bytes: Vec<u8>,
) -> Result<ContextRecord, ContextError> {
    Ok(ContextRecord {
        id: format!("terminal-{owner}-{generation}"),
        provider: "terminal".to_owned(),
        owner: StableId::new(owner)?,
        generation,
        collected_at_ms: now_ms,
        priority: ContextPriority::Normal,
        summary: "bounded terminal snapshot".to_owned(),
        body: RecordBody::Inline(bytes),
        supersedes: None,
        is_untrusted_surface: true,
    })
}

/// Collect one bounded terminal snapshot through the real [`IpcBridge`] and
/// adapt it to a runtime [`ContextRecord`].
///
/// # Errors
///
/// - [`SliceError`] from the bridge (unknown method, missing scope/consent,
///   host refusal) or when the snapshot exceeds `request.max_bytes`.
/// - [`SliceError::ContextUnavailable`] when the bytes cannot form a valid
///   runtime record (e.g. a malformed owner [`StableId`]).
pub fn collect_terminal_context(
    bridge: &mut IpcBridge,
    peer: &mut dyn HostPeer,
    request: &SnapshotRequest,
    generation: u64,
    now_ms: u64,
) -> Result<ContextRecord, SliceError> {
    let bytes = bridge.call("terminal.snapshot", &request.params_json(), now_ms, peer)?;
    if bytes.len() > request.max_bytes {
        return Err(SliceError::ContextBudgetExceeded {
            limit: request.max_bytes,
            actual: bytes.len(),
        });
    }
    terminal_record(&request.owner, generation, now_ms, bytes).map_err(|error| {
        SliceError::ContextUnavailable {
            reason: error.to_string(),
        }
    })
}

/// The harness read-only registry: exactly one tool, [`HARNESS_TOOL`].
///
/// # Errors
///
/// Returns [`ToolError`] when the registry or spec bounds reject the
/// declaration (unreachable for these constants; kept fallible so the
/// harness never panics on a bound).
pub fn test_tool_registry() -> Result<ToolRegistry, ToolError> {
    let mut registry = ToolRegistry::new();
    registry.register(ToolSpec::new(
        HARNESS_TOOL,
        "Read a bounded terminal semantic zone (read-only)",
        br#"{"type":"object"}"#.to_vec(),
        "terminal.inspect",
        true,
    )?)?;
    Ok(registry)
}

/// Authorizer that allows read-only tools and denies mutating ones.
///
/// This is a test-harness stand-in for the host capability/consent check,
/// not an accepted security mechanism: the real grant lives behind the
/// runtime [`ToolAuthorizer`] seam on the host side.
#[derive(Debug, Default)]
pub struct AllowReadOnly;

impl ToolAuthorizer for AllowReadOnly {
    fn authorize(&self, ctx: &AuthContext) -> AuthDecision {
        if ctx.read_only {
            AuthDecision::Allow
        } else {
            AuthDecision::Deny {
                reason: format!("harness denies mutating tool {}", ctx.tool),
            }
        }
    }
}

/// Build a scripted [`FakeProvider`] replaying one deterministic turn.
///
/// # Errors
///
/// Returns [`ProviderError::InvalidProviderId`] when the harness provider id
/// violates the `MP-2` shape (unreachable for the constant id).
pub fn scripted_provider(
    answer: &str,
    tool_arguments: Option<Vec<u8>>,
) -> Result<FakeProvider, ProviderError> {
    let mut provider = FakeProvider::new(HARNESS_PROVIDER_ID)?;
    let tool_calls = match tool_arguments {
        Some(arguments) => vec![ToolCallRequest {
            name: HARNESS_TOOL.to_owned(),
            arguments,
        }],
        None => Vec::new(),
    };
    provider.push_turn(ProviderTurn {
        text: answer.to_owned(),
        tool_calls,
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    Ok(provider)
}

/// Fresh deterministic session (inspect tier, `Active`, generation 1).
#[must_use]
pub fn test_session() -> AgentSession {
    let mut ids = IdIssuer::default();
    AgentSession::new(ids.agent_instance(), ids.run(), ids.session())
}

/// Wire a deterministic single-agent runtime: `provider` plus the harness
/// read-only tool bus at `context_budget_bytes`.
///
/// # Errors
///
/// Returns [`ToolError`] when the harness registry violates a tool bound
/// (unreachable for these constants).
pub fn harness_agent(
    provider: FakeProvider,
    context_budget_bytes: usize,
) -> Result<Agent<FakeProvider>, ToolError> {
    let tools = ToolBus::new(test_tool_registry()?).with_authorizer(AllowReadOnly);
    let config = AgentConfig {
        context_budget_bytes,
        ..AgentConfig::default()
    };
    Ok(Agent::new(provider, tools, test_session(), config))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode one JSON string body (without surrounding quotes).
    ///
    /// Test-only counterpart to [`escape_json_string`]: handles the short
    /// forms the encoder emits plus `\uXXXX` BMP escapes. Returns `None` on
    /// unterminated input or bad escapes.
    fn decode_json_string(body: &str) -> Option<String> {
        let mut out = String::with_capacity(body.len());
        let mut chars = body.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{08}'),
                'f' => out.push('\u{0C}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let mut code: u32 = 0;
                    for _ in 0..4 {
                        code = code * 16 + chars.next()?.to_digit(16)?;
                    }
                    out.push(char::from_u32(code)?);
                }
                _ => return None,
            }
        }
        Some(out)
    }

    /// Extract the raw (still-escaped) body of the string value for `key`,
    /// starting the scan at `from`. Handles `\"` / `\\` so an escaped quote
    /// inside the value does not end the scan early.
    fn raw_field<'a>(json: &'a str, key: &str, from: usize) -> Option<(&'a str, usize)> {
        let needle = format!("\"{key}\":\"");
        let start = json[from..].find(&needle)? + from + needle.len();
        let bytes = json.as_bytes();
        let mut index = start;
        while index < bytes.len() {
            match bytes[index] {
                b'\\' => index += 2,
                b'"' => return Some((&json[start..index], index + 1)),
                _ => index += 1,
            }
        }
        None
    }

    /// Parse the exact `params_json` shape back into its four fields,
    /// decoding the three strings. Panics on any structural deviation, so a
    /// hostile value that breaks the envelope fails the test.
    fn parse_params(json: &str) -> (String, String, String, usize) {
        let (instance_raw, after_instance) =
            raw_field(json, "instance", 0).expect("instance field parses");
        let (terminal_raw, after_terminal) =
            raw_field(json, "terminal", after_instance).expect("terminal field parses");
        let (zone_raw, after_zone) =
            raw_field(json, "zone", after_terminal).expect("zone field parses");
        let max_key = "\"max_bytes\":";
        let max_start = json[after_zone..]
            .find(max_key)
            .map(|pos| after_zone + pos + max_key.len())
            .expect("max_bytes field parses");
        let max_end = json[max_start..]
            .find('}')
            .map(|pos| max_start + pos)
            .expect("params object closes");
        let max_bytes: usize = json[max_start..max_end]
            .parse()
            .expect("max_bytes parses as usize");
        assert!(
            json[max_end + 1..].is_empty(),
            "no trailing bytes after params object"
        );
        (
            decode_json_string(instance_raw).expect("instance escapes decode"),
            decode_json_string(terminal_raw).expect("terminal escapes decode"),
            decode_json_string(zone_raw).expect("zone escapes decode"),
            max_bytes,
        )
    }

    #[test]
    fn params_json_escapes_quote_backslash_and_control_chars() {
        let instance = "inst\"\\-1";
        let terminal = "term-\n-\u{1}";
        let zone = "zo\"ne\\test\n\r\t\u{8}\u{c}\u{1}\u{1f}\u{7f}";
        let request = SnapshotRequest::new(instance, terminal, zone, "term-1", 4096);
        let bytes = request.params_json();
        let json = String::from_utf8(bytes).expect("params are UTF-8");

        // The hostile bytes must appear only in escaped form.
        assert!(json.contains("\\\""), "quote is escaped: {json}");
        assert!(json.contains("\\\\"), "backslash is escaped: {json}");
        assert!(json.contains("\\n"), "LF is escaped: {json}");
        assert!(json.contains("\\r"), "CR is escaped: {json}");
        assert!(json.contains("\\t"), "TAB is escaped: {json}");
        assert!(
            json.contains("\\b") || json.contains("\\u0008"),
            "BS is escaped: {json}"
        );
        assert!(
            json.contains("\\f") || json.contains("\\u000c"),
            "FF is escaped: {json}"
        );
        assert!(json.contains("\\u0001"), "C0 is \\u-escaped: {json}");
        assert!(json.contains("\\u001f"), "C0 is \\u-escaped: {json}");
        assert!(json.contains("\\u007f"), "DEL is \\u-escaped: {json}");
        assert!(
            !json.bytes().any(|b| b < 0x20),
            "no raw C0 byte survives: {json:?}"
        );

        // Well-formed: the exact envelope round-trips to the inputs.
        let (instance_got, terminal_got, zone_got, max_got) = parse_params(&json);
        assert_eq!(instance_got, instance);
        assert_eq!(terminal_got, terminal);
        assert_eq!(zone_got, zone);
        assert_eq!(max_got, 4096);
    }

    #[test]
    fn params_json_plain_values_keep_stable_shape() {
        let request = SnapshotRequest::new("inst-1", "term-1", "output", "term-1", 4096);
        let json = String::from_utf8(request.params_json()).expect("params are UTF-8");
        assert_eq!(
            json,
            r#"{"instance":"inst-1","terminal":"term-1","zone":"output","max_bytes":4096}"#
        );
    }

    #[test]
    fn terminal_record_ids_distinguish_sources_per_generation() {
        // P2-2 (AI-0061): turn-scoped ids must be unique or assembly
        // rejects them as DuplicateRecordId. Distinct terminals sampled in
        // one generation get distinct ids; re-sampling one terminal in the
        // same generation reuses its id, so callers advance the generation
        // per sample instead of assembling same-id pairs.
        let first = terminal_record("term-1", 1, 0, b"a".to_vec()).expect("record");
        let second = terminal_record("term-2", 1, 0, b"b".to_vec()).expect("record");
        assert_ne!(first.id, second.id);
        let resample = terminal_record("term-1", 1, 1, b"c".to_vec()).expect("record");
        assert_eq!(first.id, resample.id);
        let next_gen = terminal_record("term-1", 2, 2, b"d".to_vec()).expect("record");
        assert_ne!(first.id, next_gen.id);
    }
}

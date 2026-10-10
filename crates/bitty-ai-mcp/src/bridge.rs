//! [`McpToolAdapter`]: the runtime [`ToolExecutor`] seam over MCP tools.
//!
//! Gate order per call (fail-closed, zero transport contact on refusal):
//! tool-list staleness (AI-0211 deny-stale), read-only per-call narrowing
//! (AI-0216 [`McpToolAdapter::execute_narrowed`], OpenAI cookbook-201
//! equivalent), allowlisted import lookup, `inspect`-tier read-only gate,
//! runtime [`ToolAuthorizer`], runtime [`ConsentLedger`] (exact `(protocol,
//! tool, scope=spec.required_scope)` triple at `now_ms`), then bounded
//! `tools/call` dispatch. Secrets never pass through the adapter: child
//! environments are built once by [`crate::supervise::spawn_server`] from
//! [`CredentialRef`] names, and every error below quotes names only.
//!
//! Dynamic invalidation (AI-0211, AIQ-08 facet): the host polls for server
//! messages (HTTP `poll_server_messages` first; stdio drains inline), routes
//! each queued line through
//! [`crate::tools::is_tools_list_changed_notification`], and marks that
//! server's adapter stale via [`McpToolAdapter::mark_stale`] (or
//! [`McpToolAdapter::observe_notification`]). While stale, `execute` denies
//! with [`ToolError::Denied`] (`tool list stale; re-list required`) and zero
//! transport contact. An explicit host-driven [`McpToolAdapter::relist`]
//! re-runs `tools/list`, diffs digests, swaps the snapshot on success,
//! bumps the version, and clears stale — unless a
//! `notifications/tools/list_changed` arrived mid-round-trip, in which case
//! the surfaced frame is routed through
//! [`McpToolAdapter::observe_notification`] after the refresh and stale is
//! preserved (AI-0212). Mid-call notifications are likewise routed on the
//! adapter without changing the call's outcome. No auto-relist, no background
//! threads, no [`ToolSpec`](bitty_ai_runtime::tool::ToolSpec) registry
//! mutation. `notifications/resources/list_changed` and
//! `notifications/prompts/list_changed` have pure classifiers
//! ([`crate::tools::is_resources_list_changed_notification`],
//! [`crate::tools::is_prompts_list_changed_notification`]) for taxonomy
//! symmetry only — [`McpToolAdapter::observe_notification`], `relist`, and
//! `execute` stay tools-only with no version/stale/relist until a consumer
//! exists.
//!
//! Seam mapping for server answers: an MCP `isError: true` answer (or a
//! JSON-RPC `error` answer outside the pre-execution protocol set) maps to
//! [`ToolError::Failed`] with the reason
//! `tool reported failure: <bounded server text>`. `Failed` means the tool
//! executed and the host reported failure, distinct from [`ToolError::Denied`]
//! (a policy refusal with no effect: per-call narrowing omission, allowlist
//! miss, `inspect`-tier, authorizer, or consent). A JSON-RPC `error` answer with code `-32600`,
//! `-32601`, or `-32602` maps to [`ToolError::ProtocolRejected`]: the server
//! refused the frame before any effect, so the call never executed: never
//! `Denied` (policy), never `EffectUnknown` (uncertain effect, reconcile),
//! never `Failed` (executed-then-failed). The bus records a
//! model-observable failed status and the turn continues; the model may
//! retry only as a new call with fixed params. The reason preserves the
//! true attribution (rejected before execution, with the protocol code).
//! Transport faults mid-call map to `EffectUnknown` (in-flight effect
//! uncertain).

use bitty_ai_runtime::bridge::{
    ConsentLedger, ConsentQuery, ensure_consented, validate_protocol_id,
};
use bitty_ai_runtime::session::AgentLevel;
use bitty_ai_runtime::tool::{
    AuthBase, AuthContext, AuthDecision, ToolAuthorizer, ToolError, ToolExecutor, ToolSuccess,
};

use crate::call::{McpCallError, call_tool};
use crate::error::{McpError, McpFailure, McpStage, bound_error_text};
use crate::tools::{
    ImportedTool, ToolListDiff, ToolListSnapshot, diff_tool_snapshots,
    is_tools_list_changed_notification, list_tools,
};
use crate::{MAX_TIMEOUT_MS, McpTransport};

/// Adapter construction parameters (keeps [`McpToolAdapter::new`] under the
/// argument-count lint while staying a plain struct).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpAdapterParams {
    /// Wire principal (`owner.name`) charged for consent.
    pub protocol_id: String,
    /// Caller identity and tier for authorization.
    pub base: AuthBase,
    /// Pinned `cwd` echoed to `roots/list` during calls.
    pub cwd: String,
    /// Whole-call timeout in milliseconds (`1..=MAX_TIMEOUT_MS`).
    pub timeout_ms: u64,
}

/// Runtime [`ToolExecutor`] over one server's allowlisted MCP imports.
///
/// Synchronous only (`&mut self`, no threads, no async). Owns the transport,
/// the request-id counter, and the policy hooks; the host owns the
/// [`ToolSpec`](bitty_ai_runtime::tool::ToolSpec) registry built from the
/// same [`ImportedTool`] set.
pub struct McpToolAdapter {
    server_id: String,
    tools: Vec<ImportedTool>,
    transport: Box<dyn McpTransport>,
    authorizer: Box<dyn ToolAuthorizer>,
    consent: Box<dyn ConsentLedger>,
    params: McpAdapterParams,
    next_id: u64,
    list_version: u64,
    list_stale: bool,
}

impl McpToolAdapter {
    /// Build an adapter over already-imported allowlisted tools.
    ///
    /// Validates the wire principal and the timeout fail-closed; tools must
    /// be non-empty (an adapter with nothing to dispatch is a config error).
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::InvalidConfig`] for a malformed principal, a
    /// zero/over-ceiling timeout, or an empty tool set.
    pub fn new(
        server_id: &str,
        tools: Vec<ImportedTool>,
        transport: Box<dyn McpTransport>,
        authorizer: Box<dyn ToolAuthorizer>,
        consent: Box<dyn ConsentLedger>,
        params: McpAdapterParams,
    ) -> Result<Self, McpError> {
        validate_protocol_id(&params.protocol_id).map_err(|_| {
            McpError::new(
                McpStage::Bridge,
                McpFailure::InvalidConfig {
                    reason: bound_error_text(
                        "protocol id must be owner.name",
                        crate::error::MAX_ERROR_TEXT_BYTES,
                    ),
                },
            )
        })?;
        if params.timeout_ms == 0 || params.timeout_ms > MAX_TIMEOUT_MS {
            return Err(McpError::new(
                McpStage::Bridge,
                McpFailure::InvalidConfig {
                    reason: bound_error_text(
                        "timeout_ms must be within 1..=30000",
                        crate::error::MAX_ERROR_TEXT_BYTES,
                    ),
                },
            ));
        }
        if tools.is_empty() {
            return Err(McpError::new(
                McpStage::Bridge,
                McpFailure::InvalidConfig {
                    reason: bound_error_text(
                        "adapter imports no allowlisted tools",
                        crate::error::MAX_ERROR_TEXT_BYTES,
                    ),
                },
            ));
        }
        Ok(Self {
            server_id: server_id.to_owned(),
            tools,
            transport,
            authorizer,
            consent,
            params,
            next_id: 2,
            list_version: 0,
            list_stale: false,
        })
    }

    /// Imported tools backing this adapter.
    #[must_use]
    pub fn imported(&self) -> &[ImportedTool] {
        &self.tools
    }

    /// Server id.
    #[must_use]
    pub fn server_id(&self) -> &str {
        &self.server_id
    }

    /// Monotonic tool-list version (AI-0211).
    ///
    /// Starts at `0`. Each host-detected `notifications/tools/list_changed`
    /// bumps it saturating, and each successful explicit [`McpToolAdapter::relist`]
    /// bumps it saturating. Failed re-lists leave it unchanged.
    #[must_use]
    pub fn list_version(&self) -> u64 {
        self.list_version
    }

    /// Whether the tool list is stale (AI-0211 deny-stale).
    ///
    /// While true, [`ToolExecutor::execute`] denies with
    /// [`ToolError::Denied`] and zero transport contact until an explicit
    /// [`McpToolAdapter::relist`] succeeds.
    #[must_use]
    pub fn is_stale(&self) -> bool {
        self.list_stale
    }

    /// Next JSON-RPC request id (counter discipline probe for tests).
    #[must_use]
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Host-side digest snapshot of the current imports (AI-0211 diff
    /// handle; see [`ToolListSnapshot`]).
    #[must_use]
    pub fn snapshot(&self) -> ToolListSnapshot {
        ToolListSnapshot::from_imported(&self.tools)
    }

    /// Mark the tool list stale after a host-routed
    /// `notifications/tools/list_changed` signal.
    ///
    /// Whole-list stale, monotonic saturating version bump, idempotent
    /// effect (repeated signals keep bumping the version while staying
    /// stale). Synchronous, no transport contact, no auto-relist.
    pub fn mark_stale(&mut self) {
        self.list_version = self.list_version.saturating_add(1);
        self.list_stale = true;
    }

    /// Route one host-polled line through the shared
    /// [`is_tools_list_changed_notification`] classifier, marking stale on
    /// true.
    ///
    /// Returns whether the line marked the list stale. Host duty: poll
    /// (`poll_server_messages` first on HTTP; stdio drains inline), drain
    /// with `recv_line`, and call this per line. `405`/disabled polls queue
    /// nothing, so they never reach here and never stale.
    pub fn observe_notification(&mut self, line: &str) -> bool {
        if is_tools_list_changed_notification(line) {
            self.mark_stale();
            true
        } else {
            false
        }
    }

    /// Explicit host-driven re-list (AI-0211, AI-0212).
    ///
    /// Re-runs `tools/list` with `allowlist` over the owned transport using
    /// the shared `next_id` counter discipline, diffs digests against the
    /// current snapshot, and on success swaps the snapshot, bumps the
    /// version saturating, clears stale, and returns the diff. On failure
    /// the old snapshot is kept and the adapter stays stale (version
    /// unchanged apart from routed signals below). No [`ToolSpec`](bitty_ai_runtime::tool::ToolSpec)
    /// registry mutation: the host owns the registry and re-registers from
    /// [`McpToolAdapter::imported`] under its own refusal-only policy.
    ///
    /// Mid-round-trip signals (AI-0212): notifications observed while
    /// waiting for the `tools/list` answers are routed through
    /// [`McpToolAdapter::observe_notification`] after the refresh clears
    /// stale. A `notifications/tools/list_changed` seen mid-round-trip
    /// therefore keeps `list_stale` true — the completed refresh still bumps
    /// the version, and routing marks stale with a second saturating bump —
    /// while the diff for the completed refresh returns normally. Any other
    /// surfaced frame (ping, resources/prompts signals, `id`-carrying
    /// echoes) never stales. On a failed re-list the surfaced frames are
    /// still routed through [`McpToolAdapter::observe_notification`] before
    /// the error returns: the snapshot stays old (no swap) and the version
    /// bumps only via `mark_stale` for observed signals, so a mid-flight
    /// `list_changed` keeps a non-stale adapter fail-closed instead of
    /// fail-open on the deny-stale gate (PX-0913: only frames observed
    /// during this call are routable; nothing reconciles against server
    /// state beyond them).
    ///
    /// # Errors
    ///
    /// Returns pagination, bound, collision, unknown-tool, timeout, or
    /// transport errors from [`list_tools`].
    pub fn relist(&mut self, allowlist: &[String]) -> Result<ToolListDiff, McpError> {
        self.next_id = self.next_id.max(2);
        let before = ToolListSnapshot::from_imported(&self.tools);
        let mut surfaced: Vec<String> = Vec::new();
        let fresh = match list_tools(
            self.transport.as_mut(),
            &self.server_id,
            &self.params.cwd,
            allowlist,
            &mut self.next_id,
            self.params.timeout_ms,
            &mut surfaced,
        ) {
            Ok(fresh) => fresh,
            Err(error) => {
                // Route first, then fail: a `list_changed` seen before the
                // failure must still stale the adapter (fail-closed), even
                // when the adapter was not stale going in.
                for line in &surfaced {
                    self.observe_notification(line);
                }
                return Err(error);
            }
        };
        self.next_id = self.next_id.max(2);
        let after = ToolListSnapshot::from_imported(&fresh);
        let diff = diff_tool_snapshots(&before, &after);
        self.tools = fresh;
        self.list_version = self.list_version.saturating_add(1);
        self.list_stale = false;
        for line in &surfaced {
            self.observe_notification(line);
        }
        Ok(diff)
    }

    /// Credential discipline pin: secrets resolve into the closed child
    /// environment at spawn time ([`crate::supervise::resolve_env_refs`])
    /// and this adapter only ever carries reference names. Tests pin the
    /// wording so the invariant cannot drift silently.
    #[must_use]
    pub fn credential_discipline() -> &'static str {
        "secrets resolve into the closed child environment at spawn; the adapter carries names only"
    }

    fn lookup(&self, tool: &str) -> Option<&ImportedTool> {
        self.tools.iter().find(|entry| entry.spec.name == tool)
    }

    /// Read-only per-call tool narrowing at execute time (AI-0216, OpenAI
    /// cookbook-201 equivalent).
    ///
    /// `allowed` is the caller-scoped allow-set for this call only
    /// (`None` = no constraint, same as [`ToolExecutor::execute`];
    /// `Some(list)` = only the listed sanitized names may dispatch). A tool
    /// omitted from a constrained set denies with [`ToolError::Denied`]
    /// (never `UnknownTool`/`Failed`/`EffectUnknown`/`ProtocolRejected`, no
    /// new variant) with zero transport contact and no state mutation:
    /// snapshot, digests, `list_version`, `list_stale`, and `next_id` are
    /// unchanged, and narrowing never re-lists. Gate order is
    /// stale -> per-call-omit -> allowlist-lookup -> inspect-tier ->
    /// authorizer -> consent -> dispatch, so a stale adapter still reports
    /// the stale denial first and `relist` stays the sole un-staling path.
    pub fn execute_narrowed(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
        allowed: Option<&[String]>,
    ) -> Result<ToolSuccess, ToolError> {
        self.execute_inner(tool, arguments, now_ms, allowed)
    }

    /// Whether `tool` is narrowed away by a per-call allow-set.
    fn narrowing_omits(allowed: Option<&[String]>, tool: &str) -> bool {
        match allowed {
            None => false,
            Some(list) => !list.iter().any(|entry| entry == tool),
        }
    }

    fn execute_inner(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
        allowed: Option<&[String]>,
    ) -> Result<ToolSuccess, ToolError> {
        // AI-0211 deny-stale first gate: whole-list stale denies with
        // `Denied` (never `EffectUnknown`/`ProtocolRejected`) and zero
        // transport contact. The host must `relist` explicitly.
        if self.list_stale {
            return Err(self.deny(tool, "tool list stale; re-list required"));
        }
        // AI-0216 per-call narrowing gate (read-only, zero contact): a tool
        // omitted from the caller-scoped allow-set denies before any
        // allowlist lookup, with the reason naming the tool.
        if Self::narrowing_omits(allowed, tool) {
            return Err(self.deny(
                tool,
                &format!("tool '{tool}' omitted by per-call narrowing"),
            ));
        }
        let (raw_name, required_scope, read_only) = match self.lookup(tool) {
            Some(entry) => (
                entry.raw_name.clone(),
                entry.spec.required_scope.clone(),
                entry.spec.read_only,
            ),
            None => {
                return Err(self.deny(
                    tool,
                    &format!(
                        "tool '{tool}' is not an allowlisted import of '{}'",
                        self.server_id
                    ),
                ));
            }
        };
        if self.params.base.level == AgentLevel::Inspect && !read_only {
            return Err(self.deny(tool, "inspect tier is read-only; elevation required"));
        }
        let decision = self.authorizer.authorize(&AuthContext {
            base: self.params.base,
            tool,
            required_scope: &required_scope,
            read_only,
        });
        if let AuthDecision::Deny { reason } = decision {
            return Err(self.deny(tool, &format!("authorizer denied: {reason}")));
        }
        if let Err(error) = ensure_consented(
            self.consent.as_ref(),
            &ConsentQuery {
                protocol_id: &self.params.protocol_id,
                agent_instance: self.params.base.agent_instance_id,
                tool,
                scope: &required_scope,
                now_ms,
            },
        ) {
            return Err(self.deny(tool, &format!("consent denied: {error}")));
        }
        let id = self.next_id.max(2);
        self.next_id = id.wrapping_add(1).max(2);
        let mut surfaced: Vec<String> = Vec::new();
        let outcome = call_tool(
            self.transport.as_mut(),
            id,
            &raw_name,
            tool,
            arguments,
            &self.params.cwd,
            self.params.timeout_ms,
            &mut surfaced,
        );
        // AI-0212: mid-call notifications reach the adapter instead of being
        // dropped. Routing marks stale for the *next* call; this call's
        // outcome is unchanged — no auto-relist, no retry.
        for line in &surfaced {
            self.observe_notification(line);
        }
        match outcome {
            Ok(success) => Ok(success),
            Err(McpCallError::Transport(_)) => Err(ToolError::EffectUnknown {
                name: tool.to_owned(),
                reason: "transport failed mid-call; reconcile before retry".to_owned(),
            }),
            Err(McpCallError::ArgumentsTooLarge { limit, actual }) => {
                Err(ToolError::ArgumentsTooLarge { limit, actual })
            }
            Err(McpCallError::ResultRejected { limit, actual }) => {
                Err(ToolError::ResultTooLarge { limit, actual })
            }
            Err(McpCallError::Failed { reason, .. }) => {
                Err(self.failed(tool, &format!("tool reported failure: {reason}")))
            }
            // Pre-execution protocol refusal: never `Denied` (policy), never
            // `EffectUnknown` (uncertain effect), never `Failed`
            // (executed-then-failed). The bus records it model-observable
            // with turn-continue semantics under its own attribution.
            Err(McpCallError::ProtocolRejected { code, message, .. }) => {
                Err(self.protocol_rejected(tool, code, &message))
            }
            Err(McpCallError::Unknown { reason, .. }) => Err(ToolError::EffectUnknown {
                name: tool.to_owned(),
                reason,
            }),
        }
    }

    fn deny(&self, tool: &str, reason: &str) -> ToolError {
        ToolError::Denied {
            name: tool.to_owned(),
            reason: reason.to_owned(),
        }
        .normalized()
    }

    fn failed(&self, tool: &str, reason: &str) -> ToolError {
        ToolError::Failed {
            name: tool.to_owned(),
            reason: reason.to_owned(),
        }
        .normalized()
    }

    fn protocol_rejected(&self, tool: &str, code: i32, message: &str) -> ToolError {
        ToolError::ProtocolRejected {
            name: tool.to_owned(),
            reason: format!("protocol rejected before execution (code {code}): {message}"),
        }
        .normalized()
    }
}

impl ToolExecutor for McpToolAdapter {
    fn execute(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<ToolSuccess, ToolError> {
        // Shared gate chain with no per-call constraint (`None`): stale ->
        // per-call-omit (inactive) -> allowlist-lookup -> inspect-tier ->
        // authorizer -> consent -> dispatch. The [`ToolExecutor`] trait stays
        // frozen (v0.1); narrowing travels only on the adapter-native
        // [`McpToolAdapter::execute_narrowed`].
        self.execute_narrowed(tool, arguments, now_ms, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_ai_runtime::bridge::FakeConsentLedger;
    use bitty_ai_runtime::session::IdIssuer;
    use bitty_ai_runtime::tool::{
        AuthContext, ToolBus, ToolCall, ToolRegistry, ToolSpec, ToolStatus,
    };
    use std::collections::VecDeque;

    struct FakeTransport {
        inbound: VecDeque<String>,
        sent: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    }

    impl FakeTransport {
        fn ok_answer(id: u64, text: &str) -> String {
            format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{text}\"}}]}}}}"
            )
        }

        fn fresh(inbound: Vec<String>) -> (Self, std::rc::Rc<std::cell::RefCell<Vec<String>>>) {
            let sent = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            (
                Self {
                    inbound: inbound.into_iter().collect(),
                    sent: std::rc::Rc::clone(&sent),
                },
                sent,
            )
        }
    }

    impl McpTransport for FakeTransport {
        fn send_line(&mut self, line: &str) -> Result<(), McpError> {
            self.sent.borrow_mut().push(line.to_owned());
            Ok(())
        }

        fn recv_line(&mut self, _timeout_ms: u64) -> Result<Option<String>, McpError> {
            Ok(self.inbound.pop_front())
        }
    }

    struct Allow;
    impl ToolAuthorizer for Allow {
        fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
            AuthDecision::Allow
        }
    }

    struct Deny;
    impl ToolAuthorizer for Deny {
        fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
            AuthDecision::Deny {
                reason: "test denial".to_owned(),
            }
        }
    }

    fn imported_echo() -> ImportedTool {
        let spec = ToolSpec::new(
            "mcp_demo_echo",
            "Echo",
            br#"{"type":"object"}"#.to_vec(),
            "mcp.demo",
            true,
        )
        .expect("valid");
        ImportedTool {
            digest: spec.schema_digest(),
            server_id: "demo".to_owned(),
            raw_name: "echo".to_owned(),
            spec,
        }
    }

    fn base(level: AgentLevel) -> AuthBase {
        let mut issuer = IdIssuer::default();
        AuthBase {
            agent_instance_id: issuer.agent_instance(),
            session_id: issuer.session(),
            level,
        }
    }

    fn consented() -> FakeConsentLedger {
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "mcp_demo_echo", "mcp.demo", 9_000)
            .expect("grant");
        ledger
    }

    fn make_adapter(
        transport: FakeTransport,
        authorizer: impl ToolAuthorizer + 'static,
        consent: impl ConsentLedger + 'static,
        level: AgentLevel,
    ) -> McpToolAdapter {
        McpToolAdapter::new(
            "demo",
            vec![imported_echo()],
            Box::new(transport),
            Box::new(authorizer),
            Box::new(consent),
            McpAdapterParams {
                protocol_id: "local.assistant".to_owned(),
                base: base(level),
                cwd: "/tmp/bitty".to_owned(),
                timeout_ms: 1_000,
            },
        )
        .expect("adapter")
    }

    #[test]
    fn allowlist_miss_denies_with_zero_contact() {
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let error = adapter
            .execute("mcp_demo_ghost", b"{}", 1_000)
            .expect_err("allowlist miss must deny");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert!(sent.borrow().is_empty());
    }

    #[test]
    fn authorizer_and_consent_deny_without_contact() {
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Deny, consented(), AgentLevel::Workspace);
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("authorizer must deny");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert!(sent.borrow().is_empty());

        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(
            transport,
            Allow,
            FakeConsentLedger::new(),
            AgentLevel::Workspace,
        );
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("consent must deny");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert!(sent.borrow().is_empty());
    }

    #[test]
    fn inspect_tier_denies_mutating_tools() {
        let mut spec = imported_echo();
        spec.spec.read_only = false;
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = McpToolAdapter::new(
            "demo",
            vec![spec],
            Box::new(transport),
            Box::new(Allow),
            Box::new(consented()),
            McpAdapterParams {
                protocol_id: "local.assistant".to_owned(),
                base: base(AgentLevel::Inspect),
                cwd: "/tmp/bitty".to_owned(),
                timeout_ms: 1_000,
            },
        )
        .expect("adapter");
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("inspect must deny mutating");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert!(sent.borrow().is_empty());
    }

    #[test]
    fn allowed_call_dispatches_and_returns_text() {
        let (transport, _sent) = FakeTransport::fresh(vec![FakeTransport::ok_answer(2, "hi")]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let success = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect("dispatch");
        assert_eq!(success.data, b"hi");
    }

    #[test]
    fn tool_reported_failure_maps_to_failed_with_attribution() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"kaput\"}],\"isError\":true}}";
        let (transport, _sent) = FakeTransport::fresh(vec![line.to_owned()]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("isError must fail");
        match error {
            ToolError::Failed { name, reason } => {
                assert_eq!(name, "mcp_demo_echo");
                assert!(reason.contains("tool reported failure"));
                assert!(reason.contains("kaput"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn rpc_error_answer_maps_to_failed_not_denied() {
        // `-32603` is outside the pre-execution protocol set: it keeps the
        // executed-failure mapping, never a policy denial.
        let line =
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32603,\"message\":\"bad params\"}}";
        let (transport, _sent) = FakeTransport::fresh(vec![line.to_owned()]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("RPC error must fail");
        match error {
            ToolError::Failed { reason, .. } => {
                assert!(reason.contains("tool reported failure"));
                assert!(reason.contains("bad params"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn pre_execution_error_maps_to_protocol_rejected_not_failed() {
        // `-32600`/`-32601`/`-32602` never executed: the adapter reports the
        // distinct protocol outcome, never `Denied`, `EffectUnknown`, or
        // `Failed`.
        for code in [-32600, -32601, -32602] {
            let line = format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{{\"code\":{code},\"message\":\"bad frame\"}}}}"
            );
            let (transport, _sent) = FakeTransport::fresh(vec![line]);
            let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
            let error = adapter
                .execute("mcp_demo_echo", b"{}", 1_000)
                .expect_err("protocol error must reject");
            match &error {
                ToolError::ProtocolRejected { name, reason } => {
                    assert_eq!(name, "mcp_demo_echo");
                    assert!(reason.contains("protocol rejected before execution"));
                    assert!(reason.contains(&code.to_string()));
                    assert!(reason.contains("bad frame"));
                }
                other => panic!("code {code}: expected ProtocolRejected, got {other:?}"),
            }
            assert!(!matches!(error, ToolError::Denied { .. }));
            assert!(!matches!(error, ToolError::EffectUnknown { .. }));
            assert!(!matches!(error, ToolError::Failed { .. }));
        }
    }

    #[test]
    fn bus_dispatch_maps_is_error_failure_to_failed_status() {
        // End-to-end through the bus: an `isError: true` answer is an
        // executed-then-failed effect, so dispatch records
        // `ToolStatus::Failed`, never `Denied` (policy refusal with no
        // effect).
        let line = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"kaput\"}],\"isError\":true}}";
        let (transport, _sent) = FakeTransport::fresh(vec![line.to_owned()]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolSpec::new(
                    "mcp_demo_echo",
                    "Echo",
                    br#"{"type":"object"}"#.to_vec(),
                    "mcp.demo",
                    true,
                )
                .expect("valid"),
            )
            .expect("capacity");
        let mut bus = ToolBus::new(registry).with_authorizer(Allow);
        let call = ToolCall {
            name: "mcp_demo_echo".to_owned(),
            arguments: b"{}".to_vec(),
        };
        let mut ids = IdIssuer::default();
        let base_workspace = base(AgentLevel::Workspace);
        let execution = bus
            .dispatch(&mut adapter, &call, &base_workspace, ids.execution(), 1_000)
            .expect("host failure is a recorded status, not a bus error");
        match &execution.status {
            ToolStatus::Failed { reason } => {
                assert!(reason.contains("kaput"));
            }
            other => panic!("expected Failed status, got {other:?}"),
        }
        assert!(execution.data.is_empty());
        assert!(execution.is_untrusted_surface);
    }

    #[test]
    fn bus_dispatch_maps_protocol_rejection_with_continue_semantics() {
        // End-to-end through the bus: a pre-execution protocol rejection is
        // a recorded non-executed outcome with turn-continue semantics (an
        // `Ok` execution, counted, model-observable, untrusted), never a
        // bus error, never `Denied` or `Unknown`.
        let line =
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32602,\"message\":\"bad params\"}}";
        let (transport, _sent) = FakeTransport::fresh(vec![line.to_owned()]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolSpec::new(
                    "mcp_demo_echo",
                    "Echo",
                    br#"{"type":"object"}"#.to_vec(),
                    "mcp.demo",
                    true,
                )
                .expect("valid"),
            )
            .expect("capacity");
        let mut bus = ToolBus::new(registry).with_authorizer(Allow);
        let call = ToolCall {
            name: "mcp_demo_echo".to_owned(),
            arguments: b"{}".to_vec(),
        };
        let mut ids = IdIssuer::default();
        let base_workspace = base(AgentLevel::Workspace);
        let execution = bus
            .dispatch(&mut adapter, &call, &base_workspace, ids.execution(), 1_000)
            .expect("protocol rejection is a recorded status, not a bus error");
        match &execution.status {
            ToolStatus::Failed { reason } => {
                assert!(reason.contains("protocol rejected before execution"));
            }
            other => panic!("expected Failed status, got {other:?}"),
        }
        assert!(
            execution.summary.contains("before execution"),
            "summary must mark non-executed, got {:?}",
            execution.summary
        );
        assert!(!matches!(execution.status, ToolStatus::Denied { .. }));
        assert!(!matches!(execution.status, ToolStatus::Unknown { .. }));
        assert!(execution.data.is_empty());
        assert!(execution.is_untrusted_surface);
        assert_eq!(bus.calls_this_turn(), 1);
    }

    #[test]
    fn credential_discipline_names_only() {
        // Pin the credential-discipline invariant wording.
        assert!(McpToolAdapter::credential_discipline().contains("names only"));
    }

    #[test]
    fn stale_gate_denies_with_zero_contact_and_version_bump() {
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        assert_eq!(adapter.list_version(), 0);
        assert!(!adapter.is_stale());
        adapter.mark_stale();
        assert!(adapter.is_stale());
        assert_eq!(adapter.list_version(), 1);
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("stale must deny");
        match &error {
            ToolError::Denied { reason, .. } => {
                assert!(reason.contains("tool list stale; re-list required"));
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        assert!(!matches!(error, ToolError::EffectUnknown { .. }));
        assert!(sent.borrow().is_empty());
    }

    #[test]
    fn observe_notification_routes_only_bare_list_changed() {
        let (transport, _sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        assert!(adapter.observe_notification(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}"
        ));
        assert!(adapter.is_stale());
        let version = adapter.list_version();
        for line in [
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/ping\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/resources/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/prompts/list_changed\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"notifications/tools/list_changed\"}",
            "not json",
        ] {
            assert!(!adapter.observe_notification(line));
        }
        assert_eq!(adapter.list_version(), version);
    }

    #[test]
    fn relist_swaps_snapshot_bumps_version_and_keeps_next_id() {
        let list_frame = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo\",\"inputSchema\":{\"type\":\"object\"}}]}}";
        let call_frame = FakeTransport::ok_answer(3, "hi");
        let (transport, sent) = FakeTransport::fresh(vec![list_frame.to_owned(), call_frame]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        adapter.mark_stale();
        assert_eq!(adapter.next_id(), 2);
        let diff = adapter
            .relist(&["echo".to_owned()])
            .expect("relist succeeds");
        assert!(diff.is_empty());
        assert!(!adapter.is_stale());
        assert_eq!(adapter.list_version(), 2);
        assert_eq!(adapter.next_id(), 3);
        let sent_before_call = sent.borrow().len();
        assert_eq!(sent_before_call, 1);
        assert!(sent.borrow()[0].contains("\"id\":2"));
        let success = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect("post-relist allows");
        assert_eq!(success.data, b"hi");
        assert!(sent.borrow()[1].contains("\"id\":3"));
    }

    #[test]
    fn failed_relist_keeps_snapshot_and_stays_stale() {
        let error_frame =
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32603,\"message\":\"boom\"}}";
        let (transport, sent) = FakeTransport::fresh(vec![error_frame.to_owned()]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let before = adapter.snapshot();
        adapter.mark_stale();
        let version = adapter.list_version();
        let error = adapter.relist(&["echo".to_owned()]).expect_err("must fail");
        let _ = error;
        assert!(adapter.is_stale());
        assert_eq!(adapter.list_version(), version);
        assert_eq!(adapter.snapshot(), before);
        let denied = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("still stale");
        assert!(matches!(denied, ToolError::Denied { .. }));
        assert_eq!(sent.borrow().len(), 1);
    }

    // ── AI-0216: read-only per-call narrowing (cookbook-201 equivalent) ──

    #[test]
    fn narrowed_away_denies_with_zero_contact_and_unchanged_state() {
        // Omission denies as `Denied` (never `UnknownTool`/`Failed`/
        // `EffectUnknown`/`ProtocolRejected`, no new variant) with zero
        // transport contact and no state mutation.
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let before_snapshot = adapter.snapshot();
        let before_digest = before_snapshot.digest_of("mcp_demo_echo");
        let before_version = adapter.list_version();
        let before_next_id = adapter.next_id();
        assert!(!adapter.is_stale());

        let allowed = vec!["mcp_demo_other".to_owned()];
        let error = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&allowed))
            .expect_err("narrowed-away must deny");
        match &error {
            ToolError::Denied { name, reason } => {
                assert_eq!(name, "mcp_demo_echo");
                assert!(
                    reason.contains("mcp_demo_echo"),
                    "reason must name tool: {reason:?}"
                );
                assert!(
                    reason.contains("narrowing"),
                    "reason must mark narrowing: {reason:?}"
                );
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        assert!(!matches!(error, ToolError::UnknownTool { .. }));
        assert!(!matches!(error, ToolError::Failed { .. }));
        assert!(!matches!(error, ToolError::EffectUnknown { .. }));
        assert!(!matches!(error, ToolError::ProtocolRejected { .. }));
        assert!(sent.borrow().is_empty(), "zero transport contact");
        assert_eq!(adapter.snapshot(), before_snapshot);
        assert_eq!(adapter.snapshot().digest_of("mcp_demo_echo"), before_digest);
        assert_eq!(adapter.list_version(), before_version);
        assert!(!adapter.is_stale());
        assert_eq!(adapter.next_id(), before_next_id);

        // Explicit empty allow-set narrows everything away the same way.
        let empty: Vec<String> = Vec::new();
        let error = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&empty))
            .expect_err("empty set must deny");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert!(sent.borrow().is_empty());
        assert_eq!(adapter.snapshot(), before_snapshot);
        assert_eq!(adapter.list_version(), before_version);
        assert!(!adapter.is_stale());
    }

    #[test]
    fn narrowed_allowed_dispatches_and_none_matches_execute() {
        let allowed = vec!["mcp_demo_echo".to_owned()];
        let (transport, sent) = FakeTransport::fresh(vec![FakeTransport::ok_answer(2, "hi")]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let success = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&allowed))
            .expect("allowed must dispatch");
        assert_eq!(success.data, b"hi");
        assert_eq!(sent.borrow().len(), 1);

        // `None` behaves exactly like `execute` (shared gate chain).
        let (transport, sent) = FakeTransport::fresh(vec![FakeTransport::ok_answer(2, "hi")]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let via_narrowed = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, None)
            .expect("None dispatches");
        assert_eq!(via_narrowed.data, b"hi");
        assert_eq!(sent.borrow().len(), 1);

        let (transport, sent) = FakeTransport::fresh(vec![FakeTransport::ok_answer(2, "hi")]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let via_execute = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect("execute dispatches");
        assert_eq!(via_execute, via_narrowed);
        assert_eq!(sent.borrow().len(), 1);
    }

    #[test]
    fn stale_wins_over_narrowing() {
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        adapter.mark_stale();
        let version = adapter.list_version();

        // Omitted tool still reports stale first (relist is the sole
        // un-staling path).
        let omitted = vec!["mcp_demo_other".to_owned()];
        let error = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&omitted))
            .expect_err("stale must win");
        match &error {
            ToolError::Denied { reason, .. } => {
                assert!(reason.contains("stale"), "stale must win: {reason:?}");
                assert!(
                    !reason.contains("narrowing"),
                    "narrowing must not win: {reason:?}"
                );
            }
            other => panic!("expected Denied, got {other:?}"),
        }

        // Even an allowed tool reports stale first.
        let allowed = vec!["mcp_demo_echo".to_owned()];
        let error = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&allowed))
            .expect_err("stale must win");
        match &error {
            ToolError::Denied { reason, .. } => assert!(reason.contains("stale")),
            other => panic!("expected Denied, got {other:?}"),
        }
        assert_eq!(adapter.list_version(), version);
        assert!(adapter.is_stale());
        assert!(sent.borrow().is_empty());
    }

    #[test]
    fn narrowed_mid_call_list_changed_stales_next_call_only() {
        // Allowed narrowed call with a `list_changed` mid-wait succeeds; the
        // signal stales only the next call (AI-0212 preserved under
        // narrowing). Narrowing itself never re-lists.
        let signal =
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}".to_owned();
        let call_frame = FakeTransport::ok_answer(2, "hi");
        let (transport, sent) = FakeTransport::fresh(vec![signal, call_frame]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let allowed = vec!["mcp_demo_echo".to_owned()];
        let success = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&allowed))
            .expect("narrowed call succeeds");
        assert_eq!(success.data, b"hi");
        assert_eq!(sent.borrow().len(), 1);
        assert!(adapter.is_stale());
        assert_eq!(adapter.list_version(), 1);

        let error = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&allowed))
            .expect_err("next call must deny stale");
        match &error {
            ToolError::Denied { reason, .. } => assert!(reason.contains("stale")),
            other => panic!("expected Denied, got {other:?}"),
        }
        assert_eq!(sent.borrow().len(), 1);
    }

    #[test]
    fn narrowing_never_relists_or_mutates_snapshot() {
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let before_snapshot = adapter.snapshot();
        let before_version = adapter.list_version();
        let before_next_id = adapter.next_id();

        // Repeated narrowed-away denials mutate nothing.
        let omitted = vec!["mcp_demo_other".to_owned()];
        for _ in 0..3 {
            let error = adapter
                .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&omitted))
                .expect_err("narrowed-away must deny");
            assert!(matches!(error, ToolError::Denied { .. }));
        }
        assert_eq!(adapter.snapshot(), before_snapshot);
        assert_eq!(adapter.list_version(), before_version);
        assert!(!adapter.is_stale());
        assert_eq!(adapter.next_id(), before_next_id);
        assert!(sent.borrow().is_empty());
    }

    #[test]
    fn bus_dispatch_narrowed_refuses_pre_contact_with_denied_cause() {
        // Bus threading: a narrowed-away call is an admission refusal
        // (`Refused{Denied}`) before any executor contact.
        let (transport, sent) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolSpec::new(
                    "mcp_demo_echo",
                    "Echo",
                    br#"{"type":"object"}"#.to_vec(),
                    "mcp.demo",
                    true,
                )
                .expect("valid"),
            )
            .expect("capacity");
        let mut bus = ToolBus::new(registry).with_authorizer(Allow);
        let call = ToolCall {
            name: "mcp_demo_echo".to_owned(),
            arguments: b"{}".to_vec(),
        };
        let mut ids = IdIssuer::default();
        let base_workspace = base(AgentLevel::Workspace);
        let omitted = vec!["mcp_demo_other".to_owned()];
        let execution = bus
            .dispatch_narrowed(
                &mut adapter,
                &call,
                &base_workspace,
                ids.execution(),
                1_000,
                Some(&omitted),
            )
            .expect("narrowed refusal is a recorded status, not a bus error");
        match &execution.status {
            ToolStatus::Refused {
                cause: ToolError::Denied { name, reason },
            } => {
                assert_eq!(name, "mcp_demo_echo");
                assert!(reason.contains("narrowing"), "got {reason:?}");
            }
            other => panic!("expected Refused{{Denied}}, got {other:?}"),
        }
        assert!(!execution.is_untrusted_surface);
        assert!(execution.data.is_empty());
        assert!(sent.borrow().is_empty(), "zero executor contact");
        assert_eq!(bus.calls_this_turn(), 0);

        // The narrowed precheck mirrors the same taxonomy without dispatch.
        let error = bus
            .precheck_narrowed(
                std::slice::from_ref(&call),
                &base_workspace,
                8,
                Some(&omitted),
            )
            .expect_err("precheck must deny narrowed-away");
        assert!(matches!(error, ToolError::Denied { .. }));
        assert_eq!(bus.calls_this_turn(), 0);
    }

    /// Determinism pin for identical canonical bytes (not a narrowing
    /// invariance proof).
    ///
    /// Stability-by-construction: the narrowed allow-set travels as a
    /// separate `execute_narrowed`/`dispatch_narrowed` parameter (this file,
    /// `execute_narrowed` takes `allowed: Option<&[String]>` alongside
    /// `tool`/`arguments`/`now_ms`) and never enters `CacheKey` inputs:
    /// `CacheKey::new` takes only `(provider_id, model_id, scope,
    /// canonical: &[u8])` and hashes exactly the leading `prefix_len`
    /// canonical bytes (`crates/bitty-ai-runtime/src/cache_key.rs:154-186`,
    /// `stable_prefix_len` at `:202-220`, key fields at `:128-140`). There
    /// is therefore no narrowing input to vary here by construction, so
    /// re-keying identical bytes asserts determinism only. State
    /// immutability under narrowing is covered by the snapshot-digest
    /// assertions in the sibling narrowing tests
    /// (`narrowed_away_denies_with_zero_contact_and_unchanged_state`,
    /// `narrowing_never_relists_or_mutates_snapshot`); the digest check
    /// below is retained as a local pin that the denial leaves this
    /// adapter's imports untouched.
    #[test]
    fn cache_key_deterministic_for_identical_canonical_bytes() {
        use bitty_ai_runtime::{CacheKey, CacheScope, LayerInput, PromptLayer, PromptSnapshot};
        let snapshot = PromptSnapshot::new(
            "bitty-core-prompt@1",
            vec![
                LayerInput::text_only(PromptLayer::CoreContract, "stable core"),
                LayerInput::text_only(PromptLayer::RuntimeTurn, "turn"),
            ],
        )
        .expect("valid snapshot");
        let bytes = bitty_ai_runtime::assemble_prompt(&snapshot)
            .expect("assembles")
            .canonical_bytes()
            .to_vec();
        let first =
            CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &bytes).expect("key");
        // Determinism only: same inputs key identically. This does not vary
        // narrowing (there is no narrowing input to `CacheKey::new`); the
        // canonical bytes above are built by `assemble_prompt(&snapshot)`
        // with no allow-set in scope.
        let second =
            CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, &bytes).expect("key");
        assert_eq!(first, second);

        let (transport, _) = FakeTransport::fresh(Vec::new());
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let digest_before = adapter.snapshot().digest_of("mcp_demo_echo");
        let omitted = vec!["mcp_demo_other".to_owned()];
        let _ = adapter
            .execute_narrowed("mcp_demo_echo", b"{}", 1_000, Some(&omitted))
            .expect_err("narrowed-away must deny");
        assert_eq!(adapter.snapshot().digest_of("mcp_demo_echo"), digest_before);
    }
}

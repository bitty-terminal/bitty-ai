//! [`McpToolAdapter`]: the runtime [`ToolExecutor`] seam over MCP tools.
//!
//! Gate order per call (fail-closed, zero transport contact on refusal):
//! tool-list staleness (AI-0211 deny-stale), allowlisted import lookup,
//! `inspect`-tier read-only gate, runtime [`ToolAuthorizer`], runtime
//! [`ConsentLedger`] (exact `(protocol, tool, scope=spec.required_scope)`
//! triple at `now_ms`), then bounded `tools/call` dispatch. Secrets never
//! pass through the adapter: child environments are built once by
//! [`crate::supervise::spawn_server`] from [`CredentialRef`] names, and
//! every error below quotes names only.
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
//! bumps the version, and clears stale. No auto-relist, no background
//! threads, no [`ToolSpec`](bitty_ai_runtime::tool::ToolSpec) registry
//! mutation, no resources/prompts versioning.
//!
//! Seam mapping for server answers: an MCP `isError: true` answer (or a
//! JSON-RPC `error` answer outside the pre-execution protocol set) maps to
//! [`ToolError::Failed`] with the reason
//! `tool reported failure: <bounded server text>`. `Failed` means the tool
//! executed and the host reported failure, distinct from [`ToolError::Denied`]
//! (a policy refusal with no effect: allowlist miss, `inspect`-tier,
//! authorizer, or consent). A JSON-RPC `error` answer with code `-32600`,
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

    /// Explicit host-driven re-list (AI-0211).
    ///
    /// Re-runs `tools/list` with `allowlist` over the owned transport using
    /// the shared `next_id` counter discipline, diffs digests against the
    /// current snapshot, and on success swaps the snapshot, bumps the
    /// version saturating, clears stale, and returns the diff. On failure
    /// the old snapshot is kept and the adapter stays stale (version
    /// unchanged). No [`ToolSpec`](bitty_ai_runtime::tool::ToolSpec)
    /// registry mutation: the host owns the registry and re-registers from
    /// [`McpToolAdapter::imported`] under its own refusal-only policy.
    ///
    /// # Errors
    ///
    /// Returns pagination, bound, collision, unknown-tool, timeout, or
    /// transport errors from [`list_tools`].
    pub fn relist(&mut self, allowlist: &[String]) -> Result<ToolListDiff, McpError> {
        self.next_id = self.next_id.max(2);
        let before = ToolListSnapshot::from_imported(&self.tools);
        let fresh = list_tools(
            self.transport.as_mut(),
            &self.server_id,
            &self.params.cwd,
            allowlist,
            &mut self.next_id,
            self.params.timeout_ms,
        )?;
        self.next_id = self.next_id.max(2);
        let after = ToolListSnapshot::from_imported(&fresh);
        let diff = diff_tool_snapshots(&before, &after);
        self.tools = fresh;
        self.list_version = self.list_version.saturating_add(1);
        self.list_stale = false;
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
        // AI-0211 deny-stale first gate: whole-list stale denies with
        // `Denied` (never `EffectUnknown`/`ProtocolRejected`) and zero
        // transport contact. The host must `relist` explicitly.
        if self.list_stale {
            return Err(self.deny(tool, "tool list stale; re-list required"));
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
        match call_tool(
            self.transport.as_mut(),
            id,
            &raw_name,
            tool,
            arguments,
            &self.params.cwd,
            self.params.timeout_ms,
        ) {
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
}

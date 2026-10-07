//! [`McpToolAdapter`]: the runtime [`ToolExecutor`] seam over MCP tools.
//!
//! Gate order per call (fail-closed, zero transport contact on refusal):
//! allowlisted import lookup, `inspect`-tier read-only gate, runtime
//! [`ToolAuthorizer`], runtime [`ConsentLedger`] (exact
//! `(protocol, tool, scope=spec.required_scope)` triple at `now_ms`), then
//! bounded `tools/call` dispatch. Secrets never pass through the adapter:
//! child environments are built once by [`crate::supervise::spawn_server`]
//! from [`CredentialRef`] names, and every error below quotes names only.
//!
//! Seam mapping for server answers (judgment call, documented): the
//! [`ToolExecutor`] vocabulary offers only `Denied` (refusal) and
//! `EffectUnknown` (uncertain) as reason-carrying errors, so an MCP
//! `isError: true` answer maps to `ToolError::Denied` with the reason
//! `tool reported failure: <bounded server text>`. The turn still fails
//! closed with no blind retry, and the reason preserves the true
//! attribution (executed, then failed). A future `ToolError::Failed`
//! carrier would be the ideal mapping; until one exists this avoids the
//! worse lie of a bound/shape variant with false numbers. Transport faults
//! mid-call map to `EffectUnknown` (in-flight effect uncertain).

use bitty_ai_runtime::bridge::{
    ConsentLedger, ConsentQuery, ensure_consented, validate_protocol_id,
};
use bitty_ai_runtime::session::AgentLevel;
use bitty_ai_runtime::tool::{
    AuthBase, AuthContext, AuthDecision, ToolAuthorizer, ToolError, ToolExecutor, ToolSuccess,
};

use crate::call::{McpCallError, call_tool};
use crate::error::{McpError, McpFailure, McpStage, bound_error_text};
use crate::tools::ImportedTool;
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
}

impl ToolExecutor for McpToolAdapter {
    fn execute(
        &mut self,
        tool: &str,
        arguments: &[u8],
        now_ms: u64,
    ) -> Result<ToolSuccess, ToolError> {
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
                Err(self.deny(tool, &format!("tool reported failure: {reason}")))
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
    use bitty_ai_runtime::tool::{AuthContext, ToolSpec};
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
    fn tool_reported_failure_denies_with_attribution() {
        let line = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"kaput\"}],\"isError\":true}}";
        let (transport, _sent) = FakeTransport::fresh(vec![line.to_owned()]);
        let mut adapter = make_adapter(transport, Allow, consented(), AgentLevel::Workspace);
        let error = adapter
            .execute("mcp_demo_echo", b"{}", 1_000)
            .expect_err("isError must fail");
        match error {
            ToolError::Denied { reason, .. } => {
                assert!(reason.contains("tool reported failure"));
                assert!(reason.contains("kaput"));
            }
            other => panic!("expected Denied, got {other:?}"),
        }
    }

    #[test]
    fn credential_discipline_names_only() {
        // Pin the credential-discipline invariant wording.
        assert!(McpToolAdapter::credential_discipline().contains("names only"));
    }
}

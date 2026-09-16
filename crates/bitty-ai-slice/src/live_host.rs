//! Live host adapter for the BII-09 rendezvous (`LiveBittyHost`).
//!
//! This module owns the `bitty-ai` side of the meet-at-the-bridge step
//! ([`bitty-side-integration-input.md` BII-09][bii09]): the same
//! [`BittyHost`](crate::fake_host::BittyHost) trait that [`FakeHost`] serves
//! deterministically is now also served by a live adapter that delegates to
//! the real `bitty-ipc` host services (`SnapshotService`, `ToolDispatchService`,
//! `ExecutionService`) with the real DTOs, bounds, and `validate()` methods.
//! Dispatch prefix order, consent attribution, and `Unknown` reconcile/resolve
//! semantics therefore hold by construction, not by mirrored steps.
//!
//! [bii09]: https://github.com/bitty-terminal/bitty-ai-docs/blob/main/specifications/bitty-side-integration-input.md
//!
//! ## Direction basis (input only, not accepted architecture)
//!
//! BII-01 (bounded `terminal.snapshot` under `terminal.inspect`), BII-02
//! (generic host tool dispatch with per-tool consent), BII-03 (one common
//! authorization gate before every real effect), BII-04 (generic supervised
//! execution backend), and BII-05 (structured outcome with `Unknown`
//! reconciliation) are the direction basis for this adapter. They are
//! **draft handoff inputs** from `bitty-side-integration-input.md`: they
//! record no `bitty` decision, authorize no shipped behavior, and close no
//! open question. Likewise `execution-ownership-r1` (draft) and
//! `tool-transport-r2` (draft) are direction inputs only; the accepted
//! [`ipc-agent-rfc.md`][rfc] framing, scopes, and lifecycle contracts remain
//! the overriding authority.
//!
//! [rfc]: https://github.com/bitty-terminal/bitty-ai-docs/blob/main/specifications/ipc-agent-rfc.md
//!
//! ## Read-only mirror pin
//!
//! No file in the `bitty` repository is modified by this task. The services
//! used here are read through the pinned `bitty-ipc` revision `be6e63c`
//! (AI-0073 verified `2cbb1fb..be6e63c`: `auth.rs` token-free reasons plus
//! `devtools` surface; existing DTO/validate/bounds shapes byte-identical):
//!
//! - Snapshot DTO/service: `crates/bitty-ipc/src/snapshot.rs` `SNAPSHOT_METHOD`
//!   (`L64`), `SnapshotRequest` (`L235`, `validate` `L288`), `SnapshotData`
//!   (`L317`), `TerminalSnapshot` (`L338`, `validate` `L371`),
//!   `SnapshotService::dispatch` (`L548-580`).
//! - Tool dispatch: `crates/bitty-ipc/src/tool_dispatch.rs` `ToolSpec`
//!   (`L197`), `ToolRequest` (`L271`, `validate` `L316`), `ToolOutput`
//!   (`L346`, `validate` `L362`), `ToolExecution` (`L393`, `validate` `L427`),
//!   `ToolDispatchService::dispatch` (`L564-639`).
//! - Execution + `Unknown`: `crates/bitty-ipc/src/execution.rs`
//!   `EXECUTION_SCOPE` (`L155`), `ExecutionRequest` (`L444`, `validate`
//!   `L547`, `effective_stream_budget` `L526`), `RawExecutionOutput` (`L654`,
//!   `validate` `L685`), `ExecutionResult` (`L781`, `validate` `L870`),
//!   `ExecutionService::dispatch` (`L1047`), `reconcile` (`L1149`), `resolve`
//!   (`L1169`).
//! - Consent + scopes: `crates/bitty-ipc/src/scope.rs`
//!   `validate_method_name` (`L308`), `required_scope_for_method` (`L380`),
//!   `authorize_method` (`L425`), `ConsentLedger` (`L477`).
//!
//! If a live shape ever diverges from the expectation encoded in
//! `tests/host_conformance.rs`, the adapter fails closed (the service error
//! surfaces as [`SliceError::Ipc`] or the facade consent/method mapping) and
//! the divergence is recorded as gap input for the `bitty` track. There is no
//! bypass path around a refused dispatch.
//!
//! ## Transport seam (no live wiring)
//!
//! Connection and transport exist only as an injectable seam: provider `fn`
//! pointers (`SnapshotProvider`, `ToolProvider`, `ExecutionProvider`),
//! server-evaluated `granted` scopes, and the real `ConsentLedger`. This file
//! contains no code that connects to a real terminal, process, PTY, socket,
//! or network peer (no `std::net`, `std::process`, `std::fs`, async runtime,
//! or IPC transport), reads no wall clock (every method takes
//! caller-supplied `now_ms`), and carries no secret or credential field.
//! Tests assert exactly this seam: what is tested is mapping proof through
//! the real services with canned providers, never a claim about live data.
//!
//! ## Live-provider mapping (AI-0040, mapping proof only)
//!
//! The same seam can point at the real live providers from
//! `bitty-ipc/src/host_bridge.rs` at pin `be6e63c`: `live_snapshot_provider`
//! for snapshots plus `inspect_text_provider` / `inspect_status_provider`
//! (via `register_live_inspect_tools`) for the read-only inspect tools.
//! There is no live execution provider in this pin, so the execution seam
//! stays injected (canned in tests). Wiring is explicit:
//! [`LiveBittyHost::with_live_snapshot`] selects the live snapshot provider,
//! [`LiveBittyHost::register_live_inspect_tools`] registers the live inspect
//! tools; tests publish fixtures with `publish_live_snapshot` (real
//! `SnapshotData` DTOs) then dispatch through the live providers over the
//! unchanged dispatch paths. This is live-provider conformance (mapping
//! proof), never a live-wiring claim: no socket, process, PTY, network,
//! clock, or secret is involved, and trust binding (`HostCaller::bind` with
//! `VerifiedPeer`) stays host-side. The slice uses only the already-bound
//! caller identity (`client_id` plus server-evaluated `granted` scopes plus
//! caller-supplied `now_ms` via the consent seam); it never constructs
//! `VerifiedPeer`, never asserts scope allocation, and never rewinds the
//! clock.
//!
//! ## Live-store serial discipline (process-global `OnceLock`)
//!
//! The live store is process-global (`OnceLock<Mutex<BTreeMap>>`, bounded
//! 64, same-id overwrite, full-reject). Tests touching it must serialize:
//! acquire the module-local `live_test_lock` (`OnceLock<Mutex<()>>`, same
//! pattern as `host_bridge.rs` tests), use unique terminal ids under the
//! `t:41xx` prefix reserved for AI-0040 live-provider conformance, open with
//! a front fence (`clear_live_snapshots_for_tests` then
//! `live_snapshot_count() == 0`), and close with cleanup plus a back fence
//! (`clear_live_snapshots_for_tests`, assert `0`). Front fences prove the
//! previous test left no residue; back fences prove this test leaves none,
//! so successive runs show no cross-talk. Canned-provider tests never touch
//! the store and need no lock.
//!
//! ## Determinism and scope
//!
//! - Std-only, deterministic for a given provider set plus `now_ms`.
//! - Single-agent: one `LiveBittyHost` serves one `client_id` with its own
//!   server-evaluated `granted` scopes and consent ledger.
//! - `FakeHost` is retained for the deterministic scripted path; shared
//!   conformance (`tests/host_conformance.rs`) runs the same assertions
//!   against both hosts.

use bitty_ipc::error::IpcError;
use bitty_ipc::execution::{
    ExecutionProvider, ExecutionRequest, ExecutionResult, ExecutionService,
};
use bitty_ipc::host_bridge::{
    live_snapshot_provider, register_live_inspect_tools as register_live_tools,
};
use bitty_ipc::scope::{
    ConsentLedger, Scope, ScopeSet, authorize_method, required_scope_for_method,
    validate_method_name,
};
use bitty_ipc::snapshot::{SnapshotProvider, SnapshotRequest, SnapshotService, TerminalSnapshot};
use bitty_ipc::tool_dispatch::{
    MAX_TOOL_CLIENT_ID_BYTES, ToolDispatchService, ToolExecution, ToolProvider, ToolRequest,
    ToolSpec,
};

use bitty_ai_runtime::bridge::IdentityBridge;

use crate::bridge::wire_client_id;
use crate::error::SliceError;
use crate::fake_host::BittyHost;

/// Live adapter over the real `bitty-ipc` host services (BII-09 rendezvous).
///
/// Owns one client's server-evaluated scopes, the real consent ledger, and
/// the real snapshot / tool-dispatch / execution services. Providers are the
/// injectable transport seam: canned `fn` pointers in tests, host-wired
/// providers in a future deployment. There is no socket, process, network,
/// clock, or secret in this type.
#[derive(Debug)]
pub struct LiveBittyHost {
    client_id: String,
    granted: ScopeSet,
    consent: ConsentLedger,
    snapshots: SnapshotService,
    tools: ToolDispatchService,
    executions: ExecutionService,
}

impl LiveBittyHost {
    /// Construct a live host for `client_id` with server-evaluated `granted`
    /// scopes, an optional snapshot provider (`None` leaves the snapshot
    /// table empty so the missing-handler path stays fail-closed and
    /// observable), and the supervised-execution provider.
    ///
    /// This is the raw seam: `client_id` is taken verbatim and only
    /// length-bounded. Product callers must use [`Self::from_binding`] so the
    /// id is derived from a bound protocol principal (`AI-0065`).
    ///
    /// Tool providers are registered per tool via [`Self::register_tool`];
    /// the tool table starts empty (unknown tools fail closed).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] when `client_id` is empty or exceeds the
    /// host client bound (mirrors the dispatch client checks in
    /// `tool_dispatch.rs:564` and `execution.rs:1047`; both bounds derive
    /// from the same scoped-id ceiling).
    pub fn new(
        client_id: impl Into<String>,
        granted: ScopeSet,
        snapshot_provider: Option<SnapshotProvider>,
        execution_provider: ExecutionProvider,
    ) -> Result<Self, SliceError> {
        let client_id = client_id.into();
        if client_id.is_empty() {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: "tool client_id must not be empty".to_owned(),
            }));
        }
        if client_id.len() > MAX_TOOL_CLIENT_ID_BYTES {
            return Err(SliceError::Ipc(IpcError::LimitExceeded {
                field: "tool client_id".to_owned(),
                limit: MAX_TOOL_CLIENT_ID_BYTES,
                actual: client_id.len(),
            }));
        }
        let snapshots = match snapshot_provider {
            Some(provider) => SnapshotService::with_defaults(provider),
            None => SnapshotService::new(),
        };
        Ok(Self {
            client_id,
            granted,
            consent: ConsentLedger::new(),
            snapshots,
            tools: ToolDispatchService::new(),
            executions: ExecutionService::with_provider(execution_provider),
        })
    }

    /// Construct a live host whose client identity is derived from the bound
    /// protocol principal of `identity` (`AI-0065`).
    ///
    /// # Errors
    ///
    /// Returns the failures of [`wire_client_id`]: an unbound identity or an
    /// over-long derived id is refused before any host state exists (no
    /// consent ledger, snapshot table, tool registry, or execution store).
    pub fn from_binding(
        identity: &IdentityBridge,
        granted: ScopeSet,
        snapshot_provider: Option<SnapshotProvider>,
        execution_provider: ExecutionProvider,
    ) -> Result<Self, SliceError> {
        Self::new(
            wire_client_id(identity)?,
            granted,
            snapshot_provider,
            execution_provider,
        )
    }

    /// Construct a live host wired to the real live snapshot provider
    /// (`host_bridge::live_snapshot_provider` at pin `be6e63c`).
    ///
    /// This is live-provider conformance (mapping proof), not live wiring:
    /// tests publish real `SnapshotData` fixtures with
    /// `publish_live_snapshot`, then dispatch through the live provider over
    /// the unchanged snapshot path. The caller identity stays the
    /// already-bound seam (`client_id` plus server-evaluated `granted`
    /// scopes plus consent at caller-supplied `now_ms`); trust binding
    /// (`HostCaller::bind` with `VerifiedPeer`) remains host-side and is
    /// never constructed here. There is no live execution provider in this
    /// pin, so `execution_provider` stays injected (canned in tests).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] under the same client-id bounds as
    /// [`Self::new`].
    pub fn with_live_snapshot(
        client_id: impl Into<String>,
        granted: ScopeSet,
        execution_provider: ExecutionProvider,
    ) -> Result<Self, SliceError> {
        Self::new(
            client_id,
            granted,
            Some(live_snapshot_provider),
            execution_provider,
        )
    }

    /// Authenticated client identity served by this host.
    #[must_use]
    pub fn host_client_id(&self) -> &str {
        &self.client_id
    }

    /// Server-evaluated granted scopes (read-only view for tests).
    #[must_use]
    pub fn granted(&self) -> &ScopeSet {
        &self.granted
    }

    /// Number of registered tools.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tools.tool_count()
    }

    /// Number of stored execution outcomes.
    #[must_use]
    pub fn execution_count(&self) -> usize {
        self.executions.len()
    }

    /// Number of registered snapshot methods (0 or 1 in this adapter).
    #[must_use]
    pub fn snapshot_method_count(&self) -> usize {
        self.snapshots.method_count()
    }

    /// Register one tool declaration with its host provider (mirrors
    /// `ToolDispatchService::register`, `tool_dispatch.rs:512`).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] when the name is already registered or
    /// the host registry is at capacity (`MAX_TOOLS_PER_HOST`, fail-closed,
    /// no silent eviction).
    pub fn register_tool(
        &mut self,
        spec: ToolSpec,
        provider: ToolProvider,
    ) -> Result<(), SliceError> {
        self.tools.register(spec, provider).map_err(SliceError::Ipc)
    }

    /// Register the live read-only inspect tools (`terminal_text` plus
    /// `terminal_status`, both `terminal.inspect`) backed by the real live
    /// providers at pin `be6e63c`.
    ///
    /// This is the tool-side half of live-provider conformance: tests publish
    /// real `SnapshotData` fixtures with `publish_live_snapshot`, then
    /// dispatch `terminal_text` / `terminal_status` through the live
    /// providers over the unchanged tool-dispatch path (grammar, registry,
    /// scope, consent, echo-match, bounds, DTO-validate). Effect tools stay
    /// deny-by-default (`NotFound`) until their own slice wires them; there
    /// is no live execution provider in this pin. Trust binding stays
    /// host-side (already-bound caller seam only, no `VerifiedPeer`).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] when a live tool name is already
    /// registered or the host registry is at capacity (fail-closed, no
    /// silent overwrite or eviction).
    pub fn register_live_inspect_tools(&mut self) -> Result<(), SliceError> {
        register_live_tools(&mut self.tools).map_err(SliceError::Ipc)
    }

    /// Grant `scope` to this host's client for `ttl_ms` from `now_ms`
    /// (delegates to the real `ConsentLedger::grant`, `scope.rs:520`).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] when the ledger is at capacity or the
    /// grant is malformed (fail-closed, no silent eviction).
    pub fn grant_consent(
        &mut self,
        scope: Scope,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<(), SliceError> {
        self.consent
            .grant(
                self.client_id.clone(),
                scope,
                now_ms,
                ttl_ms,
                "live-host".to_owned(),
            )
            .map_err(SliceError::Ipc)
    }

    /// Revoke `scope` from this host's client immediately.
    pub fn revoke_consent(&mut self, scope: Scope) -> bool {
        self.consent.revoke(&self.client_id, scope)
    }

    /// Whether `scope` is currently granted to this host's client at `now_ms`.
    #[must_use]
    pub fn consent_active(&self, scope: Scope, now_ms: u64) -> bool {
        self.consent.is_granted(&self.client_id, scope, now_ms)
    }

    /// Drain grants expired at `now_ms`, returning the expired keys.
    pub fn drain_expired(&mut self, now_ms: u64) -> Vec<(String, Scope)> {
        self.consent.drain_expired(now_ms)
    }
}

impl BittyHost for LiveBittyHost {
    fn client_id(&self) -> &str {
        &self.client_id
    }

    fn snapshot(
        &mut self,
        method: &str,
        request: &SnapshotRequest,
        now_ms: u64,
    ) -> Result<TerminalSnapshot, SliceError> {
        // Prefix matches `FakeHost::snapshot` and the `IpcBridge` gate: the
        // headless `SnapshotService::dispatch` checks scope only, so the
        // facade enforces per-client consent like `bridge.rs:113-173` before
        // delegating to the real service for validation, provider call,
        // terminal match, bounding, and DTO validation.
        validate_method_name(method).map_err(SliceError::Ipc)?;
        let required =
            required_scope_for_method(method).ok_or_else(|| SliceError::UnsupportedHostMethod {
                method: method.to_owned(),
            })?;
        request.validate().map_err(SliceError::Ipc)?;
        authorize_method(method, &self.granted).map_err(SliceError::Ipc)?;
        if !self.consent.is_granted(&self.client_id, required, now_ms) {
            return Err(SliceError::ConsentRequired {
                scope: required.as_str(),
            });
        }
        self.snapshots
            .dispatch(method, request, &self.granted)
            .map_err(SliceError::Ipc)
    }

    fn dispatch_tool(
        &mut self,
        request: &ToolRequest,
        now_ms: u64,
        execution_id: u64,
    ) -> Result<ToolExecution, SliceError> {
        // Real prefix order by construction:
        // `ToolDispatchService::dispatch` (`tool_dispatch.rs:564-639`).
        // Any refusal stores nothing.
        self.tools
            .dispatch(
                request,
                &self.granted,
                &self.consent,
                &self.client_id.clone(),
                now_ms,
                execution_id,
            )
            .map_err(SliceError::Ipc)
    }

    fn execute(
        &mut self,
        request: &ExecutionRequest,
        now_ms: u64,
        execution_id: u64,
    ) -> Result<ExecutionResult, SliceError> {
        // Real prefix order by construction:
        // `ExecutionService::dispatch` (`execution.rs:1047`). Refusals store
        // nothing; `Unknown` agreement, budgeting, attribution, and store are
        // enforced by the service.
        self.executions
            .dispatch(
                request,
                &self.granted,
                &self.consent,
                &self.client_id.clone(),
                now_ms,
                execution_id,
            )
            .map_err(SliceError::Ipc)
    }

    fn reconcile(&self, execution_id: u64) -> Result<ExecutionResult, SliceError> {
        // Real query path by construction:
        // `ExecutionService::reconcile` (`execution.rs:1149`).
        self.executions
            .reconcile(execution_id)
            .map_err(SliceError::Ipc)
    }

    fn resolve(&mut self, execution_id: u64, result: ExecutionResult) -> Result<(), SliceError> {
        // Real close-out by construction:
        // `ExecutionService::resolve` (`execution.rs:1169`).
        self.executions
            .resolve(execution_id, result)
            .map_err(SliceError::Ipc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_ipc::execution::{EffectState, ExecutionStatus, RawExecutionOutput};
    use bitty_ipc::snapshot::{DetailLevel, SNAPSHOT_METHOD, SnapshotData};
    use bitty_ipc::tool_dispatch::ToolOutput;

    const NOW_MS: u64 = 1_000;
    const TTL_MS: u64 = 60_000;

    fn snapshot_provider(request: &SnapshotRequest) -> Result<SnapshotData, IpcError> {
        Ok(SnapshotData {
            terminal_id: request.terminal_id.clone(),
            generation: 7,
            cwd: "/work".to_owned(),
            semantic_zones: Vec::new(),
            text: "hello".to_owned(),
        })
    }

    fn tool_provider(request: &ToolRequest) -> Result<ToolOutput, IpcError> {
        Ok(ToolOutput {
            target_id: request.target.clone(),
            data: b"zone bytes".to_vec(),
            summary: "ok".to_owned(),
        })
    }

    fn exec_provider(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
        Ok(RawExecutionOutput {
            target_id: request.target.clone(),
            status: ExecutionStatus::Completed,
            exit_code: Some(0),
            stdout: String::new(),
            stderr: String::new(),
            evidence_refs: Vec::new(),
            effect_state: EffectState::Completed,
        })
    }

    fn live_host() -> LiveBittyHost {
        let mut host = LiveBittyHost::new(
            "agent-live",
            ScopeSet::single(Scope::TerminalInspect),
            Some(snapshot_provider),
            exec_provider,
        )
        .expect("live host construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");
        host
    }

    #[test]
    fn snapshot_delegates_to_real_service_with_trust_label() {
        let mut host = live_host();
        let request = SnapshotRequest::new("t:1", DetailLevel::Standard);
        let snapshot = host
            .snapshot(SNAPSHOT_METHOD, &request, NOW_MS)
            .expect("snapshot serves");
        assert_eq!(snapshot.terminal_id, "t:1");
        assert_eq!(snapshot.text, "hello");
        assert!(snapshot.is_untrusted_surface);
        snapshot.validate().expect("dto validates");
    }

    #[test]
    fn dispatch_delegates_to_real_service_with_attribution() {
        let mut host = LiveBittyHost::new(
            "agent-live",
            ScopeSet::single(Scope::TerminalInspect),
            Some(snapshot_provider),
            exec_provider,
        )
        .expect("construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent");
        let spec = ToolSpec::new(
            "terminal_read_zone",
            "read-only zone read",
            br#"{"type":"object"}"#.to_vec(),
            Scope::TerminalInspect,
            true,
        )
        .expect("spec");
        host.register_tool(spec, tool_provider).expect("register");
        let request = ToolRequest::new("terminal_read_zone", br#"{"zone":"output"}"#.to_vec());
        let execution = host
            .dispatch_tool(&request, NOW_MS, 41)
            .expect("dispatch serves");
        assert_eq!(execution.execution_id, 41);
        assert_eq!(execution.client_id, "agent-live");
        assert!(execution.is_untrusted_surface);
        execution.validate().expect("outcome validates");
    }

    #[test]
    fn execution_unknown_reconciles_through_real_service() {
        fn unknown_provider(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
            Ok(RawExecutionOutput {
                target_id: request.target.clone(),
                status: ExecutionStatus::Unknown,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                evidence_refs: Vec::new(),
                effect_state: EffectState::Unknown,
            })
        }

        let mut host = LiveBittyHost::new(
            "agent-live",
            ScopeSet::single(Scope::ProcessSpawn),
            Some(snapshot_provider),
            unknown_provider,
        )
        .expect("construction");
        host.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
            .expect("consent");
        let request =
            ExecutionRequest::new("git", vec!["diff".to_owned()]).with_allow_effects(true);
        let result = host.execute(&request, NOW_MS, 7).expect("unknown stores");
        assert!(result.needs_reconciliation());
        assert_eq!(host.reconcile(7).expect("reconcile"), result);
    }

    #[test]
    fn missing_snapshot_handler_fails_closed_without_bypass() {
        let mut host = LiveBittyHost::new(
            "agent-live",
            ScopeSet::single(Scope::TerminalInspect),
            None,
            exec_provider,
        )
        .expect("construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent");
        let request = SnapshotRequest::new("t:1", DetailLevel::Standard);
        let error = host
            .snapshot(SNAPSHOT_METHOD, &request, NOW_MS)
            .expect_err("missing handler must fail closed");
        assert!(
            matches!(error, SliceError::Ipc(IpcError::NotFound { .. })),
            "got {error:?}"
        );
    }
}

/// Live-provider conformance through the process-global live store (AI-0040).
///
/// Mapping proof only, never live wiring: each test publishes real
/// `SnapshotData` fixtures with `publish_live_snapshot`, then dispatches
/// through [`LiveBittyHost::with_live_snapshot`] plus
/// [`LiveBittyHost::register_live_inspect_tools`] over the unchanged
/// `bitty-ipc` dispatch paths. The caller stays the already-bound seam
/// (`client_id` plus server-evaluated scopes plus consent at
/// caller-supplied `now_ms`); no test constructs `VerifiedPeer`, asserts
/// scope allocation, or reads a clock. Execution keeps its injected
/// (canned) provider: pin `be6e63c` defines no live execution provider.
///
/// Serial discipline: the live store is process-global, so every test here
/// holds `live_test_lock`, uses a unique `t:41xx` terminal id, opens with a
/// front fence (`count == 0`) and closes with cleanup plus a back fence
/// (`count == 0`). Front fences fail when a previous test leaks; back fences
/// fail when this test leaks, so green runs prove no cross-talk.
#[cfg(test)]
mod live_provider_tests {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use bitty_ipc::error::IpcError;
    use bitty_ipc::execution::{
        EffectState, ExecutionRequest, ExecutionStatus, RawExecutionOutput,
    };
    use bitty_ipc::host_bridge::{
        INSPECT_STATUS_TOOL, INSPECT_TEXT_TOOL, clear_live_snapshots_for_tests,
        live_snapshot_count, publish_live_snapshot,
    };
    use bitty_ipc::scope::{Scope, ScopeSet};
    use bitty_ipc::snapshot::{
        DetailLevel, SNAPSHOT_METHOD, SemanticZone, SnapshotData, SnapshotRequest, ZoneKind,
    };
    use bitty_ipc::tool_dispatch::ToolRequest;

    use super::LiveBittyHost;
    use crate::error::SliceError;
    use crate::fake_host::BittyHost;

    const NOW_MS: u64 = 1_000;
    const TTL_MS: u64 = 60_000;

    /// Serialize the live-store tests in this module (the live store is
    /// process-global; parallel tests must not interleave publishes).
    fn live_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn canned_exec_provider(request: &ExecutionRequest) -> Result<RawExecutionOutput, IpcError> {
        Ok(RawExecutionOutput {
            target_id: request.target.clone(),
            status: ExecutionStatus::Completed,
            exit_code: Some(0),
            stdout: String::new(),
            stderr: String::new(),
            evidence_refs: Vec::new(),
            effect_state: EffectState::Completed,
        })
    }

    fn fixture(terminal_id: &str, generation: u64, text: &str) -> SnapshotData {
        SnapshotData {
            terminal_id: terminal_id.to_owned(),
            generation,
            cwd: "/work".to_owned(),
            semantic_zones: vec![SemanticZone {
                kind: ZoneKind::Output,
                line_start: 0,
                line_end: 2,
            }],
            text: text.to_owned(),
        }
    }

    fn live_inspect_host(client_id: &str) -> LiveBittyHost {
        let mut host = LiveBittyHost::with_live_snapshot(
            client_id,
            ScopeSet::single(Scope::TerminalInspect),
            canned_exec_provider,
        )
        .expect("live host construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");
        host.register_live_inspect_tools()
            .expect("live inspect tools register");
        host
    }

    #[test]
    fn live_snapshot_serves_published_fixture() {
        let _guard = live_test_lock();
        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "front fence: previous live test leaked"
        );

        let terminal_id = "t:4101";
        assert!(
            publish_live_snapshot(fixture(terminal_id, 9, "hello live")).expect("publish serves"),
            "fresh id stores"
        );
        assert_eq!(live_snapshot_count(), 1);

        let mut host = LiveBittyHost::with_live_snapshot(
            "agent-live-4101",
            ScopeSet::single(Scope::TerminalInspect),
            canned_exec_provider,
        )
        .expect("live host construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");

        let snapshot = host
            .snapshot(
                SNAPSHOT_METHOD,
                &SnapshotRequest::new(terminal_id, DetailLevel::Standard),
                NOW_MS,
            )
            .expect("live snapshot serves");
        assert_eq!(snapshot.terminal_id, terminal_id);
        assert_eq!(snapshot.generation, 9);
        assert_eq!(snapshot.text, "hello live");
        assert!(snapshot.is_untrusted_surface);
        snapshot.validate().expect("dto validates");
        assert_eq!(host.client_id(), "agent-live-4101");

        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "back fence: this test must leave no residue"
        );
    }

    #[test]
    fn live_inspect_text_serves_published_bytes() {
        let _guard = live_test_lock();
        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "front fence: previous live test leaked"
        );

        let terminal_id = "t:4102";
        publish_live_snapshot(fixture(terminal_id, 4, "typed bytes")).expect("publish serves");
        assert_eq!(live_snapshot_count(), 1);

        let mut host = live_inspect_host("agent-live-4102");
        assert_eq!(host.tool_count(), 2, "both live inspect tools register");

        let execution = host
            .dispatch_tool(
                &ToolRequest::new(INSPECT_TEXT_TOOL, b"{}".to_vec()).with_target(terminal_id),
                NOW_MS,
                41,
            )
            .expect("live inspect text serves");
        assert_eq!(execution.tool, INSPECT_TEXT_TOOL);
        assert_eq!(execution.target.as_deref(), Some(terminal_id));
        assert_eq!(execution.data, b"typed bytes");
        assert!(execution.summary.contains(terminal_id));
        assert!(execution.is_untrusted_surface);
        assert_eq!(execution.execution_id, 41);
        assert_eq!(execution.client_id, "agent-live-4102");
        execution.validate().expect("outcome validates");

        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "back fence: this test must leave no residue"
        );
    }

    #[test]
    fn live_inspect_status_reports_generation() {
        let _guard = live_test_lock();
        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "front fence: previous live test leaked"
        );

        let terminal_id = "t:4103";
        publish_live_snapshot(fixture(terminal_id, 6, "hello")).expect("publish serves");

        let mut host = live_inspect_host("agent-live-4103");
        let execution = host
            .dispatch_tool(
                &ToolRequest::new(INSPECT_STATUS_TOOL, b"{}".to_vec()).with_target(terminal_id),
                NOW_MS,
                42,
            )
            .expect("live inspect status serves");
        let body = String::from_utf8(execution.data).expect("status is UTF-8");
        assert!(body.contains("generation: 6"), "got {body:?}");
        assert!(body.contains("cwd: /work"), "got {body:?}");
        assert!(body.contains("zones: 1"), "got {body:?}");
        assert!(execution.is_untrusted_surface);
        assert_eq!(execution.client_id, "agent-live-4103");

        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "back fence: this test must leave no residue"
        );
    }

    #[test]
    fn live_miss_is_not_found_without_crosstalk() {
        let _guard = live_test_lock();
        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "front fence: previous live test leaked"
        );

        let terminal_id = "t:4104";
        publish_live_snapshot(fixture(terminal_id, 1, "gated")).expect("publish");

        let mut host = live_inspect_host("agent-live-4104");

        // Earlier live ids must stay absent: no cross-talk from prior tests.
        for absent in ["t:4101", "t:4102", "t:4103", "t:4199"] {
            let error = host
                .snapshot(
                    SNAPSHOT_METHOD,
                    &SnapshotRequest::new(absent, DetailLevel::Standard),
                    NOW_MS,
                )
                .expect_err("unpublished terminal must fail closed");
            assert!(
                matches!(error, SliceError::Ipc(IpcError::NotFound { .. })),
                "snapshot {absent}: got {error:?}"
            );
        }

        let error = host
            .dispatch_tool(
                &ToolRequest::new(INSPECT_TEXT_TOOL, b"{}".to_vec()).with_target("t:4199"),
                NOW_MS,
                43,
            )
            .expect_err("unpublished inspect target must fail closed");
        assert!(
            matches!(error, SliceError::Ipc(IpcError::NotFound { .. })),
            "got {error:?}"
        );

        // The published id still serves, proving the miss is selective.
        let snapshot = host
            .snapshot(
                SNAPSHOT_METHOD,
                &SnapshotRequest::new(terminal_id, DetailLevel::Standard),
                NOW_MS,
            )
            .expect("published id serves");
        assert_eq!(snapshot.terminal_id, terminal_id);

        clear_live_snapshots_for_tests();
        assert_eq!(
            live_snapshot_count(),
            0,
            "back fence: this test must leave no residue"
        );
    }
}

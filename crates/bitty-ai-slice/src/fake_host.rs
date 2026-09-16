//! Deterministic host double for the BII-09 route (`FakeHost`).
//!
//! This module owns the `bitty-ai` side of the BII-09 parallel-work split
//! ([`bitty-side-integration-input.md` BII-09][bii09]): the agent kernel runs
//! against a scripted deterministic double while the `bitty` track owns the
//! live capability gateway. Swapping to the live host is mechanical:
//! [`crate::live_host::LiveBittyHost`] implements [`BittyHost`] over the real
//! `bitty-ipc` services, so only the construction site changes; callers keep
//! calling the same trait methods.
//!
//! [bii09]: https://github.com/bitty-terminal/bitty-ai-docs/blob/main/specifications/bitty-side-integration-input.md
//!
//! ## Read-only mirror of merged `bitty` shapes
//!
//! No file in the `bitty` repository is modified by this task. Shapes below
//! mirror `bitty` `main` at `be6e63c` (AI-0073 inspection point), read-only.
//! The `2cbb1fb..be6e63c` window changes `crates/bitty-ipc` `auth.rs`
//! (`ChildTokenStore::verify` token-free static reasons plus
//! `child_token_errors_are_token_free`) and the `devtools` surface;
//! existing DTO/validate/bounds/scope/wire shapes are byte-identical, so the
//! mirror needs no shape change beyond this pin reference:
//!
//! - Dispatch prefix + per-tool consent: `crates/bitty-ipc/src/tool_dispatch.rs`
//!   `ToolSpec` (`L197`), `ToolRequest` (`L271`, `validate` `L316`),
//!   `ToolOutput` (`L346`, `validate` `L362`), `ToolExecution` (`L393`,
//!   `validate` `L427`), `ToolDispatchService::dispatch` (`L564-639`),
//!   bounds `MAX_TOOL_*` (`L106-130`).
//! - Execution + `Unknown` reconcile/resolve:
//!   `crates/bitty-ipc/src/execution.rs` `EXECUTION_SCOPE` (`L155`),
//!   `ExecutionStatus` (`L161`), `EffectState` (`L220`),
//!   `ExecutionRequest` (`L444`, `validate` `L547`, `effective_stream_budget`
//!   `L526`), `RawExecutionOutput` (`L654`, `validate` `L671`),
//!   `ExecutionResult` (`L781`, `validate` `L806`), `ExecutionService::dispatch`
//!   (`L1047`), `reconcile` (`L1149`), `resolve` (`L1169`), `Unknown` agreement
//!   (`L754`), truncation (`L929`), bounds `MAX_EXEC_*` (`L102-152`).
//! - Snapshot DTO: `crates/bitty-ipc/src/snapshot.rs` `SNAPSHOT_METHOD`
//!   (`L64`), `DetailLevel` (`L93`), `ZoneKind` (`L148`), `SemanticZone`
//!   (`L203`), `SnapshotRequest` (`L235`, `validate` `L288`),
//!   `SnapshotData` (`L317`), `TerminalSnapshot` (`L338`, `validate`
//!   `L371`), bounding (`bound_snapshot` `L424`, `truncate_to_budget`
//!   `L412`), `SnapshotService::dispatch` (`L548-580`), bounds
//!   `MAX_SNAPSHOT_*` (`L68-83`).
//! - Consent + scopes: `crates/bitty-ipc/src/scope.rs`
//!   `validate_method_name` (`L308`), `required_scope_for_method` (`L380`,
//!   `terminal.snapshot` under `terminal.inspect`), `authorize_method`
//!   (`L425`), `ConsentGrant` (`L449`, `is_expired` `L465`),
//!   `ConsentLedger` (`L477`, `MAX_GRANTS` `L484`, `is_granted` `L506`,
//!   `grant` `L520`, `revoke` `L552`, `drain_expired` `L560`).
//!
//! Fidelity strategy: this file reuses the real `bitty-ipc` DTOs, bounds,
//! and `validate()` methods directly (no copied budgets, no drift). Only the
//! dispatch order is mirrored as explicit steps with the same denial classes
//! and no-partial-state guarantee; provider calls pop per-instance scripted
//! queues instead of calling `fn`-pointer tables so tests stay deterministic
//! without globals.
//!
//! ## Determinism and scope
//!
//! - Std-only, no network, no threads, no filesystem, no wall clock.
//!   Every method takes caller-supplied `now_ms`; expiry is evaluated against
//!   it, never against system time.
//! - Single-agent: one `FakeHost` serves one `client_id` with its own
//!   server-evaluated `granted` scopes and consent ledger.
//! - Live wiring is explicitly out of scope: this file contains no code that
//!   connects to a real terminal, process, PTY, socket, or network peer
//!   (no `std::net`, `std::process`, `std::fs`, async runtime, or IPC
//!   transport). The live path is [`crate::live_host::LiveBittyHost`], which
//!   implements [`BittyHost`] over the real `bitty-ipc` services with an
//!   injectable provider seam (still no socket, process, network, or secret).

use std::collections::{BTreeMap, VecDeque};

use bitty_ipc::error::IpcError;
use bitty_ipc::execution::{
    EXECUTION_SCOPE, ExecutionRequest, ExecutionResult, MAX_EXEC_CLIENT_ID_BYTES,
    MAX_TRACKED_EXECUTIONS, RawExecutionOutput,
};
use bitty_ipc::scope::{
    ConsentLedger, Scope, ScopeSet, authorize_method, required_scope_for_method,
    validate_method_name,
};
use bitty_ipc::snapshot::{
    MAX_SNAPSHOT_CWD_BYTES, MAX_SNAPSHOT_ZONES, SnapshotData, SnapshotRequest, TerminalSnapshot,
};
use bitty_ipc::tool_dispatch::{
    MAX_TOOL_CLIENT_ID_BYTES, MAX_TOOLS_PER_HOST, ToolExecution, ToolOutput, ToolRequest, ToolSpec,
};

use bitty_ai_runtime::bridge::IdentityBridge;

use crate::bridge::wire_client_id;
use crate::error::SliceError;

/// Host capability surface shared by [`FakeHost`] and the live host.
///
/// A generic consumer drives the agent kernel through this trait only, so
/// replacing the construction site (`FakeHost::new` vs
/// `LiveBittyHost::new`) is mechanical and no call site changes shape.
///
/// All methods are deterministic for a given script plus `now_ms`. Failures
/// are fail-closed with no partial state: a denied dispatch stores nothing
/// and consumes no scripted output beyond the popped entry that produced a
/// provider-level fault.
pub trait BittyHost {
    /// Authenticated client identity this host serves (single-agent).
    fn client_id(&self) -> &str;

    /// Serve one bounded snapshot read (mirrors `SnapshotService::dispatch`
    /// plus the bridge consent gate).
    ///
    /// # Errors
    ///
    /// - [`SliceError::UnsupportedHostMethod`] for unknown methods.
    /// - [`SliceError::ConsentRequired`] when the host consent ledger lacks
    ///   an active grant for the required scope at `now_ms`.
    /// - [`SliceError::Ipc`] for scope, bound, terminal-mismatch, missing
    ///   scripted-data, or DTO-validation refusals.
    fn snapshot(
        &mut self,
        method: &str,
        request: &SnapshotRequest,
        now_ms: u64,
    ) -> Result<TerminalSnapshot, SliceError>;

    /// Serve one bounded tool dispatch (mirrors
    /// `ToolDispatchService::dispatch` order).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] for validation, scope, opt-in, consent,
    /// unknown-tool, target-mismatch, bound, missing-script, or outcome
    /// refusals. No partial state is stored on denial.
    fn dispatch_tool(
        &mut self,
        request: &ToolRequest,
        now_ms: u64,
        execution_id: u64,
    ) -> Result<ToolExecution, SliceError>;

    /// Serve one bounded supervised execution (mirrors
    /// `ExecutionService::dispatch` order, stores the attributed outcome).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] for validation, id-reuse, capacity,
    /// scope, opt-in, consent, target-mismatch, bound, missing-script, or
    /// `Unknown`-agreement refusals. Refusals store nothing.
    fn execute(
        &mut self,
        request: &ExecutionRequest,
        now_ms: u64,
        execution_id: u64,
    ) -> Result<ExecutionResult, SliceError>;

    /// Return the stored outcome for `execution_id` without re-executing
    /// (the only query path for `Unknown`).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] with `NotFound` when no outcome is
    /// stored for `execution_id`.
    fn reconcile(&self, execution_id: u64) -> Result<ExecutionResult, SliceError>;

    /// Close an `Unknown` entry with a terminal outcome, preserving
    /// attribution. There is no retry primitive: re-dispatch under a tracked
    /// id fails; a new id is an explicit re-execution.
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] for unknown ids, already-terminal stored
    /// outcomes, id/attribution mismatches, `Unknown` resolutions, or
    /// DTO-validation refusals.
    fn resolve(&mut self, execution_id: u64, result: ExecutionResult) -> Result<(), SliceError>;
}

/// Scripted deterministic host double (BII-09 `FakeHost`).
///
/// Owns one client's server-evaluated scopes, the real `bitty-ipc` consent
/// ledger, a bounded tool registry, FIFO scripted provider outputs, and the
/// bounded execution store. Tests seed scripts via `push_*`, grant consent
/// via `grant_consent`, then drive the [`BittyHost`] trait.
#[derive(Debug)]
pub struct FakeHost {
    client_id: String,
    granted: ScopeSet,
    consent: ConsentLedger,
    tools: BTreeMap<String, ToolSpec>,
    tool_scripts: VecDeque<Result<ToolOutput, IpcError>>,
    snapshot_scripts: VecDeque<Result<SnapshotData, IpcError>>,
    exec_scripts: VecDeque<Result<RawExecutionOutput, IpcError>>,
    executions: BTreeMap<u64, ExecutionResult>,
}

impl FakeHost {
    /// Construct a host for `client_id` with server-evaluated `granted`
    /// scopes and empty scripts, ledger, registry, and execution store.
    ///
    /// This is the raw seam: `client_id` is taken verbatim and only
    /// length-bounded. Product callers must use [`Self::from_binding`] so the
    /// id is derived from a bound protocol principal (`AI-0065`).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] when `client_id` is empty or exceeds the
    /// host client bound (mirrors the dispatch client checks in
    /// `tool_dispatch.rs:564` and `execution.rs:1047`).
    pub fn new(client_id: impl Into<String>, granted: ScopeSet) -> Result<Self, SliceError> {
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
        Ok(Self {
            client_id,
            granted,
            consent: ConsentLedger::new(),
            tools: BTreeMap::new(),
            tool_scripts: VecDeque::new(),
            snapshot_scripts: VecDeque::new(),
            exec_scripts: VecDeque::new(),
            executions: BTreeMap::new(),
        })
    }

    /// Construct a host whose client identity is derived from the bound
    /// protocol principal of `identity` (`AI-0065`).
    ///
    /// # Errors
    ///
    /// Returns the failures of [`wire_client_id`]: an unbound identity or an
    /// over-long derived id is refused before any host state exists (no
    /// registry, ledger, script, or execution store).
    pub fn from_binding(identity: &IdentityBridge, granted: ScopeSet) -> Result<Self, SliceError> {
        Self::new(wire_client_id(identity)?, granted)
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
        self.tools.len()
    }

    /// Number of stored execution outcomes.
    #[must_use]
    pub fn execution_count(&self) -> usize {
        self.executions.len()
    }

    /// Number of queued tool provider outputs.
    #[must_use]
    pub fn pending_tool_scripts(&self) -> usize {
        self.tool_scripts.len()
    }

    /// Number of queued snapshot provider payloads.
    #[must_use]
    pub fn pending_snapshot_scripts(&self) -> usize {
        self.snapshot_scripts.len()
    }

    /// Number of queued execution provider outputs.
    #[must_use]
    pub fn pending_exec_scripts(&self) -> usize {
        self.exec_scripts.len()
    }

    /// Register one tool declaration (mirrors
    /// `ToolDispatchService::register`, `tool_dispatch.rs:512`).
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::Ipc`] when the name is already registered or
    /// the host registry is at capacity (`MAX_TOOLS_PER_HOST`, fail-closed,
    /// no silent eviction).
    pub fn register_tool(&mut self, spec: ToolSpec) -> Result<(), SliceError> {
        if self.tools.contains_key(&spec.name) {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!("duplicate tool '{}'", spec.name),
            }));
        }
        if self.tools.len() >= MAX_TOOLS_PER_HOST {
            return Err(SliceError::Ipc(IpcError::LimitExceeded {
                field: "tool registry".to_owned(),
                limit: MAX_TOOLS_PER_HOST,
                actual: self.tools.len() + 1,
            }));
        }
        self.tools.insert(spec.name.clone(), spec);
        Ok(())
    }

    /// Queue one snapshot provider payload (FIFO).
    pub fn push_snapshot_data(&mut self, data: SnapshotData) {
        self.snapshot_scripts.push_back(Ok(data));
    }

    /// Queue one snapshot provider fault (FIFO, fail-closed, no partial state).
    pub fn push_snapshot_error(&mut self, error: IpcError) {
        self.snapshot_scripts.push_back(Err(error));
    }

    /// Queue one tool provider output (FIFO).
    pub fn push_tool_output(&mut self, output: ToolOutput) {
        self.tool_scripts.push_back(Ok(output));
    }

    /// Queue one tool provider fault (FIFO, fail-closed, no partial state).
    pub fn push_tool_error(&mut self, error: IpcError) {
        self.tool_scripts.push_back(Err(error));
    }

    /// Queue one execution provider output (FIFO, may carry `Unknown`).
    pub fn push_execution_output(&mut self, output: RawExecutionOutput) {
        self.exec_scripts.push_back(Ok(output));
    }

    /// Queue one execution provider fault (FIFO, fail-closed, no partial state).
    pub fn push_execution_error(&mut self, error: IpcError) {
        self.exec_scripts.push_back(Err(error));
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
                "fake-host".to_owned(),
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

impl BittyHost for FakeHost {
    fn client_id(&self) -> &str {
        &self.client_id
    }

    fn snapshot(
        &mut self,
        method: &str,
        request: &SnapshotRequest,
        now_ms: u64,
    ) -> Result<TerminalSnapshot, SliceError> {
        // Prefix mirrors `SnapshotService::dispatch` (`snapshot.rs:548`) plus
        // the bridge consent gate (`bridge.rs:113-173`): the headless service
        // checks scope only, so the facade enforces consent like `IpcBridge`.
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
        let data = self.snapshot_scripts.pop_front().ok_or_else(|| {
            SliceError::Ipc(IpcError::NotFound {
                reason: format!("no host provider for '{method}'"),
            })
        })??;
        if data.terminal_id != request.terminal_id {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!(
                    "snapshot provider returned '{}', want '{}'",
                    data.terminal_id, request.terminal_id
                ),
            }));
        }
        let snapshot = bound_snapshot(request, data);
        snapshot.validate().map_err(SliceError::Ipc)?;
        Ok(snapshot)
    }

    fn dispatch_tool(
        &mut self,
        request: &ToolRequest,
        now_ms: u64,
        execution_id: u64,
    ) -> Result<ToolExecution, SliceError> {
        // Prefix mirrors `ToolDispatchService::dispatch`
        // (`tool_dispatch.rs:564-639`): validation, client bounds, registry
        // lookup, scope, effect opt-in, consent, provider call, target match,
        // output validation, attributed outcome. Any refusal stores nothing.
        request.validate().map_err(SliceError::Ipc)?;
        if self.client_id.is_empty() {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: "tool client_id must not be empty".to_owned(),
            }));
        }
        if self.client_id.len() > MAX_TOOL_CLIENT_ID_BYTES {
            return Err(SliceError::Ipc(IpcError::LimitExceeded {
                field: "tool client_id".to_owned(),
                limit: MAX_TOOL_CLIENT_ID_BYTES,
                actual: self.client_id.len(),
            }));
        }
        let spec = self.tools.get(&request.tool).cloned().ok_or_else(|| {
            SliceError::Ipc(IpcError::NotFound {
                reason: format!("unknown tool '{}'", request.tool),
            })
        })?;
        if !self.granted.contains(spec.required_scope) {
            return Err(SliceError::Ipc(IpcError::ScopeDenied {
                scope: spec.required_scope.as_str().to_owned(),
                action: request.tool.clone(),
            }));
        }
        if !spec.read_only && !request.allow_effects {
            return Err(SliceError::Ipc(IpcError::Denied {
                code: "EffectRequiresExplicitConsent".to_owned(),
                reason: format!(
                    "effect tool '{}' requires explicit allow_effects",
                    request.tool
                ),
            }));
        }
        if !self
            .consent
            .is_granted(&self.client_id, spec.required_scope, now_ms)
        {
            return Err(SliceError::Ipc(IpcError::Denied {
                code: "ConsentRequired".to_owned(),
                reason: format!(
                    "missing consent for '{}' on scope '{}'",
                    request.tool,
                    spec.required_scope.as_str()
                ),
            }));
        }
        let output = self.tool_scripts.pop_front().ok_or_else(|| {
            SliceError::Ipc(IpcError::NotFound {
                reason: format!("no scripted output for tool '{}'", request.tool),
            })
        })??;
        if output.target_id != request.target {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!(
                    "tool provider returned target '{}', want '{}'",
                    output.target_id.as_deref().unwrap_or("<none>"),
                    request.target.as_deref().unwrap_or("<none>")
                ),
            }));
        }
        output.validate().map_err(SliceError::Ipc)?;
        let execution = ToolExecution {
            execution_id,
            client_id: self.client_id.clone(),
            tool: request.tool.clone(),
            target: request.target.clone(),
            summary: output.summary,
            data: output.data,
            is_untrusted_surface: true,
        };
        execution.validate().map_err(SliceError::Ipc)?;
        Ok(execution)
    }

    fn execute(
        &mut self,
        request: &ExecutionRequest,
        now_ms: u64,
        execution_id: u64,
    ) -> Result<ExecutionResult, SliceError> {
        // Prefix mirrors `ExecutionService::dispatch` (`execution.rs:1047`):
        // validation, client bounds, id-reuse rejection (never blind-retry),
        // capacity, scope, effect opt-in, consent, provider call, target
        // match, stream budgeting with char-boundary truncation, `Unknown`
        // agreement, attribution, store. Refusals store nothing.
        request.validate().map_err(SliceError::Ipc)?;
        if self.client_id.is_empty() {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: "execution client_id must not be empty".to_owned(),
            }));
        }
        if self.client_id.len() > MAX_EXEC_CLIENT_ID_BYTES {
            return Err(SliceError::Ipc(IpcError::LimitExceeded {
                field: "execution client_id".to_owned(),
                limit: MAX_EXEC_CLIENT_ID_BYTES,
                actual: self.client_id.len(),
            }));
        }
        if self.executions.contains_key(&execution_id) {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!(
                    "execution id {execution_id} is already tracked (reconcile, never blind-retry)"
                ),
            }));
        }
        if self.executions.len() >= MAX_TRACKED_EXECUTIONS {
            return Err(SliceError::Ipc(IpcError::LimitExceeded {
                field: "tracked executions".to_owned(),
                limit: MAX_TRACKED_EXECUTIONS,
                actual: self.executions.len() + 1,
            }));
        }
        if !self.granted.contains(EXECUTION_SCOPE) {
            return Err(SliceError::Ipc(IpcError::ScopeDenied {
                scope: EXECUTION_SCOPE.as_str().to_owned(),
                action: "exec".to_owned(),
            }));
        }
        if !request.allow_effects {
            return Err(SliceError::Ipc(IpcError::Denied {
                code: "EffectRequiresExplicitConsent".to_owned(),
                reason: "supervised execution requires explicit allow_effects".to_owned(),
            }));
        }
        if !self
            .consent
            .is_granted(&self.client_id, EXECUTION_SCOPE, now_ms)
        {
            return Err(SliceError::Ipc(IpcError::Denied {
                code: "ConsentRequired".to_owned(),
                reason: format!(
                    "missing consent for exec on scope '{}'",
                    EXECUTION_SCOPE.as_str()
                ),
            }));
        }
        let output = self.exec_scripts.pop_front().ok_or_else(|| {
            SliceError::Ipc(IpcError::NotFound {
                reason: "no scripted execution output (backend absence)".to_owned(),
            })
        })??;
        if output.target_id != request.target {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!(
                    "execution provider returned target '{}', want '{}'",
                    output.target_id.as_deref().unwrap_or("<none>"),
                    request.target.as_deref().unwrap_or("<none>")
                ),
            }));
        }
        let budget = request.effective_stream_budget();
        let (stdout_summary, stdout_truncated) = truncate_to_budget(&output.stdout, budget);
        let (stderr_summary, stderr_truncated) = truncate_to_budget(&output.stderr, budget);
        let bounded = RawExecutionOutput {
            stdout: stdout_summary,
            stderr: stderr_summary,
            ..output
        };
        bounded.validate().map_err(SliceError::Ipc)?;
        let result = ExecutionResult {
            execution_id,
            client_id: self.client_id.clone(),
            target: request.target.clone(),
            status: bounded.status,
            exit_code: bounded.exit_code,
            stdout_summary: bounded.stdout,
            stderr_summary: bounded.stderr,
            truncated: stdout_truncated || stderr_truncated,
            evidence_refs: bounded.evidence_refs,
            effect_state: bounded.effect_state,
            is_untrusted_surface: true,
        };
        result.validate().map_err(SliceError::Ipc)?;
        self.executions.insert(execution_id, result.clone());
        Ok(result)
    }

    fn reconcile(&self, execution_id: u64) -> Result<ExecutionResult, SliceError> {
        // Mirrors `ExecutionService::reconcile` (`execution.rs:1149`): the
        // only query path for `Unknown`. Never re-executes, never mutates.
        self.executions.get(&execution_id).cloned().ok_or_else(|| {
            SliceError::Ipc(IpcError::NotFound {
                reason: format!("unknown execution id {execution_id}"),
            })
        })
    }

    fn resolve(&mut self, execution_id: u64, result: ExecutionResult) -> Result<(), SliceError> {
        // Mirrors `ExecutionService::resolve` (`execution.rs:1169`):
        // `Unknown` closes only to a terminal outcome preserving
        // attribution; terminal entries never overwrite; resolutions are
        // never themselves `Unknown`.
        let stored = self.executions.get(&execution_id).cloned().ok_or_else(|| {
            SliceError::Ipc(IpcError::NotFound {
                reason: format!("unknown execution id {execution_id}"),
            })
        })?;
        if !stored.needs_reconciliation() {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!("execution id {execution_id} is already terminal"),
            }));
        }
        if result.execution_id != execution_id {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: format!(
                    "resolution carries id {}, want {execution_id}",
                    result.execution_id
                ),
            }));
        }
        if result.client_id != stored.client_id || result.target != stored.target {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: "resolution must preserve client_id and target".to_owned(),
            }));
        }
        if result.needs_reconciliation() {
            return Err(SliceError::Ipc(IpcError::InvalidRequest {
                reason: "resolution must be terminal, not Unknown".to_owned(),
            }));
        }
        result.validate().map_err(SliceError::Ipc)?;
        self.executions.insert(execution_id, result);
        Ok(())
    }
}

/// Truncate `text` to `budget` bytes at a char boundary (mirrors
/// `snapshot.rs:412` and `execution.rs:929`: zero budget yields empty text
/// with `truncated` set when the input is non-empty).
fn truncate_to_budget(text: &str, budget: usize) -> (String, bool) {
    if text.len() <= budget {
        return (text.to_owned(), false);
    }
    let mut end = budget;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_owned(), true)
}

/// Apply request budgets to raw provider data (mirrors `bound_snapshot`,
/// `snapshot.rs:424-452`: effective budget is `min(detail, caller ceiling)`,
/// text and `cwd` truncate at char boundaries, zones keep newest
/// `MAX_SNAPSHOT_ZONES` then narrow to the requested zone, `truncated`
/// covers text, `cwd`, or zone overflow, trust label always set).
fn bound_snapshot(request: &SnapshotRequest, data: SnapshotData) -> TerminalSnapshot {
    let budget = request.effective_budget();
    let (text, text_truncated) = truncate_to_budget(&data.text, budget);
    let (cwd, cwd_truncated) = truncate_to_budget(&data.cwd, MAX_SNAPSHOT_CWD_BYTES);
    let zones_truncated = data.semantic_zones.len() > MAX_SNAPSHOT_ZONES;
    let semantic_zones = if zones_truncated {
        data.semantic_zones[data.semantic_zones.len() - MAX_SNAPSHOT_ZONES..].to_vec()
    } else {
        data.semantic_zones
    };
    let semantic_zones = match request.zone {
        Some(kind) => semantic_zones
            .into_iter()
            .filter(|zone| zone.kind == kind)
            .collect(),
        None => semantic_zones,
    };
    TerminalSnapshot {
        terminal_id: request.terminal_id.clone(),
        generation: data.generation,
        cwd,
        semantic_zones,
        text,
        truncated: text_truncated || cwd_truncated || zones_truncated,
        is_untrusted_surface: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_ipc::execution::{EffectState, ExecutionStatus};
    use bitty_ipc::snapshot::{DetailLevel, SNAPSHOT_METHOD, SemanticZone, ZoneKind};

    const NOW_MS: u64 = 1_000;
    const TTL_MS: u64 = 60_000;

    fn granted_inspect() -> ScopeSet {
        ScopeSet::single(Scope::TerminalInspect)
    }

    fn granted_spawn() -> ScopeSet {
        ScopeSet::single(Scope::ProcessSpawn)
    }

    fn host_with_inspect() -> FakeHost {
        let mut host =
            FakeHost::new("agent-fakehost", granted_inspect()).expect("host construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");
        host
    }

    fn snapshot_request() -> SnapshotRequest {
        SnapshotRequest::new("t:1", DetailLevel::Standard)
    }

    fn snapshot_data(text: &str) -> SnapshotData {
        SnapshotData {
            terminal_id: "t:1".to_owned(),
            generation: 7,
            cwd: "/work".to_owned(),
            semantic_zones: vec![SemanticZone {
                kind: ZoneKind::Output,
                line_start: 0,
                line_end: 3,
            }],
            text: text.to_owned(),
        }
    }

    #[test]
    fn snapshot_returns_bounded_dto_with_trust_label() {
        let mut host = host_with_inspect();
        host.push_snapshot_data(snapshot_data("hello"));
        let snapshot = host
            .snapshot(SNAPSHOT_METHOD, &snapshot_request(), NOW_MS)
            .expect("snapshot serves");
        assert_eq!(snapshot.terminal_id, "t:1");
        assert_eq!(snapshot.generation, 7);
        assert_eq!(snapshot.text, "hello");
        assert!(!snapshot.truncated);
        assert!(snapshot.is_untrusted_surface);
        assert!(snapshot.is_untrusted_surface());
        snapshot.validate().expect("dto validates");
    }

    #[test]
    fn snapshot_truncates_at_char_boundary_and_flags() {
        let mut host = host_with_inspect();
        let request = SnapshotRequest::new("t:1", DetailLevel::Minimal).with_max_bytes(7);
        host.push_snapshot_data(snapshot_data("ééééé"));
        let snapshot = host
            .snapshot(SNAPSHOT_METHOD, &request, NOW_MS)
            .expect("snapshot serves");
        assert!(snapshot.truncated);
        assert!(snapshot.text.len() <= 7);
        assert!("ééééé".starts_with(snapshot.text.as_str()));
    }

    #[test]
    fn dispatch_serves_with_attribution_and_trust_label() {
        let mut host = FakeHost::new("agent-fakehost", ScopeSet::single(Scope::TerminalInspect))
            .expect("host construction");
        host.grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");
        let spec = ToolSpec::new(
            "terminal_read_zone",
            "read-only zone read",
            br#"{"type":"object"}"#.to_vec(),
            Scope::TerminalInspect,
            true,
        )
        .expect("spec");
        host.register_tool(spec).expect("register");
        host.push_tool_output(ToolOutput {
            target_id: None,
            data: b"zone bytes".to_vec(),
            summary: "ok".to_owned(),
        });
        let request = ToolRequest::new("terminal_read_zone", br#"{"zone":"output"}"#.to_vec());
        let execution = host
            .dispatch_tool(&request, NOW_MS, 41)
            .expect("dispatch serves");
        assert_eq!(execution.execution_id, 41);
        assert_eq!(execution.client_id, "agent-fakehost");
        assert_eq!(execution.tool, "terminal_read_zone");
        assert!(execution.is_untrusted_surface);
        execution.validate().expect("outcome validates");
    }

    #[test]
    fn execution_unknown_reconciles_and_resolves() {
        let mut host = FakeHost::new("agent-fakehost", granted_spawn()).expect("host construction");
        host.grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
            .expect("consent grant");
        host.push_execution_output(RawExecutionOutput {
            target_id: None,
            status: ExecutionStatus::Unknown,
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            evidence_refs: Vec::new(),
            effect_state: EffectState::Unknown,
        });
        let request =
            ExecutionRequest::new("git", vec!["diff".to_owned()]).with_allow_effects(true);
        let result = host.execute(&request, NOW_MS, 7).expect("unknown stores");
        assert!(result.needs_reconciliation());
        let reconciled = host.reconcile(7).expect("reconcile queries");
        assert_eq!(reconciled, result);
        let terminal = ExecutionResult::new(
            7,
            "agent-fakehost".to_owned(),
            None,
            ExecutionStatus::Completed,
            Some(0),
            String::new(),
            String::new(),
            false,
            Vec::new(),
            EffectState::Completed,
        )
        .expect("terminal result");
        host.resolve(7, terminal.clone())
            .expect("resolve closes unknown");
        assert_eq!(host.reconcile(7).expect("reconciled"), terminal);
    }
}

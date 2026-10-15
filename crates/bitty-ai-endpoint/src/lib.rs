//! Helper-side `ai.*` handler endpoint skeleton (`bitty-ai-endpoint`).
//!
//! **Status: E0 skeleton of program AI-0302 (Transport S-E).** This crate is
//! the shipped helper-side `ai.*` handler endpoint in skeleton form: it runs
//! the six-gate S-E dispatcher contract over caller-supplied inputs and stops
//! at the admit point. Per-method handler wiring is E1, not here.
//!
//! ## S-E handler contract
//!
//! Every `ai.*` call passes six gates in order, refusing at the first
//! failure without running later gates or contacting a handler (mirrors the
//! S-A `gate_pipeline` ordering test in `bitty-ipc`, read-only reference):
//!
//! 1. Method resolution: unknown methods refuse `NotFound` before any scope
//!    or consent state is consulted (fail-closed).
//! 2. Scope authorization: known methods without a server-derived grant
//!    refuse `ScopeDenied` before the consent ledger is consulted.
//! 3. Per-op consent: a missing, revoked, or expired ledger grant refuses
//!    `ScopeDenied` before deadline or JSON work.
//! 4. Deadline: an expired caller-supplied deadline refuses `Timeout`
//!    (no-exec-after-timeout) before JSON validation.
//! 5. Params JSON: malformed or schema-violating params refuse with a
//!    validation error and no handler contact.
//! 6. Admit: only a call that passes every gate reaches the admit point.
//!
//! [`admit`] runs gates 1-6 and returns the typed
//! [`DispatchPosture::Admitted`] value. [`dispatch`] is the E0 entry shape:
//! it admits, then returns the single [`DispatchOutcome::NotWired`] typed
//! refusal for every admitted call until E1 wires per-method handlers.
//!
//! ## Agnostic-`LocalEndpoint`: no routing choice
//!
//! S-E stays agnostic on the `LocalEndpoint` route question: this crate makes
//! no provider-selection, transport, or endpoint-kind decision. Routing stays
//! behind a `ModelProvider`-injection seam with kind-as-data (E1c scope); no
//! `local_provider` module moves here and no helper-side route is hardcoded.
//! Stop-and-escalate condition: if E1 wiring finds the agnostic posture
//! impossible (a handler cannot be written without choosing a route), stop
//! and escalate to the owning task instead of hardcoding one.
//!
//! ## Bytes in, bytes out only
//!
//! The resident-singleton helper *process* (spawn, digest, stdio lifecycle)
//! is owned terminal-side per Core DIR-030 (`bitty` #2123, separate
//! cross-repo work): this crate never spawns, owns no socket, frame, clock,
//! or thread, and keeps no transport state. An `IpcRequest` comes in as
//! caller-supplied fields ([`DispatchInput`]); a typed posture goes out.
//! Time comes from caller-supplied `now_ms`, never a wall clock.
//!
//! ## Roadmap pointers
//!
//! - E1a: read-only wiring (`model_select`, fallback, cost, `provider_list`,
//!   `cache_hint`, `mcp_pin`, `mcp_snapshot`, extension echoes).
//! - E1b: `context_read` wiring.
//! - E1c: `provider_complete` plus the injection seam.
//! - E1d: `tool_dispatch` digest-only plus host per-tool consent on top.
//! - E2: params-schema plug-in plus the tolerant S-1 correlator field.
//! - E3: loopback admit/deny matrix, reassembler TTL, forged-tier vectors.
//! - S-F (separate follow-up, deferred): correlator enforcement and the full
//!   host-boundary lint gate re-establishment. No producer emits snapshot
//!   versions yet, so enforcing now would deny everything.
//!
//! ## Normative precedence
//!
//! On any conflict between this skeleton and the accepted
//! `ipc-agent-rfc` (`bitty-ai-docs`, `specifications/ipc-agent-rfc.md`), the
//! RFC wins and this crate must be corrected. This crate introduces no new
//! trust boundary and weakens no normative control.

#![deny(unsafe_code)]

use bitty_ipc_api::{
    ConsentLedger, IpcError, Scope, ScopeSet, authorize_method, domain_deadline_ms,
    validate_ai_params, validate_domain_timeout,
};

// ── inputs ──────────────────────────────────────────────────────────────────

/// Caller-supplied inputs for one dispatch decision.
///
/// Everything the six gates need arrives here as values: no socket, no
/// endpoint handle, no clock read. `now_ms`, `created_at_ms`, and
/// `timeout_ms` are caller-supplied milliseconds on the same deterministic
/// clock the `bitty-ipc-api` deadline helpers assume.
#[derive(Debug, Clone, Copy)]
pub struct DispatchInput<'a> {
    /// Method name (for example `ai.model_select`).
    pub method: &'a str,
    /// Caller correlation id, echoed in the `Timeout` refusal when gate 4 fires.
    pub request_id: u64,
    /// Server-derived granted scopes for the authenticated caller.
    pub granted: &'a ScopeSet,
    /// Read-only consent ledger consulted at gate 3.
    pub ledger: &'a ConsentLedger,
    /// Authenticated client identity for the ledger lookup.
    pub client_id: &'a str,
    /// Caller `now_ms` for consent liveness and deadline expiry.
    pub now_ms: u64,
    /// Caller `created_at_ms` anchoring the absolute deadline.
    pub created_at_ms: u64,
    /// Caller `timeout_ms`, range-checked then added to `created_at_ms`.
    pub timeout_ms: u64,
    /// Raw params bytes validated at gate 5.
    pub params: &'a [u8],
}

// ── outcomes ────────────────────────────────────────────────────────────────

/// Admit-point posture: what the six gates decided.
///
/// The E0 skeleton stops here. E1 per-method wiring owns what happens after
/// admission; until then [`dispatch`] maps this to the `NotWired` refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPosture {
    /// All six gates passed for a known `ai.*` method.
    Admitted {
        /// Scope the call was admitted under (per-method assignment).
        scope: Scope,
    },
}

/// E0 terminal outcome of [`dispatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// A gate refused before admission; carries the gate's typed error.
    Refused(IpcError),
    /// Admitted, but no per-method handler is wired yet.
    ///
    /// // POLICY (AI-0302-E0): exactly one variant for every admitted call,
    /// regardless of method. Per-method handler wiring is E1, not here, so
    /// the skeleton must not branch on method past admission.
    NotWired {
        /// Method that was admitted.
        method: String,
        /// Scope the call was admitted under.
        scope: Scope,
    },
}

// ── gate probes (test diagnostics) ──────────────────────────────────────────

/// Records which gates ran for one dispatch.
///
/// Test and diagnostic support only: each flag is set when its gate runs, so
/// order tests can prove an early refusal short-circuits later gates. The
/// public [`admit`] and [`dispatch`] entries use a throwaway probe.
#[derive(Debug, Default)]
struct GateProbe {
    consent_consulted: std::cell::Cell<bool>,
    deadline_checked: std::cell::Cell<bool>,
    json_validated: std::cell::Cell<bool>,
    admit_reached: std::cell::Cell<bool>,
}

// ── dispatcher ──────────────────────────────────────────────────────────────

/// Run gates 1-6 over `input`, returning the admitted scope.
///
/// Gates 1-2 run as one [`authorize_method`] call: resolution refuses
/// `NotFound` before the grant check can run. Each later gate sets its probe
/// flag when it runs and returns its refusal early.
fn run_gates(input: &DispatchInput<'_>, probe: &GateProbe) -> Result<Scope, IpcError> {
    // POLICY (AI-0302-E0): gates 1-2 stay fused in `authorize_method`, the
    // same composition the S-A gate_pipeline reference uses. Splitting
    // resolution from authorization would let a scope check observe an
    // unregistered name.
    let required = authorize_method(input.method, input.granted)?;
    // POLICY (AI-0302-E0): gate 3 consults the read-only S-G inspection
    // (`ConsentLedger::inspect`), never a mutating ledger call. Liveness is
    // evaluated by the caller from the inspected record, so this gate
    // grants, extends, refreshes, drains, nor evicts nothing.
    probe.consent_consulted.set(true);
    let live = input
        .ledger
        .inspect(input.client_id, required)
        .is_some_and(|grant| !grant.is_expired(input.now_ms));
    if !live {
        return Err(IpcError::ScopeDenied {
            scope: required.as_str().into(),
            action: input.method.into(),
        });
    }
    // POLICY (AI-0302-E0): gate 4 checks the construction range, then expiry
    // from caller-supplied `now_ms` only. No `IpcEndpoint` is owned here
    // (that would be transport state); the expiry comparison mirrors
    // `drain_expired` semantics (expired when `now >= deadline`) without
    // queueing anything.
    probe.deadline_checked.set(true);
    validate_domain_timeout(input.timeout_ms)?;
    if input.now_ms >= domain_deadline_ms(input.created_at_ms, input.timeout_ms) {
        return Err(IpcError::Timeout {
            request_id: input.request_id,
            timeout_ms: input.timeout_ms,
        });
    }
    // POLICY (AI-0302-E0): gate 5 runs the per-method schema validator,
    // which layers the canonical params-JSON rules underneath (frame bound,
    // UTF-8, depth, no ambient authority) before the field schema. One call
    // keeps the two from drifting apart.
    probe.json_validated.set(true);
    validate_ai_params(input.method, input.params)?;
    // POLICY (AI-0302-E0): gate 6 is the admit point. Reaching it sets the
    // flag and returns the scope; handler contact stays E1-owned.
    probe.admit_reached.set(true);
    Ok(required)
}

/// Run the six S-E gates and stop at the admit point.
///
/// Returns the typed [`DispatchPosture::Admitted`] value when every gate
/// passes, or the refusing gate's [`IpcError`] otherwise. Never panics on
/// adversarial input: every refusal is a typed value.
pub fn admit(input: &DispatchInput<'_>) -> Result<DispatchPosture, IpcError> {
    let probe = GateProbe::default();
    run_gates(input, &probe).map(|scope| DispatchPosture::Admitted { scope })
}

/// E0 dispatch entry: admit, then refuse as not-wired.
///
/// Lawful traffic that passes all six gates receives the single
/// [`DispatchOutcome::NotWired`] typed refusal until E1 wires per-method
/// handlers. Gate refusals pass through unchanged as
/// [`DispatchOutcome::Refused`].
pub fn dispatch(input: &DispatchInput<'_>) -> DispatchOutcome {
    let probe = GateProbe::default();
    match run_gates(input, &probe) {
        // POLICY (AI-0302-E0): the admitted arm carries no method branch.
        // Adding per-method behavior here would be E1 scope creep.
        Ok(scope) => DispatchOutcome::NotWired {
            method: input.method.to_string(),
            scope,
        },
        Err(error) => DispatchOutcome::Refused(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLIENT: &str = "agent-7";
    const INSPECT_METHOD: &str = "ai.model_select";
    const INVOKE_METHOD: &str = "ai.tool_dispatch";
    const LIST_METHOD: &str = "ai.provider_list";
    const UNKNOWN_METHOD: &str = "ai.evil";
    const MALFORMED_PARAMS: &[u8] = br#"{"scope":"ai.invoke"}"#;

    fn input<'a>(
        method: &'a str,
        granted: &'a ScopeSet,
        ledger: &'a ConsentLedger,
        now_ms: u64,
        created_at_ms: u64,
        timeout_ms: u64,
        params: &'a [u8],
    ) -> DispatchInput<'a> {
        DispatchInput {
            method,
            request_id: 7,
            granted,
            ledger,
            client_id: CLIENT,
            now_ms,
            created_at_ms,
            timeout_ms,
            params,
        }
    }

    /// Ledger with one live grant for `(CLIENT, scope)` covering `now_ms`.
    fn ledger_with_live_grant(scope: Scope, now_ms: u64) -> ConsentLedger {
        let mut ledger = ConsentLedger::new();
        ledger
            .grant(CLIENT.to_owned(), scope, 0, 10_000, "user".to_owned())
            .expect("grant fits in an empty ledger");
        assert!(ledger.is_granted(CLIENT, scope, now_ms));
        ledger
    }

    fn assert_no_gate_ran(probe: &GateProbe) {
        assert!(!probe.consent_consulted.get());
        assert!(!probe.deadline_checked.get());
        assert!(!probe.json_validated.get());
        assert!(!probe.admit_reached.get());
    }

    #[test]
    fn unknown_method_refuses_not_found_first() {
        // POLICY (AI-0302-E0): grammar-valid but unregistered. Resolution
        // fails before any grant check, so even a poisoned later pipeline
        // (expired deadline, malformed params) cannot change the error.
        let granted = ScopeSet::all();
        let ledger = ledger_with_live_grant(Scope::AiInspect, 500);
        let probe = GateProbe::default();
        let scoped = input(
            UNKNOWN_METHOD,
            &granted,
            &ledger,
            1_000,
            0,
            1_000,
            MALFORMED_PARAMS,
        );
        let err = run_gates(&scoped, &probe).unwrap_err();
        assert!(
            matches!(&err, IpcError::NotFound { .. }),
            "unexpected: {err:?}"
        );
        assert_eq!(err.error_class(), bitty_ipc_api::ErrorClass::NotFound);
        assert_no_gate_ran(&probe);
    }

    #[test]
    fn missing_grant_refuses_scope_denied_before_consent() {
        // POLICY (AI-0302-E0): the CLI default carries no ai scope. The
        // ledger would allow, but the scope gate denies first without
        // consulting it.
        let granted = ScopeSet::cli_default();
        assert!(!granted.contains(Scope::AiInspect));
        let ledger = ledger_with_live_grant(Scope::AiInspect, 500);
        let probe = GateProbe::default();
        let scoped = input(
            INSPECT_METHOD,
            &granted,
            &ledger,
            1_000,
            0,
            1_000,
            MALFORMED_PARAMS,
        );
        let err = run_gates(&scoped, &probe).unwrap_err();
        assert!(
            matches!(
                &err,
                IpcError::ScopeDenied { scope, action }
                if scope == "ai.inspect" && action == INSPECT_METHOD
            ),
            "unexpected: {err:?}"
        );
        assert_eq!(err.error_class(), bitty_ipc_api::ErrorClass::Scope);
        assert_no_gate_ran(&probe);
    }

    #[test]
    fn consent_denial_refuses_before_deadline_and_json() {
        // POLICY (AI-0302-E0): never-granted, expired, and revoked ledgers
        // each deny at gate 3 while the poisoned deadline (at expiry) and
        // malformed params would also fail if those gates ran.
        let granted = ScopeSet::single(Scope::AiInspect);
        let never = ConsentLedger::new();
        let mut expired = ConsentLedger::new();
        expired
            .grant(
                CLIENT.to_owned(),
                Scope::AiInspect,
                0,
                100,
                "user".to_owned(),
            )
            .unwrap();
        let mut revoked = ConsentLedger::new();
        revoked
            .grant(
                CLIENT.to_owned(),
                Scope::AiInspect,
                0,
                10_000,
                "user".to_owned(),
            )
            .unwrap();
        assert!(revoked.revoke(CLIENT, Scope::AiInspect));
        let cases: [(&str, ConsentLedger, u64); 3] = [
            ("never-granted", never, 1_000),
            ("expired", expired, 1_000),
            ("revoked", revoked, 1_000),
        ];
        for (label, ledger, now_ms) in &cases {
            let probe = GateProbe::default();
            let scoped = input(
                INSPECT_METHOD,
                &granted,
                ledger,
                *now_ms,
                0,
                1_000,
                MALFORMED_PARAMS,
            );
            let err = run_gates(&scoped, &probe).unwrap_err();
            assert!(
                matches!(
                    &err,
                    IpcError::ScopeDenied { scope, .. } if scope == "ai.inspect"
                ),
                "{label}: unexpected {err:?}"
            );
            assert_eq!(
                err.error_class(),
                bitty_ipc_api::ErrorClass::Scope,
                "{label}"
            );
            assert!(probe.consent_consulted.get(), "{label}");
            assert!(!probe.deadline_checked.get(), "{label}");
            assert!(!probe.json_validated.get(), "{label}");
            assert!(!probe.admit_reached.get(), "{label}");
        }
    }

    #[test]
    fn expired_deadline_refuses_timeout_before_json() {
        // POLICY (AI-0302-E0): caller-now at the absolute deadline refuses
        // `Timeout` even though the params would also fail JSON validation.
        let granted = ScopeSet::single(Scope::AiInspect);
        let ledger = ledger_with_live_grant(Scope::AiInspect, 500);
        let probe = GateProbe::default();
        let scoped = input(
            INSPECT_METHOD,
            &granted,
            &ledger,
            1_000,
            0,
            1_000,
            MALFORMED_PARAMS,
        );
        let err = run_gates(&scoped, &probe).unwrap_err();
        assert!(
            matches!(
                &err,
                IpcError::Timeout {
                    request_id: 7,
                    timeout_ms: 1_000
                }
            ),
            "unexpected: {err:?}"
        );
        assert_eq!(err.error_class(), bitty_ipc_api::ErrorClass::Timeout);
        assert!(probe.consent_consulted.get());
        assert!(probe.deadline_checked.get());
        assert!(!probe.json_validated.get());
        assert!(!probe.admit_reached.get());
    }

    #[test]
    fn oversized_timeout_refuses_invalid_request_before_expiry() {
        // POLICY (AI-0302-E0): the construction-range arm of gate 4 fires
        // before any expiry comparison: `0` and past-ceiling timeouts are
        // `InvalidRequest`, never queued, never compared.
        let granted = ScopeSet::single(Scope::AiInspect);
        let ledger = ledger_with_live_grant(Scope::AiInspect, 1_000);
        for timeout_ms in [0, 30_001] {
            let probe = GateProbe::default();
            let scoped = input(
                INSPECT_METHOD,
                &granted,
                &ledger,
                1_000,
                0,
                timeout_ms,
                b"{}",
            );
            let err = run_gates(&scoped, &probe).unwrap_err();
            assert!(
                matches!(&err, IpcError::InvalidRequest { .. }),
                "timeout {timeout_ms}: unexpected {err:?}"
            );
            assert!(probe.consent_consulted.get());
            assert!(probe.deadline_checked.get());
            assert!(!probe.json_validated.get());
            assert!(!probe.admit_reached.get());
        }
    }

    #[test]
    fn malformed_params_refuse_without_admit() {
        // POLICY (AI-0302-E0): live consent and live deadline throughout, so
        // gate 5 owns each refusal and the admit point is never reached.
        let granted = ScopeSet::single(Scope::AiInspect);
        let ledger = ledger_with_live_grant(Scope::AiInspect, 1_000);
        let non_utf8: &[u8] = &[0x7b, 0x22, 0xff, 0x22, 0x7d];
        let cases: [(&str, &[u8], bitty_ipc_api::ErrorClass); 3] = [
            (
                "ambient-authority",
                MALFORMED_PARAMS,
                bitty_ipc_api::ErrorClass::Validation,
            ),
            ("non-utf8", non_utf8, bitty_ipc_api::ErrorClass::Framing),
            (
                "unknown-field",
                br#"{"frobnicate":1}"#,
                bitty_ipc_api::ErrorClass::Validation,
            ),
        ];
        for (label, params, class) in cases {
            let probe = GateProbe::default();
            let scoped = input(LIST_METHOD, &granted, &ledger, 1_000, 0, 5_000, params);
            let err = run_gates(&scoped, &probe).unwrap_err();
            assert!(
                matches!(
                    &err,
                    IpcError::InvalidRequest { .. } | IpcError::InvalidFrame { .. }
                ),
                "{label}: unexpected {err:?}"
            );
            assert_eq!(err.error_class(), class, "{label}");
            assert!(probe.consent_consulted.get(), "{label}");
            assert!(probe.deadline_checked.get(), "{label}");
            assert!(probe.json_validated.get(), "{label}");
            assert!(!probe.admit_reached.get(), "{label}");
        }
    }

    #[test]
    fn fully_authorized_request_reaches_admit() {
        // POLICY (AI-0302-E0): lawful traffic in both ai families passes
        // every gate: exact grant, live consent, live deadline, valid
        // params. The unknown-field vector for `ai.provider_list` above
        // proves gate 5 runs the per-method schema, so these vectors prove
        // the admit point is reachable, not vacuous.
        let tool_params =
            br#"{"tool":"t","args_digest":1,"caller":{"agent_instance_id":1,"session_id":2,"level":"inspect"},"now_ms":3}"#;
        let cases: [(&str, Scope, &[u8]); 3] = [
            (LIST_METHOD, Scope::AiInspect, b"{}"),
            (
                INSPECT_METHOD,
                Scope::AiInspect,
                br#"{"required":["text"]}"#,
            ),
            (INVOKE_METHOD, Scope::AiInvoke, tool_params),
        ];
        for (method, scope, params) in cases {
            let granted = ScopeSet::single(scope);
            let ledger = ledger_with_live_grant(scope, 1_000);
            let probe = GateProbe::default();
            let scoped = input(method, &granted, &ledger, 1_000, 0, 5_000, params);
            let out = run_gates(&scoped, &probe);
            assert_eq!(out, Ok(scope), "{method}");
            assert!(probe.consent_consulted.get(), "{method}");
            assert!(probe.deadline_checked.get(), "{method}");
            assert!(probe.json_validated.get(), "{method}");
            assert!(probe.admit_reached.get(), "{method}");
            assert_eq!(
                admit(&scoped),
                Ok(DispatchPosture::Admitted { scope }),
                "{method}"
            );
        }
    }

    #[test]
    fn admitted_dispatch_returns_not_wired() {
        // POLICY (AI-0302-E0): E0 admits, then refuses as not-wired. One
        // variant for every admitted call: inspect and invoke methods share
        // it, and gate refusals pass through unchanged.
        let granted = ScopeSet::single(Scope::AiInspect);
        let ledger = ledger_with_live_grant(Scope::AiInspect, 1_000);
        let lawful = input(LIST_METHOD, &granted, &ledger, 1_000, 0, 5_000, b"{}");
        assert_eq!(
            dispatch(&lawful),
            DispatchOutcome::NotWired {
                method: LIST_METHOD.to_string(),
                scope: Scope::AiInspect,
            }
        );
        let refused = input(UNKNOWN_METHOD, &granted, &ledger, 1_000, 0, 5_000, b"{}");
        assert!(matches!(
            dispatch(&refused),
            DispatchOutcome::Refused(IpcError::NotFound { .. })
        ));
    }

    // ── day-one boundary coverage ───────────────────────────────────────────

    /// Host manifest text, read at compile time.
    ///
    /// // POLICY (AI-0302-E0): `include_str!` keeps this hermetic (no
    /// process spawn, no filesystem read at test time). The workspace-
    /// relative path is load-bearing: moving either crate must update it,
    /// which is the point — silent relocation must break loudly.
    const HOST_MANIFEST: &str = include_str!("../../bitty-ai-host/Cargo.toml");

    /// Endpoint manifest text, read at compile time (pin guard below).
    const ENDPOINT_MANIFEST: &str = include_str!("../Cargo.toml");

    #[test]
    fn host_manifest_declares_zero_dependencies() {
        // POLICY (AI-0302-E0): day-one minimal boundary coverage per
        // PX-1298. `bitty-ai-host` must stay dependency-free so the
        // endpoint -> host edge cannot smuggle a second IPC path; the full
        // host-boundary lint-gate re-establishment is a separate follow-up,
        // recorded, not waived.
        assert!(
            HOST_MANIFEST.contains("[dependencies]"),
            "host manifest must keep its [dependencies] section"
        );
        let mut in_deps = false;
        for line in HOST_MANIFEST.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_deps = trimmed == "[dependencies]";
                continue;
            }
            if in_deps {
                assert!(
                    trimmed.is_empty() || trimmed.starts_with('#'),
                    "host manifest must stay dependency-free, found: {trimmed}"
                );
            }
        }
    }

    #[test]
    fn endpoint_ipc_pin_stays_at_owner_decision() {
        // POLICY (AI-0302-E0): the `bitty-ipc-api` git pin is the owner
        // decision in PX-1298 (`d0c4ecd`, 7-char per workspace git-pin
        // policy). A stale-pin bump is a separate follow-up, never silent
        // drift inside S-E.
        assert!(
            ENDPOINT_MANIFEST.contains(r#"rev = "d0c4ecd""#),
            "endpoint manifest must pin bitty-ipc-api at rev d0c4ecd"
        );
    }
}

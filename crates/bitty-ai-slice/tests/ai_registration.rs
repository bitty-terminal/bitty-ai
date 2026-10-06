//! CTX-0916 S6 registration acceptance (AI-0172).
//!
//! `bitty-ai`-side counterpart of the Core cutover: the `agent` / `mcp` /
//! `ai` families plus their seven heads, the Commander/Implementer ceiling
//! rows, and the provider-credential adapter removed from the Core seed must
//! all have a verified home here. Core-only rejection stays on the `bitty`
//! side (`core_seed_rejects_ai_families_fail_closed` and the ceiling pins at
//! `271d662b`); these tests prove the ai-loaded side restores AI authority
//! additively with Core defaults unchanged.
//!
//! Deterministic: no wall clock, no network, no filesystem; the credential
//! command runner is exercised through missing-program and refusal paths
//! only, never a live spawn.

use std::collections::BTreeMap;

use bitty_ai_runtime::bridge::{IdentityBridge, ProtocolAgentId};
use bitty_ai_runtime::session::AgentInstanceId;
use bitty_ai_slice::{
    AI_FAMILY_CONTRIBUTIONS, AI_ROLE_CEILINGS, LiveBittyHost, ProviderCredentialConfig,
    check_provider_override, execute_credential_cmd, register_ai_capabilities,
    register_ai_ceilings, register_ai_families, resolve_provider_credential,
};
use bitty_ipc::error::IpcError;
use bitty_ipc::execution::{EffectState, ExecutionRequest, ExecutionStatus, RawExecutionOutput};
use bitty_ipc::scope::{Scope, ScopeSet};
use bitty_package::CapabilityCatalog;
use bitty_package::manifest::CapabilityId;
use bitty_plugin_host::{AgentRole, CredentialRef, RoleCeilingCatalog};

// ── helpers ──────────────────────────────────────────────────────────────────

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

fn bound_identity() -> IdentityBridge {
    let mut identity = IdentityBridge::new();
    identity
        .bind(
            ProtocolAgentId::new("owner.name").expect("principal validates"),
            AgentInstanceId(1),
        )
        .expect("bind succeeds");
    identity
}

// ── test 1: families register and validate with exact polarity ───────────────

#[test]
fn ai_families_register_and_validate() {
    // Core-only seed knows none of the AI families or heads (the Core-side
    // matrix pins the rejection; here the absence is the precondition).
    let core = CapabilityCatalog::core();
    for family in ["ai", "mcp", "agent"] {
        assert!(
            !core.contains_family(family),
            "Core seed must not carry family '{family}'"
        );
    }
    for head in [
        "ai.provider",
        "ai.stream",
        "ai.model",
        "mcp.invoke",
        "agent.context.terminal",
        "agent.context.workspace",
        "agent.memory",
    ] {
        assert!(
            !core.contains_head(head),
            "Core seed must not carry head '{head}'"
        );
        assert!(
            CapabilityId::new(head).is_err(),
            "static path must reject '{head}'"
        );
    }

    // Register: three families, seven heads, exact pre-extraction polarity.
    let mut catalog = CapabilityCatalog::core();
    register_ai_capabilities(&mut catalog).expect("fixed table registers");
    for family in ["ai", "mcp", "agent"] {
        assert!(catalog.contains_family(family));
    }
    for (head, requires_param) in [
        ("ai.provider", false),
        ("ai.stream", false),
        ("ai.model", false),
        ("mcp.invoke", true),
        ("agent.context.terminal", false),
        ("agent.context.workspace", false),
        ("agent.memory", true),
    ] {
        assert!(catalog.contains_head(head));
        assert_eq!(
            catalog.requires_param(head),
            Some(requires_param),
            "polarity diverged for '{head}'"
        );
    }

    // The contribution table itself pins the counterpart shape and the
    // verbatim pre-extraction effect wordings (PX-5010).
    assert_eq!(AI_FAMILY_CONTRIBUTIONS.len(), 3);
    let heads: Vec<(&str, bool, &str)> = AI_FAMILY_CONTRIBUTIONS
        .iter()
        .flat_map(|family| {
            family
                .heads
                .iter()
                .map(|head| (head.head, head.requires_param, head.effect))
        })
        .collect();
    assert_eq!(
        heads,
        [
            ("ai.provider", false, "Use allowlisted AI provider"),
            ("ai.stream", false, "Stream AI responses for this agent"),
            (
                "ai.model",
                false,
                "Select AI model for this agent (bounded)"
            ),
            (
                "mcp.invoke",
                true,
                "Invoke allowlisted MCP tool (per-tool, bounded frame 256KiB)"
            ),
            (
                "agent.context.terminal",
                false,
                "Observe terminal context for this agent (bounded 32KiB)"
            ),
            (
                "agent.context.workspace",
                false,
                "Observe workspace context for this agent (bounded 32KiB)"
            ),
            (
                "agent.memory",
                true,
                "Persist agent conversational memory (opt-in, 0600, <=7 days)"
            ),
        ]
    );

    // Extended catalog accepts all seven heads with exact polarity ...
    for raw in [
        "ai.provider",
        "ai.stream",
        "ai.model",
        "mcp.invoke:mail.list",
        "agent.context.terminal",
        "agent.context.workspace",
        "agent.memory:record-1",
    ] {
        assert!(
            CapabilityId::parse_with(&catalog, raw).is_ok(),
            "extended catalog must accept '{raw}'"
        );
    }
    // ... and denies every polarity violation plus unknown heads.
    for raw in [
        "ai.provider:extra",
        "ai.stream:extra",
        "ai.model:extra",
        "mcp.invoke",
        "agent.memory",
        "agent.context.terminal:param",
        "agent.evil",
        "ai.unknown",
    ] {
        assert!(
            CapabilityId::parse_with(&catalog, raw).is_err(),
            "extended catalog must deny '{raw}'"
        );
    }

    // Core defaults are unchanged: the static paths still fail closed ...
    assert!(CapabilityId::new("ai.provider").is_err());
    assert!(CapabilityId::new("mcp.invoke:mail.list").is_err());
    assert!(CapabilityId::new("agent.memory:record-1").is_err());
    // ... a fresh Core seed is unaffected ...
    let fresh = CapabilityCatalog::core();
    assert!(!fresh.contains_family("ai"));
    assert!(!fresh.contains_family("mcp"));
    assert!(!fresh.contains_family("agent"));
    // ... and re-registration is a duplicate, never an overwrite.
    assert!(register_ai_capabilities(&mut catalog).is_err());
    assert_eq!(catalog.requires_param("mcp.invoke"), Some(true));
}

// ── test 2: ceilings restore Commander/Implementer rows only ─────────────────

#[test]
fn ai_ceilings_contribute_commander_implementer() {
    // Core-only seed admits no AI family for any role (Core-side matrix pins
    // the rejection; here the absence is the precondition).
    let core = RoleCeilingCatalog::core();
    for role in [
        AgentRole::Commander,
        AgentRole::Implementer,
        AgentRole::Tester,
        AgentRole::Reviewer,
    ] {
        for family in ["ai", "mcp", "agent"] {
            assert!(
                !core.allows(role, family),
                "Core seed must deny {role} family '{family}'"
            );
        }
    }

    // Register: Commander regains ai/mcp/agent, Implementer regains agent.
    let mut ceilings = RoleCeilingCatalog::core();
    register_ai_ceilings(&mut ceilings).expect("fixed rows register");
    for family in ["ai", "mcp", "agent"] {
        assert!(
            ceilings.allows(AgentRole::Commander, family),
            "Commander must admit '{family}'"
        );
    }
    assert!(ceilings.allows(AgentRole::Implementer, "agent"));
    assert!(!ceilings.allows(AgentRole::Implementer, "ai"));
    assert!(!ceilings.allows(AgentRole::Implementer, "mcp"));

    // No leak: Tester and Reviewer admit no AI family.
    for role in [AgentRole::Tester, AgentRole::Reviewer] {
        for family in ["ai", "mcp", "agent"] {
            assert!(
                !ceilings.allows(role, family),
                "{role} must not admit '{family}'"
            );
        }
    }

    // The contribution table pins the counterpart rows (others empty).
    assert_eq!(AI_ROLE_CEILINGS.len(), 4);
    let rows: Vec<(AgentRole, Vec<&str>)> = AI_ROLE_CEILINGS
        .iter()
        .map(|row| (row.role, row.families.to_vec()))
        .collect();
    assert_eq!(
        rows,
        [
            (AgentRole::Commander, vec!["ai", "mcp", "agent"]),
            (AgentRole::Implementer, vec!["agent"]),
            (AgentRole::Tester, vec![]),
            (AgentRole::Reviewer, vec![]),
        ]
    );

    // Effective Commander ceiling carries the contributions ...
    let effective = ceilings.effective_for(AgentRole::Commander);
    for family in ["ai", "mcp", "agent"] {
        assert!(effective.contains(&family), "effective lacks '{family}'");
    }
    // ... while a fresh Core seed is unaffected ...
    let fresh = RoleCeilingCatalog::core();
    assert!(!fresh.allows(AgentRole::Commander, "ai"));
    assert!(!fresh.allows(AgentRole::Implementer, "agent"));
    // ... and re-registration is a duplicate, never an overwrite.
    assert!(register_ai_ceilings(&mut ceilings).is_err());
}

// ── test 3: credential exclusive-or, kind gates, narrow-only, names-only ─────

#[test]
fn provider_credential_exclusive_or_narrow_only() {
    let env = CredentialRef::from_env("MY_KEY").expect("valid env");
    let cmd = CredentialRef::from_cmd("pass", vec!["show".to_string()]).expect("valid cmd");

    // Construction gate: each ref must match its kind.
    assert!(
        ProviderCredentialConfig::new(Some(cmd.clone()), None).is_err(),
        "api_key_env must refuse Cmd refs"
    );
    assert!(
        ProviderCredentialConfig::new(None, Some(env.clone())).is_err(),
        "api_key_cmd must refuse Env refs"
    );

    // Neither set resolves to no credential.
    let unset = ProviderCredentialConfig::unset();
    assert!(unset.is_unset());
    assert_eq!(unset.source(), Ok(None));
    let empty_lookup: BTreeMap<String, String> = BTreeMap::new();
    assert_eq!(
        resolve_provider_credential(&unset, |var| empty_lookup.get(var).cloned()),
        Ok(None)
    );
    assert_eq!(unset.to_string(), "credential:unset");

    // Env resolves through the injected lookup; missing/empty denies
    // names-only (the name appears, the value never does).
    let env_only = ProviderCredentialConfig::new(Some(env.clone()), None).expect("valid");
    let mut vars = BTreeMap::new();
    vars.insert("MY_KEY".to_string(), "live-value".to_string());
    vars.insert("OTHER_KEY".to_string(), "super-secret-value".to_string());
    assert_eq!(
        resolve_provider_credential(&env_only, |var| vars.get(var).cloned()),
        Ok(Some("live-value".to_string()))
    );
    let missing =
        resolve_provider_credential(&env_only, |_| None).expect_err("missing env must deny");
    let missing_text = missing.to_string();
    assert!(missing_text.contains("MY_KEY"), "{missing_text}");
    assert!(
        !missing_text.contains("live-value") && !missing_text.contains("super-secret-value"),
        "diagnostic leaked a value: {missing_text}"
    );

    // Both set denies as conflict before anything is read: the command is a
    // missing program, yet the denial names both references (proving the
    // exclusive-or gate runs first, with no spawn attempted).
    let missing_cmd =
        CredentialRef::from_cmd("bitty-definitely-missing-xyz", Vec::new()).expect("valid cmd");
    let both = ProviderCredentialConfig::new(Some(env), Some(missing_cmd)).expect("kept");
    assert!(!both.is_unset());
    let conflict = resolve_provider_credential(&both, |var| vars.get(var).cloned())
        .expect_err("conflict must deny");
    let conflict_text = conflict.to_string();
    assert!(
        conflict_text.contains("api_key_env:MY_KEY"),
        "{conflict_text}"
    );
    assert!(
        conflict_text.contains("api_key_cmd:bitty-definitely-missing-xyz"),
        "{conflict_text}"
    );
    assert_eq!(
        both.to_string(),
        "conflict:api_key_env:MY_KEY+api_key_cmd:bitty-definitely-missing-xyz"
    );

    // Missing command programs deny names-only (no spawn target exists).
    let absent = execute_credential_cmd("bitty-definitely-missing-xyz", &[])
        .expect_err("missing program must deny");
    assert!(
        absent.to_string().contains("bitty-definitely-missing-xyz"),
        "{absent}"
    );

    // Narrow-only overlay: keep-identical or remove admits; add, rename, or
    // source-switch denies — checked independently per field.
    let base = ProviderCredentialConfig::new(
        Some(CredentialRef::from_env("BASE_KEY").expect("valid")),
        None,
    )
    .expect("valid");
    let same = ProviderCredentialConfig::new(
        Some(CredentialRef::from_env("BASE_KEY").expect("valid")),
        None,
    )
    .expect("valid");
    let renamed = ProviderCredentialConfig::new(
        Some(CredentialRef::from_env("OTHER_KEY").expect("valid")),
        None,
    )
    .expect("valid");
    let switched = ProviderCredentialConfig::new(None, Some(cmd)).expect("valid");
    assert!(base.check_overlay(&same).is_ok());
    assert!(
        base.check_overlay(&ProviderCredentialConfig::unset())
            .is_ok()
    );
    assert!(base.check_overlay(&renamed).is_err());
    assert!(base.check_overlay(&switched).is_err());
    assert!(
        ProviderCredentialConfig::unset()
            .check_overlay(&base)
            .is_err(),
        "empty base gains nothing"
    );
    assert!(check_provider_override(&base, &same).is_ok());
    assert!(check_provider_override(&base, &renamed).is_err());
}

// ── wiring: from_binding carries the AI extension set (Q1) ───────────────────

#[test]
fn from_binding_registers_ai_contributions() {
    let identity = bound_identity();
    let host = LiveBittyHost::from_binding(
        &identity,
        ScopeSet::single(Scope::TerminalInspect),
        None,
        canned_exec_provider,
    )
    .expect("bound identity builds");
    assert_eq!(host.host_client_id(), "owner.name");

    // All three families with all seven heads at exact polarity.
    for family in ["ai", "mcp", "agent"] {
        assert!(
            host.capabilities().contains_family(family),
            "host lacks family '{family}'"
        );
    }
    for (head, requires_param) in [
        ("ai.provider", false),
        ("ai.stream", false),
        ("ai.model", false),
        ("mcp.invoke", true),
        ("agent.context.terminal", false),
        ("agent.context.workspace", false),
        ("agent.memory", true),
    ] {
        assert_eq!(
            host.capabilities().requires_param(head),
            Some(requires_param),
            "host polarity diverged for '{head}'"
        );
    }

    // Commander and Implementer ceilings restored; Tester/Reviewer leak-free.
    assert!(host.ceilings().allows(AgentRole::Commander, "ai"));
    assert!(host.ceilings().allows(AgentRole::Commander, "mcp"));
    assert!(host.ceilings().allows(AgentRole::Commander, "agent"));
    assert!(host.ceilings().allows(AgentRole::Implementer, "agent"));
    assert!(!host.ceilings().allows(AgentRole::Implementer, "ai"));
    assert!(!host.ceilings().allows(AgentRole::Tester, "agent"));
    assert!(!host.ceilings().allows(AgentRole::Reviewer, "agent"));

    // Combined entry registers both halves against fresh Core seeds.
    let mut catalog = CapabilityCatalog::core();
    let mut ceilings = RoleCeilingCatalog::core();
    register_ai_families(&mut catalog, &mut ceilings).expect("combined registers");
    assert!(catalog.contains_family("ai"));
    assert!(ceilings.allows(AgentRole::Commander, "ai"));
}

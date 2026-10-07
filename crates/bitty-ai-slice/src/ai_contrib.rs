//! AI capability registration against the Core typed catalogs (CTX-0916 S6).
//!
//! This module is the `bitty-ai` counterpart of the Core cutover that removed
//! every AI-shaped symbol from the Core seed: S4 dropped the `agent` / `mcp` /
//! `ai` families, their seven heads, and the Commander/Implementer ceiling rows
//! (`bitty` `df82e350`), and S5 added the ceiling-contribution API plus the
//! staged credential surface (`bitty` `271d662b`). Core-only installs reject
//! all of it fail-closed; this module restores AI authority additively through
//! the Core extension hooks ([`CapabilityCatalog::register`] and
//! [`RoleCeilingCatalog::register_ceiling`]) without changing any Core
//! default. Owner authorization for this cross-repo registration (handoff
//! PX-5009–PX-5014 on `bitty` CTX-0990, decisions PX-5014: Q1/Q2/Q3 all YES).
//!
//! ## Counterpart map (removed-Core symbol -> contribution here)
//!
//! | Core removal (pre-S4 source) | Contribution here |
//! |---|---|
//! | `agent` family (`CAPABILITY_FAMILIES`, `CapabilityFamily::Agent`) | `AI_FAMILY_CONTRIBUTIONS` `agent` row |
//! | `mcp` family | `AI_FAMILY_CONTRIBUTIONS` `mcp` row |
//! | `ai` family | `AI_FAMILY_CONTRIBUTIONS` `ai` row |
//! | `agent.context.terminal` head, param `false` | same head, same polarity |
//! | `agent.context.workspace` head, param `false` | same head, same polarity |
//! | `agent.memory` head, param `true` | same head, same polarity |
//! | `mcp.invoke` head, param `true` | same head, same polarity |
//! | `ai.provider` head, param `false` | same head, same polarity |
//! | `ai.stream` head, param `false` | same head, same polarity |
//! | `ai.model` head, param `false` | same head, same polarity |
//! | Commander ceiling `Agent`/`Mcp`/`Ai` | `AI_ROLE_CEILINGS` Commander row |
//! | Implementer ceiling `Agent` | `AI_ROLE_CEILINGS` Implementer row |
//! | `ProviderCredentialConfig` (S5 staged surface) | [`ProviderCredentialConfig`] canonical home |
//!
//! Effect wordings in [`AI_FAMILY_CONTRIBUTIONS`] are verbatim from the
//! pre-extraction Core consent strings (`bitty-plugin-host/src/capability.rs`
//! `effect_statement_with` at `df82e350^`); presentation ownership for Core
//! callers stays Core-side, this table only pins the counterpart wording so
//! the ai-loaded accepts set stays byte-identical to the pre-extraction set.
//!
//! ## What is NOT contributed
//!
//! The registration contributes exactly families, heads, parameter rules, role
//! ceilings, and the credential adapter. It contributes no trust domains, no
//! Core `EventKind`, and no new secret tier: [`register_ai_families`] touches
//! only the two Core contribution catalogs, and the credential adapter reads
//! through the Core reference mechanism only.
//!
//! ## Wiring (Q1) and layering (Q2, Q3)
//!
//! The extension-load call site is [`LiveBittyHost::from_binding`](crate::live_host::LiveBittyHost::from_binding):
//! it delegates to `LiveBittyHost::new`, which performs
//! [`register_ai_families`] before any grant parse, authorization, or tool
//! registration, so every live host carries the AI extension set and a failed
//! registration fails host construction closed (no host exists without its
//! contributions). `FakeHost` is deliberately not wired: it is the
//! deterministic scripted twin over the `bitty-ipc` seams, and its tests keep
//! string parity with these heads as opaque labels.
//!
//! Per Q2 only this slice takes the typed Core dependency
//! (`bitty-package` / `bitty-plugin-host` at the S5 merge rev `271d662b`);
//! `bitty-ai-runtime` stays `std`-only. Per Q3 the provider-credential adapter
//! lives in this slice module: [`ProviderCredentialConfig`] is the canonical
//! home moved from Core (the Core shim is removed in S7), built on Core's
//! [`CredentialRef`](bitty_plugin_host::CredentialRef), with verbatim
//! exclusive-or, narrow-only, and names-only semantics. It composes with the
//! runtime [`SecretField`](bitty_ai_runtime::secret::SecretField) typed
//! container (values, redacted everywhere) without overlapping it: the config
//! names *where* a credential comes from and never carries a value.
//!
//! ## Determinism and scope
//!
//! Registration is pure and bounded (3 families, 7 heads, 4 ceiling rows);
//! credential resolution reads through an injected environment lookup and one
//! bounded, shell-free command runner (direct spawn, null stdin, discarded
//! stderr, [`MAX_CREDENTIAL_CMD_OUTPUT_BYTES`] cap,
//! [`CREDENTIAL_CMD_TIMEOUT`] deadline). Diagnostics quote
//! reference names only — values never enter an error, log, or audit detail.

use std::fmt;
use std::time::Duration;

use bitty_package::CapabilityCatalog;
use bitty_plugin_host::{
    AgentRole, CredentialRef, CredentialSource, PluginError, RoleCeilingCatalog,
    check_project_override, resolve_choice,
};

use crate::error::SliceError;

// Re-export the Core output bound so the adapter and its callers cannot drift
// from the host store ceiling the moved semantic was reviewed against.
pub use bitty_plugin_host::MAX_CREDENTIAL_CMD_OUTPUT_BYTES;

/// Deadline bounding one credential-helper execution (AI-0173).
///
/// Covers stdout collection *and* process completion together from just
/// after spawn: a helper that hangs silently, dribbles output forever, or
/// exits without closing stdout resolves to a names-only timeout error
/// instead of blocking the caller without bound. Core has no equivalent
/// (`Command::output()` there waits unbounded); this bound is owned here.
pub const CREDENTIAL_CMD_TIMEOUT: Duration = Duration::from_secs(10);

/// Poll interval while waiting for a credential helper: small enough that
/// deadline overshoot stays negligible against [`CREDENTIAL_CMD_TIMEOUT`],
/// large enough to avoid a hot spin on `try_wait`/`try_recv`.
const CREDENTIAL_CMD_POLL_INTERVAL: Duration = Duration::from_millis(10);

// ── AI family contributions (M1/M2/M3, H1–H7, P1/P2) ─────────────────────────

/// One contributed capability head: its identifier, its parameter rule, and
/// its verbatim pre-extraction effect wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AiHeadContribution {
    /// Bare head identifier (for example `"ai.provider"`).
    pub head: &'static str,
    /// Whether the head requires a `:PARAMETER` (P1/P2 polarity).
    pub requires_param: bool,
    /// Verbatim pre-extraction Core effect wording (counterpart pin only;
    /// Core presentation ownership stays Core-side).
    pub effect: &'static str,
}

/// One contributed capability family with its heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AiFamilyContribution {
    /// Family label (for example `"ai"`).
    pub family: &'static str,
    /// Heads belonging to this family.
    pub heads: &'static [AiHeadContribution],
}

/// AI capability families removed from the Core seed, contributed back here.
///
/// Three families, seven heads, exact pre-extraction polarity: `ai` carries
/// three param-`false` heads, `mcp` carries one param-`true` head, `agent`
/// carries two param-`false` heads plus one param-`true` head. Effect strings
/// are verbatim from the handoff manifest (PX-5010).
pub const AI_FAMILY_CONTRIBUTIONS: &[AiFamilyContribution] = &[
    AiFamilyContribution {
        family: "ai",
        heads: &[
            AiHeadContribution {
                head: "ai.provider",
                requires_param: false,
                effect: "Use allowlisted AI provider",
            },
            AiHeadContribution {
                head: "ai.stream",
                requires_param: false,
                effect: "Stream AI responses for this agent",
            },
            AiHeadContribution {
                head: "ai.model",
                requires_param: false,
                effect: "Select AI model for this agent (bounded)",
            },
        ],
    },
    AiFamilyContribution {
        family: "mcp",
        heads: &[AiHeadContribution {
            head: "mcp.invoke",
            requires_param: true,
            effect: "Invoke allowlisted MCP tool (per-tool, bounded frame 256KiB)",
        }],
    },
    AiFamilyContribution {
        family: "agent",
        heads: &[
            AiHeadContribution {
                head: "agent.context.terminal",
                requires_param: false,
                effect: "Observe terminal context for this agent (bounded 32KiB)",
            },
            AiHeadContribution {
                head: "agent.context.workspace",
                requires_param: false,
                effect: "Observe workspace context for this agent (bounded 32KiB)",
            },
            AiHeadContribution {
                head: "agent.memory",
                requires_param: true,
                effect: "Persist agent conversational memory (opt-in, 0600, <=7 days)",
            },
        ],
    },
];

// ── AI role ceilings (C1/C2) ─────────────────────────────────────────────────

/// Role-ceiling contribution for one role: the families the role additionally
/// admits over its AI-free Core default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AiRoleCeiling {
    /// Role receiving the contribution.
    pub role: AgentRole,
    /// Families contributed for this role (empty rows contribute nothing and
    /// are skipped by [`register_ai_ceilings`]).
    pub families: &'static [&'static str],
}

/// AI role ceilings removed from the Core seed, contributed back here.
///
/// Commander regains `ai`/`mcp`/`agent`, Implementer regains `agent`; Tester
/// and Reviewer rows are empty by counterpart (their pre-S4 rows carried no
/// AI family, so there is nothing to restore).
pub const AI_ROLE_CEILINGS: &[AiRoleCeiling] = &[
    AiRoleCeiling {
        role: AgentRole::Commander,
        families: &["ai", "mcp", "agent"],
    },
    AiRoleCeiling {
        role: AgentRole::Implementer,
        families: &["agent"],
    },
    AiRoleCeiling {
        role: AgentRole::Tester,
        families: &[],
    },
    AiRoleCeiling {
        role: AgentRole::Reviewer,
        families: &[],
    },
];

// ── registration entry points ────────────────────────────────────────────────

/// Register the AI capability families against `catalog`.
///
/// Declares every head in [`AI_FAMILY_CONTRIBUTIONS`] with its exact
/// pre-extraction parameter polarity. Additive only: any invalid or duplicate
/// head rejects the whole call through the Core fail-closed rules (the catalog
/// keeps whatever the failing call did not touch, per Core per-call
/// atomicity).
///
/// # Errors
///
/// Returns [`SliceError::ExtensionRejected`] when Core rejects a contribution
/// (shape, duplicate, or bound). Against a fresh Core seed the fixed table
/// always registers; a rejection therefore signals catalog drift or a double
/// registration, and callers must fail closed.
pub fn register_ai_capabilities(catalog: &mut CapabilityCatalog) -> Result<(), SliceError> {
    for contribution in AI_FAMILY_CONTRIBUTIONS {
        let heads: Vec<(&str, bool)> = contribution
            .heads
            .iter()
            .map(|head| (head.head, head.requires_param))
            .collect();
        catalog
            .register(contribution.family, &heads)
            .map_err(|err| SliceError::ExtensionRejected {
                reason: err.to_string(),
            })?;
    }
    Ok(())
}

/// Register the AI role ceilings against `ceilings`.
///
/// Contributes every non-empty row in [`AI_ROLE_CEILINGS`]; empty rows
/// (Tester, Reviewer) are skipped because [`RoleCeilingCatalog`] rejects empty
/// registrations fail-closed. Additive only, same failure discipline as
/// [`register_ai_capabilities`].
///
/// # Errors
///
/// Returns [`SliceError::ExtensionRejected`] when Core rejects a contribution.
pub fn register_ai_ceilings(ceilings: &mut RoleCeilingCatalog) -> Result<(), SliceError> {
    for ceiling in AI_ROLE_CEILINGS {
        if ceiling.families.is_empty() {
            continue;
        }
        ceilings
            .register_ceiling(ceiling.role, ceiling.families)
            .map_err(|err| SliceError::ExtensionRejected {
                reason: err.to_string(),
            })?;
    }
    Ok(())
}

/// Register the full AI extension set: families plus role ceilings.
///
/// Calls [`register_ai_capabilities`] then [`register_ai_ceilings`]. The two
/// calls are sequential, not atomic: on a ceiling failure the catalog keeps
/// the registered families, so callers must discard both catalogs (host
/// construction does this by failing closed before any host state exists).
///
/// # Errors
///
/// Returns [`SliceError::ExtensionRejected`] from either half.
pub fn register_ai_families(
    catalog: &mut CapabilityCatalog,
    ceilings: &mut RoleCeilingCatalog,
) -> Result<(), SliceError> {
    register_ai_capabilities(catalog)?;
    register_ai_ceilings(ceilings)?;
    Ok(())
}

// ── provider credential adapter (R1, canonical home) ─────────────────────────

/// Provider credential config: the accepted `api_key_env` / `api_key_cmd`
/// surface (OQ-054 MPC-1..MPC-4).
///
/// Canonical home moved from Core in CTX-0916 S6 (DEC-0100; staged at
/// `bitty-plugin-host/src/provider_credential.rs` in S5, whose deprecated
/// shim is removed in S7). Semantics are verbatim from that source: at most
/// one field may resolve — both set denies as conflict at resolution
/// ([`resolve_choice`]), neither set resolves to no credential. Construction
/// preserves both fields so the conflict stays observable (and auditable)
/// instead of being rejected silently at parse time. `Display`/`Debug` quote
/// reference names only — values never enter this type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProviderCredentialConfig {
    /// `api_key_env` reference, if configured.
    pub api_key_env: Option<CredentialRef>,
    /// `api_key_cmd` reference, if configured.
    pub api_key_cmd: Option<CredentialRef>,
}

impl ProviderCredentialConfig {
    /// Build a config from already-validated references.
    ///
    /// Both-present is preserved (resolution denies as conflict); each
    /// reference must still be the matching source kind (`Env` for
    /// `api_key_env`, `Cmd` for `api_key_cmd`), otherwise construction
    /// denies fail-closed.
    ///
    /// # Errors
    ///
    /// Returns a registry error when a reference has the wrong source kind.
    pub fn new(
        api_key_env: Option<CredentialRef>,
        api_key_cmd: Option<CredentialRef>,
    ) -> Result<Self, PluginError> {
        if let Some(ref reference) = api_key_env {
            if reference.source_kind() != CredentialSource::Env {
                return Err(PluginError::registry(format!(
                    "api_key_env must name an env reference, got '{reference}'"
                )));
            }
        }
        if let Some(ref reference) = api_key_cmd {
            if reference.source_kind() != CredentialSource::Cmd {
                return Err(PluginError::registry(format!(
                    "api_key_cmd must name a command reference, got '{reference}'"
                )));
            }
        }
        Ok(Self {
            api_key_env,
            api_key_cmd,
        })
    }

    /// Empty config (no credential).
    #[must_use]
    pub fn unset() -> Self {
        Self {
            api_key_env: None,
            api_key_cmd: None,
        }
    }

    /// Whether neither reference is configured.
    #[must_use]
    pub fn is_unset(&self) -> bool {
        self.api_key_env.is_none() && self.api_key_cmd.is_none()
    }

    /// Which source wins under the exclusive-or order, if any.
    ///
    /// Both set denies fail-closed with a names-only grant error.
    ///
    /// # Errors
    ///
    /// Returns a grant error naming both references when both are set.
    pub fn source(&self) -> Result<Option<CredentialSource>, PluginError> {
        resolve_choice(self.api_key_env.as_ref(), self.api_key_cmd.as_ref())
    }

    /// Narrow-only project check for a whole provider config.
    ///
    /// Each field is checked with [`check_project_override`]: the
    /// overlay may keep each reference identical or remove it, never
    /// add, rename, or switch sources.
    ///
    /// # Errors
    ///
    /// Returns a grant error when either field widens.
    pub fn check_overlay(&self, overlay: &Self) -> Result<(), PluginError> {
        check_provider_override(self, overlay)
    }
}

impl fmt::Display for ProviderCredentialConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.api_key_env, &self.api_key_cmd) {
            (None, None) => f.write_str("credential:unset"),
            (Some(env), None) => write!(f, "{env}"),
            (None, Some(cmd)) => write!(f, "{cmd}"),
            (Some(env), Some(cmd)) => write!(f, "conflict:{env}+{cmd}"),
        }
    }
}

/// Provider-level project override (OQ-054 adopted narrow-only).
///
/// Checks the `api_key_env` and `api_key_cmd` fields independently with
/// [`check_project_override`]; any widening on either field denies.
///
/// # Errors
///
/// Returns a grant error when the overlay adds, renames, or switches sources.
pub fn check_provider_override(
    base: &ProviderCredentialConfig,
    overlay: &ProviderCredentialConfig,
) -> Result<(), PluginError> {
    check_project_override(base.api_key_env.as_ref(), overlay.api_key_env.as_ref())?;
    check_project_override(base.api_key_cmd.as_ref(), overlay.api_key_cmd.as_ref())
}

/// Resolve a provider credential to its value, if configured.
///
/// Exclusive-or first ([`ProviderCredentialConfig::source`]): conflict
/// denies before anything is read. `Env` resolves through `env_lookup`
/// (injected so tests stay hermetic; the live wrapper below supplies
/// `std::env`); missing or empty variables deny with a names-only
/// error. `Cmd` resolves through the bounded shell-free runner
/// ([`execute_credential_cmd`]). Values are returned, never logged —
/// callers must inject them into child environments only, per ADR-0006.
///
/// # Errors
///
/// Returns a grant error on conflict or on a missing/empty variable, a
/// registry error on a misconfigured or failing command, or a limit error on
/// oversize command output.
pub fn resolve_provider_credential(
    config: &ProviderCredentialConfig,
    env_lookup: impl Fn(&str) -> Option<String>,
) -> Result<Option<String>, PluginError> {
    match config.source()? {
        None => Ok(None),
        Some(CredentialSource::Env) => {
            let var = match config.api_key_env.as_ref() {
                Some(CredentialRef::Env { var }) => var.as_str(),
                _ => {
                    return Err(PluginError::registry(
                        "api_key_env misconfigured (deny by default)".to_string(),
                    ));
                }
            };
            match env_lookup(var) {
                Some(value) if !value.is_empty() => Ok(Some(value)),
                _ => Err(PluginError::grant(format!(
                    "credential env '{var}' is missing or empty (deny by default)"
                ))),
            }
        }
        Some(CredentialSource::Cmd) => {
            let (program, args) = match config.api_key_cmd.as_ref() {
                Some(CredentialRef::Cmd { program, args }) => (program.as_str(), args.as_slice()),
                _ => {
                    return Err(PluginError::registry(
                        "api_key_cmd misconfigured (deny by default)".to_string(),
                    ));
                }
            };
            execute_credential_cmd(program, args).map(Some)
        }
    }
}

/// Live-environment resolution (`std::env` lookup).
///
/// Thin wrapper so the core ([`resolve_provider_credential`]) stays
/// hermetic in tests.
///
/// # Errors
///
/// Returns the failures of [`resolve_provider_credential`].
pub fn resolve_provider_credential_live(
    config: &ProviderCredentialConfig,
) -> Result<Option<String>, PluginError> {
    resolve_provider_credential(config, |var| std::env::var(var).ok())
}

/// Execute a credential command and return its output value.
///
/// Shell-free, bounded, names-only discipline: direct spawn, null stdin,
/// OS-discarded stderr, stdout read with a cap, one trailing newline
/// stripped, empty/NUL/non-UTF-8/oversize/non-zero-spawn results deny.
/// Errors quote `program` only. This is the one bounded-spawn path in the
/// slice (mirroring how `local_provider` is the one socket path).
///
/// Hardening delta versus the Core source this adapter was moved from
/// (`bitty-plugin-host/src/provider_credential.rs` at `271d662b`, which
/// shares the shape): Core's `Command::output()` pipes stderr into memory
/// unbounded and checks the stdout cap only after the child exits. Here
/// stderr is `Stdio::null` — helper diagnostics, which may echo secret
/// material, are never buffered, logged, or retained — and stdout is read
/// capped at [`MAX_CREDENTIAL_CMD_OUTPUT_BYTES`] plus one probe byte, so at
/// most that much is ever retained; an oversize report therefore carries
/// `actual` of cap-plus-one rather than the true stream size.
///
/// Deadline delta (AI-0173): Core waits for the helper unbounded. Here a
/// single [`CREDENTIAL_CMD_TIMEOUT`] deadline from just after spawn covers
/// stdout collection *and* process completion together — stdout drains on a
/// worker thread while the caller polls the reader handoff and `try_wait`,
/// so a silent hang, an endless dribble, or an exit without pipe close all
/// resolve on time. Past the deadline the child is killed and reaped and a
/// names-only timeout error is returned (no output bytes in diagnostics).
///
/// # Errors
///
/// Returns a registry error when the program fails to spawn, exits non-zero,
/// exceeds [`CREDENTIAL_CMD_TIMEOUT`], or produces empty/NUL/non-UTF-8/
/// unreadable output, or a limit error past
/// [`MAX_CREDENTIAL_CMD_OUTPUT_BYTES`].
pub fn execute_credential_cmd(program: &str, args: &[String]) -> Result<String, PluginError> {
    use std::io::Read;
    use std::process::{Command, ExitStatus, Stdio};
    use std::sync::mpsc;
    use std::time::Instant;
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| {
            PluginError::registry(format!("api_key_cmd '{program}' failed to spawn: {err}"))
        })?;
    let pipe = match child.stdout.take() {
        Some(pipe) => pipe,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(PluginError::registry(format!(
                "api_key_cmd '{program}' output unreadable"
            )));
        }
    };
    // Drain stdout on a worker thread so a voluminous helper can never fill
    // the pipe while the caller waits for exit, and a silent helper can
    // never wedge the caller in the read. The cap-plus-one probe is
    // unchanged: at most that much is ever retained.
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut stdout = Vec::new();
        let read = pipe
            .take(MAX_CREDENTIAL_CMD_OUTPUT_BYTES as u64 + 1)
            .read_to_end(&mut stdout);
        let _ = done_tx.send((read, stdout));
    });
    let deadline = Instant::now() + CREDENTIAL_CMD_TIMEOUT;
    let mut output: Option<(std::io::Result<usize>, Vec<u8>)> = None;
    let mut status: Option<ExitStatus> = None;
    loop {
        if output.is_none() {
            match done_rx.try_recv() {
                Ok(done) => output = Some(done),
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Reader thread died without delivering: same
                    // unreadable denial as a failed sequential read.
                    if status.is_none() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    return Err(PluginError::registry(format!(
                        "api_key_cmd '{program}' output unreadable"
                    )));
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if status.is_none() {
            match child.try_wait() {
                Ok(Some(exit)) => status = Some(exit),
                Ok(None) => {}
                Err(err) => {
                    return Err(PluginError::registry(format!(
                        "api_key_cmd '{program}' wait failed: {err}"
                    )));
                }
            }
        }
        // Validate arrived output; once it is readable and in-bounds, the
        // known exit status decides between parse and nonzero-exit denial.
        // Fall-through means at least one side is still pending (or full
        // output arrived while the helper stays alive with the pipe
        // closed): keep polling until the deadline below.
        let completed: Option<ExitStatus> = match output.as_ref() {
            Some((read, stdout)) => {
                if read.is_err() {
                    if status.is_none() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    return Err(PluginError::registry(format!(
                        "api_key_cmd '{program}' output unreadable"
                    )));
                }
                if stdout.len() > MAX_CREDENTIAL_CMD_OUTPUT_BYTES {
                    if status.is_none() {
                        // Still producing past the probe byte: stop and
                        // reap before denying, as the sequential shape did.
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    return Err(PluginError::LimitExceeded {
                        field: "api_key_cmd.output".to_string(),
                        limit: MAX_CREDENTIAL_CMD_OUTPUT_BYTES,
                        actual: stdout.len(),
                    });
                }
                status
            }
            None => None,
        };
        if let Some(exit) = completed {
            if !exit.success() {
                return Err(PluginError::registry(format!(
                    "api_key_cmd '{program}' exited with {exit}"
                )));
            }
            let (_, stdout) = output.take().expect("output validated present");
            // The capped read above retains at most cap-plus-one bytes and
            // the oversize arm already returned, so `stdout` is in-bounds
            // here; the remaining checks (NUL, UTF-8, strip, empty) are
            // unchanged from Core.
            if stdout.contains(&0) {
                return Err(PluginError::registry(format!(
                    "api_key_cmd '{program}' output must not contain NUL bytes"
                )));
            }
            let mut text = String::from_utf8(stdout).map_err(|_| {
                PluginError::registry(format!("api_key_cmd '{program}' output is not UTF-8"))
            })?;
            if text.ends_with('\n') {
                text.pop();
                if text.ends_with('\r') {
                    text.pop();
                }
            }
            if text.is_empty() {
                return Err(PluginError::registry(format!(
                    "api_key_cmd '{program}' produced empty output"
                )));
            }
            return Ok(text);
        }
        if Instant::now() >= deadline {
            if status.is_none() {
                let _ = child.kill();
                let _ = child.wait();
            }
            // Names-only: the program, never helper output bytes.
            return Err(PluginError::registry(format!(
                "api_key_cmd '{program}' timed out after {}s",
                CREDENTIAL_CMD_TIMEOUT.as_secs()
            )));
        }
        std::thread::sleep(CREDENTIAL_CMD_POLL_INTERVAL);
    }
}

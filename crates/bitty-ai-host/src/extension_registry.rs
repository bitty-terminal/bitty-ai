//! Host-side pure-type mirror of the extension-registry seam.
//!
//! **Status: Phase 2 S1 pure-type mirror per AI-0275 (survey AI-0267).**
//! This module mirrors the *shapes* of the runtime `extension.rs` contract —
//! the closed [`ExtensionPoint`] 8-enum, the data-only [`Manifest`], and one
//! opaque-string descriptor struct per point mirroring the corresponding
//! `*Point` trait method return — so the host can *talk about* extension
//! declarations with the same vocabulary the runtime declares. All values
//! are plain data with no clock, threads, filesystem, network, IPC, or
//! ambient authority: `std` only, like the rest of this crate.
//!
//! Mirrored sources (read-only, shapes kept identical):
//!
//! - `extension.rs` [`ExtensionPoint`] enum (8 variants
//!   Model/Tool/Context/Memory/Compactor/Agent/Command/Ui,
//!   exhaustive-match-closed).
//! - `extension.rs` [`Manifest`] (`name`/`version`/`host` opaque strings plus
//!   `points: Vec<ExtensionPoint>`; `host` stored verbatim, never parsed;
//!   a declaration never doubles as a permission grant).
//! - `extension.rs` per-point method returns as opaque-string descriptor
//!   shapes: `ModelPoint::declared_model` → [`ModelDescriptor`],
//!   `ToolPoint::effect_class` → [`ToolDescriptor`],
//!   `ContextPoint::read_scope` → [`ContextDescriptor`],
//!   `MemoryPoint::retention_class` → [`MemoryDescriptor`],
//!   `CompactorPoint::strategy_id` → [`CompactorDescriptor`],
//!   `AgentPoint::capability_id` → [`AgentDescriptor`],
//!   `CommandPoint::command_name` → [`CommandDescriptor`],
//!   `UiPoint::descriptor` → [`UiDescriptor`] (plain text, never rendered).
//!
//! Non-inheritance holds by construction here as it does runtime-side: the
//! eight descriptors are separate structs with no shared supertype, no
//! shared trait, and no blanket impl, so constructing one contributes no
//! value of any other point's type. A holder wanting two points declares
//! two fields explicitly (pinned by
//! `combined_holder_needs_one_field_per_point`); there is no cross-point
//! coercion.
//!
//! Registration versus activation stays split: REGISTRATION is the
//! [`Manifest`] declaration (which points an extension contributes) plus
//! per-point descriptor data; ACTIVATION — grant parsing, authorization,
//! tool registration, consent evaluation — lives elsewhere and is absent
//! here. A manifest or descriptor never grants anything: there is no
//! registry to admit into, no resolver to convince, and no executor to
//! reach.
//!
//! Explicitly **not** mirrored here (each needs its own scoped task):
//!
//! - Registry, resolution algorithm, discovery/loading, version-predicate
//!   parsing or evaluation (`host` stays an opaque string), activation
//!   policy, and auto-activation on declare.
//! - IPC envelopes or transport, filesystem or network access, threads,
//!   clocks, secrets or credentials (including `ProviderCredentialConfig`,
//!   whose canonical home stays the R1 credential adapter, not this seam),
//!   MCP transport, UI rendering, command execution, and tool dispatch.
//! - Closed vocabularies or bounds for the descriptor strings: every
//!   descriptor payload is opaque data with no allowlist, no length check,
//!   and no parity check against any other seam.
//! - `Cargo.toml` stays `std`-only.
//!
//! S-1/S-2 posture (S-1 adopted 2026-10-15, bitty-ai-docs#277): this mirror is terminal-side
//! declaration metadata only — shapes a host uses to *talk about*
//! extensions, never a channel carrying them and never a store admitting
//! them.
//!
//! Deny-all posture holds by construction: constructing these values
//! performs no I/O and bestows no capability. An empty `points` list admits
//! nothing, a populated one still grants nothing, and there is no item here
//! that executes, renders, resolves, or authorizes.
//!
//! Phase 2 S2 appendage (AI-0282): scoped-IPC envelope types, no transport.
//! [`DeclarationEnvelope`] echoes a [`Manifest`] (host verbatim, never
//! parsed) and [`ContributionEnvelope`] echoes one per-point descriptor per
//! value. Both carry declarations only and grant nothing: activation — grant
//! parsing, authorization, tool registration, consent evaluation — is NOT
//! modeled here (the deny-all stub stays out; the S3 trigger is unmet).
//! Shared S2 convention: types only, `std` only, no socket/transport/dep
//! change, NO version field, no behavior beyond trivial construction and
//! borrowing, no thread/clock/net/fs.

/// The eight declared extension-point categories.
///
/// Host-side mirror of the runtime `ExtensionPoint`: a plain exhaustive
/// enum (data only). Match with no wildcard so adding a ninth point breaks
/// at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtensionPoint {
    /// Model capability declaration surface.
    Model,
    /// Tool effect-class declaration surface.
    Tool,
    /// Scoped read declaration surface.
    Context,
    /// Retention-class declaration surface.
    Memory,
    /// Compaction-strategy declaration surface.
    Compactor,
    /// Agent-capability declaration surface.
    Agent,
    /// Command registration declaration surface.
    Command,
    /// Presentation contribution declaration surface (text only).
    Ui,
}

/// Extension manifest: data only, no parsing, no resolution.
///
/// Host-side mirror of the runtime `Manifest`. `host` is an opaque version
/// predicate over the host (`bitty-ai`); stored verbatim, never parsed here.
/// `points` lists the contributed extension points. This declaration never
/// doubles as a permission grant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Extension name.
    pub name: String,
    /// Extension version (opaque string).
    pub version: String,
    /// Host version predicate (opaque string, never parsed).
    pub host: String,
    /// Contributed extension points.
    pub points: Vec<ExtensionPoint>,
}

/// Model capability descriptor (data mirror of `ModelPoint::declared_model`).
///
/// Opaque string only; no validation, no lookup, no execution. Grants
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// Declared model capability identifier (opaque descriptor).
    pub declared_model: String,
}

impl ModelDescriptor {
    /// Wrap an opaque model identifier as data.
    #[must_use]
    pub fn new(declared_model: String) -> Self {
        Self { declared_model }
    }

    /// Borrow the declared identifier.
    #[must_use]
    pub fn declared_model(&self) -> &str {
        &self.declared_model
    }
}

/// Tool effect-class descriptor (data mirror of `ToolPoint::effect_class`).
///
/// Opaque string only; no validation, no authorization, no execution.
/// Grants nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDescriptor {
    /// Declared tool effect class (opaque descriptor).
    pub effect_class: String,
}

impl ToolDescriptor {
    /// Wrap an opaque effect class as data.
    #[must_use]
    pub fn new(effect_class: String) -> Self {
        Self { effect_class }
    }

    /// Borrow the declared effect class.
    #[must_use]
    pub fn effect_class(&self) -> &str {
        &self.effect_class
    }
}

/// Scoped read descriptor (data mirror of `ContextPoint::read_scope`).
///
/// Opaque string only; no assembly, no redaction, no consent. Grants
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextDescriptor {
    /// Declared read scope (opaque descriptor).
    pub read_scope: String,
}

impl ContextDescriptor {
    /// Wrap an opaque read scope as data.
    #[must_use]
    pub fn new(read_scope: String) -> Self {
        Self { read_scope }
    }

    /// Borrow the declared read scope.
    #[must_use]
    pub fn read_scope(&self) -> &str {
        &self.read_scope
    }
}

/// Retention-class descriptor (data mirror of `MemoryPoint::retention_class`).
///
/// Opaque string only; no retention enforcement, no store access. Grants
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDescriptor {
    /// Declared retention class (opaque descriptor).
    pub retention_class: String,
}

impl MemoryDescriptor {
    /// Wrap an opaque retention class as data.
    #[must_use]
    pub fn new(retention_class: String) -> Self {
        Self { retention_class }
    }

    /// Borrow the declared retention class.
    #[must_use]
    pub fn retention_class(&self) -> &str {
        &self.retention_class
    }
}

/// Compaction-strategy descriptor (data mirror of
/// `CompactorPoint::strategy_id`).
///
/// Opaque string only; no compaction work, no budget enforcement. Grants
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactorDescriptor {
    /// Declared compaction strategy identifier (opaque descriptor).
    pub strategy_id: String,
}

impl CompactorDescriptor {
    /// Wrap an opaque strategy identifier as data.
    #[must_use]
    pub fn new(strategy_id: String) -> Self {
        Self { strategy_id }
    }

    /// Borrow the declared strategy identifier.
    #[must_use]
    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }
}

/// Agent-capability descriptor (data mirror of `AgentPoint::capability_id`).
///
/// Opaque string only; no tier/level gate, no session access. Grants
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentDescriptor {
    /// Declared agent capability identifier (opaque descriptor).
    pub capability_id: String,
}

impl AgentDescriptor {
    /// Wrap an opaque capability identifier as data.
    #[must_use]
    pub fn new(capability_id: String) -> Self {
        Self { capability_id }
    }

    /// Borrow the declared capability identifier.
    #[must_use]
    pub fn capability_id(&self) -> &str {
        &self.capability_id
    }
}

/// Command registration descriptor (data mirror of
/// `CommandPoint::command_name`).
///
/// Opaque string only; no registration, no dispatch, no execution. Grants
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandDescriptor {
    /// Declared command name (opaque descriptor).
    pub command_name: String,
}

impl CommandDescriptor {
    /// Wrap an opaque command name as data.
    #[must_use]
    pub fn new(command_name: String) -> Self {
        Self { command_name }
    }

    /// Borrow the declared command name.
    #[must_use]
    pub fn command_name(&self) -> &str {
        &self.command_name
    }
}

/// Presentation descriptor (data mirror of `UiPoint::descriptor`).
///
/// Plain text only; never rendered here. Grants nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiDescriptor {
    /// Declared presentation descriptor (plain text; never rendered here).
    pub descriptor: String,
}

impl UiDescriptor {
    /// Wrap an opaque presentation descriptor as data.
    #[must_use]
    pub fn new(descriptor: String) -> Self {
        Self { descriptor }
    }

    /// Borrow the declared presentation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &str {
        &self.descriptor
    }
}

// ---------------------------------------------------------------------------
// Phase 2 S2: scoped-IPC envelope types (no transport).
// ---------------------------------------------------------------------------

/// Declaration envelope: a [`Manifest`] echo for scoped IPC.
///
/// Carries the manifest — name, version, verbatim `host` predicate, and
/// contributed `points` — as data only. `host` stays an opaque string:
/// stored verbatim, never parsed here. This envelope declares which points
/// an extension contributes; it grants nothing. Activation (grant parsing,
/// authorization, tool registration, consent evaluation) is NOT modeled:
/// there is no registry to admit into, no resolver to convince, and no
/// executor to reach.
///
/// Shared S2 convention: types only, `std` only, no socket/transport, no
/// dependency change, NO version field, no thread/clock/net/fs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclarationEnvelope {
    /// Echoed manifest declaration (host verbatim, never parsed).
    pub manifest: Manifest,
}

impl DeclarationEnvelope {
    /// Wrap a manifest declaration as data.
    #[must_use]
    pub fn new(manifest: Manifest) -> Self {
        Self { manifest }
    }

    /// Borrow the echoed manifest declaration.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}

/// Contribution envelope: one per-point descriptor echo per value.
///
/// Each variant carries exactly one S1 descriptor struct verbatim — the
/// same opaque-string shapes the runtime `*Point` traits return — so
/// independence holds as it does for the descriptors: constructing one
/// variant contributes no value of any other point's type. A holder wanting
/// two points sends two envelopes explicitly; there is no cross-point
/// coercion. Like [`DeclarationEnvelope`], this carries declarations only
/// and grants nothing: activation is NOT modeled here.
///
/// Shared S2 convention: types only, `std` only, no socket/transport, no
/// dependency change, NO version field, no thread/clock/net/fs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContributionEnvelope {
    /// Model capability descriptor echo.
    Model(ModelDescriptor),
    /// Tool effect-class descriptor echo.
    Tool(ToolDescriptor),
    /// Scoped read descriptor echo.
    Context(ContextDescriptor),
    /// Retention-class descriptor echo.
    Memory(MemoryDescriptor),
    /// Compaction-strategy descriptor echo.
    Compactor(CompactorDescriptor),
    /// Agent-capability descriptor echo.
    Agent(AgentDescriptor),
    /// Command registration descriptor echo.
    Command(CommandDescriptor),
    /// Presentation descriptor echo (plain text; never rendered here).
    Ui(UiDescriptor),
}

impl ContributionEnvelope {
    /// Which extension point this contribution declares.
    ///
    /// Pure projection over the variant; no I/O, no parsing, no
    /// authorization. Match is exhaustive with no wildcard so adding a
    /// ninth point breaks at compile time.
    #[must_use]
    pub fn point(&self) -> ExtensionPoint {
        match self {
            Self::Model(_) => ExtensionPoint::Model,
            Self::Tool(_) => ExtensionPoint::Tool,
            Self::Context(_) => ExtensionPoint::Context,
            Self::Memory(_) => ExtensionPoint::Memory,
            Self::Compactor(_) => ExtensionPoint::Compactor,
            Self::Agent(_) => ExtensionPoint::Agent,
            Self::Command(_) => ExtensionPoint::Command,
            Self::Ui(_) => ExtensionPoint::Ui,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2 S3: deny-all activation stub (AI-0294, CTX-0152 candidate).
// ---------------------------------------------------------------------------

// Deny-all stub for the activation-policy candidate section of
// `architecture/host-boundary-trait-design.md` (candidate only, no acceptance
// claim, no identifier minted). `std`-only, offline, no behavior beyond
// refusal: no registry, no resolution, no discovery/loading, no
// version-predicate parsing or evaluation, no auto-activate, no state, no
// threads, no clock, no network, no filesystem, no `Cargo.toml` change.
// Reuses the S1 mirror ([`ExtensionPoint`]) and introduces no re-mirrored
// vocabulary and no new byte bounds.
//
// Gate summary (re-stated as a gating contract, not executed): a candidate
// point activation would run strictly after the R2 gates, in the candidate
// order recorded in [`ACTIVATION_ORDER_DOC`] — grant parse, then
// authorization, then tool registration, then consent evaluation — with each
// step a precondition for the next. A point declaration (one of the eight
// [`ExtensionPoint`] variants, a [`Manifest`], or a [`ContributionEnvelope`])
// never substitutes for a grant, an authorization, a registration, or a
// consent evaluation: the declaration names the capability shape while grant,
// authorization, registration, and consent each remain a separate check owned
// elsewhere. This stub performs none of those steps; it records the order as
// the contract and denies every activation — known points even when
// explicitly enabled (fail-closed) and unknown point names (typed refusal).

/// Candidate activation order as a gating contract (documented, not executed).
///
/// ```text
/// grant parse -> authorization -> tool registration -> consent evaluation
/// ```
///
/// Each step is a precondition for the next, and the sequence starts only
/// after the R2 gates have passed. No step runs here: there is no grant
/// parsing, no authorization, no tool registration, and no consent
/// evaluation in this module.
pub const ACTIVATION_ORDER_DOC: &str =
    "grant parse -> authorization -> tool registration -> consent evaluation";

/// Shared deny-all activation reason (byte-identical for every point).
///
/// Constant by construction so any two [`ActivationError::Denied`] refusals
/// compare byte-equal on the `reason` field regardless of the requested
/// point or enable list.
pub const DENY_ALL_ACTIVATION_REASON: &str = "default-deny: no activation grant";

/// Typed activation refusal.
///
/// `Denied` is a known [`ExtensionPoint`] refused fail-closed (even when the
/// caller lists it as explicitly enabled); `UnknownPoint` is a point name
/// outside the closed eight-point vocabulary, refused without synthesizing
/// a point. Neither variant grants anything and neither performs any
/// registry or resolution step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    /// Known point refused fail-closed (no auto-activate, no enable grants).
    Denied {
        /// Refused point.
        point: ExtensionPoint,
        /// Shared refusal reason ([`DENY_ALL_ACTIVATION_REASON`]).
        reason: String,
    },
    /// Point name outside the closed vocabulary (never synthesized).
    UnknownPoint {
        /// Rejected point name (stored verbatim).
        name: String,
    },
}

impl std::fmt::Display for ActivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied { point, reason } => write!(
                f,
                "extension point '{}' activation denied: {reason}",
                activation_point_name(*point)
            ),
            Self::UnknownPoint { name } => {
                write!(f, "unknown extension point: {name}")
            }
        }
    }
}

impl std::error::Error for ActivationError {}

/// Vocabulary name for one [`ExtensionPoint`].
///
/// Pure projection; no I/O, no parsing, no authorization. Match is
/// exhaustive with no wildcard so adding a ninth point breaks at compile
/// time.
#[must_use]
pub fn activation_point_name(point: ExtensionPoint) -> &'static str {
    match point {
        ExtensionPoint::Model => "model",
        ExtensionPoint::Tool => "tool",
        ExtensionPoint::Context => "context",
        ExtensionPoint::Memory => "memory",
        ExtensionPoint::Compactor => "compactor",
        ExtensionPoint::Agent => "agent",
        ExtensionPoint::Command => "command",
        ExtensionPoint::Ui => "ui",
    }
}

/// Parse the closed eight-point activation vocabulary.
///
/// # Errors
///
/// Returns [`ActivationError::UnknownPoint`] carrying the rejected (verbatim)
/// string for anything outside the closed vocabulary. Never synthesizes a
/// point.
pub fn parse_activation_point(name: &str) -> Result<ExtensionPoint, ActivationError> {
    match name {
        "model" => Ok(ExtensionPoint::Model),
        "tool" => Ok(ExtensionPoint::Tool),
        "context" => Ok(ExtensionPoint::Context),
        "memory" => Ok(ExtensionPoint::Memory),
        "compactor" => Ok(ExtensionPoint::Compactor),
        "agent" => Ok(ExtensionPoint::Agent),
        "command" => Ok(ExtensionPoint::Command),
        "ui" => Ok(ExtensionPoint::Ui),
        _ => Err(ActivationError::UnknownPoint {
            name: name.to_owned(),
        }),
    }
}

/// Typed activation refusal for one known point.
///
/// [`ActivationError::Denied`] naming the point with the shared
/// [`DENY_ALL_ACTIVATION_REASON`], so the same point denied with or without
/// an explicit enable compares byte-equal.
#[must_use]
pub fn activation_denial(point: ExtensionPoint) -> ActivationError {
    ActivationError::Denied {
        point,
        reason: DENY_ALL_ACTIVATION_REASON.to_owned(),
    }
}

/// Deny-all activation stub (no auto-activate, no state).
///
/// Holds no enabled set and performs no activation step: construction grants
/// nothing, and every check denies — known points even when the caller lists
/// them as explicitly enabled (the enable list is accepted for signature
/// compatibility and then ignored, fail-closed), unknown point names via
/// [`ActivationError::UnknownPoint`]. Zero-sized by construction: no grant
/// table, no registry, no resolution cache, no expiry queue, no state of any
/// kind.
#[derive(Debug, Default)]
pub struct ActivationStub;

impl ActivationStub {
    /// Empty deny-all stub. Pure constructor; no activation runs here and no
    /// enable list is stored.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Check one point activation: always denies.
    ///
    /// The `_explicitly_enabled` slice exists so callers pass the per-point
    /// enable list the candidate policy names; it is ignored fail-closed —
    /// a listed point still denies. No registry lookup, no resolution, no
    /// I/O.
    ///
    /// # Errors
    ///
    /// Always returns [`ActivationError::Denied`] for `point`.
    pub fn check(
        &self,
        point: ExtensionPoint,
        _explicitly_enabled: &[ExtensionPoint],
    ) -> Result<(), ActivationError> {
        Err(activation_denial(point))
    }

    /// Check one named point activation: known names deny, unknown names
    /// refuse typed.
    ///
    /// The `_explicitly_enabled` slice is ignored fail-closed like
    /// [`ActivationStub::check`]: a listed known name still denies. No
    /// registry lookup, no resolution, no I/O.
    ///
    /// # Errors
    ///
    /// Returns [`ActivationError::UnknownPoint`] for a name outside the
    /// closed vocabulary, and [`ActivationError::Denied`] for a known point
    /// (even when listed in `_explicitly_enabled`).
    pub fn check_named(
        &self,
        name: &str,
        _explicitly_enabled: &[&str],
    ) -> Result<(), ActivationError> {
        let point = parse_activation_point(name)?;
        Err(activation_denial(point))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ACTIVATION_ORDER_DOC, ActivationError, ActivationStub, AgentDescriptor, CommandDescriptor,
        CompactorDescriptor, ContextDescriptor, ContributionEnvelope, DENY_ALL_ACTIVATION_REASON,
        DeclarationEnvelope, ExtensionPoint, Manifest, MemoryDescriptor, ModelDescriptor,
        ToolDescriptor, UiDescriptor, activation_denial, activation_point_name,
        parse_activation_point,
    };

    /// Exhaustive match with no wildcard: adding a ninth point breaks this
    /// function at compile time, so the eight-point set stays closed by
    /// construction.
    fn point_name(point: ExtensionPoint) -> &'static str {
        match point {
            ExtensionPoint::Model => "model",
            ExtensionPoint::Tool => "tool",
            ExtensionPoint::Context => "context",
            ExtensionPoint::Memory => "memory",
            ExtensionPoint::Compactor => "compactor",
            ExtensionPoint::Agent => "agent",
            ExtensionPoint::Command => "command",
            ExtensionPoint::Ui => "ui",
        }
    }

    #[test]
    fn extension_point_enum_is_exhaustive() {
        let all = [
            ExtensionPoint::Model,
            ExtensionPoint::Tool,
            ExtensionPoint::Context,
            ExtensionPoint::Memory,
            ExtensionPoint::Compactor,
            ExtensionPoint::Agent,
            ExtensionPoint::Command,
            ExtensionPoint::Ui,
        ];
        assert_eq!(all.len(), 8);
        let names: Vec<&'static str> = all.iter().map(|point| point_name(*point)).collect();
        assert_eq!(
            names,
            [
                "model",
                "tool",
                "context",
                "memory",
                "compactor",
                "agent",
                "command",
                "ui"
            ]
        );
    }

    #[test]
    fn manifest_data_round_trip() {
        let manifest = Manifest {
            name: "example-extension".to_owned(),
            version: "0.1.0".to_owned(),
            host: "bitty-ai >=0.1.0".to_owned(),
            points: vec![ExtensionPoint::Tool, ExtensionPoint::Command],
        };
        let round_tripped = manifest.clone();
        assert_eq!(manifest, round_tripped);
        assert_eq!(manifest.name, "example-extension");
        assert_eq!(manifest.version, "0.1.0");
        // The host predicate is opaque data: stored verbatim, never parsed.
        assert_eq!(manifest.host, "bitty-ai >=0.1.0");
        assert!(manifest.points.contains(&ExtensionPoint::Tool));
        assert!(!manifest.points.contains(&ExtensionPoint::Model));
    }

    #[test]
    fn manifest_with_no_points_is_well_formed() {
        let manifest = Manifest {
            name: "empty-extension".to_owned(),
            version: "0.0.0".to_owned(),
            host: "bitty-ai *".to_owned(),
            points: Vec::new(),
        };
        assert!(manifest.points.is_empty());
        assert_eq!(manifest, manifest.clone());
    }

    #[test]
    fn manifest_host_predicate_is_verbatim_never_parsed() {
        // A predicate no semver parser would accept round-trips byte for
        // byte: this mirror performs no normalization or evaluation.
        let host = "bitty-ai >=0.1.0, <2.0.0 || custom-predicate (never parsed)".to_owned();
        let manifest = Manifest {
            name: "verbatim-extension".to_owned(),
            version: "0.1.0".to_owned(),
            host: host.clone(),
            points: vec![ExtensionPoint::Model],
        };
        assert_eq!(manifest.host, host);
        assert_eq!(manifest, manifest.clone());
    }

    #[test]
    fn each_descriptor_is_independent() {
        let model = ModelDescriptor::new("fake-chat".to_owned());
        let tool = ToolDescriptor::new("read-only".to_owned());
        let context = ContextDescriptor::new("workspace.read".to_owned());
        let memory = MemoryDescriptor::new("session".to_owned());
        let compactor = CompactorDescriptor::new("truncate-head".to_owned());
        let agent = AgentDescriptor::new("single-turn".to_owned());
        let command = CommandDescriptor::new("ai.ask".to_owned());
        let ui = UiDescriptor::new("panel: transcript".to_owned());

        assert_eq!(model.declared_model(), "fake-chat");
        assert_eq!(tool.effect_class(), "read-only");
        assert_eq!(context.read_scope(), "workspace.read");
        assert_eq!(memory.retention_class(), "session");
        assert_eq!(compactor.strategy_id(), "truncate-head");
        assert_eq!(agent.capability_id(), "single-turn");
        assert_eq!(command.command_name(), "ai.ask");
        assert_eq!(ui.descriptor(), "panel: transcript");
    }

    /// Separate structs, no shared supertype: a holder wanting two points
    /// declares two fields explicitly. One descriptor contributes no value
    /// of any other point's type.
    struct ToolAndModel {
        tool: ToolDescriptor,
        model: ModelDescriptor,
    }

    #[test]
    fn combined_holder_needs_one_field_per_point() {
        let both = ToolAndModel {
            tool: ToolDescriptor::new("read-only".to_owned()),
            model: ModelDescriptor::new("fake-chat".to_owned()),
        };
        // Both accessors resolve only because both fields exist above.
        assert_eq!(both.tool.effect_class(), "read-only");
        assert_eq!(both.model.declared_model(), "fake-chat");
    }

    fn example_manifest() -> Manifest {
        Manifest {
            name: "example-extension".to_owned(),
            version: "0.1.0".to_owned(),
            host: "bitty-ai >=0.1.0".to_owned(),
            points: vec![ExtensionPoint::Tool, ExtensionPoint::Command],
        }
    }

    #[test]
    fn declaration_envelope_round_trip() {
        let envelope = DeclarationEnvelope::new(example_manifest());
        let round_tripped = envelope.clone();
        assert_eq!(envelope, round_tripped);
        assert_eq!(envelope.manifest().name, "example-extension");
        assert_eq!(envelope.manifest().version, "0.1.0");
        assert_eq!(envelope.manifest().host, "bitty-ai >=0.1.0");
        assert_eq!(
            envelope.manifest().points,
            [ExtensionPoint::Tool, ExtensionPoint::Command]
        );
        // The echo is the manifest: no version field, no transport, no grant.
        assert_eq!(envelope.manifest, example_manifest());
    }

    #[test]
    fn declaration_envelope_host_is_verbatim_never_parsed() {
        // A predicate no semver parser would accept round-trips byte for
        // byte through the envelope: no normalization or evaluation.
        let host = "bitty-ai >=0.1.0, <2.0.0 || custom-predicate (never parsed)".to_owned();
        let envelope = DeclarationEnvelope::new(Manifest {
            name: "verbatim-extension".to_owned(),
            version: "0.1.0".to_owned(),
            host: host.clone(),
            points: vec![ExtensionPoint::Model],
        });
        assert_eq!(envelope.manifest().host, host);
        assert_eq!(envelope, envelope.clone());
    }

    #[test]
    fn declaration_envelope_empty_points_admits_nothing_grants_nothing() {
        let envelope = DeclarationEnvelope::new(Manifest {
            name: "empty-extension".to_owned(),
            version: "0.0.0".to_owned(),
            host: "bitty-ai *".to_owned(),
            points: Vec::new(),
        });
        assert!(envelope.manifest().points.is_empty());
        assert_eq!(envelope, envelope.clone());
    }

    /// Exhaustive match with no wildcard: adding a ninth contribution
    /// variant breaks this function at compile time, so the eight-point
    /// envelope set stays closed by construction.
    fn contribution_point_name(envelope: &ContributionEnvelope) -> &'static str {
        match envelope {
            ContributionEnvelope::Model(_) => "model",
            ContributionEnvelope::Tool(_) => "tool",
            ContributionEnvelope::Context(_) => "context",
            ContributionEnvelope::Memory(_) => "memory",
            ContributionEnvelope::Compactor(_) => "compactor",
            ContributionEnvelope::Agent(_) => "agent",
            ContributionEnvelope::Command(_) => "command",
            ContributionEnvelope::Ui(_) => "ui",
        }
    }

    fn all_contributions() -> Vec<ContributionEnvelope> {
        vec![
            ContributionEnvelope::Model(ModelDescriptor::new("fake-chat".to_owned())),
            ContributionEnvelope::Tool(ToolDescriptor::new("read-only".to_owned())),
            ContributionEnvelope::Context(ContextDescriptor::new("workspace.read".to_owned())),
            ContributionEnvelope::Memory(MemoryDescriptor::new("session".to_owned())),
            ContributionEnvelope::Compactor(CompactorDescriptor::new("truncate-head".to_owned())),
            ContributionEnvelope::Agent(AgentDescriptor::new("single-turn".to_owned())),
            ContributionEnvelope::Command(CommandDescriptor::new("ai.ask".to_owned())),
            ContributionEnvelope::Ui(UiDescriptor::new("panel: transcript".to_owned())),
        ]
    }

    #[test]
    fn contribution_envelopes_round_trip() {
        let envelopes = all_contributions();
        assert_eq!(envelopes.len(), 8);
        for envelope in &envelopes {
            assert_eq!(envelope, &envelope.clone());
        }
    }

    #[test]
    fn contribution_envelope_point_projection_is_exhaustive() {
        let envelopes = all_contributions();
        let points: Vec<ExtensionPoint> =
            envelopes.iter().map(ContributionEnvelope::point).collect();
        assert_eq!(
            points,
            [
                ExtensionPoint::Model,
                ExtensionPoint::Tool,
                ExtensionPoint::Context,
                ExtensionPoint::Memory,
                ExtensionPoint::Compactor,
                ExtensionPoint::Agent,
                ExtensionPoint::Command,
                ExtensionPoint::Ui,
            ]
        );
        let names: Vec<&'static str> = envelopes.iter().map(contribution_point_name).collect();
        assert_eq!(
            names,
            [
                "model",
                "tool",
                "context",
                "memory",
                "compactor",
                "agent",
                "command",
                "ui"
            ]
        );
    }

    #[test]
    fn contribution_envelopes_stay_independent() {
        // One variant contributes no value of any other point's type: each
        // envelope matches exactly its own variant and projects exactly its
        // own point.
        let model = ContributionEnvelope::Model(ModelDescriptor::new("fake-chat".to_owned()));
        assert_eq!(model.point(), ExtensionPoint::Model);
        assert!(matches!(model, ContributionEnvelope::Model(_)));
        assert!(!matches!(model, ContributionEnvelope::Tool(_)));

        let tool = ContributionEnvelope::Tool(ToolDescriptor::new("read-only".to_owned()));
        assert_eq!(tool.point(), ExtensionPoint::Tool);
        assert!(matches!(tool, ContributionEnvelope::Tool(_)));
        assert!(!matches!(tool, ContributionEnvelope::Model(_)));

        // A holder wanting two points sends two envelopes explicitly: one
        // envelope never stands in for another point.
        let pair = [
            ContributionEnvelope::Tool(ToolDescriptor::new("read-only".to_owned())),
            ContributionEnvelope::Model(ModelDescriptor::new("fake-chat".to_owned())),
        ];
        assert_eq!(pair[0].point(), ExtensionPoint::Tool);
        assert_eq!(pair[1].point(), ExtensionPoint::Model);
        assert_ne!(pair[0], pair[1]);
    }

    fn all_points() -> [ExtensionPoint; 8] {
        [
            ExtensionPoint::Model,
            ExtensionPoint::Tool,
            ExtensionPoint::Context,
            ExtensionPoint::Memory,
            ExtensionPoint::Compactor,
            ExtensionPoint::Agent,
            ExtensionPoint::Command,
            ExtensionPoint::Ui,
        ]
    }

    #[test]
    fn s3_activation_denies_every_known_point_even_when_explicitly_enabled() {
        let stub = ActivationStub::new();
        let all = all_points();
        // No auto-activate: an empty enable list denies every known point.
        for point in all {
            let err = stub.check(point, &[]).unwrap_err();
            assert_eq!(err, activation_denial(point));
            assert!(matches!(err, ActivationError::Denied { .. }));
            match &err {
                ActivationError::Denied {
                    point: refused,
                    reason,
                } => {
                    assert_eq!(*refused, point);
                    assert_eq!(reason, DENY_ALL_ACTIVATION_REASON);
                }
                ActivationError::UnknownPoint { .. } => {
                    panic!("known point must deny, not refuse unknown")
                }
            }
        }
        // Explicit enable per point still denies: the enable list is
        // accepted and then ignored fail-closed.
        for point in all {
            let err = stub.check(point, &all).unwrap_err();
            assert_eq!(err, activation_denial(point));
            assert!(matches!(err, ActivationError::Denied { .. }));
        }
        // A manifest declaration listing points grants nothing: passing the
        // declared points as the enable list still denies each one, and the
        // stub performs no registry or resolution step.
        let manifest = Manifest {
            name: "example-extension".to_owned(),
            version: "0.1.0".to_owned(),
            host: "bitty-ai >=0.1.0".to_owned(),
            points: vec![ExtensionPoint::Tool, ExtensionPoint::Command],
        };
        for point in &manifest.points {
            let err = stub.check(*point, &manifest.points).unwrap_err();
            assert!(matches!(err, ActivationError::Denied { .. }));
        }
    }

    #[test]
    fn s3_activation_named_check_denies_known_even_when_listed() {
        let stub = ActivationStub::new();
        let names = [
            "model",
            "tool",
            "context",
            "memory",
            "compactor",
            "agent",
            "command",
            "ui",
        ];
        for name in names {
            // Listed explicitly, yet still denied: no enable grants.
            let err = stub.check_named(name, &names).unwrap_err();
            let point = parse_activation_point(name).expect("closed name parses");
            assert_eq!(err, activation_denial(point));
            assert!(matches!(err, ActivationError::Denied { .. }));
            // Unlisted denies identically: the enable list changes nothing.
            let unlisted = stub.check_named(name, &[]).unwrap_err();
            assert_eq!(err, unlisted);
        }
    }

    #[test]
    fn s3_activation_unknown_point_is_typed_refusal() {
        let stub = ActivationStub::new();
        let unknowns = ["admin", "", "TOOL", "model ", "tool/model", "compactor2"];
        for name in unknowns {
            let err = stub.check_named(name, &[]).unwrap_err();
            assert_eq!(
                err,
                ActivationError::UnknownPoint {
                    name: name.to_owned(),
                }
            );
            assert!(matches!(err, ActivationError::UnknownPoint { .. }));
            assert!(!matches!(err, ActivationError::Denied { .. }));
            // Unknown stays unknown even when the enable list names it: the
            // stub never synthesizes a point.
            let listed = stub.check_named(name, &[name]).unwrap_err();
            assert_eq!(err, listed);
            // Direct parse agrees and keeps the rejected name verbatim.
            let parsed = parse_activation_point(name).unwrap_err();
            assert_eq!(parsed, err);
            assert!(format!("{err}").contains("unknown extension point"));
            assert!(format!("{err}").contains(name));
        }
    }

    #[test]
    fn s3_activation_vocabulary_is_closed_and_round_trips() {
        // Closed eight-point vocabulary shared with the S1 mirror: every
        // point projects to its name and every name parses back.
        let all = all_points();
        let mut names = Vec::new();
        for point in all {
            let name = activation_point_name(point);
            assert_eq!(parse_activation_point(name), Ok(point));
            names.push(name);
        }
        assert_eq!(
            names,
            [
                "model",
                "tool",
                "context",
                "memory",
                "compactor",
                "agent",
                "command",
                "ui"
            ]
        );
        // Denial keeps the point identity: same point denies byte-equal,
        // different points differ only in the point field.
        let stub = ActivationStub::new();
        let first = stub.check(ExtensionPoint::Tool, &[]).unwrap_err();
        let again = stub.check(ExtensionPoint::Tool, &[]).unwrap_err();
        assert_eq!(first, again);
        let other = stub.check(ExtensionPoint::Model, &[]).unwrap_err();
        assert_ne!(first, other);
        assert!(format!("{first}").contains("'tool'"));
        assert!(format!("{first}").contains(DENY_ALL_ACTIVATION_REASON));
    }

    #[test]
    fn s3_activation_stub_has_no_state() {
        // Zero-sized by construction: no grant table, no registry, no
        // resolution cache, no expiry queue.
        assert_eq!(std::mem::size_of::<ActivationStub>(), 0);
        let first = ActivationStub::new();
        let second = ActivationStub;
        for point in all_points() {
            // Deterministic: separate instances deny identically (no hidden
            // state, no clock, enable list ignored).
            assert_eq!(first.check(point, &[]), second.check(point, &[]));
            assert_eq!(first.check(point, &[]), first.check(point, &[]));
        }
        let probe = "tool";
        assert_eq!(
            first.check_named(probe, &[]),
            second.check_named(probe, &[])
        );
    }

    #[test]
    fn s3_activation_order_doc_states_gating_contract() {
        // Sequence document present: the four candidate stages appear once
        // and in candidate order (grant parse first, consent last).
        let doc = ACTIVATION_ORDER_DOC;
        let grant = doc.find("grant parse").expect("grant parse stage");
        let authorization = doc.find("authorization").expect("authorization stage");
        let registration = doc.find("tool registration").expect("registration stage");
        let consent = doc.find("consent evaluation").expect("consent stage");
        assert!(grant < authorization);
        assert!(authorization < registration);
        assert!(registration < consent);
        assert_eq!(
            doc,
            "grant parse -> authorization -> tool registration -> consent evaluation"
        );
    }
}

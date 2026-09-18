//! Deterministic five-layer prompt assembly with narrowing-only policy.
//!
//! Mirrors the draft `docs/specifications/prompt-layering-design.md` (status:
//! draft, not an accepted contract): five layers from most stable to most
//! dynamic — Core Contract, User, Project/`.wheel`, Skills/Profile, and
//! Runtime/Turn — assembled stable-before-dynamic and aligned with the
//! prefix-cache layering in
//! `docs/specifications/prefix-cache-context-design.md`.
//!
//! # Precedence and the no-grant rule
//!
//! Precedence is `Core > User > Project > Skills/Profile > Runtime/Turn`.
//! Later layers narrow, never widen:
//!
//! - Text sections concatenate in stable order; text never grants.
//! - `denied_tools` union wins: a lower layer cannot clear an upper denial.
//! - `allowed_tools` (`None` = no constraint) intersect: a lower layer cannot
//!   add a tool the upper set omits.
//! - `budget_ceiling_bytes` takes the minimum: a lower layer cannot raise the
//!   ceiling.
//! - `allowed_scopes` (`None` = no constraint) intersect: a lower layer
//!   cannot add a scope the upper set omits.
//! - Generic `directives` with the same key but different values resolve by
//!   precedence: the higher-precedence (lower-rank) layer wins, and every
//!   override is recorded in [`AssembledPrompt::merge_overrides`] and in the
//!   canonical bytes. Same-layer conflicts (no precedence difference) and
//!   conflicts on [`NEVER_MERGE_DIRECTIVE_KEYS`] stay fail-closed with
//!   [`PromptError::UnresolvableConflict`].
//!
//! > **A prompt never grants a capability.** Prompt text describes what should
//! > be done; the dispatcher decides what can be done. Even when the prompt
//! > allows a tool, [`check_dispatch`] still requires the dispatcher grant;
//! > when the dispatcher denies, dispatch is refused. When the prompt denies
//! > or omits (under a constrained allow-set), dispatch is refused even when
//! > the dispatcher would grant. All refusals are fail-closed with no partial
//! > assembly or dispatch.
//!
//! # Directive precedence and the never-merge list
//!
//! When two layers set the same directive key to different values, the
//! higher-precedence layer (smaller [`PromptLayer::rank`]) wins: Core
//! overrides User, User overrides Project, and so on down to Runtime/Turn.
//! Same-value duplicates merge silently with no record. Every cross-layer
//! override appends one [`DirectiveOverride`] per overridden layer to
//! [`AssembledPrompt::merge_overrides`] (sorted by key, overridden rank and
//! value, then winning rank and value) and to the canonical bytes as a
//! sorted `override:` line, so a reviewer can audit exactly which lower-layer value lost and to which
//! upper-layer value.
//!
//! [`NEVER_MERGE_DIRECTIVE_KEYS`] names the minimal policy set of keys that
//! never resolve by precedence: `agent.identity` (layer-asserted identity;
//! a silent override would let a lower, less-trusted layer re-label the
//! agent or mask which layer spoke) and `capability.grant` (grant-shaped
//! advisory text; a silent merge would blur the prompt-never-grants
//! boundary). Conflicts on those keys — and same-layer conflicts with no
//! precedence difference — stay fail-closed with
//! [`PromptError::UnresolvableConflict`].
//!
//! # Deterministic serialization (AIQ-12 mechanism evidence)
//!
//! [`assemble`] produces [`AssembledPrompt::canonical_bytes`], a canonical
//! byte form with fixed layer order, fixed field order, LF-only newlines,
//! length-prefixed text sections, and lexicographically sorted policy lists.
//! The same [`PromptSnapshot`] (regardless of input layer order) yields
//! identical bytes; changing only the trailing Runtime/Turn layer preserves
//! the leading prefix bytes, which is the stable-before-dynamic property the
//! prefix-cache design needs.
//!
//! This module provides enforcement evidence toward AIQ-12 ("Canonical
//! serialization and stable-prefix ordering — Design: deterministic encoding
//! is prerequisite for any prefix-cache claim") and toward the AIQ-31/AIQ-34
//! merge-semantics facet (precedence-override + audit-record evidence). It
//! does not close any AIQ: canonical owner, digest algorithm, cache-key
//! scope, epoch schema, and reviewer acceptance stay open in the register.
//!
//! # Bounds and determinism rules
//!
//! - Std only. No network, filesystem, clock, threads, async runtime, or
//!   secrets. Single-agent (v0.1): profiles compose one agent only.
//! - Every bound is enforced fail-closed with [`PromptError`]; no partial
//!   output is returned on error.
//! - Missing layers assemble as empty text with no constraints, so the
//!   canonical form always carries exactly five text sections in rank order.
//! - Declarative loaders ([`LayerInput::project_from_str`],
//!   [`LayerInput::project_from_files_under_roots`],
//!   [`LayerInput::skills_from_str`]) are pure `&str`-in / [`LayerInput`]-out:
//!   the host owns all filesystem reads and passes bytes in; this module
//!   performs no filesystem I/O, never walks directories, and never follows
//!   symlinks. [`admit_project_path`] fail-closes any lexical root escape.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Maximum prompt text bytes per layer.
pub const MAX_LAYER_TEXT_BYTES: usize = 16 * 1024;
/// Maximum directives per layer.
pub const MAX_DIRECTIVES_PER_LAYER: usize = 32;
/// Maximum directive key length in bytes.
pub const MAX_DIRECTIVE_KEY_LEN: usize = 64;
/// Maximum directive value length in bytes.
pub const MAX_DIRECTIVE_VALUE_LEN: usize = 256;
/// Maximum tool entries per list (`allowed_tools`, `denied_tools`) per layer.
pub const MAX_TOOL_ENTRIES_PER_LAYER: usize = 32;
/// Maximum scopes per layer when constrained.
pub const MAX_SCOPES_PER_LAYER: usize = 16;
/// Maximum scope string length in bytes.
pub const MAX_SCOPE_LEN: usize = 128;
/// Maximum core-version string length in bytes.
pub const MAX_CORE_VERSION_LEN: usize = 64;
/// Maximum budget ceiling in bytes (upper bound for any layer ceiling).
pub const MAX_BUDGET_CEILING_BYTES: usize = 256 * 1024;
/// Maximum canonical byte form length in bytes.
pub const MAX_CANONICAL_BYTES: usize = 96 * 1024;
/// Maximum declarative project files merged into one Project layer.
pub const MAX_PROJECT_FILES: usize = 16;
/// Maximum bytes per declarative project file.
pub const MAX_PROJECT_FILE_BYTES: usize = 8 * 1024;
/// Maximum path length in bytes for a declarative project file candidate.
pub const MAX_PROJECT_PATH_LEN: usize = 256;
/// Maximum skill/profile entries merged into one SkillsProfile layer.
pub const MAX_SKILL_ENTRIES: usize = 16;
/// Maximum bytes per skill/profile fragment.
pub const MAX_SKILL_ENTRY_BYTES: usize = 8 * 1024;
/// Maximum bytes for a whole skill/profile registry document.
pub const MAX_SKILL_REGISTRY_BYTES: usize = 64 * 1024;
/// Maximum skill/profile entry name length in bytes.
pub const MAX_SKILL_NAME_LEN: usize = 64;
/// Maximum skill/profile version string length in bytes.
pub const MAX_SKILL_VERSION_LEN: usize = 32;
/// Current skill/profile registry format version (the only supported one).
pub const SKILL_REGISTRY_VERSION_1: &str = "1";
/// Supported skill/profile format versions (exactly one today).
pub const SUPPORTED_SKILL_VERSIONS: &[&str] = &[SKILL_REGISTRY_VERSION_1];

/// Sentinel tool name carried by [`PromptError::PromptNotAllowed`] when
/// [`assemble`] itself observes an empty effective allow-set (`Some([])`).
///
/// Assembly-time empty intersection (an explicit empty list or disjoint
/// per-layer allow-sets) has no single dispatched tool to name, while the
/// existing `PromptNotAllowed` variant requires a `tool` field. This sentinel
/// reuses that variant per S-13 without adding a new error variant: it
/// contains `<`, space, and `>` so it can never collide with a `TB-2` tool
/// name (`^[a-z][a-z0-9_]*$`), making the assembly-time origin unambiguous.
pub const EMPTY_ALLOW_SET_SENTINEL: &str = "<empty allow-set>";

/// Prompt layer from most stable (Core) to most dynamic (Runtime/Turn).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromptLayer {
    /// Built-in runtime contract: versioned, session-pinned, non-overridable.
    CoreContract,
    /// User-owned global instructions; preferences only, no capability effect.
    User,
    /// Project-owned `.wheel` manifest intent; untrusted until reviewed.
    Project,
    /// Single-agent skills/profile pack (v0.1 single-agent only).
    SkillsProfile,
    /// Per-turn facts plus the current turn instruction (most dynamic).
    RuntimeTurn,
}

impl PromptLayer {
    /// Stable rank: lower assembles first (0 = Core, 4 = Runtime/Turn).
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            Self::CoreContract => 0,
            Self::User => 1,
            Self::Project => 2,
            Self::SkillsProfile => 3,
            Self::RuntimeTurn => 4,
        }
    }

    /// Short stable name used in diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::CoreContract => "core-contract",
            Self::User => "user",
            Self::Project => "project",
            Self::SkillsProfile => "skills-profile",
            Self::RuntimeTurn => "runtime-turn",
        }
    }

    /// Canonical section header name (same as [`PromptLayer::name`]).
    #[must_use]
    pub fn header(self) -> &'static str {
        self.name()
    }

    /// All layers in stable assembly order.
    #[must_use]
    pub fn stable_order() -> [Self; 5] {
        [
            Self::CoreContract,
            Self::User,
            Self::Project,
            Self::SkillsProfile,
            Self::RuntimeTurn,
        ]
    }

    /// Parse a stable header name back to a layer (canonical round-trip).
    #[must_use]
    pub fn from_header(header: &str) -> Option<Self> {
        match header {
            "core-contract" => Some(Self::CoreContract),
            "user" => Some(Self::User),
            "project" => Some(Self::Project),
            "skills-profile" => Some(Self::SkillsProfile),
            "runtime-turn" => Some(Self::RuntimeTurn),
            _ => None,
        }
    }
}

impl Display for PromptLayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.name())
    }
}

/// One generic key-value directive within a layer.
///
/// Directives carry advisory text policy (for example `tone = concise`).
/// They never grant capability. The same key with the same value across
/// layers merges to one entry; the same key with different values resolves
/// by layer precedence (higher-precedence layer wins, recorded in
/// [`AssembledPrompt::merge_overrides`]), except keys in
/// [`NEVER_MERGE_DIRECTIVE_KEYS`] and same-layer conflicts, which fail
/// closed at assembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Directive {
    /// Directive key (`^[a-z][a-z0-9_.-]*$`, bounded).
    pub key: String,
    /// Directive value (bounded, no newlines, no NUL).
    pub value: String,
}

impl Directive {
    /// Construct and validate one directive.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for a malformed key or value.
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Result<Self, PromptError> {
        let key = key.into();
        let value = value.into();
        validate_directive_key(&key)?;
        validate_directive_value(&key, &value)?;
        Ok(Self { key, value })
    }
}

/// Directive keys that never resolve by precedence (minimal policy set).
///
/// A same-key-different-value conflict on any of these keys stays
/// fail-closed with [`PromptError::UnresolvableConflict`] even across
/// layers with a clear precedence difference:
///
/// - `agent.identity`: layer-asserted identity. A silent override would let
///   a lower (less trusted) layer re-label the agent or mask which layer
///   spoke for a directive.
/// - `capability.grant`: grant-shaped advisory text. A silent merge would
///   blur the prompt-never-grants boundary by letting directive text look
///   authoritative about capabilities.
///
/// Same-value duplicates on these keys still merge silently (no conflict,
/// no audit record); only differing values fail closed. Keep this list
/// minimal: every entry must be justified as identity- or
/// capability-adjacent, and additions are a policy change requiring review.
pub const NEVER_MERGE_DIRECTIVE_KEYS: &[&str] = &["agent.identity", "capability.grant"];

/// Whether `key` is a never-merge directive key (see
/// [`NEVER_MERGE_DIRECTIVE_KEYS`]).
#[must_use]
pub fn is_never_merge_directive_key(key: &str) -> bool {
    NEVER_MERGE_DIRECTIVE_KEYS.contains(&key)
}

/// One recorded precedence override from directive merging.
///
/// When a higher-precedence layer and a lower-precedence layer set the same
/// (mergeable) directive key to different values, the higher-precedence
/// value wins and one record per overridden layer is stored on
/// [`AssembledPrompt::merge_overrides`] and rendered into the canonical
/// bytes. Records sort by (key, overridden-layer rank, overridden value,
/// winning-layer rank, winning value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectiveOverride {
    /// Conflicting directive key.
    pub key: String,
    /// Winning value (from the higher-precedence layer).
    pub winning_value: String,
    /// Layer that supplied the winning value.
    pub winning_layer: PromptLayer,
    /// Overridden value (from the lower-precedence layer).
    pub overridden_value: String,
    /// Layer that supplied the overridden value.
    pub overridden_layer: PromptLayer,
}

/// One layer's contribution to a prompt snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerInput {
    /// Which layer this input fills.
    pub layer: PromptLayer,
    /// Prompt text for this layer (bounded, LF-only, no NUL).
    pub text: String,
    /// Narrowing allow-set (`None` = no constraint from this layer).
    /// Each entry must satisfy the tool-name shape; sorted/deduped at
    /// assembly and intersected across layers.
    pub allowed_tools: Option<Vec<String>>,
    /// Deny-set for this layer; unioned across layers (deny wins).
    pub denied_tools: Vec<String>,
    /// Budget ceiling from this layer; the minimum across layers wins.
    pub budget_ceiling_bytes: Option<usize>,
    /// Narrowing scope-set (`None` = no constraint from this layer);
    /// intersected across layers.
    pub allowed_scopes: Option<Vec<String>>,
    /// Generic directives for this layer.
    pub directives: Vec<Directive>,
}

impl LayerInput {
    /// Minimal layer input: text only, no structured constraints.
    #[must_use]
    pub fn text_only(layer: PromptLayer, text: impl Into<String>) -> Self {
        Self {
            layer,
            text: text.into(),
            allowed_tools: None,
            denied_tools: Vec::new(),
            budget_ceiling_bytes: None,
            allowed_scopes: None,
            directives: Vec::new(),
        }
    }

    /// Validate bounds and shapes fail-closed (no partial acceptance).
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for over-bound or malformed fields.
    pub fn validate(&self) -> Result<(), PromptError> {
        validate_layer_text(self.layer, &self.text)?;
        if let Some(allowed) = &self.allowed_tools {
            if allowed.len() > MAX_TOOL_ENTRIES_PER_LAYER {
                return Err(PromptError::TooManyTools {
                    layer: self.layer,
                    limit: MAX_TOOL_ENTRIES_PER_LAYER,
                });
            }
            for name in allowed {
                validate_prompt_tool_name(name)?;
            }
        }
        if self.denied_tools.len() > MAX_TOOL_ENTRIES_PER_LAYER {
            return Err(PromptError::TooManyTools {
                layer: self.layer,
                limit: MAX_TOOL_ENTRIES_PER_LAYER,
            });
        }
        for name in &self.denied_tools {
            validate_prompt_tool_name(name)?;
        }
        if let Some(budget) = self.budget_ceiling_bytes {
            validate_budget_ceiling(budget)?;
        }
        if let Some(scopes) = &self.allowed_scopes {
            if scopes.len() > MAX_SCOPES_PER_LAYER {
                return Err(PromptError::TooManyScopes {
                    layer: self.layer,
                    limit: MAX_SCOPES_PER_LAYER,
                });
            }
            for scope in scopes {
                validate_scope(scope)?;
            }
        }
        if self.directives.len() > MAX_DIRECTIVES_PER_LAYER {
            return Err(PromptError::TooManyDirectives {
                layer: self.layer,
                limit: MAX_DIRECTIVES_PER_LAYER,
            });
        }
        for directive in &self.directives {
            validate_directive_key(&directive.key)?;
            validate_directive_value(&directive.key, &directive.value)?;
        }
        Ok(())
    }
}

/// Caller-supplied prompt snapshot: one pinned core version plus up to one
/// input per layer. Input order is insignificant; assembly sorts by rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSnapshot {
    /// Pinned core-contract version (for example `bitty-core-prompt@1`);
    /// recorded for replay and audit. The sketch is illustrative, not an
    /// adopted naming scheme.
    pub core_version: String,
    /// Layer inputs in any order (at most one per [`PromptLayer`]).
    pub layers: Vec<LayerInput>,
}

impl PromptSnapshot {
    /// Construct and validate a snapshot (duplicate layers fail closed).
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for a bad core version, a duplicate layer,
    /// or any invalid layer input.
    pub fn new(
        core_version: impl Into<String>,
        layers: Vec<LayerInput>,
    ) -> Result<Self, PromptError> {
        let snapshot = Self {
            core_version: core_version.into(),
            layers,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Validate without constructing the assembled form.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for a bad core version, a duplicate layer,
    /// too many layers, or any invalid layer input.
    pub fn validate(&self) -> Result<(), PromptError> {
        validate_core_version(&self.core_version)?;
        if self.layers.len() > PromptLayer::stable_order().len() {
            return Err(PromptError::TooManyLayers {
                limit: PromptLayer::stable_order().len(),
            });
        }
        let mut seen = [false; 5];
        for input in &self.layers {
            let rank = input.layer.rank() as usize;
            if seen[rank] {
                return Err(PromptError::DuplicateLayer { layer: input.layer });
            }
            seen[rank] = true;
            input.validate()?;
        }
        Ok(())
    }
}

/// One assembled text section in stable order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledSection {
    /// Layer for this section.
    pub layer: PromptLayer,
    /// Text for this layer (empty when the snapshot omits the layer).
    pub text: String,
}

/// Deterministic assembly result for one snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledPrompt {
    /// Pinned core version from the snapshot.
    pub core_version: String,
    /// Five text sections in stable rank order.
    pub sections: Vec<AssembledSection>,
    /// Effective allow-set (`None` = prompt places no allow constraint;
    /// dispatch still needs the dispatcher grant). Sorted, deduped.
    pub effective_allowed_tools: Option<Vec<String>>,
    /// Effective deny-set (union across layers). Sorted, deduped.
    pub effective_denied_tools: Vec<String>,
    /// Effective budget ceiling (minimum across layers, when any).
    pub effective_budget_ceiling_bytes: Option<usize>,
    /// Effective scope-set (`None` = no prompt scope constraint). Sorted.
    pub effective_scopes: Option<Vec<String>>,
    /// Effective directives sorted by key (deduped on equal values).
    pub effective_directives: Vec<Directive>,
    /// Precedence override audit record, sorted by (key, overridden-layer
    /// rank, overridden value, winning-layer rank, winning value). Empty
    /// when no cross-layer directive value conflict occurred. Also rendered
    /// into [`AssembledPrompt::canonical_bytes`] as sorted `override:` lines
    /// (absent when empty, so override-free snapshots keep byte-identical
    /// canonical forms).
    pub merge_overrides: Vec<DirectiveOverride>,
    /// Canonical byte form (AIQ-12 mechanism evidence). See module docs.
    pub canonical_bytes: Vec<u8>,
}

impl AssembledPrompt {
    /// Borrow the canonical byte form.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    /// Borrow the text of one layer (always present; possibly empty).
    #[must_use]
    pub fn section_text(&self, layer: PromptLayer) -> &str {
        self.sections
            .iter()
            .find(|section| section.layer == layer)
            .map(|section| section.text.as_str())
            .unwrap_or("")
    }
}

/// Prompt assembly and dispatch errors. Every variant fails closed with no
/// partial assembly or dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptError {
    /// Core version violates the pinned-version shape.
    InvalidCoreVersion {
        /// Rejected version string.
        version: String,
    },
    /// More than one input names the same layer.
    DuplicateLayer {
        /// Duplicated layer.
        layer: PromptLayer,
    },
    /// More than five layer inputs were supplied.
    TooManyLayers {
        /// Bound (always 5).
        limit: usize,
    },
    /// Layer text exceeds [`MAX_LAYER_TEXT_BYTES`].
    LayerTextTooLarge {
        /// Owning layer.
        layer: PromptLayer,
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Layer text carries non-canonical bytes (`\r` or NUL).
    InvalidLayerText {
        /// Owning layer.
        layer: PromptLayer,
        /// Why the text was rejected.
        reason: String,
    },
    /// Tool entry violates the tool-name shape.
    InvalidToolName {
        /// Rejected name.
        name: String,
    },
    /// A tool list exceeds [`MAX_TOOL_ENTRIES_PER_LAYER`].
    TooManyTools {
        /// Owning layer.
        layer: PromptLayer,
        /// Bound.
        limit: usize,
    },
    /// Scope entry violates the scope shape.
    InvalidScope {
        /// Rejected scope.
        scope: String,
    },
    /// A scope list exceeds [`MAX_SCOPES_PER_LAYER`].
    TooManyScopes {
        /// Owning layer.
        layer: PromptLayer,
        /// Bound.
        limit: usize,
    },
    /// Directive key violates its shape.
    InvalidDirectiveKey {
        /// Rejected key.
        key: String,
    },
    /// Directive value violates its bound/shape.
    InvalidDirectiveValue {
        /// Owning key (for diagnostics).
        key: String,
    },
    /// A directive list exceeds [`MAX_DIRECTIVES_PER_LAYER`].
    TooManyDirectives {
        /// Owning layer.
        layer: PromptLayer,
        /// Bound.
        limit: usize,
    },
    /// Budget ceiling is zero or exceeds [`MAX_BUDGET_CEILING_BYTES`], or the
    /// assembled canonical bytes exceed the effective budget ceiling (the
    /// minimum across layers). The second use reuses this variant per S-13
    /// without adding a new one: `actual` then carries the observed canonical
    /// length that overran the effective budget.
    InvalidBudget {
        /// Rejected ceiling, or observed canonical length when the effective
        /// budget is overrun at assembly.
        actual: usize,
    },
    /// The same directive key carries different values with no precedence
    /// resolution: either the key is in [`NEVER_MERGE_DIRECTIVE_KEYS`] or
    /// both values come from the same layer (no precedence difference). The
    /// assembler refuses to guess and fails closed. Cross-layer conflicts on
    /// mergeable keys instead resolve by precedence and are recorded in
    /// [`AssembledPrompt::merge_overrides`].
    UnresolvableConflict {
        /// Conflicting key.
        key: String,
        /// First observed value.
        first: String,
        /// Conflicting value.
        second: String,
    },
    /// Canonical form exceeds [`MAX_CANONICAL_BYTES`].
    CanonicalTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Dispatch refused: the dispatcher (capability registry / runtime
    /// policy) does not grant this tool. The prompt never overrides this.
    DispatcherDenied {
        /// Requested tool.
        tool: String,
    },
    /// Dispatch refused: the assembled prompt denies this tool (deny wins).
    PromptDenied {
        /// Requested tool.
        tool: String,
    },
    /// Dispatch refused: the assembled prompt constrains dispatch to an
    /// allow-set that omits this tool. When [`assemble`] itself observes an
    /// empty effective allow-set (`Some([])` from an explicit empty list or
    /// a disjoint intersection), it fails closed with this variant carrying
    /// [`EMPTY_ALLOW_SET_SENTINEL`], which can never be a real tool name.
    PromptNotAllowed {
        /// Requested tool, or [`EMPTY_ALLOW_SET_SENTINEL`] for an
        /// assembly-time empty allow-set with no single tool to name.
        tool: String,
    },
    /// Declarative project path rejected: not lexically under any
    /// caller-supplied root, or carries an unsafe shape (empty, absolute
    /// without a matching root, `.`/`..` segment, empty segment,
    /// backslash, CR, or NUL). Fail-closed with no file admitted.
    InvalidProjectPath {
        /// Rejected candidate path.
        path: String,
    },
    /// Declarative project file violates the documented line format
    /// (unknown header key, missing `=`, bad budget, duplicate `text:`,
    /// duplicate directive with different values, CR/NUL bytes, ...).
    MalformedProject {
        /// Why the file was rejected.
        reason: String,
    },
    /// More than [`MAX_PROJECT_FILES`] project files supplied for one layer.
    TooManyProjectFiles {
        /// Bound.
        limit: usize,
    },
    /// One declarative project file exceeds [`MAX_PROJECT_FILE_BYTES`].
    ProjectFileTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Skill/profile registry document violates the documented line format.
    MalformedSkill {
        /// Why the document was rejected.
        reason: String,
    },
    /// Skill/profile entry without the required `version` field.
    MissingSkillVersion {
        /// Entry name (or `<unknown>` when the name is missing too).
        name: String,
    },
    /// Skill/profile version that is not in [`SUPPORTED_SKILL_VERSIONS`].
    UnsupportedSkillVersion {
        /// Rejected version string.
        version: String,
    },
    /// More than [`MAX_SKILL_ENTRIES`] skill/profile entries supplied.
    TooManySkills {
        /// Bound.
        limit: usize,
    },
    /// One skill/profile fragment exceeds [`MAX_SKILL_ENTRY_BYTES`].
    SkillEntryTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Whole skill/profile registry exceeds [`MAX_SKILL_REGISTRY_BYTES`].
    SkillRegistryTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Two skill/profile entries share the same name.
    DuplicateSkill {
        /// Duplicated entry name.
        name: String,
    },
}

impl Display for PromptError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidCoreVersion { version } => {
                write!(f, "invalid core version: {version}")
            }
            Self::DuplicateLayer { layer } => {
                write!(f, "duplicate prompt layer: {layer}")
            }
            Self::TooManyLayers { limit } => {
                write!(f, "too many prompt layers (max {limit})")
            }
            Self::LayerTextTooLarge {
                layer,
                limit,
                actual,
            } => write!(
                f,
                "prompt layer {layer} text of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::InvalidLayerText { layer, reason } => {
                write!(f, "invalid prompt layer {layer} text: {reason}")
            }
            Self::InvalidToolName { name } => write!(f, "invalid prompt tool name: {name}"),
            Self::TooManyTools { layer, limit } => {
                write!(f, "prompt layer {layer} exceeds {limit} tool entries")
            }
            Self::InvalidScope { scope } => write!(f, "invalid prompt scope: {scope}"),
            Self::TooManyScopes { layer, limit } => {
                write!(f, "prompt layer {layer} exceeds {limit} scopes")
            }
            Self::InvalidDirectiveKey { key } => write!(f, "invalid directive key: {key}"),
            Self::InvalidDirectiveValue { key } => {
                write!(f, "invalid directive value for key: {key}")
            }
            Self::TooManyDirectives { layer, limit } => {
                write!(f, "prompt layer {layer} exceeds {limit} directives")
            }
            Self::InvalidBudget { actual } => write!(
                f,
                "invalid prompt budget ceiling {actual} (must be 1..={MAX_BUDGET_CEILING_BYTES})"
            ),
            Self::UnresolvableConflict { key, first, second } => write!(
                f,
                "unresolvable directive conflict for '{key}': '{first}' vs '{second}'"
            ),
            Self::CanonicalTooLarge { limit, actual } => write!(
                f,
                "canonical prompt of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::DispatcherDenied { tool } => {
                write!(f, "dispatcher denies tool '{tool}': prompt never grants")
            }
            Self::PromptDenied { tool } => {
                write!(f, "prompt denies tool '{tool}'")
            }
            Self::PromptNotAllowed { tool } => {
                write!(f, "prompt allow-set omits tool '{tool}'")
            }
            Self::InvalidProjectPath { path } => {
                write!(f, "project path escapes caller-supplied roots: '{path}'")
            }
            Self::MalformedProject { reason } => {
                write!(f, "malformed project file: {reason}")
            }
            Self::TooManyProjectFiles { limit } => {
                write!(f, "too many project files (max {limit})")
            }
            Self::ProjectFileTooLarge { limit, actual } => write!(
                f,
                "project file of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::MalformedSkill { reason } => {
                write!(f, "malformed skill registry: {reason}")
            }
            Self::MissingSkillVersion { name } => {
                write!(f, "skill entry '{name}' misses required version field")
            }
            Self::UnsupportedSkillVersion { version } => {
                write!(f, "unsupported skill version: '{version}'")
            }
            Self::TooManySkills { limit } => {
                write!(f, "too many skill entries (max {limit})")
            }
            Self::SkillEntryTooLarge { limit, actual } => write!(
                f,
                "skill fragment of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::SkillRegistryTooLarge { limit, actual } => write!(
                f,
                "skill registry of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::DuplicateSkill { name } => {
                write!(f, "duplicate skill entry: '{name}'")
            }
        }
    }
}

impl std::error::Error for PromptError {}

/// Validate a pinned core version: non-empty, at most
/// [`MAX_CORE_VERSION_LEN`] bytes, `^[a-z][a-z0-9_.@-]*$`.
///
/// # Errors
///
/// Returns [`PromptError::InvalidCoreVersion`] when the shape is violated.
pub fn validate_core_version(version: &str) -> Result<(), PromptError> {
    let valid = !version.is_empty()
        && version.len() <= MAX_CORE_VERSION_LEN
        && version
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_lowercase())
        && version.bytes().all(|b| {
            b.is_ascii_lowercase()
                || b.is_ascii_digit()
                || b == b'-'
                || b == b'_'
                || b == b'.'
                || b == b'@'
        });
    if valid {
        Ok(())
    } else {
        Err(PromptError::InvalidCoreVersion {
            version: version.to_owned(),
        })
    }
}

/// Validate one layer text: at most [`MAX_LAYER_TEXT_BYTES`] bytes, LF-only
/// (no `\r`), no NUL.
///
/// # Errors
///
/// Returns [`PromptError::LayerTextTooLarge`] or
/// [`PromptError::InvalidLayerText`] on violation.
pub fn validate_layer_text(layer: PromptLayer, text: &str) -> Result<(), PromptError> {
    if text.len() > MAX_LAYER_TEXT_BYTES {
        return Err(PromptError::LayerTextTooLarge {
            layer,
            limit: MAX_LAYER_TEXT_BYTES,
            actual: text.len(),
        });
    }
    if text.contains('\r') {
        return Err(PromptError::InvalidLayerText {
            layer,
            reason: "text must use LF newlines only (no CR)".to_owned(),
        });
    }
    if text.contains('\0') {
        return Err(PromptError::InvalidLayerText {
            layer,
            reason: "text must not contain NUL".to_owned(),
        });
    }
    Ok(())
}

/// Validate a prompt tool entry against the `TB-2` name shape
/// (`^[a-z][a-z0-9_]*$`, bounded).
///
/// # Errors
///
/// Returns [`PromptError::InvalidToolName`] when the shape is violated.
pub fn validate_prompt_tool_name(name: &str) -> Result<(), PromptError> {
    match crate::tool::validate_tool_name(name) {
        Ok(()) => Ok(()),
        Err(_) => Err(PromptError::InvalidToolName {
            name: name.to_owned(),
        }),
    }
}

/// Validate a skill/profile entry name: non-empty, at most
/// [`MAX_SKILL_NAME_LEN`] bytes, `^[a-z][a-z0-9_-]*$`.
///
/// Independent of the tool namespace ([`validate_prompt_tool_name`] / `TB-2`,
/// which forbids `-`): skill names allow hyphen so conventional names like
/// `my-skill` are accepted without widening the tool charset. Dots stay
/// rejected in both namespaces. Invalid names reuse
/// [`PromptError::InvalidToolName`] to avoid a new variant for the same shape
/// failure; the charset difference lives in the validator, not the error
/// type.
///
/// # Errors
///
/// Returns [`PromptError::InvalidToolName`] when the shape is violated.
pub fn validate_skill_name(name: &str) -> Result<(), PromptError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_SKILL_NAME_LEN
        && name.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if valid {
        Ok(())
    } else {
        Err(PromptError::InvalidToolName {
            name: name.to_owned(),
        })
    }
}

/// Validate a scope string: non-empty, at most [`MAX_SCOPE_LEN`] bytes,
/// `^[a-z][a-z0-9_.-]*$` (for example `workspace.read`).
///
/// # Errors
///
/// Returns [`PromptError::InvalidScope`] when the shape is violated.
pub fn validate_scope(scope: &str) -> Result<(), PromptError> {
    let valid = !scope.is_empty()
        && scope.len() <= MAX_SCOPE_LEN
        && scope.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && scope.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.' || b == b'-'
        });
    if valid {
        Ok(())
    } else {
        Err(PromptError::InvalidScope {
            scope: scope.to_owned(),
        })
    }
}

/// Validate a directive key: non-empty, at most [`MAX_DIRECTIVE_KEY_LEN`]
/// bytes, `^[a-z][a-z0-9_.-]*$`, no `=`, no newlines.
///
/// # Errors
///
/// Returns [`PromptError::InvalidDirectiveKey`] when the shape is violated.
pub fn validate_directive_key(key: &str) -> Result<(), PromptError> {
    let valid = !key.is_empty()
        && key.len() <= MAX_DIRECTIVE_KEY_LEN
        && !key.contains(['=', '\n', '\r', '\0'])
        && key.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && key.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.' || b == b'-'
        });
    if valid {
        Ok(())
    } else {
        Err(PromptError::InvalidDirectiveKey {
            key: key.to_owned(),
        })
    }
}

/// Validate a directive value: at most [`MAX_DIRECTIVE_VALUE_LEN`] bytes, no
/// newlines, no NUL.
///
/// # Errors
///
/// Returns [`PromptError::InvalidDirectiveValue`] when the shape is violated.
pub fn validate_directive_value(key: &str, value: &str) -> Result<(), PromptError> {
    let valid = value.len() <= MAX_DIRECTIVE_VALUE_LEN && !value.contains(['\n', '\r', '\0']);
    if valid {
        Ok(())
    } else {
        Err(PromptError::InvalidDirectiveValue {
            key: key.to_owned(),
        })
    }
}

/// Validate a budget ceiling: `1..=[`MAX_BUDGET_CEILING_BYTES`]`.
///
/// # Errors
///
/// Returns [`PromptError::InvalidBudget`] when out of range.
pub fn validate_budget_ceiling(ceiling: usize) -> Result<(), PromptError> {
    if (1..=MAX_BUDGET_CEILING_BYTES).contains(&ceiling) {
        Ok(())
    } else {
        Err(PromptError::InvalidBudget { actual: ceiling })
    }
}

/// Length of the common byte prefix of `left` and `right`.
///
/// Deterministic helper for prefix-stability evidence: changing only the
/// trailing Runtime/Turn layer must preserve the leading prefix bytes.
#[must_use]
pub fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let mut len = 0usize;
    for (a, b) in left.iter().zip(right.iter()) {
        if a != b {
            break;
        }
        len += 1;
    }
    len
}

/// Assemble `snapshot` into its deterministic [`AssembledPrompt`].
///
/// Policy, in order: validate the snapshot fail-closed (bad version,
/// duplicate layer, over-bound or malformed input); order layers by stable
/// rank (input order is insignificant); fill missing layers with empty text
/// and no constraints; merge `denied_tools` by union, `allowed_tools` and
/// `allowed_scopes` by intersection (`None` = no constraint), budgets by
/// minimum, and directives by key with precedence override (higher-precedence
/// layer wins, one [`DirectiveOverride`] per overridden layer) except
/// [`NEVER_MERGE_DIRECTIVE_KEYS`] and same-layer conflicts, which fail
/// closed; fail closed on an empty effective allow-set (`Some([])`) and on a
/// canonical form that exceeds the effective budget; then render the
/// canonical byte form.
///
/// Content text is never interpreted: no keyword, directive, or instruction
/// scan runs over layer text, and text bytes never select, widen, or
/// reallocate policy. Enforcement evidence toward AIQ-12 (and the AIQ-31/34
/// merge facet); no register entry is closed.
///
/// # Errors
///
/// Returns [`PromptError`] for any validation failure, an unresolvable
/// directive conflict, an empty effective allow-set
/// ([`PromptError::PromptNotAllowed`] with [`EMPTY_ALLOW_SET_SENTINEL`]), a
/// canonical form that exceeds the effective budget
/// ([`PromptError::InvalidBudget`]), or an over-bound canonical form. No
/// partial assembly is returned.
pub fn assemble(snapshot: &PromptSnapshot) -> Result<AssembledPrompt, PromptError> {
    snapshot.validate()?;

    // Rank-order the inputs; input order never affects output bytes.
    let mut ordered: Vec<&LayerInput> = snapshot.layers.iter().collect();
    ordered.sort_by_key(|input| input.layer.rank());

    // Fill the five stable sections (missing layers are empty/unconstrained).
    let mut sections = Vec::with_capacity(5);
    for layer in PromptLayer::stable_order() {
        let text = ordered
            .iter()
            .find(|input| input.layer == layer)
            .map(|input| input.text.clone())
            .unwrap_or_default();
        sections.push(AssembledSection { layer, text });
    }

    // Denied: union (deny wins). Sorted + deduped for determinism.
    let mut denied: Vec<String> = ordered
        .iter()
        .flat_map(|input| input.denied_tools.iter().cloned())
        .collect();
    denied.sort();
    denied.dedup();

    // Allowed: intersection over layers that constrain (None = skip).
    // P2-3: an empty intersection (explicit `Some([])` or disjoint sets)
    // fails closed here instead of returning `Ok(Some([]))` and deferring
    // the refusal to `check_dispatch`. Reuses `PromptNotAllowed` per S-13
    // with the non-colliding `EMPTY_ALLOW_SET_SENTINEL`; no new variant.
    let mut allowed: Option<Vec<String>> = None;
    for input in &ordered {
        if let Some(list) = &input.allowed_tools {
            let mut set = list.clone();
            set.sort();
            set.dedup();
            allowed = Some(match allowed {
                None => set,
                Some(current) => current
                    .into_iter()
                    .filter(|item| set.contains(item))
                    .collect(),
            });
        }
    }
    if let Some(list) = &allowed {
        if list.is_empty() {
            return Err(PromptError::PromptNotAllowed {
                tool: EMPTY_ALLOW_SET_SENTINEL.to_owned(),
            });
        }
    }

    // Budget: minimum across layers that constrain.
    let mut budget: Option<usize> = None;
    for input in &ordered {
        if let Some(ceiling) = input.budget_ceiling_bytes {
            budget = Some(match budget {
                None => ceiling,
                Some(current) => current.min(ceiling),
            });
        }
    }

    // Scopes: intersection over layers that constrain.
    let mut scopes: Option<Vec<String>> = None;
    for input in &ordered {
        if let Some(list) = &input.allowed_scopes {
            let mut set = list.clone();
            set.sort();
            set.dedup();
            scopes = Some(match scopes {
                None => set,
                Some(current) => current
                    .into_iter()
                    .filter(|item| set.contains(item))
                    .collect(),
            });
        }
    }

    // Directives: same key + same value merges silently; same key +
    // different value resolves by precedence (higher-precedence layer wins,
    // one audit record per overridden layer), except never-merge keys and
    // same-layer conflicts, which fail closed.
    let mut by_key: BTreeMap<String, (String, PromptLayer)> = BTreeMap::new();
    let mut overrides: Vec<DirectiveOverride> = Vec::new();
    for input in &ordered {
        for directive in &input.directives {
            match by_key.get(&directive.key) {
                None => {
                    by_key.insert(
                        directive.key.clone(),
                        (directive.value.clone(), input.layer),
                    );
                }
                Some((first_value, first_layer)) => {
                    if first_value != &directive.value {
                        if is_never_merge_directive_key(&directive.key)
                            || *first_layer == input.layer
                        {
                            return Err(PromptError::UnresolvableConflict {
                                key: directive.key.clone(),
                                first: first_value.clone(),
                                second: directive.value.clone(),
                            });
                        }
                        // `ordered` is rank-ascending with duplicate layers
                        // rejected, so the stored entry is always the
                        // strictly higher-precedence winner.
                        overrides.push(DirectiveOverride {
                            key: directive.key.clone(),
                            winning_value: first_value.clone(),
                            winning_layer: *first_layer,
                            overridden_value: directive.value.clone(),
                            overridden_layer: input.layer,
                        });
                    }
                }
            }
        }
    }
    overrides.sort_by(|left, right| {
        (
            &left.key,
            left.overridden_layer.rank(),
            &left.overridden_value,
            left.winning_layer.rank(),
            &left.winning_value,
        )
            .cmp(&(
                &right.key,
                right.overridden_layer.rank(),
                &right.overridden_value,
                right.winning_layer.rank(),
                &right.winning_value,
            ))
    });
    overrides.dedup();
    let directives: Vec<Directive> = by_key
        .into_iter()
        .map(|(key, (value, _layer))| Directive { key, value })
        .collect();

    let mut assembled = AssembledPrompt {
        core_version: snapshot.core_version.clone(),
        sections,
        effective_allowed_tools: allowed,
        effective_denied_tools: denied,
        effective_budget_ceiling_bytes: budget,
        effective_scopes: scopes,
        effective_directives: directives,
        merge_overrides: overrides,
        canonical_bytes: Vec::new(),
    };
    let bytes = render_canonical(&assembled)?;
    // P2-3: enforce the effective budget minimum against the canonical bytes.
    // Only the 96 KiB hard cap was checked before (`CanonicalTooLarge`);
    // a policy ceiling like `budget=1000` with 10 KiB of text wrongly
    // returned `Ok`. Reuses `InvalidBudget` per S-13 with `actual` carrying
    // the observed canonical length; no new variant. `render_canonical`
    // still reports `CanonicalTooLarge` first when both caps are exceeded.
    if let Some(ceiling) = budget {
        if bytes.len() > ceiling {
            return Err(PromptError::InvalidBudget {
                actual: bytes.len(),
            });
        }
    }
    assembled.canonical_bytes = bytes;
    Ok(assembled)
}

/// Render the canonical byte form for `prompt` (AIQ-12 mechanism evidence).
///
/// Layout (all lines LF-terminated, no trailing spaces):
///
/// ```text
/// prompt/1\n
/// core-version:<version>\n
/// [layer:core-contract len=<n>]\n<text>\n
/// [layer:user len=<n>]\n<text>\n
/// [layer:project len=<n>]\n<text>\n
/// [layer:skills-profile len=<n>]\n<text>\n
/// [layer:runtime-turn len=<n>]\n<text>\n
/// [effective]\n
/// allowed:<tool>\n | allowed:*\n
/// denied:<tool>\n
/// budget:<n>\n | budget:*\n
/// scope:<s>\n | scope:*\n
/// directive:<key>=<value>\n
/// override:<key> winner:<layer> wlen=<n>:<value> overridden:<layer> olen=<m>:<value>\n
/// end\n
/// ```
///
/// Text sections always appear in stable rank order with byte-length
/// prefixes so embedded newlines or section-like text never shift parsing.
/// Policy lists are lexicographically sorted. Override lines are sorted by
/// (key, overridden-layer rank, overridden value) and absent when no
/// override occurred, so override-free snapshots keep byte-identical forms.
/// Values are length-prefixed (`wlen`/`olen`) because directive values may
/// contain spaces, `=`, or `:` (only `\n`, `\r`, NUL are forbidden); keys
/// carry no spaces by construction and layer names come from a fixed set,
/// so each line parses unambiguously. The trailing `[effective]`
/// block keeps leading text bytes prefix-stable when only trailing layers
/// change.
///
/// # Errors
///
/// Returns [`PromptError::CanonicalTooLarge`] when the form exceeds
/// [`MAX_CANONICAL_BYTES`].
fn render_canonical(prompt: &AssembledPrompt) -> Result<Vec<u8>, PromptError> {
    let mut out: Vec<u8> = Vec::new();
    let mut push = |text: &str| out.extend_from_slice(text.as_bytes());

    push("prompt/1\n");
    push("core-version:");
    push(&prompt.core_version);
    push("\n");
    for section in &prompt.sections {
        push("[layer:");
        push(section.layer.header());
        push(" len=");
        push(&section.text.len().to_string());
        push("]\n");
        push(&section.text);
        push("\n");
    }
    push("[effective]\n");
    match &prompt.effective_allowed_tools {
        None => push("allowed:*\n"),
        Some(list) => {
            for tool in list {
                push("allowed:");
                push(tool);
                push("\n");
            }
        }
    }
    for tool in &prompt.effective_denied_tools {
        push("denied:");
        push(tool);
        push("\n");
    }
    match prompt.effective_budget_ceiling_bytes {
        None => push("budget:*\n"),
        Some(ceiling) => {
            push("budget:");
            push(&ceiling.to_string());
            push("\n");
        }
    }
    match &prompt.effective_scopes {
        None => push("scope:*\n"),
        Some(list) => {
            for scope in list {
                push("scope:");
                push(scope);
                push("\n");
            }
        }
    }
    for directive in &prompt.effective_directives {
        push("directive:");
        push(&directive.key);
        push("=");
        push(&directive.value);
        push("\n");
    }
    // Override audit lines: sorted deterministically (defensive re-sort so
    // even a hand-built `AssembledPrompt` renders canonically), absent when
    // empty so override-free snapshots keep byte-identical forms.
    let mut overrides: Vec<&DirectiveOverride> = prompt.merge_overrides.iter().collect();
    overrides.sort_by(|left, right| {
        (
            &left.key,
            left.overridden_layer.rank(),
            &left.overridden_value,
            left.winning_layer.rank(),
            &left.winning_value,
        )
            .cmp(&(
                &right.key,
                right.overridden_layer.rank(),
                &right.overridden_value,
                right.winning_layer.rank(),
                &right.winning_value,
            ))
    });
    for item in overrides {
        push("override:");
        push(&item.key);
        push(" winner:");
        push(item.winning_layer.header());
        push(" wlen=");
        push(&item.winning_value.len().to_string());
        push(":");
        push(&item.winning_value);
        push(" overridden:");
        push(item.overridden_layer.header());
        push(" olen=");
        push(&item.overridden_value.len().to_string());
        push(":");
        push(&item.overridden_value);
        push("\n");
    }
    push("end\n");

    if out.len() > MAX_CANONICAL_BYTES {
        return Err(PromptError::CanonicalTooLarge {
            limit: MAX_CANONICAL_BYTES,
            actual: out.len(),
        });
    }
    Ok(out)
}

/// Whether `tool` may dispatch under `prompt` plus the dispatcher grant.
///
/// The prompt never grants: `dispatcher_grants == false` always denies,
/// even when the prompt text or allow-set names the tool. The prompt only
/// narrows: a denied tool or a tool omitted from a constrained allow-set
/// denies even when the dispatcher would grant.
#[must_use]
pub fn is_dispatch_allowed(prompt: &AssembledPrompt, tool: &str, dispatcher_grants: bool) -> bool {
    check_dispatch(prompt, tool, dispatcher_grants).is_ok()
}

/// Check one dispatch against the assembled prompt plus dispatcher grant.
///
/// Fail-closed: dispatcher denial, prompt denial, or allow-set omission
/// each refuse with a typed error and no partial effect.
///
/// # Errors
///
/// Returns [`PromptError::DispatcherDenied`] when the dispatcher does not
/// grant, [`PromptError::PromptDenied`] when the prompt denies, or
/// [`PromptError::PromptNotAllowed`] when a constrained allow-set omits the
/// tool.
pub fn check_dispatch(
    prompt: &AssembledPrompt,
    tool: &str,
    dispatcher_grants: bool,
) -> Result<(), PromptError> {
    if !dispatcher_grants {
        return Err(PromptError::DispatcherDenied {
            tool: tool.to_owned(),
        });
    }
    if prompt
        .effective_denied_tools
        .iter()
        .any(|denied| denied == tool)
    {
        return Err(PromptError::PromptDenied {
            tool: tool.to_owned(),
        });
    }
    if let Some(allowed) = &prompt.effective_allowed_tools {
        if !allowed.iter().any(|item| item == tool) {
            return Err(PromptError::PromptNotAllowed {
                tool: tool.to_owned(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Declarative loading: `.wheel` project files + skill/profile registry.
//
// Design note (AI-0045): these loaders EXTEND the narrowing-only assembly
// above. They are pure `&str`-in / [`LayerInput`]-out constructors. The host
// (caller) owns every filesystem read and passes bytes in; this module
// performs no filesystem I/O (`std::fs` must stay out of `runtime/src` so
// the crate stays embeddable and sandboxable), never walks directories,
// never scans home directories, and never follows symlinks. The host must
// resolve symlinks under its explicit roots before passing bytes; any
// lexical escape that still reaches this module fail-closes in
// [`admit_project_path`].
//
// Untrusted-content rule: loaded project/skill bytes are data. They fill
// layer text plus structured fields through the SAME validators as
// programmatic layers ([`LayerInput::validate`] runs on every constructed
// layer), they can only narrow (deny-union, allow/scope-union within one
// layer with cross-layer intersection at assembly, budget-minimum,
// precedence-override with the never-merge list fail-closed), and they can
// never bypass [`NEVER_MERGE_DIRECTIVE_KEYS`].
//
// Documented line formats (both LF-only; any `\r` or NUL fail-closes):
//
// Project file (`LayerInput::project_from_str`):
//
// ```text
// # comment lines (first byte `#`) and blank lines are ignored
// directive.tone = concise
// allow_tool = panel_open
// deny_tool = panel_close
// scope = workspace.read
// budget = 4096
// text:
// <literal layer text, verbatim, to end of file>
// ```
//
// Header keys are exactly `directive.<name>`, `allow_tool`, `deny_tool`,
// `scope`, and `budget`; anything else in header position (including a
// `text:` line with trailing content) is malformed. A second `text:` line
// never occurs in header position: after the single `text:` marker every
// line is literal data — even lines that look like headers. Values run
// the same validators as programmatic input. A duplicate directive key
// with a different value in one file is malformed (same-layer conflicts
// fail closed at assembly anyway). After the single `text:` marker every
// line is literal data — even lines that look like headers.
//
// Skill/profile registry (`LayerInput::skills_from_str`):
//
// ```text
// # registry of single-agent skill/profile fragments, format version 1
// version = 1
// ---
// name = summaries
// version = 1
// directive.tone = concise
// text:
// <literal fragment, verbatim, to the next `---` line or end of file>
// ---
// name = planner
// version = 1
// text:
// <literal fragment>
// ```
//
// The registry-level `version` line is required first, and every entry
// requires its own `name` plus `version`. Any unknown/unsupported version
// is refused with a typed error. A line with exactly `---` always
// separates entries and therefore cannot appear inside a fragment.
// Entries merge deterministically in name order; same-key-different-value
// directives across entries fail closed with
// [`PromptError::UnresolvableConflict`] (same-layer semantics, never-merge
// keys included).
//
// No AIQ is closed by these loaders: canonical owner, digest algorithm,
// and reviewer acceptance stay open in the register.

/// Whether `version` is a supported skill/profile format version.
///
/// Today exactly `"1"` ([`SUPPORTED_SKILL_VERSIONS`]).
#[must_use]
pub fn is_supported_skill_version(version: &str) -> bool {
    SUPPORTED_SKILL_VERSIONS.contains(&version)
}

/// Lexical admission gate for declarative project file paths.
///
/// Pure string check, no filesystem access: `candidate` is admitted only
/// when it equals one of `roots` or sits directly under one (`root/..`
/// prefix). Both roots and candidates must be explicit slash-separated
/// paths with no empty segments and no `.`/`..` segments; backslash, CR,
/// LF, and NUL are rejected outright (Windows-separator and
/// control-character bypasses stay closed). Absolute and relative forms
/// never mix: an absolute candidate needs an absolute root, a relative
/// candidate a relative root. Malformed roots never match. Anything else
/// fail-closes with [`PromptError::InvalidProjectPath`].
///
/// # Errors
///
/// Returns [`PromptError::InvalidProjectPath`] when `candidate` is
/// malformed or escapes every caller-supplied root.
pub fn admit_project_path(roots: &[&str], candidate: &str) -> Result<(), PromptError> {
    let rejected = || PromptError::InvalidProjectPath {
        path: candidate.to_owned(),
    };
    let Some((candidate_abs, candidate_rest)) = split_rooted_path(candidate) else {
        return Err(rejected());
    };
    if candidate_rest.len() > MAX_PROJECT_PATH_LEN {
        return Err(rejected());
    }
    for root in roots {
        let Some((root_abs, root_rest)) = split_rooted_path(root) else {
            continue;
        };
        if root_abs != candidate_abs || root_rest.len() > MAX_PROJECT_PATH_LEN {
            continue;
        }
        if candidate_rest == root_rest {
            return Ok(());
        }
        if candidate_rest.len() > root_rest.len()
            && candidate_rest.starts_with(root_rest)
            && candidate_rest.as_bytes().get(root_rest.len()) == Some(&b'/')
        {
            return Ok(());
        }
    }
    Err(rejected())
}

/// Split a rooted path into (is_absolute, rest) after lexical hygiene.
///
/// Returns `None` for empty paths, over-long paths, paths with
/// backslash/CR/LF/NUL bytes, paths with empty segments (covers `//`,
/// leading `/` beyond the single rooted slash, and trailing `/`), and
/// paths with `.`/`..` segments. A single leading `/` marks an absolute
/// path; anything else is relative.
fn split_rooted_path(path: &str) -> Option<(bool, &str)> {
    if path.is_empty() || path.len() > MAX_PROJECT_PATH_LEN {
        return None;
    }
    if path.contains(['\0', '\r', '\n', '\\']) {
        return None;
    }
    let (absolute, rest) = match path.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, path),
    };
    if rest.is_empty() {
        return None;
    }
    if rest.split('/').any(is_unsafe_path_segment) {
        return None;
    }
    Some((absolute, rest))
}

/// Whether a slash-separated path segment is unsafe: empty (covers `//`,
/// leading `/`, trailing `/`) or a dot-segment (`.`/`..`).
fn is_unsafe_path_segment(segment: &str) -> bool {
    segment.is_empty() || segment == "." || segment == ".."
}

/// Parsed body of one declarative project file, before layer construction.
#[derive(Debug, Clone, Default)]
struct ProjectParts {
    /// Literal text lines after `text:` (joined with `\n`).
    text_lines: Vec<String>,
    /// `allow_tool` values in file order.
    allowed: Vec<String>,
    /// `deny_tool` values in file order.
    denied: Vec<String>,
    /// `budget` value, when present.
    budget: Option<usize>,
    /// `scope` values in file order.
    scopes: Vec<String>,
    /// `directive.<name>` values by name (sorted by construction).
    directives: BTreeMap<String, String>,
}

/// Split a header line on its first `=` into trimmed `(key, value)`.
fn split_header_line(line: &str) -> Option<(&str, &str)> {
    let (raw_key, raw_value) = line.split_once('=')?;
    Some((raw_key.trim(), raw_value.trim()))
}

/// Parse one declarative project file body into its parts.
///
/// Fail-closed with typed errors: structural problems yield
/// [`PromptError::MalformedProject`], while value-shape problems propagate
/// the same validator errors programmatic layers get
/// ([`PromptError::InvalidToolName`], [`PromptError::InvalidScope`],
/// [`PromptError::InvalidBudget`], [`PromptError::InvalidDirectiveKey`],
/// [`PromptError::InvalidDirectiveValue`]).
fn parse_project_source(source: &str) -> Result<ProjectParts, PromptError> {
    if source.len() > MAX_PROJECT_FILE_BYTES {
        return Err(PromptError::ProjectFileTooLarge {
            limit: MAX_PROJECT_FILE_BYTES,
            actual: source.len(),
        });
    }
    if source.contains('\r') {
        return Err(PromptError::MalformedProject {
            reason: "project file must use LF newlines only (no CR)".to_owned(),
        });
    }
    if source.contains('\0') {
        return Err(PromptError::MalformedProject {
            reason: "project file must not contain NUL".to_owned(),
        });
    }
    let malformed = |lineno: usize, detail: &str| PromptError::MalformedProject {
        reason: format!("line {lineno}: {detail}"),
    };
    let mut parts = ProjectParts::default();
    let mut in_text = false;
    for (index, line) in source.lines().enumerate() {
        let lineno = index + 1;
        if in_text {
            // Literal data: never interpreted, even when header-shaped.
            parts.text_lines.push(line.to_owned());
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed == "text:" {
            in_text = true;
            continue;
        }
        let Some((key, value)) = split_header_line(trimmed) else {
            return Err(malformed(lineno, "expected `key = value` or `text:`"));
        };
        if key.is_empty() {
            return Err(malformed(lineno, "empty header key"));
        }
        if let Some(name) = key.strip_prefix("directive.") {
            if name.is_empty() {
                return Err(malformed(lineno, "empty directive name"));
            }
            validate_directive_key(name)?;
            validate_directive_value(name, value)?;
            match parts.directives.get(name) {
                None => {
                    parts.directives.insert(name.to_owned(), value.to_owned());
                }
                Some(first) if first == value => {}
                Some(_) => {
                    return Err(malformed(
                        lineno,
                        "duplicate directive with different values",
                    ));
                }
            }
        } else if key == "allow_tool" {
            if value.is_empty() {
                return Err(malformed(lineno, "empty allow_tool value"));
            }
            validate_prompt_tool_name(value)?;
            parts.allowed.push(value.to_owned());
        } else if key == "deny_tool" {
            if value.is_empty() {
                return Err(malformed(lineno, "empty deny_tool value"));
            }
            validate_prompt_tool_name(value)?;
            parts.denied.push(value.to_owned());
        } else if key == "scope" {
            if value.is_empty() {
                return Err(malformed(lineno, "empty scope value"));
            }
            validate_scope(value)?;
            parts.scopes.push(value.to_owned());
        } else if key == "budget" {
            let ceiling = value
                .parse::<usize>()
                .map_err(|_| malformed(lineno, "budget must be a decimal byte count"))?;
            validate_budget_ceiling(ceiling)?;
            match parts.budget {
                None => parts.budget = Some(ceiling),
                Some(first) if first == ceiling => {}
                Some(_) => {
                    return Err(malformed(lineno, "duplicate budget with different values"));
                }
            }
        } else {
            return Err(malformed(lineno, "unknown project header key"));
        }
    }
    Ok(parts)
}

/// Build a validated Project [`LayerInput`] from parsed parts.
///
/// Runs [`LayerInput::validate`], the same gate programmatic layers pass.
fn project_parts_to_layer(mut parts: ProjectParts) -> Result<LayerInput, PromptError> {
    parts.allowed.sort();
    parts.allowed.dedup();
    parts.denied.sort();
    parts.denied.dedup();
    parts.scopes.sort();
    parts.scopes.dedup();
    let input = LayerInput {
        layer: PromptLayer::Project,
        text: parts.text_lines.join("\n"),
        allowed_tools: if parts.allowed.is_empty() {
            None
        } else {
            Some(parts.allowed)
        },
        denied_tools: parts.denied,
        budget_ceiling_bytes: parts.budget,
        allowed_scopes: if parts.scopes.is_empty() {
            None
        } else {
            Some(parts.scopes)
        },
        directives: parts
            .directives
            .into_iter()
            .map(|(key, value)| Directive { key, value })
            .collect(),
    };
    input.validate()?;
    Ok(input)
}

impl LayerInput {
    /// Parse one declarative `.wheel` project file body into a Project
    /// layer.
    ///
    /// The caller (host) reads the file and passes its bytes; this
    /// constructor never touches the filesystem. Loaded text is untrusted
    /// data validated exactly like programmatic input. See the module
    /// section on declarative loading for the documented format.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for over-bound, malformed, or invalid
    /// content. No partial layer is returned.
    pub fn project_from_str(source: &str) -> Result<Self, PromptError> {
        let parts = parse_project_source(source)?;
        project_parts_to_layer(parts)
    }

    /// Merge declarative `.wheel` project files into one Project layer.
    ///
    /// `roots` are explicit caller-supplied roots (for example `".wheel"`);
    /// every candidate path in `files` must pass [`admit_project_path`]
    /// against them — escapes fail closed before any byte is parsed. Each
    /// body parses via [`LayerInput::project_from_str`] rules. Merge is
    /// deterministic (sorted by path): texts concatenate in path order,
    /// deny-sets union, allow/scope-sets union within the layer
    /// (cross-layer assembly still intersects, so the layer can never
    /// widen an upper layer), budgets take the minimum, and directives
    /// merge on equal values while differing values fail closed with
    /// [`PromptError::UnresolvableConflict`] (same-layer semantics; the
    /// never-merge list included). Duplicate paths fail closed as malformed
    /// (ambiguous discovery).
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for root escapes, over-bound or malformed
    /// files, duplicate paths, directive conflicts, or invalid merged
    /// content. No partial layer is returned.
    pub fn project_from_files_under_roots(
        roots: &[&str],
        files: &[(&str, &str)],
    ) -> Result<Self, PromptError> {
        if files.len() > MAX_PROJECT_FILES {
            return Err(PromptError::TooManyProjectFiles {
                limit: MAX_PROJECT_FILES,
            });
        }
        let mut parsed: Vec<(&str, ProjectParts)> = Vec::with_capacity(files.len());
        for &(path, bytes) in files {
            admit_project_path(roots, path)?;
            let parts = parse_project_source(bytes).map_err(|error| match error {
                PromptError::MalformedProject { reason } => PromptError::MalformedProject {
                    reason: format!("{path}: {reason}"),
                },
                other => other,
            })?;
            parsed.push((path, parts));
        }
        parsed.sort_by(|left, right| left.0.cmp(right.0));
        let mut merged = ProjectParts::default();
        let mut seen_paths: Vec<&str> = Vec::with_capacity(parsed.len());
        for (path, parts) in parsed {
            if seen_paths.contains(&path) {
                return Err(PromptError::MalformedProject {
                    reason: format!("{path}: duplicate project path"),
                });
            }
            seen_paths.push(path);
            // Line vectors concatenate directly: the join in
            // `project_parts_to_layer` separates files with single `\n`.
            merged.text_lines.extend(parts.text_lines);
            merged.allowed.extend(parts.allowed);
            merged.denied.extend(parts.denied);
            merged.scopes.extend(parts.scopes);
            merged.budget = match (merged.budget, parts.budget) {
                (Some(current), Some(next)) => Some(current.min(next)),
                (Some(current), None) => Some(current),
                (None, next) => next,
            };
            for (key, value) in parts.directives {
                match merged.directives.get(&key) {
                    None => {
                        merged.directives.insert(key, value);
                    }
                    Some(first) if first == &value => {}
                    Some(first) => {
                        return Err(PromptError::UnresolvableConflict {
                            key: key.clone(),
                            first: first.clone(),
                            second: value,
                        });
                    }
                }
            }
        }
        // Duplicate paths fail closed above; merge deterministically.
        project_parts_to_layer(merged)
    }

    /// Parse a versioned skill/profile registry document into a
    /// SkillsProfile layer.
    ///
    /// The caller (host) reads the registry and passes its bytes; this
    /// constructor never touches the filesystem. The registry-level
    /// `version` field and every entry-level `version` field are required
    /// and must name a version in [`SUPPORTED_SKILL_VERSIONS`]; anything
    /// else is refused with a typed error. Fragments are untrusted data
    /// validated exactly like programmatic input. See the module section
    /// on declarative loading for the documented format.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] for over-bound, malformed, versionless, or
    /// unsupported-version content, duplicate names, directive conflicts,
    /// or invalid merged content. No partial layer is returned.
    pub fn skills_from_str(source: &str) -> Result<Self, PromptError> {
        if source.len() > MAX_SKILL_REGISTRY_BYTES {
            return Err(PromptError::SkillRegistryTooLarge {
                limit: MAX_SKILL_REGISTRY_BYTES,
                actual: source.len(),
            });
        }
        if source.contains('\r') {
            return Err(PromptError::MalformedSkill {
                reason: "skill registry must use LF newlines only (no CR)".to_owned(),
            });
        }
        if source.contains('\0') {
            return Err(PromptError::MalformedSkill {
                reason: "skill registry must not contain NUL".to_owned(),
            });
        }
        let mut lines = source.lines().enumerate().peekable();
        // First substantive line must be the registry version header.
        loop {
            let Some((index, line)) = lines.next() else {
                return Err(PromptError::MalformedSkill {
                    reason: "skill registry misses required `version` header".to_owned(),
                });
            };
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((key, value)) = split_header_line(trimmed) else {
                return Err(PromptError::MalformedSkill {
                    reason: format!("line {}: first line must be `version = <n>`", index + 1),
                });
            };
            if key != "version" || value.is_empty() {
                return Err(PromptError::MalformedSkill {
                    reason: format!("line {}: first line must be `version = <n>`", index + 1),
                });
            }
            if value.len() > MAX_SKILL_VERSION_LEN || !is_supported_skill_version(value) {
                return Err(PromptError::UnsupportedSkillVersion {
                    version: value.to_owned(),
                });
            }
            break;
        }
        // Split the remainder into entry chunks on `---` lines.
        let mut chunks: Vec<Vec<(usize, &str)>> = vec![Vec::new()];
        for (index, line) in lines {
            if line.trim() == "---" {
                chunks.push(Vec::new());
            } else {
                chunks
                    .last_mut()
                    .expect("chunks never empty")
                    .push((index + 1, line));
            }
        }
        // A trailing `---` leaves an empty final chunk; ignore that one
        // only. Any other blank chunk is skipped by `parse_skill_chunk`.
        if chunks.last().is_some_and(|last| last.is_empty()) {
            chunks.pop();
        }
        let mut entries: Vec<SkillEntry> = Vec::new();
        for chunk in &chunks {
            if let Some(entry) = parse_skill_chunk(chunk)? {
                entries.push(entry);
            }
        }
        if entries.len() > MAX_SKILL_ENTRIES {
            return Err(PromptError::TooManySkills {
                limit: MAX_SKILL_ENTRIES,
            });
        }
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        // Sorted names make duplicates adjacent: deterministic detection.
        if let Some(duplicate) = entries.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(PromptError::DuplicateSkill {
                name: duplicate[0].name.clone(),
            });
        }
        let mut merged = ProjectParts::default();
        for entry in entries {
            if !entry.fragment.is_empty() {
                merged.text_lines.push(entry.fragment);
            }
            merged.allowed.extend(entry.allowed);
            merged.denied.extend(entry.denied);
            merged.scopes.extend(entry.scopes);
            for (key, value) in entry.directives {
                match merged.directives.get(&key) {
                    None => {
                        merged.directives.insert(key, value);
                    }
                    Some(first) if first == &value => {}
                    Some(first) => {
                        return Err(PromptError::UnresolvableConflict {
                            key: key.clone(),
                            first: first.clone(),
                            second: value,
                        });
                    }
                }
            }
        }
        let input = LayerInput {
            layer: PromptLayer::SkillsProfile,
            text: merged.text_lines.join("\n"),
            allowed_tools: if merged.allowed.is_empty() {
                None
            } else {
                merged.allowed.sort();
                merged.allowed.dedup();
                Some(merged.allowed)
            },
            denied_tools: {
                merged.denied.sort();
                merged.denied.dedup();
                merged.denied
            },
            budget_ceiling_bytes: None,
            allowed_scopes: if merged.scopes.is_empty() {
                None
            } else {
                merged.scopes.sort();
                merged.scopes.dedup();
                Some(merged.scopes)
            },
            directives: merged
                .directives
                .into_iter()
                .map(|(key, value)| Directive { key, value })
                .collect(),
        };
        // Entry fragments are individually bounded; the joined text still
        // runs the same layer gate as programmatic input.
        input.validate()?;
        Ok(input)
    }
}

/// One parsed skill/profile registry entry.
#[derive(Debug, Clone)]
struct SkillEntry {
    /// Entry name (`^[a-z][a-z0-9_-]*$`, bounded; hyphen allowed independent
    /// of the tool namespace).
    name: String,
    /// Literal fragment lines after `text:` (joined with `\n`).
    fragment: String,
    /// `allow_tool` values in entry order.
    allowed: Vec<String>,
    /// `deny_tool` values in entry order.
    denied: Vec<String>,
    /// `scope` values in entry order.
    scopes: Vec<String>,
    /// `directive.<name>` values by name.
    directives: BTreeMap<String, String>,
}

/// Whether an entry chunk carries no substantive lines at all.
fn chunk_is_blank(chunk: &[(usize, &str)]) -> bool {
    chunk.iter().all(|(_, line)| {
        let trimmed = line.trim();
        trimmed.is_empty() || trimmed.starts_with('#')
    })
}

/// Parse one `---`-separated registry chunk.
///
/// Returns `Ok(None)` for a blank chunk (skipped silently); any chunk with
/// substantive content must carry a valid `name` plus a required,
/// supported `version`, else it fail-closes with a typed error.
fn parse_skill_chunk(chunk: &[(usize, &str)]) -> Result<Option<SkillEntry>, PromptError> {
    if chunk_is_blank(chunk) {
        return Ok(None);
    }
    let malformed = |lineno: usize, detail: &str| PromptError::MalformedSkill {
        reason: format!("line {lineno}: {detail}"),
    };
    let mut name: Option<String> = None;
    let mut version: Option<String> = None;
    let mut allowed: Vec<String> = Vec::new();
    let mut denied: Vec<String> = Vec::new();
    let mut scopes: Vec<String> = Vec::new();
    let mut directives: BTreeMap<String, String> = BTreeMap::new();
    let mut fragment_lines: Vec<&str> = Vec::new();
    let mut in_text = false;
    for (lineno, line) in chunk {
        if in_text {
            fragment_lines.push(line);
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed == "text:" {
            in_text = true;
            continue;
        }
        let Some((key, value)) = split_header_line(trimmed) else {
            return Err(malformed(*lineno, "expected `key = value` or `text:`"));
        };
        if key == "name" {
            if value.is_empty() {
                return Err(malformed(*lineno, "empty skill name"));
            }
            if value.len() > MAX_SKILL_NAME_LEN {
                return Err(malformed(*lineno, "skill name exceeds length bound"));
            }
            // Skill namespace is independent of the tool namespace: hyphens
            // accepted here, still rejected by `validate_prompt_tool_name`.
            validate_skill_name(value)?;
            match &name {
                None => name = Some(value.to_owned()),
                Some(first) if first == value => {}
                Some(_) => return Err(malformed(*lineno, "duplicate name lines differ")),
            }
        } else if key == "version" {
            if value.is_empty() {
                return Err(malformed(*lineno, "empty skill version"));
            }
            if value.len() > MAX_SKILL_VERSION_LEN {
                return Err(malformed(*lineno, "skill version exceeds length bound"));
            }
            version = Some(value.to_owned());
        } else if let Some(directive_name) = key.strip_prefix("directive.") {
            if directive_name.is_empty() {
                return Err(malformed(*lineno, "empty directive name"));
            }
            validate_directive_key(directive_name)?;
            validate_directive_value(directive_name, value)?;
            match directives.get(directive_name) {
                None => {
                    directives.insert(directive_name.to_owned(), value.to_owned());
                }
                Some(first) if first == value => {}
                Some(_) => {
                    return Err(malformed(
                        *lineno,
                        "duplicate directive with different values",
                    ));
                }
            }
        } else if key == "allow_tool" {
            if value.is_empty() {
                return Err(malformed(*lineno, "empty allow_tool value"));
            }
            validate_prompt_tool_name(value)?;
            allowed.push(value.to_owned());
        } else if key == "deny_tool" {
            if value.is_empty() {
                return Err(malformed(*lineno, "empty deny_tool value"));
            }
            validate_prompt_tool_name(value)?;
            denied.push(value.to_owned());
        } else if key == "scope" {
            if value.is_empty() {
                return Err(malformed(*lineno, "empty scope value"));
            }
            validate_scope(value)?;
            scopes.push(value.to_owned());
        } else {
            return Err(malformed(*lineno, "unknown skill entry key"));
        }
    }
    let Some(name) = name else {
        return Err(malformed(
            chunk.first().map(|(lineno, _)| *lineno).unwrap_or(0),
            "skill entry misses required `name`",
        ));
    };
    let Some(version) = version else {
        return Err(PromptError::MissingSkillVersion { name });
    };
    if !is_supported_skill_version(&version) {
        return Err(PromptError::UnsupportedSkillVersion { version });
    }
    let fragment = fragment_lines.join("\n");
    if fragment.len() > MAX_SKILL_ENTRY_BYTES {
        return Err(PromptError::SkillEntryTooLarge {
            limit: MAX_SKILL_ENTRY_BYTES,
            actual: fragment.len(),
        });
    }
    Ok(Some(SkillEntry {
        name,
        fragment,
        allowed,
        denied,
        scopes,
        directives,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_layer(layer: PromptLayer, text: &str) -> LayerInput {
        LayerInput::text_only(layer, text)
    }

    fn snapshot_with(layers: Vec<LayerInput>) -> PromptSnapshot {
        PromptSnapshot {
            core_version: "bitty-core-prompt@1".to_owned(),
            layers,
        }
    }

    fn full_layer(
        layer: PromptLayer,
        text: &str,
        allowed: Option<Vec<&str>>,
        denied: Vec<&str>,
        budget: Option<usize>,
        scopes: Option<Vec<&str>>,
        directives: Vec<(&str, &str)>,
    ) -> LayerInput {
        LayerInput {
            layer,
            text: text.to_owned(),
            allowed_tools: allowed.map(|list| list.into_iter().map(str::to_owned).collect()),
            denied_tools: denied.into_iter().map(str::to_owned).collect(),
            budget_ceiling_bytes: budget,
            allowed_scopes: scopes.map(|list| list.into_iter().map(str::to_owned).collect()),
            directives: directives
                .into_iter()
                .map(|(key, value)| Directive {
                    key: key.to_owned(),
                    value: value.to_owned(),
                })
                .collect(),
        }
    }

    #[test]
    fn stable_order_is_core_to_runtime() {
        let order = PromptLayer::stable_order();
        assert_eq!(
            order,
            [
                PromptLayer::CoreContract,
                PromptLayer::User,
                PromptLayer::Project,
                PromptLayer::SkillsProfile,
                PromptLayer::RuntimeTurn,
            ]
        );
        let mut ranks: Vec<u8> = order.iter().map(|layer| layer.rank()).collect();
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            PromptLayer::from_header("core-contract"),
            Some(PromptLayer::CoreContract)
        );
        assert_eq!(
            PromptLayer::from_header("runtime-turn"),
            Some(PromptLayer::RuntimeTurn)
        );
        assert_eq!(PromptLayer::from_header("nope"), None);
    }

    #[test]
    fn assembly_orders_layers_regardless_of_input_order() {
        let forward = snapshot_with(vec![
            text_layer(PromptLayer::CoreContract, "core"),
            text_layer(PromptLayer::User, "user"),
            text_layer(PromptLayer::Project, "project"),
            text_layer(PromptLayer::SkillsProfile, "skills"),
            text_layer(PromptLayer::RuntimeTurn, "turn"),
        ]);
        let reverse = snapshot_with(vec![
            text_layer(PromptLayer::RuntimeTurn, "turn"),
            text_layer(PromptLayer::SkillsProfile, "skills"),
            text_layer(PromptLayer::Project, "project"),
            text_layer(PromptLayer::User, "user"),
            text_layer(PromptLayer::CoreContract, "core"),
        ]);
        let left = assemble(&forward).expect("forward assembles");
        let right = assemble(&reverse).expect("reverse assembles");
        assert_eq!(left.canonical_bytes, right.canonical_bytes);
        assert_eq!(left.sections[0].layer, PromptLayer::CoreContract);
        assert_eq!(left.sections[4].layer, PromptLayer::RuntimeTurn);
    }

    #[test]
    fn same_snapshot_yields_identical_bytes() {
        let snapshot = snapshot_with(vec![text_layer(PromptLayer::CoreContract, "core text")]);
        let first = assemble(&snapshot).expect("first assembles");
        let second = assemble(&snapshot).expect("second assembles");
        assert_eq!(first.canonical_bytes, second.canonical_bytes);
        assert_eq!(first, second);
    }

    #[test]
    fn canonical_form_is_byte_exact_for_minimal_snapshot() {
        let snapshot = snapshot_with(vec![text_layer(PromptLayer::CoreContract, "core")]);
        let assembled = assemble(&snapshot).expect("assembles");
        let text = String::from_utf8(assembled.canonical_bytes.clone()).expect("utf8");
        let expected = concat!(
            "prompt/1\n",
            "core-version:bitty-core-prompt@1\n",
            "[layer:core-contract len=4]\ncore\n",
            "[layer:user len=0]\n\n",
            "[layer:project len=0]\n\n",
            "[layer:skills-profile len=0]\n\n",
            "[layer:runtime-turn len=0]\n\n",
            "[effective]\n",
            "allowed:*\n",
            "budget:*\n",
            "scope:*\n",
            "end\n",
        );
        assert_eq!(text, expected);
    }

    #[test]
    fn stable_before_dynamic_text_order() {
        let snapshot = snapshot_with(vec![
            text_layer(PromptLayer::RuntimeTurn, "turn"),
            text_layer(PromptLayer::CoreContract, "core"),
        ]);
        let assembled = assemble(&snapshot).expect("assembles");
        let text = String::from_utf8(assembled.canonical_bytes.clone()).expect("utf8");
        let core = text.find("[layer:core-contract").expect("core header");
        let user = text.find("[layer:user").expect("user header");
        let project = text.find("[layer:project").expect("project header");
        let skills = text.find("[layer:skills-profile").expect("skills header");
        let runtime = text.find("[layer:runtime-turn").expect("runtime header");
        let effective = text.find("[effective]").expect("effective block");
        assert!(core < user && user < project && project < skills && skills < runtime);
        assert!(runtime < effective);
    }

    #[test]
    fn canonical_lists_are_sorted_and_lf_only() {
        let snapshot = snapshot_with(vec![full_layer(
            PromptLayer::CoreContract,
            "core",
            Some(vec!["tool_b", "tool_a"]),
            vec!["tool_z", "tool_m"],
            Some(4096),
            Some(vec!["terminal.read", "workspace.read"]),
            vec![("tone", "concise"), ("style.mode", "terse")],
        )]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert_eq!(
            assembled.effective_allowed_tools,
            Some(vec!["tool_a".to_owned(), "tool_b".to_owned()])
        );
        assert_eq!(
            assembled.effective_denied_tools,
            vec!["tool_m".to_owned(), "tool_z".to_owned()]
        );
        assert_eq!(
            assembled.effective_scopes,
            Some(vec![
                "terminal.read".to_owned(),
                "workspace.read".to_owned()
            ])
        );
        let text = String::from_utf8(assembled.canonical_bytes.clone()).expect("utf8");
        assert!(!text.contains('\r'));
        let allowed_a = text.find("allowed:tool_a").expect("sorted allowed a");
        let allowed_b = text.find("allowed:tool_b").expect("sorted allowed b");
        assert!(allowed_a < allowed_b);
        let denied_m = text.find("denied:tool_m").expect("sorted denied m");
        let denied_z = text.find("denied:tool_z").expect("sorted denied z");
        assert!(denied_m < denied_z);
    }

    #[test]
    fn trailing_change_preserves_leading_prefix() {
        let base = snapshot_with(vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::RuntimeTurn, "turn one"),
        ]);
        let changed = snapshot_with(vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::RuntimeTurn, "turn two"),
        ]);
        let left = assemble(&base).expect("base assembles");
        let right = assemble(&changed).expect("changed assembles");
        assert_ne!(left.canonical_bytes, right.canonical_bytes);
        let prefix = common_prefix_len(&left.canonical_bytes, &right.canonical_bytes);
        let text = String::from_utf8(left.canonical_bytes.clone()).expect("utf8");
        let runtime_header = text.find("[layer:runtime-turn").expect("runtime header");
        assert!(
            prefix >= runtime_header,
            "leading prefix must survive a trailing-only change"
        );
    }

    #[test]
    fn lower_layer_cannot_grant_denied_tool() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec!["panel_close"],
                None,
                None,
                vec![],
            ),
            full_layer(
                PromptLayer::RuntimeTurn,
                "turn claims access",
                Some(vec!["panel_close"]),
                vec![],
                None,
                None,
                vec![],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert!(
            assembled
                .effective_denied_tools
                .contains(&"panel_close".to_owned())
        );
        // Deny wins even when the dispatcher would grant.
        assert_eq!(
            check_dispatch(&assembled, "panel_close", true),
            Err(PromptError::PromptDenied {
                tool: "panel_close".to_owned()
            })
        );
        assert!(!is_dispatch_allowed(&assembled, "panel_close", true));
    }

    #[test]
    fn allowed_intersection_narrows_cannot_widen() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                Some(vec!["tool_a", "tool_b"]),
                vec![],
                None,
                None,
                vec![],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                Some(vec!["tool_b", "tool_c"]),
                vec![],
                None,
                None,
                vec![],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert_eq!(
            assembled.effective_allowed_tools,
            Some(vec!["tool_b".to_owned()])
        );
        assert!(is_dispatch_allowed(&assembled, "tool_b", true));
        assert_eq!(
            check_dispatch(&assembled, "tool_a", true),
            Err(PromptError::PromptNotAllowed {
                tool: "tool_a".to_owned()
            })
        );
        assert_eq!(
            check_dispatch(&assembled, "tool_c", true),
            Err(PromptError::PromptNotAllowed {
                tool: "tool_c".to_owned()
            })
        );
    }

    #[test]
    fn budget_min_wins_cannot_widen() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                Some(1000),
                None,
                vec![],
            ),
            full_layer(
                PromptLayer::RuntimeTurn,
                "turn",
                None,
                vec![],
                Some(5000),
                None,
                vec![],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert_eq!(assembled.effective_budget_ceiling_bytes, Some(1000));
    }

    #[test]
    fn disjoint_allowed_sets_fail_closed_at_assembly() {
        // P2-3: disjoint per-layer allow-sets intersect to `Some([])` and
        // must fail at assembly, not defer to `check_dispatch`. Reuses
        // `PromptNotAllowed` per S-13 with the sentinel (no new variant:
        // the empty set has no single tool to name, and the sentinel can
        // never collide with a `TB-2` name).
        let disjoint = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                Some(vec!["tool_a"]),
                vec![],
                None,
                None,
                vec![],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                Some(vec!["tool_b"]),
                vec![],
                None,
                None,
                vec![],
            ),
        ]);
        assert_eq!(
            assemble(&disjoint).expect_err("disjoint allow-sets must fail"),
            PromptError::PromptNotAllowed {
                tool: EMPTY_ALLOW_SET_SENTINEL.to_owned(),
            }
        );
        // An explicit empty list is the same configuration error.
        let explicit_empty = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "u",
            Some(vec![]),
            vec![],
            None,
            None,
            vec![],
        )]);
        assert_eq!(
            assemble(&explicit_empty).expect_err("explicit empty allow-set must fail"),
            PromptError::PromptNotAllowed {
                tool: EMPTY_ALLOW_SET_SENTINEL.to_owned(),
            }
        );
        // The sentinel is unambiguous: it can never be a real tool name.
        assert!(validate_prompt_tool_name(EMPTY_ALLOW_SET_SENTINEL).is_err());
    }

    #[test]
    fn canonical_exceeding_effective_budget_fails_closed() {
        // P2-3: the effective budget minimum is enforced against the
        // canonical bytes, not just the 96 KiB hard cap. Reuses
        // `InvalidBudget` per S-13 with `actual` carrying the observed
        // canonical length; no new variant.
        let tight = snapshot_with(vec![full_layer(
            PromptLayer::CoreContract,
            "core",
            None,
            vec![],
            Some(100),
            None,
            vec![],
        )]);
        // Same text without a budget assembles, proving the fixture text
        // alone already overruns the tight ceiling (the budgeted canonical
        // form is 2 bytes longer via `budget:100` vs `budget:*`, so exact
        // equality with the unbounded length would be off by the header).
        let unbounded = snapshot_with(vec![text_layer(PromptLayer::CoreContract, "core")]);
        let baseline = assemble(&unbounded).expect("unbounded assembles");
        let observed = baseline.canonical_bytes.len();
        assert!(
            observed > 100,
            "fixture must overrun the tight budget (got {observed} bytes)"
        );
        match assemble(&tight).expect_err("over-budget canonical must fail") {
            PromptError::InvalidBudget { actual } => {
                assert!(
                    actual > 100,
                    "over-budget actual must exceed the ceiling (got {actual})"
                );
                // Budgeted form differs from the unbounded baseline only in
                // the `budget:` line (`budget:100` vs `budget:*`).
                assert_eq!(actual, observed + "100".len() - "*".len());
            }
            other => panic!("expected InvalidBudget, got {other:?}"),
        }
        // A ceiling above the canonical length still assembles.
        let roomy = snapshot_with(vec![full_layer(
            PromptLayer::CoreContract,
            "core",
            None,
            vec![],
            Some(4096),
            None,
            vec![],
        )]);
        let assembled = assemble(&roomy).expect("roomy budget assembles");
        assert_eq!(assembled.effective_budget_ceiling_bytes, Some(4096));
    }

    #[test]
    fn scopes_intersection_narrows_cannot_widen() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                Some(vec!["workspace.read", "terminal.read"]),
                vec![],
            ),
            full_layer(
                PromptLayer::RuntimeTurn,
                "turn",
                None,
                vec![],
                None,
                Some(vec!["workspace.read", "terminal.write"]),
                vec![],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert_eq!(
            assembled.effective_scopes,
            Some(vec!["workspace.read".to_owned()])
        );
    }

    // AI-0044 replaced the old fail-closed-on-any-difference rule with
    // precedence override: the higher-precedence (lower-rank) layer wins and
    // the override is audited. Only never-merge keys and same-layer
    // conflicts still fail closed (see the `never_merge_*` and
    // `same_layer_*` tests below).
    #[test]
    fn higher_precedence_directive_overrides_lower() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::User,
                "user",
                None,
                vec![],
                None,
                None,
                vec![("tone", "formal")],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                None,
                vec![],
                None,
                None,
                vec![("tone", "casual")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("precedence resolves");
        assert_eq!(
            assembled.effective_directives,
            vec![Directive {
                key: "tone".to_owned(),
                value: "formal".to_owned()
            }]
        );
        assert_eq!(
            assembled.merge_overrides,
            vec![DirectiveOverride {
                key: "tone".to_owned(),
                winning_value: "formal".to_owned(),
                winning_layer: PromptLayer::User,
                overridden_value: "casual".to_owned(),
                overridden_layer: PromptLayer::Project,
            }]
        );
    }

    #[test]
    fn lower_layer_cannot_override_upper() {
        // Core (rank 0) beats RuntimeTurn (rank 4) even when the lower layer
        // is listed first in the input: input order never affects output.
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::RuntimeTurn,
                "turn",
                None,
                vec![],
                None,
                None,
                vec![("tone", "chatty")],
            ),
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                None,
                vec![("tone", "formal")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("precedence resolves");
        assert_eq!(
            assembled.effective_directives,
            vec![Directive {
                key: "tone".to_owned(),
                value: "formal".to_owned()
            }]
        );
        assert_eq!(
            assembled.merge_overrides,
            vec![DirectiveOverride {
                key: "tone".to_owned(),
                winning_value: "formal".to_owned(),
                winning_layer: PromptLayer::CoreContract,
                overridden_value: "chatty".to_owned(),
                overridden_layer: PromptLayer::RuntimeTurn,
            }]
        );
    }

    #[test]
    fn override_audit_record_exact_contents() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                None,
                vec![("tone", "formal"), ("style.mode", "terse")],
            ),
            full_layer(
                PromptLayer::SkillsProfile,
                "skills",
                None,
                vec![],
                None,
                None,
                vec![("tone", "casual"), ("style.mode", "terse")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("precedence resolves");
        // Same-value `style.mode` merges with no record; differing `tone`
        // records exactly one override with winner + overridden value/layer.
        assert_eq!(assembled.merge_overrides.len(), 1);
        let record = &assembled.merge_overrides[0];
        assert_eq!(record.key, "tone");
        assert_eq!(record.winning_value, "formal");
        assert_eq!(record.winning_layer, PromptLayer::CoreContract);
        assert_eq!(record.overridden_value, "casual");
        assert_eq!(record.overridden_layer, PromptLayer::SkillsProfile);
        assert_eq!(
            assembled.effective_directives,
            vec![
                Directive {
                    key: "style.mode".to_owned(),
                    value: "terse".to_owned()
                },
                Directive {
                    key: "tone".to_owned(),
                    value: "formal".to_owned()
                },
            ]
        );
    }

    #[test]
    fn multi_layer_conflict_records_each_overridden_layer() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                None,
                vec![("tone", "formal")],
            ),
            full_layer(
                PromptLayer::User,
                "user",
                None,
                vec![],
                None,
                None,
                vec![("tone", "brief")],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                None,
                vec![],
                None,
                None,
                vec![("tone", "casual")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("precedence resolves");
        // Core wins; both lower layers are audited, sorted by overridden
        // rank (User before Project).
        assert_eq!(
            assembled.merge_overrides,
            vec![
                DirectiveOverride {
                    key: "tone".to_owned(),
                    winning_value: "formal".to_owned(),
                    winning_layer: PromptLayer::CoreContract,
                    overridden_value: "brief".to_owned(),
                    overridden_layer: PromptLayer::User,
                },
                DirectiveOverride {
                    key: "tone".to_owned(),
                    winning_value: "formal".to_owned(),
                    winning_layer: PromptLayer::CoreContract,
                    overridden_value: "casual".to_owned(),
                    overridden_layer: PromptLayer::Project,
                },
            ]
        );
        assert_eq!(
            assembled.effective_directives,
            vec![Directive {
                key: "tone".to_owned(),
                value: "formal".to_owned()
            }]
        );
    }

    #[test]
    fn never_merge_keys_stay_fail_closed() {
        assert_eq!(
            NEVER_MERGE_DIRECTIVE_KEYS,
            &["agent.identity", "capability.grant"]
        );
        assert!(is_never_merge_directive_key("agent.identity"));
        assert!(is_never_merge_directive_key("capability.grant"));
        assert!(!is_never_merge_directive_key("tone"));
        for key in NEVER_MERGE_DIRECTIVE_KEYS {
            let snapshot = snapshot_with(vec![
                full_layer(
                    PromptLayer::CoreContract,
                    "core",
                    None,
                    vec![],
                    None,
                    None,
                    vec![(*key, "upper")],
                ),
                full_layer(
                    PromptLayer::Project,
                    "project",
                    None,
                    vec![],
                    None,
                    None,
                    vec![(*key, "lower")],
                ),
            ]);
            let err = assemble(&snapshot).expect_err("never-merge must fail");
            assert_eq!(
                err,
                PromptError::UnresolvableConflict {
                    key: (*key).to_owned(),
                    first: "upper".to_owned(),
                    second: "lower".to_owned(),
                }
            );
        }
    }

    #[test]
    fn never_merge_keys_with_same_value_merge_silently() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                None,
                vec![("agent.identity", "bitty-core")],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                None,
                vec![],
                None,
                None,
                vec![("agent.identity", "bitty-core")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("same value merges");
        assert_eq!(
            assembled.effective_directives,
            vec![Directive {
                key: "agent.identity".to_owned(),
                value: "bitty-core".to_owned()
            }]
        );
        assert!(assembled.merge_overrides.is_empty());
    }

    #[test]
    fn same_layer_conflict_stays_fail_closed() {
        // No precedence difference within one layer: the assembler refuses
        // to guess which in-layer entry wins.
        let snapshot = PromptSnapshot {
            core_version: "bitty-core-prompt@1".to_owned(),
            layers: vec![LayerInput {
                layer: PromptLayer::User,
                text: "user".to_owned(),
                allowed_tools: None,
                denied_tools: Vec::new(),
                budget_ceiling_bytes: None,
                allowed_scopes: None,
                directives: vec![
                    Directive {
                        key: "tone".to_owned(),
                        value: "formal".to_owned(),
                    },
                    Directive {
                        key: "tone".to_owned(),
                        value: "casual".to_owned(),
                    },
                ],
            }],
        };
        let err = assemble(&snapshot).expect_err("same-layer conflict must fail");
        assert_eq!(
            err,
            PromptError::UnresolvableConflict {
                key: "tone".to_owned(),
                first: "formal".to_owned(),
                second: "casual".to_owned(),
            }
        );
    }

    #[test]
    fn override_canonical_bytes_are_byte_exact() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                None,
                vec![("tone", "formal")],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                None,
                vec![],
                None,
                None,
                vec![("tone", "casual")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("precedence resolves");
        let text = String::from_utf8(assembled.canonical_bytes.clone()).expect("utf8");
        let expected = concat!(
            "prompt/1\n",
            "core-version:bitty-core-prompt@1\n",
            "[layer:core-contract len=4]\ncore\n",
            "[layer:user len=0]\n\n",
            "[layer:project len=7]\nproject\n",
            "[layer:skills-profile len=0]\n\n",
            "[layer:runtime-turn len=0]\n\n",
            "[effective]\n",
            "allowed:*\n",
            "budget:*\n",
            "scope:*\n",
            "directive:tone=formal\n",
            "override:tone winner:core-contract wlen=6:formal overridden:project olen=6:casual\n",
            "end\n",
        );
        assert_eq!(text, expected);
        assert!(!text.contains('\r'));
    }

    #[test]
    fn override_free_canonical_has_no_override_lines() {
        // Same-value directives merge with no audit record and no canonical
        // change: existing override-free byte forms are preserved.
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::User,
                "u",
                None,
                vec![],
                None,
                None,
                vec![("tone", "concise")],
            ),
            full_layer(
                PromptLayer::Project,
                "p",
                None,
                vec![],
                None,
                None,
                vec![("tone", "concise")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("same value merges");
        assert!(assembled.merge_overrides.is_empty());
        let text = String::from_utf8(assembled.canonical_bytes.clone()).expect("utf8");
        assert!(!text.contains("override:"));
        assert!(text.contains("directive:tone=concise\n"));
    }

    #[test]
    fn override_record_and_bytes_are_input_order_invariant() {
        let layers_forward = vec![
            full_layer(
                PromptLayer::CoreContract,
                "core",
                None,
                vec![],
                None,
                None,
                vec![("tone", "formal"), ("style.mode", "terse")],
            ),
            full_layer(
                PromptLayer::Project,
                "project",
                None,
                vec![],
                None,
                None,
                vec![("tone", "casual"), ("style.mode", "loose")],
            ),
            full_layer(
                PromptLayer::RuntimeTurn,
                "turn",
                None,
                vec![],
                None,
                None,
                vec![("style.mode", "chatty")],
            ),
        ];
        let layers_reverse: Vec<LayerInput> = layers_forward.clone().into_iter().rev().collect();
        let left = assemble(&snapshot_with(layers_forward)).expect("forward assembles");
        let right = assemble(&snapshot_with(layers_reverse)).expect("reverse assembles");
        assert_eq!(left.merge_overrides, right.merge_overrides);
        assert_eq!(left.canonical_bytes, right.canonical_bytes);
        assert_eq!(left, right);
        // Two keys overridden: `tone` once (Project), `style.mode` twice
        // (Project + RuntimeTurn), sorted by key then overridden rank.
        assert_eq!(
            left.merge_overrides,
            vec![
                DirectiveOverride {
                    key: "style.mode".to_owned(),
                    winning_value: "terse".to_owned(),
                    winning_layer: PromptLayer::CoreContract,
                    overridden_value: "loose".to_owned(),
                    overridden_layer: PromptLayer::Project,
                },
                DirectiveOverride {
                    key: "style.mode".to_owned(),
                    winning_value: "terse".to_owned(),
                    winning_layer: PromptLayer::CoreContract,
                    overridden_value: "chatty".to_owned(),
                    overridden_layer: PromptLayer::RuntimeTurn,
                },
                DirectiveOverride {
                    key: "tone".to_owned(),
                    winning_value: "formal".to_owned(),
                    winning_layer: PromptLayer::CoreContract,
                    overridden_value: "casual".to_owned(),
                    overridden_layer: PromptLayer::Project,
                },
            ]
        );
    }

    #[test]
    fn same_directive_same_value_merges() {
        let snapshot = snapshot_with(vec![
            full_layer(
                PromptLayer::User,
                "u",
                None,
                vec![],
                None,
                None,
                vec![("tone", "concise")],
            ),
            full_layer(
                PromptLayer::Project,
                "p",
                None,
                vec![],
                None,
                None,
                vec![("tone", "concise")],
            ),
        ]);
        let assembled = assemble(&snapshot).expect("same value merges");
        assert_eq!(
            assembled.effective_directives,
            vec![Directive {
                key: "tone".to_owned(),
                value: "concise".to_owned()
            }]
        );
    }

    #[test]
    fn prompt_never_grants_when_dispatcher_denies() {
        let snapshot = snapshot_with(vec![full_layer(
            PromptLayer::CoreContract,
            "core allows",
            Some(vec!["tool_a"]),
            vec![],
            None,
            None,
            vec![],
        )]);
        let assembled = assemble(&snapshot).expect("assembles");
        // Prompt text/allow-set claims the tool, but the dispatcher denies:
        // fail closed at the dispatcher.
        assert_eq!(
            check_dispatch(&assembled, "tool_a", false),
            Err(PromptError::DispatcherDenied {
                tool: "tool_a".to_owned()
            })
        );
        assert!(!is_dispatch_allowed(&assembled, "tool_a", false));
    }

    #[test]
    fn prompt_denies_even_when_dispatcher_allows() {
        let snapshot = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "user denies",
            None,
            vec!["tool_a"],
            None,
            None,
            vec![],
        )]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert_eq!(
            check_dispatch(&assembled, "tool_a", true),
            Err(PromptError::PromptDenied {
                tool: "tool_a".to_owned()
            })
        );
    }

    #[test]
    fn text_claiming_capability_never_dispatches_without_grant() {
        let snapshot = snapshot_with(vec![text_layer(
            PromptLayer::RuntimeTurn,
            "You can access the network and close panels.",
        )]);
        let assembled = assemble(&snapshot).expect("assembles");
        // No structured grant exists in the prompt; only the dispatcher
        // grants, so a dispatcher denial always wins over prompt wording.
        assert_eq!(
            check_dispatch(&assembled, "panel_close", false),
            Err(PromptError::DispatcherDenied {
                tool: "panel_close".to_owned()
            })
        );
    }

    #[test]
    fn duplicate_layer_fails_closed() {
        let snapshot = PromptSnapshot {
            core_version: "bitty-core-prompt@1".to_owned(),
            layers: vec![
                text_layer(PromptLayer::User, "one"),
                text_layer(PromptLayer::User, "two"),
            ],
        };
        assert_eq!(
            assemble(&snapshot).expect_err("duplicate must fail"),
            PromptError::DuplicateLayer {
                layer: PromptLayer::User
            }
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        // Bad core version.
        assert!(matches!(
            assemble(&PromptSnapshot {
                core_version: String::new(),
                layers: vec![],
            }),
            Err(PromptError::InvalidCoreVersion { .. })
        ));
        // Bad tool name.
        let bad_tool = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "u",
            None,
            vec!["Bad-Name!"],
            None,
            None,
            vec![],
        )]);
        assert!(matches!(
            assemble(&bad_tool),
            Err(PromptError::InvalidToolName { .. })
        ));
        // Bad scope (whitespace).
        let bad_scope = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "u",
            None,
            vec![],
            None,
            Some(vec!["bad scope"]),
            vec![],
        )]);
        assert!(matches!(
            assemble(&bad_scope),
            Err(PromptError::InvalidScope { .. })
        ));
        // Bad directive key (`=` forbidden).
        let bad_key = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "u",
            None,
            vec![],
            None,
            None,
            vec![("bad=key", "v")],
        )]);
        assert!(matches!(
            assemble(&bad_key),
            Err(PromptError::InvalidDirectiveKey { .. })
        ));
        // Bad directive value (newline).
        let bad_value = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "u",
            None,
            vec![],
            None,
            None,
            vec![("tone", "a\nb")],
        )]);
        assert!(matches!(
            assemble(&bad_value),
            Err(PromptError::InvalidDirectiveValue { .. })
        ));
        // Zero budget.
        let bad_budget = snapshot_with(vec![full_layer(
            PromptLayer::User,
            "u",
            None,
            vec![],
            Some(0),
            None,
            vec![],
        )]);
        assert!(matches!(
            assemble(&bad_budget),
            Err(PromptError::InvalidBudget { .. })
        ));
        // CR in text.
        let bad_text = snapshot_with(vec![text_layer(PromptLayer::User, "a\rb")]);
        assert!(matches!(
            assemble(&bad_text),
            Err(PromptError::InvalidLayerText { .. })
        ));
    }

    #[test]
    fn bounds_fail_closed() {
        // Oversized text.
        let big = "x".repeat(MAX_LAYER_TEXT_BYTES + 1);
        let snapshot = snapshot_with(vec![text_layer(PromptLayer::User, &big)]);
        assert!(matches!(
            assemble(&snapshot),
            Err(PromptError::LayerTextTooLarge { .. })
        ));
        // Too many directives.
        let many: Vec<Directive> = (0..MAX_DIRECTIVES_PER_LAYER + 1)
            .map(|index| Directive {
                key: format!("k{index}"),
                value: "v".to_owned(),
            })
            .collect();
        let snapshot = PromptSnapshot {
            core_version: "bitty-core-prompt@1".to_owned(),
            layers: vec![LayerInput {
                layer: PromptLayer::User,
                text: String::new(),
                allowed_tools: None,
                denied_tools: Vec::new(),
                budget_ceiling_bytes: None,
                allowed_scopes: None,
                directives: many,
            }],
        };
        assert!(matches!(
            assemble(&snapshot),
            Err(PromptError::TooManyDirectives { .. })
        ));
        // Too many tools.
        let many_tools: Vec<String> = (0..MAX_TOOL_ENTRIES_PER_LAYER + 1)
            .map(|index| format!("tool_{index}"))
            .collect();
        let snapshot = PromptSnapshot {
            core_version: "bitty-core-prompt@1".to_owned(),
            layers: vec![LayerInput {
                layer: PromptLayer::User,
                text: String::new(),
                allowed_tools: Some(many_tools),
                denied_tools: Vec::new(),
                budget_ceiling_bytes: None,
                allowed_scopes: None,
                directives: Vec::new(),
            }],
        };
        assert!(matches!(
            assemble(&snapshot),
            Err(PromptError::TooManyTools { .. })
        ));
    }

    #[test]
    fn missing_layers_assemble_as_empty_deterministic() {
        let empty = PromptSnapshot {
            core_version: "bitty-core-prompt@1".to_owned(),
            layers: vec![],
        };
        let assembled = assemble(&empty).expect("empty assembles");
        assert_eq!(assembled.sections.len(), 5);
        for section in &assembled.sections {
            assert_eq!(section.text, "");
        }
        assert_eq!(assembled.effective_allowed_tools, None);
        assert!(assembled.effective_denied_tools.is_empty());
        assert_eq!(assembled.effective_budget_ceiling_bytes, None);
        assert_eq!(assembled.effective_scopes, None);
        assert!(assembled.effective_directives.is_empty());
        // Unconstrained prompt still needs the dispatcher grant.
        assert!(!is_dispatch_allowed(&assembled, "tool_a", false));
        assert!(is_dispatch_allowed(&assembled, "tool_a", true));
    }

    #[test]
    fn validation_helpers_reject_shapes() {
        assert!(validate_core_version("bitty-core-prompt@1").is_ok());
        assert!(validate_core_version("").is_err());
        assert!(validate_core_version("Bad Version!").is_err());
        assert!(validate_scope("workspace.read").is_ok());
        assert!(validate_scope("bad scope").is_err());
        assert!(validate_directive_key("tone").is_ok());
        assert!(validate_directive_key("").is_err());
        assert!(validate_directive_value("tone", "concise").is_ok());
        assert!(validate_directive_value("tone", "a\nb").is_err());
        assert!(validate_budget_ceiling(1).is_ok());
        assert!(validate_budget_ceiling(0).is_err());
        assert_eq!(common_prefix_len(b"abc", b"abd"), 2);
        assert_eq!(common_prefix_len(b"", b"abc"), 0);
    }

    #[test]
    fn section_text_returns_layer_text() {
        let snapshot = snapshot_with(vec![text_layer(PromptLayer::Project, "proj")]);
        let assembled = assemble(&snapshot).expect("assembles");
        assert_eq!(assembled.section_text(PromptLayer::Project), "proj");
        assert_eq!(assembled.section_text(PromptLayer::User), "");
    }

    // AI-0045 declarative loading tests: inline fixture strings only (no
    // filesystem access anywhere, not even in tests). Every assertion pins
    // fail-closed typed errors, narrowing preservation, and determinism.

    #[test]
    fn project_from_str_parses_documented_format() {
        let source = concat!(
            "# project intent\n",
            "directive.tone = concise\n",
            "allow_tool = panel_open\n",
            "deny_tool = panel_close\n",
            "scope = workspace.read\n",
            "budget = 4096\n",
            "text:\n",
            "Ship the widget.\n",
            "Second line.\n",
        );
        let layer = LayerInput::project_from_str(source).expect("parses");
        assert_eq!(layer.layer, PromptLayer::Project);
        assert_eq!(layer.text, "Ship the widget.\nSecond line.");
        assert_eq!(layer.allowed_tools, Some(vec!["panel_open".to_owned()]));
        assert_eq!(layer.denied_tools, vec!["panel_close".to_owned()]);
        assert_eq!(layer.budget_ceiling_bytes, Some(4096));
        assert_eq!(
            layer.allowed_scopes,
            Some(vec!["workspace.read".to_owned()])
        );
        assert_eq!(
            layer.directives,
            vec![Directive {
                key: "tone".to_owned(),
                value: "concise".to_owned()
            }]
        );
        let assembled = assemble(&snapshot_with(vec![layer])).expect("assembles");
        assert_eq!(
            assembled.section_text(PromptLayer::Project),
            "Ship the widget.\nSecond line."
        );
    }

    #[test]
    fn project_from_str_rejects_malformed_inputs() {
        // Unknown header key.
        assert!(matches!(
            LayerInput::project_from_str("frobnicate = 1\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Header line without `=`.
        assert!(matches!(
            LayerInput::project_from_str("just words\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Non-numeric budget is structural, not a range error.
        assert!(matches!(
            LayerInput::project_from_str("budget = lots\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Out-of-range budget propagates the shared validator error.
        assert!(matches!(
            LayerInput::project_from_str("budget = 0\n"),
            Err(PromptError::InvalidBudget { .. })
        ));
        // Bad tool shape propagates the shared validator error.
        assert!(matches!(
            LayerInput::project_from_str("allow_tool = Bad-Name!\n"),
            Err(PromptError::InvalidToolName { .. })
        ));
        // Same-file directive conflict fail-closes (same-layer semantics).
        assert!(matches!(
            LayerInput::project_from_str("directive.tone = a\ndirective.tone = b\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Conflicting budgets in one file are an authoring error.
        assert!(matches!(
            LayerInput::project_from_str("budget = 1\nbudget = 2\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // A `text:` line with trailing content is not the marker.
        assert!(matches!(
            LayerInput::project_from_str("text: hello\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Empty header key.
        assert!(matches!(
            LayerInput::project_from_str("= value\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Non-LF bytes fail closed.
        assert!(matches!(
            LayerInput::project_from_str("text:\r\nhi\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        assert!(matches!(
            LayerInput::project_from_str("text:\nhi\0\n"),
            Err(PromptError::MalformedProject { .. })
        ));
        // Empty file is valid: empty Project layer, no constraints.
        let empty = LayerInput::project_from_str("").expect("empty parses");
        assert_eq!(empty.layer, PromptLayer::Project);
        assert_eq!(empty.text, "");
        assert_eq!(empty.allowed_tools, None);
    }

    #[test]
    fn project_paths_admit_only_explicit_roots() {
        let roots = [".wheel"];
        assert!(admit_project_path(&roots, ".wheel/prompt.conf").is_ok());
        assert!(admit_project_path(&roots, ".wheel/nested/file.conf").is_ok());
        // Absolute-root style works for hosts that pass absolute paths.
        assert!(admit_project_path(&["/repo/.wheel"], "/repo/.wheel/a.conf").is_ok());
        let escapes = [
            "../evil.conf",
            ".wheel/../evil.conf",
            ".wheel/./sneaky.conf",
            ".wheel//double.conf",
            ".wheel/",
            "/etc/passwd",
            "/repo/other.conf",
            "other/file.conf",
            "",
            ".",
            ".wheel\\win.conf",
            ".wheel/bad\0conf",
            ".wheel/bad\rc",
            ".wheel/bad\nc",
        ];
        for bad in escapes {
            assert!(
                matches!(
                    admit_project_path(&roots, bad),
                    Err(PromptError::InvalidProjectPath { .. })
                ),
                "must refuse {bad:?}"
            );
        }
        // Empty roots admit nothing (fail-closed default).
        assert!(matches!(
            admit_project_path(&[], ".wheel/a.conf"),
            Err(PromptError::InvalidProjectPath { .. })
        ));
    }

    #[test]
    fn project_multi_file_merge_is_deterministic_and_narrowing() {
        let files = [
            (
                ".wheel/b.conf",
                "deny_tool = panel_close\nbudget = 1024\ntext:\nBee.\n",
            ),
            (
                ".wheel/a.conf",
                "allow_tool = panel_open\ndirective.tone = concise\nbudget = 4096\ntext:\nAye.\n",
            ),
        ];
        let forward =
            LayerInput::project_from_files_under_roots(&[".wheel"], &files).expect("merges");
        let mut reversed = files;
        reversed.reverse();
        let backward =
            LayerInput::project_from_files_under_roots(&[".wheel"], &reversed).expect("merges");
        assert_eq!(forward, backward);
        // Sorted by path: a.conf text first.
        assert_eq!(forward.text, "Aye.\nBee.");
        assert_eq!(forward.allowed_tools, Some(vec!["panel_open".to_owned()]));
        assert_eq!(forward.denied_tools, vec!["panel_close".to_owned()]);
        // Budgets take the minimum across files.
        assert_eq!(forward.budget_ceiling_bytes, Some(1024));
        assert_eq!(forward.directives.len(), 1);
        // An escape anywhere in the set refuses the whole merge.
        assert!(matches!(
            LayerInput::project_from_files_under_roots(
                &[".wheel"],
                &[
                    (".wheel/a.conf", "text:\nHi.\n"),
                    ("../evil.conf", "text:\nEvil.\n")
                ]
            ),
            Err(PromptError::InvalidProjectPath { .. })
        ));
        // Duplicate paths are ambiguous discovery: fail closed.
        assert!(matches!(
            LayerInput::project_from_files_under_roots(
                &[".wheel"],
                &[
                    (".wheel/a.conf", "text:\nOne.\n"),
                    (".wheel/a.conf", "text:\nTwo.\n")
                ]
            ),
            Err(PromptError::MalformedProject { .. })
        ));
    }

    #[test]
    fn project_same_layer_directive_conflict_fails_closed() {
        let conflict = [
            (".wheel/a.conf", "directive.tone = concise\n"),
            (".wheel/b.conf", "directive.tone = casual\n"),
        ];
        assert!(matches!(
            LayerInput::project_from_files_under_roots(&[".wheel"], &conflict),
            Err(PromptError::UnresolvableConflict { key, .. }) if key == "tone"
        ));
        // Same key with the same value merges silently.
        let agreed = [
            (".wheel/a.conf", "directive.tone = concise\n"),
            (".wheel/b.conf", "directive.tone = concise\n"),
        ];
        let layer =
            LayerInput::project_from_files_under_roots(&[".wheel"], &agreed).expect("merges");
        assert_eq!(layer.directives.len(), 1);
    }

    #[test]
    fn skills_from_str_parses_sorted_entries() {
        let source = concat!(
            "version = 1\n",
            "---\n",
            "name = zeta\n",
            "version = 1\n",
            "text:\n",
            "Zed fragment.\n",
            "---\n",
            "name = alpha\n",
            "version = 1\n",
            "directive.tone = terse\n",
            "text:\n",
            "Alpha fragment.\n",
        );
        let layer = LayerInput::skills_from_str(source).expect("parses");
        assert_eq!(layer.layer, PromptLayer::SkillsProfile);
        // Deterministic name order regardless of document order.
        assert_eq!(layer.text, "Alpha fragment.\nZed fragment.");
        assert_eq!(
            layer.directives,
            vec![Directive {
                key: "tone".to_owned(),
                value: "terse".to_owned()
            }]
        );
        // Document order swapped: identical layer.
        let swapped = concat!(
            "version = 1\n",
            "---\n",
            "name = alpha\n",
            "version = 1\n",
            "directive.tone = terse\n",
            "text:\n",
            "Alpha fragment.\n",
            "---\n",
            "name = zeta\n",
            "version = 1\n",
            "text:\n",
            "Zed fragment.\n",
        );
        let again = LayerInput::skills_from_str(swapped).expect("parses");
        assert_eq!(layer, again);
        let assembled = assemble(&snapshot_with(vec![layer])).expect("assembles");
        assert_eq!(
            assembled.section_text(PromptLayer::SkillsProfile),
            "Alpha fragment.\nZed fragment."
        );
    }

    #[test]
    fn skills_registry_requires_supported_versions() {
        // Missing registry header.
        assert!(matches!(
            LayerInput::skills_from_str("---\nname = a\nversion = 1\n"),
            Err(PromptError::MalformedSkill { .. })
        ));
        // Empty document misses the header too.
        assert!(matches!(
            LayerInput::skills_from_str(""),
            Err(PromptError::MalformedSkill { .. })
        ));
        // Unknown registry version refused with a typed error.
        assert!(matches!(
            LayerInput::skills_from_str("version = 2\n"),
            Err(PromptError::UnsupportedSkillVersion { version }) if version == "2"
        ));
        // Entry without a version field.
        assert!(matches!(
            LayerInput::skills_from_str("version = 1\n---\nname = a\ntext:\nhi\n"),
            Err(PromptError::MissingSkillVersion { name }) if name == "a"
        ));
        // Entry with an unknown version.
        assert!(matches!(
            LayerInput::skills_from_str("version = 1\n---\nname = a\nversion = 9\n"),
            Err(PromptError::UnsupportedSkillVersion { version }) if version == "9"
        ));
        // Header-only registry is valid: empty SkillsProfile layer.
        let bare = LayerInput::skills_from_str("version = 1\n").expect("bare parses");
        assert_eq!(bare.layer, PromptLayer::SkillsProfile);
        assert_eq!(bare.text, "");
    }

    #[test]
    fn skills_registry_rejects_malformed_inputs() {
        // Unknown entry key.
        assert!(matches!(
            LayerInput::skills_from_str(
                "version = 1\n---\nname = a\nversion = 1\nfrobnicate = x\n"
            ),
            Err(PromptError::MalformedSkill { .. })
        ));
        // Entry without a name.
        assert!(matches!(
            LayerInput::skills_from_str("version = 1\n---\nversion = 1\ntext:\nhi\n"),
            Err(PromptError::MalformedSkill { .. })
        ));
        // Entry name runs the independent skill-name validator (hyphen
        // allowed, dots still rejected); shape failures reuse
        // `InvalidToolName` with no new variant.
        assert!(matches!(
            LayerInput::skills_from_str("version = 1\n---\nname = Bad!\nversion = 1\n"),
            Err(PromptError::InvalidToolName { .. })
        ));
        // Duplicate entry names fail closed.
        assert!(matches!(
            LayerInput::skills_from_str(concat!(
                "version = 1\n",
                "---\n",
                "name = dup\n",
                "version = 1\n",
                "---\n",
                "name = dup\n",
                "version = 1\n",
            )),
            Err(PromptError::DuplicateSkill { name }) if name == "dup"
        ));
        // Same-key-different-value directives across entries: same-layer
        // conflict, fail closed.
        assert!(matches!(
            LayerInput::skills_from_str(concat!(
                "version = 1\n",
                "---\n",
                "name = a_one\n",
                "version = 1\n",
                "directive.tone = concise\n",
                "---\n",
                "name = b_two\n",
                "version = 1\n",
                "directive.tone = casual\n",
            )),
            Err(PromptError::UnresolvableConflict { key, .. }) if key == "tone"
        ));
    }

    #[test]
    fn hyphenated_skill_name_accepted_independent_of_tool_namespace() {
        // P2-4: skill names allow `-` independent of the tool namespace
        // (`TB-2` forbids it), so `my-skill` assembles while the tool
        // validator still rejects it.
        assert!(validate_skill_name("my-skill").is_ok());
        assert!(validate_prompt_tool_name("my-skill").is_err());
        // Dots stay rejected in both namespaces; bad shapes still reuse
        // `InvalidToolName` with no new variant.
        assert!(validate_skill_name("bad.name").is_err());
        assert!(matches!(
            validate_skill_name("Bad!"),
            Err(PromptError::InvalidToolName { .. })
        ));
        let layer = LayerInput::skills_from_str(concat!(
            "version = 1\n",
            "---\n",
            "name = my-skill\n",
            "version = 1\n",
            "text:\n",
            "Hyphenated fragment.\n",
        ))
        .expect("hyphenated skill accepted");
        assert_eq!(layer.layer, PromptLayer::SkillsProfile);
        let assembled = assemble(&snapshot_with(vec![layer])).expect("assembles");
        assert!(
            assembled
                .section_text(PromptLayer::SkillsProfile)
                .contains("Hyphenated fragment.")
        );
    }

    #[test]
    fn loaded_project_cannot_widen_upper_layers() {
        let user = full_layer(
            PromptLayer::User,
            "user",
            Some(vec!["tool_a"]),
            vec!["tool_b"],
            Some(1000),
            Some(vec!["workspace.read"]),
            vec![],
        );
        let project = LayerInput::project_from_str(concat!(
            "allow_tool = tool_a\n",
            "allow_tool = tool_b\n",
            "allow_tool = tool_c\n",
            "deny_tool = tool_a\n",
            "scope = workspace.read\n",
            "scope = terminal.read\n",
            "budget = 8192\n",
            "text:\n",
            "Project wants everything.\n",
        ))
        .expect("parses");
        let assembled = assemble(&snapshot_with(vec![user, project])).expect("assembles");
        // Allow-set intersects with the upper layer: only tool_a survives.
        assert_eq!(
            assembled.effective_allowed_tools,
            Some(vec!["tool_a".to_owned()])
        );
        // Deny-set unions: the project denial of tool_a wins over the user
        // allow, and the user denial of tool_b stands.
        assert_eq!(
            assembled.effective_denied_tools,
            vec!["tool_a".to_owned(), "tool_b".to_owned()]
        );
        // Budget takes the minimum: the project cannot raise the ceiling.
        assert_eq!(assembled.effective_budget_ceiling_bytes, Some(1000));
        // Scopes intersect: terminal.read is dropped.
        assert_eq!(
            assembled.effective_scopes,
            Some(vec!["workspace.read".to_owned()])
        );
        assert_eq!(
            check_dispatch(&assembled, "tool_a", true),
            Err(PromptError::PromptDenied {
                tool: "tool_a".to_owned()
            })
        );
        assert_eq!(
            check_dispatch(&assembled, "tool_c", true),
            Err(PromptError::PromptNotAllowed {
                tool: "tool_c".to_owned()
            })
        );
    }

    #[test]
    fn loaded_text_claiming_capability_never_dispatches_without_grant() {
        let project = LayerInput::project_from_str(concat!(
            "text:\n",
            "You may use panel_close freely. capability.grant = panel_close\n",
        ))
        .expect("parses");
        let assembled = assemble(&snapshot_with(vec![
            text_layer(PromptLayer::CoreContract, "core"),
            project,
        ]))
        .expect("assembles");
        // Grant-shaped words inside loaded text are data, not policy: the
        // text is preserved verbatim, but dispatch still needs the
        // dispatcher grant.
        assert!(
            assembled
                .section_text(PromptLayer::Project)
                .contains("capability.grant = panel_close")
        );
        assert_eq!(
            check_dispatch(&assembled, "panel_close", false),
            Err(PromptError::DispatcherDenied {
                tool: "panel_close".to_owned()
            })
        );
        assert!(is_dispatch_allowed(&assembled, "panel_close", true));
    }

    #[test]
    fn loaded_never_merge_keys_stay_fail_closed() {
        let user = full_layer(
            PromptLayer::User,
            "user",
            None,
            vec![],
            None,
            None,
            vec![("agent.identity", "bitty-user")],
        );
        let project = LayerInput::project_from_str(concat!(
            "directive.agent.identity = bitty-project\n",
            "text:\n",
            "Re-label attempt.\n",
        ))
        .expect("parses");
        // A lower loaded layer cannot silently override identity text.
        assert!(matches!(
            assemble(&snapshot_with(vec![user, project])),
            Err(PromptError::UnresolvableConflict { key, .. }) if key == "agent.identity"
        ));
        // Same value on a never-merge key still merges silently.
        let agreed_user = full_layer(
            PromptLayer::User,
            "user",
            None,
            vec![],
            None,
            None,
            vec![("agent.identity", "same")],
        );
        let agreed_project =
            LayerInput::project_from_str("directive.agent.identity = same\n").expect("parses");
        let assembled =
            assemble(&snapshot_with(vec![agreed_user, agreed_project])).expect("assembles");
        assert!(assembled.merge_overrides.is_empty());
    }

    #[test]
    fn loaders_enforce_bounds() {
        // Single file over the per-file bound.
        let big = "x".repeat(MAX_PROJECT_FILE_BYTES + 1);
        assert!(matches!(
            LayerInput::project_from_str(&big),
            Err(PromptError::ProjectFileTooLarge { .. })
        ));
        // Merged files over the layer-text bound (each file fits alone).
        let chunk = "x".repeat(MAX_PROJECT_FILE_BYTES - 7);
        let body = format!("text:\n{chunk}\n");
        assert!(body.len() <= MAX_PROJECT_FILE_BYTES);
        let files = [
            (".wheel/a.conf", body.as_str()),
            (".wheel/b.conf", body.as_str()),
            (".wheel/c.conf", body.as_str()),
        ];
        assert!(matches!(
            LayerInput::project_from_files_under_roots(&[".wheel"], &files),
            Err(PromptError::LayerTextTooLarge { .. })
        ));
        // More files than the discovery bound.
        let one = "text:\nhi\n";
        let many: Vec<(&str, &str)> = (0..MAX_PROJECT_FILES + 1)
            .map(|_| (".wheel/a.conf", one))
            .collect();
        assert!(matches!(
            LayerInput::project_from_files_under_roots(&[".wheel"], &many),
            Err(PromptError::TooManyProjectFiles { .. })
        ));
        // Exactly the discovery bound is accepted (distinct paths; duplicate
        // paths fail closed regardless of count).
        let exact: Vec<(String, &str)> = (0..MAX_PROJECT_FILES)
            .map(|index| (format!(".wheel/f{index:02}.conf"), one))
            .collect();
        let exact_refs: Vec<(&str, &str)> = exact
            .iter()
            .map(|(path, body)| (path.as_str(), *body))
            .collect();
        assert!(
            LayerInput::project_from_files_under_roots(&[".wheel"], &exact_refs).is_ok(),
            "exactly MAX_PROJECT_FILES must be accepted"
        );
        // Registry over the whole-document bound.
        let huge = "x".repeat(MAX_SKILL_REGISTRY_BYTES + 1);
        assert!(matches!(
            LayerInput::skills_from_str(&huge),
            Err(PromptError::SkillRegistryTooLarge { .. })
        ));
        // Fragment over the per-entry bound.
        let fat = "x".repeat(MAX_SKILL_ENTRY_BYTES + 1);
        let registry = format!("version = 1\n---\nname = fat\nversion = 1\ntext:\n{fat}\n");
        assert!(matches!(
            LayerInput::skills_from_str(&registry),
            Err(PromptError::SkillEntryTooLarge { .. })
        ));
        // More entries than the registry bound.
        let mut crowded = "version = 1\n".to_owned();
        for index in 0..MAX_SKILL_ENTRIES + 1 {
            crowded.push_str(&format!("---\nname = skill{index:02}\nversion = 1\n"));
        }
        assert!(matches!(
            LayerInput::skills_from_str(&crowded),
            Err(PromptError::TooManySkills { .. })
        ));
        // Exactly the registry bound is accepted.
        let mut exact = "version = 1\n".to_owned();
        for index in 0..MAX_SKILL_ENTRIES {
            exact.push_str(&format!("---\nname = skill{index:02}\nversion = 1\n"));
        }
        assert!(
            LayerInput::skills_from_str(&exact).is_ok(),
            "exactly MAX_SKILL_ENTRIES must be accepted"
        );
    }
}

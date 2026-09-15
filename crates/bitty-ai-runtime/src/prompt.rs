//! Deterministic five-layer prompt assembly with narrowing-only policy.
//!
//! Mirrors the draft `docs/specifications/prompt-layering-design.md` (status:
//! draft, not an accepted contract): five layers from most stable to most
//! dynamic — Core Contract, User, Project/`.bitty`, Skills/Profile, and
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
//! - Generic `directives` with the same key but different values fail closed
//!   with [`PromptError::UnresolvableConflict`]: the assembler refuses to
//!   guess which instruction wins.
//!
//! > **A prompt never grants a capability.** Prompt text describes what should
//! > be done; the dispatcher decides what can be done. Even when the prompt
//! > allows a tool, [`check_dispatch`] still requires the dispatcher grant;
//! > when the dispatcher denies, dispatch is refused. When the prompt denies
//! > or omits (under a constrained allow-set), dispatch is refused even when
//! > the dispatcher would grant. All refusals are fail-closed with no partial
//! > assembly or dispatch.
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
//! merge-semantics facet (conflict diagnostics). It does not close any AIQ:
//! canonical owner, digest algorithm, cache-key scope, epoch schema, and
//! reviewer acceptance stay open in the register.
//!
//! # Bounds and determinism rules
//!
//! - Std only. No network, filesystem, clock, threads, async runtime, or
//!   secrets. Single-agent (v0.1): profiles compose one agent only.
//! - Every bound is enforced fail-closed with [`PromptError`]; no partial
//!   output is returned on error.
//! - Missing layers assemble as empty text with no constraints, so the
//!   canonical form always carries exactly five text sections in rank order.

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

/// Prompt layer from most stable (Core) to most dynamic (Runtime/Turn).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromptLayer {
    /// Built-in runtime contract: versioned, session-pinned, non-overridable.
    CoreContract,
    /// User-owned global instructions; preferences only, no capability effect.
    User,
    /// Project-owned `.bitty` manifest intent; untrusted until reviewed.
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
/// layers merges to one entry; the same key with different values is an
/// unresolvable conflict and fails closed at assembly.
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
    /// Budget ceiling is zero or exceeds [`MAX_BUDGET_CEILING_BYTES`].
    InvalidBudget {
        /// Rejected ceiling.
        actual: usize,
    },
    /// The same directive key carries different values: the assembler
    /// refuses to guess and fails closed.
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
    /// allow-set that omits this tool.
    PromptNotAllowed {
        /// Requested tool.
        tool: String,
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
/// minimum, and directives by key with fail-closed conflict on differing
/// values; then render the canonical byte form.
///
/// Content text is never interpreted: no keyword, directive, or instruction
/// scan runs over layer text, and text bytes never select, widen, or
/// reallocate policy. Enforcement evidence toward AIQ-12 (and the AIQ-31/34
/// merge facet); no register entry is closed.
///
/// # Errors
///
/// Returns [`PromptError`] for any validation failure, an unresolvable
/// directive conflict, or an over-bound canonical form. No partial assembly
/// is returned.
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

    // Directives: same key + same value merges; same key + different value
    // fails closed (no precedence guess).
    let mut by_key: BTreeMap<String, String> = BTreeMap::new();
    for input in &ordered {
        for directive in &input.directives {
            match by_key.get(&directive.key) {
                None => {
                    by_key.insert(directive.key.clone(), directive.value.clone());
                }
                Some(first) => {
                    if first != &directive.value {
                        return Err(PromptError::UnresolvableConflict {
                            key: directive.key.clone(),
                            first: first.clone(),
                            second: directive.value.clone(),
                        });
                    }
                }
            }
        }
    }
    let directives: Vec<Directive> = by_key
        .into_iter()
        .map(|(key, value)| Directive { key, value })
        .collect();

    let mut assembled = AssembledPrompt {
        core_version: snapshot.core_version.clone(),
        sections,
        effective_allowed_tools: allowed,
        effective_denied_tools: denied,
        effective_budget_ceiling_bytes: budget,
        effective_scopes: scopes,
        effective_directives: directives,
        canonical_bytes: Vec::new(),
    };
    let bytes = render_canonical(&assembled)?;
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
/// end\n
/// ```
///
/// Text sections always appear in stable rank order with byte-length
/// prefixes so embedded newlines or section-like text never shift parsing.
/// Policy lists are lexicographically sorted. The trailing `[effective]`
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

    #[test]
    fn conflicting_directives_fail_closed() {
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
        let err = assemble(&snapshot).expect_err("conflict must fail");
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
}

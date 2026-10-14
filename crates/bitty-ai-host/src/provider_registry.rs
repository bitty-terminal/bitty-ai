//! Host-side pure-type mirror of the provider-registry seam (AI-0271).
//!
//! Phase 2 S1 per the AI-0263 survey: a dependency-free mirror of the
//! registry/selection/fallback shapes from `bitty-ai-runtime`
//! (`selection.rs` plus the `MP-2` support shapes from `provider.rs`) for
//! host-boundary traversal. This module is data plus pure functions only:
//! no `ModelProvider` trait, no `list_models`/`complete` wiring, no `MP-4`
//! caller-scope filter, no closed-vocabulary enforcement, no currency, no
//! IPC, no threads, no filesystem or network access, and no dependencies
//! beyond `std`.
//!
//! ## Mirror discipline
//!
//! Every bound, variant, grammar, refusal order, gate order, and cost rule
//! here matches the runtime source exactly so host and runtime agree on the
//! wire shape without the host crate depending on the runtime crate:
//!
//! - Bounds: [`MAX_MODEL_NAME_LEN`], [`MAX_REGISTERED_MODELS`],
//!   [`MAX_ALIASES`], [`MAX_ALIAS_CANDIDATES`] (plus [`MAX_PROVIDER_ID_LEN`]
//!   for the mirrored `MP-2` id check).
//! - Errors: [`SelectionError`] with the same variants and [`Display`]
//!   strings; [`ProviderError`] with the same variants and [`Display`]
//!   strings so [`fallback_directive`] stays total.
//! - Grammar: [`validate_model_name`] (same shape) and
//!   [`validate_provider_id`] (same `MP-2` shape).
//! - Registry: [`ModelRegistration`] (`new`/`snapshot_from`),
//!   [`RegisteredModel`], [`ProviderRegistry`] (`register`/`register_alias`/
//!   `lookup`/`entries`/`resolve_alias`/`select`) with the identical
//!   fail-closed refusal order (duplicate/conflict before capacity, no state
//!   change on any refusal).
//! - Selection: [`SelectRequest`] (`matches`/`describe`), [`SelectedModel`]
//!   (`from_entry`), [`estimate_cost`] plus `effective_cost_weight`
//!   (uncalibrated `0` counts as baseline `1`).
//! - Fallback: [`FallbackDirective`] (`Advance`/`Stop`, `Unknown` maps to
//!   `Stop`) plus [`fallback_directive`] classification.
//! - Mediation: [`TerminalModelMetadata`] (`from_descriptor`, one-way
//!   detached snapshot).
//!
//! The supporting [`ModelCapability`], [`ModelDescriptor`], and
//! [`ProviderError`] types are minimal local mirrors of the runtime
//! `provider.rs` shapes the seam needs (capability sets, descriptor bridge,
//! failure taxonomy). They are not provider implementations, hold no
//! credentials, and perform no I/O.
//!
//! Deny-all posture holds by construction: nothing here grants execution
//! authority — selection returns detached snapshots, aliases are data (never
//! fall-through), and every refusal fails closed with no partial state.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Maximum model name length in bytes (mirrors `selection.rs`).
pub const MAX_MODEL_NAME_LEN: usize = 128;
/// Maximum registered models per registry (mirrors `selection.rs`).
pub const MAX_REGISTERED_MODELS: usize = 64;
/// Maximum semantic aliases per registry (mirrors `selection.rs`).
pub const MAX_ALIASES: usize = 32;
/// Maximum candidates per alias (mirrors `selection.rs`).
pub const MAX_ALIAS_CANDIDATES: usize = 8;
/// Maximum provider id length in bytes (mirrors the `MP-2` bound in
/// `provider.rs`).
pub const MAX_PROVIDER_ID_LEN: usize = 64;

/// Model capability flags (mirrors `provider.rs`).
///
/// Capabilities are routing data, never authority: selection matches on
/// required capability sets and execution still goes through the provider
/// boundary. No closed-vocabulary rejection is pinned here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCapability {
    /// Plain text completion.
    Text,
    /// Incremental streaming output.
    Streaming,
    /// Model may request tool calls.
    ToolUse,
    /// Image understanding (vision input alongside text).
    ImageInput,
    /// Audio understanding (speech/sound input alongside text).
    AudioInput,
    /// Audio generation (speech/sound output).
    AudioOutput,
    /// Video understanding (moving-image input alongside text).
    VideoInput,
    /// Model may emit reasoning/thinking tokens (routing bit only).
    Reasoning,
}

/// One registry-known model descriptor (mirrors the `MP-2` skeleton subset
/// in `provider.rs`): `name` plus `capabilities`. Routing metadata
/// (`provider_id`, context window, cost weights) lives on
/// [`ModelRegistration`], not here; bridging stays host-side via
/// [`ModelRegistration::snapshot_from`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// Model name as referenced by selection.
    pub name: String,
    /// Capability flags for this model.
    pub capabilities: Vec<ModelCapability>,
}

/// Terminal-side generic model metadata mediation placeholder (mirrors
/// `provider.rs`).
///
/// A model name plus a capability snapshot, derivable from a
/// [`ModelDescriptor`], with no I/O methods, no secret fields, and no clock.
/// The snapshot is one-way and detached: later descriptor changes do not
/// leak through a previously derived view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalModelMetadata {
    /// Registry-known model name (mirrors [`ModelDescriptor::name`]).
    pub name: String,
    /// Capability snapshot (mirrors [`ModelDescriptor::capabilities`]).
    pub capabilities: Vec<ModelCapability>,
}

impl TerminalModelMetadata {
    /// Snapshot generic metadata from a registry descriptor. No credentials
    /// are read (there are none to read) and no I/O happens.
    #[must_use]
    pub fn from_descriptor(descriptor: &ModelDescriptor) -> Self {
        Self {
            name: descriptor.name.clone(),
            capabilities: descriptor.capabilities.clone(),
        }
    }
}

/// Provider boundary errors (mirrors `provider.rs`).
///
/// Every variant fails closed: no partial turn is produced. The taxonomy
/// exists here so [`fallback_directive`] stays pure and total over it:
/// adding a variant breaks compilation there, which is the fail-closed
/// response to a new failure mode (it must be classified before any
/// fallback uses it). No variant carries secrets, credentials, or key
/// material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// `model` names no registry-known model.
    UnknownModel {
        /// Requested model name.
        name: String,
    },
    /// Request would exceed the context budget in bytes.
    BudgetExceeded {
        /// Budget in bytes.
        limit: usize,
        /// Observed message bytes.
        actual: usize,
    },
    /// Scripted latency exceeded the caller timeout.
    Timeout {
        /// Requested timeout in milliseconds.
        timeout_ms: u64,
        /// Observed latency in milliseconds.
        latency_ms: u64,
    },
    /// Requested timeout exceeds the hard ceiling.
    TimeoutTooLarge {
        /// Hard ceiling in milliseconds.
        max: u64,
        /// Requested timeout in milliseconds.
        actual: u64,
    },
    /// Provider id violates the `MP-2` shape.
    InvalidProviderId {
        /// Rejected id.
        id: String,
    },
    /// Transport failure before any model effect. No request bytes are known
    /// to have been applied; the fallback policy may advance.
    Transport {
        /// Owning provider id.
        provider: String,
        /// What failed (no secrets, no credentials, no key material).
        reason: String,
    },
    /// Provider refused authorization. Reports the refusal only; the
    /// fallback policy stops so the operator reconciles credentials instead
    /// of spraying retries across providers.
    Auth {
        /// Owning provider id.
        provider: String,
        /// Refusal detail (no secrets, no credentials, no key material).
        reason: String,
    },
    /// Provider rate limit hit.
    RateLimited {
        /// Owning provider id.
        provider: String,
        /// Advised wait in milliseconds; `None` when the provider gave none.
        retry_after_ms: Option<u64>,
    },
    /// Selected model lacks capabilities the request required. A routing bug
    /// or registry drift, never a reason to try another model blindly.
    CapabilityMismatch {
        /// Owning provider id.
        provider: String,
        /// Requested model name.
        model: String,
        /// Required capabilities the model does not advertise.
        missing: Vec<ModelCapability>,
    },
    /// Model is known but unavailable. The fallback policy may advance.
    ModelUnavailable {
        /// Owning provider id.
        provider: String,
        /// Unavailable model name.
        model: String,
    },
    /// The effect may have happened but acknowledgement was lost. Reconcile
    /// (status inspection or user direction) before retry; never fall back
    /// blindly.
    Unknown {
        /// Owning provider id.
        provider: String,
        /// What is uncertain.
        reason: String,
    },
    /// Declared sampling contract violates its validated ranges. No value is
    /// echoed, so logs stay static.
    InvalidSampling,
    /// Declared sampling contract is valid but the backend cannot carry one
    /// or more declared fields. Names the backend field kind with a static
    /// label; no caller value is echoed.
    UnsupportedSampling {
        /// Static backend field label (never caller input).
        field: &'static str,
    },
}

impl Display for ProviderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::UnknownModel { name } => write!(f, "unknown model: {name}"),
            Self::BudgetExceeded { limit, actual } => write!(
                f,
                "context budget exceeded: {actual} bytes over {limit} byte limit"
            ),
            Self::Timeout {
                timeout_ms,
                latency_ms,
            } => write!(
                f,
                "provider timed out: latency {latency_ms}ms over {timeout_ms}ms timeout"
            ),
            Self::TimeoutTooLarge { max, actual } => {
                write!(f, "timeout {actual}ms exceeds ceiling {max}ms")
            }
            Self::InvalidProviderId { id } => write!(f, "invalid provider id: {id}"),
            Self::Transport { provider, reason } => {
                write!(f, "provider {provider} transport failure: {reason}")
            }
            Self::Auth { provider, reason } => {
                write!(f, "provider {provider} refused authorization: {reason}")
            }
            Self::RateLimited {
                provider,
                retry_after_ms,
            } => match retry_after_ms {
                Some(ms) => write!(f, "provider {provider} rate limited: retry after {ms}ms"),
                None => write!(f, "provider {provider} rate limited"),
            },
            Self::CapabilityMismatch {
                provider,
                model,
                missing,
            } => write!(
                f,
                "provider {provider} model {model} lacks {} capabilities",
                missing.len()
            ),
            Self::ModelUnavailable { provider, model } => {
                write!(f, "provider {provider} model {model} unavailable")
            }
            Self::Unknown { provider, reason } => {
                write!(f, "provider {provider} effect unknown: {reason}")
            }
            Self::InvalidSampling => write!(f, "invalid sampling parameters"),
            Self::UnsupportedSampling { field } => {
                write!(f, "sampling option unsupported by backend: {field}")
            }
        }
    }
}

impl std::error::Error for ProviderError {}

/// Validate a provider id (`MP-2`, mirrors `provider.rs`): non-empty, at
/// most [`MAX_PROVIDER_ID_LEN`] bytes, `^[a-z][a-z0-9_-]*$`.
///
/// # Errors
///
/// Returns [`ProviderError::InvalidProviderId`] when the shape is violated.
pub fn validate_provider_id(id: &str) -> Result<(), ProviderError> {
    let valid = !id.is_empty()
        && id.len() <= MAX_PROVIDER_ID_LEN
        && id.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if valid {
        Ok(())
    } else {
        Err(ProviderError::InvalidProviderId { id: id.to_owned() })
    }
}

/// Selection policy errors (mirrors `selection.rs`). Every variant fails
/// closed: the registry (or the pending chain) is left unchanged and no
/// candidate is executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// Provider id violates the `MP-2` shape.
    InvalidProviderId {
        /// Rejected id.
        id: String,
    },
    /// Model or alias name violates the registry shape.
    InvalidName {
        /// Rejected name.
        name: String,
    },
    /// Registration advertises no capabilities. A capability-less entry can
    /// never match (selection always requires at least one capability), so
    /// it is refused at registration instead of lingering unselectable.
    NoCapabilities {
        /// Rejected model name.
        name: String,
    },
    /// Exact re-registration (same provider, name, capabilities, and
    /// metadata). Refused with no replacement.
    DuplicateModel {
        /// Owning provider id.
        provider: String,
        /// Rejected model name.
        model: String,
    },
    /// Conflicting re-registration: same provider and name with different
    /// capabilities or metadata. Refused with no shadowing: the kept entry
    /// is untouched.
    ConflictingModel {
        /// Owning provider id.
        provider: String,
        /// Rejected model name.
        model: String,
    },
    /// Registry exceeds [`MAX_REGISTERED_MODELS`].
    RegistryFull {
        /// Bound.
        limit: usize,
    },
    /// Alias name is already registered. No replacement, no merge.
    DuplicateAlias {
        /// Rejected alias.
        alias: String,
    },
    /// Alias declares no candidates.
    EmptyAlias {
        /// Rejected alias.
        alias: String,
    },
    /// Alias declares more than [`MAX_ALIAS_CANDIDATES`] candidates.
    TooManyCandidates {
        /// Rejected alias.
        alias: String,
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// Registry exceeds [`MAX_ALIASES`] aliases.
    AliasesFull {
        /// Bound.
        limit: usize,
    },
    /// Alias is not registered.
    UnknownAlias {
        /// Requested alias.
        alias: String,
    },
    /// Selection requested zero capabilities. Matching by name or alias
    /// alone is refused: every selection states at least one required
    /// capability.
    EmptyRequirement,
    /// No registry entry satisfies the request after every gate.
    NoCandidate {
        /// Which requirement set matched nothing.
        detail: String,
    },
}

impl Display for SelectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidProviderId { id } => write!(f, "invalid provider id: {id}"),
            Self::InvalidName { name } => write!(f, "invalid model name: {name}"),
            Self::NoCapabilities { name } => {
                write!(f, "model {name} advertises no capabilities")
            }
            Self::DuplicateModel { provider, model } => {
                write!(f, "duplicate model: provider {provider} model {model}")
            }
            Self::ConflictingModel { provider, model } => {
                write!(
                    f,
                    "conflicting model registration: provider {provider} model {model}"
                )
            }
            Self::RegistryFull { limit } => {
                write!(f, "provider registry full at {limit} models")
            }
            Self::DuplicateAlias { alias } => write!(f, "duplicate alias: {alias}"),
            Self::EmptyAlias { alias } => write!(f, "alias {alias} declares no candidates"),
            Self::TooManyCandidates {
                alias,
                limit,
                actual,
            } => write!(
                f,
                "alias {alias} declares {actual} candidates over {limit} limit"
            ),
            Self::AliasesFull { limit } => {
                write!(f, "alias table full at {limit} aliases")
            }
            Self::UnknownAlias { alias } => write!(f, "unknown alias: {alias}"),
            Self::EmptyRequirement => write!(
                f,
                "selection requires at least one capability (never by name alone)"
            ),
            Self::NoCandidate { detail } => write!(f, "no candidate satisfies: {detail}"),
        }
    }
}

impl std::error::Error for SelectionError {}

/// Validate a model (or alias) name (mirrors `selection.rs`): non-empty, at
/// most [`MAX_MODEL_NAME_LEN`] bytes, starting with lowercase alphanumeric,
/// continuing with lowercase alphanumeric or `_`, `-`, `.`, `:`, `/`
/// (covers `fake-chat`, `llama3.1:8b`, `org/model`). Names are routing data,
/// and capability gates still apply after resolution.
///
/// # Errors
///
/// Returns [`SelectionError::InvalidName`] when the shape is violated.
pub fn validate_model_name(name: &str) -> Result<(), SelectionError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_MODEL_NAME_LEN
        && name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && name.bytes().all(|b| {
            b.is_ascii_lowercase()
                || b.is_ascii_digit()
                || b == b'_'
                || b == b'-'
                || b == b'.'
                || b == b':'
                || b == b'/'
        });
    if valid {
        Ok(())
    } else {
        Err(SelectionError::InvalidName {
            name: name.to_owned(),
        })
    }
}

/// One model registration: provider association, capability set, and routing
/// metadata (mirrors `selection.rs`). Context-window and cost fields are data
/// for routing/budget decisions, not enforcement by themselves; a
/// `context_window_tokens` of `0` means unknown and satisfies no minimum,
/// and a cost weight of `0` means uncalibrated: it counts as the baseline
/// weight `1` in [`estimate_cost`] and turn accounting (never a currency).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRegistration {
    /// Owning provider id (`MP-2` shape).
    pub provider_id: String,
    /// Registry-known model name.
    pub name: String,
    /// Capability flags (at least one).
    pub capabilities: Vec<ModelCapability>,
    /// Context window in tokens (`0` = unknown).
    pub context_window_tokens: u32,
    /// Relative input cost weight (`0` = uncalibrated, counts as baseline `1`).
    pub input_cost_weight: u32,
    /// Relative output cost weight (`0` = uncalibrated, counts as baseline `1`).
    pub output_cost_weight: u32,
}

impl ModelRegistration {
    /// Construct and validate a registration.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] for a malformed provider id or name, or
    /// for an empty capability set.
    pub fn new(
        provider_id: impl Into<String>,
        name: impl Into<String>,
        capabilities: Vec<ModelCapability>,
        context_window_tokens: u32,
        input_cost_weight: u32,
        output_cost_weight: u32,
    ) -> Result<Self, SelectionError> {
        let provider_id = provider_id.into();
        let name = name.into();
        validate_provider_id(&provider_id).map_err(|_| SelectionError::InvalidProviderId {
            id: provider_id.clone(),
        })?;
        validate_model_name(&name)?;
        if capabilities.is_empty() {
            return Err(SelectionError::NoCapabilities { name });
        }
        Ok(Self {
            provider_id,
            name,
            capabilities,
            context_window_tokens,
            input_cost_weight,
            output_cost_weight,
        })
    }

    /// Bridge a registry descriptor into a registration by attaching routing
    /// metadata. The descriptor is read only; no I/O happens and no provider
    /// is touched.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] for a malformed provider id or descriptor
    /// name, or for an empty capability set.
    pub fn snapshot_from(
        provider_id: &str,
        descriptor: &ModelDescriptor,
        context_window_tokens: u32,
        input_cost_weight: u32,
        output_cost_weight: u32,
    ) -> Result<Self, SelectionError> {
        Self::new(
            provider_id,
            descriptor.name.clone(),
            descriptor.capabilities.clone(),
            context_window_tokens,
            input_cost_weight,
            output_cost_weight,
        )
    }
}

/// One `(provider, model)` reference used as an alias candidate (mirrors
/// `selection.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    /// Owning provider id.
    pub provider_id: String,
    /// Registry-known model name.
    pub model: String,
}

impl ModelRef {
    /// Construct and validate a reference.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] for a malformed provider id or name.
    pub fn new(
        provider_id: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, SelectionError> {
        let provider_id = provider_id.into();
        let model = model.into();
        validate_provider_id(&provider_id).map_err(|_| SelectionError::InvalidProviderId {
            id: provider_id.clone(),
        })?;
        validate_model_name(&model)?;
        Ok(Self { provider_id, model })
    }
}

/// One registry-kept model entry (validated registration; mirrors
/// `selection.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredModel {
    /// Owning provider id.
    pub provider_id: String,
    /// Registry-known model name.
    pub name: String,
    /// Capability flags.
    pub capabilities: Vec<ModelCapability>,
    /// Context window in tokens (`0` = unknown).
    pub context_window_tokens: u32,
    /// Relative input cost weight (`0` = uncalibrated, counts as baseline `1`).
    pub input_cost_weight: u32,
    /// Relative output cost weight (`0` = uncalibrated, counts as baseline `1`).
    pub output_cost_weight: u32,
}

/// One semantic alias: a name resolving to ordered candidates (mirrors
/// `selection.rs`). The table is data, not authority: candidates still pass
/// every selection gate in the declared order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAlias {
    /// Alias name.
    pub alias: String,
    /// Ordered candidates (highest preference first).
    pub candidates: Vec<ModelRef>,
}

/// Deterministic provider/model registry plus alias table (mirrors
/// `selection.rs`).
///
/// Entries keep registration order internally; selection output order is
/// always derived (alias order, then provider/model lexicographic), never
/// insertion or hash order.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    models: Vec<RegisteredModel>,
    aliases: Vec<ModelAlias>,
}

impl ProviderRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one validated model entry.
    ///
    /// Fail-closed: an exact re-registration is refused with
    /// [`SelectionError::DuplicateModel`]; the same provider and name with
    /// different capabilities or metadata is refused with
    /// [`SelectionError::ConflictingModel`]; malformed entries are refused
    /// with their typed error even when built without
    /// [`ModelRegistration::new`]; a full registry refuses with
    /// [`SelectionError::RegistryFull`]. Every refusal leaves the registry
    /// unchanged. Duplicate/conflict checks run before the capacity check,
    /// so a duplicate reports as a duplicate even when the registry is full.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] as described above.
    pub fn register(&mut self, registration: ModelRegistration) -> Result<(), SelectionError> {
        if let Some(kept) = self.models.iter().find(|kept| {
            kept.provider_id == registration.provider_id && kept.name == registration.name
        }) {
            let same = kept.capabilities == registration.capabilities
                && kept.context_window_tokens == registration.context_window_tokens
                && kept.input_cost_weight == registration.input_cost_weight
                && kept.output_cost_weight == registration.output_cost_weight;
            if same {
                return Err(SelectionError::DuplicateModel {
                    provider: registration.provider_id,
                    model: registration.name,
                });
            }
            return Err(SelectionError::ConflictingModel {
                provider: registration.provider_id,
                model: registration.name,
            });
        }
        validate_provider_id(&registration.provider_id).map_err(|_| {
            SelectionError::InvalidProviderId {
                id: registration.provider_id.clone(),
            }
        })?;
        validate_model_name(&registration.name)?;
        if registration.capabilities.is_empty() {
            return Err(SelectionError::NoCapabilities {
                name: registration.name,
            });
        }
        if self.models.len() >= MAX_REGISTERED_MODELS {
            return Err(SelectionError::RegistryFull {
                limit: MAX_REGISTERED_MODELS,
            });
        }
        self.models.push(RegisteredModel {
            provider_id: registration.provider_id,
            name: registration.name,
            capabilities: registration.capabilities,
            context_window_tokens: registration.context_window_tokens,
            input_cost_weight: registration.input_cost_weight,
            output_cost_weight: registration.output_cost_weight,
        });
        Ok(())
    }

    /// Register one semantic alias.
    ///
    /// Fail-closed: a duplicate alias name is refused with
    /// [`SelectionError::DuplicateAlias`] (no replacement, no merge); an
    /// empty candidate list is refused with [`SelectionError::EmptyAlias`];
    /// over-long lists with [`SelectionError::TooManyCandidates`]; a full
    /// table with [`SelectionError::AliasesFull`]. Candidates must be
    /// well-formed but need not be registered yet: unregistered candidates
    /// are skipped in declared order at selection time, so a stale entry
    /// degrades to the next candidate; a fully missed alias fails closed
    /// with [`SelectionError::NoCandidate`] instead of falling through to
    /// non-alias entries.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] as described above.
    pub fn register_alias(
        &mut self,
        alias: impl Into<String>,
        candidates: Vec<ModelRef>,
    ) -> Result<(), SelectionError> {
        let alias = alias.into();
        if self.aliases.iter().any(|kept| kept.alias == alias) {
            return Err(SelectionError::DuplicateAlias { alias });
        }
        validate_model_name(&alias)?;
        if candidates.is_empty() {
            return Err(SelectionError::EmptyAlias { alias });
        }
        if candidates.len() > MAX_ALIAS_CANDIDATES {
            return Err(SelectionError::TooManyCandidates {
                alias,
                limit: MAX_ALIAS_CANDIDATES,
                actual: candidates.len(),
            });
        }
        for candidate in &candidates {
            validate_provider_id(&candidate.provider_id).map_err(|_| {
                SelectionError::InvalidProviderId {
                    id: candidate.provider_id.clone(),
                }
            })?;
            validate_model_name(&candidate.model)?;
        }
        if self.aliases.len() >= MAX_ALIASES {
            return Err(SelectionError::AliasesFull { limit: MAX_ALIASES });
        }
        self.aliases.push(ModelAlias { alias, candidates });
        Ok(())
    }

    /// Registered model count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry holds no model.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Registered alias count.
    #[must_use]
    pub fn alias_len(&self) -> usize {
        self.aliases.len()
    }

    /// Look up an entry by provider and model name.
    #[must_use]
    pub fn lookup(&self, provider_id: &str, name: &str) -> Option<&RegisteredModel> {
        self.models
            .iter()
            .find(|entry| entry.provider_id == provider_id && entry.name == name)
    }

    /// Registered models in `(provider, name)` lexicographic order.
    #[must_use]
    pub fn entries(&self) -> Vec<&RegisteredModel> {
        let mut entries: Vec<&RegisteredModel> = self.models.iter().collect();
        entries.sort_by(|a, b| (&a.provider_id, &a.name).cmp(&(&b.provider_id, &b.name)));
        entries
    }

    /// Resolve an alias to its declared candidates in order.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError::UnknownAlias`] for an unregistered alias.
    pub fn resolve_alias(&self, alias: &str) -> Result<Vec<&ModelRef>, SelectionError> {
        self.aliases
            .iter()
            .find(|entry| entry.alias == alias)
            .map(|entry| entry.candidates.iter().collect())
            .ok_or_else(|| SelectionError::UnknownAlias {
                alias: alias.to_owned(),
            })
    }

    /// Select the ordered fallback chain for `request`.
    ///
    /// Gate order: alias resolution (unknown alias fails), capability-subset
    /// match, context-window minimum, cost ceiling, then ordering
    /// (alias-declared order when an alias is requested, else provider/model
    /// lexicographic). The primary is element `0`; every element
    /// independently satisfies the request, so advancing on a typed
    /// transient error never weakens the requirement set.
    ///
    /// An alias is exclusive: when [`SelectRequest::alias`] is set, only the
    /// alias candidates are eligible (each still passing every gate;
    /// unregistered candidates are skipped in declared order) and non-alias
    /// entries are never appended, so a full alias miss fails closed with
    /// [`SelectionError::NoCandidate`] instead of falling through to an
    /// arbitrary capability match.
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError::EmptyRequirement`] when no capability is
    /// required, [`SelectionError::UnknownAlias`] for an unregistered alias,
    /// or [`SelectionError::NoCandidate`] when nothing satisfies the request
    /// (including a fully missed alias).
    pub fn select(&self, request: &SelectRequest) -> Result<Vec<SelectedModel>, SelectionError> {
        if request.required.is_empty() {
            return Err(SelectionError::EmptyRequirement);
        }
        let mut chain: Vec<SelectedModel> = Vec::new();
        if let Some(alias) = &request.alias {
            let candidates = self.resolve_alias(alias)?;
            for candidate in candidates {
                match self.lookup(&candidate.provider_id, &candidate.model) {
                    Some(entry) if request.matches(entry) => {
                        chain.push(SelectedModel::from_entry(entry));
                    }
                    Some(_) | None => {}
                }
            }
        } else {
            for entry in self.entries() {
                if request.matches(entry) {
                    chain.push(SelectedModel::from_entry(entry));
                }
            }
        }
        if chain.is_empty() {
            return Err(SelectionError::NoCandidate {
                detail: request.describe(),
            });
        }
        Ok(chain)
    }
}

/// One selection request: required capabilities plus optional narrowing
/// (mirrors `selection.rs`). Every field is a filter; narrowing never widens
/// the requirement set. No closed-vocabulary enforcement is pinned here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectRequest {
    /// Required capabilities: every candidate must advertise all of them.
    /// Must be non-empty (selection never matches by name or alias alone).
    pub required: Vec<ModelCapability>,
    /// Optional semantic alias. When set it is exclusive: only the declared
    /// candidates are eligible (each still passing every other gate, in
    /// declared order) and a full alias miss is
    /// [`SelectionError::NoCandidate`]; non-alias entries are never appended.
    pub alias: Option<String>,
    /// Optional minimum context window in tokens. Entries with an unknown
    /// (`0`) window never satisfy a minimum.
    pub min_context_window_tokens: Option<u32>,
    /// Optional cost ceiling: both effective input and output weights of a
    /// candidate must fit. An uncalibrated weight (`0`) counts as baseline
    /// `1`, so `Some(0)` admits nothing. Routing data only, not enforcement.
    pub max_cost_weight: Option<u32>,
}

impl SelectRequest {
    /// Build a capability-only request.
    #[must_use]
    pub fn capabilities(required: Vec<ModelCapability>) -> Self {
        Self {
            required,
            alias: None,
            min_context_window_tokens: None,
            max_cost_weight: None,
        }
    }

    /// Whether `entry` passes every gate of this request.
    #[must_use]
    pub fn matches(&self, entry: &RegisteredModel) -> bool {
        if !self
            .required
            .iter()
            .all(|cap| entry.capabilities.contains(cap))
        {
            return false;
        }
        if let Some(min) = self.min_context_window_tokens {
            if entry.context_window_tokens < min {
                return false;
            }
        }
        if let Some(max) = self.max_cost_weight {
            if effective_cost_weight(entry.input_cost_weight) > max
                || effective_cost_weight(entry.output_cost_weight) > max
            {
                return false;
            }
        }
        true
    }

    /// Human-readable requirement summary for [`SelectionError::NoCandidate`].
    #[must_use]
    pub fn describe(&self) -> String {
        let mut detail = format!("{} capabilities", self.required.len());
        if let Some(alias) = &self.alias {
            detail.push_str(&format!(" alias={alias}"));
        }
        if let Some(min) = self.min_context_window_tokens {
            detail.push_str(&format!(" min_window={min}"));
        }
        if let Some(max) = self.max_cost_weight {
            detail.push_str(&format!(" max_cost={max}"));
        }
        detail
    }
}

/// One selected candidate: a detached snapshot of a registry entry (mirrors
/// `selection.rs`, following the one-way-snapshot pattern of
/// [`TerminalModelMetadata`]). Later registry changes do not leak into
/// previously selected chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedModel {
    /// Owning provider id.
    pub provider_id: String,
    /// Registry-known model name.
    pub name: String,
    /// Capability snapshot.
    pub capabilities: Vec<ModelCapability>,
    /// Context window in tokens (`0` = unknown).
    pub context_window_tokens: u32,
    /// Relative input cost weight (`0` = uncalibrated, counts as baseline `1`).
    pub input_cost_weight: u32,
    /// Relative output cost weight (`0` = uncalibrated, counts as baseline `1`).
    pub output_cost_weight: u32,
}

impl SelectedModel {
    /// Snapshot a registry entry. The snapshot is detached: later registry
    /// changes do not leak into previously selected chains.
    #[must_use]
    pub fn from_entry(entry: &RegisteredModel) -> Self {
        Self {
            provider_id: entry.provider_id.clone(),
            name: entry.name.clone(),
            capabilities: entry.capabilities.clone(),
            context_window_tokens: entry.context_window_tokens,
            input_cost_weight: entry.input_cost_weight,
            output_cost_weight: entry.output_cost_weight,
        }
    }

    /// Estimated routing cost for an observed usage pair, in relative
    /// routing units (never currency). Delegates to [`estimate_cost`].
    #[must_use]
    pub fn estimate_cost(&self, input_tokens: u64, output_tokens: u64) -> u64 {
        estimate_cost(
            input_tokens,
            output_tokens,
            self.input_cost_weight,
            self.output_cost_weight,
        )
    }
}

/// Estimate routing cost in relative units (never currency) from observed
/// token counts and per-model cost weights (mirrors `selection.rs`):
///
/// ```text
/// cost = input_tokens * effective(input_weight)
///      + output_tokens * effective(output_weight)
/// effective(weight) = if weight == 0 { 1 } else { weight }
/// ```
///
/// Both weights are routing data held on the model registration/selection
/// ([`ModelRegistration`], [`RegisteredModel`], [`SelectedModel`]); a weight
/// of `0` means uncalibrated and counts as the baseline weight `1`, so an
/// unset weight is never free. The function itself performs no I/O, reads no
/// clock, and authorizes nothing. Saturates on overflow rather than
/// wrapping.
#[must_use]
pub fn estimate_cost(
    input_tokens: u64,
    output_tokens: u64,
    input_weight: u32,
    output_weight: u32,
) -> u64 {
    (input_tokens.saturating_mul(u64::from(effective_cost_weight(input_weight)))).saturating_add(
        output_tokens.saturating_mul(u64::from(effective_cost_weight(output_weight))),
    )
}

/// Effective cost weight shared by selection estimation/filtering and turn
/// accounting (mirrors `selection.rs`): a configured `0` means uncalibrated
/// and counts as the baseline weight `1` so an uncalibrated model is never
/// free and accounting cannot under-count.
fn effective_cost_weight(weight: u32) -> u32 {
    if weight == 0 { 1 } else { weight }
}

/// Fallback directive for one provider failure (mirrors `selection.rs`):
/// advance to the next chain element or stop. Pure and total over
/// [`ProviderError`]: adding a variant breaks compilation here, which is the
/// fail-closed response to a new failure mode (it must be classified before
/// any fallback uses it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackDirective {
    /// The error is transient or provider-local: try the next chain element
    /// with the unchanged request.
    Advance,
    /// Do not try another candidate. Reconcile first: fix the caller or
    /// configuration for typed caller/config errors, reconcile by status
    /// inspection or user direction for [`ProviderError::Unknown`].
    Stop,
}

/// Classify one provider failure into a fallback directive (mirrors
/// `selection.rs`).
///
/// - Advance (transient or provider-local, retry elsewhere is safe):
///   [`Transport`](ProviderError::Transport),
///   [`RateLimited`](ProviderError::RateLimited),
///   [`ModelUnavailable`](ProviderError::ModelUnavailable),
///   [`Timeout`](ProviderError::Timeout).
/// - Stop (caller, configuration, or authorization bugs that another
///   candidate cannot fix):
///   [`UnknownModel`](ProviderError::UnknownModel) (registry drift),
///   [`BudgetExceeded`](ProviderError::BudgetExceeded),
///   [`TimeoutTooLarge`](ProviderError::TimeoutTooLarge),
///   [`InvalidProviderId`](ProviderError::InvalidProviderId),
///   [`InvalidSampling`](ProviderError::InvalidSampling) (caller-declared
///   sampling contract),
///   [`UnsupportedSampling`](ProviderError::UnsupportedSampling) (reconcile
///   the declaration or backend choice rather than spraying),
///   [`Auth`](ProviderError::Auth) (reconcile credentials; never spray),
///   [`CapabilityMismatch`](ProviderError::CapabilityMismatch) (routing bug).
/// - Stop with reconcile (effect uncertain; blind fallback may double-apply):
///   [`Unknown`](ProviderError::Unknown).
#[must_use]
pub fn fallback_directive(error: &ProviderError) -> FallbackDirective {
    match error {
        ProviderError::Transport { .. }
        | ProviderError::RateLimited { .. }
        | ProviderError::ModelUnavailable { .. }
        | ProviderError::Timeout { .. } => FallbackDirective::Advance,
        ProviderError::UnknownModel { .. }
        | ProviderError::BudgetExceeded { .. }
        | ProviderError::TimeoutTooLarge { .. }
        | ProviderError::InvalidProviderId { .. }
        | ProviderError::InvalidSampling
        | ProviderError::UnsupportedSampling { .. }
        | ProviderError::Auth { .. }
        | ProviderError::CapabilityMismatch { .. }
        | ProviderError::Unknown { .. } => FallbackDirective::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ModelCapability as Cap;

    fn registration(
        provider: &str,
        name: &str,
        capabilities: Vec<ModelCapability>,
        window: u32,
        input: u32,
        output: u32,
    ) -> ModelRegistration {
        ModelRegistration::new(provider, name, capabilities, window, input, output)
            .expect("valid registration")
    }

    fn text_model(provider: &str, name: &str) -> ModelRegistration {
        registration(provider, name, vec![Cap::Text], 4_096, 1, 2)
    }

    fn registry_two() -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        registry
            .register(text_model("bitty-b", "chat"))
            .expect("capacity");
        registry
            .register(registration(
                "bitty-a",
                "vision",
                vec![Cap::Text, Cap::ImageInput],
                32_768,
                3,
                4,
            ))
            .expect("capacity");
        registry
    }

    #[test]
    fn mirror_bounds_match_survey_values() {
        assert_eq!(MAX_MODEL_NAME_LEN, 128);
        assert_eq!(MAX_REGISTERED_MODELS, 64);
        assert_eq!(MAX_ALIASES, 32);
        assert_eq!(MAX_ALIAS_CANDIDATES, 8);
        assert_eq!(MAX_PROVIDER_ID_LEN, 64);
    }

    #[test]
    fn model_name_shape() {
        assert!(validate_model_name("chat").is_ok());
        assert!(validate_model_name("fake-chat").is_ok());
        assert!(validate_model_name("llama3.1:8b").is_ok());
        assert!(validate_model_name("org/model").is_ok());
        assert!(validate_model_name("a1").is_ok());
        assert!(validate_model_name("a".repeat(128).as_str()).is_ok());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("Chat").is_err());
        assert!(validate_model_name("-chat").is_err());
        assert!(validate_model_name(".chat").is_err());
        assert!(validate_model_name("has space").is_err());
        assert!(validate_model_name("a".repeat(129).as_str()).is_err());
    }

    #[test]
    fn provider_id_shape() {
        assert!(validate_provider_id("bitty-fake").is_ok());
        assert!(validate_provider_id("a1_-b").is_ok());
        assert!(validate_provider_id("").is_err());
        assert!(validate_provider_id("1abc").is_err());
        assert!(validate_provider_id("ABC").is_err());
        assert!(validate_provider_id("local.deterministic").is_err());
        assert!(validate_provider_id("a".repeat(65).as_str()).is_err());
    }

    #[test]
    fn register_and_lookup() {
        let mut registry = ProviderRegistry::new();
        assert!(registry.is_empty());
        registry
            .register(text_model("bitty-fake", "fake-chat"))
            .expect("capacity");
        assert_eq!(registry.len(), 1);
        let kept = registry.lookup("bitty-fake", "fake-chat").expect("kept");
        assert_eq!(kept.capabilities, vec![Cap::Text]);
        assert!(registry.lookup("bitty-fake", "other").is_none());
        assert!(registry.lookup("other", "fake-chat").is_none());
    }

    #[test]
    fn same_name_on_different_providers_coexists() {
        // Identity is (provider, name): two providers may serve the same
        // model name without conflicting.
        let mut registry = ProviderRegistry::new();
        registry
            .register(text_model("bitty-a", "chat"))
            .expect("capacity");
        registry
            .register(text_model("bitty-b", "chat"))
            .expect("capacity");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn duplicate_registration_is_fail_closed() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(registration(
                "bitty-fake",
                "chat",
                vec![Cap::Text],
                4_096,
                1,
                2,
            ))
            .expect("capacity");
        let retry = registration("bitty-fake", "chat", vec![Cap::Text], 4_096, 1, 2);
        let error = registry.register(retry).expect_err("duplicate must fail");
        assert_eq!(
            error,
            SelectionError::DuplicateModel {
                provider: "bitty-fake".to_owned(),
                model: "chat".to_owned(),
            }
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn conflicting_reregistration_keeps_original() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(registration(
                "bitty-fake",
                "chat",
                vec![Cap::Text],
                4_096,
                1,
                2,
            ))
            .expect("capacity");
        // Same key, wider capabilities: conflict, not an upgrade.
        let shadow = registration(
            "bitty-fake",
            "chat",
            vec![Cap::Text, Cap::ToolUse],
            4_096,
            1,
            2,
        );
        let error = registry.register(shadow).expect_err("conflict must fail");
        assert_eq!(
            error,
            SelectionError::ConflictingModel {
                provider: "bitty-fake".to_owned(),
                model: "chat".to_owned(),
            }
        );
        // Same key, same caps, different metadata: still a conflict.
        let repriced = registration("bitty-fake", "chat", vec![Cap::Text], 8_192, 1, 2);
        assert!(matches!(
            registry.register(repriced),
            Err(SelectionError::ConflictingModel { .. })
        ));
        let kept = registry
            .lookup("bitty-fake", "chat")
            .expect("original kept");
        assert_eq!(kept.capabilities, vec![Cap::Text]);
        assert_eq!(kept.context_window_tokens, 4_096);
    }

    #[test]
    fn duplicate_reports_even_when_registry_is_full() {
        let mut registry = ProviderRegistry::new();
        for index in 0..MAX_REGISTERED_MODELS {
            registry
                .register(
                    ModelRegistration::new(
                        "bitty-fake",
                        format!("model-{index}"),
                        vec![Cap::Text],
                        4_096,
                        1,
                        2,
                    )
                    .expect("valid registration"),
                )
                .expect("capacity");
        }
        let retry = ModelRegistration::new("bitty-fake", "model-0", vec![Cap::Text], 4_096, 1, 2)
            .expect("valid registration");
        assert_eq!(
            registry.register(retry).expect_err("duplicate must fail"),
            SelectionError::DuplicateModel {
                provider: "bitty-fake".to_owned(),
                model: "model-0".to_owned(),
            }
        );
        let overflow =
            ModelRegistration::new("bitty-fake", "one-more", vec![Cap::Text], 4_096, 1, 2)
                .expect("valid registration");
        assert_eq!(
            registry.register(overflow).expect_err("overflow must fail"),
            SelectionError::RegistryFull {
                limit: MAX_REGISTERED_MODELS,
            }
        );
        assert_eq!(registry.len(), MAX_REGISTERED_MODELS);
    }

    #[test]
    fn register_revalidates_entries_built_without_new() {
        let mut registry = ProviderRegistry::new();
        let bad_provider = ModelRegistration {
            provider_id: "BITTY".to_owned(),
            name: "chat".to_owned(),
            capabilities: vec![Cap::Text],
            context_window_tokens: 0,
            input_cost_weight: 0,
            output_cost_weight: 0,
        };
        assert!(matches!(
            registry.register(bad_provider),
            Err(SelectionError::InvalidProviderId { .. })
        ));
        let no_caps = ModelRegistration {
            provider_id: "bitty-fake".to_owned(),
            name: "chat".to_owned(),
            capabilities: Vec::new(),
            context_window_tokens: 0,
            input_cost_weight: 0,
            output_cost_weight: 0,
        };
        assert_eq!(
            registry.register(no_caps).expect_err("no caps must fail"),
            SelectionError::NoCapabilities {
                name: "chat".to_owned(),
            }
        );
        assert!(registry.is_empty());
    }

    #[test]
    fn selection_error_display_strings() {
        assert_eq!(
            SelectionError::DuplicateModel {
                provider: "p".to_owned(),
                model: "m".to_owned(),
            }
            .to_string(),
            "duplicate model: provider p model m"
        );
        assert_eq!(
            SelectionError::RegistryFull { limit: 64 }.to_string(),
            "provider registry full at 64 models"
        );
        assert_eq!(
            SelectionError::EmptyRequirement.to_string(),
            "selection requires at least one capability (never by name alone)"
        );
        assert_eq!(
            SelectionError::UnknownAlias {
                alias: "chat".to_owned(),
            }
            .to_string(),
            "unknown alias: chat"
        );
        assert_eq!(
            SelectionError::NoCandidate {
                detail: "1 capabilities".to_owned(),
            }
            .to_string(),
            "no candidate satisfies: 1 capabilities"
        );
    }

    #[test]
    fn capability_matching_is_subset_never_name_only() {
        let registry = registry_two();
        // Vision model matches Text+ImageInput; text-only model does not.
        let chain = registry
            .select(&SelectRequest::capabilities(vec![
                Cap::Text,
                Cap::ImageInput,
            ]))
            .expect("vision matches");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "vision");
        // Bare Text matches both (superset satisfies subset).
        let chain = registry
            .select(&SelectRequest::capabilities(vec![Cap::Text]))
            .expect("both match");
        assert_eq!(chain.len(), 2);
        // Name alone never selects: empty requirement set is refused even
        // though entries exist.
        assert_eq!(
            registry
                .select(&SelectRequest::capabilities(Vec::new()))
                .expect_err("empty requirement must fail"),
            SelectionError::EmptyRequirement
        );
    }

    #[test]
    fn deterministic_tiebreak_is_lexicographic() {
        // Registered b-before-a; the chain still orders a-before-b.
        let registry = registry_two();
        let chain = registry
            .select(&SelectRequest::capabilities(vec![Cap::Text]))
            .expect("matches");
        let order: Vec<(&str, &str)> = chain
            .iter()
            .map(|item| (item.provider_id.as_str(), item.name.as_str()))
            .collect();
        assert_eq!(order, vec![("bitty-a", "vision"), ("bitty-b", "chat")]);
    }

    #[test]
    fn alias_chain_is_alias_candidates_in_declared_order() {
        let mut registry = registry_two();
        registry
            .register(text_model("bitty-c", "fast"))
            .expect("capacity");
        registry
            .register_alias(
                "chat",
                vec![
                    ModelRef::new("bitty-c", "fast").expect("valid ref"),
                    ModelRef::new("bitty-b", "chat").expect("valid ref"),
                ],
            )
            .expect("capacity");
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: Some("chat".to_owned()),
            min_context_window_tokens: None,
            max_cost_weight: None,
        };
        let chain = registry.select(&request).expect("alias resolves");
        let order: Vec<(&str, &str)> = chain
            .iter()
            .map(|item| (item.provider_id.as_str(), item.name.as_str()))
            .collect();
        // The alias is exclusive: only declared candidates, in declared
        // order. The matching non-alias entry ("bitty-a", "vision") is never
        // appended.
        assert_eq!(order, vec![("bitty-c", "fast"), ("bitty-b", "chat")]);
    }

    #[test]
    fn alias_candidates_still_require_capabilities() {
        // The alias resolves, but its only candidate lacks ImageInput. The
        // non-alias "vision" entry would satisfy the capability set, yet an
        // alias is a hard filter: the request fails closed instead of
        // falling through to a different model.
        let mut registry = registry_two();
        registry
            .register_alias(
                "chat",
                vec![ModelRef::new("bitty-b", "chat").expect("valid ref")],
            )
            .expect("capacity");
        let request = SelectRequest {
            required: vec![Cap::Text, Cap::ImageInput],
            alias: Some("chat".to_owned()),
            min_context_window_tokens: None,
            max_cost_weight: None,
        };
        match registry.select(&request) {
            Err(SelectionError::NoCandidate { detail }) => {
                assert!(detail.contains("alias=chat"), "unexpected detail: {detail}");
            }
            other => panic!("alias miss must not fall through: {other:?}"),
        }
    }

    #[test]
    fn unknown_alias_fails_closed() {
        let registry = registry_two();
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: Some("nope".to_owned()),
            min_context_window_tokens: None,
            max_cost_weight: None,
        };
        assert_eq!(
            registry
                .select(&request)
                .expect_err("unknown alias must fail"),
            SelectionError::UnknownAlias {
                alias: "nope".to_owned(),
            }
        );
        assert!(matches!(
            registry.resolve_alias("nope"),
            Err(SelectionError::UnknownAlias { .. })
        ));
    }

    #[test]
    fn alias_skips_stale_candidates_in_order() {
        let mut registry = registry_two();
        registry
            .register_alias(
                "chat",
                vec![
                    ModelRef::new("bitty-gone", "ghost").expect("valid ref"),
                    ModelRef::new("bitty-b", "chat").expect("valid ref"),
                ],
            )
            .expect("capacity");
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: Some("chat".to_owned()),
            min_context_window_tokens: None,
            max_cost_weight: None,
        };
        // The stale candidate is skipped in declared order and the chain
        // stays alias-only, so the matching non-alias "bitty-a"/"vision"
        // entry is not appended.
        let chain = registry.select(&request).expect("stale skipped");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "chat");
        assert_eq!(chain[0].provider_id, "bitty-b");
    }

    #[test]
    fn all_stale_alias_yields_no_candidate() {
        let mut registry = registry_two();
        registry
            .register_alias(
                "chat",
                vec![ModelRef::new("bitty-gone", "ghost").expect("valid ref")],
            )
            .expect("capacity");
        // Every alias candidate is unregistered while the registry does hold
        // entries satisfying the capability set. The alias is exclusive: a
        // full miss is `NoCandidate`, never a silent fall-through to a
        // non-alias model.
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: Some("chat".to_owned()),
            min_context_window_tokens: None,
            max_cost_weight: None,
        };
        match registry.select(&request) {
            Err(SelectionError::NoCandidate { detail }) => {
                assert!(detail.contains("alias=chat"), "unexpected detail: {detail}");
            }
            other => panic!("fully stale alias must fail closed: {other:?}"),
        }
    }

    #[test]
    fn alias_table_is_fail_closed() {
        let mut registry = registry_two();
        registry
            .register_alias(
                "chat",
                vec![ModelRef::new("bitty-b", "chat").expect("valid ref")],
            )
            .expect("capacity");
        assert_eq!(
            registry
                .register_alias(
                    "chat",
                    vec![ModelRef::new("bitty-a", "vision").expect("valid ref")],
                )
                .expect_err("duplicate alias must fail"),
            SelectionError::DuplicateAlias {
                alias: "chat".to_owned(),
            }
        );
        assert!(matches!(
            registry.register_alias("empty", Vec::new()),
            Err(SelectionError::EmptyAlias { .. })
        ));
        let burst: Vec<ModelRef> = (0..MAX_ALIAS_CANDIDATES + 1)
            .map(|index| ModelRef::new("bitty-b", format!("model-{index}")).expect("valid ref"))
            .collect();
        assert!(matches!(
            registry.register_alias("burst", burst),
            Err(SelectionError::TooManyCandidates { .. })
        ));
        // No partial state: only the first alias kept.
        assert_eq!(registry.alias_len(), 1);
        assert_eq!(registry.resolve_alias("chat").expect("kept").len(), 1);
    }

    #[test]
    fn context_window_minimum_filters_unknown() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(registration(
                "bitty-fake",
                "mystery",
                vec![Cap::Text],
                0,
                0,
                0,
            ))
            .expect("capacity");
        registry
            .register(registration(
                "bitty-fake",
                "roomy",
                vec![Cap::Text],
                32_768,
                0,
                0,
            ))
            .expect("capacity");
        // Unknown (`0`) window never satisfies a stated minimum.
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: None,
            min_context_window_tokens: Some(4_096),
            max_cost_weight: None,
        };
        let chain = registry.select(&request).expect("roomy matches");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "roomy");
        // No minimum: both are routing-eligible.
        let chain = registry
            .select(&SelectRequest::capabilities(vec![Cap::Text]))
            .expect("both match");
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn cost_ceiling_filters_both_weights() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(registration(
                "bitty-fake",
                "cheap",
                vec![Cap::Text],
                4_096,
                1,
                1,
            ))
            .expect("capacity");
        registry
            .register(registration(
                "bitty-fake",
                "pricey",
                vec![Cap::Text],
                4_096,
                1,
                9,
            ))
            .expect("capacity");
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: None,
            min_context_window_tokens: None,
            max_cost_weight: Some(2),
        };
        let chain = registry.select(&request).expect("cheap fits");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "cheap");
    }

    #[test]
    fn cost_ceiling_filter_uses_effective_weight() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(registration(
                "bitty-fake",
                "uncalibrated",
                vec![Cap::Text],
                4_096,
                0,
                0,
            ))
            .expect("capacity");
        registry
            .register(registration(
                "bitty-fake",
                "calibrated",
                vec![Cap::Text],
                4_096,
                2,
                2,
            ))
            .expect("capacity");
        // An uncalibrated weight (`0`) counts as baseline `1`, so a ceiling
        // of `1` admits it and rejects the calibrated 2/2 entry.
        let request = SelectRequest {
            required: vec![Cap::Text],
            alias: None,
            min_context_window_tokens: None,
            max_cost_weight: Some(1),
        };
        let chain = registry
            .select(&request)
            .expect("uncalibrated fits baseline");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "uncalibrated");
        // A zero ceiling admits nothing: baseline `1` is the floor, `0` is
        // never free anywhere in routing or accounting.
        let request = SelectRequest {
            max_cost_weight: Some(0),
            ..request
        };
        assert!(matches!(
            registry.select(&request),
            Err(SelectionError::NoCandidate { .. })
        ));
    }

    #[test]
    fn uncalibrated_weight_counts_as_baseline_one() {
        assert_eq!(effective_cost_weight(0), 1);
        assert_eq!(effective_cost_weight(1), 1);
        assert_eq!(effective_cost_weight(7), 7);
        // Relative routing units, never currency: exact on small values.
        assert_eq!(estimate_cost(10, 5, 2, 3), 35);
        assert_eq!(estimate_cost(0, 0, 2, 3), 0);
        // A `0` weight means uncalibrated and counts as the baseline `1`:
        // an unset weight is never free.
        assert_eq!(estimate_cost(10, 5, 0, 0), 15);
        assert_eq!(estimate_cost(10, 5, 0, 3), 25);
        assert_eq!(estimate_cost(10, 5, 0, 0), estimate_cost(10, 5, 1, 1));
        assert_eq!(estimate_cost(10, 5, 0, 3), estimate_cost(10, 5, 1, 3));
    }

    #[test]
    fn estimate_cost_saturates_instead_of_wrapping() {
        assert_eq!(estimate_cost(u64::MAX, 1, 2, u32::MAX), u64::MAX);
        assert_eq!(
            estimate_cost(u64::MAX, u64::MAX, u32::MAX, u32::MAX),
            u64::MAX
        );
    }

    #[test]
    fn selected_model_delegates_to_estimate_cost() {
        let entry = RegisteredModel {
            provider_id: "bitty-fake".to_owned(),
            name: "m".to_owned(),
            capabilities: vec![Cap::Text],
            context_window_tokens: 4_096,
            input_cost_weight: 2,
            output_cost_weight: 4,
        };
        let selected = SelectedModel::from_entry(&entry);
        assert_eq!(selected.estimate_cost(3, 7), 3 * 2 + 7 * 4);
    }

    #[test]
    fn selected_snapshots_are_detached() {
        let mut registry = registry_two();
        let chain = registry
            .select(&SelectRequest::capabilities(vec![Cap::Text]))
            .expect("matches");
        // Later registry state cannot leak into a selected chain: the
        // snapshot keeps its own capability copy.
        registry
            .register(registration(
                "bitty-c",
                "fresh",
                vec![Cap::Text, Cap::Streaming],
                4_096,
                1,
                1,
            ))
            .expect("capacity");
        assert_eq!(chain.len(), 2);
        assert!(!chain[0].capabilities.contains(&Cap::Streaming));
    }

    #[test]
    fn snapshot_from_bridges_descriptor_without_io() {
        let descriptor = ModelDescriptor {
            name: "fake-chat".to_owned(),
            capabilities: vec![Cap::Text, Cap::ToolUse],
        };
        let bridged = ModelRegistration::snapshot_from("bitty-fake", &descriptor, 4_096, 1, 2)
            .expect("valid snapshot");
        assert_eq!(bridged.provider_id, "bitty-fake");
        assert_eq!(bridged.name, "fake-chat");
        assert_eq!(bridged.capabilities, vec![Cap::Text, Cap::ToolUse]);
        let mut registry = ProviderRegistry::new();
        registry.register(bridged).expect("capacity");
        let chain = registry
            .select(&SelectRequest::capabilities(vec![Cap::Text, Cap::ToolUse]))
            .expect("fake-chat matches text+tools");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider_id, "bitty-fake");
        assert_eq!(chain[0].name, "fake-chat");
    }

    #[test]
    fn terminal_metadata_is_detached_one_way() {
        let descriptor = ModelDescriptor {
            name: "fake-chat".to_owned(),
            capabilities: vec![Cap::Text, Cap::ToolUse],
        };
        let mediated = TerminalModelMetadata::from_descriptor(&descriptor);
        assert_eq!(
            mediated,
            TerminalModelMetadata {
                name: "fake-chat".to_owned(),
                capabilities: vec![Cap::Text, Cap::ToolUse],
            }
        );
        // The snapshot is detached: later descriptor changes do not leak
        // through the mediation view.
        let mut changed = descriptor.clone();
        changed.capabilities.push(Cap::Streaming);
        assert_ne!(TerminalModelMetadata::from_descriptor(&changed), mediated);
    }

    #[test]
    fn fallback_advances_only_on_transient_errors() {
        let advance = [
            ProviderError::Transport {
                provider: "p".to_owned(),
                reason: "reset".to_owned(),
            },
            ProviderError::RateLimited {
                provider: "p".to_owned(),
                retry_after_ms: Some(1_000),
            },
            ProviderError::RateLimited {
                provider: "p".to_owned(),
                retry_after_ms: None,
            },
            ProviderError::ModelUnavailable {
                provider: "p".to_owned(),
                model: "m".to_owned(),
            },
            ProviderError::Timeout {
                timeout_ms: 5_000,
                latency_ms: 6_000,
            },
        ];
        for error in &advance {
            assert_eq!(
                fallback_directive(error),
                FallbackDirective::Advance,
                "must advance: {error}"
            );
        }
    }

    #[test]
    fn fallback_stops_on_caller_config_and_auth_errors() {
        let stop = [
            ProviderError::UnknownModel {
                name: "m".to_owned(),
            },
            ProviderError::BudgetExceeded {
                limit: 1,
                actual: 2,
            },
            ProviderError::TimeoutTooLarge {
                max: 30_000,
                actual: 60_000,
            },
            ProviderError::InvalidProviderId {
                id: "BAD".to_owned(),
            },
            ProviderError::InvalidSampling,
            ProviderError::UnsupportedSampling { field: "top_k" },
            ProviderError::Auth {
                provider: "p".to_owned(),
                reason: "revoked".to_owned(),
            },
            ProviderError::CapabilityMismatch {
                provider: "p".to_owned(),
                model: "m".to_owned(),
                missing: vec![Cap::ImageInput],
            },
        ];
        for error in &stop {
            assert_eq!(
                fallback_directive(error),
                FallbackDirective::Stop,
                "must stop: {error}"
            );
        }
    }

    #[test]
    fn fallback_never_advances_on_unknown_without_reconcile() {
        // Unknown means the effect may have happened: advancing blindly
        // could double-apply. Stop and reconcile first.
        let unknown = ProviderError::Unknown {
            provider: "p".to_owned(),
            reason: "ack lost".to_owned(),
        };
        assert_eq!(fallback_directive(&unknown), FallbackDirective::Stop);
        assert_eq!(unknown.to_string(), "provider p effect unknown: ack lost");
    }
}

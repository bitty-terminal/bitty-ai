//! Host-side pure-type mirror of the provider-registry seam (AI-0271).
//!
//! Phase 2 S1 per the AI-0263 survey: a dependency-free mirror of the
//! registry/selection/fallback shapes from `bitty-ai-runtime`
//! (`selection.rs` plus the `MP-2` support shapes from `provider.rs`) for
//! host-boundary traversal. This module is data plus pure functions only:
//! no `ModelProvider` trait, no `list_models`/`complete` wiring, no `MP-4`
//! caller-scope filter, no closed-vocabulary enforcement, no currency, no
//! IPC transport, no threads, no filesystem or network access, and no
//! dependencies beyond `std`.
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
//!
//! Phase 2 S2 (AI-0278) appends scoped-IPC envelope types for the provider
//! operations below: `list_models` / `complete` request/response shapes plus
//! a bounded [`ProviderIpcError`] envelope form of the [`ProviderError`]
//! taxonomy. They are still data plus pure bound checks only: no socket, no
//! transport, no encoding, no version field, no threads, no clock reads, no
//! filesystem or network access, and no dependencies beyond `std`.
//! Selection-side IPC envelopes (select request/response, matches views,
//! fallback-chain envelopes) belong to AI-0279 and appear here by name only.

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

// ---------------------------------------------------------------------------
// Phase 2 S2: scoped-IPC envelope types (AI-0278).
//
// Request/response data shapes for the two provider operations
// (`list_models` / `complete`) plus a bounded envelope form of the S1
// [`ProviderError`] taxonomy. Types plus pure bound checks only: no socket,
// no transport, no encoding, no version field on any envelope, no behavior
// beyond fail-closed shape validation, no threads, no clock reads
// (`now_ms` is caller-supplied), no filesystem or network access, and no
// dependencies beyond `std`.
//
// What is reused, not re-mirrored: [`ModelRegistration`], [`ModelRef`],
// [`ProviderRegistry`], [`ModelCapability`], [`ProviderError`], the
// [`MAX_REGISTERED_MODELS`] / [`MAX_MODEL_NAME_LEN`] / [`MAX_PROVIDER_ID_LEN`]
// bounds, and the `validate_model_name` / `validate_provider_id` shapes.
// What is new here: the `list_models` / `complete` envelope structs, the
// `MAX_IPC_*` payload bounds below (every value equals the cited runtime
// bound), the [`IpcEnvelopeError`] shape refusal, and the
// [`ProviderIpcError`] bounded error envelope with its total
// [`ProviderIpcError::from_error`] / [`ProviderIpcError::into_error`]
// mapping. What stays out: selection-side IPC envelopes (select
// request/response, matches views, fallback-chain envelopes — AI-0279 owns
// those and they appear here by name only), transport of any kind, and any
// `version` field (envelope versioning is deferred; adding a field later is
// a scoped task, never a silent widening).
//
// Normalization rule for IPC crossing: identifier fields keep their shaped
// values truncated to their shape bound on a character boundary (shape
// validation itself stays at [`ModelRef::new`] and at execution time);
// free-text reason fields are scrubbed (every character outside printable
// ASCII becomes `?`) and truncated to [`MAX_IPC_REASON_BYTES`], mirroring
// the runtime `bridge.rs` `bound_reason` transform. No variant carries
// secrets, credentials, or key material — neither in the S1 taxonomy nor in
// the envelope form — and sampling refusals echo no values at all.
// ---------------------------------------------------------------------------

/// Maximum caller timeout in milliseconds (mirrors `provider.rs`
/// `MAX_REQUEST_TIMEOUT_MS`; enforced by [`CompleteRequest::check`]).
pub const MAX_REQUEST_TIMEOUT_MS: u64 = 30_000;
/// Maximum messages per [`CompleteRequest`] (same value as `context.rs`
/// `MAX_CONTEXT_RECORDS`).
pub const MAX_IPC_MESSAGES: usize = 128;
/// Maximum bytes per envelope message content (same value as `context.rs`
/// `MAX_RECORD_BODY_BYTES`; enforced by [`CompleteRequest::check`]).
pub const MAX_IPC_MESSAGE_BYTES: usize = 16 * 1024;
/// Maximum stable-id references per [`CompleteRequest`] (same value as
/// `context.rs` `MAX_CONTEXT_RECORDS`).
pub const MAX_IPC_CONTEXT_REFS: usize = 128;
/// Maximum bytes per stable-id reference (same value as `context.rs`
/// `MAX_STABLE_ID_LEN`; references resolve server-side per `CP-2`).
pub const MAX_IPC_CONTEXT_REF_BYTES: usize = 64;
/// Maximum tool names per [`CompleteRequest`] (same value as `tool.rs`
/// `MAX_TOOLS_PER_SESSION`; names validate against tool consent at
/// execution, never here).
pub const MAX_IPC_TOOLS: usize = 32;
/// Maximum bytes per tool name (same value as `tool.rs`
/// `MAX_TOOL_NAME_LEN`).
pub const MAX_IPC_TOOL_NAME_BYTES: usize = 64;
/// Maximum assistant-text bytes per [`CompleteResponse`] (same value as
/// `context.rs` `MAX_RECORD_BODY_BYTES`).
pub const MAX_IPC_TEXT_BYTES: usize = 16 * 1024;
/// Maximum tool calls per [`CompleteResponse`] (same value as `tool.rs`
/// `MAX_TOOL_CALLS_PER_TURN`, the hard bus ceiling).
pub const MAX_IPC_TOOL_CALLS: usize = 8;
/// Maximum argument bytes per envelope tool call (same value as `tool.rs`
/// `MAX_TOOL_ARGUMENTS_BYTES`; arguments stay opaque bounded JSON).
pub const MAX_IPC_TOOL_ARGUMENTS_BYTES: usize = 16 * 1024;
/// Maximum reason bytes kept by [`ProviderIpcError::from_error`] (same value
/// as the runtime `bridge.rs` `MAX_REASON_BYTES`; same scrub transform).
pub const MAX_IPC_REASON_BYTES: usize = 512;

/// Typed refusal for over-bound S2 envelope construction. Every variant fails
/// closed with no envelope value and no partial state. Variants carry counts
/// only — never names, reasons, or payload bytes — so logging the error
/// cannot leak caller input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcEnvelopeError {
    /// A caller scope was declared on a `list_models` request. `MP-4`
    /// filtering is deferred, so any scope is denied; only
    /// [`ListModelsRequest::unfiltered`] crosses.
    ReservedScopeDenied,
    /// A `list_models` snapshot declares more than [`MAX_REGISTERED_MODELS`]
    /// models.
    TooManyModels {
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// A `complete` request declares more than [`MAX_IPC_MESSAGES`] messages.
    TooManyMessages {
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// A `complete` message exceeds [`MAX_IPC_MESSAGE_BYTES`] bytes.
    MessageTooLarge {
        /// Bound.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// A `complete` request declares more than [`MAX_IPC_CONTEXT_REFS`]
    /// stable-id references.
    TooManyContextRefs {
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// A `complete` stable-id reference exceeds [`MAX_IPC_CONTEXT_REF_BYTES`]
    /// bytes.
    ContextRefTooLarge {
        /// Bound.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// A `complete` request declares more than [`MAX_IPC_TOOLS`] tool names.
    TooManyTools {
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// A `complete` tool name (or response tool-call name) exceeds
    /// [`MAX_IPC_TOOL_NAME_BYTES`] bytes.
    ToolNameTooLarge {
        /// Bound.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// A `complete` timeout exceeds [`MAX_REQUEST_TIMEOUT_MS`] (same shape as
    /// the [`ProviderError::TimeoutTooLarge`] execution check; the envelope
    /// refuses early and execution re-checks authoritatively).
    TimeoutTooLarge {
        /// Hard ceiling in milliseconds.
        max: u64,
        /// Requested timeout in milliseconds.
        actual: u64,
    },
    /// A `complete` request's message bytes exceed its budget (same shape as
    /// the [`ProviderError::BudgetExceeded`] pre-I/O check; the envelope
    /// refuses early and execution re-checks authoritatively).
    BudgetExceeded {
        /// Budget in bytes.
        limit: usize,
        /// Observed message bytes.
        actual: usize,
    },
    /// A `complete` response text exceeds [`MAX_IPC_TEXT_BYTES`] bytes.
    ResponseTextTooLarge {
        /// Bound.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// A `complete` response declares more than [`MAX_IPC_TOOL_CALLS`] tool
    /// calls.
    TooManyToolCalls {
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// A `complete` tool-call arguments payload exceeds
    /// [`MAX_IPC_TOOL_ARGUMENTS_BYTES`] bytes.
    ToolArgumentsTooLarge {
        /// Bound.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
}

impl Display for IpcEnvelopeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::ReservedScopeDenied => write!(
                f,
                "list_models caller scope is reserved and denied (MP-4 deferred)"
            ),
            Self::TooManyModels { limit, actual } => write!(
                f,
                "list_models snapshot declares {actual} models over {limit} limit"
            ),
            Self::TooManyMessages { limit, actual } => write!(
                f,
                "complete request declares {actual} messages over {limit} limit"
            ),
            Self::MessageTooLarge { limit, actual } => write!(
                f,
                "complete message is {actual} bytes over {limit} byte limit"
            ),
            Self::TooManyContextRefs { limit, actual } => write!(
                f,
                "complete request declares {actual} context refs over {limit} limit"
            ),
            Self::ContextRefTooLarge { limit, actual } => write!(
                f,
                "complete context ref is {actual} bytes over {limit} byte limit"
            ),
            Self::TooManyTools { limit, actual } => write!(
                f,
                "complete request declares {actual} tools over {limit} limit"
            ),
            Self::ToolNameTooLarge { limit, actual } => write!(
                f,
                "complete tool name is {actual} bytes over {limit} byte limit"
            ),
            Self::TimeoutTooLarge { max, actual } => {
                write!(f, "timeout {actual}ms exceeds ceiling {max}ms")
            }
            Self::BudgetExceeded { limit, actual } => write!(
                f,
                "context budget exceeded: {actual} bytes over {limit} byte limit"
            ),
            Self::ResponseTextTooLarge { limit, actual } => {
                write!(f, "complete text is {actual} bytes over {limit} byte limit")
            }
            Self::TooManyToolCalls { limit, actual } => write!(
                f,
                "complete response declares {actual} tool calls over {limit} limit"
            ),
            Self::ToolArgumentsTooLarge { limit, actual } => write!(
                f,
                "complete tool arguments are {actual} bytes over {limit} byte limit"
            ),
        }
    }
}

impl std::error::Error for IpcEnvelopeError {}

/// Reserved `MP-4` caller-scope marker for [`ListModelsRequest`].
///
/// An opaque scope label that is never interpreted: any scoped request is
/// denied by [`ListModelsRequest::check_scope`] because caller-scope
/// filtering is deferred host-side. The label is not echoed by the refusal
/// ([`IpcEnvelopeError::ReservedScopeDenied`] carries no payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservedCallerScope {
    /// Opaque scope label (never interpreted; presence alone denies).
    pub scope: String,
}

/// Scoped-IPC request for `list_models`.
///
/// Mirrors the unfiltered runtime `list_models` (which takes no arguments
/// and returns the whole snapshot): the request carries no filter fields,
/// only the reserved [`ReservedCallerScope`] marker. `None` is the
/// unfiltered shape and the only accepted one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ListModelsRequest {
    /// Reserved `MP-4` caller scope. `None` is the unfiltered shape;
    /// any `Some` value is denied by [`ListModelsRequest::check_scope`].
    pub caller_scope: Option<ReservedCallerScope>,
}

impl ListModelsRequest {
    /// The only accepted shape: an unfiltered `list_models` request.
    #[must_use]
    pub fn unfiltered() -> Self {
        Self { caller_scope: None }
    }

    /// Declare a (reserved) caller scope. The request builds but never
    /// crosses: [`ListModelsRequest::check_scope`] denies it fail-closed.
    #[must_use]
    pub fn scoped(scope: impl Into<String>) -> Self {
        Self {
            caller_scope: Some(ReservedCallerScope {
                scope: scope.into(),
            }),
        }
    }

    /// Fail-closed scope gate: unfiltered passes, any declared scope is
    /// denied because `MP-4` filtering is deferred.
    ///
    /// # Errors
    ///
    /// Returns [`IpcEnvelopeError::ReservedScopeDenied`] when a scope is
    /// declared.
    pub fn check_scope(&self) -> Result<(), IpcEnvelopeError> {
        if self.caller_scope.is_some() {
            Err(IpcEnvelopeError::ReservedScopeDenied)
        } else {
            Ok(())
        }
    }
}

/// Scoped-IPC response for `list_models`: detached [`ModelRegistration`]
/// snapshots with a count bound.
///
/// The snapshots reuse the S1 registration type (provider association plus
/// routing metadata) and are detached: later registry changes do not leak
/// into a previously built response. Order follows
/// [`ProviderRegistry::entries`] (provider/model lexicographic), never
/// insertion order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ListModelsResponse {
    /// Detached model snapshots (at most [`MAX_REGISTERED_MODELS`]).
    pub models: Vec<ModelRegistration>,
}

impl ListModelsResponse {
    /// Build a response, refusing an over-count snapshot fail-closed with no
    /// partial value.
    ///
    /// # Errors
    ///
    /// Returns [`IpcEnvelopeError::TooManyModels`] when `models` exceeds
    /// [`MAX_REGISTERED_MODELS`].
    pub fn new(models: Vec<ModelRegistration>) -> Result<Self, IpcEnvelopeError> {
        if models.len() > MAX_REGISTERED_MODELS {
            return Err(IpcEnvelopeError::TooManyModels {
                limit: MAX_REGISTERED_MODELS,
                actual: models.len(),
            });
        }
        Ok(Self { models })
    }

    /// Snapshot a registry into a detached response. Infallible: the registry
    /// already enforces [`MAX_REGISTERED_MODELS`], and every entry is cloned
    /// so later registration cannot leak into this response.
    #[must_use]
    pub fn from_registry(registry: &ProviderRegistry) -> Self {
        Self {
            models: registry
                .entries()
                .into_iter()
                .map(|entry| ModelRegistration {
                    provider_id: entry.provider_id.clone(),
                    name: entry.name.clone(),
                    capabilities: entry.capabilities.clone(),
                    context_window_tokens: entry.context_window_tokens,
                    input_cost_weight: entry.input_cost_weight,
                    output_cost_weight: entry.output_cost_weight,
                })
                .collect(),
        }
    }

    /// Snapshot model count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the snapshot holds no model.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Envelope message role (the `provider.rs` `Role` vocabulary: human or host
/// instruction, model output, tool observation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcRole {
    /// Human or host instruction.
    User,
    /// Model output.
    Assistant,
    /// Tool observation or assembled context record (may carry untrusted
    /// observation data).
    Tool,
}

/// One bounded envelope message (the `provider.rs` `Message` shape: role plus
/// content bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcMessage {
    /// Message role.
    pub role: IpcRole,
    /// Message bytes (UTF-8; may carry untrusted observation data).
    pub content: String,
}

impl IpcMessage {
    /// Build a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: IpcRole::User,
            content: content.into(),
        }
    }

    /// Build an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: IpcRole::Assistant,
            content: content.into(),
        }
    }

    /// Build a tool/observation message.
    #[must_use]
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: IpcRole::Tool,
            content: content.into(),
        }
    }

    /// Byte length of the message content.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.content.len()
    }
}

/// Sampling declaration note for [`CompleteRequest`] (`MP-5`).
///
/// Records only whether a sampling contract was declared caller-side. Values
/// stay caller-side and validation stays execution-side (`validate_sampling`
/// before any provider I/O): an absent contract (`Undeclared`) flows through
/// untouched, and a declared one (`Declared`) is validated there — never
/// defaulted, never hashed into keys, never echoed by refusals. The envelope
/// therefore carries no float fields and no sampling vocabulary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SamplingNote {
    /// No sampling contract declared; flows through untouched.
    #[default]
    Undeclared,
    /// A sampling contract was declared caller-side; validated at execution.
    Declared,
}

/// Scoped-IPC request for `complete`: `TurnRequest`-equivalent fields.
///
/// The model reference reuses the S1 [`ModelRef`] (shape validation stays at
/// [`ModelRef::new`]; execution re-checks authoritatively with
/// [`ProviderError::UnknownModel`] / [`ProviderError::InvalidProviderId`]).
/// Build the struct literally, then [`CompleteRequest::check`] it before
/// crossing: the check enforces the envelope payload bounds, the timeout
/// ceiling, and the message-bytes budget pre-check in runtime `complete()`
/// order (timeout before budget, mirroring `check_timeout` before the
/// budget comparison). Execution re-checks every gate authoritatively —
/// including unknown-model, sampling (`InvalidSampling` /
/// `UnsupportedSampling`, which echo no values), and latency-vs-timeout —
/// so an envelope pass never implies execution admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteRequest {
    /// Target model reference (owning provider plus registry-known name).
    pub model: ModelRef,
    /// Bounded conversation messages (at most [`MAX_IPC_MESSAGES`], each at
    /// most [`MAX_IPC_MESSAGE_BYTES`] bytes).
    pub messages: Vec<IpcMessage>,
    /// Stable-id references resolved server-side (`CP-2`; at most
    /// [`MAX_IPC_CONTEXT_REFS`], each at most [`MAX_IPC_CONTEXT_REF_BYTES`]
    /// bytes).
    pub context_refs: Vec<String>,
    /// Tool Bus names validated against tool consent at execution (at most
    /// [`MAX_IPC_TOOLS`], each at most [`MAX_IPC_TOOL_NAME_BYTES`] bytes).
    pub tools: Vec<String>,
    /// Per-turn context ceiling in bytes; carries the effective bound (see
    /// `TurnRequest::budget_bytes`), enforced pre-I/O at execution.
    pub budget_bytes: usize,
    /// Caller deadline in milliseconds from [`CompleteRequest::now_ms`]
    /// (`MP-8`; at most [`MAX_REQUEST_TIMEOUT_MS`]).
    pub timeout_ms: u64,
    /// Caller-supplied timestamp; no wall clock is read anywhere on this
    /// path (`CP-7`).
    pub now_ms: u64,
    /// Sampling declaration note (declaration only, never values).
    pub sampling: SamplingNote,
}

impl CompleteRequest {
    /// Total message payload bytes (envelope mirror of
    /// `TurnRequest::total_message_bytes`, used for the pre-I/O budget
    /// check).
    #[must_use]
    pub fn total_message_bytes(&self) -> usize {
        self.messages.iter().map(IpcMessage::byte_len).sum()
    }

    /// Fail-closed envelope shape gate. Every refusal leaves no partial
    /// state; the caller fixes the shape and retries the same request.
    ///
    /// # Errors
    ///
    /// Returns [`IpcEnvelopeError`] for an over-count/over-byte payload, a
    /// timeout past [`MAX_REQUEST_TIMEOUT_MS`], or message bytes over
    /// `budget_bytes`.
    pub fn check(&self) -> Result<(), IpcEnvelopeError> {
        if self.messages.len() > MAX_IPC_MESSAGES {
            return Err(IpcEnvelopeError::TooManyMessages {
                limit: MAX_IPC_MESSAGES,
                actual: self.messages.len(),
            });
        }
        for message in &self.messages {
            if message.content.len() > MAX_IPC_MESSAGE_BYTES {
                return Err(IpcEnvelopeError::MessageTooLarge {
                    limit: MAX_IPC_MESSAGE_BYTES,
                    actual: message.content.len(),
                });
            }
        }
        if self.context_refs.len() > MAX_IPC_CONTEXT_REFS {
            return Err(IpcEnvelopeError::TooManyContextRefs {
                limit: MAX_IPC_CONTEXT_REFS,
                actual: self.context_refs.len(),
            });
        }
        for reference in &self.context_refs {
            if reference.len() > MAX_IPC_CONTEXT_REF_BYTES {
                return Err(IpcEnvelopeError::ContextRefTooLarge {
                    limit: MAX_IPC_CONTEXT_REF_BYTES,
                    actual: reference.len(),
                });
            }
        }
        if self.tools.len() > MAX_IPC_TOOLS {
            return Err(IpcEnvelopeError::TooManyTools {
                limit: MAX_IPC_TOOLS,
                actual: self.tools.len(),
            });
        }
        for tool in &self.tools {
            if tool.len() > MAX_IPC_TOOL_NAME_BYTES {
                return Err(IpcEnvelopeError::ToolNameTooLarge {
                    limit: MAX_IPC_TOOL_NAME_BYTES,
                    actual: tool.len(),
                });
            }
        }
        if self.timeout_ms > MAX_REQUEST_TIMEOUT_MS {
            return Err(IpcEnvelopeError::TimeoutTooLarge {
                max: MAX_REQUEST_TIMEOUT_MS,
                actual: self.timeout_ms,
            });
        }
        let actual = self.total_message_bytes();
        if actual > self.budget_bytes {
            return Err(IpcEnvelopeError::BudgetExceeded {
                limit: self.budget_bytes,
                actual,
            });
        }
        Ok(())
    }
}

/// One envelope tool call (the `provider.rs` `ToolCallRequest` shape:
/// registered tool name plus opaque bounded JSON arguments, dispatched by
/// the Tool Bus and never executed here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteToolCall {
    /// Registered tool name (at most [`MAX_IPC_TOOL_NAME_BYTES`] bytes).
    pub name: String,
    /// Opaque bounded JSON arguments (at most
    /// [`MAX_IPC_TOOL_ARGUMENTS_BYTES`] bytes).
    pub arguments: Vec<u8>,
}

impl CompleteToolCall {
    /// Build a tool call, refusing over-bound payloads fail-closed.
    ///
    /// # Errors
    ///
    /// Returns [`IpcEnvelopeError::ToolNameTooLarge`] or
    /// [`IpcEnvelopeError::ToolArgumentsTooLarge`] for over-bound input.
    pub fn new(name: impl Into<String>, arguments: Vec<u8>) -> Result<Self, IpcEnvelopeError> {
        let name = name.into();
        if name.len() > MAX_IPC_TOOL_NAME_BYTES {
            return Err(IpcEnvelopeError::ToolNameTooLarge {
                limit: MAX_IPC_TOOL_NAME_BYTES,
                actual: name.len(),
            });
        }
        if arguments.len() > MAX_IPC_TOOL_ARGUMENTS_BYTES {
            return Err(IpcEnvelopeError::ToolArgumentsTooLarge {
                limit: MAX_IPC_TOOL_ARGUMENTS_BYTES,
                actual: arguments.len(),
            });
        }
        Ok(Self { name, arguments })
    }
}

/// Estimated token usage for one envelope turn, in whole tokens (mirrors the
/// `provider.rs` `ProviderUsage` shape: observed usage, not billed usage; a
/// zero pair means unreported and the turn loop falls back to a
/// deterministic byte-based estimate. Cache slots are provider-reported
/// prompt-cache accounting in whole tokens; `0` means unreported, never a
/// measured zero-hit. No currency is implied or carried).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompleteUsage {
    /// Estimated input tokens observed for this turn.
    pub input_tokens: u32,
    /// Estimated output tokens observed for this turn.
    pub output_tokens: u32,
    /// Prompt tokens served from cache (`0` = unreported).
    pub cached_tokens: u32,
    /// Prompt tokens written to cache (`0` = unreported).
    pub cache_write_tokens: u32,
}

/// Scoped-IPC response for `complete`: `ProviderTurn`-equivalent (assistant
/// text plus follow-up tool calls, token usage, and observed latency).
/// Owns every payload byte, so a built response is detached by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteResponse {
    /// Assistant text for this turn (at most [`MAX_IPC_TEXT_BYTES`] bytes).
    pub text: String,
    /// Tool calls requested after the text, bounded by the Tool Bus at
    /// dispatch (at most [`MAX_IPC_TOOL_CALLS`]).
    pub tool_calls: Vec<CompleteToolCall>,
    /// Estimated token usage observed for this turn.
    pub usage: CompleteUsage,
    /// Observed provider latency in milliseconds, checked deterministically
    /// against the request timeout at execution.
    pub latency_ms: u64,
}

impl CompleteResponse {
    /// Build a response, revalidating every tool call (entries built without
    /// [`CompleteToolCall::new`] are re-checked here, mirroring the
    /// registry's revalidation of entries built without `new`).
    ///
    /// # Errors
    ///
    /// Returns [`IpcEnvelopeError`] for over-bound text, too many tool
    /// calls, or any over-bound tool call.
    pub fn new(
        text: String,
        tool_calls: Vec<CompleteToolCall>,
        usage: CompleteUsage,
        latency_ms: u64,
    ) -> Result<Self, IpcEnvelopeError> {
        if text.len() > MAX_IPC_TEXT_BYTES {
            return Err(IpcEnvelopeError::ResponseTextTooLarge {
                limit: MAX_IPC_TEXT_BYTES,
                actual: text.len(),
            });
        }
        if tool_calls.len() > MAX_IPC_TOOL_CALLS {
            return Err(IpcEnvelopeError::TooManyToolCalls {
                limit: MAX_IPC_TOOL_CALLS,
                actual: tool_calls.len(),
            });
        }
        for call in &tool_calls {
            if call.name.len() > MAX_IPC_TOOL_NAME_BYTES {
                return Err(IpcEnvelopeError::ToolNameTooLarge {
                    limit: MAX_IPC_TOOL_NAME_BYTES,
                    actual: call.name.len(),
                });
            }
            if call.arguments.len() > MAX_IPC_TOOL_ARGUMENTS_BYTES {
                return Err(IpcEnvelopeError::ToolArgumentsTooLarge {
                    limit: MAX_IPC_TOOL_ARGUMENTS_BYTES,
                    actual: call.arguments.len(),
                });
            }
        }
        Ok(Self {
            text,
            tool_calls,
            usage,
            latency_ms,
        })
    }
}

/// Truncate an identifier to a byte bound on a character boundary (never
/// splits a character). Values within the bound pass through untouched.
fn truncate_to_bound(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_owned();
    }
    let mut end = limit;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

/// Scrub and bound a free-text reason for IPC crossing (mirrors the runtime
/// `bridge.rs` `bound_reason` transform with the same value): every
/// character outside printable ASCII (`0x20..=0x7E`, so spaces are kept)
/// becomes `?`, which defuses CR/LF log injection, terminal escapes, and
/// confusable non-ASCII bytes; the result is then truncated to
/// [`MAX_IPC_REASON_BYTES`] bytes. Deterministic, silent (no ellipsis
/// marker that could itself exceed the bound), and never splits a
/// character, so the normalized value is safe to log directly.
fn scrub_reason(reason: &str) -> String {
    let mut bounded = String::with_capacity(reason.len().min(MAX_IPC_REASON_BYTES));
    for character in reason.chars() {
        if bounded.len() >= MAX_IPC_REASON_BYTES {
            break;
        }
        if character.is_ascii_graphic() || character == ' ' {
            bounded.push(character);
        } else {
            bounded.push('?');
        }
    }
    bounded
}

/// Bounded scoped-IPC envelope form of [`ProviderError`]: the same 13
/// variants, normalized for boundary crossing.
///
/// Identifier fields keep their shaped values truncated to their shape bound
/// ([`MAX_MODEL_NAME_LEN`] for model names, [`MAX_PROVIDER_ID_LEN`] for
/// provider ids) on a character boundary; free-text reason fields are
/// scrubbed and truncated to [`MAX_IPC_REASON_BYTES`] (see
/// [`ProviderIpcError::from_error`]); counts, timeouts, budgets, and the
/// `retry_after_ms` hint cross unchanged; `missing` capability bits cross as
/// typed routing data; the `UnsupportedSampling` backend label is a static
/// adapter string, never caller input. Sampling refusals echo no values, so
/// logs and reconcile reports stay static. No variant carries secrets,
/// credentials, or key material.
///
/// The [`ProviderIpcError::from_error`] / [`ProviderIpcError::into_error`]
/// mapping is total and exhaustive (no wildcard arm): adding a
/// [`ProviderError`] variant breaks compilation here, which is the
/// fail-closed response to a new failure mode (it must be classified before
/// any envelope carries it). Fallback classification over these errors stays
/// on the selection side: convert back with [`ProviderIpcError::into_error`]
/// and classify there (AI-0279 owns the select/matches/fallback envelopes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderIpcError {
    /// `model` names no registry-known model (name truncated to
    /// [`MAX_MODEL_NAME_LEN`]).
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
    /// Provider id violates the `MP-2` shape (id truncated to
    /// [`MAX_PROVIDER_ID_LEN`]).
    InvalidProviderId {
        /// Rejected id.
        id: String,
    },
    /// Transport failure before any model effect (provider truncated to
    /// [`MAX_PROVIDER_ID_LEN`], reason scrubbed to [`MAX_IPC_REASON_BYTES`]).
    Transport {
        /// Owning provider id.
        provider: String,
        /// What failed (no secrets, no credentials, no key material).
        reason: String,
    },
    /// Provider refused authorization (provider truncated, reason scrubbed;
    /// reports the refusal only, never credentials).
    Auth {
        /// Owning provider id.
        provider: String,
        /// Refusal detail (no secrets, no credentials, no key material).
        reason: String,
    },
    /// Provider rate limit hit (provider truncated to
    /// [`MAX_PROVIDER_ID_LEN`]).
    RateLimited {
        /// Owning provider id.
        provider: String,
        /// Advised wait in milliseconds; `None` when the provider gave none.
        retry_after_ms: Option<u64>,
    },
    /// Selected model lacks capabilities the request required (provider and
    /// model truncated; `missing` crosses as typed routing bits).
    CapabilityMismatch {
        /// Owning provider id.
        provider: String,
        /// Requested model name.
        model: String,
        /// Required capabilities the model does not advertise.
        missing: Vec<ModelCapability>,
    },
    /// Model is known but unavailable (provider and model truncated).
    ModelUnavailable {
        /// Owning provider id.
        provider: String,
        /// Unavailable model name.
        model: String,
    },
    /// The effect may have happened but acknowledgement was lost (provider
    /// truncated, reason scrubbed; reconcile before retry, never fall back
    /// blindly).
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

impl ProviderIpcError {
    /// Normalize a [`ProviderError`] for IPC crossing: identifiers truncated
    /// to their shape bound, reasons scrubbed and truncated to
    /// [`MAX_IPC_REASON_BYTES`]. Total over the source taxonomy: every one
    /// of the 13 variants maps, with no wildcard arm.
    #[must_use]
    pub fn from_error(error: &ProviderError) -> Self {
        match error {
            ProviderError::UnknownModel { name } => Self::UnknownModel {
                name: truncate_to_bound(name, MAX_MODEL_NAME_LEN),
            },
            ProviderError::BudgetExceeded { limit, actual } => Self::BudgetExceeded {
                limit: *limit,
                actual: *actual,
            },
            ProviderError::Timeout {
                timeout_ms,
                latency_ms,
            } => Self::Timeout {
                timeout_ms: *timeout_ms,
                latency_ms: *latency_ms,
            },
            ProviderError::TimeoutTooLarge { max, actual } => Self::TimeoutTooLarge {
                max: *max,
                actual: *actual,
            },
            ProviderError::InvalidProviderId { id } => Self::InvalidProviderId {
                id: truncate_to_bound(id, MAX_PROVIDER_ID_LEN),
            },
            ProviderError::Transport { provider, reason } => Self::Transport {
                provider: truncate_to_bound(provider, MAX_PROVIDER_ID_LEN),
                reason: scrub_reason(reason),
            },
            ProviderError::Auth { provider, reason } => Self::Auth {
                provider: truncate_to_bound(provider, MAX_PROVIDER_ID_LEN),
                reason: scrub_reason(reason),
            },
            ProviderError::RateLimited {
                provider,
                retry_after_ms,
            } => Self::RateLimited {
                provider: truncate_to_bound(provider, MAX_PROVIDER_ID_LEN),
                retry_after_ms: *retry_after_ms,
            },
            ProviderError::CapabilityMismatch {
                provider,
                model,
                missing,
            } => Self::CapabilityMismatch {
                provider: truncate_to_bound(provider, MAX_PROVIDER_ID_LEN),
                model: truncate_to_bound(model, MAX_MODEL_NAME_LEN),
                missing: missing.clone(),
            },
            ProviderError::ModelUnavailable { provider, model } => Self::ModelUnavailable {
                provider: truncate_to_bound(provider, MAX_PROVIDER_ID_LEN),
                model: truncate_to_bound(model, MAX_MODEL_NAME_LEN),
            },
            ProviderError::Unknown { provider, reason } => Self::Unknown {
                provider: truncate_to_bound(provider, MAX_PROVIDER_ID_LEN),
                reason: scrub_reason(reason),
            },
            ProviderError::InvalidSampling => Self::InvalidSampling,
            ProviderError::UnsupportedSampling { field } => Self::UnsupportedSampling { field },
        }
    }

    /// Recover the [`ProviderError`] carried by this envelope. Total: every
    /// one of the 13 envelope variants maps back, with no wildcard arm.
    #[must_use]
    pub fn into_error(self) -> ProviderError {
        match self {
            Self::UnknownModel { name } => ProviderError::UnknownModel { name },
            Self::BudgetExceeded { limit, actual } => {
                ProviderError::BudgetExceeded { limit, actual }
            }
            Self::Timeout {
                timeout_ms,
                latency_ms,
            } => ProviderError::Timeout {
                timeout_ms,
                latency_ms,
            },
            Self::TimeoutTooLarge { max, actual } => ProviderError::TimeoutTooLarge { max, actual },
            Self::InvalidProviderId { id } => ProviderError::InvalidProviderId { id },
            Self::Transport { provider, reason } => ProviderError::Transport { provider, reason },
            Self::Auth { provider, reason } => ProviderError::Auth { provider, reason },
            Self::RateLimited {
                provider,
                retry_after_ms,
            } => ProviderError::RateLimited {
                provider,
                retry_after_ms,
            },
            Self::CapabilityMismatch {
                provider,
                model,
                missing,
            } => ProviderError::CapabilityMismatch {
                provider,
                model,
                missing,
            },
            Self::ModelUnavailable { provider, model } => {
                ProviderError::ModelUnavailable { provider, model }
            }
            Self::Unknown { provider, reason } => ProviderError::Unknown { provider, reason },
            Self::InvalidSampling => ProviderError::InvalidSampling,
            Self::UnsupportedSampling { field } => ProviderError::UnsupportedSampling { field },
        }
    }
}

impl Display for ProviderIpcError {
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

impl std::error::Error for ProviderIpcError {}

// ---------------------------------------------------------------------------
// Phase 2 S3: MP-4 caller-scope filter (AI-0290).
//
// Host-side intersection-only narrowing over the S1 mirrored registry per the
// `MP-4 caller-scope filter` candidate in `provider-plugin-boundary.md`
// (scope vocabulary plus filter semantics paragraphs; the `MCP placement
// policy` paragraphs belong to AI-0296 and appear here by name only).
//
// What is reused, not re-mirrored: [`ModelRegistration`],
// [`ProviderRegistry`], [`ListModelsResponse`], [`ListModelsRequest`] (its
// unfiltered shape), [`MAX_REGISTERED_MODELS`], [`MAX_PROVIDER_ID_LEN`],
// `validate_provider_id`, and `truncate_to_bound`. What is new here:
// [`MAX_CALLER_SCOPE_LEN`], [`CallerScopeEcho`], [`ScopeFilterError`],
// [`validate_caller_scope`], and
// [`ProviderRegistry::list_models_filtered`]. What stays out: selection-side
// envelopes (AI-0291 owns those), transport of any kind, any `version` field,
// and any change to the S1/S2 paths below.
//
// Contract summary (re-stated, not re-implemented):
//
// - Scope vocabulary stays narrow and reuses existing authority. A scope
//   string is a bounded, printable-ASCII, slash-separated name with a closed
//   character set and a stated length ceiling ([`MAX_CALLER_SCOPE_LEN`]); an
//   empty, overlong, or out-of-grammar value stays invalid with
//   [`ScopeFilterError::InvalidScope`]. The `provider/<provider_id>` semantic
//   prefix used here is the S3 placement for this seam; any other well-formed
//   prefix stays unknown, never invalid.
// - Filter semantics are intersection-only narrowing with deny-by-default.
//   The unfiltered snapshot ([`ListModelsResponse::from_registry`], plus the
//   S2 [`ListModelsRequest::unfiltered`] / `check_scope` path) stays preserved
//   as-is; a filtered view is a subset of that snapshot for the same dispatch
//   `now_ms` and never wider. Adding a scope never widens visibility. An
//   unknown scope fails closed with [`ScopeFilterError::UnknownScope`]
//   carrying names only (requested scope plus `provider/model` entry names),
//   never credential values or raw vendor text.
// - Enforcement stays host-side at dispatch `now_ms`: the request envelope
//   carries a [`CallerScopeEcho`] for audit while the host — never the
//   adapter — applies the filter before selection and provider I/O. The echo
//   is observation data, not authority: constructing an echo grants nothing,
//   and `now_ms` (caller-supplied, `CP-7`) does not widen or narrow the set.
//   The filter reads no clock and performs no I/O.
// - Types plus pure checks only: `std`-only, no threads, no clock reads, no
//   filesystem or network access, no `Cargo.toml` changes, and no behavior
//   outside the filter.
// ---------------------------------------------------------------------------

/// Maximum caller-scope string length in bytes (S3 ceiling for the `MP-4`
/// scope vocabulary; chosen to match [`MAX_PROVIDER_ID_LEN`] so a
/// `provider/<provider_id>` scope always fits when the id fits).
pub const MAX_CALLER_SCOPE_LEN: usize = 64;

/// Caller scope echo for audit (observation data, not authority).
///
/// Carries the caller-declared scope string plus the caller-supplied dispatch
/// timestamp (`CP-7`, same convention as `CompleteRequest::now_ms`: no wall
/// clock is read anywhere on this path). Constructing this value grants
/// nothing: only [`ProviderRegistry::list_models_filtered`] enforces
/// host-side, and `now_ms` never widens or narrows the set (the same scope
/// over the same registry yields the same subset for any `now_ms`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallerScopeEcho {
    /// Caller-declared scope observation (validated at filter time).
    pub scope: String,
    /// Caller-supplied dispatch timestamp in milliseconds (audit only).
    pub now_ms: u64,
}

impl CallerScopeEcho {
    /// Record a scope observation for audit. No validation happens here and
    /// no authority is granted; validation plus enforcement stay host-side in
    /// [`ProviderRegistry::list_models_filtered`].
    #[must_use]
    pub fn new(scope: impl Into<String>, now_ms: u64) -> Self {
        Self {
            scope: scope.into(),
            now_ms,
        }
    }
}

/// Typed refusal for the S3 caller-scope filter. Every variant fails closed
/// with no partial view. Variants carry names only (scope strings plus
/// `provider/model` entry names, all validated printable-ASCII); no variant
/// carries secrets, credentials, key material, or raw vendor text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeFilterError {
    /// Scope string violates the narrow vocabulary: empty, overlong, or
    /// out-of-grammar (missing slash, leading/trailing slash, `//`, or a
    /// byte outside `a-z0-9_-/` or a segment starting outside `a-z0-9`).
    /// Carries the rejected string truncated to [`MAX_CALLER_SCOPE_LEN`] on
    /// a character boundary plus a static reason (`"empty"`, `"overlong"`,
    /// or `"grammar"`); no caller value beyond that bounded echo.
    InvalidScope {
        /// Rejected scope truncated to the ceiling.
        scope: String,
        /// Static reason label (never caller input).
        reason: &'static str,
    },
    /// Scope string is well-formed but matches no registry entry (unknown
    /// provider, non-`provider/` prefix, or a known provider with zero
    /// entries in this snapshot). Carries the requested scope plus the
    /// snapshot entry names as `provider/model` pairs in
    /// [`ProviderRegistry::entries`] order for reconcile; the `Display`
    /// shows the requested scope plus the entry count only, so logs stay
    /// bounded while the struct keeps the full names for audit.
    UnknownScope {
        /// Requested scope name (bounded, well-formed).
        requested: String,
        /// Snapshot entry names (`provider/model`, lexicographic).
        available: Vec<String>,
    },
}

impl Display for ScopeFilterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidScope { scope, reason } => {
                write!(f, "invalid caller scope ({reason}): {scope}")
            }
            Self::UnknownScope {
                requested,
                available,
            } => write!(
                f,
                "unknown caller scope: {requested} ({} known entries)",
                available.len()
            ),
        }
    }
}

impl std::error::Error for ScopeFilterError {}

/// Validate a caller-scope string against the narrow S3 vocabulary:
/// non-empty, at most [`MAX_CALLER_SCOPE_LEN`] bytes, containing at least one
/// `/`, never leading/trailing `/`, never `//`, every byte in
/// `a-z0-9_-/`, and every slash-separated segment starting with `a-z0-9`.
///
/// Pure shape check only: well-formed does not mean known (a well-formed
/// non-`provider/` scope or a `provider/` scope with no entries still fails
/// with [`ScopeFilterError::UnknownScope`] at filter time).
///
/// # Errors
///
/// Returns [`ScopeFilterError::InvalidScope`] with reason `"empty"`,
/// `"overlong"`, or `"grammar"` when the shape is violated.
pub fn validate_caller_scope(scope: &str) -> Result<(), ScopeFilterError> {
    if scope.is_empty() {
        return Err(ScopeFilterError::InvalidScope {
            scope: String::new(),
            reason: "empty",
        });
    }
    if scope.len() > MAX_CALLER_SCOPE_LEN {
        return Err(ScopeFilterError::InvalidScope {
            scope: truncate_to_bound(scope, MAX_CALLER_SCOPE_LEN),
            reason: "overlong",
        });
    }
    let grammar_invalid = !scope.contains('/')
        || scope.starts_with('/')
        || scope.ends_with('/')
        || scope.contains("//")
        || !scope.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-' || b == b'/'
        })
        || scope.split('/').any(|segment| {
            segment.is_empty()
                || !segment
                    .bytes()
                    .next()
                    .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        });
    if grammar_invalid {
        return Err(ScopeFilterError::InvalidScope {
            scope: truncate_to_bound(scope, MAX_CALLER_SCOPE_LEN),
            reason: "grammar",
        });
    }
    Ok(())
}

/// Snapshot entry names as `provider/model` pairs in
/// [`ProviderRegistry::entries`] order (names only, never secrets).
fn available_entry_names(registry: &ProviderRegistry) -> Vec<String> {
    registry
        .entries()
        .into_iter()
        .map(|entry| format!("{}/{}", entry.provider_id, entry.name))
        .collect()
}

/// Build the fail-closed unknown-scope refusal for `requested` over
/// `registry` (names only).
fn unknown_scope_error(requested: String, registry: &ProviderRegistry) -> ScopeFilterError {
    ScopeFilterError::UnknownScope {
        requested,
        available: available_entry_names(registry),
    }
}

impl ProviderRegistry {
    /// Filtered `list_models` view for one caller-scope echo (S3, `MP-4`).
    ///
    /// Intersection-only narrowing with deny-by-default, enforced host-side:
    /// the `provider/<provider_id>` scope selects the snapshot entries whose
    /// `provider_id` equals `<provider_id>`, in [`ProviderRegistry::entries`]
    /// order, as detached [`ModelRegistration`] clones (later registry
    /// changes do not leak into the returned view). The unfiltered path
    /// ([`ListModelsResponse::from_registry`]) is untouched: the filtered
    /// view is always a subset of that snapshot for the same dispatch
    /// `now_ms` and never wider. `echo.now_ms` is audit observation only
    /// (caller-supplied `CP-7`, no clock read): it does not affect
    /// membership. The echo alone grants nothing; only this host-side call
    /// enforces.
    ///
    /// # Errors
    ///
    /// Returns [`ScopeFilterError::InvalidScope`] for a malformed scope
    /// string, or [`ScopeFilterError::UnknownScope`] (names only) when the
    /// scope is well-formed but matches no entry.
    pub fn list_models_filtered(
        &self,
        echo: &CallerScopeEcho,
    ) -> Result<ListModelsResponse, ScopeFilterError> {
        validate_caller_scope(&echo.scope)?;
        let provider = echo
            .scope
            .strip_prefix("provider/")
            .ok_or_else(|| unknown_scope_error(echo.scope.clone(), self))?;
        if validate_provider_id(provider).is_err() {
            return Err(unknown_scope_error(echo.scope.clone(), self));
        }
        let filtered: Vec<ModelRegistration> = self
            .entries()
            .into_iter()
            .filter(|entry| entry.provider_id == provider)
            .map(|entry| ModelRegistration {
                provider_id: entry.provider_id.clone(),
                name: entry.name.clone(),
                capabilities: entry.capabilities.clone(),
                context_window_tokens: entry.context_window_tokens,
                input_cost_weight: entry.input_cost_weight,
                output_cost_weight: entry.output_cost_weight,
            })
            .collect();
        if filtered.is_empty() {
            return Err(unknown_scope_error(echo.scope.clone(), self));
        }
        // Bound holds by construction: `filtered` is a subset of a registry
        // that already enforces `MAX_REGISTERED_MODELS`, so no count refusal
        // is reachable here.
        Ok(ListModelsResponse { models: filtered })
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

#[cfg(test)]
mod ipc_envelope_tests {
    use super::*;

    fn complete_request() -> CompleteRequest {
        CompleteRequest {
            model: ModelRef::new("bitty-fake", "fake-chat").expect("valid ref"),
            messages: vec![IpcMessage::user("hello")],
            context_refs: vec!["stable-id-1".to_owned()],
            tools: vec!["read".to_owned()],
            budget_bytes: 32 * 1024,
            timeout_ms: 5_000,
            now_ms: 1_000,
            sampling: SamplingNote::Undeclared,
        }
    }

    fn registration_at(index: usize) -> ModelRegistration {
        ModelRegistration::new(
            "bitty-fake",
            format!("model-{index}"),
            vec![ModelCapability::Text],
            4_096,
            1,
            2,
        )
        .expect("valid registration")
    }

    #[test]
    fn ipc_bounds_match_source_values() {
        assert_eq!(MAX_REQUEST_TIMEOUT_MS, 30_000);
        assert_eq!(MAX_IPC_MESSAGES, 128);
        assert_eq!(MAX_IPC_MESSAGE_BYTES, 16 * 1024);
        assert_eq!(MAX_IPC_CONTEXT_REFS, 128);
        assert_eq!(MAX_IPC_CONTEXT_REF_BYTES, 64);
        assert_eq!(MAX_IPC_TOOLS, 32);
        assert_eq!(MAX_IPC_TOOL_NAME_BYTES, 64);
        assert_eq!(MAX_IPC_TEXT_BYTES, 16 * 1024);
        assert_eq!(MAX_IPC_TOOL_CALLS, 8);
        assert_eq!(MAX_IPC_TOOL_ARGUMENTS_BYTES, 16 * 1024);
        assert_eq!(MAX_IPC_REASON_BYTES, 512);
    }

    #[test]
    fn list_models_scope_gate() {
        // The unfiltered shape (and the default, which is unfiltered) passes.
        assert!(ListModelsRequest::unfiltered().check_scope().is_ok());
        assert!(ListModelsRequest::default().check_scope().is_ok());
        assert_eq!(
            ListModelsRequest::default(),
            ListModelsRequest::unfiltered()
        );
        // Any declared scope is denied: MP-4 filtering is deferred.
        let scoped = ListModelsRequest::scoped("team-a");
        assert_eq!(
            scoped.check_scope().expect_err("scoped must fail"),
            IpcEnvelopeError::ReservedScopeDenied
        );
        // The refusal echoes no scope payload.
        assert_eq!(
            IpcEnvelopeError::ReservedScopeDenied.to_string(),
            "list_models caller scope is reserved and denied (MP-4 deferred)"
        );
    }

    #[test]
    fn list_models_response_count_bound() {
        let fitting: Vec<ModelRegistration> =
            (0..MAX_REGISTERED_MODELS).map(registration_at).collect();
        let response = ListModelsResponse::new(fitting).expect("64 fits");
        assert_eq!(response.len(), MAX_REGISTERED_MODELS);
        assert!(!response.is_empty());
        assert!(ListModelsResponse::default().is_empty());
        let mut burst: Vec<ModelRegistration> =
            (0..MAX_REGISTERED_MODELS).map(registration_at).collect();
        burst.push(registration_at(MAX_REGISTERED_MODELS));
        assert_eq!(
            ListModelsResponse::new(burst).expect_err("65 must fail"),
            IpcEnvelopeError::TooManyModels {
                limit: MAX_REGISTERED_MODELS,
                actual: MAX_REGISTERED_MODELS + 1,
            }
        );
    }

    #[test]
    fn list_models_response_is_detached_snapshot() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(
                ModelRegistration::new("bitty-b", "chat", vec![ModelCapability::Text], 4_096, 1, 2)
                    .expect("valid registration"),
            )
            .expect("capacity");
        registry
            .register(
                ModelRegistration::new(
                    "bitty-a",
                    "vision",
                    vec![ModelCapability::Text, ModelCapability::ImageInput],
                    32_768,
                    3,
                    4,
                )
                .expect("valid registration"),
            )
            .expect("capacity");
        let response = ListModelsResponse::from_registry(&registry);
        // Deterministic entries() order (provider/model lexicographic),
        // detached clones of the registry state.
        let order: Vec<(&str, &str)> = response
            .models
            .iter()
            .map(|item| (item.provider_id.as_str(), item.name.as_str()))
            .collect();
        assert_eq!(order, vec![("bitty-a", "vision"), ("bitty-b", "chat")]);
        // Later registry state cannot leak into the snapshot.
        registry
            .register(
                ModelRegistration::new(
                    "bitty-c",
                    "fresh",
                    vec![ModelCapability::Text],
                    4_096,
                    1,
                    1,
                )
                .expect("valid registration"),
            )
            .expect("capacity");
        assert_eq!(response.len(), 2);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn complete_request_accepts_bounded_shape() {
        let request = complete_request();
        assert_eq!(request.total_message_bytes(), "hello".len());
        assert!(request.check().is_ok());
        // Declared sampling is a note only: no values cross, nothing to
        // validate at the envelope, so the check still passes.
        let declared = CompleteRequest {
            sampling: SamplingNote::Declared,
            ..request
        };
        assert!(declared.check().is_ok());
        assert_eq!(SamplingNote::default(), SamplingNote::Undeclared);
    }

    #[test]
    fn complete_request_refuses_over_count_bounds() {
        let burst_messages = CompleteRequest {
            messages: (0..MAX_IPC_MESSAGES + 1)
                .map(|index| IpcMessage::user(format!("m{index}")))
                .collect(),
            ..complete_request()
        };
        assert_eq!(
            burst_messages.check().expect_err("too many messages"),
            IpcEnvelopeError::TooManyMessages {
                limit: MAX_IPC_MESSAGES,
                actual: MAX_IPC_MESSAGES + 1,
            }
        );
        let burst_refs = CompleteRequest {
            context_refs: (0..MAX_IPC_CONTEXT_REFS + 1)
                .map(|index| format!("ref-{index}"))
                .collect(),
            ..complete_request()
        };
        assert_eq!(
            burst_refs.check().expect_err("too many refs"),
            IpcEnvelopeError::TooManyContextRefs {
                limit: MAX_IPC_CONTEXT_REFS,
                actual: MAX_IPC_CONTEXT_REFS + 1,
            }
        );
        let burst_tools = CompleteRequest {
            tools: (0..MAX_IPC_TOOLS + 1)
                .map(|index| format!("tool-{index}"))
                .collect(),
            ..complete_request()
        };
        assert_eq!(
            burst_tools.check().expect_err("too many tools"),
            IpcEnvelopeError::TooManyTools {
                limit: MAX_IPC_TOOLS,
                actual: MAX_IPC_TOOLS + 1,
            }
        );
    }

    #[test]
    fn complete_request_refuses_over_byte_bounds() {
        let big_message = CompleteRequest {
            messages: vec![IpcMessage::user("x".repeat(MAX_IPC_MESSAGE_BYTES + 1))],
            budget_bytes: usize::MAX,
            ..complete_request()
        };
        assert_eq!(
            big_message.check().expect_err("message too large"),
            IpcEnvelopeError::MessageTooLarge {
                limit: MAX_IPC_MESSAGE_BYTES,
                actual: MAX_IPC_MESSAGE_BYTES + 1,
            }
        );
        let big_ref = CompleteRequest {
            context_refs: vec!["r".repeat(MAX_IPC_CONTEXT_REF_BYTES + 1)],
            ..complete_request()
        };
        assert_eq!(
            big_ref.check().expect_err("ref too large"),
            IpcEnvelopeError::ContextRefTooLarge {
                limit: MAX_IPC_CONTEXT_REF_BYTES,
                actual: MAX_IPC_CONTEXT_REF_BYTES + 1,
            }
        );
        let big_tool = CompleteRequest {
            tools: vec!["t".repeat(MAX_IPC_TOOL_NAME_BYTES + 1)],
            ..complete_request()
        };
        assert_eq!(
            big_tool.check().expect_err("tool too large"),
            IpcEnvelopeError::ToolNameTooLarge {
                limit: MAX_IPC_TOOL_NAME_BYTES,
                actual: MAX_IPC_TOOL_NAME_BYTES + 1,
            }
        );
    }

    #[test]
    fn complete_request_refuses_timeout_and_budget() {
        // Exact-limit values pass: the ceiling is inclusive and message
        // bytes equal to the budget are within budget.
        let edge = CompleteRequest {
            timeout_ms: MAX_REQUEST_TIMEOUT_MS,
            budget_bytes: "hello".len(),
            ..complete_request()
        };
        assert!(edge.check().is_ok());
        let slow = CompleteRequest {
            timeout_ms: MAX_REQUEST_TIMEOUT_MS + 1,
            ..complete_request()
        };
        assert_eq!(
            slow.check().expect_err("timeout too large"),
            IpcEnvelopeError::TimeoutTooLarge {
                max: MAX_REQUEST_TIMEOUT_MS,
                actual: MAX_REQUEST_TIMEOUT_MS + 1,
            }
        );
        // Same wording as the execution-side ceiling refusal.
        assert_eq!(
            IpcEnvelopeError::TimeoutTooLarge {
                max: 30_000,
                actual: 60_000,
            }
            .to_string(),
            "timeout 60000ms exceeds ceiling 30000ms"
        );
        let over_budget = CompleteRequest {
            budget_bytes: "hello".len() - 1,
            ..complete_request()
        };
        assert_eq!(
            over_budget.check().expect_err("budget exceeded"),
            IpcEnvelopeError::BudgetExceeded {
                limit: "hello".len() - 1,
                actual: "hello".len(),
            }
        );
        assert_eq!(
            IpcEnvelopeError::BudgetExceeded {
                limit: 1,
                actual: 2
            }
            .to_string(),
            "context budget exceeded: 2 bytes over 1 byte limit"
        );
    }

    #[test]
    fn complete_response_accepts_bounded_shape() {
        let response = CompleteResponse::new(
            "done".to_owned(),
            vec![CompleteToolCall::new("read", br#"{"path":"x"}"#.to_vec()).expect("bounded")],
            CompleteUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_tokens: 0,
                cache_write_tokens: 0,
            },
            120,
        )
        .expect("bounded response builds");
        assert_eq!(response.text, "done");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.latency_ms, 120);
    }

    #[test]
    fn complete_response_refuses_over_bounds() {
        let usage = CompleteUsage::default();
        assert_eq!(
            CompleteResponse::new("x".repeat(MAX_IPC_TEXT_BYTES + 1), Vec::new(), usage, 0)
                .expect_err("text too large"),
            IpcEnvelopeError::ResponseTextTooLarge {
                limit: MAX_IPC_TEXT_BYTES,
                actual: MAX_IPC_TEXT_BYTES + 1,
            }
        );
        let burst: Vec<CompleteToolCall> = (0..MAX_IPC_TOOL_CALLS)
            .map(|index| {
                CompleteToolCall::new(format!("tool-{index}"), Vec::new()).expect("bounded call")
            })
            .collect();
        assert!(CompleteResponse::new("ok".to_owned(), burst, usage, 0).is_ok());
        let too_many: Vec<CompleteToolCall> = (0..MAX_IPC_TOOL_CALLS + 1)
            .map(|index| {
                CompleteToolCall::new(format!("tool-{index}"), Vec::new()).expect("bounded call")
            })
            .collect();
        assert_eq!(
            CompleteResponse::new("ok".to_owned(), too_many, usage, 0).expect_err("too many calls"),
            IpcEnvelopeError::TooManyToolCalls {
                limit: MAX_IPC_TOOL_CALLS,
                actual: MAX_IPC_TOOL_CALLS + 1,
            }
        );
        // Entries built without `new` are revalidated here: no smuggling.
        let smuggled_name = CompleteToolCall {
            name: "n".repeat(MAX_IPC_TOOL_NAME_BYTES + 1),
            arguments: Vec::new(),
        };
        assert_eq!(
            CompleteResponse::new("ok".to_owned(), vec![smuggled_name], usage, 0)
                .expect_err("smuggled name must fail"),
            IpcEnvelopeError::ToolNameTooLarge {
                limit: MAX_IPC_TOOL_NAME_BYTES,
                actual: MAX_IPC_TOOL_NAME_BYTES + 1,
            }
        );
        let smuggled_args = CompleteToolCall {
            name: "read".to_owned(),
            arguments: vec![0u8; MAX_IPC_TOOL_ARGUMENTS_BYTES + 1],
        };
        assert_eq!(
            CompleteResponse::new("ok".to_owned(), vec![smuggled_args], usage, 0)
                .expect_err("smuggled args must fail"),
            IpcEnvelopeError::ToolArgumentsTooLarge {
                limit: MAX_IPC_TOOL_ARGUMENTS_BYTES,
                actual: MAX_IPC_TOOL_ARGUMENTS_BYTES + 1,
            }
        );
    }

    #[test]
    fn complete_tool_call_constructor_bounds() {
        assert!(CompleteToolCall::new("read", vec![1, 2, 3]).is_ok());
        assert_eq!(
            CompleteToolCall::new("n".repeat(MAX_IPC_TOOL_NAME_BYTES + 1), Vec::new())
                .expect_err("name too large"),
            IpcEnvelopeError::ToolNameTooLarge {
                limit: MAX_IPC_TOOL_NAME_BYTES,
                actual: MAX_IPC_TOOL_NAME_BYTES + 1,
            }
        );
        assert_eq!(
            CompleteToolCall::new("read", vec![0u8; MAX_IPC_TOOL_ARGUMENTS_BYTES + 1])
                .expect_err("args too large"),
            IpcEnvelopeError::ToolArgumentsTooLarge {
                limit: MAX_IPC_TOOL_ARGUMENTS_BYTES,
                actual: MAX_IPC_TOOL_ARGUMENTS_BYTES + 1,
            }
        );
    }

    #[test]
    fn provider_ipc_error_mapping_is_total() {
        // Every one of the 13 source variants maps and round-trips,
        // including Unknown (reconcile-before-retry, never blind fallback).
        let cases = [
            ProviderError::UnknownModel {
                name: "m".to_owned(),
            },
            ProviderError::BudgetExceeded {
                limit: 1,
                actual: 2,
            },
            ProviderError::Timeout {
                timeout_ms: 5_000,
                latency_ms: 6_000,
            },
            ProviderError::TimeoutTooLarge {
                max: 30_000,
                actual: 60_000,
            },
            ProviderError::InvalidProviderId {
                id: "BAD".to_owned(),
            },
            ProviderError::Transport {
                provider: "p".to_owned(),
                reason: "reset".to_owned(),
            },
            ProviderError::Auth {
                provider: "p".to_owned(),
                reason: "revoked".to_owned(),
            },
            ProviderError::RateLimited {
                provider: "p".to_owned(),
                retry_after_ms: Some(1_000),
            },
            ProviderError::RateLimited {
                provider: "p".to_owned(),
                retry_after_ms: None,
            },
            ProviderError::CapabilityMismatch {
                provider: "p".to_owned(),
                model: "m".to_owned(),
                missing: vec![ModelCapability::ImageInput],
            },
            ProviderError::ModelUnavailable {
                provider: "p".to_owned(),
                model: "m".to_owned(),
            },
            ProviderError::Unknown {
                provider: "p".to_owned(),
                reason: "ack lost".to_owned(),
            },
            ProviderError::InvalidSampling,
            ProviderError::UnsupportedSampling { field: "top_k" },
        ];
        for error in &cases {
            let envelope = ProviderIpcError::from_error(error);
            assert_eq!(envelope.clone().into_error(), *error);
            assert_eq!(envelope.to_string(), error.to_string());
        }
    }

    #[test]
    fn provider_ipc_error_truncates_and_scrubs() {
        // Over-long identifiers truncate to their shape bound on a
        // character boundary (never splitting a character).
        let long_name = "n".repeat(MAX_MODEL_NAME_LEN + 1);
        let envelope =
            ProviderIpcError::from_error(&ProviderError::UnknownModel { name: long_name });
        match envelope {
            ProviderIpcError::UnknownModel { name } => {
                assert_eq!(name.len(), MAX_MODEL_NAME_LEN);
            }
            other => panic!("unexpected envelope: {other:?}"),
        }
        let multibyte_provider = format!("{}é", "a".repeat(MAX_PROVIDER_ID_LEN - 1));
        assert_eq!(multibyte_provider.len(), MAX_PROVIDER_ID_LEN + 1);
        let envelope = ProviderIpcError::from_error(&ProviderError::Transport {
            provider: multibyte_provider,
            reason: "reset".to_owned(),
        });
        match envelope {
            ProviderIpcError::Transport { provider, .. } => {
                // Cutting the two-byte `é` would split it, so the envelope
                // keeps the 63 ASCII bytes.
                assert_eq!(provider, "a".repeat(MAX_PROVIDER_ID_LEN - 1));
            }
            other => panic!("unexpected envelope: {other:?}"),
        }
        // Free-text reasons scrub to printable ASCII and truncate.
        let hostile = format!("boom\n\u{1b}[2J{}", "x".repeat(MAX_IPC_REASON_BYTES * 2));
        let envelope = ProviderIpcError::from_error(&ProviderError::Unknown {
            provider: "p".to_owned(),
            reason: hostile,
        });
        match envelope {
            ProviderIpcError::Unknown { reason, .. } => {
                assert_eq!(reason.len(), MAX_IPC_REASON_BYTES);
                assert!(
                    reason.chars().all(|c| c.is_ascii_graphic() || c == ' '),
                    "unscrubbed reason: {reason:?}"
                );
                assert!(reason.starts_with("boom??[2J"));
            }
            other => panic!("unexpected envelope: {other:?}"),
        }
        // Sampling refusals echo no values.
        assert_eq!(
            ProviderIpcError::from_error(&ProviderError::InvalidSampling),
            ProviderIpcError::InvalidSampling
        );
        assert_eq!(
            ProviderIpcError::InvalidSampling.to_string(),
            "invalid sampling parameters"
        );
        // The Unknown display matches the source taxonomy exactly.
        assert_eq!(
            ProviderIpcError::Unknown {
                provider: "p".to_owned(),
                reason: "ack lost".to_owned(),
            }
            .to_string(),
            "provider p effect unknown: ack lost"
        );
    }
}

#[cfg(test)]
mod scope_filter_tests {
    use super::*;

    fn text_registration(provider: &str, name: &str) -> ModelRegistration {
        ModelRegistration::new(provider, name, vec![ModelCapability::Text], 4_096, 1, 2)
            .expect("valid registration")
    }

    fn fixture_registry() -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        registry
            .register(text_registration("bitty-b", "chat"))
            .expect("capacity");
        registry
            .register(
                ModelRegistration::new(
                    "bitty-a",
                    "vision",
                    vec![ModelCapability::Text, ModelCapability::ImageInput],
                    32_768,
                    3,
                    4,
                )
                .expect("valid registration"),
            )
            .expect("capacity");
        registry
    }

    #[test]
    fn scope_ceiling_matches_documented_bound() {
        assert_eq!(MAX_CALLER_SCOPE_LEN, 64);
    }

    #[test]
    fn unfiltered_path_preserved_as_is() {
        let registry = fixture_registry();
        // The mirrored unfiltered snapshot keeps its S1/S2 shape: count,
        // lexicographic order, and detached clones.
        let unfiltered = ListModelsResponse::from_registry(&registry);
        assert_eq!(unfiltered.len(), 2);
        let order: Vec<(&str, &str)> = unfiltered
            .models
            .iter()
            .map(|item| (item.provider_id.as_str(), item.name.as_str()))
            .collect();
        assert_eq!(order, vec![("bitty-a", "vision"), ("bitty-b", "chat")]);
        // The S2 unfiltered request shape still passes, and the reserved
        // scoped shape is still denied: S3 adds alongside, never rewrites.
        assert!(ListModelsRequest::unfiltered().check_scope().is_ok());
        assert!(ListModelsRequest::default().check_scope().is_ok());
        assert_eq!(
            ListModelsRequest::scoped("provider/bitty-a")
                .check_scope()
                .expect_err("reserved scoped shape stays denied"),
            IpcEnvelopeError::ReservedScopeDenied
        );
        // Later registry state cannot leak into the unfiltered snapshot.
        let mut registry = registry;
        registry
            .register(text_registration("bitty-c", "fresh"))
            .expect("capacity");
        assert_eq!(unfiltered.len(), 2);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn filtered_is_subset_never_wider_same_now_ms() {
        let registry = fixture_registry();
        let now_ms = 1_000;
        let unfiltered = ListModelsResponse::from_registry(&registry);
        let echo = CallerScopeEcho::new("provider/bitty-a", now_ms);
        let filtered = registry
            .list_models_filtered(&echo)
            .expect("known provider scope filters");
        // Intersection-only narrowing: one entry, the provider's own.
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.models[0].provider_id, "bitty-a");
        assert_eq!(filtered.models[0].name, "vision");
        // Subset of the same-`now_ms` unfiltered snapshot, never wider.
        assert!(filtered.len() <= unfiltered.len());
        for item in &filtered.models {
            assert!(
                unfiltered
                    .models
                    .iter()
                    .any(|kept| kept.provider_id == item.provider_id
                        && kept.name == item.name
                        && kept.capabilities == item.capabilities),
                "filtered entry must exist in unfiltered: {item:?}"
            );
        }
        // The other provider's scope is disjoint but still a subset.
        let other = registry
            .list_models_filtered(&CallerScopeEcho::new("provider/bitty-b", now_ms))
            .expect("second provider filters");
        assert_eq!(other.len(), 1);
        assert_eq!(other.models[0].name, "chat");
        // Detached: later registry state cannot leak into a filtered view.
        let mut registry = registry;
        registry
            .register(text_registration("bitty-a", "fresh"))
            .expect("capacity");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn unknown_scope_refuses_fail_closed_names_only() {
        let registry = fixture_registry();
        // Well-formed `provider/` scope with no entries in this snapshot.
        let echo = CallerScopeEcho::new("provider/bitty-gone", 7);
        let error = registry
            .list_models_filtered(&echo)
            .expect_err("unknown provider must refuse");
        match &error {
            ScopeFilterError::UnknownScope {
                requested,
                available,
            } => {
                assert_eq!(requested, "provider/bitty-gone");
                assert_eq!(
                    *available,
                    vec!["bitty-a/vision".to_owned(), "bitty-b/chat".to_owned(),]
                );
            }
            other => panic!("unknown must be UnknownScope: {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "unknown caller scope: provider/bitty-gone (2 known entries)"
        );
        // Well-formed but non-provider prefix is unknown, never invalid.
        let error = registry
            .list_models_filtered(&CallerScopeEcho::new("team-a/read", 7))
            .expect_err("non-provider prefix must refuse as unknown");
        assert!(matches!(error, ScopeFilterError::UnknownScope { .. }));
        // No partial state and no secrets in the refusal.
        assert_eq!(registry.len(), 2);
        let rendered = format!("{error:?}");
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("key"));
    }

    #[test]
    fn empty_scope_is_invalid() {
        let registry = fixture_registry();
        let error = registry
            .list_models_filtered(&CallerScopeEcho::new("", 1))
            .expect_err("empty scope must refuse");
        assert_eq!(
            error,
            ScopeFilterError::InvalidScope {
                scope: String::new(),
                reason: "empty",
            }
        );
        assert_eq!(error.to_string(), "invalid caller scope (empty): ");
        assert!(matches!(
            validate_caller_scope(""),
            Err(ScopeFilterError::InvalidScope {
                reason: "empty",
                ..
            })
        ));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn malformed_scopes_are_invalid_never_unknown() {
        // Overlong stays invalid with a bounded echo.
        let overlong = format!("provider/{}", "a".repeat(MAX_CALLER_SCOPE_LEN));
        assert!(overlong.len() > MAX_CALLER_SCOPE_LEN);
        match validate_caller_scope(&overlong).expect_err("overlong must fail") {
            ScopeFilterError::InvalidScope { scope, reason } => {
                assert_eq!(reason, "overlong");
                assert_eq!(scope.len(), MAX_CALLER_SCOPE_LEN);
            }
            other => panic!("overlong must be InvalidScope: {other:?}"),
        }
        // Grammar violations: missing slash, edges, doubling, charset.
        for bad in [
            "bitty-a",
            "/provider/bitty-a",
            "provider/bitty-a/",
            "provider//bitty-a",
            "Provider/bitty-a",
            "provider/Bad",
            "provider/has space",
            "provider/has.dot",
            "provider/_leading",
        ] {
            assert!(
                matches!(
                    validate_caller_scope(bad),
                    Err(ScopeFilterError::InvalidScope {
                        reason: "grammar",
                        ..
                    })
                ),
                "must be grammar-invalid: {bad:?}"
            );
            let registry = fixture_registry();
            assert!(
                matches!(
                    registry.list_models_filtered(&CallerScopeEcho::new(bad, 1)),
                    Err(ScopeFilterError::InvalidScope { .. })
                ),
                "filter must stay invalid (never unknown): {bad:?}"
            );
        }
    }

    #[test]
    fn echo_is_observation_not_authority() {
        let registry = fixture_registry();
        // The same scope at different dispatch timestamps yields the same
        // subset: `now_ms` is audit observation, never filter authority.
        let first = registry
            .list_models_filtered(&CallerScopeEcho::new("provider/bitty-a", 1_000))
            .expect("filters");
        let second = registry
            .list_models_filtered(&CallerScopeEcho::new("provider/bitty-a", 2_000))
            .expect("filters at another now_ms");
        assert_eq!(first, second);
        // Constructing an echo alone grants nothing: no host call means no
        // view, and an unknown echo still refuses host-side.
        let echo = CallerScopeEcho::new("provider/bitty-gone", 1_000);
        assert!(registry.list_models_filtered(&echo).is_err());
        // The returned view is detached: mutating the echo afterwards cannot
        // widen a previously enforced view.
        let mut echo = CallerScopeEcho::new("provider/bitty-a", 1_000);
        let view = registry
            .list_models_filtered(&echo)
            .expect("known scope filters");
        echo.scope = "provider/bitty-b".to_owned();
        assert_eq!(view.len(), 1);
        assert_eq!(view.models[0].provider_id, "bitty-a");
    }
}

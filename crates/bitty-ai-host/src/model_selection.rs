//! Deterministic provider/model selection policy, host-side pure-type mirror
//! (AI-0272, Phase 2 S1 per AI-0264 PX-1153).
//!
//! Pure-type mirror of the model-selection seam owned by the runtime
//! (`bitty-ai-runtime` selection policy, AI-0030): the same deterministic
//! [`ProviderRegistry`] over model descriptors with capability sets,
//! capability-subset matching (never name matching alone), semantic alias
//! resolution held as data, an ordered fallback chain with a deterministic
//! tie-break, and a typed fallback policy over the mirrored
//! [`ProviderError`].
//!
//! The host crate is std-only with no dependency on the runtime crate, so the
//! vocabulary this seam needs but does not own is mirrored locally as data:
//! [`ModelCapability`], [`ModelDescriptor`], [`ProviderError`] (all variants
//! plus `Display`), and the provider-id shape check. No `ModelProvider`
//! trait, no `list_models`/`complete` wiring, and no execution of any kind
//! moves here: selection returns [`SelectedModel`] snapshots
//! (descriptors/handles) and nothing else.
//!
//! ## Gate order (tool-transport-R2 precedent)
//!
//! Every selection traverses the same ordered gates; a refusal at any gate
//! fails closed with a typed [`SelectionError`] and no partial state:
//!
//! ```text
//! alias resolution (unknown alias fails; stale candidates skipped in declared
//!   order; an alias is exclusive and a full alias miss fails closed)
//!   -> capability-subset match (empty requirement set refused: never by name alone)
//!   -> context-window minimum filter (unknown `0` satisfies no minimum)
//!   -> cost-ceiling filter (effective weights must fit; `0` counts as baseline `1`)
//!   -> caller-scope intersection (Phase 2 S3 per CTX-0149: empty echo is
//!      unfiltered; otherwise the candidate must grant every requested scope;
//!      unknown scopes fail closed with `NoCandidate`)
//!   -> ordered fallback chain (alias order with an alias, else provider/model
//!      lexicographic; non-alias entries are never appended to an alias chain)
//!   -> per-candidate execution stays runtime-side via its `ModelProvider`
//!      trait (not mirrored here)
//!   -> typed-error fallback: advance on transport/rate-limit/unavailable/timeout,
//!      stop on auth/capability/budget/caller errors, reconcile on Unknown
//! ```
//!
//! ## What this module is not
//!
//! - No provider trait, no `list_models`/`complete` wiring, no execution.
//!   Selection returns snapshots; execution stays on the runtime side.
//!   Provider-side `list_models`/`complete` scope filtering stays owned by
//!   AI-0290 and is never re-mirrored here.
//! - MP-4 caller-scope filter (Phase 2 S3 per CTX-0149) is selection-side
//!   only via [`CallerScope`], [`ScopePolicy`], and [`select_filtered`]
//!   below; there is still no closed-vocabulary enforcement: the capability
//!   mapping prose on [`SelectRequest`] is routing vocabulary only, never a
//!   rejection rule.
//! - Context-window and cost metadata are data for routing and budget
//!   decisions, not enforcement by themselves. Cost weights are relative
//!   routing units, never currency.
//! - No sampling matrix, no stream/cancel operations, no `LocalEndpoint`, no
//!   IPC transport, no threads, no filesystem, no network, no secrets.
//! - Alias tables are data, not authority: resolving an alias yields ordered
//!   candidates that still pass every gate, the alias is exclusive (a full
//!   miss is [`SelectionError::NoCandidate`], never a fall-through to
//!   non-alias entries). An alias grants no execution authority.
//! - [`ProviderError::Unknown`] never advances the chain. Reconcile (status
//!   inspection or user direction) before retry.
//!
//! All behavior is deterministic from the registry contents plus the request:
//! no wall clock, no threads, no filesystem, no secrets.

//! ## Canonical split (AI-0287)
//!
//! The S1 registry/selection vocabulary this seam needs ([`ModelCapability`],
//! [`ModelDescriptor`], [`TerminalModelMetadata`], [`ProviderError`],
//! [`SelectionError`], [`ModelRegistration`], [`ModelRef`],
//! [`RegisteredModel`], [`ModelAlias`], [`ProviderRegistry`],
//! [`SelectRequest`], [`SelectedModel`], [`estimate_cost`],
//! [`fallback_directive`], [`validate_model_name`], [`validate_provider_id`],
//! and the `MAX_*` bounds) is canonically owned by the
//! [`provider_registry`](crate::provider_registry) module and only
//! re-exported here, so both paths name the same items with no logic
//! duplication. What this module owns is the selection-side S2 envelope
//! layer below, the selection-side S3 caller-scope filter
//! ([`CallerScope`], [`ScopePolicy`], [`select_filtered`] plus the
//! `MAX_CALLER_SCOPE_LEN`, `MAX_CALLER_SCOPES`, `MAX_SCOPED_MODELS`, and
//! `MAX_SCOPES_PER_MODEL` bounds) per CTX-0149, plus the gate-order contract
//! above.
//!
pub use crate::provider_registry::{
    FallbackDirective, MAX_ALIAS_CANDIDATES, MAX_ALIASES, MAX_MODEL_NAME_LEN, MAX_PROVIDER_ID_LEN,
    MAX_REGISTERED_MODELS, ModelAlias, ModelCapability, ModelDescriptor, ModelRef,
    ModelRegistration, ProviderError, ProviderRegistry, RegisteredModel, SelectRequest,
    SelectedModel, SelectionError, TerminalModelMetadata, estimate_cost, fallback_directive,
    validate_model_name, validate_provider_id,
};

// ── Phase 2 S2 scoped-IPC envelope types (no transport, AI-0279) ──
//
// Scoped-IPC request/response shapes for the model-selection seam so a
// future host-side caller can cross the boundary without touching the
// registry directly. Types-only, `std`-only, no behavior: no socket, no
// transport, no thread, no clock, no network, no filesystem, no dependency
// change (`Cargo.toml` stays `std`-only), no new byte bounds (the S1
// [`MAX_MODEL_NAME_LEN`], [`MAX_REGISTERED_MODELS`], [`MAX_ALIASES`],
// [`MAX_ALIAS_CANDIDATES`], and `MAX_PROVIDER_ID_LEN` values apply
// unchanged), no version field on any envelope, and no selection execution.
// Every envelope reuses the S1 mirror above ([`ModelCapability`],
// [`SelectRequest`], [`SelectedModel`], [`SelectionError`],
// [`ProviderError`], [`FallbackDirective`], [`estimate_cost`],
// [`fallback_directive`]) and introduces no re-mirrored vocabulary.
//
// Provider-side `list_models`/`complete` envelopes stay owned by AI-0278:
// they are named here only to mark the boundary and are not mirrored in
// this module.
//
// Gate and mapping summary (re-stated, not re-implemented): an envelope
// request narrows only (capability subset, alias-exclusive, window minimum
// where unknown `0` satisfies nothing, cost ceiling where effective `0`
// counts as baseline `1`); an envelope response carries either the detached
// [`SelectedModel`] chain (primary is element `0`) or the typed
// [`SelectionError`], with a full alias miss preserved as
// [`SelectionError::NoCandidate`] (never a fall-through); the fallback
// envelope maps [`ProviderError::Transport`], [`ProviderError::RateLimited`],
// [`ProviderError::ModelUnavailable`], and [`ProviderError::Timeout`] to
// [`FallbackDirective::Advance`] and every other variant — including
// [`ProviderError::Unknown`] (reconcile before retry, never blind advance)
// — to [`FallbackDirective::Stop`]; the cost envelope echoes its four
// [`estimate_cost`] inputs with `effective(0) = 1` in relative routing units
// (never currency, saturating on overflow).

/// Scoped-IPC selection request envelope: narrowing-only filters over the
/// S1 [`SelectRequest`] shape.
///
/// Every field is a filter; adding a filter never widens the result.
/// `required` must be non-empty (selection never matches by name or alias
/// alone); `alias` is exclusive when set (only declared candidates are
/// eligible, each still passing every other gate in declared order, stale
/// candidates skipped, full miss is [`SelectionError::NoCandidate`]);
/// `min_context_window_tokens` refuses unknown (`0`) windows;
/// `max_cost_weight` compares effective weights (`0` counts as baseline
/// `1`, so `Some(0)` admits nothing). No version field, no transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectEnvelopeRequest {
    /// Required capabilities: every candidate must advertise all of them.
    pub required: Vec<ModelCapability>,
    /// Optional exclusive semantic alias.
    pub alias: Option<String>,
    /// Optional minimum context window in tokens.
    pub min_context_window_tokens: Option<u32>,
    /// Optional cost ceiling over effective weights.
    pub max_cost_weight: Option<u32>,
}

impl SelectEnvelopeRequest {
    /// Build an envelope request from its four narrowing fields. No
    /// validation runs here: shape checks stay at registration/selection
    /// time via the S1 vocabulary.
    #[must_use]
    pub fn new(
        required: Vec<ModelCapability>,
        alias: Option<String>,
        min_context_window_tokens: Option<u32>,
        max_cost_weight: Option<u32>,
    ) -> Self {
        Self {
            required,
            alias,
            min_context_window_tokens,
            max_cost_weight,
        }
    }

    /// Build a capability-only envelope request.
    #[must_use]
    pub fn capabilities(required: Vec<ModelCapability>) -> Self {
        Self {
            required,
            alias: None,
            min_context_window_tokens: None,
            max_cost_weight: None,
        }
    }

    /// View this envelope as the S1 [`SelectRequest`] it narrows with.
    /// Pure data mapping; selection itself stays on [`ProviderRegistry`].
    #[must_use]
    pub fn to_select_request(&self) -> SelectRequest {
        SelectRequest {
            required: self.required.clone(),
            alias: self.alias.clone(),
            min_context_window_tokens: self.min_context_window_tokens,
            max_cost_weight: self.max_cost_weight,
        }
    }

    /// Wrap an S1 [`SelectRequest`] as an envelope. Pure data mapping.
    #[must_use]
    pub fn from_select_request(request: &SelectRequest) -> Self {
        Self {
            required: request.required.clone(),
            alias: request.alias.clone(),
            min_context_window_tokens: request.min_context_window_tokens,
            max_cost_weight: request.max_cost_weight,
        }
    }

    /// Human-readable requirement summary, mirroring
    /// [`SelectRequest::describe`] for the [`SelectionError::NoCandidate`]
    /// detail.
    #[must_use]
    pub fn describe(&self) -> String {
        self.to_select_request().describe()
    }
}

/// Scoped-IPC selection response envelope: either the detached
/// [`SelectedModel`] chain or the typed [`SelectionError`].
///
/// `Selected` carries the full ordered fallback chain (primary is element
/// `0`; every element independently satisfies the request; snapshots are
/// detached so later registry changes do not leak in). `Failed` carries the
/// selection refusal, with a full alias miss preserved as
/// [`SelectionError::NoCandidate`] (never a fall-through to non-alias
/// entries). No version field, no transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectEnvelopeResponse {
    /// Ordered detached snapshot chain.
    Selected {
        /// Every candidate that satisfies the envelope request.
        chain: Vec<SelectedModel>,
    },
    /// Typed selection refusal.
    Failed {
        /// Why nothing satisfies the envelope request.
        error: SelectionError,
    },
}

impl SelectEnvelopeResponse {
    /// Wrap a [`ProviderRegistry::select`](ProviderRegistry::select) outcome
    /// as an envelope. Pure data mapping.
    #[must_use]
    pub fn from_result(result: Result<Vec<SelectedModel>, SelectionError>) -> Self {
        match result {
            Ok(chain) => Self::Selected { chain },
            Err(error) => Self::Failed { error },
        }
    }

    /// Unwrap an envelope back to the selection outcome. Pure data mapping.
    pub fn into_result(self) -> Result<Vec<SelectedModel>, SelectionError> {
        match self {
            Self::Selected { chain } => Ok(chain),
            Self::Failed { error } => Err(error),
        }
    }

    /// Whether this envelope carries a selected chain.
    #[must_use]
    pub fn is_selected(&self) -> bool {
        matches!(self, Self::Selected { .. })
    }
}

/// Scoped-IPC fallback request envelope: echoes the [`ProviderError`] to
/// classify. No version field, no transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackDirectiveEnvelopeRequest {
    /// Provider failure to classify.
    pub error: ProviderError,
}

impl FallbackDirectiveEnvelopeRequest {
    /// Build a fallback request echoing `error`. No classification runs here.
    #[must_use]
    pub fn new(error: ProviderError) -> Self {
        Self { error }
    }

    /// Classify the echoed error via the S1 [`fallback_directive`] mapping
    /// (`Advance` on transport/rate-limit/unavailable/timeout, `Stop`
    /// otherwise including [`ProviderError::Unknown`]). Pure, total, no I/O.
    #[must_use]
    pub fn resolve(&self) -> FallbackDirectiveEnvelopeResponse {
        FallbackDirectiveEnvelopeResponse {
            directive: fallback_directive(&self.error),
        }
    }
}

/// Scoped-IPC fallback response envelope: the [`FallbackDirective`] for the
/// echoed error. No version field, no transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackDirectiveEnvelopeResponse {
    /// Advance to the next chain element or stop and reconcile.
    pub directive: FallbackDirective,
}

impl FallbackDirectiveEnvelopeResponse {
    /// Build a fallback response carrying `directive`.
    #[must_use]
    pub fn new(directive: FallbackDirective) -> Self {
        Self { directive }
    }
}

/// Scoped-IPC cost-estimation request envelope: echoes the four
/// [`estimate_cost`] inputs. No version field, no transport.
///
/// `input_weight`/`output_weight` are relative routing units (never
/// currency); `0` means uncalibrated and counts as the baseline weight `1`
/// (`effective(0) = 1`), so an unset weight is never free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstimateCostEnvelopeRequest {
    /// Observed input tokens.
    pub input_tokens: u64,
    /// Observed output tokens.
    pub output_tokens: u64,
    /// Relative input cost weight (`0` = uncalibrated, counts as `1`).
    pub input_weight: u32,
    /// Relative output cost weight (`0` = uncalibrated, counts as `1`).
    pub output_weight: u32,
}

impl EstimateCostEnvelopeRequest {
    /// Build a cost request echoing its four [`estimate_cost`] inputs.
    #[must_use]
    pub fn new(
        input_tokens: u64,
        output_tokens: u64,
        input_weight: u32,
        output_weight: u32,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            input_weight,
            output_weight,
        }
    }

    /// Estimate routing cost for the echoed inputs via the S1
    /// [`estimate_cost`] rule (relative units, saturating, `effective(0) =
    /// 1`). Pure, no I/O, no clock. The response echoes the inputs.
    #[must_use]
    pub fn estimate(&self) -> EstimateCostEnvelopeResponse {
        EstimateCostEnvelopeResponse {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            input_weight: self.input_weight,
            output_weight: self.output_weight,
            cost: estimate_cost(
                self.input_tokens,
                self.output_tokens,
                self.input_weight,
                self.output_weight,
            ),
        }
    }
}

/// Scoped-IPC cost-estimation response envelope: echoes the request inputs
/// plus the estimated cost in relative routing units (never currency).
/// No version field, no transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstimateCostEnvelopeResponse {
    /// Echoed input tokens.
    pub input_tokens: u64,
    /// Echoed output tokens.
    pub output_tokens: u64,
    /// Echoed input weight.
    pub input_weight: u32,
    /// Echoed output weight.
    pub output_weight: u32,
    /// Estimated cost (`effective(0) = 1`, saturating).
    pub cost: u64,
}

impl EstimateCostEnvelopeResponse {
    /// Whether the echoed cost matches a fresh [`estimate_cost`] evaluation
    /// over the echoed inputs. Pure.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.cost
            == estimate_cost(
                self.input_tokens,
                self.output_tokens,
                self.input_weight,
                self.output_weight,
            )
    }
}

// ── Phase 2 S3 caller-scope filter (no transport, AI-0291) ──
//
// Selection-side caller-scope filter per the CTX-0149 candidate in
// `providers/provider-plugin-boundary.md` (MP-4 section): scope vocabulary
// plus intersection-only narrowing with deny-by-default, enforced host-side
// at dispatch before selection and provider I/O. The request envelope
// carries a scope echo for audit while the host — never the adapter —
// applies the filter; the echo is observation data, not authority.
//
// What is reused, not re-mirrored: [`ModelCapability`], [`SelectRequest`],
// [`SelectedModel`], [`SelectionError`], [`ProviderError`],
// [`FallbackDirective`], [`ModelRef`], [`ProviderRegistry`],
// [`SelectEnvelopeRequest`], [`estimate_cost`], [`fallback_directive`], the
// S1 `matches` gate (capability-subset plus window minimum where unknown
// `0` satisfies nothing plus cost ceiling where effective `0` counts as
// baseline `1`), and the S1 `MAX_*` discipline. What is new here:
// [`CallerScope`], [`ScopePolicy`], [`select_filtered`], and the
// `MAX_CALLER_SCOPE_LEN`, `MAX_CALLER_SCOPES`, `MAX_SCOPED_MODELS`, and
// `MAX_SCOPES_PER_MODEL` bounds below. What stays out: provider-side
// `list_models`/`complete` envelopes (AI-0290 owns those and they appear
// here by name only), transport of any kind, any `version` field, threads,
// clock reads, filesystem or network access, credentials or secrets, and any
// `Cargo.toml` change (`std`-only).
//
// Scope vocabulary (candidate-exact where the candidate pins it, minimal
// concrete choice where the candidate leaves it open): caller identity
// reuses the established caller triple shape (protocol, instance, session)
// without a new authentication primitive; authentication, consent, and
// budget gates stay with the existing IPC and `ai.provider` controls. A
// scope string is bounded ([`MAX_CALLER_SCOPE_LEN`] bytes), printable-ASCII
// (the closed charset below is a printable subset), slash-separated
// (`segment(/segment)*` with no empty segment, hence no leading/trailing
// `/` and no `//`), closed-charset (each segment
// `[a-z0-9][a-z0-9_.-]*`); an empty, overlong, or out-of-grammar value is
// invalid. Registry-field placement was open: grants live selection-side in
// [`ScopePolicy`] so the S1 registry stays untouched for AI-0290.
//
// Filter semantics (intersection-only narrowing with deny-by-default): the
// unfiltered registry snapshot stays preserved as-is (an empty scope echo
// applies no filtering); a filtered view is a subset of that snapshot for
// the same request and never wider. A candidate passes the scope gate iff
// its granted scopes contain every requested scope (`requested ⊆ granted`),
// so adding a requested scope never widens visibility. A model with no
// grant entry is denied whenever filtering (deny-by-default) and allowed
// only when unfiltered. An unknown scope (granted by no candidate) fails
// closed with [`SelectionError::NoCandidate`] carrying names only
// (requirement summary plus requested scope names), never credential values
// or raw vendor text.
//
// Gate and mapping summary (re-stated, not re-implemented): [`select_filtered`]
// traverses alias resolution (unknown alias fails; stale candidates skipped
// in declared order; the alias is exclusive and a full alias miss fails
// closed) -> capability-subset match (empty requirement set refused) ->
// context-window minimum (unknown `0` satisfies no minimum) -> cost ceiling
// (effective `0` counts as baseline `1`, so `Some(0)` admits nothing) ->
// scope intersection (`requested ⊆ granted`; empty echo is unfiltered) ->
// ordered chain (alias-declared order with an alias, else provider/model
// lexicographic; non-alias entries are never appended to an alias chain);
// the fallback mapping is unchanged ([`fallback_directive`], with
// [`ProviderError::Unknown`] mapping to [`FallbackDirective::Stop`]).
// ---------------------------------------------------------------------------

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Maximum caller-scope string length in bytes (S3, MP-4 candidate).
///
/// Mirrors the S1 bound discipline ([`MAX_PROVIDER_ID_LEN`] is also 64):
/// a bounded, printable-ASCII, slash-separated name with a stated ceiling;
/// an empty, overlong, or out-of-grammar value stays invalid.
pub const MAX_CALLER_SCOPE_LEN: usize = 64;
/// Maximum caller scopes per filtered selection (S3).
///
/// Mirrors [`MAX_ALIAS_CANDIDATES`]: an over-limit echo fails closed with
/// [`SelectionError::TooManyCandidates`] (fixed `scope-echo` marker, counts
/// only) so selection bounds CPU and refusal detail.
pub const MAX_CALLER_SCOPES: usize = 8;
/// Maximum scope-granted models in a [`ScopePolicy`] (S3).
///
/// Mirrors [`MAX_REGISTERED_MODELS`]: a full policy refuses with
/// [`SelectionError::RegistryFull`]. Every refusal leaves the policy
/// unchanged.
pub const MAX_SCOPED_MODELS: usize = 64;
/// Maximum granted scopes per model in a [`ScopePolicy`] (S3).
///
/// Mirrors [`MAX_ALIAS_CANDIDATES`]: an over-limit grant fails closed with
/// [`SelectionError::TooManyCandidates`] carrying the `provider/model` key
/// plus limit and actual counts (names only).
pub const MAX_SCOPES_PER_MODEL: usize = 8;

/// One validated caller scope (S3, MP-4 candidate).
///
/// A bounded, printable-ASCII, slash-separated name with a closed charset:
/// `segment(/segment)*` where each segment is `[a-z0-9][a-z0-9_.-]*` (no
/// empty segment, hence no leading/trailing `/` and no `//`), at most
/// [`MAX_CALLER_SCOPE_LEN`] bytes. Construction validates; the echo is
/// observation data for audit, not authority (the host applies the filter
/// in [`select_filtered`] before selection and provider I/O).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallerScope {
    /// Validated scope string (never empty, overlong, or out-of-grammar).
    scope: String,
}

impl CallerScope {
    /// Validate a scope string.
    ///
    /// Fail-closed: an empty, overlong, or out-of-grammar value is refused
    /// with [`SelectionError::InvalidName`] (carries the rejected name only,
    /// never secrets).
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError::InvalidName`] when the shape is violated.
    pub fn new(scope: impl Into<String>) -> Result<Self, SelectionError> {
        let scope = scope.into();
        let valid = !scope.is_empty()
            && scope.len() <= MAX_CALLER_SCOPE_LEN
            && scope.split('/').all(|segment| {
                let mut bytes = segment.bytes();
                match bytes.next() {
                    Some(b) if b.is_ascii_lowercase() || b.is_ascii_digit() => {}
                    _ => return false,
                }
                bytes.all(|b| {
                    b.is_ascii_lowercase()
                        || b.is_ascii_digit()
                        || b == b'_'
                        || b == b'-'
                        || b == b'.'
                })
            });
        if valid {
            Ok(Self { scope })
        } else {
            Err(SelectionError::InvalidName { name: scope })
        }
    }

    /// Validated scope string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.scope
    }

    /// Consume into the validated scope string.
    #[must_use]
    pub fn into_string(self) -> String {
        self.scope
    }
}

impl Display for CallerScope {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.scope)
    }
}

/// One scope-granted model entry (S3): the `(provider, model)` key plus its
/// granted caller scopes. Private: construction and inspection go through
/// [`ScopePolicy`] so the fail-closed discipline (duplicate/conflict before
/// capacity, no state change on any refusal) holds by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ScopedModel {
    /// Owning provider id (`MP-2` shape, validated at grant time).
    provider_id: String,
    /// Registry-known model name (validated at grant time).
    model: String,
    /// Granted caller scopes (at most [`MAX_SCOPES_PER_MODEL`]; empty means
    /// visible only when unfiltered, denied whenever filtering).
    scopes: Vec<CallerScope>,
}

/// Selection-side caller-scope grants (S3, MP-4 candidate).
///
/// A sidecar table mapping each `(provider, model)` key to the caller scopes
/// that may see it. The S1 [`ProviderRegistry`] stays untouched (AI-0290
/// owns the provider side); this policy holds only the scope placements
/// whose exact registry-field placement the candidate left open. Entries
/// keep grant order internally; selection output order is always derived
/// (alias order, then provider/model lexicographic), never grant order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScopePolicy {
    /// Granted models in grant order.
    grants: Vec<ScopedModel>,
}

impl ScopePolicy {
    /// Construct an empty policy (no grants: every filtered selection misses
    /// until grants arrive; unfiltered selections still pass through).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grant caller scopes to one `(provider, model)` key.
    ///
    /// Fail-closed, mirroring the registry discipline: an exact re-grant is
    /// refused with [`SelectionError::DuplicateModel`]; the same key with
    /// different scopes is refused with [`SelectionError::ConflictingModel`]
    /// (no shadowing, no merge: the kept entry is untouched); malformed keys
    /// are refused with their typed error via [`ModelRef::new`]; an
    /// over-limit scope list with [`SelectionError::TooManyCandidates`]
    /// (alias carries the `provider/model` key plus limit and actual);
    /// a full policy with [`SelectionError::RegistryFull`]. Every refusal
    /// leaves the policy unchanged. Duplicate/conflict checks run before the
    /// capacity check, so a duplicate reports as a duplicate even when the
    /// policy is full. An empty scope list is allowed: it is visible only
    /// when unfiltered and denied whenever filtering (deny-by-default).
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] as described above.
    pub fn grant(
        &mut self,
        provider_id: impl Into<String>,
        model: impl Into<String>,
        scopes: Vec<CallerScope>,
    ) -> Result<(), SelectionError> {
        let key = ModelRef::new(provider_id, model)?;
        if let Some(kept) = self
            .grants
            .iter()
            .find(|kept| kept.provider_id == key.provider_id && kept.model == key.model)
        {
            if kept.scopes == scopes {
                return Err(SelectionError::DuplicateModel {
                    provider: key.provider_id,
                    model: key.model,
                });
            }
            return Err(SelectionError::ConflictingModel {
                provider: key.provider_id,
                model: key.model,
            });
        }
        if scopes.len() > MAX_SCOPES_PER_MODEL {
            return Err(SelectionError::TooManyCandidates {
                alias: format!("{}/{}", key.provider_id, key.model),
                limit: MAX_SCOPES_PER_MODEL,
                actual: scopes.len(),
            });
        }
        if self.grants.len() >= MAX_SCOPED_MODELS {
            return Err(SelectionError::RegistryFull {
                limit: MAX_SCOPED_MODELS,
            });
        }
        self.grants.push(ScopedModel {
            provider_id: key.provider_id,
            model: key.model,
            scopes,
        });
        Ok(())
    }

    /// Granted model count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Whether the policy holds no grant.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }

    /// Granted scopes for one `(provider, model)` key, if any.
    #[must_use]
    pub fn scopes_for(&self, provider_id: &str, model: &str) -> Option<&[CallerScope]> {
        self.grants
            .iter()
            .find(|kept| kept.provider_id == provider_id && kept.model == model)
            .map(|kept| kept.scopes.as_slice())
    }

    /// Whether one `(provider, model)` key is visible under `scope_echo`.
    ///
    /// An empty echo is unfiltered (always visible: the snapshot stays
    /// preserved as-is). Otherwise the key must have a grant entry whose
    /// scopes contain every requested scope (`requested ⊆ granted`); a key
    /// with no entry, or missing any requested scope, is denied
    /// (deny-by-default; unknown scopes fail closed downstream as
    /// [`SelectionError::NoCandidate`]). Pure, deterministic, no I/O.
    #[must_use]
    pub fn is_visible(&self, provider_id: &str, model: &str, scope_echo: &[CallerScope]) -> bool {
        if scope_echo.is_empty() {
            return true;
        }
        match self.scopes_for(provider_id, model) {
            Some(granted) => scope_echo.iter().all(|want| granted.contains(want)),
            None => false,
        }
    }
}

/// Select the ordered fallback chain for `request` under `scope_echo`.
///
/// Gate order: empty requirement refused ([`SelectionError::EmptyRequirement`];
/// never by name or alias alone) -> over-limit scope echo refused
/// ([`SelectionError::TooManyCandidates`] with the fixed `scope-echo`
/// marker) -> alias resolution (unknown alias fails with
/// [`SelectionError::UnknownAlias`]; stale candidates skipped in declared
/// order; the alias is exclusive and a full alias miss fails closed) ->
/// capability-subset match plus context-window minimum (unknown `0`
/// satisfies no minimum) plus cost ceiling (effective `0` counts as baseline
/// `1`, via [`SelectRequest::matches`]) -> scope intersection
/// (`requested ⊆ granted` per [`ScopePolicy::is_visible`]; empty echo is
/// unfiltered) -> ordered chain (alias-declared order with an alias, else
/// provider/model lexicographic; non-alias entries are never appended to an
/// alias chain). The primary is element `0`; every element independently
/// satisfies the request plus the scope echo, so advancing on a typed
/// transient error never weakens the requirement set. A full miss at any
/// gate after alias resolution fails closed with
/// [`SelectionError::NoCandidate`] whose detail carries the requirement
/// summary plus the requested scope names only (never secrets or vendor
/// text). The fallback mapping is unchanged: classify per-candidate
/// failures separately with [`fallback_directive`] (`Advance` on
/// transport/rate-limit/unavailable/timeout, `Stop` otherwise including
/// [`ProviderError::Unknown`).
///
/// All behavior is deterministic from the registry contents plus the
/// request, echo, and policy: no wall clock, no threads, no filesystem, no
/// secrets. Selection returns detached snapshots; execution stays
/// runtime-side.
///
/// # Errors
///
/// Returns [`SelectionError::EmptyRequirement`] when no capability is
/// required, [`SelectionError::TooManyCandidates`] when the echo exceeds
/// [`MAX_CALLER_SCOPES`], [`SelectionError::UnknownAlias`] for an
/// unregistered alias, or [`SelectionError::NoCandidate`] when nothing
/// satisfies the request plus the scope echo (including a fully missed or
/// fully scope-denied alias).
pub fn select_filtered(
    registry: &ProviderRegistry,
    request: &SelectEnvelopeRequest,
    scope_echo: &[CallerScope],
    policy: &ScopePolicy,
) -> Result<Vec<SelectedModel>, SelectionError> {
    if request.required.is_empty() {
        return Err(SelectionError::EmptyRequirement);
    }
    if scope_echo.len() > MAX_CALLER_SCOPES {
        return Err(SelectionError::TooManyCandidates {
            alias: "scope-echo".to_owned(),
            limit: MAX_CALLER_SCOPES,
            actual: scope_echo.len(),
        });
    }
    let base = request.to_select_request();
    let mut chain: Vec<SelectedModel> = Vec::new();
    if let Some(alias) = &request.alias {
        let candidates = registry.resolve_alias(alias)?;
        for candidate in candidates {
            match registry.lookup(&candidate.provider_id, &candidate.model) {
                Some(entry)
                    if base.matches(entry)
                        && policy.is_visible(&entry.provider_id, &entry.name, scope_echo) =>
                {
                    chain.push(SelectedModel::from_entry(entry));
                }
                Some(_) | None => {}
            }
        }
    } else {
        for entry in registry.entries() {
            if base.matches(entry) && policy.is_visible(&entry.provider_id, &entry.name, scope_echo)
            {
                chain.push(SelectedModel::from_entry(entry));
            }
        }
    }
    if chain.is_empty() {
        return Err(SelectionError::NoCandidate {
            detail: scope_detail(request, scope_echo),
        });
    }
    Ok(chain)
}

/// Human-readable requirement-plus-scope summary for
/// [`SelectionError::NoCandidate`]: the S1 [`SelectRequest::describe`] text
/// plus the requested scope names only (`scopes=[a,b]`; `scopes=[]` when
/// unfiltered). Never carries secrets or vendor text.
fn scope_detail(request: &SelectEnvelopeRequest, scope_echo: &[CallerScope]) -> String {
    let names: Vec<&str> = scope_echo.iter().map(CallerScope::as_str).collect();
    format!("{} scopes=[{}]", request.describe(), names.join(","))
}

#[cfg(test)]
mod tests {
    use super::ModelCapability as Cap;
    use super::*;

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
    fn model_name_shape() {
        assert!(validate_model_name("chat").is_ok());
        assert!(validate_model_name("fake-chat").is_ok());
        assert!(validate_model_name("llama3.1:8b").is_ok());
        assert!(validate_model_name("org/model").is_ok());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("Chat").is_err());
        assert!(validate_model_name("-chat").is_err());
        assert!(validate_model_name("a".repeat(129).as_str()).is_err());
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
    fn fake_provider_shape_bridges_into_registry_unchanged() {
        // The host carries no provider trait and no `list_models` wiring, so
        // the bridge starts one step later: from a descriptor snapshot
        // mirroring exactly what the runtime scripted provider advertises
        // (single model `fake-chat` with text, streaming, and tool-use
        // capabilities). From that point on registration and selection
        // behave unchanged.
        let descriptor = ModelDescriptor {
            name: "fake-chat".to_owned(),
            capabilities: vec![Cap::Text, Cap::Streaming, Cap::ToolUse],
        };
        let mut registry = ProviderRegistry::new();
        registry
            .register(
                ModelRegistration::snapshot_from("bitty-fake", &descriptor, 4_096, 1, 2)
                    .expect("valid snapshot"),
            )
            .expect("capacity");
        let chain = registry
            .select(&SelectRequest::capabilities(vec![Cap::Text, Cap::ToolUse]))
            .expect("fake-chat matches text+tools");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider_id, "bitty-fake");
        assert_eq!(chain[0].name, "fake-chat");
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

    #[test]
    fn taxonomy_displays_without_secrets() {
        let auth = ProviderError::Auth {
            provider: "p".to_owned(),
            reason: "grant expired".to_owned(),
        };
        assert_eq!(
            auth.to_string(),
            "provider p refused authorization: grant expired"
        );
        let mismatch = ProviderError::CapabilityMismatch {
            provider: "p".to_owned(),
            model: "m".to_owned(),
            missing: vec![Cap::ImageInput, Cap::AudioInput],
        };
        assert_eq!(
            mismatch.to_string(),
            "provider p model m lacks 2 capabilities"
        );
    }

    #[test]
    fn estimate_cost_treats_uncalibrated_weight_as_baseline_one() {
        // Relative routing units, never currency: exact on small values.
        assert_eq!(estimate_cost(10, 5, 2, 3), 35);
        assert_eq!(estimate_cost(0, 0, 2, 3), 0);
        // A `0` weight means uncalibrated and counts as the baseline `1`:
        // an unset weight is never free, so routing cannot estimate lower
        // than accounting charges.
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
    fn envelope_select_is_alias_exclusive() {
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
        let envelope =
            SelectEnvelopeRequest::new(vec![Cap::Text], Some("chat".to_owned()), None, None);
        let chain = registry
            .select(&envelope.to_select_request())
            .expect("alias resolves");
        let response = SelectEnvelopeResponse::from_result(Ok(chain.clone()));
        assert!(response.is_selected());
        match response {
            SelectEnvelopeResponse::Selected { chain: kept } => {
                let order: Vec<(&str, &str)> = kept
                    .iter()
                    .map(|item| (item.provider_id.as_str(), item.name.as_str()))
                    .collect();
                // Alias-exclusive: only declared candidates in declared order;
                // the matching non-alias entry ("bitty-a", "vision") is never
                // appended.
                assert_eq!(order, vec![("bitty-c", "fast"), ("bitty-b", "chat")]);
                // Primary is element 0 and snapshots are detached.
                assert_eq!(kept[0].name, "fast");
                assert_eq!(kept, chain);
            }
            SelectEnvelopeResponse::Failed { error } => {
                panic!("alias-exclusive select must succeed: {error:?}");
            }
        }
        // Round-trip preserves the narrowing fields and the describe text.
        let rebuilt = SelectEnvelopeRequest::from_select_request(&envelope.to_select_request());
        assert_eq!(rebuilt, envelope);
        assert!(envelope.describe().contains("alias=chat"));
    }

    #[test]
    fn envelope_stale_chain_yields_no_candidate() {
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
        // One stale candidate degrades to the next in declared order, staying
        // alias-only.
        let envelope =
            SelectEnvelopeRequest::new(vec![Cap::Text], Some("chat".to_owned()), None, None);
        let response =
            SelectEnvelopeResponse::from_result(registry.select(&envelope.to_select_request()));
        match response.clone().into_result() {
            Ok(chain) => {
                assert_eq!(chain.len(), 1);
                assert_eq!(chain[0].name, "chat");
            }
            Err(error) => panic!("stale skips to live candidate: {error:?}"),
        }
        assert!(response.is_selected());
        // Every alias candidate stale while non-alias entries satisfy the
        // caps: full miss is NoCandidate, never a fall-through.
        let mut stale_only = registry_two();
        stale_only
            .register_alias(
                "chat",
                vec![ModelRef::new("bitty-gone", "ghost").expect("valid ref")],
            )
            .expect("capacity");
        let stale_envelope = SelectEnvelopeRequest::capabilities(vec![Cap::Text]);
        let stale_envelope = SelectEnvelopeRequest::new(
            stale_envelope.required,
            Some("chat".to_owned()),
            None,
            None,
        );
        let response = SelectEnvelopeResponse::from_result(
            stale_only.select(&stale_envelope.to_select_request()),
        );
        assert!(!response.is_selected());
        match response.into_result() {
            Err(SelectionError::NoCandidate { detail }) => {
                assert!(detail.contains("alias=chat"), "unexpected detail: {detail}");
            }
            other => panic!("fully stale alias must fail closed: {other:?}"),
        }
    }

    #[test]
    fn envelope_window_and_cost_gates_narrow_only() {
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
                1,
                1,
            ))
            .expect("capacity");
        registry
            .register(registration(
                "bitty-fake",
                "pricey",
                vec![Cap::Text],
                32_768,
                1,
                9,
            ))
            .expect("capacity");
        // No narrowing: all three are routing-eligible.
        let open = SelectEnvelopeRequest::capabilities(vec![Cap::Text]);
        let open_chain = registry
            .select(&open.to_select_request())
            .expect("all match");
        assert_eq!(open_chain.len(), 3);
        // Window minimum filters the unknown (`0`) window.
        let windowed = SelectEnvelopeRequest::new(vec![Cap::Text], None, Some(4_096), None);
        assert!(windowed.describe().contains("min_window=4096"));
        let windowed_chain = registry
            .select(&windowed.to_select_request())
            .expect("windowed matches");
        assert_eq!(windowed_chain.len(), 2);
        assert!(windowed_chain.iter().all(|item| item.name != "mystery"));
        // Cost ceiling filters both weights with effective(0) = 1.
        let priced = SelectEnvelopeRequest::new(vec![Cap::Text], None, Some(4_096), Some(2));
        assert!(priced.describe().contains("max_cost=2"));
        let priced_chain = registry
            .select(&priced.to_select_request())
            .expect("priced matches");
        assert_eq!(priced_chain.len(), 1);
        assert_eq!(priced_chain[0].name, "roomy");
        // Narrowing never widens: each filtered chain is a subset.
        let open_names: Vec<&str> = open_chain.iter().map(|item| item.name.as_str()).collect();
        for item in &windowed_chain {
            assert!(open_names.contains(&item.name.as_str()));
        }
        // A zero ceiling admits nothing: baseline `1` is the floor.
        let zero = SelectEnvelopeRequest::new(vec![Cap::Text], None, None, Some(0));
        assert!(matches!(
            registry.select(&zero.to_select_request()),
            Err(SelectionError::NoCandidate { .. })
        ));
    }

    #[test]
    fn envelope_estimate_cost_echo_effective_zero_is_one() {
        // Inputs echo plus effective(0) = 1 in relative units (never currency).
        let request = EstimateCostEnvelopeRequest::new(10, 5, 0, 0);
        let response = request.estimate();
        assert_eq!(response.input_tokens, 10);
        assert_eq!(response.output_tokens, 5);
        assert_eq!(response.input_weight, 0);
        assert_eq!(response.output_weight, 0);
        assert_eq!(response.cost, 15);
        assert!(response.is_consistent());
        // Zero weights equal baseline-one weights; partial-zero echoes too.
        assert_eq!(
            response.cost,
            EstimateCostEnvelopeRequest::new(10, 5, 1, 1)
                .estimate()
                .cost
        );
        let mixed = EstimateCostEnvelopeRequest::new(10, 5, 0, 3).estimate();
        assert_eq!(mixed.cost, 25);
        assert_eq!(
            mixed.cost,
            EstimateCostEnvelopeRequest::new(10, 5, 1, 3)
                .estimate()
                .cost
        );
        assert!(mixed.is_consistent());
        // Exact small-value accounting and saturation echo the S1 rule.
        assert_eq!(
            EstimateCostEnvelopeRequest::new(10, 5, 2, 3)
                .estimate()
                .cost,
            35
        );
        assert_eq!(
            EstimateCostEnvelopeRequest::new(0, 0, 2, 3).estimate().cost,
            0
        );
        assert_eq!(
            EstimateCostEnvelopeRequest::new(u64::MAX, 1, 2, u32::MAX)
                .estimate()
                .cost,
            u64::MAX
        );
    }

    #[test]
    fn envelope_fallback_mapping_is_total_incl_unknown_stop() {
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
            let envelope = FallbackDirectiveEnvelopeRequest::new(error.clone());
            let resolved = envelope.resolve();
            assert_eq!(resolved.directive, FallbackDirective::Advance);
            assert_eq!(
                resolved,
                FallbackDirectiveEnvelopeResponse::new(FallbackDirective::Advance)
            );
            assert_eq!(resolved.directive, fallback_directive(error));
        }
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
            ProviderError::Unknown {
                provider: "p".to_owned(),
                reason: "ack lost".to_owned(),
            },
        ];
        for error in &stop {
            let envelope = FallbackDirectiveEnvelopeRequest::new(error.clone());
            let resolved = envelope.resolve();
            assert_eq!(
                resolved.directive,
                FallbackDirective::Stop,
                "must stop: {error}"
            );
            assert_eq!(resolved.directive, fallback_directive(error));
        }
        // Unknown explicitly stops: the effect may have happened, so blind
        // advance could double-apply; reconcile first.
        let unknown = FallbackDirectiveEnvelopeRequest::new(ProviderError::Unknown {
            provider: "p".to_owned(),
            reason: "ack lost".to_owned(),
        });
        assert_eq!(unknown.resolve().directive, FallbackDirective::Stop);
    }

    #[test]
    fn envelope_bounds_match_s1_and_carry_no_version() {
        // Byte bounds are the S1 values, unchanged for the envelope layer.
        assert_eq!(MAX_MODEL_NAME_LEN, 128);
        assert_eq!(MAX_REGISTERED_MODELS, 64);
        assert_eq!(MAX_ALIASES, 32);
        assert_eq!(MAX_ALIAS_CANDIDATES, 8);
        // Envelopes carry no version field by construction: exact-field
        // construction round-trips without loss.
        let request = SelectEnvelopeRequest::new(
            vec![Cap::Text],
            Some("chat".to_owned()),
            Some(4_096),
            Some(2),
        );
        assert_eq!(
            SelectEnvelopeRequest::from_select_request(&request.to_select_request()),
            request
        );
        let chain = vec![SelectedModel {
            provider_id: "bitty-fake".to_owned(),
            name: "m".to_owned(),
            capabilities: vec![Cap::Text],
            context_window_tokens: 4_096,
            input_cost_weight: 1,
            output_cost_weight: 2,
        }];
        let selected = SelectEnvelopeResponse::from_result(Ok(chain.clone()));
        assert_eq!(selected.clone().into_result().expect("selected"), chain);
        let failed = SelectEnvelopeResponse::from_result(Err(SelectionError::NoCandidate {
            detail: "1 capabilities alias=chat".to_owned(),
        }));
        assert!(!failed.is_selected());
        assert!(matches!(
            failed.into_result(),
            Err(SelectionError::NoCandidate { .. })
        ));
    }

    #[test]
    fn canonical_identity_reexport_matches_provider_registry() {
        // AI-0287: the S1 vocabulary is canonically owned by
        // `crate::provider_registry`; this module only re-exports it. Any
        // reintroduced divergent copy fails these assertions (layout
        // mismatch, cross-path assignment refusal at compile time, or
        // behavioral disagreement at run time).
        assert_eq!(
            std::mem::size_of::<super::SelectedModel>(),
            std::mem::size_of::<crate::provider_registry::SelectedModel>()
        );
        assert_eq!(
            std::mem::align_of::<super::SelectRequest>(),
            std::mem::align_of::<crate::provider_registry::SelectRequest>()
        );
        assert_eq!(
            std::mem::size_of::<super::SelectionError>(),
            std::mem::size_of::<crate::provider_registry::SelectionError>()
        );
        // Cross-path assignment compiles only when both paths name the same
        // type (compile-time identity proof).
        let unified: crate::provider_registry::SelectedModel = super::SelectedModel {
            provider_id: "bitty-fake".to_owned(),
            name: "chat".to_owned(),
            capabilities: vec![Cap::Text],
            context_window_tokens: 4_096,
            input_cost_weight: 1,
            output_cost_weight: 2,
        };
        // Behavior agrees through both paths, including the uncalibrated
        // (`0` counts as baseline `1`) and saturating rules.
        for (input, output, in_w, out_w) in [(10, 5, 1, 2), (7, 9, 0, 0), (u64::MAX, 1, 3, 4)] {
            assert_eq!(
                super::estimate_cost(input, output, in_w, out_w),
                crate::provider_registry::estimate_cost(input, output, in_w, out_w)
            );
        }
        for error in [
            ProviderError::Transport {
                provider: "p".to_owned(),
                reason: "reset".to_owned(),
            },
            ProviderError::Unknown {
                provider: "p".to_owned(),
                reason: "ack lost".to_owned(),
            },
        ] {
            assert_eq!(
                super::fallback_directive(&error),
                fallback_directive(&error)
            );
            assert_eq!(
                super::fallback_directive(&error),
                crate::provider_registry::fallback_directive(&error)
            );
        }
        for name in ["chat", "BAD NAME", ""] {
            assert_eq!(
                super::validate_model_name(name).is_ok(),
                crate::provider_registry::validate_model_name(name).is_ok()
            );
        }
        for id in ["bitty-fake", "BAD", ""] {
            assert_eq!(
                super::validate_provider_id(id).is_ok(),
                crate::provider_registry::validate_provider_id(id).is_ok()
            );
        }
        // Round-trip: a registry built via the provider path selects
        // identically through the selection path.
        let mut registry = crate::provider_registry::ProviderRegistry::new();
        registry
            .register(text_model("bitty-fake", "chat"))
            .expect("capacity");
        let request = super::SelectRequest::capabilities(vec![Cap::Text]);
        let via_selection: Vec<super::SelectedModel> =
            registry.select(&request).expect("candidate");
        let via_provider: Vec<crate::provider_registry::SelectedModel> =
            registry.select(&request).expect("candidate");
        assert_eq!(via_selection, via_provider);
        assert_eq!(via_selection[0], unified);
        assert_eq!(
            via_selection[0].estimate_cost(10, 5),
            crate::provider_registry::estimate_cost(10, 5, 1, 2)
        );
    }

    fn scope(name: &str) -> CallerScope {
        CallerScope::new(name).expect("valid scope")
    }

    fn scoped_registry() -> ProviderRegistry {
        let mut registry = registry_two();
        registry
            .register(text_model("bitty-c", "fast"))
            .expect("capacity");
        registry
    }

    fn scoped_policy() -> ScopePolicy {
        // bitty-a/vision sees team-a only; bitty-b/chat sees team-a and
        // team-b; bitty-c/fast sees team-b only. A two-scope echo therefore
        // admits only chat (requested ⊆ granted), so adding a scope never
        // widens.
        let mut policy = ScopePolicy::new();
        policy
            .grant("bitty-a", "vision", vec![scope("team-a")])
            .expect("capacity");
        policy
            .grant("bitty-b", "chat", vec![scope("team-a"), scope("team-b")])
            .expect("capacity");
        policy
            .grant("bitty-c", "fast", vec![scope("team-b")])
            .expect("capacity");
        policy
    }

    #[test]
    fn scope_string_grammar() {
        assert_eq!(MAX_CALLER_SCOPE_LEN, 64);
        assert_eq!(MAX_CALLER_SCOPES, 8);
        assert_eq!(MAX_SCOPED_MODELS, 64);
        assert_eq!(MAX_SCOPES_PER_MODEL, 8);
        for valid in [
            "team-a",
            "a",
            "a1",
            "team-a/proj-x",
            "proto/inst/session",
            "x.y-z_w",
        ] {
            let parsed = CallerScope::new(valid).expect("valid scope");
            assert_eq!(parsed.as_str(), valid);
            assert_eq!(parsed.clone().into_string(), valid);
            assert_eq!(parsed.to_string(), valid);
        }
        for invalid in [
            "",
            "Team-a",
            "team a",
            "/team-a",
            "team-a/",
            "team-a//proj",
            "-team",
            ".team",
            "_team",
            "team-😀",
            "TEAM",
        ] {
            assert!(
                matches!(
                    CallerScope::new(invalid),
                    Err(SelectionError::InvalidName { .. })
                ),
                "must refuse: {invalid:?}"
            );
        }
        let overlong = "a".repeat(MAX_CALLER_SCOPE_LEN + 1);
        assert!(matches!(
            CallerScope::new(overlong),
            Err(SelectionError::InvalidName { .. })
        ));
        let ceiling = "a".repeat(MAX_CALLER_SCOPE_LEN);
        assert!(CallerScope::new(ceiling).is_ok());
    }

    #[test]
    fn scope_policy_grant_discipline() {
        let mut policy = ScopePolicy::new();
        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
        policy
            .grant("bitty-a", "vision", vec![scope("team-a")])
            .expect("capacity");
        assert_eq!(policy.len(), 1);
        assert_eq!(
            policy.scopes_for("bitty-a", "vision"),
            Some(vec![scope("team-a")].as_slice())
        );
        assert!(policy.scopes_for("bitty-a", "other").is_none());
        // Empty grant is allowed: visible only when unfiltered, denied
        // whenever filtering (deny-by-default).
        policy
            .grant("bitty-b", "chat", Vec::new())
            .expect("capacity");
        assert!(policy.is_visible("bitty-b", "chat", &[]));
        assert!(!policy.is_visible("bitty-b", "chat", &[scope("team-a")]));
        // Ungranted keys are denied when filtering, allowed when unfiltered.
        assert!(policy.is_visible("bitty-c", "fast", &[]));
        assert!(!policy.is_visible("bitty-c", "fast", &[scope("team-a")]));
        // Exact re-grant is a duplicate; same key with different scopes is a
        // conflict (no shadowing, kept untouched).
        let retry = policy.grant("bitty-a", "vision", vec![scope("team-a")]);
        assert_eq!(
            retry.expect_err("duplicate must fail"),
            SelectionError::DuplicateModel {
                provider: "bitty-a".to_owned(),
                model: "vision".to_owned(),
            }
        );
        let shadow = policy.grant("bitty-a", "vision", vec![scope("team-b")]);
        assert_eq!(
            shadow.expect_err("conflict must fail"),
            SelectionError::ConflictingModel {
                provider: "bitty-a".to_owned(),
                model: "vision".to_owned(),
            }
        );
        assert_eq!(
            policy.scopes_for("bitty-a", "vision"),
            Some(vec![scope("team-a")].as_slice())
        );
        // Malformed keys are refused with their typed error.
        assert!(matches!(
            policy.grant("BITTY", "chat", vec![scope("team-a")]),
            Err(SelectionError::InvalidProviderId { .. })
        ));
        assert!(matches!(
            policy.grant("bitty-a", "Chat", vec![scope("team-a")]),
            Err(SelectionError::InvalidName { .. })
        ));
        assert_eq!(policy.len(), 2);
        // Over-limit scope lists fail closed with names plus counts.
        let burst: Vec<CallerScope> = (0..MAX_SCOPES_PER_MODEL + 1)
            .map(|index| scope(format!("team-{index}").as_str()))
            .collect();
        let error = policy
            .grant("bitty-c", "fast", burst)
            .expect_err("burst must fail");
        match error {
            SelectionError::TooManyCandidates {
                alias,
                limit,
                actual,
            } => {
                assert_eq!(alias, "bitty-c/fast");
                assert_eq!(limit, MAX_SCOPES_PER_MODEL);
                assert_eq!(actual, MAX_SCOPES_PER_MODEL + 1);
            }
            other => panic!("burst must report bound: {other:?}"),
        }
        assert_eq!(policy.len(), 2);
    }

    #[test]
    fn scope_filter_exclusive_alias_plus_scope() {
        let mut registry = scoped_registry();
        registry
            .register_alias(
                "chat",
                vec![
                    ModelRef::new("bitty-c", "fast").expect("valid ref"),
                    ModelRef::new("bitty-b", "chat").expect("valid ref"),
                ],
            )
            .expect("capacity");
        let policy = scoped_policy();
        // Echo team-b: both alias candidates grant it, in declared order.
        // The matching non-alias entry (bitty-a/vision) is never appended.
        let request =
            SelectEnvelopeRequest::new(vec![Cap::Text], Some("chat".to_owned()), None, None);
        let chain = select_filtered(&registry, &request, &[scope("team-b")], &policy)
            .expect("alias resolves under team-b");
        let order: Vec<(&str, &str)> = chain
            .iter()
            .map(|item| (item.provider_id.as_str(), item.name.as_str()))
            .collect();
        assert_eq!(order, vec![("bitty-c", "fast"), ("bitty-b", "chat")]);
        assert_eq!(chain[0].name, "fast");
        // Echo team-a: fast grants team-b only, so it is denied; the chain
        // stays alias-only with chat alone (no fall-through to vision).
        let chain = select_filtered(&registry, &request, &[scope("team-a")], &policy)
            .expect("alias narrows to chat under team-a");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider_id, "bitty-b");
        assert_eq!(chain[0].name, "chat");
        // Unknown scope (well-formed, granted by nothing) fails closed with
        // names only, never a fall-through to vision.
        match select_filtered(&registry, &request, &[scope("team-c")], &policy) {
            Err(SelectionError::NoCandidate { detail }) => {
                assert!(detail.contains("alias=chat"), "unexpected detail: {detail}");
                assert!(detail.contains("team-c"), "unexpected detail: {detail}");
            }
            other => panic!("scope-denied alias must fail closed: {other:?}"),
        }
    }

    #[test]
    fn scope_filter_stale_chain_yields_no_candidate() {
        let mut registry = scoped_registry();
        registry
            .register_alias(
                "chat",
                vec![
                    ModelRef::new("bitty-gone", "ghost").expect("valid ref"),
                    ModelRef::new("bitty-b", "chat").expect("valid ref"),
                ],
            )
            .expect("capacity");
        let policy = scoped_policy();
        // Stale skipped in declared order, live candidate still needs its
        // scope: team-b is granted, so chat survives (alias-only).
        let request =
            SelectEnvelopeRequest::new(vec![Cap::Text], Some("chat".to_owned()), None, None);
        let chain = select_filtered(&registry, &request, &[scope("team-b")], &policy)
            .expect("stale skipped under team-b");
        // bitty-b/chat grants team-a and team-b, so team-b passes.
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "chat");
        // Same chain under an unknown scope: the stale entry is skipped and
        // the live entry is scope-denied, so the full miss is NoCandidate
        // (never a fall-through to the matching non-alias vision entry).
        match select_filtered(&registry, &request, &[scope("team-c")], &policy) {
            Err(SelectionError::NoCandidate { detail }) => {
                assert!(detail.contains("alias=chat"), "unexpected detail: {detail}");
                assert!(detail.contains("team-c"), "unexpected detail: {detail}");
            }
            other => panic!("stale plus scope-denied must fail closed: {other:?}"),
        }
        // Fully stale alias under any echo is still NoCandidate.
        let mut stale_only = scoped_registry();
        stale_only
            .register_alias(
                "chat",
                vec![ModelRef::new("bitty-gone", "ghost").expect("valid ref")],
            )
            .expect("capacity");
        match select_filtered(&stale_only, &request, &[scope("team-a")], &policy) {
            Err(SelectionError::NoCandidate { detail }) => {
                assert!(detail.contains("alias=chat"), "unexpected detail: {detail}");
            }
            other => panic!("fully stale alias must fail closed: {other:?}"),
        }
    }

    #[test]
    fn scope_filter_narrows_only() {
        let registry = scoped_registry();
        let policy = scoped_policy();
        let request = SelectEnvelopeRequest::capabilities(vec![Cap::Text]);
        let open = select_filtered(&registry, &request, &[], &policy).expect("unfiltered");
        assert_eq!(open.len(), 3);
        let team_a =
            select_filtered(&registry, &request, &[scope("team-a")], &policy).expect("team-a");
        let team_b =
            select_filtered(&registry, &request, &[scope("team-b")], &policy).expect("team-b");
        // Each filtered view is a subset of the unfiltered snapshot (never
        // wider), in the same provider/model lexicographic order.
        let open_names: Vec<&str> = open.iter().map(|item| item.name.as_str()).collect();
        for item in team_a.iter().chain(team_b.iter()) {
            assert!(open_names.contains(&item.name.as_str()));
        }
        assert_eq!(open_names, vec!["vision", "chat", "fast"]);
        let team_a_names: Vec<&str> = team_a.iter().map(|item| item.name.as_str()).collect();
        let team_b_names: Vec<&str> = team_b.iter().map(|item| item.name.as_str()).collect();
        assert_eq!(team_a_names, vec!["vision", "chat"]);
        assert_eq!(team_b_names, vec!["chat", "fast"]);
        // Adding a requested scope never widens: team-a plus team-b admits
        // only chat (requested ⊆ granted).
        let both = select_filtered(
            &registry,
            &request,
            &[scope("team-a"), scope("team-b")],
            &policy,
        )
        .expect("both scopes");
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].name, "chat");
        let both_names: Vec<&str> = both.iter().map(|item| item.name.as_str()).collect();
        for name in &both_names {
            assert!(team_a_names.contains(name) && team_b_names.contains(name));
        }
        // Unknown scope admits nothing.
        assert!(matches!(
            select_filtered(&registry, &request, &[scope("team-c")], &policy),
            Err(SelectionError::NoCandidate { .. })
        ));
        // Over-limit echoes fail closed with counts only (never widen).
        let burst: Vec<CallerScope> = (0..MAX_CALLER_SCOPES + 1)
            .map(|index| scope(format!("team-{index}").as_str()))
            .collect();
        match select_filtered(&registry, &request, &burst, &policy) {
            Err(SelectionError::TooManyCandidates {
                alias,
                limit,
                actual,
            }) => {
                assert_eq!(alias, "scope-echo");
                assert_eq!(limit, MAX_CALLER_SCOPES);
                assert_eq!(actual, MAX_CALLER_SCOPES + 1);
            }
            other => panic!("over-limit echo must fail closed: {other:?}"),
        }
        // Empty requirement is still refused before any scope work.
        let empty = SelectEnvelopeRequest::capabilities(Vec::new());
        assert_eq!(
            select_filtered(&registry, &empty, &[scope("team-a")], &policy)
                .expect_err("empty requirement must fail"),
            SelectionError::EmptyRequirement
        );
        // Unknown alias still fails closed (scope does not rescue it).
        let unknown =
            SelectEnvelopeRequest::new(vec![Cap::Text], Some("nope".to_owned()), None, None);
        assert_eq!(
            select_filtered(&registry, &unknown, &[scope("team-a")], &policy)
                .expect_err("unknown alias must fail"),
            SelectionError::UnknownAlias {
                alias: "nope".to_owned(),
            }
        );
    }

    #[test]
    fn scope_filter_effective_zero_is_one_preserved() {
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
                1,
                1,
            ))
            .expect("capacity");
        registry
            .register(registration(
                "bitty-fake",
                "pricey",
                vec![Cap::Text],
                32_768,
                1,
                9,
            ))
            .expect("capacity");
        let mut policy = ScopePolicy::new();
        for name in ["mystery", "roomy", "pricey"] {
            policy
                .grant("bitty-fake", name, vec![scope("team-a")])
                .expect("capacity");
        }
        let echo = [scope("team-a")];
        // Window minimum refuses the unknown (`0`) window through the
        // filtered path too.
        let windowed = SelectEnvelopeRequest::new(vec![Cap::Text], None, Some(4_096), None);
        let chain = select_filtered(&registry, &windowed, &echo, &policy).expect("windowed");
        assert_eq!(chain.len(), 2);
        assert!(chain.iter().all(|item| item.name != "mystery"));
        // Cost ceiling uses effective weights through the filtered path:
        // uncalibrated (`0`) counts as baseline `1`.
        let mut uncalibrated = ProviderRegistry::new();
        uncalibrated
            .register(registration(
                "bitty-fake",
                "uncalibrated",
                vec![Cap::Text],
                4_096,
                0,
                0,
            ))
            .expect("capacity");
        uncalibrated
            .register(registration(
                "bitty-fake",
                "calibrated",
                vec![Cap::Text],
                4_096,
                2,
                2,
            ))
            .expect("capacity");
        let mut weights = ScopePolicy::new();
        for name in ["uncalibrated", "calibrated"] {
            weights
                .grant("bitty-fake", name, vec![scope("team-a")])
                .expect("capacity");
        }
        let ceiling = SelectEnvelopeRequest::new(vec![Cap::Text], None, None, Some(1));
        let chain = select_filtered(&uncalibrated, &ceiling, &echo, &weights)
            .expect("uncalibrated fits baseline");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name, "uncalibrated");
        // A zero ceiling admits nothing through the filtered path either:
        // baseline `1` is the floor, `0` is never free.
        let zero = SelectEnvelopeRequest::new(vec![Cap::Text], None, None, Some(0));
        assert!(matches!(
            select_filtered(&uncalibrated, &zero, &echo, &weights),
            Err(SelectionError::NoCandidate { .. })
        ));
        // Filtered cost agrees with the unfiltered S1 rule exactly.
        let open = SelectEnvelopeRequest::new(vec![Cap::Text], None, None, Some(1));
        let filtered = select_filtered(&uncalibrated, &open, &echo, &weights).expect("filtered");
        let unfiltered = uncalibrated
            .select(&open.to_select_request())
            .expect("unfiltered");
        assert_eq!(filtered, unfiltered);
    }

    #[test]
    fn scope_filter_fallback_mapping_is_total_incl_unknown_stop() {
        // The S3 filter adds no fallback path: the mapping stays the S1
        // total (`Advance` on transport/rate-limit/unavailable/timeout,
        // `Stop` otherwise including `Unknown`).
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
            assert_eq!(
                super::fallback_directive(error),
                crate::provider_registry::fallback_directive(error)
            );
        }
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
            ProviderError::Unknown {
                provider: "p".to_owned(),
                reason: "ack lost".to_owned(),
            },
        ];
        for error in &stop {
            assert_eq!(
                fallback_directive(error),
                FallbackDirective::Stop,
                "must stop: {error}"
            );
        }
        // Unknown explicitly stops through the selection path too: the effect
        // may have happened, so blind advance could double-apply; reconcile
        // first. A filtered chain still advances only via the same typed
        // directive, never past Unknown.
        let registry = scoped_registry();
        let policy = scoped_policy();
        let request = SelectEnvelopeRequest::capabilities(vec![Cap::Text]);
        let chain = select_filtered(&registry, &request, &[scope("team-a")], &policy)
            .expect("team-a chain");
        assert!(!chain.is_empty());
        assert_eq!(
            fallback_directive(&ProviderError::Unknown {
                provider: "p".to_owned(),
                reason: "ack lost".to_owned(),
            }),
            FallbackDirective::Stop
        );
    }
}

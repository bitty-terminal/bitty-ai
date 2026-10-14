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
//! - No MP-4 caller-scope filter and no closed-vocabulary enforcement: the
//!   capability mapping prose on [`SelectRequest`] is routing vocabulary
//!   only, never a rejection rule.
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
//! layer below plus the gate-order contract above.
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
}

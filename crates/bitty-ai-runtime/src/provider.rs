//! Deterministic model-provider boundary.
//!
//! Mirrors the `MP-4`..`MP-8` shape: a host-owned model registry surface
//! (`list_models`), one synchronous turn (`complete`), deterministic timeouts,
//! and idempotent cancellation. There are no network providers here: the only
//! implementation is the scripted [`FakeProvider`], which replays
//! caller-supplied turns so tests run offline with no secrets.
//!
//! ## Ownership (R1 draft disposition)
//!
//! Provider registry implementation and all model I/O belong on the AI helper
//! side (this crate, staging toward a `bitty-ai-host` helper behind scoped
//! IPC), per the draft disposition of `MP-1` versus `BA-2`/`BA-3` in
//! `docs/specifications/execution-ownership-r1.md`: `BA-2` (Agent versus AI
//! split) and `BA-3` (bridge process model) win on placement.
//!
//! Red line: `bitty-agent` performs no model selection, no model I/O, and no
//! API-key handling. A terminal-side registry, if retained, validates generic
//! service metadata and mediates authorized requests only (see
//! [`TerminalModelMetadata`]): it is not a provider implementation and holds
//! no credentials.
//!
//! [`FakeProvider`] is the only provider here: deterministic, scripted,
//! offline. Network providers, TLS, credentials, and real model calls are out
//! of scope by design, not deferred work.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::str::FromStr;

/// Default per-request timeout in milliseconds (`MP-8`).
pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 5_000;
/// Default timeout for tool-mediated streaming in milliseconds (`MP-8`).
pub const DEFAULT_TOOL_STREAM_TIMEOUT_MS: u64 = 10_000;
/// Hard timeout ceiling in milliseconds (`MP-8`); larger values fail closed.
pub const MAX_REQUEST_TIMEOUT_MS: u64 = 30_000;
/// Candidate default context budget in bytes (`CP-5` profile value, not a core
/// contract).
pub const DEFAULT_CONTEXT_BUDGET_BYTES: usize = 32 * 1024;
/// Maximum provider id length in bytes (`MP-2`).
pub const MAX_PROVIDER_ID_LEN: usize = 64;
/// Maximum stop sequences per sampling contract (`MP-5`, `TB-6` parity).
pub const MAX_STOP_SEQUENCES: usize = 8;
/// Maximum bytes per stop sequence (`MP-5`, house summary parity).
pub const MAX_STOP_SEQUENCE_BYTES: usize = 256;
/// Maximum response-schema bytes for [`ResponseFormat::JsonSchema`]
/// (`MP-5`, tool-schema parity).
pub const MAX_RESPONSE_SCHEMA_BYTES: usize = 16 * 1024;
/// Minimum reasoning-token budget when declared (`MP-5`, `max_tokens` parity).
pub const MIN_REASONING_TOKENS: u32 = 1;
/// Estimated share of the response budget a declared reasoning effort
/// implies, DOCUMENTED ESTIMATION ONLY (`MP-5` cost-routing input).
///
/// The mapping mirrors the OpenRouter effort-ratio evidence the task pins;
/// it feeds [`ReasoningEffort::estimated_budget_ratio`] and
/// [`crate::selection::estimate_cost`] input only. It never gates
/// validation: effort declared without any token budget validates and
/// executes, and provider-specific mapping (OpenAI effort versus Anthropic
/// `max_tokens` versus Gemini thinking level) stays host-side (`MP-4`
/// deferred).
pub const REASONING_RATIO_MINIMAL: f64 = 0.1;
/// Estimated share for [`ReasoningEffort::Low`] (estimation only).
pub const REASONING_RATIO_LOW: f64 = 0.2;
/// Estimated share for [`ReasoningEffort::Medium`] (estimation only).
pub const REASONING_RATIO_MEDIUM: f64 = 0.5;
/// Estimated share for [`ReasoningEffort::High`] (estimation only).
pub const REASONING_RATIO_HIGH: f64 = 0.8;
/// Estimated share for [`ReasoningEffort::XHigh`] (estimation only).
pub const REASONING_RATIO_XHIGH: f64 = 0.95;
/// Estimated share for [`ReasoningEffort::Max`] (estimation only).
pub const REASONING_RATIO_MAX: f64 = 0.95;
/// Minimum sampling temperature (`MP-5`, OpenRouter evidence).
pub const MIN_TEMPERATURE: f64 = 0.0;
/// Maximum sampling temperature (`MP-5`, OpenRouter evidence).
pub const MAX_TEMPERATURE: f64 = 2.0;
/// Minimum nucleus-sampling probability (`MP-5`, OpenRouter evidence).
pub const MIN_TOP_P: f64 = 0.0;
/// Maximum nucleus-sampling probability (`MP-5`, OpenRouter evidence).
pub const MAX_TOP_P: f64 = 1.0;
/// Minimum `top_k` value (`MP-5`, OpenRouter evidence; `0` disables).
pub const MIN_TOP_K: i64 = 0;
/// Minimum frequency/presence penalty (`MP-5`, OpenRouter evidence).
pub const MIN_PENALTY: f64 = -2.0;
/// Maximum frequency/presence penalty (`MP-5`, OpenRouter evidence).
pub const MAX_PENALTY: f64 = 2.0;
/// Minimum repetition penalty (`MP-5`, OpenRouter evidence).
pub const MIN_REPETITION_PENALTY: f64 = 0.0;
/// Maximum repetition penalty (`MP-5`, OpenRouter evidence).
pub const MAX_REPETITION_PENALTY: f64 = 2.0;
/// Minimum `min_p` probability (`MP-5`, OpenRouter evidence).
pub const MIN_MIN_P: f64 = 0.0;
/// Maximum `min_p` probability (`MP-5`, OpenRouter evidence).
pub const MAX_MIN_P: f64 = 1.0;
/// Minimum completion token budget when declared (`MP-5`).
pub const MIN_MAX_TOKENS: u32 = 1;

/// Model capability flags (`MP-2`, skeleton vocabulary).
///
/// Capabilities are routing data, never authority: selection matches on
/// required capability sets (see `crate::selection`) and execution still goes
/// through [`ModelProvider::complete`]. The multimodal flags (`ImageInput`,
/// `AudioInput`, `AudioOutput`, `VideoInput`) extend the text baseline with
/// the routing inputs AI-0030 requires; unknown transports or model cards
/// never synthesize a capability.
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
    /// Model may emit reasoning/thinking tokens (`MP-2` routing bit; no
    /// inference or enforcement happens here, see [`ReasoningConfig`]).
    Reasoning,
}

/// One registry-known model (`MP-2` descriptor, skeleton subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// Model name as referenced by [`TurnRequest::model`].
    pub name: String,
    /// Capability flags for this model.
    pub capabilities: Vec<ModelCapability>,
}

/// Terminal-side generic model metadata mediation placeholder (R1 draft
/// disposition).
///
/// When a terminal-side registry is retained, it validates generic service
/// metadata and mediates authorized requests only: it is not a provider
/// implementation, performs no model I/O, and holds no credentials. This
/// struct is that boundary expressed in code: a model name plus a capability
/// snapshot, derivable from a [`ModelDescriptor`], with no I/O methods, no
/// secret fields, and no clock. Real provider I/O stays on the AI helper side
/// behind [`ModelProvider`]; `bitty-agent` performs no model selection, no
/// model I/O, and no API-key handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalModelMetadata {
    /// Registry-known model name (mirrors [`ModelDescriptor::name`]).
    pub name: String,
    /// Capability snapshot (mirrors [`ModelDescriptor::capabilities`]).
    pub capabilities: Vec<ModelCapability>,
}

impl TerminalModelMetadata {
    /// Snapshot generic metadata from a helper-side registry descriptor. No
    /// credentials are read (there are none to read) and no I/O happens.
    #[must_use]
    pub fn from_descriptor(descriptor: &ModelDescriptor) -> Self {
        Self {
            name: descriptor.name.clone(),
            capabilities: descriptor.capabilities.clone(),
        }
    }
}

/// Conversation role for one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Human or host instruction.
    User,
    /// Model output.
    Assistant,
    /// Tool observation or assembled context record.
    Tool,
}

/// One bounded conversation message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Message role.
    pub role: Role,
    /// Message bytes (UTF-8; may carry untrusted observation data).
    pub content: String,
}

impl Message {
    /// Build a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Build an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// Build a tool/observation message.
    #[must_use]
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }

    /// Byte length of the message content.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.content.len()
    }
}

/// One model-requested tool call (dispatched by the Tool Bus, never executed
/// here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRequest {
    /// Registered tool name.
    pub name: String,
    /// Opaque bounded JSON arguments.
    pub arguments: Vec<u8>,
}

/// Provider-neutral reasoning effort vocabulary (`MP-5` contract label).
///
/// The contract never maps an effort to provider-specific knobs: OpenAI
/// effort versus Anthropic `max_tokens` versus Gemini thinking level stays
/// host-side (`MP-4` deferred). The contract pins the closed vocabulary plus
/// a from-str/to-str normalization table (see
/// [`ReasoningEffort::from_str`]/[`Display`]) and the documented estimation
/// ratios ([`ReasoningEffort::estimated_budget_ratio`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// No reasoning requested.
    None,
    /// Smallest reasoning share.
    Minimal,
    /// Low reasoning share.
    Low,
    /// Medium reasoning share.
    Medium,
    /// High reasoning share.
    High,
    /// Extra-high reasoning share.
    XHigh,
    /// Maximum reasoning share.
    Max,
}

/// Unknown reasoning-effort string. Returned by
/// [`ReasoningEffort::from_str`] for any label outside the provider-neutral
/// vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownReasoningEffort;

impl Display for UnknownReasoningEffort {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "unknown reasoning effort")
    }
}

impl std::error::Error for UnknownReasoningEffort {}

impl Display for ReasoningEffort {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::None => write!(f, "none"),
            Self::Minimal => write!(f, "minimal"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::XHigh => write!(f, "xhigh"),
            Self::Max => write!(f, "max"),
        }
    }
}

impl FromStr for ReasoningEffort {
    type Err = UnknownReasoningEffort;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            _ => Err(UnknownReasoningEffort),
        }
    }
}

impl ReasoningEffort {
    /// Documented estimation share of the response budget this effort
    /// implies, or `None` when no reasoning was requested. The value feeds
    /// cost-accounting estimation ([`crate::selection::estimate_cost`]
    /// input) only; it never gates validation or execution.
    #[must_use]
    pub fn estimated_budget_ratio(self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Minimal => Some(REASONING_RATIO_MINIMAL),
            Self::Low => Some(REASONING_RATIO_LOW),
            Self::Medium => Some(REASONING_RATIO_MEDIUM),
            Self::High => Some(REASONING_RATIO_HIGH),
            Self::XHigh => Some(REASONING_RATIO_XHIGH),
            Self::Max => Some(REASONING_RATIO_MAX),
        }
    }
}

/// Provider-neutral reasoning configuration (`MP-5` contract subset).
///
/// `exclude` controls whether reasoning tokens are hidden from the response
/// text. It is a routing/display hint carried to the host adapter; the
/// runtime enforces nothing about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningConfig {
    /// Declared effort, or `None` when only a token budget is set.
    pub effort: Option<ReasoningEffort>,
    /// Reasoning token budget, or `None` when only effort is declared.
    /// Effort-only is estimation-only: it validates and executes and the
    /// effort-ratio table supplies the cost estimate.
    pub max_tokens: Option<u32>,
    /// Hide reasoning tokens from the response text (host-adapter hint).
    pub exclude: bool,
}

/// Provider-neutral response-format declaration (`MP-5` contract subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Unconstrained text.
    Text,
    /// JSON object without a schema.
    JsonObject,
    /// JSON conforming to a bounded inline schema.
    JsonSchema {
        /// Inline JSON schema bytes, or `None` for an unconstrained schema
        /// object. Bounded by [`MAX_RESPONSE_SCHEMA_BYTES`].
        schema: Option<Vec<u8>>,
    },
}

/// Optional, validated sampling contract for one provider turn (`MP-5`).
///
/// Absent (`None` on [`TurnRequest::sampling`], or `None` per field here)
/// means undeclared and flows through untouched: it is never defaulted,
/// never validated, and never hashed into cache keys. `Some` means
/// explicitly declared and validated fail-closed by [`validate_sampling`]
/// before any provider I/O.
///
/// Float fields use `f64` for the validated contract range (NaN and
/// infinities rejected); exact bit equality is out of scope and this type
/// deliberately implements `PartialEq` only, mirroring the untrusted-input
/// rule that values participate in validation, never in keying or identity.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Sampling temperature in [`MIN_TEMPERATURE`]..=[`MAX_TEMPERATURE`].
    pub temperature: Option<f64>,
    /// Nucleus-sampling probability in [`MIN_TOP_P`]..=[`MAX_TOP_P`].
    pub top_p: Option<f64>,
    /// Top-k cutoff, `>= 0` (`0` disables).
    pub top_k: Option<i64>,
    /// Frequency penalty in [`MIN_PENALTY`]..=[`MAX_PENALTY`].
    pub frequency_penalty: Option<f64>,
    /// Presence penalty in [`MIN_PENALTY`]..=[`MAX_PENALTY`].
    pub presence_penalty: Option<f64>,
    /// Repetition penalty in
    /// [`MIN_REPETITION_PENALTY`]..=[`MAX_REPETITION_PENALTY`].
    pub repetition_penalty: Option<f64>,
    /// Minimum-probability cutoff in [`MIN_MIN_P`]..=[`MAX_MIN_P`].
    pub min_p: Option<f64>,
    /// Deterministic seed; any integer is accepted.
    pub seed: Option<i64>,
    /// Completion token budget, `>= 1`. When reasoning also declares a
    /// budget, the reasoning budget must be strictly smaller (Anthropic
    /// rule) or validation fails.
    pub max_tokens: Option<u32>,
    /// Stop sequences: at most [`MAX_STOP_SEQUENCES`] entries, each at most
    /// [`MAX_STOP_SEQUENCE_BYTES`] bytes.
    pub stop: Option<Vec<String>>,
    /// Declared response format.
    pub response_format: Option<ResponseFormat>,
    /// Declared reasoning configuration.
    pub reasoning: Option<ReasoningConfig>,
}

/// One synchronous provider turn request (`MP-5` shape, skeleton subset).
///
/// Sampling participates in `PartialEq` (declared versus undeclared are
/// different requests) but never in cache keys: see `cache_key.rs`, which
/// keys provider route plus canonical bytes only. `Eq` is intentionally
/// absent because [`SamplingParams`] holds validated `f64` fields.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRequest {
    /// Registry-known model name.
    pub model: String,
    /// Bounded conversation messages.
    pub messages: Vec<Message>,
    /// Stable Id references resolved server-side (`CP-2`).
    pub context_refs: Vec<String>,
    /// Tool Bus names validated against tool consent (`MP-5`).
    pub tools: Vec<String>,
    /// Per-turn context ceiling in bytes; excess fails with
    /// [`ProviderError::BudgetExceeded`] before provider I/O (`MP-5`).
    pub budget_bytes: usize,
    /// Caller deadline in milliseconds from `now_ms` (`MP-8`).
    pub timeout_ms: u64,
    /// Caller-supplied timestamp; no wall clock is read (`CP-7`).
    pub now_ms: u64,
    /// Optional sampling contract; `None` means undeclared and flows
    /// through untouched (never defaulted, never hashed into keys).
    pub sampling: Option<SamplingParams>,
}

impl TurnRequest {
    /// Total message payload bytes (used for the pre-I/O budget check).
    #[must_use]
    pub fn total_message_bytes(&self) -> usize {
        self.messages.iter().map(Message::byte_len).sum()
    }
}

/// Estimated token usage for one provider turn, in whole tokens.
///
/// Observed usage, not billed usage: the provider reports its best estimate
/// (or the caller supplies a deterministic script value in tests) and the
/// agent turn loop converts it to relative cost units via
/// [`crate::selection::estimate_cost`]. No currency is implied. A zero pair
/// means unreported: the turn loop falls back to a deterministic byte-based
/// estimate so an uncalibrated provider cannot bypass a cost ceiling by
/// reporting nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderUsage {
    /// Estimated input tokens observed for this turn.
    pub input_tokens: u32,
    /// Estimated output tokens observed for this turn.
    pub output_tokens: u32,
}

/// One scripted provider turn: assistant text plus follow-up tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderTurn {
    /// Assistant text for this turn.
    pub text: String,
    /// Tool calls requested after the text (bounded by the Tool Bus at
    /// dispatch).
    pub tool_calls: Vec<ToolCallRequest>,
    /// Simulated provider latency in milliseconds, checked deterministically
    /// against [`TurnRequest::timeout_ms`].
    pub latency_ms: u64,
    /// Estimated token usage observed for this turn (deterministic,
    /// caller-supplied in scripts; zero means unreported).
    pub usage: ProviderUsage,
}

/// Provider boundary errors. Every variant fails closed: no partial turn is
/// produced and (for [`FakeProvider`]) the script is not consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// `model` names no registry-known model.
    UnknownModel {
        /// Requested model name.
        name: String,
    },
    /// Request would exceed the Context Budget (`MP-5`).
    BudgetExceeded {
        /// Budget in bytes.
        limit: usize,
        /// Observed message bytes.
        actual: usize,
    },
    /// Scripted latency exceeded the caller timeout (`MP-8`).
    Timeout {
        /// Requested timeout in milliseconds.
        timeout_ms: u64,
        /// Scripted latency in milliseconds.
        latency_ms: u64,
    },
    /// Requested timeout exceeds the hard ceiling (`MP-8`).
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
    /// Transport failure before any model effect (connection reset, TLS
    /// failure, malformed envelope). No request bytes are known to have been
    /// applied; the fallback policy may advance to the next candidate.
    Transport {
        /// Owning provider id.
        provider: String,
        /// What failed (no secrets, no credentials, no key material).
        reason: String,
    },
    /// Provider refused authorization (unknown key scope, expired grant,
    /// disabled account). This variant reports the refusal only: no
    /// credential is stored, logged, or carried here, and the fallback
    /// policy stops so the operator reconciles credentials instead of
    /// spraying retries across providers.
    Auth {
        /// Owning provider id.
        provider: String,
        /// Refusal detail (no secrets, no credentials, no key material).
        reason: String,
    },
    /// Provider rate limit hit. The caller may back off for
    /// `retry_after_ms` (when known) or the fallback policy may advance to
    /// the next candidate.
    RateLimited {
        /// Owning provider id.
        provider: String,
        /// Advised wait in milliseconds; `None` when the provider gave none.
        retry_after_ms: Option<u64>,
    },
    /// Selected model lacks capabilities the request required. Selection is
    /// a routing bug or registry drift, never a reason to try another model
    /// blindly: the fallback policy stops so the caller reconciles.
    CapabilityMismatch {
        /// Owning provider id.
        provider: String,
        /// Requested model name.
        model: String,
        /// Required capabilities the model does not advertise.
        missing: Vec<ModelCapability>,
    },
    /// Model is known but unavailable (disabled deployment, region without
    /// capacity, withdrawn version). The fallback policy may advance to the
    /// next candidate.
    ModelUnavailable {
        /// Owning provider id.
        provider: String,
        /// Unavailable model name.
        model: String,
    },
    /// The effect may have happened but acknowledgement was lost. Returned
    /// by providers; the caller reconciles (status inspection or user
    /// direction) before retry and never falls back blindly (`MP-7`
    /// parity with [`crate::tool::ToolError::EffectUnknown`]).
    Unknown {
        /// Owning provider id.
        provider: String,
        /// What is uncertain.
        reason: String,
    },
    /// Declared sampling contract violates its validated ranges (`MP-5`).
    /// No value is echoed: the rejection carries no model, sampling, or
    /// host input, so logs and reconcile reports stay static.
    InvalidSampling,
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
        }
    }
}

impl std::error::Error for ProviderError {}

/// Validate a provider id (`MP-2`): non-empty, at most 64 bytes,
/// `^[a-z][a-z0-9_-]*$`.
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

/// Check the caller timeout against the hard ceiling (`MP-8`).
fn check_timeout(request: &TurnRequest) -> Result<(), ProviderError> {
    if request.timeout_ms > MAX_REQUEST_TIMEOUT_MS {
        Err(ProviderError::TimeoutTooLarge {
            max: MAX_REQUEST_TIMEOUT_MS,
            actual: request.timeout_ms,
        })
    } else {
        Ok(())
    }
}

/// Whether `value` is a finite number inside the closed interval
/// `[min, max]`. NaN and infinities are never in range.
fn in_closed_interval(value: f64, min: f64, max: f64) -> bool {
    value.is_finite() && value >= min && value <= max
}

/// Validate one declared sampling contract (`MP-5`).
///
/// Absent fields (`None`) are undeclared and skipped: they are never
/// defaulted and never validated. Every `Some` field is range-checked and a
/// single out-of-range declaration fails the whole contract with
/// [`ProviderError::InvalidSampling`] before any provider I/O.
///
/// # Errors
///
/// Returns [`ProviderError::InvalidSampling`] for any out-of-range declared
/// value, including a reasoning budget that is not strictly below the
/// completion budget when both are declared.
pub fn validate_sampling(params: &SamplingParams) -> Result<(), ProviderError> {
    if let Some(value) = params.temperature {
        if !in_closed_interval(value, MIN_TEMPERATURE, MAX_TEMPERATURE) {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.top_p {
        if !in_closed_interval(value, MIN_TOP_P, MAX_TOP_P) {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.top_k {
        if value < MIN_TOP_K {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.frequency_penalty {
        if !in_closed_interval(value, MIN_PENALTY, MAX_PENALTY) {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.presence_penalty {
        if !in_closed_interval(value, MIN_PENALTY, MAX_PENALTY) {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.repetition_penalty {
        if !in_closed_interval(value, MIN_REPETITION_PENALTY, MAX_REPETITION_PENALTY) {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.min_p {
        if !in_closed_interval(value, MIN_MIN_P, MAX_MIN_P) {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(value) = params.max_tokens {
        if value < MIN_MAX_TOKENS {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(stops) = &params.stop {
        if stops.len() > MAX_STOP_SEQUENCES
            || stops
                .iter()
                .any(|stop| stop.len() > MAX_STOP_SEQUENCE_BYTES)
        {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(ResponseFormat::JsonSchema {
        schema: Some(schema),
    }) = &params.response_format
    {
        if schema.len() > MAX_RESPONSE_SCHEMA_BYTES {
            return Err(ProviderError::InvalidSampling);
        }
    }
    if let Some(reasoning) = &params.reasoning {
        if let Some(budget) = reasoning.max_tokens {
            if budget < MIN_REASONING_TOKENS {
                return Err(ProviderError::InvalidSampling);
            }
            // Anthropic rule: when both budgets are declared, the completion
            // budget must be strictly higher than the reasoning budget.
            if let Some(max_tokens) = params.max_tokens {
                if budget >= max_tokens {
                    return Err(ProviderError::InvalidSampling);
                }
            }
        }
    }
    Ok(())
}

/// Validate the sampling contract carried by `request` (`MP-5`). Absent
/// (`None`) is undeclared and always passes; declared contracts delegate to
/// [`validate_sampling`].
///
/// # Errors
///
/// Returns [`ProviderError::InvalidSampling`] for an out-of-range declared
/// contract.
fn check_sampling(request: &TurnRequest) -> Result<(), ProviderError> {
    match &request.sampling {
        None => Ok(()),
        Some(params) => validate_sampling(params),
    }
}

/// Deterministic model-provider boundary.
pub trait ModelProvider {
    /// Registry owner/name of this provider (validated at construction).
    fn provider_id(&self) -> &str;

    /// Registry snapshot (skeleton: unfiltered; caller-scope filtering per
    /// `MP-4` is host-side and deferred).
    fn list_models(&self) -> Vec<ModelDescriptor>;

    /// Execute one synchronous turn.
    ///
    /// # Errors
    ///
    /// Fails closed with [`ProviderError`] for unknown models, budget
    /// overflow, or timeouts. No partial turn is produced.
    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError>;

    /// Scripted turns still queued (test observability; always 0 for real
    /// providers).
    fn scripted_turns_remaining(&self) -> usize;

    /// Completed `complete` calls so far (test observability).
    fn complete_calls(&self) -> u64;
}

/// Deterministic scripted provider for offline tests.
///
/// Turns are replayed FIFO from a caller-supplied script; when the script is
/// exhausted a deterministic empty-text turn with no tool calls is returned.
/// Validation failures ([`ProviderError`]) never consume the script. Sampling
/// is recorded, never enforced: [`FakeProvider`] validates the contract
/// (same as any host adapter must) and echoes the received value for
/// assertions. No network, no credentials, no wall clock.
#[derive(Debug)]
pub struct FakeProvider {
    id: String,
    models: Vec<ModelDescriptor>,
    script: VecDeque<ProviderTurn>,
    complete_calls: u64,
    last_sampling: Option<SamplingParams>,
}

impl FakeProvider {
    /// Construct a provider with one scripted model named `fake-chat`.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::InvalidProviderId`] for a malformed id.
    pub fn new(id: impl Into<String>) -> Result<Self, ProviderError> {
        let id = id.into();
        validate_provider_id(&id)?;
        Ok(Self {
            id,
            models: vec![ModelDescriptor {
                name: "fake-chat".to_owned(),
                capabilities: vec![
                    ModelCapability::Text,
                    ModelCapability::Streaming,
                    ModelCapability::ToolUse,
                ],
            }],
            script: VecDeque::new(),
            complete_calls: 0,
            last_sampling: None,
        })
    }

    /// Queue one scripted turn.
    pub fn push_turn(&mut self, turn: ProviderTurn) {
        self.script.push_back(turn);
    }

    /// Queued turns not yet consumed.
    #[must_use]
    pub fn script_len(&self) -> usize {
        self.script.len()
    }

    /// Sampling contract received by the most recent `complete` call, or
    /// `None` when the last request left sampling undeclared (or when no
    /// call has run yet). Echo only: the provider enforces nothing about
    /// the values beyond the shared pre-I/O validation.
    #[must_use]
    pub fn last_sampling(&self) -> Option<&SamplingParams> {
        self.last_sampling.as_ref()
    }
}

impl ModelProvider for FakeProvider {
    fn provider_id(&self) -> &str {
        &self.id
    }

    fn list_models(&self) -> Vec<ModelDescriptor> {
        self.models.clone()
    }

    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError> {
        check_timeout(request)?;
        check_sampling(request)?;
        if !self.models.iter().any(|m| m.name == request.model) {
            return Err(ProviderError::UnknownModel {
                name: request.model.clone(),
            });
        }
        let actual = request.total_message_bytes();
        if actual > request.budget_bytes {
            return Err(ProviderError::BudgetExceeded {
                limit: request.budget_bytes,
                actual,
            });
        }
        let turn = self.script.pop_front().unwrap_or(ProviderTurn {
            text: String::new(),
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
        });
        if turn.latency_ms > request.timeout_ms {
            self.script.push_front(turn);
            let latency_ms = self.script[0].latency_ms;
            return Err(ProviderError::Timeout {
                timeout_ms: request.timeout_ms,
                latency_ms,
            });
        }
        self.complete_calls += 1;
        self.last_sampling = request.sampling.clone();
        Ok(turn)
    }

    fn scripted_turns_remaining(&self) -> usize {
        self.script.len()
    }

    fn complete_calls(&self) -> u64 {
        self.complete_calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(model: &str, text: &str, budget: usize, timeout: u64) -> TurnRequest {
        TurnRequest {
            model: model.to_owned(),
            messages: vec![Message::user(text)],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: budget,
            timeout_ms: timeout,
            now_ms: 1_000,
            sampling: None,
        }
    }

    #[test]
    fn provider_id_shape() {
        assert!(validate_provider_id("bitty-fake").is_ok());
        assert!(validate_provider_id("a1_-b").is_ok());
        assert!(validate_provider_id("").is_err());
        assert!(validate_provider_id("1abc").is_err());
        assert!(validate_provider_id("ABC").is_err());
        // Legacy dotted vocabulary (`local.deterministic`, pre-AI-0012 slice)
        // never validates: the `MP-2` shape allows only `[a-z0-9_-]`.
        assert!(validate_provider_id("local.deterministic").is_err());
        assert!(validate_provider_id("a".repeat(65).as_str()).is_err());
    }

    #[test]
    fn terminal_metadata_mirrors_descriptor_only() {
        // R1 disposition: the terminal-side mediation placeholder carries a
        // generic name-plus-capabilities snapshot and nothing else.
        let descriptor = ModelDescriptor {
            name: "fake-chat".to_owned(),
            capabilities: vec![ModelCapability::Text, ModelCapability::ToolUse],
        };
        let mediated = TerminalModelMetadata::from_descriptor(&descriptor);
        assert_eq!(
            mediated,
            TerminalModelMetadata {
                name: "fake-chat".to_owned(),
                capabilities: vec![ModelCapability::Text, ModelCapability::ToolUse],
            }
        );
        // The snapshot is detached: later helper-side changes do not leak
        // through the mediation view.
        let mut changed = descriptor.clone();
        changed.capabilities.push(ModelCapability::Streaming);
        assert_ne!(TerminalModelMetadata::from_descriptor(&changed), mediated);
    }

    #[test]
    fn mediation_view_derives_from_helper_registry() {
        // Mediation flows one way: helper-side `list_models` snapshots feed
        // the terminal-side view; the view never drives provider behavior.
        let provider = FakeProvider::new("bitty-fake").expect("valid id");
        let mediated: Vec<TerminalModelMetadata> = provider
            .list_models()
            .iter()
            .map(TerminalModelMetadata::from_descriptor)
            .collect();
        assert_eq!(mediated.len(), 1);
        assert_eq!(mediated[0].name, "fake-chat");
        assert!(mediated[0].capabilities.contains(&ModelCapability::Text));
    }

    #[test]
    fn script_replays_fifo_and_leaves_remainder() {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(ProviderTurn {
            text: "first".to_owned(),
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
        });
        provider.push_turn(ProviderTurn {
            text: "second".to_owned(),
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
        });
        let req = request("fake-chat", "hi", 4096, 5_000);
        assert_eq!(
            provider.complete(&req).expect("turn").text,
            "first".to_owned()
        );
        assert_eq!(provider.scripted_turns_remaining(), 1);
        assert_eq!(provider.complete_calls(), 1);
    }

    #[test]
    fn exhausted_script_returns_empty_turn() {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        let turn = provider
            .complete(&request("fake-chat", "hi", 4096, 5_000))
            .expect("default turn");
        assert!(turn.text.is_empty() && turn.tool_calls.is_empty());
    }

    #[test]
    fn failures_do_not_consume_script() {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(ProviderTurn {
            text: "kept".to_owned(),
            tool_calls: Vec::new(),
            latency_ms: 0,
            usage: ProviderUsage::default(),
        });
        let over = request("fake-chat", "hi", 1, 5_000);
        assert!(matches!(
            provider.complete(&over),
            Err(ProviderError::BudgetExceeded { .. })
        ));
        assert_eq!(provider.scripted_turns_remaining(), 1);
        let unknown = request("nope", "hi", 4096, 5_000);
        assert!(matches!(
            provider.complete(&unknown),
            Err(ProviderError::UnknownModel { .. })
        ));
        assert_eq!(provider.scripted_turns_remaining(), 1);
    }

    #[test]
    fn timeout_is_deterministic() {
        let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
        provider.push_turn(ProviderTurn {
            text: "slow".to_owned(),
            tool_calls: Vec::new(),
            latency_ms: 6_000,
            usage: ProviderUsage::default(),
        });
        let err = provider
            .complete(&request("fake-chat", "hi", 4096, 5_000))
            .expect_err("must time out");
        assert!(matches!(err, ProviderError::Timeout { .. }));
        assert_eq!(provider.scripted_turns_remaining(), 1);
        let too_big = request("fake-chat", "hi", 4096, 60_000);
        assert!(matches!(
            provider.complete(&too_big),
            Err(ProviderError::TimeoutTooLarge { .. })
        ));
    }
}

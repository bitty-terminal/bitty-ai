//! Deterministic single-agent runtime skeleton (`bitty-ai-runtime`).
//!
//! **Status: skeleton, draft scope (AI-0009).** This crate implements the
//! single-crate runtime skeleton authorized by `bitty-ai` Issue #11: a
//! deterministic single-agent loop over the
//! `provider` / `context` / `session` / `tool` / `agent` / `stream` / `bridge`
//! / `prompt` modules with a scripted [`provider::FakeProvider`], fail-closed
//! authorization hooks, and structured execution outcomes including `Unknown`.
//!
//! The crate tracks the draft `implementation-profile-v0.1.md` (status:
//! draft, not an accepted contract; a draft disposition proposes no accepted
//! architecture), so nothing here claims conformance to it. Normative
//! architecture stays in the canonical `docs/specifications/` corpus
//! (`ai-architecture.md`, `context-management.md`,
//! `command-tool-architecture.md`, `agent-coordination.md`,
//! `implementation-profile-v0.1.md` plus the `R1`..`R6` draft dispositions);
//! bound values are
//! annotated with the draft rule they mirror (`MP-*`, `CP-*`, `TB-*`, `RS-*`,
//! `AG-*`, `FS-AI*`) and remain skeleton defaults until an accepted profile
//! says otherwise.
//!
//! ## What is working vs stub
//!
//! - Working: deterministic turn loop, cancellation at dispatch and chunk
//!   boundaries, two-phase tool validation (transactional denial per turn),
//!   L0 structured results plus L1 dedupe/supersede/externalize assembly,
//!   bounded budgets with counted truncation, sequenced stream fragments,
//!   deny-by-default authorization hooks, P1 wire-level protocol-to-instance
//!   bridge mapping plus consent-ledger seam with deny-by-default and test
//!   double.
//! - Stub (host-owned, deliberately absent): network model providers,
//!   credential handling, real consent ledger, capability enforcement, MCP
//!   transport, persistence, multi-agent/teams, LSP wiring. L2+ compression
//!   exists only as the fake-verified [`compression`] prototype (host
//!   summarizer seam, deterministic breakpoints, in-memory retention); the
//!   durable store stays out of scope (AI-0049) and AIQ-11 stays open.
//!   [`tool::ToolExecutor`] and [`tool::ToolAuthorizer`] are the seams where
//!   the host plugs those in; without them the crate refuses (`FS-AI7`).
//!   [`bridge::ConsentLedger`] is the consent seam: [`bridge::DenyAllConsent`]
//!   denies by default and [`bridge::FakeConsentLedger`] is the deterministic
//!   test double; the real ledger lives host-side.
//!
//! ## Determinism rules
//!
//! Every operation takes a caller-supplied `now_ms`. There is no wall clock,
//! thread, async runtime, network, filesystem, or secret. All behavior in
//! tests is reproducible from the seed script plus `now_ms`.

#![deny(unsafe_code)]

pub mod adoption;
pub mod agent;
pub mod bridge;
pub mod cache_key;
pub mod compression;
pub mod context;
pub mod extension;
pub mod prompt;
pub mod provider;
pub mod reconcile;
pub mod selection;
pub mod session;
pub mod stream;
pub mod tool;

pub use adoption::{
    AdoptedEffect, AdoptedHistory, AdoptionClaim, AdoptionRefusal, ClaimedUnknownEffect,
    MAX_ADOPTION_SURVIVORS, UnknownDisposition, check_adoption,
};
pub use agent::{
    Agent, AgentConfig, AgentError, ExecOutcome, ExecutionRecord, MAX_EXECUTIONS_PER_AGENT,
};
pub use bridge::{
    BridgeError, ConsentDecision, ConsentLedger, ConsentQuery, DenyAllConsent, FakeConsentLedger,
    IdentityBridge, MAX_CONSENT_GRANTS, MAX_CONSENT_SCOPE_LEN, MAX_PROTOCOL_ID_LEN,
    MAX_PROTOCOL_ID_SEGMENT_LEN, ProtocolAgentId, ensure_consented, validate_protocol_id,
};
pub use cache_key::{CacheKey, CacheKeyError, CacheScope};
pub use compression::{
    CompressedSpan, CompressedView, CompressionConfig, CompressionError, DEFAULT_MAX_SPAN_BYTES,
    FakeSummarizer, MAX_SPAN_ID_LEN, MAX_SPANS, RetentionClass, RetentionPolicy, RetentionTags,
    SpanRange, SummarizeInput, Summarizer, compress_records, inherit_retention, select_breakpoints,
};
pub use context::{
    ArtifactRef, ArtifactStore, AssembledContent, AssembledContext, AssembledRecord, ContextError,
    ContextPriority, ContextRecord, ContextRequest, DetailLevel, RecordBody, StableId, assemble,
    validate_stable_id,
};
pub use extension::{
    AgentPoint, CommandPoint, CompactorPoint, ContextPoint, ExtensionPoint, Manifest, MemoryPoint,
    ModelPoint, ToolPoint, UiPoint,
};
pub use prompt::assemble as assemble_prompt;
pub use prompt::{
    AssembledPrompt, AssembledSection, Directive, DirectiveOverride, LayerInput,
    MAX_BUDGET_CEILING_BYTES, MAX_CANONICAL_BYTES, MAX_CORE_VERSION_LEN, MAX_DIRECTIVE_KEY_LEN,
    MAX_DIRECTIVE_VALUE_LEN, MAX_DIRECTIVES_PER_LAYER, MAX_LAYER_TEXT_BYTES,
    MAX_PROJECT_FILE_BYTES, MAX_PROJECT_FILES, MAX_PROJECT_PATH_LEN, MAX_SCOPE_LEN,
    MAX_SCOPES_PER_LAYER, MAX_SKILL_ENTRIES, MAX_SKILL_ENTRY_BYTES, MAX_SKILL_NAME_LEN,
    MAX_SKILL_REGISTRY_BYTES, MAX_SKILL_VERSION_LEN, MAX_TOOL_ENTRIES_PER_LAYER,
    NEVER_MERGE_DIRECTIVE_KEYS, PromptError, PromptLayer, PromptSnapshot, SKILL_REGISTRY_VERSION_1,
    SUPPORTED_SKILL_VERSIONS, admit_project_path, check_dispatch, common_prefix_len,
    is_dispatch_allowed, is_never_merge_directive_key, is_supported_skill_version,
    validate_budget_ceiling, validate_core_version, validate_directive_key,
    validate_directive_value, validate_layer_text, validate_prompt_tool_name, validate_scope,
};
pub use provider::{
    FakeProvider, MAX_MIN_P, MAX_PENALTY, MAX_REPETITION_PENALTY, MAX_RESPONSE_SCHEMA_BYTES,
    MAX_STOP_SEQUENCE_BYTES, MAX_STOP_SEQUENCES, MAX_TEMPERATURE, MAX_TOP_P, MIN_MAX_TOKENS,
    MIN_MIN_P, MIN_PENALTY, MIN_REASONING_TOKENS, MIN_REPETITION_PENALTY, MIN_TEMPERATURE,
    MIN_TOP_K, MIN_TOP_P, ModelCapability, ModelDescriptor, ModelProvider, ProviderError,
    ProviderTurn, ProviderUsage, REASONING_RATIO_HIGH, REASONING_RATIO_LOW, REASONING_RATIO_MAX,
    REASONING_RATIO_MEDIUM, REASONING_RATIO_MINIMAL, REASONING_RATIO_XHIGH, ReasoningConfig,
    ReasoningEffort, ResponseFormat, Role, SamplingParams, TerminalModelMetadata, ToolCallRequest,
    TurnRequest, UnknownReasoningEffort, validate_provider_id, validate_sampling,
};
pub use reconcile::{
    DEFAULT_MAX_UNKNOWN_RETRIES, DEFAULT_RECONCILE_BASE_DELAY_MS, DEFAULT_RECONCILE_MAX_DELAY_MS,
    FakeReconciler, MAX_RECONCILE_ATTEMPTS, MAX_RECONCILE_DELAY_MS, MAX_RECONCILE_REASON_BYTES,
    ReconcileConfig, ReconcileOutcome, ReconcileStatus, UnknownEscalation, UnknownReconciler,
    reconcile_delay_ms,
};
pub use selection::{
    FallbackDirective, MAX_ALIAS_CANDIDATES, MAX_ALIASES, MAX_MODEL_NAME_LEN,
    MAX_REGISTERED_MODELS, ModelAlias, ModelRef, ModelRegistration, ProviderRegistry,
    RegisteredModel, SelectRequest, SelectedModel, SelectionError, estimate_cost,
    fallback_directive, validate_model_name,
};
pub use session::{
    AgentInstanceId, AgentLevel, AgentSession, DenyAllElevations, ElevationGrant, ExecutionId,
    IdIssuer, RunId, SessionError, SessionId, SessionState,
};
pub use stream::{
    Fragment, FragmentKind, StreamChunk, StreamError, StreamSink, VecSink, emit_fragments,
    fragment_text, validate_chunk,
};
pub use tool::{
    AuthBase, AuthContext, AuthDecision, DenyAllAuthorizer, FakeToolExecutor, ToolAuthorizer,
    ToolBus, ToolCall, ToolError, ToolExecution, ToolExecutor, ToolRegistry, ToolSpec, ToolStatus,
    ToolSuccess, validate_tool_name,
};

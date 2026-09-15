//! Deterministic single-agent runtime skeleton (`bitty-ai-runtime`).
//!
//! **Status: skeleton, draft scope (AI-0009).** This crate implements the
//! single-crate runtime skeleton authorized by `bitty-ai` Issue #11: a
//! deterministic single-agent loop over the
//! `provider` / `context` / `session` / `tool` / `agent` / `stream` / `bridge`
//! modules with a scripted [`provider::FakeProvider`], fail-closed
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
//!   transport, persistence, multi-agent/teams, L2+ compaction, LSP wiring.
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

pub mod agent;
pub mod bridge;
pub mod context;
pub mod provider;
pub mod session;
pub mod stream;
pub mod tool;

pub use agent::{Agent, AgentConfig, AgentError, ExecOutcome, ExecutionRecord};
pub use bridge::{
    BridgeError, ConsentDecision, ConsentLedger, ConsentQuery, DenyAllConsent, FakeConsentLedger,
    IdentityBridge, MAX_CONSENT_GRANTS, MAX_CONSENT_SCOPE_LEN, MAX_PROTOCOL_ID_LEN,
    MAX_PROTOCOL_ID_SEGMENT_LEN, ProtocolAgentId, ensure_consented, validate_protocol_id,
};
pub use context::{
    ArtifactRef, ArtifactStore, AssembledContent, AssembledContext, AssembledRecord, ContextError,
    ContextPriority, ContextRecord, ContextRequest, DetailLevel, RecordBody, StableId, assemble,
    validate_stable_id,
};
pub use provider::{
    FakeProvider, ModelCapability, ModelDescriptor, ModelProvider, ProviderError, ProviderTurn,
    Role, TerminalModelMetadata, ToolCallRequest, TurnRequest, validate_provider_id,
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

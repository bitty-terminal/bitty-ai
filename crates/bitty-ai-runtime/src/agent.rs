//! Deterministic single-agent turn loop.
//!
//! The loop drives one agent at one tier: assemble context (L0 + L1),
//! request a provider turn, stream text fragments, validate and authorize
//! the requested tool calls as a batch (transactional denial per turn), and
//! dispatch each through the Tool Bus with cancellation checked before every
//! dispatch, between stream chunks, and between rounds (`MP-7`).
//!
//! Outcomes are structured: `Completed`, `Failed` with a typed error,
//! `Canceled` with reconciled effect counts, or `Unknown` when an effect may
//! have happened without acknowledgement. `Unknown` (and any cancel that
//! lands after a dispatch) requires reconciliation before retry; the loop
//! never blindly retries and never rolls back started effects.
//!
//! Out of scope by design: multi-agent/teams/delegation, persistence,
//! L2+ compaction, and host consent/capability machinery (the hooks stay
//! deny-by-default).

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::context::{
    ArtifactStore, AssembledContent, ContextError, ContextRecord, ContextRequest, RecordBody,
    StableId, assemble,
};
use crate::provider::{
    DEFAULT_CONTEXT_BUDGET_BYTES, DEFAULT_REQUEST_TIMEOUT_MS, Message, ModelProvider,
    ProviderError, ProviderTurn, TurnRequest,
};
use crate::selection::estimate_cost;
use crate::session::{AgentSession, ExecutionId, IdIssuer, SessionError};
use crate::stream::{FragmentKind, StreamError, StreamSink, emit_fragments, fragment_text};
use crate::tool::{
    AuthBase, MAX_TOOL_CALLS_PER_TURN, ToolBus, ToolError, ToolExecution, ToolExecutor, ToolStatus,
};

/// Skeleton liveness bound: maximum provider rounds per turn. A turn that
/// keeps requesting tools past this bound fails with
/// [`AgentError::RoundLimitExceeded`] rather than looping forever.
pub const DEFAULT_MAX_ROUNDS: usize = 4;

/// Turn-loop configuration. All values are skeleton defaults with documented
/// provenance, not accepted contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// Maximum provider rounds per turn.
    pub max_rounds: usize,
    /// Maximum tool calls accepted per assistant turn (`TB-6`).
    pub max_tool_calls_per_turn: usize,
    /// Context byte budget for turn assembly (`CP-5` candidate default).
    pub context_budget_bytes: usize,
    /// Provider timeout per round in milliseconds (`MP-8`).
    pub provider_timeout_ms: u64,
    /// Per-turn cost ceiling in relative routing units (see
    /// [`crate::selection::estimate_cost`); `None` disables the fuse and
    /// preserves the pre-AI-0046 behavior. When set, the turn loop
    /// accumulates estimated cost across provider rounds and stops with
    /// [`AgentError::CostCeilingExceeded`] before dispatching further work.
    /// Routing data only: never authorizes or bypasses the byte budget or
    /// authorization gates.
    pub max_turn_cost: Option<u64>,
    /// Relative input cost weight for the active model (mirror the
    /// [`crate::selection::SelectedModel`] weights at wiring time; `0`
    /// counts as `1` in turn accounting so an uncalibrated host cannot
    /// bypass the ceiling; never currency).
    pub input_cost_weight: u32,
    /// Relative output cost weight for the active model (same rule as
    /// [`AgentConfig::input_cost_weight`]).
    pub output_cost_weight: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_rounds: DEFAULT_MAX_ROUNDS,
            max_tool_calls_per_turn: MAX_TOOL_CALLS_PER_TURN,
            context_budget_bytes: DEFAULT_CONTEXT_BUDGET_BYTES,
            provider_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            max_turn_cost: None,
            input_cost_weight: 1,
            output_cost_weight: 1,
        }
    }
}

/// Turn-loop errors. Denial and budget failures are total: the failing turn
/// produced no dispatch of its own (effects from earlier rounds are recorded
/// on the agent, never hidden).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// Provider boundary failure (unknown model, budget, timeout).
    Provider(ProviderError),
    /// Context assembly failure.
    Context(ContextError),
    /// Tool validation, authorization, cap, or bound failure.
    Tool(ToolError),
    /// Stream framing or bound failure.
    Stream(StreamError),
    /// Session lifecycle failure.
    Session(SessionError),
    /// The provider kept requesting tools past the round bound.
    RoundLimitExceeded {
        /// Configured bound.
        limit: usize,
    },
    /// Accumulated estimated turn cost exceeded the configured ceiling.
    /// Fail-closed fuse: the turn stops before dispatching further work
    /// (never mid-effect); the session stays `Active` for reconcile-and-retry,
    /// same philosophy as `Unknown`/cancel, with no rollback of earlier
    /// recorded rounds.
    CostCeilingExceeded {
        /// Configured ceiling in relative routing units.
        limit: u64,
        /// Accumulated estimated cost when the fuse tripped.
        actual: u64,
    },
}

impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Provider(error) => write!(f, "provider: {error}"),
            Self::Context(error) => write!(f, "context: {error}"),
            Self::Tool(error) => write!(f, "tool: {error}"),
            Self::Stream(error) => write!(f, "stream: {error}"),
            Self::Session(error) => write!(f, "session: {error}"),
            Self::RoundLimitExceeded { limit } => {
                write!(f, "round limit of {limit} exceeded")
            }
            Self::CostCeilingExceeded { limit, actual } => {
                write!(f, "turn cost {actual} exceeded ceiling {limit}")
            }
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Context(error) => Some(error),
            Self::Tool(error) => Some(error),
            Self::Stream(error) => Some(error),
            Self::Session(error) => Some(error),
            Self::RoundLimitExceeded { .. } | Self::CostCeilingExceeded { .. } => None,
        }
    }
}

impl From<ProviderError> for AgentError {
    fn from(error: ProviderError) -> Self {
        Self::Provider(error)
    }
}

impl From<ContextError> for AgentError {
    fn from(error: ContextError) -> Self {
        Self::Context(error)
    }
}

impl From<ToolError> for AgentError {
    fn from(error: ToolError) -> Self {
        Self::Tool(error)
    }
}

impl From<StreamError> for AgentError {
    fn from(error: StreamError) -> Self {
        Self::Stream(error)
    }
}

impl From<SessionError> for AgentError {
    fn from(error: SessionError) -> Self {
        Self::Session(error)
    }
}

/// One attributed tool dispatch owned by this turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRecord {
    /// Attribution handle.
    pub execution_id: ExecutionId,
    /// Tool name.
    pub tool: String,
    /// Recorded status (including `Unknown`).
    pub status: ToolStatus,
}

/// Structured turn outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecOutcome {
    /// Turn finished with final assistant text.
    Completed {
        /// Final text.
        text: String,
    },
    /// Turn failed with a typed error (no dispatch of its own).
    Failed {
        /// Typed cause.
        error: AgentError,
    },
    /// Cancel landed with all dispatched effects reconciled.
    Canceled {
        /// Dispatched effects with known status.
        dispatched: usize,
        /// Always 0 here; kept so shapes match `Unknown`.
        unknown: usize,
    },
    /// An effect may have happened without acknowledgement: reconcile
    /// (status inspection or user direction) before retry.
    Unknown {
        /// What needs reconciling.
        reason: String,
        /// Dispatched effects so far.
        dispatched: usize,
    },
}

/// Deterministic single-agent runtime.
pub struct Agent<P: ModelProvider> {
    provider: P,
    tools: ToolBus,
    session: AgentSession,
    artifacts: ArtifactStore,
    ids: IdIssuer,
    config: AgentConfig,
    executions: Vec<ExecutionRecord>,
    tool_history: Vec<ContextRecord>,
    turn_cost: u64,
}

impl<P: ModelProvider> Agent<P> {
    /// Wire a provider, tool bus, session, and config into one runtime.
    pub fn new(provider: P, tools: ToolBus, session: AgentSession, config: AgentConfig) -> Self {
        Self {
            provider,
            tools,
            session,
            artifacts: ArtifactStore::new(),
            ids: IdIssuer::default(),
            config,
            executions: Vec::new(),
            tool_history: Vec::new(),
            turn_cost: 0,
        }
    }

    /// Request cancellation (idempotent, visible at the next dispatch or
    /// chunk boundary).
    pub fn cancel(&self) {
        self.session.cancel();
    }

    /// Bound session handle (shares cancellation and lifecycle).
    #[must_use]
    pub fn session(&self) -> &AgentSession {
        &self.session
    }

    /// Attributed executions recorded by this turn so far.
    #[must_use]
    pub fn executions(&self) -> &[ExecutionRecord] {
        &self.executions
    }

    /// Mutable provider access (test observability: script remainder, call
    /// counts).
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    /// Structured tool-result records accumulated this turn (L0 shape, for
    /// host drill-down or test inspection).
    #[must_use]
    pub fn tool_records(&self) -> &[ContextRecord] {
        &self.tool_history
    }

    /// Accumulated estimated turn cost in relative routing units for the
    /// current (or most recent) turn. Reset to `0` at the start of every
    /// [`Agent::run_turn`]; advanced once per successful provider round.
    #[must_use]
    pub fn turn_cost(&self) -> u64 {
        self.turn_cost
    }

    /// Run one turn: assemble seed context, loop provider rounds with tool
    /// dispatch, and stream fragments into `sink`.
    ///
    /// Cancellation returns [`ExecOutcome::Canceled`] (or
    /// [`ExecOutcome::Unknown`] when dispatched effects are unreconciled);
    /// budget, validation, and authorization failures return
    /// [`ExecOutcome::Failed`] with no dispatch of their own. The cost fuse
    /// ([`AgentError::CostCeilingExceeded`]) is the exception to the
    /// `Failed`-means-terminal rule: it returns `Failed` with the session
    /// left `Active` for reconcile-and-retry, like `Unknown`/cancel, with no
    /// rollback of earlier recorded rounds. Cost accounting never authorizes
    /// or bypasses the byte budget or authorization gates.
    pub fn run_turn(
        &mut self,
        executor: &mut dyn ToolExecutor,
        model: &str,
        prompt: &str,
        seed_records: &[ContextRecord],
        sink: &mut dyn StreamSink,
        now_ms: u64,
    ) -> ExecOutcome {
        if self.session.is_cancelled() {
            return ExecOutcome::Canceled {
                dispatched: 0,
                unknown: 0,
            };
        }
        let request = ContextRequest {
            max_tokens: None,
            max_bytes: Some(self.config.context_budget_bytes as u64),
            priority: crate::context::ContextPriority::Normal,
            detail: crate::context::DetailLevel::Standard,
        };
        let assembled = match assemble(seed_records, &mut self.artifacts, &request) {
            Ok(assembled) => assembled,
            Err(error) => return self.fail(error.into()),
        };
        let mut messages: Vec<Message> = assembled
            .records
            .iter()
            .map(|record| {
                let payload = match &record.content {
                    AssembledContent::Inline(bytes) => String::from_utf8_lossy(bytes),
                    AssembledContent::Reference(reference) => format!("see {reference}").into(),
                };
                Message::tool(format!(
                    "[{}:{}] {}\n{payload}",
                    record.provider, record.id, record.summary
                ))
            })
            .collect();
        messages.push(Message::user(prompt));
        let context_refs = assembled.context_refs.clone();

        self.tools.begin_turn();
        let base = AuthBase {
            agent_instance_id: self.session.agent_instance_id(),
            session_id: self.session.session_id(),
            level: self.session.level(),
        };
        let tool_names = self.tools.tool_names();
        let mut round = 0;
        self.tool_history.clear();
        self.turn_cost = 0;
        loop {
            if self.session.is_cancelled() {
                return self.reconcile_cancel();
            }
            if round >= self.config.max_rounds {
                return self.fail(AgentError::RoundLimitExceeded {
                    limit: self.config.max_rounds,
                });
            }
            round += 1;
            let turn_request = crate::provider::TurnRequest {
                model: model.to_owned(),
                messages: messages.clone(),
                context_refs: context_refs.clone(),
                tools: tool_names.clone(),
                budget_bytes: self.config.context_budget_bytes,
                timeout_ms: self.config.provider_timeout_ms,
                now_ms,
            };
            let turn = match self.provider.complete(&turn_request) {
                Ok(turn) => turn,
                Err(error) => return self.fail(error.into()),
            };
            // Cost fuse: accumulate estimated cost for this round, then stop
            // before any further effect (no text emission, no tool dispatch
            // for this round) when the ceiling is exceeded. The session stays
            // `Active`: earlier recorded rounds are kept, nothing is rolled
            // back, and the caller may reconcile (raise the ceiling, switch
            // models) and retry.
            let (input_tokens, output_tokens) = Self::round_usage(&turn_request, &turn);
            let round_cost = estimate_cost(
                input_tokens,
                output_tokens,
                effective_cost_weight(self.config.input_cost_weight),
                effective_cost_weight(self.config.output_cost_weight),
            );
            self.turn_cost = self.turn_cost.saturating_add(round_cost);
            if let Some(limit) = self.config.max_turn_cost {
                if self.turn_cost > limit {
                    return ExecOutcome::Failed {
                        error: AgentError::CostCeilingExceeded {
                            limit,
                            actual: self.turn_cost,
                        },
                    };
                }
            }
            match self.emit_text(sink, &turn) {
                Ok(true) => {}
                Ok(false) => return self.reconcile_cancel(),
                Err(error) => return self.fail(error.into()),
            }
            if turn.tool_calls.is_empty() {
                self.session.finish(false);
                return ExecOutcome::Completed { text: turn.text };
            }
            if turn.tool_calls.len() > self.config.max_tool_calls_per_turn {
                return self.fail(
                    ToolError::CallLimitExceeded {
                        limit: self.config.max_tool_calls_per_turn,
                    }
                    .into(),
                );
            }
            let calls: Vec<crate::tool::ToolCall> = turn
                .tool_calls
                .iter()
                .map(|call| crate::tool::ToolCall {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect();
            // Transactional gate: authorize everything before dispatching
            // anything (FS-AI1).
            if let Err(error) = self.tools.precheck(&calls, &base) {
                return self.fail(error.into());
            }
            messages.push(Message::assistant(turn.text.clone()));
            for call in &calls {
                if self.session.is_cancelled() {
                    return self.reconcile_cancel();
                }
                let execution_id = self.ids.execution();
                match self
                    .tools
                    .dispatch(executor, call, &base, execution_id, now_ms)
                {
                    Ok(execution) => {
                        let unknown = matches!(execution.status, ToolStatus::Unknown { .. });
                        let denied = matches!(execution.status, ToolStatus::Denied { .. });
                        self.record_execution(&execution, now_ms);
                        messages.push(self.execution_message(&execution));
                        match self.emit_tool_card(sink, &execution) {
                            Ok(true) => {}
                            Ok(false) => return self.reconcile_cancel(),
                            Err(error) => return self.fail(error.into()),
                        }
                        if unknown {
                            return self.unknown(
                                "tool effect uncertain; reconcile before retry",
                                execution.tool.clone(),
                            );
                        }
                        if denied {
                            let reason = match &execution.status {
                                ToolStatus::Denied { reason } => reason.clone(),
                                _ => unreachable!("matched Denied above"),
                            };
                            return self.fail(
                                ToolError::Denied {
                                    name: execution.tool.clone(),
                                    reason,
                                }
                                .into(),
                            );
                        }
                    }
                    Err(error) => {
                        self.executions.push(ExecutionRecord {
                            execution_id,
                            tool: call.name.clone(),
                            status: ToolStatus::Failed {
                                reason: error.to_string(),
                            },
                        });
                        return self.fail(error.into());
                    }
                }
            }
        }
    }

    /// Stream assistant text. Returns `Ok(false)` when cancellation stopped
    /// emission (caller reconciles).
    ///
    /// # Errors
    ///
    /// Returns [`StreamError`] for oversized or misframed chunks.
    fn emit_text(
        &mut self,
        sink: &mut dyn StreamSink,
        turn: &ProviderTurn,
    ) -> Result<bool, StreamError> {
        if turn.text.is_empty() {
            return Ok(!self.session.is_cancelled());
        }
        let fragments = fragment_text(FragmentKind::Markdown, &turn.text);
        let session = &self.session;
        emit_fragments(sink, &fragments, &|| session.is_cancelled())
    }

    /// Stream one ToolCard fragment for a recorded execution.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError`] for oversized or misframed chunks.
    fn emit_tool_card(
        &mut self,
        sink: &mut dyn StreamSink,
        execution: &ToolExecution,
    ) -> Result<bool, StreamError> {
        let status = match &execution.status {
            ToolStatus::Success => "ok",
            ToolStatus::Failed { .. } => "failed",
            ToolStatus::Denied { .. } => "denied",
            ToolStatus::Unknown { .. } => "unknown",
        };
        let card = format!(
            "tool={} status={} summary={}",
            execution.tool, status, execution.summary
        );
        let fragments = fragment_text(FragmentKind::ToolCard, &card);
        let session = &self.session;
        emit_fragments(sink, &fragments, &|| session.is_cancelled())
    }

    /// Record one execution as an attributed record plus an L0
    /// [`ContextRecord`] (large payloads externalize to the artifact store).
    fn record_execution(&mut self, execution: &ToolExecution, now_ms: u64) {
        self.executions.push(ExecutionRecord {
            execution_id: execution.execution_id,
            tool: execution.tool.clone(),
            status: execution.status.clone(),
        });
        let body = if execution.data.len() > crate::context::EXTERNALIZE_THRESHOLD_BYTES {
            match self.artifacts.store(execution.data.clone()) {
                Ok(reference) => RecordBody::Artifact(reference),
                Err(_) => RecordBody::Inline(Vec::new()),
            }
        } else {
            RecordBody::Inline(execution.data.clone())
        };
        // Ids are turn-scoped and deterministic; StableId validation cannot
        // fail for this constant, but fail closed rather than invent one.
        if let Ok(owner) = StableId::new("agent-turn") {
            self.tool_history.push(ContextRecord {
                id: format!("exec-{}", execution.execution_id.0),
                provider: "workspace".to_owned(),
                owner,
                generation: self.session.generation(),
                collected_at_ms: now_ms,
                priority: crate::context::ContextPriority::Normal,
                summary: format!("tool={} {}", execution.tool, execution.summary),
                body,
                supersedes: None,
                is_untrusted_surface: true,
            });
        }
    }

    /// Render one execution as a tool message for the next provider round.
    fn execution_message(&self, execution: &ToolExecution) -> Message {
        let status = match &execution.status {
            ToolStatus::Success => "ok".to_owned(),
            ToolStatus::Failed { reason } => format!("failed: {reason}"),
            ToolStatus::Denied { reason } => format!("denied: {reason}"),
            ToolStatus::Unknown { reason } => format!("unknown: {reason}"),
        };
        Message::tool(format!(
            "{} -> {status}: {}",
            execution.tool, execution.summary
        ))
    }

    /// Cancel landed with dispatched effects: clean cancel when all are
    /// known, `Unknown` when any needs reconciling (`MP-7`).
    fn reconcile_cancel(&mut self) -> ExecOutcome {
        let unknown = self
            .executions
            .iter()
            .filter(|record| matches!(record.status, ToolStatus::Unknown { .. }))
            .count();
        if unknown > 0 {
            ExecOutcome::Unknown {
                reason: format!(
                    "cancelled with {unknown} unreconciled effects; reconcile before retry"
                ),
                dispatched: self.executions.len(),
            }
        } else {
            ExecOutcome::Canceled {
                dispatched: self.executions.len(),
                unknown: 0,
            }
        }
    }

    /// Fail the turn with a typed error and mark the session failed.
    ///
    /// The cost fuse ([`AgentError::CostCeilingExceeded`]) deliberately does
    /// not use this path: it returns `Failed` with the session left `Active`.
    fn fail(&mut self, error: AgentError) -> ExecOutcome {
        debug_assert!(
            !matches!(error, AgentError::CostCeilingExceeded { .. }),
            "cost fuse must leave the session Active; return Failed directly"
        );
        self.session.finish(true);
        ExecOutcome::Failed { error }
    }

    /// Record an unreconciled effect without failing the session: the turn
    /// stops, the session stays usable for reconciliation and retry.
    fn unknown(&mut self, reason: &str, tool: String) -> ExecOutcome {
        ExecOutcome::Unknown {
            reason: format!("{reason} (tool {tool})"),
            dispatched: self.executions.len(),
        }
    }

    /// Resolve the token pair used for cost accounting for one provider
    /// round: the provider-reported [`crate::provider::ProviderUsage`] when
    /// either count is non-zero, else a deterministic byte-based estimate
    /// (input from the request bytes, output from the response text) so an
    /// unreported usage cannot bypass the ceiling.
    fn round_usage(request: &TurnRequest, turn: &ProviderTurn) -> (u64, u64) {
        if turn.usage.input_tokens != 0 || turn.usage.output_tokens != 0 {
            return (
                u64::from(turn.usage.input_tokens),
                u64::from(turn.usage.output_tokens),
            );
        }
        (
            ContextRequest::estimate_tokens(request.total_message_bytes()),
            ContextRequest::estimate_tokens(turn.text.len()),
        )
    }
}

/// Effective cost weight for turn accounting: a configured `0` (unset)
/// counts as `1` so an uncalibrated host cannot bypass the ceiling with free
/// rounds. Selection routing still treats `0` as smallest/unset; this mapping
/// applies only to the turn-loop multiplication.
fn effective_cost_weight(weight: u32) -> u32 {
    if weight == 0 { 1 } else { weight }
}

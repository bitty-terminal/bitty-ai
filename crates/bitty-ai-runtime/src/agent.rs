//! Deterministic single-agent turn loop.
//!
//! The loop drives one agent at one tier: assemble context (L0 + L1),
//! request a provider turn, stream text fragments, admit the requested tool
//! calls as a whole batch against the logical-turn call budget
//! (all-or-nothing admission; a batch that does not fit the remaining
//! configured or hard allowance dispatches nothing), and dispatch each
//! through the Tool Bus with cancellation checked before every dispatch,
//! between stream chunks, and between rounds (`MP-7`).
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
use crate::reconcile::{
    DEFAULT_MAX_UNKNOWN_RETRIES, DEFAULT_RECONCILE_BASE_DELAY_MS, DEFAULT_RECONCILE_MAX_DELAY_MS,
    ReconcileConfig, ReconcileOutcome, ReconcileStatus, UnknownEscalation, UnknownReconciler,
    bound_reason, reconcile_delay_ms,
};
use crate::selection::estimate_cost;
use crate::session::{AgentSession, ExecutionId, IdIssuer, SessionError, SessionState};
use crate::stream::{
    Fragment, FragmentKind, StreamError, StreamSink, emit_fragments, fragment_text,
};
use crate::tool::{
    AuthBase, MAX_TOOL_CALLS_PER_TURN, ResultDisposition, ToolBus, ToolError, ToolExecution,
    ToolExecutor, ToolStatus,
};

/// Skeleton liveness bound: maximum provider rounds per turn. A turn that
/// keeps requesting tools past this bound fails with
/// [`AgentError::RoundLimitExceeded`] rather than looping forever.
pub const DEFAULT_MAX_ROUNDS: usize = 4;

/// Absolute ceiling on attributed executions retained per turn
/// (`AI-0057/P1-1`). `executions` is cleared at every [`Agent::run_turn`]
/// entry so cancel/reconcile counts never mix history; this cap bounds a
/// single turn even when [`AgentConfig::max_rounds`] or
/// [`AgentConfig::max_tool_calls_per_turn`] are raised above the defaults.
/// The default `4` rounds `* 8` calls (`TB-6`) equals `32`, so default
/// configurations never trip it. Exceeding it fails the turn with
/// [`ToolError::CallLimitExceeded`].
pub const MAX_EXECUTIONS_PER_AGENT: usize = 32;

/// Turn-loop configuration. All values are skeleton defaults with documented
/// provenance, not accepted contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// Maximum provider rounds per turn.
    pub max_rounds: usize,
    /// Maximum tool calls accepted per logical assistant turn (`TB-6`),
    /// accounted cumulatively across the turn's provider rounds. The
    /// effective allowance is
    /// `min(AgentConfig::max_tool_calls_per_turn, MAX_TOOL_CALLS_PER_TURN)`;
    /// the bus admits a whole batch only when it fits the remaining
    /// allowance, and the per-call bus counter stays as defense in depth.
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
    /// Reconcile query budget: maximum status queries per `Unknown`
    /// execution before typed escalation (`MP-7`). Counts reconcile queries
    /// only, never tool dispatches: this budget is separate from
    /// [`AgentConfig::max_tool_calls_per_turn`] and [`AgentConfig::max_rounds`]
    /// by construction (see [`ReconcileConfig`]).
    pub max_unknown_retries: usize,
    /// Base backoff delay in milliseconds for reconcile query attempt 0;
    /// doubles per attempt up to
    /// [`AgentConfig::unknown_reconcile_max_delay_ms`]. Deterministic: the
    /// driver derives each query attempt's retry instant from
    /// caller-supplied `now_ms` with saturating addition; no wall clock is
    /// read. See
    /// [`Agent::reconcile_unknown`](crate::agent::Agent::reconcile_unknown)
    /// for the full clock-advance contract (the driver never advances
    /// `now_ms`; callers do).
    pub unknown_reconcile_base_delay_ms: u64,
    /// Per-attempt backoff ceiling in milliseconds for reconcile queries.
    /// Bounds each entry of the reported `delays_ms`; the driver still never
    /// sleeps or advances the clock itself.
    pub unknown_reconcile_max_delay_ms: u64,
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
            max_unknown_retries: DEFAULT_MAX_UNKNOWN_RETRIES,
            unknown_reconcile_base_delay_ms: DEFAULT_RECONCILE_BASE_DELAY_MS,
            unknown_reconcile_max_delay_ms: DEFAULT_RECONCILE_MAX_DELAY_MS,
        }
    }
}

impl AgentConfig {
    /// Reconcile view of the `Unknown` query budget and backoff bounds.
    /// Separate object from the tool-call budget, sharing only the source
    /// values, so callers cannot mistake one budget for the other.
    #[must_use]
    pub fn reconcile_config(&self) -> ReconcileConfig {
        ReconcileConfig {
            max_unknown_retries: self.max_unknown_retries,
            base_delay_ms: self.unknown_reconcile_base_delay_ms,
            max_delay_ms: self.unknown_reconcile_max_delay_ms,
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
    /// An `Unknown` effect stayed unresolvable within the reconcile query
    /// budget and escalated to this typed report (`MP-7`). Fail-closed: the
    /// session is marked failed and no further retry is attempted under the
    /// reconcile protocol. Retry budget and tool-call budget stay separate;
    /// this variant carries only reconcile counts, never dispatch counts.
    UnknownUnresolved {
        /// Tool whose effect stayed uncertain.
        tool: String,
        /// Last observed pending reason (bounded).
        reason: String,
        /// Status queries performed (bounded by the effective budget).
        attempts: usize,
        /// Dispatched executions recorded when escalation fired.
        dispatched: usize,
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
            Self::UnknownUnresolved {
                tool,
                reason,
                attempts,
                dispatched,
            } => {
                write!(
                    f,
                    "tool {tool} effect unreconciled after {attempts} queries \
                     ({dispatched} dispatched): {reason}"
                )
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
            Self::RoundLimitExceeded { .. }
            | Self::CostCeilingExceeded { .. }
            | Self::UnknownUnresolved { .. } => None,
        }
    }
}

impl From<ProviderError> for AgentError {
    fn from(error: ProviderError) -> Self {
        // Provider errors cross from host-owned implementations of
        // `ModelProvider` into the runtime's typed error surface, carrying
        // provider ids and transport reasons that may come from network or
        // model-shaped input. Bound and scrub those strings here (`AI-0059`)
        // so `Display`, logs, and reconcile reports never receive unbounded
        // or newline-carrying text. Other variants carry already-validated
        // registry names, not host reasons.
        let error = match error {
            ProviderError::Transport { provider, reason } => ProviderError::Transport {
                provider: crate::bridge::bound_reason(&provider),
                reason: crate::bridge::bound_reason(&reason),
            },
            ProviderError::Auth { provider, reason } => ProviderError::Auth {
                provider: crate::bridge::bound_reason(&provider),
                reason: crate::bridge::bound_reason(&reason),
            },
            ProviderError::RateLimited {
                provider,
                retry_after_ms,
            } => ProviderError::RateLimited {
                provider: crate::bridge::bound_reason(&provider),
                retry_after_ms,
            },
            ProviderError::Unknown { provider, reason } => ProviderError::Unknown {
                provider: crate::bridge::bound_reason(&provider),
                reason: crate::bridge::bound_reason(&reason),
            },
            other => other,
        };
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

impl From<UnknownEscalation> for AgentError {
    fn from(report: UnknownEscalation) -> Self {
        Self::UnknownUnresolved {
            tool: report.tool,
            reason: report.reason,
            attempts: report.attempts,
            dispatched: report.dispatched,
        }
    }
}

/// One attributed tool dispatch owned by this turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRecord {
    /// Attribution handle.
    pub execution_id: ExecutionId,
    /// Tool name.
    pub tool: String,
    /// Recorded effect status (including `Unknown`). `Refused` marks a
    /// pre-dispatch admission refusal that never contacted the executor;
    /// `Success` with [`ResultDisposition::Rejected`] marks an acknowledged
    /// effect whose payload the bus refused (AI-RUN-004).
    pub status: ToolStatus,
    /// Result-acceptance axis of the dispatch outcome (AI-RUN-004),
    /// preserved alongside the effect status so record consumers can tell an
    /// executed effect from its result handling.
    pub result_disposition: ResultDisposition,
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
    /// Attribution for effects whose outcome stayed `Unknown` when their
    /// turn ended. Survives [`Agent::run_turn`] admission (unlike
    /// `executions`, which is per-turn) so a later turn cannot silently
    /// drop the lookup; reconciling through
    /// [`Agent::reconcile_unknown`] or explicitly abandoning through
    /// [`Agent::abandon_pending`] removes entries. Bounded by
    /// [`MAX_EXECUTIONS_PER_AGENT`].
    pending_unknown: Vec<ExecutionRecord>,
    turn_cost: u64,
    /// Next unused stream sequence number of the logical turn (`S-8` scheme
    /// A); reset at every [`Agent::run_turn`] entry, advanced by every
    /// completed emission batch.
    next_seq: u32,
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
            pending_unknown: Vec::new(),
            turn_cost: 0,
            next_seq: 0,
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

    /// Attributed executions recorded by this turn so far. Cleared at every
    /// [`Agent::run_turn`] entry (per-turn semantics) and bounded by
    /// [`MAX_EXECUTIONS_PER_AGENT`].
    #[must_use]
    pub fn executions(&self) -> &[ExecutionRecord] {
        &self.executions
    }

    /// Unresolved-effect attribution that survived turn admission.
    /// Entries leave this set only through [`Agent::reconcile_unknown`]
    /// (terminal resolution), [`Agent::abandon_pending`], or
    /// escalation-time capture; a new turn never removes them silently.
    /// Escalation itself does not rewrite the recorded status (the
    /// reconcile protocol's no-rewrite-on-escalation contract), so an
    /// escalated entry stays present until abandoned. Bounded by
    /// [`MAX_EXECUTIONS_PER_AGENT`].
    #[must_use]
    pub fn pending_unknown(&self) -> &[ExecutionRecord] {
        &self.pending_unknown
    }

    /// Explicitly abandon one pending unresolved effect by its attribution
    /// handle. Returns `true` when an entry was present and removed.
    ///
    /// Abandonment is deliberate host bookkeeping, not reconciliation: the
    /// caller asserts the effect needs no further status inspection, and
    /// the entry stops being queryable afterwards.
    pub fn abandon_pending(&mut self, execution_id: ExecutionId) -> bool {
        let before = self.pending_unknown.len();
        self.pending_unknown
            .retain(|record| record.execution_id != execution_id);
        self.pending_unknown.len() != before
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

    /// Fresh authorization base from server-side session state.
    ///
    /// Re-read before every `precheck`/`dispatch` so a mid-turn
    /// [`AgentSession::rotate_generation`] downgrade to `Inspect` takes
    /// effect for the remaining calls in the turn (S-1). The generation is
    /// observed alongside the tier because rotation bumps the generation and
    /// resets the tier together; the tier drives the [`AuthBase`] decision.
    fn auth_base(&self) -> AuthBase {
        let _generation = self.session.generation();
        AuthBase {
            agent_instance_id: self.session.agent_instance_id(),
            session_id: self.session.session_id(),
            level: self.session.level(),
        }
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
    ///
    /// Tool outcomes are attributed on three separate axes (AI-RUN-004): a
    /// call refused at the dispatch boundary before executor contact is
    /// recorded as [`ToolStatus::Refused`] (never `Failed`, which claims an
    /// executed effect) and fails the turn with its typed cause; an
    /// acknowledged effect whose returned payload is over-bound stays
    /// recorded as [`ToolStatus::Success`] with
    /// [`ResultDisposition::Rejected`] and fails the turn with the typed
    /// acceptance cause, never a generic execution failure.
    ///
    /// Lifecycle: a `Canceled` session returns `Canceled` without I/O; a
    /// `Completed`/`Failed` session returns
    /// `Failed(Session(AlreadyTerminated))` without I/O, preserving the
    /// terminal state for audit. Use an explicit new session (or host reset
    /// after reconcile) to continue. Per-turn state (`executions`,
    /// `tool_history`, `turn_cost`) resets after the gate, so counts never
    /// mix history; `executions` is additionally bounded by
    /// [`MAX_EXECUTIONS_PER_AGENT`]. Streamed chunks are numbered continuously
    /// across the turn's emission batches (`S-8` scheme A) so transport dedup
    /// keys stay unique; `total`/`is_final` close each emission batch, while
    /// turn completion is this method's [`ExecOutcome`].
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
        let state = self.session.state();
        if state != SessionState::Active {
            return ExecOutcome::Failed {
                error: AgentError::Session(SessionError::AlreadyTerminated { state }),
            };
        }
        let request = ContextRequest {
            max_tokens: None,
            max_bytes: Some(self.config.context_budget_bytes as u64),
            current_generation: self.session.generation(),
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

        // AI-RUN-001: carry unresolved-effect attribution across the
        // admission boundary. `executions` still holds the *previous* turn's
        // records here (it is cleared below), so snapshot its `Unknown`
        // entries into the bounded pending set first; this turn can no
        // longer drop the lookup silently.
        //
        // The loop mutates the pending set (refresh or push), so it iterates
        // over an owned snapshot of the previous turn's `Unknown` tail —
        // never over the pending set itself.
        let carried: Vec<ExecutionRecord> = self
            .executions
            .iter()
            .filter(|record| matches!(record.status, ToolStatus::Unknown { .. }))
            .cloned()
            .collect();
        for record in &carried {
            // Refresh-before-push: an execution id the host reuses across
            // turns keeps exactly one pending entry (no duplicates), and the
            // latest status/reason wins. This is refresh-only: any id already
            // present is updated in place, never appended.
            if let Some(slot) = self
                .pending_unknown
                .iter_mut()
                .find(|pending| pending.execution_id == record.execution_id)
            {
                slot.status = record.status.clone();
                slot.result_disposition = record.result_disposition.clone();
                slot.tool.clone_from(&record.tool);
                continue;
            }
            if self.pending_unknown.len() >= MAX_EXECUTIONS_PER_AGENT {
                return self.fail(AgentError::Tool(ToolError::CallLimitExceeded {
                    limit: MAX_EXECUTIONS_PER_AGENT,
                }));
            }
            self.pending_unknown.push(record.clone());
        }

        self.tools.begin_turn();
        let tool_names = self.tools.tool_names();
        let mut round = 0;
        // AI-RUN-001: the per-turn vector is cleared here, but the carry
        // loop above already snapshotted this turn's `Unknown` entries into
        // the bounded pending set, so the lookup survives admission.
        self.executions.clear();
        self.tool_history.clear();
        self.turn_cost = 0;
        self.next_seq = 0;
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
                sampling: None,
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
                // AI-RUN-005: cancellation observed at/after final text
                // delivery must reconcile as cancellation, never be
                // declared `Completed` (which would also leave the session
                // `Canceled` by disagreeing with the outcome). Recheck after
                // the full batch was accepted and before completion.
                if self.session.is_cancelled() {
                    return self.reconcile_cancel();
                }
                self.session.finish(false);
                return ExecOutcome::Completed { text: turn.text };
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
            // anything and admit the whole batch against the logical-turn
            // budget (`FS-AI1`). The accounting scope is the bus counter,
            // cumulative across the turn's rounds: a later batch is rejected
            // whole when it does not fit the remaining configured *or* hard
            // allowance, with nothing dispatched and earlier effects kept.
            // Fresh base per round so a rotation that landed between rounds
            // is already visible here.
            let precheck_base = self.auth_base();
            if let Err(error) =
                self.tools
                    .precheck(&calls, &precheck_base, self.config.max_tool_calls_per_turn)
            {
                return self.fail(error.into());
            }
            messages.push(Message::assistant(turn.text.clone()));
            for call in &calls {
                if self.session.is_cancelled() {
                    return self.reconcile_cancel();
                }
                if self.executions.len() >= MAX_EXECUTIONS_PER_AGENT {
                    return self.fail(
                        ToolError::CallLimitExceeded {
                            limit: MAX_EXECUTIONS_PER_AGENT,
                        }
                        .into(),
                    );
                }
                let execution_id = self.ids.execution();
                // S-1: per-call JIT authorization base. A mid-turn
                // `rotate_generation()` (shared session handle) downgrades
                // the remaining dispatches to `Inspect`; already-dispatched
                // effects are kept, never rolled back.
                let base = self.auth_base();
                match self
                    .tools
                    .dispatch(executor, call, &base, execution_id, now_ms)
                {
                    Ok(execution) => {
                        // AI-RUN-004: a pre-dispatch admission refusal never
                        // reached the host: it is attributed as `Refused`
                        // (never an executed `Failed`) and the turn fails
                        // closed with the carried typed cause. No L0 record
                        // is appended: there is no result to disclose.
                        if execution.status.is_admission_refusal() {
                            self.attribute_execution(&execution);
                            let cause = match execution.status {
                                ToolStatus::Refused { cause } => cause,
                                _ => unreachable!("matched an admission refusal above"),
                            };
                            return self.fail(cause.into());
                        }
                        // AI-RUN-004: the executor acknowledged an effect
                        // whose returned payload the bus could not accept.
                        // The effect stays attributed as `Success` (it
                        // happened, and `completed` counts keep it); the
                        // turn fails closed with the typed acceptance cause,
                        // never a generic executed-failure and never silent
                        // truncation (S-2).
                        if execution.result_disposition.is_rejection() {
                            self.attribute_execution(&execution);
                            let cause = match &execution.result_disposition {
                                ResultDisposition::Rejected { cause } => cause.clone(),
                                ResultDisposition::Accepted => {
                                    unreachable!("matched a rejection above")
                                }
                            };
                            return self.fail(cause.into());
                        }
                        let unknown = matches!(execution.status, ToolStatus::Unknown { .. });
                        let denied = matches!(execution.status, ToolStatus::Denied { .. });
                        // S-2: externalization failure fails the turn; never
                        // substitute empty bytes for lost tool output.
                        if let Err(error) = self.record_execution(&execution, now_ms) {
                            return self.fail(error.into());
                        }
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
                            result_disposition: ResultDisposition::Accepted,
                        });
                        return self.fail(error.into());
                    }
                }
            }
        }
    }

    /// Stream assistant text for one provider round, continuing the turn's
    /// sequence after any earlier batch. Returns `Ok(false)` when
    /// cancellation stopped emission (caller reconciles).
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
        self.emit_batch(sink, &fragments)
    }

    /// Stream one ToolCard fragment for a recorded execution, continuing the
    /// turn's sequence after the assistant text and any earlier cards.
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
            ToolStatus::Refused { .. } => "refused",
            ToolStatus::Unknown { .. } => "unknown",
        };
        let card = format!(
            "tool={} status={} summary={}",
            execution.tool, status, execution.summary
        );
        let fragments = fragment_text(FragmentKind::ToolCard, &card);
        self.emit_batch(sink, &fragments)
    }

    /// Emit one fragment batch into the turn's continuous sequence and
    /// advance [`Agent::next_seq`] on completion (`S-8` scheme A).
    ///
    /// Returns `Ok(true)` when every fragment was emitted, `Ok(false)` when
    /// cancellation stopped emission early (already-emitted chunks stay
    /// emitted; the caller reconciles).
    ///
    /// # Errors
    ///
    /// Returns [`StreamError`] for oversized or misframed chunks or an
    /// exhausted sequence space; chunks emitted before the failure stay
    /// emitted.
    fn emit_batch(
        &mut self,
        sink: &mut dyn StreamSink,
        fragments: &[Fragment],
    ) -> Result<bool, StreamError> {
        let start_seq = self.next_seq;
        let session = &self.session;
        let next_seq = emit_fragments(sink, fragments, start_seq, &|| session.is_cancelled())?;
        match next_seq {
            Some(next_seq) => {
                self.next_seq = next_seq;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Record one execution as an attributed record plus an L0
    /// [`ContextRecord`] (large payloads externalize to the artifact store).
    ///
    /// # Errors
    ///
    /// Returns the [`ContextError`] from [`ArtifactStore::store`] (for
    /// example [`ContextError::ArtifactStoreFull`]) when a large payload
    /// cannot be externalized. The attributed [`ExecutionRecord`] is already
    /// pushed (the effect happened), but no [`ContextRecord`] is appended:
    /// callers must `fail()` the turn rather than substitute empty bytes
    /// (S-2). Small payloads never fail here.
    fn record_execution(
        &mut self,
        execution: &ToolExecution,
        now_ms: u64,
    ) -> Result<(), ContextError> {
        self.attribute_execution(execution);
        let body = if execution.data.len() > crate::context::EXTERNALIZE_THRESHOLD_BYTES {
            let reference = self.artifacts.store(execution.data.clone())?;
            RecordBody::Artifact(reference)
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
        Ok(())
    }

    /// Push one attributed [`ExecutionRecord`] for a dispatch that produces
    /// no disclosable L0 payload (admission refusal or rejected result):
    /// the effect/acceptance axes are preserved even though no bytes enter
    /// the context history (AI-RUN-004).
    fn attribute_execution(&mut self, execution: &ToolExecution) {
        self.executions.push(ExecutionRecord {
            execution_id: execution.execution_id,
            tool: execution.tool.clone(),
            status: execution.status.clone(),
            result_disposition: execution.result_disposition.clone(),
        });
    }

    /// Render one execution as a tool message for the next provider round.
    ///
    /// A rejected result never reaches this mapping: rejection fails the turn
    /// before the message is built, and the separation is carried on
    /// [`ExecutionRecord::result_disposition`] instead (AI-RUN-004).
    fn execution_message(&self, execution: &ToolExecution) -> Message {
        let status = match &execution.status {
            ToolStatus::Success => "ok".to_owned(),
            ToolStatus::Failed { reason } => format!("failed: {reason}"),
            ToolStatus::Denied { reason } => format!("denied: {reason}"),
            ToolStatus::Refused { cause } => format!("refused: {cause}"),
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

    /// Reconcile one `Unknown` execution: bounded status queries with
    /// deterministic backoff ceilings, then typed escalation (`MP-7`).
    ///
    /// The driver queries `reconciler` for `execution_id` without executing
    /// anything: no [`ToolExecutor`](crate::tool::ToolExecutor) call is made
    /// here, and the tool-call budget
    /// ([`AgentConfig::max_tool_calls_per_turn`], [`ToolBus::calls_this_turn`])
    /// is neither read nor modified. Only the separate reconcile budget
    /// ([`AgentConfig::max_unknown_retries`] plus backoff bounds) applies.
    ///
    /// - The target must be recorded on this agent with
    ///   [`ToolStatus::Unknown`]; otherwise [`ReconcileOutcome::NoUnknown`]
    ///   is returned and nothing runs.
    /// - Each query attempt `attempt` (0-based) reports
    ///   `delays_ms[attempt] = reconcile_delay_ms(attempt, base, ceiling)`
    ///   via [`reconcile_delay_ms`], with `base`/`ceiling` from
    ///   [`AgentConfig::reconcile_config`] (already bounded by
    ///   [`ReconcileConfig`] and the hard caps). The driver never sleeps and
    ///   never reads a clock.
    /// - A terminal reconciler answer updates the recorded status in place
    ///   and returns [`ReconcileOutcome::Resolved`]; the session stays
    ///   `Active`. A `Resolved(ToolStatus::Unknown)` answer is treated as
    ///   still pending (fail closed).
    /// - When the budget is exhausted the driver marks the session failed
    ///   and returns [`ReconcileOutcome::Escalated`] with the typed
    ///   [`UnknownEscalation`] report (convertible to
    ///   [`AgentError::UnknownUnresolved`]). No further retry is attempted
    ///   under this protocol.
    ///
    /// # Clock-advance contract
    ///
    /// `reconcile_unknown` is a single synchronous evaluation of the whole
    /// bounded schedule and does **not** advance the clock: every query in one
    /// invocation is issued at the same caller-supplied `now_ms`, and the
    /// retry instant the driver computes for attempt `attempt`,
    ///
    /// ```text
    /// next_retry_ms(attempt) = now_ms.saturating_add(delays_ms[attempt])
    /// ```
    ///
    /// is reported, not waited on. `delays_ms` is therefore a *schedule*, not
    /// observed elapsed time. Callers that need real backoff advance their own
    /// clock by the reported delays and re-invoke:
    ///
    /// - a full-budget invocation schedules attempts up to
    ///   `next_retry_ms(delays_ms.len() - 1)`; advance to (or past) that last
    ///   scheduled instant and re-invoke to let a time-dependent reconciler
    ///   make progress;
    /// - for exactly one query per invocation, configure
    ///   [`AgentConfig::max_unknown_retries`] `= 1` and advance by the single
    ///   reported delay, `now_ms.saturating_add(delays_ms[0])`.
    ///
    /// Re-invocation is defined for as long as the target stays recorded as
    /// [`ToolStatus::Unknown`]: escalation fails the session but does not
    /// rewrite that recorded status. Reusing the same `now_ms` on every
    /// invocation is valid only when the caller deliberately freezes time
    /// (deterministic replay); the recorded `delays_ms` never reflects real
    /// time advancement on its own.
    pub fn reconcile_unknown(
        &mut self,
        reconciler: &mut dyn UnknownReconciler,
        execution_id: ExecutionId,
        now_ms: u64,
    ) -> ReconcileOutcome {
        // AI-RUN-001: the pending set outlives turn admission, so resolve
        // against it first; the current-turn vector is the fallback for
        // records not yet carried across. Both paths mutate the same
        // underlying record shape, and resolution removes the pending
        // entry as well so the two views cannot disagree.
        if let Some(pending) = self
            .pending_unknown
            .iter()
            .position(|record| record.execution_id == execution_id)
        {
            let tool = self.pending_unknown[pending].tool.clone();
            let outcome = self.reconcile_pending_at(pending, &tool, reconciler, now_ms);
            if !matches!(outcome, ReconcileOutcome::NoUnknown) {
                return outcome;
            }
        }
        let position = self.executions.iter().position(|record| {
            record.execution_id == execution_id
                && matches!(record.status, ToolStatus::Unknown { .. })
        });
        let Some(index) = position else {
            return ReconcileOutcome::NoUnknown;
        };
        let tool = self.executions[index].tool.clone();
        let outcome = self.reconcile_executions_at(index, &tool, reconciler, execution_id, now_ms);
        if !matches!(outcome, ReconcileOutcome::NoUnknown) {
            return outcome;
        }
        ReconcileOutcome::NoUnknown
    }

    /// Run the bounded reconcile schedule against one pending-set entry.
    /// Resolution (terminal answer) clears the entry; escalation captures
    /// the pending entry's tool attribution into the report so per-turn
    /// `dispatched` counts never leak across turns.
    fn reconcile_pending_at(
        &mut self,
        pending: usize,
        tool: &str,
        reconciler: &mut dyn UnknownReconciler,
        now_ms: u64,
    ) -> ReconcileOutcome {
        let dispatched = self.executions.len();
        let config = self.config.reconcile_config();
        let budget = config.effective_retries();
        let ceiling = config.effective_max_delay_ms();
        let base = config.base_delay_ms.min(ceiling);
        let current_reason = match &self.pending_unknown[pending].status {
            ToolStatus::Unknown { reason } => bound_reason(reason),
            _ => return ReconcileOutcome::NoUnknown,
        };
        let mut delays_ms: Vec<u64> = Vec::new();
        let mut last_reason = current_reason;
        let mut attempt = 0;
        while attempt < budget {
            let delay = reconcile_delay_ms(attempt, base, ceiling);
            delays_ms.push(delay);
            let _next_retry_ms = now_ms.saturating_add(delay);
            let answer =
                reconciler.reconcile(tool, self.pending_unknown[pending].execution_id, now_ms);
            match answer {
                ReconcileStatus::Resolved(status) => {
                    let terminal = match &status {
                        ToolStatus::Success
                        | ToolStatus::Failed { .. }
                        | ToolStatus::Denied { .. }
                        | ToolStatus::Refused { .. } => true,
                        ToolStatus::Unknown { .. } => false,
                    };
                    if terminal {
                        let resolved_id = self.pending_unknown[pending].execution_id;
                        self.pending_unknown.remove(pending);
                        // Keep the current-turn view consistent when it
                        // still holds the same attribution handle.
                        if let Some(current) = self
                            .executions
                            .iter_mut()
                            .find(|record| record.execution_id == resolved_id)
                        {
                            current.status = status.clone();
                        }
                        return ReconcileOutcome::Resolved {
                            status,
                            attempts: attempt + 1,
                            delays_ms,
                        };
                    }
                    match status {
                        ToolStatus::Unknown { reason } => {
                            last_reason = bound_reason(&reason);
                        }
                        _ => unreachable!("non-terminal status is Unknown"),
                    }
                }
                ReconcileStatus::Pending { reason } => {
                    last_reason = bound_reason(&reason);
                }
            }
            attempt += 1;
        }
        let report = UnknownEscalation {
            tool: tool.to_owned(),
            reason: last_reason,
            attempts: delays_ms.len(),
            dispatched,
            delays_ms,
        };
        self.session.finish(true);
        ReconcileOutcome::Escalated(report)
    }

    /// Run the bounded reconcile schedule against one current-turn entry.
    /// This is the original `reconcile_unknown` body, unchanged apart from
    /// the pending-entry removal on resolution so both views agree.
    #[allow(clippy::too_many_lines)]
    fn reconcile_executions_at(
        &mut self,
        index: usize,
        tool: &str,
        reconciler: &mut dyn UnknownReconciler,
        execution_id: ExecutionId,
        now_ms: u64,
    ) -> ReconcileOutcome {
        let dispatched = self.executions.len();
        let config = self.config.reconcile_config();
        let budget = config.effective_retries();
        let ceiling = config.effective_max_delay_ms();
        let base = config.base_delay_ms.min(ceiling);
        let current_reason = match &self.executions[index].status {
            ToolStatus::Unknown { reason } => bound_reason(reason),
            _ => unreachable!("matched Unknown above"),
        };
        let mut delays_ms: Vec<u64> = Vec::new();
        let mut last_reason = current_reason;
        let mut attempt = 0;
        while attempt < budget {
            let delay = reconcile_delay_ms(attempt, base, ceiling);
            delays_ms.push(delay);
            let _next_retry_ms = now_ms.saturating_add(delay);
            let answer = reconciler.reconcile(tool, execution_id, now_ms);
            match answer {
                ReconcileStatus::Resolved(status) => {
                    let terminal = match &status {
                        ToolStatus::Success
                        | ToolStatus::Failed { .. }
                        | ToolStatus::Denied { .. }
                        | ToolStatus::Refused { .. } => true,
                        ToolStatus::Unknown { .. } => false,
                    };
                    if terminal {
                        self.executions[index].status = status.clone();
                        // The pending set may hold the same attribution
                        // handle (carried across admission); resolution
                        // clears it there too so both views agree.
                        self.pending_unknown
                            .retain(|record| record.execution_id != execution_id);
                        return ReconcileOutcome::Resolved {
                            status,
                            attempts: attempt + 1,
                            delays_ms,
                        };
                    }
                    match status {
                        ToolStatus::Unknown { reason } => {
                            last_reason = bound_reason(&reason);
                        }
                        _ => unreachable!("non-terminal status is Unknown"),
                    }
                }
                ReconcileStatus::Pending { reason } => {
                    last_reason = bound_reason(&reason);
                }
            }
            attempt += 1;
        }
        let report = UnknownEscalation {
            tool: tool.to_owned(),
            reason: last_reason,
            attempts: delays_ms.len(),
            dispatched,
            delays_ms,
        };
        self.session.finish(true);
        ReconcileOutcome::Escalated(report)
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

/// Effective cost weight for turn accounting: a configured `0` means
/// uncalibrated and counts as the baseline `1` so an uncalibrated host cannot
/// bypass the ceiling with free rounds. This is the unified zero-weight rule:
/// [`crate::selection::estimate_cost`] applies the same mapping to routing
/// estimates and the selection cost-ceiling filter, so routing and accounting
/// never disagree and accounting never under-counts.
fn effective_cost_weight(weight: u32) -> u32 {
    if weight == 0 { 1 } else { weight }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::MAX_REASON_BYTES;

    /// P2-7 (`AI-0059`): host-shaped provider reasons are bounded and
    /// scrubbed when they enter the runtime's typed error surface.
    #[test]
    fn provider_reasons_are_bounded_at_the_agent_boundary() {
        let hostile = format!("reset\n\u{1b}[2J{}", "x".repeat(MAX_REASON_BYTES * 2));
        let long_provider = "p".repeat(MAX_REASON_BYTES * 3);
        let cases = [
            ProviderError::Transport {
                provider: long_provider.clone(),
                reason: hostile.clone(),
            },
            ProviderError::Auth {
                provider: long_provider.clone(),
                reason: hostile.clone(),
            },
            ProviderError::RateLimited {
                provider: long_provider.clone(),
                retry_after_ms: Some(250),
            },
            ProviderError::Unknown {
                provider: long_provider.clone(),
                reason: hostile.clone(),
            },
        ];
        for error in cases {
            let agent_error: AgentError = error.into();
            match &agent_error {
                AgentError::Provider(ProviderError::Transport { provider, reason })
                | AgentError::Provider(ProviderError::Auth { provider, reason })
                | AgentError::Provider(ProviderError::Unknown { provider, reason }) => {
                    assert!(provider.len() <= MAX_REASON_BYTES);
                    assert!(reason.len() <= MAX_REASON_BYTES);
                    assert!(!reason.contains(['\n', '\r', '\t']));
                }
                AgentError::Provider(ProviderError::RateLimited { provider, .. }) => {
                    assert!(provider.len() <= MAX_REASON_BYTES);
                }
                other => panic!("unexpected conversion: {other:?}"),
            }
            let display = agent_error.to_string();
            assert!(
                !display.contains('\n') && !display.contains('\r'),
                "Display must not carry newlines: {display:?}"
            );
        }
    }

    #[test]
    fn clean_provider_reasons_pass_through_unchanged() {
        let error = ProviderError::Transport {
            provider: "bitty-fake".to_owned(),
            reason: "connection reset".to_owned(),
        };
        let agent_error: AgentError = error.clone().into();
        assert_eq!(agent_error, AgentError::Provider(error));
    }
}

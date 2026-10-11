//! Prompt-dispatch composition at the agent boundary (AIQ-33 S1).
//!
//! `prompt.rs` already proves the dispatch check in isolation
//! (`check_dispatch`/`is_dispatch_allowed` unit tests) and
//! `granularity_denial.rs` pins the prompt seam as narrowing evidence, but
//! nothing proves the agent loop itself consults the assembled prompt
//! before touching the executor. This file pins that composition end to
//! end through the public API: with an assembled prompt installed via
//! [`Agent::set_prompt`], every call of a batch is checked with
//! [`check_dispatch`](bitty_ai_runtime::check_dispatch) before
//! [`ToolBus`](bitty_ai_runtime::ToolBus) precheck, and any refusal —
//! prompt denial, allow-set omission (including the disjoint-allow-set
//! [`EMPTY_ALLOW_SET_SENTINEL`](bitty_ai_runtime::EMPTY_ALLOW_SET_SENTINEL)
//! mapping), or a missing dispatcher grant — refuses pre-contact as
//! [`ToolStatus::Refused`](bitty_ai_runtime::ToolStatus) carrying the typed
//! [`ToolError::Denied`](bitty_ai_runtime::ToolError) cause, with zero
//! executor contact and no per-turn counter increment.
//!
//! Six proofs:
//!
//! 1. Allowed dispatch: a prompt that allows the requested tool dispatches
//!    normally (executor contacted once, `Success` attribution).
//! 2. Prompt denial: a `denied_tools` entry refuses pre-contact (`Refused`
//!    plus `Denied`, executor untouched, counter untouched).
//! 3. Allow-set omission: a constrained allow-set that omits the requested
//!    tool refuses the same way, with the exact omission reason (which also
//!    pins gate order: the prompt refusal fires before the tier floor, even
//!    at `Inspect`).
//! 4. Disjoint allow-set sentinel: layers with disjoint allow-sets fail
//!    assembly with `PromptNotAllowed` carrying the sentinel, and the S1
//!    error mapping preserves that identity into `Denied` (still `Refused`,
//!    never `Failed`/`Unknown`/`ProtocolRejected`).
//! 5. Dispatcher denial: an unregistered name under an `Allow` authorizer
//!    refuses as `DispatcherDenied` mapped to `Denied` — the hook would
//!    allow, so the refusal isolates the dispatcher-grant input — again
//!    with zero executor contact.
//! 6. Execution bound: the prompt-gate refusal checks
//!    `MAX_EXECUTIONS_PER_AGENT` before attributing (mirroring the dispatch
//!    loop and `provider_unknown` with the identical `CallLimitExceeded`
//!    constructor), so a full attribution vector fails closed without
//!    pushing a 33rd record; below the cap the normal `Denied` refusal is
//!    unchanged and the bound holds.
//!
//! Deterministic and offline: caller-built snapshots, scripted
//! [`FakeProvider`](bitty_ai_runtime::FakeProvider) turns, deterministic
//! executor/authorizer doubles, caller-supplied `now_ms`. No network, no
//! secrets, no wall clock, no threads, no `HashMap`.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `granularity_denial.rs`: prompt narrowing at the prompt seam only
//!   (`check_dispatch` return values, no agent loop, no executor). The
//!   narrowing denial appears here only as the composed agent-boundary
//!   refusal (attributed `Refused`, failed turn, untouched executor) that
//!   file leaves out.
//! - `batch_evidence.rs`: whole-batch `ToolBus` precheck shape (all or
//!   nothing, zero executions on `UnknownTool`). The batch gate appears
//!   here only as the prompt-first ordering (prompt refusal attributed
//!   before precheck runs), never as the bus admission itself.
//! - `mcp_fail_closed.rs`: bus-level refusals (`Refused` carrying
//!   `UnknownTool`, no executor contact) without a prompt installed. Bus
//!   refusal construction appears here only as the agent-composed prompt
//!   refusal, never as the bus path.
//! - `runtime_fail_closed.rs` / `turn_lifecycle.rs` /
//!   `result_schema_disclosure.rs`: `UnknownTool` through `run_turn`
//!   without an installed prompt. Those paths are untouched: no test here
//!   installs a prompt and then expects bus-vocabulary errors.
//!
//! CodeQL lesson from AI-0082: assert/panic/expect messages are static only;
//! tool names, reasons, and outcomes never appear in message strings.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AgentLevel, AgentSession, AuthContext, AuthDecision,
    EMPTY_ALLOW_SET_SENTINEL, ElevationGrant, ExecOutcome, ExecutionRecord, FakeProvider,
    FakeToolExecutor, IdIssuer, LayerInput, MAX_EXECUTIONS_PER_AGENT, ModelProvider, PromptError,
    PromptLayer, PromptSnapshot, ProviderTurn, ProviderUsage, ResultDisposition, SessionState,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolRegistry, ToolSpec, ToolStatus,
    VecSink, assemble_prompt, dispatch_denial_cause,
};

const NOW_MS: u64 = 1_700_000_000_000;

fn ids() -> (
    bitty_ai_runtime::AgentInstanceId,
    bitty_ai_runtime::RunId,
    bitty_ai_runtime::SessionId,
) {
    let mut issuer = IdIssuer::default();
    (issuer.agent_instance(), issuer.run(), issuer.session())
}

fn session() -> AgentSession {
    let (agent, run, session) = ids();
    AgentSession::new(agent, run, session)
}

fn read_write_tool_bus(authorizer: AllowAll) -> ToolBus {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "workspace_read",
                "Read a bounded workspace path",
                br#"{"type":"object"}"#.to_vec(),
                "workspace.read",
                true,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    registry
        .register(
            ToolSpec::new(
                "workspace_write",
                "Write a bounded workspace path",
                br#"{"type":"object"}"#.to_vec(),
                "workspace.write",
                false,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    ToolBus::new(registry).with_authorizer(authorizer)
}

/// Test-only authorizer that allows everything. Denial tests can therefore
/// attribute every refusal to the prompt/dispatcher composition under test,
/// never to hook policy; production wiring installs the host
/// capability-plus-consent hook instead.
#[derive(Debug, Clone, Copy, Default)]
struct AllowAll;

impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

struct AllowElevation;
impl ElevationGrant for AllowElevation {
    fn elevation_granted(&self, _current: AgentLevel, _requested: AgentLevel) -> bool {
        true
    }
}

fn text_layer(layer: PromptLayer, text: &str) -> LayerInput {
    LayerInput::text_only(layer, text)
}

fn constrained_layer(
    layer: PromptLayer,
    allowed: Option<Vec<String>>,
    denied: Vec<String>,
) -> LayerInput {
    LayerInput {
        layer,
        text: "stable policy".to_owned(),
        allowed_tools: allowed,
        denied_tools: denied,
        budget_ceiling_bytes: None,
        allowed_scopes: None,
        directives: Vec::new(),
    }
}

/// Unconstrained prompt fixture: no allow-set, no deny-set. Dispatch still
/// needs the dispatcher grant; the prompt itself narrows nothing.
fn permissive_prompt() -> bitty_ai_runtime::AssembledPrompt {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::Project, "stable project"),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            text_layer(PromptLayer::RuntimeTurn, "turn instruction"),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot).expect("test prompt assembles")
}

/// Prompt fixture denying one tool at the turn layer (deny wins even when
/// the dispatcher would grant).
fn deny_write_prompt() -> bitty_ai_runtime::AssembledPrompt {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::Project, "stable project"),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            constrained_layer(
                PromptLayer::RuntimeTurn,
                None,
                vec!["workspace_write".to_owned()],
            ),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot).expect("test prompt assembles")
}

/// Prompt fixture constraining dispatch to the read tool only: the write
/// tool is omitted from the allow-set without being denied outright.
fn read_only_allow_prompt() -> bitty_ai_runtime::AssembledPrompt {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::Project, "stable project"),
            text_layer(PromptLayer::SkillsProfile, "stable skills"),
            constrained_layer(
                PromptLayer::RuntimeTurn,
                Some(vec!["workspace_read".to_owned()]),
                Vec::new(),
            ),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot).expect("test prompt assembles")
}

fn write_call(path: &str) -> ToolCallRequest {
    ToolCallRequest {
        name: "workspace_write".to_owned(),
        arguments: format!(r#"{{"path":"{path}"}}"#).into_bytes(),
    }
}

fn tool_turn(text: &str, calls: Vec<ToolCallRequest>) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: calls,
        latency_ms: 0,
        usage: ProviderUsage::default(),
    }
}

fn failed_tool_error(outcome: &ExecOutcome) -> ToolError {
    match outcome {
        ExecOutcome::Failed {
            error: AgentError::Tool(error),
        } => error.clone(),
        _ => panic!("refused turn must fail with a typed tool error"),
    }
}

// Proof 1: an allowed tool dispatches normally under an installed prompt.

#[test]
fn allowed_prompt_dispatches_normally_with_executor_contact() {
    // Control case: the prompt allows the tool and the dispatcher grants
    // it, so the turn dispatches exactly once and completes. This proves
    // the S1 gate is narrowing-only: it never blocks what both sides admit.
    let sess = session();
    sess.elevate(AgentLevel::Workspace, &AllowElevation)
        .expect("grant allows elevation");
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("one write", vec![write_call("a")]));
    provider.push_turn(tool_turn("done", Vec::new()));
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowAll),
        sess.clone(),
        AgentConfig::default(),
    );
    agent.set_prompt(permissive_prompt());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("wrote a", b"ok".to_vec());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    assert!(
        matches!(outcome, ExecOutcome::Completed { .. }),
        "allowed dispatch must complete the turn"
    );
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(executor.calls()[0].0, "workspace_write");
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(agent.provider_mut().complete_calls(), 2);
    assert_eq!(sess.state(), SessionState::Completed);
}

// Proof 2: prompt denial refuses pre-contact.

#[test]
fn prompt_denied_tool_refuses_precontact_with_denied_cause() {
    // The prompt denies the write while the dispatcher (Allow hook plus a
    // registered spec) would grant it: the boundary must refuse
    // pre-contact. The session stays at Inspect on purpose, so the exact
    // prompt reason also pins gate order — the prompt refusal fires before
    // the tier floor is ever consulted.
    let sess = session();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("one write", vec![write_call("a")]));
    provider.push_turn(tool_turn("unreached", Vec::new()));
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowAll),
        sess.clone(),
        AgentConfig::default(),
    );
    agent.set_prompt(deny_write_prompt());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never run", b"nope".to_vec());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let error = failed_tool_error(&outcome);
    assert!(
        matches!(
            error,
            ToolError::Denied { ref name, ref reason }
            if name == "workspace_write"
                && reason == "prompt denies tool 'workspace_write'"
        ),
        "prompt denial must fail the turn as Denied naming the tool"
    );
    // The refusal is attributed as an admission refusal carrying the exact
    // typed cause: Refused status, never an executed failure.
    assert_eq!(agent.executions().len(), 1);
    let status = &agent.executions()[0].status;
    assert!(status.is_admission_refusal());
    assert!(
        matches!(
            status,
            ToolStatus::Refused {
                cause: ToolError::Denied { .. }
            }
        ),
        "prompt denial must attribute Refused carrying the Denied cause"
    );
    // Zero executor contact: the scripted success is unconsumed, the call
    // log is empty, and the bus counter (which advances only on executor
    // contact) is untouched by construction.
    assert!(executor.calls().is_empty());
    // The turn reached the dispatch phase, then stopped: one provider
    // round consumed, session terminal.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(sess.state(), SessionState::Failed);
}

// Proof 3: allow-set omission refuses pre-contact.

#[test]
fn allow_omitted_tool_refuses_precontact_with_denied_cause() {
    // The allow-set admits only the read tool: the write is omitted without
    // being denied outright, and the omission must still refuse pre-contact
    // with the exact omission reason.
    let sess = session();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("one write", vec![write_call("a")]));
    provider.push_turn(tool_turn("unreached", Vec::new()));
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowAll),
        sess.clone(),
        AgentConfig::default(),
    );
    agent.set_prompt(read_only_allow_prompt());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never run", b"nope".to_vec());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let error = failed_tool_error(&outcome);
    assert!(
        matches!(
            error,
            ToolError::Denied { ref name, ref reason }
            if name == "workspace_write"
                && reason == "prompt allow-set omits tool 'workspace_write'"
        ),
        "allow omission must fail the turn as Denied naming the tool"
    );
    assert_eq!(agent.executions().len(), 1);
    let status = &agent.executions()[0].status;
    assert!(status.is_admission_refusal());
    assert!(
        matches!(
            status,
            ToolStatus::Refused {
                cause: ToolError::Denied { .. }
            }
        ),
        "allow omission must attribute Refused carrying the Denied cause"
    );
    assert!(executor.calls().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(sess.state(), SessionState::Failed);
}

// Proof 4: the disjoint-allow-set sentinel maps into Denied, never an
// executed vocabulary.

#[test]
fn disjoint_allow_set_sentinel_maps_to_denied_refusal() {
    // Disjoint per-layer allow-sets fail assembly itself with the sentinel
    // (there is no single dispatched tool to name). The S1 error mapping
    // must preserve that identity into the pre-contact Denied cause, and
    // the refusal status stays admission-only.
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            text_layer(PromptLayer::CoreContract, "stable core"),
            text_layer(PromptLayer::User, "stable user"),
            text_layer(PromptLayer::Project, "stable project"),
            constrained_layer(
                PromptLayer::SkillsProfile,
                Some(vec!["workspace_read".to_owned()]),
                Vec::new(),
            ),
            constrained_layer(
                PromptLayer::RuntimeTurn,
                Some(vec!["workspace_write".to_owned()]),
                Vec::new(),
            ),
        ],
    )
    .expect("test snapshot is valid");
    let error = assemble_prompt(&snapshot).expect_err("disjoint allow-sets must deny");
    assert!(
        matches!(
            error,
            PromptError::PromptNotAllowed { ref tool } if tool == EMPTY_ALLOW_SET_SENTINEL
        ),
        "disjoint allow-sets must deny with the sentinel identity"
    );
    let cause = dispatch_denial_cause(EMPTY_ALLOW_SET_SENTINEL, &error);
    assert!(
        matches!(
            cause,
            ToolError::Denied { ref name, ref reason }
            if name == EMPTY_ALLOW_SET_SENTINEL
                && reason.contains(EMPTY_ALLOW_SET_SENTINEL)
        ),
        "sentinel mapping must stay Denied with the sentinel identity"
    );
    // The mapped cause rides a Refused status (admission-only), never an
    // executed outcome: no Failed, no Unknown, no ProtocolRejected.
    let status = ToolStatus::Refused { cause };
    assert!(status.is_admission_refusal());
    assert!(
        matches!(status, ToolStatus::Refused { .. }),
        "sentinel refusal must stay admission-only"
    );
    // Zero contact by construction: no turn ran, so no executor or
    // transport was touched.
    let executor = FakeToolExecutor::new();
    assert!(executor.calls().is_empty());
}

// Proof 5: a missing dispatcher grant refuses pre-contact.

#[test]
fn dispatcher_denied_tool_refuses_precontact_with_denied_cause() {
    // The tool name is unregistered, so the dispatcher does not grant it —
    // while the Allow hook would allow anything. The refusal therefore
    // isolates the dispatcher-grant input (not hook policy) and maps
    // DispatcherDenied into the same pre-contact Denied refusal.
    let sess = session();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn(
        "ghost call",
        vec![ToolCallRequest {
            name: "ghost_tool".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
    ));
    provider.push_turn(tool_turn("unreached", Vec::new()));
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowAll),
        sess.clone(),
        AgentConfig::default(),
    );
    agent.set_prompt(permissive_prompt());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("must never run", b"nope".to_vec());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let error = failed_tool_error(&outcome);
    assert!(
        matches!(
            error,
            ToolError::Denied { ref name, ref reason }
            if name == "ghost_tool"
                && reason == "dispatcher denies tool 'ghost_tool': prompt never grants"
        ),
        "dispatcher denial must fail the turn as Denied naming the tool"
    );
    assert_eq!(agent.executions().len(), 1);
    let status = &agent.executions()[0].status;
    assert!(status.is_admission_refusal());
    assert!(
        matches!(
            status,
            ToolStatus::Refused {
                cause: ToolError::Denied { .. }
            }
        ),
        "dispatcher denial must attribute Refused carrying the Denied cause"
    );
    assert!(executor.calls().is_empty());
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(sess.state(), SessionState::Failed);
}

// Proof 6: prompt-gate refusal checks the execution bound before attributing.

#[test]
fn prompt_gate_refusal_checks_execution_bound_before_attributing() {
    // The cap-full branch (`executions.len() >= MAX_EXECUTIONS_PER_AGENT`
    // failing with `CallLimitExceeded` carrying that limit, identical to the
    // dispatch loop and `provider_unknown`) is defense in depth: the bus hard
    // cap stops reachable turns at eight successes, so no public-API turn can
    // fill the vector to the shared bound. This test therefore pins both
    // halves: the guard constructor shape on a locally built full vector, and
    // the unchanged normal refusal after prior successes on a live turn.
    let mut issuer = IdIssuer::default();
    let mut full: Vec<ExecutionRecord> = Vec::with_capacity(MAX_EXECUTIONS_PER_AGENT);
    for _ in 0..MAX_EXECUTIONS_PER_AGENT {
        full.push(ExecutionRecord {
            execution_id: issuer.execution(),
            tool: "workspace_read".to_owned(),
            status: ToolStatus::Success,
            result_disposition: ResultDisposition::Accepted,
        });
    }
    assert_eq!(
        full.len(),
        MAX_EXECUTIONS_PER_AGENT,
        "full vector must meet the shared bound"
    );
    assert!(
        full.len() >= MAX_EXECUTIONS_PER_AGENT,
        "full vector must trip the guard condition"
    );
    let capped = ToolError::CallLimitExceeded {
        limit: MAX_EXECUTIONS_PER_AGENT,
    };
    assert!(
        matches!(
            capped,
            ToolError::CallLimitExceeded { limit } if limit == MAX_EXECUTIONS_PER_AGENT
        ),
        "cap refusal must carry the shared bound"
    );
    // Live turn below the cap: eight reads succeed, then the denied write
    // still refuses as the prompt cause (not the cap error), attributing one
    // more record while staying within the bound. The session stays at
    // `Inspect` so the prompt refusal fires before the tier floor, as in the
    // denial proof.
    let sess = session();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    for _ in 0..8 {
        provider.push_turn(tool_turn(
            "prior read",
            vec![ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            }],
        ));
    }
    provider.push_turn(tool_turn("one write", vec![write_call("a")]));
    provider.push_turn(tool_turn("unreached", Vec::new()));
    let config = AgentConfig {
        max_rounds: 12,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(AllowAll),
        sess.clone(),
        config,
    );
    agent.set_prompt(deny_write_prompt());
    let mut executor = FakeToolExecutor::new();
    for _ in 0..8 {
        executor.push_success("ok", b"data".to_vec());
    }
    executor.push_success("must never run", b"nope".to_vec());
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "fake-chat", "hi", &[], &mut sink, NOW_MS);

    let error = failed_tool_error(&outcome);
    assert!(
        matches!(
            error,
            ToolError::Denied { ref name, ref reason }
            if name == "workspace_write"
                && reason == "prompt denies tool 'workspace_write'"
        ),
        "refusal after prior successes must stay denied"
    );
    assert!(
        !matches!(error, ToolError::CallLimitExceeded { .. }),
        "refusal below the cap must not report the cap"
    );
    assert_eq!(agent.executions().len(), 9);
    assert!(
        agent.executions().len() <= MAX_EXECUTIONS_PER_AGENT,
        "refusal must keep the bound"
    );
    assert!(
        agent.executions()[..8]
            .iter()
            .all(|record| matches!(record.status, ToolStatus::Success)),
        "prior successes must stay recorded"
    );
    let status = &agent.executions()[8].status;
    assert!(status.is_admission_refusal());
    assert!(
        matches!(
            status,
            ToolStatus::Refused {
                cause: ToolError::Denied { .. }
            }
        ),
        "final refusal must stay admission-only"
    );
    assert_eq!(executor.calls().len(), 8);
    assert_eq!(agent.provider_mut().complete_calls(), 9);
    assert_eq!(sess.state(), SessionState::Failed);
}

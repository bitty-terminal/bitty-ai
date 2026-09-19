//! Structured exec-result schema disclosure proofs (AI-0077, AIQ-37 narrowing).
//!
//! Failures, truncation, and Unknown outcomes must be disclosed, never
//! substituted. The runtime already carries the schema pieces
//! ([`ExecOutcome`], [`ToolStatus`], the S-2 externalization-failure rule,
//! truncation accounting, the reason bound); this file pins the disclosure
//! surface end to end through the public API.
//!
//! Deterministic and offline: scripted [`FakeProvider`] turns (observed
//! through a request-capturing wrapper wherever the provider-visible message
//! mapping must be asserted), scripted [`FakeToolExecutor`] outcomes, and
//! caller-supplied `now_ms`. No network, no secrets, no wall clock, no
//! threads.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `agent_turn_semantics.rs`: multi-turn continuation, cancel/resume,
//!   escalated-Unknown session semantics, round/cost-fuse interaction,
//!   stream sequence continuity. Nothing here re-asserts those shapes.
//! - `runtime_fail_closed.rs`: fail-closed mechanics (S-2 store-full,
//!   denial attribution, oversized results). S-2 and denial appear here
//!   only for the disclosure fields that file leaves out (typed `Display`,
//!   card words, no-card-on-result-rejection).
//! - `context.rs` unit tests: truncation ordering/counting mechanics; here
//!   truncation appears as provider-visible disclosure plus the accounting
//!   fields through the public [`assemble`] API in a multi-provider shape.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, ArtifactStore, AuthContext, AuthDecision, ContextError,
    ContextPriority, ContextRecord, ContextRequest, ExecOutcome, FakeProvider, FakeToolExecutor,
    FragmentKind, IdIssuer, ModelDescriptor, ModelProvider, ProviderError, ProviderTurn,
    ProviderUsage, RecordBody, ResultDisposition, SessionState, StableId, StreamSink,
    ToolAuthorizer, ToolBus, ToolCallRequest, ToolError, ToolRegistry, ToolSpec, ToolStatus,
    TurnRequest, VecSink, assemble, validate_chunk,
};
use bitty_ai_runtime::{AgentSession, MAX_RECONCILE_REASON_BYTES};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook so these tests isolate the disclosure schema from
/// authorization. Production wiring installs the host capability-plus-consent
/// hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

/// Test-only provider wrapper that records every [`TurnRequest`] before
/// delegating to the scripted [`FakeProvider`]. The agent's
/// `execution_message` output is otherwise unobservable (it is consumed by
/// the next provider round), so this double is the disclosure probe for the
/// provider-visible half of the status mapping.
struct CapturingProvider {
    inner: FakeProvider,
    seen: Vec<TurnRequest>,
}

impl CapturingProvider {
    fn new(inner: FakeProvider) -> Self {
        Self {
            inner,
            seen: Vec::new(),
        }
    }

    fn requests(&self) -> &[TurnRequest] {
        &self.seen
    }
}

impl ModelProvider for CapturingProvider {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    fn list_models(&self) -> Vec<ModelDescriptor> {
        self.inner.list_models()
    }

    fn complete(&mut self, request: &TurnRequest) -> Result<ProviderTurn, ProviderError> {
        self.seen.push(request.clone());
        self.inner.complete(request)
    }

    fn scripted_turns_remaining(&self) -> usize {
        self.inner.scripted_turns_remaining()
    }

    fn complete_calls(&self) -> u64 {
        self.inner.complete_calls()
    }
}

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

fn read_tool_bus() -> ToolBus {
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
    ToolBus::new(registry).with_authorizer(AllowAll)
}

fn tool_turn(text: &str, input_tokens: u32, output_tokens: u32) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"a"}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn final_turn(text: &str, input_tokens: u32, output_tokens: u32) -> ProviderTurn {
    ProviderTurn {
        text: text.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn run<P: ModelProvider>(
    agent: &mut Agent<P>,
    executor: &mut FakeToolExecutor,
    prompt: &str,
    seeds: &[ContextRecord],
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, seeds, sink, NOW_MS)
}

fn seed_record(id: &str, summary: &str, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms: NOW_MS,
        priority: ContextPriority::Normal,
        summary: summary.to_owned(),
        body: RecordBody::Inline(vec![b'x'; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

/// Concatenated ToolCard bytes from a sink, asserting every chunk is
/// well-framed. Concatenation (not per-chunk matching) keeps the proof
/// robust to fragment splits: splits are byte-exact, so the joined text is
/// the card the runtime emitted.
fn tool_card_text(sink: &VecSink) -> String {
    let mut bytes = Vec::new();
    for chunk in sink.chunks() {
        validate_chunk(chunk).expect("disclosure must arrive well-framed");
        if chunk.fragment.kind == FragmentKind::ToolCard {
            bytes.extend_from_slice(&chunk.fragment.bytes);
        }
    }
    String::from_utf8(bytes).expect("cards are UTF-8")
}

// Area 1: every `ExecOutcome` variant renders its disclosure fields.

#[test]
fn completed_discloses_final_text_and_records() {
    let mut inner = FakeProvider::new("bitty-fake").expect("valid id");
    inner.push_turn(tool_turn("reading", 1, 0));
    inner.push_turn(final_turn("all done", 1, 0));
    let provider = CapturingProvider::new(inner);
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "summarize", &[], &mut sink);

    let ExecOutcome::Completed { text } = &outcome else {
        panic!("turn must complete, got: {outcome:?}");
    };
    assert_eq!(text, "all done");
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    // The L0 record discloses tool attribution inline with the payload.
    assert_eq!(agent.tool_records().len(), 1);
    assert_eq!(
        agent.tool_records()[0].summary,
        "tool=workspace_read read ok"
    );
    assert!(matches!(
        &agent.tool_records()[0].body,
        RecordBody::Inline(data) if data == b"data"
    ));
    // The success dispatch is disclosed to the next provider round.
    let requests = agent.provider_mut().requests();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1]
            .messages
            .iter()
            .any(|message| message.content == "workspace_read -> ok: read ok"),
        "round two must carry the success message, got: {:?}",
        requests[1].messages
    );
}

#[test]
fn failed_typed_error_display_is_single_line() {
    // Every `AgentError` variant renders a single-line typed disclosure: the
    // exact strings pin the schema, the newline check pins log safety.
    let cases: Vec<(AgentError, &str)> = vec![
        (
            AgentError::RoundLimitExceeded { limit: 4 },
            "round limit of 4 exceeded",
        ),
        (
            AgentError::CostCeilingExceeded {
                limit: 5,
                actual: 9,
            },
            "turn cost 9 exceeded ceiling 5",
        ),
        (
            AgentError::UnknownUnresolved {
                tool: "workspace_read".to_owned(),
                reason: "still writing".to_owned(),
                attempts: 2,
                dispatched: 1,
            },
            "tool workspace_read effect unreconciled after 2 queries (1 dispatched): still writing",
        ),
        (
            AgentError::Session(bitty_ai_runtime::SessionError::AlreadyTerminated {
                state: SessionState::Failed,
            }),
            "session: session already terminated (Failed)",
        ),
        (
            AgentError::Context(ContextError::ArtifactStoreFull {
                reason: "at most 262144 retained bytes".to_owned(),
            }),
            "context: artifact store full: at most 262144 retained bytes",
        ),
        (
            AgentError::Tool(ToolError::UnknownTool {
                name: "ghost_tool".to_owned(),
            }),
            "tool: unknown tool: ghost_tool",
        ),
        (
            AgentError::Tool(ToolError::Denied {
                name: "workspace_read".to_owned(),
                reason: "host policy refused".to_owned(),
            }),
            "tool: tool workspace_read denied: host policy refused",
        ),
        (
            AgentError::Provider(ProviderError::UnknownModel {
                name: "ghost-model".to_owned(),
            }),
            "provider: unknown model: ghost-model",
        ),
        (
            AgentError::Stream(bitty_ai_runtime::StreamError::OversizedFragment {
                limit: 64 * 1024,
                actual: 64 * 1024 + 1,
            }),
            "stream: fragment of 65537 bytes exceeds 65536 byte limit",
        ),
    ];
    assert_eq!(cases.len(), 9);
    for (error, expected) in &cases {
        let display = error.to_string();
        assert_eq!(&display, expected, "schema drift for {error:?}");
        assert!(
            !display.contains(['\n', '\r']),
            "typed Display must not carry newlines: {display:?}"
        );
    }
}

#[test]
fn failed_unknown_tool_renders_typed_error() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "calling ghost".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "ghost_tool".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    let ExecOutcome::Failed { error } = &outcome else {
        panic!("unknown tool must fail the turn, got: {outcome:?}");
    };
    assert_eq!(
        error,
        &AgentError::Tool(ToolError::UnknownTool {
            name: "ghost_tool".to_owned(),
        })
    );
    assert_eq!(error.to_string(), "tool: unknown tool: ghost_tool");
    // The refused turn dispatches nothing and attributes nothing.
    assert!(agent.executions().is_empty());
    assert!(executor.calls().is_empty());
    // The provider round ran (script consumed), then the gate refused.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 0);
}

#[test]
fn failed_unknown_model_renders_typed_error_without_io() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("unreached", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = agent.run_turn(&mut executor, "ghost-model", "hi", &[], &mut sink, NOW_MS);

    let ExecOutcome::Failed { error } = &outcome else {
        panic!("unknown model must fail the turn, got: {outcome:?}");
    };
    assert_eq!(
        error,
        &AgentError::Provider(ProviderError::UnknownModel {
            name: "ghost-model".to_owned(),
        })
    );
    assert_eq!(error.to_string(), "provider: unknown model: ghost-model");
    // Provider validation fails before any script, dispatch, or record.
    assert_eq!(agent.provider_mut().complete_calls(), 0);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
}

#[test]
fn canceled_counts_scale_with_dispatches() {
    // Beyond the 0/1 shapes proven elsewhere: two dispatched effects report
    // `dispatched: 2`, proving the counts are real counts, not constants.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "two reads".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{"path":"b"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(final_turn("unreached", 1, 0));
    let sess = session();
    let mut executor = FakeToolExecutor::new().with_cancel_on_call(sess.clone(), 2);
    executor.push_success("first ok", b"a".to_vec());
    executor.push_success("second ok", b"b".to_vec());
    let mut agent = Agent::new(provider, read_tool_bus(), sess, AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            outcome,
            ExecOutcome::Canceled {
                dispatched: 2,
                unknown: 0
            }
        ),
        "cancel after two dispatches must count both, got: {outcome:?}"
    );
    assert_eq!(agent.executions().len(), 2);
    assert!(
        agent
            .executions()
            .iter()
            .all(|record| matches!(record.status, ToolStatus::Success))
    );
    // Cancellation stopped the turn: no follow-up provider round ran.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
}

#[test]
fn unknown_discloses_reason_and_dispatched() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "ack lost after dispatch".to_owned(),
    });
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    let ExecOutcome::Unknown { reason, dispatched } = &outcome else {
        panic!("uncertain effect must end Unknown, got: {outcome:?}");
    };
    assert_eq!(*dispatched, 1);
    assert_eq!(
        reason,
        "tool effect uncertain; reconcile before retry (tool workspace_read)"
    );
    assert_eq!(agent.executions().len(), 1);
    // A clean executor reason passes through unmangled.
    assert!(
        matches!(
            &agent.executions()[0].status,
            ToolStatus::Unknown { reason } if reason == "ack lost after dispatch"
        ),
        "unexpected record: {:?}",
        agent.executions()[0]
    );
    assert_eq!(agent.session().state(), SessionState::Active);
}

// Area 2: every `ToolStatus` maps to its message and card word, never silent.

#[test]
fn success_maps_to_ok_word_and_message() {
    let mut inner = FakeProvider::new("bitty-fake").expect("valid id");
    inner.push_turn(tool_turn("reading", 1, 0));
    inner.push_turn(final_turn("done", 1, 0));
    let mut agent = Agent::new(
        CapturingProvider::new(inner),
        read_tool_bus(),
        session(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert_eq!(
        tool_card_text(&sink),
        "tool=workspace_read status=ok summary=read ok"
    );
}

#[test]
fn denial_maps_to_denied_word_and_record() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("reading", 1, 0));
    provider.push_turn(final_turn("unreached", 1, 0));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess.clone(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::Denied {
        name: "workspace_read".to_owned(),
        reason: "host policy refused".to_owned(),
    });
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    let ExecOutcome::Failed { error } = &outcome else {
        panic!("host denial must fail the turn, got: {outcome:?}");
    };
    assert_eq!(
        error.to_string(),
        "tool: tool workspace_read denied: host policy refused"
    );
    // The denial is disclosed on the card before the turn terminates.
    assert_eq!(
        tool_card_text(&sink),
        "tool=workspace_read status=denied summary=host denied execution"
    );
    assert!(
        matches!(
            &agent.executions()[0].status,
            ToolStatus::Denied { reason } if reason == "host policy refused"
        ),
        "unexpected record: {:?}",
        agent.executions()[0]
    );
    // No follow-up round ran; the session is terminally failed.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(sess.state(), SessionState::Failed);
}

#[test]
fn unknown_maps_to_unknown_word_and_record() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "ack lost".to_owned(),
    });
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    assert_eq!(
        tool_card_text(&sink),
        "tool=workspace_read status=unknown summary=effect uncertain; reconcile before retry"
    );
    assert!(
        matches!(
            &agent.executions()[0].status,
            ToolStatus::Unknown { reason } if reason == "ack lost"
        ),
        "unexpected record: {:?}",
        agent.executions()[0]
    );
}

#[test]
fn rejected_result_keeps_success_and_emits_no_card() {
    // AI-RUN-004: the executor acknowledged the effect but the returned
    // payload exceeded the result bound. The effect status stays `Success`
    // (the host ran it) and the acceptance axis carries the typed bound
    // failure; the turn fails closed with that same typed error. No card is
    // emitted because there is no accepted payload to disclose, and no
    // empty-bytes substitute is recorded.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("reading", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("too big", vec![b'y'; 17 * 1024]);
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    let actual = 17 * 1024;
    let limit = 16 * 1024;
    let detail = format!("tool result of {actual} bytes exceeds {limit} byte limit");
    let ExecOutcome::Failed { error } = &outcome else {
        panic!("a rejected over-bound result must fail the turn with a typed cause");
    };
    assert_eq!(error.to_string(), format!("tool: {detail}"));
    assert_eq!(tool_card_text(&sink), "");
    // The effect happened: success is preserved, never relabeled as an
    // executed failure; only the acceptance axis is rejected.
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(
        matches!(
            &agent.executions()[0].result_disposition,
            ResultDisposition::Rejected {
                cause: ToolError::ResultTooLarge { .. }
            }
        ),
        "unexpected record: {:?}",
        agent.executions()[0]
    );
    assert_eq!(agent.tool_records().len(), 0);
    assert_eq!(agent.session().state(), SessionState::Failed);
}

// Area 3: S-2 externalization failure fails the turn with a typed disclosure.

#[test]
fn s2_store_full_fails_turn_with_typed_disclosure() {
    // Mechanics (63 large seeds fill the store, the tool result cannot
    // externalize) are proven in `runtime_fail_closed.rs`; this test pins
    // the disclosure schema the turn renders instead of empty bytes: the
    // exact typed `Display`, the attributed-but-unrecorded split, and the
    // terminal session.
    let seeds: Vec<ContextRecord> = (0..63)
        .map(|index| seed_record(&format!("seed-{index}"), &format!("summary-{index}"), 4100))
        .collect();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("reading", 1, 0));
    provider.push_turn(final_turn("unreached", 1, 0));
    let sess = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        sess.clone(),
        AgentConfig {
            context_budget_bytes: 64 * 1024,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("large read", vec![b'y'; 5120]);
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "go", &seeds, &mut sink);

    let ExecOutcome::Failed { error } = &outcome else {
        panic!("store-full externalization must fail the turn, got: {outcome:?}");
    };
    assert_eq!(
        error.to_string(),
        format!(
            "context: artifact store full: at most {} retained bytes",
            256 * 1024
        )
    );
    assert!(!error.to_string().contains(['\n', '\r']));
    // The effect is attributed (it happened) but no Inline(empty)
    // substitute is recorded for the lost bytes.
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(
        agent.tool_records().is_empty(),
        "no Inline(empty) substitute may be recorded, got: {:?}",
        agent.tool_records()
    );
    assert_eq!(sess.state(), SessionState::Failed);
}

// Area 4: truncation accounting surfaces counted bytes, never silent drops.

#[test]
fn truncation_accounting_surfaces_counted_bytes() {
    fn record(
        id: &str,
        provider: &str,
        priority: ContextPriority,
        summary: &str,
        body_len: usize,
    ) -> ContextRecord {
        ContextRecord {
            id: id.to_owned(),
            provider: provider.to_owned(),
            owner: StableId::new("term-1").expect("valid stable id"),
            generation: 1,
            collected_at_ms: NOW_MS,
            priority,
            summary: summary.to_owned(),
            body: RecordBody::Inline(vec![b'x'; body_len]),
            supersedes: None,
            is_untrusted_surface: false,
        }
    }

    // Footprints: keep 4 + 100 = 104; drop-b 6 + 150 = 156; drop-c 6 + 50.
    let records = vec![
        record("keep-1", "workspace", ContextPriority::High, "keep", 100),
        record("drop-2", "project", ContextPriority::Normal, "drop-b", 150),
        record("drop-3", "git", ContextPriority::Low, "drop-c", 50),
    ];
    let mut store = ArtifactStore::new();
    let request = ContextRequest {
        max_tokens: None,
        max_bytes: Some(104),
        current_generation: 1,
    };

    let assembled = assemble(&records, &mut store, &request).expect("assemble");

    assert_eq!(assembled.records.len(), 1);
    assert_eq!(assembled.records[0].id, "keep-1");
    assert_eq!(
        assembled.omitted_ids,
        vec!["drop-2".to_owned(), "drop-3".to_owned()]
    );
    assert_eq!(assembled.truncated_bytes, 156 + 56);
    assert_eq!(
        assembled.truncated_tokens_estimate,
        ContextRequest::estimate_tokens(212)
    );
    assert_eq!(
        assembled.truncated_providers,
        vec!["project".to_owned(), "git".to_owned()]
    );
    assert_eq!(assembled.externalized, 0);
    assert!(assembled.pruned_ids.is_empty());
    assert_eq!(assembled.budget_bytes, 104);
}

#[test]
fn truncation_never_leaks_dropped_records_to_provider() {
    // End-to-end disclosure: the dropped seed's bytes never reach the
    // provider request, and the turn still completes on what survived.
    let seeds = vec![
        seed_record("keep-1", "keep me", 100),
        seed_record("drop-2", "drop me", 150),
    ];
    let mut inner = FakeProvider::new("bitty-fake").expect("valid id");
    inner.push_turn(final_turn("done", 1, 0));
    let mut agent = Agent::new(
        CapturingProvider::new(inner),
        read_tool_bus(),
        session(),
        AgentConfig {
            context_budget_bytes: 200,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "show status", &seeds, &mut sink);

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    let requests = agent.provider_mut().requests();
    assert_eq!(requests.len(), 1);
    let contents: Vec<&str> = requests[0]
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert!(
        contents.iter().any(|content| content.contains("keep me")),
        "survivor must reach the provider: {contents:?}"
    );
    assert!(
        !contents.iter().any(|content| content.contains("drop me")),
        "truncated bytes must never leak to the provider: {contents:?}"
    );
}

// Area 5: the Unknown record keeps its reason bounded.

#[test]
fn unknown_record_reason_stays_bounded() {
    // `MAX_RECONCILE_REASON_BYTES` mirrors the runtime-wide
    // `MAX_REASON_BYTES` bound (512): every runtime-owned reason surface
    // shares one bound, and the Unknown execution record is one of them.
    let hostile = format!("bad\nreason {}", "x".repeat(MAX_RECONCILE_REASON_BYTES * 4));
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(tool_turn("maybe wrote", 1, 0));
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: hostile,
    });
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    let ToolStatus::Unknown { reason } = &agent.executions()[0].status else {
        panic!("unexpected record: {:?}", agent.executions()[0]);
    };
    assert!(
        reason.len() <= MAX_RECONCILE_REASON_BYTES,
        "Unknown reason of {} bytes exceeds the bound: {reason:?}",
        reason.len()
    );
    assert!(
        !reason.contains(['\n', '\r']),
        "Unknown reason must not carry newlines: {reason:?}"
    );
}

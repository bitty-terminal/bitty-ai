//! Fail-closed integration tests for the deterministic single-agent loop.
//!
//! Every test is offline and deterministic: scripted [`FakeProvider`] turns,
//! scripted [`FakeToolExecutor`] outcomes, and caller-supplied `now_ms`. No
//! network, no secrets, no wall clock.

use bitty_ai_runtime::{
    Agent, AgentConfig, AgentError, AgentLevel, AuthContext, AuthDecision, ContextError,
    ContextPriority, ContextRecord, ElevationGrant, ExecOutcome, FakeProvider, FakeToolExecutor,
    FragmentKind, ModelProvider, ProviderError, ProviderTurn, ProviderUsage, RecordBody,
    SessionState, StableId, StreamSink, ToolAuthorizer, ToolBus, ToolCallRequest, ToolError,
    ToolExecutor, ToolRegistry, ToolSpec, ToolStatus, ToolSuccess, VecSink,
};

const NOW_MS: u64 = 1_700_000_000_000;

/// Test-only allow hook. This exists so tests can isolate each fail-closed
/// layer (tier floor, registry, budget, cancellation) from authorization.
/// It must never ship behind a real seam: production wiring installs the
/// host capability-plus-consent hook instead.
struct AllowAll;
impl ToolAuthorizer for AllowAll {
    fn authorize(&self, _ctx: &AuthContext) -> AuthDecision {
        AuthDecision::Allow
    }
}

fn ids() -> (
    bitty_ai_runtime::AgentInstanceId,
    bitty_ai_runtime::RunId,
    bitty_ai_runtime::SessionId,
) {
    let mut issuer = bitty_ai_runtime::IdIssuer::default();
    (issuer.agent_instance(), issuer.run(), issuer.session())
}

fn session() -> bitty_ai_runtime::AgentSession {
    let (agent, run, session) = ids();
    bitty_ai_runtime::AgentSession::new(agent, run, session)
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

fn read_write_tool_bus() -> ToolBus {
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
    ToolBus::new(registry).with_authorizer(AllowAll)
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

fn run(
    agent: &mut Agent<FakeProvider>,
    executor: &mut dyn ToolExecutor,
    prompt: &str,
    seeds: &[ContextRecord],
    sink: &mut VecSink,
) -> ExecOutcome {
    agent.run_turn(executor, "fake-chat", prompt, seeds, sink, NOW_MS)
}

#[test]
fn happy_path_streams_markdown_and_toolcard_then_completes() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "checking workspace".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{"path":"a"}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "done".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read 12 bytes", b"hello world!".to_vec());
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = run(
        &mut agent,
        &mut executor,
        "summarize",
        &[seed_record("s1", "manifest", 10)],
        &mut sink,
    );

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert_eq!(executor.calls().len(), 1);
    let kinds: Vec<FragmentKind> = sink.chunks().iter().map(|c| c.fragment.kind).collect();
    assert!(kinds.contains(&FragmentKind::Markdown));
    assert!(kinds.contains(&FragmentKind::ToolCard));
    // Three logical turns were streamed: assistant text, tool card, final.
    assert_eq!(sink.chunks().len(), 3);
    // Framing holds within every emitted chunk.
    for chunk in sink.chunks() {
        assert!(chunk.seq < chunk.total);
        assert_eq!(chunk.is_final, chunk.seq + 1 == chunk.total);
    }
}

#[test]
fn cancel_before_dispatch_touches_nothing() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "never seen".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let session = session();
    session.cancel();
    let mut agent = Agent::new(provider, read_tool_bus(), session, AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(
        outcome,
        ExecOutcome::Canceled {
            dispatched: 0,
            unknown: 0
        }
    ));
    assert!(sink.is_empty());
    assert!(agent.executions().is_empty());
    assert!(executor.calls().is_empty());
    // No provider I/O happened: script and call counter untouched.
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 0);
}

#[test]
fn cancel_after_dispatch_reports_reconciled_cancel() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "reading".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let session = session();
    let executor = FakeToolExecutor::new().with_cancel_on_call(session.clone(), 1);
    let mut executor = executor;
    executor.push_success("read ok", b"data".to_vec());
    let mut agent = Agent::new(provider, read_tool_bus(), session, AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(
        outcome,
        ExecOutcome::Canceled {
            dispatched: 1,
            unknown: 0
        }
    ));
    assert_eq!(agent.executions().len(), 1);
    // Cancellation landed after dispatch: the follow-up turn never ran.
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
}

#[test]
fn cancel_after_dispatch_with_unknown_effect_needs_reconcile() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "writing".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let session = session();
    let mut executor = FakeToolExecutor::new().with_cancel_on_call(session.clone(), 1);
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "ack lost after dispatch".to_owned(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session, AgentConfig::default());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
}

#[test]
fn budget_exceeded_fails_before_dispatch() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "never produced".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let config = AgentConfig {
        context_budget_bytes: 8,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(provider, read_tool_bus(), session(), config);
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(
        &mut agent,
        &mut executor,
        "this prompt alone exceeds eight bytes",
        &[],
        &mut sink,
    );

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Provider(ProviderError::BudgetExceeded { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
    // Budget failure consumed no scripted turn.
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
}

#[test]
fn unknown_tool_fails_with_no_dispatch() {
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

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::UnknownTool { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
}

#[test]
fn deny_by_default_without_authorizer() {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "workspace_read",
                "Read a bounded workspace path",
                Vec::new(),
                "workspace.read",
                true,
            )
            .expect("valid spec"),
        )
        .expect("capacity");
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "reading".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    // No authorizer installed: fail closed (FS-AI7).
    let mut agent = Agent::new(
        provider,
        ToolBus::new(registry),
        session(),
        AgentConfig::default(),
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::Denied { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
}

#[test]
fn mid_turn_generation_rotation_downgrades_remaining_writes() {
    // P0-1 (S-1): per-call JIT `AuthBase`. A `Workspace` turn with two
    // writes rotates to `Inspect` during the first dispatch; the second
    // write must be denied at the fresh tier while the first stays recorded.
    struct AllowElevation;
    impl ElevationGrant for AllowElevation {
        fn elevation_granted(&self, _current: AgentLevel, _requested: AgentLevel) -> bool {
            true
        }
    }
    struct RotatingExecutor {
        session: bitty_ai_runtime::AgentSession,
        rotate_on_call: usize,
        calls: Vec<(String, Vec<u8>)>,
    }
    impl ToolExecutor for RotatingExecutor {
        fn execute(
            &mut self,
            tool: &str,
            arguments: &[u8],
            _now_ms: u64,
        ) -> Result<ToolSuccess, ToolError> {
            self.calls.push((tool.to_owned(), arguments.to_vec()));
            if self.calls.len() == self.rotate_on_call {
                self.session.rotate_generation();
            }
            ToolSuccess::new("write ok".to_owned(), b"ok".to_vec())
        }
    }
    let session = session();
    session
        .elevate(AgentLevel::Workspace, &AllowElevation)
        .expect("grant allows elevation");
    assert_eq!(session.generation(), 1);
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "two writes".to_owned(),
        tool_calls: vec![
            ToolCallRequest {
                name: "workspace_write".to_owned(),
                arguments: br#"{"path":"a"}"#.to_vec(),
            },
            ToolCallRequest {
                name: "workspace_write".to_owned(),
                arguments: br#"{"path":"b"}"#.to_vec(),
            },
        ],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(
        provider,
        read_write_tool_bus(),
        session.clone(),
        AgentConfig::default(),
    );
    let mut executor = RotatingExecutor {
        session: session.clone(),
        rotate_on_call: 1,
        calls: Vec::new(),
    };
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::Denied { .. })
            }
        ),
        "remaining write after rotation must be denied, got: {outcome:?}"
    );
    // Rotation happened mid-turn through the shared handle.
    assert_eq!(session.level(), AgentLevel::Inspect);
    assert_eq!(session.generation(), 2);
    // Second dispatch never reached the host: fail closed before execute.
    assert_eq!(executor.calls.len(), 1);
    // Already-dispatched effect is kept, never rolled back; the denied
    // remainder is recorded as a failed attribution.
    assert_eq!(agent.executions().len(), 2);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(matches!(
        agent.executions()[1].status,
        ToolStatus::Failed { .. }
    ));
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(session.state(), SessionState::Failed);
}

#[test]
fn oversized_arguments_fail_with_no_dispatch() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "big args".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: vec![b'{'; 17 * 1024],
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::ArgumentsTooLarge { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
}

#[test]
fn oversized_result_fails_after_single_dispatch() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "reading".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("too big", vec![b'y'; 17 * 1024]);
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::ResultTooLarge { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    // The dispatch happened exactly once and is recorded as failed.
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Failed { .. }
    ));
}

#[test]
fn full_artifact_store_fails_turn_instead_of_empty_inline() {
    // P0-2 (S-2): 63 large seeds externalize during assembly (each 4100 B
    // over the 4 KiB threshold with distinct summaries so L1 dedupe keeps
    // all 63; 63 * 4100 = 258_300 retained bytes under the 256 KiB cap), so
    // the following 5120 B tool result cannot externalize. The turn must
    // fail as `Context(ArtifactStoreFull)` with the session marked failed,
    // never `Completed` with an empty inline substitute.
    let seeds: Vec<ContextRecord> = (0..63)
        .map(|index| seed_record(&format!("seed-{index}"), &format!("summary-{index}"), 4100))
        .collect();
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "reading".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let session = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session.clone(),
        AgentConfig {
            context_budget_bytes: 64 * 1024,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    executor.push_success("large read", vec![b'y'; 5120]);
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "go", &seeds, &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Context(ContextError::ArtifactStoreFull { .. })
            }
        ),
        "store-full externalization must fail the turn, got: {outcome:?}"
    );
    // The host dispatch happened (effect attributed), but no empty inline
    // record was substituted for the lost bytes.
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(agent.executions()[0].status, ToolStatus::Success));
    assert!(
        agent.tool_records().is_empty(),
        "no Inline(empty) substitute may be recorded, got: {:?}",
        agent.tool_records()
    );
    assert_eq!(session.state(), SessionState::Failed);
    assert_eq!(agent.provider_mut().complete_calls(), 1);
}

#[test]
fn oversized_chunk_rejected_by_sink() {
    let mut sink = VecSink::new();
    let big = bitty_ai_runtime::StreamChunk {
        seq: 0,
        total: 1,
        is_final: true,
        fragment: bitty_ai_runtime::Fragment::markdown(vec![
            b'z';
            bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES
                + 1
        ]),
    };
    let result = sink.emit(big);
    assert!(matches!(
        result,
        Err(bitty_ai_runtime::StreamError::OversizedFragment { .. })
    ));
    assert!(sink.is_empty());
}

#[test]
fn unknown_outcome_stops_turn_without_session_failure() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "maybe wrote".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::EffectUnknown {
        name: "workspace_read".to_owned(),
        reason: "host crashed before ack".to_owned(),
    });
    let session = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session.clone(),
        AgentConfig::default(),
    );
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(matches!(outcome, ExecOutcome::Unknown { .. }));
    // Session stays usable for reconciliation and retry (not Failed).
    assert_eq!(session.state(), bitty_ai_runtime::SessionState::Active);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Unknown { .. }
    ));
}

#[test]
fn l1_assembly_prunes_and_externalizes_inside_turn() {
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "final".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut new = seed_record("read-v2", "outline of foo", 10);
    new.supersedes = Some("read-v1".to_owned());
    let seeds = vec![seed_record("read-v1", "outline of foo", 10), new];
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session(),
        AgentConfig {
            context_budget_bytes: 64 * 1024,
            ..AgentConfig::default()
        },
    );
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "go", &seeds, &mut sink);

    assert!(matches!(outcome, ExecOutcome::Completed { .. }));
    // One provider round ran despite two seed records: L1 collapsed them.
    assert_eq!(agent.provider_mut().complete_calls(), 1);
}

#[test]
fn legacy_dotted_tool_name_from_model_fails_as_invalid() {
    // The model is untrusted: a legacy dotted name (`terminal.read_zone`,
    // pre-AI-0012 slice vocabulary) must surface as `InvalidName` (`TB-2`),
    // never as a dispatch or a silent unknown-tool substitute.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "calling legacy".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "terminal.read_zone".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::InvalidName { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
}

#[test]
fn tool_call_burst_fails_before_any_dispatch() {
    // Nine calls in one assistant turn exceed the `TB-6` per-turn cap: the
    // turn fails with `CallLimitExceeded` and dispatches nothing.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "burst".to_owned(),
        tool_calls: (0..9)
            .map(|_| ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{}"#.to_vec(),
            })
            .collect(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = Agent::new(provider, read_tool_bus(), session(), AgentConfig::default());
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::CallLimitExceeded { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
}

#[test]
fn context_request_resolves_token_first_budget() {
    let request = bitty_ai_runtime::ContextRequest {
        max_tokens: Some(64),
        max_bytes: Some(1_000_000),
        current_generation: 1,
    };
    assert_eq!(request.effective_budget_bytes(), 256);
}

#[test]
fn relaxed_config_still_clamped_by_bus_constant() {
    // P2-5 (AI-0056): effective cap = min(config, MAX_TOOL_CALLS_PER_TURN).
    // A config of 16 does not relax the hard bus ceiling of 8: nine calls in
    // one batch fail whole-batch admission in `ToolBus::precheck` with
    // limit 8 and dispatch nothing.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "burst".to_owned(),
        tool_calls: (0..9)
            .map(|_| ToolCallRequest {
                name: "workspace_read".to_owned(),
                arguments: br#"{}"#.to_vec(),
            })
            .collect(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let config = AgentConfig {
        max_tool_calls_per_turn: 16,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(provider, read_tool_bus(), session(), config);
    let mut executor = FakeToolExecutor::new();
    executor.push_success("read ok", b"data".to_vec());
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::CallLimitExceeded { limit: 8 })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert!(executor.calls().is_empty());
    assert!(agent.executions().is_empty());
}

#[test]
fn executor_denial_is_attributed_then_fails_the_turn() {
    // A host-side refusal returned by the executor becomes an attributed
    // `Denied` execution and then fails the turn (`TB-7`); the refusal is
    // never retried and the follow-up round is not consumed.
    let mut provider = FakeProvider::new("bitty-fake").expect("valid id");
    provider.push_turn(ProviderTurn {
        text: "reading".to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: "workspace_read".to_owned(),
            arguments: br#"{}"#.to_vec(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    provider.push_turn(ProviderTurn {
        text: "unreached".to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut executor = FakeToolExecutor::new();
    executor.push_error(ToolError::Denied {
        name: "workspace_read".to_owned(),
        reason: "host policy refused".to_owned(),
    });
    let session = session();
    let mut agent = Agent::new(
        provider,
        read_tool_bus(),
        session.clone(),
        AgentConfig::default(),
    );
    let mut sink = VecSink::new();

    let outcome = run(&mut agent, &mut executor, "hi", &[], &mut sink);

    assert!(
        matches!(
            &outcome,
            ExecOutcome::Failed {
                error: AgentError::Tool(ToolError::Denied { .. })
            }
        ),
        "unexpected outcome: {outcome:?}"
    );
    // The refusal was attributed exactly once and never dispatched again.
    assert_eq!(executor.calls().len(), 1);
    assert_eq!(agent.executions().len(), 1);
    assert!(matches!(
        agent.executions()[0].status,
        ToolStatus::Denied { .. }
    ));
    // The tool card carries the denial before the turn terminates.
    let cards: Vec<&bitty_ai_runtime::StreamChunk> = sink
        .chunks()
        .iter()
        .filter(|chunk| chunk.fragment.kind == FragmentKind::ToolCard)
        .collect();
    assert_eq!(cards.len(), 1);
    assert!(
        String::from_utf8_lossy(&cards[0].fragment.bytes).contains("status=denied"),
        "card: {}",
        String::from_utf8_lossy(&cards[0].fragment.bytes)
    );
    // No follow-up provider round ran; the session is terminally failed.
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 1);
    assert_eq!(agent.provider_mut().complete_calls(), 1);
    assert_eq!(session.state(), SessionState::Failed);
}

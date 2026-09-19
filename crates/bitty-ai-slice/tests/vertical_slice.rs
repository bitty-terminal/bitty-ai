//! Deterministic end-to-end and fail-closed proof for the harness.
//!
//! The real `bitty-ai-runtime` (`Agent` / `AgentSession` / `FakeProvider` /
//! context assembly / `ToolBus` / `VecSink`) is driven through the real
//! `bitty-ipc` bridge (`IpcBridge` + consent ledger + scoped method
//! registry). No network, no secrets, no wall-clock: the host peer is a
//! bounded loopback implementation and every timestamp is caller-supplied.
//! The loopback peer is a test double for the Bitty host, not a claim about
//! live data; the product path fails closed when the real host does not
//! implement a method.
//!
//! Vocabulary deltas vs the pre-AI-0012 slice (all forced by runtime
//! validated shapes; the `bitty-agent` mapping is later P1 work):
//!
//! - tool `terminal_read_zone` (was `terminal.read_zone`: runtime `TB-2`
//!   names reject `.`),
//! - provider id `local-deterministic` (was `local.deterministic`: runtime
//!   `MP-2` ids reject `.`),
//! - record owner `term-1` (was `inst-1/term-1`: a runtime `StableId` is one
//!   hierarchy level).

use bitty_ai_runtime::{
    AgentError, AgentLevel, AuthBase, ContextError, ExecOutcome, FakeToolExecutor, Fragment,
    FragmentKind, IdIssuer, ModelProvider, ProviderTurn, ProviderUsage, RecordBody, StreamChunk,
    StreamError, StreamSink, ToolBus, ToolCall, ToolCallRequest, ToolError, ToolRegistry, ToolSpec,
    ToolStatus, VecSink,
};
use bitty_ai_slice::{
    AllowReadOnly, HARNESS_MODEL, HARNESS_PROVIDER_ID, HARNESS_TOOL, HostPeer, IpcBridge,
    SliceError, SnapshotRequest, collect_terminal_context, harness_agent, scripted_provider,
    terminal_record, test_tool_registry,
};
use bitty_ipc::channel::{IpcRequest, IpcResponse};
use bitty_ipc::error::IpcError;
use bitty_ipc::rich_fragment::{FragmentData, FragmentIngestService};
use bitty_ipc::scope::{Scope, ScopeSet};

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;
const ANSWER: &str = "The last command printed `hello` with a zero exit status.";
/// Hard bus ceiling mirrored from the `TB-6` contract (`tool.rs`), asserted
/// here as a literal (not imported) because the constant is intentionally
/// crate-private to the bus boundary.
const BUS_CALL_CAP: usize = 8;

struct LoopbackHost {
    snapshot: Vec<u8>,
    serve_snapshot: bool,
}

impl LoopbackHost {
    fn serving(snapshot: Vec<u8>) -> Self {
        Self {
            snapshot,
            serve_snapshot: true,
        }
    }

    fn delegated() -> Self {
        Self {
            snapshot: Vec::new(),
            serve_snapshot: false,
        }
    }
}

impl HostPeer for LoopbackHost {
    fn serve(&mut self, request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        if request.method == "terminal.snapshot" && self.serve_snapshot {
            return Ok(IpcResponse::success(request.id, self.snapshot.clone())?);
        }
        Ok(IpcResponse::error(
            request.id,
            b"host does not implement this method".to_vec(),
        )?)
    }
}

/// Peer that counts served requests, so a refused bridge call can be shown
/// to have never reached the host.
struct CountingHost {
    served: usize,
}

impl CountingHost {
    fn new() -> Self {
        Self { served: 0 }
    }
}

impl HostPeer for CountingHost {
    fn serve(&mut self, request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        self.served += 1;
        Ok(IpcResponse::success(request.id, b"served".to_vec())?)
    }
}

fn snapshot() -> Vec<u8> {
    b"$ echo hello\nhello\n".to_vec()
}

fn request(max_bytes: usize) -> SnapshotRequest {
    SnapshotRequest::new("inst-1", "term-1", "output", "term-1", max_bytes)
}

fn bridge(granted: ScopeSet, consent: bool) -> IpcBridge {
    let mut bridge = IpcBridge::new("agent-0407", granted);
    if consent {
        bridge
            .grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");
    }
    bridge
}

fn tool_args() -> Vec<u8> {
    br#"{"zone":"output"}"#.to_vec()
}

/// One full deterministic turn: scripted provider round (answer draft + one
/// tool call), real tool dispatch of the harness tool, then a scripted final
/// round with the terminal answer and no further tool calls.
fn run_harness(context_budget_bytes: usize) -> (ExecOutcome, Vec<StreamChunk>, String, usize) {
    let mut bridge = bridge(ScopeSet::single(Scope::TerminalInspect), true);
    let mut peer = LoopbackHost::serving(snapshot());
    let seed = collect_terminal_context(&mut bridge, &mut peer, &request(4096), 1, NOW_MS)
        .expect("context collected");
    let mut provider =
        scripted_provider("Checking the terminal output.", Some(tool_args())).expect("provider");
    provider.push_turn(ProviderTurn {
        text: ANSWER.to_owned(),
        tool_calls: Vec::new(),
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let provider_id = provider.provider_id().to_owned();
    let mut agent = harness_agent(provider, context_budget_bytes).expect("agent");
    let mut executor = FakeToolExecutor::new();
    executor.push_success("terminal snapshot", snapshot());
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(
        &mut executor,
        HARNESS_MODEL,
        "what did my last command print?",
        &[seed],
        &mut sink,
        NOW_MS,
    );
    (
        outcome,
        sink.chunks().to_vec(),
        provider_id,
        executor.calls().len(),
    )
}

/// Direct runtime-to-transport projection used by the joint ingest proof: a
/// validated chunk becomes `FragmentData` carrying the runtime's own
/// continuous `seq` (`S-8` scheme A makes the runtime the turn's sequence
/// authority, so no renumbering is needed). Mirrors the mapping helper in
/// `fragment_mapping.rs` (test binaries cannot share it) minus the optional
/// zone.
fn map_chunk_to_fragment_data(
    chunk: &StreamChunk,
    terminal_id: &str,
    generation: u64,
) -> FragmentData {
    bitty_ai_runtime::validate_chunk(chunk).expect("runtime framing valid");
    FragmentData {
        terminal_id: terminal_id.to_owned(),
        generation,
        seq: u64::from(chunk.seq),
        zone: None,
        text: String::from_utf8(chunk.fragment.bytes.clone()).expect("fragment text is UTF-8"),
    }
}

#[test]
fn end_to_end_loop_is_deterministic() {
    let (first, first_chunks, first_provider, first_calls) = run_harness(32 * 1024);
    let (second, second_chunks, second_provider, second_calls) = run_harness(32 * 1024);
    assert_eq!(first, second, "the runtime must be deterministic");
    assert_eq!(first_chunks, second_chunks, "streamed chunks must match");
    assert_eq!(first_provider, HARNESS_PROVIDER_ID);
    assert_eq!(second_provider, HARNESS_PROVIDER_ID);
    assert_eq!(first_calls, 1, "exactly one tool dispatch per turn");
    assert_eq!(second_calls, 1, "exactly one tool dispatch per turn");

    let text = match &first {
        ExecOutcome::Completed { text } => text.clone(),
        other => panic!("turn must complete, got {other:?}"),
    };
    assert!(text.contains("hello"), "final answer mentions output");

    // Seed context record collected through the real bridge.
    let seed = terminal_record("term-1", 1, NOW_MS, snapshot()).expect("record");
    assert_eq!(seed.provider, "terminal");
    assert_eq!(seed.owner.as_str(), "term-1");
    assert_eq!(seed.generation, 1);
    assert_eq!(seed.collected_at_ms, NOW_MS);
    assert!(seed.is_untrusted_surface);
    match &seed.body {
        RecordBody::Inline(bytes) => assert_eq!(*bytes, snapshot()),
        RecordBody::Artifact(_) => panic!("small body stays inline"),
    }

    // Two provider rounds stream three single-fragment blocks: draft text,
    // then the tool card, then final text. `seq` is continuous across the
    // whole logical turn (S-8 scheme A) with a running water mark as `total`,
    // so the three single-fragment batches frame as [0/1, 1/2, 2/3]; each
    // block closes its own emission batch (`is_final`).
    assert_eq!(first_chunks.len(), 3);
    assert_eq!(first_chunks[0].fragment.kind, FragmentKind::Markdown);
    assert_eq!(first_chunks[1].fragment.kind, FragmentKind::ToolCard);
    assert_eq!(first_chunks[2].fragment.kind, FragmentKind::Markdown);
    for (index, chunk) in first_chunks.iter().enumerate() {
        bitty_ai_runtime::validate_chunk(chunk).expect("runtime framing valid");
        assert_eq!(
            (chunk.seq, chunk.total, chunk.is_final),
            (index as u32, index as u32 + 1, true)
        );
    }
    assert_eq!(
        first_chunks[0].fragment.bytes,
        b"Checking the terminal output.".to_vec()
    );
    let card = String::from_utf8_lossy(&first_chunks[1].fragment.bytes);
    assert!(card.contains(HARNESS_TOOL), "card names the tool");
    assert!(card.contains("status=ok"), "card records success");
    assert_eq!(first_chunks[2].fragment.bytes, ANSWER.as_bytes());
    assert!(first_chunks[1].fragment.bytes.len() <= 16 * 1024);
}

/// P1-5 joint proof: the three real blocks of one deterministic turn (draft
/// text, tool card, final text) ingest into the real `bitty-ipc`
/// `FragmentIngestService` through the direct `seq` projection. Before the
/// fix every block restarted at `seq 0`, so the second ingest failed as a
/// duplicate `(terminal_id, generation, seq)` key; the continuous runtime
/// sequence now passes the transport's own dedup check.
#[test]
fn streamed_turn_blocks_ingest_into_real_fragment_service() {
    let (outcome, chunks, _, _) = run_harness(32 * 1024);
    assert!(
        matches!(outcome, ExecOutcome::Completed { .. }),
        "turn must complete, got {outcome:?}"
    );
    assert_eq!(chunks.len(), 3, "draft text, tool card, final text");

    let mut service = FragmentIngestService::new();
    for chunk in &chunks {
        let data = map_chunk_to_fragment_data(chunk, "t:1", 1);
        let stored = service
            .ingest(data)
            .expect("continuous turn seq must pass the transport dedup key");
        assert!(stored.is_untrusted_surface);
    }
    assert_eq!(service.len(), 3);

    let drained = service.drain_bounded(8);
    assert_eq!(drained.len(), 3);
    assert_eq!(
        drained
            .iter()
            .map(|fragment| fragment.seq)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    for fragment in &drained {
        fragment.validate().expect("stored DTO validates");
        assert!(!fragment.truncated, "in-budget blocks are not truncated");
    }
    let joined: String = drained
        .iter()
        .map(|fragment| fragment.text.clone())
        .collect();
    assert!(joined.contains("Checking the terminal output."));
    assert!(joined.ends_with(ANSWER));
}

#[test]
fn unknown_host_method_fails_closed() {
    let mut bridge = bridge(ScopeSet::single(Scope::TerminalInspect), true);
    let mut host = LoopbackHost::serving(snapshot());
    let error = bridge
        .call("panel.context", b"{}", NOW_MS, &mut host)
        .expect_err("unknown method must fail closed");
    assert_eq!(
        error,
        SliceError::UnsupportedHostMethod {
            method: "panel.context".to_owned(),
        }
    );
}

#[test]
fn missing_consent_fails_closed() {
    let mut bridge = bridge(ScopeSet::single(Scope::TerminalInspect), false);
    let mut host = LoopbackHost::serving(snapshot());
    let error = bridge
        .call("terminal.snapshot", b"{}", NOW_MS, &mut host)
        .expect_err("missing consent must fail closed");
    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );
}

#[test]
fn missing_scope_fails_closed() {
    let mut bridge = bridge(ScopeSet::new(), true);
    let mut host = LoopbackHost::serving(snapshot());
    let error = bridge
        .call("terminal.snapshot", b"{}", NOW_MS, &mut host)
        .expect_err("missing scope must fail closed");
    assert!(matches!(error, SliceError::Ipc(_)), "got {error:?}");
}

#[test]
fn live_consent_never_widens_the_server_evaluated_scope_set() {
    // AIQ-22/42 negative evidence: the reader is granted only
    // `terminal.inspect` server-side while its ledger carries a live grant
    // for `process.spawn`. A live grant outside the granted set must not
    // authorize the method, and the refusal must reach no host peer.
    let mut bridge = IpcBridge::new("reader-inspect", ScopeSet::single(Scope::TerminalInspect));
    bridge
        .grant_consent(Scope::ProcessSpawn, NOW_MS, TTL_MS)
        .expect("consent grant");
    assert!(bridge.consent_active(Scope::ProcessSpawn, NOW_MS));
    let mut peer = CountingHost::new();

    let error = bridge
        .call("process.spawn", br#"{}"#, NOW_MS, &mut peer)
        .expect_err("consent must not widen the granted scope set");

    match error {
        SliceError::Ipc(IpcError::ScopeDenied { .. }) => {}
        other => panic!("expected ScopeDenied, got {other:?}"),
    }
    assert_eq!(peer.served, 0, "refused call never reached the host peer");
    assert_eq!(bridge.pending_count(), 0, "no request enqueued");
}

#[test]
fn consent_grant_is_bound_to_the_reader_that_holds_it() {
    // AIQ-58 negative evidence: two readers hold the same server-evaluated
    // scope set, but only reader A holds a grant. Reader B must not inherit
    // reader A's authorization, and only A reaches the host.
    let mut reader_a = IpcBridge::new("reader-a", ScopeSet::single(Scope::TerminalInspect));
    reader_a
        .grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("grant");
    let mut reader_b = IpcBridge::new("reader-b", ScopeSet::single(Scope::TerminalInspect));
    let mut peer = CountingHost::new();

    reader_a
        .call("terminal.snapshot", b"{}", NOW_MS, &mut peer)
        .expect("grant holder is served");
    let error = reader_b
        .call("terminal.snapshot", b"{}", NOW_MS, &mut peer)
        .expect_err("second reader must not inherit the first reader's grant");

    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );
    assert!(reader_a.consent_active(Scope::TerminalInspect, NOW_MS));
    assert!(!reader_b.consent_active(Scope::TerminalInspect, NOW_MS));
    assert_eq!(peer.served, 1, "only the grant holder reached the host");
    assert_eq!(reader_b.pending_count(), 0, "no request enqueued");
}

#[test]
fn per_reader_consent_expiry_does_not_authorize_another_reader() {
    // AIQ-58 negative evidence: each reader's grant expires on its own
    // schedule. At `later` reader A's grant is expired while reader B's is
    // live; only B is served and A enqueues nothing.
    let mut reader_a = IpcBridge::new("reader-a", ScopeSet::single(Scope::TerminalInspect));
    reader_a
        .grant_consent(Scope::TerminalInspect, NOW_MS, 100)
        .expect("grant");
    let mut reader_b = IpcBridge::new("reader-b", ScopeSet::single(Scope::TerminalInspect));
    reader_b
        .grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("grant");
    let later = NOW_MS + 100;
    assert!(!reader_a.consent_active(Scope::TerminalInspect, later));
    assert!(reader_b.consent_active(Scope::TerminalInspect, later));
    let mut peer = CountingHost::new();

    let error = reader_a
        .call("terminal.snapshot", b"{}", later, &mut peer)
        .expect_err("expired reader grant must not serve");
    assert_eq!(
        error,
        SliceError::ConsentRequired {
            scope: "terminal.inspect",
        }
    );
    reader_b
        .call("terminal.snapshot", b"{}", later, &mut peer)
        .expect("live reader grant serves");

    assert_eq!(peer.served, 1, "only the live grant reached the host");
    assert_eq!(reader_a.pending_count(), 0, "no request enqueued");
}

#[test]
fn host_without_snapshot_handler_fails_closed() {
    let mut bridge = bridge(ScopeSet::single(Scope::TerminalInspect), true);
    let mut host = LoopbackHost::delegated();
    let error = collect_terminal_context(&mut bridge, &mut host, &request(4096), 1, NOW_MS)
        .expect_err("missing host handler must fail closed");
    assert!(
        matches!(error, SliceError::ContextUnavailable { .. }),
        "got {error:?}"
    );
}

fn auth_base() -> AuthBase {
    let mut ids = IdIssuer::default();
    AuthBase {
        agent_instance_id: ids.agent_instance(),
        session_id: ids.session(),
        level: AgentLevel::Inspect,
    }
}

#[test]
fn unknown_tool_fails_closed_before_dispatch() {
    let bus = ToolBus::new(test_tool_registry().expect("registry")).with_authorizer(AllowReadOnly);
    let call = ToolCall {
        name: "terminal_destroy".to_owned(),
        arguments: br#"{}"#.to_vec(),
    };
    let error = bus
        .precheck(std::slice::from_ref(&call), &auth_base(), BUS_CALL_CAP)
        .expect_err("unknown tool must fail closed");
    assert_eq!(
        error,
        ToolError::UnknownTool {
            name: "terminal_destroy".to_owned(),
        }
    );
    assert_eq!(bus.calls_this_turn(), 0);
}

#[test]
fn write_tool_is_denied_by_default() {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolSpec::new(
                "workspace_write",
                "write files",
                br#"{"type":"object"}"#.to_vec(),
                "workspace.write",
                false,
            )
            .expect("spec"),
        )
        .expect("capacity");
    let bus = ToolBus::new(registry).with_authorizer(AllowReadOnly);
    let call = ToolCall {
        name: "workspace_write".to_owned(),
        arguments: br#"{}"#.to_vec(),
    };
    let error = bus
        .precheck(std::slice::from_ref(&call), &auth_base(), BUS_CALL_CAP)
        .expect_err("write tool must be denied by default");
    assert!(matches!(error, ToolError::Denied { .. }), "got {error:?}");
}

#[test]
fn tool_call_limit_is_enforced() {
    let mut bus =
        ToolBus::new(test_tool_registry().expect("registry")).with_authorizer(AllowReadOnly);
    let mut executor = FakeToolExecutor::new();
    for _ in 0..8 {
        executor.push_success("ok", b"data".to_vec());
    }
    let mut ids = IdIssuer::default();
    let base = auth_base();
    let call = ToolCall {
        name: HARNESS_TOOL.to_owned(),
        arguments: br#"{}"#.to_vec(),
    };
    for _ in 0..8 {
        bus.dispatch(&mut executor, &call, &base, ids.execution(), NOW_MS)
            .expect("within limit");
    }
    // AI-RUN-004: the ninth call is a pre-dispatch admission refusal carrying
    // the typed cap cause, not an executed failure; the host is never
    // contacted for it.
    let execution = bus
        .dispatch(&mut executor, &call, &base, ids.execution(), NOW_MS)
        .expect("admission refusal is a recorded status, not a bus error");
    assert!(matches!(
        execution.status,
        ToolStatus::Refused {
            cause: ToolError::CallLimitExceeded { limit: 8 }
        }
    ));
}

#[test]
fn oversized_stream_chunk_fails_closed() {
    // Exercises the runtime `MAX_FRAGMENT_BYTES` (64 KiB) gate: 70 KiB is
    // rejected before reaching the sink.
    let mut sink = VecSink::new();
    let bytes = vec![b'x'; 70 * 1024];
    let error = sink
        .emit(StreamChunk {
            seq: 0,
            total: 1,
            is_final: true,
            fragment: Fragment::markdown(bytes),
        })
        .expect_err("oversized fragment must fail closed");
    assert!(
        matches!(error, StreamError::OversizedFragment { .. }),
        "got {error:?}"
    );
    assert!(sink.is_empty());
}

#[test]
fn chunk_over_rc10_ceiling_fails_closed() {
    // The runtime is std-only, so the RC-10 ceiling is mirrored as
    // `MAX_STREAM_CHUNK_BYTES`. One fragment per chunk (`RS-3`) keeps the
    // reachable bound at `MAX_FRAGMENT_BYTES` (64 KiB), so a chunk over the
    // 256 KiB ceiling surfaces as `OversizedFragment` (P2-1: the unreachable
    // chunk-layer variant was removed), not as a `bitty-ipc` wire error. The
    // wire gate itself stays covered by the bridge path (`validate_chunk` on
    // send).
    let mut sink = VecSink::new();
    let bytes = vec![b'x'; bitty_ipc::wire::CHUNK_CEILING + 1];
    let actual = bytes.len();
    let error = sink
        .emit(StreamChunk {
            seq: 0,
            total: 1,
            is_final: true,
            fragment: Fragment::markdown(bytes),
        })
        .expect_err("a chunk over the RC-10 ceiling must fail closed");
    assert_eq!(
        error,
        StreamError::OversizedFragment {
            limit: 64 * 1024,
            actual,
        }
    );
    assert!(sink.is_empty());
}

#[test]
fn context_budget_exceeded_fails_closed() {
    let mut provider = scripted_provider(ANSWER, Some(tool_args())).expect("provider");
    provider.push_turn(ProviderTurn {
        text: ANSWER.to_owned(),
        tool_calls: vec![ToolCallRequest {
            name: HARNESS_TOOL.to_owned(),
            arguments: tool_args(),
        }],
        latency_ms: 0,
        usage: ProviderUsage::default(),
    });
    let mut agent = harness_agent(provider, 4).expect("agent");
    let seed = terminal_record("term-1", 1, NOW_MS, snapshot()).expect("record");
    let mut executor = FakeToolExecutor::new();
    let mut sink = VecSink::new();
    let outcome = agent.run_turn(
        &mut executor,
        HARNESS_MODEL,
        "what did my last command print?",
        &[seed],
        &mut sink,
        NOW_MS,
    );
    match outcome {
        ExecOutcome::Failed { error } => assert!(
            matches!(
                error,
                AgentError::Context(ContextError::BudgetExceeded { limit: 4, .. })
            ),
            "got {error:?}"
        ),
        other => panic!("over-budget context must fail closed, got {other:?}"),
    }
    // Budget fails before provider I/O: the script is unconsumed.
    assert_eq!(agent.provider_mut().scripted_turns_remaining(), 2);
    // The denied-by-status tool path stays typed: no dispatch happened.
    assert!(agent.executions().is_empty());
    assert!(sink.is_empty());
}

//! Deterministic end-to-end and fail-closed proof for the vertical slice.
//!
//! No network, no secrets, no wall-clock: the host peer is a bounded loopback
//! implementation and every timestamp is caller-supplied. The loopback peer is
//! a test double for the Bitty host, not a claim about live data; the product
//! path fails closed when the real host does not implement a method.

use bitty_ai_slice::{
    ContextRequest, DeterministicLocalProvider, FragmentKind, HostPeer, HostToolBus, IpcBridge,
    Message, PanelStreamSink, Role, SemanticZone, SliceError, StreamChunk, StreamSink, ToolBus,
    ToolDecl, ToolHost, ToolInvocation, VerticalSlice,
};
use bitty_ipc::channel::{IpcRequest, IpcResponse};
use bitty_ipc::scope::{Scope, ScopeSet};

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;

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

impl ToolHost for LoopbackHost {
    fn call_tool(
        &mut self,
        tool: &str,
        _arguments: &[u8],
        _now_ms: u64,
    ) -> Result<Vec<u8>, SliceError> {
        if tool == "terminal.read_zone" {
            Ok(self.snapshot.clone())
        } else {
            Err(SliceError::ToolDenied {
                name: tool.to_owned(),
            })
        }
    }
}

fn snapshot() -> Vec<u8> {
    b"$ echo hello\nhello\n".to_vec()
}

fn request(max_bytes: usize) -> ContextRequest {
    ContextRequest {
        instance_id: "inst-1".to_owned(),
        terminal_id: "term-1".to_owned(),
        zone: SemanticZone::Output,
        max_bytes,
    }
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

fn messages() -> Vec<Message> {
    vec![Message::new(Role::User, "what did my last command print?")]
}

#[test]
fn end_to_end_loop_is_deterministic() {
    let run = || {
        let mut slice = VerticalSlice::new(
            DeterministicLocalProvider::default_slice(),
            bridge(ScopeSet::single(Scope::TerminalInspect), true),
            HostToolBus::new(HostToolBus::read_only_registry()).expect("tool bus"),
            PanelStreamSink::new(),
        );
        let mut ipc_host = LoopbackHost::serving(snapshot());
        let mut tool_host = LoopbackHost::serving(snapshot());
        slice
            .run_turn(
                &mut ipc_host,
                &mut tool_host,
                &messages(),
                &request(4096),
                NOW_MS,
            )
            .expect("turn succeeds")
    };

    let first = run();
    let second = run();
    assert_eq!(first, second, "the slice must be deterministic");

    assert_eq!(first.provider_id, "local.deterministic");
    assert!(first.answer.contains("hello"));

    let context = first.context.as_ref().expect("context collected");
    assert_eq!(context.provider, "terminal");
    assert_eq!(context.owner, "inst-1/term-1");
    assert_eq!(context.generation, 1);
    assert_eq!(context.collected_at_ms, NOW_MS);
    assert!(context.is_untrusted_surface);
    assert_eq!(context.bytes, snapshot());

    let tool = first.tool.as_ref().expect("tool dispatched");
    assert_eq!(tool.name, "terminal.read_zone");
    assert!(tool.is_untrusted_surface);
    assert!(tool.result.len() <= 16 * 1024);

    assert_eq!(first.chunks.len(), 2);
    assert_eq!(first.chunks[0].fragment.kind, FragmentKind::Markdown);
    assert_eq!(first.chunks[1].fragment.kind, FragmentKind::ToolCard);
    for (index, chunk) in first.chunks.iter().enumerate() {
        assert_eq!(chunk.seq, index as u32);
        assert_eq!(chunk.total, first.chunks.len() as u32);
        assert_eq!(chunk.is_final, index + 1 == first.chunks.len());
    }
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
fn host_without_snapshot_handler_fails_closed() {
    let mut slice = VerticalSlice::new(
        DeterministicLocalProvider::default_slice(),
        bridge(ScopeSet::single(Scope::TerminalInspect), true),
        HostToolBus::new(HostToolBus::read_only_registry()).expect("tool bus"),
        PanelStreamSink::new(),
    );
    let mut ipc_host = LoopbackHost::delegated();
    let mut tool_host = LoopbackHost::serving(snapshot());
    let error = slice
        .run_turn(
            &mut ipc_host,
            &mut tool_host,
            &messages(),
            &request(4096),
            NOW_MS,
        )
        .expect_err("missing host handler must fail closed");
    assert!(
        matches!(error, SliceError::ContextUnavailable { .. }),
        "got {error:?}"
    );
}

#[test]
fn unknown_tool_fails_closed_before_dispatch() {
    let mut bus = HostToolBus::new(HostToolBus::read_only_registry()).expect("tool bus");
    let mut host = LoopbackHost::serving(snapshot());
    let error = bus
        .dispatch(
            &mut host,
            &ToolInvocation {
                name: "terminal.destroy".to_owned(),
                arguments: "{}".to_owned(),
            },
            NOW_MS,
        )
        .expect_err("unknown tool must fail closed");
    assert_eq!(
        error,
        SliceError::ToolNotRegistered {
            name: "terminal.destroy".to_owned(),
        }
    );
    assert_eq!(bus.calls_this_turn(), 0);
}

#[test]
fn write_tool_is_denied_by_default() {
    let mut bus = HostToolBus::new(vec![ToolDecl::new("workspace.write", "write files", false)])
        .expect("tool bus");
    let mut host = LoopbackHost::serving(snapshot());
    let error = bus
        .dispatch(
            &mut host,
            &ToolInvocation {
                name: "workspace.write".to_owned(),
                arguments: "{}".to_owned(),
            },
            NOW_MS,
        )
        .expect_err("write tool must be denied by default");
    assert_eq!(
        error,
        SliceError::ToolDenied {
            name: "workspace.write".to_owned(),
        }
    );
}

#[test]
fn tool_call_limit_is_enforced() {
    let mut bus = HostToolBus::new(HostToolBus::read_only_registry()).expect("tool bus");
    let mut host = LoopbackHost::serving(snapshot());
    let call = ToolInvocation {
        name: "terminal.read_zone".to_owned(),
        arguments: "{}".to_owned(),
    };
    for _ in 0..8 {
        bus.dispatch(&mut host, &call, NOW_MS)
            .expect("within limit");
    }
    let error = bus
        .dispatch(&mut host, &call, NOW_MS)
        .expect_err("ninth call must fail");
    assert_eq!(error, SliceError::ToolCallLimitExceeded { limit: 8 });
}

#[test]
fn oversized_stream_chunk_fails_closed() {
    // Exercises the slice-local `MAX_FRAGMENT_BYTES` (64 KiB) gate, not
    // RC-10's larger 256 KiB `CHUNK_CEILING`: 70 KiB passes `validate_chunk`
    // and is rejected by the slice-local fragment bound.
    // `chunk_over_rc10_ceiling_fails_closed` below covers the RC-10 gate.
    let mut sink = PanelStreamSink::new();
    let bytes = vec![b'x'; 70 * 1024];
    let error = sink
        .emit(StreamChunk {
            seq: 0,
            total: 1,
            is_final: true,
            fragment: bitty_ai_slice::Fragment {
                kind: FragmentKind::Markdown,
                bytes,
            },
        })
        .expect_err("oversized fragment must fail closed");
    assert!(
        matches!(error, SliceError::StreamViolation { .. }),
        "got {error:?}"
    );
    assert!(sink.chunks().is_empty());
}

#[test]
fn chunk_over_rc10_ceiling_fails_closed() {
    let mut sink = PanelStreamSink::new();
    let bytes = vec![b'x'; bitty_ipc::wire::CHUNK_CEILING + 1];
    let error = sink
        .emit(StreamChunk {
            seq: 0,
            total: 1,
            is_final: true,
            fragment: bitty_ai_slice::Fragment {
                kind: FragmentKind::Markdown,
                bytes: bytes.clone(),
            },
        })
        .expect_err("a chunk over the RC-10 ceiling must fail closed");
    assert_eq!(
        error,
        SliceError::Ipc(bitty_ipc::error::IpcError::PayloadTooLarge {
            field: "chunk.bytes".to_owned(),
            limit: bitty_ipc::wire::CHUNK_CEILING,
            actual: bytes.len(),
        })
    );
    assert!(sink.chunks().is_empty());
}

#[test]
fn context_budget_exceeded_fails_closed() {
    let mut slice = VerticalSlice::new(
        DeterministicLocalProvider::default_slice(),
        bridge(ScopeSet::single(Scope::TerminalInspect), true),
        HostToolBus::new(HostToolBus::read_only_registry()).expect("tool bus"),
        PanelStreamSink::new(),
    );
    let mut ipc_host = LoopbackHost::serving(snapshot());
    let mut tool_host = LoopbackHost::serving(snapshot());
    let error = slice
        .run_turn(
            &mut ipc_host,
            &mut tool_host,
            &messages(),
            &request(4),
            NOW_MS,
        )
        .expect_err("over-budget context must fail closed");
    assert_eq!(
        error,
        SliceError::ContextBudgetExceeded {
            limit: 4,
            actual: snapshot().len(),
        }
    );
}

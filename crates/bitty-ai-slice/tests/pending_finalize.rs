//! Pending-correlation finalization on every terminal exit (`AI-CTX-007`).
//!
//! The bridge tracks in-flight correlation state in `IpcEndpoint::pending`,
//! which is separate from the request/response queues: dequeue alone does not
//! remove pending state, and the slice exposes no expiry/drain recovery. Every
//! terminal exit after admission must therefore finalize the pending entry.
//!
//! Acceptance (`AI-CTX-007`): an inert peer transport error returns the
//! original typed error, leaves `pending_count()` unchanged, and a later
//! successful call remains possible; the mismatched-ID and correlated-refusal
//! paths keep their existing behavior.
//!
//! Deterministic: caller-supplied `now_ms`, inert scripted peers, no wall
//! clock, thread, sleep, server, network, filesystem, or secret. Assert
//! messages are static; no runtime value is formatted into a panic message.

use bitty_ai_slice::{HostPeer, IpcBridge, SliceError};
use bitty_ipc::channel::{IpcRequest, IpcResponse, RequestId};
use bitty_ipc::error::IpcError;
use bitty_ipc::scope::{Scope, ScopeSet};

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;
const METHOD: &str = "terminal.snapshot";

fn inspecting_bridge() -> IpcBridge {
    let mut bridge = IpcBridge::new(
        "agent-pending-finalize",
        ScopeSet::single(Scope::TerminalInspect),
    );
    bridge
        .grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
        .expect("consent grant");
    bridge
}

/// Inert peer whose first call fails with a typed transport fault and whose
/// later calls answer correctly, so "later service remains possible" is
/// observable through the same bridge.
struct RecoveringTransportPeer {
    fail_next: bool,
}

impl HostPeer for RecoveringTransportPeer {
    fn serve(&mut self, request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        if self.fail_next {
            self.fail_next = false;
            return Err(SliceError::Ipc(IpcError::Transport {
                reason: "inert peer transport fault".to_owned(),
            }));
        }
        Ok(IpcResponse::success(request.id, b"{}".to_vec())?)
    }
}

/// Inert peer that always fails with a typed transport fault.
struct FaultingPeer;

impl HostPeer for FaultingPeer {
    fn serve(&mut self, _request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        Err(SliceError::Ipc(IpcError::Transport {
            reason: "inert peer transport fault".to_owned(),
        }))
    }
}

/// Inert peer that answers with a fixed uncorrelated id.
struct MismatchedIdPeer;

impl HostPeer for MismatchedIdPeer {
    fn serve(&mut self, _request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        Ok(IpcResponse::success(RequestId(900_000), b"{}".to_vec())?)
    }
}

/// Inert peer that answers with a correlated application-level error.
struct RefusingPeer;

impl HostPeer for RefusingPeer {
    fn serve(&mut self, request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        Ok(IpcResponse::error(
            request.id,
            b"inert application refusal".to_vec(),
        )?)
    }
}

#[test]
fn transport_peer_error_finalizes_pending_and_allows_later_service() {
    let mut bridge = inspecting_bridge();
    let mut peer = RecoveringTransportPeer { fail_next: true };
    assert_eq!(
        bridge.pending_count(),
        0,
        "bridge starts with no pending entry"
    );

    let error = bridge
        .call(METHOD, b"{}", NOW_MS, &mut peer)
        .expect_err("an inert transport fault must surface");
    assert_eq!(
        error,
        SliceError::Ipc(IpcError::Transport {
            reason: "inert peer transport fault".to_owned(),
        }),
        "the original typed peer error must be preserved untouched"
    );
    assert_eq!(
        bridge.pending_count(),
        0,
        "a transport peer error must finalize, not leak, the pending entry"
    );

    assert_eq!(
        bridge
            .call(METHOD, b"{}", NOW_MS, &mut peer)
            .expect("a later successful call must remain possible"),
        b"{}"
    );
    assert_eq!(
        bridge.pending_count(),
        0,
        "a served call leaves no pending entry"
    );
}

#[test]
fn repeated_transport_peer_errors_do_not_accumulate_pending() {
    let mut bridge = inspecting_bridge();
    let mut peer = FaultingPeer;

    for _ in 0..8 {
        bridge
            .call(METHOD, b"{}", NOW_MS, &mut peer)
            .expect_err("every inert transport fault must surface");
        assert_eq!(
            bridge.pending_count(),
            0,
            "repeated peer faults must not accumulate pending entries"
        );
    }
}

#[test]
fn mismatched_response_id_still_leaves_no_pending_entry() {
    let mut bridge = inspecting_bridge();
    let mut peer = MismatchedIdPeer;

    let error = bridge
        .call(METHOD, b"{}", NOW_MS, &mut peer)
        .expect_err("an uncorrelated answer must be refused");
    match error {
        SliceError::ContextUnavailable { .. } => {}
        _ => panic!("an uncorrelated answer must fail as ContextUnavailable"),
    }
    assert_eq!(
        bridge.pending_count(),
        0,
        "an uncorrelated answer must not pin pending capacity"
    );
}

#[test]
fn correlated_error_response_completes_pending_before_refusal() {
    let mut bridge = inspecting_bridge();
    let mut peer = RefusingPeer;

    let error = bridge
        .call(METHOD, b"{}", NOW_MS, &mut peer)
        .expect_err("a correlated refusal must surface");
    match error {
        SliceError::ContextUnavailable { reason } => {
            assert!(
                reason.contains("inert application refusal"),
                "the application refusal reason must be preserved"
            );
        }
        _ => panic!("a correlated refusal must fail as ContextUnavailable"),
    }
    assert_eq!(
        bridge.pending_count(),
        0,
        "a correlated refusal must complete its pending entry first"
    );
}

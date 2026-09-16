//! Pinned `bitty-ipc` surface contract (`AI-0081`: rev `be6e63c` -> `cfeffa2`).
//!
//! The slice links `bitty-ipc` by Git revision and mirrors the parts it
//! consumes, so upstream changes do not follow the pin automatically. The
//! `be6e63c..cfeffa2` window changed:
//!
//! - `wire.rs` / `lib.rs` (CTX-0484): `SUPPORTED_WIRE_VERSIONS` plus
//!   `negotiate_wire_version`, and set-based `validate_wire_version` that
//!   stays fail-closed with `VersionMismatch` outside the supported set;
//! - `channel.rs` (CTX-0483): `IpcEndpoint::drain_expired` also purges
//!   not-yet-handed-over queued requests, so an expired request can never be
//!   executed after its deadline;
//! - `bridge.rs` (CTX-0483): `BridgeClient::answer` refuses unknown ids before
//!   enqueueing, so uncorrelated answers buffer nothing;
//! - `limits.rs` (CTX-0483): `RateLimiter` refills a token bucket at
//!   `limit_per_sec` instead of re-arming the full burst every window.
//!
//! The slice consumes the wire envelopes and the `IpcEndpoint` correlation
//! path directly (see `src/bridge.rs`), so the tests below pin the upstream
//! contract that mirror relies on. `BridgeClient` and `RateLimiter` have no
//! slice call site, so they are re-verified by review of the window diff
//! rather than mirrored as tests.
//!
//! Deterministic: caller-supplied `now_ms` only, no wall clock, thread,
//! network, filesystem, or secret.

use bitty_ai_slice::{HostPeer, IpcBridge, SliceError};
use bitty_ipc::channel::{IpcEndpoint, IpcRequest, IpcResponse, RequestId};
use bitty_ipc::error::IpcError;
use bitty_ipc::scope::{Scope, ScopeSet};
use bitty_ipc::wire::{
    SUPPORTED_WIRE_VERSIONS, WIRE_VERSION, negotiate_wire_version, validate_request_envelope,
    validate_response_envelope, validate_wire_version,
};

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;
const METHOD: &str = "terminal.snapshot";

fn inspecting_bridge(consent: bool) -> IpcBridge {
    let mut bridge = IpcBridge::new(
        "agent-pinned-surface",
        ScopeSet::single(Scope::TerminalInspect),
    );
    if consent {
        bridge
            .grant_consent(Scope::TerminalInspect, NOW_MS, TTL_MS)
            .expect("consent grant");
    }
    bridge
}

/// Peer that answers with the request id unless a fixed `id` is supplied,
/// which emulates a hostile peer delivering an uncorrelated answer.
struct AnswerPeer {
    id: Option<RequestId>,
}

impl HostPeer for AnswerPeer {
    fn serve(&mut self, request: &IpcRequest) -> Result<IpcResponse, SliceError> {
        let id = self.id.unwrap_or(request.id);
        Ok(IpcResponse::success(id, b"{}".to_vec())?)
    }
}

// ── wire version negotiation (CTX-0484) ─────────────────────────────────────

#[test]
fn wire_contract_is_set_based_and_fails_closed() {
    assert_eq!(
        WIRE_VERSION, 1,
        "pinned contract version moved; re-verify the slice envelope usage"
    );
    assert_eq!(
        SUPPORTED_WIRE_VERSIONS,
        [WIRE_VERSION].as_slice(),
        "the advertised set must be the current contract version at this pin"
    );
    assert!(validate_wire_version(WIRE_VERSION).is_ok());
    for unsupported in [0, WIRE_VERSION + 1, u16::MAX] {
        let err =
            validate_wire_version(unsupported).expect_err("unsupported version must fail closed");
        assert!(
            matches!(
                err,
                IpcError::VersionMismatch { expected, actual }
                    if expected == WIRE_VERSION && actual == unsupported
            ),
            "expected VersionMismatch({WIRE_VERSION}, {unsupported}), got {err:?}"
        );
    }
}

#[test]
fn negotiation_selects_highest_mutual_version_and_fails_closed_without_overlap() {
    assert_eq!(
        negotiate_wire_version(&[WIRE_VERSION]).expect("mutual v1"),
        WIRE_VERSION
    );
    assert_eq!(
        negotiate_wire_version(&[WIRE_VERSION, WIRE_VERSION + 1]).expect("overlap on v1"),
        WIRE_VERSION,
        "selection is the highest mutual version, not the highest offered"
    );
    assert_eq!(
        negotiate_wire_version(&[WIRE_VERSION + 2, WIRE_VERSION]).expect("ordering irrelevant"),
        WIRE_VERSION
    );

    let err = negotiate_wire_version(&[]).expect_err("an empty offer must fail closed");
    assert!(
        matches!(
            err,
            IpcError::VersionMismatch { expected, actual }
                if expected == WIRE_VERSION && actual == 0
        ),
        "empty offer reports actual 0, got {err:?}"
    );

    let err = negotiate_wire_version(&[WIRE_VERSION + 1, WIRE_VERSION + 2])
        .expect_err("no overlap must fail closed");
    assert!(
        matches!(
            err,
            IpcError::VersionMismatch { expected, actual }
                if expected == WIRE_VERSION && actual == WIRE_VERSION + 2
        ),
        "actual is the highest offered version, got {err:?}"
    );
}

#[test]
fn slice_envelopes_validate_only_at_a_supported_version() {
    let negotiated = negotiate_wire_version(&[WIRE_VERSION]).expect("mutual v1");
    validate_request_envelope(negotiated, "1", METHOD, b"{}")
        .expect("slice request envelope valid at the negotiated version");
    validate_response_envelope(negotiated, "1", b"{}")
        .expect("slice response envelope valid at the negotiated version");

    assert!(
        validate_request_envelope(WIRE_VERSION + 1, "1", METHOD, b"{}").is_err(),
        "unsupported version must fail closed before method/params checks"
    );
    assert!(
        validate_response_envelope(WIRE_VERSION + 1, "1", b"{}").is_err(),
        "unsupported version must fail closed"
    );
}

// ── timeout reaping (CTX-0483) ──────────────────────────────────────────────

#[test]
fn expired_queued_request_is_never_handed_over() {
    let mut endpoint = IpcEndpoint::with_capacity(8, 8);
    let expired = endpoint
        .create_request(METHOD.to_owned(), b"{}".to_vec(), NOW_MS, 500)
        .expect("request fits");
    assert_eq!(endpoint.request_len(), 1);
    assert_eq!(endpoint.pending_count(), 1);

    let drained = endpoint.drain_expired(NOW_MS + 500);
    assert_eq!(
        drained,
        vec![expired],
        "the deadline is inclusive at now_ms"
    );
    assert_eq!(endpoint.pending_count(), 0);
    assert_eq!(
        endpoint.request_len(),
        0,
        "an expired request must not remain queued for execution"
    );
    assert!(
        endpoint.recv_request().is_none(),
        "no exec-after-timeout from the queue"
    );
}

#[test]
fn timeout_reaping_keeps_live_queued_requests() {
    let mut endpoint = IpcEndpoint::with_capacity(8, 8);
    let _expired = endpoint
        .create_request(METHOD.to_owned(), b"{}".to_vec(), NOW_MS, 500)
        .expect("request fits");
    let live = endpoint
        .create_request(METHOD.to_owned(), b"{}".to_vec(), NOW_MS, 5_000)
        .expect("request fits");

    let drained = endpoint.drain_expired(NOW_MS + 500);
    assert_eq!(drained.len(), 1, "only the expired id is reaped");
    assert_eq!(endpoint.pending_count(), 1);
    assert_eq!(endpoint.request_len(), 1);
    assert_eq!(
        endpoint.recv_request().expect("live request survives").id,
        live
    );
}

// ── uncorrelated answers buffer nothing (CTX-0483) ──────────────────────────

#[test]
fn uncorrelated_answer_is_refused_and_pins_no_capacity() {
    let mut bridge = inspecting_bridge(true);
    let mut mismatched = AnswerPeer {
        id: Some(RequestId(900_000)),
    };
    let err = bridge
        .call(METHOD, b"{}", NOW_MS, &mut mismatched)
        .expect_err("an uncorrelated answer must be refused");
    match err {
        SliceError::ContextUnavailable { reason } => {
            assert!(reason.contains("does not correlate"), "got {reason}");
        }
        other => panic!("expected ContextUnavailable, got {other:?}"),
    }
    assert_eq!(
        bridge.pending_count(),
        0,
        "an uncorrelated answer must not pin pending capacity"
    );

    // The refusal is total: the same bridge still serves a correlated answer.
    let mut echo = AnswerPeer { id: None };
    assert_eq!(
        bridge
            .call(METHOD, b"{}", NOW_MS, &mut echo)
            .expect("correlated answer is served"),
        b"{}"
    );
    assert_eq!(bridge.pending_count(), 0);
}

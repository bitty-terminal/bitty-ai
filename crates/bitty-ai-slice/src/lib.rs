//! Integration harness: the real `bitty-ai-runtime` driven through the real
//! generic `bitty-ipc` primitives.
//!
//! This crate is a **pressure test**, not a shipped component. It exists to
//! answer one architecture question (CTX-0407, 017 recommendation 6, and the
//! `BA-6` pressure-test gate in the AI architecture specification): can an
//! end-to-end AI turn be expressed using the real AI runtime over only the
//! generic primitives that Core already exposes, or does it demand an
//! AI-specific Core API?
//!
//! Since AI-0012 this crate owns **no AI mechanism**: the former second
//! runtime (`provider`, `context`, `session`, `stream`, `toolbus` modules)
//! was deleted once `bitty-ai-runtime` subsumed it. What remains is the host
//! boundary ([`IpcBridge`]), typed bridge failures ([`SliceError`]), and the
//! [`harness`] fixtures that adapt real IPC snapshots to real runtime inputs.
//! `tests/vertical_slice.rs` proves the full loop (prompt -> provider ->
//! context -> tool -> streamed result) and the fail-closed paths against the
//! real runtime.
//!
//! The harness reuses the real, externally consumable generic IPC primitive
//! `bitty-ipc` (bounded wire envelope, method/scope authorization, per-client
//! consent ledger, RC-10 chunk validation, bounded channels, and the headless
//! MCP client stub) through a pinned Git revision. It never links Core's
//! in-process agent or runtime. Terminal content is requested through the
//! generic `terminal.snapshot` wire method under the `terminal.inspect` scope;
//! when the host does not implement a method or a capability is absent, the
//! harness fails closed and the gap is recorded rather than worked around with
//! a new Core API.
//!
//! The harness is deterministic: every operation takes a caller-supplied
//! `now_ms`, there is no wall-clock, thread, async runtime, network, or secret,
//! and the model provider is the runtime's scripted `FakeProvider` — except the
//! `local_provider` experiment, which opens loopback-only TCP with mandatory
//! timeouts and a caller-supplied key, and the [`journal_prototype`]
//! experiment (AI-0049), which persists an append-ordered single-writer
//! journal through SQLite in a caller-supplied file.
//!
//! [`fragment_transport`] owns the runtime-to-transport pre-split rule
//! (AI-0066) that reconciles the runtime 64 KiB fragment bound with the
//! 16 KiB `bitty-ipc` ingest ceiling, so no fragment truncates across the
//! boundary. It is the mapping layer, not a shipped transport: the runtime is
//! std-only and cannot name the transport ceiling, and `bitty-ipc` is upstream
//! and unchanged.
//!
//! [`snapshot_ingest`] (AI-0121) is the host-side ProjectSnapshot v1
//! ingestion experiment: pure bytes-in/record-out verification of `psnap`
//! canonical bytes plus an explicit per-invocation refresh authorization,
//! adapted to a runtime `ContextRecord` (provider `"project"`,
//! untrusted-surface). The host spawns `psnap` out-of-process; this crate
//! never does.

#![deny(unsafe_code)]

pub mod bridge;
pub mod error;
pub mod fake_host;
pub mod fragment_transport;
pub mod harness;
pub mod journal_prototype;
pub mod live_host;
pub mod local_provider;
pub mod snapshot_ingest;

pub use bridge::{HostPeer, IpcBridge};
pub use error::SliceError;
pub use fake_host::{BittyHost, FakeHost};
pub use harness::{
    AllowReadOnly, HARNESS_MODEL, HARNESS_PROVIDER_ID, HARNESS_TOOL, SnapshotRequest,
    collect_terminal_context, harness_agent, scripted_provider, terminal_record, test_session,
    test_tool_registry,
};
pub use live_host::LiveBittyHost;
pub use local_provider::{LocalEndpoint, LocalProvider, MonotonicClock, SystemMonotonicClock};
pub use snapshot_ingest::{
    RefreshAuthorization, SNAPSHOT_CANONICALIZATION_VERSION, SNAPSHOT_DIGEST_PREFIX_LEN,
    SNAPSHOT_PROVIDER, SNAPSHOT_SCHEMA_VERSION, SnapshotIngestError, SnapshotIngestRequest,
    ingest_snapshot, snapshot_digest_hex,
};

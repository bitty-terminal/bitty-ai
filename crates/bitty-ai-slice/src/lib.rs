//! Experimental `bitty-ai` vertical slice built only on generic Bitty
//! primitives.
//!
//! This crate is a **pressure test**, not a shipped component. It exists to
//! answer one architecture question (CTX-0407, 017 recommendation 6, and the
//! `BA-6` pressure-test gate in the AI architecture specification): can an
//! end-to-end AI turn be expressed using only the generic primitives that Core
//! already exposes, or does it demand an AI-specific Core API?
//!
//! The slice reuses the real, externally consumable generic IPC primitive
//! `bitty-ipc` (bounded wire envelope, method/scope authorization, per-client
//! consent ledger, RC-10 chunk validation, bounded channels, and the headless
//! MCP client stub) through a pinned Git revision. It never links Core's
//! in-process agent or runtime. Terminal content is requested through the
//! generic `terminal.snapshot` wire method under the `terminal.inspect` scope;
//! when the host does not implement a method or a capability is absent, the
//! slice fails closed and the gap is recorded rather than worked around with a
//! new Core API.
//!
//! The slice is deterministic: every operation takes a caller-supplied
//! `now_ms`, there is no wall-clock, thread, async runtime, network, or secret,
//! and the local model provider is scripted. `tests/vertical_slice.rs` proves
//! the full loop (prompt -> provider -> context -> tool -> streamed result) and
//! the fail-closed paths.

#![deny(unsafe_code)]

pub mod bridge;
pub mod context;
pub mod error;
pub mod provider;
pub mod session;
pub mod stream;
pub mod toolbus;

pub use bridge::{HostPeer, IpcBridge};
pub use context::{
    ContextProvider, ContextRecord, ContextRequest, IpcTerminalContext, SemanticZone,
};
pub use error::SliceError;
pub use provider::{
    DeterministicLocalProvider, Message, ModelProvider, ProviderTurn, Role, ToolCall,
};
pub use session::{SliceOutcome, VerticalSlice};
pub use stream::{Fragment, FragmentKind, PanelStreamSink, StreamChunk, StreamSink};
pub use toolbus::{HostToolBus, ToolBus, ToolDecl, ToolHost, ToolInvocation, ToolOutcome};

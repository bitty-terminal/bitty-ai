# bitty-ai-slice

Integration harness: the real `bitty-ai-runtime` driven through the real
generic `bitty-ipc` primitives.

## Status

Pressure test, not a shipped component. It exists to answer one architecture
question (CTX-0407, 017 recommendation 6, and the `BA-6` pressure-test gate in
the AI architecture specification): can an end-to-end AI turn be expressed
using the real AI runtime over only the generic primitives that Core already
exposes, or does it demand an AI-specific Core API?

Since AI-0012 this crate owns no AI mechanism: the former second runtime
(`provider`, `context`, `session`, `stream`, `toolbus` modules) was deleted
once `bitty-ai-runtime` subsumed it. What remains is the host boundary
(`IpcBridge`), typed bridge failures (`SliceError`), and the `harness`
fixtures that adapt real IPC snapshots to real runtime inputs.
`tests/vertical_slice.rs` proves the full loop (prompt -> provider ->
context -> tool -> streamed result) and the fail-closed paths against the
real runtime.

The harness reuses the real, externally consumable generic IPC primitive
`bitty-ipc` (bounded wire envelope, method/scope authorization, per-client
consent ledger, RC-10 chunk validation, bounded channels, and the headless
MCP client stub) through a pinned Git revision. It never links Core's
in-process agent or runtime. Terminal content is requested through the
generic `terminal.snapshot` wire method under the `terminal.inspect` scope;
when the host does not implement a method or a capability is absent, the
harness fails closed and the gap is recorded rather than worked around with
a new Core API.

The harness is deterministic: every operation takes a caller-supplied
`now_ms`, there is no wall-clock, thread, async runtime, network, or secret,
and the model provider is the runtime's scripted `FakeProvider`.

## Modules

```text
bridge     Generic host boundary composing real bitty-ipc primitives; unknown methods fail closed
error      Typed fail-closed slice errors (unsupported method, consent, budget and bound violations)
fake_host  Deterministic FakeHost double mirroring bitty dispatch, consent, and result shapes (BII-09)
harness    Fixtures adapting real IPC snapshots to real runtime inputs with scripted provider
live_host  LiveBittyHost adapter delegating to the real bitty-ipc services via an injectable seam (BII-09)
local_provider  Experiment (AI-0042): localhost-only LocalProvider speaking to a local Ollama/OpenAI-compatible endpoint over std TcpStream with loopback enforcement, mandatory timeouts, and bounded fail-closed JSON
```

## Dependencies

- `bitty-ai-runtime` via local path.
- `bitty-ipc` via pinned Git revision (`2cbb1fbed82814c157359b71dd8efbb4be0c36e7`
  in `Cargo.toml`).

## FakeHost and the live host

`FakeHost` implements the `BittyHost` trait defined in `src/fake_host.rs`.
`LiveBittyHost` (`src/live_host.rs`) implements the same trait by delegating
to the real `bitty-ipc` services (`SnapshotService`, `ToolDispatchService`,
`ExecutionService`) with the real DTOs, bounds, and `validate()` methods.
Swapping hosts is mechanical: replace the construction site only; callers
keep calling the same trait methods.

Live wiring is explicitly out of scope here: this crate contains no code that
connects to a real terminal, process, PTY, socket, or network peer (no
`std::net`, `std::process`, `std::fs`, async runtime, or IPC transport).
`LiveBittyHost` carries only an injectable provider seam (`fn` pointers),
server-evaluated scopes, and the real consent ledger; tests use canned
providers as mapping proof and claim no live data. Every operation takes
caller-supplied `now_ms`, and there are no secret or credential fields —
except the `local_provider` experiment, which opens loopback-only TCP with
mandatory timeouts and a caller-supplied, redacted key.

`FakeHost` is std-only and deterministic: no network, no threads, no
filesystem, no wall clock. Every method takes caller-supplied `now_ms`;
expiry is evaluated against it, never against system time. One `FakeHost`
serves one `client_id` with its own server-evaluated granted scopes and
consent ledger. It reuses the real `bitty-ipc` DTOs, bounds, and `validate()`
methods directly; only the dispatch order is mirrored as explicit steps with
the same denial classes and no-partial-state guarantee.

## Tests

```text
cargo test -p bitty-ai-slice
```

- `tests/vertical_slice.rs` (11 tests): full loop (prompt -> provider ->
  context -> tool -> streamed result) and fail-closed paths against the real
  runtime.
- `tests/fake_host.rs` (27 tests) plus `src/fake_host.rs` inline tests
  (4 tests): snapshot, tool dispatch, supervised execution
  (`execute`/`reconcile`/`resolve` including `Unknown`), consent
  grant/revoke/expiry, and the shared `BittyHost` trait surface.
- `tests/host_conformance.rs` (shared, FakeHost + LiveBittyHost): dispatch
  prefix order, consent attribution, and `ExecutionResult`/`Unknown`
  reconcile semantics against the real `bitty-ipc` shapes.
- `src/live_host.rs` inline tests (4 tests): live delegation for snapshot,
  tool dispatch, `Unknown` reconcile, and missing-handler fail-closed.

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
`now_ms`, there is no wall-clock, thread, or async runtime, and the model
provider is the runtime's scripted `FakeProvider`. The harness paths
themselves (`FakeHost`, `LiveBittyHost`, `IpcBridge`, `harness`) open no
socket and contain no secret field. Two experiments are the documented
exceptions. The `local_provider` experiment (AI-0042) opens loopback-only
`TcpStream` connections with loopback enforcement, mandatory timeouts, one
monotonic request deadline (AI-CTX-005), and a caller-supplied, redacted key;
`journal_prototype` (AI-0049) persists an append-ordered single-writer journal
through SQLite in a caller-supplied database file. The journal module is a
prototype for the draft R6 persistence profile
(`docs/specifications/persistence-profile-r6.md`); both are evidence only and
decide no open register entry.

## Modules

```text
bridge     Generic host boundary composing real bitty-ipc primitives; protocol-identity -> client_id binding rule; unknown methods fail closed
error      Typed fail-closed slice errors (unsupported method, consent, budget and bound violations)
fake_host  Deterministic FakeHost double mirroring bitty dispatch, consent, and result shapes (BII-09)
harness    Fixtures adapting real IPC snapshots to real runtime inputs with scripted provider
live_host  LiveBittyHost adapter delegating to the real bitty-ipc services via an injectable seam (BII-09)
local_provider  Experiment (AI-0042): localhost-only LocalProvider speaking to a local Ollama/OpenAI-compatible endpoint over std TcpStream with loopback enforcement, mandatory timeouts, one monotonic request deadline (AI-CTX-005), and bounded fail-closed JSON
journal_prototype  Experiment (AI-0049): append-ordered single-writer SQLite journal with deletion tombstones, bounded fields, and fail-closed corruption handling (no FTS5, no scheduler, caller-supplied timestamps)
fragment_transport  Runtime-to-transport pre-split (AI-0066, AI-0070): 64 KiB runtime fragments -> <=16 KiB parts at code-point boundaries with a continuation marker, dense seq, and a caller-bindable reassembly identity
```

## Dependencies

- `bitty-ai-runtime` via local path.
- `bitty-ipc` via pinned Git revision (`cfeffa2d8e1387029850940af2b64877dbfbe25f`
  in `Cargo.toml`). Detect drift between this pin and the local `bitty`
  checkout with `just pin-drift` (`scripts/pin-drift.sh`); it is local-only by
  default and documents the bump checklist in its header. It is intentionally
  excluded from `just check`, which must not require network access.
- `rusqlite` `=0.40.2`, `default-features = false`, `features = ["bundled"]`,
  for the `journal_prototype` experiment only. The dependency review (AI-0049
  PX-0298) covers supply-chain breadth, MIT licensing, Rust 1.85 fit, and
  ambient authority; `bundled` compiles vendored SQLite 3.53.2 and requires a
  C toolchain. `bitty-ai-runtime` stays std-only and untouched.

## FakeHost and the live host

`FakeHost` implements the `BittyHost` trait defined in `src/fake_host.rs`.
`LiveBittyHost` (`src/live_host.rs`) implements the same trait by delegating
to the real `bitty-ipc` services (`SnapshotService`, `ToolDispatchService`,
`ExecutionService`) with the real DTOs, bounds, and `validate()` methods.
Swapping hosts is mechanical: replace the construction site only; callers
keep calling the same trait methods.

Live wiring is explicitly out of scope here: the harness path contains no
code that connects to a real terminal, process, PTY, socket, or network peer
(no `std::process`, async runtime, or IPC transport). `LiveBittyHost` carries
only an injectable provider seam (`fn` pointers), server-evaluated scopes, and
the real consent ledger; tests use canned providers as mapping proof and claim
no live data. Every operation takes caller-supplied `now_ms`. The
`local_provider` experiment (AI-0042) is the one path with a real socket:
loopback-only `TcpStream` with loopback enforcement, mandatory timeouts, one
monotonic request deadline (AI-CTX-005), and a caller-supplied, redacted key —
no other secret or credential field exists in the crate. Filesystem access
appears only in the `journal_prototype` experiment (the caller-supplied
database path and SQLite's own sidecar files in that directory) and its test
scratch directories.

`FakeHost` is std-only and deterministic: no network, no threads, no
filesystem, no wall clock. Every method takes caller-supplied `now_ms`;
expiry is evaluated against it, never against system time. One `FakeHost`
serves one `client_id` with its own server-evaluated granted scopes and
consent ledger. It reuses the real `bitty-ipc` DTOs, bounds, and `validate()`
methods directly; only the dispatch order is mirrored as explicit steps with
the same denial classes and no-partial-state guarantee.

## Protocol identity to wire `client_id` binding (AI-0065)

The host keys its consent ledger and execution store by a wire `client_id`
string, while `bitty-ai-runtime`'s `IdentityBridge` owns the protocol principal
`ProtocolAgentId(owner.name)`. `bridge::wire_client_id` defines the one
deterministic rule that binds them: `client_id` is exactly the validated
`owner.name` wire principal of the bound identity (matching the upstream
`ConsentGrant.client_id` contract). The rule fails closed:

- an unbound identity is refused, so a caller cannot substitute an arbitrary
  id for a missing binding, and
- a bound principal longer than the upstream `client_id` bound
  (`bridge::MAX_WIRE_CLIENT_ID_BYTES`, re-derived from the pinned `bitty-ipc`
  `auth::MAX_SCOPED_ID_BYTES` value `64`, pinned revision
  `cfeffa2d8e1387029850940af2b64877dbfbe25f`) is refused, because the runtime
  protocol bound (128 bytes) is wider than the host wire bound.

`bridge::verify_wire_client_id` refuses a caller-supplied id that disagrees
with the bound principal. `IpcBridge::from_binding`, `FakeHost::from_binding`,
and `LiveBittyHost::from_binding` are the sanctioned construction paths; the
raw `new(client_id, ..)` constructors remain the explicit test/host seam.
Every refusal is total: no host is constructed, nothing is dispatched, and no
store entry is written.

## Fragment pre-split (`fragment_transport`)

The runtime bound `bitty_ai_runtime::stream::MAX_FRAGMENT_BYTES` (64 KiB,
reject) and the `bitty-ipc` ingest bound
`rich_fragment::MAX_FRAGMENT_TEXT_BYTES` (16 KiB, truncate with
`truncated = true`) disagree, so a full 64 KiB fragment projected verbatim into
the ingest service loses bytes. `src/fragment_transport.rs` owns the
deterministic pre-split rule that removes the loss:

1. A validated runtime chunk whose fragment is valid UTF-8 is cut greedily
   left to right into the longest prefixes that are `<= 16 KiB` and end on a
   UTF-8 code-point boundary (a code point is never split).
2. Parts of one source fragment carry a dense zero-based `part_index`
   (`0..part_count`), `part_count >= 1`, and `is_continuation =
part_index > 0`.
3. Transport `seq` values are assigned contiguously from a cursor: part `i`
   gets `first_seq + i`, and the cursor advances by `part_count`, so keys stay
   unique under `(terminal_id, generation, seq)`.
4. Concatenating the parts in order reproduces the source bytes exactly and
   every part stays under the 16 KiB ceiling, so the ingest service never
   truncates a pre-split part.
5. Both entry points enforce the 64 KiB runtime bound: `pre_split_chunk` through
   `validate_chunk`, and the fragment-level `pre_split_fragment` directly, so a
   raw `Fragment` larger than one runtime fragment is refused rather than split.
6. `reassemble` refuses parts that do not share one source identity: every part
   must carry the first part's `(terminal_id, generation, source_seq)` and a
   transport `seq` of `first_seq + part_index`, so a foreign or reordered part
   fails closed instead of being concatenated.
7. `reassemble_expected` binds the whole part set to a caller-supplied
   `FragmentIdentity` (AI-0070). Because rule 6 only checks the parts against
   each other, a wholly foreign but internally consistent set reassembles
   silently; binding the first part absolutely to the caller's expectation and
   every later part to the first rejects it with
   `FragmentTransportError::IdentityMismatch`. A later part whose recorded
   `part_count` disagrees with the first reports
   `FragmentTransportError::InconsistentPartCount`, kept distinct from the
   supplied-length `FragmentTransportError::PartCountMismatch`.
8. `reassemble` and `reassemble_expected` enforce aggregate admission bounds
   against `MAX_FRAGMENT_BYTES` (64 KiB) before allocating reconstructed output
   (AI-0104): all parts are validated and their text lengths summed into
   `total_bytes`; if `total_bytes > MAX_FRAGMENT_BYTES`, reassembly fails closed
   with `FragmentTransportError::OversizedFragment` before any output allocation.

This module is the runtime-to-transport mapping layer, not a shipped
transport. `bitty-ai-runtime` stays std-only with zero dependencies and cannot
name the transport ceiling; `bitty-ipc` is upstream and unchanged; no `rich.*`
wire method is registered. The rule above is the contract the future
production mapper must follow.

## Tests

```text
cargo test -p bitty-ai-slice
```

- `tests/vertical_slice.rs` (11 tests): full loop (prompt -> provider ->
  context -> tool -> streamed result) and fail-closed paths against the real
  runtime.
- `tests/client_id_binding.rs` (8 tests): protocol identity -> wire
  `client_id` binding rule (deterministic derivation, single upstream length
  bound, and the unbound / over-long / disagreeing negatives with no dispatch
  or store entry).
- `tests/fake_host.rs` (27 tests) plus `src/fake_host.rs` inline tests
  (4 tests): snapshot, tool dispatch, supervised execution
  (`execute`/`reconcile`/`resolve` including `Unknown`), consent
  grant/revoke/expiry, and the shared `BittyHost` trait surface.
- `tests/host_conformance.rs` (shared, FakeHost + LiveBittyHost): dispatch
  prefix order, consent attribution, and `ExecutionResult`/`Unknown`
  reconcile semantics against the real `bitty-ipc` shapes.
- `tests/pinned_surface.rs` (6 tests): the pinned `bitty-ipc` contract the
  mirror relies on (`AI-0081` window `be6e63c` -> `cfeffa2`): set-based
  `validate_wire_version` plus `negotiate_wire_version` selection and
  fail-closed no-overlap behavior, envelope validation at the negotiated
  version, `IpcEndpoint::drain_expired` purging expired queued requests while
  live ones survive, and an uncorrelated peer answer refused without pinning
  pending capacity.
- `tests/fragment_mapping.rs`: runtime fragment -> transport `FragmentData`
  mapping, including the AI-0066 pre-split proof that a 64 KiB multi-byte
  UTF-8 block reassembles byte-identically through the real ingest service
  (the counterfactual direct projection truncates), plus the AI-0069 negatives:
  a foreign `terminal_id`/`generation`/`source_seq` part, a non-contiguous
  transport `seq`, a mismatched later-part `part_count`, and an oversized raw
  fragment at the fragment-level entry point are all refused. The AI-0070
  additions prove `reassemble_expected` rejects a foreign but internally
  consistent part set (`IdentityMismatch`) that `reassemble` alone accepts, and
  that the two `part_count` failure modes stay distinguishable
  (`PartCountMismatch` vs `InconsistentPartCount`). The AI-0104 additions prove
  that aggregate parts exceeding `MAX_FRAGMENT_BYTES` fail closed with
  `OversizedFragment` before output allocation while exact-bound sets reassemble.
- `src/live_host.rs` inline tests (4 tests): live delegation for snapshot,
  tool dispatch, `Unknown` reconcile, and missing-handler fail-closed.
- `src/journal_prototype.rs` inline tests (8 tests): append/read-back order
  across reopen, tombstone visibility without history rewrite, tombstone
  idempotency and unknown-id no-op, duplicate-id rejection, bounded-field
  fail-closed, garbage-file corruption fail-closed (file untouched),
  incompatible-schema fail-closed (rows preserved), and single-writer
  rejection until the first writer closes.

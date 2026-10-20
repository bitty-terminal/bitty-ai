# bitty-ai-control

Helper-side launch-scoped mutual-certificate control transport and
provider-publication state machine, plus the `bitty-ai-helper` launch
entrypoint.

**Status:** pre-review implementation (AI-0333 transport, AI-0337 state
machine, AI-0338 entrypoint). Strict provider publication remains disabled.
This crate does not claim integrated authority, provider availability, or
production readiness. Canonical architecture and security requirements live in
`bitty-ai-docs` (`architecture/authenticated-provider-publication.md`); this
README records only the repository-local launch and binary contract.

## Launch contract

The Core supervisor starts `bitty-ai-helper` through its provider-publication
launch seam with two inherited descriptors:

| Descriptor | Constant                     | Channel                                                                   |
| ---------- | ---------------------------- | ------------------------------------------------------------------------- |
| `3`        | `CHILD_BOOTSTRAP_DESCRIPTOR` | one-shot bootstrap read end: exactly one launch manifest record, then EOF |
| `4`        | `CHILD_CONTROL_DESCRIPTOR`   | dedicated duplex control stream, TLS 1.3 from its first byte              |
| `5`        | not defined or used          | reserved for the data channel in a later slice                            |

The bootstrap channel is supervisor-to-helper only. Manifest bytes never
appear on the control stream, and control records never appear on the bootstrap
channel. Descriptor `5` is never adopted; `Admit`/`Admitted`, provider routing,
and credential contact are not implemented.

The constants above mirror the Core launch seam
(`bitty-runtime` `ai_publication_launch::CHILD_BOOTSTRAP_DESCRIPTOR` and
`CHILD_CONTROL_DESCRIPTOR`). `bitty-ai-control` has no dependency edge on the
Core crate, so the values are named and documented here rather than imported.

## `bitty-ai-helper`

- Source path: `crates/bitty-ai-control/src/bin/bitty-ai-helper.rs`
- Binary name: `bitty-ai-helper`
- Built artifact (Cargo default): `target/<profile>/bitty-ai-helper`
- Workspace target: `cargo build -p bitty-ai-control --bin bitty-ai-helper --locked`

On Unix the binary adopts descriptors `3` and `4` through one narrow audited
`unsafe` function and calls the library entrypoint `run_helper`. On every
non-Unix target it fails closed with the `UnsupportedPlatform` diagnostic and
exit code `2` before adopting or reading any descriptor. No packaging, install
script, service definition, or Core `ai_helper` wiring is part of this
repository or this slice.

### `executable_sha256` binding

Core's launch manifest `executable_sha256` must identify the exact
`bitty-ai-helper` executable produced from the launch's verified build. The
digest binding belongs to Core's trusted descriptor and launch path, which
already verifies the component before spawn; the helper does not read its own
image, compute a digest, or compare `executable_sha256` locally. There is no
trusted descriptor source in this crate, and the manifest field is only echoed
back in the establishment exchange. Launching an untracked or stale helper is a
Core launch defect, not something this binary can detect.

### Exit codes

The library maps every terminal outcome to `HelperExit`; diagnostics are fixed
text with no payload, identifier, path, model string, certificate, or key
material.

| Code | Outcome                 | Meaning                                                     |
| ---- | ----------------------- | ----------------------------------------------------------- |
| `0`  | `AuthorityClosed`       | Core withdrew authority (`GrantLoss` or `Shutdown`)         |
| `0`  | `PeerClosed`            | Core sent a terminal control error                          |
| `2`  | `UnsupportedPlatform`   | no inherited-descriptor launch contract on this target      |
| `3`  | `BootstrapIo`           | bootstrap read failed                                       |
| `4`  | `BootstrapManifest`     | manifest malformed, truncated, oversized, or trailing bytes |
| `5`  | `BootstrapTrailingData` | data after the bootstrap end-of-file marker                 |
| `6`  | `ServerConfiguration`   | control server configuration or key material rejected       |
| `7`  | `Handshake`             | mutual TLS establishment failed or was refused              |
| `8`  | `Establish`             | application establishment exchange failed                   |
| `9`  | `Control`               | post-establishment control step failed                      |

`ServerConfiguration` is a defensive mapping: the manifest parser and the
control-server builder both validate key material, so an ingested manifest has
already passed those checks.

### Sequence

`run_helper(bootstrap, control, registry)` performs exactly:

1. `LaunchManifest::read_from` and an explicit end-of-file check on the
   bootstrap reader;
2. drop the bootstrap reader, closing the inherited descriptor before any
   control work;
3. `ControlServer::new(manifest)`;
4. the blocking mutual-certificate `accept(control)`;
5. `HelperControl::from_session`;
6. `establish`, then `step` until the Core closes authority or a terminal
   failure occurs.

The library takes injected `Read`/`Read + Write` streams and owns no listener,
socket, thread, clock, deadline, environment, or filesystem input.

## Control-only registry adapter

`control_registry::ControlOnlyRegistry` is a fixed empty `provider-routing`
projection at feed revision `1`. It exists only so the entrypoint can drive the
state machine without provider integration:

- it is **not** the real provider registry, store, or feed;
- it never mutates, so `registry_mutated` has no production path here;
- `retained_matches` returns true only for the canonical empty projection at
  revision `1` and false for any other revision or bytes.

Real registry feed integration (provider metadata, feed revisions, mutation
invalidation, and the registry read gate) is a follow-up task.

## Tests

- `tests/entrypoint.rs` drives `run_helper` over an in-memory duplex and the
  checked-in certificate fixtures: establishment, `Publish -> Ack -> Bind ->
Bound`, malformed/extra bootstrap data, premature control closure, the
  non-Unix gate, and exit-code/diagnostic mapping. Threads exist only in the
  test harness, never in the library or binary.
- `tests/tls.rs`, `tests/state_machine.rs`, and `tests/manifest.rs` cover the
  transport, state machine, and ingestion slices below the entrypoint.

Run the crate gates through the repository justfile, or directly:

```text
cargo clippy -p bitty-ai-control --all-targets --locked -- -D warnings
cargo test -p bitty-ai-control --all-targets --locked
```

# Metadata Publication API

`metadata_publication` implements a helper-side, in-memory S-1 publication
handshake. Core publishes a `PublishRequest` containing a unique
`publication_id`, monotonically increasing `snapshot_version`, SHA-256 digest,
and `metadata_json`. The digest is lowercase hex SHA-256 over the exact UTF-8
bytes of `metadata_json`. A successful publication returns the same identity,
version, and digest as `PublicationAck`; the acknowledgement operation accepts
that tuple exactly once and only while it is the latest pending publication.
Accepted publication IDs cannot be reused during the lifetime of a state, even
after acknowledgement. The state retains at most 256 accepted IDs and rejects
further valid publications once that bound is reached.

Metadata is limited to provider IDs/names, model names, capability labels, and
routing facts. It cannot carry provider selection decisions, credentials,
provider implementations, or model requests. Frames and metadata JSON are
limited to 256 KiB, with independent collection and string bounds enforced by
the endpoint. Unknown fields, duplicate JSON keys, malformed JSON, credential-
like labels or provider IDs, mismatched digests, replayed/non-increasing
versions, reused publication IDs, exhausted publication history, and superseded
acknowledgements fail closed.

This module defines vocabulary and state validation only. It adds no transport;
Core must wire publish/ack through the resident helper IPC and establish the
producer-side metadata source before any `snapshot_version` is sent on normal
`ai.*` requests. Existing endpoint entries remain tolerant for standalone use.

`StrictEndpoint` can retain an opaque capability made from an exact
`PublicationState` acknowledgement, but strict dispatch is currently
fail-closed as `Unbound`. The `provider_binding` module adds the first pure
helper-side `provider-routing` projection for the populated runtime
`ProviderRegistry`: canonical UTF-8 bytes contain only provider IDs, model
names, sorted closed capability labels, context windows, and relative cost
weights. Provider display names, routing facts, aliases, credentials,
endpoints, provider implementations, requests, and MCP tool-list fields are
excluded. It can issue an opaque equality proof only after an exact
domain-tagged publish/ack and current-registry digest match; registry
mutation, republish, stale/replayed input, wrong domain, and mismatches fail
closed. Binding retains the bounded exact canonical publication bytes through
pending, acknowledged, and bound lifecycle state. It compares those bytes
byte-for-byte with a fresh `ProviderRegistry` projection in addition to the
SHA-256 digest, exact identity/domain/version, publication epoch, and registry
revision. This is not live Core/helper wiring or full S-F behavior.

The MCP `SnapshotEnvelope::list_version` is not treated as equivalent to the
provider publication's `snapshot_version`. `dispatch_with_strict` remains
`Unbound` for every method in this revision, and the existing standalone
`dispatch_with` path remains tolerant. No caller-supplied publication ID,
digest, version, or projection bytes are authority, and `StrictReadOnlyView`
intentionally has no such fields.

## Provider-routing byte binding

The provider-routing helper retains the bounded, exact canonical UTF-8
publication bytes from `publish_provider_routing` through pending,
acknowledged, and bound state. The acknowledgement frame carries only the
validated identity, domain, version, and digest; the helper returns the
retained publication record rather than trusting caller-supplied bytes. Binding
and current checks require both byte-for-byte equality and the lowercase
SHA-256 digest against a fresh projection of the actual `ProviderRegistry`
consumed by `ai.provider_list`.

Registry revision is a mutation guard, not an instance-identity claim. Two
registry values with the same revision are accepted only when their canonical
projections are byte-for-byte equal; a different projection fails closed. The
retained projection is bounded by the publication, model, provider, capability,
field, and string limits described above, so memory is O(N) for N bounded
projection bytes plus the bounded model/state records.

These public publication helpers remain offline, in-memory seams with **no
authenticated Core ingress**. They do not establish Core origin, add provider
I/O, or enable strict dispatch. `dispatch_with_strict` remains `Unbound`; live
Core/Lua/IPC transport and authenticated publication wiring are separate
follow-ups. The six gates remain explicit: provider-routing domain, exact
identity and version lifecycle, canonical projection, digest, exact retained
bytes, and current registry revision/epoch. MCP remains a separate version
domain, and the default tolerant dispatch path is preserved.

Replay and supersession guarantees are scoped to one `PublicationState`
instance lifetime. Reconstructing a fresh state does not provide cross-reset
identity or authenticated channel isolation; an identical publication can be
accepted again and an otherwise matching old proof can be current in that
reconstructed state. This offline limitation is explicit and is not a claim of
Core-origin authentication. Durable helper/channel lifetime isolation is tracked
by AI-0330 and blocks AI-0329 strict success.

This is only a partial, helper-side seam. The accepted contract is documented
in the [`bitty-ai-docs` execution-ownership R1 draft](https://github.com/bitty-terminal/bitty-ai-docs/blob/main/architecture/execution-ownership-r1.md#providerrouting-snapshot-binding-contract-accepted-2026-10-17);
the R1 document remains a draft overall, while its provider-routing binding
contract is accepted separately. Live authenticated Core/Lua/IPC caller
wiring, strict handler binding, and production release remain out of scope.
This API makes no claim that publication metadata equals MCP tool-list metadata.

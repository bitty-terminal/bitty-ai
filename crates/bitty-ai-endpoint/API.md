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
closed. This is not live Core/helper wiring or full S-F behavior.

The MCP `SnapshotEnvelope::list_version` is not treated as equivalent to the
provider publication's `snapshot_version`. `dispatch_with_strict` remains
`Unbound` for every method in this revision, and the existing standalone
`dispatch_with` path remains tolerant. No caller-supplied publication ID,
digest, or version is authority, and `StrictReadOnlyView` intentionally has no
such fields.

This is only a partial, helper-side seam. The candidate contract is documented
in the [`bitty-ai-docs` execution-ownership design](https://github.com/bitty-terminal/bitty-ai-docs/blob/main/architecture/execution-ownership-r1.md#provider-routing-snapshot-binding-candidate-draft-2026-10-17);
that document remains a candidate and is not an accepted product contract by
this implementation. Live Core/Lua/IPC caller wiring, strict handler binding,
and production release remain out of scope. This API makes no claim that
publication metadata equals MCP tool-list metadata.

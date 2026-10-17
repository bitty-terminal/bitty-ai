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
fail-closed as `Unbound`. The acknowledgement describes the provider/routing
metadata validated by `validate_metadata`; the current `ReadOnlyView` exposes
runtime registry data and MCP pin/tool-list envelopes. No same-domain helper
snapshot binding is present, and the MCP `SnapshotEnvelope::list_version` is
not treated as equivalent to the publication's `snapshot_version`. Therefore
`dispatch_with_strict` refuses before handlers for every method in this
revision. No caller-supplied publication ID, digest, or version is authority,
and `StrictReadOnlyView` intentionally has no such fields.

This is only a partial, helper-side seam. Core/Lua/IPC caller wiring, a
same-domain current binding, and production release remain out of scope; this
API makes no claim that publication metadata equals MCP tool-list metadata.

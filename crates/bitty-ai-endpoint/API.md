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
`ai.*` requests. Until that integration exists, the optional E2 correlator
remains tolerant and AI-0303 stale/missing enforcement remains disabled.

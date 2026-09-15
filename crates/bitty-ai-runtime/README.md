# bitty-ai-runtime

Deterministic single-agent runtime skeleton (AI-0009, draft scope).

## Status

Skeleton, not an accepted contract. This crate tracks the draft
`implementation-profile-v0.1.md` (status: draft; a draft disposition
proposes no accepted architecture); normative architecture stays in the
canonical `docs/specifications/` corpus (`ai-architecture.md`,
`context-management.md`, `command-tool-architecture.md`,
`agent-coordination.md`, `implementation-profile-v0.1.md` plus the
`R1`..`R6` draft dispositions).

## Modules

```text
provider  ModelProvider trait, deterministic FakeProvider (scripted turns)
context   Stable Ids, token-first budget, L0 structured results, L1 prune
session   AgentInstanceId/RunId/SessionId/ExecutionId, tiers, idempotent cancel
tool      Bounded registry, deny-by-default hooks, host executor seam
agent     Single-agent turn loop with structured outcomes incl. Unknown
stream    Markdown/Diff/ToolCard fragments, seq/total/final chunks
bridge    P1 wire owner.name <-> AgentInstanceId map + consent seam (deny + fake)
prompt    Five-layer deterministic assembly, narrowing-only, canonical bytes
selection Deterministic registry, capability-subset matching, data-held aliases, ordered fallback
compression L2 selective-compression prototype (host summarizer seam, deterministic breakpoints, in-memory retention); fake-verified, no durable store, AIQ-11 stays open
```

## Rules

- Every operation takes caller-supplied `now_ms`: no wall clock, threads,
  async runtime, network, filesystem, or secrets.
- Authorization hooks deny by default; without host wiring the crate refuses.
- The runtime never executes a tool; the host plugs in `ToolExecutor`.
- The real consent ledger lives host-side; this crate owns only the seam
  (`ConsentLedger` trait, `DenyAllConsent`, `FakeConsentLedger` test double).
- Protocol identity is a validated wire string (`owner.name`); this crate
  never depends on the generic `bitty-agent` crate.
- `Unknown` effects reconcile before retry; the loop never blindly retries.

## Tests

```text
cargo test -p bitty-ai-runtime
```

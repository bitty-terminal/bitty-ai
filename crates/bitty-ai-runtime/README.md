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
```

## Rules

- Every operation takes caller-supplied `now_ms`: no wall clock, threads,
  async runtime, network, filesystem, or secrets.
- Authorization hooks deny by default; without host wiring the crate refuses.
- The runtime never executes a tool; the host plugs in `ToolExecutor`.
- `Unknown` effects reconcile before retry; the loop never blindly retries.

## Tests

```text
cargo test -p bitty-ai-runtime
```

# bitty-ai-runtime

Deterministic single-agent runtime skeleton (AI-0009, draft scope).

## Status

Skeleton, not an accepted contract. This crate tracks the unmerged draft
`implementation-profile-v0.1.md` content; normative architecture stays in the
canonical `docs/specifications/` corpus (`ai-architecture.md`,
`context-management.md`, `command-tool-architecture.md`,
`agent-coordination.md`).

## Modules

```text
provider  ModelProvider trait, deterministic FakeProvider (scripted turns)
context   Stable Ids, token-first budget, L0 structured results, L1 prune
session   AgentId/RunId/SessionId/ExecutionId, tiers, idempotent cancel
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

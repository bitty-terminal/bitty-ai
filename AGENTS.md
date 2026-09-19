# bitty-ai agent guide

## Scope and authority

- This file governs only the independent `bitty-ai` Git repository.
- The umbrella directory is not a Git repository and other Bitty repositories own
  their own Git, CarryCtx, CI, releases, and agent guidance.
- All formal Bitty repositories belong under <https://github.com/bitty-terminal>.
- [`bitty-ai-docs`](https://github.com/bitty-terminal/bitty-ai-docs) is the canonical source for AI-core architecture,
  specifications, and design documentation. It is mounted at `docs/` as a Git
  submodule pinned to a commit; initialize it with
  `git submodule update --init`.
- [`bitty-docs`](https://github.com/bitty-terminal/bitty-docs) owns shared governance (decisions, security corpus, reviews,
  project state); [`bitty-terminal-docs`](https://github.com/bitty-terminal/bitty-terminal-docs) and [`bitty-plugins-docs`](https://github.com/bitty-terminal/bitty-plugins-docs) own the
  respective corpora and are linked by absolute URL.

## Current phase

- Experimental, pre-alpha status: the repository contains an experimental
  implementation inspected at `bcb8365015a5726ba4beefecde14096274878a6b`
  (2026-09-19) — two workspace crates, `bitty-ai-runtime` (deterministic
  single-agent runtime skeleton, AI-0009, draft scope) and `bitty-ai-slice`
  (pressure-test harness, not a shipped component, AI-0005). It is not the
  complete proposed runtime, and no behavior is verified,
  compatibility-guaranteed, or production-ready.
- Documentation-first: architecture, security requirements, and design decisions
  must be captured in the `docs/` submodule (bitty-ai-docs) before
  implementation.
- Do not add product code unless a later task authorizes it and its architecture
  and security gates are accepted.
- Rust components use edition 2024 and resolver 3, with MSRV 1.85 declared in
  `Cargo.toml` (`rust-version`) and `clippy.toml`, enforced by the CI `msrv`
  job; the pinned toolchain is stable `1.98.1` (`rust-toolchain.toml`) and no
  nightly is used. Further crate splits, additional dependencies, and release
  profiles remain proposal-stage (the v0.1 Implementation Profile and
  Dependency Strategy are draft; AIQ-5C stays open).
- Never describe planned behavior, a candidate dependency, or a configuration
  file as implemented evidence.

## Read before acting

1. Read this guide and the applicable files in `.carryctx/rules/`.
2. Adopt the assigned persona in `.carryctx/personas/`.
3. Read the task, team context, exact scopes, dependencies, and relevant
   canonical contracts in the `docs/` submodule (bitty-ai-docs); run
   `git submodule update --init` first when `docs/` is empty.

## CarryCtx workflow

- CarryCtx is the durable execution record; it does not spawn agents.
- Bind a named agent and session to the task before work. Record progress,
  decisions, risks, blockers, handoffs, and checkpoints while work is active.
- Map GitHub Issue intent to a CarryCtx task; repository ownership to a team;
  ordering to dependencies; edits to exact scopes; active work to a session;
  and recovery points to checkpoints.
- Subagents perform narrowly scoped implementation. The commander coordinates,
  reads durable state back, verifies the diff, and runs acceptance gates.
- Independent review is required before completion. Self-reports are not
  acceptance evidence.

## Delivery lifecycle

- The normal lifecycle is GitHub Issue, CarryCtx task, team/dependencies/scopes,
  named session, isolated worktree and branch, coherent commits, pull request,
  independent review plus CI, merge, documentation synchronization, final
  checkpoint, task completion, and Issue closure.
- After the first commit, parallel implementation uses dedicated worktrees and
  branches. Branches use `ctx-XXXX/<type>-<short-slug>` where `XXXX` is the
  owning CarryCtx task number (using AI prefix), `<type>` is one of
  `feat|fix|chore|docs`, and the slug is short kebab-case; CarryCtx-bound
  worktrees live at `.worktrees/ctx-XXXX-<type>-<short-slug>` with `/` mapped
  to `-`. One branch per task; commander housekeeping branches may use
  `cmd/<slug>`.
- Before the first commit, branches, worktrees, commits, and pull requests are
  unavailable. The commander may authorize a shared checkout only for disjoint
  scopes with CI-equivalent local checks. This exception ends at initialization.
- Do not commit, push, merge, publish, or mutate remote state unless the task or
  user explicitly authorizes it.

## Issue hygiene (labels and milestones)

- Every GitHub Issue and PR must have appropriate `labels` (e.g., `feat`, `fix`,
  `docs`, `chore`, `P0`, `area:model-provider`, `area:context`, `area:agent`,
  `area:tool-bus` etc.) and `milestone` (e.g., `v0.1.0`, `v1.0`) when
  applicable.
- Use `gh issue create --label "feat,area:model-provider" --milestone "v0.1.0"`
  and `gh issue edit`/`gh pr edit` to add labels/milestones.
- Every CarryCtx task must include
  `Priority: P0/P1/P2 | Area: xxx | Labels: feat,area:xxx,P0 | Milestone: v0.1.0 | RFC: OQ-xxx | Task: AI-XXXX`
  in description.

## Local gates before push (mandatory)

- Before pushing any branch: run repository justfile gates locally and ensure 0
  issues: `just check` (fmt-check + clippy -D warnings + test + actionlint)
  plus full `cargo test --workspace --all-targets --locked` and validate GitHub
  workflows with `act -n` for `.github/workflows/ci.yml` and
  `.github/workflows/codeql.yml`. All must pass. Never push with known local
  failures.

## Remote monitoring and merge (bitty-ai)

- After push, monitor via
  `HTTPS_PROXY=$NETWORK_PROXY gh pr checks <PR> --watch --interval 15` until
  CodeQL and Quality gates pass, mergeable==MERGEABLE, then
  `gh pr merge --squash`. Prefer `--watch` over `sleep` loops; `pty_spawn` with
  `notifyOnExit` handles long waits.

## Documentation contract

- Repository-owned documentation is English-only.
- **All AI architecture and design documentation lives in `bitty-ai-docs`,
  mounted at `docs/`, to prevent duplication and drift.** This repository
  contains only implementation-specific API docs and crate README files.
- Initialize/refresh the submodule with `git submodule update --init`; bump the
  pin with `git submodule update --remote docs` followed by `git add docs` and
  a `docs:` commit. `docs/` is external content: `just check` excludes it from
  markdownlint, so gates behave identically with and without the submodule
  initialized. Security and governance documents stay in `bitty-docs`.
- Synchronize affected canonical material in `bitty-ai-docs` (via the `docs/`
  submodule pointer) when architecture, security, public behavior,
  configuration, compatibility, or developer workflows change.
- Documentation synchronization is part of definition of done, not deferred
  cleanup.

## Architecture and security

- ModelProvider, ContextProvider, Agent, and Tool Bus boundaries are defined in
  `docs/specifications/ai-architecture.md` (bitty-ai-docs).
- Treat terminal content, plugin data, IPC/MCP clients, and model responses as
  untrusted across every boundary.
- P0 security controls are release blockers. Never add ambient authority,
  unbounded parser/resource path, or allow-all capability.
- Privacy-first: minimization, typed redaction, per-scope consent, and
  prohibition of self-acceptance.
- Agent, MCP, model streaming, context assembly, tool dispatch, secret handling,
  and consent ledger changes require focused security review.

## Performance and verification

- Performance claims require reproducible benchmarks, named workloads, context,
  baselines, and accepted budgets (e.g., Context Budget 32 KiB, RC-9/RC-10
  sharing).
- Bound queues, payloads, decoded resources, memory, and execution.
- Run checks proportionate to the change: formatting, linting, tests, platform
  checks, security gates, and documentation validation.

## Workspace hygiene

- Run Git and CarryCtx inside this repository, never at the umbrella root.
- Ephemeral scratch under `/tmp/bitty/`; durable material under this repository's
  `recording/` (gitignored). References belong
  under `recording/references/` and remain untrusted, read-only evidence.

## Handoff

- Report changed files, exact verification evidence, unresolved risks, and
  remaining work.
- Implementers request review rather than completing their own task. The
  independent reviewer records findings or acceptance before the commander
  completes it.

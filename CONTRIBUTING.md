# Contributing

Thank you for your interest in contributing to Bitty AI. This repository is in
a pre-implementation bootstrap phase: it contains a placeholder Rust workspace
scaffold and quality gates, not product behavior. Canonical product,
architecture, security, and project documentation lives in the
[bitty-docs](https://github.com/bitty-terminal/bitty-docs) repository.

Start by reading [AGENTS.md](AGENTS.md). It defines the governance model,
CarryCtx workflow, delivery lifecycle, and constraints that every contributor
and agent must follow.

## Prerequisites

- Rust — the exact channel is pinned in `rust-toolchain.toml`; `rustup`
  resolves and installs it automatically (includes `rustfmt` and Clippy)
- `just` — command runner for the quality gates
- `actionlint` — GitHub Actions workflow linting, invoked by `just actionlint`
- `lefthook` — Git hook manager, installed into the repository hooks by
  `just setup`
- `bun` — runs the pinned JavaScript dev tools (commitlint,
  markdownlint-cli2) through `bunx --bun`; no `npm`, `npx`, or `yarn`

## Setup

Clone the repository and run:

```bash
just setup
```

This fetches Cargo dependencies, installs the Lefthook Git hooks
(pre-commit format/lint/markdown gates, commit-msg Conventional Commits
check, pre-push build check), and provisions the pinned JS dev tools under
`target/dev-tools`.

## Development loop

All checks run through the justfile:

```bash
just check              # fmt-check + clippy + test + actionlint + markdownlint
just fmt-check          # cargo fmt --all -- --check
just clippy             # cargo clippy --workspace --all-targets --locked -- -D warnings
just test               # cargo test --workspace --all-targets --locked
just markdownlint       # markdownlint-cli2 over the repository Markdown
```

Run `just check` before requesting review. Formatting, lints, tests, and
workflow validation must pass without warnings.

## Delivery lifecycle

Changes follow the standard lifecycle: GitHub Issue, CarryCtx task, branch or
worktree, coherent commits, pull request, independent review plus CI, merge,
then Issue closure. Until the repository's first commit exists, branch, commit,
pull request, and merge stages are unavailable; scoped shared-checkout work is
the only authorized exception. See [AGENTS.md](AGENTS.md) for the normative
rules.

### Branch and worktree naming

Use uniform names across all Bitty repositories:

- Branches follow `ctx-XXXX/<type>-<short-slug>` where `XXXX` is the owning
  CarryCtx task number, `<type>` is one of `feat|fix|chore|docs`, and the slug
  is short kebab-case (for example `ctx-0001/feat-model-registry`).
- CarryCtx-bound worktrees live at `.worktrees/ctx-XXXX-<type>-<short-slug>`
  with `/` mapped to `-`.
- Use one branch per task; commander housekeeping branches may use `cmd/<slug>`.

## Committing

Use Conventional Commits:

```text
feat(model): add model registry skeleton
fix(ci): correct supply-chain job dependencies
docs(readme): clarify scaffold scope
```

A Conventional Commits configuration is provided in `commitlint.config.ts`.
After `just setup`, the commit-msg hook enforces it locally through
`just commit-check`; the pre-commit hooks run the same Rust and Markdown
gates as `just check`.

## Scope expectations

- Do not add product code, dependencies, or configuration unless an explicitly
  scoped task authorizes it.
- Documentation synchronization in `bitty-docs` is part of definition of done.
- Never describe scaffolding or plans as implemented behavior.

## Reporting issues

Open a GitHub Issue describing the problem, expected behavior, and evidence.
Security vulnerabilities must not be reported through public issues; see
[SECURITY.md](SECURITY.md).

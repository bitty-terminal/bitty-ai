# Bitty AI

Bitty AI is the AI subsystem for the Bitty terminal platform. This repository
contains the ModelProvider, ContextProvider, Agent runtime, and Tool Bus
components that enable terminal-native AI assistance.

**Status**: Pre-implementation. This repository is currently in the
documentation-first phase. Canonical architecture and design decisions are
maintained in
[bitty-ai-docs](https://github.com/bitty-terminal/bitty-ai-docs), mounted at
`docs/` as a Git submodule; shared governance and the security corpus live in
[bitty-docs](https://github.com/bitty-terminal/bitty-docs).

## Architecture

The AI subsystem is organized into four core components:

- **ModelProvider**: Model registry, capability negotiation, streaming inference, and budget tracking
- **ContextProvider**: Stable ID system, semantic zones, budget assembly, and provider implementations
- **Agent**: Multi-level agent runtime (AgentWorkspace L0/L1/L2), sessions, and consent ledger
- **Tool Bus**: MCP adapter, tool registry, capability gating, and secure dispatch

## Repository structure

This workspace will contain:

```text
crates/
  bitty-ai-model-provider/     # Model abstraction and registry
  bitty-ai-context-provider/   # Context system and providers
  bitty-ai-agent/              # Agent runtime and levels
  bitty-ai-tool-bus/           # Tool discovery and execution
  bitty-ai-workspace/          # AgentWorkspace and coordination
```

## Development

- Rust edition 2024, resolver 3, MSRV 1.85
- Toolchain: 1.98.1 (pinned in `rust-toolchain.toml`)
- Quality gates: `just check` (fmt, clippy, tests, actionlint, markdownlint)
- Git hooks: managed by lefthook (`lefthook install`)
- Lifecycle: managed by CarryCtx (state in `.git/carryctx/`)
- Docs submodule: `git submodule update --init` (or clone with
  `--recurse-submodules`); bump with `git submodule update --remote docs`

## Documentation

Canonical architecture, specifications, and cross-cutting design documentation
live in the [bitty-ai-docs](https://github.com/bitty-terminal/bitty-ai-docs)
repository, mounted at `docs/` as a Git submodule pinned by commit:

- New clone: `git clone --recurse-submodules …` (or
  `git submodule update --init` in an existing checkout).
- Bump the pin: `git submodule update --remote docs`, then `git add docs` and
  commit the pointer change.
- Read: [`docs/README.md`](https://github.com/bitty-terminal/bitty-ai-docs/blob/main/docs/README.md)
  is the documentation map.

Shared governance and the security corpus live in
[bitty-docs](https://github.com/bitty-terminal/bitty-docs). This repository
contains only implementation-specific documentation (API docs, crate README
files, and inline code documentation).

## License

MIT

# Bitty AI

Bitty AI is the AI subsystem for the Bitty terminal platform. This repository
contains the ModelProvider, ContextProvider, Agent runtime, and Tool Bus
components that enable terminal-native AI assistance.

**Status**: Pre-implementation. This repository is currently in the
documentation-first phase. Canonical architecture, security requirements, and
design decisions are maintained in
[bitty-docs](https://github.com/bitty-terminal/bitty-docs) under the `ai/`
directory to ensure cross-repository consistency.

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

## Documentation

All architectural decisions, security requirements, and cross-cutting design
documentation live in the
[bitty-docs repository](https://github.com/bitty-terminal/bitty-docs) under
`docs/ai/` to prevent duplication and drift across repositories.

This repository contains only implementation-specific documentation (API docs,
crate README files, and inline code documentation).

## License

MIT

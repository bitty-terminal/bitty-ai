setup:
    cargo fetch
    lefthook install
    just tools

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings

test:
    cargo test --workspace --all-targets --locked

typecheck:
    cargo check --workspace --all-targets --locked

actionlint:
    actionlint -color

markdownlint *args:
    bunx --bun markdownlint-cli2@0.23.1 {{args}}

tools:
    #!/usr/bin/env bash
    set -euo pipefail
    pins="commitlint@21.2.2 @commitlint/config-conventional@21.2.2"
    dir="target/dev-tools"
    stamp="$dir/node_modules/.pins"
    if [[ "$(cat "$stamp" 2>/dev/null)" == "$pins" ]]; then
        exit 0
    fi
    mkdir -p "$dir"
    cd "$dir"
    if [[ ! -f package.json ]]; then
        printf '{"name":"bitty-ai-dev-tools","private":true}\n' > package.json
    fi
    bun add $pins
    printf '%s\n' "$pins" > node_modules/.pins

commit-check message:
    @just tools
    @cp commitlint.config.ts target/dev-tools/commitlint.config.ts
    @msg="$(realpath "{{message}}")" && cd target/dev-tools && bunx --bun commitlint --edit "$msg"

check: fmt-check clippy test actionlint markdownlint

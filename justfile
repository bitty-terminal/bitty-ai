# Pinned gitleaks release. Single source of truth for both `just secrets` and
# the `Secret scan` CI job (.github/workflows/ci.yml), which parses this line
# and checksum-verifies the downloaded artifact against the release checksums.
gitleaks_version := "8.30.1"

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

# Scan committed Git history for secrets (gitleaks pinned via `gitleaks_version`
# above; see .gitleaks.toml). Uncommitted or untracked working-tree content is
# only covered once it is staged and committed; `secrets` is deliberately not
# part of `check` so contributors without gitleaks still pass the default gate.
# `secrets` runs whatever `gitleaks` is on PATH; CI installs the pinned release
# and checksum-verifies it in the `Secret scan` job.
secrets:
    gitleaks detect --source . --no-banner

# Report the pinned gitleaks version and the version the local binary reports.
# A binary built without ldflags version metadata prints "version is set by
# build process"; that is reported, not treated as an error.
gitleaks-version:
    #!/usr/bin/env bash
    set -euo pipefail
    printf 'pinned: %s\n' "{{gitleaks_version}}"
    if command -v gitleaks >/dev/null 2>&1; then
        printf 'local:  %s\n' "$(gitleaks version 2>&1 || true)"
    else
        printf 'local:  (gitleaks not on PATH)\n'
    fi

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

# Publish a redacted CarryCtx snapshot inside this repo (commander merge
# closeout only; never a git hook). `carryctx export --publication` redacts the
# bundle, stamps manifest.redacted, and commits one snapshot to the fixed ref
# `refs/heads/carryctx-snapshots`; the target pushes that branch only when the
# local ref advanced. Canonical closeout runs from the primary checkout on
# branch main (`cd "$BITTY_WORKSPACE/bitty-ai" && just workflow-publish`); a
# detached or feature worktree records that branch as the snapshot source. Dry
# run validates the export and writes neither the ref nor the remote.
workflow-publish *args:
    bash scripts/workflow-publish.sh {{args}}

workflow-publish-dry *args:
    bash scripts/workflow-publish.sh --dry-run {{args}}

# Restore the local CarryCtx DB from the in-repo snapshot branch
# `refs/heads/carryctx-snapshots` (fresh-clone recipe). Refuses to replace a
# non-empty local DB without --force, e.g. `just workflow-import --force`.
workflow-import *args:
    bash scripts/workflow-import.sh {{args}}

workflow-import-dry *args:
    bash scripts/workflow-import.sh --dry-run {{args}}

# Detect `bitty-ipc` pin drift against the mirrored FakeHost surface.
# Local-only by default (no network); `--remote` opts into fetching
# `origin/main` from the `bitty` checkout. Deliberately NOT part of
# `just check`, which must keep zero network dependency.
pin-drift *args:
    bash scripts/pin-drift.sh {{args}}

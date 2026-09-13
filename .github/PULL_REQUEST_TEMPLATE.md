<!-- Conventional Commit title: feat(scope): description -->

## What & why

<!-- What does this change and why? Link the issue: Closes #123 -->

## Changes

-

## Testing

- [ ] `cargo fmt --all -- --check` passes (`just fmt-check`)
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings` is clean (`just clippy`)
- [ ] `cargo test --workspace --all-targets --locked` passes (`just test`)
- [ ] `actionlint -color` passes (`just actionlint`)

## Checklist

- [ ] Public API contracts unchanged (or documented in the PR description)
- [ ] No secrets or local paths committed
- [ ] Canonical `bitty-docs` material synchronized (or explicitly not needed)

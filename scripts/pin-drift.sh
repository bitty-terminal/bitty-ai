#!/usr/bin/env bash
# pin-drift.sh — detect `bitty-ipc` pin drift against the mirrored FakeHost.
#
# The slice pin-upgrades `bitty-ipc` by Git revision
# (`crates/bitty-ai-slice/Cargo.toml`) while `FakeHost` hand-mirrors upstream
# DTO/validate/bound bytes and dispatch order. Nothing follows the bump
# automatically, so a `bitty` change drifts silently until
# `host_conformance` fails. This script is the low-cost, deterministic signal:
# given the pinned revision, it reports whether the locally available `bitty`
# checkout has advanced past that pin and which upstream hand-mirrored files
# changed in the window.
#
# It is a local-only check by default: no fetch, no network, no writes. The
# `--remote` mode is the only network path (fetch `origin/main` from `bitty`)
# and is opt-in. This script is intentionally NOT wired into `just check`, so
# the normal gate keeps zero network dependency; run it via `just pin-drift`.
#
# Usage:
#   scripts/pin-drift.sh [--bitty DIR] [--ref REV] [--remote] [--cargo FILE]
#                        [--log] [--help]
#   --bitty DIR   `bitty` checkout to inspect. Default: $BITTY_WORKSPACE/bitty.
#   --ref REV     local revision to compare against (default: HEAD).
#   --remote      fetch `origin/main` from the `bitty` remote first and compare
#                 against it instead of a local revision (network required).
#   --cargo FILE  Cargo.toml carrying the pin (default: the slice manifest).
#   --log         also print the commit subjects in the drift window.
#
# Exit codes:
#   0  in sync — the compared revision is exactly the pinned revision.
#   1  drift — the compared revision advanced past the pin (or diverged).
#   2  environment/usage error — checkout absent, pin unparseable, bad args.
#
# Bump checklist (run after changing the pin in the slice `Cargo.toml`):
#   1. Re-check the dispatch prefix order and denial classes mirrored as
#      explicit steps in `crates/bitty-ai-slice/src/fake_host.rs` against the
#      new upstream `tool_dispatch.rs`, `execution.rs`, and `snapshot.rs`.
#   2. Re-check the `MAX_*` equality tests — `MAX_WIRE_CLIENT_ID_BYTES` must
#      equal `bitty_ipc::auth::MAX_SCOPED_ID_BYTES`, `MAX_TOOL_CLIENT_ID_BYTES`,
#      and `MAX_EXEC_CLIENT_ID_BYTES` (crates/bitty-ai-slice/tests/
#      client_id_binding.rs) — plus the derived bounds used by the FakeHost.
#   3. Refresh the mirrored line-number references in the `fake_host.rs` module
#      header, which cites the pinned inspection point (`cfeffa2`) by line.
#   4. Re-run `cargo test -p bitty-ai-slice`, especially `host_conformance` and
#      `client_id_binding`, then `just check`.
#   5. Bump `rev` in `crates/bitty-ai-slice/Cargo.toml`, refresh the lockfile
#      (`cargo update -p bitty-ipc`), and update the pin rev cited in
#      `crates/bitty-ai-slice/README.md` and `fake_host.rs`/`bridge.rs` docs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO_TOML="$REPO_ROOT/crates/bitty-ai-slice/Cargo.toml"
BITTY_DIR="${BITTY_WORKSPACE:+$BITTY_WORKSPACE/bitty}"
REF="HEAD"
REMOTE=0
SHOW_LOG=0

# Hand-mirrored upstream files under `crates/bitty-ipc/src` (FakeHost surface).
MIRROR_SURFACE=(
	tool_dispatch.rs
	execution.rs
	snapshot.rs
	scope.rs
	host_bridge.rs
	wire.rs
	rich_fragment.rs
	auth.rs
	channel.rs
	error.rs
	lib.rs
)

usage() {
	sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'
}

die() {
	printf 'pin-drift: %s\n' "$*" >&2
	exit 2
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--bitty)
		BITTY_DIR="${2:?--bitty requires a directory}"
		shift 2
		;;
	--bitty=*)
		BITTY_DIR="${1#--bitty=}"
		shift
		;;
	--ref)
		REF="${2:?--ref requires a revision}"
		shift 2
		;;
	--ref=*)
		REF="${1#--ref=}"
		shift
		;;
	--cargo)
		CARGO_TOML="${2:?--cargo requires a file}"
		shift 2
		;;
	--cargo=*)
		CARGO_TOML="${1#--cargo=}"
		shift
		;;
	--remote)
		REMOTE=1
		shift
		;;
	--log)
		SHOW_LOG=1
		shift
		;;
	--help | -h)
		usage
		exit 0
		;;
	*)
		die "unknown argument: $1 (try --help)"
		;;
	esac
done

[[ -n "$BITTY_DIR" ]] || die "no bitty checkout: set \$BITTY_WORKSPACE or pass --bitty DIR"
[[ -f "$CARGO_TOML" ]] || die "manifest not found: $CARGO_TOML"

PIN="$(
	sed -nE 's/^[[:space:]]*bitty-ipc[[:space:]]*=.*rev[[:space:]]*=[[:space:]]*"([0-9a-f]{7,40})".*/\1/p' \
		"$CARGO_TOML" | head -n 1
)"
[[ -n "$PIN" ]] || die "could not parse the bitty-ipc rev from $CARGO_TOML"

git -C "$BITTY_DIR" rev-parse --git-dir >/dev/null 2>&1 ||
	die "not a git checkout: $BITTY_DIR"

if [[ "$REMOTE" -eq 1 ]]; then
	git -C "$BITTY_DIR" fetch --quiet origin main ||
		die "failed to fetch origin/main from $BITTY_DIR (network required)"
	TARGET_REF="refs/remotes/origin/main"
	TARGET_LABEL="origin/main"
else
	TARGET_REF="$REF"
	TARGET_LABEL="$REF"
fi

TARGET="$(git -C "$BITTY_DIR" rev-parse --verify --quiet "${TARGET_REF}^{commit}")" ||
	die "cannot resolve revision '$TARGET_REF' in $BITTY_DIR"
PIN_FULL="$(git -C "$BITTY_DIR" rev-parse --verify --quiet "${PIN}^{commit}")" ||
	die "pinned revision $PIN is not present in $BITTY_DIR (fetch the repo first)"

printf 'pin:    %s (%s)\n' "$PIN" "${CARGO_TOML#"$REPO_ROOT"/}"
printf 'bitty:  %s @ %s (%s)\n' "$BITTY_DIR" "${TARGET:0:7}" "$TARGET_LABEL"

if [[ "$TARGET" == "$PIN_FULL" ]]; then
	printf 'status: IN SYNC\n'
	exit 0
fi

if git -C "$BITTY_DIR" merge-base --is-ancestor "$PIN_FULL" "$TARGET"; then
	AHEAD="$(git -C "$BITTY_DIR" rev-list --count "$PIN_FULL..$TARGET")"
	printf 'status: ADVANCED by %s commit(s)\n' "$AHEAD"
else
	printf 'status: DIVERGED — %s does not contain the pinned revision\n' "$TARGET_LABEL"
fi

if [[ "$SHOW_LOG" -eq 1 ]]; then
	printf '\ncommits %s..%s:\n' "${PIN:0:7}" "${TARGET:0:7}"
	git -C "$BITTY_DIR" log --oneline "$PIN_FULL..$TARGET" | sed 's/^/  /'
fi

printf '\nchanged files under crates/bitty-ipc/src (%s..%s):\n' "${PIN:0:7}" "${TARGET:0:7}"
CHANGED="$(git -C "$BITTY_DIR" diff --name-only "$PIN_FULL..$TARGET" -- crates/bitty-ipc/src || true)"
if [[ -z "$CHANGED" ]]; then
	printf '  (none)\n'
	printf '\nno crates/bitty-ipc/src changes in the window; the pin can advance as-is.\n'
else
	MIRRORED=0
	while IFS= read -r file; do
		base="${file##*/}"
		match="other"
		for m in "${MIRROR_SURFACE[@]}"; do
			if [[ "$base" == "$m" ]]; then
				match="MIRRORED"
				MIRRORED=1
				break
			fi
		done
		printf '  %-8s %s\n' "$match" "$file"
	done <<<"$CHANGED"
	if [[ "$MIRRORED" -eq 1 ]]; then
		printf '\nmirrored surface changed — run the bump checklist in this file header.\n'
	else
		printf '\nno hand-mirrored surface file changed; still review the other ipc\nchanges before bumping the pin.\n'
	fi
fi

exit 1

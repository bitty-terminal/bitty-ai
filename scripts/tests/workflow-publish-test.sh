#!/usr/bin/env bash
# workflow-publish-test.sh — regression suite for `scripts/workflow-publish.sh`.
#
# Every case builds a throwaway bare "remote" plus a clone whose directory name
# matches it (the script's provenance name check), and prepends a `carryctx`
# stub that commits a chosen `manifest.json` to `refs/heads/carryctx-snapshots`.
# The remote is a local bare repository: the suite performs no network access
# and never touches a real remote.
#
# Cases lock the two hardening findings:
#   - source ancestry is verified for branch `main` too. A local main that is
#     ahead of or diverged from a freshly fetched origin/main must fail before
#     the export, while detached/behind checkouts stay valid and
#     `--allow-non-main` remains the explicit bypass;
#   - the manifest guard parses JSON and requires the TOP-LEVEL `redacted`
#     boolean. A nested `redacted: true`, a string, a number, or a truncated
#     document must not authorize a push.
#
# Usage: scripts/tests/workflow-publish-test.sh
set -euo pipefail

SCRIPT="${WORKFLOW_PUBLISH_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/workflow-publish.sh}"
SCRATCH_ROOT="${TMPDIR:-/tmp/bitty}"
mkdir -p -- "$SCRATCH_ROOT"
TEST_ROOT="$(mktemp -d "$SCRATCH_ROOT/workflow-publish-test.XXXXXX")"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

# Keep the fixtures hermetic: no user/system git config, no prompt, fixed
# identity for the stub's commit-tree calls.
export GIT_AUTHOR_NAME="workflow-publish-test"
export GIT_AUTHOR_EMAIL="workflow-publish-test@example.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_TERMINAL_PROMPT=0

FAILURES=0
CASE_COUNT=0
RUN_OUTPUT=""
RUN_STATUS=0

# ── carryctx stub ───────────────────────────────────────────────────────────
# Mirrors the one observable effect the publish script relies on: the export
# commits `manifest.json` to the publication ref. STUB_MANIFEST selects the
# content so each case can exercise the redaction guard.
STUB_BIN="$TEST_ROOT/bin"
mkdir -p -- "$STUB_BIN"
cat >"$STUB_BIN/carryctx" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'carryctx %s\n' "$*" >>"${STUB_CALL_LOG:-/dev/null}"
[[ "${1:-}" == "export" ]] || {
	echo "carryctx stub: unsupported subcommand: $*" >&2
	exit 64
}
project=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	export) shift ;;
	--pack-format) shift 2 ;;
	--pack-format=*) shift ;;
	-o) shift 2 ;;
	--publication | --dry-run) shift ;;
	--project) project="${2:?carryctx stub: --project needs a value}" && shift 2 ;;
	--project=*) project="${1#--project=}" && shift ;;
	*) shift ;;
	esac
done
[[ -n "$project" ]] || {
	echo "carryctx stub: missing --project" >&2
	exit 64
}
manifest="${STUB_MANIFEST:?carryctx stub: STUB_MANIFEST is required}"
blob="$(git -C "$project" hash-object -w -- "$manifest")"
tree="$(printf '100644 blob %s\tmanifest.json\n' "$blob" | git -C "$project" mktree)"
commit_args=(-m "CarryCtx-Export-Id: stub")
if git -C "$project" rev-parse -q --verify refs/heads/carryctx-snapshots >/dev/null 2>&1; then
	commit_args+=(-p "$(git -C "$project" rev-parse refs/heads/carryctx-snapshots)")
fi
commit="$(git -C "$project" commit-tree "$tree" "${commit_args[@]}")"
git -C "$project" update-ref refs/heads/carryctx-snapshots "$commit"
printf '{"exported":true}\n'
STUB
chmod +x "$STUB_BIN/carryctx"

# ── fixtures and helpers ────────────────────────────────────────────────────
new_fixture() {
	local name="$1"
	local root="$TEST_ROOT/$name" seed="$TEST_ROOT/$name-seed"
	mkdir -p -- "$root"
	git init --quiet --bare --initial-branch=main "$root/remote.git"
	git init --quiet --initial-branch=main "$seed"
	printf 'seed\n' >"$seed/README.md"
	git -C "$seed" add README.md
	git -C "$seed" commit --quiet -m seed
	git -C "$seed" push --quiet "$root/remote.git" main
	git clone --quiet "$root/remote.git" "$root/remote"
	printf '%s\n' "$root"
}

checkout_of() { printf '%s/remote\n' "$1"; }
bare_of() { printf '%s/remote.git\n' "$1"; }

# Writes the manifest the stub will commit and prints its path.
write_manifest() {
	local root="$1" json="$2" file="$1/manifest.json"
	printf '%s\n' "$json" >"$file"
	printf '%s\n' "$file"
}

run_publish() {
	local root="$1" manifest="$2"
	shift 2
	STUB_MANIFEST="$manifest" STUB_CALL_LOG="$root/carryctx-calls.log" \
		PATH="$STUB_BIN:$PATH" \
		bash "$SCRIPT" --project "$(checkout_of "$root")" "$@"
}

# Sets RUN_STATUS and RUN_OUTPUT without tripping `set -e`.
capture() {
	RUN_OUTPUT="$("$@" 2>&1)" && RUN_STATUS=0 || RUN_STATUS=$?
}

remote_has_snapshot() {
	git -C "$(bare_of "$1")" rev-parse -q --verify refs/heads/carryctx-snapshots >/dev/null 2>&1
}

export_recorded() {
	grep -q '^carryctx export' "$1/carryctx-calls.log" 2>/dev/null
}

report_failure() {
	local name="$1"
	shift
	printf 'not ok %d - %s\n' "$CASE_COUNT" "$name"
	local line
	for line in "$@"; do
		printf '    %s\n' "$line"
	done
	FAILURES=$((FAILURES + 1))
}

expect_status() {
	local name="$1" expected="$2"
	CASE_COUNT=$((CASE_COUNT + 1))
	if [[ "$RUN_STATUS" -eq "$expected" ]]; then
		printf 'ok %d - %s (exit %d)\n' "$CASE_COUNT" "$name" "$RUN_STATUS"
	else
		report_failure "$name" "expected exit $expected, got $RUN_STATUS" "$RUN_OUTPUT"
	fi
}

expect_ok() {
	local name="$1"
	shift
	CASE_COUNT=$((CASE_COUNT + 1))
	if "$@"; then
		printf 'ok %d - %s\n' "$CASE_COUNT" "$name"
	else
		report_failure "$name" "predicate failed"
	fi
}

expect_not() {
	local name="$1"
	shift
	CASE_COUNT=$((CASE_COUNT + 1))
	if "$@"; then
		report_failure "$name" "predicate unexpectedly succeeded"
	else
		printf 'ok %d - %s\n' "$CASE_COUNT" "$name"
	fi
}

output_has() {
	grep -Fq -- "$2" <<<"$1"
}

# ── accepted publications ───────────────────────────────────────────────────
root="$(new_fixture main-in-sync)"
manifest="$(write_manifest "$root" '{"redacted": true, "project": "stub"}')"
capture run_publish "$root" "$manifest"
expect_status "in-sync main with top-level redacted:true publishes" 0
expect_ok "export runs on accepted source" export_recorded "$root"
expect_ok "snapshot branch reaches the remote" remote_has_snapshot "$root"

root="$(new_fixture detached-at-main)"
detached="$(checkout_of "$root")"
git -C "$detached" checkout --quiet --detach refs/remotes/origin/main
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "detached checkout at origin/main stays publishable" 0
expect_ok "detached publication pushes the snapshot" remote_has_snapshot "$root"

root="$(new_fixture behind-main)"
seed="$TEST_ROOT/behind-main-seed"
printf 'next\n' >>"$seed/README.md"
git -C "$seed" commit --quiet -am "advance main"
git -C "$seed" push --quiet "$(bare_of "$root")" main
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "local main behind freshly fetched origin/main publishes with a warning" 0
expect_ok "behind checkout reports the stale source" output_has "$RUN_OUTPUT" "behind"
expect_ok "behind publication pushes the snapshot" remote_has_snapshot "$root"

# ── source ancestry rejections ──────────────────────────────────────────────
root="$(new_fixture diverged-local-main)"
diverged="$(checkout_of "$root")"
git -C "$diverged" commit --quiet --allow-empty -m "local-only main commit"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "divergent local main is rejected before export" 1
expect_ok "rejection names the divergent local main" output_has "$RUN_OUTPUT" "local main"
expect_not "rejected source never runs the export" export_recorded "$root"
expect_not "rejected source never pushes a snapshot" remote_has_snapshot "$root"

root="$(new_fixture feature-branch)"
feature="$(checkout_of "$root")"
git -C "$feature" checkout --quiet -b feature
git -C "$feature" commit --quiet --allow-empty -m "feature commit"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "feature branch is rejected" 1
expect_not "feature branch never pushes a snapshot" remote_has_snapshot "$root"

# ── --allow-non-main bypass ─────────────────────────────────────────────────
root="$(new_fixture allow-non-main)"
override="$(checkout_of "$root")"
git -C "$override" commit --quiet --allow-empty -m "unpublished closeout commit"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest" --allow-non-main
expect_status "--allow-non-main publishes a divergent checkout" 0
expect_ok "--allow-non-main keeps its warning" output_has "$RUN_OUTPUT" "--allow-non-main set"
expect_ok "bypassed publication pushes the snapshot" remote_has_snapshot "$root"

root="$(new_fixture unverifiable-main)"
git -C "$(bare_of "$root")" update-ref -d refs/heads/main
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "unverifiable origin/main fails closed" 1
expect_ok "unverifiable source names the remote main ref" output_has "$RUN_OUTPUT" "cannot determine"
expect_not "unverifiable source never exports" export_recorded "$root"
expect_not "unverifiable source never pushes" remote_has_snapshot "$root"

root="$(new_fixture allow-non-main-unverifiable)"
git -C "$(bare_of "$root")" update-ref -d refs/heads/main
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest" --allow-non-main
expect_status "--allow-non-main bypasses an unverifiable origin/main" 0
expect_ok "bypass reports skipped source verification" output_has "$RUN_OUTPUT" "skipping source verification"
expect_ok "bypassed publication pushes despite unverifiable main" remote_has_snapshot "$root"

# ── manifest redaction rejections ───────────────────────────────────────────
root="$(new_fixture nested-redacted)"
manifest="$(write_manifest "$root" '{"schema": 1, "nested": {"redacted": true}}')"
capture run_publish "$root" "$manifest"
expect_status "nested redacted:true does not satisfy the guard" 1
expect_ok "nested rejection names the top-level requirement" output_has "$RUN_OUTPUT" "top-level"
expect_not "nested redacted:true never pushes" remote_has_snapshot "$root"

root="$(new_fixture string-redacted)"
manifest="$(write_manifest "$root" '{"redacted": "true"}')"
capture run_publish "$root" "$manifest"
expect_status "string redacted value is rejected" 1
expect_not "string redacted value never pushes" remote_has_snapshot "$root"

root="$(new_fixture numeric-redacted)"
manifest="$(write_manifest "$root" '{"redacted": 1}')"
capture run_publish "$root" "$manifest"
expect_status "numeric redacted value is rejected" 1
expect_not "numeric redacted value never pushes" remote_has_snapshot "$root"

root="$(new_fixture malformed-manifest)"
manifest="$(write_manifest "$root" '{"redacted": true')"
capture run_publish "$root" "$manifest"
expect_status "malformed manifest JSON fails closed" 1
expect_ok "malformed manifest reports a parse error" output_has "$RUN_OUTPUT" "not readable JSON"
expect_not "malformed manifest never pushes" remote_has_snapshot "$root"

printf 'workflow-publish-test: %d cases, %d failures\n' "$CASE_COUNT" "$FAILURES"
((FAILURES == 0))

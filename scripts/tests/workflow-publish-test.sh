#!/usr/bin/env bash
# workflow-publish-test.sh — regression suite for `scripts/workflow-publish.sh`.
#
# Every case builds a throwaway bare "remote" plus a clone whose directory name
# matches it (the script's provenance name check), and prepends a `carryctx`
# stub that commits a chosen `manifest.json` to `refs/heads/carryctx-snapshots`.
# The remote is a local bare repository: the suite performs no network access
# and never touches a real remote.
#
# Cases lock the hardening findings:
#   - source ancestry is verified for branch `main` too. A local main that is
#     ahead of or diverged from a freshly fetched origin/main must fail before
#     the export, while detached/behind checkouts stay valid and
#     `--allow-non-main` remains the explicit bypass;
#   - the named remote must resolve to exactly one URL for both fetch and
#     push. An explicit pushurl identical to the fetch URL publishes; an empty
#     `remote.<name>.url` or `remote.<name>.pushurl` record (which
#     `git remote get-url` can hide next to a valid URL), a split pushurl
#     (reachable or not), a pushurl that differs from the fetch URL only by
#     trailing newline bytes, a `pushInsteadOf` rewrite, more than one push
#     URL, and an unresolvable `--remote` (a URL or path, which would skip the
#     comparison) are refused before the export with a static,
#     credential-free diagnostic, and `--allow-non-main` cannot bypass that
#     refusal;
#   - the manifest guard parses JSON and requires the TOP-LEVEL `redacted`
#     boolean. A nested `redacted: true`, a string, a number, a boolean false,
#     or a truncated document must not authorize a push, and diagnostics must
#     report the JSON type without leaking field contents;
#   - the validated commit is pinned for both the manifest read and the push:
#     a probe moves the publication ref to a decoy immediately before push and
#     asserts the remote receives the validated commit;
#   - dry runs forward `--publication`, write no publication ref, and push
#     nothing.
#
# Usage: scripts/tests/workflow-publish-test.sh
set -euo pipefail

SCRIPT="${WORKFLOW_PUBLISH_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/workflow-publish.sh}"
SCRATCH_ROOT="${TMPDIR:-/tmp/bitty}"
mkdir -p -- "$SCRATCH_ROOT"
TEST_ROOT="$(mktemp -d "$SCRATCH_ROOT/workflow-publish-test.XXXXXX")"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

# Hook-style environments (post-commit, pre-push, lefthook, CI wrappers) export
# repository selectors that would redirect every `git -C` call below at the
# parent repository instead of the throwaway fixtures. Drop them first.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE

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
# Mirrors the observable effects the publish script relies on: `--publication`
# must be requested, a real export commits `manifest.json` to the publication
# ref, and a `--dry-run` export writes nothing. STUB_MANIFEST selects the
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
publication=0
dry_run=0
while [[ $# -gt 0 ]]; do
	case "$1" in
	export) shift ;;
	--pack-format) shift 2 ;;
	--pack-format=*) shift ;;
	-o) shift 2 ;;
	--publication) publication=1 && shift ;;
	--dry-run) dry_run=1 && shift ;;
	--project) project="${2:?carryctx stub: --project needs a value}" && shift 2 ;;
	--project=*) project="${1#--project=}" && shift ;;
	*) shift ;;
	esac
done
[[ -n "$project" ]] || {
	echo "carryctx stub: missing --project" >&2
	exit 64
}
# Lock the script's contract: the publisher must request the redacted
# publication export, never the plain bundle export.
[[ "$publication" == 1 ]] || {
	echo "carryctx stub: --publication was not passed" >&2
	exit 65
}
manifest="${STUB_MANIFEST:?carryctx stub: STUB_MANIFEST is required}"
if [[ "$dry_run" == 1 ]]; then
	# Real `carryctx export --publication --dry-run` validates and writes no ref.
	printf '{"exported":false,"dry_run":true}\n'
	exit 0
fi
blob="$(git -C "$project" hash-object -w -- "$manifest")"
tree="$(printf '100644 blob %s\tmanifest.json\n' "$blob" | git -C "$project" mktree)"
commit_args=(-m "CarryCtx-Export-Id: stub")
if git -C "$project" rev-parse -q --verify refs/heads/carryctx-snapshots >/dev/null 2>&1; then
	commit_args+=(-p "$(git -C "$project" rev-parse refs/heads/carryctx-snapshots)")
fi
commit="$(git -C "$project" commit-tree "$tree" "${commit_args[@]}")"
git -C "$project" update-ref refs/heads/carryctx-snapshots "$commit"
printf '%s\n' "$commit" >>"${STUB_COMMIT_LOG:-/dev/null}"
printf '{"exported":true}\n'
STUB
chmod +x "$STUB_BIN/carryctx"

# ── git wrapper for the TOCTOU probe ────────────────────────────────────────
# Immediately before the script's push, move the local publication ref to a
# decoy commit and delegate to the real git. A push by ref name would send the
# decoy; a push by the pinned validated SHA ignores it. The wrapper logs the
# push argv so the suite can assert the source is the validated commit, not a
# ref name.
REAL_GIT="$(type -P git)"
TOCTOU_BIN="$TEST_ROOT/bin-toctou"
mkdir -p -- "$TOCTOU_BIN"
cat >"$TOCTOU_BIN/git" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-C" && "${3:-}" == "push" ]]; then
	project="$2"
	decoy_manifest="${GIT_TOCTOU_DECOY:?GIT_TOCTOU_DECOY is required}"
	blob="$("${REAL_GIT:?REAL_GIT is required}" -C "$project" hash-object -w -- "$decoy_manifest")"
	tree="$(printf '100644 blob %s\tmanifest.json\n' "$blob" | "${REAL_GIT}" -C "$project" mktree)"
	decoy="$("${REAL_GIT}" -C "$project" commit-tree "$tree" -m "decoy snapshot")"
	"${REAL_GIT}" -C "$project" update-ref refs/heads/carryctx-snapshots "$decoy"
	printf '%s\n' "$decoy" >>"${GIT_TOCTOU_LOG:?GIT_TOCTOU_LOG is required}"
	printf '%s\n' "$*" >>"${GIT_PUSH_LOG:?GIT_PUSH_LOG is required}"
fi
exec "${REAL_GIT:?REAL_GIT is required}" "$@"
WRAPPER
chmod +x "$TOCTOU_BIN/git"

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

# Replaces `remote.origin.pushurl` with exactly the given URLs; no arguments
# removes the pushurl and restores the fetch URL as the push destination.
set_pushurls() {
	local root="$1"
	shift
	local checkout url
	checkout="$(checkout_of "$root")"
	git -C "$checkout" config --unset-all remote.origin.pushurl >/dev/null 2>&1 || true
	for url in "$@"; do
		git -C "$checkout" config --add remote.origin.pushurl "$url"
	done
}

path_has_snapshot() {
	git -C "$1" rev-parse -q --verify refs/heads/carryctx-snapshots >/dev/null 2>&1
}

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

# Same run, but with the TOCTOU probe on PATH: the wrapper moves the local
# publication ref to `decoy` immediately before the push and logs the push
# source plus the decoy SHA.
run_publish_toctou() {
	local root="$1" manifest="$2" decoy="$3"
	REAL_GIT="$REAL_GIT" GIT_TOCTOU_DECOY="$decoy" \
		GIT_TOCTOU_LOG="$root/toctou-decoy.log" GIT_PUSH_LOG="$root/git-push.log" \
		STUB_MANIFEST="$manifest" STUB_COMMIT_LOG="$root/stub-commit.log" \
		STUB_CALL_LOG="$root/carryctx-calls.log" \
		PATH="$TOCTOU_BIN:$STUB_BIN:$PATH" \
		bash "$SCRIPT" --project "$(checkout_of "$root")"
}

# Sets RUN_STATUS and RUN_OUTPUT without tripping `set -e`.
capture() {
	RUN_OUTPUT="$("$@" 2>&1)" && RUN_STATUS=0 || RUN_STATUS=$?
}

remote_has_snapshot() {
	git -C "$(bare_of "$1")" rev-parse -q --verify refs/heads/carryctx-snapshots >/dev/null 2>&1
}

local_ref_exists() {
	git -C "$(checkout_of "$1")" rev-parse -q --verify refs/heads/carryctx-snapshots >/dev/null 2>&1
}

local_ref_equals() {
	[[ "$(git -C "$(checkout_of "$1")" rev-parse -q --verify refs/heads/carryctx-snapshots 2>/dev/null)" == "$2" ]]
}

# The logged push line ends with `<src>:refs/heads/carryctx-snapshots`.
push_source_equals() {
	local log="$1" expected="$2" line src
	line="$(cat "$log")" || return 1
	src="${line##* }"
	src="${src%%:*}"
	[[ "$src" == "$expected" ]]
}

remote_manifest_matches() {
	local root="$1" commit="$2" file="$3"
	git -C "$(bare_of "$root")" show "$commit:manifest.json" 2>/dev/null | cmp -s - "$file"
}

export_recorded() {
	grep -q '^carryctx export' "$1/carryctx-calls.log" 2>/dev/null
}

export_requested_publication() {
	grep -Fq -- '--publication' "$1/carryctx-calls.log" 2>/dev/null
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
expect_ok "export requests the redacted publication" export_requested_publication "$root"
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

# ── dry run ─────────────────────────────────────────────────────────────────
root="$(new_fixture dry-run)"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest" --dry-run
expect_status "dry run validates the export without publishing" 0
expect_ok "dry run reports its PASS state" output_has "$RUN_OUTPUT" "dry-run PASS"
expect_ok "dry-run export still requests --publication" export_requested_publication "$root"
expect_not "dry run writes no local publication ref" local_ref_exists "$root"
expect_not "dry run pushes no snapshot" remote_has_snapshot "$root"

# ── validated commit is the pushed source (TOCTOU) ──────────────────────────
# The wrapper moves refs/heads/carryctx-snapshots to a decoy commit immediately
# before the push. Pushing the pinned validated commit keeps the remote clean;
# pushing the ref name would publish the decoy's non-redacted manifest.
root="$(new_fixture toctou-ref-move)"
manifest="$(write_manifest "$root" '{"redacted": true}')"
decoy_manifest="$root/decoy-manifest.json"
printf '%s\n' '{"redacted": false, "note": "decoy"}' >"$decoy_manifest"
capture run_publish_toctou "$root" "$manifest" "$decoy_manifest"
expect_status "ref moved between validation and push publishes the validated commit" 0
validated="$(cat "$root/stub-commit.log")"
decoy_sha="$(cat "$root/toctou-decoy.log")"
expect_ok "probe moved the local publication ref before push" local_ref_equals "$root" "$decoy_sha"
expect_ok "push source is the pinned validated commit" push_source_equals "$root/git-push.log" "$validated"
push_tip="$(git -C "$(bare_of "$root")" rev-parse refs/heads/carryctx-snapshots 2>/dev/null || true)"
expect_ok "remote tip is the validated commit" test "$push_tip" = "$validated"
expect_not "remote tip is not the decoy" test "$push_tip" = "$decoy_sha"
expect_ok "remote manifest is the validated redacted one" remote_manifest_matches "$root" "$push_tip" "$manifest"

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

# ── fetch/push destination guard ────────────────────────────────────────────
# The script fetches main from the remote's fetch URL but `git push` writes to
# its push URL(s). A split destination is refused before the export: publishing
# into a repository whose main was never verified misattributes provenance, and
# the snapshot-source gate runs in the repository that receives the snapshot.

# Git accepts a valid URL plus an empty `remote.<name>.url` record. On some
# versions `git remote get-url --all` hides the empty record, so the resolved
# count/equality guard would pass while fetch resolves the partial
# configuration. The raw record must be refused before resolution.
root="$(new_fixture empty-fetch-url-record)"
git -C "$(checkout_of "$root")" config --add remote.origin.url ""
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "empty fetch URL record is refused before export" 1
expect_ok "empty fetch URL record reports the empty record" output_has "$RUN_OUTPUT" "empty fetch URL record"
expect_not "empty fetch URL record does not echo URL values" output_has "$RUN_OUTPUT" "$root"
expect_not "empty fetch URL record never exports" export_recorded "$root"
expect_not "empty fetch URL record never pushes" remote_has_snapshot "$root"

# Same shape on the push side: a valid pushurl plus an empty record can resolve
# to only the valid line, while push still reads the partial configuration.
root="$(new_fixture empty-push-url-record)"
fetch_url="$(git -C "$(checkout_of "$root")" remote get-url origin)"
set_pushurls "$root" "$fetch_url" ""
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "empty push URL record is refused before export" 1
expect_ok "empty push URL record reports the empty record" output_has "$RUN_OUTPUT" "empty push URL record"
expect_not "empty push URL record does not echo URL values" output_has "$RUN_OUTPUT" "$root"
expect_not "empty push URL record never exports" export_recorded "$root"
expect_not "empty push URL record never pushes" remote_has_snapshot "$root"

# An explicit pushurl identical to the fetch URL is still one destination.
root="$(new_fixture identical-pushurl)"
set_pushurls "$root" "$(git -C "$(checkout_of "$root")" remote get-url origin)"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "pushurl identical to the fetch URL publishes" 0
expect_ok "identical pushurl reaches the remote" remote_has_snapshot "$root"

# A configured pushurl whose value ends in newline bytes is a different
# destination: `git push` resolves the raw value, so the newline is part of the
# path. Command substitution used to strip that byte, making the push record
# compare equal to the fetch URL even though the destinations differ. Capture
# must stay byte-exact and refuse this split before the export.
root="$(new_fixture newline-suffixed-pushurl)"
fetch_url="$(git -C "$(checkout_of "$root")" remote get-url origin)"
set_pushurls "$root" "$fetch_url"$'\n'
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "newline-suffixed pushurl is refused before export" 1
expect_ok "newline-suffixed pushurl reports the shared-URL requirement" output_has "$RUN_OUTPUT" "exactly one shared URL"
expect_not "newline-suffixed pushurl is not echoed" output_has "$RUN_OUTPUT" "$root"
expect_not "newline-suffixed pushurl never exports" export_recorded "$root"
expect_not "newline-suffixed pushurl never pushes to the fetch remote" remote_has_snapshot "$root"

# Control for the case above: the same fixture with the exact URL still
# publishes. The refusal comes from the extra newline byte, not from declaring
# an explicit pushurl at all.
root="$(new_fixture exact-pushurl-control)"
fetch_url="$(git -C "$(checkout_of "$root")" remote get-url origin)"
set_pushurls "$root" "$fetch_url"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "exact pushurl still publishes under byte-exact capture" 0
expect_ok "exact pushurl still pushes the snapshot" remote_has_snapshot "$root"

# A reachable second repository as pushurl: even with an identical main, the
# fetch-side main is not the destination's verified main, so refuse.
root="$(new_fixture split-pushurl)"
split_dest="$root/push-destination.git"
git init --quiet --bare --initial-branch=main "$split_dest"
git -C "$TEST_ROOT/split-pushurl-seed" push --quiet "$split_dest" main
set_pushurls "$root" "$split_dest"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "split pushurl is refused before export" 1
expect_ok "split refusal names the shared-URL requirement" output_has "$RUN_OUTPUT" "exactly one shared URL"
expect_not "split refusal does not echo the push URL" output_has "$RUN_OUTPUT" "$split_dest"
expect_not "split configuration never exports" export_recorded "$root"
expect_not "split configuration never pushes to the fetch remote" remote_has_snapshot "$root"
expect_not "split configuration never pushes to the push destination" path_has_snapshot "$split_dest"

# An unreachable push destination is the same gap: the fetch works, but the
# snapshot could not be verified there. Refuse early, not after the export.
root="$(new_fixture unreachable-pushurl)"
unreachable="$root/missing-push-destination.git"
set_pushurls "$root" "$unreachable"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "unreachable push destination is refused before export" 1
expect_ok "unreachable destination reports the split" output_has "$RUN_OUTPUT" "exactly one shared URL"
expect_not "unreachable destination path is not echoed" output_has "$RUN_OUTPUT" "missing-push-destination"
expect_not "unreachable destination never exports" export_recorded "$root"
expect_not "unreachable destination never pushes" remote_has_snapshot "$root"

# Multiple push URLs push to several repositories at once; one main check
# cannot cover them, so the destination is ambiguous.
root="$(new_fixture multiple-pushurls)"
fetch_url="$(git -C "$(checkout_of "$root")" remote get-url origin)"
set_pushurls "$root" "$fetch_url" "$fetch_url"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "multiple push URLs are refused" 1
expect_ok "multiple push URLs report the destination counts" output_has "$RUN_OUTPUT" "2 push URL(s)"
expect_not "multiple push URLs never export" export_recorded "$root"
expect_not "multiple push URLs never push" remote_has_snapshot "$root"

# `url.<base>.pushInsteadOf` rewrites the push URL without touching
# `remote.origin.pushurl`; `git remote get-url --push` applies the same rewrite
# git push does, so the guard sees the split destination.
root="$(new_fixture pushinsteadof-split)"
instead_dest="$root/push-destination.git"
git init --quiet --bare --initial-branch=main "$instead_dest"
git -C "$TEST_ROOT/pushinsteadof-split-seed" push --quiet "$instead_dest" main
fetch_url="$(git -C "$(checkout_of "$root")" remote get-url origin)"
git -C "$(checkout_of "$root")" config "url.$instead_dest.pushInsteadOf" "$fetch_url"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest"
expect_status "pushInsteadOf split is refused before export" 1
expect_ok "pushInsteadOf split reports the shared-URL requirement" output_has "$RUN_OUTPUT" "exactly one shared URL"
expect_not "pushInsteadOf split does not echo the rewritten URL" output_has "$RUN_OUTPUT" "$instead_dest"
expect_not "pushInsteadOf split never exports" export_recorded "$root"
expect_not "pushInsteadOf split never pushes to the fetch remote" remote_has_snapshot "$root"
expect_not "pushInsteadOf split never pushes to the rewritten destination" path_has_snapshot "$instead_dest"

# A URL or path as --remote is not a configured remote, so the fetch/push
# comparison cannot run; `pushInsteadOf` could still redirect the push. Refuse,
# and never echo the argument (a URL can carry credentials).
root="$(new_fixture unresolvable-remote)"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest" --remote "https://user:SENTINEL-TOKEN@example.invalid/org/repo.git"
expect_status "unresolvable --remote is refused before export" 1
expect_ok "unresolvable remote reports the named-remote requirement" output_has "$RUN_OUTPUT" "named remote"
expect_not "unresolvable remote is not echoed" output_has "$RUN_OUTPUT" "SENTINEL-TOKEN"
expect_not "unresolvable remote never exports" export_recorded "$root"
expect_not "unresolvable remote never pushes" remote_has_snapshot "$root"

# --allow-non-main bypasses source provenance, not a split destination: the
# snapshot would still land in a repository whose main was never checked.
root="$(new_fixture split-pushurl-bypass)"
bypass_dest="$root/push-destination.git"
git init --quiet --bare --initial-branch=main "$bypass_dest"
set_pushurls "$root" "$bypass_dest"
manifest="$(write_manifest "$root" '{"redacted": true}')"
capture run_publish "$root" "$manifest" --allow-non-main
expect_status "--allow-non-main cannot bypass a split pushurl" 1
expect_not "--allow-non-main split refusal never exports" export_recorded "$root"
expect_not "--allow-non-main split refusal never pushes" path_has_snapshot "$bypass_dest"

# ── manifest redaction rejections ───────────────────────────────────────────
root="$(new_fixture nested-redacted)"
manifest="$(write_manifest "$root" '{"schema": 1, "nested": {"redacted": true, "token": "SENTINEL-NESTED"}}')"
capture run_publish "$root" "$manifest"
expect_status "nested redacted:true does not satisfy the guard" 1
expect_ok "nested rejection names the top-level requirement" output_has "$RUN_OUTPUT" "top-level"
expect_not "nested rejection does not leak nested field contents" output_has "$RUN_OUTPUT" "SENTINEL-NESTED"
expect_not "nested redacted:true never pushes" remote_has_snapshot "$root"

root="$(new_fixture string-redacted)"
manifest="$(write_manifest "$root" '{"redacted": "SENTINEL-STRING"}')"
capture run_publish "$root" "$manifest"
expect_status "string redacted value is rejected" 1
expect_ok "string rejection reports the JSON type" output_has "$RUN_OUTPUT" "(found string)"
expect_not "string rejection does not leak the value" output_has "$RUN_OUTPUT" "SENTINEL-STRING"
expect_not "string redacted value never pushes" remote_has_snapshot "$root"

root="$(new_fixture numeric-redacted)"
manifest="$(write_manifest "$root" '{"redacted": 1}')"
capture run_publish "$root" "$manifest"
expect_status "numeric redacted value is rejected" 1
expect_ok "numeric rejection reports the JSON type" output_has "$RUN_OUTPUT" "(found number)"
expect_not "numeric redacted value never pushes" remote_has_snapshot "$root"

root="$(new_fixture boolean-redacted)"
manifest="$(write_manifest "$root" '{"redacted": false, "token": "SENTINEL-BOOLEAN"}')"
capture run_publish "$root" "$manifest"
expect_status "boolean false redacted value is rejected" 1
expect_ok "boolean rejection reports the JSON type" output_has "$RUN_OUTPUT" "(found boolean)"
expect_not "boolean rejection does not leak sibling contents" output_has "$RUN_OUTPUT" "SENTINEL-BOOLEAN"
expect_not "boolean false redacted value never pushes" remote_has_snapshot "$root"

root="$(new_fixture malformed-manifest)"
manifest="$(write_manifest "$root" '{"redacted": true')"
capture run_publish "$root" "$manifest"
expect_status "malformed manifest JSON fails closed" 1
expect_ok "malformed manifest reports a parse error" output_has "$RUN_OUTPUT" "not readable JSON"
expect_not "malformed manifest never pushes" remote_has_snapshot "$root"

printf 'workflow-publish-test: %d cases, %d failures\n' "$CASE_COUNT" "$FAILURES"
((FAILURES == 0))

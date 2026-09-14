#!/usr/bin/env bash
# workflow-import.sh — restore the local CarryCtx DB from the in-repo snapshot.
#
# In-repo restore (no mirror repository): fetches the publication branch
# `refs/heads/carryctx-snapshots` from the remote, then runs the native
# `carryctx import --from-git <ref> --mode replace --yes`. CarryCtx never
# touches the network, so this target fetches the branch itself.
#
# Safety (thin, no Python):
#   - refuses to replace a non-empty local DB (one project row with data rows)
#     without --force, leaving the DB untouched;
#   - initializes a fresh clone's CarryCtx state (no project row) before import;
#   - --dry-run validates the snapshot and writes nothing;
#   - restores the committed `.carryctx/config.toml` byte-identically, because
#     `carryctx import` rewrites it with current config defaults;
#   - prints snapshot provenance (commit, export id, CarryCtx-Source trailer).
#
# Usage:
#   scripts/workflow-import.sh [--force] [--dry-run] [--project DIR]
#                              [--remote NAME] [--git-timeout SECS] [--help]
#   --force            replace a non-empty local DB.
#   --dry-run          fetch + validate only; no DB write.
#   --project DIR      repository to restore (default: this checkout root).
#   --remote NAME      git remote to fetch from (default: origin).
#   --git-timeout SECS timeout for every carryctx/git op (default: 120).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAP_BRANCH="refs/heads/carryctx-snapshots"
PROJECT="$REPO_ROOT"
REMOTE="${WORKFLOW_REMOTE:-origin}"
GIT_TIMEOUT="${GIT_TIMEOUT:-120}"
FORCE=0
DRY_RUN=0

usage() {
	sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--force)
		FORCE=1
		shift
		;;
	--dry-run)
		DRY_RUN=1
		shift
		;;
	--project)
		PROJECT="${2:?--project requires a directory}"
		shift 2
		;;
	--project=*)
		PROJECT="${1#--project=}"
		shift
		;;
	--remote)
		REMOTE="${2:?--remote requires a name}"
		shift 2
		;;
	--remote=*)
		REMOTE="${1#--remote=}"
		shift
		;;
	--git-timeout)
		GIT_TIMEOUT="${2:?--git-timeout requires seconds}"
		shift 2
		;;
	--git-timeout=*)
		GIT_TIMEOUT="${1#--git-timeout=}"
		shift
		;;
	--help | -h)
		usage
		exit 0
		;;
	*)
		echo "workflow-import: FAIL: unknown flag $1 (see --help)" >&2
		exit 2
		;;
	esac
done

fail() {
	echo "workflow-import: FAIL: $1" >&2
	exit 1
}

log() {
	echo "workflow-import: $1"
}

warn() {
	echo "workflow-import: WARN: $1" >&2
}

have() { command -v "$1" >/dev/null 2>&1; }

have git || fail "git not on PATH"
have carryctx || fail "carryctx not on PATH"
have timeout || fail "timeout not on PATH"

PROJECT="$(cd "$PROJECT" && pwd)" || fail "project directory $PROJECT not found"
TRACK_REF="refs/remotes/$REMOTE/carryctx-snapshots"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/workflow-import.XXXXXX")"
cleanup() {
	rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

# Read-only probe of the local CarryCtx DB. Never creates or mutates it. Prints
# "<project_rows> <data_rows>"; project_rows is 1 when a projects row exists.
probe_state() {
	local db="$1" projects=0 rows=0 expr="" table present
	[[ -f "$db" ]] || {
		printf '0 0\n'
		return 0
	}
	have sqlite3 || fail "sqlite3 not on PATH (needed to inspect the local state DB); install it or pass a fresh --project"
	present="$(timeout "$GIT_TIMEOUT" sqlite3 -readonly "$db" \
		"SELECT name FROM sqlite_master WHERE type='table';" 2>/dev/null || true)"
	if grep -qx 'projects' <<<"$present"; then
		projects="$(timeout "$GIT_TIMEOUT" sqlite3 -readonly "$db" 'SELECT COUNT(*) FROM projects;' 2>/dev/null || echo 0)"
	fi
	for table in agents tasks task_dependencies progress_items sessions worktrees \
		checkpoints checkpoint_corrections scopes decisions handoffs events \
		graph_nodes graph_edges teams team_members; do
		if grep -qx "$table" <<<"$present"; then
			expr+="+(SELECT COUNT(*) FROM \"$table\")"
		fi
	done
	if [[ -n "$expr" ]]; then
		rows="$(timeout "$GIT_TIMEOUT" sqlite3 -readonly "$db" "SELECT 0${expr};" 2>/dev/null || echo 0)"
	fi
	printf '%s %s\n' "$projects" "$rows"
}

GIT_COMMON="$(timeout "$GIT_TIMEOUT" git -C "$PROJECT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" ||
	fail "cannot resolve the git common dir for $PROJECT"
DB="$GIT_COMMON/carryctx/state.sqlite"
STATE_LINE="$(probe_state "$DB")" || fail "cannot inspect local CarryCtx state"
read -r PROJECTS ROWS <<<"$STATE_LINE"
log "local state: db_exists=$([[ -f "$DB" ]] && echo 1 || echo 0) project_rows=$PROJECTS data_rows=$ROWS"

if [[ "$PROJECTS" -gt 1 ]]; then
	fail "local DB has $PROJECTS project rows; refusing to replace an ambiguous state DB"
fi
if [[ "$DRY_RUN" == 0 && "$PROJECTS" -ge 1 && "$ROWS" -gt 0 && "$FORCE" == 0 ]]; then
	fail "local DB already holds $ROWS data row(s); refusing to overwrite non-empty state without --force (local DB untouched)"
fi

log "fetching $REMOTE $SNAP_BRANCH -> $TRACK_REF"
if ! timeout "$GIT_TIMEOUT" git -C "$PROJECT" fetch "$REMOTE" \
	"refs/heads/carryctx-snapshots:refs/remotes/$REMOTE/carryctx-snapshots"; then
	fail "git fetch failed (has refs/heads/carryctx-snapshots been published? network/auth?)"
fi

print_provenance() {
	local commit export_id source_line message
	commit="$(timeout "$GIT_TIMEOUT" git -C "$PROJECT" rev-parse --short "$TRACK_REF" 2>/dev/null || echo '?')"
	message="$(timeout "$GIT_TIMEOUT" git -C "$PROJECT" log -1 --format=%B "$TRACK_REF" 2>/dev/null || true)"
	export_id="$(sed -n 's/^CarryCtx-Export-Id: //p' <<<"$message" | head -n1)"
	source_line="$(sed -n 's/^CarryCtx-Source: //p' <<<"$message" | head -n1)"
	log "provenance: snapshot=$commit export_id=${export_id:-?} source=${source_line:-?}"
}

if [[ "$DRY_RUN" == 1 ]]; then
	if ! timeout "$GIT_TIMEOUT" carryctx import --from-git "$TRACK_REF" \
		--mode replace --yes --dry-run --project "$PROJECT" >"$TMP_ROOT/import.json" 2>"$TMP_ROOT/import.err"; then
		cat "$TMP_ROOT/import.err" >&2 2>/dev/null || true
		fail "carryctx import --dry-run failed; local DB untouched"
	fi
	print_provenance
	if [[ "$PROJECTS" -ge 1 && "$ROWS" -gt 0 ]]; then
		log "dry-run PASS: snapshot fetched + validated; local DB has $ROWS data row(s), a real run needs --force"
	else
		log "dry-run PASS: snapshot fetched + validated; nothing imported"
	fi
	exit 0
fi

CFG="$PROJECT/.carryctx/config.toml"
CFG_BACKUP=""
if [[ -f "$CFG" ]]; then
	CFG_BACKUP="$TMP_ROOT/config.toml.before"
	cp -p "$CFG" "$CFG_BACKUP" || fail "cannot back up $CFG"
fi

# Fresh clone / no project row: initialize CarryCtx state explicitly so the
# documented fresh-clone path never depends on importer auto-init. The
# committed `.carryctx/config.toml` (when present) is restored byte-identically
# below, because `carryctx init` rewrites it with current config defaults.
if [[ "$PROJECTS" -eq 0 ]]; then
	log "no project row in local DB: initializing CarryCtx state (carryctx init --non-interactive)"
	if ! timeout "$GIT_TIMEOUT" carryctx init --non-interactive --project "$PROJECT" >"$TMP_ROOT/init.log" 2>&1; then
		cat "$TMP_ROOT/init.log" >&2 2>/dev/null || true
		fail "carryctx init failed"
	fi
fi

if [[ "$PROJECTS" -ge 1 && "$ROWS" -gt 0 ]]; then
	warn "--force: replacing $ROWS existing local data row(s) in $PROJECT"
fi

log "importing snapshot $TRACK_REF into $PROJECT (replace mode)"
if ! timeout "$GIT_TIMEOUT" carryctx import --from-git "$TRACK_REF" \
	--mode replace --yes --project "$PROJECT" >"$TMP_ROOT/import.json" 2>"$TMP_ROOT/import.err"; then
	cat "$TMP_ROOT/import.err" >&2 2>/dev/null || true
	fail "carryctx import failed; local DB is left in the state carryctx reports above"
fi

if [[ -n "$CFG_BACKUP" && -f "$CFG" ]] && ! cmp -s "$CFG_BACKUP" "$CFG"; then
	cp -p "$CFG_BACKUP" "$CFG" || fail "cannot restore $CFG after import"
	log "restored pre-existing .carryctx/config.toml (import rewrites local config defaults)"
fi

print_provenance
log "restore complete; local counts:"
timeout "$GIT_TIMEOUT" carryctx stats --project "$PROJECT" || fail "carryctx stats failed"

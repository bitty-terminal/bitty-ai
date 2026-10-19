#!/usr/bin/env bash
# host-boundary-gate-v2.sh — host-boundary lint gate v2 (fail-closed textual gate).
#
# Design authority: bitty-ai-docs `architecture/host-boundary-trait-design.md`,
# `## Classification delta for gate-v2` (candidate basis recorded in
# bitty-ai-docs#282; that section is candidate wording, not accepted text). That section inverts the AI-0145 topology: the host
# crate is a std-only vocabulary crate (no behavior, no dependencies) and the
# endpoint crate is the privileged consumer with a declared consumption set.
# Mechanics only (fail-closed empty set, unlisted-directory failure,
# rg-error-is-error) are inherited from the AI-0145 gate lineage
# (bitty-ai 13f5861/0446724/01f610f, never merged to main); the check
# inventory below follows the delta exactly, not the old inventory.
#
# The five checks (one per delta inventory item):
#   1. Endpoint consumption allow-list: every `bitty_ai_runtime::` and
#      `bitty_ai_host::` reference in the endpoint resolves to a member of the
#      declared consumption set (the 10 lawful seams); anything else fails.
#      Ambient authority (sockets/frames/spawn/clock/threads/net/fs) and the
#      never-seams (`ToolBus`, `precheck`) fail here too.
#   2. Host zero-dep tripwire: the host manifest `[dependencies]` section
#      stays comment-and-blank-lines only. (The E0 `host_manifest_...` test in
#      the endpoint is kept as the seed per the delta; this tripwire is the
#      gate-side enforcement of the same property.)
#   3. Pin tripwire: the endpoint manifest keeps pinning `bitty-ipc-api` at
#      the owner-decided revision. (The E0 `endpoint_ipc_pin_...` test is kept
#      as the seed per the delta; this tripwire is the gate-side enforcement.)
#   4. No-new-edge without classification update: an unlisted `crates/*`
#      directory, an unlisted manifest dependency edge, a wrong edge kind, or
#      a foreign `use` in the host fails until the classification names it.
#   5. Test-only confinement: seam-trait implementations, test doubles,
#      ledger grants, registry/provider construction, and cost settlement
#      appear only under `#[cfg(test)]`; any non-test occurrence fails.
#
# Fail-closed: the gate exits nonzero on any banned hit, on any `rg`
# invocation error, on a missing binding directory, on an unlisted crate
# directory, and on an unrecognized `#[cfg(test)]` placement. A passing gate
# proves only that no banned textual reference exists; semantic confinement
# stays with review and the trait bound.
#
# Usage:
#   scripts/host-boundary-gate-v2.sh
#   HOST_BOUNDARY_GATE_V2_ROOT=/path/to/fixture scripts/host-boundary-gate-v2.sh
#
# Exit codes:
#   0  pass — all five checks green.
#   1  fail — a banned reference, tripwire breach, or unlisted edge.
#   2  environment/tool error — missing tool, missing directory, bad scan.
set -euo pipefail

# POLICY (AI-0310): the scan root defaults to this repository, but the
# regression suite overrides it per fixture via the environment. The override
# selects which tree is scanned; it cannot turn a violation in the scanned
# tree into a pass, so it is a test affordance, not a verdict switch. CI
# never sets it.
REPO_ROOT="${HOST_BOUNDARY_GATE_V2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

ENDPOINT_CRATE="crates/bitty-ai-endpoint"
HOST_CRATE="crates/bitty-ai-host"
ENDPOINT_DIR="$REPO_ROOT/$ENDPOINT_CRATE"
HOST_DIR="$REPO_ROOT/$HOST_CRATE"
ENDPOINT_MANIFEST="$ENDPOINT_DIR/Cargo.toml"
HOST_MANIFEST="$HOST_DIR/Cargo.toml"

# POLICY (AI-0310): classified crate inventory per the delta. The endpoint is
# the privileged consumer; the host is the std-only vocabulary crate; the
# remaining five are classified non-binding members. `bitty-ai-control` is its
# own TLS/bootstrap composition boundary (AI-0333), not an endpoint edge. A new
# `crates/*` directory fails check 4 until the classification names it here.
CLASSIFIED_CRATES=(
	"crates/bitty-ai-endpoint"
	"crates/bitty-ai-host"
	"crates/bitty-ai-runtime"
	"crates/bitty-ai-slice"
	"crates/bitty-ai-session"
	"crates/bitty-ai-mcp"
	"crates/bitty-ai-control"
)

# POLICY (AI-0310): classified dependency edges per the delta. The endpoint
# holds exactly three edges (runtime + host by path, bitty-ipc-api by git
# pin); the host holds none. A new edge fails check 4 until the
# classification names it here.
ENDPOINT_ALLOWED_DEPS=(
	"bitty-ai-runtime"
	"bitty-ai-host"
	"bitty-ipc-api"
)

# POLICY (AI-0310): the `bitty-ipc-api` git pin is an owner decision
# (`f201911` per AI-0307). A bump is a separately scoped owner decision,
# never silent drift; the gate and the E0 seed test change together.
EXPECTED_IPC_PIN="f201911"

# POLICY (AI-0310): endpoint-allowed `bitty_ai_runtime` items — the declared
# consumption set, i.e. the 10 lawful seams as surveyed at main 7879757:
# select/read-only views (ProviderRegistry, SelectRequest, SelectionError,
# ModelCapability, ArtifactStore, ContextError, ContextRecord, RecordBody,
# ArtifactAbsence, Message, ModelProvider, ProviderError, Role, TurnRequest,
# MAX_SUMMARY_BYTES, ToolError, ToolExecutor), test-only doubles and fixtures
# (FakeProvider, FakeToolExecutor, ModelDescriptor, ProviderTurn,
# ProviderUsage, ToolCallRequest, ToolSuccess, ModelRef, ModelRegistration,
# StableId, ArtifactRef, AssembledRecord, AssembledContent, ContextPriority,
# FallbackDirective), the drift-locked fallback reference
# (fallback_directive), and cost-settlement fixture (estimate_cost, confined
# to tests by check 5). `ToolBus` is deliberately absent: the endpoint takes
# the `ToolExecutor` handle only as an injected reference, never `ToolBus`,
# `precheck`, or `dispatch`.
RUNTIME_ALLOW=(
	"ArtifactAbsence" "ArtifactRef" "ArtifactStore" "AssembledContent"
	"AssembledRecord" "ContextError" "ContextPriority" "ContextRecord"
	"FakeProvider" "FakeToolExecutor" "FallbackDirective" "MAX_SUMMARY_BYTES"
	"Message" "ModelCapability" "ModelDescriptor" "ModelProvider" "ModelRef"
	"ModelRegistration" "ProviderError" "ProviderRegistry" "ProviderTurn"
	"ProviderUsage" "RecordBody" "Role" "SelectRequest" "SelectionError"
	"StableId" "ToolCallRequest" "ToolError" "ToolExecutor" "ToolSuccess"
	"TurnRequest" "estimate_cost" "fallback_directive"
)
RUNTIME_MODULES=(
	"context" "provider" "tool"
)

# POLICY (AI-0310): endpoint-allowed `bitty_ai_host` items — the host
# vocabulary the endpoint may consume: cache/disclosure, extension registry,
# S2 envelopes (McpFailure, PinEnvelope, SnapshotEnvelope, ToolsCapabilityShape,
# ImportedTool, ToolListSnapshot, ToolPlacement, ToolSpec, InitializeResult,
# PROTOCOL_VERSION), model selection envelopes plus the resolve seam
# (EstimateCostEnvelopeRequest, FallbackDirectiveEnvelopeRequest,
# SelectEnvelopeRequest, FallbackDirective), provider registry vocabulary,
# snapshot vocabulary (SnapshotRefusal, SnapshotVersion,
# check_snapshot_version), context providers, and the tool-dispatch seam
# (injected ConsentLedger trait object plus bounds and validators; never a
# constructed ledger).
HOST_ALLOW=(
	"ActivationStub" "ActivationError" "AgentInstanceId" "AgentLevel"
	"ArtifactAbsence" "CacheHintRequest" "CacheHintResponse" "CacheKey"
	"CacheScope" "CallerScopeEcho" "CompleteRequest" "CompleteResponse"
	"CompleteToolCall" "CompleteUsage" "ConsentDecision" "ConsentGrant"
	"ConsentLedger" "ConsentQuery" "ContextReadError" "ContextReadRequest"
	"ContextRequest" "DENY_ALL_ACTIVATION_REASON" "DeclarationEnvelope"
	"DetailLevel" "EstimateCostEnvelopeRequest" "ExtensionPoint"
	"FallbackDirective" "FallbackDirectiveEnvelopeRequest" "ImportedTool"
	"InitializeResult" "InputFingerprint" "IpcEnvelopeError" "IpcMessage"
	"IpcRole" "ListModelsRequest" "ListModelsResponse" "MAX_ARTIFACTS"
	"MAX_CONTEXT_RECORDS" "MAX_REQUEST_TIMEOUT_MS"
	"MAX_TOOL_ARGUMENTS_BYTES" "MAX_TOOL_RESULT_BYTES" "Manifest"
	"McpFailure" "ModelCapability" "ModelRef" "ModelRegistration"
	"PinEnvelope" "PROTOCOL_VERSION" "ProviderError" "SamplingNote"
	"SelectEnvelopeRequest" "SnapshotEnvelope" "SnapshotRefusal"
	"SnapshotVersion" "StableId" "ToolListSnapshot" "ToolPlacement"
	"ToolRegistry" "ToolSpec" "ToolsCapabilityShape" "UsageEvent"
	"activation_point_name" "bound_reason" "check_snapshot_version"
	"consent_grant_expiry" "contains_untrusted_markers" "narrowing_omits"
	"parse_activation_point" "validate_caller_scope" "validate_model_name"
	"validate_provider_id" "validate_tool_name" "validate_zone1_text"
)
HOST_MODULES=(
	"cache_disclosure" "context_providers" "extension_registry"
	"mcp_placement" "model_selection" "provider_registry" "snapshot"
	"tool_dispatch"
)

# POLICY (AI-0310): ambient authority banned in the endpoint. The endpoint
# takes sockets, clocks, threads, and the filesystem as caller-supplied
# values or not at all; a direct path into these subtrees is a new edge the
# classification did not declare. `ToolBus` / `precheck` are banned here
# because the declared seam is the injected `ToolExecutor`, never the bus.
# The `extern crate std` / `use std as` rename spellings are banned with the
# paths themselves: a one-line rename (`extern crate std as s;` plus
# `s::fs::read`) recompiles the banned subtrees under an alias the `std::`
# patterns cannot see (both spellings proven compilable), so the rename is
# refused outright rather than chased through alias resolution.
#
# POLICY (AI-0330): `std::sync::Arc`, `std::sync::atomic::AtomicBool`,
# `std::sync::atomic::AtomicU64`, and `std::sync::atomic::Ordering` are the
# helper's private in-memory lifetime discriminator (shared clone cell and
# monotonic sequence). They are not ambient I/O or a transport. Every other
# `std::sync` item (Mutex, RwLock, mpsc, Condvar, Once, ...) and all
# clock/fs/net/thread paths stay banned.
AMBIENT_PATTERN='std::(net|process|thread|fs|time|os|env)\b|std::sync(?!::(?:Arc\b|atomic::(?:Atomic(?:Bool|U64)\b|Ordering\b)))\b|\btokio::|\bmio::|\bsocket2::|\b(UnixStream|TcpStream|TcpListener|UdpSocket|SocketAddr)\b|\bthread::spawn\b|\bSystemTime\b|\bToolBus\b|::precheck\b|extern\s+crate\s+std\b|use\s+std\s+as\b'

fail() {
	printf 'host-boundary-gate-v2: FAIL: %s\n' "$*" >&2
	exit 1
}

tool_fail() {
	printf 'host-boundary-gate-v2: ERROR: %s\n' "$*" >&2
	exit 2
}

contains_token() {
	local needle="$1"
	shift
	local candidate
	for candidate in "$@"; do
		if [[ "$candidate" == "$needle" ]]; then
			return 0
		fi
	done
	return 1
}

command -v rg >/dev/null 2>&1 || tool_fail "rg is required"
command -v awk >/dev/null 2>&1 || tool_fail "awk is required"
[[ -d "$REPO_ROOT/crates" ]] || tool_fail "crates directory is missing: $REPO_ROOT/crates"

# Fail-closed binding set (AI-0145 mechanics): the declared binding crates
# exist with manifests. There is no bootstrap exception: the endpoint and
# host crates exist, so a missing directory is a configuration error, and an
# empty scan (no Rust sources) fails rather than passing vacuously.
[[ -d "$ENDPOINT_DIR" ]] || fail "binding crate directory is missing: $ENDPOINT_CRATE"
[[ -d "$HOST_DIR" ]] || fail "binding crate directory is missing: $HOST_CRATE"
[[ -f "$ENDPOINT_MANIFEST" ]] || fail "binding crate manifest is missing: $ENDPOINT_CRATE/Cargo.toml"
[[ -f "$HOST_MANIFEST" ]] || fail "binding crate manifest is missing: $HOST_CRATE/Cargo.toml"

ENDPOINT_RS_COUNT="$(find "$ENDPOINT_DIR" -name '*.rs' -type f | wc -l)"
HOST_RS_COUNT="$(find "$HOST_DIR" -name '*.rs' -type f | wc -l)"
((ENDPOINT_RS_COUNT > 0)) || fail "binding crate has no Rust sources: $ENDPOINT_CRATE"
((HOST_RS_COUNT > 0)) || fail "binding crate has no Rust sources: $HOST_CRATE"

# ── check 1: endpoint consumption allow-list ────────────────────────────────
#
# 1a. Every `bitty_ai_runtime::` / `bitty_ai_host::` reference in the endpoint
# resolves to a member of the declared set. Three sub-rules:
#   (i)   each `use` statement names at least one allow-listed item and no
#         foreign item (a bare-module `use` fails: it would admit
#         `module::Anything` paths that spell no crate identifier);
#   (ii)  each qualified path resolves to an allow-listed item (a trailing
#         module-only mention, e.g. a doc link, is not an item use);
#   (iii) a bare crate identifier with no qualified form is an unclassified
#         mention (macro-argument smuggling) and fails.

collect_use_statements() {
	awk '
		/^[[:space:]]*use[[:space:]]+bitty_ai_(runtime|host)::/ { in_use = 1; stmt = "" }
		in_use {
			line = $0
			gsub(/^[[:space:]]+/, "", line)
			stmt = (stmt == "" ? line : stmt " " line)
			if (line ~ /;[[:space:]]*$/) {
				print stmt
				in_use = 0
			}
		}
	' "$ENDPOINT_DIR"/src/*.rs
}

validate_use_statement() {
	local stmt="$1"
	local cratekind
	if [[ "$stmt" == *"bitty_ai_runtime::"* ]]; then
		cratekind="runtime"
	elif [[ "$stmt" == *"bitty_ai_host::"* ]]; then
		cratekind="host"
	else
		fail "check 1 (allow-list): unclassified use statement: $stmt"
	fi
	local body="${stmt#*bitty_ai_${cratekind}::}"
	body="${body%;}"
	# Drop `as Alias` renames: aliases are local names, never allow items.
	body="$(printf '%s' "$body" | sed -E 's/[[:space:]]+as[[:space:]]+[A-Za-z0-9_]+//g')"
	# Tokenize on use-tree punctuation.
	local tokens
	tokens="$(printf '%s' "$body" | tr '{},:;' '     ' | tr -s ' ' '\n' | sed '/^[[:space:]]*$/d')"
	local item_count=0
	local token
	while IFS= read -r token; do
		[[ -n "$token" ]] || continue
		if [[ "$cratekind" == "runtime" ]]; then
			contains_token "$token" "${RUNTIME_MODULES[@]}" && continue
			contains_token "$token" "${RUNTIME_ALLOW[@]}" ||
				fail "check 1 (allow-list): endpoint use of non-allow-listed runtime item: $token in: $stmt"
		else
			contains_token "$token" "${HOST_MODULES[@]}" && continue
			contains_token "$token" "${HOST_ALLOW[@]}" ||
				fail "check 1 (allow-list): endpoint use of non-allow-listed host item: $token in: $stmt"
		fi
		item_count=$((item_count + 1))
	done <<<"$tokens"
	((item_count > 0)) ||
		fail "check 1 (allow-list): endpoint use statement names no allow-listed item (bare-module import): $stmt"
}

use_statements="$(collect_use_statements)"
while IFS= read -r stmt; do
	[[ -n "$stmt" ]] || continue
	validate_use_statement "$stmt"
done <<<"$use_statements"

qual_refs="$(rg -N --no-filename --glob '!target/**' -o 'bitty_ai_(runtime|host)::[A-Za-z0-9_:]+' -- "$ENDPOINT_DIR" 2>&1)" || {
	status=$?
	((status == 1)) || tool_fail "check 1 (allow-list): rg qualified scan failed: $qual_refs"
	qual_refs=""
}
while IFS= read -r ref; do
	[[ -n "$ref" ]] || continue
	cratekind="runtime"
	rest="${ref#bitty_ai_runtime::}"
	if [[ "$ref" == bitty_ai_host::* ]]; then
		cratekind="host"
		rest="${ref#bitty_ai_host::}"
	fi
	item=""
	IFS=':' read -r -a comps <<<"$rest"
	for comp in "${comps[@]}"; do
		[[ -n "$comp" ]] || continue
		if [[ "$cratekind" == "runtime" ]]; then
			contains_token "$comp" "${RUNTIME_MODULES[@]}" && continue
		else
			contains_token "$comp" "${HOST_MODULES[@]}" && continue
		fi
		item="$comp"
		break
	done
	# A module-only mention (e.g. a doc link to a module) names no item.
	[[ -n "$item" ]] || continue
	# A trailing lowercase segment is an associated function or variant;
	# the item is the segment already taken above, so only validate the
	# item itself here.
	if [[ "$cratekind" == "runtime" ]]; then
		contains_token "$item" "${RUNTIME_ALLOW[@]}" ||
			fail "check 1 (allow-list): endpoint reference outside the declared runtime set: $ref"
	else
		contains_token "$item" "${HOST_ALLOW[@]}" ||
			fail "check 1 (allow-list): endpoint reference outside the declared host set: $ref"
	fi
done <<<"$qual_refs"

bare_mentions="$(rg -n --glob '!target/**' 'bitty_ai_runtime|bitty_ai_host' -- "$ENDPOINT_DIR" 2>&1 | rg -v 'bitty_ai_(runtime|host)::' || true)"
[[ -z "$bare_mentions" ]] ||
	fail "check 1 (allow-list): endpoint has bare crate-identifier mentions with no qualified form (unclassified route):"$'\n'"$bare_mentions"

# 1b. Ambient authority and never-seams are banned in the endpoint.
ambient_hits="$(rg -n --glob '!target/**' -P "$AMBIENT_PATTERN" -- "$ENDPOINT_DIR" 2>&1)" || {
	status=$?
	((status == 1)) || tool_fail "check 1 (allow-list): rg ambient scan failed: $ambient_hits"
	ambient_hits=""
}
[[ -z "$ambient_hits" ]] ||
	fail "check 1 (allow-list): endpoint uses banned ambient/never-seam paths:"$'\n'"$ambient_hits"

printf 'host-boundary-gate-v2: check 1 PASS (endpoint consumption allow-list)\n'

# ── check 2: host zero-dep tripwire ──────────────────────────────────────────
dep_violations="$(awk '
	/^[[:space:]]*\[/ {
		in_deps = ($0 ~ /^[[:space:]]*\[dependencies\]/)
		next
	}
	in_deps && $0 !~ /^[[:space:]]*(#|$)/ { print }
' "$HOST_MANIFEST")"
[[ -z "$dep_violations" ]] ||
	fail "check 2 (host zero-dep): host manifest must stay dependency-free, found:"$'\n'"$dep_violations"

printf 'host-boundary-gate-v2: check 2 PASS (host zero-dep tripwire)\n'

# ── check 3: pin tripwire ────────────────────────────────────────────────────
pin_line="$(grep -E 'bitty-ipc-api.*rev[[:space:]]*=' "$ENDPOINT_MANIFEST" || true)"
[[ -n "$pin_line" ]] || fail "check 3 (pin): endpoint manifest has no bitty-ipc-api rev pin"
pin_rev="$(printf '%s' "$pin_line" | sed -nE 's/.*rev[[:space:]]*=[[:space:]]*"([0-9a-f]+)".*/\1/p')"
[[ -n "$pin_rev" ]] || fail "check 3 (pin): could not parse the bitty-ipc-api rev from the endpoint manifest"
[[ "$pin_rev" == "$EXPECTED_IPC_PIN" ]] ||
	fail "check 3 (pin): endpoint bitty-ipc-api pin is $pin_rev, owner decision is $EXPECTED_IPC_PIN"

printf 'host-boundary-gate-v2: check 3 PASS (pin tripwire at %s)\n' "$EXPECTED_IPC_PIN"

# ── check 4: no-new-edge without classification update ──────────────────────
shopt -s nullglob
for candidate in "$REPO_ROOT"/crates/*; do
	[[ -d "$candidate" ]] || continue
	relative="${candidate#"$REPO_ROOT"/}"
	contains_token "$relative" "${CLASSIFIED_CRATES[@]}" ||
		fail "check 4 (no-new-edge): unlisted crate directory — update the classification first: $relative"
done
shopt -u nullglob
for classified in "${CLASSIFIED_CRATES[@]}"; do
	[[ -d "$REPO_ROOT/$classified" ]] || fail "check 4 (no-new-edge): classified crate directory is missing: $classified"
	[[ -f "$REPO_ROOT/$classified/Cargo.toml" ]] || fail "check 4 (no-new-edge): classified crate manifest is missing: $classified/Cargo.toml"
done

manifest_dep_names() {
	awk '
		/^[[:space:]]*\[/ {
			in_deps = ($0 ~ /^[[:space:]]*\[dependencies\]/)
			next
		}
		in_deps && /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/ {
			line = $0
			sub(/^[[:space:]]*/, "", line)
			sub(/[[:space:]]*=.*/, "", line)
			print line
		}
	' "$1"
}

while IFS= read -r dep; do
	[[ -n "$dep" ]] || continue
	contains_token "$dep" "${ENDPOINT_ALLOWED_DEPS[@]}" ||
		fail "check 4 (no-new-edge): endpoint has unlisted dependency edge — update the classification first: $dep"
done < <(manifest_dep_names "$ENDPOINT_MANIFEST")

endpoint_runtime_line="$(grep -E '^[[:space:]]*bitty-ai-runtime[[:space:]]*=' "$ENDPOINT_MANIFEST" || true)"
endpoint_host_line="$(grep -E '^[[:space:]]*bitty-ai-host[[:space:]]*=' "$ENDPOINT_MANIFEST" || true)"
endpoint_ipc_line="$(grep -E '^[[:space:]]*bitty-ipc-api[[:space:]]*=' "$ENDPOINT_MANIFEST" || true)"
[[ "$endpoint_runtime_line" == *"path"* ]] || fail "check 4 (no-new-edge): endpoint runtime edge must be a path dependency"
[[ "$endpoint_host_line" == *"path"* ]] || fail "check 4 (no-new-edge): endpoint host edge must be a path dependency"
[[ "$endpoint_ipc_line" == *"git"* ]] || fail "check 4 (no-new-edge): endpoint bitty-ipc-api edge must be a git dependency"

host_dep_names="$(manifest_dep_names "$HOST_MANIFEST")"
[[ -z "$host_dep_names" ]] ||
	fail "check 4 (no-new-edge): host must hold no dependency edges, found:"$'\n'"$host_dep_names"

# The host is std-only vocabulary: any foreign `use` in host source is a new
# edge the manifests above do not declare.
host_foreign_uses="$(rg -n --glob '!target/**' 'use\s+bitty_ai_runtime\b|use\s+bitty_ai_host\b|use\s+bitty_ipc|extern\s+crate\s+bitty_ai_runtime|extern\s+crate\s+bitty_ipc|bitty_ipc_api::' -- "$HOST_DIR" 2>&1)" || {
	status=$?
	((status == 1)) || tool_fail "check 4 (no-new-edge): rg host-edge scan failed: $host_foreign_uses"
	host_foreign_uses=""
}
[[ -z "$host_foreign_uses" ]] ||
	fail "check 4 (no-new-edge): host source has foreign-crate uses:"$'\n'"$host_foreign_uses"

printf 'host-boundary-gate-v2: check 4 PASS (no-new-edge)\n'

# ── check 5: test-only confinement ───────────────────────────────────────────
#
# Seam-trait implementations, test doubles, ledger grants, registry/provider
# construction, and cost settlement appear only under `#[cfg(test)]`. The
# strip below removes `#[cfg(test)] mod ... { ... }` blocks (brace-counted)
# and full-line comments; a `#[cfg(test)]` attribute on any other item is an
# unrecognized placement and fails closed.
stripped="$(awk '
	function nopen(s) { return gsub(/\{/, "{", s) }
	function nclose(s) { return gsub(/\}/, "}", s) }
	/^[[:space:]]*#\[cfg\(test\)\][[:space:]]*$/ {
		if ((getline nextline) > 0) {
			if (nextline ~ /(^|[^A-Za-z0-9_])mod[[:space:]]+[A-Za-z0-9_]+/) {
				depth = nopen(nextline) - nclose(nextline)
				in_test = 1
				next
			}
		}
		print "UNRECOGNIZED-CFG-TEST-PLACEMENT:" $0
		bad = 1
		next
	}
	in_test {
		depth += nopen($0) - nclose($0)
		if (depth <= 0) { in_test = 0 }
		next
	}
	{ print }
	END { if (bad) exit 3 }
' "$ENDPOINT_DIR"/src/*.rs)" || awk_status=$?
awk_status="${awk_status:-0}"
((awk_status != 3)) || fail "check 5 (test-only): unrecognized #[cfg(test)] placement — update the gate first"
((awk_status == 0)) || tool_fail "check 5 (test-only): failed to strip test modules"

# POLICY (AI-0310): test-confined symbols per the delta. Each is lawful
# inside `#[cfg(test)]` (doubles, ledger grants, registry/provider
# construction, cost settlement, the drift-locked `fallback_directive`
# reference) and a smuggling route outside it. `HostStableId::new` and
# `HostModelRef::new` stay lawful: the bare-construction ban below excludes
# alias-qualified (`Host*::new`) and host-crate-qualified spellings, so only
# runtime-qualified or bare construction fails.
TEST_ONLY_PATTERN='FakeProvider|FakeToolExecutor|estimate_cost|fallback_directive|ProviderRegistry::new|register_alias|\.register\(|\.grant\(|ConsentLedger::new|bitty_ai_runtime::(StableId|ModelRef|ModelRegistration)\b|(?<![A-Za-z0-9_:])(StableId|ModelRef|ModelRegistration)::new'
test_only_hits="$(printf '%s' "$stripped" | grep -v '^[[:space:]]*//' | rg -n -P "$TEST_ONLY_PATTERN" 2>&1)" || {
	status=$?
	((status == 1)) || tool_fail "check 5 (test-only): rg test-confinement scan failed: $test_only_hits"
	test_only_hits=""
}
[[ -z "$test_only_hits" ]] ||
	fail "check 5 (test-only): non-test endpoint code uses test-confined symbols:"$'\n'"$test_only_hits"

# Seam-trait implementations never land in the endpoint outside tests; the
# multiline form matches `impl ... Trait ... for` across line breaks.
seam_impl_hits="$(printf '%s' "$stripped" | grep -v '^[[:space:]]*//' | rg -n -U -P '(?s)\bimpl\b[^{};]*\b(?:ToolExecutor|ToolAuthorizer|ModelProvider)\b\s+for\b' 2>&1)" || {
	status=$?
	((status == 1)) || tool_fail "check 5 (test-only): rg seam-implementation scan failed: $seam_impl_hits"
	seam_impl_hits=""
}
[[ -z "$seam_impl_hits" ]] ||
	fail "check 5 (test-only): non-test endpoint code implements a Core seam trait:"$'\n'"$seam_impl_hits"

printf 'host-boundary-gate-v2: check 5 PASS (test-only confinement)\n'

printf 'host-boundary-gate-v2: PASS (5 checks, endpoint %s, host %s, pin %s)\n' \
	"$ENDPOINT_CRATE" "$HOST_CRATE" "$EXPECTED_IPC_PIN"

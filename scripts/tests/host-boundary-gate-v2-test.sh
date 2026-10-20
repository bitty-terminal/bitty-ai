#!/usr/bin/env bash
# host-boundary-gate-v2-test.sh — regression suite for the host-boundary
# lint gate v2 (`scripts/host-boundary-gate-v2.sh`).
#
# Pattern (in miniature) of the AI-0145 54-case suite: each case builds a
# synthetic repository fixture, runs the gate with `HOST_BOUNDARY_GATE_V2_ROOT`
# pointed at it, and asserts the exit code. Positive cases must pass (exit 0);
# hostile cases must fail (exit 1). The suite also asserts the gate is green
# on the real tree, so day-one-red is impossible: the allow-list admits the
# existing endpoint consumption, including the `ToolExecutor` import a verbatim
# AI-0145 port would have flagged.
#
# Usage: scripts/tests/host-boundary-gate-v2-test.sh
set -euo pipefail

# POLICY (AI-0310): the suite runs the gate under test against synthetic
# fixtures via HOST_BOUNDARY_GATE_V2_ROOT (a scan-root selector, not a verdict
# switch) and asserts the gate is also green on the real tree, so the
# allow-list admits the existing endpoint consumption with no day-one red.
GATE="${HOST_BOUNDARY_GATE_V2_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/host-boundary-gate-v2.sh}"
SCRATCH_ROOT="${TMPDIR:-/tmp/bitty}"
mkdir -p -- "$SCRATCH_ROOT"
TEST_ROOT="$(mktemp -d "$SCRATCH_ROOT/host-boundary-gate-v2-test.XXXXXX")"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

FAILURES=0
CASE_COUNT=0

pass_case() {
	CASE_COUNT=$((CASE_COUNT + 1))
	local name="$1"
	local root="$2"
	if HOST_BOUNDARY_GATE_V2_ROOT="$root" bash "$GATE" >/dev/null 2>&1; then
		printf 'ok %d - %s (pass as designed)\n' "$CASE_COUNT" "$name"
	else
		printf 'not ok %d - %s (expected PASS, gate failed)\n' "$CASE_COUNT" "$name"
		FAILURES=$((FAILURES + 1))
	fi
}

fail_case() {
	CASE_COUNT=$((CASE_COUNT + 1))
	local name="$1"
	local root="$2"
	local output
	if output="$(HOST_BOUNDARY_GATE_V2_ROOT="$root" bash "$GATE" 2>&1)"; then
		printf 'not ok %d - %s (expected FAIL, gate passed)\n' "$CASE_COUNT" "$name"
		FAILURES=$((FAILURES + 1))
	else
		local status=$?
		if ((status == 1)); then
			printf 'ok %d - %s (fail as designed)\n' "$CASE_COUNT" "$name"
		else
			printf 'not ok %d - %s (expected FAIL/1, got ERROR/%d):\n%s\n' \
				"$CASE_COUNT" "$name" "$status" "$output"
			FAILURES=$((FAILURES + 1))
		fi
	fi
}

# Builds a minimal classified tree: seven classified crates, an endpoint with
# the day-one-red import plus lawful host consumption, and a host with no
# dependencies. Prints the fixture root.
new_fixture() {
	local name="$1"
	local root="$TEST_ROOT/$name"
	local crate
	mkdir -p "$root/crates/bitty-ai-endpoint/src" "$root/crates/bitty-ai-host/src"
	printf '[workspace]\nmembers = [\n  "crates/bitty-ai-slice",\n  "crates/bitty-ai-runtime",\n  "crates/bitty-ai-session",\n  "crates/bitty-ai-mcp",\n  "crates/bitty-ai-control",\n  "crates/bitty-ai-host",\n  "crates/bitty-ai-endpoint",\n]\nresolver = "3"\n' >"$root/Cargo.toml"
	for crate in bitty-ai-runtime bitty-ai-slice bitty-ai-session bitty-ai-mcp bitty-ai-control; do
		mkdir -p "$root/crates/$crate/src"
		printf '[package]\nname = "%s"\nversion = "0.0.1"\nedition = "2024"\n' "$crate" >"$root/crates/$crate/Cargo.toml"
		printf '// stub\n' >"$root/crates/$crate/src/lib.rs"
	done
	printf '[package]\nname = "bitty-ai-host"\nversion = "0.0.1"\nedition = "2024"\n\n[dependencies]\n# No dependencies: std only.\n' >"$root/crates/bitty-ai-host/Cargo.toml"
	# The host doc link below names the runtime crate without consuming it;
	# host docs are exempt from the endpoint-scoped allow-list.
	printf '//! host stub\n#![deny(unsafe_code)]\n/// See [`ExecutionContext`](bitty_ai_runtime::tool::ExecutionContext).\npub mod snapshot;\n' >"$root/crates/bitty-ai-host/src/lib.rs"
	printf '//! snapshot vocabulary\npub struct SnapshotVersion(u32);\npub struct SnapshotRefusal;\npub fn check_snapshot_version(a: u32, b: u32) -> bool { a == b }\n' >"$root/crates/bitty-ai-host/src/snapshot.rs"
	printf '[package]\nname = "bitty-ai-endpoint"\nversion = "0.0.1"\nedition = "2024"\n\n[dependencies]\nbitty-ai-runtime = { path = "../bitty-ai-runtime" }\nbitty-ai-host = { path = "../bitty-ai-host" }\nbitty-ipc-api = { git = "https://github.com/bitty-terminal/bitty-ipc", rev = "e6a2c0c" }\n' >"$root/crates/bitty-ai-endpoint/Cargo.toml"
	# POLICY (AI-0310): fixture pins mirror the owner decisions under test
	# (pin e6a2c0c); the pin-drift case below mutates the rev to prove the
	# tripwire fails closed.
	cat >"$root/crates/bitty-ai-endpoint/src/lib.rs" <<'EOF'
#![deny(unsafe_code)]
//! Day-one-red import: a verbatim AI-0145 port bans `tool::` imports, but
//! gate-v2 allow-lists the injected `ToolExecutor` seam, so this passes.
//! No socket, no clock, no threads here.
use bitty_ai_runtime::tool::{ToolError as RuntimeToolError, ToolExecutor as RuntimeToolExecutor};
use bitty_ai_host::snapshot::{SnapshotRefusal, SnapshotVersion, check_snapshot_version};

pub struct ToolView<'a> {
    pub executor: &'a mut dyn RuntimeToolExecutor,
}

pub fn versions_match(a: SnapshotVersion, b: SnapshotVersion) -> bool {
    let _ = SnapshotRefusal;
    let _ = RuntimeToolError;
    check_snapshot_version(1, 1)
}

#[cfg(test)]
mod tests {
    use bitty_ai_runtime::{FakeProvider, ModelDescriptor};

    #[test]
    fn doubles_stay_in_tests() {
        let _ = FakeProvider::new("bitty-fake");
        let _ = ModelDescriptor;
    }
}
EOF
	printf '%s\n' "$root"
}

endpoint_lib() {
	printf '%s/crates/bitty-ai-endpoint/src/lib.rs' "$1"
}

# ── positive cases (must PASS) ──────────────────────────────────────────────
root="$(new_fixture clean)"
pass_case "clean classified tree passes" "$root"

root="$(new_fixture lifetime-atomics)"
cat >>"$(endpoint_lib "$root")" <<'EOF'

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

pub struct LifetimeCell {
    sequence: AtomicU64,
    revoked: AtomicBool,
}

pub fn share(cell: &Arc<LifetimeCell>) -> bool {
    let _ = Ordering::SeqCst;
    !cell.revoked.load(Ordering::SeqCst) && cell.sequence.load(Ordering::Relaxed) > 0
}
EOF
pass_case "AI-0330 lifetime Arc/atomics pass" "$root"

root="$(new_fixture lawful-alias-construction)"
cat >>"$(endpoint_lib "$root")" <<'EOF'

use bitty_ai_host::context_providers::StableId as HostStableId;
use bitty_ai_host::provider_registry::ModelRef as HostModelRef;

pub fn lawful_construction(id: &str) {
    let _ = HostStableId::new(id);
    let _ = HostModelRef::new("p", "m");
}
EOF
mkdir -p "$root/crates/bitty-ai-host/src"
pass_case "lawful host-alias construction passes" "$root"

# ── hostile cases (must FAIL) ───────────────────────────────────────────────
# POLICY (AI-0310): each hostile case names the smuggling route it locks:
# ambient authority (socket/fs/threads), unclassified routes (ToolBus import,
# bare-identifier macro argument), test-only escape (doubles, grants, cost,
# construction, seam impls), tripwire breaches (host dep, pin drift), and
# unclassified edges (new crate, new dep, missing binding set).
root="$(new_fixture smuggled-socket-import)"
printf '\nuse std::net::TcpStream;\n' >>"$(endpoint_lib "$root")"
fail_case "smuggled socket import fails" "$root"

root="$(new_fixture smuggled-toolbus)"
printf '\nuse bitty_ai_runtime::tool::ToolBus;\n' >>"$(endpoint_lib "$root")"
fail_case "smuggled ToolBus import fails (outside allow-list)" "$root"

root="$(new_fixture smuggled-fs)"
printf '\npub fn read_it(p: &str) { let _ = std::fs::read(p); }\n' >>"$(endpoint_lib "$root")"
fail_case "smuggled filesystem read fails" "$root"

root="$(new_fixture smuggled-thread-spawn)"
printf '\npub fn bg() { let _ = std::thread::spawn(|| {}); }\n' >>"$(endpoint_lib "$root")"
fail_case "smuggled thread spawn fails" "$root"

root="$(new_fixture smuggled-sync-mutex)"
printf '\nuse std::sync::Mutex;\npub fn guard(m: &Mutex<u8>) -> u8 { *m.lock().unwrap() }\n' >>"$(endpoint_lib "$root")"
fail_case "smuggled std::sync::Mutex fails" "$root"

root="$(new_fixture smuggled-extern-crate-rename)"
printf '\nextern crate std as s;\npub fn exfil(p: &str) { let _ = s::fs::read(p); }\n' >>"$(endpoint_lib "$root")"
fail_case "extern-crate std rename fails" "$root"

root="$(new_fixture smuggled-use-as-rename)"
printf '\nuse std as s;\npub fn exfil(p: &str) { let _ = s::fs::read(p); }\n' >>"$(endpoint_lib "$root")"
fail_case "use-as std rename fails" "$root"

root="$(new_fixture smuggled-macro-route)"
printf '\nmy_macro!(bitty_ai_runtime);\n' >>"$(endpoint_lib "$root")"
fail_case "bare crate-identifier macro route fails" "$root"

root="$(new_fixture nontest-fake-provider)"
printf '\npub fn evil() { let _ = bitty_ai_runtime::FakeProvider::new("x"); }\n' >>"$(endpoint_lib "$root")"
fail_case "non-test FakeProvider construction fails" "$root"

root="$(new_fixture nontest-grant)"
printf '\npub fn evil(ledger: &mut ConsentLedger) { ledger.grant("c".into(), scope, 0, 1, "u".into()); }\n' >>"$(endpoint_lib "$root")"
fail_case "non-test ledger grant fails" "$root"

root="$(new_fixture nontest-estimate-cost)"
printf '\npub fn evil_cost() { let _ = bitty_ai_runtime::estimate_cost(1, 2, 0, 0); }\n' >>"$(endpoint_lib "$root")"
fail_case "non-test cost settlement fails" "$root"

root="$(new_fixture nontest-bare-construction)"
printf '\npub fn evil_new(id: &str) { let _ = StableId::new(id); }\n' >>"$(endpoint_lib "$root")"
fail_case "non-test bare StableId construction fails" "$root"

root="$(new_fixture seam-impl)"
printf '\nstruct Mine;\nimpl ModelProvider for Mine { }\n' >>"$(endpoint_lib "$root")"
fail_case "non-test seam-trait implementation fails" "$root"

root="$(new_fixture host-adds-dep)"
printf 'serde = "1"\n' >>"$root/crates/bitty-ai-host/Cargo.toml"
fail_case "host dependency fails the zero-dep tripwire" "$root"

root="$(new_fixture pin-drift)"
sed -i 's/rev = "e6a2c0c"/rev = "aaaaaaa"/' "$root/crates/bitty-ai-endpoint/Cargo.toml"
fail_case "pin drift fails the pin tripwire" "$root"

root="$(new_fixture unlisted-crate)"
mkdir -p "$root/crates/bitty-ai-evil/src"
printf '[package]\nname = "bitty-ai-evil"\nversion = "0.0.1"\nedition = "2024"\n' >"$root/crates/bitty-ai-evil/Cargo.toml"
printf '// evil\n' >"$root/crates/bitty-ai-evil/src/lib.rs"
fail_case "unlisted crate directory fails no-new-edge" "$root"

root="$(new_fixture unlisted-edge)"
printf 'tokio = "1"\n' >>"$root/crates/bitty-ai-endpoint/Cargo.toml"
fail_case "unlisted dependency edge fails no-new-edge" "$root"

root="$(new_fixture missing-binding)"
rm -rf "$root/crates/bitty-ai-endpoint"
fail_case "missing binding crate fails closed" "$root"

# ── real tree (must PASS: no day-one red) ────────────────────────────────────
CASE_COUNT=$((CASE_COUNT + 1))
if bash "$GATE" >/dev/null 2>&1; then
	printf 'ok %d - gate green on the real tree (pass as designed)\n' "$CASE_COUNT"
else
	printf 'not ok %d - gate green on the real tree (expected PASS, gate failed)\n' "$CASE_COUNT"
	FAILURES=$((FAILURES + 1))
fi

printf 'host-boundary-gate-v2-test: %d cases, %d failures\n' "$CASE_COUNT" "$FAILURES"
((FAILURES == 0))

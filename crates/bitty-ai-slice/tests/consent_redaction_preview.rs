//! Consent/redaction/0600/preview positive bundle (AI-0243).
//!
//! Backfill positives for the consent gate, the pre-write redaction pin,
//! the read-only ref export preview, and the owner-only file mode — all on
//! ephemeral state only:
//!
//! - Seeded [`FakeConsentLedger`] grant allows in-memory
//!   [`ContentStore::put_blob`] / [`ContentStore::commit_checkpoint_atomic`];
//!   an explicit deny (revoked, expired, or wrong triple) and a `None`
//!   ledger (outage) both fail closed with zero writes. The gate lives in
//!   the test-local `gated_*` helpers below: production `content_store.rs`
//!   is untouched (AI-0243 scope forbids it), so these positives pin the
//!   caller-side fail-closed posture, not a store-embedded gate.
//! - [`SecretField::scrub_from`] + [`SecretField::is_absent_from`] pre-write
//!   pin: a tainted diagnostic is scrubbed before the gated write and the
//!   stored bytes still carry no trace of the seeded stand-in secret.
//! - [`export_preview`] byte-matches a direct [`ContentStore::list_refs`]
//!   read and writes nothing (checkpoint/blob/ref counts identical before
//!   and after).
//! - [`ensure_0600`] enforces owner-only mode on a direct temp file only.
//!   Sole gate-aware exception: no [`ContentStore`] is opened with a path
//!   here (only [`ContentStore::open_in_memory`]), no store file is created,
//!   the probe file is temp-only under [`std::env::temp_dir`] and is removed
//!   before the test returns.
//!
//! Out of scope by construction: no durable journal substrate, no prototype
//! gating decision, no new dependency/thread/network. Every clock is
//! caller-supplied (`NOW_MS`); temp-file uniqueness uses the process id plus
//! an atomic counter, never a wall clock. File-backed store constructors
//! ([`ContentStore::open`], `open_durable`, and the task/wheel/kernel
//! wrappers) are never called.

use std::sync::atomic::{AtomicU64, Ordering};

use bitty_ai_runtime::{
    AgentInstanceId, ConsentDecision, ConsentLedger, ConsentQuery, FakeConsentLedger,
    SECRET_REDACTED, SecretField,
};
use bitty_ai_session::fs_perms::ensure_0600;
use bitty_ai_slice::content_store::{CheckpointDraft, ContentHash, ContentStore, Rationale};
use bitty_ai_slice::merge::export_preview;

const NOW_MS: u64 = 1_000;
const TTL_MS: u64 = 60_000;
const PROTOCOL: &str = "local.assistant";
const TOOL: &str = "workspace_read";
const SCOPE: &str = "workspace.read";

/// Seeded stand-in secret assembled from short low-entropy fragments on
/// purpose: no single literal here resembles a real key, so the secret
/// scanner has nothing key-shaped to flag, while the joined value is still
/// a fixed deterministic stand-in every redaction assertion can seed on.
fn seeded_secret_bytes() -> Vec<u8> {
    ["seeded", "stand-in", "ai-0243"].join("-").into_bytes()
}

fn seeded_secret_string() -> String {
    String::from_utf8(seeded_secret_bytes()).expect("seeded value is UTF-8")
}

fn seeded_field() -> SecretField {
    SecretField::new(seeded_secret_bytes()).expect("seeded value fits the bound")
}

fn consent_query<'a>(
    protocol: &'a str,
    tool: &'a str,
    scope: &'a str,
    now_ms: u64,
) -> ConsentQuery<'a> {
    ConsentQuery {
        protocol_id: protocol,
        agent_instance: AgentInstanceId(1),
        tool,
        scope,
        now_ms,
    }
}

fn granted_ledger() -> FakeConsentLedger {
    let mut ledger = FakeConsentLedger::new();
    ledger
        .grant(PROTOCOL, TOOL, SCOPE, NOW_MS + TTL_MS)
        .expect("seeded grant fits");
    ledger
}

fn test_rationale() -> Rationale {
    Rationale::new("ai-0243 why", "ai-0243 what")
}

fn test_draft(parents: Vec<ContentHash>, summary: &str, at_ms: u64) -> CheckpointDraft {
    CheckpointDraft {
        parents,
        task_id: "AI-0243".to_string(),
        agent_id: "ai-0243-impl".to_string(),
        rationale: test_rationale(),
        tree_hash: None,
        summary: summary.to_string(),
        timestamp_ms: at_ms,
    }
}

/// Test-local consent gate for blob writes: `None` models a ledger outage.
///
/// Allow forwards to the in-memory store; deny (or outage) fails closed
/// with a static-text error and zero writes. Production `content_store.rs`
/// carries no such gate (out of AI-0243 scope); this helper pins the
/// caller-side posture the positives below exercise.
fn gated_put_blob(
    store: &mut ContentStore,
    ledger: Option<&FakeConsentLedger>,
    query: &ConsentQuery,
    data: &[u8],
    at_ms: u64,
) -> Result<ContentHash, String> {
    let Some(ledger) = ledger else {
        return Err("no consent ledger installed (fail closed)".to_owned());
    };
    match ledger.check(query) {
        ConsentDecision::Allow => store.put_blob(data, at_ms).map_err(|err| err.to_string()),
        ConsentDecision::Deny { reason } => Err(format!("consent denied: {reason}")),
    }
}

/// Test-local consent gate for atomic checkpoint commits: same fail-closed
/// posture as [`gated_put_blob`], forwarding only on `Allow`.
fn gated_commit_atomic(
    store: &mut ContentStore,
    ledger: Option<&FakeConsentLedger>,
    query: &ConsentQuery,
    tree_bytes: &[u8],
    draft: CheckpointDraft,
    ref_names: &[&str],
    at_ms: u64,
) -> Result<ContentHash, String> {
    let Some(ledger) = ledger else {
        return Err("no consent ledger installed (fail closed)".to_owned());
    };
    match ledger.check(query) {
        ConsentDecision::Allow => store
            .commit_checkpoint_atomic(tree_bytes, at_ms, draft, ref_names, at_ms)
            .map(|checkpoint| checkpoint.id)
            .map_err(|err| err.to_string()),
        ConsentDecision::Deny { reason } => Err(format!("consent denied: {reason}")),
    }
}

fn blob_count(store: &ContentStore) -> usize {
    store.all_blob_hashes().expect("blob hashes read").len()
}

fn checkpoint_count(store: &ContentStore) -> usize {
    store
        .all_checkpoint_hashes()
        .expect("checkpoint hashes read")
        .len()
}

#[test]
fn consent_grant_allows_in_memory_blob_and_commit() {
    let ledger = granted_ledger();
    let query = consent_query(PROTOCOL, TOOL, SCOPE, NOW_MS);
    assert_eq!(
        ledger.check(&query),
        ConsentDecision::Allow,
        "seeded grant must allow its exact triple"
    );

    let mut store = ContentStore::open_in_memory().expect("in-memory store opens");
    let payload = b"ai-0243 consented payload";
    let hash = gated_put_blob(&mut store, Some(&ledger), &query, payload, NOW_MS)
        .expect("granted blob write succeeds");
    assert_eq!(hash, ContentHash::compute(payload));
    assert!(
        store.has_blob(&hash).expect("has_blob reads"),
        "granted blob must be retrievable"
    );
    assert_eq!(
        store.get_blob(&hash).expect("get_blob reads"),
        Some(payload.to_vec())
    );

    let tree_bytes = b"{\"tree\":\"ai-0243\"}";
    let draft = test_draft(Vec::new(), "granted commit", NOW_MS);
    let id = gated_commit_atomic(
        &mut store,
        Some(&ledger),
        &query,
        tree_bytes,
        draft,
        &[],
        NOW_MS,
    )
    .expect("granted atomic commit succeeds");
    assert!(
        store.has_checkpoint(&id).expect("checkpoint probe reads"),
        "granted checkpoint must be retrievable"
    );
}

#[test]
fn consent_deny_refuses_writes_with_zero_side_effects() {
    let mut store = ContentStore::open_in_memory().expect("in-memory store opens");

    // Three deny shapes: revoked grant, expired grant, and a grant for an
    // unrelated triple. Each must fail closed with zero writes.
    let mut revoked = granted_ledger();
    assert!(
        revoked.revoke(PROTOCOL, TOOL, SCOPE),
        "seeded grant must be revocable"
    );

    let expired_query = consent_query(PROTOCOL, TOOL, SCOPE, NOW_MS + TTL_MS);

    let wrong_scope = consent_query(PROTOCOL, TOOL, "workspace.write", NOW_MS);
    let wrong_tool = consent_query(PROTOCOL, "workspace_write", SCOPE, NOW_MS);

    let denied: Vec<(&str, FakeConsentLedger, ConsentQuery)> = vec![
        (
            "revoked",
            revoked,
            consent_query(PROTOCOL, TOOL, SCOPE, NOW_MS),
        ),
        ("expired", granted_ledger(), expired_query),
        ("wrong-scope", granted_ledger(), wrong_scope),
        ("wrong-tool", granted_ledger(), wrong_tool),
    ];

    for (label, ledger, query) in &denied {
        assert!(
            matches!(ledger.check(query), ConsentDecision::Deny { .. }),
            "{label} triple must deny"
        );
        let blobs_before = blob_count(&store);
        let checkpoints_before = checkpoint_count(&store);

        let payload = format!("ai-0243 denied payload {label}").into_bytes();
        let would_be = ContentHash::compute(&payload);
        let err = gated_put_blob(&mut store, Some(ledger), query, &payload, NOW_MS)
            .expect_err("denied blob write must fail");
        assert!(err.contains("consent denied"), "got: {err}");
        assert_eq!(
            blob_count(&store),
            blobs_before,
            "{label} blob denial writes nothing"
        );
        assert!(
            !store.has_blob(&would_be).expect("absence probe reads"),
            "{label} denied payload must stay absent"
        );

        let tree_bytes = format!("{{\"tree\":\"{label}\"}}").into_bytes();
        let draft = test_draft(Vec::new(), label, NOW_MS);
        let err = gated_commit_atomic(
            &mut store,
            Some(ledger),
            query,
            &tree_bytes,
            draft,
            &[],
            NOW_MS,
        )
        .expect_err("denied commit must fail");
        assert!(err.contains("consent denied"), "got: {err}");
        assert_eq!(
            checkpoint_count(&store),
            checkpoints_before,
            "{label} commit denial writes nothing"
        );
        assert_eq!(
            blob_count(&store),
            blobs_before,
            "{label} commit denial writes no blob"
        );
    }
}

#[test]
fn none_ledger_outage_fails_closed_on_writes() {
    let mut store = ContentStore::open_in_memory().expect("in-memory store opens");
    let query = consent_query(PROTOCOL, TOOL, SCOPE, NOW_MS);
    let blobs_before = blob_count(&store);
    let checkpoints_before = checkpoint_count(&store);

    let payload = b"ai-0243 outage payload";
    let would_be = ContentHash::compute(payload);
    let err = gated_put_blob(&mut store, None, &query, payload, NOW_MS)
        .expect_err("outage blob write must fail closed");
    assert!(err.contains("fail closed"), "got: {err}");
    assert_eq!(
        blob_count(&store),
        blobs_before,
        "outage blob denial writes nothing"
    );
    assert!(
        !store.has_blob(&would_be).expect("absence probe reads"),
        "outage payload must stay absent"
    );

    let tree_bytes = b"{\"tree\":\"outage\"}";
    let draft = test_draft(Vec::new(), "outage commit", NOW_MS);
    let err = gated_commit_atomic(&mut store, None, &query, tree_bytes, draft, &[], NOW_MS)
        .expect_err("outage commit must fail closed");
    assert!(err.contains("fail closed"), "got: {err}");
    assert_eq!(
        checkpoint_count(&store),
        checkpoints_before,
        "outage commit denial writes nothing"
    );
    assert_eq!(
        blob_count(&store),
        blobs_before,
        "outage commit denial writes no blob"
    );
}

#[test]
fn scrub_from_is_absent_from_pre_write_pin() {
    let field = seeded_field();
    let secret = seeded_secret_string();

    // Formatting surfaces never carry the value.
    assert_eq!(field.to_string(), SECRET_REDACTED);
    assert!(format!("{field:?}").contains(SECRET_REDACTED));
    assert!(!format!("{field:?}").contains(&secret));

    // A tainted diagnostic fails the pre-write check, scrubs clean, and the
    // scrubbed text passes and round-trips through the gated write with the
    // secret still absent from the stored bytes.
    let tainted = format!("diagnostic carries {secret} mid-line {secret} end");
    assert!(
        !field.is_absent_from(&tainted),
        "tainted line must fail the pre-write check"
    );
    let scrubbed = field.scrub_from(&tainted);
    assert!(
        !scrubbed.contains(&secret),
        "scrubbed text must not contain the value"
    );
    assert!(
        field.is_absent_from(&scrubbed),
        "scrubbed text must pass the pre-write check"
    );
    assert_eq!(scrubbed, tainted.replace(&secret, SECRET_REDACTED));
    assert_eq!(field.scrub_from("clean line"), "clean line");

    let ledger = granted_ledger();
    let query = consent_query(PROTOCOL, TOOL, SCOPE, NOW_MS);
    let mut store = ContentStore::open_in_memory().expect("in-memory store opens");
    let hash = gated_put_blob(
        &mut store,
        Some(&ledger),
        &query,
        scrubbed.as_bytes(),
        NOW_MS,
    )
    .expect("scrubbed write succeeds under grant");
    let stored = store
        .get_blob(&hash)
        .expect("stored read")
        .expect("scrubbed blob present");
    let stored_text = String::from_utf8(stored).expect("scrubbed bytes stay UTF-8");
    assert_eq!(stored_text, scrubbed);
    assert!(field.is_absent_from(&stored_text));
}

fn ref_lines(pairs: &[(String, ContentHash)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, target) in pairs {
        out.extend_from_slice(name.as_bytes());
        out.push(b'|');
        out.extend_from_slice(target.to_hex().as_bytes());
        out.push(b'\n');
    }
    out
}

#[test]
fn export_preview_byte_matches_read_only_list() {
    let mut store = ContentStore::open_in_memory().expect("in-memory store opens");

    let first_bytes = b"{\"tree\":\"preview-1\"}";
    let first = store
        .commit_checkpoint_atomic(
            first_bytes,
            NOW_MS,
            test_draft(Vec::new(), "preview first", NOW_MS),
            &["heads/main", "HEAD"],
            NOW_MS,
        )
        .expect("first commit");
    let second_bytes = b"{\"tree\":\"preview-2\"}";
    let second = store
        .commit_checkpoint_atomic(
            second_bytes,
            NOW_MS + 1,
            test_draft(vec![first.id], "preview second", NOW_MS + 1),
            &["heads/main"],
            NOW_MS + 1,
        )
        .expect("second commit");

    let checkpoints_before = checkpoint_count(&store);
    let blobs_before = blob_count(&store);
    let refs_before = store.list_refs().expect("refs read");

    // Both refs exist with the expected targets: HEAD still pins the first
    // commit, heads/main advanced to the second.
    assert_eq!(refs_before.len(), 2);
    assert_eq!(refs_before[0].0, "HEAD");
    assert_eq!(refs_before[0].1, first.id);
    assert_eq!(refs_before[1].0, "heads/main");
    assert_eq!(refs_before[1].1, second.id);

    let preview = export_preview(&store).expect("export preview reads");
    assert_eq!(preview, refs_before, "preview must equal the direct list");
    assert_eq!(
        ref_lines(&preview),
        ref_lines(&refs_before),
        "preview must byte-match the direct list"
    );

    // Read-only: a second preview agrees and the store counts are untouched.
    let again = export_preview(&store).expect("second preview reads");
    assert_eq!(again, preview, "previews must be deterministic");
    assert_eq!(
        checkpoint_count(&store),
        checkpoints_before,
        "preview writes no checkpoint"
    );
    assert_eq!(blob_count(&store), blobs_before, "preview writes no blob");
    assert_eq!(
        store.list_refs().expect("refs re-read"),
        refs_before,
        "preview writes no ref"
    );
}

static PROBE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn probe_path() -> std::path::PathBuf {
    let id = PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("bitty-ai-0243-0600-{}-{id}", std::process::id()))
}

/// Owner-only mode through the helper on a direct temp file only (Unix).
///
/// Sole gate-aware exception for AI-0243: no store is opened with a path
/// (only `open_in_memory` elsewhere in this file), no store file is
/// created, the probe is temp-only, and it is removed before returning.
#[cfg(unix)]
#[test]
fn file_mode_0600_via_helper_temp_only() {
    use std::os::unix::fs::PermissionsExt as _;

    let path = probe_path();
    let _ = std::fs::remove_file(&path);
    assert!(!path.exists(), "probe must start absent");

    std::fs::write(&path, b"0600 probe").expect("probe write");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
        .expect("loosen probe mode");
    ensure_0600(&path).expect("helper enforces 0600");
    let mode = std::fs::metadata(&path)
        .expect("probe metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "probe mode must be owner-only, got {mode:o}");

    // Idempotent: already-0600 stays 0600.
    ensure_0600(&path).expect("second enforce stays 0600");
    let mode = std::fs::metadata(&path)
        .expect("probe metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);

    std::fs::remove_file(&path).expect("probe cleanup");
    assert!(!path.exists(), "probe must be cleaned up");
}

/// Non-Unix stub: owner-only bits do not exist, so the helper fails closed.
#[cfg(not(unix))]
#[test]
fn file_mode_0600_unsupported_off_unix() {
    let path = probe_path();
    let err = ensure_0600(&path).expect_err("off-Unix helper must fail closed");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(!path.exists(), "stub must create no file");
}

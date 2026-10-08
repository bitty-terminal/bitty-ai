//! Integration tests for the refs plane (AI-0180): typed branch verbs plus
//! automatic reflog. All stores are hermetic in-memory instances except the
//! file-backed schema migration cases, which need a legacy on-disk layout.

use bitty_ai_slice::content_store::{
    CheckpointDraft, ContentHash, ContentStore, ContentStoreError, Rationale,
};
use bitty_ai_slice::session_refs::{
    BranchName, MAX_REFLOG_READ_LIMIT, MAX_REFLOG_REASON_BYTES, MIN_REFLOG_FLOOR, PruneReport,
    RefError, ReflogEntry, commit_checkpoint_with_branch, create_branch, delete_branch, get_branch,
    list_branches, prune_reflog, read_reflog, rename_branch, update_branch,
};
use bitty_ai_slice::wheel_bridge::{BridgeResponse, WheelBridge};
use bitty_ai_slice::wheel_kernel::WheelKernel;

fn commit_linear(
    store: &mut ContentStore,
    parents: Vec<ContentHash>,
    summary: &str,
    timestamp_ms: u64,
) -> ContentHash {
    store
        .commit_checkpoint(CheckpointDraft {
            parents,
            task_id: "AI-0180".to_string(),
            agent_id: "ctx-0180-impl".to_string(),
            rationale: Rationale::new("Session graph refs plane", "Exercise branch verbs"),
            tree_hash: None,
            summary: summary.to_string(),
            timestamp_ms,
        })
        .expect("commit test checkpoint")
        .id
}

#[test]
fn branch_full_lifecycle_with_reflog() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let first = commit_linear(&mut store, Vec::new(), "first", 1000);
    let second = commit_linear(&mut store, vec![first], "second", 2000);

    // Create.
    let branch = create_branch(
        &mut store,
        "heads/main",
        &first,
        "create main",
        "alice",
        1100,
    )
    .expect("create");
    assert_eq!(branch.as_str(), "heads/main");
    assert_eq!(get_branch(&store, "heads/main").expect("get"), Some(first));

    // Update.
    update_branch(
        &mut store,
        "heads/main",
        &second,
        "forward main",
        "bob",
        2100,
        false,
    )
    .expect("update");
    assert_eq!(get_branch(&store, "heads/main").expect("get"), Some(second));

    // Reflog so far: newest-first [update, create].
    let rows = read_reflog(&store, "heads/main", 10).expect("read");
    assert_eq!(rows.len(), 2);
    assert!(rows[0].seq > rows[1].seq);
    assert_eq!(
        rows[1],
        ReflogEntry {
            seq: rows[1].seq,
            ref_name: "heads/main".to_string(),
            old_hash: None,
            new_hash: first,
            reason: "create main".to_string(),
            actor: "alice".to_string(),
            at_ms: 1100,
        }
    );
    assert_eq!(rows[0].old_hash, Some(first));
    assert_eq!(rows[0].new_hash, second);
    assert_eq!(rows[0].reason, "forward main");
    assert_eq!(rows[0].actor, "bob");
    assert_eq!(rows[0].at_ms, 2100);

    // Rename: atomic delete + create + two rows.
    let moved = rename_branch(
        &mut store,
        "heads/main",
        "heads/next",
        "rename",
        "carol",
        3000,
    )
    .expect("rename");
    assert_eq!(moved, second);
    assert_eq!(get_branch(&store, "heads/main").expect("get old"), None);
    assert_eq!(
        get_branch(&store, "heads/next").expect("get new"),
        Some(second)
    );
    let names: Vec<String> = list_branches(&store)
        .expect("list")
        .into_iter()
        .map(|(name, _)| name.to_string())
        .collect();
    assert_eq!(names, vec!["heads/next".to_string()]);

    // Old-name history keeps the tombstone; new-name history starts fresh.
    let old_rows = read_reflog(&store, "heads/main", 10).expect("read old");
    assert_eq!(old_rows.len(), 3);
    let tombstone = &old_rows[0];
    assert_eq!(tombstone.old_hash, Some(second));
    assert_eq!(tombstone.new_hash, ContentHash::from_bytes([0u8; 32]));
    assert_eq!(tombstone.reason, "rename");
    assert_eq!(tombstone.actor, "carol");
    assert_eq!(tombstone.at_ms, 3000);
    let new_rows = read_reflog(&store, "heads/next", 10).expect("read new");
    assert_eq!(new_rows.len(), 1);
    assert_eq!(new_rows[0].old_hash, None);
    assert_eq!(new_rows[0].new_hash, second);

    // Delete: typed tip return, branch gone, tombstone appended.
    let deleted = delete_branch(&mut store, "heads/next", "retire", "dave", 4000).expect("delete");
    assert_eq!(deleted, second);
    assert_eq!(get_branch(&store, "heads/next").expect("get"), None);
    assert!(list_branches(&store).expect("list").is_empty());
    let retired = read_reflog(&store, "heads/next", 10).expect("read retired");
    assert_eq!(retired.len(), 2);
    assert_eq!(retired[0].old_hash, Some(second));
    assert_eq!(retired[0].new_hash, ContentHash::from_bytes([0u8; 32]));
}

#[test]
fn branch_namespace_and_existence_refusals_are_typed() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    let missing = ContentHash::compute(b"no such checkpoint");

    // Bare names are invalid; HEAD is protected; tags are reserved.
    for bad in ["main", "", "heads/", "heads"] {
        assert_eq!(
            create_branch(&mut store, bad, &tip, "r", "a", 1000).expect_err("bare must fail"),
            RefError::InvalidName,
            "name {bad:?} must be InvalidName"
        );
    }
    assert_eq!(
        create_branch(&mut store, "HEAD", &tip, "r", "a", 1000).expect_err("HEAD must fail"),
        RefError::ProtectedHead
    );
    for reserved in ["tags/v1", "refs/heads/main", "other/ns"] {
        assert_eq!(
            create_branch(&mut store, reserved, &tip, "r", "a", 1000)
                .expect_err("reserved must fail"),
            RefError::ReservedNamespace,
            "name {reserved:?} must be ReservedNamespace"
        );
    }
    // Namespace rules apply to reads and updates too.
    assert_eq!(
        get_branch(&store, "HEAD").expect_err("HEAD get"),
        RefError::ProtectedHead
    );
    assert_eq!(
        update_branch(&mut store, "tags/v1", &tip, "r", "a", 1000, false).expect_err("tags update"),
        RefError::ReservedNamespace
    );
    assert_eq!(
        delete_branch(&mut store, "HEAD", "r", "a", 1000).expect_err("HEAD delete"),
        RefError::ProtectedHead
    );

    // Missing targets reuse MissingTarget.
    let err = create_branch(&mut store, "heads/main", &missing, "r", "a", 1000)
        .expect_err("missing target");
    assert_eq!(err, RefError::MissingTarget(missing));

    // Duplicate create, missing update/delete/rename.
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
    assert_eq!(
        create_branch(&mut store, "heads/main", &tip, "again", "alice", 1001)
            .expect_err("duplicate"),
        RefError::AlreadyExists
    );
    assert_eq!(
        update_branch(&mut store, "heads/gone", &tip, "r", "a", 1000, false)
            .expect_err("missing update"),
        RefError::NotFound
    );
    assert_eq!(
        delete_branch(&mut store, "heads/gone", "r", "a", 1000).expect_err("missing delete"),
        RefError::NotFound
    );
    assert_eq!(
        rename_branch(&mut store, "heads/gone", "heads/new", "r", "a", 1000)
            .expect_err("missing rename source"),
        RefError::NotFound
    );
    assert_eq!(
        rename_branch(&mut store, "heads/main", "heads/main", "r", "a", 1000)
            .expect_err("self rename"),
        RefError::InvalidName
    );
    create_branch(&mut store, "heads/other", &tip, "create", "alice", 1000).expect("second branch");
    assert_eq!(
        rename_branch(&mut store, "heads/main", "heads/other", "r", "a", 1000)
            .expect_err("rename onto taken name"),
        RefError::AlreadyExists
    );
    // Failed renames leave both sides untouched with no extra rows.
    assert_eq!(get_branch(&store, "heads/main").expect("get"), Some(tip));
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        1
    );
}

#[test]
fn fast_forward_only_refuses_divergence_and_accepts_descendants() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let root = commit_linear(&mut store, Vec::new(), "root", 1000);
    let fork_a = commit_linear(&mut store, vec![root], "fork-a", 1100);
    let fork_b = commit_linear(&mut store, vec![root], "fork-b", 1200);
    let child_a = commit_linear(&mut store, vec![fork_a], "child-a", 1300);

    create_branch(&mut store, "heads/main", &fork_a, "create", "alice", 1400).expect("create");

    // Divergent target with ff-only: refused, ref and reflog untouched.
    let err = update_branch(
        &mut store,
        "heads/main",
        &fork_b,
        "diverge",
        "bob",
        1500,
        true,
    )
    .expect_err("divergent ff-only update must fail");
    assert!(
        matches!(err, RefError::Storage(_)),
        "ff refusal is a store-level refusal, got {err:?}"
    );
    assert_eq!(get_branch(&store, "heads/main").expect("get"), Some(fork_a));
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        1
    );

    // Descendant with ff-only: accepted with a history row.
    update_branch(
        &mut store,
        "heads/main",
        &child_a,
        "forward",
        "bob",
        1600,
        true,
    )
    .expect("descendant ff-only update");
    assert_eq!(
        get_branch(&store, "heads/main").expect("get"),
        Some(child_a)
    );
    let rows = read_reflog(&store, "heads/main", 10).expect("read");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].old_hash, Some(fork_a));
    assert_eq!(rows[0].new_hash, child_a);

    // Same call without ff-only allows the divergent move.
    update_branch(
        &mut store,
        "heads/main",
        &fork_b,
        "force",
        "bob",
        1700,
        false,
    )
    .expect("non-ff update with flag off");
    assert_eq!(get_branch(&store, "heads/main").expect("get"), Some(fork_b));
}

#[test]
fn reflog_read_is_bounded_newest_first() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
    for i in 1..=4 {
        update_branch(
            &mut store,
            "heads/main",
            &tip,
            &format!("move {i}"),
            "bob",
            1000 + i as u64,
            false,
        )
        .expect("update");
    }
    // Five rows total; limit 2 returns the two newest.
    let capped = read_reflog(&store, "heads/main", 2).expect("capped read");
    assert_eq!(capped.len(), 2);
    assert_eq!(capped[0].reason, "move 4");
    assert_eq!(capped[1].reason, "move 3");
    assert!(capped[0].seq > capped[1].seq);
    // Empty and unknown names read back empty (no error).
    assert!(
        read_reflog(&store, "heads/main", 0)
            .expect("zero")
            .is_empty()
    );
    assert!(
        read_reflog(&store, "heads/never-created", 10)
            .expect("unknown")
            .is_empty()
    );
}

#[test]
fn reflog_read_is_capped_at_the_hard_limit() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
    for i in 1..=(MAX_REFLOG_READ_LIMIT as u64 + 5) {
        update_branch(
            &mut store,
            "heads/main",
            &tip,
            "tick",
            "bob",
            1000 + i,
            false,
        )
        .expect("update");
    }
    // An unbounded request still returns at most the hard cap, newest-first.
    let rows = read_reflog(&store, "heads/main", usize::MAX).expect("huge limit");
    assert_eq!(rows.len(), MAX_REFLOG_READ_LIMIT);
    assert!(rows.windows(2).all(|pair| pair[0].seq > pair[1].seq));
}

#[test]
fn wheel_commit_checkpoint_routes_branch_through_verbs() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");
    kernel.put_slot("a.txt", b"v1", 1000).expect("slot");
    let first = kernel
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("First", "Baseline"),
            Some("heads/main"),
            1010,
        )
        .expect("first commit");
    assert_eq!(kernel.head_checkpoint(), Some(&first.id));
    assert_eq!(
        kernel.get_branch("heads/main").expect("branch"),
        Some(first.id)
    );
    // The verb path recorded history with the wheel reason/actor.
    let rows = kernel.read_reflog("heads/main", 10).expect("reflog");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].old_hash, None);
    assert_eq!(rows[0].new_hash, first.id);
    assert_eq!(rows[0].reason, "wheel commit");
    assert_eq!(rows[0].actor, "wheel-kernel");

    kernel.put_slot("a.txt", b"v2", 1020).expect("slot");
    let second = kernel
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("Second", "Advance"),
            Some("heads/main"),
            1030,
        )
        .expect("second commit");
    assert_eq!(
        kernel.get_branch("heads/main").expect("branch"),
        Some(second.id)
    );
    let rows = kernel.read_reflog("heads/main", 10).expect("reflog");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].old_hash, Some(first.id));
    assert_eq!(rows[0].new_hash, second.id);

    // Thin passthroughs reach the same plane.
    kernel
        .create_branch("heads/side", &first.id, "side", "tester", 1040)
        .expect("kernel create");
    assert_eq!(
        kernel.get_branch("heads/side").expect("side"),
        Some(first.id)
    );
    let names: Vec<String> = kernel
        .list_branches()
        .expect("list")
        .into_iter()
        .map(|(name, _)| name.to_string())
        .collect();
    assert_eq!(
        names,
        vec!["heads/main".to_string(), "heads/side".to_string()]
    );
    kernel
        .delete_branch("heads/side", "drop", "tester", 1050)
        .expect("kernel delete");
    assert_eq!(kernel.get_branch("heads/side").expect("side"), None);

    // A bad branch name fails before anything is persisted.
    let before = kernel.head_checkpoint().expect("head").to_owned();
    let err = kernel
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("Bad", "Bad branch"),
            Some("tags/v1"),
            1060,
        )
        .expect_err("reserved namespace must fail");
    assert!(matches!(err, bitty_ai_slice::facade::FacadeError::Store(_)));
    assert_eq!(kernel.head_checkpoint(), Some(&before));
}

#[test]
fn branch_name_parse_rules() {
    assert_eq!(
        BranchName::parse("heads/main").expect("valid").as_str(),
        "heads/main"
    );
    assert_eq!(
        BranchName::parse("HEAD").expect_err("HEAD"),
        RefError::ProtectedHead
    );
    assert_eq!(
        BranchName::parse("tags/v1").expect_err("tags"),
        RefError::ReservedNamespace
    );
    assert_eq!(
        BranchName::parse("main").expect_err("bare"),
        RefError::InvalidName
    );
    assert_eq!(
        BranchName::parse("heads/").expect_err("empty leaf"),
        RefError::InvalidName
    );
    // Errors never echo caller text.
    let err = BranchName::parse("main").expect_err("bare");
    assert!(!err.to_string().contains("main"));
}

fn legacy_four_table_db(path: &std::path::Path) {
    let conn = rusqlite::Connection::open(path).expect("raw open");
    conn.execute_batch(
        "CREATE TABLE blobs (
            hash TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            data BLOB NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE TABLE checkpoints (
            hash TEXT PRIMARY KEY,
            parents_json TEXT NOT NULL,
            task_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            rationale_json TEXT NOT NULL,
            tree_hash TEXT,
            summary TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE TABLE refs (
            name TEXT PRIMARY KEY,
            target_hash TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE durable_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO durable_meta (key, value) VALUES ('profile', 'durable-v1');",
    )
    .expect("legacy schema");
}

#[test]
fn legacy_store_without_reflog_migrates() {
    let dir = std::env::temp_dir().join(format!("bitty_test_refs_migrate_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let db_path = dir.join("legacy.db");
    let _ = std::fs::remove_file(&db_path);
    legacy_four_table_db(&db_path);

    let mut store = ContentStore::open(&db_path).expect("legacy store migrates");
    assert!(list_branches(&store).expect("list").is_empty());
    // The migrated store records history like any other.
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000)
        .expect("create after migrate");
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        1
    );
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn partial_schema_fails_closed() {
    let dir = std::env::temp_dir().join(format!("bitty_test_refs_partial_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);

    // Only the blobs table: partial, must refuse.
    let partial_path = dir.join("partial.db");
    let _ = std::fs::remove_file(&partial_path);
    {
        let conn = rusqlite::Connection::open(&partial_path).expect("raw open");
        conn.execute_batch(
            "CREATE TABLE blobs (
                hash TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                data BLOB NOT NULL,
                created_at_ms INTEGER NOT NULL
            );",
        )
        .expect("partial schema");
    }
    assert!(
        matches!(
            ContentStore::open(&partial_path),
            Err(ContentStoreError::Corrupt { .. })
        ),
        "blobs-only store must fail closed"
    );

    // Core triple plus a malformed reflog: fail closed, never silently kept.
    let bad_reflog_path = dir.join("bad-reflog.db");
    let _ = std::fs::remove_file(&bad_reflog_path);
    {
        let conn = rusqlite::Connection::open(&bad_reflog_path).expect("raw open");
        conn.execute_batch(
            "CREATE TABLE blobs (
                hash TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                data BLOB NOT NULL,
                created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE checkpoints (
                hash TEXT PRIMARY KEY,
                parents_json TEXT NOT NULL,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                rationale_json TEXT NOT NULL,
                tree_hash TEXT,
                summary TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE refs (
                name TEXT PRIMARY KEY,
                target_hash TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE reflog (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ref_name TEXT NOT NULL,
                old_hash TEXT,
                new_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                at_ms INTEGER NOT NULL
            );
            CREATE TABLE durable_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO durable_meta (key, value) VALUES ('profile', 'durable-v1');",
        )
        .expect("bad reflog schema");
    }
    assert!(
        matches!(
            ContentStore::open(&bad_reflog_path),
            Err(ContentStoreError::Corrupt { .. })
        ),
        "malformed reflog must fail closed"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn commit_with_branch_rolls_back_head_branch_reflog_together() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let tree_v1 = b"{\"v\":1}";
    let draft_v1 = CheckpointDraft {
        parents: Vec::new(),
        task_id: "AI-0180".to_string(),
        agent_id: "ctx-0180-impl".to_string(),
        rationale: Rationale::new("Session graph refs plane", "Baseline"),
        tree_hash: None,
        summary: "v1".to_string(),
        timestamp_ms: 1000,
    };
    let first = commit_checkpoint_with_branch(
        &mut store,
        tree_v1,
        1000,
        draft_v1,
        "heads/main",
        "init",
        "tester",
        1000,
    )
    .expect("first commit");
    assert_eq!(store.get_ref("HEAD").expect("head"), Some(first.id));
    assert_eq!(
        get_branch(&store, "heads/main").expect("branch"),
        Some(first.id)
    );
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        1
    );

    // Oversize reason fails inside the same transaction: HEAD, branch, blob,
    // checkpoint, and reflog must all roll back together.
    let tree_v2 = b"{\"v\":2}";
    let draft_v2 = CheckpointDraft {
        parents: vec![first.id],
        task_id: "AI-0180".to_string(),
        agent_id: "ctx-0180-impl".to_string(),
        rationale: Rationale::new("Session graph refs plane", "Advance"),
        tree_hash: None,
        summary: "v2".to_string(),
        timestamp_ms: 2000,
    };
    let long_reason = "r".repeat(MAX_REFLOG_REASON_BYTES + 1);
    let err = commit_checkpoint_with_branch(
        &mut store,
        tree_v2,
        2000,
        draft_v2,
        "heads/main",
        &long_reason,
        "tester",
        2000,
    )
    .expect_err("oversize reason must fail the whole commit");
    assert_eq!(err, RefError::InvalidName);
    assert_eq!(
        store.get_ref("HEAD").expect("head"),
        Some(first.id),
        "failed commit must not advance HEAD"
    );
    assert_eq!(
        get_branch(&store, "heads/main").expect("branch"),
        Some(first.id),
        "failed commit must not move the branch"
    );
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        1,
        "failed commit must append no reflog row"
    );
    let blob_v2 = ContentHash::compute(tree_v2);
    assert!(
        !store.has_blob(&blob_v2).expect("has blob"),
        "failed commit must not persist the tree blob"
    );
}

#[test]
fn ff_only_refuses_after_concurrent_branch_move() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let root = commit_linear(&mut store, Vec::new(), "root", 1000);
    let fork_a = commit_linear(&mut store, vec![root], "fork-a", 1100);
    let fork_b = commit_linear(&mut store, vec![root], "fork-b", 1200);
    let child_a = commit_linear(&mut store, vec![fork_a], "child-a", 1300);

    create_branch(&mut store, "heads/main", &fork_a, "create", "alice", 1400).expect("create");
    // Simulate a concurrent writer moving the tip between the pre-read and
    // the write transaction (another handle/path): the stale fast-forward
    // attempt below must be refused.
    update_branch(
        &mut store,
        "heads/main",
        &fork_b,
        "concurrent",
        "mallory",
        1450,
        false,
    )
    .expect("concurrent move");
    let err = update_branch(
        &mut store,
        "heads/main",
        &child_a,
        "stale ff",
        "bob",
        1500,
        true,
    )
    .expect_err("stale ff-only update must fail");
    assert!(
        matches!(err, RefError::Storage(_)),
        "stale refusal is store-level, got {err:?}"
    );
    assert!(
        err.to_string().contains("non-fast-forward"),
        "stale refusal must mention non-fast-forward, got {err}"
    );
    assert_eq!(get_branch(&store, "heads/main").expect("get"), Some(fork_b));
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        2,
        "failed stale update must append no reflog row"
    );
}

#[test]
fn prune_floor_keeps_latest_row_despite_age() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
    for i in 1..=3 {
        update_branch(
            &mut store,
            "heads/main",
            &tip,
            &format!("move {i}"),
            "bob",
            1000 + i as u64,
            false,
        )
        .expect("update");
    }
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        4
    );
    // Every row is age-eligible, but the floor keeps the newest one.
    let report = prune_reflog(&mut store, "heads/main", u64::MAX, 100, 0, 0).expect("prune");
    assert_eq!(report.pruned, 3);
    assert_eq!(report.floor_kept, MIN_REFLOG_FLOOR);
    assert_eq!(report.tombstone_survived, 0);
    let rows = read_reflog(&store, "heads/main", 10).expect("read after");
    assert_eq!(rows.len(), 1, "floor keeps exactly the latest row");
    assert_eq!(rows[0].reason, "move 3");

    // A single-row ref never prunes: floor covers the only row.
    let mut single = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut single, Vec::new(), "tip", 1000);
    create_branch(&mut single, "heads/solo", &tip, "create", "alice", 1000).expect("create");
    let report = prune_reflog(&mut single, "heads/solo", u64::MAX, 100, 0, 0).expect("prune solo");
    assert_eq!(report.pruned, 0);
    assert_eq!(report.floor_kept, 1);
    assert_eq!(
        read_reflog(&single, "heads/solo", 10)
            .expect("read solo")
            .len(),
        1
    );

    // Kernel passthrough reaches the same plane with no extra policy.
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");
    kernel.put_slot("a.txt", b"v1", 1000).expect("slot");
    kernel
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("First", "Baseline"),
            Some("heads/main"),
            1010,
        )
        .expect("commit");
    kernel.put_slot("a.txt", b"v2", 1020).expect("slot");
    kernel
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("Second", "Advance"),
            Some("heads/main"),
            1030,
        )
        .expect("commit");
    let kernel_report = kernel
        .prune_reflog("heads/main", u64::MAX, 100, 0, 0)
        .expect("kernel prune");
    assert_eq!(kernel_report.pruned, 1);
    assert_eq!(kernel_report.floor_kept, MIN_REFLOG_FLOOR);
    assert_eq!(
        kernel
            .read_reflog("heads/main", 10)
            .expect("kernel read")
            .len(),
        1
    );
}

#[test]
fn prune_tombstone_grace_survives_and_out_of_grace_prunes() {
    fn setup_with_recreate() -> ContentStore {
        let mut store = ContentStore::open_in_memory().expect("open");
        let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
        create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
        update_branch(&mut store, "heads/main", &tip, "move", "bob", 2000, false).expect("move");
        delete_branch(&mut store, "heads/main", "retire", "dave", 9000).expect("delete");
        // Recreate so the tombstone is not the floor-protected newest row.
        create_branch(&mut store, "heads/main", &tip, "recreate", "alice", 9500).expect("recreate");
        store
    }

    // In-grace tombstone survives: threshold 8000, tombstone at 9000 is kept.
    let mut store = setup_with_recreate();
    let report = prune_reflog(&mut store, "heads/main", 9500, 100, 2000, 10_000).expect("prune");
    assert_eq!(report.floor_kept, MIN_REFLOG_FLOOR);
    assert_eq!(report.tombstone_survived, 1);
    assert_eq!(report.pruned, 2, "create + move prune, tombstone survives");
    let rows = read_reflog(&store, "heads/main", 10).expect("read");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].reason, "recreate");
    assert_eq!(rows[0].at_ms, 9500);
    assert_eq!(
        rows[1].new_hash,
        ContentHash::from_bytes([0u8; 32]),
        "surviving row is the tombstone"
    );

    // Out-of-grace tombstone prunes like any other row: threshold 18000.
    let mut store = setup_with_recreate();
    let report = prune_reflog(&mut store, "heads/main", 9500, 100, 2000, 20_000).expect("prune");
    assert_eq!(report.tombstone_survived, 0);
    assert_eq!(report.pruned, 3, "tombstone out of grace is prunable");
    let rows = read_reflog(&store, "heads/main", 10).expect("read");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].reason, "recreate");
}

#[test]
fn prune_max_rows_bounds_deletion_oldest_first() {
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
    for i in 1..=4 {
        update_branch(
            &mut store,
            "heads/main",
            &tip,
            &format!("move {i}"),
            "bob",
            1000 + i as u64,
            false,
        )
        .expect("update");
    }
    // Five rows; floor protects move 4, the four older rows are eligible.
    let report = prune_reflog(&mut store, "heads/main", u64::MAX, 2, 0, 0).expect("prune");
    assert_eq!(report.pruned, 2);
    assert_eq!(report.floor_kept, MIN_REFLOG_FLOOR);
    let rows = read_reflog(&store, "heads/main", 10).expect("read");
    assert_eq!(rows.len(), 3, "bounded prune leaves three newest rows");
    assert_eq!(rows[0].reason, "move 4");
    assert_eq!(rows[1].reason, "move 3");
    assert_eq!(rows[2].reason, "move 2");
}

#[test]
fn prune_corrupt_row_fails_closed() {
    // A stored negative at_ms (u64::MAX wraps to -1 on the i64 column) is
    // corrupt: prune must fail closed with zero writes, matching read_reflog.
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", u64::MAX).expect("create");
    let err = prune_reflog(&mut store, "heads/main", u64::MAX, 100, 0, 0).expect_err("corrupt");
    assert_eq!(err, RefError::Corrupt);
    // No partial progress: a second call still fails closed instead of
    // succeeding on a trimmed table.
    let err =
        prune_reflog(&mut store, "heads/main", u64::MAX, 100, 0, 0).expect_err("still corrupt");
    assert_eq!(err, RefError::Corrupt);
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect_err("read corrupt"),
        RefError::Corrupt
    );

    // Malformed names never reach the store scan (shape only, like
    // read_reflog: bare but well-formed names such as "main" are readable and
    // prune to an empty report, while bad charset or empty names are Invalid).
    let mut clean = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut clean, Vec::new(), "tip", 1000);
    create_branch(&mut clean, "heads/main", &tip, "create", "alice", 1000).expect("create");
    let empty_report = prune_reflog(&mut clean, "heads/never-created", u64::MAX, 100, 0, 0)
        .expect("unknown well-formed name prunes nothing");
    assert_eq!(empty_report.pruned, 0);
    for bad in ["", "bad name with spaces", "bad:name"] {
        assert_eq!(
            prune_reflog(&mut clean, bad, u64::MAX, 100, 0, 0).expect_err("bad shape"),
            RefError::InvalidName,
            "name {bad:?} must be InvalidName"
        );
    }
}

fn dispatch_ok(
    bridge: &mut WheelBridge,
    command: &str,
    payload: &serde_json::Value,
) -> serde_json::Value {
    let raw = bridge.dispatch(command, &payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&raw).expect("bridge response parses");
    assert!(resp.success, "command {command} failed: {:?}", resp.error);
    resp.data.expect("response data present")
}

fn dispatch_err(bridge: &mut WheelBridge, command: &str, payload: &serde_json::Value) -> String {
    let raw = bridge.dispatch(command, &payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&raw).expect("bridge response parses");
    assert!(!resp.success, "command {command} unexpectedly succeeded");
    resp.error.expect("error present")
}

#[test]
fn bridge_reflog_prune_wrong_typed_fields_fail_closed() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let base = serde_json::json!({
        "ref_name": "heads/main",
        "older_than_ms": 5000,
        "max_rows": 10,
        "tombstone_grace_ms": 1000,
        "now_ms": 10000
    });
    // Absent or null numerics keep fail-safe defaults (no error).
    let mut nulls = base.clone();
    nulls["older_than_ms"] = serde_json::Value::Null;
    nulls["max_rows"] = serde_json::Value::Null;
    nulls["tombstone_grace_ms"] = serde_json::Value::Null;
    nulls["now_ms"] = serde_json::Value::Null;
    let data = dispatch_ok(&mut bridge, "reflog.prune", &nulls);
    assert_eq!(data["pruned"], 0, "null defaults prune nothing");
    // Present-but-wrong-typed fields fail closed naming the field.
    for (field, bad) in [
        ("ref_name", serde_json::json!(42)),
        ("older_than_ms", serde_json::json!("5000")),
        ("max_rows", serde_json::json!("10")),
        ("tombstone_grace_ms", serde_json::json!("1000")),
        ("now_ms", serde_json::json!("10000")),
    ] {
        let mut payload = base.clone();
        payload[field] = bad;
        let err = dispatch_err(&mut bridge, "reflog.prune", &payload);
        assert!(
            err.contains(field),
            "wrong-typed {field} must name the field, got: {err}"
        );
    }
    // Missing ref_name is an error, never a default.
    let err = dispatch_err(&mut bridge, "reflog.prune", &serde_json::json!({}));
    assert!(
        err.contains("ref_name"),
        "missing ref_name must error, got: {err}"
    );
}

#[test]
fn bridge_reflog_prune_default_noop_safe() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    bridge
        .kernel_mut()
        .put_slot("a.txt", b"v1", 1000)
        .expect("slot");
    bridge
        .kernel_mut()
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("First", "Baseline"),
            Some("heads/main"),
            1010,
        )
        .expect("commit");
    bridge
        .kernel_mut()
        .put_slot("a.txt", b"v2", 1020)
        .expect("slot");
    bridge
        .kernel_mut()
        .commit_checkpoint(
            bitty_ai_slice::content_store::Rationale::new("Second", "Advance"),
            Some("heads/main"),
            1030,
        )
        .expect("commit");
    let before = bridge
        .kernel()
        .read_reflog("heads/main", 10)
        .expect("read before");
    assert_eq!(before.len(), 2);

    // Bare ref_name uses fail-safe numeric defaults (0): pruning with a zero
    // bound matches nothing, so the call is a no-op.
    let data = dispatch_ok(
        &mut bridge,
        "reflog.prune",
        &serde_json::json!({
            "ref_name": "heads/main"
        }),
    );
    assert_eq!(data["pruned"], 0);
    assert_eq!(data["floor_kept"], 1);
    assert_eq!(data["tombstone_survived"], 0);
    let after = bridge
        .kernel()
        .read_reflog("heads/main", 10)
        .expect("read after");
    assert_eq!(after, before, "default prune deletes nothing");

    // Explicit zeros behave identically.
    let data = dispatch_ok(
        &mut bridge,
        "reflog.prune",
        &serde_json::json!({
            "ref_name": "heads/main",
            "older_than_ms": 0,
            "max_rows": 100,
            "tombstone_grace_ms": 0,
            "now_ms": 9999
        }),
    );
    assert_eq!(data["pruned"], 0);
    assert_eq!(
        bridge
            .kernel()
            .read_reflog("heads/main", 10)
            .expect("read")
            .len(),
        2
    );

    // Direct free-function no-op parity: zero bounds delete nothing.
    let mut store = ContentStore::open_in_memory().expect("open");
    let tip = commit_linear(&mut store, Vec::new(), "tip", 1000);
    create_branch(&mut store, "heads/main", &tip, "create", "alice", 1000).expect("create");
    let report: PruneReport = prune_reflog(&mut store, "heads/main", 0, 0, 0, 0).expect("noop");
    assert_eq!(report.pruned, 0);
    assert_eq!(
        read_reflog(&store, "heads/main", 10).expect("read").len(),
        1
    );
}

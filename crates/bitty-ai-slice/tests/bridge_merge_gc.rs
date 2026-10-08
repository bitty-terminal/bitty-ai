//! Bridge merge + GC dispatch coverage (AI-0190).
//!
//! Hermetic `open_in_memory` tests exercising `merge_commit`,
//! `collect_garbage`, and `gc_preview` through `WheelBridge::dispatch`
//! (`merge.commit` / `gc.preview` / `gc.collect`). Setup uses the kernel
//! content store directly; every merge/GC operation under test goes through
//! the bridge. Clocks are caller-supplied throughout.

use bitty_ai_slice::content_store::{Checkpoint, CheckpointDraft, ContentHash, Rationale};
use bitty_ai_slice::context_compiler::{ContextTree, EntryKind, TreeEntry};
use bitty_ai_slice::session_refs::dump_reflog_all;
use bitty_ai_slice::task_dag::{TaskDraft, TaskId};
use bitty_ai_slice::wheel_bridge::{BridgeResponse, WheelBridge};

fn test_rationale() -> Rationale {
    Rationale::new("merge why", "merge what")
}

fn slot_tree(pairs: &[(&str, &str)]) -> ContextTree {
    let mut tree = ContextTree::new();
    for (name, content) in pairs {
        let hash = ContentHash::compute(content.as_bytes());
        tree.insert(
            TreeEntry::new(*name, hash, EntryKind::Blob, content.len()).expect("valid test slot"),
        );
    }
    tree
}

fn commit_tree(
    bridge: &mut WheelBridge,
    parents: Vec<ContentHash>,
    tree: &ContextTree,
    summary: &str,
    at_ms: u64,
) -> Checkpoint {
    let bytes = tree.canonical_bytes();
    let draft = CheckpointDraft {
        parents,
        task_id: "AI-0190".to_string(),
        agent_id: "bridge-test".to_string(),
        rationale: test_rationale(),
        tree_hash: None,
        summary: summary.to_string(),
        timestamp_ms: at_ms,
    };
    bridge
        .kernel_mut()
        .content_store_mut()
        .commit_checkpoint_atomic(&bytes, at_ms, draft, &[], at_ms)
        .expect("setup commit")
}

fn merge_payload(
    ours: &ContentHash,
    theirs: &ContentHash,
    target_branch: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ours": ours.to_hex(),
        "theirs": theirs.to_hex(),
        "target_branch": target_branch,
        "task_id": "AI-0190",
        "agent_id": "bridge-test",
        "rationale": {"why": "merge why", "what": "merge what"},
        "summary": "merge summary",
        "actor": "tester",
        "reason": "merge test",
        "at_ms": 9000
    })
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct StoreSnapshot {
    checkpoints: Vec<String>,
    blobs: Vec<String>,
    refs: Vec<String>,
    reflog: Vec<String>,
}

fn snapshot(bridge: &WheelBridge) -> StoreSnapshot {
    let store = bridge.kernel().content_store();
    let checkpoints: Vec<String> = store
        .all_checkpoint_hashes()
        .expect("checkpoint hashes")
        .iter()
        .map(ContentHash::to_hex)
        .collect();
    let blobs: Vec<String> = store
        .all_blob_hashes()
        .expect("blob hashes")
        .iter()
        .map(ContentHash::to_hex)
        .collect();
    let live_refs = store.list_refs().expect("refs");
    let refs: Vec<String> = live_refs
        .iter()
        .map(|(name, target)| format!("{name}|{}", target.to_hex()))
        .collect();
    let entries = dump_reflog_all(store).expect("reflog");
    let reflog: Vec<String> = entries
        .iter()
        .map(|entry| {
            format!(
                "{}|{}|{}|{}|{}|{}|{}",
                entry.seq,
                entry.ref_name,
                entry
                    .old_hash
                    .as_ref()
                    .map(ContentHash::to_hex)
                    .as_deref()
                    .unwrap_or("-"),
                entry.new_hash.to_hex(),
                entry.reason,
                entry.actor,
                entry.at_ms
            )
        })
        .collect();
    StoreSnapshot {
        checkpoints,
        blobs,
        refs,
        reflog,
    }
}

fn setup_clean_pair(bridge: &mut WheelBridge) -> (Checkpoint, Checkpoint) {
    let base = commit_tree(
        bridge,
        Vec::new(),
        &slot_tree(&[("shared", "v0")]),
        "base",
        1000,
    );
    let ours = commit_tree(
        bridge,
        vec![base.id],
        &slot_tree(&[("shared", "v0"), ("ours-only", "o")]),
        "ours",
        2000,
    );
    let theirs = commit_tree(
        bridge,
        vec![base.id],
        &slot_tree(&[("shared", "v0"), ("theirs-only", "t")]),
        "theirs",
        3000,
    );
    (ours, theirs)
}

#[test]
fn bridge_merge_clean_two_parent_head_branch_reflog() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let (ours, theirs) = setup_clean_pair(&mut bridge);
    let payload = merge_payload(&ours.id, &theirs.id, "heads/main");
    let data = dispatch_ok(&mut bridge, "merge.commit", &payload);
    let merged_hex = data["id"].as_str().expect("merged id");
    let parents: Vec<String> = serde_json::from_value(data["parents"].clone()).expect("parents");
    assert_eq!(parents, vec![ours.id.to_hex(), theirs.id.to_hex()]);
    let merged = ContentHash::from_hex(merged_hex).expect("merged hash");
    assert_eq!(
        bridge.kernel().head_checkpoint(),
        Some(&merged),
        "kernel HEAD mirrors the merge"
    );
    assert_eq!(
        bridge
            .kernel()
            .content_store()
            .get_ref("HEAD")
            .expect("head read"),
        Some(merged)
    );
    assert_eq!(
        bridge
            .kernel()
            .get_branch("heads/main")
            .expect("branch read"),
        Some(merged)
    );
    let history = bridge
        .kernel()
        .read_reflog("heads/main", 10)
        .expect("reflog");
    assert_eq!(history.len(), 1, "creation writes exactly one reflog row");
    assert_eq!(history[0].old_hash, None);
    assert_eq!(history[0].new_hash, merged);
    let bytes = bridge
        .kernel()
        .content_store()
        .get_blob(
            &bridge
                .kernel()
                .content_store()
                .get_checkpoint(&merged)
                .expect("checkpoint read")
                .expect("checkpoint present")
                .tree_hash
                .expect("tree link"),
        )
        .expect("blob read")
        .expect("tree blob present");
    let tree = ContextTree::from_canonical_bytes(&bytes).expect("merged tree decodes");
    assert!(tree.get("shared").is_some());
    assert!(tree.get("ours-only").is_some());
    assert!(tree.get("theirs-only").is_some());
}

#[test]
fn bridge_merge_conflict_writes_nothing() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let base = commit_tree(
        &mut bridge,
        Vec::new(),
        &slot_tree(&[("s", "v0")]),
        "base",
        1000,
    );
    let ours = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("s", "v1")]),
        "ours",
        2000,
    );
    let theirs = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("s", "v2")]),
        "theirs",
        3000,
    );
    let before = snapshot(&bridge);
    let payload = merge_payload(&ours.id, &theirs.id, "heads/main");
    let err = dispatch_err(&mut bridge, "merge.commit", &payload);
    assert!(
        err.contains("merge conflicts"),
        "conflict taxonomy preserved, got: {err}"
    );
    assert_eq!(snapshot(&bridge), before, "conflict must write nothing");
}

#[test]
fn bridge_merge_disjoint_no_common_ancestor() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let left = commit_tree(
        &mut bridge,
        Vec::new(),
        &slot_tree(&[("slot", "left")]),
        "left",
        1000,
    );
    let right = commit_tree(
        &mut bridge,
        Vec::new(),
        &slot_tree(&[("slot", "right")]),
        "right",
        2000,
    );
    let before = snapshot(&bridge);
    let payload = merge_payload(&left.id, &right.id, "heads/main");
    let err = dispatch_err(&mut bridge, "merge.commit", &payload);
    assert!(
        err.contains("no common ancestor"),
        "taxonomy preserved, got: {err}"
    );
    assert_eq!(snapshot(&bridge), before, "failure must write nothing");
}

#[test]
fn bridge_merge_criss_cross_refused() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let base = commit_tree(
        &mut bridge,
        Vec::new(),
        &slot_tree(&[("shared", "v0")]),
        "base",
        1000,
    );
    let x = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("shared", "v0"), ("x", "1")]),
        "x",
        2000,
    );
    let y = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("shared", "v0"), ("y", "1")]),
        "y",
        2500,
    );
    let m1 = commit_tree(
        &mut bridge,
        vec![x.id, y.id],
        &slot_tree(&[("shared", "v0"), ("x", "1"), ("y", "1"), ("m", "1")]),
        "m1",
        3000,
    );
    let m2 = commit_tree(
        &mut bridge,
        vec![y.id, x.id],
        &slot_tree(&[("shared", "v0"), ("x", "1"), ("y", "1"), ("m", "2")]),
        "m2",
        3500,
    );
    assert_ne!(m1.id, m2.id, "merges must differ to form criss-cross");
    let before = snapshot(&bridge);
    let payload = merge_payload(&m1.id, &m2.id, "heads/main");
    let err = dispatch_err(&mut bridge, "merge.commit", &payload);
    assert!(
        err.contains("criss-cross"),
        "taxonomy preserved, got: {err}"
    );
    assert_eq!(snapshot(&bridge), before, "failure must write nothing");
}

#[test]
fn bridge_merge_stale_generation_refused_before_write() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let (ours, theirs) = setup_clean_pair(&mut bridge);
    let task_id = TaskId::new("AI-0190").expect("task id");
    let node = bridge
        .kernel_mut()
        .create_task(
            TaskDraft {
                id: task_id,
                title: "merge task".to_string(),
                description: String::new(),
                priority: 0,
                dependencies: Vec::new(),
            },
            1000,
        )
        .expect("create task");
    assert_eq!(node.generation, 0);
    let before = snapshot(&bridge);
    let mut payload = merge_payload(&ours.id, &theirs.id, "heads/main");
    payload["expected_task_generation"] = serde_json::json!(node.generation + 99);
    let err = dispatch_err(&mut bridge, "merge.commit", &payload);
    assert!(
        err.contains("stale task generation"),
        "taxonomy preserved, got: {err}"
    );
    assert_eq!(snapshot(&bridge), before, "stale fence must write nothing");
    payload["expected_task_generation"] = serde_json::json!(node.generation);
    let data = dispatch_ok(&mut bridge, "merge.commit", &payload);
    assert!(data["id"].as_str().is_some(), "fenced merge succeeds");
}

#[test]
fn bridge_gc_preview_matches_collect_with_truncated_resume() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let base = commit_tree(
        &mut bridge,
        Vec::new(),
        &slot_tree(&[("slot", "base")]),
        "base",
        1000,
    );
    let live1 = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("slot", "base"), ("live", "1")]),
        "live1",
        2000,
    );
    let live2 = commit_tree(
        &mut bridge,
        vec![live1.id],
        &slot_tree(&[("slot", "base"), ("live", "2")]),
        "live2",
        3000,
    );
    bridge
        .kernel_mut()
        .content_store_mut()
        .update_refs_atomic(&[("HEAD", &live2.id), ("heads/main", &live2.id)], 3000)
        .expect("live refs");
    let fork1 = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("slot", "base"), ("fork", "1")]),
        "fork1",
        2000,
    );
    let fork2 = commit_tree(
        &mut bridge,
        vec![fork1.id],
        &slot_tree(&[("slot", "base"), ("fork", "2")]),
        "fork2",
        2500,
    );
    let pinned = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("slot", "base"), ("pinned", "1")]),
        "pinned",
        2000,
    );
    bridge
        .kernel_mut()
        .create_branch("heads/tmp", &pinned.id, "pin", "tester", 9000)
        .expect("pin branch");
    bridge
        .kernel_mut()
        .delete_branch("heads/tmp", "unpin", "tester", 9100)
        .expect("unpin branch");
    let full = serde_json::json!({
        "now_ms": 10_000,
        "reflog_grace_ms": 2000,
        "max_deletes_per_call": 1000
    });
    let preview = dispatch_ok(&mut bridge, "gc.preview", &full);
    assert_eq!(preview["reachable_checkpoints"], 4);
    assert_eq!(preview["deleted_checkpoints"], 2);
    assert_eq!(preview["deleted_blobs"], 2);
    assert_eq!(preview["truncated"], false);
    let tiny = serde_json::json!({
        "now_ms": 10_000,
        "reflog_grace_ms": 2000,
        "max_deletes_per_call": 1
    });
    let tiny_preview = dispatch_ok(&mut bridge, "gc.preview", &tiny);
    assert_eq!(tiny_preview["truncated"], true);
    let first = dispatch_ok(&mut bridge, "gc.collect", &tiny);
    assert_eq!(first, tiny_preview, "preview must byte-match the batch");
    let mut total_checkpoints = first["deleted_checkpoints"].as_u64().expect("count");
    let mut total_blobs = first["deleted_blobs"].as_u64().expect("count");
    let mut calls = 1;
    loop {
        let report = dispatch_ok(&mut bridge, "gc.collect", &tiny);
        total_checkpoints += report["deleted_checkpoints"].as_u64().expect("count");
        total_blobs += report["deleted_blobs"].as_u64().expect("count");
        calls += 1;
        if !report["truncated"].as_bool().expect("truncated") {
            break;
        }
        assert!(calls < 10, "bounded resume must converge");
    }
    assert_eq!(total_checkpoints, 2);
    assert_eq!(total_blobs, 2);
    for kept in [base.id, live1.id, live2.id, pinned.id] {
        assert!(
            bridge
                .kernel()
                .content_store()
                .has_checkpoint(&kept)
                .expect("kept check"),
            "reachable checkpoint must survive"
        );
    }
    for gone in [fork1.id, fork2.id] {
        assert!(
            !bridge
                .kernel()
                .content_store()
                .has_checkpoint(&gone)
                .expect("gone check"),
            "abandoned fork must be pruned"
        );
    }
    assert_eq!(
        bridge
            .kernel()
            .content_store()
            .get_ref("heads/main")
            .expect("main"),
        Some(live2.id)
    );
}

#[test]
fn bridge_gc_dry_run_parity_and_reflog_intact() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let base = commit_tree(
        &mut bridge,
        Vec::new(),
        &slot_tree(&[("slot", "base")]),
        "base",
        1000,
    );
    let live = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("slot", "live")]),
        "live",
        2000,
    );
    bridge
        .kernel_mut()
        .content_store_mut()
        .update_refs_atomic(&[("HEAD", &live.id), ("heads/main", &live.id)], 2000)
        .expect("live refs");
    let gone_tip = commit_tree(
        &mut bridge,
        vec![base.id],
        &slot_tree(&[("slot", "gone")]),
        "gone",
        1500,
    );
    bridge
        .kernel_mut()
        .create_branch("heads/gone", &gone_tip.id, "create", "tester", 1600)
        .expect("gone branch");
    bridge
        .kernel_mut()
        .delete_branch("heads/gone", "remove", "tester", 1700)
        .expect("delete branch");
    let reflog_before = snapshot(&bridge).reflog;
    assert!(!reflog_before.is_empty(), "setup must leave reflog rows");
    let options = serde_json::json!({
        "now_ms": 100_000,
        "reflog_grace_ms": 1000,
        "max_deletes_per_call": 1000
    });
    let preview = dispatch_ok(&mut bridge, "gc.preview", &options);
    let mut dry = options.clone();
    dry["dry_run"] = serde_json::json!(true);
    let dry_report = dispatch_ok(&mut bridge, "gc.collect", &dry);
    assert_eq!(dry_report, preview, "dry run must equal preview");
    assert!(
        bridge
            .kernel()
            .content_store()
            .has_checkpoint(&gone_tip.id)
            .expect("fork survives dry run"),
        "dry run deletes nothing"
    );
    let destructive = dispatch_ok(&mut bridge, "gc.collect", &options);
    assert_eq!(destructive, preview, "destructive batch matches preview");
    assert!(
        !bridge
            .kernel()
            .content_store()
            .has_checkpoint(&gone_tip.id)
            .expect("gone check"),
        "unpinned tombstoned tip is pruned"
    );
    assert!(
        bridge
            .kernel()
            .content_store()
            .has_checkpoint(&live.id)
            .expect("live check"),
        "live tip survives"
    );
    assert_eq!(
        snapshot(&bridge).reflog,
        reflog_before,
        "reflog rows are never pruned"
    );
    let zero = ContentHash::from_bytes([0u8; 32]);
    assert!(
        bridge
            .kernel()
            .content_store()
            .get_blob(&zero)
            .expect("zero blob")
            .is_none(),
        "tombstone payload stays absent"
    );
    let alias = serde_json::json!({
        "now_ms": 100_000,
        "reflog_grace_ms": 1000,
        "max_deletes_per_call": 1000
    });
    let alias_report = dispatch_ok(&mut bridge, "gc.collect_garbage", &alias);
    assert_eq!(
        alias_report["truncated"], false,
        "alias verb drains to the same terminal state"
    );
}

#[test]
fn bridge_merge_clean_advances_working_tree() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let (ours, theirs) = setup_clean_pair(&mut bridge);
    assert!(
        bridge.kernel().active_tree().is_empty(),
        "working tree starts empty"
    );
    let payload = merge_payload(&ours.id, &theirs.id, "heads/main");
    let data = dispatch_ok(&mut bridge, "merge.commit", &payload);
    let merged_hex = data["id"].as_str().expect("merged id");
    let merged = ContentHash::from_hex(merged_hex).expect("merged hash");
    let tree = bridge.kernel().active_tree();
    assert!(tree.get("shared").is_some(), "merged shared survives");
    assert!(tree.get("ours-only").is_some(), "merged ours-only survives");
    assert!(
        tree.get("theirs-only").is_some(),
        "clean merge must advance working tree"
    );
    assert_eq!(
        bridge.kernel().head_checkpoint(),
        Some(&merged),
        "kernel HEAD mirrors the merge"
    );
    let commit_payload = serde_json::json!({
        "rationale": {"why": "merge why", "what": "merge what"},
        "now_ms": 9500
    });
    let commit_data = dispatch_ok(&mut bridge, "checkpoint.commit", &commit_payload);
    let next_hex = commit_data["id"].as_str().expect("commit id");
    let next = ContentHash::from_hex(next_hex).expect("next hash");
    let next_cp = bridge
        .kernel()
        .content_store()
        .get_checkpoint(&next)
        .expect("checkpoint read")
        .expect("checkpoint present");
    assert!(
        next_cp.parents.contains(&merged),
        "next commit must descend from the merge"
    );
    let tree_hash = next_cp.tree_hash.expect("tree link");
    let bytes = bridge
        .kernel()
        .content_store()
        .get_blob(&tree_hash)
        .expect("blob read")
        .expect("tree blob present");
    let next_tree = ContextTree::from_canonical_bytes(&bytes).expect("tree decodes");
    assert!(
        next_tree.get("theirs-only").is_some(),
        "next commit must keep theirs-only slots with zero loss"
    );
    assert!(
        next_tree.get("ours-only").is_some(),
        "next commit must keep ours-only slots with zero loss"
    );
}

#[test]
fn bridge_merge_dirty_refused_zero_write() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    bridge
        .kernel_mut()
        .put_slot("local", b"v", 1000)
        .expect("setup slot");
    let setup_payload = serde_json::json!({
        "rationale": {"why": "merge why", "what": "merge what"},
        "now_ms": 1500
    });
    dispatch_ok(&mut bridge, "checkpoint.commit", &setup_payload);
    let head_before = bridge
        .kernel()
        .head_checkpoint()
        .cloned()
        .expect("HEAD established");
    let (ours, theirs) = setup_clean_pair(&mut bridge);
    bridge
        .kernel_mut()
        .put_slot("uncommitted", b"dirty", 8000)
        .expect("dirty slot");
    let before = snapshot(&bridge);
    let before_head_store = bridge
        .kernel()
        .content_store()
        .get_ref("HEAD")
        .expect("head read");
    assert_eq!(
        before_head_store,
        Some(head_before),
        "durable HEAD matches kernel HEAD before merge"
    );
    let payload = merge_payload(&ours.id, &theirs.id, "heads/main");
    let err = dispatch_err(&mut bridge, "merge.commit", &payload);
    assert!(
        err.contains("uncommitted"),
        "dirty refusal must name uncommitted, got: {err}"
    );
    assert_eq!(snapshot(&bridge), before, "dirty merge must write nothing");
    assert_eq!(
        bridge.kernel().head_checkpoint(),
        Some(&head_before),
        "in-memory HEAD untouched"
    );
    assert_eq!(
        bridge
            .kernel()
            .content_store()
            .get_ref("HEAD")
            .expect("head read"),
        Some(head_before),
        "durable HEAD untouched"
    );
    let dirty = bridge
        .kernel()
        .get_slot("uncommitted")
        .expect("slot read")
        .expect("dirty slot survives");
    assert_eq!(dirty, b"dirty", "dirty payload survives refusal");
    let local = bridge
        .kernel()
        .get_slot("local")
        .expect("slot read")
        .expect("committed slot survives");
    assert_eq!(local, b"v", "committed payload survives refusal");
}

#[test]
fn bridge_merge_ours_not_head_refused_zero_write() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    bridge
        .kernel_mut()
        .put_slot("local", b"v", 1000)
        .expect("setup slot");
    let setup_payload = serde_json::json!({
        "rationale": {"why": "merge why", "what": "merge what"},
        "now_ms": 1500
    });
    dispatch_ok(&mut bridge, "checkpoint.commit", &setup_payload);
    let head_before = bridge
        .kernel()
        .head_checkpoint()
        .cloned()
        .expect("HEAD established");
    // Working tree is clean here: the commit snapshotted `local`, so the
    // dirty gate passes and only the ours-vs-HEAD guard can refuse.
    let (ours, theirs) = setup_clean_pair(&mut bridge);
    assert_ne!(ours.id, head_before, "pair must not descend from HEAD");
    let before = snapshot(&bridge);
    let payload = merge_payload(&ours.id, &theirs.id, "heads/main");
    let err = dispatch_err(&mut bridge, "merge.commit", &payload);
    assert!(
        err.contains("not HEAD"),
        "ours-diverged refusal must name HEAD, got: {err}"
    );
    assert_eq!(
        snapshot(&bridge),
        before,
        "refused merge must write nothing"
    );
    assert_eq!(
        bridge.kernel().head_checkpoint(),
        Some(&head_before),
        "in-memory HEAD untouched"
    );
    assert_eq!(
        bridge
            .kernel()
            .content_store()
            .get_ref("HEAD")
            .expect("head read"),
        Some(head_before),
        "durable HEAD untouched"
    );
    let local = bridge
        .kernel()
        .get_slot("local")
        .expect("slot read")
        .expect("HEAD-only slot survives");
    assert_eq!(local, b"v", "HEAD-only payload survives refusal");
}

#[test]
fn bridge_gc_wrong_typed_options_fail_closed() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let base = serde_json::json!({
        "now_ms": 100_000,
        "reflog_grace_ms": 1000,
        "max_deletes_per_call": 1000
    });
    // Absent or null fields keep their defaults.
    let mut nulls = base.clone();
    nulls["dry_run"] = serde_json::Value::Null;
    nulls["now_ms"] = serde_json::Value::Null;
    dispatch_ok(&mut bridge, "gc.preview", &nulls);
    // Present-but-wrong-typed fields fail closed instead of silently
    // defaulting: a mistyped dry_run must never become a destructive collect.
    for (field, bad) in [
        ("dry_run", serde_json::json!("true")),
        ("dry_run", serde_json::json!(1)),
        ("max_deletes_per_call", serde_json::json!("1")),
        ("now_ms", serde_json::json!("100")),
        ("reflog_grace_ms", serde_json::json!("100")),
    ] {
        let mut payload = base.clone();
        payload[field] = bad;
        let err = dispatch_err(&mut bridge, "gc.collect", &payload);
        assert!(
            err.contains(field),
            "wrong-typed {field} must name the field, got: {err}"
        );
    }
}

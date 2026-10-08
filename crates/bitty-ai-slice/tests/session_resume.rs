//! Session resume/fork surface tests (AI-0197).
//!
//! Hermetic coverage for the kernel `new_session` / `resume_session` /
//! `fork_branch` / `list_sessions` verbs and their `session.*` / `branch.*`
//! bridge spellings. File-backed tests use a fresh `temp_dir` scratch
//! database per test with caller-supplied clocks throughout; no wall clock,
//! no network, no threads. Scratch directories are removed at test end.

use bitty_ai_slice::content_store::Rationale;
use bitty_ai_slice::task_dag::{TaskDraft, TaskId};
use bitty_ai_slice::wheel_bridge::{BridgeResponse, WheelBridge};
use bitty_ai_slice::wheel_kernel::WheelKernel;

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bitty-ai-0197-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn test_task(id: &str) -> TaskDraft {
    TaskDraft {
        id: TaskId::new(id).expect("valid test task id"),
        title: format!("task {id}"),
        description: "session resume test task".to_owned(),
        priority: 1,
        dependencies: Vec::new(),
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
fn crash_mid_turn_resume_reports_head_and_live_generation() {
    let dir = scratch_dir("crash-resume");
    let db_path = dir.join("wheel.db");
    let head_hex: String;
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel open");
        kernel
            .create_task(test_task("sess-task-1"), 1000)
            .expect("create task");
        let task_id = TaskId::new("sess-task-1").expect("valid task id");
        let started = kernel
            .start_task(&task_id, "worker-1", 1010)
            .expect("start task");
        assert_eq!(started.generation, 1, "assign must bump generation to 1");
        kernel
            .set_active_task(Some(task_id))
            .expect("set active task");
        kernel
            .put_slot("notes.txt", b"crash-mid-turn payload", 1020)
            .expect("put slot");
        let checkpoint = kernel
            .commit_checkpoint(
                Rationale::new("Persist progress", "Save durable state"),
                Some("heads/main"),
                1030,
            )
            .expect("commit");
        head_hex = checkpoint.id.to_hex();
        assert_eq!(checkpoint.task_id, "sess-task-1");
        kernel
            .new_session("sess-crash-1", "heads/main", 1040)
            .expect("bind session");
        // Drop the kernel without further writes: the crash-mid-turn point.
        // In-memory state (active task, recent actions, working tree) is lost;
        // the durable triple (HEAD, checkpoint, tree) plus the session row
        // must survive.
    }
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel reopen");
        let report = kernel
            .resume_session("sess-crash-1", 2, 2000)
            .expect("resume after reopen");
        assert_eq!(report.checkpoint.to_hex(), head_hex);
        assert_eq!(
            report.generation, 1,
            "report must carry the live task generation, not a default"
        );
        assert!(report.pending_unknowns.is_empty());
        assert!(report.pending_log_absent);
        assert_eq!(report.fence_token, 2);
        assert_eq!(
            report.session_id.as_ref().map(|id| id.as_str()),
            Some("sess-crash-1")
        );
        // The resumed working tree continues the lineage: the payload slot
        // survives the reopen through the sync, not through memory.
        let slot = kernel
            .get_slot("notes.txt")
            .expect("slot read")
            .expect("slot restored after resume");
        assert_eq!(slot.as_slice(), b"crash-mid-turn payload");
        // Bare-branch resume is an unfenced read of the same tip.
        let by_branch = kernel
            .resume_session("heads/main", 0, 2010)
            .expect("branch resume");
        assert_eq!(by_branch.checkpoint.to_hex(), head_hex);
        assert_eq!(by_branch.session_id, None);
        assert_eq!(by_branch.fence_token, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resume_reports_zero_generation_without_task_binding() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel
        .put_slot("loose.txt", b"unassigned work", 1000)
        .expect("put slot");
    // No active task: the checkpoint records `task-unassigned`, which names
    // no task row, so the honest generation is 0.
    kernel
        .commit_checkpoint(
            Rationale::new("Loose work", "No task bound"),
            Some("heads/main"),
            1010,
        )
        .expect("commit");
    kernel
        .new_session("sess-loose-1", "heads/main", 1020)
        .expect("bind session");
    let report = kernel
        .resume_session("sess-loose-1", 2, 2000)
        .expect("resume");
    assert_eq!(report.generation, 0);
    assert!(report.pending_log_absent);
}

#[test]
fn fork_shares_parent_blobs_by_construction() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel
        .put_slot("shared.txt", b"shared payload", 1000)
        .expect("put slot");
    kernel
        .commit_checkpoint(
            Rationale::new("Base state", "Parent tip"),
            Some("heads/main"),
            1010,
        )
        .expect("commit");
    let main_tip = kernel
        .get_branch("heads/main")
        .expect("get branch")
        .expect("main tip present");
    kernel
        .fork_branch("heads/exp", &main_tip, "experiment", "tester", 1020)
        .expect("fork");
    let exp_tip = kernel
        .get_branch("heads/exp")
        .expect("get fork")
        .expect("fork tip present");
    assert_eq!(main_tip, exp_tip, "fork starts at the parent tip");
    let main_cp = kernel
        .get_checkpoint(&main_tip)
        .expect("get main")
        .expect("main checkpoint present");
    let exp_cp = kernel
        .get_checkpoint(&exp_tip)
        .expect("get exp")
        .expect("fork checkpoint present");
    assert_eq!(
        main_cp.tree_hash, exp_cp.tree_hash,
        "both tips must reach the same tree blob (no copy)"
    );
    let tree_hash = main_cp.tree_hash.expect("tree link present");
    assert!(
        kernel
            .content_store()
            .get_blob(&tree_hash)
            .expect("blob read")
            .is_some(),
        "shared tree blob must be reachable from both tips"
    );
}

#[test]
fn stale_fence_second_resume_is_refused() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel.put_slot("w.txt", b"work", 1000).expect("put slot");
    kernel
        .commit_checkpoint(Rationale::new("Work", "State"), Some("heads/main"), 1010)
        .expect("commit");
    kernel
        .new_session("sess-fence-1", "heads/main", 1020)
        .expect("bind session");
    let first = kernel
        .resume_session("sess-fence-1", 2, 2000)
        .expect("first resume");
    assert_eq!(first.fence_token, 2);
    // Equal and lower claims refuse as stale with zero row movement.
    for stale in [2, 1, 0] {
        let err = kernel
            .resume_session("sess-fence-1", stale, 2010)
            .expect_err("stale claim must refuse");
        assert!(
            err.to_string().contains("stale"),
            "stale refusal must name itself, got: {err}"
        );
    }
    // A greater claim still advances afterwards.
    let advanced = kernel
        .resume_session("sess-fence-1", 3, 2020)
        .expect("greater claim resumes");
    assert_eq!(advanced.fence_token, 3);
}

#[test]
fn duplicate_session_id_is_refused() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel.put_slot("w.txt", b"work", 1000).expect("put slot");
    kernel
        .commit_checkpoint(Rationale::new("Work", "State"), Some("heads/main"), 1010)
        .expect("commit");
    kernel
        .new_session("sess-dup-1", "heads/main", 1020)
        .expect("first bind");
    let err = kernel
        .new_session("sess-dup-1", "heads/main", 1030)
        .expect_err("duplicate session id must refuse");
    assert!(
        err.to_string().contains("already bound"),
        "duplicate refusal must name itself, got: {err}"
    );
}

#[test]
fn unknown_ref_resume_is_refused() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel.put_slot("w.txt", b"work", 1000).expect("put slot");
    kernel
        .commit_checkpoint(Rationale::new("Work", "State"), Some("heads/main"), 1010)
        .expect("commit");
    // Unknown session id (valid shape, never bound).
    let err = kernel
        .resume_session("sess-ghost-1", 5, 2000)
        .expect_err("unknown session must refuse");
    assert!(
        err.to_string().contains("not found"),
        "unknown session must report not-found, got: {err}"
    );
    // Unknown branch (valid shape, never created).
    let err = kernel
        .resume_session("heads/ghost", 5, 2000)
        .expect_err("unknown branch must refuse");
    assert!(
        err.to_string().contains("not found"),
        "unknown branch must report not-found, got: {err}"
    );
    // Malformed ref (neither namespace).
    assert!(
        kernel.resume_session("not a ref!!", 5, 2000).is_err(),
        "malformed ref must refuse"
    );
}

#[test]
fn session_list_is_global_and_ordered() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel.put_slot("w.txt", b"work", 1000).expect("put slot");
    kernel
        .commit_checkpoint(Rationale::new("Work", "State"), Some("heads/main"), 1010)
        .expect("commit");
    assert!(
        kernel.list_sessions().expect("list").is_empty(),
        "fresh database lists zero sessions"
    );
    let main_tip = kernel
        .get_branch("heads/main")
        .expect("get")
        .expect("tip present");
    kernel
        .fork_branch("heads/other", &main_tip, "second line", "tester", 1020)
        .expect("fork");
    kernel
        .new_session("sess-b", "heads/other", 1030)
        .expect("bind b");
    kernel
        .new_session("sess-a", "heads/main", 1040)
        .expect("bind a");
    let listed = kernel.list_sessions().expect("list");
    let names: Vec<&str> = listed
        .iter()
        .map(|binding| binding.session_id.as_str())
        .collect();
    assert_eq!(names, vec!["sess-a", "sess-b"]);
}

#[test]
fn dirty_working_tree_refuses_resume_with_zero_writes() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel.put_slot("w.txt", b"work", 1000).expect("put slot");
    kernel
        .commit_checkpoint(Rationale::new("Work", "State"), Some("heads/main"), 1010)
        .expect("commit");
    kernel
        .new_session("sess-dirty-1", "heads/main", 1020)
        .expect("bind session");
    // An uncommitted slot dirties the tree: both resume paths (fenced
    // session id and unfenced branch) must refuse before any store write,
    // matching the merge_commit / GC dirty gates.
    kernel
        .put_slot("draft.txt", b"uncommitted draft", 1030)
        .expect("put slot");
    for refused in ["sess-dirty-1", "heads/main"] {
        let err = kernel
            .resume_session(refused, 2, 2000)
            .expect_err("dirty resume must refuse");
        assert!(
            err.to_string().contains("uncommitted"),
            "dirty refusal must name itself, got: {err}"
        );
    }
    // The uncommitted slot survives the refusals and the session row never
    // moved: no epoch was consumed, so the same claim stays admissible.
    let slot = kernel
        .get_slot("draft.txt")
        .expect("slot read")
        .expect("dirty slot preserved across refused resumes");
    assert_eq!(slot.as_slice(), b"uncommitted draft");
    let row = kernel
        .list_sessions()
        .expect("list")
        .into_iter()
        .find(|binding| binding.session_id.as_str() == "sess-dirty-1")
        .expect("session row present");
    assert_eq!(
        row.epoch, 1,
        "dirty refusal must write nothing to the session row"
    );
    // After committing the draft the same claim succeeds.
    kernel
        .commit_checkpoint(Rationale::new("Draft", "Flush"), Some("heads/main"), 1040)
        .expect("commit");
    let report = kernel
        .resume_session("sess-dirty-1", 2, 2050)
        .expect("resume once clean");
    assert_eq!(report.fence_token, 2);
}

#[test]
fn stale_claim_refusal_leaves_epoch_untouched() {
    let mut kernel = WheelKernel::open_in_memory().expect("open");
    kernel.put_slot("w.txt", b"work", 1000).expect("put slot");
    kernel
        .commit_checkpoint(Rationale::new("Work", "State"), Some("heads/main"), 1010)
        .expect("commit");
    kernel
        .new_session("sess-fence-2", "heads/main", 1020)
        .expect("bind session");
    let first = kernel
        .resume_session("sess-fence-2", 2, 2000)
        .expect("first resume");
    assert_eq!(first.fence_token, 2);
    // A stale claim refuses and must not move the row: the admitted epoch
    // stays 2, so a greater claim still advances afterwards.
    let err = kernel
        .resume_session("sess-fence-2", 2, 2010)
        .expect_err("stale claim must refuse");
    assert!(
        err.to_string().contains("stale"),
        "stale refusal must name itself, got: {err}"
    );
    let row = kernel
        .list_sessions()
        .expect("list")
        .into_iter()
        .find(|binding| binding.session_id.as_str() == "sess-fence-2")
        .expect("session row present");
    assert_eq!(
        row.epoch, 2,
        "stale refusal must write nothing to the session row"
    );
    let advanced = kernel
        .resume_session("sess-fence-2", 3, 2020)
        .expect("greater claim resumes");
    assert_eq!(advanced.fence_token, 3);
}
#[test]
fn bridge_session_verbs_round_trip_with_strict_types() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    // Setup through the kernel: task + slot + commit so generation is live.
    bridge
        .kernel_mut()
        .create_task(test_task("bridge-task-1"), 1000)
        .expect("create task");
    let task_id = TaskId::new("bridge-task-1").expect("valid task id");
    bridge
        .kernel_mut()
        .start_task(&task_id, "worker-1", 1010)
        .expect("start task");
    bridge
        .kernel_mut()
        .set_active_task(Some(task_id))
        .expect("set active");
    bridge
        .kernel_mut()
        .put_slot("b.txt", b"bridge work", 1020)
        .expect("put slot");
    bridge
        .kernel_mut()
        .commit_checkpoint(
            Rationale::new("Bridge work", "State"),
            Some("heads/main"),
            1030,
        )
        .expect("commit");

    // session.new requires session_id + branch; missing fields fail.
    let bound = dispatch_ok(
        &mut bridge,
        "session.new",
        &serde_json::json!({
            "session_id": "sess-bridge-1",
            "branch": "heads/main",
            "now_ms": 1040
        }),
    );
    assert_eq!(bound["session_id"], "sess-bridge-1");
    assert_eq!(bound["branch"], "heads/main");
    assert_eq!(bound["epoch"], 1);
    assert_eq!(bound["generation"], 1);
    let err = dispatch_err(
        &mut bridge,
        "session.new",
        &serde_json::json!({"branch": "heads/main"}),
    );
    assert!(err.contains("session_id"), "missing id must name it: {err}");
    // Duplicate bind through the bridge refuses.
    let err = dispatch_err(
        &mut bridge,
        "session.new",
        &serde_json::json!({"session_id": "sess-bridge-1", "branch": "heads/main"}),
    );
    assert!(
        err.contains("already bound"),
        "duplicate must refuse: {err}"
    );

    // session.resume requires claim_epoch as u64: present-wrong-type errors.
    let err = dispatch_err(
        &mut bridge,
        "session.resume",
        &serde_json::json!({"ref": "sess-bridge-1", "claim_epoch": "2"}),
    );
    assert!(
        err.contains("claim_epoch"),
        "wrong-typed epoch must name the field: {err}"
    );
    let report = dispatch_ok(
        &mut bridge,
        "session.resume",
        &serde_json::json!({"ref": "sess-bridge-1", "claim_epoch": 2, "now_ms": 2000}),
    );
    assert_eq!(report["session_id"], "sess-bridge-1");
    assert_eq!(report["generation"], 1);
    assert_eq!(report["fence_token"], 2);
    assert_eq!(report["pending_unknowns"].as_array().unwrap().len(), 0);
    assert_eq!(report["pending_log_absent"], true);
    // Stale claim through the bridge refuses.
    let err = dispatch_err(
        &mut bridge,
        "session.resume",
        &serde_json::json!({"ref": "sess-bridge-1", "claim_epoch": 2}),
    );
    assert!(err.contains("stale"), "stale fence must refuse: {err}");

    // session.fork shares the tip; session.list shows the binding.
    let main_tip = bridge
        .kernel()
        .get_branch("heads/main")
        .expect("get")
        .expect("tip present");
    let forked = dispatch_ok(
        &mut bridge,
        "session.fork",
        &serde_json::json!({
            "name": "heads/bridge-exp",
            "from_tip": main_tip.to_hex(),
            "reason": "bridge experiment",
            "actor": "tester",
            "now_ms": 2010
        }),
    );
    assert_eq!(forked["name"], "heads/bridge-exp");
    assert_eq!(forked["target"], main_tip.to_hex());
    let listed = dispatch_ok(&mut bridge, "session.list", &serde_json::json!({}));
    assert_eq!(listed.as_array().unwrap().len(), 1);

    // branch.create + branch.list through the bridge (session UX minimum).
    let created = dispatch_ok(
        &mut bridge,
        "branch.create",
        &serde_json::json!({
            "name": "heads/bridge-second",
            "target": main_tip.to_hex(),
            "reason": "second line",
            "actor": "tester",
            "now_ms": 2020
        }),
    );
    assert_eq!(created["name"], "heads/bridge-second");
    let branches = dispatch_ok(&mut bridge, "branch.list", &serde_json::json!({}));
    assert_eq!(branches.as_array().unwrap().len(), 3);
    // Strict hash shape: malformed hex refuses instead of defaulting.
    let err = dispatch_err(
        &mut bridge,
        "branch.create",
        &serde_json::json!({"name": "heads/bad", "target": "not-hex"}),
    );
    assert!(!err.is_empty());

    // reflog.read observes the fork history newest (bridge UX read path).
    let entries = dispatch_ok(
        &mut bridge,
        "reflog.read",
        &serde_json::json!({"ref_name": "heads/bridge-exp", "limit": 8}),
    );
    assert_eq!(entries.as_array().unwrap().len(), 1);
    assert_eq!(entries[0]["reason"], "bridge experiment");
}

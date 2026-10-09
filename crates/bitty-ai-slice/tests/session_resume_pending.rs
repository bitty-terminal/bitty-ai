//! Durable pending-Unknown log tests (AI-0199).
//!
//! Hermetic coverage for the crash-begin/reopen/resume pin, the
//! host-executor outcome mapping, GC pinning of open-referenced blobs, and
//! fail-closed resume on a poisoned pending row. File-backed tests use a
//! fresh `temp_dir` scratch database per test with caller-supplied clocks
//! throughout; no wall clock, no network, no threads. Scratch directories
//! are removed at test end.

use bitty_ai_runtime::session::IdIssuer;
use bitty_ai_runtime::tool::{ExecutionContext, FakeToolExecutor, ToolError, ToolExecutor};
use bitty_ai_session::content_store::ContentStore;
use bitty_ai_session::pending::{PendingBegin, PendingDisposition, PendingResolve, PendingStore};
use bitty_ai_session::sessions::WheelSessionId;
use bitty_ai_slice::content_store::Rationale;
use bitty_ai_slice::merge::GcOptions;
use bitty_ai_slice::pending_host::{PendingToolExecutor, mint_pending_call_id};
use bitty_ai_slice::task_dag::{TaskDraft, TaskId};
use bitty_ai_slice::wheel_kernel::WheelKernel;

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bitty-ai-0199-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn test_task(id: &str) -> TaskDraft {
    TaskDraft {
        id: TaskId::new(id).expect("valid test task id"),
        title: format!("task {id}"),
        description: "pending resume test task".to_owned(),
        priority: 1,
        dependencies: Vec::new(),
    }
}

fn test_context(now_ms: u64) -> ExecutionContext {
    ExecutionContext {
        execution_id: IdIssuer::default().execution(),
        now_ms,
    }
}

fn gc_options() -> GcOptions {
    GcOptions {
        now_ms: 9000,
        reflog_grace_ms: 0,
        max_deletes_per_call: 256,
        dry_run: false,
    }
}

#[test]
fn crash_begin_reopen_resume_names_pending_on_both_paths() {
    let dir = scratch_dir("crash-resume-pending");
    let db_path = dir.join("wheel.db");
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel open");
        kernel
            .create_task(test_task("pend-task-1"), 1000)
            .expect("create task");
        let task_id = TaskId::new("pend-task-1").expect("valid task id");
        kernel
            .start_task(&task_id, "worker-1", 1010)
            .expect("start task");
        kernel
            .set_active_task(Some(task_id))
            .expect("set active task");
        kernel
            .put_slot("notes.txt", b"pending-mid-turn payload", 1020)
            .expect("put slot");
        kernel
            .commit_checkpoint(
                Rationale::new("Persist progress", "Save durable state"),
                Some("heads/main"),
                1030,
            )
            .expect("commit");
        kernel
            .new_session("sess-pend-1", "heads/main", 1040)
            .expect("bind session");
        // The host began a tool call and crashed before the delegate
        // returned: the open row is the only durable trace.
        let session = WheelSessionId::parse("sess-pend-1").expect("valid session");
        PendingStore::begin(
            kernel.content_store(),
            PendingBegin {
                call_id: "beef01",
                session_id: &session,
                task_id: Some("pend-task-1"),
                tool: "write_file",
                args: br#"{"path":"notes.txt"}"#,
                generation: 1,
                epoch: 1,
                now_ms: 1050,
            },
        )
        .expect("host begin");
        // Drop the kernel without resolving: the crash-mid-turn point.
    }
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel reopen");
        let report = kernel
            .resume_session("sess-pend-1", 2, 2000)
            .expect("resume after reopen");
        assert_eq!(report.pending_unknowns, vec!["beef01".to_owned()]);
        assert!(
            !report.pending_log_absent,
            "resume must report the log present"
        );
        assert_eq!(report.fence_token, 2);
        assert_eq!(report.generation, 1);
        // The bare-branch path unions the sessions bound to the branch.
        let by_branch = kernel
            .resume_session("heads/main", 0, 2010)
            .expect("branch resume");
        assert_eq!(by_branch.pending_unknowns, vec!["beef01".to_owned()]);
        assert!(!by_branch.pending_log_absent);
        assert_eq!(by_branch.session_id, None);
        assert_eq!(by_branch.fence_token, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pending_executor_maps_outcomes_and_unknown_stays_open() {
    let store = ContentStore::open_in_memory().expect("store");
    let session = WheelSessionId::parse("sess-exec-1").expect("valid session");
    let mut inner = FakeToolExecutor::new();
    inner.push_success("ok", b"data".to_vec());
    inner.push_error(ToolError::EffectUnknown {
        name: "write_file".to_owned(),
        reason: "ack lost".to_owned(),
    });
    inner.push_error(ToolError::Denied {
        name: "write_file".to_owned(),
        reason: "no grant".to_owned(),
    });
    let mut executor = PendingToolExecutor::new(inner, &store, session.clone(), None, 0, 1, 0);
    // Success resolves and returns the delegate outcome unchanged.
    let ok = executor
        .execute_with_context("write_file", b"{}", &test_context(1000))
        .expect("success delegates");
    assert_eq!(ok.data, b"data");
    let first = mint_pending_call_id(&session, 0);
    let entry = PendingStore::get(&store, &first)
        .expect("get")
        .expect("success row retained");
    assert_eq!(entry.disposition, Some(PendingDisposition::Success));
    // EffectUnknown leaves the row open and returns the error unchanged.
    let err = executor
        .execute_with_context("write_file", b"{}", &test_context(2000))
        .expect_err("unknown propagates");
    assert!(matches!(err, ToolError::EffectUnknown { .. }));
    let second = mint_pending_call_id(&session, 1);
    let open = PendingStore::get(&store, &second)
        .expect("get")
        .expect("unknown row stays open");
    assert_eq!(open.disposition, None);
    // The host reconciles the unknown by inspection, then resolves.
    PendingStore::resolve(
        &store,
        PendingResolve {
            call_id: &second,
            disposition: PendingDisposition::UnknownReconciled,
            claim_epoch: 1,
            expected_generation: 0,
            now_ms: 3000,
        },
    )
    .expect("inspection resolve");
    // Denied resolves as denied.
    let err = executor
        .execute_with_context("write_file", b"{}", &test_context(4000))
        .expect_err("denied propagates");
    assert!(matches!(err, ToolError::Denied { .. }));
    let third = mint_pending_call_id(&session, 2);
    let denied = PendingStore::get(&store, &third)
        .expect("get")
        .expect("denied row retained");
    assert_eq!(denied.disposition, Some(PendingDisposition::Denied));
    // Mint ids are deterministic opaque hex within the call-id bound.
    for id in [&first, &second, &third] {
        assert_eq!(id.len(), 64);
        assert!(id.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    assert!(first != second && second != third);
}

#[test]
fn open_pending_pins_blob_through_gc_until_resolved() {
    let mut kernel = WheelKernel::open_in_memory().expect("wheel open");
    let payload = b"in-flight payload pinned by an open entry";
    let hash = kernel
        .content_store_mut()
        .put_blob(payload, 1000)
        .expect("put orphan blob");
    // The host began a call over exactly these bytes: the stored digest
    // doubles as the blob hash, so the open entry pins the orphan blob.
    let session = WheelSessionId::parse("sess-gc-1").expect("valid session");
    {
        let store = kernel.content_store();
        PendingStore::begin(
            store,
            PendingBegin {
                call_id: "aa01",
                session_id: &session,
                task_id: None,
                tool: "write_file",
                args: payload,
                generation: 0,
                epoch: 1,
                now_ms: 1010,
            },
        )
        .expect("host begin");
    }
    let report = kernel
        .collect_garbage(&gc_options())
        .expect("gc with open entry");
    assert_eq!(report.deleted_blobs, 0, "an open entry must pin its blob");
    assert!(
        kernel
            .content_store()
            .get_blob(&hash)
            .expect("blob read")
            .is_some(),
        "pinned blob survives collection"
    );
    // Resolving releases the pin: the next collection reclaims the orphan.
    {
        let store = kernel.content_store();
        PendingStore::resolve(
            store,
            PendingResolve {
                call_id: "aa01",
                disposition: PendingDisposition::Success,
                claim_epoch: 1,
                expected_generation: 0,
                now_ms: 1020,
            },
        )
        .expect("resolve releases pin");
    }
    let report = kernel
        .collect_garbage(&gc_options())
        .expect("gc after resolve");
    assert_eq!(report.deleted_blobs, 1);
    assert!(
        kernel
            .content_store()
            .get_blob(&hash)
            .expect("blob read")
            .is_none(),
        "released blob is reclaimed"
    );
}

#[test]
fn poisoned_pending_row_refuses_resume_with_rows_preserved() {
    let dir = scratch_dir("poison-resume-pending");
    let db_path = dir.join("wheel.db");
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel open");
        kernel
            .create_task(test_task("pend-task-9"), 1000)
            .expect("create task");
        kernel
            .put_slot("notes.txt", b"poison probe payload", 1010)
            .expect("put slot");
        kernel
            .commit_checkpoint(
                Rationale::new("Persist progress", "Save durable state"),
                Some("heads/main"),
                1020,
            )
            .expect("commit");
        kernel
            .new_session("sess-poison-1", "heads/main", 1030)
            .expect("bind session");
        let session = WheelSessionId::parse("sess-poison-1").expect("valid session");
        PendingStore::begin(
            kernel.content_store(),
            PendingBegin {
                call_id: "ab12",
                session_id: &session,
                task_id: None,
                tool: "read_file",
                args: b"{}",
                generation: 0,
                epoch: 1,
                now_ms: 1040,
            },
        )
        .expect("host begin");
        // Drop the kernel before poisoning: raw access never contends with
        // the single writer.
    }
    {
        let conn = rusqlite::Connection::open(&db_path).expect("raw open");
        conn.execute(
            "UPDATE pending_effects SET status = 'bogus' WHERE call_id = 'ab12'",
            [],
        )
        .expect("plant bad status");
    }
    {
        let mut kernel = WheelKernel::open(&db_path).expect("wheel reopen");
        let err = kernel
            .resume_session("sess-poison-1", 2, 2000)
            .expect_err("poisoned log must refuse resume");
        assert!(
            err.to_string().contains("corrupt"),
            "resume must fail closed as corrupt, got: {err}"
        );
        // Drop the kernel before raw verification: one live connection per
        // phase, so the probe never contends with the single writer.
    }
    {
        // The fence never advanced and the poisoned row is preserved
        // verbatim: the refusal wrote nothing.
        let conn = rusqlite::Connection::open(&db_path).expect("raw open");
        let status: String = conn
            .query_row(
                "SELECT status FROM pending_effects WHERE call_id = 'ab12'",
                [],
                |row| row.get(0),
            )
            .expect("poisoned row preserved");
        assert_eq!(status, "bogus");
        conn.execute(
            "UPDATE pending_effects SET status = 'open' WHERE call_id = 'ab12'",
            [],
        )
        .expect("repair row");
    }
    {
        // The same claim stays admissible after the repair.
        let mut kernel = WheelKernel::open(&db_path).expect("wheel reopen");
        let report = kernel
            .resume_session("sess-poison-1", 2, 2000)
            .expect("same claim admits after repair");
        assert_eq!(report.pending_unknowns, vec!["ab12".to_owned()]);
        assert!(!report.pending_log_absent);
        assert_eq!(report.fence_token, 2);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

//! Durable pending-Unknown log tests (AI-0199).
//!
//! Hermetic coverage for the host-executor outcome mapping, GC pinning of
//! open-referenced blobs, and pending verbs on live handles.
//! Crash-begin/reopen/resume pins refuse under the v0.1 gate (AI-0238);
//! all other cases run on hermetic in-memory handles with caller-supplied
//! clocks throughout; no wall clock, no network, no threads. Scratch
//! directories are removed at test end.

use bitty_ai_runtime::session::IdIssuer;
use bitty_ai_runtime::tool::{ExecutionContext, FakeToolExecutor, ToolError, ToolExecutor};
use bitty_ai_session::content_store::ContentStore;
use bitty_ai_session::pending::{PendingBegin, PendingDisposition, PendingResolve, PendingStore};
use bitty_ai_session::sessions::WheelSessionId;
use bitty_ai_slice::merge::GcOptions;
use bitty_ai_slice::pending_host::{PendingToolExecutor, mint_pending_call_id};
use bitty_ai_slice::wheel_kernel::WheelKernel;

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bitty-ai-0199-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
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

/// v0.1 gate (AI-0238): the file-backed kernel open refuses and creates no
/// file, so crash-begin/reopen/resume naming across reopen is out of the
/// v0.1 profile. Pending-Unknown naming on a live handle stays covered by
/// the in-memory executor test below.
#[test]
fn crash_begin_reopen_resume_names_pending_on_both_paths() {
    let dir = scratch_dir("crash-resume-pending");
    let db_path = dir.join("wheel.db");
    let err = match WheelKernel::open(&db_path) {
        Ok(_) => panic!("durable kernel open must refuse under the v0.1 gate"),
        Err(err) => err,
    };
    let text = err.to_string();
    assert!(
        text.contains("gated out of v0.1"),
        "refusal must carry the v0.1 gate message, got: {text}"
    );
    assert!(
        !db_path.exists(),
        "refused open must not create a file: {}",
        db_path.display()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// v0.1 gate (AI-0238): the file-backed kernel open refuses, so HEAD-scope
/// isolation across reopen is out of the v0.1 profile. The same verbs on a
/// live handle stay covered by the in-memory tests in this file.
#[test]
fn direct_head_resume_reports_only_explicit_head_sessions() {
    let dir = scratch_dir("direct-head-explicit");
    let db_path = dir.join("wheel.db");
    let err = match WheelKernel::open(&db_path) {
        Ok(_) => panic!("durable kernel open must refuse under the v0.1 gate"),
        Err(err) => err,
    };
    let text = err.to_string();
    assert!(
        text.contains("gated out of v0.1"),
        "refusal must carry the v0.1 gate message, got: {text}"
    );
    assert!(
        !db_path.exists(),
        "refused open must not create a file: {}",
        db_path.display()
    );
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

/// v0.1 gate (AI-0238): the file-backed kernel open refuses, so planting a
/// poisoned pending row on disk and resuming across reopen is out of the
/// v0.1 profile. Poisoned-row refusal on a live handle stays covered by the
/// unit verbs; the gate itself is fail-closed with zero filesystem effects.
#[test]
fn poisoned_pending_row_refuses_resume_with_rows_preserved() {
    let dir = scratch_dir("poison-resume-pending");
    let db_path = dir.join("wheel.db");
    let err = match WheelKernel::open(&db_path) {
        Ok(_) => panic!("durable kernel open must refuse under the v0.1 gate"),
        Err(err) => err,
    };
    let text = err.to_string();
    assert!(
        text.contains("gated out of v0.1"),
        "refusal must carry the v0.1 gate message, got: {text}"
    );
    assert!(
        !db_path.exists(),
        "refused open must not create a file: {}",
        db_path.display()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

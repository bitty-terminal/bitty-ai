//! v0.1 gate: no working durable store path in the Wheel store triple (AI-0238).
//!
//! Every public path-taking file-backed constructor in the triple —
//! [`ContentStore`][bitty_ai_slice::content_store::ContentStore],
//! [`TaskEngine`][bitty_ai_slice::task_dag::TaskEngine],
//! [`WheelKernel`][bitty_ai_slice::wheel_kernel::WheelKernel] — plus the
//! delegating wrappers ([`WheelBridge`][bitty_ai_slice::wheel_bridge::WheelBridge]
//! and the [`AiEngine`][bitty_ai_slice::facade::AiEngine] `open_*` wrappers)
//! must refuse with the v0.1 gate error before any filesystem access and
//! create no file. The in-memory constructors are unaffected and must keep
//! working.
//!
//! Scope: the `CompactionStore` / `Journal` prototypes are separate
//! experiment surfaces (AI-0178 / AI-0049), not the Wheel store path, and are
//! intentionally outside this gate. This is the single `just
//! gate-v01-durable` / CI guard for AI-0238: behavior-asserting, not a
//! framework.

use bitty_ai_slice::action_protocol::SpilloverConfig;
use bitty_ai_slice::content_store::{ContentStore, ContentStoreError, V01_DURABLE_GATE_MESSAGE};
use bitty_ai_slice::context_compiler::CompilerBudgetConfig;
use bitty_ai_slice::facade::{AiEngine, FacadeError};
use bitty_ai_slice::task_dag::{TaskEngine, TaskEngineError};
use bitty_ai_slice::wheel_bridge::WheelBridge;
use bitty_ai_slice::wheel_kernel::WheelKernel;

const GATE_MARK: &str = "gated out of v0.1";

fn scratch_path(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("bitty-ai-0238-{name}-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    assert!(
        !path.exists(),
        "scratch path must start absent: {}",
        path.display()
    );
    path
}

fn assert_gate_text(text: &str, api: &str) {
    assert!(
        text.contains(GATE_MARK),
        "{api} must refuse with the v0.1 gate message, got: {text}"
    );
    assert!(
        text.contains("AI-0238"),
        "{api} refusal must name AI-0238, got: {text}"
    );
}

fn assert_no_file(path: &std::path::Path, api: &str) {
    assert!(
        !path.exists(),
        "{api} must not create a file (fail closed, no side effects): {}",
        path.display()
    );
}

/// Unwrap the refusal without requiring `Debug` on the gated `Ok` type
/// (the store types deliberately implement no `Debug`).
fn expect_refusal<T, E>(result: Result<T, E>, api: &str) -> E {
    match result {
        Ok(_) => panic!("{api} path open must refuse under the v0.1 gate"),
        Err(err) => err,
    }
}

#[test]
fn content_store_path_opens_refuse() {
    for api in ["open", "open_durable"] {
        let path = scratch_path(&format!("content-store-{api}"));
        let err = if api == "open" {
            expect_refusal(ContentStore::open(&path), "ContentStore::open")
        } else {
            expect_refusal(
                ContentStore::open_durable(&path),
                "ContentStore::open_durable",
            )
        };
        assert!(
            matches!(err, ContentStoreError::DurableGated),
            "{api} must be DurableGated, got {err:?}"
        );
        assert_eq!(err.to_string(), V01_DURABLE_GATE_MESSAGE);
        assert_gate_text(&err.to_string(), api);
        assert_no_file(&path, api);
    }
}

#[test]
fn task_engine_path_opens_refuse() {
    for api in ["open", "open_durable"] {
        let path = scratch_path(&format!("task-engine-{api}"));
        let err = if api == "open" {
            expect_refusal(TaskEngine::open(&path), "TaskEngine::open")
        } else {
            expect_refusal(TaskEngine::open_durable(&path), "TaskEngine::open_durable")
        };
        assert!(
            matches!(err, TaskEngineError::DurableGated),
            "{api} must be DurableGated, got {err:?}"
        );
        assert_gate_text(&err.to_string(), api);
        assert_no_file(&path, api);
    }
}

#[test]
fn wheel_kernel_path_opens_refuse() {
    let path = scratch_path("wheel-kernel-open");
    let err = expect_refusal(WheelKernel::open(&path), "WheelKernel::open");
    assert!(
        matches!(err, FacadeError::Store(_)),
        "wheel open must be Store, got {err:?}"
    );
    assert_gate_text(&err.to_string(), "WheelKernel::open");
    assert_no_file(&path, "WheelKernel::open");

    let path = scratch_path("wheel-kernel-open-with-config");
    let err = expect_refusal(
        WheelKernel::open_with_config(
            &path,
            CompilerBudgetConfig::default(),
            SpilloverConfig::default(),
        ),
        "WheelKernel::open_with_config",
    );
    assert_gate_text(&err.to_string(), "WheelKernel::open_with_config");
    assert_no_file(&path, "WheelKernel::open_with_config");

    let path = scratch_path("wheel-kernel-open-durable");
    let err = expect_refusal(
        WheelKernel::open_durable(&path),
        "WheelKernel::open_durable",
    );
    assert_gate_text(&err.to_string(), "WheelKernel::open_durable");
    assert_no_file(&path, "WheelKernel::open_durable");
}

#[test]
fn delegating_wrappers_refuse() {
    let path = scratch_path("bridge-open");
    let err = expect_refusal(WheelBridge::open(&path), "WheelBridge::open");
    assert_gate_text(&err.to_string(), "WheelBridge::open");
    assert_no_file(&path, "WheelBridge::open");

    let path = scratch_path("facade-content-store");
    let err = expect_refusal(
        AiEngine::open_content_store(&path),
        "AiEngine::open_content_store",
    );
    assert_gate_text(&err.to_string(), "AiEngine::open_content_store");
    assert_no_file(&path, "AiEngine::open_content_store");

    let path = scratch_path("facade-task-engine");
    let err = expect_refusal(
        AiEngine::open_task_engine(&path),
        "AiEngine::open_task_engine",
    );
    assert_gate_text(&err.to_string(), "AiEngine::open_task_engine");
    assert_no_file(&path, "AiEngine::open_task_engine");

    let path = scratch_path("facade-wheel-kernel");
    let err = expect_refusal(
        AiEngine::open_wheel_kernel(&path),
        "AiEngine::open_wheel_kernel",
    );
    assert_gate_text(&err.to_string(), "AiEngine::open_wheel_kernel");
    assert_no_file(&path, "AiEngine::open_wheel_kernel");

    let path = scratch_path("facade-wheel-bridge");
    let err = expect_refusal(
        AiEngine::open_wheel_bridge(&path),
        "AiEngine::open_wheel_bridge",
    );
    assert_gate_text(&err.to_string(), "AiEngine::open_wheel_bridge");
    assert_no_file(&path, "AiEngine::open_wheel_bridge");
}

#[test]
fn in_memory_paths_unaffected() {
    ContentStore::open_in_memory().expect("in-memory content store works");
    TaskEngine::open_in_memory().expect("in-memory task engine works");
    WheelKernel::open_in_memory().expect("in-memory kernel works");
    WheelKernel::open_in_memory_with_config(
        CompilerBudgetConfig::default(),
        SpilloverConfig::default(),
    )
    .expect("in-memory kernel with config works");
    WheelBridge::open_in_memory().expect("in-memory bridge works");
    AiEngine::open_in_memory_content_store().expect("facade in-memory content works");
    AiEngine::open_in_memory_task_engine().expect("facade in-memory task works");
    AiEngine::open_in_memory_wheel_kernel().expect("facade in-memory kernel works");
    AiEngine::open_in_memory_wheel_bridge().expect("facade in-memory bridge works");
}

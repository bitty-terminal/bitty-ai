//! Integration tests for Action / Intent / Outcome Protocol and Blob Spillover (AI-0166).

use bitty_ai_slice::action_protocol::{
    Action, ActionEngine, ActionError, ActionIntent, ActionOutcome, InMemoryBlobSink,
    MAX_ACTION_ID_BYTES, MAX_TOOL_NAME_BYTES, SpilloverConfig,
};
use bitty_ai_slice::content_store::ContentHash;
use bitty_ai_slice::context_compiler::{CompilerBudgetConfig, ContextCompiler};
use bitty_ai_slice::facade::AiEngine;

#[test]
fn action_intent_and_action_construction_validation() {
    // Valid intents
    let i_inspect = ActionIntent::Inspect("crates/bitty-ai-slice/src/lib.rs".to_string());
    assert_eq!(i_inspect.intent_type(), "inspect");
    assert!(i_inspect.validate().is_ok());

    let i_modify = ActionIntent::Modify("crates/bitty-ai-slice/src/facade.rs".to_string());
    assert_eq!(i_modify.intent_type(), "modify");
    assert!(i_modify.validate().is_ok());

    let i_exec = ActionIntent::Execute("cargo test".to_string());
    assert_eq!(i_exec.intent_type(), "execute");
    assert!(i_exec.validate().is_ok());

    let i_verify = ActionIntent::Verify("all quality gates pass".to_string());
    assert_eq!(i_verify.intent_type(), "verify");
    assert!(i_verify.validate().is_ok());

    let i_custom = ActionIntent::Custom {
        category: "benchmark".to_string(),
        description: "profile memory usage across turns".to_string(),
    };
    assert_eq!(i_custom.intent_type(), "custom");
    assert!(i_custom.validate().is_ok());

    // Invalid intents
    assert!(matches!(
        ActionIntent::Inspect("".to_string()).validate(),
        Err(ActionError::InvalidIntent(_))
    ));
    assert!(matches!(
        ActionIntent::Inspect("   ".to_string()).validate(),
        Err(ActionError::InvalidIntent(_))
    ));
    assert!(matches!(
        ActionIntent::Inspect("a".repeat(1025)).validate(),
        Err(ActionError::InvalidIntent(_))
    ));

    // Valid action
    let action = Action::new(
        "act-001",
        i_exec.clone(),
        "run_command",
        r#"{"command":"cargo test"}"#,
    )
    .expect("valid action");
    assert_eq!(action.id, "act-001");
    assert_eq!(action.tool_name, "run_command");

    // Invalid action ID characters or length
    assert!(matches!(
        Action::new("bad action id!", i_exec.clone(), "tool", "{}"),
        Err(ActionError::InvalidActionId(_))
    ));
    assert!(matches!(
        Action::new("", i_exec.clone(), "tool", "{}"),
        Err(ActionError::InvalidActionId(_))
    ));
    assert!(matches!(
        Action::new(
            "a".repeat(MAX_ACTION_ID_BYTES + 1),
            i_exec.clone(),
            "tool",
            "{}"
        ),
        Err(ActionError::InvalidActionId(_))
    ));

    // Invalid tool name
    assert!(matches!(
        Action::new("act-002", i_exec.clone(), "", "{}"),
        Err(ActionError::InvalidToolName(_))
    ));
    assert!(matches!(
        Action::new(
            "act-002",
            i_exec.clone(),
            "a".repeat(MAX_TOOL_NAME_BYTES + 1),
            "{}"
        ),
        Err(ActionError::InvalidToolName(_))
    ));

    // Invalid arguments JSON
    assert!(matches!(
        Action::new("act-003", i_exec, "tool", "{invalid-json"),
        Err(ActionError::InvalidArguments(_))
    ));
}

#[test]
fn inline_action_outcome_processing() {
    let engine = ActionEngine::default();
    let mut sink = InMemoryBlobSink::new();

    let stdout_raw = "All 15 tests passed.\nFinished in 0.25s.";
    let stderr_raw = "";

    let outcome = engine
        .process_outcome(
            "act-inline-1",
            true,
            Some(0),
            250,
            stdout_raw,
            stderr_raw,
            &mut sink,
            1000,
        )
        .expect("outcome processing should succeed");

    assert_eq!(outcome.action_id, "act-inline-1");
    assert!(outcome.success);
    assert_eq!(outcome.exit_code, Some(0));
    assert_eq!(outcome.duration_ms, 250);

    // Stdout was inline
    assert!(!outcome.stdout.is_spilled());
    assert_eq!(outcome.stdout.text(), stdout_raw);
    assert_eq!(outcome.stdout.total_bytes(), stdout_raw.len());
    assert_eq!(outcome.stdout.blob_pointer(), None);

    // Stderr was inline (empty)
    assert!(!outcome.stderr.is_spilled());
    assert_eq!(outcome.stderr.text(), "");
    assert_eq!(outcome.stderr.total_bytes(), 0);

    // No blobs were written to the sink
    assert!(sink.is_empty());
}

#[test]
fn auto_spillover_for_large_payload() {
    let config = SpilloverConfig {
        max_inline_bytes: 500, // Small limit for testing spillover
        max_preview_bytes: 100,
    };
    let engine = ActionEngine::new(config);
    let mut sink = InMemoryBlobSink::new();

    // Generate large stdout (5000 bytes)
    let large_stdout = "Line 001: Initialization...\n".to_string().repeat(180); // ~4860 bytes

    let outcome = engine
        .process_outcome(
            "act-spill-1",
            true,
            Some(0),
            1200,
            &large_stdout,
            "",
            &mut sink,
            2000,
        )
        .expect("outcome processing should succeed");

    // Stdout must be spilled
    assert!(outcome.stdout.is_spilled());
    assert_eq!(outcome.stdout.total_bytes(), large_stdout.len());

    let pointer = outcome.stdout.blob_pointer().expect("pointer must exist");
    assert_eq!(pointer.size_bytes, large_stdout.len());

    let expected_hash = ContentHash::compute(large_stdout.as_bytes());
    assert_eq!(pointer.hash, expected_hash);

    // Verify blob in sink
    assert_eq!(sink.len(), 1);
    let stored_bytes = sink
        .get_blob(&pointer.hash)
        .expect("blob must exist in sink");
    assert_eq!(stored_bytes, large_stdout.as_bytes());

    // Verify preview snippet has bounded length and contains truncation notice
    let preview = outcome.stdout.text();
    assert!(preview.contains("[... skipped "));
    assert!(preview.len() <= 200); // Bounded preview
}

#[test]
fn utf8_character_boundary_preview_safety() {
    let raw = "🦀 Rust 🦀 计算机 🦀 Antigravity 🦀 🚀 "
        .to_string()
        .repeat(50); // Large multibyte UTF-8 string

    let preview = ActionEngine::generate_preview(&raw, 40);

    // Verify string does not panic on UTF-8 char boundary and contains marker
    assert!(preview.contains("[... skipped "));
    assert!(std::str::from_utf8(preview.as_bytes()).is_ok());
}

#[test]
fn action_outcome_formatting_and_context_compiler_integration() {
    let config = SpilloverConfig {
        max_inline_bytes: 200,
        max_preview_bytes: 80,
    };
    let engine = ActionEngine::new(config);
    let mut sink = InMemoryBlobSink::new();

    // 1. Successful small action
    let o1 = engine
        .process_outcome(
            "act-read-file",
            true,
            None,
            15,
            "pub fn main() {}",
            "",
            &mut sink,
            1000,
        )
        .unwrap();

    let obs1 = o1.format_for_context();
    assert!(obs1.contains("[ACTION OUTCOME] id: act-read-file | status: success"));
    assert!(obs1.contains("stdout:\n  pub fn main() {}"));
    assert!(obs1.contains("stderr: (empty)"));

    // 2. Failed large action that spilled
    let large_err = "Error: compilation failed\nDetails: missing semicolon\n".repeat(20);
    let o2 = engine
        .process_outcome(
            "act-cargo-build",
            false,
            Some(101),
            850,
            "Building crate target...",
            &large_err,
            &mut sink,
            1005,
        )
        .unwrap();

    let obs2 = o2.format_for_context();
    assert!(
        obs2.contains("[ACTION OUTCOME] id: act-cargo-build | status: failed | exit_code: 101")
    );
    assert!(obs2.contains("stderr: [SPILLED to blob: "));

    // 3. Directly feed these observations into ContextCompiler Zone 3
    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "You are a software engineer agent.".to_string();
    compiler.turn_prompt = "Fix the compilation error.".to_string();
    compiler.uncollapsed_observations = vec![obs1, obs2];

    let budget = CompilerBudgetConfig::default();
    let compiled = compiler
        .compile(&budget)
        .expect("compilation should succeed");

    assert!(compiled.zone3_tail.contains("act-read-file"));
    assert!(compiled.zone3_tail.contains("act-cargo-build"));
    assert!(compiled.zone3_tail.contains("[SPILLED to blob:"));
}

#[test]
fn facade_process_action_outcome_with_content_store() {
    let mut store = AiEngine::open_in_memory_content_store().expect("open store");
    let config = SpilloverConfig {
        max_inline_bytes: 300,
        max_preview_bytes: 100,
    };

    let large_output = "Build log line 001\n".repeat(50); // ~950 bytes > 300

    let outcome: ActionOutcome = AiEngine::process_action_outcome(
        &config,
        "act-facade-test",
        true,
        Some(0),
        500,
        &large_output,
        "",
        &mut store,
        1700000000,
    )
    .expect("facade processing should succeed");

    assert!(outcome.stdout.is_spilled());
    let pointer = outcome.stdout.blob_pointer().expect("blob pointer");

    // Verify blob is permanently persisted in SQLite store and recoverable
    let retrieved_blob = store
        .get_blob(&pointer.hash)
        .expect("blob query must succeed")
        .expect("blob must exist in store");
    assert_eq!(retrieved_blob, large_output.as_bytes());
}

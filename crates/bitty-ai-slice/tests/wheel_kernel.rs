//! Integration tests for the unified [`WheelKernel`] facade (AI-0167).

use bitty_ai_slice::action_protocol::SpilloverConfig;
use bitty_ai_slice::content_store::Rationale;
use bitty_ai_slice::context_compiler::CompilerBudgetConfig;
use bitty_ai_slice::task_dag::{TaskDraft, TaskId, TaskStatus};
use bitty_ai_slice::wheel_kernel::WheelKernel;

#[test]
fn test_kernel_initialization_and_status() {
    let kernel = WheelKernel::open_in_memory().expect("in-memory kernel opens");
    assert!(kernel.active_task().is_none());
    assert!(kernel.head_checkpoint().is_none());
    assert_eq!(kernel.active_tree().len(), 0);
    assert_eq!(kernel.recent_actions().len(), 0);
}

#[test]
fn test_kernel_task_lifecycle_and_cascade_readiness() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    let t1_id = TaskId::new("task-01").expect("valid id");
    let t2_id = TaskId::new("task-02").expect("valid id");

    // Task 1: No dependencies
    let d1 = TaskDraft {
        id: t1_id.clone(),
        title: "Setup foundation".to_string(),
        description: "Initial scaffolding".to_string(),
        priority: 10,
        dependencies: vec![],
    };
    let n1 = kernel.create_task(d1, 1000).expect("create t1");
    assert_eq!(n1.status, TaskStatus::Ready);

    // Task 2: Depends on Task 1
    let d2 = TaskDraft {
        id: t2_id.clone(),
        title: "Build component".to_string(),
        description: "Dependent task".to_string(),
        priority: 5,
        dependencies: vec![t1_id.clone()],
    };
    let n2 = kernel.create_task(d2, 1001).expect("create t2");
    assert_eq!(n2.status, TaskStatus::Pending);

    // Topological sort
    let order = kernel.topological_sort().expect("topological sort");
    assert_eq!(order, vec![t1_id.clone(), t2_id.clone()]);

    // Start Task 1
    let running_n1 = kernel
        .start_task(&t1_id, "worker-a", 1010)
        .expect("start t1");
    assert_eq!(running_n1.status, TaskStatus::Running);
    assert_eq!(running_n1.generation, 1);

    // Complete Task 1 -> triggers cascade readiness on Task 2
    let completed_n1 = kernel
        .complete_task(&t1_id, 1, None, 1020)
        .expect("complete t1");
    assert_eq!(completed_n1.status, TaskStatus::Succeeded);

    // Verify Task 2 is now Ready
    let ready_n2 = kernel
        .get_task(&t2_id)
        .expect("query t2")
        .expect("task exists");
    assert_eq!(ready_n2.status, TaskStatus::Ready);
}

#[test]
fn test_kernel_task_failure_cascade_blocking() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    let t1_id = TaskId::new("task-root").expect("valid id");
    let t2_id = TaskId::new("task-child").expect("valid id");

    kernel
        .create_task(
            TaskDraft {
                id: t1_id.clone(),
                title: "Root".to_string(),
                description: "Root".to_string(),
                priority: 1,
                dependencies: vec![],
            },
            1000,
        )
        .expect("create t1");

    kernel
        .create_task(
            TaskDraft {
                id: t2_id.clone(),
                title: "Child".to_string(),
                description: "Child".to_string(),
                priority: 1,
                dependencies: vec![t1_id.clone()],
            },
            1001,
        )
        .expect("create t2");

    kernel.start_task(&t1_id, "worker", 1010).expect("start t1");
    kernel
        .fail_task(&t1_id, 1, "disk failure", 1020)
        .expect("fail t1");

    let child = kernel.get_task(&t2_id).unwrap().unwrap();
    assert_eq!(child.status, TaskStatus::Blocked);
}

#[test]
fn test_kernel_merkle_tree_slots() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    let data1 = b"fn run() { println!(\"bitty\"); }";
    let hash1 = kernel
        .put_slot("src/main.rs", data1, 1000)
        .expect("put slot");

    let retrieved = kernel
        .get_slot("src/main.rs")
        .expect("get slot")
        .expect("slot exists");
    assert_eq!(retrieved, data1);

    assert_eq!(kernel.active_tree().len(), 1);
    assert_eq!(kernel.list_slots().len(), 1);
    assert_eq!(kernel.list_slots()[0].name, "src/main.rs");
    assert_eq!(kernel.list_slots()[0].hash, hash1);

    // Remove slot
    let removed = kernel.remove_slot("src/main.rs");
    assert!(removed);
    assert_eq!(kernel.active_tree().len(), 0);
    assert!(kernel.get_slot("src/main.rs").unwrap().is_none());
}

#[test]
fn test_kernel_cognitive_checkpoint_chaining() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    kernel
        .put_slot("config.toml", b"[settings]\nmode = 'fast'", 1000)
        .unwrap();

    let r1 = Rationale::new("Initial architecture", "Draft configuration file");
    let cp1 = kernel
        .commit_checkpoint(r1, Some("heads/main"), 1010)
        .unwrap();
    assert_eq!(kernel.head_checkpoint(), Some(&cp1.id));

    kernel
        .put_slot("src/lib.rs", b"pub fn init() {}", 1020)
        .unwrap();

    let r2 = Rationale::new("Add library", "Created library entrypoint");
    let cp2 = kernel
        .commit_checkpoint(r2, Some("heads/main"), 1030)
        .unwrap();
    assert_eq!(kernel.head_checkpoint(), Some(&cp2.id));
    assert_eq!(cp2.parents, vec![cp1.id]);

    // Check log
    let history = kernel.log(10).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].id, cp2.id);
    assert_eq!(history[1].id, cp1.id);
}

#[test]
fn test_kernel_action_outcome_auto_spillover() {
    let spillover = SpilloverConfig {
        max_inline_bytes: 64,
        max_preview_bytes: 32,
    };
    let mut kernel =
        WheelKernel::open_in_memory_with_config(CompilerBudgetConfig::default(), spillover)
            .expect("kernel opens");

    // Short output: Inline
    let outcome1 = kernel
        .record_action_outcome("act-01", true, Some(0), 12, "cargo test ok", "", 1000)
        .unwrap();
    assert!(outcome1.stdout.is_inline());
    assert_eq!(kernel.recent_actions().len(), 1);

    // Oversized output: Spilled to ContentStore
    let long_output = "A".repeat(256);
    let outcome2 = kernel
        .record_action_outcome("act-02", true, Some(0), 150, &long_output, "", 1010)
        .unwrap();
    assert!(outcome2.stdout.is_spilled());
    assert_eq!(kernel.recent_actions().len(), 2);

    // Verify blob is stored in content store
    if let bitty_ai_slice::action_protocol::ActionPayload::Spilled { pointer, .. } =
        &outcome2.stdout
    {
        let blob = kernel
            .content_store()
            .get_blob(&pointer.hash)
            .unwrap()
            .unwrap();
        assert_eq!(blob, long_output.as_bytes());
    } else {
        panic!("expected spilled payload");
    }
}

#[test]
fn test_kernel_three_zone_context_compilation() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    // Set up active task
    let task_id = TaskId::new("ctx-task").unwrap();
    kernel
        .create_task(
            TaskDraft {
                id: task_id.clone(),
                title: "Refactor engine".to_string(),
                description: "Implement unified facade".to_string(),
                priority: 10,
                dependencies: vec![],
            },
            1000,
        )
        .unwrap();
    kernel.set_active_task(Some(task_id)).unwrap();

    // Set up slot and checkpoint
    kernel.put_slot("rules.md", b"Do not panic.", 1010).unwrap();
    let r = Rationale::new("Baseline", "Save baseline context");
    kernel.commit_checkpoint(r, None, 1020).unwrap();

    // Record an action
    kernel
        .record_action_outcome("act-99", true, Some(0), 45, "build successful", "", 1030)
        .unwrap();

    // Compile context
    let compiled = kernel
        .compile_context(
            "You are Wheel, the Bitty terminal coding assistant.",
            &["Never guess code.", "Run quality gates."],
            &["tool: git_commit(message: string)"],
            "What is our next milestone?",
        )
        .expect("compiles cleanly");

    // Verify Zone 1 (Stable Prefix)
    assert!(compiled.zone1_prefix.contains("SYSTEM INSTRUCTIONS"));
    assert!(compiled.zone1_prefix.contains("You are Wheel"));
    assert!(compiled.zone1_prefix.contains("PROJECT RULES"));
    assert!(compiled.zone1_prefix.contains("TOOL SCHEMAS"));

    // Verify Zone 2 (Structured State)
    assert!(compiled.zone2_state.contains("ACTIVE TASK"));
    assert!(compiled.zone2_state.contains("Refactor engine"));
    assert!(compiled.zone2_state.contains("COGNITIVE CHECKPOINTS"));
    assert!(compiled.zone2_state.contains("Baseline"));
    assert!(compiled.zone2_state.contains("CONTEXT SLOTS"));
    assert!(compiled.zone2_state.contains("rules.md"));

    // Verify Zone 3 (Dynamic Tail)
    assert!(compiled.zone3_tail.contains("OBSERVATION"));
    assert!(compiled.zone3_tail.contains("act-99"));
    assert!(compiled.zone3_tail.contains("build successful"));
    assert!(compiled.zone3_tail.contains("TURN PROMPT"));
    assert!(compiled.zone3_tail.contains("What is our next milestone?"));

    assert!(compiled.total_bytes > 0);
}

#[test]
fn test_kernel_persistence_across_reopen() {
    let temp_dir =
        std::env::temp_dir().join(format!("bitty_test_wheel_persist_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let db_path = temp_dir.join("wheel.db");
    let _ = std::fs::remove_file(&db_path);

    // Session 1: Create state
    {
        let mut kernel = WheelKernel::open(&db_path).expect("open persistent kernel");
        let task_id = TaskId::new("task-persisted").unwrap();
        kernel
            .create_task(
                TaskDraft {
                    id: task_id.clone(),
                    title: "Durable Task".to_string(),
                    description: "Must survive reopen".to_string(),
                    priority: 5,
                    dependencies: vec![],
                },
                1000,
            )
            .unwrap();

        kernel
            .put_slot("data.txt", b"durable payload", 1010)
            .unwrap();
        let r = Rationale::new("Persist", "Save durable state");
        let cp = kernel
            .commit_checkpoint(r, Some("heads/main"), 1020)
            .unwrap();
        assert_eq!(kernel.head_checkpoint(), Some(&cp.id));
    }

    // Session 2: Re-open from same file
    {
        let kernel = WheelKernel::open(&db_path).expect("reopen kernel");
        let task_id = TaskId::new("task-persisted").unwrap();
        let task = kernel.get_task(&task_id).unwrap().expect("task persisted");
        assert_eq!(task.title, "Durable Task");

        assert!(kernel.head_checkpoint().is_some());
        let history = kernel.log(5).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].rationale.why, "Persist");

        // Merkle context tree slot restored from HEAD checkpoint blob
        let slot = kernel
            .get_slot("data.txt")
            .unwrap()
            .expect("slot restored on reopen");
        assert_eq!(slot.as_slice(), b"durable payload");
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_kernel_task_view_dependencies() {
    let mut kernel = WheelKernel::open_in_memory().expect("kernel opens");

    let t1 = TaskId::new("kernel-task-a").unwrap();
    let t2 = TaskId::new("kernel-task-b").unwrap();

    kernel
        .create_task(
            TaskDraft {
                id: t1.clone(),
                title: "Task A".to_string(),
                description: "".to_string(),
                priority: 10,
                dependencies: vec![],
            },
            1000,
        )
        .unwrap();

    kernel
        .create_task(
            TaskDraft {
                id: t2.clone(),
                title: "Task B".to_string(),
                description: "".to_string(),
                priority: 5,
                dependencies: vec![t1.clone()],
            },
            1010,
        )
        .unwrap();

    let view_a = kernel.get_task_view(&t1).unwrap().expect("view A exists");
    assert_eq!(view_a.id, t1);
    assert!(view_a.dependencies.is_empty());

    let view_b = kernel.get_task_view(&t2).unwrap().expect("view B exists");
    assert_eq!(view_b.id, t2);
    assert_eq!(view_b.dependencies, vec![t1.clone()]);

    let views = kernel.list_task_views().unwrap();
    assert_eq!(views.len(), 2);
    assert_eq!(views[0].id, t1);
    assert!(views[0].dependencies.is_empty());
    assert_eq!(views[1].id, t2);
    assert_eq!(views[1].dependencies, vec![t1]);

    let non_existent = TaskId::new("unknown-id").unwrap();
    assert!(kernel.get_task_view(&non_existent).unwrap().is_none());
}

//! Integration tests for Task DAG engine, generation fencing, and control plane (AI-0164).

use std::str::FromStr;

use bitty_ai_slice::{
    AiEngine, ContentHash, TaskDraft, TaskEngine, TaskEngineError, TaskId, TaskStatus,
};

#[test]
fn task_id_validation_and_bounds() {
    // Valid IDs
    assert!(TaskId::new("task-1").is_ok());
    assert!(TaskId::new("CTX-0164").is_ok());
    assert!(TaskId::new("agent_1/subtask.2").is_ok());

    // Invalid IDs
    assert!(TaskId::new("").is_err());
    assert!(TaskId::new("   ").is_err());
    assert!(TaskId::new("/leading-slash").is_err());
    assert!(TaskId::new("trailing-slash/").is_err());
    assert!(TaskId::new(".leading-dot").is_err());
    assert!(TaskId::new("trailing-dot.").is_err());
    assert!(TaskId::new("double//slash").is_err());
    assert!(TaskId::new("double..dot").is_err());
    assert!(TaskId::new("invalid char").is_err());
    assert!(TaskId::new("invalid*char").is_err());

    // Oversized ID (> 128 bytes)
    let long_id = "a".repeat(129);
    assert!(TaskId::new(long_id).is_err());
}

#[test]
fn task_creation_and_lifecycle_state_machine() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    // Create root task (no dependencies) -> starts in Ready
    let t1 = engine.create_task(
        TaskDraft {
            id: TaskId::new("task-root")?,
            title: "Root Task".to_string(),
            description: "Initial task without prerequisites".to_string(),
            priority: 10,
            dependencies: vec![],
        },
        1000,
    )?;

    assert_eq!(t1.status, TaskStatus::Ready);
    assert_eq!(t1.generation, 0);
    assert_eq!(t1.assigned_agent, None);

    // Assign to worker -> Running, generation 1
    let gen1 = engine.assign_task(&t1.id, "worker-alpha", 1100)?;
    assert_eq!(gen1, 1);

    let running_node = engine.get_task(&t1.id)?;
    assert_eq!(running_node.status, TaskStatus::Running);
    assert_eq!(running_node.assigned_agent.as_deref(), Some("worker-alpha"));
    assert_eq!(running_node.generation, 1);

    // Cannot re-assign a running task
    assert!(matches!(
        engine.assign_task(&t1.id, "worker-beta", 1150),
        Err(TaskEngineError::InvalidStatusTransition { .. })
    ));

    // Complete task with dummy checkpoint hash
    let dummy_hash =
        ContentHash::from_str("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            .expect("valid hash");
    let completed_node = engine.complete_task(&t1.id, gen1, Some(dummy_hash), 1200)?;

    assert_eq!(completed_node.status, TaskStatus::Succeeded);
    assert_eq!(completed_node.checkpoint, Some(dummy_hash));

    // Terminal task cannot be completed again
    assert!(matches!(
        engine.complete_task(&t1.id, gen1, None, 1300),
        Err(TaskEngineError::InvalidStatusTransition { .. })
    ));

    Ok(())
}

#[test]
fn generation_fencing_prevents_stale_worker_completion() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    let t = engine.create_task(
        TaskDraft {
            id: TaskId::new("task-fenced")?,
            title: "Fenced Task".to_string(),
            description: "Test worker fencing".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1000,
    )?;

    // Worker 1 starts task at generation 1
    let gen_worker1 = engine.assign_task(&t.id, "worker-1", 1010)?;
    assert_eq!(gen_worker1, 1);

    // Supervisor cancels or retries the task due to worker timeout
    let cancelled = engine.cancel_task(&t.id, 1050)?;
    assert_eq!(cancelled.status, TaskStatus::Cancelled);
    assert_eq!(cancelled.generation, 2);

    // Supervisor retries the task
    let retried = engine.retry_task(&t.id, 1060)?;
    assert_eq!(retried.status, TaskStatus::Ready);
    assert_eq!(retried.generation, 3);

    // Worker 2 takes over at generation 4
    let gen_worker2 = engine.assign_task(&t.id, "worker-2", 1070)?;
    assert_eq!(gen_worker2, 4);

    // Stale Worker 1 wakes up and attempts to complete task with generation 1 -> REJECTED
    let stale_attempt = engine.complete_task(&t.id, gen_worker1, None, 1080);
    assert!(matches!(
        stale_attempt,
        Err(TaskEngineError::StaleGeneration {
            task_id: _,
            expected: 4,
            found: 1,
        })
    ));

    // Worker 2 completes task with generation 4 -> ACCEPTED
    let success = engine.complete_task(&t.id, gen_worker2, None, 1090)?;
    assert_eq!(success.status, TaskStatus::Succeeded);

    Ok(())
}

#[test]
fn dependency_dag_and_cascade_readiness() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    // Construct Diamond DAG:
    //      A (root)
    //     / \
    //    B   C
    //     \ /
    //      D
    let a_id = TaskId::new("task-a")?;
    let b_id = TaskId::new("task-b")?;
    let c_id = TaskId::new("task-c")?;
    let d_id = TaskId::new("task-d")?;

    engine.create_task(
        TaskDraft {
            id: a_id.clone(),
            title: "Task A".to_string(),
            description: "Root".to_string(),
            priority: 10,
            dependencies: vec![],
        },
        1000,
    )?;

    engine.create_task(
        TaskDraft {
            id: b_id.clone(),
            title: "Task B".to_string(),
            description: "Branch 1".to_string(),
            priority: 5,
            dependencies: vec![a_id.clone()],
        },
        1010,
    )?;

    engine.create_task(
        TaskDraft {
            id: c_id.clone(),
            title: "Task C".to_string(),
            description: "Branch 2".to_string(),
            priority: 5,
            dependencies: vec![a_id.clone()],
        },
        1020,
    )?;

    engine.create_task(
        TaskDraft {
            id: d_id.clone(),
            title: "Task D".to_string(),
            description: "Join".to_string(),
            priority: 0,
            dependencies: vec![b_id.clone(), c_id.clone()],
        },
        1030,
    )?;

    // Initial assertions:
    // A is Ready. B, C, D are Pending.
    assert_eq!(engine.get_task(&a_id)?.status, TaskStatus::Ready);
    assert_eq!(engine.get_task(&b_id)?.status, TaskStatus::Pending);
    assert_eq!(engine.get_task(&c_id)?.status, TaskStatus::Pending);
    assert_eq!(engine.get_task(&d_id)?.status, TaskStatus::Pending);

    // Ready queue has only A
    let ready = engine.ready_tasks()?;
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].id, a_id);

    // Complete A -> B and C become Ready, D remains Pending
    let gen_a = engine.assign_task(&a_id, "agent-a", 1040)?;
    engine.complete_task(&a_id, gen_a, None, 1050)?;

    assert_eq!(engine.get_task(&b_id)?.status, TaskStatus::Ready);
    assert_eq!(engine.get_task(&c_id)?.status, TaskStatus::Ready);
    assert_eq!(engine.get_task(&d_id)?.status, TaskStatus::Pending);

    // Complete B -> D is STILL Pending because C has not finished
    let gen_b = engine.assign_task(&b_id, "agent-b", 1060)?;
    engine.complete_task(&b_id, gen_b, None, 1070)?;

    assert_eq!(engine.get_task(&d_id)?.status, TaskStatus::Pending);

    // Complete C -> D now has ALL prerequisites satisfied -> becomes Ready!
    let gen_c = engine.assign_task(&c_id, "agent-c", 1080)?;
    engine.complete_task(&c_id, gen_c, None, 1090)?;

    assert_eq!(engine.get_task(&d_id)?.status, TaskStatus::Ready);

    // Complete D
    let gen_d = engine.assign_task(&d_id, "agent-d", 1100)?;
    let d_done = engine.complete_task(&d_id, gen_d, None, 1110)?;
    assert_eq!(d_done.status, TaskStatus::Succeeded);

    Ok(())
}

#[test]
fn failure_and_cancellation_cascade_blocking() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    let p_id = TaskId::new("prereq")?;
    let dep_id = TaskId::new("dependent")?;

    engine.create_task(
        TaskDraft {
            id: p_id.clone(),
            title: "Prereq".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1000,
    )?;

    engine.create_task(
        TaskDraft {
            id: dep_id.clone(),
            title: "Dep".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![p_id.clone()],
        },
        1010,
    )?;

    assert_eq!(engine.get_task(&dep_id)?.status, TaskStatus::Pending);

    // Fail prerequisite
    let generation = engine.assign_task(&p_id, "worker", 1020)?;
    engine.fail_task(&p_id, generation, "compiler error", 1030)?;

    // Dependent automatically cascaded to Blocked
    assert_eq!(engine.get_task(&dep_id)?.status, TaskStatus::Blocked);

    // Retry prerequisite
    engine.retry_task(&p_id, 1040)?;
    assert_eq!(engine.get_task(&p_id)?.status, TaskStatus::Ready);

    // Complete prerequisite -> Dependent transitions from Blocked to Ready
    let gen_new = engine.assign_task(&p_id, "worker", 1050)?;
    engine.complete_task(&p_id, gen_new, None, 1060)?;

    assert_eq!(engine.get_task(&dep_id)?.status, TaskStatus::Ready);

    Ok(())
}

#[test]
fn cycle_detection_prevents_deadlocks() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    let t1 = TaskId::new("task-1")?;
    let t2 = TaskId::new("task-2")?;
    let t3 = TaskId::new("task-3")?;

    engine.create_task(
        TaskDraft {
            id: t1.clone(),
            title: "1".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1000,
    )?;

    engine.create_task(
        TaskDraft {
            id: t2.clone(),
            title: "2".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![t1.clone()],
        },
        1010,
    )?;

    engine.create_task(
        TaskDraft {
            id: t3.clone(),
            title: "3".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![t2.clone()],
        },
        1020,
    )?;

    // Try creating cycle: 3 -> 1 (since 1 -> 2 -> 3)
    let cycle_err = engine.add_dependency(&t3, &t1);
    assert!(matches!(
        cycle_err,
        Err(TaskEngineError::CycleDetected { .. })
    ));

    // Self dependency
    let self_err = engine.add_dependency(&t1, &t1);
    assert!(matches!(self_err, Err(TaskEngineError::SelfDependency(_))));

    Ok(())
}

#[test]
fn topological_sort_and_priority_queue() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    let low_root = TaskId::new("low-root")?;
    let high_root = TaskId::new("high-root")?;
    let child_high = TaskId::new("child-high")?;

    engine.create_task(
        TaskDraft {
            id: low_root.clone(),
            title: "Low Root".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1000,
    )?;

    engine.create_task(
        TaskDraft {
            id: high_root.clone(),
            title: "High Root".to_string(),
            description: "".to_string(),
            priority: 100,
            dependencies: vec![],
        },
        1010,
    )?;

    engine.create_task(
        TaskDraft {
            id: child_high.clone(),
            title: "Child of High".to_string(),
            description: "".to_string(),
            priority: 50,
            dependencies: vec![high_root.clone()],
        },
        1020,
    )?;

    // Ready queue ordering: high_root (100) before low_root (0)
    let next = engine.next_ready_task()?.expect("ready task exists");
    assert_eq!(next.id, high_root);

    // Topological sort ordering
    let order = engine.topological_sort()?;
    assert_eq!(order.len(), 3);

    let pos_high = order.iter().position(|id| *id == high_root).unwrap();
    let pos_child = order.iter().position(|id| *id == child_high).unwrap();
    let pos_low = order.iter().position(|id| *id == low_root).unwrap();

    // high_root must precede its child
    assert!(pos_high < pos_child);
    // high_root must precede low_root because of higher priority
    assert!(pos_high < pos_low);

    Ok(())
}

#[test]
fn facade_and_sqlite_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir().join(format!("bitty-test-task-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;
    let db_path = temp_dir.join("tasks.db");

    // Open via AiEngine facade
    {
        let mut engine = AiEngine::open_task_engine(&db_path)?;
        let t_id = TaskId::new("persist-task")?;
        engine.create_task(
            TaskDraft {
                id: t_id.clone(),
                title: "Persisted Task".to_string(),
                description: "Checks disk persistence".to_string(),
                priority: 42,
                dependencies: vec![],
            },
            1000,
        )?;
        let generation = engine.assign_task(&t_id, "persistent-agent", 1010)?;
        assert_eq!(generation, 1);
    }

    // Reopen and verify persisted state
    {
        let engine = AiEngine::open_task_engine(&db_path)?;
        let t_id = TaskId::new("persist-task")?;
        let task = engine.get_task(&t_id)?;
        assert_eq!(task.title, "Persisted Task");
        assert_eq!(task.priority, 42);
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.assigned_agent.as_deref(), Some("persistent-agent"));
        assert_eq!(task.generation, 1);
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[test]
fn transitive_cascade_blocking_across_multi_hop_chain() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    // Chain: A -> B -> C -> D
    let a_id = TaskId::new("chain-a")?;
    let b_id = TaskId::new("chain-b")?;
    let c_id = TaskId::new("chain-c")?;
    let d_id = TaskId::new("chain-d")?;

    engine.create_task(
        TaskDraft {
            id: a_id.clone(),
            title: "A".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1000,
    )?;
    engine.create_task(
        TaskDraft {
            id: b_id.clone(),
            title: "B".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![a_id.clone()],
        },
        1010,
    )?;
    engine.create_task(
        TaskDraft {
            id: c_id.clone(),
            title: "C".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![b_id.clone()],
        },
        1020,
    )?;
    engine.create_task(
        TaskDraft {
            id: d_id.clone(),
            title: "D".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![c_id.clone()],
        },
        1030,
    )?;

    // Initially A is Ready, B, C, D are Pending
    assert_eq!(engine.get_task(&a_id)?.status, TaskStatus::Ready);
    assert_eq!(engine.get_task(&b_id)?.status, TaskStatus::Pending);
    assert_eq!(engine.get_task(&c_id)?.status, TaskStatus::Pending);
    assert_eq!(engine.get_task(&d_id)?.status, TaskStatus::Pending);

    // Cancel A
    engine.cancel_task(&a_id, 1040)?;

    // Entire downstream chain (B, C, D) must be transitively Blocked!
    assert_eq!(engine.get_task(&a_id)?.status, TaskStatus::Cancelled);
    assert_eq!(engine.get_task(&b_id)?.status, TaskStatus::Blocked);
    assert_eq!(engine.get_task(&c_id)?.status, TaskStatus::Blocked);
    assert_eq!(engine.get_task(&d_id)?.status, TaskStatus::Blocked);

    Ok(())
}

#[test]
fn running_or_terminal_task_rejects_new_dependency() -> Result<(), TaskEngineError> {
    let mut engine = TaskEngine::open_in_memory()?;

    let t1 = TaskId::new("task-1")?;
    let t2 = TaskId::new("task-2")?;

    engine.create_task(
        TaskDraft {
            id: t1.clone(),
            title: "1".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1000,
    )?;
    engine.create_task(
        TaskDraft {
            id: t2.clone(),
            title: "2".to_string(),
            description: "".to_string(),
            priority: 0,
            dependencies: vec![],
        },
        1010,
    )?;

    // t1 is running
    engine.assign_task(&t1, "agent-1", 1020)?;

    // Cannot add dependency to running task t1
    let err = engine.add_dependency(&t2, &t1);
    assert!(matches!(
        err,
        Err(TaskEngineError::InvalidStatusTransition { .. })
    ));

    Ok(())
}

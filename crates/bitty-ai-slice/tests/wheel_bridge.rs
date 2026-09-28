//! Integration tests for the [`WheelBridge`] JSON-RPC boundary dispatch (AI-0167).

use bitty_ai_slice::wheel_bridge::{BridgeResponse, WheelBridge};

#[test]
fn test_bridge_kernel_status() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");
    let resp_str = bridge.dispatch("kernel.status", "{}");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).expect("parses JSON");
    assert!(resp.success);
    let data = resp.data.expect("data present");
    assert_eq!(data["slot_count"], 0);
    assert_eq!(data["task_count"], 0);
    assert_eq!(data["recent_action_count"], 0);
}

#[test]
fn test_bridge_task_lifecycle() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");

    // 1. Create Task
    let create_payload = serde_json::json!({
        "id": "bridge-task-1",
        "title": "Bridge Task",
        "description": "Tested over JSON bridge",
        "priority": 5,
        "dependencies": [],
        "now_ms": 1000
    });
    let resp_str = bridge.dispatch("task.create", &create_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(resp.data.unwrap()["id"], "bridge-task-1");

    // 2. Set active task
    let set_active_payload = serde_json::json!({ "id": "bridge-task-1" });
    let resp_str = bridge.dispatch("task.set_active", &set_active_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(resp.data.unwrap()["active_task"], "bridge-task-1");

    // 3. Start task
    let start_payload = serde_json::json!({
        "id": "bridge-task-1",
        "worker_id": "worker-bridge",
        "now_ms": 1010
    });
    let resp_str = bridge.dispatch("task.start", &start_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let task = resp.data.unwrap();
    assert_eq!(task["status"], "running");
    let generation = task["generation"].as_u64().unwrap();

    // 4. Complete task
    let complete_payload = serde_json::json!({
        "id": "bridge-task-1",
        "expected_generation": generation,
        "now_ms": 1020
    });
    let resp_str = bridge.dispatch("task.complete", &complete_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(resp.data.unwrap()["status"], "succeeded");
}

#[test]
fn test_bridge_slot_operations() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");

    // Put slot
    let put_payload = serde_json::json!({
        "name": "docs/architecture.md",
        "content": "# Wheel Architecture\nConvergence of wheels.",
        "now_ms": 1000
    });
    let resp_str = bridge.dispatch("slot.put", &put_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let slot_data = resp.data.unwrap();
    assert_eq!(slot_data["name"], "docs/architecture.md");
    assert!(slot_data["hash"].as_str().is_some());

    // Get slot
    let get_payload = serde_json::json!({ "name": "docs/architecture.md" });
    let resp_str = bridge.dispatch("slot.get", &get_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let retrieved = resp.data.unwrap();
    assert_eq!(retrieved["found"], true);
    assert!(
        retrieved["content"]
            .as_str()
            .unwrap()
            .contains("Wheel Architecture")
    );

    // List slots
    let resp_str = bridge.dispatch("slot.list", "{}");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let list = resp.data.unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);

    // Remove slot
    let remove_payload = serde_json::json!({ "name": "docs/architecture.md" });
    let resp_str = bridge.dispatch("slot.remove", &remove_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(resp.data.unwrap()["removed"], true);
}

#[test]
fn test_bridge_checkpoint_operations() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");

    // Commit checkpoint
    let cp_payload = serde_json::json!({
        "rationale": {
            "why": "Bridge test checkpoint",
            "what": "Verifying JSON-RPC commit"
        },
        "branch": "heads/main",
        "now_ms": 1000
    });
    let resp_str = bridge.dispatch("checkpoint.commit", &cp_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let cp_data = resp.data.unwrap();
    let hash = cp_data["id"].as_str().unwrap();

    // Get checkpoint
    let get_payload = serde_json::json!({ "hash": hash });
    let resp_str = bridge.dispatch("checkpoint.get", &get_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(
        resp.data.unwrap()["rationale"]["why"],
        "Bridge test checkpoint"
    );

    // Log checkpoints
    let log_payload = serde_json::json!({ "max_depth": 5 });
    let resp_str = bridge.dispatch("checkpoint.log", &log_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(resp.data.unwrap().as_array().unwrap().len(), 1);
}

#[test]
fn test_bridge_action_and_context_compile() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");

    // Record action
    let act_payload = serde_json::json!({
        "action_id": "act-bridge-1",
        "success": true,
        "exit_code": 0,
        "duration_ms": 35,
        "raw_stdout": "bridge test output",
        "raw_stderr": "",
        "timestamp_ms": 1000
    });
    let resp_str = bridge.dispatch("action.record", &act_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);

    // Compile context
    let compile_payload = serde_json::json!({
        "system_instruction": "You are Wheel AI.",
        "project_rules": ["Strict bounds", "Clean JSON"],
        "tool_schemas": ["tool: test()"],
        "turn_prompt": "Synthesize status."
    });
    let resp_str = bridge.dispatch("context.compile", &compile_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let compiled = resp.data.unwrap();
    assert!(
        compiled["prompt_string"]
            .as_str()
            .unwrap()
            .contains("You are Wheel AI")
    );
    assert!(
        compiled["prompt_string"]
            .as_str()
            .unwrap()
            .contains("bridge test output")
    );
    assert!(
        compiled["prompt_string"]
            .as_str()
            .unwrap()
            .contains("Synthesize status.")
    );
}

#[test]
fn test_bridge_error_handling() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");

    // Invalid JSON
    let resp_str = bridge.dispatch("kernel.status", "{ invalid json");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(!resp.success);
    assert!(resp.error.unwrap().contains("invalid JSON"));

    // Unknown command
    let resp_str = bridge.dispatch("unknown.command", "{}");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(!resp.success);
    assert!(resp.error.unwrap().contains("unknown bridge command"));

    // Missing field
    let resp_str = bridge.dispatch("task.get", "{}");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(!resp.success);
    assert!(
        resp.error
            .unwrap()
            .contains("missing or invalid 'id' field")
    );

    // Invalid TaskId deserialization rejected
    let invalid_task_payload = serde_json::json!({
        "draft": {
            "id": "/invalid/starting/slash",
            "title": "Title",
            "description": "Desc",
            "priority": 1,
            "dependencies": []
        },
        "now_ms": 1000
    });
    let resp_str = bridge.dispatch("task.create", &invalid_task_payload.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(!resp.success);
    assert!(resp.error.unwrap().contains("invalid"));

    // Action record with exit_code exceeding i32 range rejected
    let out_of_range_action = serde_json::json!({
        "action_id": "act-invalid-exit",
        "exit_code": 9_999_999_999_i64
    });
    let resp_str = bridge.dispatch("action.record", &out_of_range_action.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(!resp.success);
    assert!(resp.error.unwrap().contains("32-bit"));
}

#[test]
fn test_bridge_task_dependencies_inlined() {
    let mut bridge = WheelBridge::open_in_memory().expect("bridge opens");

    // 1. Create root task (empty dependencies)
    let create_root = serde_json::json!({
        "id": "bridge-root",
        "title": "Root Task",
        "description": "Scaffolding",
        "priority": 10,
        "dependencies": [],
        "now_ms": 1000
    });
    let resp_str = bridge.dispatch("task.create", &create_root.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);

    // 2. Create child task 1
    let create_child1 = serde_json::json!({
        "id": "bridge-child-1",
        "title": "Child Task 1",
        "description": "Depends on root",
        "priority": 5,
        "dependencies": ["bridge-root"],
        "now_ms": 1010
    });
    let resp_str = bridge.dispatch("task.create", &create_child1.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);

    // 3. Create child task 2
    let create_child2 = serde_json::json!({
        "id": "bridge-child-2",
        "title": "Child Task 2",
        "description": "Depends on root",
        "priority": 5,
        "dependencies": ["bridge-root"],
        "now_ms": 1020
    });
    let resp_str = bridge.dispatch("task.create", &create_child2.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);

    // 4. Create join leaf task (depends on child 1 and child 2)
    let create_leaf = serde_json::json!({
        "id": "bridge-leaf",
        "title": "Leaf Task",
        "description": "Diamond join",
        "priority": 1,
        "dependencies": ["bridge-child-1", "bridge-child-2"],
        "now_ms": 1030
    });
    let resp_str = bridge.dispatch("task.create", &create_leaf.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);

    // 5. Query task.get for root
    let get_root = serde_json::json!({ "id": "bridge-root" });
    let resp_str = bridge.dispatch("task.get", &get_root.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let root_data = resp.data.unwrap();
    assert_eq!(root_data["id"], "bridge-root");
    assert_eq!(root_data["title"], "Root Task");
    assert_eq!(root_data["status"], "ready");
    assert_eq!(root_data["dependencies"], serde_json::json!([]));

    // 6. Query task.get for leaf (inlined multiple dependencies)
    let get_leaf = serde_json::json!({ "id": "bridge-leaf" });
    let resp_str = bridge.dispatch("task.get", &get_leaf.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let leaf_data = resp.data.unwrap();
    assert_eq!(leaf_data["id"], "bridge-leaf");
    assert_eq!(leaf_data["status"], "pending");
    assert_eq!(
        leaf_data["dependencies"],
        serde_json::json!(["bridge-child-1", "bridge-child-2"])
    );

    // 7. Query task.get_view alias
    let resp_str = bridge.dispatch("task.get_view", &get_leaf.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert_eq!(
        resp.data.unwrap()["dependencies"],
        serde_json::json!(["bridge-child-1", "bridge-child-2"])
    );

    // 8. Query task.get for nonexistent task -> returns data: null
    let get_missing = serde_json::json!({ "id": "nonexistent" });
    let resp_str = bridge.dispatch("task.get", &get_missing.to_string());
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    assert!(resp.data.is_none() || resp.data.unwrap().is_null());

    // 9. Query task.list -> all tasks have dependencies inlined
    let resp_str = bridge.dispatch("task.list", "{}");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let list = resp.data.unwrap();
    let tasks = list.as_array().expect("array of tasks");
    assert_eq!(tasks.len(), 4);

    assert_eq!(tasks[0]["id"], "bridge-root");
    assert_eq!(tasks[0]["dependencies"], serde_json::json!([]));

    assert_eq!(tasks[1]["id"], "bridge-child-1");
    assert_eq!(tasks[1]["dependencies"], serde_json::json!(["bridge-root"]));

    assert_eq!(tasks[2]["id"], "bridge-child-2");
    assert_eq!(tasks[2]["dependencies"], serde_json::json!(["bridge-root"]));

    assert_eq!(tasks[3]["id"], "bridge-leaf");
    assert_eq!(
        tasks[3]["dependencies"],
        serde_json::json!(["bridge-child-1", "bridge-child-2"])
    );

    // 10. Query task.list_views alias
    let resp_str = bridge.dispatch("task.list_views", "{}");
    let resp: BridgeResponse = serde_json::from_str(&resp_str).unwrap();
    assert!(resp.success);
    let views_list = resp.data.unwrap();
    assert_eq!(views_list.as_array().unwrap().len(), 4);
}

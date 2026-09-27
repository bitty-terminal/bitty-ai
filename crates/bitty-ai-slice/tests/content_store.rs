//! Integration tests for ContentStore, ContentHash, Rationale, and Checkpoint DAG engine (AI-0162).

use bitty_ai_slice::content_store::{
    CheckpointDraft, ContentHash, ContentStore, ContentStoreError, Rationale,
};
use bitty_ai_slice::facade::AiEngine;

#[test]
fn blob_deduplication_and_integrity() {
    let mut store = ContentStore::open_in_memory().expect("open in-memory store");

    let payload = b"fn main() { println!(\"hello bitty wheel\"); }";
    let hash1 = store.put_blob(payload, 1000).expect("put blob first time");
    let hash2 = store.put_blob(payload, 2000).expect("put blob second time");

    assert_eq!(hash1, hash2, "identical payload must yield identical hash");
    assert!(store.has_blob(&hash1).expect("has_blob"));

    let retrieved = store
        .get_blob(&hash1)
        .expect("get_blob")
        .expect("blob must exist");
    assert_eq!(retrieved, payload);

    let nonexistent = ContentHash::compute(b"does-not-exist");
    assert!(!store.has_blob(&nonexistent).expect("has_blob false"));
    assert!(
        store
            .get_blob(&nonexistent)
            .expect("get_blob none")
            .is_none()
    );
}

#[test]
fn rationale_validation_and_json_roundtrip() {
    let r = Rationale::new("Fix compiler error in parser", "Add bounds checking")
        .with_where("crates/bitty-ai-slice/src/chat_stream.rs")
        .with_how("Introduce MAX_STREAM_TOOL_CALLS constant")
        .with_expected("Zero unbounded vector allocations")
        .with_observed("All 15 vertical slice tests pass green");

    assert!(r.validate().is_ok());

    let json = serde_json::to_string(&r).expect("serialize rationale");
    let deserialized: Rationale = serde_json::from_str(&json).expect("deserialize rationale");
    assert_eq!(r, deserialized);

    // Empty why fails closed
    let invalid_why = Rationale::new("", "Something");
    assert!(matches!(
        invalid_why.validate(),
        Err(ContentStoreError::EmptyField("why"))
    ));

    // Empty what fails closed
    let invalid_what = Rationale::new("Reason", "   ");
    assert!(matches!(
        invalid_what.validate(),
        Err(ContentStoreError::EmptyField("what"))
    ));
}

#[test]
fn checkpoint_dag_creation_and_integrity() {
    let mut store = ContentStore::open_in_memory().expect("open store");

    // Put a project tree blob
    let tree_data = b"{\"files\": [\"src/main.rs\", \"Cargo.toml\"]}";
    let tree_hash = store.put_blob(tree_data, 100).expect("put tree");

    // Commit root checkpoint
    let root_draft = CheckpointDraft {
        parents: Vec::new(),
        task_id: "AI-0162".to_string(),
        agent_id: "ai-0162-impl".to_string(),
        rationale: Rationale::new("Initialize workspace", "Initial commit"),
        tree_hash: Some(tree_hash),
        summary: "Root checkpoint".to_string(),
        timestamp_ms: 1000,
    };

    let root_cp = store
        .commit_checkpoint(root_draft)
        .expect("commit root checkpoint");
    assert!(root_cp.parents.is_empty());
    assert_eq!(root_cp.task_id, "AI-0162");

    // Missing parent fails closed
    let fake_parent = ContentHash::compute(b"missing parent");
    let invalid_child_draft = CheckpointDraft {
        parents: vec![fake_parent],
        task_id: "AI-0162".to_string(),
        agent_id: "ai-0162-impl".to_string(),
        rationale: Rationale::new("Child step", "Invalid parent test"),
        tree_hash: None,
        summary: "Should fail".to_string(),
        timestamp_ms: 1050,
    };

    let err = store
        .commit_checkpoint(invalid_child_draft)
        .expect_err("must reject missing parent");
    assert!(matches!(err, ContentStoreError::MissingParent(p) if p == fake_parent));

    // Missing tree blob fails closed
    let fake_tree = ContentHash::compute(b"missing tree");
    let invalid_tree_draft = CheckpointDraft {
        parents: vec![root_cp.id],
        task_id: "AI-0162".to_string(),
        agent_id: "ai-0162-impl".to_string(),
        rationale: Rationale::new("Child step", "Invalid tree test"),
        tree_hash: Some(fake_tree),
        summary: "Should fail".to_string(),
        timestamp_ms: 1060,
    };

    let err = store
        .commit_checkpoint(invalid_tree_draft)
        .expect_err("must reject missing tree");
    assert!(matches!(err, ContentStoreError::MissingTree(t) if t == fake_tree));

    // Valid child checkpoint
    let valid_child_draft = CheckpointDraft {
        parents: vec![root_cp.id],
        task_id: "AI-0162".to_string(),
        agent_id: "ai-0162-impl".to_string(),
        rationale: Rationale::new("Add content store", "Implement Phase 1"),
        tree_hash: Some(tree_hash),
        summary: "Child checkpoint".to_string(),
        timestamp_ms: 1100,
    };

    let child_cp = store
        .commit_checkpoint(valid_child_draft)
        .expect("commit child checkpoint");
    assert_eq!(child_cp.parents, vec![root_cp.id]);

    // Retrieve checkpoints
    let retrieved_root = store
        .get_checkpoint(&root_cp.id)
        .expect("get root")
        .expect("root exists");
    assert_eq!(retrieved_root.id, root_cp.id);

    let retrieved_child = store
        .get_checkpoint(&child_cp.id)
        .expect("get child")
        .expect("child exists");
    assert_eq!(retrieved_child.id, child_cp.id);
}

#[test]
fn refs_and_log_traversal() {
    let mut store = ContentStore::open_in_memory().expect("open store");

    let cp1 = store
        .commit_checkpoint(CheckpointDraft {
            parents: Vec::new(),
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Step 1", "Initial setup"),
            tree_hash: None,
            summary: "C1".to_string(),
            timestamp_ms: 1000,
        })
        .expect("cp1");

    let cp2 = store
        .commit_checkpoint(CheckpointDraft {
            parents: vec![cp1.id],
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Step 2", "Implement feature"),
            tree_hash: None,
            summary: "C2".to_string(),
            timestamp_ms: 2000,
        })
        .expect("cp2");

    let cp3 = store
        .commit_checkpoint(CheckpointDraft {
            parents: vec![cp2.id],
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Step 3", "Finalize testing"),
            tree_hash: None,
            summary: "C3".to_string(),
            timestamp_ms: 3000,
        })
        .expect("cp3");

    // Update ref HEAD -> cp3
    store
        .update_ref("HEAD", &cp3.id, 3050)
        .expect("update ref HEAD");
    store
        .update_ref("heads/feature", &cp2.id, 3050)
        .expect("update ref heads/feature");

    assert_eq!(store.get_ref("HEAD").expect("get_ref HEAD"), Some(cp3.id));
    assert_eq!(
        store
            .get_ref("heads/feature")
            .expect("get_ref heads/feature"),
        Some(cp2.id)
    );

    let refs = store.list_refs().expect("list_refs");
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].0, "HEAD");
    assert_eq!(refs[1].0, "heads/feature");

    // Traverse log from HEAD
    let log = store.log(&cp3.id, 10).expect("log from cp3");
    assert_eq!(log.len(), 3);
    assert_eq!(log[0].id, cp3.id);
    assert_eq!(log[1].id, cp2.id);
    assert_eq!(log[2].id, cp1.id);

    // Delete ref
    assert!(store.delete_ref("heads/feature").expect("delete_ref"));
    assert!(
        store
            .get_ref("heads/feature")
            .expect("get deleted ref")
            .is_none()
    );
}

#[test]
fn merge_base_lowest_common_ancestor() {
    let mut store = ContentStore::open_in_memory().expect("open store");

    // DAG topology:
    //      Root (C0)
    //      /      \
    //    C1        C2
    //    |         |
    //    C3        C4
    let c0 = store
        .commit_checkpoint(CheckpointDraft {
            parents: Vec::new(),
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Root", "Init"),
            tree_hash: None,
            summary: "C0".to_string(),
            timestamp_ms: 100,
        })
        .expect("C0");

    let c1 = store
        .commit_checkpoint(CheckpointDraft {
            parents: vec![c0.id],
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Branch A step 1", "Work A1"),
            tree_hash: None,
            summary: "C1".to_string(),
            timestamp_ms: 200,
        })
        .expect("C1");

    let c3 = store
        .commit_checkpoint(CheckpointDraft {
            parents: vec![c1.id],
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Branch A step 2", "Work A2"),
            tree_hash: None,
            summary: "C3".to_string(),
            timestamp_ms: 300,
        })
        .expect("C3");

    let c2 = store
        .commit_checkpoint(CheckpointDraft {
            parents: vec![c0.id],
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Branch B step 1", "Work B1"),
            tree_hash: None,
            summary: "C2".to_string(),
            timestamp_ms: 250,
        })
        .expect("C2");

    let c4 = store
        .commit_checkpoint(CheckpointDraft {
            parents: vec![c2.id],
            task_id: "AI-0162".to_string(),
            agent_id: "ai-impl".to_string(),
            rationale: Rationale::new("Branch B step 2", "Work B2"),
            tree_hash: None,
            summary: "C4".to_string(),
            timestamp_ms: 350,
        })
        .expect("C4");

    // LCA of C3 and C4 should be C0
    let lca = store.merge_base(&c3.id, &c4.id).expect("merge_base C3 C4");
    assert_eq!(lca, Some(c0.id));

    // LCA of C3 and C1 should be C1
    let lca_self = store.merge_base(&c3.id, &c1.id).expect("merge_base C3 C1");
    assert_eq!(lca_self, Some(c1.id));
}

#[test]
fn facade_engine_content_store() {
    let mut store = AiEngine::open_in_memory_content_store().expect("open via AiEngine");
    let hash = store
        .put_blob(b"facade store test", 500)
        .expect("put blob via facade store");
    assert!(store.has_blob(&hash).expect("has blob"));
}

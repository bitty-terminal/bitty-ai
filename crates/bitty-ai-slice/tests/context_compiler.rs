//! Integration tests for Three-Zone Context Compiler, Merkle Tree, and Budget Pipeline (AI-0165).

use bitty_ai_slice::content_store::{Checkpoint, ContentHash, Rationale};
use bitty_ai_slice::context_compiler::{
    CompiledContext, CompilerBudgetConfig, CompilerError, ContextCompiler, ContextTree, EntryKind,
    MAX_SLOT_NAME_BYTES, TreeEntry,
};
use bitty_ai_slice::facade::AiEngine;
use bitty_ai_slice::task_dag::{TaskDraft, TaskEngine, TaskId};

#[test]
fn merkle_tree_canonical_digest_and_sorting() {
    let mut tree1 = ContextTree::new();
    let mut tree2 = ContextTree::new();

    let hash_a = ContentHash::compute(b"content-a");
    let hash_b = ContentHash::compute(b"content-b");
    let hash_c = ContentHash::compute(b"content-c");

    let entry_a = TreeEntry::new("alpha.rs", hash_a, EntryKind::Blob, 100).expect("valid entry");
    let entry_b = TreeEntry::new("beta.rs", hash_b, EntryKind::Blob, 200).expect("valid entry");
    let entry_c = TreeEntry::new("charlie.rs", hash_c, EntryKind::Blob, 300).expect("valid entry");

    // Insert in forward order in tree1
    tree1.insert(entry_a.clone());
    tree1.insert(entry_b.clone());
    tree1.insert(entry_c.clone());

    // Insert in reverse order in tree2
    tree2.insert(entry_c.clone());
    tree2.insert(entry_a.clone());
    tree2.insert(entry_b.clone());

    // Canonical digest must be identical regardless of insertion order
    let digest1 = tree1.digest();
    let digest2 = tree2.digest();
    assert_eq!(digest1, digest2);

    // Modifying one entry alters the root digest
    let entry_a_mod = TreeEntry::new(
        "alpha.rs",
        ContentHash::compute(b"modified"),
        EntryKind::Blob,
        105,
    )
    .expect("valid entry");
    tree2.insert(entry_a_mod);
    assert_ne!(tree1.digest(), tree2.digest());
}

#[test]
fn slot_name_bounds_and_validation() {
    let dummy_hash = ContentHash::compute(b"dummy");

    // Valid slot names
    assert!(TreeEntry::new("src/main.rs", dummy_hash, EntryKind::Blob, 10).is_ok());
    assert!(TreeEntry::new("scratch/notes", dummy_hash, EntryKind::Blob, 10).is_ok());
    assert!(TreeEntry::new("temp/output.log", dummy_hash, EntryKind::Blob, 10).is_ok());

    // Empty slot name fails
    assert!(matches!(
        TreeEntry::new("", dummy_hash, EntryKind::Blob, 10),
        Err(CompilerError::InvalidSlotName(_))
    ));

    // Whitespace-only slot name fails
    assert!(matches!(
        TreeEntry::new("   ", dummy_hash, EntryKind::Blob, 10),
        Err(CompilerError::InvalidSlotName(_))
    ));

    // NUL or control characters fail
    assert!(matches!(
        TreeEntry::new("bad\0name", dummy_hash, EntryKind::Blob, 10),
        Err(CompilerError::InvalidSlotName(_))
    ));

    // Oversized slot name fails
    let long_name = "a".repeat(MAX_SLOT_NAME_BYTES + 1);
    assert!(matches!(
        TreeEntry::new(long_name, dummy_hash, EntryKind::Blob, 10),
        Err(CompilerError::OversizedSlotName { .. })
    ));
}

#[test]
fn tree_diff_computation() {
    let mut base = ContextTree::new();
    let mut target = ContextTree::new();

    let h1 = ContentHash::compute(b"1");
    let h2 = ContentHash::compute(b"2");
    let h3 = ContentHash::compute(b"3");

    let e_unchanged = TreeEntry::new("common.rs", h1, EntryKind::Blob, 50).unwrap();
    let e_modified_base = TreeEntry::new("mod.rs", h1, EntryKind::Blob, 50).unwrap();
    let e_modified_target = TreeEntry::new("mod.rs", h2, EntryKind::Blob, 75).unwrap();
    let e_removed = TreeEntry::new("old.rs", h3, EntryKind::Blob, 100).unwrap();
    let e_added = TreeEntry::new("new.rs", h3, EntryKind::Blob, 120).unwrap();

    base.insert(e_unchanged.clone());
    base.insert(e_modified_base.clone());
    base.insert(e_removed.clone());

    target.insert(e_unchanged);
    target.insert(e_modified_target.clone());
    target.insert(e_added.clone());

    let diff = base.diff(&target);
    assert_eq!(diff.added.len(), 1);
    assert_eq!(diff.added[0], e_added);

    assert_eq!(diff.modified.len(), 1);
    assert_eq!(diff.modified[0], (e_modified_base, e_modified_target));

    assert_eq!(diff.removed.len(), 1);
    assert_eq!(diff.removed[0], e_removed);
}

#[test]
fn three_way_semantic_slot_merge() {
    let mut base = ContextTree::new();
    let mut ours = ContextTree::new();
    let mut theirs = ContextTree::new();

    let h0 = ContentHash::compute(b"v0");
    let h_ours = ContentHash::compute(b"v-ours");
    let h_theirs = ContentHash::compute(b"v-theirs");

    // Clean case 1: slot unchanged in both
    let e_same = TreeEntry::new("same.rs", h0, EntryKind::Blob, 10).unwrap();
    base.insert(e_same.clone());
    ours.insert(e_same.clone());
    theirs.insert(e_same.clone());

    // Clean case 2: changed only in ours
    let e_ours_base = TreeEntry::new("our_change.rs", h0, EntryKind::Blob, 10).unwrap();
    let e_ours_new = TreeEntry::new("our_change.rs", h_ours, EntryKind::Blob, 20).unwrap();
    base.insert(e_ours_base.clone());
    ours.insert(e_ours_new.clone());
    theirs.insert(e_ours_base);

    // Clean case 3: changed only in theirs
    let e_theirs_base = TreeEntry::new("their_change.rs", h0, EntryKind::Blob, 10).unwrap();
    let e_theirs_new = TreeEntry::new("their_change.rs", h_theirs, EntryKind::Blob, 30).unwrap();
    base.insert(e_theirs_base.clone());
    ours.insert(e_theirs_base);
    theirs.insert(e_theirs_new.clone());

    // Clean case 4: non-conflicting disjoint additions
    let e_add_ours = TreeEntry::new("added_ours.rs", h_ours, EntryKind::Blob, 15).unwrap();
    let e_add_theirs = TreeEntry::new("added_theirs.rs", h_theirs, EntryKind::Blob, 25).unwrap();
    ours.insert(e_add_ours.clone());
    theirs.insert(e_add_theirs.clone());

    // Conflict case: both modified concurrently to differing hashes
    let e_conf_base = TreeEntry::new("conflict.rs", h0, EntryKind::Blob, 10).unwrap();
    let e_conf_ours = TreeEntry::new("conflict.rs", h_ours, EntryKind::Blob, 18).unwrap();
    let e_conf_theirs = TreeEntry::new("conflict.rs", h_theirs, EntryKind::Blob, 19).unwrap();
    base.insert(e_conf_base.clone());
    ours.insert(e_conf_ours.clone());
    theirs.insert(e_conf_theirs.clone());

    let result = ContextTree::merge_3way(&base, &ours, &theirs);

    // Check clean merges
    assert_eq!(result.merged.get("same.rs"), Some(&e_same));
    assert_eq!(result.merged.get("our_change.rs"), Some(&e_ours_new));
    assert_eq!(result.merged.get("their_change.rs"), Some(&e_theirs_new));
    assert_eq!(result.merged.get("added_ours.rs"), Some(&e_add_ours));
    assert_eq!(result.merged.get("added_theirs.rs"), Some(&e_add_theirs));

    // Check conflict detection
    assert_eq!(result.conflicts.len(), 1);
    assert_eq!(result.conflicts[0].slot, "conflict.rs");
    assert_eq!(result.conflicts[0].base, Some(e_conf_base));
    assert_eq!(result.conflicts[0].ours, Some(e_conf_ours));
    assert_eq!(result.conflicts[0].theirs, Some(e_conf_theirs));
}

#[test]
fn three_zone_compilation_and_prefix_cache_stability() {
    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "You are a deterministic coding assistant.".to_string();
    compiler.project_rules = "No unsafe code. Enforce strict type bounds.".to_string();
    compiler.tool_schemas = r#"{"tools": ["read_file", "write_file"]}"#.to_string();

    let budget = CompilerBudgetConfig::default();

    // First compilation with prompt 1
    compiler.turn_prompt = "Run cargo check".to_string();
    compiler.uncollapsed_observations = vec!["Build output line 1".to_string()];
    let res1 = compiler.compile(&budget).expect("compile should succeed");

    // Second compilation with a completely different prompt and observation in Zone 3
    compiler.turn_prompt = "Now run cargo test on unit tests".to_string();
    compiler.uncollapsed_observations = vec!["Running 10 tests... ok".to_string()];
    let res2 = compiler.compile(&budget).expect("compile should succeed");

    // Zone 1 content and prefix hash MUST be completely identical!
    assert_eq!(res1.zone1_prefix, res2.zone1_prefix);
    assert_eq!(res1.prefix_hash, res2.prefix_hash);

    // Zone 3 tails must reflect their specific turn prompts
    assert!(res1.zone3_tail.contains("Run cargo check"));
    assert!(res2.zone3_tail.contains("Now run cargo test"));
}

#[test]
fn zone1_budget_exceeded_fails_closed() {
    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "Huge system prompt that exceeds strict zone 1 limit.".to_string();

    let budget = CompilerBudgetConfig {
        max_total_bytes: 64 * 1024,
        max_zone1_bytes: 30, // Intentionally tiny Zone 1 limit
        max_zone2_bytes: 32 * 1024,
        max_zone3_bytes: 16 * 1024,
    };

    let err = compiler.compile(&budget).unwrap_err();
    match err {
        CompilerError::Zone1BudgetExceeded { size, max } => {
            assert!(size > max);
            assert_eq!(max, 30);
        }
        other => panic!("expected Zone1BudgetExceeded, got {other:?}"),
    }
}

#[test]
fn multi_tier_budget_pipeline() {
    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "Instructions".to_string();

    // Insert persistent slots and scratch slots into context tree
    let h = ContentHash::compute(b"data");
    compiler
        .context_tree
        .insert(TreeEntry::new("src/lib.rs", h, EntryKind::Blob, 500).unwrap());
    compiler
        .context_tree
        .insert(TreeEntry::new("scratch/temp_note", h, EntryKind::Blob, 200).unwrap());
    compiler
        .context_tree
        .insert(TreeEntry::new("temp/output.log", h, EntryKind::Blob, 300).unwrap());

    // Add multiple checkpoints with rich rationales
    let r1 = Rationale {
        why: "Investigating bug".to_string(),
        what: "Inspecting error log".to_string(),
        where_focus: Some("crates/core".to_string()),
        how: Some("grep".to_string()),
        expected: Some("Found error line".to_string()),
        observed: Some("Found missing match arm".to_string()),
    };
    let cp1 = Checkpoint {
        id: ContentHash::compute(b"cp1"),
        parents: vec![],
        task_id: "AI-0165".to_string(),
        agent_id: "agent-1".to_string(),
        tree_hash: Some(compiler.context_tree.digest()),
        rationale: r1,
        summary: "Step 1: Found error line".to_string(),
        timestamp_ms: 1000,
    };

    let r2 = Rationale {
        why: "Applying fix".to_string(),
        what: "Added missing match arm".to_string(),
        where_focus: Some("crates/core".to_string()),
        how: Some("edit".to_string()),
        expected: Some("Compiler passes".to_string()),
        observed: Some("Compiler passes with 0 errors".to_string()),
    };
    let cp2 = Checkpoint {
        id: ContentHash::compute(b"cp2"),
        parents: vec![ContentHash::compute(b"cp1")],
        task_id: "AI-0165".to_string(),
        agent_id: "agent-1".to_string(),
        tree_hash: Some(compiler.context_tree.digest()),
        rationale: r2,
        summary: "Step 2: Applied fix".to_string(),
        timestamp_ms: 2000,
    };

    compiler.checkpoints = vec![cp1, cp2];

    // Case A: Generous budget, no pruning or compression
    let generous_budget = CompilerBudgetConfig::default();
    let res_generous = compiler
        .compile(&generous_budget)
        .expect("generous compile succeeds");
    assert!(res_generous.pruned_slots.is_empty());
    assert_eq!(res_generous.summarized_checkpoints, 0);
    assert!(!res_generous.truncated_tail);

    // Case B: Tight Zone 2 budget triggers Tier 1 (scratch pruning)
    let tight_zone2_tier1 = CompilerBudgetConfig {
        max_total_bytes: 64 * 1024,
        max_zone1_bytes: 16 * 1024,
        max_zone2_bytes: 400, // triggers Tier 1 pruning
        max_zone3_bytes: 16 * 1024,
    };
    let res_tier1 = compiler
        .compile(&tight_zone2_tier1)
        .expect("tier 1 compile succeeds");
    assert!(
        res_tier1
            .pruned_slots
            .contains(&"scratch/temp_note".to_string())
    );
    assert!(
        res_tier1
            .pruned_slots
            .contains(&"temp/output.log".to_string())
    );
    assert!(!res_tier1.zone2_state.contains("scratch/temp_note"));
    assert!(res_tier1.zone2_state.contains("src/lib.rs"));

    // Case C: Tighter Zone 2 budget triggers Tier 2 (checkpoint rationale compression)
    let tight_zone2_tier2 = CompilerBudgetConfig {
        max_total_bytes: 64 * 1024,
        max_zone1_bytes: 16 * 1024,
        max_zone2_bytes: 250, // triggers Tier 2 summarization
        max_zone3_bytes: 16 * 1024,
    };
    let res_tier2 = compiler
        .compile(&tight_zone2_tier2)
        .expect("tier 2 compile succeeds");
    assert_eq!(res_tier2.summarized_checkpoints, 1);
    assert!(res_tier2.zone2_state.contains("(compressed)"));

    // Case D: Small total/zone3 budget triggers Tier 3 (clean tail truncation)
    compiler.turn_prompt = "User prompt: please implement the feature".to_string();
    compiler.uncollapsed_observations = vec!["A".repeat(1000)];
    let tight_zone3 = CompilerBudgetConfig {
        max_total_bytes: 1000,
        max_zone1_bytes: 200,
        max_zone2_bytes: 500,
        max_zone3_bytes: 150, // triggers Tier 3 truncation
    };
    let res_tier3 = compiler
        .compile(&tight_zone3)
        .expect("tier 3 compile succeeds");
    assert!(res_tier3.truncated_tail);
    assert!(res_tier3.total_bytes <= tight_zone3.max_total_bytes);
    assert!(
        res_tier3
            .zone3_tail
            .contains("[...OBSERVATIONS TRUNCATED...]")
    );
}

#[test]
fn facade_compile_context_integration() {
    let mut task_engine = TaskEngine::open_in_memory().unwrap();
    let draft = TaskDraft {
        id: TaskId::new("task-compiler").unwrap(),
        title: "Active task for compiler".to_string(),
        description: "Task under test".to_string(),
        priority: 10,
        dependencies: vec![],
    };
    let task = task_engine.create_task(draft, 1000).unwrap();

    let mut compiler = ContextCompiler::new();
    compiler.system_prompt = "System prompt via facade".to_string();
    compiler.active_task = Some(task);
    compiler.turn_prompt = "Next step in task".to_string();

    let budget = CompilerBudgetConfig::default();
    let compiled: CompiledContext = AiEngine::compile_context(&compiler, &budget).unwrap();

    assert!(compiled.zone1_prefix.contains("System prompt via facade"));
    assert!(compiled.zone2_state.contains("Active task for compiler"));
    assert!(compiled.zone3_tail.contains("Next step in task"));
    assert_eq!(compiled.pruned_slots.len(), 0);
    assert_eq!(compiled.summarized_checkpoints, 0);
    assert!(!compiled.truncated_tail);
}

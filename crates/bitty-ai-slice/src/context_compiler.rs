//! Three-zone context compiler, Merkle context tree, and multi-tier budget pipeline (AI-0165).
//!
//! Provides the compilation plane for Wheel:
//! - [`ContextTree`]: Merkle tree mapping named context slots to immutable [`ContentHash`] blobs
//!   with deterministic canonical digests, tree diffing, and 3-way semantic slot merging.
//! - [`ContextCompiler`]: Three-zone context compilation separating invariant stable prefix
//!   (for LLM vendor prefix-cache hits) from structured cognitive state and dynamic turn tails.
//! - [`CompilerBudgetConfig`]: Multi-tier budget limiter enforcing hard ceilings via L1 pruning,
//!   L2 structured rationale compression, and L3 clean dynamic tail truncation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::content_store::{Checkpoint, ContentHash};
use crate::task_dag::TaskNode;

/// Maximum allowed byte length for a context slot name (256 bytes).
pub const MAX_SLOT_NAME_BYTES: usize = 256;

/// Default maximum total context budget in bytes (64 KiB).
pub const DEFAULT_MAX_TOTAL_BYTES: usize = 64 * 1024;

/// Default maximum Zone 1 (Stable Prefix) budget in bytes (16 KiB).
pub const DEFAULT_MAX_ZONE1_BYTES: usize = 16 * 1024;

/// Default maximum Zone 2 (Structured State) budget in bytes (32 KiB).
pub const DEFAULT_MAX_ZONE2_BYTES: usize = 32 * 1024;

/// Default maximum Zone 3 (Dynamic Tail) budget in bytes (16 KiB).
pub const DEFAULT_MAX_ZONE3_BYTES: usize = 16 * 1024;

/// Errors arising from context compilation or Merkle tree operations.
#[derive(Debug)]
pub enum CompilerError {
    /// Invariant Zone 1 (Stable Prefix) exceeded its strict allocation limit.
    Zone1BudgetExceeded {
        /// Actual size of Zone 1 in bytes.
        size: usize,
        /// Maximum allowed limit in bytes.
        max: usize,
    },
    /// Total context size exceeds maximum budget even after all pruning and compression tiers.
    TotalBudgetExceeded {
        /// Actual total bytes.
        size: usize,
        /// Configured limit in bytes.
        max: usize,
    },
    /// Slot name exceeds length limit.
    OversizedSlotName {
        /// The offending slot name.
        name: String,
        /// Maximum allowed bytes.
        max: usize,
    },
    /// Slot name contains invalid control characters or forbidden sequences.
    InvalidSlotName(String),
    /// Slot was not found in the context tree.
    SlotNotFound(String),
    /// JSON serialization or deserialization failure.
    Serialization(serde_json::Error),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zone1BudgetExceeded { size, max } => {
                write!(
                    f,
                    "zone 1 (stable prefix) exceeds budget: {size} bytes > {max} bytes"
                )
            }
            Self::TotalBudgetExceeded { size, max } => {
                write!(
                    f,
                    "compiled context exceeds total budget: {size} bytes > {max} bytes"
                )
            }
            Self::OversizedSlotName { name, max } => {
                write!(f, "slot name '{name}' exceeds length limit of {max} bytes")
            }
            Self::InvalidSlotName(name) => write!(f, "invalid slot name: {name:?}"),
            Self::SlotNotFound(name) => write!(f, "context slot '{name}' not found"),
            Self::Serialization(err) => write!(f, "serialization error: {err}"),
        }
    }
}

impl std::error::Error for CompilerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(err) => Some(err),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for CompilerError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err)
    }
}

/// Kind of entry referenced in a [`ContextTree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// An immutable leaf blob of data.
    Blob,
    /// A nested subtree of slots.
    Tree,
}

impl EntryKind {
    /// Return byte identifier for Merkle hashing.
    #[must_use]
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Blob => 1,
            Self::Tree => 2,
        }
    }
}

/// A typed entry within a [`ContextTree`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeEntry {
    /// Slot name identifier (e.g. `system`, `rules`, `workspace/src/lib.rs`).
    pub name: String,
    /// Cryptographic content address of the entry.
    pub hash: ContentHash,
    /// Kind of entry (blob or subtree).
    pub kind: EntryKind,
    /// Size of the underlying content in bytes.
    pub size_bytes: usize,
}

impl TreeEntry {
    /// Construct and validate a new [`TreeEntry`].
    pub fn new(
        name: impl AsRef<str>,
        hash: ContentHash,
        kind: EntryKind,
        size_bytes: usize,
    ) -> Result<Self, CompilerError> {
        let name_str = name.as_ref().trim();
        if name_str.is_empty() {
            return Err(CompilerError::InvalidSlotName(name_str.to_string()));
        }
        if name_str.len() > MAX_SLOT_NAME_BYTES {
            return Err(CompilerError::OversizedSlotName {
                name: name_str.to_string(),
                max: MAX_SLOT_NAME_BYTES,
            });
        }
        for b in name_str.bytes() {
            if b < 0x20 || b == 0x7f {
                return Err(CompilerError::InvalidSlotName(name_str.to_string()));
            }
        }

        Ok(Self {
            name: name_str.to_string(),
            hash,
            kind,
            size_bytes,
        })
    }
}

/// Difference between two [`ContextTree`] snapshots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TreeDiff {
    /// Entries present in target but absent in base.
    pub added: Vec<TreeEntry>,
    /// Entries present in both with mismatched hash, kind, or size.
    pub modified: Vec<(TreeEntry, TreeEntry)>,
    /// Entries present in base but absent in target.
    pub removed: Vec<TreeEntry>,
}

impl TreeDiff {
    /// Check if there are no differences between the two trees.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.removed.is_empty()
    }
}

/// A conflict detected during a 3-way merge of context trees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotConflict {
    /// Slot name that encountered a concurrent modification.
    pub slot: String,
    /// Base version, if present.
    pub base: Option<TreeEntry>,
    /// Our branch version, if present.
    pub ours: Option<TreeEntry>,
    /// Their branch version, if present.
    pub theirs: Option<TreeEntry>,
}

/// Result of a 3-way merge on two context trees relative to a common ancestor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeMergeResult {
    /// The merged context tree containing cleanly merged slots.
    pub merged: ContextTree,
    /// Conflicts where both sides modified the same slot incompatibly.
    pub conflicts: Vec<SlotConflict>,
}

impl TreeMergeResult {
    /// Check whether the merge was clean without any slot conflicts.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}

/// A Merkle tree representing a snapshot of named context slots.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextTree {
    entries: BTreeMap<String, TreeEntry>,
}

impl ContextTree {
    /// Create a new, empty context tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Insert or replace a slot entry in the tree.
    pub fn insert(&mut self, entry: TreeEntry) -> Option<TreeEntry> {
        self.entries.insert(entry.name.clone(), entry)
    }

    /// Retrieve an entry by slot name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TreeEntry> {
        self.entries.get(name)
    }

    /// Remove an entry by slot name.
    pub fn remove(&mut self, name: &str) -> Option<TreeEntry> {
        self.entries.remove(name)
    }

    /// Return the number of slot entries in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the tree contains no slot entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over references to all slot entries in canonical lexicographical order.
    pub fn entries(&self) -> impl Iterator<Item = &TreeEntry> {
        self.entries.values()
    }

    /// Calculate the deterministic Merkle digest [`ContentHash`] of this tree.
    ///
    /// Serializes entries in sorted order with canonical length prefixing:
    /// `tree:v1\0` + `[entry_name_len, entry_name, hash, kind, size_bytes]`.
    #[must_use]
    pub fn digest(&self) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(b"tree:v1\0");

        for entry in self.entries.values() {
            let name_bytes = entry.name.as_bytes();
            hasher.update((name_bytes.len() as u32).to_be_bytes());
            hasher.update(name_bytes);
            hasher.update(entry.hash.as_bytes());
            hasher.update([entry.kind.as_u8()]);
            hasher.update((entry.size_bytes as u64).to_be_bytes());
        }

        let raw: [u8; 32] = hasher.finalize().into();
        ContentHash::from_bytes(raw)
    }

    /// Compute the difference between `self` (base) and `target`.
    #[must_use]
    pub fn diff(&self, target: &Self) -> TreeDiff {
        let mut added = Vec::new();
        let mut modified = Vec::new();
        let mut removed = Vec::new();

        for (name, target_entry) in &target.entries {
            match self.entries.get(name) {
                None => added.push(target_entry.clone()),
                Some(base_entry) => {
                    if base_entry != target_entry {
                        modified.push((base_entry.clone(), target_entry.clone()));
                    }
                }
            }
        }

        for (name, base_entry) in &self.entries {
            if !target.entries.contains_key(name) {
                removed.push(base_entry.clone());
            }
        }

        TreeDiff {
            added,
            modified,
            removed,
        }
    }

    /// Perform a 3-way semantic merge of `ours` and `theirs` against `base`.
    #[must_use]
    pub fn merge_3way(base: &Self, ours: &Self, theirs: &Self) -> TreeMergeResult {
        let mut merged = Self::new();
        let mut conflicts = Vec::new();

        let all_keys: BTreeSet<&str> = base
            .entries
            .keys()
            .chain(ours.entries.keys())
            .chain(theirs.entries.keys())
            .map(|s| s.as_str())
            .collect();

        for key in all_keys {
            let b = base.entries.get(key);
            let o = ours.entries.get(key);
            let t = theirs.entries.get(key);

            match (b, o, t) {
                // Unchanged in both
                (Some(bv), Some(ov), Some(tv)) if ov == bv && tv == bv => {
                    merged.insert(bv.clone());
                }
                // Changed only in ours
                (Some(bv), Some(ov), Some(tv)) if tv == bv && ov != bv => {
                    merged.insert(ov.clone());
                }
                // Changed only in theirs
                (Some(bv), Some(ov), Some(tv)) if ov == bv && tv != bv => {
                    merged.insert(tv.clone());
                }
                // Both changed identically
                (Some(_), Some(ov), Some(tv)) if ov == tv => {
                    merged.insert(ov.clone());
                }
                // Added only in ours
                (None, Some(ov), None) => {
                    merged.insert(ov.clone());
                }
                // Added only in theirs
                (None, None, Some(tv)) => {
                    merged.insert(tv.clone());
                }
                // Added identically in both
                (None, Some(ov), Some(tv)) if ov == tv => {
                    merged.insert(ov.clone());
                }
                // Deleted only in ours (theirs unchanged)
                (Some(bv), None, Some(tv)) if tv == bv => {}
                // Deleted only in theirs (ours unchanged)
                (Some(bv), Some(ov), None) if ov == bv => {}
                // Deleted in both
                (Some(_), None, None) => {}
                // Conflict: concurrent incompatible modification or add/delete conflict
                _ => {
                    conflicts.push(SlotConflict {
                        slot: key.to_string(),
                        base: b.cloned(),
                        ours: o.cloned(),
                        theirs: t.cloned(),
                    });
                }
            }
        }

        TreeMergeResult { merged, conflicts }
    }
}

/// Budget configuration controlling multi-tier context compilation thresholds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerBudgetConfig {
    /// Maximum allowed aggregate context size in bytes.
    pub max_total_bytes: usize,
    /// Maximum allowed allocation for Zone 1 (Stable Prefix) in bytes.
    pub max_zone1_bytes: usize,
    /// Maximum allowed allocation for Zone 2 (Structured State) in bytes.
    pub max_zone2_bytes: usize,
    /// Maximum allowed allocation for Zone 3 (Dynamic Tail) in bytes.
    pub max_zone3_bytes: usize,
}

impl Default for CompilerBudgetConfig {
    fn default() -> Self {
        Self {
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
            max_zone1_bytes: DEFAULT_MAX_ZONE1_BYTES,
            max_zone2_bytes: DEFAULT_MAX_ZONE2_BYTES,
            max_zone3_bytes: DEFAULT_MAX_ZONE3_BYTES,
        }
    }
}

/// Compiled context ready for model dispatch, detailing per-zone buffers and metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledContext {
    /// Zone 1: Byte-stable prefix (system prompt, tools, rules).
    pub zone1_prefix: String,
    /// Zone 2: Structured state (active task, checkpoints, rationales, pinned slots).
    pub zone2_state: String,
    /// Zone 3: Dynamic tail (uncollapsed observations, turn prompt).
    pub zone3_tail: String,
    /// Cryptographic digest of Zone 1 for vendor prefix-cache validation.
    pub prefix_hash: ContentHash,
    /// Total compiled context size in bytes.
    pub total_bytes: usize,
    /// Slot names pruned during Tier 1 reduction.
    pub pruned_slots: Vec<String>,
    /// Number of historical checkpoints summarized during Tier 2 reduction.
    pub summarized_checkpoints: usize,
    /// Whether Tier 3 hard truncation occurred on the dynamic tail.
    pub truncated_tail: bool,
}

impl CompiledContext {
    /// Render the full assembled prompt by joining the three zones with newlines.
    #[must_use]
    pub fn to_prompt_string(&self) -> String {
        let mut out = String::with_capacity(self.total_bytes + 8);
        if !self.zone1_prefix.is_empty() {
            out.push_str(&self.zone1_prefix);
            out.push('\n');
        }
        if !self.zone2_state.is_empty() {
            out.push_str(&self.zone2_state);
            out.push('\n');
        }
        if !self.zone3_tail.is_empty() {
            out.push_str(&self.zone3_tail);
        }
        out
    }

    /// Check if context had to undergo hard tail truncation.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.truncated_tail
    }
}

/// Three-zone context compiler optimizing for prefix-cache hit rate and high-density state.
#[derive(Debug, Clone, Default)]
pub struct ContextCompiler {
    // Zone 1: Stable Prefix
    /// Invariant system prompt.
    pub system_prompt: String,
    /// Pinned tool declarations or schemas.
    pub tool_schemas: String,
    /// Invariant project rules and policies.
    pub project_rules: String,

    // Zone 2: Structured State
    /// Active task state machine node, if any.
    pub active_task: Option<TaskNode>,
    /// Predecessor cognitive checkpoints in chronological order.
    pub checkpoints: Vec<Checkpoint>,
    /// Snapshot of current context tree slots.
    pub context_tree: ContextTree,

    // Zone 3: Dynamic Tail
    /// Immediate user or agent turn query.
    pub turn_prompt: String,
    /// Recent uncollapsed observations or command outputs.
    pub uncollapsed_observations: Vec<String>,
}

impl ContextCompiler {
    /// Create a new, blank context compiler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compile context under the provided budget configuration.
    ///
    /// Executes the multi-tier budget pipeline:
    /// - Tier 1: Prune unpinned scratchpad slots (`scratch/*`, `temp/*`).
    /// - Tier 2: Summarize older historical rationales into 1-line digests.
    /// - Tier 3: Hard truncate dynamic observations while preserving Zone 1 and the active task.
    pub fn compile(&self, budget: &CompilerBudgetConfig) -> Result<CompiledContext, CompilerError> {
        // --- Assemble Zone 1: Stable Prefix ---
        let mut z1 = String::new();
        if !self.system_prompt.trim().is_empty() {
            z1.push_str("=== SYSTEM INSTRUCTIONS ===\n");
            z1.push_str(self.system_prompt.trim());
            z1.push('\n');
        }
        if !self.project_rules.trim().is_empty() {
            z1.push_str("=== PROJECT RULES ===\n");
            z1.push_str(self.project_rules.trim());
            z1.push('\n');
        }
        if !self.tool_schemas.trim().is_empty() {
            z1.push_str("=== TOOL SCHEMAS ===\n");
            z1.push_str(self.tool_schemas.trim());
            z1.push('\n');
        }

        if z1.len() > budget.max_zone1_bytes {
            return Err(CompilerError::Zone1BudgetExceeded {
                size: z1.len(),
                max: budget.max_zone1_bytes,
            });
        }

        let prefix_hash = ContentHash::compute(z1.as_bytes());

        // --- Assemble Zone 2: Structured State (with Tier 1 & Tier 2) ---
        let mut pruned_slots = Vec::new();
        let mut summarized_checkpoints = 0;

        let mut z2 = self.render_zone2(&[], 0);

        // Tier 1 Reduction: Prune scratchpad slots if over budget
        if z2.len() > budget.max_zone2_bytes {
            let scratch_slots: Vec<String> = self
                .context_tree
                .entries()
                .filter(|e| e.name.starts_with("scratch/") || e.name.starts_with("temp/"))
                .map(|e| e.name.clone())
                .collect();

            if !scratch_slots.is_empty() {
                pruned_slots = scratch_slots;
                z2 = self.render_zone2(&pruned_slots, 0);
            }
        }

        // Tier 2 Reduction: Summarize older checkpoints if still over budget
        if z2.len() > budget.max_zone2_bytes && self.checkpoints.len() > 1 {
            summarized_checkpoints = self.checkpoints.len() - 1;
            z2 = self.render_zone2(&pruned_slots, summarized_checkpoints);
        }

        // --- Assemble Zone 3: Dynamic Tail (with Tier 3) ---
        let mut z3 = String::new();
        let mut truncated_tail = false;

        let mut obs_text = String::new();
        for (i, obs) in self.uncollapsed_observations.iter().enumerate() {
            let trimmed = obs.trim();
            if !trimmed.is_empty() {
                obs_text.push_str(&format!("[OBSERVATION {i}]: {trimmed}\n"));
            }
        }

        let mut turn_text = String::new();
        if !self.turn_prompt.trim().is_empty() {
            turn_text.push_str("=== TURN PROMPT ===\n");
            turn_text.push_str(self.turn_prompt.trim());
            turn_text.push('\n');
        }

        let separator_bytes = usize::from(!z1.is_empty()) + usize::from(!z2.is_empty());
        let current_z3_len = obs_text.len() + turn_text.len();
        let remaining_total_budget = budget
            .max_total_bytes
            .saturating_sub(z1.len() + z2.len() + separator_bytes);
        let effective_z3_max = budget.max_zone3_bytes.min(remaining_total_budget);

        if current_z3_len <= effective_z3_max {
            if !obs_text.is_empty() {
                z3.push_str("=== OBSERVATIONS ===\n");
                z3.push_str(&obs_text);
            }
            z3.push_str(&turn_text);
        } else {
            // Tier 3 Hard Truncation: clamp observations first
            truncated_tail = true;
            if turn_text.len() >= effective_z3_max {
                // Even turn prompt alone exceeds or fills budget: clamp turn prompt
                let clamp_len = effective_z3_max.saturating_sub(32);
                let safe_slice = Self::truncate_utf8(turn_text.trim(), clamp_len);
                z3.push_str("=== TURN PROMPT ===\n");
                z3.push_str(safe_slice);
                z3.push_str("\n[...TRUNCATED...]\n");
            } else {
                let obs_budget = effective_z3_max.saturating_sub(turn_text.len() + 32);
                if obs_budget > 0 {
                    z3.push_str("=== OBSERVATIONS ===\n");
                    let safe_obs = Self::truncate_utf8(obs_text.trim(), obs_budget);
                    z3.push_str(safe_obs);
                    z3.push_str("\n[...OBSERVATIONS TRUNCATED...]\n");
                }
                z3.push_str(&turn_text);
            }
        }

        let total_bytes = z1.len() + z2.len() + z3.len() + separator_bytes;
        if total_bytes > budget.max_total_bytes {
            return Err(CompilerError::TotalBudgetExceeded {
                size: total_bytes,
                max: budget.max_total_bytes,
            });
        }

        Ok(CompiledContext {
            zone1_prefix: z1,
            zone2_state: z2,
            zone3_tail: z3,
            prefix_hash,
            total_bytes,
            pruned_slots,
            summarized_checkpoints,
            truncated_tail,
        })
    }

    /// Internal helper: Render Zone 2 content taking pruning and summarization counts into account.
    fn render_zone2(&self, pruned_slots: &[String], summarize_older_than: usize) -> String {
        let mut z2 = String::new();

        // Active task section
        if let Some(task) = &self.active_task {
            z2.push_str("=== ACTIVE TASK ===\n");
            z2.push_str(&format!(
                "ID: {}\nTitle: {}\nPriority: {}\nStatus: {}\nGeneration: {}\n",
                task.id, task.title, task.priority, task.status, task.generation
            ));
            if !task.description.trim().is_empty() {
                z2.push_str(&format!("Description: {}\n", task.description.trim()));
            }
            if let Some(reason) = &task.failure_reason {
                z2.push_str(&format!("Failure Reason: {}\n", reason.trim()));
            }
            z2.push('\n');
        }

        // Cognitive checkpoints section
        if !self.checkpoints.is_empty() {
            z2.push_str("=== COGNITIVE CHECKPOINTS ===\n");
            for (idx, cp) in self.checkpoints.iter().enumerate() {
                if idx < summarize_older_than {
                    // L2 compressed summary
                    let obs = cp.rationale.observed.as_deref().unwrap_or("-");
                    z2.push_str(&format!(
                        "[{idx}] (compressed) Summary: {} | Why: {} -> Observed: {}\n",
                        cp.summary, cp.rationale.why, obs
                    ));
                } else {
                    // Full rationale
                    z2.push_str(&format!(
                        "[{idx}] Summary: {}\n  Why: {}\n  What: {}\n",
                        cp.summary, cp.rationale.why, cp.rationale.what
                    ));
                    if let Some(focus) = &cp.rationale.where_focus {
                        z2.push_str(&format!("  Where: {focus}\n"));
                    }
                    if let Some(how) = &cp.rationale.how {
                        z2.push_str(&format!("  How: {how}\n"));
                    }
                    if let Some(exp) = &cp.rationale.expected {
                        z2.push_str(&format!("  Expected: {exp}\n"));
                    }
                    if let Some(obs) = &cp.rationale.observed {
                        z2.push_str(&format!("  Observed: {obs}\n"));
                    }
                }
            }
            z2.push('\n');
        }

        // Context tree slots section
        if !self.context_tree.is_empty() {
            let pruned_set: BTreeSet<&str> = pruned_slots.iter().map(|s| s.as_str()).collect();
            z2.push_str("=== CONTEXT SLOTS ===\n");
            for entry in self.context_tree.entries() {
                if !pruned_set.contains(entry.name.as_str()) {
                    z2.push_str(&format!(
                        "Slot: {} | Hash: {} | Size: {} bytes\n",
                        entry.name, entry.hash, entry.size_bytes
                    ));
                }
            }
            z2.push('\n');
        }

        z2
    }

    /// Internal helper: safely truncate a UTF-8 string at a character boundary.
    fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        let mut boundary = max_bytes;
        while boundary > 0 && !s.is_char_boundary(boundary) {
            boundary -= 1;
        }
        &s[..boundary]
    }
}

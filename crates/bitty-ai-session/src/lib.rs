//! Session plane: content-addressed object store, typed ref verbs, and content hash.
//!
//! Extracted from `bitty-ai-slice` (AI-0186, DEC-0008 step 5) as a pure move:
//! [`ContentHash`] (leaf, std plus `sha2`/`hex`/`serde` only),
//! [`ContentStore`] (transactional SQLite content/checkpoint/ref store), and the
//! typed branch verbs in [`session_refs`] (free functions over
//! [`ContentStore`], moved atomically with the type). No behavior change.
//!
//! The slice keeps byte-identical re-export shims, so every existing
//! `bitty_ai_slice::content_store::*` / `::session_refs::*` /
//! `::content_hash::*` path still resolves. `bitty-ai-slice` depends on this
//! crate, never the reverse.

#![deny(unsafe_code)]

pub mod content_hash;
pub mod content_store;
pub mod session_refs;

pub use content_hash::{ContentHash, ContentHashError};
pub use content_store::{
    Checkpoint, CheckpointDraft, ContentStore, ContentStoreError, DURABLE_BUSY_TIMEOUT_MS,
    DURABLE_PROFILE, MAX_AGENT_ID_BYTES, MAX_BLOB_BYTES, MAX_CHECKPOINT_PARENTS,
    MAX_RATIONALE_FIELD_BYTES, MAX_RATIONALE_TOTAL_BYTES, MAX_REF_NAME_BYTES, MAX_SUMMARY_BYTES,
    MAX_TASK_ID_BYTES, Rationale,
};
pub use session_refs::{
    BranchName, MAX_REFLOG_ACTOR_BYTES, MAX_REFLOG_READ_LIMIT, MAX_REFLOG_REASON_BYTES, RefError,
    ReflogEntry, commit_checkpoint_with_branch, create_branch, delete_branch, dump_reflog_all,
    get_branch, list_branches, read_reflog, rename_branch, update_branch,
};

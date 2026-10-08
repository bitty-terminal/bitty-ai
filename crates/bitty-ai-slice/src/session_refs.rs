//! Re-export shim (AI-0186, DEC-0008 step 5): the refs plane now lives in
//! `bitty-ai-session`. This module preserves every existing
//! `bitty_ai_slice::session_refs::*` path byte-identically.
//!
//! The `From<RefError> for FacadeError` impl cannot move with the type
//! (the session crate cannot name `FacadeError` without depending on the
//! slice, which would cycle: slice -> session, never the reverse),
//! so it is re-homed in [`crate::facade`].

pub use bitty_ai_session::session_refs::{
    BranchName, MAX_REFLOG_ACTOR_BYTES, MAX_REFLOG_READ_LIMIT, MAX_REFLOG_REASON_BYTES, RefError,
    ReflogEntry, commit_checkpoint_with_branch, create_branch, delete_branch, dump_reflog_all,
    get_branch, list_branches, read_reflog, rename_branch, update_branch,
};

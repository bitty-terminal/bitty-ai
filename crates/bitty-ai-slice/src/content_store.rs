//! Re-export shim (AI-0186, DEC-0008 step 5): the content store now lives in
//! `bitty-ai-session`. This module preserves every existing
//! `bitty_ai_slice::content_store::*` path byte-identically.

pub use bitty_ai_session::content_store::{
    CONTENT_SCHEMA, Checkpoint, CheckpointDraft, ContentHash, ContentHashError, ContentStore,
    ContentStoreError, DURABLE_BUSY_TIMEOUT_MS, DURABLE_PROFILE, MAX_AGENT_ID_BYTES,
    MAX_BLOB_BYTES, MAX_CHECKPOINT_PARENTS, MAX_RATIONALE_FIELD_BYTES, MAX_RATIONALE_TOTAL_BYTES,
    MAX_REF_NAME_BYTES, MAX_SUMMARY_BYTES, MAX_TASK_ID_BYTES, Rationale, admit_or_init,
    apply_durable_pragmas, check_sqlite_magic, claim_writer_fast, init_content_schema,
    map_busy_for_facade, verify_profile,
};

//! Re-export shim (AI-0186, DEC-0008 step 5): [`ContentHash`] now lives in
//! `bitty-ai-session`. This module preserves every existing
//! `bitty_ai_slice::content_hash::*` path byte-identically.

pub use bitty_ai_session::content_hash::{ContentHash, ContentHashError};

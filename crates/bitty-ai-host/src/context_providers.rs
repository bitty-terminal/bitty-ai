//! Future seam: context providers.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no provider contract, no assembly logic, no
//! redaction or consent evaluation. By construction it grants nothing: there
//! is no item to call and no ambient authority to inherit. Corresponds to
//! the context-providers item in the canonical design's Phase 2 backlog.

//! Future seam: ToolBus and dispatch.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no registry, no spec surface, no precheck, no
//! dispatch, no authorizer. By construction it grants nothing: there is no
//! item to call and no ambient authority to inherit. Corresponds to the
//! tool surface in the canonical design's call-site inventory (Seam 2).

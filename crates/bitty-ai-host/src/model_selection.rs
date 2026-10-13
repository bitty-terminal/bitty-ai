//! Future seam: model selection.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no registry lookup, no alias resolution, no
//! fallback policy. By construction it grants nothing: there is no item to
//! call and no ambient authority to inherit. Corresponds to the selection
//! surface in the canonical design's call-site inventory (Seam 3, selection
//! side).

//! Future seam: provider registry plus model I/O.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no registry, no provider trait, no model I/O,
//! no selection logic. By construction it grants nothing: there is no item
//! to call and no ambient authority to inherit. Corresponds to the
//! provider surface in the canonical design's call-site inventory (Seam 3,
//! provider side).

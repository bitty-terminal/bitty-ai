//! Future seam: extension-registry wiring.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no manifest resolution, no point wiring, no
//! versioning logic, no permission grant. By construction it grants nothing:
//! there is no item to call and no ambient authority to inherit.
//! Corresponds to the extension-point declarations in the canonical design's
//! call-site inventory (Seam 1).

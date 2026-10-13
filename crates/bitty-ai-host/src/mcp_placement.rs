//! Future seam: MCP placement.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no transport, no placement decision, no routing.
//! By construction it grants nothing: there is no item to call and no
//! ambient authority to inherit. Corresponds to the MCP-placement item in
//! the canonical design's Phase 2 backlog (per AIQ-36 and AIQ-38); placement
//! stays open and no choice is made here.

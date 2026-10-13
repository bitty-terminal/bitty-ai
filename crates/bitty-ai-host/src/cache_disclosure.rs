//! Future seam: cache and disclosure primitives.
//!
//! A separately scoped Phase 2 task could traverse this seam; this module
//! declares the name only — no cache-key construction, no fingerprint reuse
//! checks, no writer-lease fencing, no fallback envelopes. By construction
//! it grants nothing: there is no item to call and no ambient authority to
//! inherit. Corresponds to the cache-key, fingerprint, fencing, and fallback
//! primitives in the canonical design's call-site inventory (Seam 4).

//! Host-boundary shell skeleton (`bitty-ai-host`).
//!
//! **Status: experimental shell, Phase 1 per CTX-0147.** This crate declares
//! the seam skeleton a later phase could traverse — module names only, no
//! signatures, no logic, no wiring — and grants no behavior: no registry, no
//! selection logic, no dispatch, no provider or context wiring, no IPC, no
//! threads, no filesystem or network access, no dependencies beyond `std`.
//!
//! Deny-all posture holds by construction: the shell carries no ambient
//! authority because there is nothing to call — every seam module below is
//! documentation only, with zero items — and `#![deny(unsafe_code)]`
//! (honored via the workspace lints) forbids unsafe blocks crate-wide.
//!
//! Explicitly open (no choice made here or in any module below):
//!
//! - S-1 versus S-2 registry-split choice (stays open under AIQ-38).
//! - DEC-0004 acceptance (still requires architecture, docs-curator, and
//!   security review).
//! - `LocalEndpoint` route question (stays with the Phase 2 backlog).
//! - Per-seam behavior (stays with separately scoped Phase 2 tasks).
//!
//! Shape note: the seam modules are intentionally doc-only with no items and
//! no tests — except `context_providers`, which carries the Phase 2 S1
//! pure-type/zone mirror, and `extension_registry`, which carries the Phase 2
//! S1 pure-type mirror (both std-only, still no I/O, IPC, threads, or ambient
//! authority, each with offline unit tests). A test elsewhere would either assert
//! a vacuous truth or require introducing an item (constant or function) to
//! assert on, and any such item would itself be behavior-adjacent surface
//! contradicting the no-types, no-functions, no-behavior rule above. The
//! deny-all default is structural (empty shell apart from the mirror,
//! std-only, deny unsafe), not asserted.
//!
//! Canonical design text lives in `bitty-ai-docs`
//! (`architecture/host-boundary-trait-design.md`, Phase 1 scope section);
//! this crate adds no per-seam item and rewrites none.

#![deny(unsafe_code)]

pub mod cache_disclosure;
pub mod context_providers;
pub mod extension_registry;
pub mod mcp_placement;
pub mod model_selection;
pub mod provider_registry;
pub mod tool_dispatch;

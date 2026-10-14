//! Host-boundary shell skeleton (`bitty-ai-host`).
//!
//! **Status: experimental shell, Phase 1 per CTX-0147.** This crate declares
//! the seam skeleton a later phase could traverse — module names only, no
//! signatures, no logic, no wiring — and grants no behavior: no registry, no
//! selection logic, no dispatch, no provider or context wiring, no IPC, no
//! threads, no filesystem or network access, no dependencies beyond `std`.
//!
//! Deny-all posture holds by construction: the shell carries no ambient
//! authority because each seam module below carries only types plus pure
//! checks — no behavior, no transport — and `#![deny(unsafe_code)]`
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
//! Shape note: each of the seven seam modules below carries its Phase 2 S1
//! pure-type mirror plus its Phase 2 S2 scoped-IPC envelope types — types
//! plus pure `&str`/`&[u8]`-in/value-out checks only, `std`-only, with no
//! behavior and no transport (no registry execution, no dispatch, no provider
//! or context wiring, no socket, no encoding, no version field, no threads,
//! no clock, no filesystem or network access, no ambient authority), each
//! with offline unit tests. Each module's own header documents its S-scope
//! (what is mirrored, what stays out, and which S2 task owns its envelopes).
//! The deny-all default stays structural (mirrors plus envelopes only,
//! `std`-only, deny unsafe), not asserted.
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

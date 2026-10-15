//! Shared snapshot-version correlator vocabulary (S-1 follow-up, AI-0301).
//!
//! Offline, `std`-only, types-only vocabulary slice: an opaque ordered
//! [`SnapshotVersion`], a fail-closed stale/missing refusal taxonomy
//! ([`SnapshotRefusal`]), and pure checks ([`is_stale`] /
//! [`check_snapshot_version`]). No clock, no I/O, no threads, no transport,
//! no serialization, no envelope change, no `bitty-ipc` dependency
//! (`Cargo.toml` stays `std`-only): versions are caller-supplied data,
//! mirroring the crate's deterministic caller-supplied `now_ms` discipline
//! (see `tool_dispatch.rs`, where `now_ms` travels on the query and no wall
//! clock is read).
//!
//! Normative basis: the R1 ruling (2026-10-15) adopted S-1 (metadata-only
//! terminal registry with a resident singleton helper) and REQUIRED an
//! invalidation/refresh path with fail-closed behavior on stale or missing
//! snapshot versions (AI-0211 deny-stale pattern as read-only reference), so
//! a drifted snapshot view cannot authorize or route as if current. This
//! module is the vocabulary slice of that required path: it refuses stale or
//! missing versions fail-closed and carries versions only (never payload), so
//! its diagnostics stay safe to log directly. Refresh/invalidation execution
//! itself stays producer-owned (same split as the S2 [`SnapshotEnvelope`]
//! convention in `mcp_placement.rs`, where re-list stays the sole MCP-owned
//! un-staling path and the envelope only echoes version data).
//!
//! Plug-in contract (candidate until transport scoping defines the params
//! JSON schema): when that schema lands, the correlator rides as an optional
//! params field citing [`SnapshotVersion`]; an absent field maps to `None` in
//! [`check_snapshot_version`] and therefore refuses fail-closed as
//! [`SnapshotRefusal::MissingVersion`]. Until then, envelope-identifier-only
//! correlation remains valid (S-2-compatible fallback preserved: per the R1
//! candidate text, S-2 reconciliation correlates by envelope identifier
//! alone). No S2 envelope struct carries the field now, and none is added
//! here.
//!
//! S2-echo positioning (read-only, no refactor): the version-like `u64`s
//! already in this crate stay as they are. `SnapshotEnvelope.list_version` /
//! `list_stale` (`mcp_placement.rs`) echo MCP list data with no enforcement,
//! and `ContextReadResponse.current_generation` (`context_providers.rs`) is
//! an exact-match generation fence echo, not an ordered comparison. Neither
//! is refactored here: the generation fence keeps exact-equality semantics
//! (it must NOT migrate to ordered comparison), while the list-version echo
//! data may cite [`SnapshotVersion`] when the params schema lands (candidate,
//! owned by transport scoping, not this task).
//!
//! Explicitly not in scope: S2 envelope changes, serialization, any
//! `bitty-ipc` dependency, refresh/invalidation execution, admit/deny routing,
//! and any thread, clock, network, or filesystem use.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Initial snapshot version.
///
/// The producer starts at `0` (mirrors the S2 `SnapshotEnvelope` convention
/// in `mcp_placement.rs`, where `list_version` starts at `0` with saturating
/// bumps producer-side). `0` is a legitimate version, never a missing
/// marker: absence is `None` at the check boundary, never a sentinel value.
pub const SNAPSHOT_VERSION_INITIAL: u64 = 0;

/// Opaque ordered snapshot version for the S-1 metadata snapshot correlator.
///
/// Wraps a caller-supplied monotonic `u64` (per the R1 ruling, 2026-10-15:
/// the terminal side pins a versioned snapshot, and boundary checks cite the
/// pinned version for audit). Opaque so no caller fabricates ordering
/// outside [`is_stale`] / [`check_snapshot_version`]; ordered so staleness
/// is a pure comparison. Construction is total over the full `u64` range —
/// no silent truncation, no wrap — because bound discipline here is
/// representational (every version round-trips), not restrictive; saturating
/// bumps stay producer-owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotVersion(u64);

impl SnapshotVersion {
    /// Initial version (equals [`SNAPSHOT_VERSION_INITIAL`]).
    pub const INITIAL: Self = Self(SNAPSHOT_VERSION_INITIAL);

    /// Wrap a caller-supplied version. Total: every `u64` is valid and
    /// round-trips through [`SnapshotVersion::get`].
    #[must_use]
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Borrow the raw version (audit citation only, never payload).
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }

    /// Whether this is the initial version.
    #[must_use]
    pub fn is_initial(self) -> bool {
        self.0 == SNAPSHOT_VERSION_INITIAL
    }
}

impl Display for SnapshotVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "snapshot version {}", self.0)
    }
}

/// Fail-closed refusal taxonomy for snapshot-version checks.
///
/// Per the R1 ruling (2026-10-15), stale or missing snapshot versions must
/// refuse fail-closed so a drifted view cannot authorize or route as if
/// current (AI-0211 deny-stale pattern as read-only reference). Both variants
/// carry versions only (`u64`), never payload bytes, mirroring the crate's
/// names/bounds/counts-only diagnostic policy — `Display` output is safe to
/// log directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotRefusal {
    /// The supplied version is behind the caller-supplied current version.
    StaleVersion {
        /// Version the caller presented.
        seen: SnapshotVersion,
        /// Version the caller holds as current.
        current: SnapshotVersion,
    },
    /// No version was supplied (for example, the future optional params
    /// field is absent). Carries the caller-supplied current version so the
    /// audit trail still cites what was current at refusal time.
    MissingVersion {
        /// Version the caller holds as current.
        current: SnapshotVersion,
    },
}

impl Display for SnapshotRefusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::StaleVersion { seen, current } => write!(
                f,
                "stale snapshot version {} (current {})",
                seen.get(),
                current.get()
            ),
            Self::MissingVersion { current } => {
                write!(f, "missing snapshot version (current {})", current.get())
            }
        }
    }
}

impl std::error::Error for SnapshotRefusal {}

/// Pure staleness comparison: `seen` is stale exactly when behind `current`.
///
/// Deterministic, offline, no I/O. Equal versions are current, never stale;
/// an ahead `seen` is accepted as not-stale (candidate rationale: versions
/// are monotonic producer data, so an ahead sighting means the caller's
/// `current` is itself behind — the caller should refresh its current, but
/// the sighting is not drift authorizing as current).
#[must_use]
pub fn is_stale(seen: SnapshotVersion, current: SnapshotVersion) -> bool {
    seen < current
}

/// Fail-closed version gate over an optional seen version against a
/// caller-supplied current version.
///
/// `None` (no version supplied) refuses as [`SnapshotRefusal::MissingVersion`];
/// a behind version refuses as [`SnapshotRefusal::StaleVersion`]; an equal or
/// ahead version passes. Never panics: the missing path is a typed refusal,
/// never an unwrap. Pure, deterministic, offline.
///
/// # Errors
///
/// Returns [`SnapshotRefusal::StaleVersion`] when `seen` is behind `current`,
/// and [`SnapshotRefusal::MissingVersion`] when no version was supplied.
pub fn check_snapshot_version(
    seen: Option<SnapshotVersion>,
    current: SnapshotVersion,
) -> Result<(), SnapshotRefusal> {
    match seen {
        None => Err(SnapshotRefusal::MissingVersion { current }),
        Some(version) if is_stale(version, current) => Err(SnapshotRefusal::StaleVersion {
            seen: version,
            current,
        }),
        Some(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_ordering_and_equality() {
        assert!(SnapshotVersion::new(1) < SnapshotVersion::new(2));
        assert!(SnapshotVersion::new(2) > SnapshotVersion::new(1));
        assert_eq!(SnapshotVersion::new(5), SnapshotVersion::new(5));
        assert_ne!(SnapshotVersion::new(5), SnapshotVersion::new(6));
        // Ordering is total: equal versions are never stale in either
        // direction.
        assert!(!is_stale(SnapshotVersion::new(5), SnapshotVersion::new(5)));
    }

    #[test]
    fn version_bounds_represent_full_range() {
        // Explicit bound discipline: the initial version is named (mirrors the
        // S2 `list_version` starts-at-0 convention), and construction is
        // total over the full u64 range with no silent truncation.
        assert_eq!(SNAPSHOT_VERSION_INITIAL, 0);
        assert_eq!(SnapshotVersion::INITIAL, SnapshotVersion::new(0));
        assert!(SnapshotVersion::INITIAL.is_initial());
        assert!(!SnapshotVersion::new(1).is_initial());
        assert_eq!(SnapshotVersion::new(0).get(), 0);
        assert_eq!(SnapshotVersion::new(u64::MAX).get(), u64::MAX);
    }

    #[test]
    fn stale_version_refuses_fail_closed() {
        let seen = SnapshotVersion::new(3);
        let current = SnapshotVersion::new(5);
        assert!(is_stale(seen, current));
        assert_eq!(
            check_snapshot_version(Some(seen), current),
            Err(SnapshotRefusal::StaleVersion { seen, current })
        );
        // Refusal carries versions only, never payload: exact text cited.
        let refusal = SnapshotRefusal::StaleVersion { seen, current };
        assert_eq!(refusal.to_string(), "stale snapshot version 3 (current 5)");
    }

    #[test]
    fn missing_version_refuses_fail_closed() {
        let current = SnapshotVersion::new(5);
        assert_eq!(
            check_snapshot_version(None, current),
            Err(SnapshotRefusal::MissingVersion { current })
        );
        let refusal = SnapshotRefusal::MissingVersion { current };
        assert_eq!(refusal.to_string(), "missing snapshot version (current 5)");
        // The taxonomy is a real error type (fail-closed plumbing).
        let error: &dyn std::error::Error = &refusal;
        assert!(error.to_string().contains("missing snapshot version"));
    }

    #[test]
    fn current_and_ahead_versions_pass() {
        let current = SnapshotVersion::new(5);
        assert!(check_snapshot_version(Some(current), current).is_ok());
        assert!(check_snapshot_version(Some(SnapshotVersion::new(7)), current).is_ok());
        assert!(!is_stale(SnapshotVersion::new(7), current));
    }

    #[test]
    fn missing_path_never_panics_at_extremes() {
        // No-version comparison is panics-free at every current, including
        // the range extremes: absence refuses, never unwraps.
        for current in [0, 1, SNAPSHOT_VERSION_INITIAL, u64::MAX] {
            let current = SnapshotVersion::new(current);
            assert_eq!(
                check_snapshot_version(None, current),
                Err(SnapshotRefusal::MissingVersion { current })
            );
            assert!(!is_stale(current, current));
        }
    }
}

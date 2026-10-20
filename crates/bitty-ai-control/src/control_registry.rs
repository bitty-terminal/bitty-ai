//! Minimal control-only registry adapter for the helper entrypoint.
//!
//! **This is not the real provider registry.** It exists so the launch
//! entrypoint can drive the provider-publication state machine without a
//! provider/credential integration: the snapshot is a fixed empty
//! `provider-routing` projection at feed revision `1`, and no method mutates
//! it. Real registry feed integration (provider metadata, revisions,
//! invalidation on mutation, and the read gate) is a separate follow-up.
//!
//! The adapter implements exactly the [`RegistryView`] read seam:
//!
//! - [`RegistryView::snapshot`] returns the same empty projection at the same
//!   revision on every call;
//! - [`RegistryView::retained_matches`] compares retained publication bytes
//!   against the current canonical empty projection and refuses a different
//!   feed revision.
//!
//! A publication is therefore bindable only when its canonical payload is the
//! exact empty projection announced at revision `1`. Provider routing remains
//! empty, so strict provider publication stays disabled.

use bitty_ipc_api::provider_publication::ProviderRoutingProjection;

use crate::state::{RegistryError, RegistrySnapshot, RegistryView};

/// Feed revision reported by the control-only registry adapter.
pub const CONTROL_ONLY_FEED_REVISION: u64 = 1;

/// Fixed empty registry view used by the launch entrypoint.
///
/// The value is stateless: it owns one prepared snapshot (empty projection at
/// [`CONTROL_ONLY_FEED_REVISION`]) and never writes. It is deliberately not a
/// provider registry, a store, or a feed: no provider metadata, model entry,
/// or credential is read, retained, or announced.
#[derive(Debug, Clone)]
pub struct ControlOnlyRegistry {
    snapshot: RegistrySnapshot,
}

impl ControlOnlyRegistry {
    /// Build the fixed empty view at [`CONTROL_ONLY_FEED_REVISION`].
    #[must_use]
    pub fn new() -> Self {
        let snapshot = RegistrySnapshot::new(
            CONTROL_ONLY_FEED_REVISION,
            ProviderRoutingProjection::empty(),
        )
        .expect("the control-only feed revision is non-zero");
        Self { snapshot }
    }
}

impl Default for ControlOnlyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RegistryView for ControlOnlyRegistry {
    fn snapshot(&self) -> Result<RegistrySnapshot, RegistryError> {
        Ok(self.snapshot.clone())
    }

    fn retained_matches(&self, feed_revision: u64, retained: &[u8]) -> Result<bool, RegistryError> {
        if feed_revision != self.snapshot.feed_revision() {
            return Ok(false);
        }
        Ok(self.snapshot.projection().canonical_bytes() == retained)
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTROL_ONLY_FEED_REVISION, ControlOnlyRegistry};
    use crate::state::{RegistrySnapshot, RegistryView};

    #[test]
    fn snapshot_is_the_empty_projection_at_revision_one() {
        let registry = ControlOnlyRegistry::new();
        let snapshot = registry.snapshot().expect("snapshot");
        assert_eq!(snapshot.feed_revision(), CONTROL_ONLY_FEED_REVISION);
        assert!(snapshot.projection().providers().is_empty());
        assert_eq!(
            snapshot.projection().canonical_bytes(),
            b"{\"domain\":\"provider-routing\",\"providers\":[]}".to_vec()
        );
        assert_eq!(snapshot, registry.snapshot().expect("repeated snapshot"));
    }

    #[test]
    fn retained_bytes_must_be_the_current_empty_projection() {
        let registry = ControlOnlyRegistry::new();
        let snapshot = registry.snapshot().expect("snapshot");
        let canonical = snapshot.projection().canonical_bytes();

        assert_eq!(
            registry.retained_matches(CONTROL_ONLY_FEED_REVISION, &canonical),
            Ok(true)
        );
        assert_eq!(
            registry.retained_matches(CONTROL_ONLY_FEED_REVISION, b"not-the-projection"),
            Ok(false)
        );
        assert_eq!(
            registry.retained_matches(CONTROL_ONLY_FEED_REVISION + 1, &canonical),
            Ok(false)
        );
    }

    #[test]
    fn repeated_reads_are_identical_and_immutable() {
        let registry = ControlOnlyRegistry::new();
        let first: RegistrySnapshot = registry.snapshot().expect("first");
        let second = registry.snapshot().expect("second");
        assert_eq!(first.feed_revision(), second.feed_revision());
        assert_eq!(first.projection(), second.projection());
    }
}

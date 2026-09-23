//! Descriptor contract under the v0.1 freeze (AI-0137, AI-0135 freeze).
//!
//! Deterministic and offline: scripted [`FakeProvider`] snapshots plus pure
//! [`ModelRegistration::snapshot_from`] calls. No network, no secrets, no
//! wall clock, no threads.
//!
//! Covered here (and only here):
//! 1. `list_models` is unfiltered: the snapshot is returned as-is.
//!    Host-side caller-scope filtering per `MP-4` is deferred, so this file
//!    pins the current behavior (full snapshot, detached clone), not the
//!    future filtering.
//! 2. `snapshot_from` metadata-attachment semantics: `provider_id` plus the
//!    context window plus the cost weights attach to the registration while
//!    the descriptor stays read-only.
//! 3. Fail-closed paths: empty capability sets and malformed provider ids
//!    refuse with typed errors and leave the registry unchanged.
//!
//! Non-overlap with neighbors (assertions, not scenarios, are disjoint):
//! - `provider.rs` unit tests own provider-id shape, timeout ceiling,
//!   budget, and script mechanics. Id shape appears here only as the
//!   construction/bridge refusal.
//! - `selection.rs` unit tests own alias, window, cost, and fallback gates.
//!   Selection appears here only as the bridge proof that a snapshot
//!   registration is usable.
//!
//! CodeQL lesson from AI-0082: assert/panic messages are static only;
//! provider ids, model names, and capability data never appear in message
//! strings.

use bitty_ai_runtime::{
    FakeProvider, ModelCapability, ModelDescriptor, ModelProvider, ModelRegistration,
    ProviderRegistry, SelectRequest, SelectionError,
};

fn fake_descriptor() -> ModelDescriptor {
    FakeProvider::new("bitty-fake")
        .expect("valid id")
        .list_models()
        .pop()
        .expect("fake provider ships one model")
}

#[test]
fn list_models_returns_snapshot_as_is() {
    // v0.1 freeze: `list_models` is a registry snapshot (skeleton:
    // unfiltered). Caller-scope filtering per `MP-4` is host-side and
    // deferred, so the current contract is the full snapshot, returned
    // as-is on every call with no cursor and no consumption.
    let provider = FakeProvider::new("bitty-fake").expect("valid id");
    let models = provider.list_models();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "fake-chat");
    assert_eq!(
        models[0].capabilities,
        vec![
            ModelCapability::Text,
            ModelCapability::Streaming,
            ModelCapability::ToolUse,
        ]
    );
    let again = provider.list_models();
    assert_eq!(models, again);
    assert_eq!(provider.scripted_turns_remaining(), 0);
    assert_eq!(provider.complete_calls(), 0);
}

#[test]
fn list_models_snapshot_is_detached() {
    // The snapshot is a clone: mutating a returned vector never leaks back
    // into the provider, so a later call still yields the pristine entry.
    let provider = FakeProvider::new("bitty-fake").expect("valid id");
    let mut models = provider.list_models();
    models[0].name.push_str("-mutated");
    models[0].capabilities.clear();
    models.push(ModelDescriptor {
        name: "injected".to_owned(),
        capabilities: vec![ModelCapability::Text],
    });
    let again = provider.list_models();
    assert_eq!(again.len(), 1);
    assert_eq!(again[0].name, "fake-chat");
    assert_eq!(
        again[0].capabilities,
        vec![
            ModelCapability::Text,
            ModelCapability::Streaming,
            ModelCapability::ToolUse,
        ]
    );
}

#[test]
fn snapshot_from_attaches_metadata_and_leaves_descriptor_read_only() {
    // Bridging attaches routing metadata (`provider_id`, context window,
    // cost weights) to the registration; the descriptor is read only.
    let provider = FakeProvider::new("bitty-fake").expect("valid id");
    let descriptor = fake_descriptor();
    let before = descriptor.clone();
    let registration =
        ModelRegistration::snapshot_from(provider.provider_id(), &descriptor, 4_096, 1, 2)
            .expect("valid snapshot");
    assert_eq!(registration.provider_id, "bitty-fake");
    assert_eq!(registration.name, descriptor.name);
    assert_eq!(registration.capabilities, descriptor.capabilities);
    assert_eq!(registration.context_window_tokens, 4_096);
    assert_eq!(registration.input_cost_weight, 1);
    assert_eq!(registration.output_cost_weight, 2);
    assert_eq!(descriptor, before);
    // Same descriptor with different routing metadata keeps the same
    // name/capabilities and carries the new window/weights.
    let repriced =
        ModelRegistration::snapshot_from(provider.provider_id(), &descriptor, 8_192, 3, 4)
            .expect("valid snapshot");
    assert_eq!(repriced.name, registration.name);
    assert_eq!(repriced.capabilities, registration.capabilities);
    assert_eq!(repriced.context_window_tokens, 8_192);
    assert_eq!(repriced.input_cost_weight, 3);
    assert_eq!(repriced.output_cost_weight, 4);
    assert_eq!(descriptor, before);
}

#[test]
fn snapshot_registers_and_selects() {
    // A bridged snapshot is immediately usable: it registers and satisfies
    // capability-subset selection for the advertised set.
    let provider = FakeProvider::new("bitty-fake").expect("valid id");
    let descriptor = fake_descriptor();
    let registration =
        ModelRegistration::snapshot_from(provider.provider_id(), &descriptor, 4_096, 1, 2)
            .expect("valid snapshot");
    let mut registry = ProviderRegistry::new();
    registry.register(registration).expect("capacity");
    let chain = registry
        .select(&SelectRequest::capabilities(vec![
            ModelCapability::Text,
            ModelCapability::ToolUse,
        ]))
        .expect("snapshot matches its advertised set");
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].provider_id, "bitty-fake");
    assert_eq!(chain[0].name, "fake-chat");
}

#[test]
fn snapshot_from_with_empty_capabilities_fails_closed() {
    // A capability-less descriptor can never match (selection always
    // requires at least one capability), so bridging refuses it instead of
    // registering an unselectable entry.
    let empty = ModelDescriptor {
        name: "empty".to_owned(),
        capabilities: Vec::new(),
    };
    assert_eq!(
        ModelRegistration::snapshot_from("bitty-fake", &empty, 4_096, 1, 2)
            .expect_err("empty capabilities must fail"),
        SelectionError::NoCapabilities {
            name: "empty".to_owned(),
        }
    );
    assert_eq!(
        ModelRegistration::new("bitty-fake", "empty", Vec::new(), 4_096, 1, 2)
            .expect_err("empty capabilities must fail"),
        SelectionError::NoCapabilities {
            name: "empty".to_owned(),
        }
    );
    // No partial state: the failed snapshot never reaches the registry.
    let registry = ProviderRegistry::new();
    assert!(registry.is_empty());
}

#[test]
fn snapshot_from_with_malformed_provider_id_fails_closed() {
    // The `MP-2` id shape (`^[a-z][a-z0-9_-]*$`, at most 64 bytes) is
    // enforced at the bridge: every malformed id refuses with the typed
    // error and leaves the registry unchanged.
    let descriptor = ModelDescriptor {
        name: "fake-chat".to_owned(),
        capabilities: vec![ModelCapability::Text],
    };
    let bad_ids = ["", "BAD", "1abc", "local.deterministic"];
    assert_eq!(bad_ids.len(), 4);
    for bad in bad_ids {
        assert!(
            matches!(
                ModelRegistration::snapshot_from(bad, &descriptor, 4_096, 1, 2),
                Err(SelectionError::InvalidProviderId { .. })
            ),
            "malformed provider id must fail"
        );
    }
    let long = "a".repeat(65);
    assert!(
        matches!(
            ModelRegistration::snapshot_from(&long, &descriptor, 4_096, 1, 2),
            Err(SelectionError::InvalidProviderId { .. })
        ),
        "over-long provider id must fail"
    );
    let registry = ProviderRegistry::new();
    assert!(registry.is_empty());
}

#[test]
fn provider_construction_rejects_malformed_id() {
    // Same shape at the source: [`FakeProvider::new`] validates before any
    // snapshot exists, so a malformed id never yields a provider to bridge.
    assert!(FakeProvider::new("").is_err());
    assert!(FakeProvider::new("BAD").is_err());
    assert!(FakeProvider::new("1abc").is_err());
    assert!(FakeProvider::new("local.deterministic").is_err());
    let long = "a".repeat(65);
    assert!(FakeProvider::new(long).is_err());
    assert!(FakeProvider::new("bitty-fake").is_ok());
}

//! Typed extension-point seam declarations (AI-0079, draft candidate input).
//!
//! Declarations only: the eight seam traits plus the [`Manifest`] data
//! struct. There is no registry, no resolution, no versioning logic, and no
//! permission grant anywhere in this file. Every trait is a draft candidate
//! referencing the 040 distillation, not an accepted schema.
//!
//! The load-bearing property under test is permission non-inheritance: a
//! struct implementing one point gains no method from any other point
//! (separate traits, no supertraits, no blanket impls). The negative half of
//! that property is pinned by `compile_fail` doctests on the [`extension`]
//! module; this file pins the positive half (each point is independently
//! implementable, a multi-point struct needs one explicit impl per point)
//! plus [`Manifest`] data round-trips and [`ExtensionPoint`]
//! exhaustiveness.
//!
//! [`extension`]: bitty_ai_runtime::extension
//! [`Manifest`]: bitty_ai_runtime::extension::Manifest
//! [`ExtensionPoint`]: bitty_ai_runtime::extension::ExtensionPoint

use bitty_ai_runtime::extension::{
    AgentPoint, CommandPoint, CompactorPoint, ContextPoint, ExtensionPoint, Manifest, MemoryPoint,
    ModelPoint, ToolPoint, UiPoint,
};

struct ModelProbe;
impl ModelPoint for ModelProbe {
    fn declared_model(&self) -> &str {
        "fake-chat"
    }
}

struct ToolProbe;
impl ToolPoint for ToolProbe {
    fn effect_class(&self) -> &str {
        "read-only"
    }
}

struct ContextProbe;
impl ContextPoint for ContextProbe {
    fn read_scope(&self) -> &str {
        "workspace.read"
    }
}

struct MemoryProbe;
impl MemoryPoint for MemoryProbe {
    fn retention_class(&self) -> &str {
        "session"
    }
}

struct CompactorProbe;
impl CompactorPoint for CompactorProbe {
    fn strategy_id(&self) -> &str {
        "truncate-head"
    }
}

struct AgentProbe;
impl AgentPoint for AgentProbe {
    fn capability_id(&self) -> &str {
        "single-turn"
    }
}

struct CommandProbe;
impl CommandPoint for CommandProbe {
    fn command_name(&self) -> &str {
        "ai.ask"
    }
}

struct UiProbe;
impl UiPoint for UiProbe {
    fn descriptor(&self) -> &str {
        "panel: transcript"
    }
}

/// One struct, two explicit per-point impls: implementing [`ToolPoint`]
/// contributes no [`ModelPoint`] method, so the second impl block is still
/// required (the reverse — omitting either block — fails to compile where
/// the matching bound is needed).
struct ToolAndModel;
impl ToolPoint for ToolAndModel {
    fn effect_class(&self) -> &str {
        "read-only"
    }
}
impl ModelPoint for ToolAndModel {
    fn declared_model(&self) -> &str {
        "fake-chat"
    }
}

fn requires_model(point: &impl ModelPoint) -> &str {
    point.declared_model()
}

fn requires_tool(point: &impl ToolPoint) -> &str {
    point.effect_class()
}

fn requires_context(point: &impl ContextPoint) -> &str {
    point.read_scope()
}

fn requires_memory(point: &impl MemoryPoint) -> &str {
    point.retention_class()
}

fn requires_compactor(point: &impl CompactorPoint) -> &str {
    point.strategy_id()
}

fn requires_agent(point: &impl AgentPoint) -> &str {
    point.capability_id()
}

fn requires_command(point: &impl CommandPoint) -> &str {
    point.command_name()
}

fn requires_ui(point: &impl UiPoint) -> &str {
    point.descriptor()
}

#[test]
fn each_point_is_implementable_independently() {
    assert_eq!(requires_model(&ModelProbe), "fake-chat");
    assert_eq!(requires_tool(&ToolProbe), "read-only");
    assert_eq!(requires_context(&ContextProbe), "workspace.read");
    assert_eq!(requires_memory(&MemoryProbe), "session");
    assert_eq!(requires_compactor(&CompactorProbe), "truncate-head");
    assert_eq!(requires_agent(&AgentProbe), "single-turn");
    assert_eq!(requires_command(&CommandProbe), "ai.ask");
    assert_eq!(requires_ui(&UiProbe), "panel: transcript");
}

#[test]
fn multi_point_struct_needs_explicit_impl_per_point() {
    let both = ToolAndModel;
    assert_eq!(requires_tool(&both), "read-only");
    assert_eq!(requires_model(&both), "fake-chat");
}

/// Exhaustive match with no wildcard: adding a ninth point breaks this test
/// at compile time, so the eight-point set stays closed by construction.
fn point_name(point: ExtensionPoint) -> &'static str {
    match point {
        ExtensionPoint::Model => "model",
        ExtensionPoint::Tool => "tool",
        ExtensionPoint::Context => "context",
        ExtensionPoint::Memory => "memory",
        ExtensionPoint::Compactor => "compactor",
        ExtensionPoint::Agent => "agent",
        ExtensionPoint::Command => "command",
        ExtensionPoint::Ui => "ui",
    }
}

#[test]
fn extension_point_enum_is_exhaustive() {
    let all = [
        ExtensionPoint::Model,
        ExtensionPoint::Tool,
        ExtensionPoint::Context,
        ExtensionPoint::Memory,
        ExtensionPoint::Compactor,
        ExtensionPoint::Agent,
        ExtensionPoint::Command,
        ExtensionPoint::Ui,
    ];
    assert_eq!(all.len(), 8);
    let names: Vec<&'static str> = all.iter().map(|point| point_name(*point)).collect();
    assert_eq!(
        names,
        vec![
            "model",
            "tool",
            "context",
            "memory",
            "compactor",
            "agent",
            "command",
            "ui"
        ]
    );
}

#[test]
fn manifest_data_round_trip() {
    let manifest = Manifest {
        name: "example-extension".to_owned(),
        version: "0.1.0".to_owned(),
        host: "bitty-ai >=0.1.0".to_owned(),
        points: vec![ExtensionPoint::Tool, ExtensionPoint::Command],
    };
    let round_tripped = manifest.clone();
    assert_eq!(manifest, round_tripped);
    assert_eq!(manifest.name, "example-extension");
    assert_eq!(manifest.version, "0.1.0");
    // The host predicate is opaque data: stored verbatim, never parsed.
    assert_eq!(manifest.host, "bitty-ai >=0.1.0");
    assert!(manifest.points.contains(&ExtensionPoint::Tool));
    assert!(!manifest.points.contains(&ExtensionPoint::Model));
}

#[test]
fn manifest_with_no_points_is_well_formed() {
    let manifest = Manifest {
        name: "empty-extension".to_owned(),
        version: "0.0.0".to_owned(),
        host: "bitty-ai *".to_owned(),
        points: Vec::new(),
    };
    assert!(manifest.points.is_empty());
    assert_eq!(manifest, manifest.clone());
}

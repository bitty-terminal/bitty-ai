//! Typed extension-point seam declarations (draft candidate input).
//!
//! Mirrors the Extension Points / Manifest / permission non-inheritance
//! direction of
//! `research-distillation-040-bitty-ai.md`: a fixed set of declaration-only
//! seams, one per capability category, that extensions attach through. Point
//! names and method shapes are proposals, not an accepted schema; the stable
//! claim is only typed attachment (no capability outside a declared point)
//! plus dependency-is-not-authority (an extension implementing one point
//! gains nothing from any other point).
//!
//! # What this module is not
//!
//! There is no registry, no resolution algorithm, no versioning logic, and
//! no permission grant in this file. [`Manifest`] is data only: the host
//! predicate is an opaque string that is stored verbatim and never parsed.
//! Every trait method is required (no provided methods that do work), there
//! are no supertraits between the point traits, and there are no blanket
//! impls, so the negative non-inheritance half holds: a probe struct
//! implementing one point does not satisfy any other point's bound.
//!
//! ```compile_fail
//! use bitty_ai_runtime::extension::{ModelPoint, ToolPoint};
//!
//! struct OnlyTool;
//! impl ToolPoint for OnlyTool {
//!     fn effect_class(&self) -> &str { "read-only" }
//! }
//!
//! // `OnlyTool` implements `ToolPoint` but not `ModelPoint`: a generic
//! // bound on `ModelPoint` must not resolve for it, so this fails.
//! fn requires_model(point: &impl ModelPoint) -> &str {
//!     point.declared_model()
//! }
//!
//! requires_model(&OnlyTool);
//! ```
//!
//! ```compile_fail
//! use bitty_ai_runtime::extension::{AgentPoint, ContextPoint};
//!
//! struct OnlyContext;
//! impl ContextPoint for OnlyContext {
//!     fn read_scope(&self) -> &str { "workspace.read" }
//! }
//!
//! // Same separation across the context/agent pair: one point's impl
//! // contributes no method from the other.
//! fn requires_agent(point: &impl AgentPoint) -> &str {
//!     point.capability_id()
//! }
//!
//! requires_agent(&OnlyContext);
//! ```

/// The eight declared extension-point categories.
///
/// Kept as a plain exhaustive enum (data only): matching with no wildcard
/// breaks at compile time if a ninth point is ever added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtensionPoint {
    /// Model capability declaration surface. Mirrors the
    /// [`provider::ModelProvider`](crate::provider::ModelProvider) boundary;
    /// does not replace it.
    Model,
    /// Tool effect-class declaration. Mirrors the
    /// [`tool::ToolAuthorizer`](crate::tool::ToolAuthorizer) /
    /// [`tool::ToolExecutor`](crate::tool::ToolExecutor) boundary.
    Tool,
    /// Scoped read declaration. Mirrors context assembly inputs.
    Context,
    /// Retention-class declaration. Mirrors
    /// [`compression::RetentionPolicy`](crate::compression::RetentionPolicy)
    /// inputs.
    Memory,
    /// Compaction-strategy declaration. Mirrors the compression seam.
    Compactor,
    /// Agent-capability declaration. Mirrors session tier/level gates;
    /// grants nothing.
    Agent,
    /// Command registration declaration. Mirrors R2 declared placement.
    Command,
    /// Presentation contribution declaration. Text-only descriptors;
    /// no rendering.
    Ui,
}

/// Extension manifest: data only, no parsing, no resolution.
///
/// `host` is an opaque version predicate over the host (`bitty-ai`);
/// stored verbatim, never parsed here. `points` lists the contributed
/// extension points as an enum set. This declaration never doubles as a
/// permission grant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Extension name.
    pub name: String,
    /// Extension version (opaque string).
    pub version: String,
    /// Host version predicate (opaque string, never parsed).
    pub host: String,
    /// Contributed extension points.
    pub points: Vec<ExtensionPoint>,
}

/// Model capability declaration surface. Draft candidate input referencing
/// the 040 distillation; not an accepted schema. Mirrors the
/// [`provider::ModelProvider`](crate::provider::ModelProvider) boundary;
/// does not replace it. Grants nothing.
pub trait ModelPoint {
    /// Declared model capability identifier (opaque descriptor).
    #[must_use]
    fn declared_model(&self) -> &str;
}

/// Tool effect-class declaration. Draft candidate input referencing the 040
/// distillation; not an accepted schema. Mirrors the
/// [`tool::ToolAuthorizer`](crate::tool::ToolAuthorizer) /
/// [`tool::ToolExecutor`](crate::tool::ToolExecutor) boundary. Grants
/// nothing.
pub trait ToolPoint {
    /// Declared tool effect class (opaque descriptor).
    #[must_use]
    fn effect_class(&self) -> &str;
}

/// Scoped read declaration. Draft candidate input referencing the 040
/// distillation; not an accepted schema. Mirrors context assembly inputs.
/// Grants nothing.
pub trait ContextPoint {
    /// Declared read scope (opaque descriptor).
    #[must_use]
    fn read_scope(&self) -> &str;
}

/// Retention-class declaration. Draft candidate input referencing the 040
/// distillation; not an accepted schema. Mirrors
/// [`compression::RetentionPolicy`](crate::compression::RetentionPolicy)
/// inputs. Grants nothing.
pub trait MemoryPoint {
    /// Declared retention class (opaque descriptor).
    #[must_use]
    fn retention_class(&self) -> &str;
}

/// Compaction-strategy declaration. Draft candidate input referencing the
/// 040 distillation; not an accepted schema. Mirrors the compression seam.
/// Grants nothing.
pub trait CompactorPoint {
    /// Declared compaction strategy identifier (opaque descriptor).
    #[must_use]
    fn strategy_id(&self) -> &str;
}

/// Agent-capability declaration. Draft candidate input referencing the 040
/// distillation; not an accepted schema. Mirrors session tier/level gates;
/// grants nothing.
pub trait AgentPoint {
    /// Declared agent capability identifier (opaque descriptor).
    #[must_use]
    fn capability_id(&self) -> &str;
}

/// Command registration declaration. Draft candidate input referencing the
/// 040 distillation; not an accepted schema. Mirrors R2 declared placement.
/// Grants nothing.
pub trait CommandPoint {
    /// Declared command name (opaque descriptor).
    #[must_use]
    fn command_name(&self) -> &str;
}

/// Presentation contribution declaration. Draft candidate input referencing
/// the 040 distillation; not an accepted schema. Text-only descriptors;
/// no rendering. Grants nothing.
pub trait UiPoint {
    /// Declared presentation descriptor (plain text; never rendered here).
    #[must_use]
    fn descriptor(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::{
        AgentPoint, CommandPoint, CompactorPoint, ContextPoint, ExtensionPoint, Manifest,
        MemoryPoint, ModelPoint, ToolPoint, UiPoint,
    };

    struct ToolOnly;
    impl ToolPoint for ToolOnly {
        fn effect_class(&self) -> &str {
            "read-only"
        }
    }

    struct ModelOnly;
    impl ModelPoint for ModelOnly {
        fn declared_model(&self) -> &str {
            "fake-chat"
        }
    }

    struct ContextOnly;
    impl ContextPoint for ContextOnly {
        fn read_scope(&self) -> &str {
            "workspace.read"
        }
    }

    struct MemoryOnly;
    impl MemoryPoint for MemoryOnly {
        fn retention_class(&self) -> &str {
            "session"
        }
    }

    struct CompactorOnly;
    impl CompactorPoint for CompactorOnly {
        fn strategy_id(&self) -> &str {
            "truncate-head"
        }
    }

    struct AgentOnly;
    impl AgentPoint for AgentOnly {
        fn capability_id(&self) -> &str {
            "single-turn"
        }
    }

    struct CommandOnly;
    impl CommandPoint for CommandOnly {
        fn command_name(&self) -> &str {
            "ai.ask"
        }
    }

    struct UiOnly;
    impl UiPoint for UiOnly {
        fn descriptor(&self) -> &str {
            "panel: transcript"
        }
    }

    /// One struct, two explicit impl blocks: the [`ToolPoint`] impl
    /// contributes no [`ModelPoint`] method, so the second block is still
    /// required to satisfy a `ModelPoint` bound.
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
    fn each_point_is_implementable_independently() {
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

        assert_eq!(requires_model(&ModelOnly), "fake-chat");
        assert_eq!(requires_tool(&ToolOnly), "read-only");
        assert_eq!(requires_context(&ContextOnly), "workspace.read");
        assert_eq!(requires_memory(&MemoryOnly), "session");
        assert_eq!(requires_compactor(&CompactorOnly), "truncate-head");
        assert_eq!(requires_agent(&AgentOnly), "single-turn");
        assert_eq!(requires_command(&CommandOnly), "ai.ask");
        assert_eq!(requires_ui(&UiOnly), "panel: transcript");
    }

    #[test]
    fn multi_point_struct_needs_explicit_impl_per_point() {
        fn requires_model(point: &impl ModelPoint) -> &str {
            point.declared_model()
        }
        fn requires_tool(point: &impl ToolPoint) -> &str {
            point.effect_class()
        }
        let both = ToolAndModel;
        // Both bounds resolve only because both impl blocks exist above.
        assert_eq!(requires_tool(&both), "read-only");
        assert_eq!(requires_model(&both), "fake-chat");
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
            [
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
    fn manifest_is_data_only() {
        let manifest = Manifest {
            name: "example-extension".to_owned(),
            version: "0.1.0".to_owned(),
            host: "bitty-ai >=0.1.0".to_owned(),
            points: vec![ExtensionPoint::Tool],
        };
        // The host predicate round-trips verbatim: stored, never parsed.
        assert_eq!(manifest, manifest.clone());
        assert_eq!(manifest.host, "bitty-ai >=0.1.0");
        assert_eq!(manifest.points, [ExtensionPoint::Tool]);
    }
}

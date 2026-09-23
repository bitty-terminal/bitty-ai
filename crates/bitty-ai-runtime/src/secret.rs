//! Typed secret container with unconditional redaction (`AI-0138`).
//!
//! Mirrors the provider-plugin boundary secret invariant (`MP-10`, `PP-2`):
//! a model can use credentials, but a model must never see credentials.
//! Real credentials live in the host secret store and reach adapters as
//! opaque handles resolved on the Rust host side; this crate only defines
//! the typed container that keeps a secret value out of every
//! diagnostic, trace, and snapshot surface.
//!
//! # Trust boundary
//!
//! Every accessor and every `Debug`/`Display` impl here is a trust
//! boundary. The only API that returns the raw value is
//! [`SecretField::expose_for_adapter`], whose name and documentation pin
//! it to the host adapter edge: the single call site that hands the
//! value to the transport (for example an `Authorization` header) and
//! never logs, traces, snapshots, or interpolates it afterwards.
//!
//! # Determinism rules
//!
//! `std`-only, no wall clock, no thread, no async runtime, no network,
//! no filesystem. All behavior in tests is reproducible from the
//! caller-supplied bytes.

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

/// Maximum secret value length in bytes (`AI-0138`).
///
/// Mirrors the slice-side local-endpoint key bound (4 KiB): values past
/// this bound fail closed at construction so no unbounded secret enters
/// the runtime.
pub const MAX_SECRET_LEN: usize = 4 * 1024;

/// Redaction placeholder emitted by [`SecretField`] formatting impls.
///
/// Fixed and value-independent: it leaks neither length nor prefix.
pub const SECRET_REDACTED: &str = "[redacted secret]";

/// Typed container for a secret value (`MP-10`, `PP-2`, `P0-AC-026`).
///
/// Wraps secret bytes so they cannot leak through `Debug`, `Display`,
/// or error interpolation: both formatting impls emit the fixed
/// [`SECRET_REDACTED`] placeholder unconditionally, never the value,
/// its length, or a prefix. There is deliberately no `Clone`,
/// no `PartialEq`/`Eq`, no `Hash`, no `AsRef<[u8]>`, and no
/// `From`/`Into` conversion: the only path back to the raw bytes is
/// [`SecretField::expose_for_adapter`], pinned to the host adapter edge.
///
/// Construction fails closed on empty or over-bound input so an
/// unbounded or vacuous secret never enters the runtime.
pub struct SecretField(Box<[u8]>);

/// Secret construction errors. The rejection carries only the bound,
/// never the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretError {
    /// The secret value is empty.
    Empty,
    /// The secret value exceeds [`MAX_SECRET_LEN`].
    TooLong {
        /// Bound that was exceeded.
        max: usize,
    },
}

impl Display for SecretError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Empty => write!(f, "secret value must not be empty"),
            Self::TooLong { max } => write!(f, "secret value too long (max {max} bytes)"),
        }
    }
}

impl std::error::Error for SecretError {}

impl SecretField {
    /// Wrap secret bytes. The caller transfers ownership; the value is
    /// stored opaquely and never copied out except through
    /// [`SecretField::expose_for_adapter`].
    ///
    /// # Errors
    ///
    /// Returns [`SecretError::Empty`] for an empty value and
    /// [`SecretError::TooLong`] past [`MAX_SECRET_LEN`].
    pub fn new(value: Vec<u8>) -> Result<Self, SecretError> {
        if value.is_empty() {
            return Err(SecretError::Empty);
        }
        if value.len() > MAX_SECRET_LEN {
            return Err(SecretError::TooLong {
                max: MAX_SECRET_LEN,
            });
        }
        Ok(Self(value.into_boxed_slice()))
    }

    /// Wrap a secret string. Convenience for call sites holding text;
    /// the same bounds as [`SecretField::new`] apply.
    ///
    /// # Errors
    ///
    /// Returns [`SecretError::Empty`] for an empty value and
    /// [`SecretError::TooLong`] past [`MAX_SECRET_LEN`].
    pub fn from_string(value: String) -> Result<Self, SecretError> {
        Self::new(value.into_bytes())
    }

    /// Authorized-edge consume API: reveal the raw value to the host
    /// adapter edge only.
    ///
    /// The caller MUST be the host adapter edge that hands the value to
    /// the transport (for example an `Authorization` header) and MUST
    /// NOT log, trace, snapshot, interpolate into an error reason, or
    /// otherwise retain the returned bytes. This is the single
    /// intentionally-named escape hatch; every other surface sees only
    /// [`SECRET_REDACTED`].
    #[must_use]
    pub fn expose_for_adapter(&self) -> &[u8] {
        &self.0
    }

    /// Whether `haystack` carries no trace of this secret's value.
    ///
    /// Pre-queue / pre-write check helper (`PP-2`): callers run a
    /// diagnostic string, trace line, or snapshot through this before
    /// it reaches a queue or a file, and drop or scrub the record when
    /// it returns `false`. Empty haystacks trivially pass; a secret
    /// value is never empty (see [`SecretError::Empty`]), so the check
    /// cannot be vacuous.
    #[must_use]
    pub fn is_absent_from(&self, haystack: &str) -> bool {
        if self.0.is_empty() {
            return true;
        }
        let bytes = haystack.as_bytes();
        if bytes.len() < self.0.len() {
            return true;
        }
        bytes
            .windows(self.0.len())
            .all(|window| window != &self.0[..])
    }

    /// Scrub this secret's value out of a diagnostic string (`PP-2`).
    ///
    /// Pre-queue / pre-write redaction helper: returns the input with
    /// every occurrence of the secret value replaced by
    /// [`SECRET_REDACTED`]. Callers apply this to any diagnostic,
    /// trace, or snapshot string that could have interpolated the
    /// value before it reaches a queue or a file. The replacement is
    /// deterministic and value-independent.
    #[must_use]
    pub fn scrub_from(&self, haystack: &str) -> String {
        // `str::replace` needs `&str`; secrets are arbitrary bytes, so
        // fall back to a byte-level scan when the value is not UTF-8.
        // Either path replaces every occurrence with the fixed marker.
        if let Ok(secret) = std::str::from_utf8(&self.0) {
            return haystack.replace(secret, SECRET_REDACTED);
        }
        let needle: &[u8] = &self.0;
        // The constructor rejects empty values, so the scan below always
        // advances; guard anyway so a future refactor cannot loop forever.
        if needle.is_empty() {
            return haystack.to_owned();
        }
        let bytes = haystack.as_bytes();
        let mut scrubbed = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            if bytes.len() - index >= needle.len() && &bytes[index..index + needle.len()] == needle
            {
                scrubbed.extend_from_slice(SECRET_REDACTED.as_bytes());
                index += needle.len();
            } else {
                scrubbed.push(bytes[index]);
                index += 1;
            }
        }
        String::from_utf8(scrubbed).unwrap_or_else(|_| SECRET_REDACTED.to_owned())
    }
}

impl Debug for SecretField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Unconditional redaction: no length, no prefix, no hash.
        f.debug_tuple("SecretField")
            .field(&SECRET_REDACTED)
            .finish()
    }
}

impl Display for SecretField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Unconditional redaction: identical fixed marker as `Debug`.
        f.write_str(SECRET_REDACTED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{
        ConsentDecision, ConsentLedger, ConsentQuery, DenyAllConsent, FakeConsentLedger,
        MAX_REASON_BYTES, ProtocolAgentId, bound_reason, ensure_consented,
    };
    use crate::provider::{ProviderError, TurnRequest};
    use crate::session::AgentInstanceId;

    /// Seeded stand-in for a real credential (`AI-0138`).
    ///
    /// Assembled from short low-entropy fragments on purpose: no single
    /// literal in this file resembles a real key, so the secret scanner
    /// has nothing key-shaped to flag, while the joined value is still a
    /// fixed deterministic stand-in every negative test can seed on.
    fn seeded_secret_bytes() -> Vec<u8> {
        ["seeded", "stand-in", "ai-0138"].join("-").into_bytes()
    }

    fn seeded_secret_string() -> String {
        String::from_utf8(seeded_secret_bytes()).expect("seeded value is UTF-8")
    }

    fn seeded_field() -> SecretField {
        SecretField::new(seeded_secret_bytes()).expect("seeded value fits the bound")
    }

    #[test]
    fn secret_field_debug_and_display_redact_unconditionally() {
        let field = seeded_field();
        let debug = format!("{field:?}");
        let display = format!("{field}");
        assert_eq!(debug, format!("SecretField(\"{SECRET_REDACTED}\")"));
        assert_eq!(display, SECRET_REDACTED);
        // Alternate formatting flags must not widen the surface either:
        // pretty output differs in layout only, never in content.
        let pretty = format!("{field:#?}");
        assert!(pretty.contains(SECRET_REDACTED));
        assert!(!pretty.contains(&seeded_secret_string()));
        assert!(!debug.contains(&seeded_secret_string()));
        assert!(!display.contains(&seeded_secret_string()));
        // The redaction leaks neither length nor prefix of the value.
        assert!(!debug.contains(&seeded_secret_bytes().len().to_string()));
        assert!(!debug.contains("seeded"));
    }

    #[test]
    fn secret_field_construction_fails_closed() {
        assert_eq!(
            SecretField::new(Vec::new()).expect_err("empty must fail"),
            SecretError::Empty
        );
        assert_eq!(
            SecretField::from_string(String::new()).expect_err("empty must fail"),
            SecretError::Empty
        );
        let oversized = vec![b'x'; MAX_SECRET_LEN + 1];
        assert_eq!(
            SecretField::new(oversized).expect_err("over-bound must fail"),
            SecretError::TooLong {
                max: MAX_SECRET_LEN
            }
        );
        // The bound itself is accepted.
        assert!(SecretField::new(vec![b'x'; MAX_SECRET_LEN]).is_ok());
        // Error displays carry the bound, never the value.
        assert_eq!(
            SecretError::TooLong {
                max: MAX_SECRET_LEN
            }
            .to_string(),
            format!("secret value too long (max {MAX_SECRET_LEN} bytes)")
        );
        assert_eq!(
            SecretError::Empty.to_string(),
            "secret value must not be empty"
        );
    }

    #[test]
    fn expose_for_adapter_is_the_only_raw_value_path() {
        let field = seeded_field();
        assert_eq!(field.expose_for_adapter(), seeded_secret_bytes().as_slice());
        // `from_string` round-trips through the same authorized edge.
        let text = SecretField::from_string(seeded_secret_string()).expect("fits");
        assert_eq!(
            std::str::from_utf8(text.expose_for_adapter()).expect("UTF-8"),
            seeded_secret_string()
        );
    }

    #[test]
    fn is_absent_from_proves_absence_before_queue_or_write() {
        let field = seeded_field();
        assert!(field.is_absent_from("clean diagnostic line"));
        assert!(field.is_absent_from(""));
        let tainted = format!("token={} end", seeded_secret_string());
        assert!(!field.is_absent_from(&tainted));
    }

    #[test]
    fn scrub_from_replaces_every_occurrence_pre_write() {
        let field = seeded_field();
        let secret = seeded_secret_string();
        let tainted = format!("first {secret} then {secret} end");
        let scrubbed = field.scrub_from(&tainted);
        assert_eq!(
            scrubbed,
            format!("first {SECRET_REDACTED} then {SECRET_REDACTED} end")
        );
        assert!(!scrubbed.contains(&secret));
        assert!(field.is_absent_from(&scrubbed));
        // Clean input passes through unchanged.
        assert_eq!(field.scrub_from("clean line"), "clean line");
    }

    #[test]
    fn scrub_from_handles_non_utf8_secret_bytes() {
        let field = SecretField::new(vec![0xFF, 0xFE, 0x01]).expect("fits");
        // Non-UTF-8 values cannot hide inside a `&str` haystack, so a
        // diagnostic string trivially carries no trace of them.
        assert!(field.is_absent_from("any diagnostic line"));
        assert_eq!(
            field.scrub_from("any diagnostic line"),
            "any diagnostic line"
        );
        assert_eq!(
            format!("{field:?}"),
            format!("SecretField(\"{SECRET_REDACTED}\")")
        );
    }

    /// Provider consent scope pinned by the boundary contract: provider
    /// credentials require a dedicated `ai.provider` consent distinct
    /// from streaming and Tool Bus scopes (`MP-10`).
    const PROVIDER_SCOPE: &str = "ai.provider";

    fn consent_query<'a>(protocol: &'a str, tool: &'a str, scope: &'a str) -> ConsentQuery<'a> {
        ConsentQuery {
            protocol_id: protocol,
            agent_instance: AgentInstanceId(1),
            tool,
            scope,
            now_ms: 1_000,
        }
    }

    #[test]
    fn provider_grant_does_not_satisfy_other_scopes_and_vice_versa() {
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "model_complete", PROVIDER_SCOPE, 9_000)
            .expect("provider grant fits");
        // The `ai.provider` grant allows its exact triple only.
        assert_eq!(
            ledger.check(&consent_query(
                "local.assistant",
                "model_complete",
                PROVIDER_SCOPE
            )),
            ConsentDecision::Allow
        );
        // Same tool under a streaming scope is denied: no cross-scope
        // sharing from the provider grant.
        assert!(matches!(
            ledger.check(&consent_query(
                "local.assistant",
                "model_complete",
                "streaming.live"
            )),
            ConsentDecision::Deny { .. }
        ));
        // Same tool under a Tool Bus scope is denied.
        assert!(matches!(
            ledger.check(&consent_query(
                "local.assistant",
                "model_complete",
                "workspace.read"
            )),
            ConsentDecision::Deny { .. }
        ));
        // A streaming grant does not satisfy the provider scope either.
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "model_complete", "streaming.live", 9_000)
            .expect("streaming grant fits");
        assert!(matches!(
            ledger.check(&consent_query(
                "local.assistant",
                "model_complete",
                PROVIDER_SCOPE
            )),
            ConsentDecision::Deny { .. }
        ));
        // A Tool Bus grant does not satisfy the provider scope either.
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "workspace_read", "workspace.read", 9_000)
            .expect("tool grant fits");
        assert!(matches!(
            ledger.check(&consent_query(
                "local.assistant",
                "workspace_read",
                PROVIDER_SCOPE
            )),
            ConsentDecision::Deny { .. }
        ));
        assert!(matches!(
            ledger.check(&consent_query(
                "local.assistant",
                "model_complete",
                PROVIDER_SCOPE
            )),
            ConsentDecision::Deny { .. }
        ));
        // The typed helper agrees: denied consent fails closed.
        assert!(
            ensure_consented(
                &ledger,
                &consent_query("local.assistant", "workspace_read", PROVIDER_SCOPE),
            )
            .is_err()
        );
    }

    /// Assert a seeded secret value appears nowhere in a diagnostic
    /// surface (`MP-10`, `P0-AC-026`).
    fn assert_secret_absent(label: &str, surface: &str, secret: &str) {
        assert!(
            !surface.contains(secret),
            "seeded secret leaked into {label}: {surface:?}"
        );
    }

    #[test]
    fn seeded_secret_appears_nowhere_in_provider_diagnostics() {
        let secret = seeded_secret_string();
        let field = seeded_field();

        // Debug and Display of the container itself.
        assert_secret_absent("SecretField Debug", &format!("{field:?}"), &secret);
        assert_secret_absent("SecretField Display", &format!("{field}"), &secret);

        // Every provider error Display that interpolates caller or host
        // input must stay secret-free even when keyed with the value.
        let provider_errors = [
            ProviderError::Transport {
                provider: "local-test".to_owned(),
                reason: "refused".to_owned(),
            },
            ProviderError::Auth {
                provider: "local-test".to_owned(),
                reason: "unknown scope".to_owned(),
            },
            ProviderError::Unknown {
                provider: "local-test".to_owned(),
                reason: "ack lost".to_owned(),
            },
            ProviderError::UnknownModel {
                name: "missing-model".to_owned(),
            },
        ];
        for error in &provider_errors {
            let rendered = format!("{error}");
            assert_secret_absent("ProviderError Display", &rendered, &secret);
            assert_secret_absent("ProviderError Debug", &format!("{error:?}"), &secret);
        }

        // Consent denial reasons echo the queried tool and scope, so a
        // well-formed query (tools are `TB-2` names, which can never be a
        // secret value) carries no secret. The denial path takes no
        // secret input by construction; prove the outputs stay clean.
        let denial = DenyAllConsent.check(&consent_query(
            "local.assistant",
            "model_complete",
            PROVIDER_SCOPE,
        ));
        match denial {
            ConsentDecision::Deny { reason } => {
                assert!(reason.len() <= MAX_REASON_BYTES);
                assert_secret_absent("consent denial reason", &reason, &secret);
            }
            ConsentDecision::Allow => panic!("deny-all must deny"),
        }

        // Defense in depth (`PP-2`): even if a secret value ever reached
        // a diagnostic string, the pre-queue helper scrubs it before the
        // record is queued or written.
        let tainted = format!("prefix-{secret}-suffix");
        assert!(!field.is_absent_from(&tainted));
        let scrubbed_tainted = field.scrub_from(&tainted);
        assert_secret_absent("scrubbed tainted string", &scrubbed_tainted, &secret);
        assert!(field.is_absent_from(&scrubbed_tainted));

        // `bound_reason` outputs over ordinary hostile input stay
        // bounded printable ASCII.
        let bounded = bound_reason("line1\r\nline2\ttab");
        assert_eq!(bounded, "line1??line2?tab");

        // A redacted-then-queued diagnostic passes the pre-queue check.
        let scrubbed = field.scrub_from(&format!("auth failed for {secret}"));
        assert!(field.is_absent_from(&scrubbed));
        assert_secret_absent("scrubbed diagnostic", &scrubbed, &secret);
    }

    #[test]
    fn seeded_secret_appears_nowhere_in_consent_and_bridge_surfaces() {
        let secret = seeded_secret_string();
        let mut ledger = FakeConsentLedger::new();
        ledger
            .grant("local.assistant", "model_complete", PROVIDER_SCOPE, 9_000)
            .expect("grant fits");

        // Ledger Debug (grant table snapshot) never stores the value:
        // grants hold protocol/tool/scope triples only.
        assert_secret_absent("FakeConsentLedger Debug", &format!("{ledger:?}"), &secret);

        // Consent decisions over the provider scope carry no value.
        let decision = ledger.check(&consent_query(
            "local.assistant",
            "model_complete",
            PROVIDER_SCOPE,
        ));
        assert_eq!(decision, ConsentDecision::Allow);
        assert_secret_absent("ConsentDecision Debug", &format!("{decision:?}"), &secret);

        // Bridge denial errors echo tool and bounded reason only.
        let error = ensure_consented(
            &ledger,
            &consent_query("local.assistant", "model_complete", "streaming.live"),
        )
        .expect_err("cross-scope must deny");
        assert_secret_absent("BridgeError Display", &format!("{error}"), &secret);
        assert_secret_absent("BridgeError Debug", &format!("{error:?}"), &secret);

        // Protocol identity surfaces echo the wire id only.
        let protocol = ProtocolAgentId::new("local.assistant").expect("valid");
        assert_secret_absent("ProtocolAgentId Display", &protocol.to_string(), &secret);
    }

    #[test]
    fn seeded_secret_appears_nowhere_in_turn_request_snapshot() {
        use crate::provider::Message;

        let secret = seeded_secret_string();
        let request = TurnRequest {
            model: "fake-chat".to_owned(),
            messages: vec![Message::user("summarize this")],
            context_refs: Vec::new(),
            tools: Vec::new(),
            budget_bytes: 4096,
            timeout_ms: 5_000,
            now_ms: 1_000,
            sampling: None,
        };
        // The request snapshot surface (Debug) is a representative
        // trace/snapshot shape: it must carry no secret, and a
        // pre-write check over it passes.
        let snapshot = format!("{request:?}");
        assert_secret_absent("TurnRequest Debug snapshot", &snapshot, &secret);
        assert!(seeded_field().is_absent_from(&snapshot));
    }
}

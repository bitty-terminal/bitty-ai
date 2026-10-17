//! Helper-side provider/routing snapshot projection and equality proof.
//!
//! This is a pure, in-memory seam for the first `ai.provider_list` candidate.
//! It does not wire strict dispatch, transport, provider implementations, or
//! credentials. The generic metadata publication handshake remains separate.

use bitty_ai_runtime::{ModelCapability, ProviderRegistry};

use crate::JVal;

use super::metadata_publication::{
    MAX_CAPABILITIES_PER_MODEL, MAX_METADATA_FIELD_BYTES, MAX_MODELS_PER_PROVIDER, MAX_PROVIDERS,
    MAX_PUBLICATION_FRAME_BYTES,
};

/// Domain tag for the provider/routing projection.
// POLICY (AI-0324): this domain is never interchangeable with MCP/tool-list
// metadata, even when numeric snapshot versions happen to match.
pub const PROVIDER_ROUTING_DOMAIN: &str = "provider-routing";

/// Maximum total models in one provider/routing projection.
pub const MAX_PROVIDER_ROUTING_MODELS: usize = MAX_MODELS_PER_PROVIDER;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderRoutingError {
    FrameTooLarge,
    InvalidJson,
    InvalidDomain,
    InvalidIdentity,
    InvalidDigest,
    InvalidProjection,
    NonCanonicalProjection,
    StaleOrReplay,
    PublicationIdReused,
    PublicationLimit,
    Superseded,
    NoPublication,
    NoAcknowledgement,
    ProjectionMismatch,
}

/// Exact domain-tagged publication identity accepted by the helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRoutingPublicationAck {
    publication_id: String,
    domain: String,
    snapshot_version: u64,
    digest: String,
    canonical_bytes: Vec<u8>,
}

impl ProviderRoutingPublicationAck {
    #[must_use]
    pub fn publication_id(&self) -> &str {
        &self.publication_id
    }

    #[must_use]
    pub fn domain(&self) -> &str {
        &self.domain
    }

    #[must_use]
    pub fn snapshot_version(&self) -> u64 {
        self.snapshot_version
    }

    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    #[must_use]
    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub(crate) fn same_identity(&self, other: &Self) -> bool {
        self.publication_id == other.publication_id
            && self.domain == other.domain
            && self.snapshot_version == other.snapshot_version
            && self.digest == other.digest
    }
}

/// Canonical projection and its lowercase SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRoutingProjection {
    canonical_bytes: Vec<u8>,
    digest: String,
}

impl ProviderRoutingProjection {
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// Opaque same-domain equality proof. It contains no registry mutation path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRoutingBinding {
    publication_id: String,
    domain: String,
    snapshot_version: u64,
    digest: String,
    canonical_bytes: Vec<u8>,
    registry_revision: u64,
    publication_epoch: u64,
}

impl ProviderRoutingBinding {
    pub(crate) fn new(
        ack: &ProviderRoutingPublicationAck,
        publication_epoch: u64,
        registry_revision: u64,
    ) -> Self {
        Self {
            publication_id: ack.publication_id.clone(),
            domain: ack.domain.clone(),
            snapshot_version: ack.snapshot_version,
            digest: ack.digest.clone(),
            canonical_bytes: ack.canonical_bytes.clone(),
            registry_revision,
            publication_epoch,
        }
    }

    pub(crate) fn registry_revision(&self) -> u64 {
        self.registry_revision
    }

    pub(crate) fn publication_epoch(&self) -> u64 {
        self.publication_epoch
    }
    pub(crate) fn publication_id(&self) -> &str {
        &self.publication_id
    }
    pub(crate) fn domain(&self) -> &str {
        &self.domain
    }
    pub(crate) fn snapshot_version(&self) -> u64 {
        self.snapshot_version
    }
    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

/// Project the populated runtime helper registry into exact canonical bytes.
pub fn project_registry(
    registry: &ProviderRegistry,
) -> Result<ProviderRoutingProjection, ProviderRoutingError> {
    // POLICY (AI-0324): aliases and provider implementation state are excluded
    // from this first projection; alias mutation still invalidates the proof
    // through ProviderRegistry::revision.
    let entries = registry.entries();
    if entries.len() > MAX_PROVIDER_ROUTING_MODELS {
        return Err(ProviderRoutingError::InvalidProjection);
    }

    for entry in &entries {
        validate_text(&entry.provider_id)?;
        validate_text(&entry.name)?;
        if entry.capabilities.len() > MAX_CAPABILITIES_PER_MODEL {
            return Err(ProviderRoutingError::InvalidProjection);
        }
        let mut capabilities: Vec<&str> = entry.capabilities.iter().map(capability_label).collect();
        capabilities.sort_unstable();
        if capabilities.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ProviderRoutingError::InvalidProjection);
        }
    }

    // POLICY (AI-0324): field order and bytewise provider/model/capability
    // ordering are part of the digest, never an incidental serializer choice.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(br#"{"domain":"provider-routing","providers":["#);
    let mut provider_index = 0;
    let mut entry_index = 0;
    while entry_index < entries.len() {
        let provider_id = &entries[entry_index].provider_id;
        let provider_start = entry_index;
        while entry_index < entries.len() && entries[entry_index].provider_id == *provider_id {
            entry_index += 1;
        }
        let models = &entries[provider_start..entry_index];
        if provider_index >= MAX_PROVIDERS || models.len() > MAX_MODELS_PER_PROVIDER {
            return Err(ProviderRoutingError::InvalidProjection);
        }
        if provider_index > 0 {
            bytes.push(b',');
        }
        bytes.extend_from_slice(br#"{"id":"#);
        append_json_string(&mut bytes, provider_id);
        bytes.extend_from_slice(br#","models":["#);
        for (model_index, model) in models.iter().enumerate() {
            if model_index > 0 {
                bytes.push(b',');
            }
            bytes.extend_from_slice(br#"{"name":"#);
            append_json_string(&mut bytes, &model.name);
            bytes.extend_from_slice(br#","capabilities":["#);
            let mut capabilities: Vec<&str> =
                model.capabilities.iter().map(capability_label).collect();
            capabilities.sort_unstable();
            for (capability_index, capability) in capabilities.iter().enumerate() {
                if capability_index > 0 {
                    bytes.push(b',');
                }
                append_json_string(&mut bytes, capability);
            }
            bytes.extend_from_slice(b"],\"context_window_tokens\":");
            bytes.extend_from_slice(model.context_window_tokens.to_string().as_bytes());
            bytes.extend_from_slice(b",\"input_cost_weight\":");
            bytes.extend_from_slice(model.input_cost_weight.to_string().as_bytes());
            bytes.extend_from_slice(b",\"output_cost_weight\":");
            bytes.extend_from_slice(model.output_cost_weight.to_string().as_bytes());
            bytes.push(b'}');
        }
        bytes.extend_from_slice(b"]}");
        provider_index += 1;
    }
    bytes.extend_from_slice(b"]}");
    if bytes.len() > MAX_PUBLICATION_FRAME_BYTES {
        return Err(ProviderRoutingError::FrameTooLarge);
    }
    let digest = super::metadata_publication::sha256_hex(&bytes);
    Ok(ProviderRoutingProjection {
        canonical_bytes: bytes,
        digest,
    })
}

fn validate_text(value: &str) -> Result<(), ProviderRoutingError> {
    if value.is_empty()
        || value.len() > MAX_METADATA_FIELD_BYTES
        || value.chars().any(char::is_control)
    {
        Err(ProviderRoutingError::InvalidProjection)
    } else {
        Ok(())
    }
}

fn capability_label(capability: &ModelCapability) -> &'static str {
    match capability {
        ModelCapability::Text => "text",
        ModelCapability::Streaming => "streaming",
        ModelCapability::ToolUse => "tool_use",
        ModelCapability::ImageInput => "image_input",
        ModelCapability::AudioInput => "audio_input",
        ModelCapability::AudioOutput => "audio_output",
        ModelCapability::VideoInput => "video_input",
        ModelCapability::Reasoning => "reasoning",
    }
}

fn append_json_string(out: &mut Vec<u8>, value: &str) {
    out.push(b'"');
    for character in value.chars() {
        match character {
            '"' => out.extend_from_slice(br#"\"#),
            '\\' => out.extend_from_slice(br#"\\"#),
            '\u{0008}' => out.extend_from_slice(br#"\b"#),
            '\u{000c}' => out.extend_from_slice(br#"\f"#),
            '\n' => out.extend_from_slice(br#"\n"#),
            '\r' => out.extend_from_slice(br#"\r"#),
            '\t' => out.extend_from_slice(br#"\t"#),
            '\u{0000}'..='\u{001f}' => {
                out.extend_from_slice(format!(r#"\u{:04x}"#, character as u32).as_bytes())
            }
            _ => out.extend_from_slice(character.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
    out.push(b'"');
}

pub(crate) fn parse_provider_routing_publication(
    frame: &[u8],
) -> Result<(ProviderRoutingPublicationAck, ProviderRoutingProjection), ProviderRoutingError> {
    if frame.len() > MAX_PUBLICATION_FRAME_BYTES {
        return Err(ProviderRoutingError::FrameTooLarge);
    }
    let top = super::metadata_publication::parse_object(frame)
        .map_err(|_| ProviderRoutingError::InvalidJson)?;
    super::metadata_publication::reject_duplicate_keys(&top)
        .map_err(|_| ProviderRoutingError::InvalidJson)?;
    exact_keys(
        &top,
        &[
            "publication_id",
            "snapshot_version",
            "domain",
            "digest",
            "metadata_json",
        ],
    )?;
    let publication_id = string_field(&top, "publication_id")?.to_owned();
    let domain = string_field(&top, "domain")?;
    let digest = string_field(&top, "digest")?.to_owned();
    let version = snapshot_version_field(&top)?;
    let payload = string_field(&top, "metadata_json")?;
    if !super::metadata_publication::valid_identity(&publication_id)
        || publication_id.len() > MAX_METADATA_FIELD_BYTES
    {
        return Err(ProviderRoutingError::InvalidIdentity);
    }
    if domain != PROVIDER_ROUTING_DOMAIN || digest.len() != 64 || !is_lower_hex(digest.as_bytes()) {
        return Err(if domain != PROVIDER_ROUTING_DOMAIN {
            ProviderRoutingError::InvalidDomain
        } else {
            ProviderRoutingError::InvalidDigest
        });
    }
    if payload.len() > MAX_PUBLICATION_FRAME_BYTES {
        return Err(ProviderRoutingError::FrameTooLarge);
    }
    let payload_value = super::metadata_publication::parse_object(payload.as_bytes())
        .map_err(|_| ProviderRoutingError::InvalidProjection)?;
    let projection = project_payload(&payload_value)?;
    if projection.canonical_bytes() != payload.as_bytes() {
        return Err(ProviderRoutingError::NonCanonicalProjection);
    }
    if projection.digest() != digest {
        return Err(ProviderRoutingError::InvalidDigest);
    }
    Ok((
        ProviderRoutingPublicationAck {
            publication_id,
            domain: domain.to_owned(),
            snapshot_version: version,
            digest,
            canonical_bytes: projection.canonical_bytes().to_vec(),
        },
        projection,
    ))
}

pub(crate) fn parse_provider_routing_ack(
    frame: &[u8],
) -> Result<ProviderRoutingPublicationAck, ProviderRoutingError> {
    if frame.len() > MAX_PUBLICATION_FRAME_BYTES {
        return Err(ProviderRoutingError::FrameTooLarge);
    }
    let top = super::metadata_publication::parse_object(frame)
        .map_err(|_| ProviderRoutingError::InvalidJson)?;
    super::metadata_publication::reject_duplicate_keys(&top)
        .map_err(|_| ProviderRoutingError::InvalidJson)?;
    exact_keys(
        &top,
        &["publication_id", "snapshot_version", "domain", "digest"],
    )?;
    let publication_id = string_field(&top, "publication_id")?.to_owned();
    let domain = string_field(&top, "domain")?.to_owned();
    let digest = string_field(&top, "digest")?.to_owned();
    if !super::metadata_publication::valid_identity(&publication_id) {
        return Err(ProviderRoutingError::InvalidIdentity);
    }
    if domain != PROVIDER_ROUTING_DOMAIN {
        return Err(ProviderRoutingError::InvalidDomain);
    }
    if digest.len() != 64 || !is_lower_hex(digest.as_bytes()) {
        return Err(ProviderRoutingError::InvalidDigest);
    }
    Ok(ProviderRoutingPublicationAck {
        publication_id,
        domain,
        snapshot_version: snapshot_version_field(&top)?,
        digest,
        canonical_bytes: Vec::new(),
    })
}

fn project_payload(value: &JVal) -> Result<ProviderRoutingProjection, ProviderRoutingError> {
    let fields = value
        .as_obj()
        .ok_or(ProviderRoutingError::InvalidProjection)?;
    if fields.len() != 2 {
        return Err(ProviderRoutingError::InvalidProjection);
    }
    if string_field(value, "domain")? != PROVIDER_ROUTING_DOMAIN {
        return Err(ProviderRoutingError::InvalidDomain);
    }
    let providers = value
        .field("providers")
        .and_then(JVal::as_arr)
        .ok_or(ProviderRoutingError::InvalidProjection)?;
    if providers.len() > MAX_PROVIDERS {
        return Err(ProviderRoutingError::InvalidProjection);
    }
    let mut last_provider = None;
    let mut model_count = 0;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(br#"{"domain":"provider-routing","providers":["#);
    for (provider_index, provider) in providers.iter().enumerate() {
        let provider_fields = provider
            .as_obj()
            .ok_or(ProviderRoutingError::InvalidProjection)?;
        if provider_fields.len() != 2 {
            return Err(ProviderRoutingError::InvalidProjection);
        }
        let id = string_field(provider, "id")?;
        validate_text(id)?;
        if last_provider.is_some_and(|last: &str| last >= id) {
            return Err(ProviderRoutingError::InvalidProjection);
        }
        last_provider = Some(id);
        let models = provider
            .field("models")
            .and_then(JVal::as_arr)
            .ok_or(ProviderRoutingError::InvalidProjection)?;
        if models.len() > MAX_MODELS_PER_PROVIDER {
            return Err(ProviderRoutingError::InvalidProjection);
        }
        if provider_index > 0 {
            bytes.push(b',');
        }
        bytes.extend_from_slice(br#"{"id":"#);
        append_json_string(&mut bytes, id);
        bytes.extend_from_slice(br#","models":["#);
        let mut last_model = None;
        for (model_index, model) in models.iter().enumerate() {
            model_count += 1;
            if model_count > MAX_PROVIDER_ROUTING_MODELS {
                return Err(ProviderRoutingError::InvalidProjection);
            }
            let model_fields = model
                .as_obj()
                .ok_or(ProviderRoutingError::InvalidProjection)?;
            if model_fields.len() != 5 {
                return Err(ProviderRoutingError::InvalidProjection);
            }
            let name = string_field(model, "name")?;
            validate_text(name)?;
            if last_model.is_some_and(|last: &str| last >= name) {
                return Err(ProviderRoutingError::InvalidProjection);
            }
            last_model = Some(name);
            let capabilities = model
                .field("capabilities")
                .and_then(JVal::as_arr)
                .ok_or(ProviderRoutingError::InvalidProjection)?;
            if capabilities.is_empty() || capabilities.len() > MAX_CAPABILITIES_PER_MODEL {
                return Err(ProviderRoutingError::InvalidProjection);
            }
            let mut labels = Vec::new();
            for capability in capabilities {
                let label = capability
                    .as_str()
                    .ok_or(ProviderRoutingError::InvalidProjection)?;
                validate_text(label)?;
                if ![
                    "text",
                    "streaming",
                    "tool_use",
                    "image_input",
                    "audio_input",
                    "audio_output",
                    "video_input",
                    "reasoning",
                ]
                .contains(&label)
                {
                    return Err(ProviderRoutingError::InvalidProjection);
                }
                labels.push(label);
            }
            if labels.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(ProviderRoutingError::InvalidProjection);
            }
            let context_window_tokens = integer_field(model, "context_window_tokens")?;
            let input_cost_weight = integer_field(model, "input_cost_weight")?;
            let output_cost_weight = integer_field(model, "output_cost_weight")?;
            if model_index > 0 {
                bytes.push(b',');
            }
            bytes.extend_from_slice(br#"{"name":"#);
            append_json_string(&mut bytes, name);
            bytes.extend_from_slice(br#","capabilities":["#);
            for (index, label) in labels.iter().enumerate() {
                if index > 0 {
                    bytes.push(b',');
                }
                append_json_string(&mut bytes, label);
            }
            bytes.extend_from_slice(b"],\"context_window_tokens\":");
            bytes.extend_from_slice(context_window_tokens.to_string().as_bytes());
            bytes.extend_from_slice(b",\"input_cost_weight\":");
            bytes.extend_from_slice(input_cost_weight.to_string().as_bytes());
            bytes.extend_from_slice(b",\"output_cost_weight\":");
            bytes.extend_from_slice(output_cost_weight.to_string().as_bytes());
            bytes.push(b'}');
        }
        bytes.extend_from_slice(b"]}");
    }
    bytes.extend_from_slice(b"]}");
    if bytes.len() > MAX_PUBLICATION_FRAME_BYTES {
        return Err(ProviderRoutingError::FrameTooLarge);
    }
    let digest = super::metadata_publication::sha256_hex(&bytes);
    Ok(ProviderRoutingProjection {
        canonical_bytes: bytes,
        digest,
    })
}

fn exact_keys(value: &JVal, expected: &[&str]) -> Result<(), ProviderRoutingError> {
    let fields = value.as_obj().ok_or(ProviderRoutingError::InvalidJson)?;
    if fields.len() != expected.len()
        || fields
            .iter()
            .any(|(key, _)| !expected.contains(&key.as_str()))
        || fields
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != expected.len()
    {
        return Err(ProviderRoutingError::InvalidJson);
    }
    Ok(())
}

fn string_field<'a>(value: &'a JVal, name: &str) -> Result<&'a str, ProviderRoutingError> {
    value
        .field(name)
        .and_then(JVal::as_str)
        .ok_or(ProviderRoutingError::InvalidJson)
}

fn integer_field(value: &JVal, name: &str) -> Result<u64, ProviderRoutingError> {
    value
        .field(name)
        .and_then(JVal::as_u64)
        .filter(|value| u32::try_from(*value).is_ok())
        .ok_or(ProviderRoutingError::InvalidProjection)
}

fn snapshot_version_field(value: &JVal) -> Result<u64, ProviderRoutingError> {
    value
        .field("snapshot_version")
        .and_then(JVal::as_u64)
        .ok_or(ProviderRoutingError::InvalidJson)
}

fn is_lower_hex(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata_publication::PublicationState;
    use bitty_ai_runtime::{ModelCapability, ModelRegistration};

    fn registry() -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        registry
            .register(
                ModelRegistration::new(
                    "zeta",
                    "model-z",
                    vec![ModelCapability::ToolUse, ModelCapability::Text],
                    8192,
                    2,
                    3,
                )
                .unwrap(),
            )
            .unwrap();
        registry
            .register(
                ModelRegistration::new(
                    "alpha",
                    "model-a",
                    vec![ModelCapability::Streaming, ModelCapability::Text],
                    4096,
                    1,
                    1,
                )
                .unwrap(),
            )
            .unwrap();
        registry
    }

    #[test]
    fn canonical_projection_is_sorted_and_domain_tagged() {
        let projection = project_registry(&registry()).unwrap();
        assert_eq!(
            projection.canonical_bytes(),
            br#"{"domain":"provider-routing","providers":[{"id":"alpha","models":[{"name":"model-a","capabilities":["streaming","text"],"context_window_tokens":4096,"input_cost_weight":1,"output_cost_weight":1}]},{"id":"zeta","models":[{"name":"model-z","capabilities":["text","tool_use"],"context_window_tokens":8192,"input_cost_weight":2,"output_cost_weight":3}]}]}"#
        );
        assert_eq!(projection.digest().len(), 64);
        assert_eq!(
            projection.digest(),
            "12502a3569f98452cb43ceeeb90d23b20776797b1f926646997a0990f603d286"
        );
        assert!(
            projection
                .digest()
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
    }

    #[test]
    fn registration_order_does_not_change_canonical_digest() {
        let mut reverse = ProviderRegistry::new();
        reverse
            .register(
                ModelRegistration::new(
                    "alpha",
                    "model-a",
                    vec![ModelCapability::Text, ModelCapability::Streaming],
                    4096,
                    1,
                    1,
                )
                .unwrap(),
            )
            .unwrap();
        reverse
            .register(
                ModelRegistration::new(
                    "zeta",
                    "model-z",
                    vec![ModelCapability::Text, ModelCapability::ToolUse],
                    8192,
                    2,
                    3,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            project_registry(&reverse).unwrap(),
            project_registry(&registry()).unwrap()
        );
    }

    fn one_model_registry(
        provider: &str,
        model: &str,
        capability: ModelCapability,
        window: u32,
        input_weight: u32,
        output_weight: u32,
    ) -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        registry
            .register(
                ModelRegistration::new(
                    provider,
                    model,
                    vec![capability],
                    window,
                    input_weight,
                    output_weight,
                )
                .unwrap(),
            )
            .unwrap();
        registry
    }

    #[test]
    fn every_projected_model_field_mismatch_refuses_binding() {
        let base = one_model_registry("alpha", "model-a", ModelCapability::Text, 4096, 1, 1);
        let variants = [
            one_model_registry("beta", "model-a", ModelCapability::Text, 4096, 1, 1),
            one_model_registry("alpha", "model-b", ModelCapability::Text, 4096, 1, 1),
            one_model_registry("alpha", "model-a", ModelCapability::Streaming, 4096, 1, 1),
            one_model_registry("alpha", "model-a", ModelCapability::Text, 8192, 1, 1),
            one_model_registry("alpha", "model-a", ModelCapability::Text, 4096, 2, 1),
            one_model_registry("alpha", "model-a", ModelCapability::Text, 4096, 1, 2),
        ];

        for (index, variant) in variants.iter().enumerate() {
            let projection = project_registry(variant).unwrap();
            let frame = publication_frame(
                index as u64 + 1,
                &format!("field-mismatch-{index}"),
                PROVIDER_ROUTING_DOMAIN,
                projection.canonical_bytes(),
            );
            let mut state = PublicationState::default();
            let ack = state.publish_provider_routing(&frame).unwrap();
            let ack = state
                .acknowledge_provider_routing(&acknowledgement_frame(&ack))
                .unwrap();
            assert_eq!(
                state.bind_provider_routing(&ack, &base),
                Err(ProviderRoutingError::ProjectionMismatch),
                "variant {index} unexpectedly proved equal"
            );
        }
    }

    #[test]
    fn same_revision_different_registry_projection_refuses_binding() {
        let published_registry =
            one_model_registry("alpha", "model-a", ModelCapability::Text, 4096, 1, 1);
        let current_registry =
            one_model_registry("beta", "model-a", ModelCapability::Text, 4096, 1, 1);
        assert_eq!(published_registry.revision(), current_registry.revision());
        assert_ne!(
            project_registry(&published_registry)
                .unwrap()
                .canonical_bytes(),
            project_registry(&current_registry)
                .unwrap()
                .canonical_bytes()
        );

        let projection = project_registry(&published_registry).unwrap();
        let frame = publication_frame(
            1,
            "same-revision-different-projection",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let mut state = PublicationState::default();
        let ack = state.publish_provider_routing(&frame).unwrap();
        let ack = state
            .acknowledge_provider_routing(&acknowledgement_frame(&ack))
            .unwrap();

        assert_eq!(
            state.bind_provider_routing(&ack, &current_registry),
            Err(ProviderRoutingError::ProjectionMismatch)
        );
    }

    #[test]
    fn identical_projection_in_distinct_registry_values_is_accepted() {
        let published_registry =
            one_model_registry("alpha", "model-a", ModelCapability::Text, 4096, 1, 1);
        let current_registry =
            one_model_registry("alpha", "model-a", ModelCapability::Text, 4096, 1, 1);
        assert_eq!(published_registry.revision(), current_registry.revision());
        assert_eq!(
            project_registry(&published_registry).unwrap(),
            project_registry(&current_registry).unwrap()
        );

        let projection = project_registry(&published_registry).unwrap();
        let frame = publication_frame(
            1,
            "identical-distinct-registry",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let mut state = PublicationState::default();
        let ack = state.publish_provider_routing(&frame).unwrap();
        let ack = state
            .acknowledge_provider_routing(&acknowledgement_frame(&ack))
            .unwrap();

        assert!(
            state.bind_provider_routing(&ack, &current_registry).is_ok(),
            "binding is projection-based, not an instance-identity claim"
        );
    }

    #[test]
    fn aliases_are_excluded_but_alias_mutation_changes_revision() {
        let mut registry = registry();
        let before = project_registry(&registry).unwrap();
        registry
            .register_alias(
                "default",
                vec![bitty_ai_runtime::ModelRef::new("alpha", "model-a").unwrap()],
            )
            .unwrap();
        let after = project_registry(&registry).unwrap();
        assert_eq!(before, after);
        assert_ne!(registry.revision(), 2);
    }

    #[test]
    fn capability_duplicates_refuse_projection() {
        let mut registry = ProviderRegistry::new();
        registry
            .register(ModelRegistration {
                provider_id: "alpha".into(),
                name: "model-a".into(),
                capabilities: vec![ModelCapability::Text, ModelCapability::Text],
                context_window_tokens: 1,
                input_cost_weight: 1,
                output_cost_weight: 1,
            })
            .unwrap();
        assert_eq!(
            project_registry(&registry),
            Err(ProviderRoutingError::InvalidProjection)
        );
    }

    fn publication_frame(version: u64, id: &str, domain: &str, payload: &[u8]) -> Vec<u8> {
        let digest = super::super::metadata_publication::sha256_hex(payload);
        let payload = String::from_utf8(payload.to_vec()).unwrap();
        let payload = payload.replace('\\', "\\\\").replace('"', "\\\"");
        let metadata_json = format!("\"{payload}\"");
        format!(
            r#"{{"publication_id":"{id}","snapshot_version":{version},"domain":"{domain}","digest":"{digest}","metadata_json":{}}}"#,
            metadata_json
        )
        .into_bytes()
    }

    fn acknowledgement_frame(ack: &ProviderRoutingPublicationAck) -> Vec<u8> {
        format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"domain":"{}","digest":"{}"}}"#,
            ack.publication_id(),
            ack.snapshot_version(),
            ack.domain(),
            ack.digest()
        )
        .into_bytes()
    }

    #[test]
    fn exact_publish_ack_and_binding_prove_registry_equality() {
        let registry = registry();
        let projection = project_registry(&registry).unwrap();
        let frame = publication_frame(
            1,
            "provider-pub-1",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let mut state = PublicationState::default();
        let ack = state.publish_provider_routing(&frame).unwrap();
        let acknowledged = state
            .acknowledge_provider_routing(&acknowledgement_frame(&ack))
            .unwrap();
        let binding = state
            .bind_provider_routing(&acknowledged, &registry)
            .unwrap();
        assert!(state.provider_routing_binding_is_current(&binding, &registry));
        assert_eq!(acknowledged.domain(), PROVIDER_ROUTING_DOMAIN);
        assert_eq!(acknowledged.digest(), projection.digest());
        assert_eq!(acknowledged.canonical_bytes(), projection.canonical_bytes());
        assert_eq!(binding.canonical_bytes(), projection.canonical_bytes());
    }

    #[test]
    fn caller_ack_clone_bytes_cannot_replace_retained_publication_bytes() {
        let registry = registry();
        let projection = project_registry(&registry).unwrap();
        let frame = publication_frame(
            1,
            "provider-pub-caller-clone",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let mut state = PublicationState::default();
        let published = state.publish_provider_routing(&frame).unwrap();
        let acknowledged = state
            .acknowledge_provider_routing(&acknowledgement_frame(&published))
            .unwrap();
        let mut caller_clone = acknowledged.clone();
        caller_clone.canonical_bytes = b"caller-owned-bytes".to_vec();

        let binding = state
            .bind_provider_routing(&caller_clone, &registry)
            .expect("identity-only caller clone selects retained helper record");
        assert_eq!(binding.canonical_bytes(), projection.canonical_bytes());
        assert!(state.provider_routing_binding_is_current(&binding, &registry));
    }

    #[test]
    fn binding_rejects_changed_retained_bytes_even_with_same_digest_and_revision() {
        let registry = registry();
        let projection = project_registry(&registry).unwrap();
        let frame = publication_frame(
            1,
            "provider-pub-bytes",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let mut state = PublicationState::default();
        let ack = state.publish_provider_routing(&frame).unwrap();
        let acknowledged = state
            .acknowledge_provider_routing(&acknowledgement_frame(&ack))
            .unwrap();
        let mut binding = state
            .bind_provider_routing(&acknowledged, &registry)
            .unwrap();

        binding.canonical_bytes = b"same-digest-different-bytes".to_vec();

        assert!(!state.provider_routing_binding_is_current(&binding, &registry));
    }

    #[test]
    fn wrong_domain_digest_and_projection_refuse() {
        let registry = registry();
        let projection = project_registry(&registry).unwrap();
        let mut state = PublicationState::default();
        let wrong_domain = publication_frame(
            1,
            "wrong-domain",
            "mcp-tool-list",
            projection.canonical_bytes(),
        );
        assert_eq!(
            state.publish_provider_routing(&wrong_domain),
            Err(ProviderRoutingError::InvalidDomain)
        );

        let wrong_digest = publication_frame(
            1,
            "wrong-digest",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        )
        .iter()
        .map(|byte| *byte as char)
        .collect::<String>()
        .replacen(&projection.digest().to_string(), &"0".repeat(64), 1);
        assert_eq!(
            state.publish_provider_routing(wrong_digest.as_bytes()),
            Err(ProviderRoutingError::InvalidDigest)
        );

        let changed_payload = br#"{"domain":"provider-routing","providers":[]}"#;
        let changed = publication_frame(
            1,
            "wrong-projection",
            PROVIDER_ROUTING_DOMAIN,
            changed_payload,
        );
        let changed_ack = state.publish_provider_routing(&changed).unwrap();
        let changed_ack = state
            .acknowledge_provider_routing(&acknowledgement_frame(&changed_ack))
            .unwrap();
        assert_eq!(
            state.bind_provider_routing(&changed_ack, &registry),
            Err(ProviderRoutingError::ProjectionMismatch)
        );
    }

    #[test]
    fn mutation_invalidates_binding_and_replay_is_refused() {
        let mut registry = registry();
        let projection = project_registry(&registry).unwrap();
        let frame = publication_frame(
            1,
            "provider-pub-1",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let mut state = PublicationState::default();
        let ack = state.publish_provider_routing(&frame).unwrap();
        let acknowledged = state
            .acknowledge_provider_routing(&acknowledgement_frame(&ack))
            .unwrap();
        let binding = state
            .bind_provider_routing(&acknowledged, &registry)
            .unwrap();
        assert!(state.provider_routing_binding_is_current(&binding, &registry));

        registry
            .register(
                ModelRegistration::new("alpha", "model-b", vec![ModelCapability::Text], 1, 1, 1)
                    .unwrap(),
            )
            .unwrap();
        assert!(!state.provider_routing_binding_is_current(&binding, &registry));
        assert_eq!(
            state.publish_provider_routing(&frame),
            Err(ProviderRoutingError::StaleOrReplay)
        );

        let second_projection = project_registry(&registry).unwrap();
        let second_frame = publication_frame(
            2,
            "provider-pub-2",
            PROVIDER_ROUTING_DOMAIN,
            second_projection.canonical_bytes(),
        );
        let second_ack = state.publish_provider_routing(&second_frame).unwrap();
        let second_ack = state
            .acknowledge_provider_routing(&acknowledgement_frame(&second_ack))
            .unwrap();
        let second_binding = state.bind_provider_routing(&second_ack, &registry).unwrap();
        assert!(state.provider_routing_binding_is_current(&second_binding, &registry));
        assert!(!state.provider_routing_binding_is_current(&binding, &registry));
    }

    #[test]
    fn superseded_and_duplicate_publications_fail_closed() {
        let registry = registry();
        let projection = project_registry(&registry).unwrap();
        let mut state = PublicationState::default();
        let first = publication_frame(
            1,
            "provider-pub-1",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let first_ack = state.publish_provider_routing(&first).unwrap();
        let second = publication_frame(
            2,
            "provider-pub-2",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        let second_ack = state.publish_provider_routing(&second).unwrap();
        assert_eq!(
            state.acknowledge_provider_routing(&acknowledgement_frame(&first_ack)),
            Err(ProviderRoutingError::Superseded)
        );
        let second_ack = state
            .acknowledge_provider_routing(&acknowledgement_frame(&second_ack))
            .unwrap();
        assert_eq!(
            state.acknowledge_provider_routing(&acknowledgement_frame(&second_ack)),
            Err(ProviderRoutingError::NoPublication)
        );
        let duplicate_id = publication_frame(
            3,
            "provider-pub-2",
            PROVIDER_ROUTING_DOMAIN,
            projection.canonical_bytes(),
        );
        assert_eq!(
            state.publish_provider_routing(&duplicate_id),
            Err(ProviderRoutingError::PublicationIdReused)
        );
    }
}

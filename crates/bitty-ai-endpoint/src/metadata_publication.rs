//! In-memory S-1 metadata publication handshake. It adds no transport and does
//! not select providers, access credentials, or perform model I/O. Normal
//! `ai.*` correlators remain optional/tolerant. The acknowledgement capability
//! is retained by the helper-side strict seam, but no strict method is enabled
//! until a current snapshot in this same provider/routing metadata domain is
//! bound to it.

use super::{JVal, parse_params_object};
use crate::provider_binding;
use std::collections::BTreeSet;

pub const MAX_PUBLICATION_FRAME_BYTES: usize = 256 * 1024;
pub const MAX_ACCEPTED_PUBLICATION_IDS: usize = 256;
pub const MAX_PROVIDERS: usize = 32;
pub const MAX_MODELS_PER_PROVIDER: usize = 64;
pub const MAX_CAPABILITIES_PER_MODEL: usize = 8;
pub const MAX_ROUTING_FACTS: usize = 32;
pub const MAX_METADATA_FIELD_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishRequest {
    pub publication_id: String,
    pub snapshot_version: u64,
    pub digest: String,
    /// Exact UTF-8 JSON payload bytes hashed by `digest`.
    pub metadata_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicationAck {
    pub publication_id: String,
    pub snapshot_version: u64,
    pub digest: String,
}

/// Opaque proof that one exact publication acknowledgement was accepted.
///
/// The fields are intentionally private: endpoint strict mode can only be
/// entered with a value issued by [`PublicationState::acknowledge_for_endpoint`]
/// after the state has validated and consumed the matching pending ack. This
/// is an in-memory capability, not a transport credential or a caller-supplied
/// version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcknowledgedPublication {
    publication_id: String,
    snapshot_version: u64,
    digest: String,
}

impl AcknowledgedPublication {
    /// The validated publication identity, safe for bounded diagnostics.
    #[must_use]
    pub fn publication_id(&self) -> &str {
        &self.publication_id
    }

    /// The validated snapshot version carried by the publication.
    #[must_use]
    pub fn snapshot_version(&self) -> u64 {
        self.snapshot_version
    }

    /// The validated lowercase SHA-256 digest of the metadata payload.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationError {
    FrameTooLarge,
    InvalidJson,
    InvalidMetadata,
    CredentialLikeValue,
    InvalidIdentity,
    InvalidDigest,
    PublicationIdReused,
    PublicationLimit,
    StaleOrReplay,
    Superseded,
    NoPublication,
}

#[derive(Debug, Default)]
pub struct PublicationState {
    latest_version: Option<u64>,
    pending: Option<PublicationAck>,
    accepted_publication_ids: BTreeSet<String>,
    provider_latest_version: Option<u64>,
    provider_pending: Option<provider_binding::ProviderRoutingPublicationAck>,
    provider_acknowledged: Option<provider_binding::ProviderRoutingPublicationAck>,
    provider_bound: Option<provider_binding::ProviderRoutingPublicationAck>,
    provider_accepted_publication_ids: BTreeSet<String>,
    provider_epoch: u64,
}

impl PublicationState {
    pub fn publish(&mut self, frame: &[u8]) -> Result<PublicationAck, PublicationError> {
        if frame.len() > MAX_PUBLICATION_FRAME_BYTES {
            return Err(PublicationError::FrameTooLarge);
        }
        let top = parse_object(frame)?;
        reject_duplicate_keys(&top)?;
        exact_keys(
            &top,
            &[
                "publication_id",
                "snapshot_version",
                "digest",
                "metadata_json",
            ],
        )?;
        let id = string_field(&top, "publication_id")?.to_owned();
        let version = integer_field(&top, "snapshot_version")?;
        let digest = string_field(&top, "digest")?.to_owned();
        let payload = string_field(&top, "metadata_json")?;
        if !valid_identity(&id) {
            return Err(PublicationError::InvalidIdentity);
        }
        if self.latest_version.is_some_and(|last| version <= last) {
            return Err(PublicationError::StaleOrReplay);
        }
        if payload.len() > MAX_PUBLICATION_FRAME_BYTES {
            return Err(PublicationError::FrameTooLarge);
        }
        let metadata =
            parse_object(payload.as_bytes()).map_err(|_| PublicationError::InvalidMetadata)?;
        reject_duplicate_keys(&metadata).map_err(|_| PublicationError::InvalidMetadata)?;
        validate_metadata(&metadata)?;
        let expected = sha256_hex(payload.as_bytes());
        if digest != expected {
            return Err(PublicationError::InvalidDigest);
        }
        if self.accepted_publication_ids.contains(&id) {
            return Err(PublicationError::PublicationIdReused);
        }
        if self.accepted_publication_ids.len() >= MAX_ACCEPTED_PUBLICATION_IDS {
            return Err(PublicationError::PublicationLimit);
        }
        let ack = PublicationAck {
            publication_id: id,
            snapshot_version: version,
            digest,
        };
        self.accepted_publication_ids
            .insert(ack.publication_id.clone());
        self.latest_version = Some(version);
        self.pending = Some(ack.clone());
        Ok(ack)
    }

    pub fn acknowledge(&mut self, frame: &[u8]) -> Result<PublicationAck, PublicationError> {
        if frame.len() > MAX_PUBLICATION_FRAME_BYTES {
            return Err(PublicationError::FrameTooLarge);
        }
        let top = parse_object(frame)?;
        reject_duplicate_keys(&top)?;
        exact_keys(&top, &["publication_id", "snapshot_version", "digest"])?;
        let ack = PublicationAck {
            publication_id: string_field(&top, "publication_id")?.to_owned(),
            snapshot_version: integer_field(&top, "snapshot_version")?,
            digest: string_field(&top, "digest")?.to_owned(),
        };
        let Some(pending) = &self.pending else {
            return Err(PublicationError::NoPublication);
        };
        if &ack != pending {
            return Err(PublicationError::Superseded);
        }
        self.pending = None;
        Ok(ack)
    }

    /// Consume the exact pending acknowledgement and issue an opaque endpoint
    /// capability. Strict dispatch remains `Unbound` until a same-domain
    /// current snapshot is supplied by an accepted contract.
    ///
    /// This is deliberately separate from [`Self::acknowledge`], whose public
    /// tuple is retained for protocol tests and callers that only need the
    /// handshake result. The capability cannot be constructed from that tuple.
    pub fn acknowledge_for_endpoint(
        &mut self,
        frame: &[u8],
    ) -> Result<AcknowledgedPublication, PublicationError> {
        let ack = self.acknowledge(frame)?;
        Ok(AcknowledgedPublication {
            publication_id: ack.publication_id,
            snapshot_version: ack.snapshot_version,
            digest: ack.digest,
        })
    }

    /// Publish one domain-tagged provider/routing projection. This lifecycle is
    /// separate from the generic metadata publication above: provider display
    /// names and routing facts are not the `ai.provider_list` projection.
    pub fn publish_provider_routing(
        &mut self,
        frame: &[u8],
    ) -> Result<
        provider_binding::ProviderRoutingPublicationAck,
        provider_binding::ProviderRoutingError,
    > {
        let (ack, _) = provider_binding::parse_provider_routing_publication(frame)?;
        if self
            .provider_latest_version
            .is_some_and(|last| ack.snapshot_version() <= last)
        {
            return Err(provider_binding::ProviderRoutingError::StaleOrReplay);
        }
        if self
            .provider_accepted_publication_ids
            .contains(ack.publication_id())
        {
            return Err(provider_binding::ProviderRoutingError::PublicationIdReused);
        }
        if self.provider_accepted_publication_ids.len() >= MAX_ACCEPTED_PUBLICATION_IDS {
            return Err(provider_binding::ProviderRoutingError::PublicationLimit);
        }
        self.provider_accepted_publication_ids
            .insert(ack.publication_id().to_owned());
        self.provider_latest_version = Some(ack.snapshot_version());
        self.provider_pending = Some(ack.clone());
        self.provider_acknowledged = None;
        self.provider_bound = None;
        self.provider_epoch = self.provider_epoch.saturating_add(1);
        Ok(ack)
    }

    /// Consume the exact pending provider/routing acknowledgement.
    pub fn acknowledge_provider_routing(
        &mut self,
        frame: &[u8],
    ) -> Result<
        provider_binding::ProviderRoutingPublicationAck,
        provider_binding::ProviderRoutingError,
    > {
        let ack = provider_binding::parse_provider_routing_ack(frame)?;
        let Some(pending) = &self.provider_pending else {
            return Err(provider_binding::ProviderRoutingError::NoPublication);
        };
        if &ack != pending {
            return Err(provider_binding::ProviderRoutingError::Superseded);
        }
        self.provider_pending = None;
        self.provider_acknowledged = Some(ack.clone());
        Ok(ack)
    }

    /// Prove equality between an exact acknowledged publication and the
    /// current helper registry. The proof is opaque and carries no mutation
    /// authority. A failed comparison never enables strict dispatch.
    pub fn bind_provider_routing(
        &mut self,
        ack: &provider_binding::ProviderRoutingPublicationAck,
        registry: &bitty_ai_runtime::ProviderRegistry,
    ) -> Result<provider_binding::ProviderRoutingBinding, provider_binding::ProviderRoutingError>
    {
        let Some(acknowledged) = &self.provider_acknowledged else {
            return Err(provider_binding::ProviderRoutingError::NoAcknowledgement);
        };
        if acknowledged != ack {
            return Err(provider_binding::ProviderRoutingError::Superseded);
        }
        let projection = provider_binding::project_registry(registry)?;
        if projection.digest() != ack.digest() {
            self.provider_acknowledged = None;
            return Err(provider_binding::ProviderRoutingError::ProjectionMismatch);
        }
        self.provider_bound = Some(ack.clone());
        self.provider_acknowledged = None;
        Ok(provider_binding::ProviderRoutingBinding::new(
            ack,
            projection,
            self.provider_epoch,
            registry.revision(),
        ))
    }

    /// Check a proof against the publication lifecycle and current registry.
    #[must_use]
    pub fn provider_routing_binding_is_current(
        &self,
        binding: &provider_binding::ProviderRoutingBinding,
        registry: &bitty_ai_runtime::ProviderRegistry,
    ) -> bool {
        let Some(ack) = &self.provider_bound else {
            return false;
        };
        ack.publication_id() == binding.publication_id()
            && ack.domain() == binding.domain()
            && ack.snapshot_version() == binding.snapshot_version()
            && ack.digest() == binding.digest()
            && binding.publication_epoch() == self.provider_epoch
            && binding.registry_revision() == registry.revision()
            && provider_binding::project_registry(registry)
                .is_ok_and(|projection| projection.digest() == binding.digest())
    }
}

pub(crate) fn parse_object(bytes: &[u8]) -> Result<JVal, PublicationError> {
    parse_params_object(bytes).map_err(|_| PublicationError::InvalidJson)
}

pub(crate) fn reject_duplicate_keys(value: &JVal) -> Result<(), PublicationError> {
    match value {
        JVal::Obj(fields) => {
            let mut keys = BTreeSet::new();
            for (key, value) in fields {
                if !keys.insert(key) {
                    return Err(PublicationError::InvalidJson);
                }
                reject_duplicate_keys(value)?;
            }
        }
        JVal::Arr(items) => {
            for item in items {
                reject_duplicate_keys(item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn exact_keys(value: &JVal, expected: &[&str]) -> Result<(), PublicationError> {
    let fields = value.as_obj().ok_or(PublicationError::InvalidJson)?;
    if fields.len() != expected.len()
        || fields
            .iter()
            .any(|(key, _)| !expected.contains(&key.as_str()))
    {
        return Err(PublicationError::InvalidJson);
    }
    Ok(())
}

fn string_field<'a>(value: &'a JVal, name: &str) -> Result<&'a str, PublicationError> {
    value
        .field(name)
        .and_then(JVal::as_str)
        .ok_or(PublicationError::InvalidJson)
}

fn integer_field(value: &JVal, name: &str) -> Result<u64, PublicationError> {
    value
        .field(name)
        .and_then(JVal::as_u64)
        .ok_or(PublicationError::InvalidJson)
}

pub(crate) fn valid_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_METADATA_FIELD_BYTES
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
}

fn check_identity(value: &str) -> Result<(), PublicationError> {
    if !valid_identity(value) {
        return Err(PublicationError::InvalidMetadata);
    }
    if credential_like_identity(value) {
        return Err(PublicationError::CredentialLikeValue);
    }
    Ok(())
}

fn validate_metadata(top: &JVal) -> Result<(), PublicationError> {
    exact_keys(top, &["providers", "routing_facts"])
        .map_err(|_| PublicationError::InvalidMetadata)?;
    let providers = top
        .field("providers")
        .and_then(JVal::as_arr)
        .ok_or(PublicationError::InvalidMetadata)?;
    let facts = top
        .field("routing_facts")
        .and_then(JVal::as_arr)
        .ok_or(PublicationError::InvalidMetadata)?;
    if providers.len() > MAX_PROVIDERS || facts.len() > MAX_ROUTING_FACTS {
        return Err(PublicationError::InvalidMetadata);
    }
    let mut ids = BTreeSet::new();
    for provider in providers {
        exact_keys(provider, &["id", "name", "models"])
            .map_err(|_| PublicationError::InvalidMetadata)?;
        let id = string_field(provider, "id").map_err(|_| PublicationError::InvalidMetadata)?;
        let name = string_field(provider, "name").map_err(|_| PublicationError::InvalidMetadata)?;
        check_identity(id)?;
        if !ids.insert(id.to_owned()) {
            return Err(PublicationError::InvalidMetadata);
        }
        check_text(name)?;
        let models = provider
            .field("models")
            .and_then(JVal::as_arr)
            .ok_or(PublicationError::InvalidMetadata)?;
        if models.len() > MAX_MODELS_PER_PROVIDER {
            return Err(PublicationError::InvalidMetadata);
        }
        let mut names = BTreeSet::new();
        for model in models {
            exact_keys(model, &["name", "capabilities"])
                .map_err(|_| PublicationError::InvalidMetadata)?;
            let name =
                string_field(model, "name").map_err(|_| PublicationError::InvalidMetadata)?;
            check_text(name)?;
            if !names.insert(name.to_owned()) {
                return Err(PublicationError::InvalidMetadata);
            }
            let caps = model
                .field("capabilities")
                .and_then(JVal::as_arr)
                .ok_or(PublicationError::InvalidMetadata)?;
            if caps.len() > MAX_CAPABILITIES_PER_MODEL {
                return Err(PublicationError::InvalidMetadata);
            }
            let mut unique = BTreeSet::new();
            for cap in caps {
                let cap = cap.as_str().ok_or(PublicationError::InvalidMetadata)?;
                check_text(cap)?;
                if !unique.insert(cap.to_owned()) {
                    return Err(PublicationError::InvalidMetadata);
                }
            }
        }
    }
    for fact in facts {
        check_text(fact.as_str().ok_or(PublicationError::InvalidMetadata)?)?;
    }
    Ok(())
}

fn check_text(value: &str) -> Result<(), PublicationError> {
    if credential_like_text(value) {
        return Err(PublicationError::CredentialLikeValue);
    }
    if value.is_empty()
        || value.len() > MAX_METADATA_FIELD_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(PublicationError::InvalidMetadata);
    }
    Ok(())
}

const CREDENTIAL_MARKERS: [&str; 7] = [
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "credential",
];

fn credential_like_text(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    CREDENTIAL_MARKERS
        .iter()
        .any(|marker| value.contains(marker))
}

fn credential_like_identity(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    CREDENTIAL_MARKERS.iter().any(|marker| {
        value.match_indices(marker).any(|(start, _)| {
            let end = start + marker.len();
            (start == 0 || !value.as_bytes()[start - 1].is_ascii_alphanumeric())
                && (end == value.len() || !value.as_bytes()[end].is_ascii_alphanumeric())
        })
    })
}

// SHA-256 per FIPS 180-4. Fixed-size state and bounded caller input only.
pub(crate) fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = input.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for block in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10));
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut x) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let t1 = x
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            x = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, x]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata() -> &'static str {
        r#"{"providers":[{"id":"local","name":"Local","models":[{"name":"model-a","capabilities":["text"]}]}],"routing_facts":["local-only"]}"#
    }
    fn metadata_for_provider(id: &str) -> String {
        format!(
            r#"{{"providers":[{{"id":"{id}","name":"Provider","models":[]}}],"routing_facts":[]}}"#
        )
    }
    fn frame(version: u64, id: &str, payload: &str) -> Vec<u8> {
        format!(r#"{{"publication_id":"{id}","snapshot_version":{version},"digest":"{}","metadata_json":{}}}"#,sha256_hex(payload.as_bytes()),json_quote(payload)).into_bytes()
    }
    fn json_quote(s: &str) -> String {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                _ => out.push(c),
            }
        }
        out.push('"');
        out
    }
    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
    #[test]
    fn exact_digest_identity_and_single_ack() {
        let mut s = PublicationState::default();
        let ack = s.publish(&frame(1, "pub-1", metadata())).unwrap();
        let wrong_digest = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            ack.publication_id,
            ack.snapshot_version,
            "0".repeat(64)
        );
        assert_eq!(
            s.acknowledge(wrong_digest.as_bytes()),
            Err(PublicationError::Superseded)
        );
        let bytes = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            ack.publication_id, ack.snapshot_version, ack.digest
        );
        assert_eq!(s.acknowledge(bytes.as_bytes()), Ok(ack.clone()));
        assert_eq!(
            s.acknowledge(bytes.as_bytes()),
            Err(PublicationError::NoPublication)
        );
    }

    #[test]
    fn endpoint_capability_requires_exact_single_ack() {
        let mut state = PublicationState::default();
        let publication = frame(1, "pub-1", metadata());
        let ack = state.publish(&publication).unwrap();
        let ack_frame = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            ack.publication_id, ack.snapshot_version, ack.digest
        );
        let capability = state
            .acknowledge_for_endpoint(ack_frame.as_bytes())
            .expect("exact acknowledgement issues endpoint capability");
        assert_eq!(capability.publication_id(), "pub-1");
        assert_eq!(capability.snapshot_version(), 1);
        assert_eq!(capability.digest(), ack.digest);
        assert_eq!(
            state.acknowledge_for_endpoint(ack_frame.as_bytes()),
            Err(PublicationError::NoPublication)
        );
        // A replayed publication is rejected by the same bounded state even
        // after the endpoint capability has been issued.
        assert_eq!(
            state.publish(&publication),
            Err(PublicationError::StaleOrReplay)
        );
    }

    #[test]
    fn superseded_ack_cannot_issue_endpoint_capability() {
        let mut state = PublicationState::default();
        let first = state.publish(&frame(1, "pub-1", metadata())).unwrap();
        let second = state.publish(&frame(2, "pub-2", metadata())).unwrap();
        let first_ack = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            first.publication_id, first.snapshot_version, first.digest
        );

        assert_eq!(
            state.acknowledge_for_endpoint(first_ack.as_bytes()),
            Err(PublicationError::Superseded)
        );
        let second_ack = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            second.publication_id, second.snapshot_version, second.digest
        );
        assert!(
            state
                .acknowledge_for_endpoint(second_ack.as_bytes())
                .is_ok()
        );
    }
    #[test]
    fn replay_superseded_and_mismatch_refuse() {
        let mut s = PublicationState::default();
        let a = s.publish(&frame(1, "a", metadata())).unwrap();
        assert_eq!(
            s.publish(&frame(1, "x", metadata())),
            Err(PublicationError::StaleOrReplay)
        );
        let b = s.publish(&frame(2, "b", metadata())).unwrap();
        let old = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            a.publication_id, a.snapshot_version, a.digest
        );
        assert_eq!(
            s.acknowledge(old.as_bytes()),
            Err(PublicationError::Superseded)
        );
        assert_eq!(
            s.acknowledge(
                format!(
                    r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
                    b.publication_id, b.snapshot_version, b.digest
                )
                .as_bytes()
            ),
            Ok(b)
        );
    }

    #[test]
    fn publication_id_cannot_be_reused_for_a_new_version_or_payload() {
        let mut state = PublicationState::default();
        let first_payload = metadata();
        let second_payload = r#"{"providers":[{"id":"local","name":"Local","models":[{"name":"model-b","capabilities":["text"]}]}],"routing_facts":["local-only"]}"#;
        let first_ack = state
            .publish(&frame(1, "pub-1", first_payload))
            .expect("first publication is accepted");

        assert_ne!(first_payload, second_payload);
        assert_eq!(
            state.publish(&frame(2, "pub-1", second_payload)),
            Err(PublicationError::PublicationIdReused)
        );

        let first_ack_frame = format!(
            r#"{{"publication_id":"{}","snapshot_version":{},"digest":"{}"}}"#,
            first_ack.publication_id, first_ack.snapshot_version, first_ack.digest
        );
        assert_eq!(state.acknowledge(first_ack_frame.as_bytes()), Ok(first_ack));
        assert!(state.publish(&frame(2, "pub-2", second_payload)).is_ok());
    }

    #[test]
    fn accepted_publication_identity_history_is_bounded() {
        let mut state = PublicationState::default();
        for version in 1..=MAX_ACCEPTED_PUBLICATION_IDS {
            state
                .publish(&frame(
                    version as u64,
                    &format!("pub-{version}"),
                    metadata(),
                ))
                .expect("publication is within the identity history bound");
        }

        let pending_before = state.pending.clone();
        assert_eq!(
            state.publish(&frame(
                MAX_ACCEPTED_PUBLICATION_IDS as u64 + 1,
                "pub-over-limit",
                metadata()
            )),
            Err(PublicationError::PublicationLimit)
        );
        assert_eq!(
            state.latest_version,
            Some(MAX_ACCEPTED_PUBLICATION_IDS as u64)
        );
        assert_eq!(state.pending, pending_before);
    }
    #[test]
    fn malformed_duplicate_credential_and_oversized_refuse() {
        let mut s = PublicationState::default();
        assert_eq!(
            s.publish(br#"{"publication_id":"a","publication_id":"b"}"#),
            Err(PublicationError::InvalidJson)
        );
        let cred = r#"{"providers":[{"id":"p","name":"password","models":[]}],"routing_facts":[]}"#;
        assert_eq!(
            s.publish(&frame(1, "a", cred)),
            Err(PublicationError::CredentialLikeValue)
        );
        assert_eq!(
            s.publish(&vec![b' '; MAX_PUBLICATION_FRAME_BYTES + 1]),
            Err(PublicationError::FrameTooLarge)
        );
        let duplicate = r#"{"providers":[],"providers":[],"routing_facts":[]}"#;
        assert_eq!(
            s.publish(&frame(1, "a", duplicate)),
            Err(PublicationError::InvalidMetadata)
        );
    }

    #[test]
    fn malformed_snapshot_numbers_do_not_mutate_publication_state() {
        let mut state = PublicationState::default();
        state
            .publish(&frame(1, "pub-1", metadata()))
            .expect("baseline publication is accepted");
        let latest_before = state.latest_version;
        let pending_before = state.pending.clone();
        let ids_before = state.accepted_publication_ids.clone();

        for number in ["01", "-01", "1.", "1e", "1e+", "1e-"] {
            let valid = String::from_utf8(frame(2, "pub-malformed", metadata())).unwrap();
            let malformed = valid.replace(
                "\"snapshot_version\":2",
                &format!("\"snapshot_version\":{number}"),
            );
            assert_eq!(
                state.publish(malformed.as_bytes()),
                Err(PublicationError::InvalidJson),
                "{number}"
            );
            assert_eq!(state.latest_version, latest_before, "{number}");
            assert_eq!(state.pending, pending_before, "{number}");
            assert_eq!(state.accepted_publication_ids, ids_before, "{number}");
        }
    }

    #[test]
    fn provider_ids_reject_credential_like_values_but_allow_bounded_ids() {
        let mut state = PublicationState::default();
        for (version, id) in [(1, "api_key"), (2, "secret.token")] {
            let payload = metadata_for_provider(id);
            assert_eq!(
                state.publish(&frame(version, &format!("pub-{version}"), &payload)),
                Err(PublicationError::CredentialLikeValue)
            );
        }

        let payload = metadata_for_provider("local.deterministic");
        assert_eq!(
            state
                .publish(&frame(3, "pub-3", &payload))
                .expect("bounded provider identifier is accepted")
                .publication_id,
            "pub-3"
        );
    }
}

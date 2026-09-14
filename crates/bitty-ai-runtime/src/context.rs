//! Bounded context assembly (L0 + L1 only).
//!
//! Mirrors the `CP-1`..`CP-7` shape: validated Stable Ids, a closed v1
//! provider set, a token-first request contract resolved against a byte
//! budget, and deterministic assembly for a given provider snapshot.
//!
//! Implemented levels (from `context-management.md`):
//!
//! - **L0 structured output**: records carry a bounded `summary` plus a body;
//!   large bodies are externalized to [`ArtifactStore`] references instead of
//!   flat text dumps.
//! - **L1 lossless pruning**: superseded records are dropped, exact-duplicate
//!   records collapse to the newest, and large inline bodies move to artifact
//!   references.
//!
//! Explicitly out of scope: L2+ selective compression/compaction, provider
//! summarization, retrieval/ranking, and durable retention. Missing, expired,
//! or deleted content resolves to typed [`ContextError::ArtifactUnavailable`],
//! never to a silent substitute.

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::provider::DEFAULT_CONTEXT_BUDGET_BYTES;

/// Maximum Stable Id length in bytes (`CP-1`).
pub const MAX_STABLE_ID_LEN: usize = 64;
/// Maximum structured summary length in bytes (L0).
pub const MAX_SUMMARY_BYTES: usize = 4 * 1024;
/// Maximum inline record body length in bytes (`TB-6` result-cap parity).
pub const MAX_RECORD_BODY_BYTES: usize = 16 * 1024;
/// Inline bodies larger than this move to an artifact reference (L1
/// externalize).
pub const EXTERNALIZE_THRESHOLD_BYTES: usize = 4 * 1024;
/// Maximum records accepted per assembly call.
pub const MAX_CONTEXT_RECORDS: usize = 128;
/// Maximum artifacts retained per store.
pub const MAX_ARTIFACTS: usize = 64;
/// Maximum bytes per artifact.
pub const MAX_ARTIFACT_BYTES: usize = 64 * 1024;
/// Maximum total artifact bytes per store.
pub const MAX_ARTIFACT_STORE_BYTES: usize = 256 * 1024;
/// Conservative bytes-per-token estimate for budget math. This is a documented
/// skeleton heuristic for enforcing a token-first ceiling, not billed usage.
pub const BYTES_PER_TOKEN_ESTIMATE: usize = 4;

/// Closed v1 ContextProvider set (`CP` provider table). New providers require
/// a reviewed spec amendment; unknown names fail closed.
pub const KNOWN_PROVIDERS: [&str; 5] = ["workspace", "project", "git", "diagnostics", "terminal"];

/// Context assembly errors. All variants fail closed with no partial
/// assembly returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// Stable Id violates the `CP-1` shape.
    InvalidStableId {
        /// Rejected id.
        id: String,
    },
    /// Provider name is outside the closed v1 set.
    UnknownProvider {
        /// Rejected provider name.
        name: String,
    },
    /// Structured summary exceeds [`MAX_SUMMARY_BYTES`].
    SummaryTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Inline body exceeds [`MAX_RECORD_BODY_BYTES`].
    BodyTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Record count exceeds [`MAX_CONTEXT_RECORDS`].
    TooManyRecords {
        /// Bound.
        limit: usize,
    },
    /// Single artifact exceeds [`MAX_ARTIFACT_BYTES`].
    ArtifactTooLarge {
        /// Bound in bytes.
        limit: usize,
        /// Observed bytes.
        actual: usize,
    },
    /// Store cannot retain another artifact (`MAX_ARTIFACTS` or
    /// [`MAX_ARTIFACT_STORE_BYTES`]).
    ArtifactStoreFull {
        /// Which bound was hit.
        reason: String,
    },
    /// Reference names no retained artifact. Missing, expired, or deleted
    /// content is explicitly unavailable, never silently substituted.
    ArtifactUnavailable {
        /// Dangling reference.
        reference: String,
    },
    /// Even the smallest record does not fit the resolved budget.
    BudgetExceeded {
        /// Resolved budget in bytes.
        limit: usize,
        /// Smallest record footprint in bytes.
        actual: usize,
    },
}

impl Display for ContextError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidStableId { id } => write!(f, "invalid stable id: {id}"),
            Self::UnknownProvider { name } => write!(f, "unknown context provider: {name}"),
            Self::SummaryTooLarge { limit, actual } => write!(
                f,
                "context summary of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::BodyTooLarge { limit, actual } => write!(
                f,
                "context body of {actual} bytes exceeds {limit} byte limit"
            ),
            Self::TooManyRecords { limit } => {
                write!(f, "context record count exceeds limit {limit}")
            }
            Self::ArtifactTooLarge { limit, actual } => {
                write!(f, "artifact of {actual} bytes exceeds {limit} byte limit")
            }
            Self::ArtifactStoreFull { reason } => {
                write!(f, "artifact store full: {reason}")
            }
            Self::ArtifactUnavailable { reference } => {
                write!(f, "artifact unavailable: {reference}")
            }
            Self::BudgetExceeded { limit, actual } => write!(
                f,
                "context budget exceeded: smallest record {actual} bytes over {limit} byte budget"
            ),
        }
    }
}

impl std::error::Error for ContextError {}

/// Validate a Stable Id (`CP-1`): non-empty, at most 64 bytes,
/// `^[a-z0-9_-]+$`.
///
/// # Errors
///
/// Returns [`ContextError::InvalidStableId`] when the shape is violated.
pub fn validate_stable_id(id: &str) -> Result<(), ContextError> {
    let valid = !id.is_empty()
        && id.len() <= MAX_STABLE_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if valid {
        Ok(())
    } else {
        Err(ContextError::InvalidStableId { id: id.to_owned() })
    }
}

/// A validated Stable Id: one level of the
/// `Instance -> Window -> Workspace -> View -> Terminal` hierarchy (`CP-1`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StableId(String);

impl StableId {
    /// Construct and validate.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::InvalidStableId`] for a malformed id.
    pub fn new(id: impl Into<String>) -> Result<Self, ContextError> {
        let id = id.into();
        validate_stable_id(&id)?;
        Ok(Self(id))
    }

    /// Borrow the id text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Priority used for truncation order (`CP-5`): lower priority truncates
/// first. Callers assign terminal/diagnostics extremes accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextPriority {
    /// Truncate first (for example terminal scrape, diagnostics).
    Low,
    /// Default.
    Normal,
    /// Truncate late (for example project manifest, git stat).
    High,
    /// Truncate last.
    Critical,
}

/// Requested detail depth (`CP-5`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailLevel {
    /// Summaries and references only.
    Summary,
    /// Summaries plus bounded inline bodies.
    Standard,
    /// Inline bodies up to the record bound (still bounded, still budgeted).
    Full,
}

/// Bounded context request (`CP-5` shape, skeleton subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextRequest {
    /// Token-first ceiling; enforced via [`ContextRequest::estimate_tokens`].
    pub max_tokens: Option<u32>,
    /// Byte ceiling; defaults to the candidate 32 KiB profile.
    pub max_bytes: Option<u64>,
    /// Caller priority hint (kept for the host contract; assembly truncates
    /// per-record priority, not this field).
    pub priority: ContextPriority,
    /// Requested detail depth.
    pub detail: DetailLevel,
}

impl ContextRequest {
    /// Conservative token estimate for `bytes` (heuristic, not billed usage).
    #[must_use]
    pub fn estimate_tokens(bytes: usize) -> u64 {
        bytes.div_ceil(BYTES_PER_TOKEN_ESTIMATE) as u64
    }

    /// Resolve the effective byte budget: the explicit byte cap (when set),
    /// further limited by the token cap converted at the estimate rate, else
    /// the candidate default.
    #[must_use]
    pub fn effective_budget_bytes(&self) -> usize {
        let mut budget = self
            .max_bytes
            .and_then(|bytes| usize::try_from(bytes).ok())
            .unwrap_or(DEFAULT_CONTEXT_BUDGET_BYTES);
        if let Some(tokens) = self.max_tokens {
            budget = budget.min(tokens as usize * BYTES_PER_TOKEN_ESTIMATE);
        }
        budget
    }
}

/// Opaque reference to retained bytes (`artifact://<n>`). The string form is
/// an index over retained records, never a filesystem or network capability.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactRef(String);

impl ArtifactRef {
    /// Borrow the reference text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ArtifactRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

/// Bounded in-memory artifact store (L1 externalize target). Retention is
/// session-scoped and consent-bounded by the host; the store itself only
/// enforces byte/count caps and typed absence.
#[derive(Debug, Default)]
pub struct ArtifactStore {
    entries: Vec<(String, Vec<u8>)>,
    total_bytes: usize,
    next_id: u64,
}

impl ArtifactStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retained artifact count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store holds no artifact.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Retained bytes across all artifacts.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Retain `bytes` and return its reference.
    ///
    /// # Errors
    ///
    /// Fails closed with [`ContextError::ArtifactTooLarge`] or
    /// [`ContextError::ArtifactStoreFull`]; the store is unchanged on error.
    pub fn store(&mut self, bytes: Vec<u8>) -> Result<ArtifactRef, ContextError> {
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(ContextError::ArtifactTooLarge {
                limit: MAX_ARTIFACT_BYTES,
                actual: bytes.len(),
            });
        }
        if self.entries.len() >= MAX_ARTIFACTS {
            return Err(ContextError::ArtifactStoreFull {
                reason: format!("at most {MAX_ARTIFACTS} artifacts"),
            });
        }
        if self.total_bytes + bytes.len() > MAX_ARTIFACT_STORE_BYTES {
            return Err(ContextError::ArtifactStoreFull {
                reason: format!("at most {MAX_ARTIFACT_STORE_BYTES} retained bytes"),
            });
        }
        self.next_id += 1;
        let reference = format!("artifact://{}", self.next_id);
        self.total_bytes += bytes.len();
        self.entries.push((reference.clone(), bytes));
        Ok(ArtifactRef(reference))
    }

    /// Resolve a reference to retained bytes (`CP-6` drill-down primitive).
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::ArtifactUnavailable`] for dangling references.
    pub fn resolve(&self, reference: &ArtifactRef) -> Result<&[u8], ContextError> {
        self.entries
            .iter()
            .find(|(name, _)| name == &reference.0)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| ContextError::ArtifactUnavailable {
                reference: reference.0.clone(),
            })
    }
}

/// Record payload: inline bounded bytes or an artifact reference (L0/L1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordBody {
    /// Bounded inline bytes.
    Inline(Vec<u8>),
    /// Reference to retained bytes.
    Artifact(ArtifactRef),
}

/// One bounded context record (`CP-3` attribution shape plus L0 summary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextRecord {
    /// Record id, unique within the turn (used by `supersedes` and omission
    /// reports).
    pub id: String,
    /// Source provider; must name the closed v1 set.
    pub provider: String,
    /// Owning Stable Id path head (`CP-3`).
    pub owner: StableId,
    /// Source generation for invalidation (`CP-3`).
    pub generation: u64,
    /// Caller-supplied collection timestamp (`CP-3`, `CP-7`).
    pub collected_at_ms: u64,
    /// Truncation priority: lower drops first (`CP-5`).
    pub priority: ContextPriority,
    /// L0 structured summary (bounded, always inline).
    pub summary: String,
    /// Record payload.
    pub body: RecordBody,
    /// L1 supersede link: this record replaces the named record id.
    pub supersedes: Option<String>,
    /// Terminal/tool content is untrusted observation data (`CP-10`).
    pub is_untrusted_surface: bool,
}

impl ContextRecord {
    /// Validate one record against the closed provider set and the L0 bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError`] for unknown providers, over-bound summaries,
    /// or over-bound inline bodies.
    pub fn validate(&self) -> Result<(), ContextError> {
        if !KNOWN_PROVIDERS.contains(&self.provider.as_str()) {
            return Err(ContextError::UnknownProvider {
                name: self.provider.clone(),
            });
        }
        if self.summary.len() > MAX_SUMMARY_BYTES {
            return Err(ContextError::SummaryTooLarge {
                limit: MAX_SUMMARY_BYTES,
                actual: self.summary.len(),
            });
        }
        if let RecordBody::Inline(bytes) = &self.body {
            if bytes.len() > MAX_RECORD_BODY_BYTES {
                return Err(ContextError::BodyTooLarge {
                    limit: MAX_RECORD_BODY_BYTES,
                    actual: bytes.len(),
                });
            }
        }
        Ok(())
    }

    /// Budget footprint: summary bytes plus inline bytes (or reference text
    /// for externalized bodies).
    #[must_use]
    pub fn footprint_bytes(&self) -> usize {
        let body = match &self.body {
            RecordBody::Inline(bytes) => bytes.len(),
            RecordBody::Artifact(reference) => reference.as_str().len(),
        };
        self.summary.len() + body
    }
}

/// Assembled payload: inline bytes or a reference for drill-down (`CP-6`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssembledContent {
    /// Bounded inline bytes.
    Inline(Vec<u8>),
    /// Drill down with [`ArtifactStore::resolve`] under the same consent and
    /// budget.
    Reference(ArtifactRef),
}

impl AssembledContent {
    /// Budget footprint in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Inline(bytes) => bytes.len(),
            Self::Reference(reference) => reference.as_str().len(),
        }
    }

    /// Whether the content is inline bytes.
    #[must_use]
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }
}

/// One record selected for the provider turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledRecord {
    /// Source record id.
    pub id: String,
    /// Source provider.
    pub provider: String,
    /// L0 summary (always inline).
    pub summary: String,
    /// Payload.
    pub content: AssembledContent,
    /// Whether the record arrived via the untrusted surface.
    pub is_untrusted_surface: bool,
}

/// Deterministic assembly result for one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledContext {
    /// Selected records in caller order.
    pub records: Vec<AssembledRecord>,
    /// Ids of selected records (provider `context_refs` input).
    pub context_refs: Vec<String>,
    /// Ids dropped by budget truncation (whole-record, counted).
    pub omitted_ids: Vec<String>,
    /// Ids dropped by L1 supersede/dedupe.
    pub pruned_ids: Vec<String>,
    /// Bodies externalized to the artifact store during this call.
    pub externalized: usize,
    /// Bytes removed by truncation (omitted records).
    pub truncated_bytes: u64,
    /// Token estimate of truncated bytes (heuristic, not billed usage).
    pub truncated_tokens_estimate: u64,
    /// Providers that lost bytes, first-seen order, deduplicated.
    pub truncated_providers: Vec<String>,
    /// Resolved budget this assembly enforced.
    pub budget_bytes: usize,
}

/// Assemble `records` under `request` (L0 + L1, deterministic).
///
/// Policy, in order: validate all records (fail closed, no partial output);
/// drop L1-superseded records; collapse exact `(provider, owner, summary)`
/// duplicates to the newest; externalize inline bodies over
/// [`EXTERNALIZE_THRESHOLD_BYTES`]; then greedily include records
/// highest-priority-first while the budget holds, omitting the rest with
/// counted truncation. Output order follows the caller order.
///
/// # Errors
///
/// Returns [`ContextError`] for invalid records, record-count overflow,
/// artifact failures, or when even the smallest record exceeds the budget.
pub fn assemble(
    records: &[ContextRecord],
    store: &mut ArtifactStore,
    request: &ContextRequest,
) -> Result<AssembledContext, ContextError> {
    if records.len() > MAX_CONTEXT_RECORDS {
        return Err(ContextError::TooManyRecords {
            limit: MAX_CONTEXT_RECORDS,
        });
    }
    for record in records {
        record.validate()?;
    }

    let mut pruned_ids: Vec<String> = Vec::new();
    let superseded: Vec<&str> = records
        .iter()
        .filter_map(|record| record.supersedes.as_deref())
        .collect();
    // L1 dedupe: exact (provider, owner, summary) duplicates collapse to the
    // newest collected_at; ties keep the first record.
    let mut deduped: Vec<(usize, ContextRecord)> = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if superseded.contains(&record.id.as_str()) {
            pruned_ids.push(record.id.clone());
            continue;
        }
        if let Some(slot) = deduped.iter_mut().find(|(_, kept)| {
            kept.provider == record.provider
                && kept.owner == record.owner
                && kept.summary == record.summary
        }) {
            if record.collected_at_ms > slot.1.collected_at_ms {
                pruned_ids.push(slot.1.id.clone());
                *slot = (index, record.clone());
            } else {
                pruned_ids.push(record.id.clone());
            }
        } else {
            deduped.push((index, record.clone()));
        }
    }

    // L1 externalize: large inline bodies become artifact references.
    let mut externalized = 0usize;
    let mut staged: Vec<(usize, ContextRecord)> = Vec::with_capacity(deduped.len());
    for (index, mut record) in deduped {
        let externalize = matches!(&record.body, RecordBody::Inline(bytes) if bytes.len() > EXTERNALIZE_THRESHOLD_BYTES);
        if externalize {
            // Guarded by the check above: the body is Inline here.
            let bytes = match &mut record.body {
                RecordBody::Inline(bytes) => std::mem::take(bytes),
                RecordBody::Artifact(_) => Vec::new(),
            };
            let reference = store.store(bytes)?;
            record.body = RecordBody::Artifact(reference);
            externalized += 1;
        }
        staged.push((index, record));
    }

    // Greedy include, highest priority first; ties keep caller order.
    let budget = request.effective_budget_bytes();
    let mut order: Vec<usize> = (0..staged.len()).collect();
    order.sort_by_key(|&position| {
        let (_, record) = &staged[position];
        (std::cmp::Reverse(record.priority as u8), position)
    });
    let mut included = vec![false; staged.len()];
    let mut used = 0usize;
    let mut omitted_ids: Vec<String> = Vec::new();
    let mut truncated_bytes: u64 = 0;
    let mut truncated_providers: Vec<String> = Vec::new();
    for position in order {
        let (_, record) = &staged[position];
        let need = record.footprint_bytes();
        if used + need <= budget {
            included[position] = true;
            used += need;
        } else {
            omitted_ids.push(record.id.clone());
            truncated_bytes += need as u64;
            if !truncated_providers.contains(&record.provider) {
                truncated_providers.push(record.provider.clone());
            }
        }
    }
    if !staged.is_empty() && !included.iter().any(|kept| *kept) {
        let smallest = staged
            .iter()
            .map(|(_, record)| record.footprint_bytes())
            .min()
            .unwrap_or(0);
        return Err(ContextError::BudgetExceeded {
            limit: budget,
            actual: smallest,
        });
    }

    let mut selected: Vec<(usize, ContextRecord)> = staged
        .into_iter()
        .enumerate()
        .filter(|(position, _)| included[*position])
        .map(|(_, staged_record)| staged_record)
        .collect();
    selected.sort_by_key(|(index, _)| *index);

    let mut assembled_records = Vec::with_capacity(selected.len());
    let mut context_refs = Vec::with_capacity(selected.len());
    for (_, record) in selected {
        context_refs.push(record.id.clone());
        let content = match record.body {
            RecordBody::Inline(bytes) => AssembledContent::Inline(bytes),
            RecordBody::Artifact(reference) => AssembledContent::Reference(reference),
        };
        assembled_records.push(AssembledRecord {
            id: record.id,
            provider: record.provider,
            summary: record.summary,
            content,
            is_untrusted_surface: record.is_untrusted_surface,
        });
    }

    Ok(AssembledContext {
        records: assembled_records,
        context_refs,
        omitted_ids,
        pruned_ids,
        externalized,
        truncated_bytes,
        truncated_tokens_estimate: ContextRequest::estimate_tokens(truncated_bytes as usize),
        truncated_providers,
        budget_bytes: budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, provider: &str, summary: &str, body_len: usize) -> ContextRecord {
        ContextRecord {
            id: id.to_owned(),
            provider: provider.to_owned(),
            owner: StableId::new("term-1").expect("valid stable id"),
            generation: 1,
            collected_at_ms: 100,
            priority: ContextPriority::Normal,
            summary: summary.to_owned(),
            body: RecordBody::Inline(vec![b'x'; body_len]),
            supersedes: None,
            is_untrusted_surface: false,
        }
    }

    fn budget(bytes: usize) -> ContextRequest {
        ContextRequest {
            max_tokens: None,
            max_bytes: Some(bytes as u64),
            priority: ContextPriority::Normal,
            detail: DetailLevel::Standard,
        }
    }

    #[test]
    fn stable_id_shape() {
        assert!(validate_stable_id("term-1").is_ok());
        assert!(validate_stable_id("").is_err());
        assert!(validate_stable_id("Term").is_err());
        assert!(validate_stable_id("a b").is_err());
        assert!(validate_stable_id("a".repeat(65).as_str()).is_err());
    }

    #[test]
    fn unknown_provider_fails_closed() {
        let mut store = ArtifactStore::new();
        let err = assemble(&[record("r1", "nope", "s", 10)], &mut store, &budget(4096))
            .expect_err("unknown provider must fail");
        assert!(matches!(err, ContextError::UnknownProvider { .. }));
    }

    #[test]
    fn token_cap_limits_byte_budget() {
        let request = ContextRequest {
            max_tokens: Some(100),
            max_bytes: Some(1_000_000),
            priority: ContextPriority::Normal,
            detail: DetailLevel::Standard,
        };
        assert_eq!(request.effective_budget_bytes(), 400);
        assert_eq!(ContextRequest::estimate_tokens(9), 3);
    }

    #[test]
    fn supersede_and_duplicate_collapse_to_newest() {
        let mut old = record("read-v1", "workspace", "outline of foo", 10);
        old.collected_at_ms = 50;
        let mut new = record("read-v2", "workspace", "outline of foo", 12);
        new.collected_at_ms = 150;
        new.supersedes = Some("read-v1".to_owned());
        let mut store = ArtifactStore::new();
        let assembled = assemble(&[old, new], &mut store, &budget(8192)).expect("assemble");
        assert_eq!(assembled.records.len(), 1);
        assert_eq!(assembled.records[0].id, "read-v2");
        assert!(assembled.pruned_ids.contains(&"read-v1".to_owned()));
    }

    #[test]
    fn large_bodies_externalize_to_artifacts() {
        let mut store = ArtifactStore::new();
        let assembled = assemble(
            &[record("big", "terminal", "zone dump", 8_192)],
            &mut store,
            &budget(32_768),
        )
        .expect("assemble");
        assert_eq!(assembled.externalized, 1);
        assert!(matches!(
            assembled.records[0].content,
            AssembledContent::Reference(_)
        ));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn truncation_prefers_low_priority_and_counts() {
        let mut low = record("low", "diagnostics", "lint notes", 100);
        low.priority = ContextPriority::Low;
        let mut high = record("high", "project", "manifest", 100);
        high.priority = ContextPriority::High;
        let mut store = ArtifactStore::new();
        // Budget fits exactly one record footprint (summary 10 + body 100 = 110).
        let assembled = assemble(&[low, high], &mut store, &budget(110)).expect("assemble");
        assert_eq!(assembled.records.len(), 1);
        assert_eq!(assembled.records[0].id, "high");
        assert_eq!(assembled.omitted_ids, vec!["low".to_owned()]);
        assert_eq!(assembled.truncated_bytes, 110);
        assert_eq!(
            assembled.truncated_providers,
            vec!["diagnostics".to_owned()]
        );
        assert_eq!(
            assembled.truncated_tokens_estimate,
            ContextRequest::estimate_tokens(110)
        );
    }

    #[test]
    fn dangling_reference_is_typed_absence() {
        let store = ArtifactStore::new();
        let missing = ArtifactRef("artifact://9".to_owned());
        assert!(matches!(
            store.resolve(&missing),
            Err(ContextError::ArtifactUnavailable { .. })
        ));
    }
}

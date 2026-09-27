//! Content-addressed object store, rationale protocol, and structured checkpoint engine (AI-0162).
//!
//! Provides the immutable data plane for Wheel:
//! - [`ContentHash`]: Type-safe SHA-256 content address with strict hex validation.
//! - [`BlobStore`]: Deduplicating immutable blob storage in SQLite.
//! - [`Rationale`]: Structured cognitive intent and observation record replacing raw CoT.
//! - [`Checkpoint`]: Content-addressed DAG commit node binding parent hashes, task ID, agent, and rationale.
//! - [`ContentStore`]: Unified transactional store managing blobs, checkpoints, refs, and DAG log/merge-base traversal.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Maximum allowed size for a single blob (16 MiB).
pub const MAX_BLOB_BYTES: usize = 16 * 1024 * 1024;

/// Maximum number of parent checkpoints in a single commit (16).
pub const MAX_CHECKPOINT_PARENTS: usize = 16;

/// Maximum allowed length for task IDs (128 bytes).
pub const MAX_TASK_ID_BYTES: usize = 128;

/// Maximum allowed length for agent IDs (128 bytes).
pub const MAX_AGENT_ID_BYTES: usize = 128;

/// Maximum allowed length for checkpoint summaries (2048 bytes).
pub const MAX_SUMMARY_BYTES: usize = 2048;

/// Maximum allowed length for a ref name (256 bytes).
pub const MAX_REF_NAME_BYTES: usize = 256;

/// Maximum byte length for a single rationale field (4096 bytes).
pub const MAX_RATIONALE_FIELD_BYTES: usize = 4096;

/// Maximum aggregate byte length for a rationale (16384 bytes).
pub const MAX_RATIONALE_TOTAL_BYTES: usize = 16384;

/// Errors arising from content store, rationale, or checkpoint operations.
#[derive(Debug)]
pub enum ContentStoreError {
    /// SQLite storage error.
    Sqlite(rusqlite::Error),
    /// JSON serialization or deserialization failure.
    Json(serde_json::Error),
    /// Hash string is not a valid 64-character lowercase hex SHA-256 digest.
    InvalidHash(String),
    /// Retrieved blob failed integrity verification against its expected content hash.
    CorruptData {
        expected: ContentHash,
        found: ContentHash,
    },
    /// Retrieved checkpoint failed integrity verification against its expected content hash.
    CorruptCheckpoint {
        expected: ContentHash,
        found: ContentHash,
    },
    /// A referenced parent checkpoint does not exist in the store.
    MissingParent(ContentHash),
    /// A referenced context tree blob does not exist in the store.
    MissingTree(ContentHash),
    /// A ref target checkpoint does not exist in the store.
    MissingTarget(ContentHash),
    /// Field exceeded its strict byte size limit.
    OversizedField {
        field: &'static str,
        size: usize,
        max: usize,
    },
    /// A mandatory field was empty or whitespace-only.
    EmptyField(&'static str),
    /// An invalid ref name was provided.
    InvalidRefName(String),
}

impl fmt::Display for ContentStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(err) => write!(f, "sqlite error: {err}"),
            Self::Json(err) => write!(f, "json serialization error: {err}"),
            Self::InvalidHash(hex) => write!(f, "invalid SHA-256 hex string: {hex:?}"),
            Self::CorruptData { expected, found } => {
                write!(
                    f,
                    "blob data corrupt: expected hash {expected}, found {found}"
                )
            }
            Self::CorruptCheckpoint { expected, found } => {
                write!(
                    f,
                    "checkpoint data corrupt: expected hash {expected}, found {found}"
                )
            }
            Self::MissingParent(hash) => {
                write!(f, "referenced parent checkpoint {hash} does not exist")
            }
            Self::MissingTree(hash) => {
                write!(f, "referenced context tree blob {hash} does not exist")
            }
            Self::MissingTarget(hash) => {
                write!(f, "target checkpoint {hash} for ref does not exist")
            }
            Self::OversizedField { field, size, max } => {
                write!(
                    f,
                    "field '{field}' exceeds size limit: {size} bytes > {max} bytes"
                )
            }
            Self::EmptyField(field) => write!(f, "field '{field}' must not be empty"),
            Self::InvalidRefName(name) => write!(f, "invalid ref name {name:?}"),
        }
    }
}

impl std::error::Error for ContentStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(err) => Some(err),
            Self::Json(err) => Some(err),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for ContentStoreError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Sqlite(err)
    }
}

impl From<serde_json::Error> for ContentStoreError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

/// Type-safe SHA-256 content address (32 bytes).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Compute the SHA-256 digest of the given byte slice.
    #[must_use]
    pub fn compute(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        Self(bytes)
    }

    /// Construct a `ContentHash` from raw 32 bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Access the underlying 32-byte digest array.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Parse a 64-character lowercase hex string into a `ContentHash`.
    pub fn from_hex(s: &str) -> Result<Self, ContentStoreError> {
        if s.len() != 64 {
            return Err(ContentStoreError::InvalidHash(s.to_string()));
        }

        // Validate strictly lowercase hex digits
        for b in s.bytes() {
            if !matches!(b, b'0'..=b'9' | b'a'..=b'f') {
                return Err(ContentStoreError::InvalidHash(s.to_string()));
            }
        }

        let mut bytes = [0u8; 32];
        hex::decode_to_slice(s, &mut bytes)
            .map_err(|_| ContentStoreError::InvalidHash(s.to_string()))?;

        Ok(Self(bytes))
    }

    /// Render the content hash as a 64-character lowercase hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({})", self.to_hex())
    }
}

impl FromStr for ContentHash {
    type Err = ContentStoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_hex(s)
    }
}

impl Serialize for ContentHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Structured cognitive intent and observation record replacing raw CoT dumps.
///
/// Bounded to avoid token bloat and cognitive inertia across model inferences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rationale {
    /// Motivation / Intent behind this turn or decision (why this action is taken).
    pub why: String,
    /// Specific action, hypothesis, or modification (what is being done).
    pub what: String,
    /// Optional target scope, symbol, or path boundary (where focus lies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub where_focus: Option<String>,
    /// Optional methodology, tool choice, or algorithmic technique.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub how: Option<String>,
    /// Expected result or hypothesis verification criteria.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Observed findings, evidence, or actual test results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
}

impl Rationale {
    /// Construct a basic rationale with mandatory `why` and `what` fields.
    pub fn new(why: impl Into<String>, what: impl Into<String>) -> Self {
        Self {
            why: why.into(),
            what: what.into(),
            where_focus: None,
            how: None,
            expected: None,
            observed: None,
        }
    }

    /// Builder method for `where_focus`.
    #[must_use]
    pub fn with_where(mut self, where_focus: impl Into<String>) -> Self {
        self.where_focus = Some(where_focus.into());
        self
    }

    /// Builder method for `how`.
    #[must_use]
    pub fn with_how(mut self, how: impl Into<String>) -> Self {
        self.how = Some(how.into());
        self
    }

    /// Builder method for `expected`.
    #[must_use]
    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    /// Builder method for `observed`.
    #[must_use]
    pub fn with_observed(mut self, observed: impl Into<String>) -> Self {
        self.observed = Some(observed.into());
        self
    }

    /// Validate fields against bounds and non-empty invariants.
    pub fn validate(&self) -> Result<(), ContentStoreError> {
        if self.why.trim().is_empty() {
            return Err(ContentStoreError::EmptyField("why"));
        }
        if self.what.trim().is_empty() {
            return Err(ContentStoreError::EmptyField("what"));
        }

        Self::check_field("why", &self.why)?;
        Self::check_field("what", &self.what)?;

        if let Some(ref s) = self.where_focus {
            Self::check_field("where_focus", s)?;
        }
        if let Some(ref s) = self.how {
            Self::check_field("how", s)?;
        }
        if let Some(ref s) = self.expected {
            Self::check_field("expected", s)?;
        }
        if let Some(ref s) = self.observed {
            Self::check_field("observed", s)?;
        }

        let total = self.why.len()
            + self.what.len()
            + self.where_focus.as_ref().map_or(0, String::len)
            + self.how.as_ref().map_or(0, String::len)
            + self.expected.as_ref().map_or(0, String::len)
            + self.observed.as_ref().map_or(0, String::len);

        if total > MAX_RATIONALE_TOTAL_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "rationale_total",
                size: total,
                max: MAX_RATIONALE_TOTAL_BYTES,
            });
        }

        Ok(())
    }

    fn check_field(field: &'static str, val: &str) -> Result<(), ContentStoreError> {
        if val.len() > MAX_RATIONALE_FIELD_BYTES {
            return Err(ContentStoreError::OversizedField {
                field,
                size: val.len(),
                max: MAX_RATIONALE_FIELD_BYTES,
            });
        }
        Ok(())
    }
}

/// Draft data used to create a new content-addressed [`Checkpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointDraft {
    /// Zero or more parent checkpoint hashes in the DAG.
    pub parents: Vec<ContentHash>,
    /// The associated task or issue ID (e.g. "AI-0162").
    pub task_id: String,
    /// Identity of the acting agent authoring this checkpoint.
    pub agent_id: String,
    /// Structured rationale explaining the intent and findings of this state.
    pub rationale: Rationale,
    /// Optional content hash of a referenced context tree / project snapshot blob.
    pub tree_hash: Option<ContentHash>,
    /// Short human-readable summary of this checkpoint.
    pub summary: String,
    /// Timestamp in milliseconds since unix epoch.
    pub timestamp_ms: u64,
}

impl CheckpointDraft {
    /// Compute the deterministic canonical digest for this checkpoint draft.
    ///
    /// Uses length-prefixed field encoding to guarantee unambiguous canonical hashing,
    /// preventing delimiter-collision vulnerabilities across draft fields.
    pub fn canonical_hash(&self) -> Result<ContentHash, ContentStoreError> {
        let rationale_json = serde_json::to_string(&self.rationale)?;

        let mut sorted_parents = self.parents.clone();
        sorted_parents.sort_unstable();

        let parents_str = sorted_parents
            .iter()
            .map(ContentHash::to_hex)
            .collect::<Vec<_>>()
            .join(",");

        let tree_hex = self.tree_hash.as_ref().map(ContentHash::to_hex);
        let tree_str = tree_hex.as_deref().unwrap_or("");
        let ts_str = self.timestamp_ms.to_string();

        let mut hasher = Sha256::new();
        hasher.update(b"checkpoint:v2\0");
        for field in [
            parents_str.as_bytes(),
            self.task_id.as_bytes(),
            self.agent_id.as_bytes(),
            tree_str.as_bytes(),
            self.summary.as_bytes(),
            ts_str.as_bytes(),
            rationale_json.as_bytes(),
        ] {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field);
        }

        let digest: [u8; 32] = hasher.finalize().into();
        Ok(ContentHash::from_bytes(digest))
    }
}

/// An immutable, content-addressed commit node in the context DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Content-addressed SHA-256 hash identifying this checkpoint.
    pub id: ContentHash,
    /// Parent checkpoint hashes.
    pub parents: Vec<ContentHash>,
    /// Associated task / issue ID.
    pub task_id: String,
    /// Authoring agent identity.
    pub agent_id: String,
    /// Structured rationale.
    pub rationale: Rationale,
    /// Optional context tree / snapshot blob hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_hash: Option<ContentHash>,
    /// Summary of changes / cognitive state.
    pub summary: String,
    /// Creation timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
}

/// Transactional SQLite-backed store for content-addressed blobs, checkpoints, and refs.
pub struct ContentStore {
    conn: Connection,
}

impl ContentStore {
    /// Open a content store at the given SQLite database path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContentStoreError> {
        let conn = Connection::open(path)?;
        Self::init_with_connection(conn)
    }

    /// Open an in-memory content store (useful for testing and ephemeral workflows).
    pub fn open_in_memory() -> Result<Self, ContentStoreError> {
        let conn = Connection::open_in_memory()?;
        Self::init_with_connection(conn)
    }

    fn init_with_connection(conn: Connection) -> Result<Self, ContentStoreError> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;

             CREATE TABLE IF NOT EXISTS blobs (
                 hash TEXT PRIMARY KEY,
                 size INTEGER NOT NULL,
                 data BLOB NOT NULL,
                 created_at_ms INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS checkpoints (
                 hash TEXT PRIMARY KEY,
                 parents_json TEXT NOT NULL,
                 task_id TEXT NOT NULL,
                 agent_id TEXT NOT NULL,
                 rationale_json TEXT NOT NULL,
                 tree_hash TEXT,
                 summary TEXT NOT NULL,
                 created_at_ms INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id);

             CREATE TABLE IF NOT EXISTS refs (
                 name TEXT PRIMARY KEY,
                 target_hash TEXT NOT NULL,
                 updated_at_ms INTEGER NOT NULL
             );",
        )?;

        Ok(Self { conn })
    }

    /// Store a raw data blob.
    ///
    /// If the blob already exists, it is deduplicated and the existing [`ContentHash`]
    /// is returned without disk duplication.
    pub fn put_blob(
        &mut self,
        data: &[u8],
        created_at_ms: u64,
    ) -> Result<ContentHash, ContentStoreError> {
        if data.len() > MAX_BLOB_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "blob_data",
                size: data.len(),
                max: MAX_BLOB_BYTES,
            });
        }

        let hash = ContentHash::compute(data);
        let hash_hex = hash.to_hex();

        // Check for deduplication
        let mut exists_stmt = self.conn.prepare("SELECT 1 FROM blobs WHERE hash = ?1")?;
        let exists = exists_stmt.exists(params![hash_hex])?;

        if !exists {
            self.conn.execute(
                "INSERT OR IGNORE INTO blobs (hash, size, data, created_at_ms) VALUES (?1, ?2, ?3, ?4)",
                params![hash_hex, data.len() as i64, data, created_at_ms as i64],
            )?;
        }

        Ok(hash)
    }

    /// Retrieve a blob by its content hash.
    ///
    /// Verifies the cryptographic digest before returning to ensure data integrity.
    pub fn get_blob(&self, hash: &ContentHash) -> Result<Option<Vec<u8>>, ContentStoreError> {
        let hash_hex = hash.to_hex();
        let mut stmt = self
            .conn
            .prepare("SELECT data FROM blobs WHERE hash = ?1")?;

        let maybe_data: Option<Vec<u8>> = stmt
            .query_row(params![hash_hex], |row| row.get::<_, Vec<u8>>(0))
            .optional()?;

        match maybe_data {
            Some(data) => {
                let computed = ContentHash::compute(&data);
                if computed != *hash {
                    return Err(ContentStoreError::CorruptData {
                        expected: *hash,
                        found: computed,
                    });
                }
                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    /// Check if a blob exists in the store without reading its payload.
    pub fn has_blob(&self, hash: &ContentHash) -> Result<bool, ContentStoreError> {
        let hash_hex = hash.to_hex();
        let mut stmt = self.conn.prepare("SELECT 1 FROM blobs WHERE hash = ?1")?;
        Ok(stmt.exists(params![hash_hex])?)
    }

    /// Create and persist a new [`Checkpoint`] from a validated draft.
    ///
    /// Ensures all referenced parents and context trees exist before committing.
    pub fn commit_checkpoint(
        &mut self,
        draft: CheckpointDraft,
    ) -> Result<Checkpoint, ContentStoreError> {
        if draft.parents.len() > MAX_CHECKPOINT_PARENTS {
            return Err(ContentStoreError::OversizedField {
                field: "parents",
                size: draft.parents.len(),
                max: MAX_CHECKPOINT_PARENTS,
            });
        }
        if draft.task_id.trim().is_empty() {
            return Err(ContentStoreError::EmptyField("task_id"));
        }
        if draft.task_id.len() > MAX_TASK_ID_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "task_id",
                size: draft.task_id.len(),
                max: MAX_TASK_ID_BYTES,
            });
        }
        if draft.agent_id.trim().is_empty() {
            return Err(ContentStoreError::EmptyField("agent_id"));
        }
        if draft.agent_id.len() > MAX_AGENT_ID_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "agent_id",
                size: draft.agent_id.len(),
                max: MAX_AGENT_ID_BYTES,
            });
        }
        if draft.summary.len() > MAX_SUMMARY_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "summary",
                size: draft.summary.len(),
                max: MAX_SUMMARY_BYTES,
            });
        }

        draft.rationale.validate()?;

        // Verify parents exist
        for parent in &draft.parents {
            if !self.has_checkpoint(parent)? {
                return Err(ContentStoreError::MissingParent(*parent));
            }
        }

        // Verify tree blob exists if provided
        if let Some(ref tree_hash) = draft.tree_hash {
            if !self.has_blob(tree_hash)? {
                return Err(ContentStoreError::MissingTree(*tree_hash));
            }
        }

        let id = draft.canonical_hash()?;
        let id_hex = id.to_hex();

        // Idempotent commit: if checkpoint hash already exists, return it
        if !self.has_checkpoint(&id)? {
            let rationale_json = serde_json::to_string(&draft.rationale)?;
            let parents_json = serde_json::to_string(&draft.parents)?;
            let tree_hex = draft.tree_hash.as_ref().map(ContentHash::to_hex);

            self.conn.execute(
                "INSERT OR IGNORE INTO checkpoints (hash, parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    id_hex,
                    parents_json,
                    draft.task_id,
                    draft.agent_id,
                    rationale_json,
                    tree_hex,
                    draft.summary,
                    draft.timestamp_ms as i64
                ],
            )?;
        }

        Ok(Checkpoint {
            id,
            parents: draft.parents,
            task_id: draft.task_id,
            agent_id: draft.agent_id,
            rationale: draft.rationale,
            tree_hash: draft.tree_hash,
            summary: draft.summary,
            timestamp_ms: draft.timestamp_ms,
        })
    }

    /// Retrieve a checkpoint by its content hash.
    pub fn get_checkpoint(
        &self,
        hash: &ContentHash,
    ) -> Result<Option<Checkpoint>, ContentStoreError> {
        let hash_hex = hash.to_hex();
        let mut stmt = self.conn.prepare(
            "SELECT parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms
             FROM checkpoints WHERE hash = ?1",
        )?;

        let row = stmt
            .query_row(params![hash_hex], |row| {
                let parents_json: String = row.get::<_, String>(0)?;
                let task_id: String = row.get::<_, String>(1)?;
                let agent_id: String = row.get::<_, String>(2)?;
                let rationale_json: String = row.get::<_, String>(3)?;
                let tree_hash_hex: Option<String> = row.get::<_, Option<String>>(4)?;
                let summary: String = row.get::<_, String>(5)?;
                let created_at_ms: i64 = row.get::<_, i64>(6)?;

                Ok((
                    parents_json,
                    task_id,
                    agent_id,
                    rationale_json,
                    tree_hash_hex,
                    summary,
                    created_at_ms,
                ))
            })
            .optional()?;

        match row {
            Some((
                parents_json,
                task_id,
                agent_id,
                rationale_json,
                tree_hash_hex,
                summary,
                created_at_ms,
            )) => {
                let parents: Vec<ContentHash> = serde_json::from_str(&parents_json)?;
                let rationale: Rationale = serde_json::from_str(&rationale_json)?;
                let tree_hash = match tree_hash_hex {
                    Some(ref hex) => Some(ContentHash::from_hex(hex)?),
                    None => None,
                };

                let draft = CheckpointDraft {
                    parents,
                    task_id,
                    agent_id,
                    rationale,
                    tree_hash,
                    summary,
                    timestamp_ms: created_at_ms as u64,
                };

                let computed_hash = draft.canonical_hash()?;
                if computed_hash != *hash {
                    return Err(ContentStoreError::CorruptCheckpoint {
                        expected: *hash,
                        found: computed_hash,
                    });
                }

                Ok(Some(Checkpoint {
                    id: *hash,
                    parents: draft.parents,
                    task_id: draft.task_id,
                    agent_id: draft.agent_id,
                    rationale: draft.rationale,
                    tree_hash: draft.tree_hash,
                    summary: draft.summary,
                    timestamp_ms: draft.timestamp_ms,
                }))
            }
            None => Ok(None),
        }
    }

    /// Check if a checkpoint exists in the store.
    pub fn has_checkpoint(&self, hash: &ContentHash) -> Result<bool, ContentStoreError> {
        let hash_hex = hash.to_hex();
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM checkpoints WHERE hash = ?1")?;
        Ok(stmt.exists(params![hash_hex])?)
    }

    /// Update or create a named reference (e.g. "HEAD", "heads/main", "tags/v1").
    pub fn update_ref(
        &mut self,
        name: &str,
        target: &ContentHash,
        updated_at_ms: u64,
    ) -> Result<(), ContentStoreError> {
        Self::validate_ref_name(name)?;

        if !self.has_checkpoint(target)? {
            return Err(ContentStoreError::MissingTarget(*target));
        }

        let target_hex = target.to_hex();
        self.conn.execute(
            "INSERT INTO refs (name, target_hash, updated_at_ms)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET target_hash = excluded.target_hash, updated_at_ms = excluded.updated_at_ms",
            params![name, target_hex, updated_at_ms as i64],
        )?;

        Ok(())
    }

    /// Get the target checkpoint hash of a named reference.
    pub fn get_ref(&self, name: &str) -> Result<Option<ContentHash>, ContentStoreError> {
        let mut stmt = self
            .conn
            .prepare("SELECT target_hash FROM refs WHERE name = ?1")?;
        let maybe_hex: Option<String> = stmt
            .query_row(params![name], |row| row.get::<_, String>(0))
            .optional()?;

        match maybe_hex {
            Some(hex) => Ok(Some(ContentHash::from_hex(&hex)?)),
            None => Ok(None),
        }
    }

    /// Delete a named reference. Returns `true` if the reference existed and was deleted.
    pub fn delete_ref(&mut self, name: &str) -> Result<bool, ContentStoreError> {
        let affected = self
            .conn
            .execute("DELETE FROM refs WHERE name = ?1", params![name])?;
        Ok(affected > 0)
    }

    /// List all named references and their target checkpoint hashes.
    pub fn list_refs(&self) -> Result<Vec<(String, ContentHash)>, ContentStoreError> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, target_hash FROM refs ORDER BY name ASC")?;

        let rows = stmt.query_map([], |row| {
            let name: String = row.get::<_, String>(0)?;
            let hash_hex: String = row.get::<_, String>(1)?;
            Ok((name, hash_hex))
        })?;

        let mut result = Vec::new();
        for item in rows {
            let (name, hash_hex) = item?;
            result.push((name, ContentHash::from_hex(&hash_hex)?));
        }

        Ok(result)
    }

    /// Traverse backward through checkpoint DAG parent links from `head`, returning linear history.
    pub fn log(
        &self,
        head: &ContentHash,
        limit: usize,
    ) -> Result<Vec<Checkpoint>, ContentStoreError> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(*head);
        visited.insert(*head);

        while let Some(current_hash) = queue.pop_front() {
            if result.len() >= limit {
                break;
            }

            if let Some(cp) = self.get_checkpoint(&current_hash)? {
                for parent in &cp.parents {
                    if visited.insert(*parent) {
                        queue.push_back(*parent);
                    }
                }
                result.push(cp);
            }
        }

        Ok(result)
    }

    /// Find the lowest common ancestor (LCA / `merge_base`) of two checkpoints in the DAG.
    ///
    /// Returns `Ok(Some(hash))` if a common ancestor exists, or `Ok(None)` if the graphs are disjoint.
    pub fn merge_base(
        &self,
        a: &ContentHash,
        b: &ContentHash,
    ) -> Result<Option<ContentHash>, ContentStoreError> {
        if a == b {
            return Ok(Some(*a));
        }

        // Collect all ancestors of `a`
        let mut ancestors_a = HashSet::new();
        let mut queue_a = VecDeque::new();
        queue_a.push_back(*a);
        ancestors_a.insert(*a);

        while let Some(curr) = queue_a.pop_front() {
            if let Some(cp) = self.get_checkpoint(&curr)? {
                for parent in cp.parents {
                    if ancestors_a.insert(parent) {
                        queue_a.push_back(parent);
                    }
                }
            }
        }

        // Traverse BFS from `b`, returning the first node found in `ancestors_a`
        let mut visited_b = HashSet::new();
        let mut queue_b = VecDeque::new();
        queue_b.push_back(*b);
        visited_b.insert(*b);

        while let Some(curr) = queue_b.pop_front() {
            if ancestors_a.contains(&curr) {
                return Ok(Some(curr));
            }

            if let Some(cp) = self.get_checkpoint(&curr)? {
                for parent in cp.parents {
                    if visited_b.insert(parent) {
                        queue_b.push_back(parent);
                    }
                }
            }
        }

        Ok(None)
    }

    fn validate_ref_name(name: &str) -> Result<(), ContentStoreError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(ContentStoreError::EmptyField("ref_name"));
        }
        if name.len() > MAX_REF_NAME_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "ref_name",
                size: name.len(),
                max: MAX_REF_NAME_BYTES,
            });
        }

        for b in name.bytes() {
            let valid =
                matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'/' | b'-' | b'_' | b'.');
            if !valid {
                return Err(ContentStoreError::InvalidRefName(name.to_string()));
            }
        }

        Ok(())
    }
}

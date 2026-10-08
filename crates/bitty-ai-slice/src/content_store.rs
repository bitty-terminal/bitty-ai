//! Content-addressed object store, rationale protocol, and structured checkpoint engine (AI-0162).
//!
//! Provides the immutable data plane for Wheel:
//! - [`ContentHash`]: Type-safe SHA-256 content address with strict hex validation.
//! - [`BlobStore`]: Deduplicating immutable blob storage in SQLite.
//! - [`Rationale`]: Structured cognitive intent and observation record replacing raw CoT.
//! - [`Checkpoint`]: Content-addressed DAG commit node binding parent hashes, task ID, agent, and rationale.
//! - [`ContentStore`]: Unified transactional store managing blobs, checkpoints, refs, and DAG log/merge-base traversal.
//!
//! ## Durable pragma profile (AI-0178, formal path)
//!
//! Every open applies one unified profile before any schema statement:
//! `journal_mode=WAL`, `synchronous=FULL`, `foreign_keys=ON`, and a nonzero
//! `busy_timeout` of [`DURABLE_BUSY_TIMEOUT_MS`] milliseconds, plus
//! `locking_mode=EXCLUSIVE` to hold the single-writer lock for the handle's
//! lifetime (mirroring `journal_prototype` admission). Choices:
//! - WAL keeps multi-statement commit transactions crash-atomic (torn writes
//!   replay or roll back as a unit) while readers proceed during checkpoints.
//! - FULL flushes each commit to the OS before it returns, so an acknowledged
//!   HEAD survives a host crash (NORMAL could lose the tail).
//! - Foreign keys enforce DAG integrity (parents, trees, ref targets) at the
//!   SQL layer, failing closed on dangling links.
//! - A 5 s busy timeout tolerates transient WAL checkpoint contention on CI
//!   without masking a true second writer, which surfaces as
//!   [`ContentStoreError::WriterBusy`] after the timeout. The initial
//!   single-writer claim uses a zero timeout to fail fast; steady state is
//!   always nonzero.
//! - EXCLUSIVE holds the writer lock so a second `open`/`open_durable` while
//!   the first lives fails with [`ContentStoreError::WriterBusy`] instead of
//!   silently serializing.
//!
//! ## Single-Connection ownership (AI-0178)
//!
//! [`ContentStore`] owns its SQLite [`Connection`] behind
//! `Arc<Mutex<..>>` so [`crate::wheel_kernel::WheelKernel`] can open the
//! database file exactly once, initialize content and task schemas on that
//! one connection, and share the handle into both [`ContentStore`] and
//! [`crate::task_dag::TaskEngine`] via [`ContentStore::from_shared`]. Two
//! independent `Connection`s on the same file (the pre-0178 pattern) split
//! pragmas, split transactions, and let recovery observe divergent snapshots:
//! a crash between the tree-blob write on one connection and the HEAD update
//! on the other leaves a half-advanced HEAD. One connection plus one
//! `transaction()` per commit keeps tree blob, checkpoint row, and HEAD/branch
//! refs atomic.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

/// Nonzero SQLite busy timeout for the durable profile (5 s).
///
/// Tolerates transient WAL checkpoint contention; a true second writer still
/// fails closed as [`ContentStoreError::WriterBusy`] after the timeout. The
/// initial writer-claim probe uses a zero timeout to fail fast; steady state
/// is always this nonzero value.
pub const DURABLE_BUSY_TIMEOUT_MS: u64 = 5000;

/// Durable profile marker stored in `durable_meta.profile`.
pub const DURABLE_PROFILE: &str = "durable-v1";

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
    /// Malformed, corrupt, or schema-incompatible database state. Fail closed:
    /// the file is never reset, truncated, or repaired implicitly.
    Corrupt { detail: String },
    /// Another live writer holds this database file's single-writer lock.
    WriterBusy,
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
            Self::Corrupt { detail } => {
                write!(f, "content store corrupt or incompatible: {detail}")
            }
            Self::WriterBusy => write!(f, "another writer holds the single-writer lock"),
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
///
/// The connection is shared behind `Arc<Mutex<..>>` so the Wheel durable
/// path opens the file exactly once and splits the handle into the content
/// store and the task engine (see module docs). Standalone `open` callers
/// get a private handle; `from_shared` joins an already-admitted handle.
pub struct ContentStore {
    conn: Arc<Mutex<Connection>>,
}

/// Durable pragma profile applied on every open (see module docs).
pub(crate) fn apply_durable_pragmas(conn: &Connection) -> Result<(), ContentStoreError> {
    conn.busy_timeout(Duration::from_millis(DURABLE_BUSY_TIMEOUT_MS))
        .map_err(map_busy)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = FULL;
         PRAGMA foreign_keys = ON;
         PRAGMA locking_mode = EXCLUSIVE;",
    )
    .map_err(map_busy)?;
    Ok(())
}

fn is_busy(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, _)
            if matches!(
                e.code,
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
            )
    )
}

/// Map a raw SQLite open error to a facade-ready string preserving the
/// WriterBusy/Corrupt distinction for [`crate::facade::FacadeError`].
pub(crate) fn map_busy_for_facade(err: rusqlite::Error) -> String {
    match map_busy(err) {
        ContentStoreError::WriterBusy => {
            "writer busy: another writer holds the single-writer lock".to_owned()
        }
        ContentStoreError::Corrupt { detail } => format!("corrupt: {detail}"),
        ContentStoreError::Sqlite(inner) => format!("sqlite error: {inner}"),
        other => other.to_string(),
    }
}

fn map_busy(err: rusqlite::Error) -> ContentStoreError {
    if is_busy(&err) {
        ContentStoreError::WriterBusy
    } else if let rusqlite::Error::SqliteFailure(e, _) = &err {
        if matches!(
            e.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return ContentStoreError::Corrupt {
                detail: err.to_string(),
            };
        }
        ContentStoreError::Sqlite(err)
    } else {
        ContentStoreError::Sqlite(err)
    }
}

pub(crate) fn check_sqlite_magic(path: &Path) -> Result<(), ContentStoreError> {
    // Bounded header probe: only the first 16 bytes are ever read, so the
    // check costs O(1) memory no matter how large the database grows.
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(ContentStoreError::Corrupt {
                detail: format!("unreadable database file: {e}"),
            });
        }
    };
    let mut header = [0u8; 16];
    let mut read = 0;
    while read < header.len() {
        match file.read(&mut header[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) => {
                return Err(ContentStoreError::Corrupt {
                    detail: format!("unreadable database file: {e}"),
                });
            }
        }
    }
    if read == 0 {
        return Ok(());
    }
    if read < 16 || header != *b"SQLite format 3\0" {
        return Err(ContentStoreError::Corrupt {
            detail: "file is not a SQLite database".to_owned(),
        });
    }
    Ok(())
}

pub(crate) const CONTENT_SCHEMA: &str = "
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
);
CREATE TABLE IF NOT EXISTS reflog (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_name TEXT NOT NULL,
    old_hash TEXT,
    new_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reflog_ref_seq ON reflog(ref_name, seq);
CREATE TABLE IF NOT EXISTS durable_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);";

pub(crate) fn init_content_schema(conn: &Connection) -> Result<(), ContentStoreError> {
    conn.execute_batch(CONTENT_SCHEMA).map_err(map_busy)?;
    Ok(())
}

pub(crate) fn claim_writer_fast(conn: &Connection) -> Result<(), ContentStoreError> {
    conn.busy_timeout(Duration::ZERO).map_err(map_busy)?;
    let claim = conn.execute(
        "INSERT OR REPLACE INTO durable_meta (key, value) VALUES ('profile', ?1)",
        params![DURABLE_PROFILE],
    );
    conn.busy_timeout(Duration::from_millis(DURABLE_BUSY_TIMEOUT_MS))
        .map_err(map_busy)?;
    claim.map_err(map_busy)?;
    Ok(())
}

pub(crate) fn verify_profile(conn: &Connection) -> Result<(), ContentStoreError> {
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM durable_meta WHERE key = 'profile'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(map_busy)?;
    match stored.as_deref() {
        Some(DURABLE_PROFILE) => Ok(()),
        Some(other) => Err(ContentStoreError::Corrupt {
            detail: format!("unrecognized durable profile '{other}'; no migration exists"),
        }),
        // Missing marker means a pre-0178 database without the marker table
        // row (or a fresh init that has not claimed yet): the caller writes
        // it via claim_writer_fast, so admission succeeds (migration, not
        // refusal). A partially created store is caught by admit_or_init
        // before this runs.
        None => Ok(()),
    }
}

pub(crate) fn admit_or_init(conn: &Connection) -> Result<(), ContentStoreError> {
    let mut stmt = conn
        .prepare(
            "SELECT type, name FROM sqlite_master
             WHERE name COLLATE NOCASE IN ('blobs', 'checkpoints', 'refs', 'reflog', 'durable_meta')",
        )
        .map_err(map_busy)?;
    let mut matches: Vec<(String, String)> = Vec::new();
    let mut rows = stmt.query([]).map_err(map_busy)?;
    while let Some(row) = rows.next().map_err(map_busy)? {
        matches.push((
            row.get::<_, String>(0).map_err(map_busy)?,
            row.get::<_, String>(1).map_err(map_busy)?,
        ));
    }
    drop(rows);
    drop(stmt);

    const EXPECTED: [&str; 5] = ["blobs", "checkpoints", "refs", "reflog", "durable_meta"];
    let mut present: Vec<&str> = Vec::new();
    for expected in EXPECTED {
        if matches
            .iter()
            .any(|(ty, name)| ty == "table" && name == expected)
        {
            present.push(expected);
            continue;
        }
        if let Some((ty, name)) = matches.iter().find(|(ty, name)| {
            matches!(ty.as_str(), "table" | "view" | "index") && name.eq_ignore_ascii_case(expected)
        }) {
            return Err(ContentStoreError::Corrupt {
                detail: format!(
                    "durable name '{expected}' is occupied by {ty} '{name}', not the exact-case table"
                ),
            });
        }
    }
    // Fresh or foreign-but-empty of durable names: no schema statement has
    // run yet, so initializing now cannot clobber anything. Unrelated tables
    // are preserved (no exclusive ownership claim over the file).
    if present.is_empty() {
        return Ok(());
    }
    // Additive-table migration (AI-0178 durable_meta, AI-0180 reflog): the
    // core triple (blobs, checkpoints, refs) must be fully present; the only
    // tables allowed to be missing are the ones introduced after it. A store
    // missing just `reflog` is the AI-0180 migration (init creates it); a
    // store missing `durable_meta`, or both, is the pre-0178 path (init
    // creates them). Every present table is still shape-checked first, so a
    // malformed `reflog` fails closed here instead of being silently kept.
    // Anything missing from the core triple is a partial schema and fails
    // closed below.
    let core_present = ["blobs", "checkpoints", "refs"]
        .iter()
        .all(|table| present.contains(table));
    if core_present && present.len() < EXPECTED.len() {
        for table in ["blobs", "checkpoints", "refs"] {
            verify_content_table_shape(conn, table)?;
        }
        if present.contains(&"reflog") {
            verify_content_table_shape(conn, "reflog")?;
        }
        return Ok(());
    }
    if present.len() != EXPECTED.len() {
        return Err(ContentStoreError::Corrupt {
            detail: format!("partial durable schema; present tables: {present:?}"),
        });
    }
    for table in ["blobs", "checkpoints", "refs", "reflog"] {
        verify_content_table_shape(conn, table)?;
    }
    verify_profile(conn)?;
    Ok(())
}

fn verify_content_table_shape(conn: &Connection, table: &str) -> Result<(), ContentStoreError> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(map_busy)?;
    let mut cols: Vec<(String, String, bool, u32)> = Vec::new();
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)? != 0,
                row.get::<_, i64>(5)? as u32,
            ))
        })
        .map_err(map_busy)?;
    for row in rows {
        cols.push(row.map_err(map_busy)?);
    }
    drop(stmt);
    let expected: &[(&str, &str, bool, u32)] = match table {
        "blobs" => &[
            ("hash", "TEXT", false, 1),
            ("size", "INTEGER", true, 0),
            ("data", "BLOB", true, 0),
            ("created_at_ms", "INTEGER", true, 0),
        ],
        "checkpoints" => &[
            ("hash", "TEXT", false, 1),
            ("parents_json", "TEXT", true, 0),
            ("task_id", "TEXT", true, 0),
            ("agent_id", "TEXT", true, 0),
            ("rationale_json", "TEXT", true, 0),
            ("tree_hash", "TEXT", false, 0),
            ("summary", "TEXT", true, 0),
            ("created_at_ms", "INTEGER", true, 0),
        ],
        "refs" => &[
            ("name", "TEXT", false, 1),
            ("target_hash", "TEXT", true, 0),
            ("updated_at_ms", "INTEGER", true, 0),
        ],
        "reflog" => &[
            ("seq", "INTEGER", false, 1),
            ("ref_name", "TEXT", true, 0),
            ("old_hash", "TEXT", false, 0),
            ("new_hash", "TEXT", true, 0),
            ("reason", "TEXT", true, 0),
            ("actor", "TEXT", true, 0),
            ("at_ms", "INTEGER", true, 0),
        ],
        _ => return Ok(()),
    };
    if cols.len() != expected.len() {
        return Err(ContentStoreError::Corrupt {
            detail: format!(
                "table {table} has {} columns, expected {}",
                cols.len(),
                expected.len()
            ),
        });
    }
    for (i, (name, ty, notnull, pk)) in expected.iter().enumerate() {
        let (aname, aty, anotnull, apk) = &cols[i];
        if aname != *name || aty != *ty || anotnull != notnull || apk != pk {
            return Err(ContentStoreError::Corrupt {
                detail: format!("table {table} column {i} shape mismatch"),
            });
        }
    }
    Ok(())
}

impl ContentStore {
    pub(crate) fn lock_conn(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Connection>, ContentStoreError> {
        self.conn.lock().map_err(|_| ContentStoreError::Corrupt {
            detail: "content store lock poisoned; fail closed".to_owned(),
        })
    }

    /// Open a content store at the given SQLite database path.
    ///
    /// Applies the durable pragma profile, admits the store (new files are
    /// initialized; complete stores are verified; partial, occupied, or
    /// foreign stores fail with [`ContentStoreError::Corrupt`] unchanged),
    /// and claims the single-writer lock. The busy timeout stays zero through
    /// admission and schema init so a second open while the first lives fails
    /// fast with [`ContentStoreError::WriterBusy`]; steady state restores the
    /// nonzero profile timeout only after the writer claim succeeds.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContentStoreError> {
        let path = path.as_ref();
        check_sqlite_magic(path)?;
        let conn = Connection::open(path).map_err(map_busy)?;
        conn.busy_timeout(Duration::ZERO).map_err(map_busy)?;
        // Non-mutating gate: foreign_keys is per-connection state.
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(map_busy)?;
        admit_or_init(&conn)?;
        apply_durable_pragmas(&conn)?;
        conn.busy_timeout(Duration::ZERO).map_err(map_busy)?;
        init_content_schema(&conn)?;
        verify_profile(&conn)?;
        claim_writer_fast(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Open a durable content store (formal path, same profile as [`Self::open`]).
    ///
    /// Kept as a named alias so call sites spell the formal-path intent;
    /// behavior matches [`Self::open`] exactly.
    pub fn open_durable(path: impl AsRef<Path>) -> Result<Self, ContentStoreError> {
        Self::open(path)
    }

    /// Open an in-memory content store (useful for testing and ephemeral workflows).
    pub fn open_in_memory() -> Result<Self, ContentStoreError> {
        let conn = Connection::open_in_memory().map_err(map_busy)?;
        apply_durable_pragmas(&conn)?;
        init_content_schema(&conn)?;
        conn.execute(
            "INSERT OR REPLACE INTO durable_meta (key, value) VALUES ('profile', ?1)",
            params![DURABLE_PROFILE],
        )
        .map_err(map_busy)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Join an already-admitted shared connection (Wheel single-open path).
    ///
    /// The caller opened the file once, applied the durable profile, and
    /// initialized content and task schemas. This re-runs the idempotent
    /// schema creation and joins the handle without re-claiming the writer
    /// lock (the opener already holds it).
    pub(crate) fn from_shared(shared: Arc<Mutex<Connection>>) -> Result<Self, ContentStoreError> {
        {
            let guard = shared.lock().map_err(|_| ContentStoreError::Corrupt {
                detail: "content store lock poisoned; fail closed".to_owned(),
            })?;
            guard.execute_batch(CONTENT_SCHEMA).map_err(map_busy)?;
        }
        Ok(Self { conn: shared })
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

        // Check for deduplication (scoped lock so the guard drops before insert)
        let exists = {
            let guard = self.lock_conn()?;
            let mut exists_stmt = guard
                .prepare("SELECT 1 FROM blobs WHERE hash = ?1")
                .map_err(map_busy)?;
            exists_stmt.exists(params![hash_hex]).map_err(map_busy)?
        };

        if !exists {
            let guard = self.lock_conn()?;
            guard
                .execute(
                    "INSERT OR IGNORE INTO blobs (hash, size, data, created_at_ms) VALUES (?1, ?2, ?3, ?4)",
                    params![hash_hex, data.len() as i64, data, created_at_ms as i64],
                )
                .map_err(map_busy)?;
        }

        Ok(hash)
    }

    /// Retrieve a blob by its content hash.
    ///
    /// Verifies the cryptographic digest before returning to ensure data integrity.
    pub fn get_blob(&self, hash: &ContentHash) -> Result<Option<Vec<u8>>, ContentStoreError> {
        let hash_hex = hash.to_hex();
        let maybe_data: Option<Vec<u8>> = {
            let guard = self.lock_conn()?;
            let mut stmt = guard
                .prepare("SELECT data FROM blobs WHERE hash = ?1")
                .map_err(map_busy)?;

            stmt.query_row(params![hash_hex], |row| row.get::<_, Vec<u8>>(0))
                .optional()
                .map_err(map_busy)?
        };
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
        let guard = self.lock_conn()?;
        let mut stmt = guard
            .prepare("SELECT 1 FROM blobs WHERE hash = ?1")
            .map_err(map_busy)?;
        stmt.exists(params![hash_hex]).map_err(map_busy)
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

            let guard = self.lock_conn()?;
            guard
                .execute(
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
                )
                .map_err(map_busy)?;
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

    /// Atomically persist tree blob, checkpoint row, and HEAD/branch refs in one transaction.
    ///
    /// This is the Wheel durable commit path (AI-0178): the tree blob insert,
    /// the checkpoint insert, and every ref update commit or roll back as one
    /// SQLite transaction, so a crash between the blob insert and the HEAD
    /// update can never leave a half-advanced HEAD. HEAD advances only when
    /// the transaction commits. Idempotent: re-committing an existing
    /// checkpoint still advances the named refs to it.
    pub fn commit_checkpoint_atomic(
        &mut self,
        tree_bytes: &[u8],
        tree_created_at_ms: u64,
        draft: CheckpointDraft,
        ref_names: &[&str],
        updated_at_ms: u64,
    ) -> Result<Checkpoint, ContentStoreError> {
        if tree_bytes.len() > MAX_BLOB_BYTES {
            return Err(ContentStoreError::OversizedField {
                field: "blob_data",
                size: tree_bytes.len(),
                max: MAX_BLOB_BYTES,
            });
        }
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
        for name in ref_names {
            Self::validate_ref_name(name)?;
        }

        let tree_hash = ContentHash::compute(tree_bytes);
        let tree_hex = tree_hash.to_hex();
        let effective = CheckpointDraft {
            parents: draft.parents.clone(),
            task_id: draft.task_id.clone(),
            agent_id: draft.agent_id.clone(),
            rationale: draft.rationale.clone(),
            tree_hash: Some(tree_hash),
            summary: draft.summary.clone(),
            timestamp_ms: draft.timestamp_ms,
        };
        let id = effective.canonical_hash()?;
        let id_hex = id.to_hex();
        let rationale_json = serde_json::to_string(&effective.rationale)?;
        let parents_json = serde_json::to_string(&effective.parents)?;

        let mut guard = self.lock_conn()?;
        let tx = guard.transaction().map_err(map_busy)?;
        // Parents must exist inside the same transaction (immutable rows, so
        // the check cannot race with deletion; checkpoints are never deleted).
        for parent in &effective.parents {
            let hex = parent.to_hex();
            let exists: bool = tx
                .query_row(
                    "SELECT 1 FROM checkpoints WHERE hash = ?1",
                    params![hex],
                    |_| Ok(()),
                )
                .optional()
                .map_err(map_busy)?
                .is_some();
            if !exists {
                return Err(ContentStoreError::MissingParent(*parent));
            }
        }
        tx.execute(
            "INSERT OR IGNORE INTO blobs (hash, size, data, created_at_ms) VALUES (?1, ?2, ?3, ?4)",
            params![
                tree_hex,
                tree_bytes.len() as i64,
                tree_bytes,
                tree_created_at_ms as i64
            ],
        )
        .map_err(map_busy)?;
        tx.execute(
            "INSERT OR IGNORE INTO checkpoints (hash, parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id_hex,
                parents_json,
                effective.task_id,
                effective.agent_id,
                rationale_json,
                tree_hex,
                effective.summary,
                effective.timestamp_ms as i64
            ],
        )
        .map_err(map_busy)?;
        for name in ref_names {
            tx.execute(
                "INSERT INTO refs (name, target_hash, updated_at_ms)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(name) DO UPDATE SET target_hash = excluded.target_hash, updated_at_ms = excluded.updated_at_ms",
                params![name, id_hex, updated_at_ms as i64],
            )
            .map_err(map_busy)?;
        }
        tx.commit().map_err(map_busy)?;

        Ok(Checkpoint {
            id,
            parents: effective.parents,
            task_id: effective.task_id,
            agent_id: effective.agent_id,
            rationale: effective.rationale,
            tree_hash: effective.tree_hash,
            summary: effective.summary,
            timestamp_ms: effective.timestamp_ms,
        })
    }

    /// Retrieve a checkpoint by its content hash.
    pub fn get_checkpoint(
        &self,
        hash: &ContentHash,
    ) -> Result<Option<Checkpoint>, ContentStoreError> {
        let hash_hex = hash.to_hex();
        let row = {
            let guard = self.lock_conn()?;
            let mut stmt = guard
                .prepare(
                    "SELECT parents_json, task_id, agent_id, rationale_json, tree_hash, summary, created_at_ms
                     FROM checkpoints WHERE hash = ?1",
                )
                .map_err(map_busy)?;

            stmt.query_row(params![hash_hex], |row| {
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
            .optional()
            .map_err(map_busy)?
        };
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
        let guard = self.lock_conn()?;
        let mut stmt = guard
            .prepare("SELECT 1 FROM checkpoints WHERE hash = ?1")
            .map_err(map_busy)?;
        stmt.exists(params![hash_hex]).map_err(map_busy)
    }

    /// Update or create multiple named references atomically within a single SQLite transaction.
    pub fn update_refs_atomic(
        &mut self,
        refs: &[(&str, &ContentHash)],
        updated_at_ms: u64,
    ) -> Result<(), ContentStoreError> {
        if refs.is_empty() {
            return Ok(());
        }

        for (name, target) in refs {
            Self::validate_ref_name(name)?;
            if !self.has_checkpoint(target)? {
                return Err(ContentStoreError::MissingTarget(**target));
            }
        }

        let mut guard = self.lock_conn()?;
        let tx = guard.transaction().map_err(map_busy)?;
        for (name, target) in refs {
            let target_hex = target.to_hex();
            tx.execute(
                "INSERT INTO refs (name, target_hash, updated_at_ms)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(name) DO UPDATE SET target_hash = excluded.target_hash, updated_at_ms = excluded.updated_at_ms",
                params![name, target_hex, updated_at_ms as i64],
            )
            .map_err(map_busy)?;
        }
        tx.commit().map_err(map_busy)?;

        Ok(())
    }

    /// Update or create a named reference (e.g. "HEAD", "heads/main", "tags/v1").
    pub fn update_ref(
        &mut self,
        name: &str,
        target: &ContentHash,
        updated_at_ms: u64,
    ) -> Result<(), ContentStoreError> {
        self.update_refs_atomic(&[(name, target)], updated_at_ms)
    }

    /// Get the target checkpoint hash of a named reference.
    pub fn get_ref(&self, name: &str) -> Result<Option<ContentHash>, ContentStoreError> {
        let maybe_hex: Option<String> = {
            let guard = self.lock_conn()?;
            let mut stmt = guard
                .prepare("SELECT target_hash FROM refs WHERE name = ?1")
                .map_err(map_busy)?;
            stmt.query_row(params![name], |row| row.get::<_, String>(0))
                .optional()
                .map_err(map_busy)?
        };

        match maybe_hex {
            Some(hex) => Ok(Some(ContentHash::from_hex(&hex)?)),
            None => Ok(None),
        }
    }

    /// Delete a named reference. Returns `true` if the reference existed and was deleted.
    pub fn delete_ref(&mut self, name: &str) -> Result<bool, ContentStoreError> {
        let guard = self.lock_conn()?;
        let affected = guard
            .execute("DELETE FROM refs WHERE name = ?1", params![name])
            .map_err(map_busy)?;
        Ok(affected > 0)
    }

    /// List all named references and their target checkpoint hashes.
    pub fn list_refs(&self) -> Result<Vec<(String, ContentHash)>, ContentStoreError> {
        let pairs: Vec<(String, String)> = {
            let guard = self.lock_conn()?;
            let mut stmt = guard
                .prepare("SELECT name, target_hash FROM refs ORDER BY name ASC")
                .map_err(map_busy)?;
            let rows = stmt
                .query_map([], |row| {
                    let name: String = row.get::<_, String>(0)?;
                    let hash_hex: String = row.get::<_, String>(1)?;
                    Ok((name, hash_hex))
                })
                .map_err(map_busy)?;
            let mut out = Vec::new();
            for item in rows {
                out.push(item.map_err(map_busy)?);
            }
            out
        };

        let mut result = Vec::new();
        for (name, hash_hex) in pairs {
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

    pub(crate) fn validate_ref_name(name: &str) -> Result<(), ContentStoreError> {
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

#[cfg(test)]
mod durable_profile_tests {
    use super::*;

    /// `PRAGMA synchronous` and `PRAGMA foreign_keys` are per-connection
    /// state, so they must be asserted on the same connection that applied
    /// them (a fresh connection only shows build defaults).
    #[test]
    fn pragmas_apply_on_the_same_connection() {
        // A temp file (not memory): in-memory databases cannot use WAL, so
        // journal_mode would read back "memory" regardless of the request.
        let dir = std::env::temp_dir().join(format!(
            "bitty_test_profile_{}_{}",
            std::process::id(),
            "same-conn"
        ));
        let _ = std::fs::create_dir_all(&dir);
        let db_path = dir.join("profile.db");
        let _ = std::fs::remove_file(&db_path);
        let conn = Connection::open(&db_path).expect("open");
        apply_durable_pragmas(&conn).expect("pragmas");
        let journal: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .expect("journal_mode");
        assert_eq!(journal.to_lowercase(), "wal");
        let synchronous: i64 = conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .expect("synchronous");
        assert_eq!(synchronous, 2, "synchronous must be FULL");
        let foreign_keys: i64 = conn
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .expect("foreign_keys");
        assert_eq!(foreign_keys, 1, "foreign_keys must be ON");
        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

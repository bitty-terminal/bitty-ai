//! Graph-theoretic task DAG engine with generation fencing and SQLite persistence (AI-0164).
//!
//! Provides the control plane for Wheel:
//! - [`TaskId`]: Type-safe task identifier with strict character and length validation.
//! - [`TaskStatus`]: State machine tracking task execution lifecycle.
//! - [`TaskNode`]: Immutable view of a task record in the DAG.
//! - [`TaskDraft`]: Specification for creating a new task with initial dependencies.
//! - [`TaskEngine`]: Database-backed task scheduler providing cycle detection,
//!   topological sorting, priority queueing, cascade readiness, and generation fencing.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::content_store::ContentHash;

/// Maximum allowed byte length for a task ID (128 bytes).
pub const MAX_TASK_ID_BYTES: usize = 128;

/// Maximum allowed byte length for a task title (256 bytes).
pub const MAX_TASK_TITLE_BYTES: usize = 256;

/// Maximum allowed byte length for a task description (4096 bytes).
pub const MAX_TASK_DESC_BYTES: usize = 4096;

/// Maximum allowed byte length for an agent ID (128 bytes).
pub const MAX_AGENT_ID_BYTES: usize = 128;

/// Maximum allowed byte length for a failure reason (2048 bytes).
pub const MAX_FAILURE_REASON_BYTES: usize = 2048;

/// Maximum allowed dependencies per task (128).
pub const MAX_DEPENDENCIES_PER_TASK: usize = 128;

/// Errors arising from task DAG and scheduling operations.
#[derive(Debug)]
pub enum TaskEngineError {
    /// SQLite storage error.
    Sqlite(rusqlite::Error),
    /// A referenced task was not found.
    TaskNotFound(TaskId),
    /// Task identifier string is invalid.
    InvalidTaskId(String),
    /// State machine transition is forbidden.
    InvalidStatusTransition {
        /// Task identifier.
        task_id: TaskId,
        /// Current status.
        from: TaskStatus,
        /// Target status.
        to: TaskStatus,
    },
    /// Worker operation was rejected because its generation token is obsolete.
    StaleGeneration {
        /// Task identifier.
        task_id: TaskId,
        /// Expected active generation.
        expected: u64,
        /// Generation provided by the worker.
        found: u64,
    },
    /// A cycle was detected in the task dependency graph.
    CycleDetected {
        /// Path forming the cycle.
        cycle: Vec<TaskId>,
    },
    /// A task cannot depend on itself.
    SelfDependency(TaskId),
    /// A task with this identifier already exists.
    DuplicateTask(TaskId),
    /// A dependency referenced during creation or addition does not exist.
    DependencyNotFound {
        /// Dependent task ID.
        task_id: TaskId,
        /// Missing prerequisite task ID.
        prerequisite_id: TaskId,
    },
    /// Field exceeded its strict byte size limit.
    OversizedField {
        /// Name of the field.
        field: &'static str,
        /// Actual size in bytes.
        size: usize,
        /// Maximum allowed bytes.
        max: usize,
    },
    /// A mandatory field was empty or whitespace-only.
    EmptyField(&'static str),
    /// Content hash parsing error.
    InvalidHash(String),
    /// Malformed, corrupt, or schema-incompatible database state.
    Corrupt { detail: String },
    /// Another live writer holds this database file's single-writer lock.
    WriterBusy,
}

impl fmt::Display for TaskEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(err) => write!(f, "sqlite error: {err}"),
            Self::TaskNotFound(id) => write!(f, "task '{id}' not found"),
            Self::InvalidTaskId(id) => write!(f, "invalid task id {id:?}"),
            Self::InvalidStatusTransition { task_id, from, to } => {
                write!(
                    f,
                    "invalid transition for task '{task_id}': cannot move from {from:?} to {to:?}"
                )
            }
            Self::StaleGeneration {
                task_id,
                expected,
                found,
            } => {
                write!(
                    f,
                    "stale generation for task '{task_id}': expected {expected}, found {found}"
                )
            }
            Self::CycleDetected { cycle } => {
                let cycle_str = cycle
                    .iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .join(" -> ");
                write!(f, "cycle detected in task graph: {cycle_str}")
            }
            Self::SelfDependency(id) => write!(f, "task '{id}' cannot depend on itself"),
            Self::DuplicateTask(id) => write!(f, "task '{id}' already exists"),
            Self::DependencyNotFound {
                task_id,
                prerequisite_id,
            } => {
                write!(
                    f,
                    "prerequisite task '{prerequisite_id}' required by '{task_id}' not found"
                )
            }
            Self::OversizedField { field, size, max } => {
                write!(
                    f,
                    "field '{field}' exceeds limit: {size} bytes > {max} bytes"
                )
            }
            Self::EmptyField(field) => write!(f, "field '{field}' must not be empty"),
            Self::InvalidHash(err) => write!(f, "invalid checkpoint hash: {err}"),
            Self::Corrupt { detail } => write!(f, "task engine corrupt or incompatible: {detail}"),
            Self::WriterBusy => write!(f, "another writer holds the single-writer lock"),
        }
    }
}

impl std::error::Error for TaskEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(err) => Some(err),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for TaskEngineError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Sqlite(err)
    }
}

/// Type-safe task identifier.
///
/// Must be 1 to 128 characters, containing only ASCII alphanumeric characters,
/// hyphens, underscores, dots, or forward slashes, without leading/trailing dots or slashes.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TaskId(String);

impl<'de> Deserialize<'de> for TaskId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

impl TaskId {
    /// Construct and validate a new [`TaskId`].
    pub fn new(s: impl AsRef<str>) -> Result<Self, TaskEngineError> {
        let s = s.as_ref().trim();
        if s.is_empty() {
            return Err(TaskEngineError::EmptyField("task_id"));
        }
        if s.len() > MAX_TASK_ID_BYTES {
            return Err(TaskEngineError::OversizedField {
                field: "task_id",
                size: s.len(),
                max: MAX_TASK_ID_BYTES,
            });
        }
        if s.starts_with('/') || s.starts_with('.') || s.ends_with('/') || s.ends_with('.') {
            return Err(TaskEngineError::InvalidTaskId(s.to_string()));
        }
        if s.contains("//") || s.contains("..") {
            return Err(TaskEngineError::InvalidTaskId(s.to_string()));
        }
        for b in s.bytes() {
            if !(b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'/') {
                return Err(TaskEngineError::InvalidTaskId(s.to_string()));
            }
        }
        Ok(Self(s.to_string()))
    }

    /// Access the underlying string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId({:?})", self.0)
    }
}

impl FromStr for TaskId {
    type Err = TaskEngineError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Execution status of a task in the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Task has unfinished prerequisites and is waiting.
    Pending,
    /// All prerequisites are satisfied; task is ready to be dispatched.
    Ready,
    /// A worker is actively executing the task.
    Running,
    /// One or more prerequisites failed or were cancelled; task cannot proceed.
    Blocked,
    /// Task successfully finished.
    Succeeded,
    /// Task execution failed.
    Failed,
    /// Task was explicitly aborted.
    Cancelled,
}

impl TaskStatus {
    /// Return the canonical string name for database persistence.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Check if the task is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    /// Parse a status string from database records.
    pub fn parse(s: &str) -> Result<Self, TaskEngineError> {
        match s {
            "pending" => Ok(Self::Pending),
            "ready" => Ok(Self::Ready),
            "running" => Ok(Self::Running),
            "blocked" => Ok(Self::Blocked),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(TaskEngineError::InvalidTaskId(format!(
                "unknown status: {other}"
            ))),
        }
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

fn deserialize_task_deps<'de, D>(deserializer: D) -> Result<Vec<TaskId>, D::Error>
where
    D: Deserializer<'de>,
{
    struct TaskDepsVisitor;

    impl<'de> Visitor<'de> for TaskDepsVisitor {
        type Value = Vec<TaskId>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a sequence of task IDs, an empty map, or null")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut list = Vec::new();
            while let Some(elem) = seq.next_element()? {
                list.push(elem);
            }
            Ok(list)
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let entry: Option<(String, de::IgnoredAny)> = map.next_entry()?;
            if let Some((key, _)) = entry {
                return Err(de::Error::custom(format!(
                    "expected empty map or sequence for dependencies, found map with key '{key}'"
                )));
            }
            Ok(Vec::new())
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }
    }

    deserializer.deserialize_any(TaskDepsVisitor)
}

/// Specification for creating a new task in the DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskDraft {
    /// Unique task identifier.
    pub id: TaskId,
    /// Short descriptive title.
    pub title: String,
    /// Detailed description or acceptance criteria.
    #[serde(default)]
    pub description: String,
    /// Scheduling priority (higher values are executed earlier).
    #[serde(default)]
    pub priority: i32,
    /// Initial prerequisite task IDs.
    #[serde(default, deserialize_with = "deserialize_task_deps")]
    pub dependencies: Vec<TaskId>,
}

/// An immutable record of a task stored in the DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskNode {
    /// Unique task identifier.
    pub id: TaskId,
    /// Task title.
    pub title: String,
    /// Task description.
    pub description: String,
    /// Priority level.
    pub priority: i32,
    /// Current execution status.
    pub status: TaskStatus,
    /// Currently assigned agent identity, if any.
    pub assigned_agent: Option<String>,
    /// Monotonically increasing generation number for worker fencing.
    pub generation: u64,
    /// Associated checkpoint content hash upon completion.
    pub checkpoint: Option<ContentHash>,
    /// Failure reason if status is [`TaskStatus::Failed`].
    pub failure_reason: Option<String>,
    /// Timestamp in epoch milliseconds when task was created.
    pub created_at_ms: u64,
    /// Timestamp in epoch milliseconds when task was last updated.
    pub updated_at_ms: u64,
}

/// An enriched view of a task record in the DAG including its inlined prerequisite dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskView {
    /// Inner task node metadata.
    #[serde(flatten)]
    pub task: TaskNode,
    /// Prerequisite task IDs that must succeed before this task becomes ready.
    #[serde(default, deserialize_with = "deserialize_task_deps")]
    pub dependencies: Vec<TaskId>,
}

impl TaskView {
    /// Create a new task view from a task node and its dependencies.
    #[must_use]
    pub fn new(task: TaskNode, dependencies: Vec<TaskId>) -> Self {
        Self { task, dependencies }
    }

    /// Access the underlying task node.
    #[must_use]
    pub fn task(&self) -> &TaskNode {
        &self.task
    }

    /// Access the prerequisite dependencies.
    #[must_use]
    pub fn dependencies(&self) -> &[TaskId] {
        &self.dependencies
    }

    /// Consume the view and return the task node and its dependencies.
    #[must_use]
    pub fn into_parts(self) -> (TaskNode, Vec<TaskId>) {
        (self.task, self.dependencies)
    }
}

impl std::ops::Deref for TaskView {
    type Target = TaskNode;

    fn deref(&self) -> &Self::Target {
        &self.task
    }
}

/// Persistent, transactional Task DAG engine.
///
/// Shares the Wheel single-connection handle behind `Arc<Mutex<..>>` (see
/// `crate::content_store` docs): the database file is opened exactly once
/// and the handle is split into the content store and this engine.
pub struct TaskEngine {
    conn: Arc<Mutex<Connection>>,
}

fn task_is_busy(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, _)
            if matches!(
                e.code,
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
            )
    )
}

/// Durable task schema shared by standalone and shared opens.
pub(crate) const TASK_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,
    assigned_agent TEXT,
    generation INTEGER NOT NULL,
    checkpoint TEXT,
    failure_reason TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_priority
    ON tasks(status, priority DESC, created_at_ms ASC);
CREATE TABLE IF NOT EXISTS task_dependencies (
    prerequisite_id TEXT NOT NULL,
    dependent_id TEXT NOT NULL,
    PRIMARY KEY (prerequisite_id, dependent_id),
    FOREIGN KEY (prerequisite_id) REFERENCES tasks(task_id) ON DELETE CASCADE,
    FOREIGN KEY (dependent_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_task_deps_dependent
    ON task_dependencies(dependent_id);
CREATE INDEX IF NOT EXISTS idx_task_deps_prerequisite
    ON task_dependencies(prerequisite_id);
"#;

pub(crate) fn admit_task_or_init(conn: &rusqlite::Connection) -> Result<(), TaskEngineError> {
    let mut stmt = conn
        .prepare(
            "SELECT type, name FROM sqlite_master
             WHERE name COLLATE NOCASE IN ('tasks', 'task_dependencies')",
        )
        .map_err(task_map_busy)?;
    let mut matches: Vec<(String, String)> = Vec::new();
    let mut rows = stmt.query([]).map_err(task_map_busy)?;
    while let Some(row) = rows.next().map_err(task_map_busy)? {
        matches.push((
            row.get::<_, String>(0).map_err(task_map_busy)?,
            row.get::<_, String>(1).map_err(task_map_busy)?,
        ));
    }
    drop(rows);
    drop(stmt);
    let mut present = Vec::new();
    for expected in ["tasks", "task_dependencies"] {
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
            return Err(TaskEngineError::Corrupt {
                detail: format!(
                    "task name '{expected}' is occupied by {ty} '{name}', not the exact-case table"
                ),
            });
        }
    }
    if present.is_empty() {
        return Ok(());
    }
    if present.len() != 2 {
        return Err(TaskEngineError::Corrupt {
            detail: format!("partial task schema; present tables: {present:?}"),
        });
    }
    // Verify tasks columns (names/types/NOT NULL/PK position)
    let mut stmt = conn
        .prepare("PRAGMA table_info(tasks)")
        .map_err(task_map_busy)?;
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
        .map_err(task_map_busy)?;
    for row in rows {
        cols.push(row.map_err(task_map_busy)?);
    }
    drop(stmt);
    let expected: &[(&str, &str, bool, u32)] = &[
        ("task_id", "TEXT", true, 1),
        ("title", "TEXT", true, 0),
        ("description", "TEXT", true, 0),
        ("priority", "INTEGER", true, 0),
        ("status", "TEXT", true, 0),
        ("assigned_agent", "TEXT", false, 0),
        ("generation", "INTEGER", true, 0),
        ("checkpoint", "TEXT", false, 0),
        ("failure_reason", "TEXT", false, 0),
        ("created_at_ms", "INTEGER", true, 0),
        ("updated_at_ms", "INTEGER", true, 0),
    ];
    if cols.len() != expected.len() {
        return Err(TaskEngineError::Corrupt {
            detail: format!(
                "table tasks has {} columns, expected {}",
                cols.len(),
                expected.len()
            ),
        });
    }
    for (i, (name, ty, notnull, pk)) in expected.iter().enumerate() {
        let (aname, aty, anotnull, apk) = &cols[i];
        if aname != *name || aty != *ty || anotnull != notnull || apk != pk {
            return Err(TaskEngineError::Corrupt {
                detail: format!("table tasks column {i} shape mismatch"),
            });
        }
    }
    Ok(())
}

fn task_map_busy(err: rusqlite::Error) -> TaskEngineError {
    if task_is_busy(&err) {
        TaskEngineError::WriterBusy
    } else if let rusqlite::Error::SqliteFailure(e, _) = &err {
        if matches!(
            e.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return TaskEngineError::Corrupt {
                detail: err.to_string(),
            };
        }
        TaskEngineError::Sqlite(err)
    } else {
        TaskEngineError::Sqlite(err)
    }
}

/// Bounded SQLite header probe for standalone task opens: only the first 16
/// bytes are ever read, so the check costs O(1) memory on any file size.
fn check_task_magic(path: &Path) -> Result<(), TaskEngineError> {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(TaskEngineError::Corrupt {
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
                return Err(TaskEngineError::Corrupt {
                    detail: format!("unreadable database file: {e}"),
                });
            }
        }
    }
    if read == 0 {
        return Ok(());
    }
    if read < 16 || header != *b"SQLite format 3\0" {
        return Err(TaskEngineError::Corrupt {
            detail: "file is not a SQLite database".to_owned(),
        });
    }
    Ok(())
}

/// Durable pragma profile for task connections (WAL, FULL, FK ON, EXCLUSIVE),
/// sharing the content-store timeout by construction.
fn apply_task_pragmas(conn: &Connection) -> Result<(), TaskEngineError> {
    conn.busy_timeout(Duration::from_millis(
        crate::content_store::DURABLE_BUSY_TIMEOUT_MS,
    ))
    .map_err(task_map_busy)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = FULL;
         PRAGMA foreign_keys = ON;
         PRAGMA locking_mode = EXCLUSIVE;",
    )
    .map_err(task_map_busy)?;
    Ok(())
}

/// Claim the single-writer lock on a standalone open.
///
/// `CREATE TABLE IF NOT EXISTS` is read-only when the schema already exists,
/// so the EXCLUSIVE locking mode alone takes no lock until the first real
/// write. This zero-timeout `BEGIN EXCLUSIVE` probe takes the lock at open:
/// a second open while the first lives fails fast with
/// [`TaskEngineError::WriterBusy`]. Steady state restores the nonzero profile
/// timeout only after the claim succeeds.
fn claim_task_writer(conn: &Connection) -> Result<(), TaskEngineError> {
    conn.execute_batch("BEGIN EXCLUSIVE; COMMIT;")
        .map_err(task_map_busy)?;
    conn.busy_timeout(Duration::from_millis(
        crate::content_store::DURABLE_BUSY_TIMEOUT_MS,
    ))
    .map_err(task_map_busy)?;
    Ok(())
}

impl TaskEngine {
    fn lock_conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>, TaskEngineError> {
        self.conn.lock().map_err(|_| TaskEngineError::Corrupt {
            detail: "task engine lock poisoned; fail closed".to_owned(),
        })
    }

    /// Open or create a persistent Task DAG SQLite database at the specified path.
    ///
    /// Applies the durable pragma profile shared with
    /// `crate::content_store` (WAL, FULL, FK ON, nonzero busy timeout,
    /// EXCLUSIVE single-writer). The file header is magic-checked, the task
    /// schema is admitted (foreign or partial shapes fail closed), and the
    /// single-writer lock is claimed before returning, so a second open while
    /// the first lives fails with [`TaskEngineError::WriterBusy`]. The busy
    /// timeout stays zero through admission and init and restores the nonzero
    /// profile timeout only after the writer claim succeeds.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TaskEngineError> {
        let path = path.as_ref();
        check_task_magic(path)?;
        let conn = Connection::open(path).map_err(task_map_busy)?;
        conn.busy_timeout(Duration::ZERO).map_err(task_map_busy)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(task_map_busy)?;
        admit_task_or_init(&conn)?;
        apply_task_pragmas(&conn)?;
        conn.busy_timeout(Duration::ZERO).map_err(task_map_busy)?;
        conn.execute_batch(TASK_SCHEMA).map_err(task_map_busy)?;
        claim_task_writer(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Open a durable task engine (formal path, same profile as `Self::open`).
    pub fn open_durable(path: impl AsRef<Path>) -> Result<Self, TaskEngineError> {
        Self::open(path)
    }

    /// Open an in-memory Task DAG SQLite database for ephemeral sessions and testing.
    pub fn open_in_memory() -> Result<Self, TaskEngineError> {
        let conn = Connection::open_in_memory().map_err(task_map_busy)?;
        Self::from_connection(conn)
    }

    /// Join an already-admitted shared connection (Wheel single-open path).
    pub(crate) fn from_shared(shared: Arc<Mutex<Connection>>) -> Result<Self, TaskEngineError> {
        {
            let guard = shared.lock().map_err(|_| TaskEngineError::Corrupt {
                detail: "task engine lock poisoned; fail closed".to_owned(),
            })?;
            guard.execute_batch(TASK_SCHEMA).map_err(task_map_busy)?;
        }
        Ok(Self { conn: shared })
    }

    /// Initialize tables and indexes on an open connection.
    ///
    /// Applies the same durable pragma profile as the content store (WAL,
    /// FULL, FK ON, nonzero busy timeout, EXCLUSIVE single-writer) via
    /// [`apply_task_pragmas`]; the timeout value is
    /// `crate::content_store::DURABLE_BUSY_TIMEOUT_MS` by construction.
    fn from_connection(conn: Connection) -> Result<Self, TaskEngineError> {
        apply_task_pragmas(&conn)?;
        conn.execute_batch(TASK_SCHEMA).map_err(task_map_busy)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create and persist a new task in the DAG.
    ///
    /// Validates field bounds, uniqueness, existence of prerequisites, and absence of cycles.
    /// Automatically sets status to [`TaskStatus::Ready`] if all prerequisites are satisfied,
    /// [`TaskStatus::Blocked`] if any prerequisite failed/cancelled, or [`TaskStatus::Pending`].
    pub fn create_task(
        &mut self,
        draft: TaskDraft,
        now_ms: u64,
    ) -> Result<TaskNode, TaskEngineError> {
        let title = draft.title.trim();
        if title.is_empty() {
            return Err(TaskEngineError::EmptyField("title"));
        }
        if title.len() > MAX_TASK_TITLE_BYTES {
            return Err(TaskEngineError::OversizedField {
                field: "title",
                size: title.len(),
                max: MAX_TASK_TITLE_BYTES,
            });
        }
        if draft.description.len() > MAX_TASK_DESC_BYTES {
            return Err(TaskEngineError::OversizedField {
                field: "description",
                size: draft.description.len(),
                max: MAX_TASK_DESC_BYTES,
            });
        }
        if draft.dependencies.len() > MAX_DEPENDENCIES_PER_TASK {
            return Err(TaskEngineError::OversizedField {
                field: "dependencies",
                size: draft.dependencies.len(),
                max: MAX_DEPENDENCIES_PER_TASK,
            });
        }

        if self.has_task(&draft.id)? {
            return Err(TaskEngineError::DuplicateTask(draft.id));
        }

        let mut deduped_deps = BTreeSet::new();
        for dep in &draft.dependencies {
            if *dep == draft.id {
                return Err(TaskEngineError::SelfDependency(draft.id));
            }
            if !self.has_task(dep)? {
                return Err(TaskEngineError::DependencyNotFound {
                    task_id: draft.id.clone(),
                    prerequisite_id: dep.clone(),
                });
            }
            deduped_deps.insert(dep.clone());
        }

        // Determine initial status based on prerequisites
        let mut has_unfinished = false;
        let mut has_blocked = false;

        for dep_id in &deduped_deps {
            let dep_node = self.get_task(dep_id)?;
            match dep_node.status {
                TaskStatus::Succeeded => {}
                TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked => {
                    has_blocked = true;
                }
                TaskStatus::Pending | TaskStatus::Ready | TaskStatus::Running => {
                    has_unfinished = true;
                }
            }
        }

        let initial_status = if has_blocked {
            TaskStatus::Blocked
        } else if has_unfinished {
            TaskStatus::Pending
        } else {
            TaskStatus::Ready
        };

        // Transactional insert
        let mut guard = self.lock_conn()?;
        let tx = guard.transaction().map_err(task_map_busy)?;
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO tasks (
                    task_id, title, description, priority, status,
                    assigned_agent, generation, checkpoint, failure_reason,
                    created_at_ms, updated_at_ms
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                "#,
            )?;
            stmt.execute(params![
                draft.id.as_str(),
                title,
                draft.description.as_str(),
                draft.priority,
                initial_status.as_str(),
                Option::<String>::None,
                0i64,
                Option::<String>::None,
                Option::<String>::None,
                now_ms as i64,
                now_ms as i64,
            ])?;
        }

        {
            let mut dep_stmt = tx.prepare(
                r#"
                INSERT INTO task_dependencies (prerequisite_id, dependent_id)
                VALUES (?1, ?2)
                "#,
            )?;
            for dep in &deduped_deps {
                dep_stmt.execute(params![dep.as_str(), draft.id.as_str()])?;
            }
        }

        tx.commit().map_err(task_map_busy)?;

        Ok(TaskNode {
            id: draft.id,
            title: title.to_string(),
            description: draft.description,
            priority: draft.priority,
            status: initial_status,
            assigned_agent: None,
            generation: 0,
            checkpoint: None,
            failure_reason: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        })
    }

    /// Add a prerequisite dependency edge between two existing tasks.
    ///
    /// Validates that both tasks exist, prevents self-dependencies, and verifies that
    /// no cyclic dependency path is introduced.
    pub fn add_dependency(
        &mut self,
        prerequisite_id: &TaskId,
        dependent_id: &TaskId,
    ) -> Result<(), TaskEngineError> {
        if prerequisite_id == dependent_id {
            return Err(TaskEngineError::SelfDependency(dependent_id.clone()));
        }
        if !self.has_task(prerequisite_id)? {
            return Err(TaskEngineError::TaskNotFound(prerequisite_id.clone()));
        }
        if !self.has_task(dependent_id)? {
            return Err(TaskEngineError::TaskNotFound(dependent_id.clone()));
        }

        // Check if edge already exists
        let exists: bool = {
            let guard = self.lock_conn()?;
            guard
                .query_row(
                    r#"
            SELECT 1 FROM task_dependencies
            WHERE prerequisite_id = ?1 AND dependent_id = ?2
            "#,
                    params![prerequisite_id.as_str(), dependent_id.as_str()],
                    |_| Ok(true),
                )
                .optional()
                .map_err(task_map_busy)?
                .unwrap_or(false)
        };

        if exists {
            return Ok(());
        }

        // Forbid adding prerequisites to a task that has already started or completed
        let dep_node = self.get_task(dependent_id)?;
        if dep_node.status == TaskStatus::Running || dep_node.status.is_terminal() {
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: dependent_id.clone(),
                from: dep_node.status,
                to: TaskStatus::Pending,
            });
        }

        // Cycle check: can we reach prerequisite_id starting from dependent_id?
        if let Some(cycle) = self.find_path(dependent_id, prerequisite_id)? {
            let mut full_cycle = cycle;
            full_cycle.push(dependent_id.clone());
            return Err(TaskEngineError::CycleDetected { cycle: full_cycle });
        }

        // Insert dependency
        {
            let guard = self.lock_conn()?;
            guard
                .execute(
                    r#"
            INSERT INTO task_dependencies (prerequisite_id, dependent_id)
            VALUES (?1, ?2)
            "#,
                    params![prerequisite_id.as_str(), dependent_id.as_str()],
                )
                .map_err(task_map_busy)?;
        }

        // Re-evaluate dependent task status
        let prereq_node = self.get_task(prerequisite_id)?;

        if dep_node.status == TaskStatus::Ready {
            if matches!(
                prereq_node.status,
                TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
            ) {
                self.set_status(dependent_id, TaskStatus::Blocked, dep_node.updated_at_ms)?;
            } else if prereq_node.status != TaskStatus::Succeeded {
                self.set_status(dependent_id, TaskStatus::Pending, dep_node.updated_at_ms)?;
            }
        }

        Ok(())
    }

    /// Assign a ready task to an agent, transitioning it to [`TaskStatus::Running`].
    ///
    /// Increments the task's generation number and returns the new generation token.
    /// Fails with [`TaskEngineError::InvalidStatusTransition`] if the task is not [`TaskStatus::Ready`].
    pub fn assign_task(
        &mut self,
        task_id: &TaskId,
        agent_id: &str,
        now_ms: u64,
    ) -> Result<u64, TaskEngineError> {
        let agent_id = agent_id.trim();
        if agent_id.is_empty() {
            return Err(TaskEngineError::EmptyField("agent_id"));
        }
        if agent_id.len() > MAX_AGENT_ID_BYTES {
            return Err(TaskEngineError::OversizedField {
                field: "agent_id",
                size: agent_id.len(),
                max: MAX_AGENT_ID_BYTES,
            });
        }

        let task = self.get_task(task_id)?;
        if task.status != TaskStatus::Ready {
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: task.status,
                to: TaskStatus::Running,
            });
        }

        let new_generation = task.generation + 1;
        let rows = {
            let guard = self.lock_conn()?;
            guard
                .execute(
                    r#"
            UPDATE tasks
            SET status = ?1, assigned_agent = ?2, generation = ?3, updated_at_ms = ?4
            WHERE task_id = ?5 AND status = 'ready' AND generation = ?6
            "#,
                    params![
                        TaskStatus::Running.as_str(),
                        agent_id,
                        new_generation as i64,
                        now_ms as i64,
                        task_id.as_str(),
                        task.generation as i64,
                    ],
                )
                .map_err(task_map_busy)?
        };

        if rows == 0 {
            let current = self.get_task(task_id)?;
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: current.status,
                to: TaskStatus::Running,
            });
        }

        Ok(new_generation)
    }

    /// Complete a running task with an optional output checkpoint.
    ///
    /// Validates the worker's generation token. If generation does not match active generation,
    /// fails closed with [`TaskEngineError::StaleGeneration`].
    /// Cascades completion to all dependent tasks: any dependent whose prerequisites are all
    /// now [`TaskStatus::Succeeded`] transitions from [`TaskStatus::Pending`] or
    /// [`TaskStatus::Blocked`] to [`TaskStatus::Ready`].
    pub fn complete_task(
        &mut self,
        task_id: &TaskId,
        generation: u64,
        checkpoint: Option<ContentHash>,
        now_ms: u64,
    ) -> Result<TaskNode, TaskEngineError> {
        let task = self.get_task(task_id)?;
        if task.status != TaskStatus::Running {
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: task.status,
                to: TaskStatus::Succeeded,
            });
        }
        if task.generation != generation {
            return Err(TaskEngineError::StaleGeneration {
                task_id: task_id.clone(),
                expected: task.generation,
                found: generation,
            });
        }

        let checkpoint_str = checkpoint.as_ref().map(|h| h.to_string());

        let mut guard = self.lock_conn()?;
        let tx = guard.transaction().map_err(task_map_busy)?;
        let rows = {
            let mut stmt = tx.prepare(
                r#"
                UPDATE tasks
                SET status = ?1, checkpoint = ?2, updated_at_ms = ?3
                WHERE task_id = ?4 AND status = 'running' AND generation = ?5
                "#,
            )?;
            stmt.execute(params![
                TaskStatus::Succeeded.as_str(),
                checkpoint_str,
                now_ms as i64,
                task_id.as_str(),
                generation as i64,
            ])?
        };

        if rows == 0 {
            let current = Self::query_task_on_conn(&tx, task_id)?;
            if current.generation != generation {
                return Err(TaskEngineError::StaleGeneration {
                    task_id: task_id.clone(),
                    expected: current.generation,
                    found: generation,
                });
            }
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: current.status,
                to: TaskStatus::Succeeded,
            });
        }

        // Cascade readiness check for downstream dependents
        let dependents = Self::query_dependents_on_conn(&tx, task_id)?;
        for dep_id in dependents {
            let dep_node = Self::query_task_on_conn(&tx, &dep_id)?;
            if matches!(dep_node.status, TaskStatus::Pending | TaskStatus::Blocked) {
                let prereqs = Self::query_prerequisites_on_conn(&tx, &dep_id)?;
                let mut all_succeeded = true;
                let mut any_failed = false;

                for prereq_id in prereqs {
                    let prereq = Self::query_task_on_conn(&tx, &prereq_id)?;
                    match prereq.status {
                        TaskStatus::Succeeded => {}
                        TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked => {
                            any_failed = true;
                            all_succeeded = false;
                        }
                        _ => {
                            all_succeeded = false;
                        }
                    }
                }

                if all_succeeded {
                    let mut stmt = tx.prepare(
                        "UPDATE tasks SET status = ?1, updated_at_ms = ?2 WHERE task_id = ?3",
                    )?;
                    stmt.execute(params![
                        TaskStatus::Ready.as_str(),
                        now_ms as i64,
                        dep_id.as_str()
                    ])?;
                } else if any_failed && dep_node.status != TaskStatus::Blocked {
                    let mut stmt = tx.prepare(
                        "UPDATE tasks SET status = ?1, updated_at_ms = ?2 WHERE task_id = ?3",
                    )?;
                    stmt.execute(params![
                        TaskStatus::Blocked.as_str(),
                        now_ms as i64,
                        dep_id.as_str()
                    ])?;
                }
            }
        }

        tx.commit().map_err(task_map_busy)?;
        drop(guard);
        self.get_task(task_id)
    }

    /// Record task execution failure with a diagnostic reason.
    ///
    /// Validates generation token for running tasks. Transitively cascades [`TaskStatus::Blocked`]
    /// across the entire downstream dependency graph for tasks currently in [`TaskStatus::Pending`]
    /// or [`TaskStatus::Ready`].
    pub fn fail_task(
        &mut self,
        task_id: &TaskId,
        generation: u64,
        reason: &str,
        now_ms: u64,
    ) -> Result<TaskNode, TaskEngineError> {
        let reason = reason.trim();
        if reason.len() > MAX_FAILURE_REASON_BYTES {
            return Err(TaskEngineError::OversizedField {
                field: "failure_reason",
                size: reason.len(),
                max: MAX_FAILURE_REASON_BYTES,
            });
        }

        let task = self.get_task(task_id)?;
        if task.status != TaskStatus::Running {
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: task.status,
                to: TaskStatus::Failed,
            });
        }
        if task.generation != generation {
            return Err(TaskEngineError::StaleGeneration {
                task_id: task_id.clone(),
                expected: task.generation,
                found: generation,
            });
        }

        let mut guard = self.lock_conn()?;
        let tx = guard.transaction().map_err(task_map_busy)?;
        let rows = {
            let mut stmt = tx.prepare(
                r#"
                UPDATE tasks
                SET status = ?1, failure_reason = ?2, updated_at_ms = ?3
                WHERE task_id = ?4 AND status = 'running' AND generation = ?5
                "#,
            )?;
            stmt.execute(params![
                TaskStatus::Failed.as_str(),
                reason,
                now_ms as i64,
                task_id.as_str(),
                generation as i64,
            ])?
        };

        if rows == 0 {
            let current = Self::query_task_on_conn(&tx, task_id)?;
            if current.generation != generation {
                return Err(TaskEngineError::StaleGeneration {
                    task_id: task_id.clone(),
                    expected: current.generation,
                    found: generation,
                });
            }
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: current.status,
                to: TaskStatus::Failed,
            });
        }

        // Transitive cascade block to all downstream dependents
        Self::cascade_block_downstream_on_conn(&tx, task_id, now_ms)?;

        tx.commit().map_err(task_map_busy)?;
        drop(guard);
        self.get_task(task_id)
    }

    /// Cancel a task prior to or during execution.
    ///
    /// Bumps generation to fence off any in-flight workers. Transitively cascades [`TaskStatus::Blocked`]
    /// to all downstream dependents.
    pub fn cancel_task(
        &mut self,
        task_id: &TaskId,
        now_ms: u64,
    ) -> Result<TaskNode, TaskEngineError> {
        let task = self.get_task(task_id)?;
        if task.status.is_terminal() {
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: task.status,
                to: TaskStatus::Cancelled,
            });
        }

        let new_generation = task.generation + 1;
        let mut guard = self.lock_conn()?;
        let tx = guard.transaction().map_err(task_map_busy)?;
        let rows = {
            let mut stmt = tx.prepare(
                r#"
                UPDATE tasks
                SET status = ?1, generation = ?2, updated_at_ms = ?3
                WHERE task_id = ?4 AND status NOT IN ('succeeded', 'failed', 'cancelled')
                "#,
            )?;
            stmt.execute(params![
                TaskStatus::Cancelled.as_str(),
                new_generation as i64,
                now_ms as i64,
                task_id.as_str(),
            ])?
        };

        if rows == 0 {
            let current = Self::query_task_on_conn(&tx, task_id)?;
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: current.status,
                to: TaskStatus::Cancelled,
            });
        }

        // Transitive cascade block to all downstream dependents
        Self::cascade_block_downstream_on_conn(&tx, task_id, now_ms)?;

        tx.commit().map_err(task_map_busy)?;
        drop(guard);
        self.get_task(task_id)
    }

    /// Internal helper: recursively cascade [`TaskStatus::Blocked`] to all downstream
    /// transitive dependents using BFS within the active transaction.
    fn cascade_block_downstream_on_conn(
        tx: &rusqlite::Transaction<'_>,
        root_id: &TaskId,
        now_ms: u64,
    ) -> Result<(), TaskEngineError> {
        let mut visited: HashSet<TaskId> = HashSet::new();
        let mut queue: VecDeque<TaskId> = VecDeque::new();

        let initial_dependents = Self::query_dependents_on_conn(tx, root_id)?;
        for dep in initial_dependents {
            if visited.insert(dep.clone()) {
                queue.push_back(dep);
            }
        }

        while let Some(curr_id) = queue.pop_front() {
            let node = Self::query_task_on_conn(tx, &curr_id)?;
            if matches!(node.status, TaskStatus::Pending | TaskStatus::Ready) {
                let mut stmt = tx.prepare(
                    "UPDATE tasks SET status = ?1, updated_at_ms = ?2 WHERE task_id = ?3",
                )?;
                stmt.execute(params![
                    TaskStatus::Blocked.as_str(),
                    now_ms as i64,
                    curr_id.as_str()
                ])?;

                let next_dependents = Self::query_dependents_on_conn(tx, &curr_id)?;
                for next_dep in next_dependents {
                    if visited.insert(next_dep.clone()) {
                        queue.push_back(next_dep);
                    }
                }
            }
        }

        Ok(())
    }

    /// Retry a failed or cancelled task.
    ///
    /// Bumps generation, clears failure diagnostics, and re-evaluates dependency readiness.
    pub fn retry_task(
        &mut self,
        task_id: &TaskId,
        now_ms: u64,
    ) -> Result<TaskNode, TaskEngineError> {
        let task = self.get_task(task_id)?;
        if !matches!(task.status, TaskStatus::Failed | TaskStatus::Cancelled) {
            return Err(TaskEngineError::InvalidStatusTransition {
                task_id: task_id.clone(),
                from: task.status,
                to: TaskStatus::Pending,
            });
        }

        let prereqs = self.get_dependencies(task_id)?;
        let mut all_succeeded = true;
        let mut any_blocked = false;

        for prereq_id in &prereqs {
            let prereq = self.get_task(prereq_id)?;
            match prereq.status {
                TaskStatus::Succeeded => {}
                TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked => {
                    any_blocked = true;
                    all_succeeded = false;
                }
                _ => {
                    all_succeeded = false;
                }
            }
        }

        let next_status = if any_blocked {
            TaskStatus::Blocked
        } else if all_succeeded {
            TaskStatus::Ready
        } else {
            TaskStatus::Pending
        };

        let new_generation = task.generation + 1;
        {
            let guard = self.lock_conn()?;
            guard
                .execute(
                    r#"
            UPDATE tasks
            SET status = ?1, generation = ?2, failure_reason = NULL,
                assigned_agent = NULL, checkpoint = NULL, updated_at_ms = ?3
            WHERE task_id = ?4
            "#,
                    params![
                        next_status.as_str(),
                        new_generation as i64,
                        now_ms as i64,
                        task_id.as_str()
                    ],
                )
                .map_err(task_map_busy)?;
        }

        self.get_task(task_id)
    }

    /// Retrieve the highest priority [`TaskStatus::Ready`] task from the queue.
    pub fn next_ready_task(&self) -> Result<Option<TaskNode>, TaskEngineError> {
        let guard = self.lock_conn()?;
        let result = guard
            .query_row(
                r#"
            SELECT task_id, title, description, priority, status,
                   assigned_agent, generation, checkpoint, failure_reason,
                   created_at_ms, updated_at_ms
            FROM tasks
            WHERE status = 'ready'
            ORDER BY priority DESC, created_at_ms ASC, task_id ASC
            LIMIT 1
            "#,
                [],
                Self::row_to_task_node,
            )
            .optional()
            .map_err(task_map_busy)?;

        Ok(result)
    }

    /// Retrieve all tasks currently in [`TaskStatus::Ready`] state, sorted by priority descending.
    pub fn ready_tasks(&self) -> Result<Vec<TaskNode>, TaskEngineError> {
        let guard = self.lock_conn()?;
        let mut stmt = guard
            .prepare(
                r#"
            SELECT task_id, title, description, priority, status,
                   assigned_agent, generation, checkpoint, failure_reason,
                   created_at_ms, updated_at_ms
            FROM tasks
            WHERE status = 'ready'
            ORDER BY priority DESC, created_at_ms ASC, task_id ASC
            "#,
            )
            .map_err(task_map_busy)?;
        let rows = stmt
            .query_map([], Self::row_to_task_node)
            .map_err(task_map_busy)?;
        let mut tasks = Vec::new();
        for r in rows {
            tasks.push(r.map_err(task_map_busy)?);
        }
        Ok(tasks)
    }

    /// Retrieve a task record by its identifier.
    pub fn get_task(&self, id: &TaskId) -> Result<TaskNode, TaskEngineError> {
        let guard = self.lock_conn()?;
        let task = guard
            .query_row(
                r#"
            SELECT task_id, title, description, priority, status,
                   assigned_agent, generation, checkpoint, failure_reason,
                   created_at_ms, updated_at_ms
            FROM tasks
            WHERE task_id = ?1
            "#,
                params![id.as_str()],
                Self::row_to_task_node,
            )
            .optional()
            .map_err(task_map_busy)?;

        task.ok_or_else(|| TaskEngineError::TaskNotFound(id.clone()))
    }

    /// Check if a task exists.
    pub fn has_task(&self, id: &TaskId) -> Result<bool, TaskEngineError> {
        let guard = self.lock_conn()?;
        let exists: bool = guard
            .query_row(
                "SELECT 1 FROM tasks WHERE task_id = ?1",
                params![id.as_str()],
                |_| Ok(true),
            )
            .optional()
            .map_err(task_map_busy)?
            .unwrap_or(false);

        Ok(exists)
    }

    /// List all tasks in the store.
    pub fn list_tasks(&self) -> Result<Vec<TaskNode>, TaskEngineError> {
        let guard = self.lock_conn()?;
        let mut stmt = guard
            .prepare(
                r#"
            SELECT task_id, title, description, priority, status,
                   assigned_agent, generation, checkpoint, failure_reason,
                   created_at_ms, updated_at_ms
            FROM tasks
            ORDER BY created_at_ms ASC, task_id ASC
            "#,
            )
            .map_err(task_map_busy)?;
        let rows = stmt
            .query_map([], Self::row_to_task_node)
            .map_err(task_map_busy)?;
        let mut tasks = Vec::new();
        for r in rows {
            tasks.push(r.map_err(task_map_busy)?);
        }
        Ok(tasks)
    }

    /// Retrieve an enriched view of a task including its inlined prerequisite dependencies.
    pub fn get_task_view(&self, id: &TaskId) -> Result<TaskView, TaskEngineError> {
        let task = self.get_task(id)?;
        let guard = self.lock_conn()?;
        let dependencies = Self::query_prerequisites_on_conn(&guard, id)?;
        Ok(TaskView::new(task, dependencies))
    }

    /// List all tasks in the store with their inlined prerequisite dependencies.
    ///
    /// Executes in O(|V| + |E|) time using a single query on tasks and a single query
    /// on dependencies, avoiding N+1 round trips.
    pub fn list_task_views(&self) -> Result<Vec<TaskView>, TaskEngineError> {
        let tasks = self.list_tasks()?;
        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        let mut deps_map: HashMap<TaskId, Vec<TaskId>> = HashMap::with_capacity(tasks.len());
        {
            let guard = self.lock_conn()?;
            let mut stmt = guard
                .prepare(
                    "SELECT prerequisite_id, dependent_id FROM task_dependencies ORDER BY prerequisite_id ASC",
                )
                .map_err(task_map_busy)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(task_map_busy)?;

            for r in rows {
                let (prereq_str, dep_str) = r.map_err(task_map_busy)?;
                if let (Ok(prereq), Ok(dep)) = (TaskId::new(&prereq_str), TaskId::new(&dep_str)) {
                    deps_map.entry(dep).or_default().push(prereq);
                }
            }
        }

        let views = tasks
            .into_iter()
            .map(|task| {
                let dependencies = deps_map.remove(&task.id).unwrap_or_default();
                TaskView::new(task, dependencies)
            })
            .collect();

        Ok(views)
    }

    /// Retrieve direct prerequisite task IDs for a task.
    pub fn get_dependencies(&self, id: &TaskId) -> Result<Vec<TaskId>, TaskEngineError> {
        if !self.has_task(id)? {
            return Err(TaskEngineError::TaskNotFound(id.clone()));
        }
        let guard = self.lock_conn()?;
        Self::query_prerequisites_on_conn(&guard, id)
    }

    /// Retrieve direct downstream dependent task IDs for a task.
    pub fn get_dependents(&self, id: &TaskId) -> Result<Vec<TaskId>, TaskEngineError> {
        if !self.has_task(id)? {
            return Err(TaskEngineError::TaskNotFound(id.clone()));
        }
        let guard = self.lock_conn()?;
        Self::query_dependents_on_conn(&guard, id)
    }

    /// Compute a topological sort of all tasks in the DAG.
    ///
    /// Respects dependency ordering, with independent tasks prioritized by
    /// [`TaskNode::priority`] descending and creation order ascending.
    pub fn topological_sort(&self) -> Result<Vec<TaskId>, TaskEngineError> {
        let all_tasks = self.list_tasks()?;
        if all_tasks.is_empty() {
            return Ok(Vec::new());
        }

        let mut in_degrees: HashMap<TaskId, usize> = HashMap::new();
        let mut adj_list: HashMap<TaskId, Vec<TaskId>> = HashMap::new();
        let mut task_priorities: HashMap<TaskId, (i32, u64)> = HashMap::new();

        for t in &all_tasks {
            in_degrees.insert(t.id.clone(), 0);
            adj_list.insert(t.id.clone(), Vec::new());
            task_priorities.insert(t.id.clone(), (t.priority, t.created_at_ms));
        }

        let guard = self.lock_conn()?;
        let mut stmt = guard
            .prepare("SELECT prerequisite_id, dependent_id FROM task_dependencies")
            .map_err(task_map_busy)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(task_map_busy)?;

        for r in rows {
            let (prereq_str, dep_str) = r.map_err(task_map_busy)?;
            if let (Ok(prereq), Ok(dep)) = (TaskId::new(&prereq_str), TaskId::new(&dep_str)) {
                if let Some(entry) = adj_list.get_mut(&prereq) {
                    entry.push(dep.clone());
                }
                if let Some(deg) = in_degrees.get_mut(&dep) {
                    *deg += 1;
                }
            }
        }

        // Priority queue of nodes with in_degree == 0
        // Sort key: (priority DESC, created_at_ms ASC, task_id ASC)
        let mut ready: BTreeSet<(i32, u64, TaskId)> = BTreeSet::new();
        for (id, &deg) in &in_degrees {
            if deg == 0 {
                let &(prio, created) = &task_priorities[id];
                ready.insert((-prio, created, id.clone()));
            }
        }

        let mut sorted = Vec::with_capacity(all_tasks.len());

        while let Some(item) = ready.iter().next().cloned() {
            ready.remove(&item);
            let (_, _, u) = item;
            sorted.push(u.clone());

            if let Some(neighbors) = adj_list.get(&u) {
                for v in neighbors {
                    if let Some(deg) = in_degrees.get_mut(v) {
                        *deg -= 1;
                        if *deg == 0 {
                            let &(prio, created) = &task_priorities[v];
                            ready.insert((-prio, created, v.clone()));
                        }
                    }
                }
            }
        }

        if sorted.len() != all_tasks.len() {
            return Err(TaskEngineError::CycleDetected { cycle: Vec::new() });
        }

        Ok(sorted)
    }

    /// Internal helper: Find a directed path from `from` to `to` using BFS.
    fn find_path(
        &self,
        from: &TaskId,
        to: &TaskId,
    ) -> Result<Option<Vec<TaskId>>, TaskEngineError> {
        let mut visited: HashSet<TaskId> = HashSet::new();
        let mut parent: HashMap<TaskId, TaskId> = HashMap::new();
        let mut queue: VecDeque<TaskId> = VecDeque::new();

        visited.insert(from.clone());
        queue.push_back(from.clone());

        let guard = self.lock_conn()?;
        let mut stmt = guard
            .prepare("SELECT dependent_id FROM task_dependencies WHERE prerequisite_id = ?1")
            .map_err(task_map_busy)?;

        while let Some(current) = queue.pop_front() {
            if current == *to {
                // Reconstruct path
                let mut path = Vec::new();
                let mut curr = to.clone();
                while curr != *from {
                    path.push(curr.clone());
                    curr = parent.get(&curr).cloned().expect("parent must exist");
                }
                path.push(from.clone());
                path.reverse();
                return Ok(Some(path));
            }

            let rows = stmt.query_map(params![current.as_str()], |row| row.get::<_, String>(0))?;

            for r in rows {
                let dep_id = TaskId::new(&r?)?;
                if !visited.contains(&dep_id) {
                    visited.insert(dep_id.clone());
                    parent.insert(dep_id.clone(), current.clone());
                    queue.push_back(dep_id);
                }
            }
        }

        Ok(None)
    }

    /// Internal helper: update task status.
    fn set_status(
        &mut self,
        id: &TaskId,
        status: TaskStatus,
        now_ms: u64,
    ) -> Result<(), TaskEngineError> {
        {
            let guard = self.lock_conn()?;
            guard
                .execute(
                    "UPDATE tasks SET status = ?1, updated_at_ms = ?2 WHERE task_id = ?3",
                    params![status.as_str(), now_ms as i64, id.as_str()],
                )
                .map_err(task_map_busy)?;
        }
        Ok(())
    }

    /// Internal helper: convert SQLite row to [`TaskNode`].
    fn row_to_task_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<TaskNode> {
        let id_str: String = row.get(0)?;
        let title: String = row.get(1)?;
        let description: String = row.get(2)?;
        let priority: i32 = row.get(3)?;
        let status_str: String = row.get(4)?;
        let assigned_agent: Option<String> = row.get(5)?;
        let generation: i64 = row.get(6)?;
        let checkpoint_str: Option<String> = row.get(7)?;
        let failure_reason: Option<String> = row.get(8)?;
        let created_at_ms: i64 = row.get(9)?;
        let updated_at_ms: i64 = row.get(10)?;

        let id = TaskId::new(&id_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let status = TaskStatus::parse(&status_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let checkpoint = match checkpoint_str {
            Some(hex) => Some(ContentHash::from_str(&hex).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    7,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?),
            None => None,
        };

        Ok(TaskNode {
            id,
            title,
            description,
            priority,
            status,
            assigned_agent,
            generation: generation as u64,
            checkpoint,
            failure_reason,
            created_at_ms: created_at_ms as u64,
            updated_at_ms: updated_at_ms as u64,
        })
    }

    /// Internal helper: query prerequisites on an active connection or transaction.
    fn query_prerequisites_on_conn(
        conn: &Connection,
        dependent_id: &TaskId,
    ) -> Result<Vec<TaskId>, TaskEngineError> {
        let mut stmt = conn.prepare(
            r#"
            SELECT prerequisite_id FROM task_dependencies
            WHERE dependent_id = ?1
            ORDER BY prerequisite_id ASC
            "#,
        )?;
        let rows = stmt.query_map(params![dependent_id.as_str()], |row| {
            row.get::<_, String>(0)
        })?;
        let mut res = Vec::new();
        for r in rows {
            res.push(TaskId::new(&r?)?);
        }
        Ok(res)
    }

    /// Internal helper: query dependents on an active connection or transaction.
    fn query_dependents_on_conn(
        conn: &Connection,
        prerequisite_id: &TaskId,
    ) -> Result<Vec<TaskId>, TaskEngineError> {
        let mut stmt = conn.prepare(
            r#"
            SELECT dependent_id FROM task_dependencies
            WHERE prerequisite_id = ?1
            ORDER BY dependent_id ASC
            "#,
        )?;
        let rows = stmt.query_map(params![prerequisite_id.as_str()], |row| {
            row.get::<_, String>(0)
        })?;
        let mut res = Vec::new();
        for r in rows {
            res.push(TaskId::new(&r?)?);
        }
        Ok(res)
    }

    /// Internal helper: query a task on an active connection or transaction.
    fn query_task_on_conn(conn: &Connection, id: &TaskId) -> Result<TaskNode, TaskEngineError> {
        let task = conn
            .query_row(
                r#"
            SELECT task_id, title, description, priority, status,
                   assigned_agent, generation, checkpoint, failure_reason,
                   created_at_ms, updated_at_ms
            FROM tasks
            WHERE task_id = ?1
            "#,
                params![id.as_str()],
                Self::row_to_task_node,
            )
            .optional()?;

        task.ok_or_else(|| TaskEngineError::TaskNotFound(id.clone()))
    }
}

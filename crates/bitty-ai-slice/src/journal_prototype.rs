//! Experimental R6 single-writer journal prototype (AI-0049, slice-side only).
//!
//! **Prototype, not a product path.** This module is implementation evidence
//! for the draft R6 persistence profile
//! (`docs/specifications/persistence-profile-r6.md`): one append-ordered
//! journal over a transactional store with exactly one writer, where logical
//! append order never overrides deletion. It is re-exported by nothing,
//! wired into no production path, and decides none of AIQ-51 through AIQ-5C:
//! those register entries stay open, and this module is evidence, never a
//! decision.
//!
//! ## Why a plain module instead of a Cargo feature
//!
//! The std-only invariant belongs to `bitty-ai-runtime`, which carries no
//! dependencies and is untouched here. `bitty-ai-slice` is already the
//! workspace's dependency-bearing experiment crate (`bitty-ipc` through a
//! pinned revision), so its default build is not std-only to begin with. A
//! non-default feature would hide this module from the mandatory
//! `cargo test --workspace --all-targets` gate (silent rot), while a
//! default-on feature would add `cfg` complexity without adding isolation,
//! because the dependency stays compiled either way. The module is therefore
//! a plain, clearly documented experiment.
//!
//! ## Storage profile (prototype)
//!
//! - One SQLite database file at a caller-supplied path. The schema is
//!   created only for a store with no prototype tables; a store that already
//!   carries prototype tables is admitted only as a complete, current
//!   prototype journal (see "Admission policy and migrations"). Journal mode
//!   is SQLite's default rollback journal; WAL selection stays open under
//!   AIQ-53.
//! - Append order: `journal_records.seq` is a strictly increasing
//!   `INTEGER PRIMARY KEY AUTOINCREMENT`; reads return surviving records in
//!   ascending `seq` order.
//! - Deletion: tombstones live in `journal_tombstones`, keyed by record id;
//!   reads exclude tombstoned ids. Record rows and payload bytes are never
//!   rewritten or removed, so history is preserved and tombstoning is
//!   idempotent.
//! - Single writer: the opening connection claims SQLite's write lock in
//!   exclusive locking mode and holds it for its lifetime; a second
//!   [`Journal::open`] on the same file fails with
//!   [`JournalError::WriterBusy`].
//! - No wall clock: every persisted timestamp is caller-supplied
//!   (`recorded_at_ms`, `tombstoned_at_ms`).
//!
//! ## Bounds (prototype policy)
//!
//! - [`MAX_ID_BYTES`], [`MAX_KIND_BYTES`], and [`MAX_PAYLOAD_BYTES`] bound
//!   each appended field; out-of-bound values fail closed before any write.
//! - FTS5 is **excluded**: no search index is created and no FTS5 syntax is
//!   used (the bundled SQLite compiles FTS5 in; this module never touches
//!   it). Retention, projection, replay, export, backup, and compaction
//!   policies do not exist here.
//!
//! ## Admission policy and migrations
//!
//! [`Journal::open`] distinguishes four store states before it writes
//! anything:
//!
//! 1. **New store.** The database file is missing or already exists as a
//!    SQLite database with none of the three prototype names occupied. The
//!    prototype schema is created, the profile marker is written, and the
//!    store is admitted. This is the only implicit state change.
//! 2. **Existing complete journal.** All three prototype tables are present.
//!    Each table must match the expected column shape *including* declared
//!    column types, `NOT NULL` flags, primary-key position and table-level
//!    keys (`journal_records.seq` must be a true `INTEGER PRIMARY KEY` rowid
//!    alias, plus a non-partial unique index on exactly
//!    `journal_records.id`; single-column primary keys on
//!    `journal_meta.key` and `journal_tombstones.id`), and
//!    `journal_meta.profile` must read exactly `r6-prototype`. A partial
//!    unique index or a composite key such as `UNIQUE (id, kind)` is refused:
//!    both let duplicate ids coexist. A descending alias
//!    (`INTEGER PRIMARY KEY DESC`) or any other non-alias primary-key form
//!    is refused: `seq` would stay NULL while `last_insert_rowid()` returned
//!    a value, so the append-order contract would break after mutation. Only
//!    then is the store admitted;
//!    missing tables are never grafted onto a partial journal.
//! 3. **Occupied prototype name.** A case-variant table, a view, or any other
//!    object occupying a prototype name. Refused before any schema statement.
//!    SQLite matches object names case-insensitively, so
//!    `CREATE TABLE IF NOT EXISTS` would otherwise silently no-op under the
//!    foreign object while later operations run against it, or fail with a
//!    name-collision error; neither outcome is admission.
//! 4. **Anything else.** A foreign database, a partial prototype store, a
//!    schema with altered types/constraints, or a missing or unrecognized
//!    profile fails closed with [`JournalError::Corrupt`] (or
//!    [`JournalError::UnsupportedSchema`] when the store is a complete
//!    prototype journal under a different, identified profile key) and is
//!    left byte-for-byte unchanged.
//!
//!    A foreign table that does not exist under the exact case-sensitive name
//!    is never a prototype table: SQLite matches table names
//!    case-insensitively, so a table named `Journal_Records` (or any other
//!    case variant) must not be adopted, shadowed, or used to satisfy schema
//!    creation. Prototype-name occupancy is decided by one case-insensitive
//!    scan of `sqlite_master` plus an exact (case-sensitive) name comparison,
//!    and the name an object actually carries is what refusals report.
//!
//! The `AUTOINCREMENT` property of `journal_records.seq` is not part of the
//! admission predicate: `PRAGMA table_info` exposes neither it nor the
//! `sqlite_sequence` bookkeeping it requires, and the prototype does not
//! audit declared `CREATE TABLE` bodies beyond the properties listed above.
//! A hand-written schema that drops `AUTOINCREMENT` is therefore admitted;
//! the observable append-order contract still holds under the checked
//! `INTEGER PRIMARY KEY` + `last_insert_rowid()` path. A future profile that
//! must reject it needs a declared schema identity, not a name comparison.
//!
//! Admission checks the three prototype tables only. A database carrying
//! unrelated tables *in addition* to a complete prototype journal is
//! admitted; the prototype does not claim exclusive ownership of the file.
//! "Prototype table" means the exact-case name; a case variant is a foreign
//! object under a prototype name and is refused (see above).
//!
//! Partial-store policy (explicit): a store that carries *some* prototype
//! tables but not all of them is refused. The prototype has no repair path by
//! design; a partially created schema is indistinguishable from an
//! interrupted or foreign store, so admitting it would let the prototype run
//! against constraints it cannot verify.
//!
//! Migrations (explicit): there are none in the prototype. A future schema
//! change must be expressed as a new profile key plus a declared forward
//! migration; `CREATE TABLE IF NOT EXISTS` and column renames are not
//! migrations. Until such a migration exists, a recognized older profile is
//! refused unmodified. The current store is therefore admitted at exactly one
//! schema identity, and unknown columns in a prototype table are refused
//! rather than ignored because append order and duplicate-id protection
//! depend on the exact constraints.
//!
//! ## Fail-closed rules
//!
//! - Malformed, corrupt, or schema-incompatible database state yields a
//!   typed [`JournalError::Corrupt`] (or
//!   [`JournalError::UnsupportedSchema`] for an identified foreign profile).
//!   The prototype never deletes, truncates, recreates, or silently resets an
//!   existing database file and never repairs state implicitly. Schema
//!   verification is atomic: no schema statement runs unless every
//!   pre-existing prototype table is compatible and the profile matches.
//! - All failures are typed: no `panic!`, `unwrap`, or `expect` outside
//!   tests, and no `unsafe` (the crate denies it).
//! - No network, no background work, and no process spawning at runtime; the
//!   only I/O is the caller-supplied database file and SQLite's own sidecar
//!   files in that directory.
//!
//! ## Residual risks (recorded, not mitigated here)
//!
//! - SQLite detects corruption structurally; a corruption it silently
//!   tolerates would not surface as [`JournalError::Corrupt`]. A durable
//!   profile needs an explicit integrity policy (AIQ-51/AIQ-55 scope).
//! - Crash recovery between an append and its acknowledgement is SQLite's
//!   rollback-journal behavior, not a reconciliation protocol; unknown-effect
//!   handling stays with AIQ-59 and is untested here.

use std::fmt;
use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};

/// Prototype bound for record ids (128 bytes, mirroring the consent-scope
/// bound used elsewhere in the slice).
pub const MAX_ID_BYTES: usize = 128;

/// Prototype bound for record kinds (64 bytes).
pub const MAX_KIND_BYTES: usize = 64;

/// Prototype bound for record payloads (8 KiB).
pub const MAX_PAYLOAD_BYTES: usize = 8 * 1024;

/// Current profile key stored in `journal_meta.profile` and checked on
/// admission. A different value identifies a different schema identity; the
/// prototype performs no migration between identities.
const PROFILE: &str = "r6-prototype";

/// Prototype schema, applied only to a new empty store. `CREATE TABLE IF NOT
/// EXISTS` never repairs constraints on an existing table, which is why
/// existing stores are verified against [`REQUIRED_TABLES`] before this runs.
/// It also never runs while any object occupies a prototype name (see
/// [`classify_store`]): SQLite resolves names case-insensitively, so a foreign
/// object under any case variant of a prototype name would otherwise make
/// these statements no-op behind the object or fail on the name.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS journal_records (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    id             TEXT NOT NULL UNIQUE,
    kind           TEXT NOT NULL,
    payload        TEXT NOT NULL,
    recorded_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS journal_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal_tombstones (
    id               TEXT PRIMARY KEY,
    tombstoned_at_ms INTEGER NOT NULL
);
";

/// One expected column: name, declared type, and `NOT NULL` flag, compared
/// against `PRAGMA table_info` (which reports declared types, not runtime
/// affinity).
struct ExpectedColumn {
    name: &'static str,
    declared_type: &'static str,
    not_null: bool,
}

/// Expected per-table shape: columns in declaration order plus the primary-key
/// column position `PRAGMA table_info` reports (`1`-based, `0` when absent).
struct ExpectedTable {
    table: &'static str,
    columns: &'static [ExpectedColumn],
    primary_key: u32,
    /// The `journal_records.id` unique index must be present; it is the
    /// duplicate-id protection the append path depends on. The index must be
    /// unique, non-partial, and cover exactly `("id")`.
    unique_id_index: bool,
    /// The primary key must be a true `INTEGER PRIMARY KEY` rowid alias.
    /// `PRAGMA table_info` reports identical `type`, `notnull`, and `pk`
    /// values for `INTEGER PRIMARY KEY DESC`, a table-level
    /// `PRIMARY KEY (seq)`, a non-`INTEGER` key, and `WITHOUT ROWID`, none of
    /// which is a rowid alias; each of those materializes an index with
    /// `origin = 'pk'` instead, so requiring no such index proves the alias.
    rowid_alias: bool,
}

const fn column(name: &'static str, declared_type: &'static str, not_null: bool) -> ExpectedColumn {
    ExpectedColumn {
        name,
        declared_type,
        not_null,
    }
}

/// Required table shapes, verified before any schema statement runs so a
/// foreign or incompatible database is rejected without mutation. Types,
/// `NOT NULL`, and primary-key shape are checked; `journal_meta` carries no
/// unique-id index, but its `key` primary key provides the id uniqueness the
/// marker write depends on.
const REQUIRED_TABLES: [ExpectedTable; 3] = [
    ExpectedTable {
        table: "journal_records",
        columns: &[
            column("seq", "INTEGER", false),
            column("id", "TEXT", true),
            column("kind", "TEXT", true),
            column("payload", "TEXT", true),
            column("recorded_at_ms", "INTEGER", true),
        ],
        primary_key: 1,
        unique_id_index: true,
        rowid_alias: true,
    },
    ExpectedTable {
        table: "journal_meta",
        columns: &[column("key", "TEXT", false), column("value", "TEXT", true)],
        primary_key: 1,
        unique_id_index: false,
        rowid_alias: false,
    },
    ExpectedTable {
        table: "journal_tombstones",
        columns: &[
            column("id", "TEXT", false),
            column("tombstoned_at_ms", "INTEGER", true),
        ],
        primary_key: 1,
        unique_id_index: false,
        rowid_alias: false,
    },
];

/// One journal record as read back from the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalRecord {
    /// Strictly increasing append position assigned by the store.
    pub seq: i64,
    /// Caller-supplied record id, unique per journal.
    pub id: String,
    /// Caller-supplied record kind (a bounded label, not a schema).
    pub kind: String,
    /// Caller-supplied payload.
    pub payload: String,
    /// Caller-supplied timestamp in milliseconds; no clock is ever read.
    pub recorded_at_ms: i64,
}

/// An open single-writer journal.
///
/// At most one `Journal` may hold a given database file at a time; see
/// [`Journal::open`]. Every method is deterministic: no wall clock, threads,
/// network, or background work is involved.
#[derive(Debug)]
pub struct Journal {
    conn: Connection,
}

/// Typed fail-closed journal failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalError {
    /// A caller-supplied field was empty.
    EmptyField { field: &'static str },
    /// A caller-supplied field exceeded its prototype bound.
    FieldTooLong {
        field: &'static str,
        limit: usize,
        actual: usize,
    },
    /// The journal already holds a record with this id.
    DuplicateId { id: String },
    /// Another live writer holds this database file's single-writer lock.
    WriterBusy,
    /// The database file could not be opened as supplied.
    Open { detail: String },
    /// Malformed, corrupt, or schema-incompatible database state. Fail
    /// closed: the file is never reset, truncated, or repaired implicitly.
    Corrupt { detail: String },
    /// The store is a complete, externally written prototype journal under a
    /// different, identified profile key. No migration exists, so the store
    /// is refused unchanged rather than upgraded or reset.
    UnsupportedSchema { profile: String },
    /// Any other storage failure (reported, never retried silently).
    Storage { detail: String },
}

impl fmt::Display for JournalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => {
                write!(f, "field '{field}' must not be empty")
            }
            Self::FieldTooLong {
                field,
                limit,
                actual,
            } => write!(
                f,
                "field '{field}' exceeds its {limit}-byte bound: {actual} bytes"
            ),
            Self::DuplicateId { id } => {
                write!(f, "record id '{id}' already exists")
            }
            Self::WriterBusy => {
                write!(f, "another writer holds the single-writer lock")
            }
            Self::Open { detail } => {
                write!(f, "journal database could not be opened: {detail}")
            }
            Self::Corrupt { detail } => {
                write!(f, "journal database is malformed or incompatible: {detail}")
            }
            Self::UnsupportedSchema { profile } => {
                write!(
                    f,
                    "journal profile '{profile}' is not supported and no migration exists"
                )
            }
            Self::Storage { detail } => {
                write!(f, "journal storage failure: {detail}")
            }
        }
    }
}

impl std::error::Error for JournalError {}

impl Journal {
    /// Open (or create) the journal database at `path` and claim the
    /// single-writer lock for the returned handle's lifetime.
    ///
    /// Admission is decided before any write; see the module documentation
    /// for the full policy and the migration stance. In short:
    ///
    /// - A missing file or a SQLite database with no prototype objects is a
    ///   new empty store: the prototype schema is created and the profile is
    ///   marked.
    /// - An existing store is admitted only when every prototype table is
    ///   present under its exact-case name and matches the expected column
    ///   types, `NOT NULL` flags, primary-key shape, non-partial unique index
    ///   over exactly `("id")`, and the stored profile is exactly
    ///   `r6-prototype`. A case-variant table, a view, or any other object
    ///   occupying a prototype name is refused.
    /// - A foreign database, a partial prototype store, a store with altered
    ///   constraints, or an occupied prototype name fails closed with
    ///   [`JournalError::Corrupt`] and is left unchanged; a complete prototype
    ///   journal under a different, identified profile key fails with
    ///   [`JournalError::UnsupportedSchema`] (no migration exists).
    /// - A second open while another [`Journal`] holds the file fails with
    ///   [`JournalError::WriterBusy`].
    pub fn open(path: &Path) -> Result<Self, JournalError> {
        let conn = Connection::open(path).map_err(|err| JournalError::Open {
            detail: err.to_string(),
        })?;
        conn.busy_timeout(Duration::ZERO).map_err(map_sqlite)?;

        let mode: String = conn
            .pragma_update_and_check(None, "locking_mode", "EXCLUSIVE", |row| row.get(0))
            .map_err(map_sqlite)?;
        if !mode.eq_ignore_ascii_case("exclusive") {
            return Err(JournalError::Storage {
                detail: format!("locking mode not applied: {mode}"),
            });
        }

        admit_or_initialize(&conn)?;

        // Reaching here is always a write: schema creation for a new store or
        // the idempotent profile marker for an admitted one. A real page
        // write claims the exclusive lock; `INSERT OR REPLACE` always
        // performs a delete and an insert, so the write happens even when the
        // marker row is already present.
        conn.execute(
            "INSERT OR REPLACE INTO journal_meta (key, value) VALUES ('profile', ?1)",
            [PROFILE],
        )
        .map_err(map_sqlite)?;

        Ok(Self { conn })
    }

    /// Append one bounded record and return it with its assigned `seq`.
    ///
    /// `recorded_at_ms` is stored verbatim (no clock is read). Appending a
    /// duplicate id fails with [`JournalError::DuplicateId`] and leaves the
    /// existing record untouched.
    pub fn append(
        &self,
        id: &str,
        kind: &str,
        payload: &str,
        recorded_at_ms: i64,
    ) -> Result<JournalRecord, JournalError> {
        check_field("id", id, MAX_ID_BYTES)?;
        check_field("kind", kind, MAX_KIND_BYTES)?;
        check_field("payload", payload, MAX_PAYLOAD_BYTES)?;

        match self.conn.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, kind, payload, recorded_at_ms],
        ) {
            Ok(_) => Ok(JournalRecord {
                seq: self.conn.last_insert_rowid(),
                id: id.to_owned(),
                kind: kind.to_owned(),
                payload: payload.to_owned(),
                recorded_at_ms,
            }),
            Err(err) if is_unique_violation(&err) => {
                Err(JournalError::DuplicateId { id: id.to_owned() })
            }
            Err(err) => Err(map_sqlite(err)),
        }
    }

    /// Read all surviving records in append order.
    ///
    /// Tombstoned ids are excluded from the result; their rows and payloads
    /// remain in the store as history.
    pub fn read_all(&self) -> Result<Vec<JournalRecord>, JournalError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT seq, id, kind, payload, recorded_at_ms
                 FROM journal_records AS r
                 WHERE NOT EXISTS (
                     SELECT 1 FROM journal_tombstones AS t WHERE t.id = r.id
                 )
                 ORDER BY seq ASC",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| {
                Ok(JournalRecord {
                    seq: row.get(0)?,
                    id: row.get(1)?,
                    kind: row.get(2)?,
                    payload: row.get(3)?,
                    recorded_at_ms: row.get(4)?,
                })
            })
            .map_err(map_sqlite)?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row.map_err(map_sqlite)?);
        }
        Ok(records)
    }

    /// Tombstone a record id without rewriting history.
    ///
    /// Idempotent: tombstoning an already-tombstoned or unknown id succeeds
    /// and leaves earlier state untouched (the first tombstone timestamp
    /// wins). Reads exclude tombstoned ids.
    pub fn tombstone(&self, id: &str, tombstoned_at_ms: i64) -> Result<(), JournalError> {
        check_field("id", id, MAX_ID_BYTES)?;
        self.conn
            .execute(
                "INSERT OR IGNORE INTO journal_tombstones (id, tombstoned_at_ms)
                 VALUES (?1, ?2)",
                params![id, tombstoned_at_ms],
            )
            .map_err(map_sqlite)?;
        Ok(())
    }
}

fn check_field(field: &'static str, value: &str, limit: usize) -> Result<(), JournalError> {
    if value.is_empty() {
        return Err(JournalError::EmptyField { field });
    }
    let actual = value.len();
    if actual > limit {
        return Err(JournalError::FieldTooLong {
            field,
            limit,
            actual,
        });
    }
    Ok(())
}

fn is_unique_violation(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, _)
            if e.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
                || e.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_PRIMARYKEY
    )
}

/// Map a storage error from normal operation (append, read, tombstone).
fn map_sqlite(err: rusqlite::Error) -> JournalError {
    let detail = err.to_string();
    if let rusqlite::Error::SqliteFailure(e, _) = &err {
        return match e.code {
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked => {
                JournalError::WriterBusy
            }
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase => {
                JournalError::Corrupt { detail }
            }
            _ => JournalError::Storage { detail },
        };
    }
    JournalError::Storage { detail }
}

/// Map a failure while establishing or verifying the schema. Anything other
/// than contention is treated as malformed state: the journal does not run
/// against a database it cannot recognize.
fn schema_error(err: rusqlite::Error) -> JournalError {
    let detail = err.to_string();
    if let rusqlite::Error::SqliteFailure(e, _) = &err {
        if matches!(
            e.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        ) {
            return JournalError::WriterBusy;
        }
    }
    JournalError::Corrupt { detail }
}

/// What a store looks like before admission decides what to do with it.
enum StoreState {
    /// No prototype name is occupied: a missing file, a fresh database, or a
    /// SQLite database with no prototype objects. The only state in which the
    /// prototype creates its schema.
    Empty,
    /// Every prototype name is occupied by its exact-case table; structural
    /// and profile verification decides admission.
    Complete,
    /// Some, but not all, prototype names are occupied by their exact-case
    /// tables. Always refused: a partial store is indistinguishable from an
    /// interrupted or foreign database and has no repair path.
    Partial { present: Vec<&'static str> },
    /// A prototype name is occupied by something that is not its exact-case
    /// table: a case-variant table, a view, or any other object type. Always
    /// refused before any schema statement: SQLite name resolution is
    /// case-insensitive, so `CREATE TABLE IF NOT EXISTS` would silently no-op
    /// under the foreign object and later operations would run against it.
    Occupied { detail: String },
}

/// Decide whether `conn` is a new store to initialize, an existing journal to
/// admit, or a store to refuse. Runs before any schema statement or marker
/// write, so a refusal leaves the database byte-for-byte unchanged.
fn admit_or_initialize(conn: &Connection) -> Result<(), JournalError> {
    match classify_store(conn)? {
        StoreState::Empty => conn.execute_batch(SCHEMA).map_err(schema_error),
        StoreState::Complete => {
            for table in &REQUIRED_TABLES {
                verify_table_shape(conn, table)?;
            }
            verify_profile(conn)
        }
        StoreState::Partial { present } => Err(JournalError::Corrupt {
            detail: format!("partial journal schema; present tables: {present:?}"),
        }),
        StoreState::Occupied { detail } => Err(JournalError::Corrupt { detail }),
    }
}

/// Classify the store by which prototype names are occupied and by what.
/// Names come from the fixed [`REQUIRED_TABLES`] list, never from caller
/// input.
///
/// SQLite resolves table/view/index names case-insensitively, so for each
/// prototype name the scan collects every case-insensitive match and decides
/// per name:
///
/// - the exact-case table is present: `present`;
/// - otherwise any table/view/index match (a case variant, a view, or an
///   index) is an occupancy, because `CREATE TABLE IF NOT EXISTS` would
///   no-op under it or fail on its name;
/// - otherwise absent. A `trigger` match alone is ignored: SQLite keeps
///   triggers in a separate namespace, so a trigger cannot block the schema
///   statement and cannot be the target of a table read or write.
fn classify_store(conn: &Connection) -> Result<StoreState, JournalError> {
    let mut stmt = conn
        .prepare(
            "SELECT type, name FROM sqlite_master
             WHERE name COLLATE NOCASE IN (?1, ?2, ?3)",
        )
        .map_err(schema_error)?;
    let mut rows = stmt
        .query(params![
            REQUIRED_TABLES[0].table,
            REQUIRED_TABLES[1].table,
            REQUIRED_TABLES[2].table
        ])
        .map_err(schema_error)?;

    let mut matches: Vec<(String, String)> = Vec::new();
    while let Some(row) = rows.next().map_err(schema_error)? {
        matches.push((
            row.get(0).map_err(schema_error)?,
            row.get(1).map_err(schema_error)?,
        ));
    }

    let mut present: Vec<&'static str> = Vec::new();
    let mut occupied: Vec<(&'static str, String, String)> = Vec::new();
    for table in &REQUIRED_TABLES {
        // Each `find` inspects only rows the query already filtered to a
        // prototype name (`COLLATE NOCASE`): `None` means absent, or that the
        // only matches are triggers, which are a separate namespace.
        let exact_table = matches
            .iter()
            .find(|(object_type, name)| object_type == "table" && name == table.table);
        if exact_table.is_some() {
            present.push(table.table);
            continue;
        }
        if let Some((object_type, name)) = matches.iter().find(|(object_type, name)| {
            matches!(object_type.as_str(), "table" | "view" | "index")
                && name.eq_ignore_ascii_case(table.table)
        }) {
            occupied.push((table.table, object_type.clone(), name.clone()));
        }
    }

    if let Some((expected, object_type, name)) = occupied.first() {
        return Ok(StoreState::Occupied {
            detail: format!(
                "prototype name '{expected}' is occupied by {object_type} '{name}', not the exact-case prototype table"
            ),
        });
    }
    if present.is_empty() {
        return Ok(StoreState::Empty);
    }
    if present.len() == REQUIRED_TABLES.len() {
        return Ok(StoreState::Complete);
    }
    Ok(StoreState::Partial { present })
}

/// Compare one existing table against its expected shape: column names and
/// order, declared types, `NOT NULL` flags, primary-key position, the
/// rowid-alias property of the `journal_records.seq` key, and (for
/// `journal_records`) a non-partial unique index over exactly `("id")`.
/// Everything `PRAGMA` `table_info`/`index_list`/`index_info` exposes is
/// checked; generated columns, CHECK constraints, foreign keys, collations,
/// partial-index predicates, and the `AUTOINCREMENT` property are not,
/// because the prototype schema declares none of them and the pragmas do not
/// expose `AUTOINCREMENT` (see the module-level admission policy). Partial
/// and composite indexes are not inspected in detail: any unique index that
/// is not exactly `("id")` fails the predicate, so they are refused.
///
/// `PRAGMA table_info` alone cannot tell a rowid alias from a
/// non-alias key: `INTEGER PRIMARY KEY DESC` reports `type=INTEGER`,
/// `notnull=0`, `pk=1`, identical to the accepted form, yet SQLite does not
/// alias it to the rowid. The rowid-alias predicate therefore reads
/// `PRAGMA index_list` and requires that no `origin = 'pk'` index exist, the
/// one observable difference every non-alias primary key shares.
fn verify_table_shape(conn: &Connection, expected: &ExpectedTable) -> Result<(), JournalError> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({})", expected.table))
        .map_err(schema_error)?;
    let mut actual_columns: Vec<(String, String, bool, u32)> = Vec::new();
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)? != 0,
                row.get::<_, i64>(5)? as u32,
            ))
        })
        .map_err(schema_error)?;
    for row in rows {
        actual_columns.push(row.map_err(schema_error)?);
    }

    if actual_columns.len() != expected.columns.len() {
        return Err(JournalError::Corrupt {
            detail: format!(
                "table {} has {} columns, expected {}",
                expected.table,
                actual_columns.len(),
                expected.columns.len()
            ),
        });
    }

    for (index, column) in expected.columns.iter().enumerate() {
        let (name, declared_type, not_null, primary_key) = &actual_columns[index];
        if name != column.name {
            return Err(JournalError::Corrupt {
                detail: format!(
                    "table {} column {} is named '{name}', expected '{}'",
                    expected.table, index, column.name
                ),
            });
        }
        if declared_type != column.declared_type {
            return Err(JournalError::Corrupt {
                detail: format!(
                    "table {} column '{name}' has type '{declared_type}', expected '{}'",
                    expected.table, column.declared_type
                ),
            });
        }
        if *not_null != column.not_null {
            return Err(JournalError::Corrupt {
                detail: format!(
                    "table {} column '{name}' has wrong NOT NULL constraint",
                    expected.table
                ),
            });
        }
        // `PRAGMA table_info` reports the primary-key position (`1`-based,
        // `0` when a column is not part of the key); a composite key shows up
        // as positions `2..n` on later columns and is refused here.
        let expected_primary_key = if index == 0 { expected.primary_key } else { 0 };
        if *primary_key != expected_primary_key {
            return Err(JournalError::Corrupt {
                detail: format!(
                    "table {} column '{name}' has unexpected primary-key position",
                    expected.table
                ),
            });
        }
    }

    if expected.rowid_alias {
        // A rowid alias is the one primary-key form SQLite materializes with
        // no separate index: `PRAGMA index_list` then reports no index with
        // `origin = 'pk'`. `INTEGER PRIMARY KEY DESC`, a table-level
        // `PRIMARY KEY (seq)`, a `WITHOUT ROWID` table, and every
        // non-`INTEGER` or composite key all create a `pk`-origin index
        // instead, which is exactly what `PRAGMA table_info` cannot
        // distinguish. Requiring the absence of that index rejects them
        // before any mutation: without the alias `seq` is not populated from
        // `last_insert_rowid()`, so `append` would return a `seq` that is
        // never stored (NULL) and `read_all` would later fail on the NULL.
        let has_separate_primary_key_index: bool = conn
            .query_row(
                "SELECT EXISTS(
                     SELECT 1 FROM pragma_index_list(?1) AS i
                     WHERE i.origin = 'pk'
                 )",
                [expected.table],
                |row| row.get(0),
            )
            .map_err(schema_error)?;
        if has_separate_primary_key_index {
            return Err(JournalError::Corrupt {
                detail: format!(
                    "table {} primary key is not an INTEGER PRIMARY KEY rowid alias; append order would not be stored",
                    expected.table
                ),
            });
        }
    }

    if expected.unique_id_index {
        // The duplicate-id predicate is exact: a unique index whose full
        // column set is exactly `("id")`, with `partial = 0`. A partial
        // unique index (`WHERE ...`) or a composite key such as
        // `UNIQUE (id, kind)` would satisfy "unique and first column is id"
        // while still letting duplicate ids coexist, so both are refused.
        let unique_on_id: bool = conn
            .query_row(
                "SELECT EXISTS(
                     SELECT 1 FROM pragma_index_list(?1) AS i
                     WHERE i.\"unique\" = 1
                       AND i.partial = 0
                       AND (SELECT COUNT(*) FROM pragma_index_info(i.name)) = 1
                       AND 'id' = (
                           SELECT c.name FROM pragma_index_info(i.name) AS c
                           ORDER BY c.seqno ASC LIMIT 1
                       )
                 )",
                [expected.table],
                |row| row.get(0),
            )
            .map_err(schema_error)?;
        if !unique_on_id {
            return Err(JournalError::Corrupt {
                detail: format!(
                    "table {} lacks a non-partial unique index on exactly ('id'); duplicate-id protection is not enforceable",
                    expected.table
                ),
            });
        }
    }

    Ok(())
}

/// Read and compare the stored profile. A missing row means initialization
/// never completed (or the store is foreign): refused. An identified but
/// different profile is a distinct schema identity with no migration path.
fn verify_profile(conn: &Connection) -> Result<(), JournalError> {
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM journal_meta WHERE key = 'profile'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(schema_error)?;

    match stored.as_deref() {
        Some(PROFILE) => Ok(()),
        Some(other) => Err(JournalError::UnsupportedSchema {
            profile: other.to_owned(),
        }),
        None => Err(JournalError::Corrupt {
            detail: "journal_meta has no 'profile' marker; store is not an initialized journal"
                .to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Deterministic, caller-owned scratch directory: no new test dependency,
    /// unique per test name and process, removed on drop.
    struct ScratchDir {
        path: PathBuf,
    }

    impl ScratchDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("bitty-ai-journal-{}-{name}", std::process::id()));
            if path.exists() {
                fs::remove_dir_all(&path).expect("clear previous scratch dir");
            }
            fs::create_dir_all(&path).expect("create scratch dir");
            Self { path }
        }

        fn db_path(&self) -> PathBuf {
            self.path.join("journal.sqlite3")
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn append_and_read_back_in_append_order_across_reopen() {
        let scratch = ScratchDir::new("append-order");
        let path = scratch.db_path();
        let journal = Journal::open(&path).expect("open fresh journal");

        let first = journal
            .append("rec-1", "note", "first", 1_000)
            .expect("append first");
        let second = journal
            .append("rec-2", "note", "second", 1_000)
            .expect("append second with the same caller timestamp");
        let third = journal
            .append("rec-3", "tool", "third", 3_500)
            .expect("append third");
        assert!(first.seq < second.seq);
        assert!(second.seq < third.seq);
        assert_eq!(first.seq + 1, second.seq);
        assert_eq!(second.seq + 1, third.seq);
        drop(journal);

        let reopened = Journal::open(&path).expect("reopen existing journal");
        let records = reopened.read_all().expect("read all");
        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["rec-1", "rec-2", "rec-3"]
        );
        assert_eq!(records[0].payload, "first");
        assert_eq!(records[1].recorded_at_ms, 1_000);
        assert_eq!(records[2].kind, "tool");
    }

    #[test]
    fn tombstone_hides_record_without_rewriting_history() {
        let scratch = ScratchDir::new("tombstone-visibility");
        let path = scratch.db_path();
        let journal = Journal::open(&path).expect("open");
        journal.append("rec-1", "note", "first", 10).unwrap();
        journal.append("rec-2", "note", "second", 20).unwrap();
        journal.append("rec-3", "note", "third", 30).unwrap();

        journal.tombstone("rec-2", 99).expect("tombstone rec-2");
        let surviving: Vec<String> = journal
            .read_all()
            .expect("read surviving")
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert_eq!(surviving, ["rec-1", "rec-3"]);
        drop(journal);

        let raw = Connection::open(&path).expect("raw open after close");
        let records: i64 = raw
            .query_row("SELECT COUNT(*) FROM journal_records", [], |row| row.get(0))
            .unwrap();
        let tombstones: i64 = raw
            .query_row("SELECT COUNT(*) FROM journal_tombstones", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(records, 3, "record rows must not be rewritten or removed");
        assert_eq!(tombstones, 1);
        let payload: String = raw
            .query_row(
                "SELECT payload FROM journal_records WHERE id = 'rec-2'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(payload, "second", "tombstoned payload stays in history");
    }

    #[test]
    fn tombstone_is_idempotent_and_tolerates_unknown_ids() {
        let scratch = ScratchDir::new("tombstone-idempotent");
        let path = scratch.db_path();
        let journal = Journal::open(&path).expect("open");
        journal.append("rec-1", "note", "first", 10).unwrap();
        journal.append("rec-2", "note", "second", 20).unwrap();

        journal.tombstone("rec-1", 111).expect("first tombstone");
        journal
            .tombstone("rec-1", 222)
            .expect("second tombstone is idempotent");
        journal
            .tombstone("never-recorded", 333)
            .expect("unknown id is a no-op");
        let surviving: Vec<String> = journal
            .read_all()
            .expect("read surviving")
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert_eq!(surviving, ["rec-2"]);
        drop(journal);

        let raw = Connection::open(&path).expect("raw open after close");
        let tombstones: i64 = raw
            .query_row("SELECT COUNT(*) FROM journal_tombstones", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(tombstones, 2, "repeat tombstone must not duplicate rows");
        let stamp: i64 = raw
            .query_row(
                "SELECT tombstoned_at_ms FROM journal_tombstones WHERE id = 'rec-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stamp, 111, "first tombstone timestamp wins");
        let unknown: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM journal_tombstones WHERE id = 'never-recorded'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(unknown, 1);
    }

    #[test]
    fn corrupt_database_fails_closed_without_reset() {
        let scratch = ScratchDir::new("corrupt-garbage");
        let path = scratch.db_path();
        let garbage =
            b"this is not a sqlite database; deterministic noise padded past the 100-byte header";
        fs::write(&path, garbage).expect("seed garbage file");

        let err = Journal::open(&path).expect_err("garbage must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            fs::read(&path).expect("read back"),
            garbage,
            "the corrupt file must not be truncated, reset, or recreated"
        );
    }

    #[test]
    fn incompatible_schema_fails_closed_and_preserves_rows() {
        let scratch = ScratchDir::new("corrupt-schema");
        let path = scratch.db_path();
        {
            let raw = Connection::open(&path).expect("raw create");
            raw.execute_batch(
                "CREATE TABLE journal_records (seq INTEGER PRIMARY KEY, wrong TEXT NOT NULL);",
            )
            .expect("seed wrong schema");
            raw.execute("INSERT INTO journal_records (wrong) VALUES ('keep-me')", [])
                .expect("seed row");
        }

        let err = Journal::open(&path).expect_err("incompatible schema must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "unexpected error: {err:?}"
        );

        let raw = Connection::open(&path).expect("raw reopen");
        let kept: String = raw
            .query_row("SELECT wrong FROM journal_records", [], |row| row.get(0))
            .expect("existing row survives");
        assert_eq!(kept, "keep-me");
    }

    /// Full prototype table shapes as raw SQL, used to seed incompatible
    /// fixtures that differ from [`SCHEMA`] by exactly one property.
    const VALID_TABLES_SQL: &str = "
CREATE TABLE journal_records (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    id             TEXT NOT NULL UNIQUE,
    kind           TEXT NOT NULL,
    payload        TEXT NOT NULL,
    recorded_at_ms INTEGER NOT NULL
);
CREATE TABLE journal_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE journal_tombstones (
    id               TEXT PRIMARY KEY,
    tombstoned_at_ms INTEGER NOT NULL
);
";

    /// Seed a raw store with `sql` and the marker row that matches the shape
    /// the fixture declares. No prototype code participates, so a fixture is
    /// never admitted through the path under test.
    fn seed_fixture(path: &Path, sql: &str, profile: Option<&str>) {
        let raw = Connection::open(path).expect("raw create");
        raw.execute_batch(sql).expect("seed fixture schema");
        if let Some(profile) = profile {
            raw.execute(
                "INSERT INTO journal_meta (key, value) VALUES ('profile', ?1)",
                [profile],
            )
            .expect("seed profile marker");
        }
    }

    /// Assert a refused open left every seeded row in place. The fixture row
    /// is the same in every fixture, so the checks are static.
    fn assert_fixture_row_survived(path: &Path) {
        let raw = Connection::open(path).expect("raw reopen");
        let kept: String = raw
            .query_row(
                "SELECT payload FROM journal_records WHERE id = 'rec-1'",
                [],
                |row| row.get(0),
            )
            .expect("seeded row must survive refusal");
        assert_eq!(kept, "keep-me");
    }

    #[test]
    fn wrong_column_type_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-wrong-type");
        let path = scratch.db_path();
        // Same names and constraints as the prototype, but `recorded_at_ms`
        // is declared `TEXT`; the row stays readable only under the wrong
        // declared type.
        seed_fixture(
            &path,
            "CREATE TABLE journal_records (
                 seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                 id             TEXT NOT NULL UNIQUE,
                 kind           TEXT NOT NULL,
                 payload        TEXT NOT NULL,
                 recorded_at_ms TEXT NOT NULL
             );
             CREATE TABLE journal_meta (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE journal_tombstones (
                 id               TEXT PRIMARY KEY,
                 tombstoned_at_ms INTEGER NOT NULL
             );",
            Some(PROFILE),
        );
        let raw = Connection::open(&path).expect("raw insert");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', '1700')",
            [],
        )
        .expect("seed row");

        let err = Journal::open(&path).expect_err("wrong column type must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a wrong column type"
        );
        assert_fixture_row_survived(&path);
    }

    #[test]
    fn missing_not_null_constraint_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-missing-not-null");
        let path = scratch.db_path();
        seed_fixture(&path, VALID_TABLES_SQL, Some(PROFILE));
        let raw = Connection::open(&path).expect("raw alter");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', 1700)",
            [],
        )
        .expect("seed row");
        // Rebuild `journal_records` without NOT NULL on `payload`; names,
        // types, and key shape stay identical, so only the constraint
        // differs from the expected schema.
        raw.execute_batch(
            "ALTER TABLE journal_records RENAME TO journal_records_old;
             CREATE TABLE journal_records (
                 seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                 id             TEXT NOT NULL UNIQUE,
                 kind           TEXT NOT NULL,
                 payload        TEXT,
                 recorded_at_ms INTEGER NOT NULL
             );
             INSERT INTO journal_records
                 SELECT seq, id, kind, payload, recorded_at_ms FROM journal_records_old;
             DROP TABLE journal_records_old;",
        )
        .expect("rebuild without NOT NULL");

        let err = Journal::open(&path).expect_err("missing NOT NULL must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a missing NOT NULL constraint"
        );
        assert_fixture_row_survived(&path);
    }

    #[test]
    fn altered_primary_key_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-altered-pk");
        let path = scratch.db_path();
        seed_fixture(&path, VALID_TABLES_SQL, Some(PROFILE));
        let raw = Connection::open(&path).expect("raw alter");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', 1700)",
            [],
        )
        .expect("seed row");
        // Move the rowid key off `seq` and onto `kind`: names and types
        // still match, but the primary-key shape no longer does, so append
        // order would be assigned by a column the journal does not control.
        raw.execute_batch(
            "ALTER TABLE journal_records RENAME TO journal_records_old;
             CREATE TABLE journal_records (
                 seq            INTEGER,
                 id             TEXT NOT NULL UNIQUE,
                 kind           TEXT NOT NULL PRIMARY KEY,
                 payload        TEXT NOT NULL,
                 recorded_at_ms INTEGER NOT NULL
             );
             INSERT INTO journal_records
                 SELECT seq, id, kind, payload, recorded_at_ms FROM journal_records_old;
             DROP TABLE journal_records_old;",
        )
        .expect("rebuild with altered primary key");

        let err = Journal::open(&path).expect_err("altered primary key must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for an altered primary key"
        );
        assert_fixture_row_survived(&path);
    }

    #[test]
    fn descending_rowid_alias_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-desc-alias");
        let path = scratch.db_path();
        // `seq INTEGER PRIMARY KEY DESC` is NOT a rowid alias: `PRAGMA
        // table_info` reports exactly the same `type`/`notnull`/`pk` as the
        // accepted `INTEGER PRIMARY KEY`, so only the `pk`-origin index it
        // materializes distinguishes it. Without the alias an INSERT leaves
        // `seq` NULL while `last_insert_rowid()` still returns 1, 2, ...;
        // `append` would report a `seq` never stored and `read_all` would
        // later fail on the NULL. It must be refused before any mutation.
        seed_fixture(
            &path,
            "CREATE TABLE journal_records (
                 seq            INTEGER PRIMARY KEY DESC,
                 id             TEXT NOT NULL UNIQUE,
                 kind           TEXT NOT NULL,
                 payload        TEXT NOT NULL,
                 recorded_at_ms INTEGER NOT NULL
             );
             CREATE TABLE journal_meta (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE journal_tombstones (
                 id               TEXT PRIMARY KEY,
                 tombstoned_at_ms INTEGER NOT NULL
             );",
            Some(PROFILE),
        );
        let raw = Connection::open(&path).expect("raw insert");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', 1700)",
            [],
        )
        .expect("seed row");

        let err = Journal::open(&path).expect_err("descending rowid alias must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a non-rowid primary key"
        );
        assert_fixture_row_survived(&path);

        // The refusal wrote nothing: `seq` is still NULL for the seeded row,
        // and no row exists under a journal-assigned sequence.
        let raw = Connection::open(&path).expect("raw reopen");
        let null_seq: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM journal_records WHERE seq IS NULL",
                [],
                |row| row.get(0),
            )
            .expect("count null-seq rows");
        assert_eq!(
            null_seq, 1,
            "refusal must not have rewritten the seeded row"
        );
    }

    #[test]
    fn wrong_profile_is_refused_unchanged() {
        let scratch = ScratchDir::new("schema-wrong-profile");
        let path = scratch.db_path();
        seed_fixture(&path, VALID_TABLES_SQL, Some("r5-prototype"));
        let raw = Connection::open(&path).expect("raw insert");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', 1700)",
            [],
        )
        .expect("seed row");

        let err = Journal::open(&path).expect_err("foreign profile must fail closed");
        assert!(
            matches!(
                err,
                JournalError::UnsupportedSchema { ref profile } if profile == "r5-prototype"
            ),
            "expected a typed UnsupportedSchema refusal carrying the stored profile"
        );
        assert_fixture_row_survived(&path);

        let raw = Connection::open(&path).expect("raw reopen");
        let stored: String = raw
            .query_row(
                "SELECT value FROM journal_meta WHERE key = 'profile'",
                [],
                |row| row.get(0),
            )
            .expect("profile marker must survive refusal");
        assert_eq!(stored, "r5-prototype", "a refused profile is not rewritten");
    }

    #[test]
    fn partial_store_is_refused_and_not_completed() {
        let scratch = ScratchDir::new("schema-partial");
        let path = scratch.db_path();
        seed_fixture(
            &path,
            "CREATE TABLE journal_records (
                 seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                 id             TEXT NOT NULL UNIQUE,
                 kind           TEXT NOT NULL,
                 payload        TEXT NOT NULL,
                 recorded_at_ms INTEGER NOT NULL
             );
             INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
                 VALUES ('rec-1', 'note', 'keep-me', 1700);",
            None,
        );

        let err = Journal::open(&path).expect_err("partial store must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a partial prototype store"
        );
        assert_fixture_row_survived(&path);

        let raw = Connection::open(&path).expect("raw reopen");
        let tables: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name IN ('journal_records', 'journal_meta', 'journal_tombstones')",
                [],
                |row| row.get(0),
            )
            .expect("count journal tables");
        assert_eq!(
            tables, 1,
            "missing tables must not be grafted onto a refusal"
        );
    }

    #[test]
    fn case_variant_prototype_table_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-case-collision");
        let path = scratch.db_path();
        // SQLite matches table names case-insensitively, so a store holding
        // `Journal_Records` would previously classify Empty, no-op the schema
        // creation, and adopt the foreign table. The exact-case name is what
        // marks a prototype table; a case variant is an occupancy.
        seed_fixture(
            &path,
            "CREATE TABLE Journal_Records (
                 seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                 id             TEXT NOT NULL UNIQUE,
                 kind           TEXT NOT NULL,
                 payload        TEXT NOT NULL,
                 recorded_at_ms INTEGER NOT NULL
             );
             INSERT INTO Journal_Records (id, kind, payload, recorded_at_ms)
                 VALUES ('rec-1', 'note', 'keep-me', 1700);",
            None,
        );

        let err = Journal::open(&path).expect_err("case-variant table must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a case-variant prototype name"
        );
        assert_fixture_row_survived(&path);

        let raw = Connection::open(&path).expect("raw reopen");
        let journal_records: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'journal_records'",
                [],
                |row| row.get(0),
            )
            .expect("count exact-case table");
        let case_variant: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'Journal_Records'",
                [],
                |row| row.get(0),
            )
            .expect("count case-variant table");
        assert_eq!(
            journal_records, 0,
            "the prototype must not create its table under an occupied name"
        );
        assert_eq!(case_variant, 1, "the foreign table must stay untouched");
    }

    #[test]
    fn case_variant_index_collision_is_refused_without_mutation() {
        let scratch = ScratchDir::new("schema-index-collision");
        let path = scratch.db_path();
        // An index named `Journal_Records` (case variant of a prototype
        // table) cannot coexist with the prototype's table under SQLite's
        // case-insensitive index/table namespace: `CREATE TABLE IF NOT
        // EXISTS journal_records` fails with a name-collision error. That
        // must surface as a typed refusal before any write, not as untyped
        // storage failure after partial schema creation.
        seed_fixture(
            &path,
            "CREATE TABLE unrelated (value TEXT NOT NULL);
             CREATE UNIQUE INDEX Journal_Records ON unrelated (value);
             INSERT INTO unrelated (value) VALUES ('keep-me');",
            None,
        );

        let err = Journal::open(&path).expect_err("index collision must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for an index occupying a prototype name"
        );

        let raw = Connection::open(&path).expect("raw reopen");
        let prototype_tables: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table'
                   AND name IN ('journal_records', 'journal_meta', 'journal_tombstones')",
                [],
                |row| row.get(0),
            )
            .expect("count exact-case prototype tables");
        assert_eq!(
            prototype_tables, 0,
            "a refusal must not leave partially created prototype tables"
        );
        let colliding_index: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'Journal_Records'",
                [],
                |row| row.get(0),
            )
            .expect("count colliding index");
        assert_eq!(colliding_index, 1, "the foreign index must stay untouched");
        let kept: String = raw
            .query_row("SELECT value FROM unrelated", [], |row| row.get(0))
            .expect("seed row must survive refusal");
        assert_eq!(kept, "keep-me");
    }

    #[test]
    fn partial_unique_index_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-partial-unique");
        let path = scratch.db_path();
        // A partial unique index on `id` (`WHERE kind = 'note'`) satisfies
        // "unique and first column is id" but lets the same id exist under
        // another kind, so it must not be admitted as duplicate-id
        // protection.
        seed_fixture(
            &path,
            "CREATE TABLE journal_records (
                 seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                 id             TEXT NOT NULL,
                 kind           TEXT NOT NULL,
                 payload        TEXT NOT NULL,
                 recorded_at_ms INTEGER NOT NULL
             );
             CREATE UNIQUE INDEX note_only_id ON journal_records (id) WHERE kind = 'note';
             CREATE TABLE journal_meta (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE journal_tombstones (
                 id               TEXT PRIMARY KEY,
                 tombstoned_at_ms INTEGER NOT NULL
             );",
            Some(PROFILE),
        );
        let raw = Connection::open(&path).expect("raw insert");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', 1700)",
            [],
        )
        .expect("seed row");

        let err = Journal::open(&path).expect_err("partial unique index must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a partial unique index"
        );
        assert_fixture_row_survived(&path);
    }

    #[test]
    fn composite_unique_key_is_refused_and_preserves_rows() {
        let scratch = ScratchDir::new("schema-composite-unique");
        let path = scratch.db_path();
        // A composite `UNIQUE (id, kind)` reports `id` at `seqno = 0`, so it
        // previously satisfied the duplicate-id predicate while two rows
        // with the same id in different kinds coexist. The unique index must
        // cover exactly `("id")`.
        seed_fixture(
            &path,
            "CREATE TABLE journal_records (
                 seq            INTEGER PRIMARY KEY AUTOINCREMENT,
                 id             TEXT NOT NULL,
                 kind           TEXT NOT NULL,
                 payload        TEXT NOT NULL,
                 recorded_at_ms INTEGER NOT NULL,
                 UNIQUE (id, kind)
             );
             CREATE TABLE journal_meta (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE journal_tombstones (
                 id               TEXT PRIMARY KEY,
                 tombstoned_at_ms INTEGER NOT NULL
             );",
            Some(PROFILE),
        );
        let raw = Connection::open(&path).expect("raw insert");
        raw.execute(
            "INSERT INTO journal_records (id, kind, payload, recorded_at_ms)
             VALUES ('rec-1', 'note', 'keep-me', 1700)",
            [],
        )
        .expect("seed row");

        let err = Journal::open(&path).expect_err("composite unique key must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a composite unique key"
        );
        assert_fixture_row_survived(&path);
    }

    #[test]
    fn view_backed_prototype_name_is_refused_without_mutation() {
        let scratch = ScratchDir::new("schema-view");
        let path = scratch.db_path();
        // A view named `journal_meta` reads like a prototype table under
        // SQLite's case-insensitive name lookup but cannot be written. It
        // must be refused as a typed occupancy, not classified Empty and
        // mutated until the marker write fails with an untyped error.
        seed_fixture(
            &path,
            "CREATE VIEW journal_meta AS SELECT 'profile' AS key, 'r6-prototype' AS value;",
            None,
        );

        let err = Journal::open(&path).expect_err("view-backed name must fail closed");
        assert!(
            matches!(err, JournalError::Corrupt { .. }),
            "expected a typed Corrupt refusal for a view occupying a prototype name"
        );

        let raw = Connection::open(&path).expect("raw reopen");
        let objects: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE name IN ('journal_records', 'journal_meta', 'journal_tombstones')",
                [],
                |row| row.get(0),
            )
            .expect("count prototype names");
        assert_eq!(
            objects, 1,
            "a refused view store must not have prototype tables grafted onto it"
        );
        let view_sql: String = raw
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'view' AND name = 'journal_meta'",
                [],
                |row| row.get(0),
            )
            .expect("view must survive refusal");
        assert_eq!(view_sql, "journal_meta");
    }

    #[test]
    fn valid_existing_journal_with_data_reopens_and_appends() {
        let scratch = ScratchDir::new("schema-valid-existing");
        let path = scratch.db_path();
        let journal = Journal::open(&path).expect("open fresh journal");
        journal.append("rec-1", "note", "first", 10).unwrap();
        journal.append("rec-2", "note", "second", 20).unwrap();
        journal.tombstone("rec-2", 30).expect("tombstone rec-2");
        drop(journal);

        let reopened = Journal::open(&path).expect("valid existing journal must be admitted");
        let records = reopened.read_all().expect("read surviving");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "rec-1");
        assert_eq!(records[0].payload, "first");

        let appended = reopened
            .append("rec-3", "note", "third", 40)
            .expect("append to an admitted journal");
        assert_eq!(appended.seq, 3, "append order continues from stored seq");
        let records = reopened.read_all().expect("read after append");
        assert_eq!(
            records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["rec-1", "rec-3"]
        );
    }

    #[test]
    fn second_writer_is_rejected_until_first_closes() {
        let scratch = ScratchDir::new("single-writer");
        let path = scratch.db_path();
        let first = Journal::open(&path).expect("first writer");
        let err = Journal::open(&path).expect_err("second writer must be rejected");
        assert!(
            matches!(err, JournalError::WriterBusy),
            "unexpected error: {err:?}"
        );
        drop(first);
        let third = Journal::open(&path).expect("lock must be released when the writer closes");
        drop(third);
    }

    #[test]
    fn bounded_fields_fail_closed_before_any_write() {
        let scratch = ScratchDir::new("bounds");
        let path = scratch.db_path();
        let journal = Journal::open(&path).expect("open");

        let long_payload = "x".repeat(MAX_PAYLOAD_BYTES + 1);
        let err = journal
            .append("rec-1", "note", &long_payload, 1)
            .expect_err("over-long payload must fail");
        assert!(
            matches!(
                err,
                JournalError::FieldTooLong {
                    field: "payload",
                    limit: MAX_PAYLOAD_BYTES,
                    actual,
                } if actual == MAX_PAYLOAD_BYTES + 1
            ),
            "unexpected error: {err:?}"
        );
        let err = journal
            .append("", "note", "x", 1)
            .expect_err("empty id must fail");
        assert!(
            matches!(err, JournalError::EmptyField { field: "id" }),
            "unexpected error: {err:?}"
        );
        assert!(
            journal.read_all().expect("read all").is_empty(),
            "failed appends must not write"
        );
    }

    #[test]
    fn duplicate_id_is_rejected_and_original_record_survives() {
        let scratch = ScratchDir::new("duplicate-id");
        let path = scratch.db_path();
        let journal = Journal::open(&path).expect("open");
        journal.append("rec-1", "note", "first", 1).unwrap();

        let err = journal
            .append("rec-1", "note", "second", 2)
            .expect_err("duplicate id must fail");
        assert!(
            matches!(err, JournalError::DuplicateId { ref id } if id == "rec-1"),
            "unexpected error: {err:?}"
        );
        let records = journal.read_all().expect("read all");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload, "first");
    }
}

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
//! - One SQLite database file at a caller-supplied path; this module creates
//!   the schema on first open (journal mode is SQLite's default rollback
//!   journal; WAL selection stays open under AIQ-53).
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
//! ## Fail-closed rules
//!
//! - Malformed, corrupt, or schema-incompatible database state yields a
//!   typed [`JournalError::Corrupt`]. The prototype never deletes, truncates,
//!   recreates, or silently resets an existing database file and never
//!   repairs state implicitly.
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

use rusqlite::{Connection, params};

/// Prototype bound for record ids (128 bytes, mirroring the consent-scope
/// bound used elsewhere in the slice).
pub const MAX_ID_BYTES: usize = 128;

/// Prototype bound for record kinds (64 bytes).
pub const MAX_KIND_BYTES: usize = 64;

/// Prototype bound for record payloads (8 KiB).
pub const MAX_PAYLOAD_BYTES: usize = 8 * 1024;

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

/// Expected column names per table, verified before any schema statement runs
/// so a foreign or incompatible database is rejected without mutation.
const REQUIRED_TABLES: [(&str, &[&str]); 3] = [
    (
        "journal_records",
        &["seq", "id", "kind", "payload", "recorded_at_ms"],
    ),
    ("journal_meta", &["key", "value"]),
    ("journal_tombstones", &["id", "tombstoned_at_ms"]),
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
    /// Creating a missing database is the only implicit state change. A
    /// pre-existing database with malformed content or an incompatible
    /// schema fails closed with [`JournalError::Corrupt`] and is left in
    /// place; a second open while another [`Journal`] holds the file fails
    /// with [`JournalError::WriterBusy`].
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

        verify_existing_schema(&conn)?;
        conn.execute_batch(SCHEMA).map_err(schema_error)?;

        // A real page write claims the exclusive lock. `INSERT OR REPLACE`
        // always performs a delete and an insert, so the write happens even
        // when the marker row is already present.
        conn.execute(
            "INSERT OR REPLACE INTO journal_meta (key, value) VALUES ('profile', ?1)",
            ["r6-prototype"],
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

/// Verify the column shape of any pre-existing journal table before the
/// schema statements run, so a foreign or incompatible database is rejected
/// without mutation.
fn verify_existing_schema(conn: &Connection) -> Result<(), JournalError> {
    for (table, expected) in REQUIRED_TABLES {
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(
                     SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1
                 )",
                [table],
                |row| row.get(0),
            )
            .map_err(schema_error)?;
        if !exists {
            continue;
        }

        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(schema_error)?;
        let mut columns = Vec::new();
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(schema_error)?;
        for row in rows {
            columns.push(row.map_err(schema_error)?);
        }

        let expected: Vec<String> = expected.iter().map(|name| String::from(*name)).collect();
        if columns != expected {
            return Err(JournalError::Corrupt {
                detail: format!("table {table} has incompatible columns: {columns:?}"),
            });
        }
    }
    Ok(())
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

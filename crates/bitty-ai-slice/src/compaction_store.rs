//! Durable compaction slice: formal durable path for L2 selective compaction (AI-0178).
//!
//! Persists [`CompressedView`] span metadata, tombstones, retention tags, and
//! compaction driver state (`previous_summary`, `ineffective_strikes`,
//! `generation`, `window_bytes`) in SQLite. This is the formal durable path
//! per owner decision DEC-0005, not a prototype: [`crate::journal_prototype`]
//! (AI-0049, `r6-prototype`) is implementation evidence only and is wired into
//! no production path; this module is the shippable slice.
//!
//! ## Schema
//!
//! - `compaction_spans(span_id PK, source_ids_json, summary, retention,
//!   source_deadline_ms, created_at_ms, is_untrusted)`: one row per
//!   [`CompressedSpan`]. `is_untrusted` is a required extension beyond the
//!   minimal column list: untrusted-surface marking is security-critical
//!   (summaries of untrusted sources stay untrusted) and must survive a
//!   reopen, so it is stored, not recomputed.
//! - `compaction_tombstones(id PK)`: absence metadata only, never payload bytes.
//! - `compaction_meta(key PK, value)`: `profile`, `previous_summary`,
//!   `ineffective_strikes`, `generation`, `window_bytes`, plus `records_json`
//!   and `retention_tags_json`. Records and tags ride in meta (rather than a
//!   fourth table) to keep the three-table contract exact while achieving
//!   identical replay: synthetic/passthrough [`ContextRecord`]s carry
//!   provider/owner/generation/priority/body needed for assembly, and
//!   [`RetentionTags`] entries are required for expiry.
//!
//! ## Retention tags without runtime changes
//!
//! `bitty-ai-runtime` is read-only here and exposes no tag iterator. Tags are
//! persisted by probing: for every id in the view (record ids, span source
//! ids, tombstones), `tags.get(id)` is read and only non-`Normal` entries are
//! stored. Untagged ids correctly read back as `Normal`, so the persisted map
//! is complete without a runtime iterator.
//!
//! ## Bounds (reused semantics)
//!
//! - Span count per view: [`MAX_SPANS`] (32), enforced on write and on load.
//! - Span id length: [`MAX_SPAN_ID_LEN`] (64).
//! - Span sources: at most [`MAX_CONTEXT_RECORDS`] ids per span, each
//!   non-empty and at most [`MAX_TOMBSTONE_ID_LEN`] bytes.
//! - Summary text (span summaries and `previous_summary`):
//!   runtime [`MAX_SUMMARY_BYTES`] (4 KiB, the summarizer output bound).
//! - Record bodies: inline only; artifact references fail closed (their bytes
//!   live in the in-memory [`ArtifactStore`], not here).
//! - Record count per view: [`MAX_CONTEXT_RECORDS`] (128), enforced on write
//!   and on load.
//! - Stored `records_json` image: at most [`MAX_RECORDS_JSON_BYTES`] (8 MiB),
//!   size-checked before parsing on load. Tombstoned records are stripped
//!   from the image on write and filtered on load, so absence metadata never
//!   carries payload bytes.
//! - Tombstones: at most [`MAX_STORED_TOMBSTONES`] ids (each non-empty, at
//!   most [`MAX_TOMBSTONE_ID_LEN`] bytes), enforced on the merged total on
//!   write and on load. The load-time retain filter hashes the tombstone set,
//!   so the pass costs O(spans x sources) lookups within the caps above.
//!
//! ## Fail-closed rules
//!
//! - Malformed, corrupt, or schema-incompatible state yields
//!   [`DurabilityError::Corrupt`]; the file is never reset or repaired.
//! - A second open while the first lives yields [`DurabilityError::WriterBusy`].
//! - Generation mismatch yields [`DurabilityError::StaleGeneration`]; nothing is written.
//! - No wall clock: every timestamp is caller-supplied.
//!
//! ## Pragmas
//!
//! Same durable profile as the content store (WAL, FULL, FK ON, nonzero busy
//! timeout 5 s, EXCLUSIVE single-writer); see `content_store` docs.

use std::collections::HashSet;
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use bitty_ai_runtime::compression::{
    CompressedSpan, CompressedView, MAX_SPAN_ID_LEN, MAX_SPANS, RetentionClass, RetentionTags,
};
use bitty_ai_runtime::context::{
    ContextPriority, ContextRecord, MAX_CONTEXT_RECORDS, MAX_SUMMARY_BYTES as RUNTIME_MAX_SUMMARY,
    RecordBody, StableId,
};
use rusqlite::{Connection, OptionalExtension, params};

use crate::content_store::DURABLE_BUSY_TIMEOUT_MS;

/// Durable profile marker stored in `compaction_meta.profile`.
pub const COMPACTION_PROFILE: &str = "durable-v1";

/// Maximum stored tombstone ids, enforced on write and on load.
///
/// Tombstones merge monotonically (never removed), so an explicit fail-closed
/// cap keeps the tombstone table, the in-memory set, and the span/record
/// retain filters bounded. `4096 * 128 B` ids stay well under one megabyte.
pub const MAX_STORED_TOMBSTONES: usize = 4096;

/// Maximum bytes for one tombstone id, enforced on write and on load.
pub const MAX_TOMBSTONE_ID_LEN: usize = 128;

/// Maximum stored `records_json` bytes, checked before parsing on load.
///
/// A valid view holds at most [`MAX_CONTEXT_RECORDS`] records with inline
/// bodies bounded by the runtime record-body limit; 8 MiB covers that worst
/// case with headroom while keeping a hostile file from exhausting memory.
pub const MAX_RECORDS_JSON_BYTES: usize = 8 * 1024 * 1024;

/// Typed fail-closed durability failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurabilityError {
    /// Malformed, corrupt, or schema-incompatible state. Fail closed.
    Corrupt { detail: String },
    /// Another live writer holds this file's single-writer lock.
    WriterBusy,
    /// Generation fence mismatch; nothing was written.
    StaleGeneration { expected: u64, found: u64 },
    /// Any other storage failure (reported, never retried silently).
    Storage { detail: String },
}

impl fmt::Display for DurabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Corrupt { detail } => write!(f, "compaction store corrupt: {detail}"),
            Self::WriterBusy => write!(f, "another writer holds the single-writer lock"),
            Self::StaleGeneration { expected, found } => {
                write!(
                    f,
                    "stale compaction generation: expected {expected}, found {found}"
                )
            }
            Self::Storage { detail } => write!(f, "compaction storage failure: {detail}"),
        }
    }
}

impl std::error::Error for DurabilityError {}

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

fn map_storage(err: rusqlite::Error) -> DurabilityError {
    if is_busy(&err) {
        return DurabilityError::WriterBusy;
    }
    if let rusqlite::Error::SqliteFailure(e, _) = &err {
        if matches!(
            e.code,
            rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
        ) {
            return DurabilityError::Corrupt {
                detail: err.to_string(),
            };
        }
    }
    DurabilityError::Storage {
        detail: err.to_string(),
    }
}

impl From<rusqlite::Error> for DurabilityError {
    fn from(err: rusqlite::Error) -> Self {
        map_storage(err)
    }
}

fn check_magic(path: &Path) -> Result<(), DurabilityError> {
    // Bounded header probe: only the first 16 bytes are ever read, so the
    // check costs O(1) memory no matter how large the database grows.
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(DurabilityError::Corrupt {
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
                return Err(DurabilityError::Corrupt {
                    detail: format!("unreadable database file: {e}"),
                });
            }
        }
    }
    if read == 0 {
        return Ok(());
    }
    if read < 16 || header != *b"SQLite format 3\0" {
        return Err(DurabilityError::Corrupt {
            detail: "file is not a SQLite database".to_owned(),
        });
    }
    Ok(())
}

const COMPACTION_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS compaction_spans (
    span_id TEXT PRIMARY KEY,
    source_ids_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    retention TEXT NOT NULL,
    source_deadline_ms INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    is_untrusted INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS compaction_tombstones (
    id TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS compaction_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);";

fn apply_pragmas(conn: &Connection) -> Result<(), DurabilityError> {
    conn.busy_timeout(Duration::from_millis(DURABLE_BUSY_TIMEOUT_MS))
        .map_err(map_storage)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = FULL;
         PRAGMA foreign_keys = ON;
         PRAGMA locking_mode = EXCLUSIVE;",
    )
    .map_err(map_storage)?;
    Ok(())
}

fn admit_or_init(conn: &Connection) -> Result<(), DurabilityError> {
    let mut stmt = conn
        .prepare(
            "SELECT type, name FROM sqlite_master
             WHERE name COLLATE NOCASE IN ('compaction_spans', 'compaction_tombstones', 'compaction_meta')",
        )
        .map_err(map_storage)?;
    let mut matches: Vec<(String, String)> = Vec::new();
    let mut rows = stmt.query([]).map_err(map_storage)?;
    while let Some(row) = rows.next().map_err(map_storage)? {
        matches.push((
            row.get::<_, String>(0).map_err(map_storage)?,
            row.get::<_, String>(1).map_err(map_storage)?,
        ));
    }
    drop(rows);
    drop(stmt);
    let mut present = Vec::new();
    for expected in [
        "compaction_spans",
        "compaction_tombstones",
        "compaction_meta",
    ] {
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
            return Err(DurabilityError::Corrupt {
                detail: format!(
                    "compaction name '{expected}' is occupied by {ty} '{name}', not the exact-case table"
                ),
            });
        }
    }
    if present.is_empty() {
        return Ok(());
    }
    if present.len() != 3 {
        return Err(DurabilityError::Corrupt {
            detail: format!("partial compaction schema; present tables: {present:?}"),
        });
    }
    verify_span_shape(conn)?;
    verify_profile(conn)?;
    Ok(())
}

fn verify_span_shape(conn: &Connection) -> Result<(), DurabilityError> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(compaction_spans)")
        .map_err(map_storage)?;
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
        .map_err(map_storage)?;
    for row in rows {
        cols.push(row.map_err(map_storage)?);
    }
    drop(stmt);
    let expected: &[(&str, &str, bool, u32)] = &[
        ("span_id", "TEXT", false, 1),
        ("source_ids_json", "TEXT", true, 0),
        ("summary", "TEXT", true, 0),
        ("retention", "TEXT", true, 0),
        ("source_deadline_ms", "INTEGER", true, 0),
        ("created_at_ms", "INTEGER", true, 0),
        ("is_untrusted", "INTEGER", true, 0),
    ];
    if cols.len() != expected.len() {
        return Err(DurabilityError::Corrupt {
            detail: format!(
                "table compaction_spans has {} columns, expected {}",
                cols.len(),
                expected.len()
            ),
        });
    }
    for (i, (name, ty, notnull, pk)) in expected.iter().enumerate() {
        let (aname, aty, anotnull, apk) = &cols[i];
        if aname != *name || aty != *ty || anotnull != notnull || apk != pk {
            return Err(DurabilityError::Corrupt {
                detail: format!("table compaction_spans column {i} shape mismatch"),
            });
        }
    }
    Ok(())
}

fn verify_profile(conn: &Connection) -> Result<(), DurabilityError> {
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM compaction_meta WHERE key = 'profile'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(map_storage)?;
    match stored.as_deref() {
        Some(COMPACTION_PROFILE) => Ok(()),
        Some(other) => Err(DurabilityError::Corrupt {
            detail: format!("unrecognized compaction profile '{other}'; no migration exists"),
        }),
        None => Ok(()),
    }
}

fn claim_writer(conn: &Connection) -> Result<(), DurabilityError> {
    conn.busy_timeout(Duration::ZERO).map_err(map_storage)?;
    let claim = conn.execute(
        "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('profile', ?1)",
        params![COMPACTION_PROFILE],
    );
    conn.busy_timeout(Duration::from_millis(DURABLE_BUSY_TIMEOUT_MS))
        .map_err(map_storage)?;
    claim.map_err(map_storage)?;
    Ok(())
}

fn retention_to_str(class: RetentionClass) -> &'static str {
    match class {
        RetentionClass::Pinned => "Pinned",
        RetentionClass::Recent => "Recent",
        RetentionClass::Normal => "Normal",
        RetentionClass::Ephemeral => "Ephemeral",
    }
}

fn retention_from_str(s: &str) -> Result<RetentionClass, DurabilityError> {
    match s {
        "Pinned" => Ok(RetentionClass::Pinned),
        "Recent" => Ok(RetentionClass::Recent),
        "Normal" => Ok(RetentionClass::Normal),
        "Ephemeral" => Ok(RetentionClass::Ephemeral),
        other => Err(DurabilityError::Corrupt {
            detail: format!("unknown retention class '{other}'"),
        }),
    }
}

fn priority_to_str(p: ContextPriority) -> &'static str {
    match p {
        ContextPriority::Low => "Low",
        ContextPriority::Normal => "Normal",
        ContextPriority::High => "High",
        ContextPriority::Critical => "Critical",
    }
}

fn priority_from_str(s: &str) -> Result<ContextPriority, DurabilityError> {
    match s {
        "Low" => Ok(ContextPriority::Low),
        "Normal" => Ok(ContextPriority::Normal),
        "High" => Ok(ContextPriority::High),
        "Critical" => Ok(ContextPriority::Critical),
        other => Err(DurabilityError::Corrupt {
            detail: format!("unknown priority '{other}'"),
        }),
    }
}

/// Durable compaction store: spans, tombstones, tags, records, and driver state.
pub struct CompactionStore {
    conn: Connection,
}

impl CompactionStore {
    /// Open (or create) the compaction database at `path` and claim the single-writer lock.
    ///
    /// The busy timeout stays zero through admission and schema init so a
    /// second open while the first lives fails fast with
    /// [`DurabilityError::WriterBusy`]; steady state restores the nonzero
    /// profile timeout only after the writer claim succeeds.
    pub fn open(path: &Path) -> Result<Self, DurabilityError> {
        check_magic(path)?;
        let conn = Connection::open(path).map_err(map_storage)?;
        conn.busy_timeout(Duration::ZERO).map_err(map_storage)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(map_storage)?;
        admit_or_init(&conn)?;
        apply_pragmas(&conn)?;
        conn.busy_timeout(Duration::ZERO).map_err(map_storage)?;
        conn.execute_batch(COMPACTION_SCHEMA).map_err(map_storage)?;
        verify_profile(&conn)?;
        claim_writer(&conn)?;
        Ok(Self { conn })
    }

    /// Open the formal durable path (same profile as [`Self::open`]).
    pub fn open_durable(path: &Path) -> Result<Self, DurabilityError> {
        Self::open(path)
    }

    /// Open an in-memory store (testing only; no writer contention).
    pub fn open_in_memory() -> Result<Self, DurabilityError> {
        let conn = Connection::open_in_memory().map_err(map_storage)?;
        apply_pragmas(&conn)?;
        conn.execute_batch(COMPACTION_SCHEMA).map_err(map_storage)?;
        conn.execute(
            "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('profile', ?1)",
            params![COMPACTION_PROFILE],
        )
        .map_err(map_storage)?;
        Ok(Self { conn })
    }

    /// Atomically record a compressed view: spans, tombstones, tags, and records.
    ///
    /// Fails closed (nothing written) on bound violations or generation
    /// mismatch against the stored `generation` marker. Tombstones in the
    /// view are merged into the stored set (never removed); spans are
    /// replaced wholesale for the recorded view. Records whose ids are
    /// tombstoned are stripped before persisting, so deleted or expired
    /// payload bytes never reach the disk image.
    pub fn record_view(&mut self, view: &CompressedView) -> Result<(), DurabilityError> {
        if view.spans.len() > MAX_SPANS {
            return Err(DurabilityError::Storage {
                detail: format!(
                    "compressed span count {} exceeds limit {MAX_SPANS}",
                    view.spans.len()
                ),
            });
        }
        if view.tombstones.len() > MAX_STORED_TOMBSTONES {
            return Err(DurabilityError::Storage {
                detail: format!(
                    "tombstone count {} exceeds limit {MAX_STORED_TOMBSTONES}",
                    view.tombstones.len()
                ),
            });
        }
        for span in &view.spans {
            if span.span_id.len() > MAX_SPAN_ID_LEN || span.span_id.is_empty() {
                return Err(DurabilityError::Corrupt {
                    detail: format!("span id {:?} violates length bound", span.span_id),
                });
            }
            if span.summary.len() > RUNTIME_MAX_SUMMARY {
                return Err(DurabilityError::Storage {
                    detail: format!(
                        "span {} summary {} bytes exceeds {RUNTIME_MAX_SUMMARY} byte limit",
                        span.span_id,
                        span.summary.len()
                    ),
                });
            }
            if span.source_ids.len() > MAX_CONTEXT_RECORDS {
                return Err(DurabilityError::Storage {
                    detail: format!(
                        "span {} source count {} exceeds limit {MAX_CONTEXT_RECORDS}",
                        span.span_id,
                        span.source_ids.len()
                    ),
                });
            }
            for source in &span.source_ids {
                if source.is_empty() || source.len() > MAX_TOMBSTONE_ID_LEN {
                    return Err(DurabilityError::Corrupt {
                        detail: format!("span {} source id violates length bound", span.span_id),
                    });
                }
            }
        }
        for id in &view.tombstones {
            if id.is_empty() || id.len() > MAX_TOMBSTONE_ID_LEN {
                return Err(DurabilityError::Storage {
                    detail: "tombstone id violates bound".to_owned(),
                });
            }
        }
        // Strip tombstoned records up front: absence wins over payload, so
        // deleted or expired bodies never reach the disk image.
        let tomb_set: HashSet<&str> = view.tombstones.iter().map(String::as_str).collect();
        let live_records: Vec<ContextRecord> = view
            .records
            .iter()
            .filter(|record| !tomb_set.contains(record.id.as_str()))
            .cloned()
            .collect();
        if live_records.len() > MAX_CONTEXT_RECORDS {
            return Err(DurabilityError::Storage {
                detail: format!(
                    "record count {} exceeds limit {MAX_CONTEXT_RECORDS}",
                    live_records.len()
                ),
            });
        }
        // Generation fence: the live view's max record generation must match
        // the stored marker when one is set; otherwise refuse without writing.
        let view_generation = live_records.iter().map(|r| r.generation).max().unwrap_or(0);
        if let Some(stored) = self.get_generation_inner()? {
            if stored != view_generation && !live_records.is_empty() {
                return Err(DurabilityError::StaleGeneration {
                    expected: stored,
                    found: view_generation,
                });
            }
        }
        let stored_view = CompressedView {
            records: live_records,
            spans: view.spans.clone(),
            tombstones: view.tombstones.clone(),
            tags: view.tags.clone(),
        };
        let records_json = serialize_records(&stored_view.records)?;
        let tags_json = serialize_tags(&stored_view)?;

        let tx = self.conn.transaction().map_err(map_storage)?;
        tx.execute("DELETE FROM compaction_spans", [])
            .map_err(map_storage)?;
        for span in &stored_view.spans {
            let source_json =
                serde_json::to_string(&span.source_ids).map_err(|e| DurabilityError::Storage {
                    detail: format!("span source_ids serialization: {e}"),
                })?;
            tx.execute(
                "INSERT OR REPLACE INTO compaction_spans
                 (span_id, source_ids_json, summary, retention, source_deadline_ms, created_at_ms, is_untrusted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    span.span_id,
                    source_json,
                    span.summary,
                    retention_to_str(span.retention),
                    span.source_deadline_ms as i64,
                    span.created_at_ms as i64,
                    i64::from(span.is_untrusted_surface),
                ],
            )
            .map_err(map_storage)?;
        }
        for id in &stored_view.tombstones {
            tx.execute(
                "INSERT OR IGNORE INTO compaction_tombstones (id) VALUES (?1)",
                params![id],
            )
            .map_err(map_storage)?;
        }
        // The tombstone set merges monotonically, so enforce the cap on the
        // merged total: roll back (drop the transaction uncommitted) on exceed.
        let merged_tombstones: i64 = tx
            .query_row("SELECT COUNT(*) FROM compaction_tombstones", [], |row| {
                row.get(0)
            })
            .map_err(map_storage)?;
        if merged_tombstones > MAX_STORED_TOMBSTONES as i64 {
            return Err(DurabilityError::Storage {
                detail: format!(
                    "merged tombstone count {merged_tombstones} exceeds limit {MAX_STORED_TOMBSTONES}"
                ),
            });
        }
        tx.execute(
            "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('records_json', ?1)",
            params![records_json],
        )
        .map_err(map_storage)?;
        tx.execute(
            "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('retention_tags_json', ?1)",
            params![tags_json],
        )
        .map_err(map_storage)?;
        if stored_view.records.is_empty() {
            // Keep an existing generation marker when recording an empty view
            // (tombstone-only update); otherwise set it from the view.
        } else {
            tx.execute(
                "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('generation', ?1)",
                params![view_generation.to_string()],
            )
            .map_err(map_storage)?;
        }
        tx.commit().map_err(map_storage)?;
        Ok(())
    }

    /// Reload the recorded view (spans, tombstones, tags, records).
    ///
    /// Never contacts a summarizer: summaries come from storage verbatim.
    /// Tombstoned spans stay tombstoned (never resurrected), and tombstoned
    /// or expired records never come back with payload bytes: records whose
    /// ids are tombstoned are dropped on load, and the tombstone appliers
    /// strip them from the persisted image so they do not survive on disk.
    ///
    /// Fail-closed load bounds: more than [`MAX_SPANS`] stored spans, more
    /// than [`MAX_STORED_TOMBSTONES`] tombstones, a `records_json` image
    /// larger than [`MAX_RECORDS_JSON_BYTES`], more than
    /// [`MAX_CONTEXT_RECORDS`] stored records, or any out-of-bound
    /// id/summary field yields [`DurabilityError::Corrupt`].
    pub fn load_view(&self) -> Result<CompressedView, DurabilityError> {
        let span_rows: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM compaction_spans", [], |row| {
                row.get(0)
            })
            .map_err(map_storage)?;
        if span_rows > MAX_SPANS as i64 {
            return Err(DurabilityError::Corrupt {
                detail: format!("stored span count {span_rows} exceeds limit {MAX_SPANS}"),
            });
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT span_id, source_ids_json, summary, retention, source_deadline_ms, created_at_ms, is_untrusted
                 FROM compaction_spans ORDER BY span_id ASC",
            )
            .map_err(map_storage)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            })
            .map_err(map_storage)?;
        let mut spans = Vec::new();
        for row in rows {
            let (span_id, source_json, summary, retention, deadline, created, untrusted) =
                row.map_err(map_storage)?;
            if span_id.is_empty() || span_id.len() > MAX_SPAN_ID_LEN {
                return Err(DurabilityError::Corrupt {
                    detail: "stored span id violates bound".to_owned(),
                });
            }
            if summary.len() > RUNTIME_MAX_SUMMARY {
                return Err(DurabilityError::Corrupt {
                    detail: "stored span summary exceeds bound".to_owned(),
                });
            }
            if source_json.len() > MAX_RECORDS_JSON_BYTES {
                return Err(DurabilityError::Corrupt {
                    detail: "stored source_ids_json exceeds bound".to_owned(),
                });
            }
            let source_ids: Vec<String> =
                serde_json::from_str(&source_json).map_err(|e| DurabilityError::Corrupt {
                    detail: format!("stored source_ids_json malformed: {e}"),
                })?;
            if source_ids.len() > MAX_CONTEXT_RECORDS {
                return Err(DurabilityError::Corrupt {
                    detail: "stored span source count exceeds bound".to_owned(),
                });
            }
            for source in &source_ids {
                if source.is_empty() || source.len() > MAX_TOMBSTONE_ID_LEN {
                    return Err(DurabilityError::Corrupt {
                        detail: "stored span source id violates bound".to_owned(),
                    });
                }
            }
            spans.push(CompressedSpan {
                span_id,
                source_ids,
                summary,
                is_untrusted_surface: untrusted != 0,
                retention: retention_from_str(&retention)?,
                source_deadline_ms: deadline as u64,
                created_at_ms: created as u64,
            });
        }
        let tombstone_rows: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM compaction_tombstones", [], |row| {
                row.get(0)
            })
            .map_err(map_storage)?;
        if tombstone_rows > MAX_STORED_TOMBSTONES as i64 {
            return Err(DurabilityError::Corrupt {
                detail: format!(
                    "stored tombstone count {tombstone_rows} exceeds limit {MAX_STORED_TOMBSTONES}"
                ),
            });
        }
        let mut tstmt = self
            .conn
            .prepare("SELECT id FROM compaction_tombstones ORDER BY id ASC")
            .map_err(map_storage)?;
        let trows = tstmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(map_storage)?;
        let mut tombstones = Vec::new();
        for row in trows {
            let id: String = row.map_err(map_storage)?;
            if id.is_empty() || id.len() > MAX_TOMBSTONE_ID_LEN {
                return Err(DurabilityError::Corrupt {
                    detail: "stored tombstone id violates bound".to_owned(),
                });
            }
            tombstones.push(id);
        }
        // Drop spans killed by tombstones (deletion propagation survives reload):
        // a tombstoned span id or any tombstoned source id invalidates the span.
        // The tombstone set is hashed, so the filter costs O(spans x sources)
        // hash lookups; both dimensions are capped above (MAX_SPANS spans of at
        // most MAX_CONTEXT_RECORDS sources over at most MAX_STORED_TOMBSTONES
        // ids), keeping the pass trivially cheap.
        let tomb_set: HashSet<&str> = tombstones.iter().map(String::as_str).collect();
        spans.retain(|span| {
            if tomb_set.contains(span.span_id.as_str()) {
                return false;
            }
            !span
                .source_ids
                .iter()
                .any(|source| tomb_set.contains(source.as_str()))
        });
        let mut records = self.load_records_inner()?;
        // Absence wins over payload: tombstoned or expired ids (expired ids
        // reach the store as tombstones) never reload with body bytes, even
        // if an older image still carries them.
        records.retain(|record| !tomb_set.contains(record.id.as_str()));
        let tags = self.load_tags_inner()?;
        Ok(CompressedView {
            records,
            spans,
            tombstones,
            tags,
        })
    }

    fn load_records_inner(&self) -> Result<Vec<ContextRecord>, DurabilityError> {
        // Check the image size in SQL first so a hostile records_json cannot
        // force an unbounded allocation before the cap is enforced.
        let byte_len: Option<i64> = self
            .conn
            .query_row(
                "SELECT LENGTH(value) FROM compaction_meta WHERE key = 'records_json'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        if let Some(len) = byte_len {
            if len > MAX_RECORDS_JSON_BYTES as i64 {
                return Err(DurabilityError::Corrupt {
                    detail: format!(
                        "stored records_json {len} bytes exceeds limit {MAX_RECORDS_JSON_BYTES}"
                    ),
                });
            }
        }
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'records_json'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        let Some(json) = raw else {
            return Ok(Vec::new());
        };
        let records = deserialize_records(&json)?;
        if records.len() > MAX_CONTEXT_RECORDS {
            return Err(DurabilityError::Corrupt {
                detail: format!(
                    "stored record count {} exceeds limit {MAX_CONTEXT_RECORDS}",
                    records.len()
                ),
            });
        }
        Ok(records)
    }

    fn load_tags_inner(&self) -> Result<RetentionTags, DurabilityError> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'retention_tags_json'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        let Some(json) = raw else {
            return Ok(RetentionTags::new());
        };
        let map: std::collections::BTreeMap<String, String> =
            serde_json::from_str(&json).map_err(|e| DurabilityError::Corrupt {
                detail: format!("stored retention tags malformed: {e}"),
            })?;
        let mut tags = RetentionTags::new();
        for (id, class) in map {
            tags.set(id, retention_from_str(&class)?);
        }
        Ok(tags)
    }

    /// Persist tombstones (idempotent merge; first write wins per id set).
    ///
    /// Records whose ids are tombstoned are stripped from the persisted
    /// `records_json` image in the same transaction, so deleted or expired
    /// payload bytes stop surviving on disk the moment absence is recorded.
    pub fn add_tombstones(&mut self, ids: &[&str]) -> Result<(), DurabilityError> {
        if ids.is_empty() {
            return Ok(());
        }
        let tx = self.conn.transaction().map_err(map_storage)?;
        for id in ids {
            if id.is_empty() || id.len() > MAX_TOMBSTONE_ID_LEN {
                return Err(DurabilityError::Storage {
                    detail: "tombstone id violates bound".to_owned(),
                });
            }
            tx.execute(
                "INSERT OR IGNORE INTO compaction_tombstones (id) VALUES (?1)",
                params![id],
            )
            .map_err(map_storage)?;
        }
        // Enforce the cap on the merged total; dropping the transaction
        // uncommitted rolls everything back.
        let merged_tombstones: i64 = tx
            .query_row("SELECT COUNT(*) FROM compaction_tombstones", [], |row| {
                row.get(0)
            })
            .map_err(map_storage)?;
        if merged_tombstones > MAX_STORED_TOMBSTONES as i64 {
            return Err(DurabilityError::Storage {
                detail: format!(
                    "merged tombstone count {merged_tombstones} exceeds limit {MAX_STORED_TOMBSTONES}"
                ),
            });
        }
        // Strip newly absent records from the persisted image: absence wins
        // over payload, so no tombstoned body survives on disk.
        let tombed: Vec<String> = {
            let mut stmt = tx
                .prepare("SELECT id FROM compaction_tombstones")
                .map_err(map_storage)?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(map_storage)?;
            let mut all = Vec::new();
            for row in rows {
                all.push(row.map_err(map_storage)?);
            }
            all
        };
        let tomb_set: HashSet<&str> = tombed.iter().map(String::as_str).collect();
        let raw: Option<String> = tx
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'records_json'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        if let Some(json) = raw {
            let mut records = deserialize_records(&json)?;
            let kept = records.len();
            records.retain(|record| !tomb_set.contains(record.id.as_str()));
            if records.len() != kept {
                let stripped = serialize_records(&records)?;
                tx.execute(
                    "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('records_json', ?1)",
                    params![stripped],
                )
                .map_err(map_storage)?;
            }
        }
        // Purge spans derived from newly absent sources: source-delete kills
        // the derived span, so a tombstoned source must not leave its summary
        // readable in the spans table (forensic resurrection). Span rows are
        // bounded (MAX_SPANS enforced on write and load), so the scan is O(1).
        let doomed: Vec<String> = {
            let mut stmt = tx
                .prepare("SELECT span_id, source_ids_json FROM compaction_spans")
                .map_err(map_storage)?;
            let rows = stmt
                .query_map([], |row| {
                    let span_id: String = row.get(0)?;
                    let source_json: String = row.get(1)?;
                    Ok((span_id, source_json))
                })
                .map_err(map_storage)?;
            let mut doomed = Vec::new();
            for row in rows {
                let (span_id, source_json) = row.map_err(map_storage)?;
                let source_ids: Vec<String> =
                    serde_json::from_str(&source_json).map_err(|e| DurabilityError::Corrupt {
                        detail: format!("stored source_ids_json malformed: {e}"),
                    })?;
                if tomb_set.contains(span_id.as_str())
                    || source_ids
                        .iter()
                        .any(|source| tomb_set.contains(source.as_str()))
                {
                    doomed.push(span_id);
                }
            }
            doomed
        };
        for span_id in &doomed {
            tx.execute(
                "DELETE FROM compaction_spans WHERE span_id = ?1",
                params![span_id],
            )
            .map_err(map_storage)?;
        }
        tx.commit().map_err(map_storage)?;
        Ok(())
    }

    /// Read the ineffective-compression strike counter (0 when unset).
    pub fn get_ineffective_strikes(&self) -> Result<u64, DurabilityError> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'ineffective_strikes'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        match raw {
            None => Ok(0),
            Some(s) => s.parse::<u64>().map_err(|_| DurabilityError::Corrupt {
                detail: "stored ineffective_strikes malformed".to_owned(),
            }),
        }
    }

    /// Overwrite the strike counter.
    pub fn set_ineffective_strikes(&mut self, strikes: u64) -> Result<(), DurabilityError> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('ineffective_strikes', ?1)",
                params![strikes.to_string()],
            )
            .map_err(map_storage)?;
        Ok(())
    }

    /// Read the previous compaction summary, if one was stored.
    pub fn get_previous_summary(&self) -> Result<Option<String>, DurabilityError> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'previous_summary'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        Ok(raw)
    }

    /// Store the previous summary (bounded like all summaries).
    pub fn set_previous_summary(&mut self, summary: Option<&str>) -> Result<(), DurabilityError> {
        match summary {
            None => {
                self.conn
                    .execute(
                        "DELETE FROM compaction_meta WHERE key = 'previous_summary'",
                        [],
                    )
                    .map_err(map_storage)?;
                Ok(())
            }
            Some(text) => {
                if text.len() > RUNTIME_MAX_SUMMARY {
                    return Err(DurabilityError::Storage {
                        detail: format!(
                            "previous summary {} bytes exceeds {RUNTIME_MAX_SUMMARY} byte limit",
                            text.len()
                        ),
                    });
                }
                self.conn
                    .execute(
                        "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('previous_summary', ?1)",
                        params![text],
                    )
                    .map_err(map_storage)?;
                Ok(())
            }
        }
    }

    fn get_generation_inner(&self) -> Result<Option<u64>, DurabilityError> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'generation'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        match raw {
            None => Ok(None),
            Some(s) => s
                .parse::<u64>()
                .map_err(|_| DurabilityError::Corrupt {
                    detail: "stored generation malformed".to_owned(),
                })
                .map(Some),
        }
    }

    /// Read the stored generation marker, if one was recorded.
    pub fn get_generation(&self) -> Result<Option<u64>, DurabilityError> {
        self.get_generation_inner()
    }

    /// Read the stored window size, if one was recorded.
    pub fn get_window_bytes(&self) -> Result<Option<usize>, DurabilityError> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM compaction_meta WHERE key = 'window_bytes'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_storage)?;
        match raw {
            None => Ok(None),
            Some(s) => s
                .parse::<usize>()
                .map_err(|_| DurabilityError::Corrupt {
                    detail: "stored window_bytes malformed".to_owned(),
                })
                .map(Some),
        }
    }

    /// Store the window size.
    pub fn set_window_bytes(&mut self, window: usize) -> Result<(), DurabilityError> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO compaction_meta (key, value) VALUES ('window_bytes', ?1)",
                params![window.to_string()],
            )
            .map_err(map_storage)?;
        Ok(())
    }
}

fn serialize_records(records: &[ContextRecord]) -> Result<String, DurabilityError> {
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let body = match &record.body {
            RecordBody::Inline(bytes) => {
                serde_json::json!({"type": "inline", "hex": hex::encode(bytes)})
            }
            RecordBody::Artifact(reference) => {
                return Err(DurabilityError::Storage {
                    detail: format!(
                        "record {} has an artifact body ({}); durable compaction stores inline bodies only",
                        record.id,
                        reference.as_str()
                    ),
                });
            }
        };
        out.push(serde_json::json!({
            "id": record.id,
            "provider": record.provider,
            "owner": record.owner.as_str(),
            "generation": record.generation,
            "collected_at_ms": record.collected_at_ms,
            "priority": priority_to_str(record.priority),
            "summary": record.summary,
            "body": body,
            "supersedes": record.supersedes,
            "is_untrusted_surface": record.is_untrusted_surface,
        }));
    }
    serde_json::to_string(&out).map_err(|e| DurabilityError::Storage {
        detail: format!("records serialization: {e}"),
    })
}

fn deserialize_records(json: &str) -> Result<Vec<ContextRecord>, DurabilityError> {
    let corrupt = |e: String| DurabilityError::Corrupt {
        detail: format!("stored records malformed: {e}"),
    };
    let items: Vec<serde_json::Value> =
        serde_json::from_str(json).map_err(|e| corrupt(e.to_string()))?;
    let mut records = Vec::with_capacity(items.len());
    for item in items {
        let id = item
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt("record id missing".to_owned()))?
            .to_owned();
        let provider = item
            .get("provider")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt("provider missing".to_owned()))?
            .to_owned();
        let owner = item
            .get("owner")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt("owner missing".to_owned()))?
            .to_owned();
        let generation = item
            .get("generation")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| corrupt("generation missing".to_owned()))?;
        let collected_at_ms = item
            .get("collected_at_ms")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| corrupt("collected_at_ms missing".to_owned()))?;
        let priority = item
            .get("priority")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt("priority missing".to_owned()))?;
        let summary = item
            .get("summary")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt("summary missing".to_owned()))?
            .to_owned();
        let is_untrusted_surface = item
            .get("is_untrusted_surface")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| corrupt("is_untrusted_surface missing or not a boolean".to_owned()))?;
        let supersedes = match item.get("supersedes") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(_) => return Err(corrupt("supersedes malformed".to_owned())),
        };
        let body = item
            .get("body")
            .ok_or_else(|| corrupt("body missing".to_owned()))?;
        let body_type = body
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt("body type missing".to_owned()))?;
        let body = match body_type {
            "inline" => {
                let hexed = body
                    .get("hex")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| corrupt("body hex missing".to_owned()))?;
                let bytes =
                    hex::decode(hexed).map_err(|e| corrupt(format!("body hex malformed: {e}")))?;
                RecordBody::Inline(bytes)
            }
            other => return Err(corrupt(format!("unknown body type '{other}'"))),
        };
        let record = ContextRecord {
            id,
            provider,
            owner: StableId::new(owner).map_err(|e| corrupt(format!("owner invalid: {e}")))?,
            generation,
            collected_at_ms,
            priority: priority_from_str(priority)?,
            summary,
            body,
            supersedes,
            is_untrusted_surface,
        };
        record.validate().map_err(|e| DurabilityError::Corrupt {
            detail: format!("stored record invalid: {e}"),
        })?;
        records.push(record);
    }
    Ok(records)
}

fn serialize_tags(view: &CompressedView) -> Result<String, DurabilityError> {
    let mut ids: Vec<&str> = Vec::new();
    for record in &view.records {
        if !ids.contains(&record.id.as_str()) {
            ids.push(record.id.as_str());
        }
    }
    for span in &view.spans {
        for source in &span.source_ids {
            if !ids.contains(&source.as_str()) {
                ids.push(source.as_str());
            }
        }
    }
    for id in &view.tombstones {
        if !ids.contains(&id.as_str()) {
            ids.push(id.as_str());
        }
    }
    let mut map = std::collections::BTreeMap::new();
    for id in ids {
        let class = view.tags.get(id);
        if class != RetentionClass::Normal {
            map.insert(id.to_owned(), retention_to_str(class).to_owned());
        }
    }
    serde_json::to_string(&map).map_err(|e| DurabilityError::Storage {
        detail: format!("tags serialization: {e}"),
    })
}

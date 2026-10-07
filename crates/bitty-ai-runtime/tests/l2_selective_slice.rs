//! Minimal L2 selective-compaction slice (AI-0177).
//!
//! Covers the trigger ([`effective_reserve_bytes`], [`should_compact`]),
//! the head/recent selection ([`select_compaction_window`]), and one
//! guarded driver pass ([`compact_selective`]) over the `compression`
//! prototype:
//!
//! - Trigger boundary: equality with the window does not compact, one byte
//!   over does, and any zero input fails closed.
//! - Tail protection: the recent tail keeps the byte budget AND at least
//!   [`MIN_KEEP_RECORDS`]; protected ids always survive regardless of
//!   budget; all-protected or empty input selects an empty head and never
//!   contacts the summarizer.
//! - Fail-closed: stale generations, over-bound records, exhausted scripts,
//!   empty or whitespace-only summaries, duplicate ids, over-bound previous
//!   summaries, and doomed windows all refuse without partial output.
//! - Provenance: untrusted-surface OR-marking, most-restrictive retention
//!   inheritance, and `source_deadline_ms` anchored at the minimum source
//!   `collected_at_ms`.
//! - Cache stability: summaries land in the dynamic turn region only, so
//!   the [`CacheKey`] stable prefix is unchanged by compaction.
//!
//! Std only. No clock, no threads, no async: every timestamp is
//! caller-supplied (`now_ms`), every lookup is a linear scan over a `Vec`
//! (no `HashMap`, no `RandomState`).

use std::cell::RefCell;

use bitty_ai_runtime::{
    CacheKey, CacheScope, CompactionOutcome, CompressionError, ContextError, ContextPriority,
    ContextRecord, DEFAULT_KEEP_RECENT_BYTES, DEFAULT_RESERVE_BYTES, FakeSummarizer, LayerInput,
    MAX_SUMMARY_BYTES, MIN_KEEP_RECORDS, PromptLayer, PromptSnapshot, RESERVE_FRACTION_DENOMINATOR,
    RESERVE_FRACTION_NUMERATOR, RecordBody, RetentionClass, RetentionTags,
    SelectiveCompactionConfig, SpanRange, StableId, SummarizeInput, Summarizer, assemble_prompt,
    compact_selective, compress_records, effective_reserve_bytes, select_compaction_window,
    should_compact,
};

/// One record with an exact footprint of `summary_len + body_len` bytes.
fn record(id: &str, summary_len: usize, body_len: usize) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "workspace".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms: 100,
        priority: ContextPriority::Normal,
        summary: "s".repeat(summary_len),
        body: RecordBody::Inline(vec![b'x'; body_len]),
        supersedes: None,
        is_untrusted_surface: false,
    }
}

fn untrusted(id: &str, collected_at_ms: u64) -> ContextRecord {
    ContextRecord {
        id: id.to_owned(),
        provider: "terminal".to_owned(),
        owner: StableId::new("term-1").expect("valid stable id"),
        generation: 1,
        collected_at_ms,
        priority: ContextPriority::Normal,
        summary: "tool output".to_owned(),
        body: RecordBody::Inline(vec![b'e'; 10]),
        supersedes: None,
        is_untrusted_surface: true,
    }
}

/// Test-only recording double: serves queued outputs in call order, fails
/// closed when exhausted, and records every input for plumbing assertions.
struct RecordingSummarizer {
    outputs: Vec<String>,
    seen: RefCell<Vec<SummarizeInput>>,
}

impl RecordingSummarizer {
    fn new(outputs: &[&str]) -> Self {
        Self {
            outputs: outputs.iter().map(|text| (*text).to_owned()).collect(),
            seen: RefCell::new(Vec::new()),
        }
    }

    fn calls(&self) -> usize {
        self.seen.borrow().len()
    }
}

impl Summarizer for RecordingSummarizer {
    fn summarize(&self, input: &SummarizeInput) -> Result<String, CompressionError> {
        let next = self.seen.borrow().len();
        if next >= self.outputs.len() {
            return Err(CompressionError::SummarizerFailed {
                reason: format!("script exhausted at call {next}"),
            });
        }
        self.seen.borrow_mut().push(input.clone());
        Ok(self.outputs[next].clone())
    }
}

fn config(window_bytes: usize) -> SelectiveCompactionConfig {
    SelectiveCompactionConfig {
        window_bytes,
        keep_recent_bytes: DEFAULT_KEEP_RECENT_BYTES,
        protected_ids: Vec::new(),
        current_generation: Some(1),
        previous_summary: None,
    }
}

fn four_hundred_footprint() -> Vec<ContextRecord> {
    // Four records, 100 bytes of footprint each (10 summary + 90 body).
    vec![
        record("r0", 10, 90),
        record("r1", 10, 90),
        record("r2", 10, 90),
        record("r3", 10, 90),
    ]
}

// --- Reserve sizing ---

#[test]
fn reserve_constants_pin_documented_values() {
    assert_eq!(DEFAULT_RESERVE_BYTES, 16 * 1024);
    assert_eq!(
        (RESERVE_FRACTION_NUMERATOR, RESERVE_FRACTION_DENOMINATOR),
        (15, 100),
        "reserve fraction is 15/100"
    );
    assert_eq!(DEFAULT_KEEP_RECENT_BYTES, 20 * 1024);
    assert_eq!(MIN_KEEP_RECORDS, 1);
}

#[test]
fn effective_reserve_is_floored_fraction_clamped_below_window() {
    assert_eq!(effective_reserve_bytes(0), 0, "zero window yields zero");
    assert_eq!(effective_reserve_bytes(1), 0, "clamped below the window");
    // 15% of 1000 is 150, below the 16 KiB floor; the floor then clamps to
    // window - 1.
    assert_eq!(effective_reserve_bytes(1_000), 999);
    // 15% of 1 MiB is 150 KiB, above the floor: the fraction wins and sits
    // strictly below the window.
    let big = effective_reserve_bytes(1_000_000);
    assert_eq!(big, 150_000);
    assert!(big < 1_000_000);
    // Fraction/floor handoff: 15% of 109_227 truncates to exactly 16_384.
    assert_eq!(effective_reserve_bytes(109_227), 16_384);
    // Just below the handoff the floor still wins.
    assert_eq!(effective_reserve_bytes(109_226), 16_384);
}

#[test]
fn effective_reserve_saturates_without_panic() {
    let reserve = effective_reserve_bytes(usize::MAX);
    assert_eq!(
        reserve,
        usize::MAX / 100,
        "saturating multiply then exact divide, no overflow"
    );
    assert!(reserve < usize::MAX);
}

// --- Trigger boundary ---

#[test]
fn trigger_boundary_is_exact_and_zero_inputs_fail_closed() {
    assert!(
        !should_compact(90, 100, 10),
        "used + reserve == window must not compact"
    );
    assert!(
        should_compact(91, 100, 10),
        "one byte over the window must compact"
    );
    assert!(
        !should_compact(0, 100, 10),
        "zero load must not compact (fail closed)"
    );
    assert!(
        !should_compact(90, 0, 10),
        "zero window must not compact (fail closed)"
    );
    assert!(
        !should_compact(90, 100, 0),
        "zero reserve must not compact (fail closed)"
    );
    assert!(
        !should_compact(0, 0, 0),
        "all-zero inputs must not compact (fail closed)"
    );
}

#[test]
fn trigger_saturates_without_panic() {
    assert!(
        !should_compact(usize::MAX, usize::MAX, usize::MAX),
        "saturating sum equals the window: no trigger, no panic"
    );
    assert!(
        should_compact(usize::MAX, usize::MAX - 1, 1),
        "saturating sum still exceeds a smaller window"
    );
}

// --- Tail protection ---

#[test]
fn recent_tail_honors_budget_and_is_deterministic() {
    let records = four_hundred_footprint();
    // 150 bytes keeps the newest two 100-byte records (100 < 150 stops only
    // after the second lands at 200).
    let first = select_compaction_window(&records, 150, &[], Some(1)).expect("select");
    let second = select_compaction_window(&records, 150, &[], Some(1)).expect("select");
    assert_eq!(first, second, "selection must be deterministic");
    assert_eq!(first.recent, vec![2, 3]);
    assert_eq!(first.head, vec![0, 1]);
    // Exact budget fit still keeps both covered records.
    let exact = select_compaction_window(&records, 200, &[], Some(1)).expect("select");
    assert_eq!(exact.recent, vec![2, 3]);
    assert_eq!(exact.head, vec![0, 1]);
}

#[test]
fn min_keep_records_protects_newest_turn_with_zero_budget() {
    let records = four_hundred_footprint();
    let window = select_compaction_window(&records, 0, &[], Some(1)).expect("select");
    assert_eq!(
        window.recent,
        vec![3],
        "zero budget still keeps the newest record (MIN_KEEP_RECORDS)"
    );
    assert_eq!(window.head, vec![0, 1, 2]);
}

#[test]
fn protected_ids_always_survive_regardless_of_budget() {
    let records = four_hundred_footprint();
    let window = select_compaction_window(&records, 0, &["r0"], Some(1)).expect("select");
    assert_eq!(window.recent, vec![0, 3]);
    assert_eq!(window.head, vec![1, 2]);
    // Unknown protected ids name no record and are ignored.
    let unknown = select_compaction_window(&records, 0, &["nope"], Some(1)).expect("select");
    assert_eq!(unknown.recent, vec![3]);
    assert_eq!(unknown.head, vec![0, 1, 2]);
}

// --- Empty head never contacts the summarizer ---

#[test]
fn all_protected_or_empty_input_selects_empty_head() {
    let records = four_hundred_footprint();
    let protected = ["r0", "r1", "r2", "r3"];
    let window = select_compaction_window(&records, 0, &protected, Some(1)).expect("select");
    assert!(window.head.is_empty());
    assert_eq!(window.recent, vec![0, 1, 2, 3]);

    let empty: Vec<ContextRecord> = Vec::new();
    let window = select_compaction_window(&empty, 0, &[], Some(1)).expect("select");
    assert!(window.head.is_empty());
    assert!(window.recent.is_empty());
}

#[test]
fn empty_head_is_noop_without_summarizer_contact() {
    let tags = RetentionTags::new();
    // Empty input.
    let exhausted = FakeSummarizer::new(Vec::new());
    assert_eq!(
        compact_selective(&[], &config(1_000_000), &exhausted, &tags, 500),
        CompactionOutcome::NoOp
    );
    assert_eq!(exhausted.calls(), 0);

    // All-protected input.
    let records = four_hundred_footprint();
    let mut all_protected = config(1_000_000);
    all_protected.protected_ids = ["r0", "r1", "r2", "r3"]
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    let starved = FakeSummarizer::new(Vec::new());
    assert_eq!(
        compact_selective(&records, &all_protected, &starved, &tags, 500),
        CompactionOutcome::NoOp
    );
    assert_eq!(starved.calls(), 0);

    // Everything fits the tail budget: nothing to compact.
    let mut roomy = config(1_000_000);
    roomy.keep_recent_bytes = usize::MAX;
    let idle = FakeSummarizer::new(Vec::new());
    assert_eq!(
        compact_selective(&records, &roomy, &idle, &tags, 500),
        CompactionOutcome::NoOp
    );
    assert_eq!(idle.calls(), 0);
}

// --- Fail-closed ---

#[test]
fn stale_generation_fails_before_selection() {
    let mut records = four_hundred_footprint();
    records[1].generation = 2;
    let err = select_compaction_window(&records, 150, &[], Some(1)).expect_err("stale must fail");
    assert!(
        matches!(
            err,
            CompressionError::Context(ContextError::StaleGeneration { .. })
        ),
        "unexpected error: {err}"
    );
    // Without a gate the same records select normally.
    let window = select_compaction_window(&records, 150, &[], None).expect("ungated select");
    assert_eq!(window.head, vec![0, 1]);

    // The driver maps the refusal to Failed without contacting the summarizer.
    let host = RecordingSummarizer::new(&["unused"]);
    let outcome = compact_selective(
        &records,
        &config(1_000_000),
        &host,
        &RetentionTags::new(),
        500,
    );
    assert!(matches!(outcome, CompactionOutcome::Failed { .. }));
    assert_eq!(host.calls(), 0);
}

#[test]
fn oversize_records_fail_before_selection() {
    let mut records = four_hundred_footprint();
    records[0].summary = "s".repeat(MAX_SUMMARY_BYTES + 1);
    let err =
        select_compaction_window(&records, 150, &[], Some(1)).expect_err("oversize must fail");
    assert!(
        matches!(
            err,
            CompressionError::Context(ContextError::SummaryTooLarge { .. })
        ),
        "unexpected error: {err}"
    );

    let host = RecordingSummarizer::new(&["unused"]);
    let outcome = compact_selective(
        &records,
        &config(1_000_000),
        &host,
        &RetentionTags::new(),
        500,
    );
    assert!(matches!(outcome, CompactionOutcome::Failed { .. }));
    assert_eq!(host.calls(), 0);
}

#[test]
fn duplicate_ids_fail_before_selection() {
    let mut records = four_hundred_footprint();
    records[2].id = "r1".to_owned();
    let err = select_compaction_window(&records, 150, &[], Some(1)).expect_err("dup must fail");
    assert!(
        matches!(
            err,
            CompressionError::Context(ContextError::DuplicateRecordId { .. })
        ),
        "unexpected error: {err}"
    );

    let host = RecordingSummarizer::new(&["unused"]);
    let outcome = compact_selective(
        &records,
        &config(1_000_000),
        &host,
        &RetentionTags::new(),
        500,
    );
    assert!(matches!(outcome, CompactionOutcome::Failed { .. }));
    assert_eq!(host.calls(), 0);
}

#[test]
fn exhausted_script_fails_without_partial_output() {
    let records = four_hundred_footprint();
    let mut tight = config(1_000_000);
    tight.keep_recent_bytes = 100;
    let starved = FakeSummarizer::new(Vec::new());
    let outcome = compact_selective(&records, &tight, &starved, &RetentionTags::new(), 500);
    assert!(
        matches!(
            &outcome,
            CompactionOutcome::Failed { reason } if !reason.is_empty()
        ),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(starved.calls(), 0);
}

#[test]
fn empty_and_whitespace_summaries_are_summarizer_failures() {
    let records = vec![record("r0", 10, 90)];
    let ranges = [SpanRange { start: 0, end: 1 }];
    for empty in ["", "   ", "\t\n "] {
        let host = FakeSummarizer::new(vec![empty.to_owned()]);
        let err = compress_records(&records, &ranges, &host, &RetentionTags::new(), 500)
            .expect_err("empty summary must fail");
        assert!(
            matches!(err, CompressionError::SummarizerFailed { .. }),
            "unexpected error for {empty:?}: {err}"
        );
    }

    // The driver maps the refusal to Failed. Two records with zero tail
    // budget select a one-record head (MIN_KEEP_RECORDS pins the newest);
    // the blank summary then refuses after exactly one contact.
    let mut tight = config(1_000_000);
    tight.keep_recent_bytes = 0;
    let pair = vec![record("r0", 10, 90), record("r1", 10, 90)];
    let blank = RecordingSummarizer::new(&["   "]);
    let outcome = compact_selective(&pair, &tight, &blank, &RetentionTags::new(), 500);
    assert!(matches!(outcome, CompactionOutcome::Failed { .. }));
    assert_eq!(blank.calls(), 1, "failing call still counts as contact");
}

#[test]
fn overbound_previous_summary_fails_without_contact() {
    let records = four_hundred_footprint();
    let mut seeded = config(1_000_000);
    seeded.keep_recent_bytes = 100;
    seeded.previous_summary = Some("p".repeat(MAX_SUMMARY_BYTES + 1));
    let host = RecordingSummarizer::new(&["unused"]);
    let outcome = compact_selective(&records, &seeded, &host, &RetentionTags::new(), 500);
    assert!(
        matches!(
            &outcome,
            CompactionOutcome::Failed { reason } if reason.contains(&MAX_SUMMARY_BYTES.to_string())
        ),
        "over-bound seed must fail closed, got: {outcome:?}"
    );
    assert_eq!(host.calls(), 0);
}

#[test]
fn doomed_window_fails_without_contact() {
    let records = four_hundred_footprint();
    // Head holds r0..r1 (200 bytes); even the smallest possible summary
    // (bounded by MAX_SUMMARY_BYTES) cannot fit a 100-byte window.
    let mut tight = config(100);
    tight.keep_recent_bytes = 150;
    let host = RecordingSummarizer::new(&["unused"]);
    let outcome = compact_selective(&records, &tight, &host, &RetentionTags::new(), 500);
    assert!(
        matches!(
            &outcome,
            CompactionOutcome::Failed { reason } if reason.contains("doomed")
        ),
        "doomed call must fail closed, got: {outcome:?}"
    );
    assert_eq!(host.calls(), 0);
}

#[test]
fn roomy_window_compacts_head_and_reports_span_count() {
    let records = four_hundred_footprint();
    let mut tight = config(1_000_000);
    tight.keep_recent_bytes = 150;
    let host = RecordingSummarizer::new(&["head summary"]);
    let outcome = compact_selective(&records, &tight, &host, &RetentionTags::new(), 500);
    assert_eq!(outcome, CompactionOutcome::Compacted { span_count: 1 });
    assert_eq!(host.calls(), 1);
}

// --- Previous-summary plumbing ---

#[test]
fn previous_summary_seed_reaches_first_span_and_rolls_forward() {
    // Protected "p" splits the head into two runs (r0) and (r2): two spans.
    // keep_recent_bytes = 100 keeps only r3; "p" is force-kept.
    let mut records = four_hundred_footprint();
    records[1].id = "p".to_owned();
    let mut chained = config(1_000_000);
    chained.keep_recent_bytes = 100;
    chained.protected_ids = vec!["p".to_owned()];
    chained.previous_summary = Some("prior work".to_owned());

    let window = select_compaction_window(&records, 100, &["p"], Some(1)).expect("select");
    assert_eq!(window.head, vec![0, 2]);
    assert_eq!(window.recent, vec![1, 3]);

    let host = RecordingSummarizer::new(&["sum-a", "sum-b"]);
    let outcome = compact_selective(&records, &chained, &host, &RetentionTags::new(), 500);
    assert_eq!(outcome, CompactionOutcome::Compacted { span_count: 2 });
    assert_eq!(host.calls(), 2);
    let seen = host.seen.borrow();
    assert_eq!(seen[0].previous_summary.as_deref(), Some("prior work"));
    assert_eq!(
        seen[1].previous_summary.as_deref(),
        Some("sum-a"),
        "later spans carry the rolling prior summary"
    );
    assert_eq!(seen[0].source_ids, vec!["r0".to_owned()]);
    assert_eq!(seen[1].source_ids, vec!["r2".to_owned()]);
}

#[test]
fn legacy_path_observes_no_previous_summary() {
    let records = four_hundred_footprint();
    let ranges = [
        SpanRange { start: 0, end: 1 },
        SpanRange { start: 1, end: 2 },
    ];
    let host = RecordingSummarizer::new(&["first", "second"]);
    compress_records(&records, &ranges, &host, &RetentionTags::new(), 500).expect("compress");
    let seen = host.seen.borrow();
    assert_eq!(seen.len(), 2);
    assert!(
        seen.iter().all(|input| input.previous_summary.is_none()),
        "legacy compress_records must observe previous_summary: None"
    );
}

// --- Provenance ---

#[test]
fn provenance_untrusted_or_retention_and_deadline_hold() {
    let clean = record("clean", 10, 90);
    let evil = untrusted("evil", 50);
    let records = vec![clean, evil];
    let ranges = [SpanRange { start: 0, end: 2 }];
    let mut tags = RetentionTags::new();
    tags.set("evil", RetentionClass::Ephemeral);
    let host = RecordingSummarizer::new(&["mixed"]);
    let view = compress_records(&records, &ranges, &host, &tags, 500).expect("compress");
    assert!(
        view.spans[0].is_untrusted_surface,
        "any untrusted source marks the summary (OR-rule)"
    );
    assert!(view.records[0].is_untrusted_surface);
    assert_eq!(
        view.spans[0].retention,
        RetentionClass::Ephemeral,
        "summaries inherit the most restrictive source class"
    );
    assert_eq!(
        view.spans[0].source_deadline_ms, 50,
        "deadline anchors at the minimum source collected_at_ms"
    );
    assert_eq!(
        view.spans[0].source_ids,
        vec!["clean".to_owned(), "evil".to_owned()]
    );
}

// --- Cache stability ---

fn canonical_bytes(turn_text: &str) -> Vec<u8> {
    let snapshot = PromptSnapshot::new(
        "bitty-core-prompt@1",
        vec![
            LayerInput::text_only(PromptLayer::CoreContract, "stable core"),
            LayerInput::text_only(PromptLayer::User, "stable user"),
            LayerInput::text_only(PromptLayer::Project, "stable project"),
            LayerInput::text_only(PromptLayer::SkillsProfile, "stable skills"),
            LayerInput::text_only(PromptLayer::RuntimeTurn, turn_text),
        ],
    )
    .expect("test snapshot is valid");
    assemble_prompt(&snapshot)
        .expect("test snapshot assembles")
        .canonical_bytes()
        .to_vec()
}

fn session_key(bytes: &[u8]) -> CacheKey {
    CacheKey::new("bitty-fake", "fake-chat", CacheScope::Session, bytes).expect("valid test key")
}

#[test]
fn summaries_in_turn_region_keep_stable_prefix_key() {
    // Before compaction the turn region carries raw record text; after, it
    // carries the compacted summary. Stable layers never move.
    let before = canonical_bytes("r0 outline\nr1 outline\nr2 outline");
    let after = canonical_bytes("summary of r0-r1; r2 verbatim outline");
    assert_ne!(
        before, after,
        "turn text differs, so canonical bytes differ"
    );
    let before_key = session_key(&before);
    let after_key = session_key(&after);
    assert_eq!(
        before_key.stable_prefix_hash, after_key.stable_prefix_hash,
        "stable-region digest must not move when only the turn region changes"
    );
    assert_eq!(
        before_key, after_key,
        "summaries land in the dynamic turn region only; stable layers untouched"
    );
}

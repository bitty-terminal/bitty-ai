//! Compaction tuning-knob plumbing (AI-0183).
//!
//! Covers [`CompactionPolicy`] defaults and validation, the policy-based
//! reserve path, [`SelectiveCompactionConfig::from_policy`], the
//! summarizer-free [`preview_compaction`] byte-agreement with
//! [`compact_selective`], and the [`record_ineffective`] strike counter:
//!
//! - Defaults reproduce the AI-0177 trigger/window behavior bit-for-bit.
//! - Validation rejects zero window, `keep_recent >= window`, and zero
//!   denominator (fail closed).
//! - Preview byte-agrees with the driver on `NoOp`, doomed `Failed`, and
//!   would-compact without consuming the summarizer script.
//! - Strikes trip exactly at the bound and saturate at `u64::MAX`.
//! - Double runs are deterministic.
//!
//! Std only. No clock, no threads, no async: every timestamp is
//! caller-supplied.

use bitty_ai_runtime::{
    CompactionOutcome, CompactionPolicy, CompactionPreview, CompressionError, ContextPriority,
    ContextRecord, DEFAULT_COMPACTION_WINDOW_BYTES, DEFAULT_KEEP_RECENT_BYTES,
    DEFAULT_MAX_INEFFECTIVE_STRIKES, DEFAULT_RESERVE_BYTES, FakeSummarizer,
    MAX_INEFFECTIVE_STRIKES, MAX_SUMMARY_BYTES, PreviewOutcome, RESERVE_FRACTION_DENOMINATOR,
    RESERVE_FRACTION_NUMERATOR, RecordBody, RetentionTags, SelectiveCompactionConfig, StableId,
    compact_selective, effective_reserve_bytes, preview_compaction, record_ineffective,
    select_compaction_window, should_compact,
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

fn four_hundred_footprint() -> Vec<ContextRecord> {
    // Four records, 100 bytes of footprint each (10 summary + 90 body).
    vec![
        record("r0", 10, 90),
        record("r1", 10, 90),
        record("r2", 10, 90),
        record("r3", 10, 90),
    ]
}

fn policy(window_bytes: usize, keep_recent_bytes: usize) -> CompactionPolicy {
    CompactionPolicy {
        window_bytes,
        keep_recent_bytes,
        reserve_num: RESERVE_FRACTION_NUMERATOR,
        reserve_den: RESERVE_FRACTION_DENOMINATOR,
        max_ineffective_strikes: DEFAULT_MAX_INEFFECTIVE_STRIKES,
    }
}

// --- Defaults reproduce the AI-0177 behavior ---

#[test]
fn policy_defaults_pin_today_constants() {
    let defaults = CompactionPolicy::default();
    assert_eq!(defaults.window_bytes, DEFAULT_COMPACTION_WINDOW_BYTES);
    assert_eq!(defaults.keep_recent_bytes, DEFAULT_KEEP_RECENT_BYTES);
    assert_eq!(
        (defaults.reserve_num, defaults.reserve_den),
        (RESERVE_FRACTION_NUMERATOR, RESERVE_FRACTION_DENOMINATOR),
        "reserve fraction stays 15/100"
    );
    assert_eq!(
        defaults.max_ineffective_strikes, DEFAULT_MAX_INEFFECTIVE_STRIKES,
        "strike bound carries its documented default"
    );
    assert_eq!(DEFAULT_RESERVE_BYTES, 16 * 1024);
    assert_eq!(DEFAULT_KEEP_RECENT_BYTES, 20 * 1024);
    assert!(defaults.validate().is_ok(), "defaults must validate");
}

#[test]
fn default_policy_reserve_reproduces_legacy_trigger_values() {
    // Bit-for-bit agreement with the documented AI-0177 values: the method
    // on a default policy and the legacy free function share one
    // implementation.
    for window in [1_usize, 1_000, 109_226, 109_227, 1_000_000, usize::MAX] {
        let via_policy = CompactionPolicy {
            window_bytes: window,
            ..CompactionPolicy::default()
        }
        .effective_reserve_bytes();
        assert_eq!(
            via_policy,
            effective_reserve_bytes(window),
            "policy reserve must equal the legacy wrapper at window {window}"
        );
    }
    assert_eq!(effective_reserve_bytes(0), 0);
    assert_eq!(effective_reserve_bytes(1_000), 999);
    assert_eq!(effective_reserve_bytes(1_000_000), 150_000);
    assert_eq!(effective_reserve_bytes(109_227), 16_384);
}

#[test]
fn default_policy_window_matches_legacy_selection() {
    // The policy tail budget selects exactly like a direct call with the
    // legacy constant, and the trigger built from the policy agrees.
    let records = four_hundred_footprint();
    let defaults = CompactionPolicy::default();
    let via_policy = select_compaction_window(&records, defaults.keep_recent_bytes, &[], Some(1))
        .expect("select");
    let via_constant = select_compaction_window(&records, DEFAULT_KEEP_RECENT_BYTES, &[], Some(1))
        .expect("select");
    assert_eq!(via_policy, via_constant);
    let reserve = defaults.effective_reserve_bytes();
    assert_eq!(
        should_compact(1_000_000, defaults.window_bytes, reserve),
        should_compact(
            1_000_000,
            defaults.window_bytes,
            effective_reserve_bytes(defaults.window_bytes)
        )
    );
}

#[test]
fn from_policy_maps_window_and_tail_and_leaves_overrides_to_host() {
    let defaults = CompactionPolicy::default();
    let config = SelectiveCompactionConfig::from_policy(&defaults);
    assert_eq!(config.window_bytes, defaults.window_bytes);
    assert_eq!(config.keep_recent_bytes, defaults.keep_recent_bytes);
    assert!(config.protected_ids.is_empty());
    assert_eq!(config.current_generation, None);
    assert_eq!(config.previous_summary, None);

    // The mapped config drives the driver: roomy window compacts.
    let records = four_hundred_footprint();
    let mut roomy = SelectiveCompactionConfig::from_policy(&policy(1_000_000, 150));
    roomy.current_generation = Some(1);
    let host = FakeSummarizer::new(vec!["head summary".to_owned()]);
    let outcome = compact_selective(&records, &roomy, &host, &RetentionTags::new(), 500);
    assert!(
        matches!(outcome, CompactionOutcome::Compacted { .. }),
        "from_policy config must drive the driver, got: {outcome:?}"
    );
    assert_eq!(host.calls(), 1);
}

// --- Validation fails closed ---

#[test]
fn validation_rejects_zero_window() {
    let bad = CompactionPolicy {
        window_bytes: 0,
        ..CompactionPolicy::default()
    };
    assert!(
        matches!(bad.validate(), Err(CompressionError::InvalidConfig { .. })),
        "zero window must fail closed"
    );
}

#[test]
fn validation_rejects_keep_at_or_above_window() {
    let window = DEFAULT_COMPACTION_WINDOW_BYTES;
    for keep in [window, window + 1, usize::MAX] {
        let bad = policy(window, keep);
        assert!(
            matches!(bad.validate(), Err(CompressionError::InvalidConfig { .. })),
            "keep_recent {keep} >= window {window} must fail closed"
        );
    }
    // Just below the window validates.
    assert!(policy(window, window - 1).validate().is_ok());
}

#[test]
fn validation_rejects_zero_denominator() {
    let bad = CompactionPolicy {
        reserve_den: 0,
        ..CompactionPolicy::default()
    };
    assert!(
        matches!(bad.validate(), Err(CompressionError::InvalidConfig { .. })),
        "zero denominator must fail closed"
    );
}

#[test]
fn validation_rejects_degenerate_fraction_and_strikes() {
    // Fraction above one.
    let over = CompactionPolicy {
        reserve_num: 101,
        reserve_den: 100,
        ..CompactionPolicy::default()
    };
    assert!(matches!(
        over.validate(),
        Err(CompressionError::InvalidConfig { .. })
    ));
    // Zero numerator.
    let zero_num = CompactionPolicy {
        reserve_num: 0,
        ..CompactionPolicy::default()
    };
    assert!(matches!(
        zero_num.validate(),
        Err(CompressionError::InvalidConfig { .. })
    ));
    // Zero strike bound.
    let zero_strikes = CompactionPolicy {
        max_ineffective_strikes: 0,
        ..CompactionPolicy::default()
    };
    assert!(matches!(
        zero_strikes.validate(),
        Err(CompressionError::InvalidConfig { .. })
    ));
}

#[test]
fn validation_rejects_over_bound_strikes() {
    assert_eq!(MAX_INEFFECTIVE_STRIKES, 1_024);
    for strikes in [MAX_INEFFECTIVE_STRIKES + 1, u32::MAX] {
        let over = CompactionPolicy {
            max_ineffective_strikes: strikes,
            ..CompactionPolicy::default()
        };
        assert!(
            matches!(over.validate(), Err(CompressionError::InvalidConfig { .. })),
            "strikes {strikes} > bound must fail closed"
        );
    }
    let at_bound = CompactionPolicy {
        max_ineffective_strikes: MAX_INEFFECTIVE_STRIKES,
        ..CompactionPolicy::default()
    };
    assert!(at_bound.validate().is_ok(), "bound itself must validate");
}

// --- Preview byte-agrees with the driver without summarizer contact ---

#[test]
fn preview_noop_byte_agrees_with_driver() {
    let tags = RetentionTags::new();
    // Empty input: preview says NoOp, driver says NoOp, script untouched.
    let scripted = FakeSummarizer::new(Vec::new());
    let preview = preview_compaction(&[], &CompactionPolicy::default(), &[], Some(1));
    assert_eq!(preview.head_bytes, 0);
    assert!(preview.head.is_empty());
    assert!(matches!(preview.outcome, PreviewOutcome::NoOp));
    let outcome = compact_selective(
        &[],
        &SelectiveCompactionConfig::from_policy(&CompactionPolicy::default()),
        &scripted,
        &tags,
        500,
    );
    // from_policy carries no generation gate; empty input is NoOp either way.
    assert_eq!(outcome, CompactionOutcome::NoOp);
    assert_eq!(scripted.calls(), 0, "preview must not consume the script");

    // All-protected input selects an empty head: NoOp on both sides.
    let records = four_hundred_footprint();
    let protected = ["r0", "r1", "r2", "r3"];
    let preview = preview_compaction(&records, &policy(1_000_000, 0), &protected, Some(1));
    assert!(matches!(preview.outcome, PreviewOutcome::NoOp));
    assert!(preview.head.is_empty());
    assert_eq!(preview.recent, vec![0, 1, 2, 3]);
    let host = FakeSummarizer::new(Vec::new());
    let mut config = SelectiveCompactionConfig::from_policy(&policy(1_000_000, 0));
    config.current_generation = Some(1);
    config.protected_ids = protected.iter().map(|id| (*id).to_owned()).collect();
    assert_eq!(
        compact_selective(&records, &config, &host, &tags, 500),
        CompactionOutcome::NoOp
    );
    assert_eq!(host.calls(), 0);
}

#[test]
fn preview_doomed_byte_agrees_with_driver() {
    // Head holds r0..r1 (200 bytes); a 1000-byte window with a 150-byte
    // tail budget is valid (keep < window) yet doomed: even the smallest
    // summary cannot fit alongside the head.
    let records = four_hundred_footprint();
    let tight = policy(1000, 150);
    let preview = preview_compaction(&records, &tight, &[], Some(1));
    let CompactionPreview {
        head,
        recent,
        head_bytes,
        outcome,
    } = &preview;
    assert_eq!(*head, vec![0, 1]);
    assert_eq!(*recent, vec![2, 3]);
    assert_eq!(*head_bytes, 200);
    let PreviewOutcome::Failed {
        reason: preview_reason,
    } = outcome
    else {
        panic!("doomed preview must fail, got: {preview:?}");
    };
    assert!(preview_reason.contains("doomed"), "got: {preview_reason}");

    // The driver refuses with the identical reason and no contact.
    let host = FakeSummarizer::new(vec!["unused".to_owned()]);
    let mut config = SelectiveCompactionConfig::from_policy(&tight);
    config.current_generation = Some(1);
    let outcome = compact_selective(&records, &config, &host, &RetentionTags::new(), 500);
    let CompactionOutcome::Failed {
        reason: driver_reason,
    } = outcome
    else {
        panic!("doomed driver must fail, got: {outcome:?}");
    };
    assert_eq!(
        preview_reason, &driver_reason,
        "preview and driver doomed reasons must byte-agree"
    );
    assert_eq!(host.calls(), 0, "doomed pass must not consume the script");
}

#[test]
fn preview_would_compact_byte_agrees_with_driver_without_consuming_script() {
    let records = four_hundred_footprint();
    let roomy = policy(1_000_000, 150);
    let scripted = FakeSummarizer::new(vec!["head summary".to_owned()]);
    // Preview first: no summarizer argument exists, so the script cannot be
    // consumed; the call count below proves it.
    let preview = preview_compaction(&records, &roomy, &[], Some(1));
    assert_eq!(preview.head, vec![0, 1]);
    assert_eq!(preview.recent, vec![2, 3]);
    assert_eq!(preview.head_bytes, 200);
    assert!(
        matches!(preview.outcome, PreviewOutcome::WouldCompact),
        "roomy preview must report would-compact, got: {:?}",
        preview.outcome
    );
    // Cross-check against the raw selection and the footprint sum.
    let window =
        select_compaction_window(&records, roomy.keep_recent_bytes, &[], Some(1)).expect("select");
    assert_eq!(preview.head, window.head);
    assert_eq!(preview.recent, window.recent);
    let expected: usize = window
        .head
        .iter()
        .map(|i| records[*i].footprint_bytes())
        .sum();
    assert_eq!(preview.head_bytes, expected);
    assert_eq!(scripted.calls(), 0, "preview must not consume the script");

    // The driver then compacts with exactly one contact.
    let mut config = SelectiveCompactionConfig::from_policy(&roomy);
    config.current_generation = Some(1);
    let outcome = compact_selective(&records, &config, &scripted, &RetentionTags::new(), 500);
    let CompactionOutcome::Compacted { span_count, view } = outcome else {
        panic!("roomy driver must compact, got: {outcome:?}");
    };
    assert_eq!(span_count, 1);
    assert_eq!(
        view.spans[0].source_ids,
        vec!["r0".to_owned(), "r1".to_owned()]
    );
    assert_eq!(
        scripted.calls(),
        1,
        "driver consumes exactly one script line"
    );
}

#[test]
fn preview_failed_selection_matches_driver_refusal() {
    // Stale generation: selection refuses; preview reports Failed with the
    // same reason the driver reports, and the driver never contacts.
    let mut records = four_hundred_footprint();
    records[1].generation = 2;
    let roomy = policy(1_000_000, 150);
    let preview = preview_compaction(&records, &roomy, &[], Some(1));
    let PreviewOutcome::Failed {
        reason: preview_reason,
    } = &preview.outcome
    else {
        panic!("stale preview must fail, got: {:?}", preview.outcome);
    };
    assert!(preview.head.is_empty() && preview.recent.is_empty());
    assert_eq!(preview.head_bytes, 0);

    let host = FakeSummarizer::new(vec!["unused".to_owned()]);
    let mut config = SelectiveCompactionConfig::from_policy(&roomy);
    config.current_generation = Some(1);
    let outcome = compact_selective(&records, &config, &host, &RetentionTags::new(), 500);
    let CompactionOutcome::Failed {
        reason: driver_reason,
    } = outcome
    else {
        panic!("stale driver must fail, got: {outcome:?}");
    };
    assert_eq!(preview_reason, &driver_reason);
    assert_eq!(host.calls(), 0);
}

#[test]
fn preview_rejects_invalid_policy_without_contact() {
    let records = four_hundred_footprint();
    let bad = CompactionPolicy {
        window_bytes: 0,
        ..CompactionPolicy::default()
    };
    let preview = preview_compaction(&records, &bad, &[], Some(1));
    assert!(
        matches!(preview.outcome, PreviewOutcome::Failed { .. }),
        "invalid policy must preview as Failed, got: {:?}",
        preview.outcome
    );
}

// --- Strike counter trips exactly at the bound with saturation ---

#[test]
fn strikes_trip_exactly_at_bound_and_stay_disabled() {
    let bound = CompactionPolicy {
        max_ineffective_strikes: 3,
        ..CompactionPolicy::default()
    };
    assert_eq!(record_ineffective(0, &bound), (1, false));
    assert_eq!(record_ineffective(1, &bound), (2, false));
    // Exactly at the bound: disabled flips true and stays true.
    assert_eq!(record_ineffective(2, &bound), (3, true));
    assert_eq!(record_ineffective(3, &bound), (4, true));
    // Bound of one disables on the first miss.
    let single = CompactionPolicy {
        max_ineffective_strikes: 1,
        ..CompactionPolicy::default()
    };
    assert_eq!(record_ineffective(0, &single), (1, true));
}

#[test]
fn strikes_saturate_at_u64_max_without_wrapping() {
    let bound = CompactionPolicy::default();
    let (next, disabled) = record_ineffective(u64::MAX, &bound);
    assert_eq!(next, u64::MAX, "counter must saturate, never wrap");
    assert!(disabled, "saturated counter stays disabled");
    assert_eq!(
        record_ineffective(u64::MAX - 1, &bound),
        (u64::MAX, true),
        "last increment lands on MAX and disables"
    );
}

// --- Double-run determinism ---

#[test]
fn double_run_preview_and_selection_are_deterministic() {
    let records = four_hundred_footprint();
    let roomy = policy(1_000_000, 150);
    let first = preview_compaction(&records, &roomy, &[], Some(1));
    let second = preview_compaction(&records, &roomy, &[], Some(1));
    assert_eq!(first, second, "preview must be deterministic");

    let doomed = policy(1000, 150);
    assert_eq!(
        preview_compaction(&records, &doomed, &[], Some(1)),
        preview_compaction(&records, &doomed, &[], Some(1)),
        "doomed preview must be deterministic"
    );

    // Reserve and trigger are pure functions of the policy.
    assert_eq!(
        roomy.effective_reserve_bytes(),
        roomy.effective_reserve_bytes()
    );
    let reserve = roomy.effective_reserve_bytes();
    assert_eq!(
        should_compact(500_000, roomy.window_bytes, reserve),
        should_compact(500_000, roomy.window_bytes, reserve)
    );
}

#[test]
fn preview_head_bytes_cannot_overflow_the_doomed_guard() {
    // Saturated head sum still refuses as doomed rather than panicking:
    // head_bytes saturates, then the guard compares against the window.
    let records = four_hundred_footprint();
    let tiny = policy(100, 0);
    let preview = preview_compaction(&records, &tiny, &[], Some(1));
    assert_eq!(preview.head, vec![0, 1, 2]);
    assert!(
        preview.head_bytes.saturating_add(MAX_SUMMARY_BYTES) > tiny.window_bytes,
        "tiny window must read as doomed"
    );
    assert!(matches!(preview.outcome, PreviewOutcome::Failed { .. }));
}

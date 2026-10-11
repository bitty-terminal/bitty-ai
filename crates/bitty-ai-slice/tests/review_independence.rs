//! Review-independence negative gate (AI-0235, R5-V4 code half, RFC AIQ-26).
//!
//! All offline and deterministic: caller-supplied identities, pointers, and
//! timestamps only. No threads, no network, no wall-clock.
//!
//! Pinned negatives (each fails closed, never `Ok`): self-review (same
//! agent), same session, shared PASS, cached PASS, self-report, and partial
//! checks (missing or identical requirement/evidence). Positive shape:
//! approval needs a different agent **and** session, reviewer-held
//! requirement+evidence pointers, and the separate [`ReviewApproval::issue`]
//! action. HARD RULE: same-model-family, shared-base, shared-data, or a
//! cached PASS alone is NOT independence and no code path smuggles it.
//! Approval is a record, never a transition or grant (R5-V1).

use bitty_ai_slice::task_dag::{
    CachedPassClaim, ReviewApproval, ReviewApprovalError, ReviewEvidence, ReviewSubject,
    ReviewerIdentity, SelfReportClaim, SharedPassClaim, TaskDraft, TaskEngine, TaskId,
};

fn subject() -> ReviewSubject {
    ReviewSubject {
        task_id: TaskId::new("review-subject").expect("valid id"),
        executor_agent: "exec-agent".to_string(),
        executor_session: "exec-session".to_string(),
    }
}

fn reviewer() -> ReviewerIdentity {
    ReviewerIdentity {
        agent: "review-agent".to_string(),
        session: "review-session".to_string(),
    }
}

fn evidence() -> ReviewEvidence {
    ReviewEvidence {
        requirement_ref: "req:AIQ-26/R5-V4".to_string(),
        evidence_ref: "ev:task-row+reflog+checkpoint".to_string(),
    }
}

#[test]
fn self_review_same_agent_never_approval() {
    let subject = subject();
    let reviewer = ReviewerIdentity {
        agent: "exec-agent".to_string(),
        session: "other-session".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &reviewer, &evidence(), 1000)
        .expect_err("self-review must fail closed");
    assert_eq!(err, ReviewApprovalError::SelfReview);
}

#[test]
fn same_session_different_agent_never_approval() {
    let subject = subject();
    let reviewer = ReviewerIdentity {
        agent: "review-agent".to_string(),
        session: "exec-session".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &reviewer, &evidence(), 1000)
        .expect_err("same session must fail closed");
    assert_eq!(err, ReviewApprovalError::SameSession);
}

#[test]
fn shared_pass_never_approval() {
    let claim = SharedPassClaim {
        task_id: TaskId::new("review-subject").expect("valid id"),
        claimant_agent: "review-agent".to_string(),
        claimant_session: "review-session".to_string(),
    };
    let err = claim
        .into_approval()
        .expect_err("shared PASS must fail closed");
    assert_eq!(err, ReviewApprovalError::SharedPass);
}

#[test]
fn cached_pass_reuse_refused_fail_closed() {
    let claim = CachedPassClaim {
        task_id: TaskId::new("review-subject").expect("valid id"),
        holder_agent: "review-agent".to_string(),
        holder_session: "review-session".to_string(),
        cache_key: "cache:pass:review-subject:v1".to_string(),
    };
    let err = claim
        .into_approval()
        .expect_err("cached PASS reuse must fail closed");
    assert_eq!(err, ReviewApprovalError::CachedPass);
}

#[test]
fn self_report_never_approval() {
    let claim = SelfReportClaim {
        task_id: TaskId::new("review-subject").expect("valid id"),
        executor_agent: "exec-agent".to_string(),
        executor_session: "exec-session".to_string(),
        report_ref: "report:exec-did-the-work".to_string(),
    };
    let err = claim
        .into_approval()
        .expect_err("self-report must fail closed");
    assert_eq!(err, ReviewApprovalError::SelfReport);
}

#[test]
fn partial_checks_never_approval() {
    let subject = subject();
    let reviewer = reviewer();

    // Missing requirement pointer.
    let missing_req = ReviewEvidence {
        requirement_ref: "   ".to_string(),
        evidence_ref: "ev:task-row".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &reviewer, &missing_req, 1000)
        .expect_err("missing requirement must fail closed");
    assert_eq!(err, ReviewApprovalError::EmptyField("requirement_ref"));

    // Missing evidence pointer.
    let missing_ev = ReviewEvidence {
        requirement_ref: "req:AIQ-26".to_string(),
        evidence_ref: "".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &reviewer, &missing_ev, 1000)
        .expect_err("missing evidence must fail closed");
    assert_eq!(err, ReviewApprovalError::EmptyField("evidence_ref"));

    // Same pointer supplied as both requirement and evidence.
    let identical = ReviewEvidence {
        requirement_ref: "ptr:single-check".to_string(),
        evidence_ref: "ptr:single-check".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &reviewer, &identical, 1000)
        .expect_err("identical pointers must fail closed as partial");
    assert_eq!(err, ReviewApprovalError::PartialChecks);
}

#[test]
fn approval_requires_different_agent_and_session_with_refs_and_separate_action() {
    let subject = subject();
    let reviewer = reviewer();
    let evidence = evidence();
    let approval = ReviewApproval::issue(&subject, &reviewer, &evidence, 1234)
        .expect("independent review approves");
    assert_eq!(approval.task_id, subject.task_id);
    assert_eq!(approval.executor_agent, "exec-agent");
    assert_eq!(approval.executor_session, "exec-session");
    assert_eq!(approval.reviewer_agent, "review-agent");
    assert_eq!(approval.reviewer_session, "review-session");
    assert_eq!(approval.requirement_ref, "req:AIQ-26/R5-V4");
    assert_eq!(approval.evidence_ref, "ev:task-row+reflog+checkpoint");
    assert_eq!(approval.issued_at_ms, 1234);
    assert_ne!(approval.reviewer_agent, approval.executor_agent);
    assert_ne!(approval.reviewer_session, approval.executor_session);
    assert_ne!(approval.requirement_ref, approval.evidence_ref);
}

#[test]
fn same_model_family_shared_base_data_do_not_confer_independence() {
    // The gate takes no model/base/data/cache inputs by design: varying an
    // external model-family (or base/data) label cannot change the verdict.
    // Same agent with a "different model" is still self-review; same session
    // with a "different model" is still the same session; a cached PASS with
    // a "different model" is still a cached PASS.
    for model_family in ["same-family", "other-family", "other-base", "other-data"] {
        let _ = model_family;

        let subject = subject();
        let same_agent = ReviewerIdentity {
            agent: "exec-agent".to_string(),
            session: "other-session".to_string(),
        };
        assert_eq!(
            ReviewApproval::issue(&subject, &same_agent, &evidence(), 1000)
                .expect_err("model label must not smuggle self-review"),
            ReviewApprovalError::SelfReview
        );

        let same_session = ReviewerIdentity {
            agent: "review-agent".to_string(),
            session: "exec-session".to_string(),
        };
        assert_eq!(
            ReviewApproval::issue(&subject, &same_session, &evidence(), 1000)
                .expect_err("model label must not smuggle session sharing"),
            ReviewApprovalError::SameSession
        );

        let cached = CachedPassClaim {
            task_id: TaskId::new("review-subject").expect("valid id"),
            holder_agent: "review-agent".to_string(),
            holder_session: "review-session".to_string(),
            cache_key: format!("cache:{model_family}:review-subject"),
        };
        assert_eq!(
            cached
                .into_approval()
                .expect_err("model label must not smuggle cached PASS"),
            ReviewApprovalError::CachedPass
        );
    }
}

#[test]
fn approval_is_record_never_transition() -> Result<(), ReviewApprovalError> {
    let mut engine = TaskEngine::open_in_memory().expect("engine opens");
    let id = TaskId::new("record-task").expect("valid id");
    engine
        .create_task(
            TaskDraft {
                id: id.clone(),
                title: "Record".to_string(),
                description: "".to_string(),
                priority: 0,
                dependencies: vec![],
            },
            1000,
        )
        .expect("create");
    let generation = engine.assign_task(&id, "exec-agent", 1010).expect("assign");
    let before = engine.get_task(&id).expect("read before");
    assert_eq!(before.generation, generation);

    // Completing the work alone yields no approval record: the only approval
    // path is the separate issue action below.
    engine
        .complete_task(&id, generation, None, 1020)
        .expect("complete");
    let completed = engine.get_task(&id).expect("read completed");

    let subject = ReviewSubject {
        task_id: id.clone(),
        executor_agent: "exec-agent".to_string(),
        executor_session: "exec-session".to_string(),
    };
    let approval = ReviewApproval::issue(&subject, &reviewer(), &evidence(), 1030)?;
    assert_eq!(approval.task_id, id);

    // Issuing the record touched no engine state: status, generation,
    // assignment, and checkpoint are byte-identical before/after.
    let after = engine.get_task(&id).expect("read after");
    assert_eq!(after.status, completed.status);
    assert_eq!(after.generation, completed.generation);
    assert_eq!(after.assigned_agent, completed.assigned_agent);
    assert_eq!(after.checkpoint, completed.checkpoint);
    assert_eq!(after.updated_at_ms, completed.updated_at_ms);
    Ok(())
}

#[test]
fn review_bounds_fail_closed() {
    let subject = subject();
    let reviewer = reviewer();

    let long_ref = "r".repeat(513);
    let oversized = ReviewEvidence {
        requirement_ref: long_ref,
        evidence_ref: "ev:ok".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &reviewer, &oversized, 1000)
        .expect_err("oversized requirement must fail closed");
    assert_eq!(
        err,
        ReviewApprovalError::OversizedField {
            field: "requirement_ref",
            size: 513,
            max: 512,
        }
    );

    let long_agent = "a".repeat(129);
    let oversized_reviewer = ReviewerIdentity {
        agent: long_agent,
        session: "review-session".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &oversized_reviewer, &evidence(), 1000)
        .expect_err("oversized reviewer agent must fail closed");
    assert!(matches!(
        err,
        ReviewApprovalError::OversizedField {
            field: "reviewer_agent",
            ..
        }
    ));

    let empty_reviewer = ReviewerIdentity {
        agent: "   ".to_string(),
        session: "review-session".to_string(),
    };
    let err = ReviewApproval::issue(&subject, &empty_reviewer, &evidence(), 1000)
        .expect_err("empty reviewer agent must fail closed");
    assert_eq!(err, ReviewApprovalError::EmptyField("reviewer_agent"));
}

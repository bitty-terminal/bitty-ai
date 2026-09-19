//! Supervisor crash-recovery adoption rule (AI-0091, AIQ-2B narrowing).
//!
//! AIQ-2B (supervisor crash recovery/adoption) is stay-open-narrowed:
//! single-agent `Unknown` handling is bounded (resolve / escalate accounting
//! in [`crate::reconcile`], multi-turn and escalation-session semantics in
//! `agent_turn_semantics.rs`), but supervisor crash adoption across processes
//! had no mechanism. The runtime is single-process sans-I/O, so cross-process
//! adoption cannot be built here; what this module pins at the seam is the
//! ADOPTION RULE: a supervisor must never adopt arbitrary survivors or repeat
//! `Unknown` effects.
//!
//! Pinned rule: adoption requires an explicit, typed [`AdoptionClaim`] and
//! the supervisor-side [`check_adoption`] refuses unless all four hold:
//! (a) the prior session is in terminal `Failed` state (live sessions are
//! never adopted; `Completed`/`Canceled` have no adoptable leftovers and a
//! different owner);
//! (b) every carried still-`Unknown` effect is named `Escalated` (claimed
//! `Reconciled` over still-uncertain evidence, omitted `Unknown`s, and any
//! claimed disposition over never-`Unknown` evidence all refuse; a
//! `Reconciled` disposition is verifiable only for evidence the pre-crash
//! protocol actually resolved, covered by the `Unknown`-free success path;
//! escalated ones adopt as quarantined declarations);
//! (c) nothing replays: the check takes no executor, returns declarative
//! data only, and adopted ids stay foreign to the fresh session (its
//! `reconcile_unknown` reports `NoUnknown` for them because they were never
//! recorded there);
//! (d) the claim's fence token equals the current supervisor epoch
//! (stale-epoch claims refused, mirroring the AIQ-2C writer-fencing
//! direction without building it).
//!
//! Shape note: the candidate's bare `unknown_effect_ids` cannot carry
//! dispositions, so the claim names `unknown_effects:
//! Vec<ClaimedUnknownEffect>` pairing each id with its [`UnknownDisposition`].
//! Survivor evidence reuses [`crate::agent::ExecutionRecord`] (the exact
//! attributed-record shape the crashed turn recorded) instead of a duplicate
//! type.
//!
//! ## What is proven vs stub
//!
//! - Proven (here plus `tests/recovery_adoption.rs`): the four refusal verbs,
//!   exact-set coverage (no stowaway, no missing, no duplicate evidence),
//!   quarantine of escalated `Unknown`s, the no-replay observable (executor
//!   call counts flat, adopted ids rejected as `NoUnknown`), and the fence
//!   verdict. Deterministic doubles only; no threads, no processes, no I/O,
//!   no clock.
//! - Stub (deliberately absent): cross-process transport of the claim and the
//!   evidence, supervisor liveness/epoch election, and persistence of the
//!   fence token. The claim and evidence are caller-supplied values at this
//!   seam; a host that fabricates them bypasses the rule, and that host-side
//!   attestation stays out of scope (stated, not built).
//!
//! ## Determinism rules
//!
//! Pure function of its inputs: no wall clock, no thread, no executor, no
//! I/O. All behavior in tests is reproducible from the claim plus evidence.

use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::agent::ExecutionRecord;
use crate::session::{ExecutionId, SessionState};
use crate::tool::ToolStatus;

/// Hard cap on survivors per adoption claim. Bounds supervisor work even
/// when a caller ships a larger crash manifest; the check refuses over-bound
/// sets fail-closed before considering any content. The same bound covers
/// every collection a claim carries — survivor ids, carried evidence, and
/// the auxiliary `Unknown` disposition list, which cannot legitimately
/// exceed the survivor set — and is the capacity bound for the check's
/// scratch vectors.
pub const MAX_ADOPTION_SURVIVORS: usize = 32;

/// Verified disposition of one carried `Unknown` effect: the pre-crash
/// reconcile protocol ran first (`reconcile_unknown` on the crashed agent),
/// and the claim attests the outcome it observed. `Reconciled` is verifiable
/// only for evidence the protocol actually resolved (the success path pins
/// this by carrying post-reconcile terminal evidence alongside the protocol
/// transcript); a never-`Unknown` record carrying `Reconciled` refuses as
/// `DispositionMismatch` because the rule cannot distinguish it from an
/// invented disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownDisposition {
    /// The effect reconciled to a terminal status before the crash (evidence
    /// carries that terminal status now).
    Reconciled,
    /// Reconcile exhausted its budget and escalated before the crash
    /// (evidence still carries `Unknown`; the survivor adopts quarantined).
    Escalated,
}

/// One `Unknown` effect named by an adoption claim, pairing the id with the
/// disposition the pre-crash protocol observed. The pairing (not a bare id
/// list) is what lets the check verify reconcile-first ordering against the
/// carried evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimedUnknownEffect {
    /// `Unknown` execution the claim names.
    pub execution_id: ExecutionId,
    /// Disposition the pre-crash protocol reported for it.
    pub disposition: UnknownDisposition,
}

/// Supervisor-side adoption claim: the explicit, typed attestation a
/// supervisor presents to adopt survivors of a crashed session. All fields
/// are caller-supplied values at this seam; the host owns attesting them
/// honestly (see the module docs: fabrication bypasses the rule by
/// construction and stays out of scope).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdoptionClaim {
    /// Lifecycle state the crashed session was observed in. Only `Failed`
    /// is adoptable; anything else refuses (`AlreadyTerminated` semantics
    /// for live sessions: never adopt a live session survivor).
    pub prior_session_state: SessionState,
    /// Survivor ids requested for adoption, in adoption order. Must name
    /// exactly the carried evidence: no stowaway, no missing, no duplicate.
    pub survivor_ids: Vec<ExecutionId>,
    /// Dispositions for the `Unknown` effects in the evidence. Must name
    /// every carried `Unknown` exactly once with a disposition the evidence
    /// verifies; surplus or contradicted entries refuse.
    pub unknown_effects: Vec<ClaimedUnknownEffect>,
    /// Fence token binding the claim to one supervisor epoch. Must equal the
    /// adopting supervisor's current epoch.
    pub fence_token: u64,
}

/// Typed adoption refusal. Every check refuses fail-closed with no partial
/// adoption: no declaration is returned, nothing is dispatched (the check
/// takes no executor, so dispatch is impossible by construction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdoptionRefusal {
    /// The prior session is still live (`Active`). Never adopt a live
    /// session survivor.
    LiveSession,
    /// The prior session is terminal but not `Failed`; carries the observed
    /// state. `Completed` sessions have no unreconciled leftovers and
    /// `Canceled` leftovers belong to the cancel-reconcile path.
    NotFailed {
        /// Observed terminal state.
        state: SessionState,
    },
    /// A carried `Unknown` has no verified disposition: it was omitted from
    /// `unknown_effects`, claimed `Reconciled` while the evidence is still
    /// `Unknown`, or carries another status the rule cannot verify.
    UnreconciledUnknown {
        /// Offending execution.
        execution_id: ExecutionId,
    },
    /// A claimed disposition contradicts the evidence: an `Escalated` claim
    /// over already-terminal evidence, a duplicate unknown entry, or a claim
    /// entry with no matching carried evidence.
    DispositionMismatch {
        /// Offending execution.
        execution_id: ExecutionId,
    },
    /// A claimed survivor id has no matching carried evidence.
    MissingEvidence {
        /// Offending execution.
        execution_id: ExecutionId,
    },
    /// Carried evidence has no matching claim: adoption is exact-set only.
    UnclaimedSurvivor {
        /// Offending execution.
        execution_id: ExecutionId,
    },
    /// A survivor or evidence id appears twice. Exact-set adoption cannot
    /// admit ambiguous attribution.
    DuplicateEffect {
        /// Offending execution.
        execution_id: ExecutionId,
    },
    /// A claim collection exceeds [`MAX_ADOPTION_SURVIVORS`]: the survivor
    /// set, the carried evidence, or the named `Unknown` dispositions.
    /// Every count is checked before any content or allocation so oversized
    /// manifests bound supervisor work.
    TooManySurvivors {
        /// Bound.
        limit: usize,
        /// Observed count.
        actual: usize,
    },
    /// The claim's fence token does not equal the current supervisor epoch.
    /// A stale supervisor cannot decide adoptions.
    StaleEpoch {
        /// Epoch carried by the claim.
        claim_epoch: u64,
        /// Epoch of the adopting supervisor.
        current_epoch: u64,
    },
}

impl Display for AdoptionRefusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::LiveSession => write!(f, "adoption refused: prior session is live"),
            Self::NotFailed { state } => {
                write!(
                    f,
                    "adoption refused: prior session is {state:?}, not failed"
                )
            }
            Self::UnreconciledUnknown { execution_id } => write!(
                f,
                "adoption refused: effect {} is unreconciled",
                execution_id.0
            ),
            Self::DispositionMismatch { execution_id } => write!(
                f,
                "adoption refused: disposition mismatch for effect {}",
                execution_id.0
            ),
            Self::MissingEvidence { execution_id } => write!(
                f,
                "adoption refused: no evidence for effect {}",
                execution_id.0
            ),
            Self::UnclaimedSurvivor { execution_id } => write!(
                f,
                "adoption refused: effect {} is not claimed",
                execution_id.0
            ),
            Self::DuplicateEffect { execution_id } => {
                write!(f, "adoption refused: duplicate effect {}", execution_id.0)
            }
            Self::TooManySurvivors { limit, actual } => {
                write!(f, "adoption refused: {actual} entries exceed limit {limit}")
            }
            Self::StaleEpoch {
                claim_epoch,
                current_epoch,
            } => write!(
                f,
                "adoption refused: stale epoch {claim_epoch}, current {current_epoch}"
            ),
        }
    }
}

impl std::error::Error for AdoptionRefusal {}

/// One adopted survivor: declarative history, never an executable effect.
/// The adopting session carries these as inert records (for example as
/// context seed data marked by `quarantined`); they must never re-enter the
/// dispatch or reconcile path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdoptedEffect {
    /// Attribution handle from the crashed turn (foreign to the adopter).
    pub execution_id: ExecutionId,
    /// Tool that ran (or was uncertain) before the crash.
    pub tool: String,
    /// Status observed pre-crash (terminal, or `Unknown` when escalated).
    pub status: ToolStatus,
    /// Whether the effect is quarantined: always true for adopted
    /// still-`Unknown` effects, always false for terminal ones. Quarantined
    /// effects are declarations of uncertainty, never redispatch candidates.
    pub quarantined: bool,
}

/// Adopted crash history: the exact survivor set as declarative data. The
/// order follows the claim's `survivor_ids`; `quarantined_ids` names the
/// subset that must never execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdoptedHistory {
    /// Adopted survivors in claim order.
    pub effects: Vec<AdoptedEffect>,
}

impl AdoptedHistory {
    /// Number of adopted survivors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Whether no survivor was adopted (vacuous crash).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Ids of the quarantined (still-`Unknown`) survivors, in claim order.
    /// These are declarations of uncertainty, never redispatch candidates.
    #[must_use]
    pub fn quarantined_ids(&self) -> Vec<ExecutionId> {
        self.effects
            .iter()
            .filter(|effect| effect.quarantined)
            .map(|effect| effect.execution_id)
            .collect()
    }
}

/// Supervisor-side adoption check: the four-part rule as a pure function.
///
/// Refuses fail-closed with a typed [`AdoptionRefusal`] unless (a) the prior
/// session is `Failed`, (b) every carried still-`Unknown` effect is named
/// `Escalated` (claimed `Reconciled` over still-uncertain evidence refuses;
/// any claimed disposition over never-`Unknown` evidence refuses because a
/// bare now-terminal record cannot prove the pre-crash protocol ran), (c)
/// coverage is exact (see below), and (d) the fence token
/// equals `current_epoch`. On success returns the exact survivor set as
/// declarative [`AdoptedHistory`]: terminal effects as plain declarations,
/// escalated `Unknown`s as quarantined declarations.
///
/// The check performs no effect: it takes no executor, runs no provider
/// round, issues no reconcile query, and reads no clock. Bounds are checked
/// before content: every collection count (survivor ids, carried evidence,
/// named `Unknown` dispositions) is validated before any content is
/// inspected or any scratch vector is allocated.
///
/// # Errors
///
/// Returns [`AdoptionRefusal`] for a live or non-failed prior, a stale
/// fence token, an over-bound set, or any claim/evidence mismatch
/// (missing or unclaimed ids, duplicates, unverified or contradicted
/// dispositions).
pub fn check_adoption(
    claim: &AdoptionClaim,
    evidence: &[ExecutionRecord],
    current_epoch: u64,
) -> Result<AdoptedHistory, AdoptionRefusal> {
    let actual = claim
        .survivor_ids
        .len()
        .max(evidence.len())
        .max(claim.unknown_effects.len());
    if actual > MAX_ADOPTION_SURVIVORS {
        return Err(AdoptionRefusal::TooManySurvivors {
            limit: MAX_ADOPTION_SURVIVORS,
            actual,
        });
    }
    if claim.fence_token != current_epoch {
        return Err(AdoptionRefusal::StaleEpoch {
            claim_epoch: claim.fence_token,
            current_epoch,
        });
    }
    if claim.prior_session_state == SessionState::Active {
        return Err(AdoptionRefusal::LiveSession);
    }
    if claim.prior_session_state != SessionState::Failed {
        return Err(AdoptionRefusal::NotFailed {
            state: claim.prior_session_state,
        });
    }
    // Every count above is validated against the survivor bound, so scratch
    // capacity is derived from that bound (never from raw untrusted lengths)
    // and stays bounded even if a future edit reorders this check.
    let mut seen_ids: Vec<ExecutionId> = Vec::with_capacity(MAX_ADOPTION_SURVIVORS);
    for record in evidence {
        if seen_ids.contains(&record.execution_id) {
            return Err(AdoptionRefusal::DuplicateEffect {
                execution_id: record.execution_id,
            });
        }
        seen_ids.push(record.execution_id);
    }
    let mut seen_claims: Vec<ExecutionId> = Vec::with_capacity(MAX_ADOPTION_SURVIVORS);
    for id in &claim.survivor_ids {
        if seen_claims.contains(id) {
            return Err(AdoptionRefusal::DuplicateEffect { execution_id: *id });
        }
        seen_claims.push(*id);
    }
    let mut seen_unknowns: Vec<ExecutionId> = Vec::with_capacity(MAX_ADOPTION_SURVIVORS);
    for named in &claim.unknown_effects {
        if seen_unknowns.contains(&named.execution_id) {
            return Err(AdoptionRefusal::DispositionMismatch {
                execution_id: named.execution_id,
            });
        }
        seen_unknowns.push(named.execution_id);
        // Every named unknown must be survivor-covered evidence; an orphan
        // entry names an effect adoption cannot declare.
        if !seen_claims.contains(&named.execution_id) || !seen_ids.contains(&named.execution_id) {
            return Err(AdoptionRefusal::MissingEvidence {
                execution_id: named.execution_id,
            });
        }
        let Some(record) = evidence
            .iter()
            .find(|record| record.execution_id == named.execution_id)
        else {
            // Unreachable by construction: membership was checked against
            // `seen_ids` (built from the same `evidence`) two lines above.
            // Fail closed rather than invent a record.
            return Err(AdoptionRefusal::MissingEvidence {
                execution_id: named.execution_id,
            });
        };
        let is_unknown = matches!(record.status, ToolStatus::Unknown { .. });
        match (named.disposition, is_unknown) {
            // Rule (b): escalated only over still-`Unknown` evidence. A
            // `Reconciled` entry over now-terminal evidence that the
            // pre-crash protocol actually resolved is covered by the
            // `Unknown`-free success path instead (see
            // `tests/recovery_adoption.rs`): a bare now-terminal record
            // cannot prove the protocol ran, so it refuses here.
            (UnknownDisposition::Escalated, true) => {}
            (UnknownDisposition::Reconciled, true) => {
                return Err(AdoptionRefusal::UnreconciledUnknown {
                    execution_id: named.execution_id,
                });
            }
            (UnknownDisposition::Reconciled, false) | (UnknownDisposition::Escalated, false) => {
                return Err(AdoptionRefusal::DispositionMismatch {
                    execution_id: named.execution_id,
                });
            }
        }
    }
    let mut effects: Vec<AdoptedEffect> = Vec::with_capacity(MAX_ADOPTION_SURVIVORS);
    for id in &claim.survivor_ids {
        let Some(record) = evidence.iter().find(|record| record.execution_id == *id) else {
            return Err(AdoptionRefusal::MissingEvidence { execution_id: *id });
        };
        let is_unknown = matches!(record.status, ToolStatus::Unknown { .. });
        if is_unknown && !seen_unknowns.contains(id) {
            return Err(AdoptionRefusal::UnreconciledUnknown { execution_id: *id });
        }
        if !is_unknown && seen_unknowns.contains(id) {
            // Any claimed disposition over never-`Unknown` evidence
            // contradicts it (verified in the loop above); adoption cannot
            // declare it.
            return Err(AdoptionRefusal::DispositionMismatch { execution_id: *id });
        }
        effects.push(AdoptedEffect {
            execution_id: record.execution_id,
            tool: record.tool.clone(),
            status: record.status.clone(),
            quarantined: is_unknown,
        });
    }
    for record in evidence {
        if !seen_claims.contains(&record.execution_id) {
            return Err(AdoptionRefusal::UnclaimedSurvivor {
                execution_id: record.execution_id,
            });
        }
    }
    Ok(AdoptedHistory { effects })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: u64, tool: &str, status: ToolStatus) -> ExecutionRecord {
        ExecutionRecord {
            execution_id: ExecutionId(id),
            tool: tool.to_owned(),
            status,
            result_disposition: crate::tool::ResultDisposition::Accepted,
        }
    }

    fn claim(
        state: SessionState,
        survivors: &[u64],
        unknowns: &[(u64, UnknownDisposition)],
        fence_token: u64,
    ) -> AdoptionClaim {
        AdoptionClaim {
            prior_session_state: state,
            survivor_ids: survivors.iter().map(|id| ExecutionId(*id)).collect(),
            unknown_effects: unknowns
                .iter()
                .map(|(id, disposition)| ClaimedUnknownEffect {
                    execution_id: ExecutionId(*id),
                    disposition: *disposition,
                })
                .collect(),
            fence_token,
        }
    }

    /// Terminal-evidence success: an all-terminal survivor set with no
    /// `unknown_effects` adopts as plain (non-quarantined) declarations.
    #[test]
    fn terminal_evidence_adopts_without_dispositions() {
        let evidence = vec![
            record(11, "workspace_read", ToolStatus::Success),
            record(
                12,
                "workspace_read",
                ToolStatus::Failed {
                    reason: "host wrote error".to_owned(),
                },
            ),
        ];
        let history = check_adoption(
            &claim(SessionState::Failed, &[11, 12], &[], 4),
            &evidence,
            4,
        )
        .expect("terminal evidence adopts");
        assert_eq!(history.len(), 2);
        assert!(history.quarantined_ids().is_empty());
        for effect in &history.effects {
            assert!(!effect.quarantined);
        }
    }

    #[test]
    fn empty_crash_adopts_vacuously() {
        let history = check_adoption(&claim(SessionState::Failed, &[], &[], 3), &[], 3)
            .expect("empty crash adopts");
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.quarantined_ids().is_empty());
    }

    #[test]
    fn bound_refuses_before_content() {
        let survivors: Vec<u64> = (1..=MAX_ADOPTION_SURVIVORS as u64 + 1).collect();
        let refusal = check_adoption(&claim(SessionState::Active, &survivors, &[], 0), &[], 1)
            .expect_err("over-bound set must refuse");
        assert!(matches!(refusal, AdoptionRefusal::TooManySurvivors { .. }));
    }

    /// Boundary table over each collection count independently: the bound
    /// admits exactly `MAX_ADOPTION_SURVIVORS` entries and refuses the next
    /// one for survivor ids, carried evidence, and the auxiliary `Unknown`
    /// disposition list. Refusals carry the observed count and the limit;
    /// content is never processed before the size gate.
    #[test]
    fn every_collection_count_refuses_over_bound_before_content() {
        let count_cases: [(usize, bool); 2] = [
            (MAX_ADOPTION_SURVIVORS, false),
            (MAX_ADOPTION_SURVIVORS + 1, true),
        ];

        for (count, over_bound) in count_cases {
            let survivors: Vec<u64> = (1..=count as u64).collect();
            let evidence: Vec<ExecutionRecord> = (1..=count as u64)
                .map(|id| record(id, "workspace_read", ToolStatus::Success))
                .collect();

            let refusal = check_adoption(
                &claim(SessionState::Failed, &survivors, &[], 7),
                &evidence,
                7,
            );
            if over_bound {
                let refusal = refusal.expect_err("over-bound survivors must refuse");
                assert!(matches!(
                    refusal,
                    AdoptionRefusal::TooManySurvivors { limit, actual }
                        if limit == MAX_ADOPTION_SURVIVORS && actual == count
                ));
            } else {
                assert!(refusal.is_ok(), "at-bound survivors must adopt");
            }

            // Survivor ids at bound, evidence over bound: the evidence count
            // alone trips the gate.
            let refusal = check_adoption(
                &claim(
                    SessionState::Failed,
                    &survivors[..survivors.len() - 1],
                    &[],
                    7,
                ),
                &evidence,
                7,
            );
            if over_bound {
                assert!(matches!(
                    refusal,
                    Err(AdoptionRefusal::TooManySurvivors { .. })
                ));
            }

            // Auxiliary disposition list over bound with a tiny survivor set
            // and no matching content: the count gate fires before the
            // membership checks that would otherwise reject these entries.
            let unknowns: Vec<(u64, UnknownDisposition)> = (1..=count as u64)
                .map(|id| (1_000 + id, UnknownDisposition::Escalated))
                .collect();
            let refusal = check_adoption(
                &claim(
                    SessionState::Failed,
                    &survivors[..survivors.len().saturating_sub(1)],
                    &unknowns,
                    7,
                ),
                &[],
                7,
            );
            if over_bound {
                let refusal = refusal.expect_err("over-bound disposition list must refuse");
                assert!(matches!(
                    refusal,
                    AdoptionRefusal::TooManySurvivors { limit, actual }
                        if limit == MAX_ADOPTION_SURVIVORS && actual == count
                ));
            } else {
                assert!(
                    matches!(refusal, Err(AdoptionRefusal::MissingEvidence { .. })),
                    "at-bound disposition list passes the size gate into content checks"
                );
            }
        }
    }

    /// The size gate outranks every content/state check: an over-bound
    /// auxiliary list refuses as `TooManySurvivors` even for a live-session
    /// claim with a stale epoch, proving no content or state inspection
    /// happens first.
    #[test]
    fn over_bound_dispositions_refuse_before_epoch_and_state_checks() {
        let unknowns: Vec<(u64, UnknownDisposition)> = (1..=MAX_ADOPTION_SURVIVORS as u64 + 1)
            .map(|id| (id, UnknownDisposition::Escalated))
            .collect();
        let refusal = check_adoption(&claim(SessionState::Active, &[1], &unknowns, 0), &[], 99)
            .expect_err("over-bound disposition list must refuse");
        assert!(matches!(
            refusal,
            AdoptionRefusal::TooManySurvivors { limit, actual }
                if limit == MAX_ADOPTION_SURVIVORS
                    && actual == MAX_ADOPTION_SURVIVORS + 1
        ));
    }

    #[test]
    fn refusal_display_names_no_untrusted_text() {
        // `Display` carries only the static reason shape plus numeric fields
        // (ids, limits, epochs): tool names and host reasons never reach the
        // message surface.
        let refusal = AdoptionRefusal::LiveSession;
        assert_eq!(
            refusal.to_string(),
            "adoption refused: prior session is live"
        );
        let refusal = AdoptionRefusal::StaleEpoch {
            claim_epoch: 7,
            current_epoch: 8,
        };
        assert_eq!(
            refusal.to_string(),
            "adoption refused: stale epoch 7, current 8"
        );
    }

    #[test]
    fn quarantined_ids_follow_claim_order() {
        let evidence = vec![
            record(
                1,
                "workspace_read",
                ToolStatus::Unknown {
                    reason: "lost ack".to_owned(),
                },
            ),
            record(2, "workspace_read", ToolStatus::Success),
            record(
                3,
                "workspace_read",
                ToolStatus::Unknown {
                    reason: "lost ack".to_owned(),
                },
            ),
        ];
        let history = check_adoption(
            &claim(
                SessionState::Failed,
                &[1, 2, 3],
                &[
                    (1, UnknownDisposition::Escalated),
                    (3, UnknownDisposition::Escalated),
                ],
                5,
            ),
            &evidence,
            5,
        )
        .expect("quarantined adoption succeeds");
        assert_eq!(history.len(), 3);
        assert_eq!(
            history.quarantined_ids(),
            vec![ExecutionId(1), ExecutionId(3)]
        );
        assert!(!history.effects[1].quarantined);
    }

    #[test]
    fn reconciled_entry_over_always_terminal_evidence_refuses() {
        // `Reconciled` is only verifiable over now-terminal evidence that the
        // pre-crash protocol actually resolved (see the success path in
        // `tests/recovery_adoption.rs`, where the evidence carries a
        // post-reconcile terminal status). A `Reconciled` disposition over
        // evidence the protocol never observed as `Unknown` contradicts the
        // claim and refuses as `DispositionMismatch`.
        let evidence = vec![record(9, "workspace_read", ToolStatus::Success)];
        let refusal = check_adoption(
            &claim(
                SessionState::Failed,
                &[9],
                &[(9, UnknownDisposition::Reconciled)],
                2,
            ),
            &evidence,
            2,
        )
        .expect_err("reconciled claim over never-unknown evidence must refuse");
        assert!(matches!(
            refusal,
            AdoptionRefusal::DispositionMismatch { .. }
        ));
    }

    #[test]
    fn orphan_unknown_entry_refuses_as_missing_evidence() {
        let evidence = vec![record(4, "workspace_read", ToolStatus::Success)];
        let refusal = check_adoption(
            &claim(
                SessionState::Failed,
                &[4],
                &[(8, UnknownDisposition::Reconciled)],
                2,
            ),
            &evidence,
            2,
        )
        .expect_err("orphan entry must refuse");
        assert!(matches!(refusal, AdoptionRefusal::MissingEvidence { .. }));
    }
}

//! Session identity, agent levels, and lifecycle.
//!
//! Mirrors the draft `AG-1`..`AG-4` and `AW-1` shape (draft disposition,
//! not an accepted contract): a fresh session starts at the
//! `inspect` read tier, elevation requires an explicit grant hook (denied by
//! default), level checks are server-side state (never client claims), and
//! cancellation is idempotent (`MP-7`). Identity separates the runtime-local
//! agent instance, the run, the session, and each execution, following the
//! ownership split in `agent-coordination.md` (agent/run owned by the AI
//! runtime; executions are attributed records reconciled by the supervising
//! backend, stubbed here).
//!
//! ## Runtime identity hierarchy (v0.1, single agent)
//!
//! ```text
//! AgentInstanceId (one runtime-local agent object)
//! ├── RunId (one turn-loop invocation)
//! ├── SessionId (one context/history scope)
//! └── ExecutionId (one side-effect attempt)
//! ```
//!
//! - [`AgentInstanceId`] names the runtime-local agent object owned by this
//!   crate. v0.1 runs a single agent, so there is no orchestration,
//!   scheduling, or multi-agent routing on top of it. Instance ids are
//!   process-unique (`AI-0059`), so co-located agents never share a handle
//!   even though routing still assumes one agent per process.
//! - [`RunId`] names the turn-loop invocation that created the session. v0.1
//!   binds one [`RunId`] per [`AgentSession`] at construction.
//! - [`SessionId`] names the context/history scope (fact store scope) carried
//!   by the [`AgentSession`] handle.
//! - [`ExecutionId`] names one dispatched tool execution. It is issued per
//!   dispatch (see [`IdIssuer::execution`]) and recorded on
//!   [`crate::agent::ExecutionRecord`] and [`crate::tool::ToolExecution`].
//!
//! ## Protocol identity vs runtime identity
//!
//! This crate is `std`-only and does not depend on the generic `bitty-agent`
//! protocol crate. The names below are deliberately distinct so a future
//! bridge can map them without a collision:
//!
//! | Identity | Owner | Shape (today) | Meaning |
//! | --- | --- | --- | --- |
//! | Protocol `AgentId` | `bitty-agent` (external) | `owner.name` principal, e.g. `"owner.name"` | External principal on the generic wire protocol |
//! | [`AgentInstanceId`] | `bitty-ai-runtime` (this crate) | `u64` handle | Runtime-local agent object |
//! | [`RunId`] | `bitty-ai-runtime` | `u64` handle | One invocation |
//! | [`SessionId`] | `bitty-ai-runtime` | `u64` handle | One context/history scope |
//! | [`ExecutionId`] | `bitty-ai-runtime` | `u64` handle | One side-effect attempt |
//!
//! Bridge mapping (later P1 work, not implemented here):
//!
//! ```text
//! protocol AgentId (owner.name)
//!       │ authorization / mapping (future bridge crate)
//!       ▼
//! AgentInstanceId
//!       ├── RunId
//!       ├── SessionId
//!       └── ExecutionId
//! ```
//!
//! Sessions are shared handles: cancelling through any clone is visible to
//! all holders, so a tool host or test peer can cancel a turn mid-dispatch
//! deterministically without threads.

use std::cell::Cell;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Runtime-local agent object handle.
///
/// This is **not** the generic `bitty-agent` protocol `AgentId`
/// (`owner.name`, the external principal). It names the single agent object
/// owned by this runtime crate; the protocol-to-instance mapping lives in a
/// future bridge crate (P1) and is intentionally absent here so this crate
/// stays `std`-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentInstanceId(pub u64);

/// One agent turn-loop invocation.
///
/// v0.1 binds one [`RunId`] per [`AgentSession`] at construction; there is
/// no run scheduler or multi-run orchestration in this skeleton phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub u64);

/// One agent session (fact store scope).
///
/// Names the context/history scope carried by [`AgentSession`]. Distinct
/// from [`RunId`] (the invocation) and [`ExecutionId`] (one dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// One dispatched tool execution (attribution handle).
///
/// Issued per dispatch via [`IdIssuer::execution`] and recorded on
/// [`crate::agent::ExecutionRecord`] and [`crate::tool::ToolExecution`].
/// An `Unknown` status on this id requires reconciliation before retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionId(pub u64);

/// Process-wide id counter behind every [`IdIssuer`] (`AI-0059`).
///
/// Ids are identity handles, never protocol-visible facts, so one monotonic
/// source per process is enough: it keeps two `Agent` instances from minting
/// colliding [`AgentInstanceId`]s. `Relaxed` is sufficient: uniqueness comes
/// from the atomic read-modify-write itself, and ids guard no other memory.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Process-level id issuer. Values start at 1; 0 is reserved as "none".
///
/// The counter is process-global (`AI-0059`): every issued value is unique
/// and strictly increasing across all [`IdIssuer`] handles and all four id
/// spaces in one process. This widens the `AI-0013` single-counter rule from
/// one issuer to the process, so a second `Agent` constructed with its own
/// `IdIssuer::default()` can no longer collide with the first one's
/// [`AgentInstanceId`]. There are no per-kind sequences: future persistence
/// or bridge code must store the `(kind, value)` pair rather than assuming
/// e.g. run 1 pairs with session 1.
///
/// The issuer itself carries no state; the private marker keeps the type
/// non-unit so [`IdIssuer::default`] stays the single construction seam
/// every existing call site already uses. Values are reproducible only for
/// the issue order inside one process run; they are local handles and must
/// never be persisted as stable external ids.
#[derive(Debug, Default)]
pub struct IdIssuer {
    /// Private marker: no state lives on the handle, the counter is
    /// [`NEXT_ID`].
    _private: (),
}

impl IdIssuer {
    /// Claim the next process-global value.
    ///
    /// Panics only if the `u64` id space is exhausted (unreachable in
    /// practice); ids never wrap back onto an already-issued value.
    fn next(&mut self) -> u64 {
        NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("id space exhausted")
            + 1
    }

    /// Issue the next agent-instance id.
    pub fn agent_instance(&mut self) -> AgentInstanceId {
        AgentInstanceId(self.next())
    }

    /// Issue the next run id.
    pub fn run(&mut self) -> RunId {
        RunId(self.next())
    }

    /// Issue the next session id.
    pub fn session(&mut self) -> SessionId {
        SessionId(self.next())
    }

    /// Issue the next execution id.
    pub fn execution(&mut self) -> ExecutionId {
        ExecutionId(self.next())
    }
}

/// Attenuated authority tier (draft `AG` levels in `ai-architecture.md`).
/// `Own` is the spec `self` tier, renamed because `self` is a Rust keyword.
/// This tier is distinct from the ephemeral `AgentWorkspace` (`AW-1`) and
/// from context levels `L0`/`L1`; the `AgentInstanceId` split stays per AI-0013.
/// This is only a policy-profile label, not a capability proof: holding a tier
/// grants nothing by itself; every dispatch still passes the `AG-1`..`AG-4`
/// checks and the `R2` unified authorization backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentLevel {
    /// Read workspace/project/git/diagnostics/zone-scoped snapshots, list
    /// models and tools. Default for every fresh session (`AG-1`).
    Inspect,
    /// `Inspect` plus ephemeral scratch and model turns.
    Own,
    /// `Own` plus reads/writes within the current workspace.
    Workspace,
    /// `Workspace` plus cross-workspace/window actions under separate
    /// per-target grants.
    All,
}

impl AgentLevel {
    /// Attenuation rank: higher includes lower tiers' read authorities.
    #[must_use]
    pub fn rank(&self) -> u8 {
        match self {
            Self::Inspect => 0,
            Self::Own => 1,
            Self::Workspace => 2,
            Self::All => 3,
        }
    }

    /// Whether this tier includes at least the authority of `other`.
    #[must_use]
    pub fn includes(&self, other: Self) -> bool {
        self.rank() >= other.rank()
    }
}

/// Session lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Accepting turns.
    Active,
    /// Turn loop finished cleanly.
    Completed,
    /// Turn loop failed.
    Failed,
    /// Cancel requested; in-flight work reconciles before retry (`MP-7`).
    Canceled,
}

/// Session lifecycle errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Elevation or transition requested on a terminal session.
    AlreadyTerminated {
        /// Observed state.
        state: SessionState,
    },
    /// Elevation grant hook refused.
    ElevationDenied {
        /// Requested tier.
        requested: AgentLevel,
    },
}

impl Display for SessionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::AlreadyTerminated { state } => {
                write!(f, "session already terminated ({state:?})")
            }
            Self::ElevationDenied { requested } => {
                write!(f, "elevation to {requested:?} denied")
            }
        }
    }
}

impl std::error::Error for SessionError {}

/// Elevation grant hook (`AG-1`/`AG-3`). The host implements per-client
/// consent grants behind this seam; the skeleton only calls it. This is not
/// the consent ledger: grants here are process-local test stubs unless the
/// host wires the real ledger.
pub trait ElevationGrant {
    /// Decide whether `requested` may replace `current`.
    fn elevation_granted(&self, current: AgentLevel, requested: AgentLevel) -> bool;
}

/// Elevation hook that denies everything (default).
#[derive(Debug, Default)]
pub struct DenyAllElevations;

impl ElevationGrant for DenyAllElevations {
    fn elevation_granted(&self, _current: AgentLevel, _requested: AgentLevel) -> bool {
        false
    }
}

#[derive(Debug)]
struct SessionInner {
    agent_instance_id: AgentInstanceId,
    run_id: RunId,
    session_id: SessionId,
    level: Cell<AgentLevel>,
    generation: Cell<u64>,
    state: Cell<SessionState>,
}

/// One agent session. Clones share identity and lifecycle: `cancel`,
/// `elevate`, `rotate_generation`, and `finish` are visible through every
/// handle.
///
/// # Single-threaded contract (`!Send`)
///
/// A session is an `Rc`-shared handle over `Cell` state, so it is
/// deliberately `!Send` and `!Sync`; the compile-time test
/// `agent_session_is_not_send` fails the build if that ever changes. Create,
/// use, and drop a session on one thread: never move it across a thread
/// boundary, never hold it across an `.await` on a multi-threaded executor,
/// and never share it between concurrent tasks.
///
/// This is a declared v0.1 contract, not an oversight (review 07 `P1-7`,
/// suggestion `S-10`). Lock-free `Cell` state keeps cancellation, elevation,
/// and generation rotation deterministic for a single driver; the trade-off
/// is that a multi-threaded async host cannot own the session directly.
///
/// ## Host wiring: drive each turn on one thread
///
/// An async host (for example a Tokio multi-threaded runtime) keeps the
/// contract by constructing and driving the session inside one blocking
/// task, so the session never crosses an `await` and only `Send` values
/// leave the task:
///
/// ```ignore
/// // Host-side sketch. The async runtime (here `tokio`) and this wiring live
/// // in the host crate; `bitty-ai-runtime` stays std-only and `!Send`.
/// let summary = tokio::task::spawn_blocking(move || {
///     let mut ids = IdIssuer::default();
///     let session = AgentSession::new(ids.agent_instance(), ids.run(), ids.session());
///     // Construction, the whole turn, and drop all happen on this one
///     // blocking thread; nothing `!Send` crosses the spawn boundary.
///     run_turn(&session, request) // returns a `Send` summary
/// })
/// .await??;
/// ```
///
/// Migrating the session to `Arc<Mutex<SessionInner>>` or an atomic state
/// machine so it can be `Send` is tracked as follow-up; it is intentionally
/// out of scope here (`S-10`) and would need cancellation-visibility ordering
/// re-proven across threads.
#[derive(Debug, Clone)]
pub struct AgentSession(Rc<SessionInner>);

impl AgentSession {
    /// Start a session at `inspect`/`Active`, generation 1 (`AG-1`).
    #[must_use]
    pub fn new(agent_instance_id: AgentInstanceId, run_id: RunId, session_id: SessionId) -> Self {
        Self(Rc::new(SessionInner {
            agent_instance_id,
            run_id,
            session_id,
            level: Cell::new(AgentLevel::Inspect),
            generation: Cell::new(1),
            state: Cell::new(SessionState::Active),
        }))
    }

    /// Bound runtime-local agent instance id.
    #[must_use]
    pub fn agent_instance_id(&self) -> AgentInstanceId {
        self.0.agent_instance_id
    }

    /// Bound run id.
    #[must_use]
    pub fn run_id(&self) -> RunId {
        self.0.run_id
    }

    /// Session id.
    #[must_use]
    pub fn session_id(&self) -> SessionId {
        self.0.session_id
    }

    /// Current authority tier (server-side state; client claims are ignored).
    #[must_use]
    pub fn level(&self) -> AgentLevel {
        self.0.level.get()
    }

    /// Current generation (bumped by rotation, invalidating elevation).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.0.generation.get()
    }

    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        self.0.state.get()
    }

    /// Whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.state.get() == SessionState::Canceled
    }

    /// Request cancellation. Idempotent: only `Active` transitions; terminal
    /// states are kept, so one waiter's cancel never rewrites another
    /// waiter's completed outcome (`MP-7`).
    pub fn cancel(&self) {
        if self.0.state.get() == SessionState::Active {
            self.0.state.set(SessionState::Canceled);
        }
    }

    /// Record terminal turn-loop completion. Only transitions from `Active`;
    /// a prior `Canceled` is never overwritten.
    pub fn finish(&self, failed: bool) {
        if self.0.state.get() == SessionState::Active {
            self.0.state.set(if failed {
                SessionState::Failed
            } else {
                SessionState::Completed
            });
        }
    }

    /// Elevate to `requested` through the grant hook (`AG-1`). Equal-or-lower
    /// tiers are a no-op success; higher tiers require the hook.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::AlreadyTerminated`] on a terminal session or
    /// [`SessionError::ElevationDenied`] when the hook refuses.
    pub fn elevate(
        &self,
        requested: AgentLevel,
        grant: &dyn ElevationGrant,
    ) -> Result<(), SessionError> {
        if self.0.state.get() != SessionState::Active {
            return Err(SessionError::AlreadyTerminated {
                state: self.0.state.get(),
            });
        }
        if requested.rank() > self.0.level.get().rank() {
            if grant.elevation_granted(self.0.level.get(), requested) {
                self.0.level.set(requested);
                Ok(())
            } else {
                Err(SessionError::ElevationDenied { requested })
            }
        } else {
            Ok(())
        }
    }

    /// Rotate the generation, resetting the tier to `inspect` (`AG-2`): a
    /// suspend/dispose/reload invalidates prior elevation and re-grant needs
    /// a fresh prompt. No-op on terminal sessions.
    pub fn rotate_generation(&self) {
        if self.0.state.get() == SessionState::Active {
            self.0.generation.set(self.0.generation.get() + 1);
            self.0.level.set(AgentLevel::Inspect);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> AgentSession {
        let mut ids = IdIssuer::default();
        AgentSession::new(ids.agent_instance(), ids.run(), ids.session())
    }

    #[test]
    fn id_issuer_increases_monotonically_across_kinds() {
        let mut ids = IdIssuer::default();
        let values = [
            ids.agent_instance().0,
            ids.run().0,
            ids.session().0,
            ids.execution().0,
            ids.execution().0,
        ];
        // 0 stays reserved as "none": no issued id may be zero.
        for value in values {
            assert_ne!(value, 0);
        }
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn id_issuer_is_process_shared_across_agents() {
        // P1-6 regression: each `Agent` builds its own `IdIssuer::default()`;
        // two co-located agents must never mint the same instance id.
        let mut first_agent = IdIssuer::default();
        let mut second_agent = IdIssuer::default();
        assert_ne!(first_agent.agent_instance(), second_agent.agent_instance());
        // Interleaved issuance is strictly increasing: one shared process
        // counter, not two independent counters restarting at 1.
        for _ in 0..16 {
            let first = first_agent.agent_instance();
            let second = second_agent.agent_instance();
            assert_ne!(first, second);
            assert!(first.0 < second.0);
        }
    }

    #[test]
    fn session_preserves_bound_identity() {
        let mut ids = IdIssuer::default();
        let instance = ids.agent_instance();
        let run = ids.run();
        let session_id = ids.session();
        let session = AgentSession::new(instance, run, session_id);
        assert_eq!(session.agent_instance_id(), instance);
        assert_eq!(session.run_id(), run);
        assert_eq!(session.session_id(), session_id);
    }

    #[test]
    fn fresh_session_is_inspect_active() {
        let session = session();
        assert_eq!(session.level(), AgentLevel::Inspect);
        assert_eq!(session.state(), SessionState::Active);
        assert_eq!(session.generation(), 1);
    }

    #[test]
    fn cancel_is_idempotent_and_shared() {
        let session = session();
        let clone = session.clone();
        session.cancel();
        assert!(clone.is_cancelled());
        clone.cancel();
        assert_eq!(session.state(), SessionState::Canceled);
        session.finish(false);
        assert_eq!(session.state(), SessionState::Canceled);
    }

    #[test]
    fn elevation_denied_by_default_and_reset_by_rotation() {
        struct Allow;
        impl ElevationGrant for Allow {
            fn elevation_granted(&self, _c: AgentLevel, _r: AgentLevel) -> bool {
                true
            }
        }
        let session = session();
        assert!(matches!(
            session.elevate(AgentLevel::Workspace, &DenyAllElevations),
            Err(SessionError::ElevationDenied { .. })
        ));
        assert_eq!(session.level(), AgentLevel::Inspect);
        session
            .elevate(AgentLevel::Workspace, &Allow)
            .expect("grant allows");
        assert_eq!(session.level(), AgentLevel::Workspace);
        session.rotate_generation();
        assert_eq!(session.level(), AgentLevel::Inspect);
        assert_eq!(session.generation(), 2);
    }

    #[test]
    fn terminal_session_rejects_elevation() {
        let session = session();
        session.cancel();
        assert!(matches!(
            session.elevate(AgentLevel::Own, &DenyAllElevations),
            Err(SessionError::AlreadyTerminated { .. })
        ));
    }

    #[test]
    fn agent_session_is_not_send() {
        // Review 07 `P1-7` / `S-10`: `AgentSession` must stay `!Send` while it
        // is an `Rc<Cell<..>>` handle. This is a compile-time assertion, not a
        // runtime check, using the classic two-blanket-impl trick (the
        // static_assertions `assert_not_impl_any` approach, hand-rolled here
        // to keep the crate dependency-free): if `AgentSession` were `Send`,
        // both `AmbiguousIfSend<()>` and `AmbiguousIfSend<[()]>` would apply
        // and resolving `AmbiguousIfSend<_>` would fail with E0283, breaking
        // the build. If this stops compiling, revisit the single-threaded
        // contract documented on `AgentSession`.
        trait AmbiguousIfSend<A: ?Sized> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<[()]> for T {}
        let _ = <AgentSession as AmbiguousIfSend<_>>::marker;
    }
}

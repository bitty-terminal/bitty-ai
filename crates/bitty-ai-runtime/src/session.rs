//! Session identity, agent levels, and lifecycle.
//!
//! Mirrors the `AG-1`..`AG-4` and `AW-1` shape: a fresh session starts at the
//! `inspect` read tier, elevation requires an explicit grant hook (denied by
//! default), level checks are server-side state (never client claims), and
//! cancellation is idempotent (`MP-7`). Identity separates the logical agent,
//! the run, the session, and each execution, following the ownership split in
//! `agent-coordination.md` (agent/run owned by the AI runtime; executions are
//! attributed records reconciled by the supervising backend, stubbed here).
//!
//! Sessions are shared handles: cancelling through any clone is visible to
//! all holders, so a tool host or test peer can cancel a turn mid-dispatch
//! deterministically without threads.

use std::cell::Cell;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

/// Logical agent handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u64);

/// One agent turn-loop invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub u64);

/// One agent session (fact store scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// One dispatched tool execution (attribution handle).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionId(pub u64);

/// Deterministic id issuer. Values start at 1; 0 is reserved as "none".
#[derive(Debug, Default)]
pub struct IdIssuer {
    next: u64,
}

impl IdIssuer {
    /// Issue the next agent id.
    pub fn agent(&mut self) -> AgentId {
        self.next += 1;
        AgentId(self.next)
    }

    /// Issue the next run id.
    pub fn run(&mut self) -> RunId {
        self.next += 1;
        RunId(self.next)
    }

    /// Issue the next session id.
    pub fn session(&mut self) -> SessionId {
        self.next += 1;
        SessionId(self.next)
    }

    /// Issue the next execution id.
    pub fn execution(&mut self) -> ExecutionId {
        self.next += 1;
        ExecutionId(self.next)
    }
}

/// Attenuated authority tier (`AG` levels). `Own` is the spec `self` tier,
/// renamed because `self` is a Rust keyword.
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
    agent_id: AgentId,
    run_id: RunId,
    session_id: SessionId,
    level: Cell<AgentLevel>,
    generation: Cell<u64>,
    state: Cell<SessionState>,
}

/// One agent session. Clones share identity and lifecycle: `cancel`,
/// `elevate`, `rotate_generation`, and `finish` are visible through every
/// handle.
#[derive(Debug, Clone)]
pub struct AgentSession(Rc<SessionInner>);

impl AgentSession {
    /// Start a session at `inspect`/`Active`, generation 1 (`AG-1`).
    #[must_use]
    pub fn new(agent_id: AgentId, run_id: RunId, session_id: SessionId) -> Self {
        Self(Rc::new(SessionInner {
            agent_id,
            run_id,
            session_id,
            level: Cell::new(AgentLevel::Inspect),
            generation: Cell::new(1),
            state: Cell::new(SessionState::Active),
        }))
    }

    /// Bound agent id.
    #[must_use]
    pub fn agent_id(&self) -> AgentId {
        self.0.agent_id
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
        AgentSession::new(ids.agent(), ids.run(), ids.session())
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
}

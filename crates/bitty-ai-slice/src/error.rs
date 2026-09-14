//! Typed failures for the vertical slice.
//!
//! Every failure is total and happen-before-side-effects where the architecture
//! requires it (for example a `BudgetExceeded` fails before provider I/O per
//! `MP-5`, and an unknown tool fails before dispatch per `TB-3`).

use core::fmt;

use bitty_ipc::error::IpcError;

/// A fail-closed slice error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceError {
    /// A generic IPC primitive rejected the operation (wire, scope, or bound).
    Ipc(IpcError),
    /// The generic method registry does not know the requested method.
    UnsupportedHostMethod { method: String },
    /// No per-client consent grant exists for the required scope.
    ConsentRequired { scope: &'static str },
    /// Assembled context exceeded the Context Budget before provider I/O.
    ContextBudgetExceeded { limit: usize, actual: usize },
    /// The host answered with an error, or the response was unusable.
    ContextUnavailable { reason: String },
    /// The tool is not in the bounded registry (unknown tool fails closed).
    ToolNotRegistered { name: String },
    /// The tool exists but the current permission profile denies it.
    ToolDenied { name: String },
    /// Tool arguments exceeded the bounded argument size (`TB-3`).
    ToolArgumentsTooLarge { limit: usize, actual: usize },
    /// The per-turn tool-call cap was exceeded (`TB-6`) or the registry cap was
    /// exceeded (`TB-2`).
    ToolCallLimitExceeded { limit: usize },
    /// A tool result exceeded the bounded result size.
    ToolResultTooLarge { limit: usize, actual: usize },
    /// A streamed chunk violated the RC-10 or fragment bound.
    StreamViolation { reason: String },
    /// A provider request exceeded a declared message bound.
    ProviderBoundExceeded {
        field: &'static str,
        limit: usize,
        actual: usize,
    },
}

impl fmt::Display for SliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ipc(err) => write!(f, "ipc primitive rejected the operation: {err}"),
            Self::UnsupportedHostMethod { method } => {
                write!(f, "host method '{method}' is not in the generic registry")
            }
            Self::ConsentRequired { scope } => {
                write!(f, "no consent grant for required scope '{scope}'")
            }
            Self::ContextBudgetExceeded { limit, actual } => write!(
                f,
                "context budget exceeded before provider I/O: limit {limit} bytes, got {actual}"
            ),
            Self::ContextUnavailable { reason } => {
                write!(f, "context unavailable from host: {reason}")
            }
            Self::ToolNotRegistered { name } => {
                write!(f, "tool '{name}' is not registered")
            }
            Self::ToolDenied { name } => {
                write!(f, "tool '{name}' is denied by the permission profile")
            }
            Self::ToolArgumentsTooLarge { limit, actual } => {
                write!(
                    f,
                    "tool arguments too large: limit {limit} bytes, got {actual}"
                )
            }
            Self::ToolCallLimitExceeded { limit } => {
                write!(f, "tool call limit exceeded: limit {limit}")
            }
            Self::ToolResultTooLarge { limit, actual } => write!(
                f,
                "tool result too large: limit {limit} bytes, got {actual}"
            ),
            Self::StreamViolation { reason } => write!(f, "stream chunk rejected: {reason}"),
            Self::ProviderBoundExceeded {
                field,
                limit,
                actual,
            } => write!(
                f,
                "provider bound exceeded for {field}: limit {limit} bytes, got {actual}"
            ),
        }
    }
}

impl std::error::Error for SliceError {}

impl From<IpcError> for SliceError {
    fn from(value: IpcError) -> Self {
        Self::Ipc(value)
    }
}

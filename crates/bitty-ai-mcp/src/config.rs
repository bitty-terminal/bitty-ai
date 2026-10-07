//! Fail-closed MCP server configuration.
//!
//! [`McpServerConfig`] is pure data plus [`McpServerConfig::validate`]: the
//! child is never spawned through a shell, the working directory is pinned
//! to an absolute path without `..` traversal, the whole-operation timeout
//! is bounded to `1..=MAX_TIMEOUT_MS`, and the tool allowlist is non-empty
//! and capped at the runtime registry bound (32). Secrets travel only as
//! [`CredentialRef`] names; values are resolved at spawn time by
//! [`crate::supervise`] and never appear in errors or logs.

use crate::error::{McpError, bound_error_text};
use crate::{MAX_TIMEOUT_MS, McpStage};

/// Maximum server id length in bytes (namespace-safe for sanitized names).
pub const MAX_SERVER_ID_LEN: usize = 32;
/// Maximum command (program) length in bytes.
pub const MAX_COMMAND_LEN: usize = 256;
/// Maximum argument count per command.
pub const MAX_COMMAND_ARGS: usize = 32;
/// Maximum bytes per command argument.
pub const MAX_COMMAND_ARG_LEN: usize = 1024;
/// Maximum credential references per server.
pub const MAX_ENV_REFS: usize = 16;
/// Maximum credential helper arguments.
pub const MAX_CREDENTIAL_ARGS: usize = 16;
/// Maximum allowlisted tools per server (runtime registry parity: 32).
pub const MAX_ALLOWLIST_LEN: usize = 32;
/// Maximum raw allowlist entry length in bytes.
pub const MAX_ALLOWLIST_ENTRY_LEN: usize = 64;
/// Maximum working-directory length in bytes.
pub const MAX_CWD_LEN: usize = 1024;
/// Maximum environment variable name length in bytes.
pub const MAX_ENV_VAR_LEN: usize = 64;

/// Bytes that would let a program string become a shell invocation.
///
/// The child spawns directly (`Command::new`, never a shell), and these
/// metacharacters are refused at validation so a config can never smuggle a
/// pipeline, substitution, redirect, glob, or background operator into
/// `argv[0]`. Spaces are allowed: a path containing spaces stays one
/// argument under direct spawn.
fn is_shell_byte(byte: u8) -> bool {
    matches!(
        byte,
        b';' | b'|'
            | b'&'
            | b'$'
            | b'`'
            | b'('
            | b')'
            | b'<'
            | b'>'
            | b'\\'
            | b'"'
            | b'\''
            | b'!'
            | b'*'
            | b'?'
            | b'~'
            | b'#'
    )
}

/// Credential reference: where a secret comes from, never its value.
///
/// Minimal local shape (the Core `CredentialRef` lives behind an external
/// crate this client must not depend on): `Env` passes one host variable
/// through into the closed child environment under the same name, `Cmd`
/// runs a bounded shell-free helper whose trimmed stdout becomes the child
/// variable `name`. `Display` quotes reference names only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialRef {
    /// Pass host variable `var` into the child environment.
    Env {
        /// Variable name (`^[A-Z][A-Z0-9_]*$`).
        var: String,
    },
    /// Run `program args`, store trimmed stdout as child variable `name`.
    Cmd {
        /// Child variable name (`^[A-Z][A-Z0-9_]*$`).
        name: String,
        /// Helper program (same no-shell rule as server commands).
        program: String,
        /// Helper arguments (bounded, NUL-free).
        args: Vec<String>,
    },
}

impl CredentialRef {
    /// Reference name for names-only diagnostics (`env:VAR` / `cmd:program`).
    #[must_use]
    pub fn ref_name(&self) -> String {
        match self {
            Self::Env { var } => format!("env:{var}"),
            Self::Cmd { program, .. } => format!("cmd:{program}"),
        }
    }

    fn validate(&self) -> Result<(), McpError> {
        match self {
            Self::Env { var } => {
                if !is_env_name(var) {
                    return Err(McpError::invalid_config(
                        "credential env var must match ^[A-Z][A-Z0-9_]*$ (1..=64 bytes)",
                    ));
                }
                Ok(())
            }
            Self::Cmd {
                name,
                program,
                args,
            } => {
                if !is_env_name(name) {
                    return Err(McpError::invalid_config(
                        "credential target name must match ^[A-Z][A-Z0-9_]*$ (1..=64 bytes)",
                    ));
                }
                validate_program(program)?;
                if args.len() > MAX_CREDENTIAL_ARGS {
                    return Err(McpError::invalid_config(
                        "credential helper carries too many arguments",
                    ));
                }
                for arg in args {
                    if arg.len() > MAX_COMMAND_ARG_LEN || arg.contains('\0') {
                        return Err(McpError::invalid_config(
                            "credential helper argument is over-bound or NUL-bearing",
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for CredentialRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.ref_name())
    }
}

fn is_env_name(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_ENV_VAR_LEN {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_uppercase() {
        return false;
    }
    bytes
        .iter()
        .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || *byte == b'_')
}

fn is_server_id(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_SERVER_ID_LEN {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_lowercase() {
        return false;
    }
    bytes
        .iter()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
}

fn validate_program(program: &str) -> Result<(), McpError> {
    if program.is_empty() || program.len() > MAX_COMMAND_LEN {
        return Err(McpError::invalid_config(
            "program path is empty or over-bound",
        ));
    }
    if program.contains('\0') || program.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(McpError::new(
            McpStage::Config,
            crate::error::McpFailure::InvalidConfig {
                reason: bound_error_text(
                    "program path carries control bytes",
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        ));
    }
    if program.bytes().any(is_shell_byte) {
        return Err(McpError::new(
            McpStage::Config,
            crate::error::McpFailure::InvalidConfig {
                reason: bound_error_text(
                    "program path carries shell metacharacters (no shell; use direct argv)",
                    crate::error::MAX_ERROR_TEXT_BYTES,
                ),
            },
        ));
    }
    Ok(())
}

fn is_allowlist_entry(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_ALLOWLIST_ENTRY_LEN {
        return false;
    }
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.' || byte == b'-')
}

/// MCP server launch configuration plus tool allowlist.
///
/// Fail-closed validation ([`McpServerConfig::validate`]): command required
/// and shell-free, `cwd` pinned absolute without `..`, timeout within
/// `1..=MAX_TIMEOUT_MS`, allowlist non-empty with at most [`MAX_ALLOWLIST_LEN`]
/// raw MCP tool names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    /// Server id (`^[a-z][a-z0-9_]*$`, 1..=32 bytes): namespaces sanitized
    /// tool names (`mcp_<id>_<tool>`).
    pub id: String,
    /// Helper program: direct spawn, never a shell.
    pub command: String,
    /// Helper arguments (bounded, NUL-free).
    pub args: Vec<String>,
    /// Credential references resolved into the closed child environment.
    pub env_refs: Vec<CredentialRef>,
    /// Pinned absolute working directory (no `..` segments).
    pub cwd: String,
    /// Whole-operation timeout in milliseconds (`1..=MAX_TIMEOUT_MS`).
    pub timeout_ms: u64,
    /// Raw MCP tool names this server may expose (non-empty, ≤32).
    pub tool_allowlist: Vec<String>,
}

impl McpServerConfig {
    /// Validate fail-closed. Every refusal names the violated bound only;
    /// no values, paths, or secrets enter the error.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] with [`McpStage::Config`] for any shape
    /// violation.
    pub fn validate(&self) -> Result<(), McpError> {
        if !is_server_id(&self.id) {
            return Err(McpError::invalid_config(
                "server id must match ^[a-z][a-z0-9_]*$ (1..=32 bytes)",
            ));
        }
        if self.command.is_empty() {
            return Err(McpError::invalid_config("command is required"));
        }
        validate_program(&self.command)?;
        if self.args.len() > MAX_COMMAND_ARGS {
            return Err(McpError::invalid_config(
                "server command carries too many arguments",
            ));
        }
        for arg in &self.args {
            if arg.len() > MAX_COMMAND_ARG_LEN || arg.contains('\0') {
                return Err(McpError::invalid_config(
                    "server argument is over-bound or NUL-bearing",
                ));
            }
        }
        if self.env_refs.len() > MAX_ENV_REFS {
            return Err(McpError::invalid_config(
                "server carries too many credential references",
            ));
        }
        for reference in &self.env_refs {
            reference.validate()?;
        }
        if self.cwd.is_empty() || self.cwd.len() > MAX_CWD_LEN {
            return Err(McpError::invalid_config("cwd is empty or over-bound"));
        }
        if !self.cwd.starts_with('/') {
            return Err(McpError::invalid_config("cwd must be an absolute path"));
        }
        if self.cwd.contains('\0') {
            return Err(McpError::invalid_config("cwd must not contain NUL"));
        }
        if self.cwd.split('/').any(|segment| segment == "..") {
            return Err(McpError::invalid_config(
                "cwd must not contain '..' segments",
            ));
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_TIMEOUT_MS {
            return Err(McpError::invalid_config(
                "timeout_ms must be within 1..=30000",
            ));
        }
        if self.tool_allowlist.is_empty() {
            return Err(McpError::invalid_config("tool allowlist must not be empty"));
        }
        if self.tool_allowlist.len() > MAX_ALLOWLIST_LEN {
            return Err(McpError::invalid_config(
                "tool allowlist exceeds 32 entries",
            ));
        }
        for entry in &self.tool_allowlist {
            if !is_allowlist_entry(entry) {
                return Err(McpError::invalid_config(
                    "allowlist entry must be 1..=64 bytes of [a-zA-Z0-9_.-]",
                ));
            }
        }
        Ok(())
    }

    /// Effective whole-operation timeout: the configured value, defaulting
    /// to [`crate::DEFAULT_MCP_TIMEOUT_MS`] when zero (callers should still
    /// prefer [`McpServerConfig::validate`], which refuses zero).
    #[must_use]
    pub fn effective_timeout_ms(&self) -> u64 {
        if self.timeout_ms == 0 {
            crate::DEFAULT_MCP_TIMEOUT_MS
        } else {
            self.timeout_ms.min(MAX_TIMEOUT_MS)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> McpServerConfig {
        McpServerConfig {
            id: "demo".to_owned(),
            command: "/bin/fake-mcp".to_owned(),
            args: vec!["--stdio".to_owned()],
            env_refs: vec![CredentialRef::Env {
                var: "DEMO_TOKEN".to_owned(),
            }],
            cwd: "/tmp/bitty".to_owned(),
            timeout_ms: 10_000,
            tool_allowlist: vec!["echo".to_owned()],
        }
    }

    #[test]
    fn valid_config_passes() {
        valid_config().validate().expect("valid config");
    }

    #[test]
    fn command_is_required_and_shell_free() {
        let mut config = valid_config();
        config.command.clear();
        assert!(config.validate().is_err());
        for shell in [
            "sh -c 'x'",
            "a;b",
            "a|b",
            "a&b",
            "$(x)",
            "`x`",
            "a>b",
            "a<b",
            "a*b",
            "~x",
        ] {
            config.command = shell.to_owned();
            assert!(config.validate().is_err(), "shell escaped: {shell}");
        }
        // Spaces stay legal: one argv element under direct spawn.
        config.command = "/opt/my tools/helper".to_owned();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn cwd_is_pinned_absolute_without_traversal() {
        let mut config = valid_config();
        for cwd in [
            "",
            "relative/dir",
            "/tmp/../etc",
            "/x/y/../../z",
            "/has\0nul",
        ] {
            config.cwd = cwd.to_owned();
            assert!(config.validate().is_err(), "cwd escaped: {cwd:?}");
        }
        config.cwd = "/tmp/bitty".to_owned();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn timeout_is_bounded() {
        let mut config = valid_config();
        for timeout in [0, 30_001, u64::MAX] {
            config.timeout_ms = timeout;
            assert!(config.validate().is_err());
        }
        config.timeout_ms = 30_000;
        assert!(config.validate().is_ok());
        config.timeout_ms = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn allowlist_is_non_empty_and_bounded() {
        let mut config = valid_config();
        config.tool_allowlist.clear();
        assert!(config.validate().is_err());
        config.tool_allowlist = vec!["t".to_owned(); MAX_ALLOWLIST_LEN + 1];
        assert!(config.validate().is_err());
        config.tool_allowlist = vec!["bad name!".to_owned()];
        assert!(config.validate().is_err());
        config.tool_allowlist = vec!["echo".to_owned(), "list.files-2".to_owned()];
        assert!(config.validate().is_ok());
    }

    #[test]
    fn credential_refs_are_validated() {
        let mut config = valid_config();
        config.env_refs = vec![CredentialRef::Env {
            var: "lowercase".to_owned(),
        }];
        assert!(config.validate().is_err());
        config.env_refs = vec![CredentialRef::Cmd {
            name: "TOKEN".to_owned(),
            program: "pass;show".to_owned(),
            args: Vec::new(),
        }];
        assert!(config.validate().is_err());
        config.env_refs = vec![CredentialRef::Env {
            var: "DEMO_TOKEN".to_owned(),
        }];
        assert!(config.validate().is_ok());
    }
}

//! Supervised stdio children: one child per server, closed environment.
//!
//! [`spawn_server`] starts a server helper with the credential-spawn
//! discipline (direct `Command` spawn, no shell; `stdin` piped; `stderr`
//! discarded at the OS level; `stdout` drained on exactly one worker thread
//! with the frame cap-plus-one probe; `cwd` pinned; closed environment plus
//! resolved credential refs only). One live child exists per server: exits
//! and timeouts kill and reap the child, in-flight calls resolve to
//! `Unknown` at the bridge, and there is deliberately NO auto-restart —
//! only an explicit [`McpHost`]-gated `spawn_server` call revives a server.
//!
//! Reconnect pressure is owned by [`McpHost`] through per-server
//! [`ServerCooldown`]: consecutive failures back off geometrically
//! (`COOLDOWN_BASE_MS << failures`, capped at `COOLDOWN_CAP_MS`), successes
//! reset the count. The cooldown map is the only timeful policy state and
//! runs entirely on caller-supplied `now_ms`.

use std::io::{BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use crate::McpTransport;
use crate::config::{CredentialRef, McpServerConfig};
use crate::error::{McpError, McpFailure, McpStage, bound_error_name, bound_error_text};
use crate::frame::{FramedLines, MAX_FRAME_BYTES};

/// Stdout-drain poll interval: small enough that deadline overshoot stays
/// negligible, large enough to avoid a hot spin (credential precedent).
pub const SUPERVISE_POLL_INTERVAL: Duration = Duration::from_millis(10);
/// Credential helper deadline (credential-spawn precedent: 10s).
pub const CREDENTIAL_CMD_TIMEOUT_MS: u64 = 10_000;
/// Credential helper stdout cap (secret-value bound precedent: 4 KiB).
pub const CREDENTIAL_CMD_MAX_OUTPUT: usize = 4 * 1024;
/// Cooldown base for [`ServerCooldown`] (first retry waits 1s).
pub const COOLDOWN_BASE_MS: u64 = 1_000;
/// Cooldown ceiling for [`ServerCooldown`] (geometric backoff caps at 60s).
pub const COOLDOWN_CAP_MS: u64 = 60_000;
/// Maximum tracked servers per [`McpHost`] (fail-closed roster bound).
pub const MAX_HOST_SERVERS: usize = 32;

/// Pipe event delivered by the stdout-drain worker.
enum PipeEvent {
    /// One well-formed frame.
    Line(String),
    /// One oversize line (frame refused; stream stayed aligned).
    Oversize {
        /// Bound in bytes.
        limit: usize,
        /// Probe value (cap-plus-one).
        actual: usize,
    },
    /// Clean EOF (child closed stdout).
    Eof,
}

/// Geometric, capped per-server connect cooldown.
///
/// Consecutive failures back off as `min(BASE << failures, CAP)` measured
/// from the failure instant on caller-supplied `now_ms`; any success resets
/// the count. Saturating arithmetic throughout: a pathological failure
/// count pins at the cap instead of overflowing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerCooldown {
    /// Consecutive failures since the last success.
    pub consecutive_failures: u32,
    /// Earliest `now_ms` that may connect.
    pub not_before_ms: u64,
}

impl ServerCooldown {
    /// Whether `now_ms` may connect.
    #[must_use]
    pub fn may_connect(&self, now_ms: u64) -> bool {
        now_ms >= self.not_before_ms
    }

    /// Milliseconds to wait from `now_ms` before connecting (zero when
    /// [`ServerCooldown::may_connect`]).
    #[must_use]
    pub fn wait_ms(&self, now_ms: u64) -> u64 {
        self.not_before_ms.saturating_sub(now_ms)
    }

    /// Record a success: reset the backoff.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.not_before_ms = 0;
    }

    /// Record a failure at `now_ms`: geometric backoff, capped.
    pub fn record_failure(&mut self, now_ms: u64) {
        let shift = self.consecutive_failures.min(16);
        let backoff = COOLDOWN_BASE_MS
            .saturating_mul(1_u64 << shift)
            .min(COOLDOWN_CAP_MS);
        self.not_before_ms = now_ms.saturating_add(backoff);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }
}

/// Host-side roster owning one [`ServerCooldown`] per server id.
///
/// The host never spawns by itself; it gates explicit reconnects. Register
/// each configured server once ([`McpHost::register`]), then check
/// [`McpHost::may_connect`] before [`spawn_server`] and record the outcome
/// after. Unknown ids fail closed.
#[derive(Debug, Default)]
pub struct McpHost {
    entries: Vec<(String, ServerCooldown)>,
}

impl McpHost {
    /// Empty roster.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a server id (idempotent for the same id).
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::Supervision`] past [`MAX_HOST_SERVERS`].
    pub fn register(&mut self, server_id: &str) -> Result<(), McpError> {
        if self.entries.iter().any(|(id, _)| id == server_id) {
            return Ok(());
        }
        if self.entries.len() >= MAX_HOST_SERVERS {
            return Err(McpError::new(
                McpStage::Supervise,
                McpFailure::Supervision {
                    detail: bound_error_text(
                        "host roster is full",
                        crate::error::MAX_ERROR_TEXT_BYTES,
                    ),
                },
            ));
        }
        self.entries
            .push((server_id.to_owned(), ServerCooldown::default()));
        Ok(())
    }

    /// Cooldown state for a registered server.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::Supervision`] for an unregistered id.
    pub fn cooldown(&self, server_id: &str) -> Result<ServerCooldown, McpError> {
        self.entries
            .iter()
            .find(|(id, _)| id == server_id)
            .map(|(_, cooldown)| *cooldown)
            .ok_or_else(|| {
                McpError::new(
                    McpStage::Supervise,
                    McpFailure::Supervision {
                        detail: bound_error_text(
                            &format!("unknown server '{server_id}'"),
                            crate::error::MAX_ERROR_TEXT_BYTES,
                        ),
                    },
                )
            })
    }

    /// Whether `server_id` may connect at `now_ms`.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::Supervision`] for an unregistered id.
    pub fn may_connect(&self, server_id: &str, now_ms: u64) -> Result<bool, McpError> {
        Ok(self.cooldown(server_id)?.may_connect(now_ms))
    }

    /// Record a successful connect or call for `server_id`.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::Supervision`] for an unregistered id.
    pub fn record_success(&mut self, server_id: &str) -> Result<(), McpError> {
        let slot = self.slot(server_id)?;
        slot.record_success();
        Ok(())
    }

    /// Record a failed connect or call for `server_id` at `now_ms`.
    ///
    /// # Errors
    ///
    /// Returns [`McpFailure::Supervision`] for an unregistered id.
    pub fn record_failure(&mut self, server_id: &str, now_ms: u64) -> Result<(), McpError> {
        let slot = self.slot(server_id)?;
        slot.record_failure(now_ms);
        Ok(())
    }

    fn slot(&mut self, server_id: &str) -> Result<&mut ServerCooldown, McpError> {
        self.entries
            .iter_mut()
            .find(|(id, _)| id == server_id)
            .map(|(_, cooldown)| cooldown)
            .ok_or_else(|| {
                McpError::new(
                    McpStage::Supervise,
                    McpFailure::Supervision {
                        detail: bound_error_text(
                            &format!("unknown server '{server_id}'"),
                            crate::error::MAX_ERROR_TEXT_BYTES,
                        ),
                    },
                )
            })
    }
}

/// Resolve credential references into child environment pairs.
///
/// `Env` resolves through `env_lookup` (injected so tests stay hermetic);
/// missing or empty variables fail closed naming the reference only.
/// `Cmd` runs the bounded shell-free helper below; helper output becomes
/// the value of `name`. Values are returned, never logged — the single
/// caller injects them into the closed child environment.
///
/// # Errors
///
/// Returns [`McpFailure::CredentialMissing`] or [`McpFailure::CredentialFailed`]
/// naming the reference only.
pub fn resolve_env_refs(
    refs: &[CredentialRef],
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, McpError> {
    let mut pairs = Vec::with_capacity(refs.len());
    for reference in refs {
        match reference {
            CredentialRef::Env { var } => match env_lookup(var) {
                Some(value) if !value.is_empty() => pairs.push((var.clone(), value)),
                _ => {
                    return Err(McpError::new(
                        McpStage::Credential,
                        McpFailure::CredentialMissing {
                            name: bound_error_name(&reference.ref_name()),
                        },
                    ));
                }
            },
            CredentialRef::Cmd {
                name,
                program,
                args,
            } => {
                let value = run_credential_cmd(program, args).map_err(|_| {
                    McpError::new(
                        McpStage::Credential,
                        McpFailure::CredentialFailed {
                            name: bound_error_name(&reference.ref_name()),
                        },
                    )
                })?;
                pairs.push((name.clone(), value));
            }
        }
    }
    Ok(pairs)
}

/// Run a credential helper and return its trimmed stdout value.
///
/// Shell-free, bounded, names-only discipline (credential-spawn precedent):
/// direct spawn, null stdin, OS-discarded stderr, stdout drained on a worker
/// thread with a cap-plus-one probe at [`CREDENTIAL_CMD_MAX_OUTPUT`], one
/// trailing newline stripped, empty/NUL/non-UTF-8/oversize/non-zero results
/// refused. The deadline ([`CREDENTIAL_CMD_TIMEOUT_MS`]) covers drain and
/// completion together; past it the child is killed and reaped.
///
/// # Errors
///
/// Returns a names-only message quoting the program (never helper output).
fn run_credential_cmd(program: &str, args: &[String]) -> Result<String, String> {
    use std::io::Read as _;
    use std::sync::mpsc;

    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| format!("api_key_cmd '{program}' failed to spawn"))?;
    let pipe = match child.stdout.take() {
        Some(pipe) => pipe,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("api_key_cmd '{program}' output unreadable"));
        }
    };
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut stdout = Vec::new();
        let read = pipe
            .take(CREDENTIAL_CMD_MAX_OUTPUT as u64 + 1)
            .read_to_end(&mut stdout);
        let _ = done_tx.send((read, stdout));
    });
    let deadline = Instant::now() + Duration::from_millis(CREDENTIAL_CMD_TIMEOUT_MS);
    let mut output: Option<(std::io::Result<usize>, Vec<u8>)> = None;
    let mut status: Option<std::process::ExitStatus> = None;
    loop {
        if output.is_none() {
            match done_rx.try_recv() {
                Ok(done) => output = Some(done),
                Err(mpsc::TryRecvError::Disconnected) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("api_key_cmd '{program}' output unreadable"));
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if status.is_none() {
            match child.try_wait() {
                Ok(Some(exit)) => status = Some(exit),
                Ok(None) => {}
                Err(_) => {
                    return Err(format!("api_key_cmd '{program}' wait failed"));
                }
            }
        }
        if let Some((read, stdout)) = output.as_ref() {
            if read.is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("api_key_cmd '{program}' output unreadable"));
            }
            if stdout.len() > CREDENTIAL_CMD_MAX_OUTPUT {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("api_key_cmd '{program}' output too large"));
            }
            if let Some(exit) = status {
                if !exit.success() {
                    return Err(format!("api_key_cmd '{program}' exited with {exit}"));
                }
                if stdout.contains(&0) {
                    return Err(format!(
                        "api_key_cmd '{program}' output must not contain NUL"
                    ));
                }
                let mut text = String::from_utf8(stdout.clone())
                    .map_err(|_| format!("api_key_cmd '{program}' output is not UTF-8"))?;
                if text.ends_with('\n') {
                    text.pop();
                    if text.ends_with('\r') {
                        text.pop();
                    }
                }
                if text.is_empty() {
                    return Err(format!("api_key_cmd '{program}' produced empty output"));
                }
                return Ok(text);
            }
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("api_key_cmd '{program}' timed out"));
        }
        std::thread::sleep(SUPERVISE_POLL_INTERVAL);
    }
}

/// One supervised server child.
///
/// Owns the child process, its stdin, and the single stdout-drain worker.
/// Implements [`McpTransport`]: sends write whole frames to stdin, receives
/// poll the drain channel within the requested timeout while watching child
/// liveness. Exits kill and reap the child and mark the connection dead;
/// there is no auto-restart. `Drop` reaps a live child so tests never leak
/// helpers.
pub struct SupervisedServer {
    server_id: String,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    lines: Receiver<PipeEvent>,
    next_id: u64,
    dead: bool,
}

impl SupervisedServer {
    /// Next JSON-RPC request id (wrapping counter; calls are single-flight).
    #[must_use]
    pub fn next_request_id(&mut self) -> u64 {
        let id = self.next_id.max(1);
        self.next_id = id.wrapping_add(1);
        if self.next_id == 0 {
            self.next_id = 1;
        }
        id
    }

    /// Server id.
    #[must_use]
    pub fn server_id(&self) -> &str {
        &self.server_id
    }

    /// Whether the child is still alive (reaps zombies as a side effect).
    #[must_use]
    pub fn is_alive(&mut self) -> bool {
        if self.dead {
            return false;
        }
        match self.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                Ok(Some(_)) => {
                    self.mark_dead();
                    false
                }
                Err(_) => false,
            },
            None => false,
        }
    }

    /// Kill (if alive) and reap the child, marking the connection dead.
    /// Idempotent: safe to call twice and on an exited child.
    pub fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.stdin.take();
        self.mark_dead();
    }

    fn mark_dead(&mut self) {
        self.dead = true;
    }
}

impl Drop for SupervisedServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl McpTransport for SupervisedServer {
    fn send_line(&mut self, line: &str) -> Result<(), McpError> {
        if self.dead {
            return Err(McpError::new(
                McpStage::Supervise,
                McpFailure::TransportClosed,
            ));
        }
        if !self.is_alive() {
            return Err(McpError::new(
                McpStage::Supervise,
                McpFailure::ChildExited {
                    detail: bound_error_text(
                        "child exited before send",
                        crate::error::MAX_ERROR_TEXT_BYTES,
                    ),
                },
            ));
        }
        let write_result = match self.stdin.as_mut() {
            None => {
                return Err(McpError::new(
                    McpStage::Supervise,
                    McpFailure::TransportClosed,
                ));
            }
            Some(stdin) => stdin
                .write_all(line.as_bytes())
                .and_then(|()| stdin.write_all(b"\n"))
                .and_then(|()| stdin.flush()),
        };
        if write_result.is_err() {
            self.shutdown();
            return Err(McpError::new(
                McpStage::Supervise,
                McpFailure::ChildExited {
                    detail: bound_error_text(
                        "stdin write failed; child reaped",
                        crate::error::MAX_ERROR_TEXT_BYTES,
                    ),
                },
            ));
        }
        Ok(())
    }

    fn recv_line(&mut self, timeout_ms: u64) -> Result<Option<String>, McpError> {
        if self.dead {
            return Err(McpError::new(
                McpStage::Supervise,
                McpFailure::TransportClosed,
            ));
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(1));
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            // Poll in small slices so child liveness is observed promptly.
            let slice = remaining.min(SUPERVISE_POLL_INTERVAL * 10);
            match self.lines.recv_timeout(slice) {
                Ok(PipeEvent::Line(line)) => return Ok(Some(line)),
                Ok(PipeEvent::Oversize { limit, actual }) => {
                    return Err(McpError::new(
                        McpStage::Frame,
                        McpFailure::FrameTooLarge { limit, actual },
                    ));
                }
                Ok(PipeEvent::Eof) => {
                    self.shutdown();
                    return Err(McpError::new(
                        McpStage::Supervise,
                        McpFailure::ChildExited {
                            detail: bound_error_text(
                                "child closed stdout",
                                crate::error::MAX_ERROR_TEXT_BYTES,
                            ),
                        },
                    ));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        if !self.is_alive() {
                            return Err(McpError::new(
                                McpStage::Supervise,
                                McpFailure::ChildExited {
                                    detail: bound_error_text(
                                        "child exited while waiting",
                                        crate::error::MAX_ERROR_TEXT_BYTES,
                                    ),
                                },
                            ));
                        }
                        return Ok(None);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    self.shutdown();
                    return Err(McpError::new(
                        McpStage::Supervise,
                        McpFailure::TransportClosed,
                    ));
                }
            }
        }
    }
}

/// Spawn one supervised server child.
///
/// Validates the config fail-closed, resolves credential refs through
/// `env_lookup` (hermetic in tests), and spawns directly with a closed
/// environment (`env_clear` plus resolved pairs only), pinned `cwd`, piped
/// `stdin`, OS-null `stderr`, and piped `stdout` drained by the single
/// worker thread.
///
/// # Errors
///
/// Returns config, credential, or spawn errors (names-only).
pub fn spawn_server(
    config: &McpServerConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<SupervisedServer, McpError> {
    config.validate()?;
    let pairs = resolve_env_refs(&config.env_refs, env_lookup)?;
    let mut command = Command::new(&config.command);
    command
        .args(&config.args)
        .env_clear()
        .envs(pairs)
        .current_dir(&config.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = command.spawn().map_err(|_| {
        McpError::new(
            McpStage::Spawn,
            McpFailure::SpawnFailed {
                program: bound_error_name(&config.command),
            },
        )
    })?;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let (event_tx, event_rx): (Sender<PipeEvent>, Receiver<PipeEvent>) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        drain_stdout(stdout, event_tx);
    });
    Ok(SupervisedServer {
        server_id: config.id.clone(),
        child: Some(child),
        stdin,
        lines: event_rx,
        next_id: 2,
        dead: false,
    })
}

/// Stdout-drain worker: the single permitted thread shape.
///
/// Decodes frames with the cap-plus-one probe and forwards well-formed
/// lines, oversize refusals, and EOF. Malformed lines drop inside
/// [`FramedLines`] (counted there; drop-count behavior is pinned by the
/// frame unit tests, not across this channel).
fn drain_stdout(stdout: Option<std::process::ChildStdout>, events: Sender<PipeEvent>) {
    let Some(pipe) = stdout else {
        let _ = events.send(PipeEvent::Eof);
        return;
    };
    let mut frames = FramedLines::new(BufReader::new(pipe));
    loop {
        match frames.next_line() {
            Ok(Some(line)) => {
                if events.send(PipeEvent::Line(line)).is_err() {
                    return;
                }
            }
            Ok(None) => {
                let _ = events.send(PipeEvent::Eof);
                return;
            }
            Err(error) => {
                let (limit, actual) = match error.failure {
                    McpFailure::FrameTooLarge { limit, actual } => (limit, actual),
                    _ => (MAX_FRAME_BYTES, MAX_FRAME_BYTES + 1),
                };
                if events.send(PipeEvent::Oversize { limit, actual }).is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_var: &str) -> Option<String> {
        None
    }

    #[test]
    fn cooldown_backs_off_geometrically_and_caps() {
        let mut cooldown = ServerCooldown::default();
        assert!(cooldown.may_connect(0));
        cooldown.record_failure(1_000);
        assert!(!cooldown.may_connect(1_500));
        assert!(cooldown.may_connect(2_000));
        assert_eq!(cooldown.wait_ms(1_500), 500);
        cooldown.record_failure(2_000);
        assert!(!cooldown.may_connect(3_999));
        assert!(cooldown.may_connect(4_000));
        // Cap: even deep failure streaks wait at most 60s.
        for _ in 0..30 {
            cooldown.record_failure(10_000);
        }
        assert_eq!(cooldown.wait_ms(10_000), COOLDOWN_CAP_MS);
        cooldown.record_success();
        assert!(cooldown.may_connect(10_000));
        assert_eq!(cooldown.wait_ms(10_000), 0);
    }

    #[test]
    fn host_gates_unknown_ids_fail_closed() {
        let mut host = McpHost::new();
        host.register("demo").expect("register");
        assert!(host.may_connect("demo", 0).expect("known"));
        assert!(host.may_connect("ghost", 0).is_err());
        host.record_failure("demo", 1_000).expect("record");
        assert!(!host.may_connect("demo", 1_500).expect("known"));
        host.record_success("demo").expect("record");
        assert!(host.may_connect("demo", 1_500).expect("known"));
    }

    #[test]
    fn missing_env_credential_names_the_reference_only() {
        let refs = vec![CredentialRef::Env {
            var: "MISSING_TOKEN".to_owned(),
        }];
        let error = resolve_env_refs(&refs, &no_env).expect_err("missing");
        let text = error.to_string();
        assert!(text.contains("env:MISSING_TOKEN"));
        assert!(!text.contains("super-secret"));
    }

    #[test]
    fn present_env_resolves_without_logging() {
        let refs = vec![CredentialRef::Env {
            var: "PRESENT_TOKEN".to_owned(),
        }];
        let pairs = resolve_env_refs(&refs, &|var| {
            if var == "PRESENT_TOKEN" {
                Some("value-abc".to_owned())
            } else {
                None
            }
        })
        .expect("resolve");
        assert_eq!(
            pairs,
            vec![("PRESENT_TOKEN".to_owned(), "value-abc".to_owned())]
        );
    }

    #[test]
    fn spawn_rejects_invalid_config_before_touching_the_os() {
        let mut config = McpServerConfig {
            id: "demo".to_owned(),
            command: String::new(),
            args: Vec::new(),
            env_refs: Vec::new(),
            cwd: "/tmp/bitty".to_owned(),
            timeout_ms: 1_000,
            tool_allowlist: vec!["echo".to_owned()],
        };
        assert!(spawn_server(&config, &no_env).is_err());
        config.command = "bitty-definitely-missing-xyz".to_owned();
        let error = match spawn_server(&config, &no_env) {
            Ok(_) => panic!("missing program must fail"),
            Err(error) => error,
        };
        assert!(matches!(error.failure, McpFailure::SpawnFailed { .. }));
        assert!(error.to_string().contains("bitty-definitely-missing-xyz"));
    }
}

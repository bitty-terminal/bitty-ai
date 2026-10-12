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
/// Fixed streaming buffer for server-binary digest verification (64 KiB).
///
/// The pin check reads the `command` binary through exactly this buffer in a
/// loop, so memory stays bounded no matter how large the binary is: the whole
/// file is never loaded.
pub const BINARY_DIGEST_BUF_LEN: usize = 64 * 1024;
/// Maximum pinned server-binary size hashed pre-spawn (8 MiB).
///
/// Value choice: 8 MiB is 128x the [`BINARY_DIGEST_BUF_LEN`] streaming buffer
/// and 32x the [`MAX_FRAME_BYTES`] frame cap (power-of-two KiB multiples, the
/// repo bound idiom). It covers typical MCP server helpers (shell/python
/// shims and small compiled servers; e.g. `bash` ~1.2 MiB) while bounding
/// pre-spawn hash work to milliseconds. Larger binaries refuse fail-closed
/// with [`McpFailure::IdentityMismatch`] — never truncate-and-accept, since a
/// partial hash cannot verify identity — so operators pin a small launcher
/// shim instead of a multi-tens-of-MiB runtime. The existing multi-chunk
/// streaming test (~72 KiB) stays well under this cap.
pub const MAX_PINNED_BINARY_BYTES: usize = 8 * 1024 * 1024;

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

/// Server-binary digest pin (AI-0248 Rank-2, pre-configured, spawn-time).
///
/// Hash-implementation decision (written justification, AI-0248 §4):
/// option (a) small pure-Rust SHA-256 in-crate is chosen. Option (b) a direct
/// `sha2 = "=0.11.0"` dep would be fewer lines (one manifest line plus ~30
/// lines of call code, reusing the exact pin `bitty-ai-session` already
/// carries), but total review surface is larger: a new registry edge from this
/// client plus transitive `digest`/`block-buffer`/`crypto-common` crates,
/// feature unification, and build-time change. Reuse via the existing
/// `bitty-ai-session` edge is rejected without further review: that crate
/// depends on `rusqlite` (bundled SQLite), so a `bitty-ai-session` dep would
/// drag a vendored C build into this stdio client for no behavioral gain.
/// Option (c) shell-out to an external verifier is rejected a priori per the
/// task (helper-identity recursion). Option (a) keeps zero `Cargo.toml` /
/// `Cargo.lock` churn, preserves the std-only posture, is `#![deny(unsafe_code)]`
/// clean and MSRV 1.85 safe, and is pinned by NIST vectors below.
///
/// Verification discipline: when [`McpServerConfig::sha256_pin`] is `Some`,
/// the command binary streams through a fixed [`BINARY_DIGEST_BUF_LEN`] buffer
/// BEFORE `Command::new`; mismatch or unreadable binary refuses spawn
/// fail-closed with [`McpFailure::IdentityMismatch`] carrying the server id
/// only (zero path bytes, zero hash bytes). When absent, no filesystem touch
/// happens (byte-identical behavior); the Rank-1 capability-set pin (AI-0244)
/// still applies at relist time. There is deliberately no TOFU, no
/// auto-update, no HTTP SPKI, no PKI, and no transparency log.
///
/// Note on TOCTOU: the digest is checked immediately before spawn, narrowing
/// but not eliminating a swap between check and `exec`. That residual is
/// accepted for Rank-2 (spawn-time pin, not a verified-exec chain) and is why
/// the pin stays optional-in-config, never required.
#[allow(clippy::unreadable_literal)]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Minimal SHA-256 (FIPS 180-4) for the digest pin: streaming `update` plus
/// one-shot `finalize`, no dependencies, no `unsafe`.
#[derive(Clone)]
struct PinSha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl PinSha256 {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        // Fill a partial block first, then compress whole blocks straight
        // from the input without extra copies.
        if self.buf_len > 0 {
            let want = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + want].copy_from_slice(&data[..want]);
            self.buf_len += want;
            self.total_len = self.total_len.wrapping_add(want as u64);
            data = &data[want..];
            if self.buf_len == 64 {
                let block = self.buf;
                Self::compress(&mut self.h, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress(&mut self.h, &block);
            self.total_len = self.total_len.wrapping_add(64);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
            self.total_len = self.total_len.wrapping_add(data.len() as u64);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        // Padding is at most two blocks; use a fixed stack array, never a
        // heap allocation tied to input size.
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        let pad_len = if self.buf_len < 56 {
            56 - self.buf_len
        } else {
            64 - self.buf_len + 56
        };
        self.update(&pad[..pad_len]);
        debug_assert_eq!(self.buf_len, 56);
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for (i, word) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
}

/// Decode a 64-character lowercase-hex pin to 32 bytes (`None` on shape
/// violation; callers fail closed without echoing the input).
fn decode_sha256_pin(pin: &str) -> Option<[u8; 32]> {
    if pin.len() != crate::config::SHA256_PIN_LEN {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in pin.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_val(chunk[0])?;
        let lo = hex_val(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

/// Single lowercase-hex digit value (`None` for anything else, including
/// uppercase: pins are strictly `0-9a-f`).
fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Constant-time 32-byte equality (accumulated difference, no early exit).
fn pin_digest_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (a, b) in left.iter().zip(right.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Resolve the `command` program to a filesystem path for digesting.
///
/// A `/`-bearing program is a path and opens directly; a bare name follows
/// the host `PATH` (the same lookup `Command::new` performs at spawn). Only
/// regular files are accepted from `PATH`; anything else falls through to a
/// path that fails closed at open time with a names-only error.
fn resolve_program_path(program: &str) -> std::path::PathBuf {
    use std::path::PathBuf;
    if program.contains('/') {
        return PathBuf::from(program);
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join(program);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(program)
}

/// Stream the file at `path` through a fixed 64 KiB buffer into the pin
/// hasher. Memory stays bounded: the whole file is never loaded.
///
/// Fail-closed bounds: the pre-open metadata and the opened-handle metadata
/// must both report a regular file, and the total hashed bytes must stay
/// within [`MAX_PINNED_BINARY_BYTES`]. Non-regular files (FIFOs, sockets,
/// char/block devices) and over-cap binaries refuse with an I/O error that
/// callers map to [`McpFailure::IdentityMismatch`]; over-cap never
/// truncates-and-accepts since a partial hash cannot verify identity. The
/// pre-open check avoids a blocking `open` on a pre-existing FIFO; the Unix
/// `O_NONBLOCK` open covers a swap between check and open; the streaming
/// total covers growth between metadata and EOF. Residual TOCTOU (swap
/// between hash and `exec`) is documented at [`verify_binary_pin`].
fn hash_command_binary(path: &std::path::Path) -> Result<[u8; 32], std::io::Error> {
    use std::io::Read as _;
    // Pre-open gate: refuse non-regular or over-cap without touching `open`,
    // so a pre-existing FIFO never blocks here.
    let pre = std::fs::metadata(path)?;
    if !pre.is_file() {
        return Err(std::io::Error::other("pinned binary is not a regular file"));
    }
    if pre.len() > MAX_PINNED_BINARY_BYTES as u64 {
        return Err(std::io::Error::other("pinned binary exceeds size bound"));
    }
    let mut file = open_pinned_binary_nonblocking(path)?;
    // Opened-handle gate: catch a swap between the pre-check and `open`.
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(std::io::Error::other("pinned binary is not a regular file"));
    }
    if meta.len() > MAX_PINNED_BINARY_BYTES as u64 {
        return Err(std::io::Error::other("pinned binary exceeds size bound"));
    }
    let mut hasher = PinSha256::new();
    let mut buf = [0u8; BINARY_DIGEST_BUF_LEN];
    let mut total: u64 = 0;
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| std::io::Error::other("pinned binary exceeds size bound"))?;
        if total > MAX_PINNED_BINARY_BYTES as u64 {
            return Err(std::io::Error::other("pinned binary exceeds size bound"));
        }
        hasher.update(&buf[..read]);
    }
    Ok(hasher.finalize())
}

/// Open the pinned binary without blocking on FIFOs where supported.
///
/// Unix opens with `O_NONBLOCK` so a FIFO swapped in between the pre-open
/// metadata gate and `open` still cannot hang the caller: the subsequent
/// `is_file` gate refuses it. Flag values are OS-specific (`Linux`/`Android`
/// `0o4000`, other Unix `0x0004`); non-Unix falls back to a plain open (no
/// FIFO semantics there). Std-only, MSRV 1.85 safe, no `unsafe`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_pinned_binary_nonblocking(path: &std::path::Path) -> Result<std::fs::File, std::io::Error> {
    use std::os::unix::fs::OpenOptionsExt as _;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0o4000)
        .open(path)
}

/// Open the pinned binary without blocking on FIFOs where supported.
///
/// Unix opens with `O_NONBLOCK` so a FIFO swapped in between the pre-open
/// metadata gate and `open` still cannot hang the caller: the subsequent
/// `is_file` gate refuses it. Flag values are OS-specific (`Linux`/`Android`
/// `0o4000`, other Unix `0x0004`); non-Unix falls back to a plain open (no
/// FIFO semantics there). Std-only, MSRV 1.85 safe, no `unsafe`.
#[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
fn open_pinned_binary_nonblocking(path: &std::path::Path) -> Result<std::fs::File, std::io::Error> {
    use std::os::unix::fs::OpenOptionsExt as _;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0x0004)
        .open(path)
}

/// Open the pinned binary without blocking on FIFOs where supported.
///
/// Unix opens with `O_NONBLOCK` so a FIFO swapped in between the pre-open
/// metadata gate and `open` still cannot hang the caller: the subsequent
/// `is_file` gate refuses it. Flag values are OS-specific (`Linux`/`Android`
/// `0o4000`, other Unix `0x0004`); non-Unix falls back to a plain open (no
/// FIFO semantics there). Std-only, MSRV 1.85 safe, no `unsafe`.
#[cfg(not(unix))]
fn open_pinned_binary_nonblocking(path: &std::path::Path) -> Result<std::fs::File, std::io::Error> {
    std::fs::File::open(path)
}

/// Verify the command binary against the pre-configured pin, when present.
///
/// Streams the binary through [`BINARY_DIGEST_BUF_LEN`] BEFORE `Command::new`;
/// mismatch or unreadable binary refuses with [`McpFailure::IdentityMismatch`]
/// carrying the server id only. Absent pin returns `Ok(None)` without touching
/// the filesystem. On success returns the verified path: callers must spawn
/// EXACTLY this path (never re-resolve the bare name), eliminating the
/// double-lookup confusion where a non-executable decoy earlier in `PATH`
/// could satisfy the pin while a different binary launches after `env_clear`.
fn verify_binary_pin(config: &McpServerConfig) -> Result<Option<std::path::PathBuf>, McpError> {
    let Some(expected_hex) = config.sha256_pin.as_deref() else {
        return Ok(None);
    };
    let identity_mismatch = || {
        McpError::new(
            McpStage::Spawn,
            McpFailure::IdentityMismatch {
                server: bound_error_name(&config.id),
            },
        )
    };
    // Shape already validated, but fail closed here too (defense in depth;
    // never echo the pin bytes).
    let Some(expected) = decode_sha256_pin(expected_hex) else {
        return Err(identity_mismatch());
    };
    let path = resolve_program_path(&config.command);
    let actual = hash_command_binary(&path).map_err(|_| identity_mismatch())?;
    if !pin_digest_eq(&actual, &expected) {
        return Err(identity_mismatch());
    }
    Ok(Some(path))
}

/// Spawn one supervised server child.
///
/// Validates the config fail-closed, verifies the optional server-binary
/// digest pin by streaming the `command` binary through a fixed 64 KiB buffer
/// BEFORE `Command::new` (mismatch refuses with `IdentityMismatch`, zero
/// child, names-only), resolves credential refs through `env_lookup`
/// (hermetic in tests), and spawns directly with a closed environment
/// (`env_clear` plus resolved pairs only), pinned `cwd`, piped `stdin`,
/// OS-null `stderr`, and piped `stdout` drained by the single worker thread.
/// An absent pin skips verification without touching the filesystem
/// (byte-identical behavior: the bare `command` passes straight to
/// `Command::new` as before). A present pin spawns EXACTLY the verified path
/// (no second `PATH` lookup after `env_clear`).
///
/// # Errors
///
/// Returns config, digest-pin, credential, or spawn errors (names-only).
pub fn spawn_server(
    config: &McpServerConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<SupervisedServer, McpError> {
    config.validate()?;
    let verified_path = verify_binary_pin(config)?;
    let pairs = resolve_env_refs(&config.env_refs, env_lookup)?;
    let program = verified_path
        .as_deref()
        .unwrap_or_else(|| std::path::Path::new(&config.command));
    let mut command = Command::new(program);
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
            sha256_pin: None,
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

    // ── AI-0248: in-crate SHA-256 pinned by NIST vectors ──

    fn pin_hex(bytes: &[u8]) -> String {
        let mut hasher = PinSha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('?'));
            hex.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('?'));
        }
        hex
    }

    #[test]
    fn pin_sha256_matches_nist_vectors() {
        // Empty string.
        assert_eq!(
            pin_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // Single-block message.
        assert_eq!(
            pin_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Two-block message (56 bytes: padding spans two blocks). Expected
        // values cross-checked against Python `hashlib.sha256` (OpenSSL).
        assert_eq!(
            pin_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn pin_sha256_streaming_matches_one_shot() {
        let message = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut chunked = PinSha256::new();
        for chunk in message.chunks(7) {
            chunked.update(chunk);
        }
        let mut whole = PinSha256::new();
        whole.update(message);
        assert_eq!(chunked.finalize(), whole.finalize());
        // Million-`a` vector exercises many blocks through `update`.
        // Cross-checked against Python `hashlib.sha256`.
        let mut million = PinSha256::new();
        for _ in 0..1_000 {
            million.update(&[b'a'; 1_000]);
        }
        let digest = million.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('?'));
            hex.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('?'));
        }
        assert_eq!(
            hex,
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn pin_hex_codec_is_strict_lowercase() {
        let good = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert!(decode_sha256_pin(good).is_some());
        for bad in [
            "",
            "abc",
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85g",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85",
        ] {
            assert!(decode_sha256_pin(bad).is_none(), "must refuse {bad:?}");
        }
        // Constant-time equality distinguishes without early-exit branches.
        let zero = [0u8; 32];
        let mut one = [0u8; 32];
        one[31] = 1;
        assert!(pin_digest_eq(&zero, &zero));
        assert!(!pin_digest_eq(&zero, &one));
    }
}

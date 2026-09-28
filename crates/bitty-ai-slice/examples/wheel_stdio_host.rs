//! Wheel JSON-RPC stdio host bridge (AI-0170 / E2E).
//!
//! Exposes [`WheelBridge`] over stdin/stdout lines for cross-process
//! host integration, foreign language clients, and integration drills.
//!
//! # Usage
//!
//! ```bash
//! # Ephemeral in-memory session:
//! cargo run -p bitty-ai-slice --example wheel_stdio_host
//!
//! # Persistent SQLite database session:
//! cargo run -p bitty-ai-slice --example wheel_stdio_host -- --db /tmp/bitty/wheel.db
//! ```
//!
//! # Protocol
//! - Each request on `stdin` is a single line:
//!   - Format 1 (JSON object): `{"command": "kernel.status", "payload": {}}`
//!   - Format 2 (Space-separated): `task.list {}` or `kernel.status`
//! - Each response on `stdout` is a single-line JSON string:
//!   `{"success": true, "data": ...}` or `{"success": false, "error": ...}`
//! - On EOF or command `exit` / `quit`, the host process terminates cleanly.

#![deny(unsafe_code)]

use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use bitty_ai_slice::WheelBridge;

fn print_help() {
    eprintln!("Usage: wheel_stdio_host [OPTIONS]");
    eprintln!();
    eprintln!("Expose WheelBridge over line-delimited JSON-RPC via stdin/stdout.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --db <PATH>       Path to SQLite database file (defaults to in-memory)");
    eprintln!("  -h, --help        Print help information");
}

fn parse_args() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut db_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--db" => {
                let path = args.next().ok_or("missing argument for --db")?;
                db_path = Some(PathBuf::from(path));
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            unknown => {
                return Err(format!("unknown option: {unknown}").into());
            }
        }
    }

    Ok(db_path)
}

fn main() -> Result<(), Box<dyn Error>> {
    let db_path = parse_args()?;

    let mut bridge = match db_path {
        Some(ref path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            eprintln!(
                "[wheel-stdio-host] Opened persistent database at {}",
                path.display()
            );
            WheelBridge::open(path)?
        }
        None => {
            eprintln!("[wheel-stdio-host] Opened in-memory ephemeral store");
            WheelBridge::open_in_memory()?
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();

    eprintln!(
        "[wheel-stdio-host] Ready for commands on stdin (format: command payload_json or JSON object)"
    );

    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[wheel-stdio-host] Error reading stdin: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "exit" || trimmed == "quit" {
            eprintln!("[wheel-stdio-host] Exit command received, shutting down cleanly");
            break;
        }

        let (command, payload_json) = if trimmed.starts_with('{') {
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(val) => {
                    let cmd = val
                        .get("command")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    let payload = val.get("payload").unwrap_or(&serde_json::Value::Null);
                    (cmd, payload.to_string())
                }
                Err(e) => {
                    let err_resp = serde_json::json!({
                        "success": false,
                        "error": format!("invalid JSON request: {e}")
                    });
                    writeln!(stdout_lock, "{err_resp}")?;
                    stdout_lock.flush()?;
                    continue;
                }
            }
        } else if let Some((cmd, rest)) = trimmed.split_once(' ') {
            (cmd.trim().to_string(), rest.trim().to_string())
        } else {
            (trimmed.to_string(), "{}".to_string())
        };

        let response = bridge.dispatch(&command, &payload_json);
        writeln!(stdout_lock, "{response}")?;
        stdout_lock.flush()?;
    }

    Ok(())
}

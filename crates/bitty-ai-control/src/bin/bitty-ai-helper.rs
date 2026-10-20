//! `bitty-ai-helper`: the AI-helper launch entrypoint.
//!
//! The Core supervisor starts this executable through its provider-publication
//! launch seam with two inherited descriptors:
//!
//! | Descriptor | Channel |
//! | --- | --- |
//! | `3` (`CHILD_BOOTSTRAP_DESCRIPTOR`) | one-shot bootstrap read end (exactly one launch manifest, then EOF) |
//! | `4` (`CHILD_CONTROL_DESCRIPTOR`) | dedicated duplex control stream, TLS from its first byte |
//!
//! Descriptor `5` remains reserved for the data channel and is not adopted.
//! This binary creates no listener, socket, environment lookup, filesystem
//! input, thread, or clock beyond the two inherited channels.
//!
//! The executable path is `target/<profile>/bitty-ai-helper`, built from this
//! file (`crates/bitty-ai-control/src/bin/bitty-ai-helper.rs`). Core's launch
//! manifest `executable_sha256` must identify the exact built helper binary;
//! that binding belongs to Core's verified descriptor and launch path. This
//! binary does not read its own executable or compute a digest, and this
//! repository does not install or package it.
//!
//! On every non-Unix target the binary fails closed with
//! [`HelperExit::UnsupportedPlatform`] before adopting or reading a descriptor.
//! On Unix, descriptor adoption is the single audited `unsafe` operation in
//! `adopt_inherited_channels`.
//!
//! Strict provider publication stays disabled. The control-only registry view
//! is deliberately empty and is not a provider registry.

#![deny(unsafe_code)]

use std::io::Write as _;
use std::process::ExitCode;

use bitty_ai_control::entrypoint::HelperExit;

#[cfg(unix)]
use bitty_ai_control::control_registry::ControlOnlyRegistry;

fn main() -> ExitCode {
    let exit = run();
    if !exit.is_success() {
        // A diagnostic failure must not replace the mapped exit status.
        let _ = writeln!(std::io::stderr(), "{}", exit.diagnostic());
    }
    ExitCode::from(exit.code())
}

#[cfg(unix)]
fn run() -> HelperExit {
    let (bootstrap, control) = adopt_inherited_channels();
    bitty_ai_control::run_helper(bootstrap, control, ControlOnlyRegistry::new())
}

#[cfg(not(unix))]
fn run() -> HelperExit {
    // No inherited-descriptor launch contract exists on this platform. Fail
    // before adopting, reading, or constructing anything.
    HelperExit::UnsupportedPlatform
}

/// Adopt the two inherited launch descriptors.
///
/// This is the only `unsafe` operation in the binary. The descriptor numbers
/// are the fixed Core launch contract, not caller input: the supervisor
/// duplicates the bootstrap pipe's read end onto descriptor `3` and the
/// control socket's child end onto descriptor `4` in `pre_exec`, immediately
/// before `execve`.
///
/// # Safety
///
/// The process must be the helper launched by the Core provider-publication
/// launch seam, so descriptors `3` and `4` are open, inherited, and owned
/// exclusively by this process for the lifetime of the adopted values.
/// Adopting a descriptor transfers ownership to the returned values, which
/// close it on drop. Manual execution is unsupported: the binary is defined
/// only for the Core launch contract above. If descriptors `3`/`4` are closed,
/// foreign, or not the expected channel types, behavior is outside this safety
/// contract and is not guaranteed to map to a specific [`HelperExit`]; callers
/// must not rely on a defined failure for unsupported invocations. Supported
/// launches only ever pass open descriptors of the documented types.
/// `OwnedFd`/`UnixStream` construction is otherwise safe: no memory is
/// reinterpreted and no allocation or I/O occurs here.
#[cfg(unix)]
#[allow(unsafe_code)] // Justification: one narrow, audited descriptor adoption.
fn adopt_inherited_channels() -> (std::fs::File, std::os::unix::net::UnixStream) {
    use std::fs::File;
    use std::os::fd::{FromRawFd, OwnedFd, RawFd};
    use std::os::unix::net::UnixStream;

    use bitty_ai_control::entrypoint::{CHILD_BOOTSTRAP_DESCRIPTOR, CHILD_CONTROL_DESCRIPTOR};

    // SAFETY: per the launch contract, descriptor 3 is an open, inherited
    // bootstrap pipe read end owned by this process. `from_raw_fd` takes that
    // ownership exactly once.
    let bootstrap =
        File::from(unsafe { OwnedFd::from_raw_fd(CHILD_BOOTSTRAP_DESCRIPTOR as RawFd) });

    // SAFETY: per the launch contract, descriptor 4 is an open, inherited
    // Unix-domain socket owned by this process. `from_raw_fd` takes that
    // ownership exactly once.
    let control = unsafe { UnixStream::from_raw_fd(CHILD_CONTROL_DESCRIPTOR as RawFd) };

    (bootstrap, control)
}

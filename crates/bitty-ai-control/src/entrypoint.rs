//! Testable helper launch entrypoint.
//!
//! [`run_helper`] is the library half of the `bitty-ai-helper` process: it
//! consumes exactly one bounded launch manifest from the injected bootstrap
//! reader, requires that channel to end at the exact record, builds the
//! launch-scoped mutual-certificate [`ControlServer`], completes the blocking
//! `accept` over the injected control stream, then drives [`HelperControl`]
//! through `establish` and `step` until the peer closes authority or any
//! terminal failure occurs.
//!
//! The function takes streams, not descriptors: tests inject in-memory
//! streams and the binary owns the single audited descriptor adoption. No
//! listener, socket, thread, clock, environment, filesystem path, or data
//! channel (`fd 5`) is created here. Deadlines remain caller policy; this
//! module implements the design's five-second establishment bound nowhere.
//!
//! Strict provider publication stays disabled: the entrypoint never sends or
//! accepts `Admit`, never contacts a provider or credential source, and the
//! only registry view available to it is the empty control-only adapter.
//!
//! # Exit mapping
//!
//! [`HelperExit`] is the small terminal vocabulary for the process exit path.
//! Every variant has one fixed diagnostic and one fixed exit code; no payload,
//! identifier, path, model string, certificate, or key material is ever part
//! of a diagnostic. A successful terminal state (the Core withdrew authority
//! or reported a terminal peer error) exits `0`; every launch or control
//! failure exits with its own non-zero code.

use std::fmt;
use std::io::{Read, Write};

use crate::manifest::{LaunchManifest, ManifestReadError};
use crate::server::ControlServer;
use crate::state::{HelperControl, RegistryView, Step};

/// Fixed child descriptor number of the one-shot launch bootstrap read end.
///
/// This mirrors the planned Core provider-publication launch contract
/// (`bitty-runtime` `ai_publication_launch::CHILD_BOOTSTRAP_DESCRIPTOR`). That
/// module is not present in the Core checkout at `aceb327e`; it lands with the
/// separately scoped Core adapter. `bitty-ai-control` has no dependency edge
/// on the Core crate, so the value is restated here with the planned contract
/// named instead of imported.
pub const CHILD_BOOTSTRAP_DESCRIPTOR: u32 = 3;

/// Fixed child descriptor number of the dedicated duplex control stream.
///
/// This mirrors the Core launch seam's `CHILD_CONTROL_DESCRIPTOR`. Descriptor
/// `5` stays reserved for the data channel and is never adopted or used in
/// this slice; there is no data-channel constant here on purpose.
pub const CHILD_CONTROL_DESCRIPTOR: u32 = 4;

/// Terminal outcome of [`run_helper`], mapped to a process exit status.
///
/// The enum carries no payloads. It is `Copy` so the binary can print the
/// fixed diagnostic and return the fixed code without cloning state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelperExit {
    /// The Core withdrew provider-publication authority (`GrantLoss` or
    /// `Shutdown`); strict publication was already disabled. Exit code `0`.
    AuthorityClosed,
    /// The Core sent a terminal control error. Exit code `0`.
    PeerClosed,
    /// The platform has no inherited-descriptor launch contract. Exit code
    /// `2`; the process fails before reading anything.
    UnsupportedPlatform,
    /// The launch bootstrap reader failed before a complete record. Exit code
    /// `3`.
    BootstrapIo,
    /// The launch manifest was malformed, truncated, oversized, or carried
    /// trailing bytes inside the read record. Exit code `4`.
    BootstrapManifest,
    /// The bootstrap channel delivered data after its end-of-file marker. Exit
    /// code `5`.
    BootstrapTrailingData,
    /// The launch key material could not be turned into a control server.
    /// Exit code `6`.
    ServerConfiguration,
    /// The mutual-certificate TLS 1.3 handshake failed or was refused. Exit
    /// code `7`.
    Handshake,
    /// The application establishment exchange failed. Exit code `8`.
    Establish,
    /// A control step failed after establishment, or a record violated the
    /// state machine. Exit code `9`.
    Control,
}

impl HelperExit {
    /// Process exit code for this outcome.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::AuthorityClosed | Self::PeerClosed => 0,
            Self::UnsupportedPlatform => 2,
            Self::BootstrapIo => 3,
            Self::BootstrapManifest => 4,
            Self::BootstrapTrailingData => 5,
            Self::ServerConfiguration => 6,
            Self::Handshake => 7,
            Self::Establish => 8,
            Self::Control => 9,
        }
    }

    /// Fixed local diagnostic text.
    ///
    /// The text never contains a payload, identifier, path, model string,
    /// certificate, or key material.
    #[must_use]
    pub const fn diagnostic(self) -> &'static str {
        match self {
            Self::AuthorityClosed => "provider-publication authority closed by the Core",
            Self::PeerClosed => "provider-publication control closed by a peer error",
            Self::UnsupportedPlatform => "helper launch is unsupported on this platform",
            Self::BootstrapIo => "launch bootstrap channel read failed",
            Self::BootstrapManifest => "launch manifest rejected",
            Self::BootstrapTrailingData => "launch bootstrap channel carried extra data",
            Self::ServerConfiguration => "control server configuration rejected",
            Self::Handshake => "mutual TLS control establishment failed",
            Self::Establish => "provider-publication control establishment failed",
            Self::Control => "provider-publication control session failed",
        }
    }

    /// Whether this is a successful terminal state (exit code `0`).
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::AuthorityClosed | Self::PeerClosed)
    }
}

impl fmt::Display for HelperExit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.diagnostic())
    }
}

/// Run the helper launch protocol over injected streams.
///
/// The sequence is fixed:
///
/// 1. read exactly one [`LaunchManifest`] with
///    [`LaunchManifest::read_from`] and require that the bootstrap reader is
///    then at end-of-file;
/// 2. drop the bootstrap reader, closing the inherited descriptor before any
///    control work;
/// 3. build [`ControlServer`] from the manifest (consuming it);
/// 4. complete the mutual-certificate `accept` over `control`;
/// 5. build [`HelperControl::from_session`] with `registry`;
/// 6. call `establish`, then `step` until authority closes.
///
/// `control` must carry TLS from its first byte; manifest bytes never appear
/// there. The function blocks on both streams and owns no deadline or clock.
///
/// # Errors
///
/// Every failure is a [`HelperExit`] with fixed diagnostics. The function does
/// not panic on malformed launch material, refused authentication, premature
/// EOF, or state-machine rejection.
#[must_use]
pub fn run_helper<B, C, R>(bootstrap: B, control: C, registry: R) -> HelperExit
where
    B: Read,
    C: Read + Write,
    R: RegistryView,
{
    run_helper_on_platform(cfg!(unix), bootstrap, control, registry)
}

/// Platform-injectable form of [`run_helper`].
///
/// **Test/adapter-only.** Production callers use [`run_helper`], which derives
/// support from the target platform. This form exists so the non-Unix
/// fail-closed path is testable on a Unix host and so a future platform
/// adapter can reuse the same sequence. `platform_supported == false` returns
/// [`HelperExit::UnsupportedPlatform`] before either stream is touched.
#[doc(hidden)]
#[must_use]
pub fn run_helper_on_platform<B, C, R>(
    platform_supported: bool,
    bootstrap: B,
    control: C,
    registry: R,
) -> HelperExit
where
    B: Read,
    C: Read + Write,
    R: RegistryView,
{
    if !platform_supported {
        return HelperExit::UnsupportedPlatform;
    }

    // The bootstrap handle is consumed and closed before any control work.
    // The block bounds the reader's lifetime so the inherited descriptor is
    // released even when the control session outlives manifest ingestion.
    let manifest = {
        let mut bootstrap = bootstrap;
        let manifest = match LaunchManifest::read_from(&mut bootstrap) {
            Ok(manifest) => manifest,
            Err(ManifestReadError::Io(_)) => return HelperExit::BootstrapIo,
            Err(ManifestReadError::Manifest(_)) => return HelperExit::BootstrapManifest,
        };
        if let Err(exit) = require_bootstrap_eof(&mut bootstrap) {
            return exit;
        }
        manifest
    };

    let server = match ControlServer::new(manifest) {
        Ok(server) => server,
        Err(_) => return HelperExit::ServerConfiguration,
    };
    let session = match server.accept(control) {
        Ok(session) => session,
        Err(_) => return HelperExit::Handshake,
    };

    let mut helper = HelperControl::from_session(session, registry);
    if helper.establish().is_err() {
        return HelperExit::Establish;
    }

    loop {
        match helper.step() {
            Ok(Step::AuthorityClosed { .. }) => return HelperExit::AuthorityClosed,
            Ok(Step::PeerError { .. }) => return HelperExit::PeerClosed,
            Ok(_) => {}
            Err(_) => return HelperExit::Control,
        }
    }
}

/// Require the bootstrap reader to be at end-of-file after the manifest.
///
/// [`LaunchManifest::read_from`] already reads to end-of-file and its parser
/// rejects trailing bytes inside that buffer, so this is defense in depth for
/// an injected reader that returns `0` and later resumes. Any read error,
/// including an interrupt, fails closed rather than retrying without a clock or
/// step budget.
fn require_bootstrap_eof<R: Read>(bootstrap: &mut R) -> Result<(), HelperExit> {
    let mut probe = [0_u8; 1];
    match bootstrap.read(&mut probe) {
        Ok(0) => Ok(()),
        Ok(_) => Err(HelperExit::BootstrapTrailingData),
        Err(_) => Err(HelperExit::BootstrapIo),
    }
}

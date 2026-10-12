//! Filesystem permission helper (AI-0243, session plane).
//!
//! Pure `0600` enforcement for secret-adjacent files: [`ensure_0600`] sets
//! the file mode to owner-read/write only via
//! [`std::os::unix::fs::PermissionsExt`]. No database, no store handle, no
//! `rusqlite`: this module is std-only plus the Unix permission extension so
//! the session crate gains no new dependency.
//!
//! Time/Space: O(1) single `chmod`; no allocation beyond the error path.
//!
//! Clocks: none. Threads: none. Network: none. Filesystem: one
//! `set_permissions` call on the caller-supplied path only.

use std::path::Path;

/// Ensure the file at `path` is owner-only (`0600`).
///
/// Sets the permission bits to `0o600` unconditionally (already-`0600` files
/// are a no-op `chmod`). A missing file fails closed with the underlying
/// I/O error; no file is created.
///
/// # Errors
///
/// Returns the underlying [`std::io::Error`] when the file cannot be
/// stat-ed or chmod-ed, and `Unsupported` on non-Unix platforms where the
/// Unix mode bits do not exist.
///
/// # Examples
///
/// ```no_run
/// # use std::path::Path;
/// # use bitty_ai_session::fs_perms::ensure_0600;
/// ensure_0600(Path::new("/tmp/example-secret")).expect("chmod 0600");
/// ```
pub fn ensure_0600(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "ensure_0600 is Unix-only (owner-only mode bits do not exist here)",
        ))
    }
}

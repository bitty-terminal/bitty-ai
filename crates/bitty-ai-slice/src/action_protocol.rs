//! Standardized Action, Intent, and Outcome execution protocol with auto-spillover (AI-0166).
//!
//! Provides the execution plane for Wheel:
//! - [`ActionIntent`]: Structured categorization of agent intent (`Inspect`, `Modify`, `Execute`, `Verify`, `Custom`).
//! - [`Action`]: Typed action descriptor with unique ID, intent, tool name, and validated arguments.
//! - [`BlobPointer`]: Cryptographic content address and byte size of spilled data in [`ContentStore`].
//! - [`ActionPayload`]: Inline vs Spilled payload with UTF-8 bounded previews.
//! - [`ActionOutcome`]: Execution result with exit code, duration, formatted observations, and auto-spillover.
//! - [`ActionEngine`]: Pipeline managing auto-spillover into content-addressed blob storage.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::content_store::{ContentHash, ContentStore, ContentStoreError, MAX_BLOB_BYTES};

/// Default threshold above which tool stdout/stderr is spilled to the blob store (4 KiB).
pub const DEFAULT_MAX_INLINE_PAYLOAD_BYTES: usize = 4096;

/// Default maximum preview byte length for spilled payloads (512 bytes).
pub const DEFAULT_MAX_PREVIEW_BYTES: usize = 512;

/// Maximum allowed byte length for an action identifier (128 bytes).
pub const MAX_ACTION_ID_BYTES: usize = 128;

/// Maximum allowed byte length for a tool name (128 bytes).
pub const MAX_TOOL_NAME_BYTES: usize = 128;

/// Errors arising from action protocol processing and spillover.
#[derive(Debug)]
pub enum ActionError {
    /// Action identifier is empty or exceeds length limit.
    InvalidActionId(String),
    /// Tool name is empty or exceeds length limit.
    InvalidToolName(String),
    /// Action intent target or description is invalid.
    InvalidIntent(String),
    /// Tool arguments exceed maximum size or fail JSON validation.
    InvalidArguments(String),
    /// Raw payload exceeds maximum blob capacity.
    PayloadTooLarge {
        /// Actual size in bytes.
        size: usize,
        /// Maximum allowed limit in bytes.
        max: usize,
    },
    /// Storage error when persisting spilled payload.
    Store(String),
    /// JSON serialization or deserialization failure.
    Serialization(serde_json::Error),
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidActionId(id) => write!(f, "invalid action id: {id:?}"),
            Self::InvalidToolName(name) => write!(f, "invalid tool name: {name:?}"),
            Self::InvalidIntent(desc) => write!(f, "invalid action intent: {desc}"),
            Self::InvalidArguments(err) => write!(f, "invalid action arguments: {err}"),
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload size {size} exceeds maximum limit {max}")
            }
            Self::Store(err) => write!(f, "content store error during spillover: {err}"),
            Self::Serialization(err) => write!(f, "action serialization error: {err}"),
        }
    }
}

impl std::error::Error for ActionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(err) => Some(err),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for ActionError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err)
    }
}

impl From<ContentStoreError> for ActionError {
    fn from(err: ContentStoreError) -> Self {
        Self::Store(err.to_string())
    }
}

/// Categorized intent behind an agent's tool invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "target", rename_all = "snake_case")]
pub enum ActionIntent {
    /// Read-only inspection of files, diagnostics, or environmental state.
    Inspect(String),
    /// Direct mutation of code, configuration, or filesystem assets.
    Modify(String),
    /// Execution of commands, build processes, or tests.
    Execute(String),
    /// Verification of hypotheses, test criteria, or quality gates.
    Verify(String),
    /// Custom intent category with descriptive explanation.
    Custom {
        /// Custom intent category name.
        category: String,
        /// Descriptive rationale of the intended action.
        description: String,
    },
}

impl ActionIntent {
    /// Validate intent target and bounds.
    pub fn validate(&self) -> Result<(), ActionError> {
        match self {
            Self::Inspect(target)
            | Self::Modify(target)
            | Self::Execute(target)
            | Self::Verify(target) => {
                if target.trim().is_empty() {
                    return Err(ActionError::InvalidIntent("target cannot be empty".into()));
                }
                if target.len() > 1024 {
                    return Err(ActionError::InvalidIntent(
                        "target exceeds maximum 1024 bytes".into(),
                    ));
                }
            }
            Self::Custom {
                category,
                description,
            } => {
                if category.trim().is_empty() || description.trim().is_empty() {
                    return Err(ActionError::InvalidIntent(
                        "category and description cannot be empty".into(),
                    ));
                }
                if category.len() > 128 || description.len() > 1024 {
                    return Err(ActionError::InvalidIntent(
                        "category or description exceeds length limits".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Return a static string representing the intent type.
    #[must_use]
    pub fn intent_type(&self) -> &'static str {
        match self {
            Self::Inspect(_) => "inspect",
            Self::Modify(_) => "modify",
            Self::Execute(_) => "execute",
            Self::Verify(_) => "verify",
            Self::Custom { .. } => "custom",
        }
    }
}

/// A structured action request initiated by an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    /// Unique identifier for this action turn.
    pub id: String,
    /// Explicit engineering intent driving this invocation.
    pub intent: ActionIntent,
    /// Name of the invoked tool.
    pub tool_name: String,
    /// JSON-encoded arguments for the tool call.
    pub arguments_json: String,
}

impl Action {
    /// Construct and validate a new [`Action`].
    pub fn new(
        id: impl Into<String>,
        intent: ActionIntent,
        tool_name: impl Into<String>,
        arguments_json: impl Into<String>,
    ) -> Result<Self, ActionError> {
        let id = id.into();
        let tool_name = tool_name.into();
        let arguments_json = arguments_json.into();

        if id.trim().is_empty() || id.len() > MAX_ACTION_ID_BYTES {
            return Err(ActionError::InvalidActionId(id));
        }
        for b in id.bytes() {
            if !(b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'/') {
                return Err(ActionError::InvalidActionId(id));
            }
        }

        if tool_name.trim().is_empty() || tool_name.len() > MAX_TOOL_NAME_BYTES {
            return Err(ActionError::InvalidToolName(tool_name));
        }

        intent.validate()?;

        // Verify arguments_json is valid JSON
        if !arguments_json.trim().is_empty() {
            let _: serde_json::Value = serde_json::from_str(&arguments_json)
                .map_err(|e| ActionError::InvalidArguments(e.to_string()))?;
        }

        Ok(Self {
            id,
            intent,
            tool_name,
            arguments_json,
        })
    }
}

/// Cryptographic content address and size pointer to a spilled blob in the [`ContentStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobPointer {
    /// SHA-256 content address of the stored data.
    pub hash: ContentHash,
    /// Exact byte size of the raw stored blob.
    pub size_bytes: usize,
}

impl BlobPointer {
    /// Create a new blob pointer.
    #[must_use]
    pub fn new(hash: ContentHash, size_bytes: usize) -> Self {
        Self { hash, size_bytes }
    }
}

/// Payload representing tool output, either fully inline or spilled to the blob store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionPayload {
    /// Output fits within inline budget and is stored verbatim.
    Inline {
        /// Verbatim content string.
        content: String,
        /// Size in bytes.
        bytes: usize,
    },
    /// Output exceeded inline threshold, stored in [`ContentStore`], with bounded preview.
    Spilled {
        /// Bounded head and tail preview for LLM prompt context.
        preview: String,
        /// Pointer to full content in blob store.
        pointer: BlobPointer,
        /// Total size of raw unclipped payload.
        total_bytes: usize,
    },
}

impl ActionPayload {
    /// Check whether this payload was spilled to content storage.
    #[must_use]
    pub fn is_spilled(&self) -> bool {
        matches!(self, Self::Spilled { .. })
    }

    /// Total byte size of the full payload.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        match self {
            Self::Inline { bytes, .. } => *bytes,
            Self::Spilled { total_bytes, .. } => *total_bytes,
        }
    }

    /// Access the displayable text (either full inline content or preview).
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Inline { content, .. } => content.as_str(),
            Self::Spilled { preview, .. } => preview.as_str(),
        }
    }

    /// Access the blob pointer if spilled.
    #[must_use]
    pub fn blob_pointer(&self) -> Option<BlobPointer> {
        match self {
            Self::Inline { .. } => None,
            Self::Spilled { pointer, .. } => Some(*pointer),
        }
    }
}

/// Structured outcome of an executed action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Identifier of the action that was executed.
    pub action_id: String,
    /// Whether the action succeeded.
    pub success: bool,
    /// Process exit code, if applicable.
    pub exit_code: Option<i32>,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Standard output payload.
    pub stdout: ActionPayload,
    /// Standard error payload.
    pub stderr: ActionPayload,
}

impl ActionOutcome {
    /// Format this outcome into a structured, readable observation for Zone 3 in [`ContextCompiler`].
    #[must_use]
    pub fn format_for_context(&self) -> String {
        let status_str = if self.success { "success" } else { "failed" };
        let code_str = self
            .exit_code
            .map_or_else(|| "none".to_string(), |c| c.to_string());

        let mut out = format!(
            "[ACTION OUTCOME] id: {} | status: {} | exit_code: {} | duration: {}ms\n",
            self.action_id, status_str, code_str, self.duration_ms
        );

        // Format stdout
        match &self.stdout {
            ActionPayload::Inline { content, bytes } => {
                if *bytes == 0 {
                    out.push_str("stdout: (empty)\n");
                } else {
                    out.push_str("stdout:\n");
                    for line in content.lines() {
                        out.push_str("  ");
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
            ActionPayload::Spilled {
                preview,
                pointer,
                total_bytes,
            } => {
                out.push_str(&format!(
                    "stdout: [SPILLED to blob: {} ({} bytes)]\n",
                    pointer.hash, total_bytes
                ));
                for line in preview.lines() {
                    out.push_str("  ");
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }

        // Format stderr
        match &self.stderr {
            ActionPayload::Inline { content, bytes } => {
                if *bytes == 0 {
                    out.push_str("stderr: (empty)\n");
                } else {
                    out.push_str("stderr:\n");
                    for line in content.lines() {
                        out.push_str("  ");
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
            ActionPayload::Spilled {
                preview,
                pointer,
                total_bytes,
            } => {
                out.push_str(&format!(
                    "stderr: [SPILLED to blob: {} ({} bytes)]\n",
                    pointer.hash, total_bytes
                ));
                for line in preview.lines() {
                    out.push_str("  ");
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }

        out
    }
}

/// Configuration governing payload auto-spillover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpilloverConfig {
    /// Maximum payload size in bytes to retain inline (default 4096).
    pub max_inline_bytes: usize,
    /// Maximum byte size for the preview snippet when spilled (default 512).
    pub max_preview_bytes: usize,
}

impl Default for SpilloverConfig {
    fn default() -> Self {
        Self {
            max_inline_bytes: DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
            max_preview_bytes: DEFAULT_MAX_PREVIEW_BYTES,
        }
    }
}

/// Pluggable storage sink for persisting spilled blobs.
pub trait BlobSink {
    /// Persist data bytes and return the deterministic content address.
    fn store_blob(&mut self, data: &[u8], timestamp_ms: u64) -> Result<ContentHash, ActionError>;
}

impl BlobSink for ContentStore {
    fn store_blob(&mut self, data: &[u8], timestamp_ms: u64) -> Result<ContentHash, ActionError> {
        self.put_blob(data, timestamp_ms).map_err(ActionError::from)
    }
}

/// In-memory blob sink for testing and ephemeral workflows.
#[derive(Debug, Default)]
pub struct InMemoryBlobSink {
    blobs: std::collections::HashMap<ContentHash, Vec<u8>>,
}

impl InMemoryBlobSink {
    /// Create a new, empty in-memory blob sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve stored blob by content hash.
    #[must_use]
    pub fn get_blob(&self, hash: &ContentHash) -> Option<&[u8]> {
        self.blobs.get(hash).map(Vec::as_slice)
    }

    /// Number of blobs stored in memory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    /// Check if the store contains no blobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }
}

impl BlobSink for InMemoryBlobSink {
    fn store_blob(&mut self, data: &[u8], _timestamp_ms: u64) -> Result<ContentHash, ActionError> {
        if data.len() > MAX_BLOB_BYTES {
            return Err(ActionError::PayloadTooLarge {
                size: data.len(),
                max: MAX_BLOB_BYTES,
            });
        }
        let hash = ContentHash::compute(data);
        self.blobs.entry(hash).or_insert_with(|| data.to_vec());
        Ok(hash)
    }
}

/// Action execution engine managing payload evaluation and auto-spillover.
#[derive(Debug, Clone)]
pub struct ActionEngine {
    config: SpilloverConfig,
}

impl Default for ActionEngine {
    fn default() -> Self {
        Self::new(SpilloverConfig::default())
    }
}

impl ActionEngine {
    /// Create a new [`ActionEngine`] with explicit configuration.
    #[must_use]
    pub fn new(config: SpilloverConfig) -> Self {
        Self { config }
    }

    /// Return reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &SpilloverConfig {
        &self.config
    }

    /// Process a raw output string into an [`ActionPayload`], automatically spilling
    /// into the provided [`BlobSink`] if its byte size exceeds `max_inline_bytes`.
    pub fn process_payload<S: BlobSink>(
        &self,
        raw: &str,
        sink: &mut S,
        timestamp_ms: u64,
    ) -> Result<ActionPayload, ActionError> {
        let bytes = raw.len();
        if bytes > MAX_BLOB_BYTES {
            return Err(ActionError::PayloadTooLarge {
                size: bytes,
                max: MAX_BLOB_BYTES,
            });
        }

        if bytes <= self.config.max_inline_bytes {
            Ok(ActionPayload::Inline {
                content: raw.to_string(),
                bytes,
            })
        } else {
            // Spilling required: store full raw bytes
            let hash = sink.store_blob(raw.as_bytes(), timestamp_ms)?;
            let pointer = BlobPointer::new(hash, bytes);
            let preview = Self::generate_preview(raw, self.config.max_preview_bytes);

            Ok(ActionPayload::Spilled {
                preview,
                pointer,
                total_bytes: bytes,
            })
        }
    }

    /// Process a completed action turn into an [`ActionOutcome`], handling spillover for
    /// both stdout and stderr.
    #[allow(clippy::too_many_arguments)]
    pub fn process_outcome<S: BlobSink>(
        &self,
        action_id: impl Into<String>,
        success: bool,
        exit_code: Option<i32>,
        duration_ms: u64,
        raw_stdout: &str,
        raw_stderr: &str,
        sink: &mut S,
        timestamp_ms: u64,
    ) -> Result<ActionOutcome, ActionError> {
        let action_id = action_id.into();
        let stdout = self.process_payload(raw_stdout, sink, timestamp_ms)?;
        let stderr = self.process_payload(raw_stderr, sink, timestamp_ms)?;

        Ok(ActionOutcome {
            action_id,
            success,
            exit_code,
            duration_ms,
            stdout,
            stderr,
        })
    }

    /// Generate a bounded head-and-tail preview of a large output string.
    ///
    /// Preserves valid UTF-8 boundaries and inserts a clear truncation indicator.
    #[must_use]
    pub fn generate_preview(raw: &str, max_bytes: usize) -> String {
        if raw.len() <= max_bytes {
            return raw.to_string();
        }

        // Half for head, half for tail
        let half = max_bytes / 2;
        let head = Self::truncate_head_utf8(raw, half);
        let tail = Self::truncate_tail_utf8(raw, half);

        let skipped = raw.len().saturating_sub(head.len() + tail.len());

        let mut out = String::with_capacity(head.len() + tail.len() + 64);
        out.push_str(head.trim_end());
        out.push_str(&format!("\n[... skipped {skipped} bytes ...]\n"));
        out.push_str(tail.trim_start());
        out
    }

    fn truncate_head_utf8(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        let mut b = max_bytes;
        while b > 0 && !s.is_char_boundary(b) {
            b -= 1;
        }
        &s[..b]
    }

    fn truncate_tail_utf8(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        let mut b = s.len().saturating_sub(max_bytes);
        while b < s.len() && !s.is_char_boundary(b) {
            b += 1;
        }
        &s[b..]
    }
}

//! High-cohesion engine facade for `bitty-ai` subsystems (AI-0161).
//!
//! Converges the discrete Rust building blocks (streaming SSE parser,
//! context budget/prompt compiler, project snapshot verification, and
//! transactional SQLite event journal) into an integrated, ergonomic facade.

use std::fmt;
use std::path::Path;

use bitty_ai_runtime::context::ContextRecord;
use bitty_ai_runtime::prompt::{AssembledPrompt, PromptError, PromptSnapshot, assemble};
use bitty_ai_runtime::provider::ProviderTurn;
use bitty_ai_runtime::stream::{StreamChunk, StreamSink, VecSink};

use crate::chat_stream::{ChatCompletionStreamParser, ChatStreamDelta, ChatStreamError};
use crate::journal_prototype::{Journal, JournalError};
use crate::snapshot_ingest::{SnapshotIngestError, SnapshotIngestRequest, ingest_snapshot};

/// Unified error enum for the [`AiEngine`] facade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FacadeError {
    /// Chat completion streaming error.
    Stream(ChatStreamError),
    /// Project snapshot ingestion or verification error.
    Snapshot(SnapshotIngestError),
    /// Transactional journal error.
    Journal(JournalError),
    /// Content-addressed store or checkpoint error.
    Store(String),
    /// Task DAG and control plane error.
    TaskDag(String),
    /// Context compiler or Merkle tree error.
    Compiler(String),
    /// Prompt layer assembly or validation error.
    Prompt(PromptError),
    /// Context or payload budget ceiling exceeded.
    BudgetExceeded {
        /// Maximum allowed limit in bytes.
        limit: usize,
        /// Actual attempted bytes.
        actual: usize,
    },
}

impl fmt::Display for FacadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream(err) => write!(f, "streaming error: {err}"),
            Self::Snapshot(err) => write!(f, "snapshot error: {err}"),
            Self::Journal(err) => write!(f, "journal error: {err}"),
            Self::Store(err) => write!(f, "content store error: {err}"),
            Self::TaskDag(err) => write!(f, "task engine error: {err}"),
            Self::Compiler(err) => write!(f, "compiler error: {err}"),
            Self::Prompt(err) => write!(f, "prompt error: {err}"),
            Self::BudgetExceeded { limit, actual } => {
                write!(f, "budget ceiling exceeded: {actual} bytes > {limit} bytes")
            }
        }
    }
}

impl std::error::Error for FacadeError {}

impl From<ChatStreamError> for FacadeError {
    fn from(err: ChatStreamError) -> Self {
        Self::Stream(err)
    }
}

impl From<SnapshotIngestError> for FacadeError {
    fn from(err: SnapshotIngestError) -> Self {
        Self::Snapshot(err)
    }
}

impl From<JournalError> for FacadeError {
    fn from(err: JournalError) -> Self {
        Self::Journal(err)
    }
}

impl From<crate::content_store::ContentStoreError> for FacadeError {
    fn from(err: crate::content_store::ContentStoreError) -> Self {
        Self::Store(err.to_string())
    }
}

impl From<crate::task_dag::TaskEngineError> for FacadeError {
    fn from(err: crate::task_dag::TaskEngineError) -> Self {
        Self::TaskDag(err.to_string())
    }
}

impl From<crate::context_compiler::CompilerError> for FacadeError {
    fn from(err: crate::context_compiler::CompilerError) -> Self {
        Self::Compiler(err.to_string())
    }
}

impl From<PromptError> for FacadeError {
    fn from(err: PromptError) -> Self {
        Self::Prompt(err)
    }
}

/// Unified chat completion streaming session.
///
/// Encapsulates the incremental SSE parser, token/tool-call delta extraction,
/// and automated piping into an attached [`StreamSink`].
pub struct AiStreamSession {
    parser: ChatCompletionStreamParser,
    sink: Box<dyn StreamSink>,
    seq: u32,
}

impl Default for AiStreamSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AiStreamSession {
    /// Create a new streaming session with a default in-memory [`VecSink`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            parser: ChatCompletionStreamParser::new(),
            sink: Box::new(VecSink::new()),
            seq: 0,
        }
    }

    /// Create a new streaming session with a caller-supplied [`StreamSink`].
    #[must_use]
    pub fn with_sink(sink: Box<dyn StreamSink>) -> Self {
        Self {
            parser: ChatCompletionStreamParser::new(),
            sink,
            seq: 0,
        }
    }

    /// Feed an arbitrary chunk of raw network bytes into the stream session.
    ///
    /// Incrementally parses SSE lines and JSON chat completion chunks,
    /// emits parsed [`ChatStreamDelta`] events, and automatically pipes
    /// content deltas into the attached sink as sequenced Markdown chunks.
    pub fn feed_chunk(&mut self, chunk: &[u8]) -> Result<Vec<ChatStreamDelta>, FacadeError> {
        let deltas = self.parser.feed(chunk).map_err(FacadeError::Stream)?;
        for delta in &deltas {
            self.parser
                .pipe_to_sink(delta, self.sink.as_mut(), &mut self.seq)
                .map_err(FacadeError::Stream)?;
        }
        Ok(deltas)
    }

    fn drain_eof_to_sink(&mut self) -> Result<(), FacadeError> {
        let eof_deltas = self.parser.drain_eof().map_err(FacadeError::Stream)?;
        for delta in &eof_deltas {
            self.parser
                .pipe_to_sink(delta, self.sink.as_mut(), &mut self.seq)
                .map_err(FacadeError::Stream)?;
        }
        Ok(())
    }

    /// Finalize the stream and obtain the consolidated [`ProviderTurn`].
    pub fn finish(mut self) -> Result<ProviderTurn, FacadeError> {
        self.drain_eof_to_sink()?;
        self.parser.finish().map_err(FacadeError::Stream)
    }

    /// Finalize the stream and obtain both the [`ProviderTurn`] and the underlying [`StreamSink`].
    pub fn finish_with_sink(mut self) -> Result<(ProviderTurn, Box<dyn StreamSink>), FacadeError> {
        self.drain_eof_to_sink()?;
        let turn = self.parser.finish().map_err(FacadeError::Stream)?;
        Ok((turn, self.sink))
    }

    /// Read-only access to all retained chunks in the attached sink.
    #[must_use]
    pub fn chunks(&self) -> &[StreamChunk] {
        self.sink.chunks()
    }

    /// Next sequence number that will be emitted.
    #[must_use]
    pub fn seq(&self) -> u32 {
        self.seq
    }

    /// Extract concatenated UTF-8 text from all accepted chunks in the sink.
    #[must_use]
    pub fn concatenated_text(&self) -> String {
        let mut out = Vec::new();
        for chunk in self.sink.chunks() {
            out.extend_from_slice(&chunk.fragment.bytes);
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// Project snapshot ingestion engine.
///
/// Encapsulates digest verification and adaptation to a runtime [`ContextRecord`]
/// with an internal [`ArtifactStore`] for externalized bodies.
#[derive(Debug, Default)]
pub struct AiSnapshotEngine {
    store: bitty_ai_runtime::context::ArtifactStore,
}

impl AiSnapshotEngine {
    /// Create a new snapshot engine with a fresh artifact store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Verify canonical snapshot bytes and adapt them to a runtime [`ContextRecord`].
    pub fn ingest(
        &mut self,
        request: &SnapshotIngestRequest<'_>,
    ) -> Result<ContextRecord, FacadeError> {
        ingest_snapshot(request, &mut self.store).map_err(FacadeError::Snapshot)
    }

    /// Read-only access to the underlying artifact store.
    #[must_use]
    pub fn store(&self) -> &bitty_ai_runtime::context::ArtifactStore {
        &self.store
    }
}

/// Top-level facade coordinating Bitty AI subsystems.
#[derive(Debug, Default)]
pub struct AiEngine;

impl AiEngine {
    /// Create a new streaming chat session with an in-memory [`VecSink`].
    #[must_use]
    pub fn new_stream_session() -> AiStreamSession {
        AiStreamSession::new()
    }

    /// Create a new streaming chat session with a caller-supplied [`StreamSink`].
    #[must_use]
    pub fn stream_session_with_sink(sink: Box<dyn StreamSink>) -> AiStreamSession {
        AiStreamSession::with_sink(sink)
    }

    /// Assemble and compile prompt layers into canonical bytes under budget.
    pub fn assemble_prompt(snapshot: &PromptSnapshot) -> Result<AssembledPrompt, FacadeError> {
        assemble(snapshot).map_err(FacadeError::Prompt)
    }

    /// Open a transactional SQLite journal at the specified path.
    pub fn open_journal(path: impl AsRef<Path>) -> Result<Journal, FacadeError> {
        Journal::open(path.as_ref()).map_err(FacadeError::Journal)
    }

    /// Open a transactional content store at the specified path.
    pub fn open_content_store(
        path: impl AsRef<Path>,
    ) -> Result<crate::content_store::ContentStore, FacadeError> {
        crate::content_store::ContentStore::open(path).map_err(FacadeError::from)
    }

    /// Open an in-memory transactional content store.
    pub fn open_in_memory_content_store() -> Result<crate::content_store::ContentStore, FacadeError>
    {
        crate::content_store::ContentStore::open_in_memory().map_err(FacadeError::from)
    }

    /// Open a persistent Task DAG SQLite database at the specified path.
    pub fn open_task_engine(
        path: impl AsRef<Path>,
    ) -> Result<crate::task_dag::TaskEngine, FacadeError> {
        crate::task_dag::TaskEngine::open(path).map_err(FacadeError::from)
    }

    /// Open an in-memory Task DAG SQLite database.
    pub fn open_in_memory_task_engine() -> Result<crate::task_dag::TaskEngine, FacadeError> {
        crate::task_dag::TaskEngine::open_in_memory().map_err(FacadeError::from)
    }

    /// Compile structured cognitive state and dynamic turns into a cached, three-zone context.
    pub fn compile_context(
        compiler: &crate::context_compiler::ContextCompiler,
        budget: &crate::context_compiler::CompilerBudgetConfig,
    ) -> Result<crate::context_compiler::CompiledContext, FacadeError> {
        compiler.compile(budget).map_err(FacadeError::from)
    }

    /// Create a new project snapshot ingestion engine.
    #[must_use]
    pub fn new_snapshot_engine() -> AiSnapshotEngine {
        AiSnapshotEngine::new()
    }
}

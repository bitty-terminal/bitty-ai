//! End-to-end slice loop: prompt -> provider -> context -> tool -> stream.
//!
//! The loop is deterministic and offline. Context assembly is bounded per
//! `CP-5` before the collected bytes are used; the tool is validated against a
//! bounded registry per `TB-3`; every streamed chunk is validated against RC-10
//! per `RS-5`. All time comes from the caller's `now_ms`.

use crate::bridge::{HostPeer, IpcBridge};
use crate::context::{
    CONTEXT_BUDGET_BYTES, ContextProvider, ContextRecord, ContextRequest, IpcTerminalContext,
};
use crate::error::SliceError;
use crate::provider::{Message, ModelProvider, ProviderTurn};
use crate::stream::{Fragment, FragmentKind, PanelStreamSink, StreamChunk, StreamSink};
use crate::toolbus::{HostToolBus, ToolBus, ToolHost, ToolInvocation, ToolOutcome};

/// Result of one completed turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceOutcome {
    /// Provider identity that produced the turn.
    pub provider_id: String,
    /// Assistant answer text.
    pub answer: String,
    /// Bounded context record, when the turn requested context.
    pub context: Option<ContextRecord>,
    /// Tool observation, when the turn requested a tool.
    pub tool: Option<ToolOutcome>,
    /// Streamed chunks accepted by the sink, in order.
    pub chunks: Vec<StreamChunk>,
}

/// The wired vertical slice.
pub struct VerticalSlice<P: ModelProvider> {
    provider: P,
    context: IpcTerminalContext,
    bridge: IpcBridge,
    toolbus: HostToolBus,
    sink: PanelStreamSink,
    generation: u64,
}

impl<P: ModelProvider> VerticalSlice<P> {
    /// Wire the four components together.
    #[must_use]
    pub fn new(
        provider: P,
        bridge: IpcBridge,
        toolbus: HostToolBus,
        sink: PanelStreamSink,
    ) -> Self {
        Self {
            provider,
            context: IpcTerminalContext,
            bridge,
            toolbus,
            sink,
            generation: 1,
        }
    }

    /// Run one full turn.
    ///
    /// # Errors
    ///
    /// Propagates provider bound failures, unsupported/denied context reads,
    /// the 32 KiB context budget, unknown/denied/over-bound tool calls, and
    /// stream chunk violations.
    pub fn run_turn(
        &mut self,
        peer: &mut dyn HostPeer,
        host: &mut dyn ToolHost,
        messages: &[Message],
        request: &ContextRequest,
        now_ms: u64,
    ) -> Result<SliceOutcome, SliceError> {
        let Self {
            provider,
            context,
            bridge,
            toolbus,
            sink,
            generation,
        } = self;
        toolbus.begin_turn();

        let turn: ProviderTurn = provider.complete(messages)?;

        let mut context_record = None;
        if turn.tool_call.is_some() {
            let record = context.collect(bridge, peer, request, *generation, now_ms)?;
            if record.bytes.len() > CONTEXT_BUDGET_BYTES {
                return Err(SliceError::ContextBudgetExceeded {
                    limit: CONTEXT_BUDGET_BYTES,
                    actual: record.bytes.len(),
                });
            }
            context_record = Some(record);
        }

        let tool_outcome = match &turn.tool_call {
            Some(call) => {
                let invocation = ToolInvocation {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                };
                Some(toolbus.dispatch(host, &invocation, now_ms)?)
            }
            None => None,
        };

        let fragments = build_fragments(&turn, tool_outcome.as_ref());
        let total = u32::try_from(fragments.len()).map_err(|_| SliceError::StreamViolation {
            reason: "more fragments than u32".to_owned(),
        })?;
        for (index, fragment) in fragments.into_iter().enumerate() {
            let seq = u32::try_from(index).map_err(|_| SliceError::StreamViolation {
                reason: "chunk index overflow".to_owned(),
            })?;
            sink.emit(StreamChunk {
                seq,
                total,
                is_final: seq + 1 == total,
                fragment,
            })?;
        }

        let chunks = sink.chunks().to_vec();
        *generation += 1;
        Ok(SliceOutcome {
            provider_id: provider.provider_id().to_owned(),
            answer: turn.text.clone(),
            context: context_record,
            tool: tool_outcome,
            chunks,
        })
    }
}

fn build_fragments(turn: &ProviderTurn, tool: Option<&ToolOutcome>) -> Vec<Fragment> {
    let mut fragments = vec![Fragment {
        kind: FragmentKind::Markdown,
        bytes: turn.text.as_bytes().to_vec(),
    }];
    if let Some(tool) = tool {
        let card = format!("{}: {}", tool.name, String::from_utf8_lossy(&tool.result));
        fragments.push(Fragment {
            kind: FragmentKind::ToolCard,
            bytes: card.into_bytes(),
        });
    }
    fragments
}

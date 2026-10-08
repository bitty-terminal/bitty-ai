//! Boundary plane bridge providing a zero-unsafe, message-based dispatch interface (AI-0167).
//!
//! Encapsulates [`WheelKernel`] and exposes standard JSON-RPC command dispatch
//! for Lua plugins and host integration without raw pointer dereferencing or memory unsafety.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::action_protocol::SpilloverConfig;
use crate::content_hash::ContentHash;
use crate::content_store::Rationale;
use crate::context_compiler::CompilerBudgetConfig;
use crate::facade::FacadeError;
use crate::merge::{GcOptions, GcReport, MergeInput};
use crate::task_dag::{TaskDraft, TaskId};
use crate::wheel_kernel::{ResumeReport, WheelKernel};
use bitty_ai_session::sessions::SessionBinding;

/// Unified bridge response payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeResponse {
    /// Whether the command succeeded.
    pub success: bool,
    /// Response payload data if successful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    /// Error message string if unsuccessful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl BridgeResponse {
    /// Create a successful response with payload.
    #[must_use]
    pub fn ok(data: serde_json::Value) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    /// Create a failure response with an error description.
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }

    /// Serialize this response to a canonical JSON string.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self)
            .unwrap_or_else(|e| format!(r#"{{"success":false,"error":"{e}"}}"#))
    }
}

/// Boundary bridge wrapping [`WheelKernel`] with JSON-RPC command dispatch.
pub struct WheelBridge {
    kernel: WheelKernel,
}

impl WheelBridge {
    /// Create a new bridge wrapping an existing [`WheelKernel`].
    #[must_use]
    pub fn new(kernel: WheelKernel) -> Self {
        Self { kernel }
    }

    /// Open a persistent bridge at the specified SQLite path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FacadeError> {
        let kernel = WheelKernel::open(path)?;
        Ok(Self::new(kernel))
    }

    /// Open an in-memory bridge with default configurations.
    pub fn open_in_memory() -> Result<Self, FacadeError> {
        let kernel = WheelKernel::open_in_memory()?;
        Ok(Self::new(kernel))
    }

    /// Open an in-memory bridge with caller-supplied budget and spillover configurations.
    pub fn open_in_memory_with_config(
        budget_config: CompilerBudgetConfig,
        spillover_config: SpilloverConfig,
    ) -> Result<Self, FacadeError> {
        let kernel = WheelKernel::open_in_memory_with_config(budget_config, spillover_config)?;
        Ok(Self::new(kernel))
    }

    /// Read-only access to the underlying [`WheelKernel`].
    #[must_use]
    pub fn kernel(&self) -> &WheelKernel {
        &self.kernel
    }

    /// Mutable access to the underlying [`WheelKernel`].
    pub fn kernel_mut(&mut self) -> &mut WheelKernel {
        &mut self.kernel
    }

    /// Dispatch a command with a JSON string payload, returning a serialized [`BridgeResponse`].
    ///
    /// This is the primary boundary entry point called by Lua scripts, FFI shims,
    /// or host IPC endpoints. It never panics and always returns a valid JSON string.
    pub fn dispatch(&mut self, command: &str, payload_json: &str) -> String {
        let payload: serde_json::Value = match serde_json::from_str(payload_json) {
            Ok(v) => v,
            Err(e) => {
                return BridgeResponse::err(format!("invalid JSON payload: {e}")).to_json_string();
            }
        };

        match self.dispatch_typed(command, payload) {
            Ok(data) => BridgeResponse::ok(data).to_json_string(),
            Err(err) => BridgeResponse::err(err).to_json_string(),
        }
    }

    /// Internal typed dispatcher executing a command against the kernel.
    fn dispatch_typed(
        &mut self,
        command: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        match command {
            "kernel.status" => {
                let status = serde_json::json!({
                    "active_task": self.kernel.active_task().map(|t| t.as_str()),
                    "head_checkpoint": self.kernel.head_checkpoint().map(|c| c.to_hex()),
                    "tree_hash": self.kernel.tree_hash().to_hex(),
                    "slot_count": self.kernel.active_tree().len(),
                    "task_count": self.kernel.list_tasks().map_err(|e| e.to_string())?.len(),
                    "recent_action_count": self.kernel.recent_actions().len(),
                    "budget_config": self.kernel.budget_config(),
                    "spillover_config": self.kernel.spillover_config(),
                });
                Ok(status)
            }
            "task.create" => {
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                let draft: TaskDraft = serde_json::from_value(payload)
                    .map_err(|e| format!("invalid task draft: {e}"))?;
                let task = self
                    .kernel
                    .create_task(draft, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.get" | "task.get_view" => {
                let id_str = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'id' field")?;
                let id = TaskId::new(id_str).map_err(|e| e.to_string())?;
                let task = self.kernel.get_task_view(&id).map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.list" | "task.list_views" => {
                let tasks = self.kernel.list_task_views().map_err(|e| e.to_string())?;
                serde_json::to_value(tasks).map_err(|e| e.to_string())
            }
            "task.set_active" => {
                let id_opt = match payload.get("id") {
                    Some(v) if v.is_null() => None,
                    Some(v) => match v.as_str() {
                        Some("") => None,
                        Some(s) => Some(TaskId::new(s).map_err(|e| e.to_string())?),
                        None => {
                            return Err("invalid 'id' type, expected string or null".to_string());
                        }
                    },
                    None => None,
                };
                self.kernel
                    .set_active_task(id_opt)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "active_task": self.kernel.active_task().map(|t| t.as_str()),
                }))
            }
            "task.start" => {
                let id_str = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'id' field")?;
                let worker_id = payload
                    .get("worker_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("worker");
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let id = TaskId::new(id_str).map_err(|e| e.to_string())?;
                let task = self
                    .kernel
                    .start_task(&id, worker_id, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.complete" => {
                let id_str = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'id' field")?;
                let generation = payload
                    .get("expected_generation")
                    .and_then(|v| v.as_u64())
                    .ok_or("missing or invalid 'expected_generation' field")?;
                let checkpoint = match payload.get("checkpoint").and_then(|v| v.as_str()) {
                    Some(s) if !s.is_empty() => {
                        Some(ContentHash::from_hex(s).map_err(|e| e.to_string())?)
                    }
                    _ => None,
                };
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let id = TaskId::new(id_str).map_err(|e| e.to_string())?;
                let task = self
                    .kernel
                    .complete_task(&id, generation, checkpoint, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.fail" => {
                let id_str = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'id' field")?;
                let error_msg = payload
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unspecified error");
                let generation = payload
                    .get("expected_generation")
                    .and_then(|v| v.as_u64())
                    .ok_or("missing or invalid 'expected_generation' field")?;
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let id = TaskId::new(id_str).map_err(|e| e.to_string())?;
                let task = self
                    .kernel
                    .fail_task(&id, generation, error_msg, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.cancel" => {
                let id_str = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'id' field")?;
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let id = TaskId::new(id_str).map_err(|e| e.to_string())?;
                let task = self
                    .kernel
                    .cancel_task(&id, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.retry" => {
                let id_str = payload
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'id' field")?;
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let id = TaskId::new(id_str).map_err(|e| e.to_string())?;
                let task = self
                    .kernel
                    .retry_task(&id, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(task).map_err(|e| e.to_string())
            }
            "task.topological_sort" => {
                let order = self.kernel.topological_sort().map_err(|e| e.to_string())?;
                let strings: Vec<&str> = order.iter().map(|id| id.as_str()).collect();
                serde_json::to_value(strings).map_err(|e| e.to_string())
            }
            "slot.put" => {
                let name = payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'name' field")?;
                let content = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'content' field")?;
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let hash = self
                    .kernel
                    .put_slot(name, content.as_bytes(), now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "name": name,
                    "hash": hash.to_hex(),
                    "size_bytes": content.len(),
                }))
            }
            "slot.get" => {
                let name = payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'name' field")?;
                match self.kernel.get_slot(name).map_err(|e| e.to_string())? {
                    Some(bytes) => {
                        let content = String::from_utf8_lossy(&bytes).into_owned();
                        Ok(serde_json::json!({
                            "name": name,
                            "found": true,
                            "content": content,
                            "size_bytes": bytes.len(),
                        }))
                    }
                    None => Ok(serde_json::json!({
                        "name": name,
                        "found": false,
                    })),
                }
            }
            "slot.remove" => {
                let name = payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'name' field")?;
                let removed = self.kernel.remove_slot(name);
                Ok(serde_json::json!({
                    "name": name,
                    "removed": removed,
                }))
            }
            "slot.list" => {
                let entries: Vec<serde_json::Value> = self
                    .kernel
                    .list_slots()
                    .into_iter()
                    .map(|e| {
                        serde_json::json!({
                            "name": e.name,
                            "hash": e.hash.to_hex(),
                            "kind": e.kind,
                            "size_bytes": e.size_bytes,
                        })
                    })
                    .collect();
                Ok(serde_json::Value::Array(entries))
            }
            "checkpoint.commit" => {
                let rationale_val = payload
                    .get("rationale")
                    .ok_or("missing 'rationale' object")?;
                let rationale: Rationale = serde_json::from_value(rationale_val.clone())
                    .map_err(|e| format!("invalid rationale: {e}"))?;
                let branch = payload.get("branch").and_then(|v| v.as_str());
                let now_ms = payload.get("now_ms").and_then(|v| v.as_u64()).unwrap_or(0);

                let cp = self
                    .kernel
                    .commit_checkpoint(rationale, branch, now_ms)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(cp).map_err(|e| e.to_string())
            }
            "checkpoint.get" => {
                let hash_str = payload
                    .get("hash")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'hash' field")?;
                let hash = ContentHash::from_hex(hash_str).map_err(|e| e.to_string())?;
                let cp = self
                    .kernel
                    .get_checkpoint(&hash)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(cp).map_err(|e| e.to_string())
            }
            "checkpoint.log" => {
                let max_depth = payload
                    .get("max_depth")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize;
                let log = self.kernel.log(max_depth).map_err(|e| e.to_string())?;
                serde_json::to_value(log).map_err(|e| e.to_string())
            }
            "action.record" => {
                let action_id = payload
                    .get("action_id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'action_id' field")?;
                let success = payload
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let exit_code = match payload.get("exit_code") {
                    Some(v) if v.is_number() => {
                        let c = v.as_i64().ok_or_else(|| {
                            "'exit_code' is outside valid integer range".to_string()
                        })?;
                        let c32 = i32::try_from(c).map_err(|_| {
                            "'exit_code' exceeds valid 32-bit signed integer range".to_string()
                        })?;
                        Some(c32)
                    }
                    _ => None,
                };
                let duration_ms = payload
                    .get("duration_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let raw_stdout = payload
                    .get("raw_stdout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let raw_stderr = payload
                    .get("raw_stderr")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let timestamp_ms = payload
                    .get("timestamp_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let outcome = self
                    .kernel
                    .record_action_outcome(
                        action_id,
                        success,
                        exit_code,
                        duration_ms,
                        raw_stdout,
                        raw_stderr,
                        timestamp_ms,
                    )
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(outcome).map_err(|e| e.to_string())
            }
            "action.recent" => {
                let actions = self.kernel.recent_actions();
                serde_json::to_value(actions).map_err(|e| e.to_string())
            }
            "action.clear" => {
                self.kernel.clear_recent_actions();
                Ok(serde_json::json!({ "cleared": true }))
            }
            "context.compile" => {
                let system_instruction = payload
                    .get("system_instruction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let project_rules_vals = payload
                    .get("project_rules")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let project_rules: Vec<&str> = project_rules_vals
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect();

                let tool_schemas_vals = payload
                    .get("tool_schemas")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let tool_schemas: Vec<&str> = tool_schemas_vals
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect();

                let turn_prompt = payload
                    .get("turn_prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let compiled = self
                    .kernel
                    .compile_context(
                        system_instruction,
                        &project_rules,
                        &tool_schemas,
                        turn_prompt,
                    )
                    .map_err(|e| e.to_string())?;

                let res = serde_json::json!({
                    "zone1_prefix": compiled.zone1_prefix,
                    "zone2_state": compiled.zone2_state,
                    "zone3_tail": compiled.zone3_tail,
                    "prefix_hash": compiled.prefix_hash.to_hex(),
                    "total_bytes": compiled.total_bytes,
                    "prompt_string": compiled.to_prompt_string(),
                    "pruned_slots": compiled.pruned_slots,
                    "summarized_checkpoints": compiled.summarized_checkpoints,
                    "truncated_tail": compiled.truncated_tail,
                });
                Ok(res)
            }
            "merge.commit" => {
                let ours_hex = payload
                    .get("ours")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'ours' field")?;
                let theirs_hex = payload
                    .get("theirs")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'theirs' field")?;
                let ours = ContentHash::from_hex(ours_hex).map_err(|e| e.to_string())?;
                let theirs = ContentHash::from_hex(theirs_hex).map_err(|e| e.to_string())?;
                let target_branch = payload
                    .get("target_branch")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'target_branch' field")?;
                let task_id = payload
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'task_id' field")?;
                let agent_id = payload
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'agent_id' field")?;
                let rationale_val = payload
                    .get("rationale")
                    .ok_or("missing 'rationale' object")?;
                let rationale: Rationale = serde_json::from_value(rationale_val.clone())
                    .map_err(|e| format!("invalid rationale: {e}"))?;
                let summary = payload
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'summary' field")?;
                let expected_task_generation = match payload.get("expected_task_generation") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(v) => Some(
                        v.as_u64()
                            .ok_or("missing or invalid 'expected_task_generation' field")?,
                    ),
                };
                let actor = payload
                    .get("actor")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'actor' field")?;
                let reason = payload
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'reason' field")?;
                let at_ms = payload.get("at_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                let input = MergeInput {
                    ours,
                    theirs,
                    target_branch: target_branch.to_string(),
                    task_id: task_id.to_string(),
                    agent_id: agent_id.to_string(),
                    rationale,
                    summary: summary.to_string(),
                    expected_task_generation,
                    actor: actor.to_string(),
                    reason: reason.to_string(),
                    at_ms,
                };
                let cp = self.kernel.merge_commit(input).map_err(|e| e.to_string())?;
                serde_json::to_value(cp).map_err(|e| e.to_string())
            }
            "gc.preview" => {
                let options = gc_options_from_payload(&payload)?;
                let report = self
                    .kernel
                    .gc_preview(&options)
                    .map_err(|e| e.to_string())?;
                Ok(gc_report_to_value(&report))
            }
            "gc.collect" | "gc.collect_garbage" => {
                let options = gc_options_from_payload(&payload)?;
                let report = self
                    .kernel
                    .collect_garbage(&options)
                    .map_err(|e| e.to_string())?;
                Ok(gc_report_to_value(&report))
            }
            "reflog.prune" => {
                let ref_name = payload
                    .get("ref_name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing or invalid 'ref_name' field")?;
                let older_than_ms = match payload.get("older_than_ms") {
                    None | Some(serde_json::Value::Null) => 0,
                    Some(v) => v
                        .as_u64()
                        .ok_or("missing or invalid 'older_than_ms' field, expected u64")?,
                };
                let max_rows = match payload.get("max_rows") {
                    None | Some(serde_json::Value::Null) => 0,
                    Some(v) => v
                        .as_u64()
                        .ok_or("missing or invalid 'max_rows' field, expected u64")?,
                } as usize;
                let tombstone_grace_ms = match payload.get("tombstone_grace_ms") {
                    None | Some(serde_json::Value::Null) => 0,
                    Some(v) => v
                        .as_u64()
                        .ok_or("missing or invalid 'tombstone_grace_ms' field, expected u64")?,
                };
                let now_ms = match payload.get("now_ms") {
                    None | Some(serde_json::Value::Null) => 0,
                    Some(v) => v
                        .as_u64()
                        .ok_or("missing or invalid 'now_ms' field, expected u64")?,
                };
                let report = self
                    .kernel
                    .prune_reflog(
                        ref_name,
                        older_than_ms,
                        max_rows,
                        tombstone_grace_ms,
                        now_ms,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(prune_report_to_value(&report))
            }
            // --- Session resume/CLI surface (AI-0197) ---
            //
            // Strict field types throughout (AI-0190 lesson): a
            // present-but-wrong-typed field is an error naming the field,
            // never a silent default. Required fields (`session_id`,
            // `ref`, `claim_epoch`, `name`, `target`, ...) error when
            // missing or null; optional clocks/metadata (`now_ms`,
            // `reason`, `actor`) default only when absent or null.
            "session.new" => {
                let session_id = req_string(&payload, "session_id")?;
                let branch = req_string(&payload, "branch")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let binding = self
                    .kernel
                    .new_session(&session_id, &branch, now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(session_binding_to_value(&binding))
            }
            "session.resume" => {
                let ref_or_branch = req_string(&payload, "ref")?;
                let claim_epoch = req_u64(&payload, "claim_epoch")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let report = self
                    .kernel
                    .resume_session(&ref_or_branch, claim_epoch, now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(resume_report_to_value(&report))
            }
            "session.fork" => {
                let name = req_string(&payload, "name")?;
                let from_tip = req_hash(&payload, "from_tip")?;
                let reason = opt_string(&payload, "reason", "")?;
                let actor = opt_string(&payload, "actor", "")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let branch = self
                    .kernel
                    .fork_branch(&name, &from_tip, &reason, &actor, now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "name": branch.as_str(),
                    "target": from_tip.to_hex(),
                }))
            }
            "session.list" => {
                let bindings = self.kernel.list_sessions().map_err(|e| e.to_string())?;
                let values: Vec<serde_json::Value> =
                    bindings.iter().map(session_binding_to_value).collect();
                Ok(serde_json::Value::Array(values))
            }
            "branch.create" => {
                let name = req_string(&payload, "name")?;
                let target = req_hash(&payload, "target")?;
                let reason = opt_string(&payload, "reason", "")?;
                let actor = opt_string(&payload, "actor", "")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let branch = self
                    .kernel
                    .create_branch(&name, &target, &reason, &actor, now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "name": branch.as_str(),
                    "target": target.to_hex(),
                }))
            }
            "branch.list" => {
                let branches = self.kernel.list_branches().map_err(|e| e.to_string())?;
                let values: Vec<serde_json::Value> = branches
                    .iter()
                    .map(|(name, target)| {
                        serde_json::json!({
                            "name": name.as_str(),
                            "target": target.to_hex(),
                        })
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            "branch.get" => {
                let name = req_string(&payload, "name")?;
                match self.kernel.get_branch(&name).map_err(|e| e.to_string())? {
                    Some(target) => Ok(serde_json::json!({
                        "name": name,
                        "found": true,
                        "target": target.to_hex(),
                    })),
                    None => Ok(serde_json::json!({
                        "name": name,
                        "found": false,
                    })),
                }
            }
            "branch.update" => {
                let name = req_string(&payload, "name")?;
                let target = req_hash(&payload, "target")?;
                let reason = opt_string(&payload, "reason", "")?;
                let actor = opt_string(&payload, "actor", "")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let fast_forward_only = opt_bool(&payload, "fast_forward_only", false)?;
                let branch = self
                    .kernel
                    .update_branch(&name, &target, &reason, &actor, now_ms, fast_forward_only)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "name": branch.as_str(),
                    "target": target.to_hex(),
                }))
            }
            "branch.delete" => {
                let name = req_string(&payload, "name")?;
                let reason = opt_string(&payload, "reason", "")?;
                let actor = opt_string(&payload, "actor", "")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let deleted = self
                    .kernel
                    .delete_branch(&name, &reason, &actor, now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "name": name,
                    "deleted_target": deleted.to_hex(),
                }))
            }
            "branch.rename" => {
                let old_name = req_string(&payload, "old_name")?;
                let new_name = req_string(&payload, "new_name")?;
                let reason = opt_string(&payload, "reason", "")?;
                let actor = opt_string(&payload, "actor", "")?;
                let now_ms = opt_u64(&payload, "now_ms", 0)?;
                let target = self
                    .kernel
                    .rename_branch(&old_name, &new_name, &reason, &actor, now_ms)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "old_name": old_name,
                    "new_name": new_name,
                    "target": target.to_hex(),
                }))
            }
            "reflog.read" => {
                let ref_name = req_string(&payload, "ref_name")?;
                let limit = opt_u64(&payload, "limit", 64)? as usize;
                let entries = self
                    .kernel
                    .read_reflog(&ref_name, limit)
                    .map_err(|e| e.to_string())?;
                let values: Vec<serde_json::Value> = entries
                    .iter()
                    .map(|entry| {
                        serde_json::json!({
                            "seq": entry.seq,
                            "ref_name": entry.ref_name,
                            "old_hash": entry.old_hash.as_ref().map(|h| h.to_hex()),
                            "new_hash": entry.new_hash.to_hex(),
                            "reason": entry.reason,
                            "actor": entry.actor,
                            "at_ms": entry.at_ms,
                        })
                    })
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            unknown => Err(format!("unknown bridge command: '{unknown}'")),
        }
    }
}

/// Parse caller-supplied GC options from a bridge payload.
///
/// Clocks are caller-supplied only (`now_ms` selects the reflog grace
/// window); missing clocks default to `0` like the other bridge verbs and
/// `max_deletes_per_call` defaults to the [`GcOptions`] default (256).
/// Fail-closed on types: a present-but-wrong-typed field is an error rather
/// than a silent default, so a mistyped `dry_run` can never turn a preview
/// into a destructive collect. Absent or null fields keep their defaults.
fn gc_options_from_payload(payload: &serde_json::Value) -> Result<GcOptions, String> {
    let now_ms = match payload.get("now_ms") {
        None | Some(serde_json::Value::Null) => 0,
        Some(v) => v
            .as_u64()
            .ok_or("missing or invalid 'now_ms' field, expected u64")?,
    };
    let reflog_grace_ms = match payload.get("reflog_grace_ms") {
        None | Some(serde_json::Value::Null) => 0,
        Some(v) => v
            .as_u64()
            .ok_or("missing or invalid 'reflog_grace_ms' field, expected u64")?,
    };
    let max_deletes_per_call = match payload.get("max_deletes_per_call") {
        None | Some(serde_json::Value::Null) => 256,
        Some(v) => v
            .as_u64()
            .ok_or("missing or invalid 'max_deletes_per_call' field, expected u64")?,
    } as usize;
    let dry_run = match payload.get("dry_run") {
        None | Some(serde_json::Value::Null) => false,
        Some(v) => v
            .as_bool()
            .ok_or("missing or invalid 'dry_run' field, expected boolean")?,
    };
    Ok(GcOptions {
        now_ms,
        reflog_grace_ms,
        max_deletes_per_call,
        dry_run,
    })
}

/// Render a [`GcReport`] as a bridge JSON value with stable field names.
fn gc_report_to_value(report: &GcReport) -> serde_json::Value {
    serde_json::json!({
        "reachable_checkpoints": report.reachable_checkpoints,
        "deleted_checkpoints": report.deleted_checkpoints,
        "deleted_blobs": report.deleted_blobs,
        "truncated": report.truncated,
    })
}

/// Render a [`PruneReport`][crate::session_refs::PruneReport] as a bridge JSON value.
///
/// Fail-safe defaults (see `reflog.prune` arm): absent or null numeric fields
/// default to `0`, so a bare `{"ref_name": ...}` call prunes nothing
/// (`older_than_ms = 0` matches no validated row; `max_rows = 0` deletes
/// nothing). Present-but-wrong-typed fields are errors naming the field, never
/// silent defaults.
fn prune_report_to_value(report: &crate::session_refs::PruneReport) -> serde_json::Value {
    serde_json::json!({
        "pruned": report.pruned,
        "floor_kept": report.floor_kept,
        "tombstone_survived": report.tombstone_survived,
    })
}

/// Require a string field: missing or null is an error, and a
/// present-but-wrong-typed value is an error (never a silent default).
fn req_string(payload: &serde_json::Value, field: &str) -> Result<String, String> {
    match payload.get(field) {
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        None | Some(serde_json::Value::Null) => Err(format!("missing '{field}' field")),
        Some(_) => Err(format!("invalid '{field}' field, expected string")),
    }
}

/// Optional string field: absent or null takes `default`, wrong-typed errors.
fn opt_string(payload: &serde_json::Value, field: &str, default: &str) -> Result<String, String> {
    match payload.get(field) {
        None | Some(serde_json::Value::Null) => Ok(default.to_owned()),
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(format!("invalid '{field}' field, expected string")),
    }
}

/// Require a u64 field: missing, null, or wrong-typed is an error.
fn req_u64(payload: &serde_json::Value, field: &str) -> Result<u64, String> {
    match payload.get(field) {
        None | Some(serde_json::Value::Null) => Err(format!("missing '{field}' field")),
        Some(v) => v
            .as_u64()
            .ok_or_else(|| format!("invalid '{field}' field, expected u64")),
    }
}

/// Optional u64 field: absent or null takes `default`, wrong-typed errors.
fn opt_u64(payload: &serde_json::Value, field: &str, default: u64) -> Result<u64, String> {
    match payload.get(field) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .ok_or_else(|| format!("invalid '{field}' field, expected u64")),
    }
}

/// Optional boolean field: absent or null takes `default`, wrong-typed errors.
fn opt_bool(payload: &serde_json::Value, field: &str, default: bool) -> Result<bool, String> {
    match payload.get(field) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("invalid '{field}' field, expected boolean")),
    }
}

/// Require a content-hash field as lowercase hex: missing, wrong-typed, or
/// malformed hex is an error.
fn req_hash(payload: &serde_json::Value, field: &str) -> Result<ContentHash, String> {
    let raw = req_string(payload, field)?;
    ContentHash::from_hex(&raw).map_err(|e| e.to_string())
}

/// Render a [`SessionBinding`] as a bridge JSON value with stable field names.
fn session_binding_to_value(binding: &SessionBinding) -> serde_json::Value {
    serde_json::json!({
        "session_id": binding.session_id.as_str(),
        "branch": binding.branch,
        "head": binding.head.to_hex(),
        "generation": binding.generation,
        "epoch": binding.epoch,
        "updated_at_ms": binding.updated_at_ms,
    })
}

/// Render a [`ResumeReport`] as a bridge JSON value with stable field names.
///
/// `session_id` is null on the unfenced branch/`HEAD` path; `fence_token` is
/// `0` there (no fence admitted). `pending_unknowns` is always empty with
/// `pending_log_absent = true` (no durable pending log exists).
fn resume_report_to_value(report: &ResumeReport) -> serde_json::Value {
    serde_json::json!({
        "session_id": report.session_id.as_ref().map(|id| id.as_str()),
        "branch": report.branch,
        "checkpoint": report.checkpoint.to_hex(),
        "generation": report.generation,
        "pending_unknowns": report.pending_unknowns,
        "pending_log_absent": report.pending_log_absent,
        "fence_token": report.fence_token,
    })
}

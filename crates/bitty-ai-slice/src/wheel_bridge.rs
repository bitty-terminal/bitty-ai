//! Boundary plane bridge providing a zero-unsafe, message-based dispatch interface (AI-0167).
//!
//! Encapsulates [`WheelKernel`] and exposes standard JSON-RPC command dispatch
//! for Lua plugins and host integration without raw pointer dereferencing or memory unsafety.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::action_protocol::SpilloverConfig;
use crate::content_store::{ContentHash, Rationale};
use crate::context_compiler::CompilerBudgetConfig;
use crate::facade::FacadeError;
use crate::task_dag::{TaskDraft, TaskId};
use crate::wheel_kernel::WheelKernel;

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
            unknown => Err(format!("unknown bridge command: '{unknown}'")),
        }
    }
}

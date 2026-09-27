-- Wheel Kernel Lua Client (AI-0167)
-- Provides an ergonomic Lua wrapper around WheelBridge JSON-RPC dispatch.
--
-- Can be driven by in-process dispatcher functions or over Bitty IPC wire methods.

local json = require("cjson")

local WheelKernel = {}
WheelKernel.__index = WheelKernel

--- Create a new WheelKernel Lua client instance.
--- @param dispatcher fun(command: string, payload_json: string): string
--- @return table
function WheelKernel.new(dispatcher)
    local self = setmetatable({}, WheelKernel)
    self.dispatcher = dispatcher
    return self
end

--- Internal helper executing a command over the dispatcher and decoding JSON response.
--- @param command string
--- @param payload table?
--- @return any
function WheelKernel:call(command, payload)
    local payload_str = json.encode(payload or {})
    local resp_str = self.dispatcher(command, payload_str)
    local resp = json.decode(resp_str)
    if not resp.success then
        error(resp.error or "unknown wheel bridge error")
    end
    return resp.data
end

--- Query kernel status and telemetry.
--- @return table
function WheelKernel:status()
    return self:call("kernel.status", {})
end

--- Create a new task in the control plane DAG.
--- @param draft table { id: string, title: string, description: string, priority: number, dependencies: string[]?, now_ms: number? }
--- @return table
function WheelKernel:create_task(draft)
    return self:call("task.create", draft)
end

--- Retrieve a task by its identifier.
--- @param id string
--- @return table
function WheelKernel:get_task(id)
    return self:call("task.get", { id = id })
end

--- List all tasks currently managed in the DAG.
--- @return table[]
function WheelKernel:list_tasks()
    return self:call("task.list", {})
end

--- Set or clear the active task driving Zone 2 context compilation.
--- @param id string?
--- @return table
function WheelKernel:set_active_task(id)
    return self:call("task.set_active", { id = id })
end

--- Start a task, binding a worker identity.
--- @param id string
--- @param worker_id string?
--- @param now_ms number?
--- @return table
function WheelKernel:start_task(id, worker_id, now_ms)
    return self:call("task.start", {
        id = id,
        worker_id = worker_id or "worker",
        now_ms = now_ms or 0,
    })
end

--- Complete a task successfully, promoting dependent tasks to Ready.
--- @param id string
--- @param expected_generation number
--- @param checkpoint string? Optional ContentHash string
--- @param now_ms number?
--- @return table
function WheelKernel:complete_task(id, expected_generation, checkpoint, now_ms)
    return self:call("task.complete", {
        id = id,
        expected_generation = expected_generation,
        checkpoint = checkpoint,
        now_ms = now_ms or 0,
    })
end

--- Mark a task failed, propagating Blocked across downstream dependents.
--- @param id string
--- @param expected_generation number
--- @param error_message string
--- @param now_ms number?
--- @return table
function WheelKernel:fail_task(id, expected_generation, error_message, now_ms)
    return self:call("task.fail", {
        id = id,
        expected_generation = expected_generation,
        error = error_message,
        now_ms = now_ms or 0,
    })
end

--- Cancel a task, propagating Blocked across downstream dependents.
--- @param id string
--- @param now_ms number?
--- @return table
function WheelKernel:cancel_task(id, now_ms)
    return self:call("task.cancel", {
        id = id,
        now_ms = now_ms or 0,
    })
end

--- Retry a failed or cancelled task.
--- @param id string
--- @param now_ms number?
--- @return table
function WheelKernel:retry_task(id, now_ms)
    return self:call("task.retry", {
        id = id,
        now_ms = now_ms or 0,
    })
end

--- Compute topological ordering of tasks.
--- @return string[]
function WheelKernel:topological_sort()
    return self:call("task.topological_sort", {})
end

--- Put a slot in the Merkle context tree and content store.
--- @param name string
--- @param content string
--- @param now_ms number?
--- @return table { name: string, hash: string, size_bytes: number }
function WheelKernel:put_slot(name, content, now_ms)
    return self:call("slot.put", {
        name = name,
        content = content,
        now_ms = now_ms or 0,
    })
end

--- Retrieve a slot from the Merkle context tree.
--- @param name string
--- @return table { name: string, found: boolean, content: string?, size_bytes: number? }
function WheelKernel:get_slot(name)
    return self:call("slot.get", { name = name })
end

--- Remove a slot from the active Merkle context tree.
--- @param name string
--- @return table { name: string, removed: boolean }
function WheelKernel:remove_slot(name)
    return self:call("slot.remove", { name = name })
end

--- List all slots in the active Merkle context tree.
--- @return table[]
function WheelKernel:list_slots()
    return self:call("slot.list", {})
end

--- Commit a cognitive checkpoint.
--- @param rationale table { why: string, what: string, where_focus: string?, how: string?, expected: string?, observed: string? }
--- @param branch string? Optional ref name
--- @param now_ms number?
--- @return table
function WheelKernel:commit_checkpoint(rationale, branch, now_ms)
    return self:call("checkpoint.commit", {
        rationale = rationale,
        branch = branch,
        now_ms = now_ms or 0,
    })
end

--- Retrieve a checkpoint by hash.
--- @param hash string
--- @return table
function WheelKernel:get_checkpoint(hash)
    return self:call("checkpoint.get", { hash = hash })
end

--- Retrieve backward checkpoint history log.
--- @param max_depth number?
--- @return table[]
function WheelKernel:log(max_depth)
    return self:call("checkpoint.log", { max_depth = max_depth or 16 })
end

--- Record an action outcome with auto-spillover.
--- @param action table { action_id: string, success: boolean?, exit_code: number?, duration_ms: number?, raw_stdout: string?, raw_stderr: string?, timestamp_ms: number? }
--- @return table
function WheelKernel:record_action(action)
    return self:call("action.record", action)
end

--- List recent action outcomes.
--- @return table[]
function WheelKernel:recent_actions()
    return self:call("action.recent", {})
end

--- Clear recent action outcomes.
function WheelKernel:clear_recent_actions()
    return self:call("action.clear", {})
end

--- Compile three-zone context prompt under multi-tier budget.
--- @param opts table { system_instruction: string?, project_rules: string[]?, tool_schemas: string[]?, turn_prompt: string? }
--- @return table { zone1_prefix: string, zone2_state: string, zone3_tail: string, prefix_hash: string, total_bytes: number, prompt_string: string, ... }
function WheelKernel:compile_context(opts)
    return self:call("context.compile", opts or {})
end

return WheelKernel

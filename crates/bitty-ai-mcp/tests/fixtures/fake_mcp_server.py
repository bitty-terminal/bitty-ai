#!/usr/bin/env python3
"""Fake MCP stdio server fixture (AI-0179, hermetic offline tests only).

Speaks just enough newline-delimited JSON-RPC for the client gates:
initialize (pinned protocol version, tools capability), tools/list
(single page: echo plus fail_now), tools/call (fixed text; fail_now
answers isError), ping, and roots/list. Unknown methods answer -32601.
Server notifications (no id) get no reply.

Bounded: serves at most 64 request lines, then exits. Exits on EOF.
Run with `python3 -u` (unbuffered): every reply is flushed immediately,
so no stdio buffering differences between shells/hosts can stall the
client handshake (POSIX sh printf may block-buffer pipes, e.g. dash).
"""

import json
import sys

MAX_LINES = 64


def main() -> int:
    count = 0
    for line in sys.stdin:
        if count >= MAX_LINES:
            break
        count += 1
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        if "id" not in msg:
            continue
        ident = msg["id"]
        method = msg.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.0.1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back input",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "fail_now",
                        "description": "Always fails",
                        "inputSchema": {"type": "object"},
                    },
                ]
            }
        elif method == "tools/call":
            params = msg.get("params") or {}
            if params.get("name") == "fail_now":
                result = {
                    "content": [{"type": "text", "text": "kaput"}],
                    "isError": True,
                }
            else:
                result = {
                    "content": [{"type": "text", "text": "fake-echo-ok"}],
                }
        elif method == "ping":
            result = {}
        elif method == "roots/list":
            result = {"roots": [{"uri": "file:///tmp/bitty"}]}
        elif isinstance(method, str):
            reply = {
                "jsonrpc": "2.0",
                "id": ident,
                "error": {"code": -32601, "message": "Method not found"},
            }
            print(json.dumps(reply), flush=True)
            continue
        else:
            # Response-shaped input (no method): no reply.
            continue
        reply = {"jsonrpc": "2.0", "id": ident, "result": result}
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/sh
# Fake MCP stdio server fixture (AI-0179, hermetic offline tests only).
#
# Speaks just enough newline-delimited JSON-RPC for the client gates:
# initialize (pinned protocol version, tools capability), tools/list
# (single page: echo plus fail_now), tools/call (fixed text; fail_now
# answers isError), ping, and roots/list. Unknown methods answer -32601.
# Server notifications (no id) get no reply.
#
# Bounded: serves at most 64 request lines, then exits. Exits on EOF.
# Reads with `IFS= read -r` so backslashes survive. POSIX sh only.

count=0
while [ "$count" -lt 64 ] && IFS= read -r line; do
	count=$((count + 1))

	# Notifications carry no "id" key: no reply, per JSON-RPC.
	case "$line" in
	*'"id"'*) ;;
	*) continue ;;
	esac

	# Extract the raw id token after the first top-level "id".
	rest=${line#*'"id"'}
	case "$rest" in
	*:*)
		rest=${rest#*:}
		;;
	*)
		continue
		;;
	esac
	# Trim leading blanks.
	while :; do
		case "$rest" in
		' '* | '	'*)
			rest=${rest#?}
			;;
		*)
			break
			;;
		esac
	done
	idquoted=0
	case "$rest" in
	'"'*)
		rest=${rest#'"'}
		id=${rest%%'"'*}
		idquoted=1
		;;
	*)
		id=${rest%%[^0-9-]*}
		;;
	esac
	if [ -z "$id" ]; then
		# Notification (no id): no reply, per JSON-RPC.
		continue
	fi
	if [ "$idquoted" = "1" ]; then
		idjson="\"$id\""
	else
		idjson="$id"
	fi

	case "$line" in
	*'"method":"initialize"'*)
		printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"fake-mcp","version":"0.0.1"}}}'
		;;
	*'"method":"tools/list"'*)
		printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"result":{"tools":[{"name":"echo","description":"Echo back input","inputSchema":{"type":"object"}},{"name":"fail_now","description":"Always fails","inputSchema":{"type":"object"}}]}}'
		;;
	*'"method":"tools/call"'*)
		case "$line" in
		*'fail_now'*)
			printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"result":{"content":[{"type":"text","text":"kaput"}],"isError":true}}'
			;;
		*)
			printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"result":{"content":[{"type":"text","text":"fake-echo-ok"}]}}'
			;;
		esac
		;;
	*'"method":"ping"'*)
		printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"result":{}}'
		;;
	*'"method":"roots/list"'*)
		printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"result":{"roots":[{"uri":"file:///tmp/bitty"}]}}'
		;;
	*'"method"'*)
		printf '%s\n' '{"jsonrpc":"2.0","id":'"$idjson"',"error":{"code":-32601,"message":"Method not found"}}'
		;;
	*)
		# Response-shaped input (no method): no reply.
		continue
		;;
	esac
done
exit 0

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENDERER="$SCRIPT_DIR/codex-render.jq"

if [ ! -f "$RENDERER" ]; then
  echo "FAIL: renderer is missing: $RENDERER" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo 'FAIL: jq is required to test the renderer' >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
EVENTS="$TMP_DIR/events.jsonl"
OUTPUT="$TMP_DIR/output.txt"

cleanup() {
  rm -f -- "$EVENTS" "$OUTPUT"
  rmdir -- "$TMP_DIR"
}
trap cleanup EXIT

cat > "$EVENTS" <<'JSONL'
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"agent_message","text":"I'll inspect the code.\nThen I'll fix it."}}
{"type":"item.started","item":{"type":"command_execution","command":"pwsh.exe -Command 'cmake --build build'","status":"in_progress"}}
{"type":"item.completed","item":{"type":"command_execution","command":"pwsh.exe -Command 'cmake --build build'","status":"failed","exit_code":1,"aggregated_output":"very noisy raw failure"}}
{"type":"item.completed","item":{"type":"file_change","status":"completed","changes":[{"path":"C:\\repo\\src\\traceweave_ingest.cpp","kind":"update"}]}}
{"type":"turn.completed","usage":{"input_tokens":12500,"cached_input_tokens":8000,"output_tokens":2100,"reasoning_output_tokens":700}}
JSONL

jq --unbuffered -rj \
  --arg color 0 \
  --arg repo_root 'C:\repo' \
  -f "$RENDERER" "$EVENTS" > "$OUTPUT"

assert_contains() {
  local expected=$1
  if ! grep -Fq "$expected" "$OUTPUT"; then
    echo "FAIL: expected rendered output to contain: $expected" >&2
    sed -n '1,120p' "$OUTPUT" >&2
    exit 1
  fi
}

assert_contains '| START Codex turn'
assert_contains "I'll inspect the code."
assert_contains '| RUN   cmake --build build'
assert_contains '| FAIL  cmake --build build (exit 1; details in diagnostics log)'
assert_contains '| EDIT  src/traceweave_ingest.cpp'
assert_contains '| DONE  turn complete | input 12.5k | cached 8k | output 2.1k | reasoning 700'

if grep -Eq '^\{' "$OUTPUT"; then
  echo 'FAIL: raw JSON leaked into rendered output' >&2
  exit 1
fi

if grep -Fq 'very noisy raw failure' "$OUTPUT"; then
  echo 'FAIL: raw command output leaked into rendered output' >&2
  exit 1
fi

if grep -Fq 'ΓÇÖ' "$OUTPUT"; then
  echo 'FAIL: mojibake leaked into rendered output' >&2
  exit 1
fi

if LC_ALL=C grep -q $'\r\r' "$OUTPUT"; then
  echo 'FAIL: doubled carriage returns leaked into rendered output' >&2
  exit 1
fi

PIPE_ENDING=$(
  printf '%s\n' '{"type":"turn.started"}' \
    | jq --unbuffered -rj \
        --arg color 0 \
        --arg repo_root 'C:\repo' \
        -f "$RENDERER" \
    | tail -c 3 \
    | od -An -tx1 \
    | tr -d ' \n'
)
if [ "$PIPE_ENDING" = "0d0d0a" ]; then
  echo 'FAIL: terminal pipeline emits CRCRLF instead of CRLF' >&2
  exit 1
fi

echo 'PASS: Codex JSONL renders as compact terminal output'

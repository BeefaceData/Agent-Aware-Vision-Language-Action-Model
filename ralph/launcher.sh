#!/usr/bin/env bash
# Internal helper, not a fourth AFK launcher.
ralph_launch() {
  local backend="$1"
  shift
  local python_bin="${RALPH_PYTHON:-python3}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    python_bin=python
  fi
  command -v "$python_bin" >/dev/null 2>&1 || {
    echo "Ralph requires Python 3.11 or newer." >&2; return 1;
  }
  exec "$python_bin" "$SCRIPT_DIR/runner.py" --backend "$backend" "$@"
}

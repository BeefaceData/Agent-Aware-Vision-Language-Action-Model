#!/usr/bin/env bash
# Internal helper, not a fourth AFK launcher.
ralph_launch() {
  local backend="$1"
  shift
  local python_bin="${RALPH_PYTHON:-}"
  if [[ -n "$python_bin" ]]; then
    "$python_bin" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1 || {
      echo "RALPH_PYTHON must name a working Python 3.11+ interpreter." >&2; return 1;
    }
  else
    local candidate
    for candidate in python3 python; do
      if "$candidate" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1; then
        python_bin="$candidate"; break
      fi
    done
    [[ -n "$python_bin" ]] || { echo "Ralph requires Python 3.11+." >&2; return 1; }
  fi
  exec "$python_bin" "$SCRIPT_DIR/runner.py" --backend "$backend" "$@"
}

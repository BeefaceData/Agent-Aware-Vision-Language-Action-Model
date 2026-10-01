#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
source "$RALPH_DIR/providers.sh"
source "$RALPH_DIR/checks.sh"
source "$RALPH_DIR/queue.sh"
source "$RALPH_DIR/publish.sh"
ralph_main codex "$@"

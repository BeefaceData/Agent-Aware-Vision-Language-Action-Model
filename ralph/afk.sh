#!/bin/bash
set -eo pipefail

# Neotix Ralph — autonomous AFK loop.
#
# Driven by an ISSUE TARGET, not an iteration count: the loop keeps running
# agent iterations until <issue-target> issues have been completed in this
# run, then stops. It also stops early when the agent reports no
# remaining agent-ready tasks, hits a blocker, or stalls past a safety cap.

# --- COLORS & UI ---
R='\033[0;31m'  # Red
G='\033[0;32m'  # Green
Y='\033[0;33m'  # Yellow
B='\033[0;34m'  # Blue
C='\033[0;36m'  # Cyan
NC='\033[0m'    # No Color

if [ -z "$1" ]; then
  echo -e "${R}Usage: $0 <issue-target>${NC}"
  echo -e "  e.g. ${C}$0 6${NC}   run until 6 issues are completed"
  echo -e "       ${C}$0 10${NC}  hike day"
  exit 1
fi
TARGET=$1

# Always run from the repo root, regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p .agent/history .agent/logs

START_TIME=$(date +%s)
ISSUES_DONE=0
COMPLETED_ISSUES=" "
# Never spin forever if the agent stops completing issues without signalling.
SAFETY_CAP=$(( TARGET * 3 + 3 ))

# On an unexpected failure we deliberately do NOT retry mid-run. Instead we
# leave a breadcrumb so the next run understands what broke and continues.
FAILURE_FILE=".agent/last-failure.md"
log_failure() {
  local ec=$?
  trap - ERR
  {
    echo "# Last run failed"
    echo
    echo "- when: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- iteration: ${iter:-0}"
    echo "- exit code: ${ec}"
    echo "- issues completed before failure: ${ISSUES_DONE}"
    if [ -n "${HISTORY_FILE:-}" ] && [ -f "${HISTORY_FILE}" ]; then
      echo "- iteration log: ${HISTORY_FILE}"
      echo
      echo "## Tail of the failed iteration"
      echo '```'
      tail -n 40 "${HISTORY_FILE}"
      echo '```'
    fi
  } > "${FAILURE_FILE}" 2>/dev/null || true
  echo -e "\n${R}✖ Iteration ${iter:-?} failed (exit ${ec}). Logged to ${FAILURE_FILE} for the next run to resume from.${NC}"
}
trap 'log_failure' ERR

echo -e "${B} Neotix Ralph (local mode) — target ${Y}${TARGET}${B} issues in ${REPO_ROOT}${NC}"

# jq filter for streaming the assistant's text out of the Codex JSONL events.
stream_text='select(.type == "item.completed" and .item.type == "agent_message") | .item.text // empty | gsub("\n"; "\r\n") | . + "\r\n\n"'

iter=0
while :; do
  # --- Stop condition: issue target reached ---
  if [ "$ISSUES_DONE" -ge "$TARGET" ]; then
    echo -e "\n${G} Issue target reached: ${ISSUES_DONE}/${TARGET}.${NC}"
    break
  fi

  iter=$((iter + 1))
  if [ "$iter" -gt "$SAFETY_CAP" ]; then
    echo -e "\n${R}⚠️  Safety cap reached (${SAFETY_CAP} iterations) with only ${ISSUES_DONE}/${TARGET} issues. Stopping so the loop doesn't spin.${NC}"
    exit 1
  fi

  SESSION_ID=$(date +%Y%m%d-%H%M%S)
  HISTORY_FILE=".agent/history/iteration-${iter}-${SESSION_ID}.log"

  echo -e "\n${B}░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░${NC}"
  echo -e "  ${Y}Iteration $iter${NC} | ${C}issues ${ISSUES_DONE}/${TARGET}${NC} | ${C}$HISTORY_FILE${NC}"
  echo -e "${B}░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░${NC}\n"

  # 1. Gather context
  echo -ne "${Y}  Fetching GitHub Issues...${NC}"
  issues=$(gh issue list --state open --json number,title,body,comments,labels)
  echo -e " ${G}Done.${NC}"

  echo -ne "${Y}  Reading Git History...${NC}"
  commits=$(git log -n 8 --format="%H%n%ad%n%B---" --date=short 2>/dev/null || echo "No commits found")
  echo -e " ${G}Done.${NC}"

  prompt=$(cat ralph/prompt.md)

  # Surface any failure from a previous run so this run understands what broke
  # and continues from there.
  prior_failure=""
  [ -f "$FAILURE_FILE" ] && prior_failure=$(cat "$FAILURE_FILE")

  # Write bulky context to a file (Windows CreateProcess caps args at ~32KB).
  CONTEXT_FILE=".agent/ralph-context.md"
  {
    printf '## Previous commits\n\n%s\n\n' "$commits"
    printf '## GitHub Issues (JSON)\n\n%s\n\n' "$issues"
    if [ -n "$prior_failure" ]; then
      printf '## A previous run failed — understand what broke, then continue\n\n%s\n\n' "$prior_failure"
    fi
    printf '## Instructions\n\n%s\n' "$prompt"
  } > "$CONTEXT_FILE"

  # 2. Run Codex locally — no sandbox, no permission prompts.
  # The prompt is passed as an arg; stdin is closed for the non-interactive run.
  codex exec --json \
    --dangerously-bypass-approvals-and-sandbox \
    "Read .agent/ralph-context.md for previous commits, open GitHub issues, and your full instructions. Then follow those instructions." \
    < /dev/null \
  | grep --line-buffered '^{' \
  | tee "$HISTORY_FILE" \
  | jq --unbuffered -rj "$stream_text"

  # 3. Inspect the result
  ITER_END=$(date +%s)

  # Extract ONLY Codex's own emitted assistant text — NOT tool output,
  # so the sentinel strings inside ralph-context.md don't trigger false exits.
  ASSISTANT_TEXT=$(jq -r "$stream_text" < "$HISTORY_FILE" 2>/dev/null || echo "")

  # Count each issue once, using only the assistant's completion signal.
  NEW=0
  if [[ "$ASSISTANT_TEXT" =~ \<promise\>ISSUE\ COMPLETED\ \#([0-9]+)\</promise\> ]]; then
    ISSUE_NUMBER=${BASH_REMATCH[1]}
    case "$COMPLETED_ISSUES" in
      *" $ISSUE_NUMBER "*) ;;
      *)
        COMPLETED_ISSUES+="$ISSUE_NUMBER "
        ISSUES_DONE=$((ISSUES_DONE + 1))
        NEW=1
        ;;
    esac
  fi

  # No more agent-ready tasks → done.
  if echo "$ASSISTANT_TEXT" | grep -q "<promise>NO MORE TASKS</promise>"; then
    rm -f "$FAILURE_FILE"
    echo -e "\n${G} No agent-ready tasks remain. ${ISSUES_DONE} issues completed this run.${NC}"
    exit 0
  fi

  # Blocked → stop for human.
  if echo "$ASSISTANT_TEXT" | grep -q "\[BLOCKED\]"; then
    echo -e "\n${R}⚠️  RALPH IS BLOCKED:${NC}"
    echo "$ASSISTANT_TEXT" | grep "\[BLOCKED\]" | tail -n 3
    exit 1
  fi

  echo -e "\n${G}└── ✓ Iteration $iter done: +${NEW} issue(s) (${ISSUES_DONE}/${TARGET}) ($((ITER_END - START_TIME))s elapsed)${NC}"
done

rm -f "$FAILURE_FILE"
echo -e "\n${G}✅ Run complete: ${ISSUES_DONE} issues completed in $(( $(date +%s) - START_TIME ))s.${NC}"

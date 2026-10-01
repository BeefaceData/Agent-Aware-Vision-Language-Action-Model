#!/usr/bin/env bash
# Shared functions, not an entry point. Inspired by ralph-inspiration/ralph.
RALPH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '%s\n' "$*"; }
die() { say "STOP: $*" >&2; exit 1; }
jqtext() { jq -r "$@" | tr -d '\r'; }
native_path() { if command -v cygpath >/dev/null; then cygpath -am "$1"; else printf '%s\n' "$1"; fi; }

# Every network command has a visible name, diagnostic file and finite timeout.
gh_call() {
  printf '[gh] %s\n' "$*" >&2
  timeout --kill-after=10s 90s gh "$@" 2> >(tee -a "$RUN/diagnostics.log" >&2)
}
git_network() {
  say "[git] $* (maximum 120 seconds)"
  timeout --kill-after=10s 120s git "$@" 2>&1 | tee -a "$RUN/diagnostics.log"
}
state_update() {
  jq "$@" "$STATE" > "$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}
active_update() {
  jq "$@" "$ACTIVE" > "$ACTIVE.tmp"
  mv "$ACTIVE.tmp" "$ACTIVE"
}
check_deadline() {
  local now; now=$(date +%s)
  (( now < $(jqtext '.deadline | floor' "$STATE") )) || die 'Queue deadline reached; see saved state.'
  if [[ -f ${ACTIVE:-/nonexistent} ]]; then
    (( now < $(jqtext '.deadline' "$ACTIVE") )) || die 'Issue deadline reached; work and evidence preserved.'
  fi
}
cleanup() {
  local result=$?
  trap - EXIT
  if [[ -n ${CHECK_CONTAINER:-} ]]; then docker rm -f "$CHECK_CONTAINER" >/dev/null 2>&1 || true; fi
  if [[ -n ${LOCK_OWNED:-} ]]; then rm -f "$RUN/lock/pid"; rmdir "$RUN/lock" || true; fi
  if (( result != 0 )); then say "Stopped. Logs: $RUN/diagnostics.log; issue evidence: $RUN/issues" >&2; fi
  exit "$result"
}
lock_run() {
  if ! mkdir "$RUN/lock" 2>/dev/null; then
    local holder; holder=$(cat "$RUN/lock/pid" 2>/dev/null || true)
    if [[ $holder =~ ^[0-9]+$ ]] && ! kill -0 "$holder" 2>/dev/null; then
      rm -f "$RUN/lock/pid"; rmdir "$RUN/lock"
      mkdir "$RUN/lock" || die 'Another loop acquired the lock.'
    else die "Another loop owns $RUN/lock (PID ${holder:-unknown})."; fi
  fi
  printf '%s\n' "$$" > "$RUN/lock/pid"
  LOCK_OWNED=1
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

load_state() {
  local fingerprint legacy active
  fingerprint=$(git -C "$ROOT" hash-object "$QUEUE")
  if [[ -f $STATE ]]; then
    jq -e --arg backend "$BACKEND" --argjson target "$TARGET" --arg queue "$fingerprint" \
      '.backend == $backend and .target == $target and .queue == $queue' "$STATE" >/dev/null ||
      die 'Saved queue options differ. Keep the original backend, target and queue.'
    say "Resuming: $(jqtext '.completed | length' "$STATE")/$TARGET completed."
    return
  fi
  legacy="$ROOT/.ralph/queues/simulation-supervisor-v1/state.json"
  if [[ -f $legacy ]]; then
    active=$(jqtext '.active // empty' "$legacy")
    [[ -z $active || ! -f $ROOT/.ralph/runs/issue-$active/state.json ]] ||
      die 'Legacy worker evidence exists; reconcile it before migrating the queue.'
    [[ $BACKEND == codex && $TARGET == 50 ]] || die 'Legacy queue was Codex / 50 issues; use those options.'
    jq --arg backend "$BACKEND" --arg queue "$fingerprint" --argjson target "$TARGET" \
      '{version:1,backend:$backend,target:$target,queue:$queue,deadline:(.deadline|floor),completed:.completed,active:.active}' \
      "$legacy" > "$STATE.tmp"
    say 'Imported the saved queue and its original deadline; no progress reset.'
  else
    jq -n --arg backend "$BACKEND" --arg queue "$fingerprint" --argjson target "$TARGET" \
      --argjson deadline "$(( $(date +%s) + 96*3600 ))" \
      '{version:1,backend:$backend,target:$target,queue:$queue,deadline:$deadline,completed:[],active:null}' > "$STATE.tmp"
  fi
  mv "$STATE.tmp" "$STATE"
}

remote_base() {
  gh_call api "repos/$REPO/git/ref/heads/$BASE_BRANCH" --jq .object.sha | tr -d '\r'
}
ensure_commit() {
  local sha=$1
  if git -C "$ROOT" cat-file -e "$sha^{commit}" 2>/dev/null; then
    say "Base $sha is already local; no fetch needed."
  else
    git_network -C "$ROOT" fetch --no-tags origin "$BASE_BRANCH"
    git -C "$ROOT" cat-file -e "$sha^{commit}" || die "Fetch did not provide $sha."
  fi
}
read_issue() {
  gh_call issue view "$1" --repo "$REPO" --json number,title,body,state,labels,assignees > "$2"
}
eligible_issue() {
  jq -e --arg actor "$ACTOR" '
    .state == "OPEN" and
    ([.labels[].name] | index("ready-for-agent") != null and index("capstone-v1") != null
      and all(.[]; . != "needs-info" and . != "needs-triage" and . != "ready-for-human" and . != "wontfix")) and
    all(.assignees[]; .login == $actor)' "$1" >/dev/null
}
dependencies_closed() {
  local issue=$1 file=$2 dep status
  gh_call api --paginate --slurp "repos/$REPO/issues/$issue/dependencies/blocked_by?per_page=100" > "$RUN/deps.json" || die 'Cannot read native dependencies.'
  jq -e 'type == "array" and all(.[]; type == "array")' "$RUN/deps.json" >/dev/null || die 'Malformed dependency response.'
  jq -e 'all(.[][]; .state == "closed")' "$RUN/deps.json" >/dev/null || return 1
  # Only references in the explicit Blocked by section are dependencies.
  jqtext '.body | gsub("\r"; "") | split("## Blocked by\n")[1] // "" | split("\n## ")[0] |
    [scan("#([0-9]+)") | .[0]] | unique[]' "$file" > "$RUN/dependency-numbers"
  while IFS= read -r dep; do
    [[ -n $dep ]] || continue
    status=$(gh_call issue view "$dep" --repo "$REPO" --json state --jq .state) || die "Cannot read dependency #$dep."
    [[ $status == CLOSED ]] || return 1
  done < "$RUN/dependency-numbers"
}
select_issue() {
  local n
  for n in $(jqtext '.issues[]' "$QUEUE"); do
    jq -e --argjson n "$n" '.completed | index($n) != null' "$STATE" >/dev/null && continue
    read_issue "$n" "$RUN/selection.json"
    eligible_issue "$RUN/selection.json" || continue
    if dependencies_closed "$n" "$RUN/selection.json"; then
      state_update --argjson n "$n" '.active=$n'
      return
    fi
  done
  die 'No eligible issues remain in this preset. Closed work by others is not counted as this run.'
}

ralph_main() {
  BACKEND=$1; shift
  TARGET=${1:-1}; [[ $# == 0 ]] || shift
  MODE=${1:-run}
  [[ $TARGET =~ ^[1-9][0-9]*$ && ${#TARGET} -le 2 && $TARGET -le 50 && $# -le 1 ]] ||
    die "Usage: $BACKEND-afk.sh <1..50> [--doctor]"
  [[ $MODE == run || $MODE == --doctor ]] || die 'Only --doctor is supported after the target.'
  local tool
  for tool in git gh jq timeout docker tar base64; do command -v "$tool" >/dev/null || die "Missing command: $tool"; done
  ROOT=$(git -C "$RALPH_DIR/.." rev-parse --show-toplevel)
  ROOT=$(cd "$ROOT" && pwd)
  RUN="$ROOT/.ralph/shell"; STATE="$RUN/state.json"
  POLICY="$RALPH_DIR/policy.json"; QUEUE="$RALPH_DIR/queues/simulation-supervisor.json"
  REPO=$(jqtext '.repository' "$POLICY"); BASE_BRANCH=$(jqtext '.base_branch' "$POLICY")
  mkdir -p "$RUN/issues"
  export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never GH_PROMPT_DISABLED=1
  cd "$ROOT"
  [[ $(gh_call repo view --json nameWithOwner --jq .nameWithOwner) == "$REPO" ]] || die 'Repository mismatch.'
  ACTOR=$(gh_call api user --jq .login)
  provider_preflight
  docker image inspect "$(jqtext '.validation.image' "$QUEUE")" >/dev/null || die 'Provision the pinned validation image first.'
  local base; base=$(remote_base); ensure_commit "$base"
  say "Neotix Ralph | $BACKEND | target $TARGET | reviewer gpt-6-astra / medium"
  say "Logs: $RUN | Client API use: disabled"
  if [[ $MODE == --doctor ]]; then say 'Preflight passed. No worker launched or GitHub state changed.'; return; fi
  lock_run
  load_state
  while (( $(jqtext '.completed | length' "$STATE") < TARGET )); do
    ACTIVE=''; check_deadline
    [[ $(jqtext '.active // empty' "$STATE") ]] || select_issue
    ISSUE=$(jqtext '.active' "$STATE")
    work_issue
    say "Completed $(jqtext '.completed | length' "$STATE")/$TARGET issues."
  done
  say "Target reached: $TARGET verified merged issues."
}

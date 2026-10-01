#!/usr/bin/env bash
set -Eeuo pipefail
TEST_DIR=$(cd "$(dirname "$0")" && pwd)
source "$TEST_DIR/../common.sh"
source "$RALPH_DIR/providers.sh"
source "$RALPH_DIR/checks.sh"
source "$RALPH_DIR/queue.sh"
source "$RALPH_DIR/publish.sh"

setup() {
  ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ralph-test.XXXXXXXX")
  ROOT=$(cd "$ROOT" && pwd)
  # Test fixtures are retained in the OS temp directory on failure for diagnosis.
  git init -q -b main "$ROOT"
  git -C "$ROOT" config user.name 'Ralph test'
  git -C "$ROOT" config user.email 'ralph@example.invalid'
  printf '.ralph/\n.ralph-run/\n' > "$ROOT/.gitignore"
  printf '# Test guidance\n' > "$ROOT/AGENTS.md"
  git -C "$ROOT" add .; git -C "$ROOT" commit -qm 'fixture'
  git init -q --bare "$ROOT/.ralph-origin.git"
  printf '\n.ralph-origin.git/\n' >> "$ROOT/.gitignore"
  git -C "$ROOT" add .gitignore; git -C "$ROOT" commit -qm 'ignore origin'
  git -C "$ROOT" remote add origin "$ROOT/.ralph-origin.git"
  git -C "$ROOT" push -q origin main
  RUN="$ROOT/.ralph/shell"; STATE="$RUN/state.json"
  mkdir -p "$RUN/issues"
  POLICY="$RALPH_DIR/policy.json"; QUEUE="$RUN/queue.json"
  jq '.issues=[2,3] | .required_checks=["fixture"]' "$RALPH_DIR/queues/simulation-supervisor.json" > "$QUEUE"
  BACKEND=codex; TARGET=2; ACTOR=tester; REPO=test/repo; BASE_BRANCH=main
  load_state
  jq -n '{number:1,body:"Fixture PRD"}' > "$RUN/prd.json"
  for number in 2 3; do
    jq -n --argjson n "$number" '{number:$n,title:"Fixture",body:"## Acceptance criteria\n\n- [ ] Public behavior works.\n\n## Blocked by\n\nNone",state:"OPEN",labels:[{name:"ready-for-agent"},{name:"capstone-v1"}],assignees:[]}' > "$RUN/issue-$number.json"
  done
}

# GitHub boundary double; Git trees, commits, worktrees and pushes remain real.
publish_branch() { git -C "$WORK" push -q origin "$HEAD:refs/heads/$BRANCH"; }
gh_call() {
  printf '%s\n' "$*" >> "$RUN/gh-calls"
  case "$1 $2" in
    'issue view')
      if [[ $3 == 1 ]]; then cat "$RUN/prd.json"; else
        if [[ ${MOCK_CLOSE_DELAY:-0} != 0 && $(jqtext '.state' "$RUN/issue-$3.json") == CLOSED ]]; then
          local count=0
          [[ ! -f $RUN/close-polls ]] || count=$(cat "$RUN/close-polls")
          printf '%s\n' "$((count+1))" > "$RUN/close-polls"
          if (( count < MOCK_CLOSE_DELAY )); then jq '.state="OPEN"' "$RUN/issue-$3.json"; return; fi
        fi
        cat "$RUN/issue-$3.json"
      fi ;;
    'issue edit') return ;;
    'api --paginate') printf '[[]]\n' ;;
    'api --method')
      if [[ $3 == POST ]]; then
        git --git-dir="$ROOT/.ralph-origin.git" update-ref "refs/heads/$BRANCH" "$BASE" ''
      else
        [[ $* == *'force=false'* ]] || return 1
        git --git-dir="$ROOT/.ralph-origin.git" update-ref refs/heads/main "$HEAD" "$BASE"
        jq '.state="CLOSED"' "$RUN/issue-$ISSUE.json" > "$RUN/closed.tmp"; mv "$RUN/closed.tmp" "$RUN/issue-$ISSUE.json"
      fi ;;
    'pr list')
      if [[ -f $EVIDENCE/pr-created ]]; then
        jq -n --arg head "$HEAD" '[{number:99,state:"OPEN",headRefOid:$head,baseRefName:"main"}]'
      else printf '[]\n'; fi ;;
    'pr create') touch "$EVIDENCE/pr-created" ;;
    'pr checks') printf '[{"name":"fixture","bucket":"pass","state":"SUCCESS"}]\n' ;;
    'pr view')
      if [[ $(git --git-dir="$ROOT/.ralph-origin.git" rev-parse main) == "$HEAD" ]]; then
        local count=0
        [[ ! -f $RUN/merge-polls ]] || count=$(cat "$RUN/merge-polls")
        printf '%s\n' "$((count+1))" > "$RUN/merge-polls"
        if (( count < ${MOCK_MERGE_DELAY:-0} )); then
          jq -n --arg head "$HEAD" '{state:"OPEN",headRefOid:$head,mergeCommit:null}'
        else
          jq -n --arg head "${MOCK_MERGED_HEAD:-$HEAD}" --arg commit "${MOCK_MERGE_COMMIT:-$HEAD}" \
            '{state:"MERGED",headRefOid:$head,mergeCommit:{oid:$commit}}'
        fi
      else
        jq -n --arg head "$HEAD" --arg base "${MOCK_BASE:-$BASE}" '{state:"OPEN",headRefOid:$head,baseRefOid:$base,mergeable:"MERGEABLE",mergeStateStatus:"CLEAN",isDraft:false}'
      fi ;;
    api*)
      case "$2" in
        */git/ref/heads/main) git --git-dir="$ROOT/.ralph-origin.git" rev-parse main ;;
        */git/ref/heads/ralph/*) git --git-dir="$ROOT/.ralph-origin.git" rev-parse "refs/heads/$BRANCH" ;;
        *) printf 'Unexpected gh call: %s\n' "$*" >&2; return 1 ;;
      esac ;;
    *) printf 'Unexpected gh call: %s\n' "$*" >&2; return 1 ;;
  esac
}
invoke_model() {
  if [[ $1 == worker ]]; then
    printf 'public behavior %s attempt %s\n' "$ISSUE" "$ATTEMPT" > "$WORK/module-$ISSUE.txt"
    jq -n --argjson issue "$ISSUE" '{issue:$issue,status:"ready_for_review",summary:"Public behavior",acceptance:[{criterion:1,evidence:"fixture"}],commit:{subject:"feat(harness): :sparkles: add fixture",why:"Exercise the shell lifecycle."},merge_danger:{door:"two-way",blast_radius:"fixture"},risks:[]}' > "$3"
  else
    jq -n --argjson issue "$ISSUE" --arg base "$BASE" --arg head "${MOCK_HEAD:-$HEAD}" --arg verdict "${MOCK_VERDICT:-pass}" \
      '{issue:$issue,base_sha:$base,head_sha:$head,verdict:$verdict,summary:"Verified fixture",acceptance:[{criterion:1,verdict:"pass",evidence:"fixture"}],findings:[]}' > "$3"
  fi
}
validate_candidate() {
  active_update '.checks += 1'
  printf 'Fixture validation for %s\n' "$1" > "$EVIDENCE/checks-$ATTEMPT.log"
}

test_lifecycle() {
  setup
  for ISSUE in 2 3; do
    state_update --argjson n "$ISSUE" '.active=$n'
    work_issue
  done
  jq -e '.completed == [2,3] and .active == null' "$STATE" >/dev/null
  [[ $(git --git-dir="$ROOT/.ralph-origin.git" rev-parse main) == "$HEAD" ]]
  [[ $(git -C "$ROOT" rev-parse main) != "$HEAD" ]] # user's checkout is untouched
  [[ $(git --git-dir="$ROOT/.ralph-origin.git" rev-list --count main) == 4 ]]
}
test_resume() {
  setup; ISSUE=2; state_update '.active=2'; prepare_issue
  worker_attempt; HEAD=$(jqtext '.head' "$ACTIVE")
  validate_candidate "$HEAD"; review_candidate
  local saved_deadline; saved_deadline=$(jqtext '.deadline' "$ACTIVE")
  load_state; work_issue
  [[ $(jqtext '.attempts' "$ACTIVE") == 1 ]]
  [[ $(jqtext '.deadline' "$ACTIVE") == "$saved_deadline" ]]
  jq -e '.completed == [2]' "$STATE" >/dev/null
}
test_check_repair() {
  setup; ISSUE=2; state_update '.active=2'
  validate_candidate() {
    active_update '.checks += 1'
    printf 'Check evidence for attempt %s\n' "$ATTEMPT" > "$EVIDENCE/checks-$ATTEMPT.log"
    [[ $ATTEMPT == 2 ]]
  }
  work_issue
  jq -e '.attempts == 2 and .reviews == 1 and .phase == "complete"' "$ACTIVE" >/dev/null
}
test_merge_propagation() {
  setup; ISSUE=2; state_update '.active=2'
  MOCK_MERGE_DELAY=1; MOCK_CLOSE_DELAY=1
  sleep() {
    jq -e '.completed == [] and .active == 2' "$STATE" >/dev/null
    jq -e '.phase == "merging"' "$ACTIVE" >/dev/null
    printf 'waiting\n' >> "$RUN/waits"
  }
  work_issue
  jq -e '.completed == [2] and .active == null' "$STATE" >/dev/null
  [[ $(wc -l < "$RUN/waits") == 2 ]]
  [[ $(grep -c 'api --method PATCH' "$RUN/gh-calls") == 1 ]]
  finish_merge 99
  jq -e '.completed == [2]' "$STATE" >/dev/null
  jq -e '.attempts == 1 and .checks == 1 and .reviews == 1' "$ACTIVE" >/dev/null
}
test_merge_never_confirms() {
  setup; ISSUE=2; state_update '.active=2'; MOCK_MERGE_DELAY=999
  sleep() { :; }
  work_issue
  die 'Unreachable: unconfirmed merge was counted.'
}
test_merge_resume() {
  setup; ISSUE=2; state_update '.active=2'; work_issue
  # Restore the journal state that would survive a stop after the remote merge.
  active_update '.phase="merging"'
  state_update '.completed=[] | .active=2'
  invoke_model() { die 'Resume reran a model.'; }
  publish_branch() { die 'Resume republished a branch.'; }
  merge_pr() { die 'Resume retried a merge.'; }
  work_issue
  jq -e '.completed == [2] and .active == null' "$STATE" >/dev/null
  [[ $(grep -c 'api --method PATCH' "$RUN/gh-calls") == 1 ]]
}
test_merged_head_changed() {
  setup; ISSUE=2; state_update '.active=2'; MOCK_MERGED_HEAD=other
  work_issue
  die 'Unreachable: changed PR head was accepted.'
}
test_wrong_merge_commit() {
  setup; ISSUE=2; state_update '.active=2'; MOCK_MERGE_COMMIT=other
  work_issue
  die 'Unreachable: wrong merged commit was accepted.'
}
test_stale_review() {
  setup; ISSUE=2; prepare_issue; worker_attempt; HEAD=$(jqtext '.head' "$ACTIVE")
  validate_candidate "$HEAD"; MOCK_HEAD=wrong; review_candidate
  die 'Unreachable: stale review accepted.'
}
test_main_moved() {
  setup; ISSUE=2; prepare_issue; worker_attempt; HEAD=$(jqtext '.head' "$ACTIVE")
  validate_candidate "$HEAD"; review_candidate; publish_pr
  MOCK_BASE=other; merge_pr
  die 'Unreachable: changed base accepted.'
}
test_revision_limit() {
  setup; ISSUE=2; state_update '.active=2'; MOCK_VERDICT=revise; work_issue
  die 'Unreachable: attempts were not bounded.'
}
test_protected_change() {
  setup; ISSUE=2; prepare_issue; worker_attempt
  printf 'bypass\n' >> "$WORK/AGENTS.md"
  ATTEMPT=2; cp "$EVIDENCE/worker-1.json" "$EVIDENCE/worker-2.json"
  make_candidate
  die 'Unreachable: protected file accepted.'
}
test_legacy_resume() {
  setup; rm "$STATE"; TARGET=50
  mkdir -p "$ROOT/.ralph/queues/simulation-supervisor-v1"
  printf '{"deadline":2000000000,"completed":[],"active":2}' > "$ROOT/.ralph/queues/simulation-supervisor-v1/state.json"
  load_state
  jq -e '.deadline == 2000000000 and .active == 2 and .completed == []' "$STATE" >/dev/null
  load_state
}
test_check_gate() {
  setup
  printf '[]' > "$RUN/checks.json"
  if checks_pass "$RUN/checks.json"; then die 'Absent checks accepted.'; fi
  printf '[{"name":"fixture","bucket":"pending"}]' > "$RUN/checks.json"
  if checks_pass "$RUN/checks.json"; then die 'Pending checks accepted.'; fi
  printf '[{"name":"fixture","bucket":"skipping"}]' > "$RUN/checks.json"
  if checks_pass "$RUN/checks.json"; then die 'Skipped required check accepted.'; fi
  printf '[{"name":"fixture","bucket":"pass"}]' > "$RUN/checks.json"
  checks_pass "$RUN/checks.json"
}
test_local_commit() {
  setup
  git_network() { die 'Unnecessary fetch attempted.'; }
  ensure_commit "$(git -C "$ROOT" rev-parse HEAD)"
}
test_keys() {
  export ANTHROPIC_API_KEY=fixture OPENAI_API_KEY=fixture GH_TOKEN=fixture
  without_keys bash -c '[[ -z ${ANTHROPIC_API_KEY:-} && -z ${OPENAI_API_KEY:-} && -z ${GH_TOKEN:-} ]]'
  [[ $ANTHROPIC_API_KEY == fixture ]] # controller environment is unchanged
}
test_dependency_failure() {
  setup
  gh_call() { return 7; }
  if dependencies_closed 2 "$RUN/issue-2.json"; then die 'Unreachable: dependency read failed open.'; fi
}
test_upload_mismatch() {
  setup; ISSUE=2; prepare_issue; worker_attempt; HEAD=$(jqtext '.head' "$ACTIVE")
  source "$RALPH_DIR/publish.sh"
  gh_call() { printf 'wrong-hash\n'; }
  publish_branch
  die 'Unreachable: unverified object was published.'
}
test_provider_command() {
  setup; ISSUE=2; prepare_issue
  # Load the real provider functions, then substitute only the executable.
  source "$RALPH_DIR/providers.sh"
  mkdir -p "$RUN/bin" "$WORK/.ralph-run"
  export TEST_CAPTURE_DIR="$RUN"
  export TEST_REQUIRE_WINDOWS_PERMISSIONS=1
  uname() { printf 'MINGW64_NT\n'; }
  cat > "$RUN/bin/codex" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ -z ${ANTHROPIC_API_KEY:-} && -z ${OPENAI_API_KEY:-} && -z ${GH_TOKEN:-} ]]
printf '%s\n' "$@" > "$TEST_CAPTURE_DIR/provider-args"
output=''; work=''; exec_seen=0; windows_sandbox=0; approval_never=0
while (( $# )); do
  case "$1" in
    exec) exec_seen=1 ;;
    -c)
      if (( exec_seen )); then
        [[ $2 != 'windows.sandbox="elevated"' ]] || windows_sandbox=1
        [[ $2 != 'approval_policy="never"' ]] || approval_never=1
      fi
      shift ;;
    --output-last-message) output=$2; shift ;;
    --cd) work=$2; shift ;;
  esac
  shift
done
if [[ ${TEST_REQUIRE_WINDOWS_PERMISSIONS:-0} == 1 ]] && (( !windows_sandbox || !approval_never )); then
  printf 'effective sandbox: read-only; exec-scoped Windows configuration missing\n' >&2
  exit 41
fi
cat >/dev/null
printf '{}' > "$output"
printf '{}' > "$work/.ralph-run/worker.json"
printf '%s\n' '{"type":"turn.started"}' '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}'
MOCK
  chmod +x "$RUN/bin/codex"
  export PATH="$RUN/bin:$PATH" ANTHROPIC_API_KEY=fixture OPENAI_API_KEY=fixture GH_TOKEN=fixture
  printf 'fixture' > "$WORK/.ralph-run/context.md"
  invoke_model worker "$WORK/.ralph-run/context.md" "$EVIDENCE/mock-worker.json"
  grep -Fxq 'gpt-6-sol' "$RUN/provider-args"
  grep -Fxq 'model_reasoning_effort="xhigh"' "$RUN/provider-args"
  grep -Fxq 'workspace-write' "$RUN/provider-args"
  invoke_model review "$WORK/.ralph-run/context.md" "$EVIDENCE/mock-review.json"
  grep -Fxq 'gpt-6-astra' "$RUN/provider-args"
  grep -Fxq 'model_reasoning_effort="medium"' "$RUN/provider-args"
  grep -Fxq 'read-only' "$RUN/provider-args"
}
test_container() {
  setup; ISSUE=2; prepare_issue; worker_attempt; HEAD=$(jqtext '.head' "$ACTIVE")
  source "$RALPH_DIR/checks.sh"
  jq '.validation.commands=[["sh","-c","test -f module-2.txt && test -z \"$ANTHROPIC_API_KEY\" && ! touch /src/unapproved && echo isolation-passed"]]' "$QUEUE" > "$QUEUE.tmp"
  mv "$QUEUE.tmp" "$QUEUE"
  export ANTHROPIC_API_KEY=fixture
  validate_candidate "$HEAD"
  grep -q isolation-passed "$EVIDENCE/checks-$ATTEMPT.log"
}

if [[ $# -gt 0 ]]; then "$1"; exit; fi
for script in "$RALPH_DIR"/*.sh "$TEST_DIR"/*.sh; do bash -n "$script"; done
for test in lifecycle resume check_repair merge_propagation merge_resume legacy_resume check_gate local_commit keys provider_command; do
  bash "$0" "test_$test" > "${TMPDIR:-/tmp}/ralph-$test.log" 2>&1 || { cat "${TMPDIR:-/tmp}/ralph-$test.log"; exit 1; }
  printf 'PASS: %s\n' "$test"
done
for item in 'stale_review:Independent review blocked' 'main_moved:Merge gate changed' \
  'revision_limit:Two worker attempts exhausted' 'protected_change:Worker changed protected path' \
  'dependency_failure:Cannot read native dependencies' 'upload_mismatch:GitHub blob differs' \
  'merge_never_confirms:Timed out waiting for GitHub merge confirmation' \
  'merged_head_changed:PR head changed during merge confirmation' \
  'wrong_merge_commit:PR merged at a different commit'; do
  test=${item%%:*}; expected=${item#*:}
  if bash "$0" "test_$test" > "${TMPDIR:-/tmp}/ralph-$test.log" 2>&1; then
    printf 'FAIL: %s unexpectedly succeeded\n' "$test"; exit 1
  fi
  grep -Fq "$expected" "${TMPDIR:-/tmp}/ralph-$test.log" || { cat "${TMPDIR:-/tmp}/ralph-$test.log"; exit 1; }
  printf 'PASS: %s stops safely\n' "$test"
done

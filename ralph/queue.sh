#!/usr/bin/env bash
prepare_issue() {
  EVIDENCE="$RUN/issues/$ISSUE"; ACTIVE="$EVIDENCE/state.json"
  WORK="$ROOT/.ralph/worktrees/issue-$ISSUE"; BRANCH="ralph/issue-$ISSUE"
  mkdir -p "$EVIDENCE"
  if [[ ! -f $ACTIVE ]]; then
    read_issue "$ISSUE" "$EVIDENCE/issue.json"
    eligible_issue "$EVIDENCE/issue.json" || die "Issue #$ISSUE is no longer eligible."
    dependencies_closed "$ISSUE" "$EVIDENCE/issue.json" || die "Issue #$ISSUE has open dependencies."
    BASE=$(remote_base); ensure_commit "$BASE"
    jq -n --arg base "$BASE" --argjson deadline "$(( $(date +%s) + 5400 ))" \
      '{phase:"claiming",base:$base,head:$base,deadline:$deadline,attempts:0,checks:0,reviews:0}' > "$ACTIVE"
    gh_call api --method POST "repos/$REPO/git/refs" -f "ref=refs/heads/$BRANCH" -f "sha=$BASE" >/dev/null
    active_update '.phase="claimed"'
  fi
  check_deadline
  BASE=$(jqtext '.base' "$ACTIVE"); HEAD=$(jqtext '.head' "$ACTIVE")
  [[ $(jqtext '.phase' "$ACTIVE") != claiming ]] || die 'Interrupted branch claim: reconcile the remote ref before continuing.'
  CRITERIA=$(jqtext '.body | gsub("\r"; "") | split("## Acceptance criteria\n")[1] // "" |
    split("\n## ")[0] | [split("\n")[] | select(test("^- \\[[ xX]\\] "))] | length' "$EVIDENCE/issue.json")
  (( CRITERIA > 0 )) || die 'Issue has no explicit acceptance checklist.'
  if [[ $(jqtext '.phase' "$ACTIVE") == claimed ]]; then
    gh_call issue edit "$ISSUE" --repo "$REPO" --add-assignee "$ACTOR"
    [[ ! -e $WORK ]] || die "Unexpected existing worktree: $WORK"
    git -C "$ROOT" worktree add -b "$BRANCH" "$WORK" "$BASE"
    active_update '.phase="working"'
  fi
  [[ -d $WORK ]] || die 'Saved worktree is missing; do not reset the queue.'
}
worker_receipt_valid() {
  jq -e --argjson issue "$ISSUE" --argjson criteria "$CRITERIA" '
    .issue == $issue and .status == "ready_for_review" and
    (.summary | type == "string" and length > 0) and
    ([.acceptance[].criterion] | sort) == [range(1;$criteria+1)] and
    all(.acceptance[]; .evidence | type == "string" and length > 0) and
    (.commit.subject | test("^(feat|fix|perf|style|refactor|test|docs|build|ci|chore|revert)(\\([a-zA-Z0-9_-]+\\))?!?: :[a-z0-9_]+: [^\\r\\n]+$")) and
    (.commit.why | type == "string" and length > 0) and
    (.merge_danger.door == "one-way" or .merge_danger.door == "two-way") and
    (.merge_danger.blast_radius | type == "string" and length > 0) and
    (.risks | type == "array")' "$1" >/dev/null
}
worker_attempt() {
  ATTEMPT=$(jqtext '.attempts + 1' "$ACTIVE")
  (( ATTEMPT <= $(jqtext '.max_attempts' "$POLICY") )) || die 'Two worker attempts exhausted; inspect the preserved evidence.'
  active_update '.attempts += 1'
  mkdir -p "$WORK/.ralph-run"
  [[ ! -L $WORK/.ralph-run ]] || die 'Worker context directory cannot be a symlink.'
  rm -f "$WORK/.ralph-run/worker.json"
  {
    cat "$RALPH_DIR/prompt.md" "$RALPH_DIR/worker-prompt.md"
    printf '\nSelected issue (%s numbered criteria, in checklist order):\n' "$CRITERIA"
    cat "$EVIDENCE/issue.json"
    printf '\nParent PRD:\n'; cat "$RUN/prd.json"
    printf '\nQueue objective and approved checks:\n'; cat "$QUEUE"
    if (( ATTEMPT > 1 )); then
      printf '\nPrevious attempt evidence (repair only this issue):\n'
      cat "$EVIDENCE/review-$((ATTEMPT-1)).json" 2>/dev/null || true
      cat "$EVIDENCE/checks-$((ATTEMPT-1)).log" 2>/dev/null || true
    fi
  } > "$WORK/.ralph-run/context.md"
  invoke_model worker "$WORK/.ralph-run/context.md" "$EVIDENCE/worker-$ATTEMPT.json"
  worker_receipt_valid "$EVIDENCE/worker-$ATTEMPT.json" || die 'Worker is blocked or its receipt does not cover the issue.'
  make_candidate
}
publish_pr() {
  local head title
  publish_branch
  head=$(gh_call api "repos/$REPO/git/ref/heads/$BRANCH" --jq .object.sha)
  [[ $head == "$HEAD" ]] || die 'Remote issue branch differs from the reviewed head.'
  gh_call pr list --repo "$REPO" --head "$BRANCH" --state all --json number,state,headRefOid,baseRefName > "$EVIDENCE/prs.json"
  if [[ $(jqtext 'length' "$EVIDENCE/prs.json") == 0 ]]; then
    {
      printf '## Summary\n\nIssue #%s -> public behavior tests -> independent Astra review\n\n' "$ISSUE"
      jqtext '.summary' "$EVIDENCE/worker-$ATTEMPT.json"
      printf '\nCloses #%s\n\n## Evidence\n\n' "$ISSUE"
      printf 'Candidate `%s` on base `%s`.\n\n' "$HEAD" "$BASE"
      printf '### Validation\n\n```text\n'; cat "$EVIDENCE/checks-$ATTEMPT.log"; printf '\n```\n\n'
      printf '### Independent review (gpt-6-astra, medium)\n\n```json\n'
      cat "$EVIDENCE/review-$ATTEMPT.json"; printf '\n```\n\n## Merge Danger\n\n'
      jqtext '.merge_danger | "**Door:** " + .door + "\n\n**Blast Radius:** " + .blast_radius + "\n\n" + (.reason // "")' "$EVIDENCE/worker-$ATTEMPT.json"
    } > "$EVIDENCE/pr-body.md"
    title=$(jqtext '.commit.subject' "$EVIDENCE/worker-$ATTEMPT.json")
    gh_call pr create --repo "$REPO" --base "$BASE_BRANCH" --head "$BRANCH" --title "$title" --body-file "$(native_path "$EVIDENCE/pr-body.md")"
    gh_call pr list --repo "$REPO" --head "$BRANCH" --state open --json number,state,headRefOid,baseRefName > "$EVIDENCE/prs.json"
  fi
  jq -e --arg head "$HEAD" --arg base "$BASE_BRANCH" \
    'length == 1 and .[0].state == "OPEN" and .[0].headRefOid == $head and .[0].baseRefName == $base' "$EVIDENCE/prs.json" >/dev/null || die 'Unexpected PR state.'
  active_update --argjson pr "$(jqtext '.[0].number' "$EVIDENCE/prs.json")" '.pr=$pr | .phase="published"'
}
checks_pass() {
  jq -e --slurpfile queue "$QUEUE" '
    . as $checks | length > 0 and
    all($queue[0].required_checks[]; . as $name | any($checks[]; .name == $name and .bucket == "pass")) and
    all(.[]; .bucket == "pass" or .bucket == "skipping")' "$1" >/dev/null
}
wait_ci() {
  local pr=$1 result until=$(( $(date +%s) + $(jqtext '.ci_timeout_seconds' "$QUEUE") ))
  while :; do
    check_deadline
    result=0
    gh_call pr checks "$pr" --repo "$REPO" --json name,bucket,state > "$EVIDENCE/ci.json" || result=$?
    [[ $result == 0 || $result == 8 || $result == 1 ]] || die 'Cannot read CI checks.'
    jq -e 'type == "array"' "$EVIDENCE/ci.json" >/dev/null || die 'Invalid CI response.'
    if checks_pass "$EVIDENCE/ci.json"; then return; fi
    jq -e 'any(.[]; .bucket == "fail" or .bucket == "cancel")' "$EVIDENCE/ci.json" >/dev/null && die 'CI failed; preserve this PR for repair.'
    (( $(date +%s) < until )) || die 'CI did not complete within its time limit.'
    say "Waiting for required checks on PR #$pr..."
    sleep 15
  done
}
finish_merge() {
  local pr=$1 poll until=$(( $(date +%s) + 120 ))
  # Updating main and updating PR/issue metadata are not immediately consistent.
  # Reconcile using reads only; never repeat the merge or count an OPEN response.
  for ((poll=0; poll<=24; poll++)); do
    check_deadline
    gh_call pr view "$pr" --repo "$REPO" --json state,headRefOid,mergeCommit > "$EVIDENCE/merged.json"
    jq -c --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{observed_at:$at,pr:.}' \
      "$EVIDENCE/merged.json" >> "$EVIDENCE/merge-confirmation.jsonl"
    jq -e --arg head "$HEAD" '.headRefOid == $head' "$EVIDENCE/merged.json" >/dev/null ||
      die 'PR head changed during merge confirmation; inspect the evidence.'
    jq -e '.state == "OPEN" or .state == "MERGED"' "$EVIDENCE/merged.json" >/dev/null ||
      die 'PR closed without a confirmed merge, or returned invalid state.'
    if [[ $(jqtext '.state' "$EVIDENCE/merged.json") == MERGED ]]; then
      jq -e --arg head "$HEAD" '.mergeCommit.oid == null or .mergeCommit.oid == $head' "$EVIDENCE/merged.json" >/dev/null ||
        die 'PR merged at a different commit; inspect the evidence.'
      if [[ $(jqtext '.mergeCommit.oid // empty' "$EVIDENCE/merged.json") == "$HEAD" ]]; then
        read_issue "$ISSUE" "$EVIDENCE/closed.json"
        jq -c --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{observed_at:$at,issue:.}' \
          "$EVIDENCE/closed.json" >> "$EVIDENCE/merge-confirmation.jsonl"
        if [[ $(jqtext '.state' "$EVIDENCE/closed.json") == CLOSED ]]; then
          active_update '.phase="complete"'
          state_update --argjson issue "$ISSUE" '.completed = ((.completed + [$issue]) | unique) | .active=null'
          return
        fi
      fi
    fi
    (( poll < 24 && $(date +%s) < until )) || break
    say "Waiting for GitHub to confirm PR #$pr merged and issue #$ISSUE closed..."
    sleep 5
  done
  die 'Timed out waiting for GitHub merge confirmation; saved merging state can be reconciled on resume.'
}
merge_pr() {
  local pr; pr=$(jqtext '.pr' "$ACTIVE")
  wait_ci "$pr"
  gh_call pr view "$pr" --repo "$REPO" --json state,headRefOid,baseRefOid,mergeable,mergeStateStatus,isDraft > "$EVIDENCE/merge-gate.json"
  jq -e --arg head "$HEAD" --arg base "$BASE" \
    '.state == "OPEN" and .headRefOid == $head and .baseRefOid == $base and .mergeable == "MERGEABLE" and .mergeStateStatus == "CLEAN" and .isDraft == false' \
    "$EVIDENCE/merge-gate.json" >/dev/null || die 'Merge gate changed or branch protection blocks merging; no bypass.'
  [[ $(remote_base) == "$BASE" ]] || die 'Main moved since review; this issue needs a fresh base and review.'
  review_passes "$EVIDENCE/review-$ATTEMPT.json" || die 'Review no longer matches this candidate.'
  active_update '.phase="merging"'
  # Exact reviewed commit; non-FF races and GitHub branch protection reject this.
  gh_call api --method PATCH "repos/$REPO/git/refs/heads/$BASE_BRANCH" -f "sha=$HEAD" -F force=false >/dev/null
  finish_merge "$pr"
}
work_issue() {
  prepare_issue
  gh_call issue view "$(jqtext '.parent_issue' "$POLICY")" --repo "$REPO" --json number,title,body > "$RUN/prd.json"
  local phase
  while :; do
    check_deadline
    phase=$(jqtext '.phase' "$ACTIVE"); ATTEMPT=$(jqtext '.attempts' "$ACTIVE"); HEAD=$(jqtext '.head' "$ACTIVE")
    say "Issue #$ISSUE | $phase | attempt $ATTEMPT/2"
    case "$phase" in
      working) worker_attempt ;;
      candidate)
        if validate_candidate "$HEAD"; then
          review_candidate
        else
          active_update '.phase="working"'
          say 'Candidate checks failed. Preserve the log and use the remaining repair attempt.'
        fi ;;
      reviewed) publish_pr ;;
      published) merge_pr; return ;;
      merging) finish_merge "$(jqtext '.pr' "$ACTIVE")"; return ;;
      complete) finish_merge "$(jqtext '.pr' "$ACTIVE")"; return ;;
      *) die "Unknown saved phase: $phase" ;;
    esac
  done
}

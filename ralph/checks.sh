#!/usr/bin/env bash
protected_path() {
  case "/${1,,}" in
    /ralph/*|/.github/*|/.agents/*|/.claude/*|/.codex/*|/.cursor/*|/docs/agents/*|/docs/adr/*|*/agents.md|*/claude.md|/context.md|/.gitignore|/.gitattributes|/.gitmodules|*/.env|*/.env.*|*/.ralph/*|*/.ralph-run/*) return 0 ;;
    *) return 1 ;;
  esac
}
make_candidate() {
  local file mode bytes tree head before
  before=$(jqtext '.head' "$ACTIVE")
  [[ $(git -C "$WORK" rev-parse HEAD) == "$before" ]] || die 'Worker changed HEAD; controller owns commits.'
  # Build a fresh private index from HEAD; ignore any worker staging decisions.
  local index="$EVIDENCE/index-$ATTEMPT"
  [[ ! -e $index ]] || die "Unexpected existing index $index."
  export GIT_INDEX_FILE; GIT_INDEX_FILE=$(native_path "$index")
  git -C "$WORK" read-tree "$before"
  git -C "$WORK" -c core.hooksPath=/dev/null add -A -- .
  git -C "$WORK" diff --cached --name-only -z > "$EVIDENCE/paths.z"
  [[ -s $EVIDENCE/paths.z ]] || die 'Worker produced no changes.'
  while IFS= read -r -d '' file; do
    protected_path "$file" && die "Worker changed protected path: $file"
    [[ $file != *$'\n'* && $file != *$'\r'* && $file != *$'\t'* ]] || die 'Unsupported filename.'
    mode=$(git -C "$WORK" ls-files --stage -- "$file" | cut -d ' ' -f1)
    [[ -z $mode || $mode == 100644 || $mode == 100755 ]] || die "Non-regular file: $file"
    if [[ -f $WORK/$file ]]; then
      bytes=$(wc -c < "$WORK/$file")
      (( bytes <= $(jqtext '.max_changed_file_bytes' "$POLICY") )) || die "File too large to review: $file"
    fi
  done < "$EVIDENCE/paths.z"
  git -C "$WORK" diff --cached --numstat > "$EVIDENCE/numstat"
  ! grep -q $'^-\t' "$EVIDENCE/numstat" || die 'Binary changes require supervised review.'
  git -C "$WORK" diff --cached --check
  tree=$(git -C "$WORK" write-tree)
  jqtext '.commit.subject + "\n\n" + .commit.why' "$EVIDENCE/worker-$ATTEMPT.json" > "$EVIDENCE/commit.txt"
  local stamp; stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  head=$(GIT_AUTHOR_DATE="$stamp" GIT_COMMITTER_DATE="$stamp" git -C "$WORK" -c core.hooksPath=/dev/null commit-tree "$tree" -p "$before" -F "$EVIDENCE/commit.txt")
  unset GIT_INDEX_FILE
  git -C "$WORK" update-ref "refs/heads/$BRANCH" "$head" "$before"
  git -C "$WORK" reset --mixed "$head" >/dev/null
  active_update --arg head "$head" '.head=$head | .phase="candidate"'
}
validate_candidate() {
  local head=$1 count snapshot image command
  count=$(jqtext '.checks' "$ACTIVE")
  (( count < 4 )) || die 'Validation attempt limit reached.'
  active_update '.checks += 1' || die 'Cannot save validation attempt.'
  snapshot="$EVIDENCE/snapshot-$((count+1))"
  mkdir "$snapshot" || die 'Cannot create validation snapshot.'
  git -C "$ROOT" archive "$head" | tar -x -C "$snapshot" || die 'Cannot materialize candidate commit.'
  image=$(jqtext '.validation.image' "$QUEUE")
  local -a args
  while IFS= read -r command; do
    mapfile -t args < <(printf '%s' "$command" | jqtext '.[]')
    CHECK_CONTAINER="neotix-ralph-$ISSUE-$$"
    say "[check] ${args[*]} (no network, no client key)"
    if MSYS_NO_PATHCONV=1 timeout --kill-after=10s "$(jqtext '.validation.timeout_seconds' "$QUEUE")s" \
      docker run --rm --name "$CHECK_CONTAINER" --network none --read-only --cap-drop ALL \
      --security-opt no-new-privileges --pids-limit 128 --memory 2g --cpus 2 --tmpfs /tmp:rw,noexec,nosuid,size=256m \
      -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/src \
      --mount "type=bind,source=$(native_path "$snapshot"),target=/src,readonly" -w /src "$image" "${args[@]}" \
      2>&1 | tee -a "$EVIDENCE/checks-$ATTEMPT.log"; then
      :
    else
      docker rm -f "$CHECK_CONTAINER" >/dev/null 2>&1 || true
      CHECK_CONTAINER=''
      return 1
    fi
    CHECK_CONTAINER=''
  done < <(jq -c '.validation.commands[]' "$QUEUE")
}
review_passes() {
  jq -e --argjson issue "$ISSUE" --arg base "$BASE" --arg head "$HEAD" --argjson criteria "$CRITERIA" '
    .issue == $issue and .base_sha == $base and .head_sha == $head and .verdict == "pass" and
    (.summary | type == "string" and length > 0) and
    ([.acceptance[].criterion] | sort) == [range(1;$criteria+1)] and
    all(.acceptance[]; .verdict == "pass" and (.evidence | type == "string" and length > 0)) and
    (.findings | type == "array") and all(.findings[]; .severity == "advisory")' "$1" >/dev/null
}
review_candidate() {
  local count file context="$EVIDENCE/review-context-$ATTEMPT.md"
  count=$(jqtext '.reviews' "$ACTIVE")
  (( count < 3 )) || die 'Review attempt limit reached.'
  active_update '.reviews += 1'
  {
    cat "$RALPH_DIR/reviewer-prompt.md"
    printf '\nIssue: %s\nBase SHA: %s\nHead SHA: %s\nCriteria: %s\n' "$ISSUE" "$BASE" "$HEAD" "$CRITERIA"
    cat "$EVIDENCE/issue.json" "$RUN/prd.json"
    printf '\nRepository guidance:\n'; git -C "$ROOT" show "$BASE:AGENTS.md"
    git -C "$ROOT" show "$BASE:CONTEXT.md" 2>/dev/null || true
    printf '\nCandidate diff:\n'; git -C "$ROOT" diff "$BASE" "$HEAD" --
    printf '\nComplete changed files:\n'
    while IFS= read -r -d '' file; do
      printf '\n--- %s ---\n' "$file"
      git -C "$ROOT" show "$HEAD:$file" 2>/dev/null || printf '(deleted)\n'
    done < <(git -C "$ROOT" diff --name-only -z "$BASE" "$HEAD")
    printf '\nActual validation:\n'; cat "$EVIDENCE/checks-$ATTEMPT.log"
  } > "$context"
  (( $(wc -c < "$context") <= $(jqtext '.max_review_bytes' "$POLICY") )) || die 'Review context exceeds the limit; split the issue.'
  invoke_model review "$context" "$EVIDENCE/review-$ATTEMPT.json"
  if review_passes "$EVIDENCE/review-$ATTEMPT.json"; then
    active_update '.phase="reviewed"'
  else
    jq -e '.verdict == "revise"' "$EVIDENCE/review-$ATTEMPT.json" >/dev/null || die 'Independent review blocked or malformed.'
    active_update '.phase="working"'
  fi
}

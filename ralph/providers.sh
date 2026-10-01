#!/usr/bin/env bash
# Personal CLI subscriptions only. Never inherit client API credentials.
without_keys() (
  local name
  while IFS= read -r name; do
    case "$name" in
      *TOKEN*|*SECRET*|*PASSWORD*|*API_KEY*|ANTHROPIC_*|OPENAI_*|GH_*|GITHUB_*|AWS_*|AZURE_*) unset "$name" ;;
    esac
  done < <(compgen -e)
  "$@"
)
provider_preflight() {
  command -v codex >/dev/null || die 'Codex CLI is required for independent Astra review.'
  local auth
  auth=$(without_keys codex login status 2>&1) || die 'Run codex login using your personal subscription.'
  [[ $auth == *ChatGPT* ]] || die 'Codex must use personal ChatGPT login, not an API key.'
  case "$BACKEND" in
    codex) ;;
    cursor)
      [[ $(uname -s) == Linux ]] || die 'Cursor requires WSL/Linux with its sandbox. Run agent login inside WSL.'
      command -v agent >/dev/null || die 'Cursor agent CLI is missing.'
      without_keys agent --list-models > "$RUN/cursor-models.txt"
      grep -Fq 'cursor-grok-4.6-xhigh' "$RUN/cursor-models.txt" || die 'Pinned Cursor model is unavailable.' ;;
    claude)
      command -v claude >/dev/null || die 'Claude CLI is missing.'
      without_keys claude auth status > "$RUN/claude-auth.json"
      jq -e '.loggedIn == true and .authMethod == "claude.ai"' "$RUN/claude-auth.json" >/dev/null ||
        die 'Claude needs a personal subscription login; the client API key cannot be used.' ;;
  esac
}
invoke_model() {
  local role=$1 context=$2 output=$3 seconds model effort provider log
  local -a cmd=(codex --ask-for-approval never)
  check_deadline
  provider=$BACKEND; model=$(jqtext --arg p "$BACKEND" '.workers[$p].model' "$POLICY")
  effort=$(jqtext --arg p "$BACKEND" '.workers[$p].effort' "$POLICY")
  seconds=$(jqtext '.worker_timeout_seconds' "$POLICY")
  if [[ $role == review ]]; then provider=codex; model=gpt-6-astra; effort=medium; seconds=$(jqtext '.review_timeout_seconds' "$POLICY"); fi
  local remaining=$(( $(jqtext '.deadline' "$ACTIVE") - $(date +%s) ))
  (( seconds <= remaining )) || seconds=$remaining
  log="${output%.json}"
  say "[$role] $model / $effort; live output and ${log}.events.jsonl"
  case "$provider" in
    codex)
      case "$(uname -s)" in MINGW*|MSYS*) cmd+=(-c 'windows.sandbox="elevated"');; esac
      cmd+=(exec --ignore-user-config --ephemeral --sandbox "$([[ $role == review ]] && echo read-only || echo workspace-write)"
        --model "$model" -c "model_reasoning_effort=\"$effort\"" --cd "$(native_path "$WORK")"
        --json --output-last-message "$(native_path "$output")" -)
      without_keys timeout --kill-after=15s "${seconds}s" "${cmd[@]}" < "$context" \
        2> >(tee -a "${log}.stderr.log" >&2) | tee "${log}.events.jsonl" |
        jq --unbuffered -rj --arg color 0 --arg repo_root "$(native_path "$WORK")" -f "$RALPH_DIR/codex-render.jq" ;;
    cursor)
      # Context and receipt paths are within the sandboxed worktree.
      without_keys timeout --kill-after=15s "${seconds}s" agent --print --output-format stream-json \
        --trust --force --sandbox enabled --workspace "$WORK" --model cursor-grok-4.6-xhigh \
        'Read .ralph-run/context.md and follow it. Write .ralph-run/worker.json, then stop.' </dev/null \
        2> >(tee -a "${log}.stderr.log" >&2) | tee "${log}.events.jsonl" |
        jq --unbuffered -r 'select(.type == "assistant" or .type == "result") | .message.content[]?.text // .result // empty' ;;
    claude)
      (cd "$WORK"; without_keys timeout --kill-after=15s "${seconds}s" claude --print --bare --no-session-persistence \
        --output-format json --model "$model" --effort "$effort" --permission-mode dontAsk \
        --tools Read,Edit,Write,Glob,Grep --allowedTools Read,Edit,Write,Glob,Grep < "$context") \
        2> >(tee -a "${log}.stderr.log" >&2) | tee "${log}.events.jsonl" ;;
  esac
  if [[ $role == worker ]]; then
    [[ -f $WORK/.ralph-run/worker.json && ! -L $WORK/.ralph-run/worker.json ]] || die 'Worker did not leave a regular receipt file.'
    cp "$WORK/.ralph-run/worker.json" "$output"
  fi
}

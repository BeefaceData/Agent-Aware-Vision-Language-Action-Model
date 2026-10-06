# Neotix Ralph

`afk.sh` is the pasted loop with an issue target, adapted for this repository and Codex.

From PowerShell:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ralph/afk.sh 6
```

From Git Bash or Linux:

```bash
bash ralph/afk.sh 6
```

Have `git`, `gh`, `jq` and `codex` on your Bash PATH, with GitHub and Codex
already signed in. Codex uses your configured model.

Each iteration fetches open issues, reads the last eight commits and
`prompt.md`, writes `.agent/ralph-context.md`, and starts a fresh Codex run.
Codex runs without sandboxing or permission prompts. Assistant text streams to
the terminal and JSONL events are saved under `.agent/history/`.

The number is the target of distinct issues completed in this run (6 means six issues).
Codex closes each finished issue and emits `<promise>ISSUE COMPLETED #123</promise>`;
the loop counts each issue number once. Commits and partial progress do not count. The loop stops at that
target, `NO MORE TASKS`, `[BLOCKED]`, an unexpected failure, or the pasted
`TARGET * 3 + 3` iteration cap. Unexpected failures leave
`.agent/last-failure.md` for the next run. There are no added runtime limits,
lookup timeouts, retries, queue controller, or separate review stages.

# Shell-loop validation

Run `bash ralph/tests/test-shell.sh` and `bash ralph/test-codex-render.sh`.

The shell lifecycle tests use actual temporary Git repositories, commits,
worktrees and pushes to a local bare remote, with deterministic GitHub/model
boundaries. They exercise sequential issues, resume, legacy queue migration,
stale review rejection, changed-main rejection, review repair limits, protected
files, required checks, credential stripping and the no-fetch local-object path.

These tests do not establish provider model quality, successful paid supervision,
or a measured VLA improvement. `codex-afk.sh 50 --doctor` checks the installed
environment read-only without launching a coding worker.

CI runs the same shell tests on Ubuntu and Windows. Candidate project tests run
separately in the pinned Docker image before independent review.

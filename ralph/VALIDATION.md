# Shell-loop validation

Run `bash ralph/tests/test-shell.sh` and `bash ralph/test-codex-render.sh`.

The shell lifecycle tests use actual temporary Git repositories, commits,
worktrees and pushes to a local bare remote, with deterministic GitHub/model
boundaries. They exercise sequential issues, resume, legacy queue migration,
stale review rejection, changed-main rejection, review repair limits, protected
files, required checks, credential stripping and the no-fetch local-object path.
Delayed PR merge visibility and delayed issue closure are tested independently;
completion is counted once, without another merge request. Unconfirmed merges
time out, and a different PR head or merged commit stops immediately.

These tests do not establish provider model quality, successful paid supervision,
or a measured VLA improvement. `codex-afk.sh 50 --doctor` checks the installed
environment without launching a coding worker or changing GitHub.

On Windows, preflight also runs a local read/write fixture through the native
Codex sandbox without a model call. The provider regression test models the
`exec` configuration scope: Windows sandbox and approval settings must follow
`exec`, including when user configuration is ignored. A real Codex 0.159.3
fixture reproduced the old read-only downgrade and passed with that placement.

CI runs the same shell tests on Ubuntu and Windows. Candidate project tests run
separately in the pinned Docker image before independent review.

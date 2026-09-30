# Issue tracker: GitHub

Issues and specs live in GitHub Issues for
`BeefaceData/Agent-Aware-Vision-Language-Action-Model`.

Use the `gh` CLI from this clone; it infers the repository from the remote.

## Conventions

- Create: `gh issue create --title "..." --body-file <path>`
- Read with discussion: `gh issue view <number> --comments`
- Read structured details: `gh issue view <number> --json number,title,body,labels,comments`
- List: `gh issue list --state open --json number,title,body,labels,comments`
  with appropriate label and state filters.
- Comment: `gh issue comment <number> --body-file <path>`
- Edit body: `gh issue edit <number> --body-file <path>`
- Apply or remove labels: `gh issue edit <number> --add-label "..."` or
  `gh issue edit <number> --remove-label "..."`
- Close: `gh issue close <number>`

For multiline bodies and comments, write the exact text to a temporary file
and pass it with `--body-file`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub shares an issue and PR number space. If a reference is ambiguous,
resolve with `gh pr view <number>` and fall back to `gh issue view <number>`.

## Skill terminology

- "Publish to the issue tracker": create a GitHub issue.
- "Fetch the relevant ticket": read the issue and its comments.

## Wayfinding operations

- Map: one issue labelled `wayfinder:map`, containing Notes,
  Decisions-so-far, and Fog.
- Child tickets: link issues to the map as GitHub sub-issues. If unavailable,
  use a task list in the map and `Part of #<map>` in each child.
  Use `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`,
  or `wayfinder:task` labels.
- Blocking: use native GitHub issue dependencies. Add an edge with
  `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`.
  Obtain the database ID with
  `gh api repos/<owner>/<repo>/issues/<number> --jq .id`.
  If dependencies are unavailable, use `Blocked by: #<number>` in the child.
- Frontier: choose the first open child in map order with no assignee and
  no open blockers. Check `issue_dependencies_summary.blocked_by`, or
  resolve the fallback references.
- Claim: `gh issue edit <number> --add-assignee "@me"`.
- Resolve: comment with the answer, close the child, and add a short
  explanation and link to the map's Decisions-so-far.

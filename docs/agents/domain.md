# Domain docs

## Layout and reading rules

This repo uses a single-context layout:

- `CONTEXT.md` at the repository root: domain terms and their meanings.
- `docs/adr/`: architectural decisions.

Before exploring the codebase, read `CONTEXT.md` and any ADRs relevant
to the area of work.

If these files do not exist, proceed silently. The `domain-modeling`
skill creates them when terms or decisions are resolved.

## Vocabulary

Use the terms defined in `CONTEXT.md` when naming domain concepts in
issues, proposals, hypotheses, and tests.

If a concept is missing, reconsider the wording or note the gap for
`domain-modeling`.

## Decision conflicts

If a proposal contradicts an ADR, identify that ADR and explain why
the decision should be reconsidered.

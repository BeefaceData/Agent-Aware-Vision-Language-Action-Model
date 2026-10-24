---
name: clean-code
description: Defaults for writing and reviewing code that is clear, easy to understand, and easy to test. Use this whenever you write new code, refactor, review a diff or pull request, or the user asks to "clean up", "tidy", "simplify", "make this readable", "make this testable", or asks whether code is well structured, even if they never say "clean code". These are judgment-based defaults to weigh, not rules to enforce.
---

# Clean code

The goal is code that the next person (or agent) can read, change, and test
without fear. These are defaults, not laws. When a guideline would make the code
harder to follow in a particular case, skip it and, if it's not obvious, say why.

When guidelines pull in different directions, use this order:
**correct > clear > testable > short**. Short tiebreak: pick the version a new
teammate would understand fastest.

## Before you change anything

- **Read the surrounding code first and match it.** Naming, comment density,
  error style, and structure should look like the code next to it. Consistency
  within a codebase matters more than any guideline below.
- **Stay in scope.** Improve the code you are already touching. If you notice
  problems elsewhere, mention them instead of fixing them in the same change.
  Large unrelated refactors make reviews harder and hide real changes.

## Clarity: can the reader tell what it does?

- **Names carry meaning.** Use domain words (`grasp_confirmed`, not `flag2`).
  Short names are fine in small scopes (a loop index). Wider scope needs a more
  descriptive name. Include units when they aren't obvious (`timeout_seconds`,
  `offset_m`).
- **One job per function, one level of detail.** If a block needs a comment to
  say *what* it does, it's often a good candidate to become a named function.
  Don't extract trivial one-liners just to make functions shorter.
- **Keep control flow flat.** Prefer early returns and guard clauses to deep
  nesting. Avoid boolean parameters that switch a function between two
  behaviors. Two named functions are usually clearer.
- **Make data shapes explicit.** When data travels beyond a function or two,
  give it a type (dataclass, typed model, named tuple) instead of a loose dict.
- **Name the numbers.** Turn unexplained literals into named constants, with
  units and, if it's not obvious, where the value came from.
- **Comments explain why, not what.** Record intent, invariants, units, safety
  reasons, and surprising decisions. Don't narrate the code line by line.
- **Delete dead code.** Remove commented-out blocks and unused helpers. Version
  control remembers them.

## Understandability: can the reader reason about it?

- **Keep related things close.** A reader shouldn't need to jump across many
  files to follow one behavior.
- **Make dependencies and side effects visible.** Prefer passing things in
  through arguments over hidden globals, singletons, or module-level state.
  A function's signature should hint at what it touches.
- **Fail clearly.** Raise or return specific, searchable errors. Don't silently
  swallow exceptions. If ignoring one is deliberate, say so in a comment.
- **Prefer boring over clever.** An obvious ten-line version beats a dense
  three-line one that needs to be decoded.

## Testability: can it be checked in isolation?

- **Separate decisions from effects.** Keep the logic that decides something
  (pure computation, validation, parsing) apart from code that does I/O
  (network, simulator, filesystem, clock, randomness). Pure logic can be tested
  with plain inputs and outputs.
- **Inject what you depend on.** Pass in clients, clocks, paths, and random
  generators instead of creating them inside the function. Tests can then hand
  in small fakes.
- **Change behavior, change a test.** New behavior should come with a test that
  shows it. A bug fix should come with a test that would have failed before.
- **Tests read like specifications.** Use descriptive test names, one behavior
  per test, and a visible arrange / act / assert shape. Keep logic out of tests.
  Prefer simple fakes to deep mock chains.
- **Use the project's existing test setup.** Follow its test runner, fixtures,
  and helpers instead of introducing new ones.

## When reviewing code

Report findings so the author can act quickly:

1. Start with a one-line overall impression.
2. List findings, most important first. For each one, give the location, what
   is unclear or hard to test, why it matters, and a concrete suggestion.
3. Label each finding **worth fixing** or **optional**. Keep optional items few,
   and skip pure taste nitpicks.
4. If the code is already clear, say so plainly. "Looks good" is a valid review.

## When to hold back

- During an urgent bug fix, make the fix minimal and mention cleanups separately.
- In generated, vendored, or third-party code, leave it alone.
- In performance-critical paths, a less obvious version can be right. Leave a
  comment that explains the trade-off.
- When a cleanup would make the diff much larger than the actual change, propose
  it as a follow-up.

For short before/after examples of these ideas, see `references/examples.md`.

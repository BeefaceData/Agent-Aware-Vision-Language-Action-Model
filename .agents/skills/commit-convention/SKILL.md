---
name: commit-convention
description: Enforces Conventional Commit messages with the full Gitmoji catalog mapped to project-approved commit types, and always writes a body explaining why. Use when writing commit messages, staging changes, or running git commit in this project.
---

# Commit Message Convention

Use Conventional Commits 1.0.0 as the structure, then add one Gitmoji after the colon-space when a table entry fits.

## Format

```text
<type>[optional scope][!]: <gitmoji-code> <description>
```

Examples:

```text
feat: :sparkles: add Fenrir checklist
fix: :bug: correct resume cursor seed drift
docs: :memo: update Fenrir QA note
build: :arrow_up: upgrade dependency pin
refactor: :recycle: simplify calibration planning
feat!: :boom: change task metadata schema
```

Use the Conventional Commit type first. The Gitmoji narrows the intent; it does not replace the type. If the type and Gitmoji appear to disagree, fix the type before picking the emoji.

## Allowed Types

| Type | When to use | SemVer signal |
|---|---|---|
| `feat` | New product, task, workflow, behavior, capability, UI/UX, i18n, analytics, business logic, healthcheck, validation or offline support. | MINOR |
| `fix` | Bug fix, security/privacy fix, hotfix, warning fix, error handling fix, compatibility fix or noisy-output fix. | PATCH |
| `perf` | Performance-only improvement with no behavior change. | none by default |
| `style` | Formatting, code style, UI styling, responsive styling, animations or transitions with no logic change. | none by default |
| `refactor` | Code restructure, architecture cleanup, type-only reshaping, dead-code removal or move/rename with no behavior change. | none by default |
| `test` | Add, update, pass, snapshot, mock or failing-test work. | none by default |
| `docs` | Documentation, comments, license, contributor metadata, text-only spelling or wording changes. | none by default |
| `build` | Dependencies, packaged artifacts, compiled outputs or build-system inputs. | none by default |
| `ci` | CI pipelines, CI build configuration or CI failure repair. | none by default |
| `chore` | Tooling, configuration, repository hygiene, scripts, releases, experiments, infrastructure, seeds or operational maintenance. | none by default |
| `revert` | Revert a previous change. | none by default |

Breaking changes use `!` after the type or scope, and may also include a `BREAKING CHANGE:` footer. Use `:boom:` with the same type when it helps humans scan the log:

```text
feat(api)!: :boom: remove legacy task upload shape

BREAKING CHANGE: task upload now requires the new metadata shape.
```

## Gitmoji Pairing Table

Source: gitmoji.dev catalog, mapped to Conventional Commit types for this project.

| Code | Project type | Use when |
|---|---|---|
| `:art:` | `style`  | Improve code structure or formatting without behavior change. |
| `:zap:` | `perf`  | Improve performance. |
| `:fire:` | `chore`  | Remove code or files that are not a behavior-preserving refactor. |
| `:bug:` | `fix`  | Fix a bug. |
| `:ambulance:` | `fix`  | Critical hotfix. |
| `:sparkles:` | `feat`  | Introduce a feature or new user-visible capability. |
| `:memo:` | `docs`  | Add or update documentation. |
| `:rocket:` | `chore`  | Deploy or prepare deployment work. |
| `:lipstick:` | `style`  | Add or update UI/style files without business-logic change. |
| `:tada:` | `chore`  | Begin a project or initialize a major repo area. |
| `:white_check_mark:` | `test`  | Add, update or pass tests. |
| `:lock:` | `fix`  | Fix security or privacy issues. Do not invent a `security` type. |
| `:closed_lock_with_key:` | `chore`  | Add or update secrets or secret-handling material. |
| `:bookmark:` | `chore`  | Release or version tag work. |
| `:rotating_light:` | `fix`  | Fix compiler or linter warnings. |
| `:construction:` | `chore`  | Work in progress checkpoint that must be committed. |
| `:green_heart:` | `ci`  | Fix CI build. |
| `:arrow_down:` | `build`  | Downgrade dependencies. |
| `:arrow_up:` | `build`  | Upgrade dependencies. |
| `:pushpin:` | `build`  | Pin dependencies to specific versions. |
| `:construction_worker:` | `ci`  | Add or update CI build system. |
| `:chart_with_upwards_trend:` | `feat`  | Add or update analytics/tracking behavior. |
| `:recycle:` | `refactor`  | Refactor code. |
| `:heavy_plus_sign:` | `build`  | Add a dependency. |
| `:heavy_minus_sign:` | `build`  | Remove a dependency. |
| `:wrench:` | `chore`  | Add or update configuration files. |
| `:hammer:` | `chore`  | Add or update development scripts. |
| `:globe_with_meridians:` | `feat`  | Add or update internationalization/localization. |
| `:pencil2:` | `docs`  | Fix typos in docs/text. Use `fix` only when the typo breaks code. |
| `:poop:` | `chore`  | Commit known-bad code that still needs cleanup. Avoid unless necessary. |
| `:rewind:` | `revert`  | Revert changes. |
| `:twisted_rightwards_arrows:` | `chore`  | Merge branches. |
| `:package:` | `build`  | Add or update compiled files or packages. |
| `:alien:` | `fix`  | Update code because an external API changed. |
| `:truck:` | `refactor`  | Move or rename files, paths or routes. |
| `:page_facing_up:` | `docs`  | Add or update license files. |
| `:boom:` | `<type>!`  | Introduce breaking changes. Pair with `!` or a `BREAKING CHANGE:` footer. |
| `:bento:` | `chore`  | Add or update assets. Use `feat` if the asset is the feature. |
| `:wheelchair:` | `feat`  | Improve accessibility. Use `fix` for accessibility regressions. |
| `:bulb:` | `docs`  | Add or update source-code comments. |
| `:beers:` | `chore`  | Commit informal or deliberately rough work. Avoid unless there is a reason. |
| `:speech_balloon:` | `feat`  | Add or update user-facing text/literals. Use `docs` for docs-only text. |
| `:card_file_box:` | `chore`  | Perform database-related changes. Use `feat`/`fix` when tied to behavior. |
| `:loud_sound:` | `chore`  | Add or update logs. |
| `:mute:` | `fix`  | Remove noisy logs/output. |
| `:busts_in_silhouette:` | `docs`  | Add or update contributor metadata. |
| `:children_crossing:` | `feat`  | Improve user experience or usability. |
| `:building_construction:` | `refactor`  | Make architectural changes. |
| `:iphone:` | `style`  | Work on responsive design. |
| `:clown_face:` | `test`  | Mock things. |
| `:egg:` | `feat`  | Add or update an easter egg. |
| `:see_no_evil:` | `chore`  | Add or update `.gitignore`. |
| `:camera_flash:` | `test`  | Add or update snapshots. |
| `:alembic:` | `chore`  | Perform experiments. |
| `:mag:` | `feat`  | Improve SEO. |
| `:label:` | `refactor`  | Add or update types without runtime behavior change. |
| `:seedling:` | `chore`  | Add or update seed files. |
| `:triangular_flag_on_post:` | `feat`  | Add, update or remove feature flags. |
| `:goal_net:` | `fix`  | Catch errors. |
| `:dizzy:` | `style`  | Add or update animations and transitions. |
| `:wastebasket:` | `refactor`  | Deprecate code that needs to be cleaned up. |
| `:passport_control:` | `feat`  | Work on authorization, roles or permissions. Use `fix` for auth bugs/security. |
| `:adhesive_bandage:` | `fix`  | Simple fix for a non-critical issue. |
| `:monocle_face:` | `chore`  | Data exploration or inspection. |
| `:coffin:` | `refactor`  | Remove dead code. |
| `:test_tube:` | `test`  | Add a failing test. |
| `:necktie:` | `feat`  | Add or update business logic. |
| `:stethoscope:` | `feat`  | Add or update healthcheck behavior. |
| `:bricks:` | `chore`  | Infrastructure-related changes. |
| `:technologist:` | `chore`  | Improve developer experience. |
| `:money_with_wings:` | `chore`  | Add sponsorship or money-related infrastructure. |
| `:thread:` | `feat`  | Add/update concurrency behavior. Use `perf` or `refactor` when that is the real intent. |
| `:safety_vest:` | `feat`  | Add or update validation behavior. Use `fix` for validation bugs. |
| `:airplane:` | `feat`  | Improve offline support. |
| `:t-rex:` | `fix`  | Add backwards compatibility or fix a compatibility regression. |

## Selection Rules

1. Pick the Conventional Commit type by intent before selecting a Gitmoji.
2. Prefer one Gitmoji per subject line. Do not stack emojis.
3. Put the Gitmoji code immediately after the colon-space: `fix: :bug: correct parser crash`.
4. `fix` commits should include the best matching fix Gitmoji code: usually `:bug:`, `:ambulance:`, `:lock:`, `:rotating_light:`, `:mute:`, `:goal_net:`, `:adhesive_bandage:` or `:t-rex:`.
5. Use `feat!`, `fix!`, `refactor!`, etc. for breaking changes; add `:boom:` when useful.
6. Do not invent non-standard types like `security`, `deps` or `types`; use `fix: `, `build: ` or `refactor: `.
7. Keep the subject under 72 characters when practical and never end it with a period.

## Workflow

When creating a commit:

1. Inspect the staged diff.
2. Choose the allowed type from the table above.
3. Choose the Gitmoji code from the pairing table.
4. Write an imperative, lower-case description where natural.
5. Commit with the final subject and a body explaining why the change exists.

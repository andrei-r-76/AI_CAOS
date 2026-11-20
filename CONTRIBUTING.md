# Contributing

## Branches
```
main  ← release: only merges from dev at milestones, tagged (v0.1-basic, v0.2, v1.0-final)
 └─ dev  ← integration: all feature PRs target dev
     ├─ feat/vision-ear
     ├─ feat/llm-alerts
     ├─ feat/dashboard-ws
     └─ docs/readme-run
```
- Branch off `dev`: `git switch dev && git pull && git switch -c feat/<area>-<thing>`
- Keep branches short-lived (a few hours). Open a PR into `dev`, get 1 review, merge, delete the branch.
- **Don't squash** feature PRs. We want the individual commits in the history.
- `dev → main`: a PR with a merge commit at each working milestone, then tag it. The final one goes in before **Sunday 12:00**.
- Never push directly to `main` or `dev`, and never force-push them.

## Commits
Small and frequent, [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(vision): compute EAR from face landmarks
fix(tts): avoid overlapping voice alerts
docs(readme): add run instructions
```
Scopes: `vision`, `llm`, `tts`, `dashboard`, `main`, `docs`, `config`.

## Ownership (who edits what)
| Area | Files | Owner |
|---|---|---|
| Vision (R1) | `src/vision.py` | Cristi |
| Voice alerts | `src/tts.py` | Cristi |
| LLM reaction (R2) + integration | `src/llm.py`, `src/main.py` | Teammate 2 |
| Dashboard (R4) | `dashboard/` | Teammate 3 |
| Docs, presentation, testing | `README.md`, `docs/`, `presentation/` | Teammate 4 |
| Shared contract | `src/events.py`, `src/config.py` | changes need a PR approved by everyone affected |

If you need a change in someone else's file, open an issue or ask them. Don't edit it yourself.

## Tasks
One GitHub Issue per task. Reference it in the PR (`Closes #12`).

## Staying in sync
```bash
git switch dev && git pull
git switch feat/my-branch && git rebase dev   # or: git merge dev
```

## AI tools
- Agents follow `CLAUDE.md` (Claude Code) / `AGENTS.md` (Codex). Both have the same rules.
- Keep the `Co-Authored-By` trailers: the rules require us to declare AI tools used.
- **Never** paste anything from `context/` (organizer materials, NDA) into an AI tool.

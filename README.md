# AI_CAOS: AI Driver Monitoring Assistant

Repo for the CodeLap Hackathon 2026 (AI Drive track), Porsche Engineering Romania.

## Idea
A laptop webcam watches the driver. Computer vision detects **drowsiness** (eyes closed, PERCLOS) and **distraction** (looking away). An LLM turns the driver state into a short, context-aware in-car alert, which is spoken aloud. A live dashboard shows every detection and alert in real time.

## Architecture (planned)
```
webcam ─► vision.py ──DriverState──► llm.py ──Alert──► tts.py (voice + beep)
                │                                  │
                └──────────────► dashboard (WebSocket) ◄┘
```

## How to run
```bash
# install uv once: https://docs.astral.sh/uv/getting-started/installation/
uv sync                                 # Python 3.11 + dependencies into .venv
uv run python scripts/webcam_test.py    # webcam test (q to quit)
```

## Project layout
| Path | Purpose |
|---|---|
| `scripts/webcam_test.py` | checks that the webcam works |
| `docs/research.md` | research on existing driver-monitoring systems |
| `presentation/` | final presentation |

## Adding a dependency
```bash
uv add <package>      # updates pyproject.toml + uv.lock, commit both
```
Don't use `pip install`: everyone must stay on the same `uv.lock`.

## Team workflow
See [CONTRIBUTING.md](CONTRIBUTING.md): feature branches → `dev` → `main`.

## Tools and libraries used
- OpenCV
- AI coding assistants: Claude Code, ChatGPT/Codex (declared per the hackathon rules)

## Git Infrastructure

# Git Commands

git status -sb                      # Current branch, local edits, ahead/behind, va arata inclusiv in file explorer-ul vscode
git branch -vv                      # Inspect local branches and their upstreams
git fetch --prune origin            # Refreshes your local view of the GitHub repository named origin. It downloads new commits, updates references such as origin/dev, and removes stale
                                    references such as origin/feat/old-task after that branch is deleted on GitHub                
git diff                            # Shows unstaged edits to tracked files. git status -sb also lists new files.
git add .                           # Stages the named file, and git diff --cached previews exactly what will go into the commit.
git commit -m "MESAJ"               # Saves the staged changes locally.
git push -u origin NUME_BRANCH      # Push
git switch NUME_BRANCH              # Change to branch
git switch -c feat/NUME_TASK        # Create new branch

# USUAL WORKFLOW:
# Pull the latest dev and create your task branch
git switch dev
git pull --ff-only origin dev
git switch -c feat/sample_branch

# Make your edits, then stage, commit, and push
git add .
git commit -m "test message"
git push -u origin feat/sample_branch
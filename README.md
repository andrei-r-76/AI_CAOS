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

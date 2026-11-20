# AI_CAOS: AI Driver Monitoring Assistant

Repo for the CodeLap Hackathon 2026 (AI Drive track), Porsche Engineering Romania.

## Idea
A laptop webcam watches the driver. Computer vision detects **drowsiness** (eyes closed, PERCLOS) and **distraction** (looking away). An LLM turns the driver state into a short, context-aware in-car alert, which is spoken aloud. A live dashboard shows every detection and alert in real time.

## Architecture (phase 1)
```
webcam ─► vision.py ──DriverState──► llm.py ──Alert──► tts.py (voice + beep)
                │                                  │
                └──────────────► dashboard (WebSocket) ◄┘
```

## How to run
```bash
# 1. install uv: https://docs.astral.sh/uv/getting-started/installation/
#    (Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync                                   # installs Python 3.11 + all dependencies into .venv
uv run python scripts/webcam_test.py      # sanity check: webcam + MediaPipe (q to quit)
cp .env.example .env          # pick the LLM provider/model
# for the default local LLM: install Ollama, then `ollama pull llama3.2:3b`
uv run python -m src.main
# dashboard: http://localhost:8000
```

## Project layout
| Path | Purpose |
|---|---|
| `src/vision.py` | face landmarks, EAR, PERCLOS, head pose |
| `src/llm.py` | driver state → alert text |
| `src/tts.py` | voice alerts |
| `src/main.py` | main loop wiring everything together |
| `src/events.py` | shared data contract between modules |
| `src/config.py` | thresholds and settings |
| `dashboard/` | live web dashboard |
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
- OpenCV, MediaPipe, FastAPI, Uvicorn, pyttsx3, Ollama
- AI coding assistants: Claude Code, ChatGPT/Codex (declared per the hackathon rules)

# Agent instructions (AI_CAOS): for Codex and other agents. Keep in sync with CLAUDE.md.

Driver-monitoring assistant for a 3-day hackathon. Pipeline:
**webcam → `src/vision.py` (driver state) → `src/llm.py` (alert) → `src/tts.py` (voice) + `dashboard/` (live view)**.
It must run **live** on a laptop webcam. No pre-recorded video.

## Hard rules
1. **Never read, quote, or use anything in `context/`.** It holds confidential organizer material (NDA). It is gitignored. Leave it that way.
2. Work on a branch off `dev` (`feat/…`, `fix/…`, `docs/…`). Never commit to `main` or `dev` directly, and never force-push them.
3. Only edit the files of the module you were asked to work on (see the ownership table in `CONTRIBUTING.md`). Changes to `src/events.py` or `src/config.py` need to be called out explicitly in the PR.
4. Small commits, Conventional Commits format (`feat(vision): …`). Keep the `Co-Authored-By` trailer.
5. Keep it simple: phase 1 is EAR/PERCLOS + head pose → LLM message → TTS → dashboard. Don't add new features or dependencies unless asked.
6. Thresholds go in `src/config.py`, not hard-coded.
7. Python env is managed by **uv** only: `uv add <pkg>` for dependencies (commit `pyproject.toml` + `uv.lock`), `uv run ...` to run. Never `pip install`.
8. Modules talk only through the dataclasses in `src/events.py`.

## Run
```bash
uv sync
cp .env.example .env
uv run python -m src.main
```

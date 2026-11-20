# Agent instructions (AI_CAOS): for Codex and other agents. Keep in sync with CLAUDE.md.

Driver-monitoring assistant for a 3-day hackathon. Planned pipeline:
**webcam → `src/vision.py` (driver state) → `src/llm.py` (alert) → `src/tts.py` (voice) + `dashboard/` (live view)**.
It must run **live** on a laptop webcam. No pre-recorded video.

## Hard rules
1. **Never read, quote, or use anything in `context/`.** It holds confidential organizer material (NDA). It is gitignored. Leave it that way.
2. Work on a branch off `dev` (`feat/…`, `fix/…`, `docs/…`). Never commit to `main` or `dev` directly, and never force-push them.
3. Only edit the files of the module you were asked to work on (see the ownership table in `CONTRIBUTING.md`).
4. Small commits, Conventional Commits format (`feat(vision): …`). Keep the `Co-Authored-By` trailer.
5. Keep it simple: build one small working step at a time. Don't add new features, files or dependencies unless asked.
6. Python env is managed by **uv** only: `uv add <pkg>` for dependencies (commit `pyproject.toml` + `uv.lock`), `uv run ...` to run. Never `pip install`.

## Run
```bash
uv sync
uv run python scripts/webcam_test.py
```

# AI_CAOS: AI Driver Monitoring Assistant

Repository for the CodeLap Hackathon 2026 (AI Drive track), Porsche Engineering Romania.

## What it does

A laptop webcam watches the driver. The vision program recognizes who is driving and detects signs of
fatigue and distraction, such as closed eyes, microsleep, yawning, a dropping head, looking away from the
road or using a phone. It writes the driver state to `state.json` several times a second.

A local LLM (through Ollama) reads that state, and decides what to tell the driver and which car actions to propose, such as seat vibration or
hazard lights. Code checks every proposal before it is applied. A web dashboard shows the driver state, the
alerts, the copilot messages and a simulated vehicle.

Everything runs live on the laptop. There is no pre-recorded video.

## How the parts connect

```
webcam -> vision/face_basic.py -> state.json -> dashboard/app.py -> browser (http://127.0.0.1:5000)
                                                      |
                                                dashboard/llm.py (Ollama)
```

1. `vision/face_basic.py` analyses each camera frame and writes `state.json` every 0.2 s.
2. `dashboard/app.py` reads `state.json`, sends the data to the browser and runs the copilot.
3. `dashboard/llm.py` decides when to react and asks the LLM for a short message and actions.
   If the LLM is slow, offline or gives an unusable answer, a fixed message is used instead.
4. If the driver's eyes stay closed for 3 seconds, emergency actions (hazard lights, light braking,
   vibration) start right away without waiting for the LLM.

## Requirements

1. [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python and dependencies.
2. [Ollama](https://ollama.com/) with a model pulled, for example:
   ```bash
   ollama pull llama3.2:3b
   ```
3. A webcam.

## Setup

```bash
uv sync      # installs Python 3.11 and the dependencies into .venv
```

Create a `.env` file in the repository root (it is gitignored):

```
LLM_MODEL=llama3.2:3b
LLM_TIMEOUT_SEC=10
# OLLAMA_HOST=http://localhost:11434   only if Ollama runs somewhere else
```

`LLM_MODEL` and `LLM_TIMEOUT_SEC` are required. `dashboard/llm.py` stops at startup if they are missing.

The MediaPipe face and hand models and the face recognition model download by themselves on the first run.

Phone detection with YOLO is optional. Without it everything else still works. To turn it on:

```bash
uv add ultralytics
```

## How to run

From the repository root:

```bash
uv run python run.py
```

This starts the vision program first. When it has written a fresh `state.json`, the dashboard starts at
http://127.0.0.1:5000. Close the camera window or press Ctrl+C to stop both.

On the first run, add yourself as a driver in the panel on the right of the camera window and follow the
calibration. After that the program recognizes you by your face. A face that is not a saved driver is shown
as unknown and is not monitored.

Keys in the camera window: `q` quits, `f` shows or hides the face drawing, `c` cancels a calibration.

### Running the parts separately

```bash
uv run python vision/face_basic.py      # vision only, writes state.json
uv run python dashboard/app.py          # dashboard and copilot, reads state.json
uv run python dashboard/llm.py          # tests the copilot on a few fixed situations (needs Ollama)
```

Run them from the repository root so they all use the same `state.json`.


## Project layout

| Path | Purpose |
|---|---|
| `run.py` | Starts the vision program and then the dashboard. |
| `vision/` | Driver monitoring from the webcam. See [vision/README.md](vision/README.md) for every module, the `state.json` keys and the alert list. |
| `dashboard/app.py` | Flask server: reads `state.json`, streams data to the browser, runs the copilot, simulates the vehicle. |
| `dashboard/llm.py` | The copilot: when to react, the prompt, the LLM call, the safety checks and the fixed fallback messages. |
| `dashboard/scenarios.py` | Simulated road events. |
| `dashboard/templates/`, `dashboard/static/` | The dashboard page. |
| `scenario_demo.py` | Turns a typed sentence into a road event name. |
| `docs/research.md` | Research on existing driver monitoring systems. |
| `presentation/` | Final presentation. |

## The driver state and the alerts

The vision program decides which alerts are active and how serious they are (`info`, `warning` or
`critical`). They are written to `state.json` as `alert_level`, `alert_reason` and `active_alerts`, along
with the details behind them (seconds with eyes closed, yawns in the last 5 minutes, gaze direction and so on).

The copilot takes the most important active alert, turns it into a sentence for the prompt and uses the
alert's level as the minimum level of its answer. The full list of alerts is in
[vision/README.md](vision/README.md#alerts). When an alert is added or renamed in `vision/alerts.py`, the
`ALERTS` and `LEVEL_OF` tables in `dashboard/llm.py` need the same change.

## Safety checks on the LLM

The LLM only proposes. The code in `dashboard/llm.py` decides what is used:

| Check | What it does |
|---|---|
| Message | Must be short, must not contain unsafe phrases (for example "close your eyes") and must mention the problem. Otherwise the fixed message is used. |
| Actions | Only hazard lights on, seat or wheel vibration on, cruise control off, and braking up to 0.3. Nothing can be switched in the unsafe direction. |
| Level | Never lower than the level decided by the code, and at most one step higher. |
| Actions per level | No actions for `info`. Braking and hazard lights only for `critical`. |
| Emergency | Eyes closed for 3 seconds triggers the emergency actions without the LLM. |

## Known issues

1. `run.py --live` passes `--live` to the dashboard, but `dashboard/app.py` does not use it yet. Road events
   only come from the `curl` command above.
2. `dashboard/llm.py --live` fails because it cannot import `scenario_demo`.
3. `scenario_demo.py --alert` fails because `dashboard/llm.py` imports `scenarios` as a top level module.
4. Some dashboard metrics in `dashboard/app.py` still read state keys that the vision rework removed
   (for example `perclos`), so they stay empty.

## Adding a dependency

```bash
uv add <package>      # updates pyproject.toml and uv.lock, commit both
```

Do not use `pip install`. Everyone must stay on the same `uv.lock`.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md). Feature branches go into `dev`, and `dev` goes into `main` at milestones.

The `dev` and `main` branches are protected from deletion, and only merges are allowed into them.
Two GitHub workflows support this:

| Workflow | What it does |
|---|---|
| `.github/workflows/pr_routing.yml` | Allows pull requests only from `feat/` branches into `dev`, and from `dev` into `main`. |
| `.github/workflows/delete_feature.yml` | Deletes old, unused `feat/` branches. |

### Usual git workflow

Start a task from the latest `dev`:

```bash
git switch dev
git pull --ff-only origin dev
git switch -c feat/my-task
```

Commit and push your work:

```bash
git add <files>
git commit -m "feat(vision): short description"
git push -u origin feat/my-task
```

Bring in the latest `dev` while working on a branch:

```bash
git fetch origin
git merge origin/dev
```

### Useful git commands

| Command | What it shows or does |
|---|---|
| `git status -sb` | Current branch, local changes, ahead or behind the remote. |
| `git branch -vv` | Local branches and the remote branch each one follows. |
| `git fetch --prune origin` | Downloads new commits and removes references to branches deleted on GitHub. |
| `git diff` | Changes not yet staged. `git diff --cached` shows what will go into the next commit. |
| `git --no-pager log -10 --oneline` | The last 10 commits. |
| `git branch -a` | All local and remote branches. |

## Tools and libraries used

| Tool | Used for |
|---|---|
| OpenCV | Camera capture, drawing, face recognition (SFace). |
| MediaPipe | Face and hand landmarks. |
| Ollama | Running the LLM locally. |
| Flask | The dashboard server. |
| Ultralytics YOLO (optional) | Phone detection. |
| Claude Code, ChatGPT/Codex | AI coding assistants, declared as required by the hackathon rules. |

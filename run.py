"""Start the webcam and dashboard from the repository root."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "state.json"


def state_mtime():
    try:
        return STATE_FILE.stat().st_mtime_ns
    except OSError:
        return None


def camera_ready(previous_mtime):
    modified = state_mtime()
    if modified is None or modified == previous_mtime or time.time_ns() - modified > 2_000_000_000:
        return False

    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and isinstance(state.get("face_found"), bool)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable typed and random road scenarios")
    args = parser.parse_args()

    face_script = ROOT / "vision" / "face_basic.py"
    if not face_script.is_file():
        face_script = ROOT / "face_basic.py"
    dashboard_script = ROOT / "dashboard" / "app.py"

    for script in (face_script, dashboard_script):
        if not script.is_file():
            print(f"Missing script: {script}", file=sys.stderr)
            return 1

    children = []
    try:
        previous_mtime = state_mtime()
        print("Starting webcam. Finish calibration in the camera window if prompted.", flush=True)
        camera = subprocess.Popen([sys.executable, "-u", str(face_script)], cwd=ROOT)
        children.append(camera)
        while camera.poll() is None:
            if camera_ready(previous_mtime):
                break
            time.sleep(0.1)
        else:
            print(f"Webcam exited before writing fresh state.json (exit {camera.returncode}).", file=sys.stderr)
            return camera.returncode or 1

        print("Camera state ready. Starting dashboard at http://127.0.0.1:5000", flush=True)
        dashboard_command = [sys.executable, "-u", str(dashboard_script)]
        if args.live:
            dashboard_command.append("--live")
        dashboard = subprocess.Popen(dashboard_command, cwd=ROOT)
        children.append(dashboard)

        while camera.poll() is None and dashboard.poll() is None:
            time.sleep(0.2)
        return dashboard.returncode or camera.returncode or 0
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Could not launch the demo: {exc}", file=sys.stderr)
        return 1
    finally:
        for process in reversed(children):
            stop(process)


if __name__ == "__main__":
    raise SystemExit(main())
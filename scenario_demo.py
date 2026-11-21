"""Recognize text-based road scenarios for the indoor demo."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dashboard.scenarios import EVENTS

ALIASES = {
    "pedestrian": ("a pedestrian is crossing the road ahead",),
    "car_braking": ("the car ahead brakes hard",),
    "cyclist": ("a cyclist is on the right",),
    "animal": ("an animal is on the road",),

    # CELE DE MAI JOS TREBUIE ADAUGATE IN scenarios.py

    # "oncoming_traffic": ("incoming traffic", "there is oncoming traffic", "there is incoming traffic"),
    
    # "lane_intrusion": (
    #     "an oncoming car merges into our lane",
    #     "an incoming car merges into our lane",
    #     "an incoming car merges to our lane",
    #     "an oncoming car enters our lane",
    #     "a vehicle enters our lane",
    # ),
    
}


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def detect_scenario(text):
    text = normalize(text)
    for key, aliases in ALIASES.items():
        candidates = (key, *aliases)
        if key in EVENTS:
            candidates += (EVENTS[key]["situation"],)
        if any(text == normalize(candidate) for candidate in candidates):
            return key
    return None


def recent_driver_state(path):
    path = Path(path)
    try:
        if time.time() - path.stat().st_mtime <= 2:
            state = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                return state
    except (OSError, ValueError):
        pass
    return {}


def report(text, preview_alert):
    scenario = detect_scenario(text)
    if scenario is None:
        print("Unknown scenario. Use one of: " + ", ".join(ALIASES), file=sys.stderr)
        return False

    kind = "road_context" if scenario == "oncoming_traffic" else "road_event"
    print(json.dumps({"scenario": scenario, "kind": kind}), flush=True)

    if preview_alert and scenario == "lane_intrusion":
        print("Vehicle entering your lane.")
    elif preview_alert and scenario in EVENTS:
        from dashboard.llm import STATE_FILE, get_alert

        print(get_alert(recent_driver_state(STATE_FILE), event=scenario))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", help="Scenario key or supported description")
    parser.add_argument("--alert", action="store_true", help="Preview the text of the alert")
    args = parser.parse_args()

    if args.scenario is not None:
        return 0 if report(args.scenario, args.alert) else 2

    print("Enter a scenario, or type quit. Keys: " + ", ".join(ALIASES))
    while True:
        try:
            text = input("scenario> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.casefold() == "quit":
            return 0
        if text:
            report(text, args.alert)


if __name__ == "__main__":
    raise SystemExit(main())
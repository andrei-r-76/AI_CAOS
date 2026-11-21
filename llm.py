"""LLM alerts: turn the driver state into one short spoken warning.

Run the test:  uv run python llm.py         (fake states, needs Ollama + the model from .env pulled)
Run live:      uv run python face_basic.py  (terminal 1, writes state.json)
               uv run python llm.py --live  (terminal 2, reads state.json and prints alerts)
Settings:      copy .env.example to .env and set LLM_MODEL (e.g. llama3.2:1b on CPU, llama3.2:3b on GPU)

Vision (and the demo scenarios) decide WHEN to alert. The LLM only decides WHAT to say.
If the LLM is slow, down or says something unsafe, a fixed message is used instead.
"""
import json
import os
import re
import sys
import time

import ollama
from dotenv import load_dotenv

from scenarios import EVENTS

load_dotenv()
MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "3"))
HOLD_SEC = 1.0             # live mode: a problem must last this long before alerting
COOLDOWN_SEC = 8.0         # live mode: min seconds before repeating the same alert
STATE_FILE = "state.json"  # written by face_basic.py

SYSTEM = ("You are a car safety assistant. You speak to the driver to keep them awake "
          "and focused, and warn them about dangers on the road. "
          "Reply with ONE short spoken sentence, max 12 words.")
EXAMPLES = [
    ("eyes closed for 2 seconds", "Wake up! Open your eyes and focus on the road."),
    ("looking away from the road for 3 seconds", "Eyes back on the road, please."),
    ("holding a phone while driving", "Please put the phone down and focus on driving."),
    ("eyes often closing, getting drowsy", "You seem tired. Let's find a place to take a break."),
    ("driver is yawning", "You're yawning, maybe it's time for a short break."),
    ("driver's face is not visible to the camera", "I can't see you, please face the road."),
    ("the car ahead is braking hard", "The car ahead is braking, slow down!"),
    ("a cyclist is riding close on the right side; looking away from the road for 3 seconds",
     "Eyes on the road, there's a cyclist on your right!"),
]
FALLBACK = {
    "drowsy": "Wake up! Open your eyes and focus on the road.",
    "distracted": "Eyes back on the road, please.",
    "phone": "Please put the phone down and focus on driving.",
    "yawning": "You're yawning. Consider a short break soon.",
    "no_face": "I can't see you. Please face the road.",
}
UNSAFE = ["keep your eyes closed", "close your eyes", "go to sleep", "get some sleep", "nap",  # dangerous advice
          "i cannot", "i can't", "sorry", "i don't", "as an ai"]          # refusals / nonsense
# the alert must mention at least one of these, or it's off-topic
RELEVANT = {
    "drowsy": ["wake", "eyes", "tired", "break", "awake", "rest stop", "open"],
    "distracted": ["road", "eyes", "focus", "look", "ahead"],
    "phone": ["phone"],
    "yawning": ["yawn", "tired", "break", "rest", "coffee"],
    "no_face": ["road", "see you", "camera", "face the"],
}

client = ollama.Client(timeout=TIMEOUT_SEC)   # OLLAMA_HOST in .env if Ollama runs elsewhere


def describe(state):
    """Return (reason, situation text) or None if the driver is fine."""
    if not state.get("face_found", True):
        return "no_face", "driver's face is not visible to the camera"
    if state.get("eyes_closed_sec", 0) >= 1.5:
        return "drowsy", f"eyes closed for {state['eyes_closed_sec']:.0f} seconds"
    if state.get("perclos", 0) >= 0.15:
        return "drowsy", "eyes often closing, getting drowsy"
    if state.get("looking_away_sec", 0) >= 2:
        return "distracted", f"looking away from the road for {state['looking_away_sec']:.0f} seconds"
    if state.get("hand_near_face"):
        return "phone", "holding a phone while driving"
    if state.get("yawning"):
        return "yawning", "driver is yawning"
    return None


def clean(text):
    """Keep only the first sentence, without quotes or 'Situation:' leftovers."""
    text = re.sub(r"(?i)situation:.*?->", "", text).strip()
    quoted = re.findall(r"['\"“](.+?)['\"”]", text)
    if quoted:
        text = quoted[0]
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip()


def is_ok(text, must_mention):
    low = text.lower()
    return (2 <= len(text.split()) <= 20
            and not any(p in low for p in UNSAFE)
            and any(k in low for k in must_mention))


def warm_up():
    """Load the model into RAM at startup so the first real alert is fast.
    Uses its own long timeout: loading a bigger model can take much longer than one alert."""
    try:
        ollama.Client(timeout=120).generate(model=MODEL, prompt="hi", options={"num_predict": 1}, keep_alive="30m")
    except Exception as e:
        print(f"[llm] warm-up failed ({type(e).__name__}): alerts will use fixed messages")


def get_alert(state, event=None):
    """Return the alert text to speak, or None if no alert is needed.

    event: optional key from scenarios.EVENTS (something happening outside the car).
    """
    found = describe(state)
    if not found and not event:
        return None

    parts = []
    if event:      # the outside danger comes first, it's the most urgent
        parts.append(EVENTS[event]["situation"])
    if found:
        parts.append(found[1])
    situation = "; ".join(parts)
    if event:
        must_mention, fallback = EVENTS[event]["must_mention"], EVENTS[event]["fallback"]
    else:
        must_mention, fallback = RELEVANT[found[0]], FALLBACK[found[0]]

    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in EXAMPLES:
        messages += [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": situation})

    try:
        reply = client.chat(model=MODEL, messages=messages,
                            options={"num_predict": 30}, keep_alive="30m")
        text = clean(reply["message"]["content"])
        if is_ok(text, must_mention):
            return text
        print(f"[llm] rejected {text!r}, using fallback")
    except Exception as e:  # LLM down, timeout, model missing...
        print(f"[llm] fallback ({type(e).__name__}: {e})")
    return fallback


def live():
    """Read the state that face_basic.py writes and print an alert when needed."""
    warm_up()
    print(f"live mode, model: {MODEL}. Start face_basic.py in another terminal. Ctrl+C to stop.")
    problem, since, last_alert, last_problem = None, 0.0, 0.0, None
    while True:
        time.sleep(0.2)
        try:
            if time.time() - os.path.getmtime(STATE_FILE) > 2:
                continue                      # face_basic.py isn't running (old file)
            with open(STATE_FILE) as f:
                state = json.load(f)
        except (OSError, ValueError):
            continue                          # file missing or being written right now
        found = describe(state)
        reason = found[0] if found else None
        if reason != problem:                 # problem changed: restart the hold timer
            problem, since = reason, time.time()
        now = time.time()
        repeat = problem == last_problem and now - last_alert < COOLDOWN_SEC
        if problem and now - since >= HOLD_SEC and not repeat:
            print(f"[{problem}] {get_alert(state)}")
            last_alert, last_problem = now, problem


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
        sys.exit()

    from state import new_state

    def fake(**values):
        s = new_state()
        s.update({"face_found": True, **values})
        return s

    warm_up()
    print(f"model: {MODEL}, timeout: {TIMEOUT_SEC}s")
    tests = [  # (driver state, outside event)
        (fake(perclos=0.05), None),                       # all good -> None
        (fake(eyes_closed_sec=2.1, perclos=0.2), None),   # drowsy
        (fake(looking_away_sec=3.4), None),               # distracted
        (fake(hand_near_face=True), None),                # phone
        (fake(yawning=True), None),                       # yawning
        (fake(face_found=False), None),                   # driver not visible
        (fake(), "pedestrian"),                           # attentive driver, pedestrian
        (fake(looking_away_sec=3.0), "pedestrian"),       # distracted + pedestrian (worst case)
        (fake(hand_near_face=True), "car_braking"),       # phone + braking car
        (fake(), "cyclist"),
        (fake(eyes_closed_sec=1.8), "animal"),
    ]
    for _ in range(2):
        for s, event in tests:
            t = time.time()
            print(f"{str(event):12} {get_alert(s, event)!r:62} ({time.time() - t:.1f}s)")

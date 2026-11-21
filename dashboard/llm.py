"""LLM copilot: decides what to say AND what to do with the car.

Inputs:  state    the driver, REAL (face_basic.py -> state.json)
         context  the world around the car, any dict (scenarios.py, made by a teammate), e.g.
                  {"trip": {...}, "environment": {...}, "vehicle": {...}, "events": [...], "notes": [...]}
         history  kept here: driver problems in the last 10 min + our last decisions and if the driver reacted
Output:  {"level", "message", "actions", "reason", ...} -> voice + car (dashboard)

Safety: the LLM only PROPOSES. Guardrails decide what is executed:
  - check_actions(): only allowed actions, only in the safer direction (switch ON / brake, never the opposite)
  - the level is at least what the code decided, and the LLM may raise it by one step at most
  - actions must fit the level (ACTIONS_BY_LEVEL): no braking for a yawn
  - eyes closed >= 3 s triggers the emergency actions without waiting for the LLM:
    the main loop must call emergency(state) on EVERY tick (decide() can take seconds)

Outside events come from scenarios.EVENTS (teammate): make_event(name) turns one into a context event.
Old API kept for scenario_demo.py: get_alert(state, event) -> just the message text.

Test:  uv run python llm.py                  (fake situations, needs Ollama + LLM_MODEL from .env)
Live:  uv run python face_basic.py           (terminal 1, writes state.json)
       uv run python llm.py --live           (terminal 2: random events every 20-45 s, or type one,
                                              e.g. "pedestrian", recognized by scenario_demo.py)
"""
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, deque

import ollama
from dotenv import load_dotenv

from scenarios import EVENTS

load_dotenv()
MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "8"))   # JSON answers are longer than one sentence
HOLD_SEC = 1.0          # a driver problem must last this long before we react
COOLDOWN_SEC = 8.0      # min seconds before reacting again to the same thing
HISTORY_SEC = 600       # history window: last 10 minutes
EMERGENCY_SEC = 3.0     # eyes closed this long -> emergency actions, no LLM needed
MAX_BRAKE = 0.3         # the system never brakes harder than this (0..1)
STATE_FILE = "state.json"

LEVELS = ["info", "warning", "critical"]
# scenarios.EVENTS has no severity yet: default per event type (unknown events -> warning)
EVENT_SEVERITY = {"pedestrian": "critical", "car_braking": "critical", "animal": "critical", "cyclist": "warning"}
EVENT_SEC = 6               # live test: how long an outside event stays active
RANDOM_EVENT_SEC = (20, 45)  # live test: a random event every 20-45 s
# driver problem -> fixed fallback message, words a good message must contain
FALLBACK = {
    "drowsy": "Wake up! Open your eyes and focus on the road.",
    "distracted": "Eyes back on the road, please.",
    "phone": "Please put the phone down and focus on driving.",
    "yawning": "You're yawning. Consider a short break soon.",
    "hypnosis": "Stay sharp. Move your eyes around and take a break soon.",
    "no_face": "I can't see you. Please face the road.",
}
RELEVANT = {
    "drowsy": ["wake", "eyes", "tired", "break", "awake", "open", "stop", "rest"],
    "distracted": ["road", "eyes", "focus", "look", "ahead"],
    "phone": ["phone"],
    "yawning": ["yawn", "tired", "break", "rest", "coffee"],
    "hypnosis": ["break", "focus", "sharp", "awake", "rest", "eyes"],
    "no_face": ["road", "see you", "camera", "face the"],
}
UNSAFE = ["keep your eyes closed", "close your eyes", "go to sleep", "get some sleep", "nap",
          "i cannot", "i can't", "sorry", "i don't", "as an ai"]
# proportionality: which actions each level may use
ACTIONS_BY_LEVEL = {"info": set(),
                    "warning": {"seat_vibration", "wheel_vibration", "cruise_control"},
                    "critical": {"seat_vibration", "wheel_vibration", "cruise_control", "hazard_lights", "brake"}}
EMERGENCY = {"hazard_lights": True, "brake": MAX_BRAKE, "cruise_control": False,
             "wheel_vibration": True, "seat_vibration": True}

SYSTEM = """You are the safety copilot of a car. You get the driver's state from a camera, what happened \
in the last minutes, your last decision, and the situation around the car. Decide how to respond.
Speak directly TO the driver ("you"). Be proportionate: a yawn needs words, not brakes.
Reply ONLY with JSON:
{"level": "info|warning|critical", "message": "one short spoken sentence for the driver, max 15 words",
 "actions": [{"name": "...", "value": ...}], "reason": "why, max 15 words"}
Allowed actions: hazard_lights true, seat_vibration true, wheel_vibration true, cruise_control false, brake 0.1-0.3.
Use actions only when needed. If the driver ignored your last warning, escalate. Use the history and the situation."""
EXAMPLE_IN = """driver now: yawning
driver last 10 min: 2x yawning, 1x drowsy
your last decision 40 s ago: "You seem tired, take a break soon." actions={} -> the driver still has this problem
trip: driving_min=150, clock=00:10
environment: road=highway, light=night"""
EXAMPLE_OUT = json.dumps({
    "level": "warning", "message": "You're still yawning after midnight. Let's stop at the next rest area.",
    "actions": [{"name": "seat_vibration", "value": True}],
    "reason": "Third fatigue sign in 10 minutes and the earlier warning was ignored."})

client = ollama.Client(timeout=TIMEOUT_SEC)   # OLLAMA_HOST in .env if Ollama runs elsewhere


def describe(state):
    """Driver problem -> (reason, text, minimum level), or None if the driver is fine."""
    if not state.get("face_found", True):
        return "no_face", "driver's face is not visible to the camera", "warning"
    if state.get("eyes_closed_sec", 0) >= 1.5:
        return "drowsy", f"eyes closed for {state['eyes_closed_sec']:.1f} seconds", "critical"
    if state.get("perclos", 0) >= 0.15:
        return "drowsy", "eyes often closing, getting drowsy", "warning"
    if state.get("looking_away_sec", 0) >= 2:
        return "distracted", f"looking away from the road for {state['looking_away_sec']:.0f} seconds", "warning"
    if state.get("vats_alert"):
        return "distracted", "glancing away from the road too often", "warning"
    if state.get("hand_near_face"):
        return "phone", "holding a phone while driving", "critical"
    if state.get("hypnosis"):
        return "hypnosis", "highway hypnosis: gaze fixed, rare blinking", "warning"
    if state.get("yawning"):
        return "yawning", "driver is yawning", "info"
    return None


class History:
    """What happened in the last HISTORY_SEC: driver problems and our own decisions."""

    def __init__(self):
        self.problems = deque()    # (time, reason)
        self.decisions = deque()   # (time, trigger, decision)

    def _trim(self, now):
        for q in (self.problems, self.decisions):
            while q and now - q[0][0] > HISTORY_SEC:
                q.popleft()

    def add_problem(self, reason, now):
        self.problems.append((now, reason))

    def add_decision(self, trigger, decision, now):
        self.decisions.append((now, trigger, decision))

    def summary(self, current, now):
        """Short text for the prompt. current = the problem/event we're reacting to now."""
        self._trim(now)
        counts = Counter(r for _, r in self.problems)
        lines = ["driver last 10 min: " + (", ".join(f"{n}x {r}" for r, n in counts.items()) or "no problems")]
        if self.decisions:
            t, trigger, d = self.decisions[-1]
            reacted = "the driver still has this problem" if trigger == current else "the driver reacted"
            lines.append(f'your last decision {now - t:.0f} s ago: "{d["message"]}" '
                         f"actions={d['actions']} -> {reacted}")
        return "\n".join(lines)


def render(context):
    """Any context dict -> short text lines. New keys show up automatically."""
    lines = []
    for key, value in context.items():
        if isinstance(value, dict) and value:
            lines.append(f"{key}: " + ", ".join(f"{k}={v}" for k, v in value.items()))
        elif isinstance(value, list) and value:
            items = [v.get("text") or v.get("situation") or str(v) if isinstance(v, dict) else str(v) for v in value]
            lines.append(f"{key}: " + "; ".join(items))
    return "\n".join(lines)


def event_info(event):
    """Read an outside event, whatever format the teammate uses (dict or plain string)."""
    if not isinstance(event, dict):
        return str(event), "warning", [], f"Careful: {event}."
    text = event.get("text") or event.get("situation") or str(event)
    level = event.get("severity") if event.get("severity") in LEVELS else "warning"
    keywords = event.get("keywords") or event.get("must_mention") or []
    return text, level, keywords, event.get("fallback") or f"Careful: {text}."


def make_event(name, now=None):
    """scenarios.EVENTS[name] -> a context event (adds type, severity and an expiry time)."""
    now = now or time.time()
    return {**EVENTS[name], "type": name, "until": now + EVENT_SEC,
            "severity": EVENTS[name].get("severity") or EVENT_SEVERITY.get(name, "warning")}


def clean(text):
    text = str(text).strip().strip("\"'")
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()


def is_ok(text, must_mention):
    low = text.lower()
    return (2 <= len(text.split()) <= 20 and not any(p in low for p in UNSAFE)
            and (not must_mention or any(k in low for k in must_mention)))


def check_actions(proposed):
    """Guardrail: keep only allowed actions, and only in the safer direction."""
    if isinstance(proposed, dict):                       # {"brake": 0.2} is also accepted
        proposed = [{"name": k, "value": v} for k, v in proposed.items()]
    safe = {}
    for a in proposed if isinstance(proposed, list) else []:
        if not isinstance(a, dict):
            continue
        name, value = a.get("name"), a.get("value")
        on = value is True or str(value).lower() in ("true", "on", "1")
        if name == "brake":
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                safe["brake"] = round(min(value, MAX_BRAKE), 2)
        elif name == "cruise_control" and not on:        # may only switch it OFF
            safe["cruise_control"] = False
        elif name in ("hazard_lights", "seat_vibration", "wheel_vibration") and on:   # may only switch ON
            safe[name] = True
    return safe


def warm_up():
    """Load the model at startup (own long timeout: loading can take much longer than one answer)."""
    try:
        ollama.Client(timeout=120).generate(model=MODEL, prompt="hi", options={"num_predict": 1}, keep_alive="30m")
    except Exception as e:
        print(f"[llm] warm-up failed ({type(e).__name__}): decisions will use fixed messages")


def decide(state, context, history, now=None):
    """Return a decision dict, or None if nothing needs a reaction."""
    now = now or time.time()
    found = describe(state)
    events = context.get("events") or []
    if not found and not events:
        return None

    # the code decides the minimum level and the fallback; an outside danger comes first
    levels = [found[2]] if found else []
    if events:     # focus on the most dangerous event; if equal, the newest one
        main = max(enumerate(events), key=lambda ie: (LEVELS.index(event_info(ie[1])[1]), ie[0]))[1]
        text, ev_level, keywords, fallback = event_info(main)
        trigger = main.get("type", text) if isinstance(main, dict) else text
        levels.append(ev_level)
    else:
        trigger, keywords, fallback = found[0], RELEVANT[found[0]], FALLBACK[found[0]]
    min_level = max(levels, key=LEVELS.index)

    prompt = "\n".join([f"driver now: {found[1] if found else 'attentive'}",
                        history.summary(trigger, now), render(context)])
    if keywords:   # tell the model the words is_ok() will check for (real words only, max 3)
        words = set(re.findall(r"[a-z']+", fallback.lower()))           # e.g. "brak" is a stem, not a word
        shown = ([k for k in keywords if k in words]
                 or [k for k in keywords if len(k) > 3 and " " not in k] or list(keywords))[:3]
        prompt += ("\nWrite the message directly to the driver, using \"you\" (never \"the driver\"). "
                   f"Use at least one of these words in the message: {', '.join(shown)}. "
                   "Do not invent distances, speeds or numbers.")
    decision = {"level": min_level, "message": fallback, "actions": {}, "reason": "fixed rule",
                "trigger": trigger, "source": "fallback"}
    try:
        reply = client.chat(model=MODEL, format="json", keep_alive="30m", options={"num_predict": 150},
                            messages=[{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": EXAMPLE_IN},
                                      {"role": "assistant", "content": EXAMPLE_OUT},
                                      {"role": "user", "content": prompt}])
        d = json.loads(reply["message"]["content"])
        message = clean(d.get("message", ""))
        if is_ok(message, keywords):
            decision.update(message=message, source="llm")
        else:
            print(f"[llm] rejected message {message!r}")
        decision["actions"] = check_actions(d.get("actions"))
        if d.get("level") in LEVELS:                       # may raise the level by one step, never lower it
            wanted = min(LEVELS.index(d["level"]), LEVELS.index(min_level) + 1)
            decision["level"] = LEVELS[max(wanted, LEVELS.index(min_level))]
        decision["actions"] = {k: v for k, v in decision["actions"].items()
                               if k in ACTIONS_BY_LEVEL[decision["level"]]}
        if d.get("reason"):
            decision["reason"] = clean(d["reason"])[:150]
    except Exception as e:                                 # LLM down, timeout, invalid JSON...
        print(f"[llm] fallback ({type(e).__name__}: {e})")

    if state.get("eyes_closed_sec", 0) >= EMERGENCY_SEC:   # deterministic emergency, LLM can't cancel it
        decision["actions"].update(EMERGENCY)
        decision["level"] = "critical"
    history.add_decision(trigger, decision, now)
    return decision


class Copilot:
    """Decides WHEN to ask the LLM: a driver problem that lasts HOLD_SEC, or a new outside event."""

    def __init__(self):
        self.history = History()
        self.problem, self.since, self.counted = None, 0.0, False
        self.last = {}            # trigger -> time of our last reaction
        self.seen_events = set()

    def should_decide(self, state, context, now=None):
        now = now or time.time()
        found = describe(state)
        reason = found[0] if found else None
        if reason != self.problem:                         # problem changed: restart the hold timer
            self.problem, self.since, self.counted = reason, now, False
        held = reason and now - self.since >= HOLD_SEC
        if held and not self.counted:                      # count each problem once in the history
            self.history.add_problem(reason, now)
            self.counted = True
        ids = {json.dumps(e, sort_keys=True, default=str) if isinstance(e, dict) else str(e)
               for e in context.get("events") or []}
        new_event = bool(ids - self.seen_events)
        self.seen_events = ids
        due = held and now - self.last.get(reason, 0) >= COOLDOWN_SEC
        if new_event or due:
            if reason:
                self.last[reason] = now
            return True
        return False

    def decide(self, state, context, now=None):
        return decide(state, context, self.history, now)


def get_alert(state, event=None):
    """Old one-sentence API, still used by scenario_demo.py --alert: returns only the message (or None)."""
    context = {"events": [make_event(event)]} if event else {}
    d = decide(state or {}, context, History())
    return d["message"] if d else None


def emergency(state):
    """Call on every tick, BEFORE/independent of decide(): returns the emergency actions or None.
    Deterministic and instant: an emergency must never wait for the LLM."""
    if state and state.get("eyes_closed_sec", 0) >= EMERGENCY_SEC:
        return dict(EMERGENCY)
    return None


def read_state(path=STATE_FILE):
    """The driver state written by face_basic.py, or None if it's missing/old (vision not running)."""
    try:
        if time.time() - os.path.getmtime(path) > 2:
            return None
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def live():
    """Test loop: real driver (state.json) + outside events, random or typed. Ctrl+C to stop."""
    from scenario_demo import detect_scenario       # teammate's text -> event recognizer

    warm_up()
    copilot = Copilot()
    context = {"vehicle": {"speed_kmh": 100}, "events": []}
    typed = []                                        # events typed in the terminal

    def read_input():
        for line in sys.stdin:
            name = detect_scenario(line.strip())
            if name:
                typed.append(name)
            elif line.strip():
                print(f"unknown scenario, try one of: {', '.join(EVENTS)}")

    threading.Thread(target=read_input, daemon=True).start()
    print(f"live mode, model: {MODEL}. Start face_basic.py in another terminal.\n"
          f"Random events every {RANDOM_EVENT_SEC[0]}-{RANDOM_EVENT_SEC[1]} s, or type one: {', '.join(EVENTS)}")
    next_random, in_emergency = time.time() + random.uniform(*RANDOM_EVENT_SEC), False
    while True:
        time.sleep(0.2)
        now = time.time()
        context["events"] = [e for e in context["events"] if e["until"] > now]   # expire old events
        while typed:
            name = typed.pop(0)
            print(f"[event] {name}")
            context["events"].append(make_event(name, now))
            next_random = now + random.uniform(*RANDOM_EVENT_SEC)

        if now >= next_random:
            name = random.choice(list(EVENTS))
            print(f"[event] {name}")
            context["events"].append(make_event(name, now))
            next_random = now + random.uniform(*RANDOM_EVENT_SEC)


        state = read_state() or {}                    # no vision running -> only events are judged
        actions = emergency(state)
        if actions and not in_emergency:
            print(f"[EMERGENCY] {FALLBACK['drowsy']} {actions}")
        in_emergency = bool(actions)
        if copilot.should_decide(state, context):
            d = copilot.decide(state, context)
            print(f"[{d['level']}] {d['message']}  actions={d['actions']}  ({d['source']}: {d['reason']})")


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
        sys.exit()

    warm_up()
    print(f"model: {MODEL}, timeout: {TIMEOUT_SEC}s")
    history = History()
    ctx = {"trip": {"driving_min": 130, "clock": "23:40"},
           "environment": {"road": "highway", "weather": "rain", "light": "night"},
           "vehicle": {"speed_kmh": 110, "cruise_control": True}, "events": []}
    ok = {"face_found": True}
    tests = [
        ("yawning, first time", {**ok, "yawning": True}, ctx),
        ("yawning again (history)", {**ok, "yawning": True}, ctx),
        ("looking away", {**ok, "looking_away_sec": 3.2}, ctx),
        ("pedestrian + attentive", ok, {**ctx, "events": [
            {"type": "pedestrian", "text": "a pedestrian is crossing ahead, not at a crosswalk",
             "severity": "critical", "keywords": ["pedestrian", "crossing", "person"]}]}),
        ("eyes closed 3.5 s (emergency)", {**ok, "eyes_closed_sec": 3.5}, ctx),
    ]
    for name, state, context in tests:
        history.add_problem(describe(state)[0], time.time()) if describe(state) else None
        t = time.time()
        d = decide(state, context, history)
        print(f"\n--- {name} ({time.time() - t:.1f}s, {d['source']})\n{json.dumps(d, indent=1)}")

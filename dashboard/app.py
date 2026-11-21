"""Demo server: feeds the dashboard until the real face program is ready.

    templates/dashboard.html   static/style.css   static/script.js   <- the dashboard files
    demo_server.py  llm.py  scenarios.py  (+ scenario_demo.py)       <- next to each other

Run:   uv run python demo_server.py        then open http://localhost:5000

Where the DRIVER data comes from (checked every tick):
  1. state.json written by face_basic.py, if it is fresh (< 2 s old)   -> real camera
  2. otherwise a built-in simulator controlled by the demo buttons      -> fake driver
So when face_basic.py starts, the dashboard switches to it by itself. Nothing else changes.

Flow every 0.2 s:  driver state -> dashboard "metrics" / "alert" events
                   emergency(state) -> actions applied instantly (never waits for the LLM)
                   Copilot.should_decide() -> llm.decide() in a background thread
                   decision -> "assistant" event + actions applied to the (simulated) vehicle
"""
import json
import queue
import random
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

from llm import EVENTS, Copilot, emergency, make_event, read_state, warm_up

app = Flask(__name__)
TICK_SEC = 0.2
METRICS_SEC = 1.0
ACTION_HOLD_SEC = 8            # how long copilot actions stay on in the simulated vehicle
LEVEL_TO_SEVERITY = {"info": 1, "warning": 2, "critical": 3}
USE_SIMULATOR = False       # True = fake driver when the camera is not running

# ---------------------------------------------------------------- live stream (SSE)
clients, clients_lock = [], threading.Lock()


def publish(event):
    data = json.dumps(event)
    with clients_lock:
        for q in clients:
            q.put(data)


@app.route("/")
def index():
    return render_template("dashboard.html", lang="en")


@app.route("/events")
def events():
    q = queue.Queue()
    with clients_lock:
        clients.append(q)

    def stream():
        try:
            while True:
                try:
                    yield f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with clients_lock:
                clients.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- simulated driver
# keys follow what llm.describe() reads from state.json
SCENARIOS = {
    "normal":     {},
    "yawn":       {"yawning": True, "mar": 0.7},
    "drowsy":     {"perclos": 0.22, "ear": 0.19, "blink_rate": 6},
    "microsleep": {"eyes_closed_sec": 3.5, "ear": 0.08},
    "distracted": {"looking_away_sec": 3.0, "yaw": 42},
    "phone":      {"hand_near_face": True, "pitch": -25},
    "eyerub":     {"eyerub": True},
    "noface":     {"face_found": False},
}
sim = {"name": "auto", "auto_until": 0.0, "auto_pick": "normal"}


def sim_state(now):
    name = sim["name"]
    if name == "auto":                                   # alternate between a problem and normal driving
        if now >= sim["auto_until"]:
            if sim["auto_pick"] == "normal":
                sim["auto_pick"] = random.choice([n for n in SCENARIOS if n != "normal"])
                sim["auto_until"] = now + 9
            else:
                sim["auto_pick"] = "normal"
                sim["auto_until"] = now + 12
        name = sim["auto_pick"]
    state = {"face_found": True, "eyes_closed_sec": 0.0, "perclos": 0.04, "looking_away_sec": 0.0,
             "yawning": False, "hand_near_face": False, "hypnosis": False, "vats_alert": False,
             "ear": 0.30, "yaw": random.uniform(-6, 6), "pitch": random.uniform(-4, 4),
             "mar": 0.10, "blink_rate": 15, "blink_ms": 150}
    state.update(SCENARIOS[name])
    return state


@app.route("/api/scenario", methods=["GET", "POST"])
def scenario():
    if request.method == "POST":
        name = (request.get_json(silent=True) or {}).get("name")
        if name != "auto" and name not in SCENARIOS:
            return jsonify(error="unknown scenario"), 400
        sim["name"], sim["auto_until"] = name, 0.0
    return jsonify(current=sim["name"], scenarios=["auto"] + list(SCENARIOS))


outside_events = []                                      # active outside events (pedestrian, ...)


@app.route("/api/event/<name>", methods=["POST"])
def trigger_event(name):
    if name not in EVENTS:
        return jsonify(error="unknown event", events=list(EVENTS)), 400
    outside_events.append(make_event(name))
    return jsonify(ok=True)


# ---------------------------------------------------------------- simulated vehicle
DEFAULT_VEHICLE = {"speed_kmh": 100.0, "throttle": 0.3, "brake": 0.0, "hazard_lights": False,
                   "seat_vibration": False, "wheel_vibration": False, "cruise_control": True}
vehicle = dict(DEFAULT_VEHICLE)
vehicle["throttle"] = 0.0          # starts locked at 0 until the driver's face has been seen (see below)
action_until = {}

# --- throttle / brake behaviour (times in seconds) ---
THROTTLE_DOWN_PER_S = 0.03    # lifting off is quick: full throttle released in about 1 s
THROTTLE_UP_PER_S = 0.06       # accelerating again is slow: 0 -> 0.3 takes about 5 s
BRAKE_UP_PER_S = 0.08           # brake builds up gradually (0 -> AUTO_BRAKE in 1 s)
BRAKE_DOWN_PER_S = 1.0         # and is released a bit faster
AUTO_BRAKE = 0.5               # brake level applied automatically while the driver's face is missing
BRAKE_AFTER_FACE_LOSS_S = 10.0 # the automatic brake starts only after the driver has been out of frame this long
COAST_DECEL_KMH_S = 2.0        # slowing down with no throttle and no brake (engine braking / drag), km/h per second
FACE_LOSS_GRACE_S = 5.0        # a face missing for less than this cuts nothing (tracking flicker); after it the throttle locks at 0
REARM_DELAY_S = 5.0            # after the face is back, wait this long before the throttle may rise again

thr = {"lost_since": None, "back_since": None, "locked": True, "braking": False}   # locked: throttle held at 0
commanded = {}                 # throttle / brake requested by assistant actions (ramped to, never jumped to)


def apply_actions(actions, now):
    for name, value in actions.items():
        if name in ("throttle", "brake"):
            commanded[name] = value                # step_vehicle ramps toward it
        else:
            vehicle[name] = value
        action_until[name] = now + ACTION_HOLD_SEC


def _ramp(value, target, up, down, dt):
    rate = up if target > value else down
    return value + max(-rate * dt, min(rate * dt, target - value))


def step_vehicle(now, dt, face_ok):
    for name in [n for n, t in action_until.items() if now > t]:     # actions wear off
        if name in ("throttle", "brake"):
            commanded.pop(name, None)
        else:
            vehicle[name] = DEFAULT_VEHICLE[name]
        del action_until[name]

    # lock logic: no face -> throttle locked at 0; it unlocks only after the face has been back for REARM_DELAY_S
    if not face_ok:
        if thr["lost_since"] is None:
            thr["lost_since"] = now
        if now - thr["lost_since"] >= FACE_LOSS_GRACE_S:
            thr["locked"] = True
        thr["back_since"] = None
    else:
        thr["lost_since"] = None
        if thr["locked"]:
            if thr["back_since"] is None:
                thr["back_since"] = now
            if now - thr["back_since"] >= REARM_DELAY_S:
                thr["locked"] = False

    # driver out of frame for BRAKE_AFTER_FACE_LOSS_S: the car brakes by itself and the hazard lights come on
    braking_for_face = (not face_ok) and thr["lost_since"] is not None \
                       and now - thr["lost_since"] >= BRAKE_AFTER_FACE_LOSS_S
    if braking_for_face != thr["braking"]:
        thr["braking"] = braking_for_face
        print("[vehicle] driver missing -> automatic braking ON" if braking_for_face
              else "[vehicle] driver back -> automatic braking OFF")
    brake_target = max(commanded.get("brake", 0.0), AUTO_BRAKE if braking_for_face else 0.0)
    vehicle["brake"] = max(0.0, min(1.0, _ramp(vehicle["brake"], brake_target, BRAKE_UP_PER_S, BRAKE_DOWN_PER_S, dt)))
    if braking_for_face:
        vehicle["hazard_lights"] = True
    elif "hazard_lights" not in action_until:
        vehicle["hazard_lights"] = DEFAULT_VEHICLE["hazard_lights"]

    # throttle target: 0 while locked; otherwise it shrinks as the brake grows (brake 100% -> throttle 0)
    desired = commanded.get("throttle", DEFAULT_VEHICLE["throttle"])
    target = 0.0 if thr["locked"] else desired * (1.0 - min(1.0, vehicle["brake"]))
    vehicle["throttle"] = max(0.0, _ramp(vehicle["throttle"], target, THROTTLE_UP_PER_S, THROTTLE_DOWN_PER_S, dt))

    # speed: the brake slows the car; otherwise it follows the throttle (no throttle = coasting down)
    if vehicle["brake"] > 0:
        vehicle["speed_kmh"] = max(0.0, vehicle["speed_kmh"] - vehicle["brake"] * 25 * dt)
    else:
        cruise = DEFAULT_VEHICLE["speed_kmh"] * vehicle["throttle"] / DEFAULT_VEHICLE["throttle"]
        if cruise >= vehicle["speed_kmh"]:           # accelerating towards the speed the throttle asks for
            vehicle["speed_kmh"] += (cruise - vehicle["speed_kmh"]) * 0.1 * dt
        else:                                        # throttle released: slow coasting, not an instant slowdown
            vehicle["speed_kmh"] = max(cruise, vehicle["speed_kmh"] - COAST_DECEL_KMH_S * dt)


# ---------------------------------------------------------------- state -> dashboard events
ALERT_TEXT = {"microsleep": "Eyes closed too long. Wake up!", "drowsy": "Signs of drowsiness detected.",
              "yawn": "Yawning detected.", "distracted": "Eyes off the road.",
              "phone": "Phone in hand while driving.", "eyerub": "Eye rubbing detected.",
              "noface": "Driver not visible."}
ALERT_SEVERITY = {"yawn": 1, "microsleep": 3, "phone": 3}
ALERT_LEVEL = {"none": 0, "ok": 0, "info": 1, "warning": 2, "critical": 3}

def to_dashboard(s):
    """llm/state.json names -> the names the dashboard reads."""
    flags = []
    eyerub = bool(s.get("eyerub") or s.get("hand_at_eyes"))
    phone = bool(s.get("phone_detected") or s.get("hand_at_ear") or (s.get("hand_near_face") and not s.get("hand_at_eyes")))

    if not s.get("face_found", True):
        flags.append("noface")
    else:
        if s.get("eyes_closed_sec", 0) >= 1.5:
            flags.append("microsleep")
        elif s.get("perclos", 0) >= 0.15:
            flags.append("drowsy")
        if s.get("looking_away_sec", 0) >= 2 or s.get("vats_alert"):
            flags.append("distracted")
        if phone:
            flags.append("phone")
        if eyerub:
            flags.append("eyerub")
        if s.get("yawning"):
            flags.append("yawn")
    state = "danger" if {"microsleep", "phone"} & set(flags) else "warn" if flags else "ok"
    ear = s.get("ear")

    metrics = {"ear": s.get("eyes_closed_sec"), "perclos": s.get("perclos"), "blink_rate": s.get("blink_rate"),
               "blink_ms": s.get("blink_ms") or (s.get("avg_blink_sec") or 0) * 1000,
               "squint": bool(s.get("squinting") or s.get("eyes_partially_closed")),
               "eyerub": bool(s.get("eyerub") or s.get("hand_at_eyes")),
               "yaw": s.get("head_yaw") if s.get("head_yaw") is not None else s.get("yaw"),
               "pitch": s.get("head_pitch") if s.get("head_pitch") is not None else s.get("pitch"),
               "mar": s.get("mar"), "yawning": bool(s.get("yawning")),
               "phone": bool(s.get("hand_near_face") or s.get("phone_use")),
               "face": s.get("face_found", True),
               "level": ALERT_LEVEL.get(s.get("alert_level"), 0)}
    return flags, state, metrics


# ---------------------------------------------------------------- main loop
def run_decision(copilot, state, context, busy):
    """llm.decide() can take seconds, so it runs off the main loop."""
    try:
        try:
            d = copilot.decide(state, context)
        except RuntimeError:        # history deque changed by the main loop while the prompt was built
            d = copilot.decide(state, context)
        if d:
            apply_actions(d["actions"], time.time())
            publish({"type": "assistant", "t": time.time(), "text": d["message"], "lang": "en",
                     "severity": LEVEL_TO_SEVERITY[d["level"]], "trigger": d.get("trigger")})
            print(f"[{d['level']}] {d['message']}  actions={d['actions']}  ({d['source']}: {d['reason']})")
    except Exception as e:
        print(f"[server] decision failed: {type(e).__name__}: {e}")
    finally:
        busy.clear()


def loop():
    copilot, busy = Copilot(), threading.Event()
    prev_flags, yawn_times, last_metrics, last_source, last_t = set(), [], 0.0, None, time.time()
    while True:
        time.sleep(TICK_SEC)
        now = time.time()
        real = read_state()
        if real is None and not USE_SIMULATOR:
            step_vehicle(now, now - last_t, False)   # no camera data = no face: throttle locked, car brakes
            last_t = now
            if now - last_metrics >= METRICS_SEC:    # the dashboard must still see the vehicle react
                last_metrics = now
                flags, level, metrics = to_dashboard({"face_found": False})
                publish({"type": "metrics", "t": now, "state": level, "flags": flags, "metrics": metrics,
                         "yawns": len(yawn_times), "source": "none",
                         "vehicle": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in vehicle.items()}})
            continue
        source = "camera" if real else "simulator"
        if source != last_source:
            print(f"[server] driver data from: {source}")
            last_source = source
        state = real or sim_state(now)

        step_vehicle(now, now - last_t, bool(state.get("face_found", False)))
        last_t = now
        urgent = emergency(state)                       # instant, independent of the LLM
        if urgent:
            apply_actions(urgent, now)

        flags, level, metrics = to_dashboard(state)
        for f in flags:                                  # one alert per newly appearing problem
            if f not in prev_flags:
                if f == "yawn":
                    yawn_times.append(now)
                publish({"type": "alert", "kind": f, "severity": ALERT_SEVERITY.get(f, 2), "t": now,
                         "text": ALERT_TEXT[f], "lang": "en"})
        prev_flags = set(flags)

        if now - last_metrics >= METRICS_SEC:
            last_metrics = now
            yawn_times[:] = [t for t in yawn_times if now - t < 300]
            publish({"type": "metrics", "t": now, "state": level, "flags": flags, "metrics": metrics,
                     "yawns": len(yawn_times), "source": source,
                     "vehicle": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in vehicle.items()}})

        outside_events[:] = [e for e in outside_events if e["until"] > now]
        context = {"trip": {"driving_min": 120}, "environment": {"road": "highway", "light": "night"},
                   "vehicle": dict(vehicle), "events": list(outside_events)}
        if not busy.is_set() and copilot.should_decide(state, context):   # not while a decision is running
            busy.set()
            threading.Thread(target=run_decision, args=(copilot, state, context, busy), daemon=True).start()


if __name__ == "__main__":
    warm_up()
    threading.Thread(target=loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
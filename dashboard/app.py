"""Driver monitoring dashboard backend.

R1 detector --POST /api/event--> analyse() --> escalation ladder + LLM alerts (R2) --SSE /events--> dashboard
No webcam? DEMO=1 (default) runs a simulator you can steer from the dashboard buttons.
Env: ASSIST_LANG=en|ro, OLLAMA_MODEL=<model> (local LLM) or ANTHROPIC_API_KEY (hosted); neither -> fixed phrases.
"""
import json, os, queue, random, threading, time, unicodedata, urllib.request
from collections import deque
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
subscribers, last_alert, lock = [], {}, threading.Lock()
latest_frame, live_until = None, 0     # newest detector JPEG; while a detector posts, the simulator stays quiet
LANG = os.getenv("ASSIST_LANG", "en")
COOLDOWN, WINDOW, CALM = 8, 120, 20    # seconds: alert repeat delay, escalation memory, calm needed to de-escalate
history, yawns = deque(), deque()      # (time, severity) of recent alerts / yawn start times
level, prev_flags, danger_since, calm_since = 0, set(), None, None

WEIGHTS = {"microsleep": 90, "phone": 70, "drowsy": 55, "nod": 45, "distracted": 40, "yawn": 30}
SEVERITY = {"microsleep": 3, "phone": 3, "drowsy": 2, "nod": 2, "distracted": 2, "yawn": 1}
ACTIONS = {0: [], 1: ["voice"], 2: ["voice", "dialog", "wheel"], 3: ["wheel", "brake", "alarm"], 4: ["stop", "sos"]}
TEXT = {
    "en": {"microsleep": "Your eyes are closing. Please pull over safely and rest now.",
           "phone": "Phone detected. Please put it down and keep your eyes on the road.",
           "drowsy": "You seem tired. Want me to find a rest area nearby?",
           "nod": "Your head is dropping. Open a window or take a short break.",
           "distracted": "You are looking away from the road. Eyes forward, please.",
           "yawn": "You've yawned {n} time(s) in the last 5 minutes. Want me to find a rest area?",
           0: "Driver recovered. Assistance off.", 1: "Gentle voice warning.",
           2: "Stay with me. What day is it today? Answer me out loud.", "awake": "Good, you sound awake. Want me to find a rest area?",
           3: "Loud alarm, steering wheel vibrating, brakes pulsing.",
           4: "Driver unresponsive. Slowing down, hazard lights on, stopping safely. SOS sent to emergency services (simulated)."},
    "ro": {"microsleep": "Ochii ți se închid. Oprește în siguranță și odihnește-te acum.",
           "phone": "Telefon detectat. Lasă-l jos și uită-te la drum.",
           "drowsy": "Pari obosit. Vrei să-ți găsesc o zonă de odihnă aproape?",
           "nod": "Îți pică capul. Deschide geamul sau fă o scurtă pauză.",
           "distracted": "Nu te uiți la drum. Privirea înainte, te rog.",
           "yawn": "Ai căscat de {n} ori în ultimele 5 minute. Vrei să-ți găsesc o zonă de odihnă?",
           0: "Șofer revenit. Asistența s-a oprit.", 1: "Avertisment vocal blând.",
           2: "Rămâi cu mine. Ce zi este azi? Răspunde-mi cu voce tare.", "awake": "Bine, pari treaz. Vrei să-ți găsesc o zonă de odihnă?",
           3: "Alarmă puternică, volan care vibrează, frâne intermitente.",
           4: "Șofer nereactiv. Încetinesc, avarii pornite, opresc în siguranță. SOS trimis către autorități (simulat)."},
}
SYSTEM = ("You are an in-car assistant. In one short spoken sentence (max 20 words), react to the "
          f"driver-monitoring detection. Be calm, direct, suggest one concrete action. Reply in {'Romanian' if LANG == 'ro' else 'English'}.")
POIS = [
    ("Cluj Botanical Garden", "Large garden with a Japanese section and greenhouses, 10 min from the centre.", {"garden", "park", "nature", "walk", "relax", "gradina", "natura", "plimbare"}),
    ("Piata Unirii and St. Michael's Church", "The historic heart of Cluj, with cafes around the square.", {"history", "church", "center", "centre", "coffee", "cafe", "istorie", "biserica", "centru", "cafea", "cafenea"}),
    ("Cetatuia Hill", "Hilltop viewpoint with a panorama over the whole city.", {"view", "panorama", "hill", "sunset", "photo", "priveliste", "deal", "apus", "poza"}),
    ("Salina Turda", "Underground salt mine turned attraction, about 40 min away.", {"salt", "mine", "unique", "underground", "sare", "mina", "salina", "subteran"}),
]
SCEN = {"normal": {}, "yawn": dict(mar=.75), "drowsy": dict(ear=.21, perclos=.3), "microsleep": dict(ear=.10, perclos=.5, pitch=-25),
        "distracted": dict(yaw=50), "phone": dict(phone=True, yaw=30)}
AUTO = [("normal", 14), ("yawn", 6), ("drowsy", 10), ("microsleep", 12), ("normal", 22), ("distracted", 6), ("phone", 6)]
scenario = "auto"


def publish(ev):
    for q in list(subscribers):
        q.put(ev)


def trim(dq, now, secs):
    while dq and now - (dq[0][0] if isinstance(dq[0], tuple) else dq[0]) > secs:
        dq.popleft()


def analyse(m):
    """R1 output -> active detections, fatigue score (0-100), overall state."""
    flags = []
    if m["perclos"] > 0.4 or m["ear"] < 0.15: flags.append("microsleep")
    elif m["perclos"] > 0.2: flags.append("drowsy")
    if m["mar"] > 0.6: flags.append("yawn")
    if m["pitch"] < -20: flags.append("nod")
    if abs(m["yaw"]) > 35: flags.append("distracted")
    if m.get("phone"): flags.append("phone")
    top = max([WEIGHTS[f] for f in flags] + [int(m["perclos"] * 100)])
    score = min(100, top + 10 * max(len(flags) - 1, 0))
    return flags, score, "danger" if score >= 70 else "warn" if score >= 35 else "ok"


def ask_llm(system, user, max_tokens=80):
    """Local Ollama (OLLAMA_MODEL) or Claude (ANTHROPIC_API_KEY). None -> caller uses fixed phrases."""
    try:
        if os.getenv("OLLAMA_MODEL"):
            body = json.dumps({"model": os.environ["OLLAMA_MODEL"], "stream": False,
                               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
            req = urllib.request.Request(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=6))["message"]["content"].strip()
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            r = anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", max_tokens=max_tokens, timeout=6, system=system,
                                                      messages=[{"role": "user", "content": user}])
            return r.content[0].text.strip()
    except Exception as e:
        print("LLM fallback:", e)
    return None


def emit_alert(kind, m, n_yawns):
    text = ask_llm(SYSTEM, f"Detection: {kind}. Yawns in last 5 min: {n_yawns}. Metrics: {json.dumps(m)}") \
        or TEXT[LANG][kind].format(n=n_yawns)
    publish({"type": "alert", "t": time.time(), "kind": kind, "severity": SEVERITY[kind], "text": text, "lang": LANG})


def set_level(new):
    global level
    level = new
    publish({"type": "action", "t": time.time(), "level": new, "actions": ACTIONS[new], "text": TEXT[LANG][new], "lang": LANG})


def handle_metrics(m):
    global prev_flags, danger_since, calm_since
    flags, score, state = analyse(m)
    now = time.time()
    with lock:
        if "yawn" in flags and "yawn" not in prev_flags:
            yawns.append((now, 1))
        prev_flags = set(flags)
        trim(yawns, now, 300); trim(history, now, WINDOW)
        fire = [f for f in flags if now - last_alert.get(f, 0) > COOLDOWN]
        for f in fire:
            last_alert[f] = now
            history.append((now, SEVERITY[f]))
        pts = sum(s for _, s in history)
        danger_since = (danger_since or now) if state == "danger" else None
        calm_since = (calm_since or now) if not flags else None
        target = 4 if danger_since and now - danger_since >= 10 else 3 if pts >= 8 else 2 if pts >= 4 else 1 if pts >= 1 else 0
        if calm_since and now - calm_since >= CALM:
            history.clear(); target = 0
            new = 0
        else:
            new = max(level, target)
        n_yawns, cur = len(yawns), level
    publish({"type": "metrics", "t": now, "metrics": m, "flags": flags, "score": score, "state": state, "level": cur})
    for f in fire:
        threading.Thread(target=emit_alert, args=(f, m, n_yawns), daemon=True).start()
    if new != cur:
        with lock:
            set_level(new)


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def chat_reply(msg):
    words = set(norm(msg).replace("?", " ").replace(",", " ").replace(".", " ").split())
    hits = [p for p in sorted(POIS, key=lambda p: -len(words & p[2])) if words & p[2]] or POIS[:1]
    context = "\n".join(f"- {n}: {d}" for n, d, _ in hits)
    return ask_llm("You are a friendly in-car guide. Answer briefly, in spoken style, using only this context:\n" + context
                   + (" Reply in Romanian." if LANG == "ro" else ""), msg, 200) \
        or ("Ți-aș recomanda: " if LANG == "ro" else "You could try: ") + "; ".join(f"{n} ({d})" for n, d, _ in hits[:2])


def simulator():
    base = dict(ear=0.30, perclos=0.05, mar=0.25, pitch=0, yaw=0, phone=False)
    t0 = time.time()
    while True:
        time.sleep(0.5)
        if time.time() < live_until:
            continue
        name = scenario
        if name == "auto":
            t = (time.time() - t0) % sum(d for _, d in AUTO)
            for name, d in AUTO:
                if t < d: break
                t -= d
        m = {**base, **SCEN[name]}
        for k in ("ear", "perclos", "mar"): m[k] = round(max(0, m[k] + random.gauss(0, .02)), 2)
        for k in ("pitch", "yaw"): m[k] = round(m[k] + random.gauss(0, 3))
        handle_metrics(m)


@app.route("/")
def index():
    return render_template("dashboard.html", lang=LANG)


@app.route("/events")
def events():
    def stream():
        q = queue.Queue()
        subscribers.append(q)
        try:
            while True:
                yield f"data: {json.dumps(q.get())}\n\n"
        finally:
            subscribers.remove(q)
    return Response(stream(), mimetype="text/event-stream")


@app.post("/api/event")
def ingest():
    """The real detector posts {ear, perclos, mar, pitch, yaw, phone} here a few times per second."""
    global live_until
    live_until = time.time() + 3
    handle_metrics(request.get_json(force=True))
    return {"ok": True}


@app.post("/api/frame")
def ingest_frame():
    global latest_frame
    latest_frame = request.get_data()
    return {"ok": True}


@app.route("/video_feed")
def video_feed():
    def gen():
        last = None
        while True:
            if latest_frame is not None and latest_frame is not last:
                last = latest_frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/scenario")
def set_scenario():
    """Steer the simulator from the dashboard: normal, yawn, drowsy, microsleep, distracted, phone, auto."""
    global scenario
    name = (request.get_json(force=True) or {}).get("name", "auto")
    if name in SCEN or name == "auto":
        scenario = name
    return {"ok": name == scenario}


@app.post("/api/ack")
def ack():
    """'I'm awake' button: stands in for the driver answering the assistant by voice."""
    global danger_since
    with lock:
        history.clear(); danger_since = None
        set_level(0)
    return {"ok": True}


@app.post("/api/chat")
def chat():
    """The driver speaks to the car (browser speech-to-text posts the transcript here)."""
    msg = request.get_json(force=True).get("message", "")
    if level >= 2:               # the driver answered the assistant's question out loud
        ack()
        return jsonify(reply=TEXT[LANG]["awake"])
    return jsonify(reply=chat_reply(msg))


if __name__ == "__main__":
    if os.getenv("DEMO", "1") == "1":
        threading.Thread(target=simulator, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

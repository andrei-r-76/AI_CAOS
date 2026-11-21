"""Driver monitoring from the webcam: MediaPipe face + hands (+ optional YOLO objects) -> state.json.

Install:   uv pip install mediapipe opencv-python numpy
           uv pip install ultralytics        (optional: phone / bottle / cup detection)
Run:       uv run python face_basic.py
Keys:      q = quit   c = recalibrate   l = toggle CLAHE (low light)   m = toggle face mesh drawing

Modules (same folder):
  camera.py     camera thread + camera health (covered / dark)
  eyes.py       ALL eye logic: open / half / closed per eye, one eye, blinks, microsleep, PERCLOS
  gaze.py       gaze zones (head + eyes), attention metrics (on-road ratio, mirrors, long glance, hypnosis)
  head.py       head yaw/pitch/roll, head drop / nodding / tilt
  hand_face.py  hands relative to the face (near face, at ear, at eyes, at mouth)
  objects.py    YOLO phone / bottle / cup (optional)
  alerts.py     scores, counters, ONE alert level + reason, event log  -> what the LLM reacts to
  state.py      the shared state schema written to state.json

Only the DRIVER (the largest face, "sticky") gets the full analysis; other faces are counted as passengers.

TUNING NOTE: every line marked "TUNE:" (here and in the modules) is a value you can adjust.
Values computed at calibration (resting jaw, eye opening open/closed, head pose on the road, gaze zones)
are stored in calib.json. The eye thresholds (eyes.py) work on the driver's own calibrated scale.
state.json gets only the keys useful for the LLM / dashboard (state.new_state); the rest stays internal.
"""
import csv           # for the optional debug log
import json          # for state.json + calib.json
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import eyes
import gaze
import objects
from alerts import AlertEngine
from camera import Camera, CameraHealth, reopen_camera
from hand_face import HandFaceTracker
from head import HeadTracker, head_angles
from hand_face import HandFilter
from state import internal_state, public

# --- models (downloaded automatically on first run) ---
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# landmark indices used for drawing
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
IRIS_POINTS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)

# --- MOUTH / YAWN (jawOpen blendshape, 0..1, measured above the calibrated resting value JAW_BASE) ---
MOUTH_OPEN_DELTA = 0.10    # TUNE: jawOpen this much above rest = mouth open
YAWN_DELTA = 0.20          # TUNE: jawOpen this much above rest...
YAWN_MIN_S = 1.2           # TUNE: ...held this long = yawn (talking opens the mouth only briefly)
YAWN_RELEASE = 0.7         # TUNE: hysteresis - once a yawn starts, it ends only below JAW_BASE + YAWN_DELTA * this

# --- CALIBRATION ---
# gaze zones learned at calibration. TUNE: add/remove zones here (names in gaze.MIRROR_ZONES count as
# mirror checks; "bord" also tells which head pitch means "down"). Look the way you would while driving.
ZONE_POSES = [
    ("oglinda_stanga", "Priveste OGLINDA STANGA (cum faci la volan)"),
    ("oglinda_dreapta", "Priveste OGLINDA DREAPTA (cum faci la volan)"),
    ("oglinda_mijloc", "Priveste OGLINDA DIN MIJLOC (sus-dreapta)"),
    ("bord", "Priveste BORDUL / TELEFONUL din poala (jos)"),
]
CALIB_FILE = "calib.json"
CALIB_VERSION = 10         # v10: eyes rebuilt (open / closed / looking-down references) -> recalibrates once
CALIB_SECONDS = 4          # TUNE: duration of the ROAD pose (eyes open, neutral posture)
CALIB_CLOSED_SECONDS = 3   # TUNE: duration of the final pose: eyes closed (between the two beeps)
POSE_SECONDS = 2.5         # TUNE: duration of each gaze-zone pose
PREP_SECONDS = 3.0         # TUNE: pause before each pose: shows what comes next + countdown, nothing is measured

# --- EYES vs HEAD POSE (the eye thresholds themselves are in eyes.py) ---
YAW_TURNED = 25.0          # TUNE: head turned more than this: a single closed eye is just the far eye hidden
YAW_EYES_MAX = 45.0        # TUNE: eyes are still checked up to this head yaw; beyond it one eye is hidden.
                           #       (A sleeping driver whose head falls sideways is still caught up to here.)
HEAD_DOWN_FRAC = 0.5       # TUNE: head pitched down at least this fraction of the "bord" pose = looking down
                           #       (used only if the head moved at least 8 degrees at the "bord" pose)

# calibrated values (defaults until calibration)
JAW_BASE = 0.0             # resting jawOpen (mouth closed), from calibration
EYE_CAL = eyes.EyeCalib()  # eye opening open / closed / looking down + iris position, from calibration
YAW_BASE = PITCH_BASE = ROLL_BASE = 0.0
GAZE_CAL = gaze.GazeCalib()
ZONE_MODEL = gaze.ZoneModel.default()
CALIBRATED = False

# --- SPEED / OUTPUT ---
DETECT_MAX_WIDTH = 640     # TUNE: frames wider than this are downscaled before detection
HAND_EVERY = 2             # TUNE: run hand detection every N frames
STATE_EMIT_S = 0.2         # TUNE: how often state.json is written (seconds)
PRINT_STATE = False        # TUNE: also print the state to the console (noisy, slightly slower)
DRAW_MESH = True           # TUNE: draw all 478 face points (key m). Off = a few FPS more
OBJECT_DETECTION = True    # TUNE: YOLO phone / bottle / cup (needs `ultralytics`; off = faster)
DEBUG_LOG = True           # TUNE: write raw driver values for every frame to debug_log.csv (for tuning)
DEBUG_LOG_FILE = "debug_log.csv"
DEBUG_FIELDS = ["t", "fps", "open_r", "open_l", "r_r", "r_l", "state_r", "state_l", "eye_pct", "blink_l", "blink_r",
                "iris_drop", "looking_down", "blink_now", "closed", "one_eye", "partial", "jaw", "jaw_base",
                "yawning", "yaw", "pitch", "roll", "gaze_h", "gaze_v", "zone", "zone_dist",
                "hand_near_face", "hand_at_ear", "phone", "alert_level", "alert_reason"]

# --- FACE TRACKING (stable IDs, so the driver is never confused with a passenger) ---
TRACK_MAX_JUMP = 0.25      # TUNE: max face-center move between frames (fraction of frame width) to keep the ID
TRACK_GRACE_S = 0.5        # TUNE: a face missing for less than this keeps its ID
DRIVER_SWITCH_RATIO = 1.5  # TUNE: another face must be this many times larger to become the driver

# --- HANDS ---
HAND_DETECT_CONF = 0.75    # TUNE: lower (0.6) if real hands are missed, higher if ghost hands appear
HAND_PRESENCE_CONF = 0.75  # TUNE   (more ghost-hand filtering in hand_face.py: HAND_MIN_SCORE, GHOST_MAX_SIZE)
HAND_TRACK_CONF = 0.75     # TUNE

USE_CLAHE = True
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
DRIVER_COLOR, PASSENGER_COLOR = (0, 255, 0), (255, 150, 0)
ALERT_COLORS = {"none": (60, 160, 60), "info": (200, 160, 0), "warning": (0, 140, 255), "critical": (0, 0, 220)}
STATE_FILE = "state.json"
WINDOW = "MediaPipe - sofer"


def ensure(path, url):
    if not os.path.exists(path):
        print(f"Descarc {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Gata.")


_last_ts = 0


def next_ts():
    """Strictly increasing timestamp in ms (required by MediaPipe VIDEO mode)."""
    global _last_ts
    _last_ts = max(int(time.monotonic() * 1000), _last_ts + 1)
    return _last_ts


def enhance(frame):
    """CLAHE on the L channel (LAB) - helps detection in low light."""
    if not USE_CLAHE:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def to_mp_image(frame):
    h, w = frame.shape[:2]
    if w > DETECT_MAX_WIDTH:
        frame = cv2.resize(frame, (DETECT_MAX_WIDTH, int(h * DETECT_MAX_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(enhance(frame), cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def blendshapes(face_res, idx):
    if face_res.face_blendshapes and idx < len(face_res.face_blendshapes):
        return {c.category_name: c.score for c in face_res.face_blendshapes[idx]}
    return {}


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def circular_mean(deg_values):
    rad = np.radians(deg_values)
    return math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


def box_area(pts):
    return ((max(p.x for p in pts) - min(p.x for p in pts)) *
            (max(p.y for p in pts) - min(p.y for p in pts)))


_tracks = {}     # face ID -> (center_x, center_y, last_seen)
_next_id = 0


def assign_ids(centers, now, w):
    """Give each detected face a stable ID by matching it to the nearest face from previous frames.

    MediaPipe can swap the order of faces between frames; without this the driver could jump
    to the passenger from one frame to the next.
    """
    global _next_id
    for tid in [k for k, v in _tracks.items() if now - v[2] > TRACK_GRACE_S]:
        del _tracks[tid]
    pairs = sorted((math.hypot(c[0] - p[0], c[1] - p[1]), i, tid)
                   for i, c in enumerate(centers) for tid, p in _tracks.items())
    ids, used = [None] * len(centers), set()
    for d, i, tid in pairs:                     # greedy: closest pairs first
        if d <= TRACK_MAX_JUMP * w and ids[i] is None and tid not in used:
            ids[i] = tid
            used.add(tid)
    for i, c in enumerate(centers):
        if ids[i] is None:                      # new face
            ids[i], _next_id = _next_id, _next_id + 1
        _tracks[ids[i]] = (c[0], c[1], now)
    return ids


def beep():
    try:
        import winsound
        winsound.Beep(1000, 250)
    except Exception:
        print("\a", end="", flush=True)


def separation(open_vals, closed_vals):
    """How well openness separates open from closed eyes (difference of medians / sum of std devs)."""
    if len(open_vals) < 5 or len(closed_vals) < 5:
        return None
    diff = float(np.median(open_vals) - np.median(closed_vals))
    return diff / (float(np.std(open_vals)) + float(np.std(closed_vals)) + 1e-6)


def write_state(state):
    """Write state.json (plain write). If another process (OneDrive, the dashboard) locks the file for
    a moment, this write is skipped; the next one comes STATE_EMIT_S later. Readers should retry on
    an empty / half-written file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except PermissionError:
        pass


# ============================================================================
#  CALIBRATION
# ============================================================================
def load_calib():
    global JAW_BASE, YAW_BASE, PITCH_BASE, ROLL_BASE, GAZE_CAL, EYE_CAL, ZONE_MODEL, CALIBRATED
    if not os.path.exists(CALIB_FILE):
        return False
    try:
        with open(CALIB_FILE) as f:
            c = json.load(f)
        if c.get("version") != CALIB_VERSION or not c.get("zones"):
            print("calib.json e de la o versiune mai veche -> recalibrez.")
            return False
        JAW_BASE = c["jaw_base"]
        EYE_CAL = eyes.EyeCalib(c["eye_open_ref"], c["eye_closed_ref"], c.get("eye_down_ref"),
                                c.get("iris_drop_road"), c.get("iris_drop_down"))
        YAW_BASE, PITCH_BASE = c["yaw_base"], c["pitch_base"]
        ROLL_BASE = c.get("roll_base", 0.0)
        GAZE_CAL = gaze.GazeCalib(c["gaze_h0"], c["gaze_v0"])
        ZONE_MODEL = gaze.ZoneModel(c["zones"])
    except (KeyError, ValueError, TypeError):
        print("calib.json corupt -> recalibrez.")
        return False
    CALIBRATED = True
    print(f"Calibrare incarcata: ochi deschis {EYE_CAL.open_ref:.2f} / inchis {EYE_CAL.closed_ref:.2f}  "
          f"gura repaus {JAW_BASE:.2f}  cap drept: yaw {YAW_BASE:+.0f} pitch {PITCH_BASE:+.0f} roll {ROLL_BASE:+.0f}  "
          f"zone: {', '.join(ZONE_MODEL.zones)}")
    return True


def _prepare(cap, face, seconds, text):
    """Pause before a calibration pose: live preview + "next: ..." + countdown. Nothing is measured.

    Returns False if the user cancelled with q.
    """
    t_end = time.time() + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = face.detect_for_video(to_mp_image(frame), next_ts())   # keeps VIDEO-mode tracking warm
        if res.face_landmarks:
            for p in res.face_landmarks[0]:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 200, 255), -1)
        left = int(t_end - time.time()) + 1
        cv2.rectangle(frame, (0, 35), (w, 135), (25, 25, 25), -1)
        cv2.putText(frame, "PREGATESTE-TE - urmeaza:", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"incepe in {left}s   (q = anuleaza)", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def _collect(cap, face, seconds, text, skip=0.0):
    """Collect per-frame measurements for `seconds`; the first `skip` seconds are ignored."""
    data = {"jaw": [], "yaw": [], "pitch": [], "roll": [], "gaze_h": [], "gaze_v": [], "open": [], "drop": []}
    t_start = time.time()
    t_end = t_start + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = face.detect_for_video(to_mp_image(frame), next_ts())
        if res.face_landmarks:
            i = max(range(len(res.face_landmarks)), key=lambda k: box_area(res.face_landmarks[k]))
            xy = [(p.x * w, p.y * h) for p in res.face_landmarks[i]]
            for x, y in xy:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            if time.time() - t_start >= skip:
                bs = blendshapes(res, i)
                if bs:
                    data["jaw"].append(bs["jawOpen"])
                ang = head_angles(res, i)
                if ang:
                    data["yaw"].append(ang[0])
                    data["pitch"].append(ang[1])
                    data["roll"].append(ang[2])
                g = gaze.gaze_ratios(xy)
                if g:
                    data["gaze_h"].append(g[0])
                    data["gaze_v"].append(g[1])
                op = eyes.eye_openness(xy)                   # iris-based openness, both eyes pooled
                if op is not None:
                    data["open"].extend(op)
                drop = eyes.iris_drop(xy)                    # iris vs eye corners (looking down)
                if drop is not None:
                    data["drop"].append(drop)
        left = int(t_end - time.time()) + 1
        cv2.putText(frame, f"{text}  {left}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, "q = anuleaza calibrarea", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
    return data


def calibrate(cap, face):
    """Gaze-zone scan + eye/head/mouth calibration for the current driver.

    Poses: ROAD (neutral) -> every zone in ZONE_POSES -> EYES CLOSED. At each zone look the
    way you normally would while driving (head, eyes or both): for every zone we store head
    yaw/pitch + iris position, so it works whether the driver turns the head or only the eyes.
    Each pose starts with a PREP_SECONDS pause and a beep. q = cancel at any time.
    """
    global JAW_BASE, YAW_BASE, PITCH_BASE, ROLL_BASE, GAZE_CAL, EYE_CAL, ZONE_MODEL, CALIBRATED
    print("\n=== SCANARE COMPLETA ===")
    n_steps = len(ZONE_POSES) + 2
    poses = [("front", CALIB_SECONDS, f"1/{n_steps} PRIVESTE DRUMUL (drept), ochii deschisi, gura inchisa")]
    poses += [(name, POSE_SECONDS, f"{i + 2}/{n_steps} {text}") for i, (name, text) in enumerate(ZONE_POSES)]
    got = {}
    for name, secs, text in poses:
        print(text)
        if not _prepare(cap, face, PREP_SECONDS, text):
            print("Scanare anulata, pastrez valorile vechi.")
            return
        beep()                                   # "start now"
        got[name] = _collect(cap, face, secs, text, skip=0.5)
        if got[name] is None:
            print("Scanare anulata, pastrez valorile vechi.")
            return
        if name == "front" and not got[name]["yaw"]:
            print("Scanare esuata (nicio fata). Pastrez valorile vechi.")
            return
    front = got["front"]

    print(f"{n_steps}/{n_steps} ({CALIB_CLOSED_SECONDS}s): la BIP inchide ochii, redeschide-i la al doilea BIP.")
    if not _prepare(cap, face, PREP_SECONDS, f"{n_steps}/{n_steps} La BIP INCHIDE OCHII, deschide-i la al doilea BIP"):
        print("Scanare anulata, pastrez valorile vechi.")
        return
    beep()
    closed = _collect(cap, face, CALIB_CLOSED_SECONDS,
                      f"{n_steps}/{n_steps} tine OCHII INCHISI pana la al doilea bip", skip=0.7)
    beep()
    if closed is None:
        print("Scanare anulata, pastrez valorile vechi.")
        return

    med = lambda lst, d: float(np.median(lst)) if lst else d

    # --- mouth / head / neutral gaze (from the ROAD pose) ---
    JAW_BASE = med(front["jaw"], 0.0)          # resting mouth; open / yawn thresholds are measured above it
    YAW_BASE, PITCH_BASE = circular_mean(front["yaw"]), circular_mean(front["pitch"])
    ROLL_BASE = circular_mean(front["roll"])
    h0, v0 = med(front["gaze_h"], 0.5), med(front["gaze_v"], 0.5)
    GAZE_CAL = gaze.GazeCalib(h0, v0)

    # --- eyes: opening open (road pose), closed (closed pose), looking down (bord pose) ---
    down = got.get("bord") or {"open": [], "drop": []}
    o_ref, c_ref = med(front["open"], None), med(closed["open"], None)
    sep = separation(front["open"], closed["open"])
    if o_ref is None or c_ref is None or o_ref - c_ref < 0.05:
        print("ATENTIE: ochii deschisi / inchisi nu se deosebesc -> apasa c si reia (lumina, fata in cadru).")
        o_ref, c_ref = (o_ref or 0.7), (0.3 * (o_ref or 0.7))
    EYE_CAL = eyes.EyeCalib(o_ref, c_ref, med(down["open"], None), med(front["drop"], None), med(down["drop"], None))
    print(f"Ochi: deschis {o_ref:.2f}  inchis {c_ref:.2f}  la bord {EYE_CAL.down_ref or 0:.2f}"
          + (f"  (separare {sep:.1f})" if sep is not None else ""))
    if sep is not None and sep < 2:
        print("ATENTIE: separare mica intre ochi deschisi si inchisi -> apasa c si reia la lumina mai buna.")

    # --- gaze zones: head yaw/pitch (relative to the road pose) + iris, per zone ---
    zones = {gaze.ROAD_ZONE: (0.0, 0.0, h0, v0)}
    for name, _ in ZONE_POSES:
        z = got[name]
        if not z["yaw"]:
            print(f"ATENTIE: zona {name}: nicio fata detectata -> o sar.")
            continue
        zy = angle_diff(circular_mean(z["yaw"]), YAW_BASE)
        zp = angle_diff(circular_mean(z["pitch"]), PITCH_BASE)
        zh, zv = med(z["gaze_h"], None), med(z["gaze_v"], None)
        zones[name] = (zy, zp, zh, zv)
        print(f"  zona {name:16s} cap yaw {zy:+5.0f} pitch {zp:+5.0f}   iris "
              + (f"h{zh:.2f} v{zv:.2f}" if zh is not None else "-"))
    ZONE_MODEL = gaze.ZoneModel(zones)
    CALIBRATED = True

    with open(CALIB_FILE, "w") as f:
        json.dump({"version": CALIB_VERSION, "jaw_base": JAW_BASE,
                   "eye_open_ref": EYE_CAL.open_ref, "eye_closed_ref": EYE_CAL.closed_ref,
                   "eye_down_ref": EYE_CAL.down_ref, "iris_drop_road": EYE_CAL.drop_road,
                   "iris_drop_down": EYE_CAL.drop_down,
                   "yaw_base": YAW_BASE, "pitch_base": PITCH_BASE, "roll_base": ROLL_BASE,
                   "gaze_h0": h0, "gaze_v0": v0,
                   "zones": ZONE_MODEL.to_json()}, f, indent=2)
    print(f"Gata. gura repaus {JAW_BASE:.2f} (deschisa >{JAW_BASE + MOUTH_OPEN_DELTA:.2f}, "
          f"cascat >{JAW_BASE + YAWN_DELTA:.2f})  "
          f"cap drept: yaw {YAW_BASE:+.0f} pitch {PITCH_BASE:+.0f} roll {ROLL_BASE:+.0f}\n")


# ============================================================================
#  DRAWING
# ============================================================================
def draw_gaze_panel(frame, gm, x, y):
    """Panel with the gaze / attention metrics."""
    lines = [
        f"Privire: {gm['direction'] or 'pe drum'}",
        f"Ochi pe drum: {gm['on_road_ratio'] * 100:.0f}%",
        f"Oglinzi: {gm['mirror_checks']}",
        f"Fixatie: {gm['fixation_s']:.0f}s" + ("  HIPNOZA" if gm["hypnosis"] else ""),
    ]
    cv2.rectangle(frame, (x - 6, y - 18), (x + 200, y + len(lines) * 18 - 4), (25, 25, 25), -1)
    for i, t in enumerate(lines):
        col = (0, 200, 255) if ("ALERTA" in t or "HIPNOZA" in t) else (220, 220, 220)
        cv2.putText(frame, t, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


def put(frame, text, org, col, scale=0.5, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)


# ============================================================================
#  MAIN LOOP
# ============================================================================
def main():
    global USE_CLAHE, DRAW_MESH
    ensure(FACE_MODEL, FACE_URL)
    ensure(HAND_MODEL, HAND_URL)

    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=HAND_DETECT_CONF,
        min_hand_presence_confidence=HAND_PRESENCE_CONF,
        min_tracking_confidence=HAND_TRACK_CONF,
    ))

    cap = Camera(0)
    if not cap.isOpened():
        raise SystemExit("Nu pot deschide camera.")

    if not load_calib():
        calibrate(cap, face)

    detector = objects.ObjectDetector(OBJECT_DETECTION)
    print(f"Detectie obiecte (YOLO): {detector.status}")

    def new_trackers():
        return gaze.GazeTracker(GAZE_CAL), eyes.EyeTracker(EYE_CAL), HeadTracker()

    gaze_tracker, eye_tracker, head_tracker = new_trackers()
    hand_tracker, health, alerts = HandFaceTracker(), CameraHealth(), AlertEngine()
    hand_filter = HandFilter()

    print("Pornit. q = iesire, c = recalibrare, l = CLAHE, m = desen puncte fata.")
    t_start = time.time()
    t0, frames, fps = t_start, 0, 0.0
    last_emit = 0.0
    frame_no, hand_res = 0, None
    driver_id = None              # stable ID of the current driver
    last_face_seen = t_start      # for face_lost_sec
    yawn_since = None
    prof = {}
    dbg_file = dbg_writer = None
    if DEBUG_LOG:
        dbg_file = open(DEBUG_LOG_FILE, "w", newline="")
        dbg_writer = csv.DictWriter(dbg_file, fieldnames=DEBUG_FIELDS)
        dbg_writer.writeheader()
        dbg_flush = t_start
        print(f"Jurnal de debug: {os.path.abspath(DEBUG_LOG_FILE)}")

    def tick(name, start):
        """Smoothed per-stage timing in ms (shown at the bottom of the window)."""
        ms = (time.perf_counter() - start) * 1000
        prof[name] = 0.9 * prof.get(name, ms) + 0.1 * ms

    while True:
        t_cam = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if cap.isOpened():
                continue                          # just a slow frame: wait for the next one
            cap = reopen_camera(cap)              # camera stopped: try to bring it back
            if cap is None:
                break
            continue
        tick("camera", t_cam)
        h, w = frame.shape[:2]
        now = time.time()
        cam = health.update(now, frame)           # before drawing anything on the frame
        detector.submit(frame)

        t = time.perf_counter()
        mp_image = to_mp_image(frame)
        ts = next_ts()
        face_res = face.detect_for_video(mp_image, ts)
        tick("fata", t)
        t = time.perf_counter()
        if hand_res is None or frame_no % HAND_EVERY == 0:
            hand_res = hands.detect_for_video(mp_image, ts)
        tick("maini", t)
        frame_no += 1
        t_draw = time.perf_counter()

        raw_hands = [[(p.x * w, p.y * h) for p in pts] for pts in hand_res.hand_landmarks]
        hand_scores = [getattr(hd[0], "score", 1.0) if hd else 0.0
                       for hd in (hand_res.handedness or [[]] * len(raw_hands))]

        # ---------- faces: pick the driver (largest face, sticky) ----------
        lms = face_res.face_landmarks
        n_faces = len(lms)
        centers = [(float(np.mean([p.x for p in pts])) * w, float(np.mean([p.y for p in pts])) * h)
                   for pts in lms]
        ids = assign_ids(centers, now, w)
        areas = [box_area(pts) for pts in lms]
        driver = None
        if n_faces:
            biggest = max(range(n_faces), key=lambda k: areas[k])
            cur = ids.index(driver_id) if driver_id in ids else None
            driver = cur if (cur is not None and areas[biggest] < DRIVER_SWITCH_RATIO * areas[cur]) else biggest
            if ids[driver] != driver_id:
                if driver_id in ids:              # a different person (still in view) became the driver
                    gaze_tracker, eye_tracker, head_tracker = new_trackers()
                    yawn_since = None
                driver_id = ids[driver]

        for idx, pts in enumerate(lms):           # passengers: box + label only
            if idx == driver:
                continue
            xs, ys = [p.x * w for p in pts], [p.y * h for p in pts]
            cv2.rectangle(frame, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))), PASSENGER_COLOR, 2)
            put(frame, "Pasager", (int(min(xs)), int(min(ys)) - 8), PASSENGER_COLOR, 0.6)

        state = internal_state()
        state["ts"] = round(now, 2)
        state["calibrated"] = CALIBRATED
        state["face_found"] = driver is not None
        state["passenger_count"] = max(0, n_faces - 1)
        state.update(cam)
        gm = em = None
        xy = None
        hands_px = hand_filter.update(raw_hands, hand_scores,
                                      [(p.x * w, p.y * h) for p in lms[driver]] if driver is not None else None)
        dbg = {}

        # ---------- driver: full analysis ----------
        if driver is not None:
            last_face_seen = now
            pts = lms[driver]
            xy = [(p.x * w, p.y * h) for p in pts]
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            cv2.rectangle(frame, (x0, y0), (x1, y1), DRIVER_COLOR, 2)
            if DRAW_MESH:
                for x, y in xy:
                    cv2.circle(frame, (int(x), int(y)), 1, DRIVER_COLOR, -1)
            bs = blendshapes(face_res, driver)

            # head
            ang = head_angles(face_res, driver)
            if ang:
                yaw_d, pitch_d, roll_d = (angle_diff(ang[0], YAW_BASE), angle_diff(ang[1], PITCH_BASE),
                                          angle_diff(ang[2], ROLL_BASE))
            else:
                yaw_d = pitch_d = roll_d = 0.0

            # eyes (eyes.py): per-eye opening on the driver's own scale -> open / half / closed, one eye,
            # blinks, microsleep. "Looking down" comes from the iris vs the eye corners + the head pitch,
            # never from the eyelids (drooping eyelids look the same as looking down).
            eyes_hidden = abs(yaw_d) > YAW_EYES_MAX
            blink_l, blink_r = (bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) if bs else (0.0, 0.0)
            op = eyes.eye_openness(xy)
            drop = eyes.iris_drop(xy)
            bord = ZONE_MODEL.zones.get("bord")
            head_down = bool(bord and abs(bord[1]) >= 8 and
                             pitch_d * ZONE_MODEL.down_sign >= HEAD_DOWN_FRAC * abs(bord[1]))
            em = eye_tracker.update(now, op, (blink_l, blink_r) if bs else None, drop, head_down,
                                    hidden=eyes_hidden, turned=abs(yaw_d) > YAW_TURNED)
            closed, one_eye = em["closed"], em["one_eye"]

            # gaze zone (head + eyes; head only while the eyes are closed)
            g = gaze.gaze_ratios(xy)
            zone, zone_dist = ZONE_MODEL.classify(yaw_d, pitch_d, g if (g and not closed) else None,
                                                  prefer=gaze_tracker.zone)   # current zone is kept more easily

            # mouth / yawn (jawOpen with hysteresis)
            jaw = bs.get("jawOpen") if bs else None
            if jaw is not None:
                above = jaw - JAW_BASE
                mouth_open = above > MOUTH_OPEN_DELTA
                wide = above > (YAWN_DELTA * YAWN_RELEASE if yawn_since is not None else YAWN_DELTA)
            else:
                mouth_open = wide = False
            yawn_since = (yawn_since or now) if wide else None
            yawning = wide and now - yawn_since >= YAWN_MIN_S

            # trackers
            gm = gaze_tracker.update(now, g, eyes_open=not closed, zone=zone,
                                     blinks_per_min=em["blinks_per_min"], warmed_up=em["warmed_up"])
            hm = head_tracker.update(now, yaw_d, pitch_d, roll_d, closed, ZONE_MODEL.down_sign)

            state.update({
                "eyes_closed_sec": round(em["closed_sec"], 1),
                "one_eye_closed": one_eye,
                "yawning": yawning,
                "head_drop": hm["head_drop"], "head_nods_1min": hm["head_nods_1min"], "head_tilt": hm["head_tilt"],
                "looking_away_sec": gm["away_sec"], "gaze": gm["direction"] or "on_road",
                "on_road_ratio": gm["on_road_ratio"], "mirror_checks": gm["mirror_checks"],
                "mirror_too_long": gm["mirror_too_long"], "long_glance": gm["long_glance"], "hypnosis": gm["hypnosis"],
            })
            for k in ("blink_rate", "slow_blinks", "perclos", "perclos_drowsy", "microsleep", "sleep"):
                state[k] = em[k]
            state["eyes_partially_closed"] = em["partial"]
            if em["pct"] is not None:
                state["eye_openness_pct"] = em["pct"]

            # driver overlay
            state_col = {"open": (0, 255, 0), "half": (0, 220, 255), "closed": (0, 0, 255)}
            eye_col = ((150, 150, 150) if eyes_hidden else (0, 0, 255) if closed else
                       (0, 200, 255) if (one_eye or em["partial"]) else (0, 255, 255))
            for k, eye in enumerate((LEFT_EYE, RIGHT_EYE)):   # LEFT_EYE = points 33..133 = eye 0 ("right")
                col = state_col[em["states"][k]] if em["states"] else (150, 150, 150)
                poly = np.array([[int(xy[i][0]), int(xy[i][1])] for i in eye], np.int32)
                cv2.polylines(frame, [poly], True, col, 2)
            for i in IRIS_POINTS:
                cv2.circle(frame, (int(xy[i][0]), int(xy[i][1])), 1, (255, 255, 0), -1)
            mouth_col = (0, 0, 255) if yawning else (0, 165, 255) if wide else (0, 255, 255) if mouth_open else (0, 255, 0)
            mt, mb = (tuple(int(v) for v in xy[MOUTH_TOP_BOTTOM[0]]), tuple(int(v) for v in xy[MOUTH_TOP_BOTTOM[1]]))
            ml, mr = (tuple(int(v) for v in xy[MOUTH_CORNERS[0]]), tuple(int(v) for v in xy[MOUTH_CORNERS[1]]))
            cv2.line(frame, mt, mb, mouth_col, 2)
            cv2.line(frame, ml, mr, mouth_col, 2)

            eye_state = ("n/a (cap intors)" if eyes_hidden else
                         f"INCHISI {em['closed_sec']:.1f}s" if closed else
                         f"un ochi inchis {em['one_eye_sec']:.1f}s" if one_eye else
                         "PE JUMATATE INCHISI" if em["partial"] else
                         "deschisi (privire jos)" if em["looking_down"] else "deschisi")
            pct = em["pct"]
            per_eye = (f"R{em['r'][0] * 100:.0f}% L{em['r'][1] * 100:.0f}%" if em["r"] else "-")
            eye_val = (f"{pct}% ({per_eye})" if pct is not None else per_eye)
            put(frame, f"Sofer  cap yaw {yaw_d:+.0f} pitch {pitch_d:+.0f} roll {roll_d:+.0f}", (x0, y0 - 68), (255, 255, 255))
            put(frame, f"Ochi {eye_val} - {eye_state}", (x0, y0 - 48), eye_col)
            put(frame, f"Privire: {gm['direction'] or 'pe drum'}", (x0, y0 - 28), (255, 255, 0))
            mouth_txt = ("CASCA!" if yawning else "cascat?" if wide else "gura deschisa" if mouth_open else "gura inchisa")
            jaw_above = (jaw - JAW_BASE) if jaw is not None else 0.0
            put(frame, f"Gura {jaw_above:.2f} (deschisa >{MOUTH_OPEN_DELTA:.2f}, cascat >{YAWN_DELTA:.2f}) - {mouth_txt}",
                (x0, y0 - 8), mouth_col)
            if closed:                                 # big and red while the eyes are closed, with the duration
                put(frame, f"OCHI INCHISI {em['closed_sec']:.1f}s", (x0, y1 + 28), (0, 0, 255), 0.9, 3)
            elif one_eye:
                put(frame, f"un ochi inchis {em['one_eye_sec']:.1f}s", (x0, y1 + 24), (0, 200, 255), 0.6, 2)
            elif em["partial"]:
                put(frame, "OCHI PE JUMATATE INCHISI", (x0, y1 + 24), (0, 200, 255), 0.6, 2)
            else:
                put(frame, f"clipiri/min {em['blink_rate']}  ultimul clipit acum {em['since_last_blink']:.0f}s",
                    (x0, y1 + 20), eye_col)
            draw_gaze_panel(frame, gm, w - 206, 72)

            dbg = {"open_r": round(op[0], 3) if op else "", "open_l": round(op[1], 3) if op else "",
                   "r_r": round(em["r"][0], 2) if em["r"] else "", "r_l": round(em["r"][1], 2) if em["r"] else "",
                   "state_r": em["states"][0] if em["states"] else "", "state_l": em["states"][1] if em["states"] else "",
                   "eye_pct": pct if pct is not None else "", "blink_l": round(blink_l, 2), "blink_r": round(blink_r, 2),
                   "iris_drop": round(drop, 3) if drop is not None else "", "looking_down": int(em["looking_down"]),
                   "blink_now": int(em["blink_now"]), "closed": int(closed), "one_eye": int(one_eye),
                   "partial": int(em["partial"]),
                   "jaw": round(jaw, 3) if jaw is not None else "", "jaw_base": round(JAW_BASE, 3),
                   "yawning": int(yawning), "yaw": round(yaw_d, 1), "pitch": round(pitch_d, 1),
                   "roll": round(roll_d, 1), "gaze_h": round(g[0], 3) if g else "",
                   "gaze_v": round(g[1], 3) if g else "", "zone": zone, "zone_dist": round(zone_dist, 2)}

        state["face_lost_sec"] = 0.0 if driver is not None else round(now - last_face_seen, 1)

        # ---------- hands (relative to the driver's face) ----------
        state.update(hand_tracker.update(now, hands_px, xy))
        for hand in hands_px:
            ipts = [(int(x), int(y)) for x, y in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, ipts[a], ipts[b], (230, 230, 230), 2)
            for p in ipts:
                cv2.circle(frame, p, 3, (255, 0, 255), -1)

        # ---------- objects (YOLO, optional) ----------
        dets = detector.latest(now)
        state.update(objects.relate(dets, xy, hands_px))
        for label, conf, bx1, by1, bx2, by2 in dets:
            col = (0, 0, 255) if label == "phone" else (255, 200, 0)
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), col, 2)
            put(frame, f"{label} {conf:.2f}", (int(bx1), int(by1) - 6), col, 0.5)

        # ---------- scores, alert level, events ----------
        alerts.update(state, now)

        if dbg_writer is not None and driver is not None:
            dbg_writer.writerow(dict(dbg, t=round(now - t_start, 3), fps=round(fps, 1),
                                     hand_near_face=int(state["hand_near_face"]),
                                     hand_at_ear=int(state["hand_at_ear"]), phone=int(state["phone_detected"]),
                                     alert_level=state["alert_level"], alert_reason=state["alert_reason"]))
            if now - dbg_flush > 1.0:             # flush once a second so the file can be read live
                dbg_file.flush()
                dbg_flush = now

        if now - last_emit > STATE_EMIT_S:
            out = public(state)                   # only what the LLM / dashboard needs
            write_state(out)
            if PRINT_STATE:
                print(out)
            last_emit = now

        # ---------- overlays: alert banner, top bar, FPS ----------
        lvl = state["alert_level"]
        if lvl != "none":
            cv2.rectangle(frame, (0, h - 95), (w, h - 55), ALERT_COLORS[lvl], -1)
            others = ", ".join(state["active_alerts"][1:4])
            put(frame, f"{lvl.upper()}: {state['alert_reason']}" + (f"  (+ {others})" if others else ""),
                (12, h - 67), (255, 255, 255), 0.6)
        cv2.rectangle(frame, (0, 0), (w, 50), (25, 25, 25), -1)
        put(frame, f"Somnolenta {state['drowsiness_score']} ({state['fatigue_level']})  "
                   f"Atentie {state['attention_score']}  PERCLOS {state['perclos'] * 100:.1f}%  "
                   f"Pasageri {state['passenger_count']}", (10, 20), (255, 255, 255), 0.5)
        flags = [k for k in ("hand_near_face", "hand_at_ear", "hand_at_eyes", "phone_detected",
                             "camera_blocked", "low_light") if state[k]]
        put(frame, f"CLAHE {'ON' if USE_CLAHE else 'OFF'}  YOLO {'ON' if detector.available else 'OFF'}  "
                   + (" ".join(flags) if flags else "[c] calib [l] CLAHE [m] mesh [q] iesire"),
            (10, 42), (200, 200, 200), 0.45, 1)

        frames += 1
        if now - t0 >= 1:
            fps, frames, t0 = frames / (now - t0), 0, now
        put(frame, f"{fps:.1f} FPS", (10, h - 15), (200, 200, 200), 0.6, 1)
        put(frame, "  ".join(f"{k} {v:.0f}" for k, v in prof.items()) + " ms", (110, h - 15), (200, 200, 200), 0.45, 1)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        tick("desen", t_draw)
        if key == ord("q"):
            break
        if key == ord("c"):
            calibrate(cap, face)
            gaze_tracker, eye_tracker, head_tracker = new_trackers()
            yawn_since = None
        if key == ord("l"):
            USE_CLAHE = not USE_CLAHE
        if key == ord("m"):
            DRAW_MESH = not DRAW_MESH

    detector.stop()
    if dbg_file is not None:
        dbg_file.close()
    cap.release()
    face.close()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Driver monitoring from the webcam: MediaPipe face + hands (+ optional YOLO objects) -> state.json.

Install:   uv pip install mediapipe opencv-python numpy
           uv pip install ultralytics        (optional: phone / bottle / cup detection)
Run:       uv run python face_basic.py
Keys:      q = quit   c = recalibrate   l = toggle CLAHE (low light)   m = toggle face mesh drawing

Modules (same folder):
  camera.py     camera thread + camera health (covered / dark)
  gaze.py       gaze zones (head + eyes), attention metrics, eye fatigue (blinks, PERCLOS, microsleep)
  head.py       head yaw/pitch/roll, head drop / nodding / tilt
  hand_face.py  hands relative to the face (near face, at ear, at eyes, at mouth)
  objects.py    YOLO phone / bottle / cup (optional)
  alerts.py     scores, counters, ONE alert level + reason, event log  -> what the LLM reacts to
  state.py      the shared state schema written to state.json

Only the DRIVER (the largest face, "sticky") gets the full analysis; other faces are counted as passengers.

TUNING NOTE: every line marked "TUNE:" (here and in the modules) is a value you can adjust.
Values computed at calibration (JAW_OPEN, EYE_OPEN_CLOSED, head pose on the road, gaze zones) are
stored in calib.json. After changing JAW_K or EYE_OPEN_K, press `c` to recalibrate.
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

import gaze
import objects
from alerts import AlertEngine
from camera import Camera, CameraHealth, reopen_camera
from hand_face import HandFaceTracker
from head import HeadTracker, head_angles
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

# --- MOUTH / YAWN (jawOpen blendshape) ---
YAWN_MIN_S = 1.2           # TUNE: mouth must stay wide open this long to count as a yawn (not speech)
YAWN_RELEASE = 0.7         # TUNE: hysteresis - once a yawn starts, it ends only when jaw < JAW_OPEN * this

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
CALIB_VERSION = 8          # same format as before (extra keys are optional) -> an existing calib.json still loads
CALIB_SECONDS = 4          # TUNE: duration of the ROAD pose (eyes open, neutral posture)
CALIB_CLOSED_SECONDS = 3   # TUNE: duration of the final pose: eyes closed (between the two beeps)
POSE_SECONDS = 2.5         # TUNE: duration of each gaze-zone pose
PREP_SECONDS = 3.0         # TUNE: pause before each pose: shows what comes next + countdown, nothing is measured
JAW_K = 0.5                # TUNE: yawn threshold = jaw_base + K * (1 - jaw_base). Lower = easier yawn (0.3 ~ 0.32)
JAW_MIN, JAW_MAX = 0.30, 0.80   # TUNE: clamp for the calibrated yawn threshold
EYE_OPEN_K = 0.4           # TUNE: eye-closed threshold = closed + K * (open - closed). Lower = harder to
                           #       count as closed (fewer false alarms when looking down)
EYE_OPEN_MIN, EYE_OPEN_MAX = 0.20, 0.70   # TUNE: clamp for the calibrated eye-closed threshold
CLOSED_MIN_S = 0.07        # TUNE: eyes must stay below the threshold this long to count as closed
CLOSED_MIN_FRAMES = 2      # TUNE: ...but never fewer frames than this (filters single-frame noise)

# --- EYES vs HEAD POSE ---
YAW_TURNED = 25.0          # TUNE: head turned more than this: a single closed eye is just the far eye hidden
YAW_EYES_MAX = 45.0        # TUNE: eyes are still checked up to this head yaw; beyond it one eye is hidden.
                           #       (A sleeping driver whose head falls sideways is still caught up to here.)
LOOK_DOWN_BS = 0.45        # TUNE: eyeLookDown blendshape above this = looking down (phone / dashboard)
LOOK_DOWN_RELAX = 0.6      # TUNE: while looking down, an eye counts as closed only below threshold * this

# calibrated values (defaults until calibration)
JAW_OPEN = 0.5
YAW_BASE = PITCH_BASE = ROLL_BASE = 0.0
GAZE_CAL = gaze.GazeCalib()
ZONE_MODEL = gaze.ZoneModel.default()
EYE_OPEN_CLOSED = gaze.EYE_OPEN_CLOSED
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
DEBUG_FIELDS = ["t", "fps", "open_r", "open_l", "eye_thr", "look_down", "closed", "jaw", "jaw_thr",
                "yawning", "yaw", "pitch", "roll", "gaze_h", "gaze_v", "zone", "zone_dist",
                "hand_near_face", "hand_at_ear", "phone", "alert_level", "alert_reason"]

# --- FACE TRACKING (stable IDs, so the driver is never confused with a passenger) ---
TRACK_MAX_JUMP = 0.25      # TUNE: max face-center move between frames (fraction of frame width) to keep the ID
TRACK_GRACE_S = 0.5        # TUNE: a face missing for less than this keeps its ID
DRIVER_SWITCH_RATIO = 1.5  # TUNE: another face must be this many times larger to become the driver

# --- HANDS ---
HAND_DETECT_CONF = 0.6     # TUNE: lower (0.5) if hands are missed, higher if ghosts appear
HAND_PRESENCE_CONF = 0.6   # TUNE
HAND_TRACK_CONF = 0.6      # TUNE

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
            json.dump(public(state), f)
    except PermissionError:
        pass


# ============================================================================
#  CALIBRATION
# ============================================================================
def load_calib():
    global JAW_OPEN, YAW_BASE, PITCH_BASE, ROLL_BASE, GAZE_CAL, EYE_OPEN_CLOSED, ZONE_MODEL, CALIBRATED
    if not os.path.exists(CALIB_FILE):
        return False
    try:
        with open(CALIB_FILE) as f:
            c = json.load(f)
        if c.get("version") != CALIB_VERSION or not c.get("zones"):
            print("calib.json e de la o versiune mai veche -> recalibrez.")
            return False
        JAW_OPEN = c["jaw_open"]
        YAW_BASE, PITCH_BASE = c["yaw_base"], c["pitch_base"]
        ROLL_BASE = c.get("roll_base", 0.0)
        GAZE_CAL = gaze.GazeCalib(c["gaze_h0"], c["gaze_v0"])
        EYE_OPEN_CLOSED = c["eye_open_closed"]
        ZONE_MODEL = gaze.ZoneModel(c["zones"])
    except (KeyError, ValueError, TypeError):
        print("calib.json corupt -> recalibrez.")
        return False
    CALIBRATED = True
    print(f"Calibrare incarcata: ochi inchisi sub {EYE_OPEN_CLOSED:.2f}  jaw>{JAW_OPEN:.2f}  "
          f"cap drept: yaw {YAW_BASE:+.0f} pitch {PITCH_BASE:+.0f} roll {ROLL_BASE:+.0f}  "
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
    data = {"jaw": [], "yaw": [], "pitch": [], "roll": [], "gaze_h": [], "gaze_v": [], "open": []}
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
                op = gaze.eye_openness(xy)                   # iris-based openness, both eyes pooled
                if op is not None:
                    data["open"].extend(op)
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
    global JAW_OPEN, YAW_BASE, PITCH_BASE, ROLL_BASE, GAZE_CAL, EYE_OPEN_CLOSED, ZONE_MODEL, CALIBRATED
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
    if front["jaw"]:
        jaw_base = med(front["jaw"], 0.0)
        JAW_OPEN = float(np.clip(jaw_base + JAW_K * (1 - jaw_base), JAW_MIN, JAW_MAX))
    YAW_BASE, PITCH_BASE = circular_mean(front["yaw"]), circular_mean(front["pitch"])
    ROLL_BASE = circular_mean(front["roll"])
    h0, v0 = med(front["gaze_h"], 0.5), med(front["gaze_v"], 0.5)
    GAZE_CAL = gaze.GazeCalib(h0, v0)

    # --- eye-closed threshold from iris openness: ROAD (open) vs CLOSED ---
    sep = separation(front["open"], closed["open"])
    if sep is not None and sep > 0:
        open_med, closed_med = med(front["open"], 0.9), med(closed["open"], 0.15)
        EYE_OPEN_CLOSED = float(np.clip(closed_med + EYE_OPEN_K * (open_med - closed_med),
                                        EYE_OPEN_MIN, EYE_OPEN_MAX))
        print(f"Deschidere ochi: deschis {open_med:.2f}  inchis {closed_med:.2f}  "
              f"-> prag {EYE_OPEN_CLOSED:.2f}  (separare {sep:.1f})")
        if sep < 2:
            print("ATENTIE: separare mica -> apasa c si reia la lumina mai buna.")
    else:
        EYE_OPEN_CLOSED = gaze.EYE_OPEN_CLOSED
        print(f"ATENTIE: deschis/inchis nu se deosebesc -> prag implicit {EYE_OPEN_CLOSED:.2f}")

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
        json.dump({"version": CALIB_VERSION, "jaw_open": JAW_OPEN,
                   "yaw_base": YAW_BASE, "pitch_base": PITCH_BASE, "roll_base": ROLL_BASE,
                   "gaze_h0": h0, "gaze_v0": v0, "eye_open_closed": EYE_OPEN_CLOSED,
                   "zones": ZONE_MODEL.to_json()}, f, indent=2)
    print(f"Gata. ochi<{EYE_OPEN_CLOSED:.2f}  jaw>{JAW_OPEN:.2f}  "
          f"cap drept: yaw {YAW_BASE:+.0f} pitch {PITCH_BASE:+.0f} roll {ROLL_BASE:+.0f}\n")


# ============================================================================
#  DRAWING
# ============================================================================
def draw_gaze_panel(frame, gm, x, y):
    """Panel with the gaze / attention metrics."""
    lines = [
        f"Privire: {gm['direction'] or 'pe drum'}",
        f"Ochi pe drum: {gm['on_road_ratio'] * 100:.0f}%",
        f"VATS: {gm['vats_s']:.0f}s / {gm['vats_glances']} priviri" + ("  ALERTA" if gm["vats_alert"] else ""),
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
        return gaze.GazeTracker(GAZE_CAL), gaze.BlinkTracker(EYE_OPEN_CLOSED), HeadTracker()

    gaze_tracker, blink_tracker, head_tracker = new_trackers()
    hand_tracker, health, alerts = HandFaceTracker(), CameraHealth(), AlertEngine()

    print("Pornit. q = iesire, c = recalibrare, l = CLAHE, m = desen puncte fata.")
    t_start = time.time()
    t0, frames, fps = t_start, 0, 0.0
    last_emit = 0.0
    frame_no, hand_res = 0, None
    driver_id = None              # stable ID of the current driver
    last_face_seen = t_start      # for face_lost_sec
    closed_frames = 0             # debounce of "both eyes below the threshold" (driver only)
    yawn_since = None
    prof = {}

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

        hands_px = [[(p.x * w, p.y * h) for p in pts] for pts in hand_res.hand_landmarks]

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
                    gaze_tracker, blink_tracker, head_tracker = new_trackers()
                    closed_frames, yawn_since = 0, None
                driver_id = ids[driver]

        for idx, pts in enumerate(lms):           # passengers: box + label only
            if idx == driver:
                continue
            xs, ys = [p.x * w for p in pts], [p.y * h for p in pts]
            cv2.rectangle(frame, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))), PASSENGER_COLOR, 2)
            put(frame, "Pasager", (int(min(xs)), int(min(ys)) - 8), PASSENGER_COLOR, 0.6)

        state = internal_state()
        state["ts"] = round(now, 2)
        state["fps"] = round(fps, 1)
        state["calibrated"] = CALIBRATED
        state["face_found"] = driver is not None
        state["passenger_count"] = max(0, n_faces - 1)
        state.update(cam)
        gm = bm = None
        xy = None
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

            # eyes: openness from the iris, per eye
            eyes_hidden = abs(yaw_d) > YAW_EYES_MAX
            look_down_score = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2 if bs else 0.0
            look_down = look_down_score > LOOK_DOWN_BS
            op = gaze.eye_openness(xy)
            thr = EYE_OPEN_CLOSED * (LOOK_DOWN_RELAX if look_down else 1.0)
            if op is not None and not eyes_hidden:
                ce_r, ce_l = op[0] < thr, op[1] < thr
                below = ce_r and ce_l                    # "eyes closed" = BOTH (one eye = blink/wink)
                one_eye = (ce_r != ce_l) and abs(yaw_d) <= YAW_TURNED   # turned head: the far eye just looks closed
            else:
                below = one_eye = False
            closed_frames = closed_frames + 1 if below else 0
            need = max(CLOSED_MIN_FRAMES, round(CLOSED_MIN_S * fps)) if fps else CLOSED_MIN_FRAMES
            closed = closed_frames >= need
            openness = (op[0] + op[1]) / 2 if op is not None else None

            # gaze zone (head + eyes; head only while the eyes are closed)
            g = gaze.gaze_ratios(xy)
            zone, zone_dist = ZONE_MODEL.classify(yaw_d, pitch_d, g if (g and not closed) else None)

            # mouth / yawn (jawOpen with hysteresis)
            jaw = bs.get("jawOpen") if bs else None
            if jaw is not None:
                wide = jaw > (JAW_OPEN * YAWN_RELEASE if yawn_since is not None else JAW_OPEN)
            else:
                wide = False
            yawn_since = (yawn_since or now) if wide else None
            yawning = wide and now - yawn_since >= YAWN_MIN_S

            # trackers
            bm = blink_tracker.update(now, closed=closed, openness=openness)
            gm = gaze_tracker.update(now, g, eyes_open=not closed, zone=zone,
                                     blinks_per_min=bm["blinks_per_min"], warmed_up=bm["warmed_up"])
            hm = head_tracker.update(now, yaw_d, pitch_d, roll_d, closed, ZONE_MODEL.down_sign)

            state.update({
                "eyes_closed_sec": round(bm["closed_sec"], 1),
                "since_last_blink_sec": round(bm["since_last_blink"], 1),
                "one_eye_closed": one_eye,
                "yawning": yawning,
                "head_yaw": round(yaw_d, 1), "head_pitch": round(pitch_d, 1), "head_roll": round(roll_d, 1),
                **hm,
            })
            for k in ("blink_rate", "avg_blink_sec", "slow_blinks", "perclos", "perclos_drowsy",
                      "microsleep", "sleep", "drowsiness_level", "squinting"):
                state[k] = bm[k]
            state.update({
                "looking_away_sec": gm["away_sec"], "gaze": gm["direction"] or "on_road",
                "on_road_ratio": gm["on_road_ratio"], "vats_s": gm["vats_s"], "vats_glances": gm["vats_glances"], "vats_alert": gm["vats_alert"],
                "mirror_checks": gm["mirror_checks"], "mirror_too_long": gm["mirror_too_long"],
                "long_glance": gm["long_glance"], "fixation_s": gm["fixation_s"], "hypnosis": gm["hypnosis"],
            })

            # driver overlay
            eye_col = (150, 150, 150) if eyes_hidden else ((0, 0, 255) if closed else (0, 255, 255))
            for eye in (LEFT_EYE, RIGHT_EYE):
                poly = np.array([[int(xy[i][0]), int(xy[i][1])] for i in eye], np.int32)
                cv2.polylines(frame, [poly], True, eye_col, 1)
            for i in IRIS_POINTS:
                cv2.circle(frame, (int(xy[i][0]), int(xy[i][1])), 1, (255, 255, 0), -1)
            mouth_col = (0, 0, 255) if yawning else ((0, 165, 255) if wide else (0, 255, 0))
            mt, mb = (tuple(int(v) for v in xy[MOUTH_TOP_BOTTOM[0]]), tuple(int(v) for v in xy[MOUTH_TOP_BOTTOM[1]]))
            ml, mr = (tuple(int(v) for v in xy[MOUTH_CORNERS[0]]), tuple(int(v) for v in xy[MOUTH_CORNERS[1]]))
            cv2.line(frame, mt, mb, mouth_col, 2)
            cv2.line(frame, ml, mr, mouth_col, 2)

            eye_state = ("n/a (cap intors)" if eyes_hidden else "INCHISI" if closed else
                         "un ochi inchis" if one_eye else "deschisi (privire jos)" if look_down else "deschisi")
            eye_val = f"R{op[0]:.2f} L{op[1]:.2f}" if op is not None else "-"
            put(frame, f"Sofer  cap yaw {yaw_d:+.0f} pitch {pitch_d:+.0f} roll {roll_d:+.0f}", (x0, y0 - 68), (255, 255, 255))
            put(frame, f"Ochi {eye_val} - {eye_state}", (x0, y0 - 48), eye_col)
            put(frame, f"Privire: {gm['direction'] or 'pe drum'}", (x0, y0 - 28), (255, 255, 0))
            mouth_txt = "CASCA!" if yawning else ("gura deschisa" if wide else "gura inchisa")
            put(frame, f"Gura jaw {jaw if jaw is not None else 0:.2f}/{JAW_OPEN:.2f} - {mouth_txt}", (x0, y0 - 8), mouth_col)
            closure_txt = (f"OCHI INCHISI {bm['closed_sec']:.1f}s" if bm["closed_sec"] > 0 else
                           f"ultimul clipit acum {bm['since_last_blink']:.0f}s")
            put(frame, closure_txt, (x0, y1 + 20), eye_col)
            draw_gaze_panel(frame, gm, w - 206, 72)

            dbg = {"open_r": round(op[0], 3) if op else "", "open_l": round(op[1], 3) if op else "",
                   "eye_thr": round(thr, 3), "look_down": round(look_down_score, 3), "closed": int(closed),
                   "jaw": round(jaw, 3) if jaw is not None else "", "jaw_thr": round(JAW_OPEN, 3),
                   "yawning": int(yawning), "yaw": round(yaw_d, 1), "pitch": round(pitch_d, 1),
                   "roll": round(roll_d, 1), "gaze_h": round(g[0], 3) if g else "",
                   "gaze_v": round(g[1], 3) if g else "", "zone": zone, "zone_dist": round(zone_dist, 2)}

        state["face_lost_sec"] = 0.0 if driver is not None else round(now - last_face_seen, 1)

        # ---------- hands (relative to the driver's face) ----------
        state.update(hand_tracker.update(now, hands_px, xy))
        state["hands_visible"] = len(hands_px)
        for hand in hands_px:
            ipts = [(int(x), int(y)) for x, y in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, ipts[a], ipts[b], (230, 230, 230), 2)
            for p in ipts:
                cv2.circle(frame, p, 3, (255, 0, 255), -1)

        # ---------- objects (YOLO, optional) ----------
        dets = detector.latest(now)
        state.update(objects.relate(dets, xy, hands_px))
        state["object_detection"] = detector.available
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
            write_state(state)
            if PRINT_STATE:
                print(state)
            last_emit = now

        # ---------- overlays: alert banner, top bar, FPS ----------
        lvl = state["alert_level"]
        if lvl != "none":
            cv2.rectangle(frame, (0, h - 95), (w, h - 55), ALERT_COLORS[lvl], -1)
            others = ", ".join(state["active_alerts"][1:4])
            put(frame, f"{lvl.upper()}: {state['alert_reason']}" + (f"  (+ {others})" if others else ""),
                (12, h - 67), (255, 255, 255), 0.6)
        cv2.rectangle(frame, (0, 0), (w, 50), (25, 25, 25), -1)

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
            gaze_tracker, blink_tracker, head_tracker = new_trackers()
            closed_frames, yawn_since = 0, None
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

"""MediaPipe - fete (2) + maini + IRIS/privire din webcam.

Instalare:   uv pip install mediapipe opencv-python
Rulare:      uv run python face_basic.py
Taste:       q = iesire   c = recalibrare   l = CLAHE (lumina slaba) pornit/oprit

Foloseste API-ul nou `mediapipe.tasks` (mp.solutions din tutorialele vechi nu mai exista).

Ce detecteaza si cum:
  - OCHI: blendshapes MediaPipe (eyeBlinkLeft/Right) sau EAR - calibrarea alege ce merge mai bine.
  - GURA / CASCAT: blendshape jawOpen (scor invatat). MAR ramane calculat (rezerva).
  - CAP: matricea de transformare a fetei -> yaw / pitch (grade) -> "priveste in lateral".
  - IRIS / PRIVIRE (gaze.py): pozitia irisului in ochi (punctele 468-477, deja date de model)
    -> directia privirii, pe drum vs in alta parte, raport ochi-pe-drum, VATS (priviri scurte
       repetate), distragere lunga, verificarea oglinzilor (pozitiv), fixatie / hipnoza de
       autostrada, saccade, gaze entropy si un mini-heatmap. Neutru calibrat la pasul 1.
  - LUMINA: CLAHE pe cadru inainte de detectie.
  - CALIBRARE per sofer, in 2 pasi (se salveaza in calib.json).
"""
import json          # pentru state.json + calib.json
import math
import os
import threading     # camera citita pe un fir separat (mai multe FPS)
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from state import new_state          # starea comuna (state.py, acelasi folder)
import gaze                          # NOU: toata logica de iris / privire (gaze.py)

# --- modele (se descarca automat prima data) ---
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# indici ochi: [colt_ext, sus1, sus2, colt_int, jos2, jos1] - ordine pentru formula EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
IRIS_POINTS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]   # NOU: punctele de iris

# gura: colturi (orizontal) + buza sus/jos (vertical), pentru MAR
MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
YAWN_MIN_S = 1.2           # gura larg deschisa atat ca sa fie cascat (nu vorbit)

# --- CALIBRARE ---
CALIB_FILE = "calib.json"
CALIB_VERSION = 5          # NOU: v5 adauga scanarea completa a privirii -> calib vechi se reface automat
CALIB_SECONDS = 4          # pozitia DREPT (ochi deschisi, postura neutra)
CALIB_CLOSED_SECONDS = 3   # pozitia finala: ochi inchisi (intre cele doua bipuri)
POSE_SECONDS = 2.5         # NOU: durata fiecarei pozitii din scanare (dreapta/stanga/sus/jos)
BLINK_K = 0.5
JAW_K = 0.5
MIN_EYE_SEPARATION = 1.0
EAR_MULT = 0.75
MAR_MULT = 1.40
EAR_CONSEC_FRAMES = 3
EYE_PEAK_WINDOW_S = 3.0

# --- CAP (grade fata de pozitia calibrata "drept inainte") ---
YAW_MAX = 25.0
PITCH_MAX = 20.0

# valori de rezerva (inainte de calibrare)
EYE_METRIC = "blink"
BLINK_CLOSED = 0.5
JAW_OPEN = 0.5
EAR_CLOSED = 0.20
MAR_OPEN = 0.55
YAW_BASE = 0.0
PITCH_BASE = 0.0
GAZE_CAL = gaze.GazeCalib()   # NOU: calibrarea privirii (neutru + raza), din scanarea completa
EYE_OPEN_CLOSED = 0.45        # deschidere_ochi / diametru_iris sub asta -> ochi inchis (per ochi)

# --- VITEZA (FPS) ---
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
CAM_DSHOW = True
DETECT_MAX_WIDTH = 640
HAND_EVERY = 2

# --- MAINI ---
HAND_DETECT_CONF = 0.7
HAND_PRESENCE_CONF = 0.7
HAND_TRACK_CONF = 0.7

USE_CLAHE = True
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# stari per fata (cheie = indexul fetei in cadru)
closed_frames = {}
closed_start = {}
open_since = {}            # NOU: de cand sunt ochii deschisi continuu
last_closure = {}
eye_hist = {}
yawn_since = {}
away_since = {}

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FACE_COLORS = [(0, 255, 0), (255, 150, 0)]
STATE_FILE = "state.json"
WINDOW = "MediaPipe - fete + maini + iris"


def ensure(path, url):
    if not os.path.exists(path):
        print(f"Descarc {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Gata.")


def eye_aspect_ratio(xy, idx):
    p1, p2, p3, p4, p5, p6 = (np.array(xy[i]) for i in idx)
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = 2.0 * np.linalg.norm(p1 - p4)
    return vert / horiz if horiz else 0.0


def mouth_aspect_ratio(xy):
    top = np.array(xy[MOUTH_TOP_BOTTOM[0]])
    bottom = np.array(xy[MOUTH_TOP_BOTTOM[1]])
    left = np.array(xy[MOUTH_CORNERS[0]])
    right = np.array(xy[MOUTH_CORNERS[1]])
    width = np.linalg.norm(left - right)
    return np.linalg.norm(top - bottom) / width if width else 0.0


def open_capture(index):
    cap = None
    if os.name == "nt" and CAM_DSHOW:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    return cap


class Camera:
    """Citeste camera pe un fir separat (cat modelele lucreaza, camera pregateste cadrul urmator)."""

    def __init__(self, index=0):
        self.cap = open_capture(index)
        self.ok, self.frame, self.id, self.last_read = False, None, 0, 0
        self.cond = threading.Condition()
        self.running = self.cap.isOpened()
        self.thread = None
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok, self.frame = ok, frame
                self.id += 1
                if not ok:
                    self.running = False
                self.cond.notify_all()

    def isOpened(self):
        return self.running

    def read(self, timeout=2.0):
        with self.cond:
            got = self.cond.wait_for(lambda: self.id > self.last_read or not self.running, timeout)
            if not got or self.id == self.last_read:
                return False, None
            self.last_read = self.id
            return self.ok, self.frame

    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()


_last_ts = 0


def next_ts():
    global _last_ts
    _last_ts = max(int(time.monotonic() * 1000), _last_ts + 1)
    return _last_ts


def to_mp_image(frame):
    h, w = frame.shape[:2]
    if w > DETECT_MAX_WIDTH:
        frame = cv2.resize(frame, (DETECT_MAX_WIDTH, int(h * DETECT_MAX_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(enhance(frame), cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def enhance(frame):
    if not USE_CLAHE:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def blendshapes(face_res, idx):
    if face_res.face_blendshapes and idx < len(face_res.face_blendshapes):
        return {c.category_name: c.score for c in face_res.face_blendshapes[idx]}
    return {}


def head_angles(face_res, idx):
    mats = face_res.facial_transformation_matrixes
    if not mats or idx >= len(mats):
        return None
    r = np.array(mats[idx])[:3, :3]
    r = r / np.linalg.norm(r, axis=0)
    yaw = math.degrees(math.asin(float(np.clip(-r[2, 0], -1.0, 1.0))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def circular_mean(deg_values):
    rad = np.radians(deg_values)
    return math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


def box_area(pts):
    return ((max(p.x for p in pts) - min(p.x for p in pts)) *
            (max(p.y for p in pts) - min(p.y for p in pts)))


def beep():
    try:
        import winsound
        winsound.Beep(1000, 250)
    except Exception:
        print("\a", end="", flush=True)


def separation(open_vals, closed_vals, closed_is_higher):
    if len(open_vals) < 5 or len(closed_vals) < 5:
        return None
    diff = float(np.median(closed_vals) - np.median(open_vals))
    if not closed_is_higher:
        diff = -diff
    return diff / (float(np.std(open_vals)) + float(np.std(closed_vals)) + 1e-6)


def draw_gaze_panel(frame, gm, x, y):
    """Panou cu metricile de privire + mini-heatmap (unde s-a uitat soferul)."""
    lines = [
        f"Privire: {gm['direction'] or 'pe drum'}",
        f"Ochi pe drum: {gm['on_road_ratio'] * 100:.0f}%",
        f"VATS: {gm['vats_s']:.0f}s" + ("  ALERTA" if gm["vats_alert"] else ""),
        f"Oglinzi: {gm['mirror_checks']}",
        f"Saccade/s: {gm['saccade_rate']:.1f}",
        f"Entropie: {gm['entropy']:.2f}",
        f"Fixatie: {gm['fixation_s']:.0f}s" + ("  HIPNOZA" if gm["hypnosis"] else ""),
    ]
    cv2.rectangle(frame, (x - 6, y - 18), (x + 200, y + len(lines) * 18 + 86), (25, 25, 25), -1)
    for i, t in enumerate(lines):
        col = (0, 200, 255) if ("ALERTA" in t or "HIPNOZA" in t) else (220, 220, 220)
        cv2.putText(frame, t, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    # mini-heatmap sub text
    grid, gmax = gm["heat"], max(gm["heat_max"], 1.0)
    cell, hy = 16, y + len(lines) * 18 + 6
    cv2.putText(frame, "Heatmap privire:", (x, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            inten = int(255 * grid[r, c] / gmax)
            cv2.rectangle(frame, (x + c * cell, hy + 6 + r * cell),
                          (x + c * cell + cell - 1, hy + 6 + r * cell + cell - 1),
                          (0, inten, inten), -1)


def load_calib():
    global EYE_METRIC, BLINK_CLOSED, JAW_OPEN, EAR_CLOSED, MAR_OPEN, YAW_BASE, PITCH_BASE, GAZE_CAL
    if not os.path.exists(CALIB_FILE):
        return False
    try:
        with open(CALIB_FILE) as f:
            c = json.load(f)
        if c.get("version") != CALIB_VERSION:
            print("calib.json e de la o versiune mai veche -> recalibrez.")
            return False
        EYE_METRIC = c["eye_metric"]
        BLINK_CLOSED, JAW_OPEN = c["blink_closed"], c["jaw_open"]
        EAR_CLOSED, MAR_OPEN = c["ear_closed"], c["mar_open"]
        YAW_BASE, PITCH_BASE = c["yaw_base"], c["pitch_base"]
        GAZE_CAL = gaze.GazeCalib(c["gaze_h0"], c["gaze_v0"], c["gaze_h_half"],
                                  c["gaze_v_half"], c["gaze_h_sign"], c["gaze_v_sign"])
    except (KeyError, ValueError):
        print("calib.json corupt -> recalibrez.")
        return False
    print(f"Calibrare incarcata: ochi pe {EYE_METRIC}  jaw>{JAW_OPEN:.2f}  "
          f"cap drept: yaw {YAW_BASE:+.0f} pitch {PITCH_BASE:+.0f}  "
          f"privire: neutru h{GAZE_CAL.h0:.2f} v{GAZE_CAL.v0:.2f} "
          f"raza h{GAZE_CAL.h_half:.2f} v{GAZE_CAL.v_half:.2f}")
    return True


def _collect(cap, face, seconds, text, skip=0.0):
    data = {"ear": [], "mar": [], "blink": [], "jaw": [], "yaw": [], "pitch": [],
            "gaze_h": [], "gaze_v": []}                              # NOU: gaze_h / gaze_v
    t_start = time.time()
    t_end = t_start + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = face.detect_for_video(to_mp_image(frame), next_ts())
        if res.face_landmarks:
            i = max(range(len(res.face_landmarks)),
                    key=lambda k: box_area(res.face_landmarks[k]))
            xy = [(p.x * w, p.y * h) for p in res.face_landmarks[i]]
            for x, y in xy:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            if time.time() - t_start >= skip:
                data["ear"].append((eye_aspect_ratio(xy, LEFT_EYE) +
                                    eye_aspect_ratio(xy, RIGHT_EYE)) / 2)
                data["mar"].append(mouth_aspect_ratio(xy))
                bs = blendshapes(res, i)
                if bs:
                    data["blink"].append((bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2)
                    data["jaw"].append(bs["jawOpen"])
                ang = head_angles(res, i)
                if ang:
                    data["yaw"].append(ang[0])
                    data["pitch"].append(ang[1])
                g = gaze.gaze_ratios(xy)                             # NOU
                if g:
                    data["gaze_h"].append(g[0])
                    data["gaze_v"].append(g[1])
        left = int(t_end - time.time()) + 1
        cv2.putText(frame, f"{text}  {left}s", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, "q = anuleaza calibrarea", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
    return data


def calibrate(cap, face):
    """Scanare completa a privirii (stil Face ID) + calibrare ochi/cap pentru soferul curent.

    Pozitii: DREPT (neutru) -> DREAPTA -> STANGA -> SUS -> JOS -> OCHI INCHISI.
    La pozitiile laterale tii capul drept si misti DOAR ochii. De aici invatam cat de
    departe iti misca ochii in fiecare directie -> praguri PERSONALE + semnul stanga/dreapta
    aflat singur (fara GAZE_H_SIGN manual). q = anuleaza in orice moment.
    """
    global EYE_METRIC, BLINK_CLOSED, JAW_OPEN, EAR_CLOSED, MAR_OPEN, YAW_BASE, PITCH_BASE, GAZE_CAL
    print("\n=== SCANARE COMPLETA ===")
    front = _collect(cap, face, CALIB_SECONDS,
                     "1/6 PRIVESTE DREPT, ochii deschisi, gura inchisa", skip=0.5)
    if front is None or not front["ear"]:
        print("Scanare anulata/esuata (nicio fata). Pastrez valorile vechi.")
        return
    right = _collect(cap, face, POSE_SECONDS, "2/6 Capul drept, PRIVESTE la DREAPTA (doar ochii)")
    left = _collect(cap, face, POSE_SECONDS, "3/6 Capul drept, PRIVESTE la STANGA (doar ochii)")
    up = _collect(cap, face, POSE_SECONDS, "4/6 Capul drept, PRIVESTE in SUS (doar ochii)")
    down = _collect(cap, face, POSE_SECONDS, "5/6 Capul drept, PRIVESTE in JOS (doar ochii)")
    if any(x is None for x in (right, left, up, down)):
        print("Scanare anulata, pastrez valorile vechi.")
        return
    print(f"6/6 ({CALIB_CLOSED_SECONDS}s): la BIP inchide ochii, redeschide-i la al doilea BIP.")
    beep()
    closed = _collect(cap, face, CALIB_CLOSED_SECONDS,
                      "6/6 tine OCHII INCHISI pana la al doilea bip", skip=0.7)
    beep()
    if closed is None:
        print("Scanare anulata, pastrez valorile vechi.")
        return

    med = lambda lst, d: float(np.median(lst)) if lst else d

    # --- ochi / gura / cap / gaze neutru (din pozitia DREPT) ---
    ear_open, mar_mean = med(front["ear"], 0.28), med(front["mar"], 0.02)
    MAR_OPEN = MAR_MULT * mar_mean
    blink_open = med(front["blink"], None) if front["blink"] else None
    jaw_base = med(front["jaw"], None) if front["jaw"] else None
    if jaw_base is not None:
        JAW_OPEN = float(np.clip(jaw_base + JAW_K * (1 - jaw_base), 0.30, 0.80))
    if front["yaw"]:
        YAW_BASE, PITCH_BASE = circular_mean(front["yaw"]), circular_mean(front["pitch"])
    h0, v0 = med(front["gaze_h"], 0.5), med(front["gaze_v"], 0.5)

    # --- ochi inchisi (blink/EAR) din DREPT vs INCHIS ---
    sep_blink = separation(front["blink"], closed["blink"], closed_is_higher=True)
    sep_ear = separation(front["ear"], closed["ear"], closed_is_higher=False)
    blink_closed_val = med(closed["blink"], None) if closed["blink"] else None
    ear_closed_val = med(closed["ear"], None) if closed["ear"] else None
    if sep_blink is not None and sep_blink > 0:
        BLINK_CLOSED = blink_open + BLINK_K * (blink_closed_val - blink_open)
    elif blink_open is not None:
        BLINK_CLOSED = float(np.clip(blink_open + BLINK_K * (1 - blink_open), 0.30, 0.85))
    if sep_ear is not None and sep_ear > 0:
        EAR_CLOSED = ear_open - BLINK_K * (ear_open - ear_closed_val)
    else:
        EAR_CLOSED = EAR_MULT * ear_open
    cands = {k: v for k, v in (("blink", sep_blink), ("ear", sep_ear)) if v is not None}
    best = max(cands, key=cands.get) if cands else None
    EYE_METRIC = best if (best and cands[best] >= MIN_EYE_SEPARATION) else \
        ("blink" if blink_open is not None else "ear")
    if not best or cands.get(best, 0) < MIN_EYE_SEPARATION:
        print("ATENTIE: ochii deschisi/inchisi se deosebesc slab -> apasa c si reia la lumina mai buna.")

    # --- raza privirii din pozitiile laterale (asta e "scanarea completa") ---
    hr, hl = med(right["gaze_h"], h0 + 0.15), med(left["gaze_h"], h0 - 0.15)
    vu, vd = med(up["gaze_v"], v0 - 0.15), med(down["gaze_v"], v0 + 0.15)
    h_half = (abs(hr - h0) + abs(hl - h0)) / 2 or 0.15
    v_half = (abs(vu - v0) + abs(vd - v0)) / 2 or 0.15
    if h_half < 0.05:
        h_half = 0.15
    if v_half < 0.05:
        v_half = 0.15
    h_sign = int(np.sign(hr - h0)) or 1
    v_sign = int(np.sign(vd - v0)) or 1
    GAZE_CAL = gaze.GazeCalib(h0, v0, h_half, v_half, h_sign, v_sign)

    with open(CALIB_FILE, "w") as f:
        json.dump({"version": CALIB_VERSION, "eye_metric": EYE_METRIC,
                   "blink_closed": BLINK_CLOSED, "jaw_open": JAW_OPEN,
                   "ear_closed": EAR_CLOSED, "mar_open": MAR_OPEN,
                   "yaw_base": YAW_BASE, "pitch_base": PITCH_BASE,
                   "gaze_h0": h0, "gaze_v0": v0, "gaze_h_half": h_half, "gaze_v_half": v_half,
                   "gaze_h_sign": h_sign, "gaze_v_sign": v_sign}, f, indent=2)
    print(f"Gata. ochi: {EYE_METRIC}  cap drept: yaw {YAW_BASE:+.0f} pitch {PITCH_BASE:+.0f}  "
          f"privire: neutru h{h0:.2f} v{v0:.2f}  raza h{h_half:.2f} v{v_half:.2f}  "
          f"semn h{h_sign:+d} v{v_sign:+d}\n")


def main():
    global USE_CLAHE
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

    gaze_tracker = gaze.GazeTracker(GAZE_CAL)             # NOU: metricile de privire pentru sofer

    print("Pornit. q = iesire, c = recalibrare, l = CLAHE pornit/oprit.")
    t0, frames, fps = time.time(), 0, 0.0
    last_emit = 0.0
    frame_no, hand_res = 0, None
    prof = {}

    def tick(name, start):
        ms = (time.perf_counter() - start) * 1000
        prof[name] = 0.9 * prof.get(name, ms) + 0.1 * ms

    while True:
        t_cam = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        tick("camera", t_cam)
        h, w = frame.shape[:2]

        t = time.perf_counter()
        mp_image = to_mp_image(frame)
        ts = next_ts()
        tick("prep", t)

        t = time.perf_counter()
        face_res = face.detect_for_video(mp_image, ts)
        tick("fata", t)

        t = time.perf_counter()
        if hand_res is None or frame_no % HAND_EVERY == 0:
            hand_res = hands.detect_for_video(mp_image, ts)
        tick("maini", t)
        frame_no += 1
        t_draw = time.perf_counter()
        now = time.time()

        # --- fete ---
        n_faces = len(face_res.face_landmarks)
        areas = []
        face_info = {}        # NOU: per fata (gaze, away, eyes_open, one_eye, yawning) pentru sofer/state
        blink_ended = set()   # NOU: fete care tocmai au clipit (inchidere incheiata)
        for idx, pts in enumerate(face_res.face_landmarks):
            col = FACE_COLORS[idx % len(FACE_COLORS)]
            xy = [(p.x * w, p.y * h) for p in pts]
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            x0, y0 = int(min(xs)), int(min(ys))
            y1 = int(max(ys))
            areas.append(box_area(pts))
            cv2.rectangle(frame, (x0, y0), (int(max(xs)), y1), col, 2)
            for x, y in xy:
                cv2.circle(frame, (int(x), int(y)), 1, col, -1)

            bs = blendshapes(face_res, idx)

            # --- cap ---
            ang = head_angles(face_res, idx)
            if ang:
                yaw_d = angle_diff(ang[0], YAW_BASE)
                pitch_d = angle_diff(ang[1], PITCH_BASE)
            else:
                yaw_d = pitch_d = 0.0
            turned = abs(yaw_d) > YAW_MAX
            away = turned or abs(pitch_d) > PITCH_MAX
            if away:
                away_since.setdefault(idx, now)
            else:
                away_since.pop(idx, None)
            away_sec = now - away_since[idx] if away else 0.0

            # --- ochi: deschidere din IRIS, PER OCHI (robust la distanta, prinde si un singur ochi) ---
            ear = (eye_aspect_ratio(xy, LEFT_EYE) + eye_aspect_ratio(xy, RIGHT_EYE)) / 2
            blink = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2 if bs else None
            use_blink = EYE_METRIC == "blink" and blink is not None
            op = gaze.eye_openness(xy)                   # (dreapta, stanga) sau None
            if op is not None:
                ce_r, ce_l = op[0] < EYE_OPEN_CLOSED, op[1] < EYE_OPEN_CLOSED
                iris_closed, one_eye = (ce_r and ce_l), (ce_r != ce_l)
                below = iris_closed                      # "ochi inchisi" = AMBII (un ochi = clipit/wink)
            else:                                        # rezerva daca lipsesc punctele de iris
                ce_r = ce_l = iris_closed = one_eye = False
                below = blink > BLINK_CLOSED if use_blink else ear < EAR_CLOSED
            if turned:
                below = False
            closed_frames[idx] = closed_frames.get(idx, 0) + 1 if below else 0
            closed = closed_frames[idx] >= EAR_CONSEC_FRAMES
            if closed:
                closed_start.setdefault(idx, now)
                open_since.pop(idx, None)               # ochii s-au inchis -> opreste cronometrul "deschis"
            else:
                open_since.setdefault(idx, now)         # de cand sunt deschisi continuu
                if idx in closed_start:
                    start = closed_start.pop(idx)
                    last_closure[idx] = (now, now - start)
                    blink_ended.add(idx)                # NOU: a clipit / a deschis ochii

            hist = eye_hist.setdefault(idx, deque())
            hist.append((now, blink if use_blink else ear))
            while hist and now - hist[0][0] > EYE_PEAK_WINDOW_S:
                hist.popleft()
            peak = (max(v for _, v in hist) if use_blink else min(v for _, v in hist))

            eye_col = ((150, 150, 150) if turned else
                       (0, 0, 255) if closed else (0, 255, 255))
            for eye in (LEFT_EYE, RIGHT_EYE):
                poly = np.array([[int(xy[i][0]), int(xy[i][1])] for i in eye], np.int32)
                cv2.polylines(frame, [poly], True, eye_col, 1)
                for i in eye:
                    cv2.circle(frame, (int(xy[i][0]), int(xy[i][1])), 2, eye_col, -1)

            # --- IRIS: desen + directia privirii (NOU) ---
            g = gaze.gaze_ratios(xy)
            for i in IRIS_POINTS:                       # punctele de iris, cyan
                cv2.circle(frame, (int(xy[i][0]), int(xy[i][1])), 1, (255, 255, 0), -1)
            cv2.circle(frame, (int(xy[468][0]), int(xy[468][1])), 3, (255, 255, 0), 1)
            cv2.circle(frame, (int(xy[473][0]), int(xy[473][1])), 3, (255, 255, 0), 1)

            # --- gura ---
            mar = mouth_aspect_ratio(xy)
            if bs:
                jaw = bs["jawOpen"]
                wide = jaw > JAW_OPEN
            else:
                jaw = None
                wide = mar > MAR_OPEN
            if wide:
                yawn_since.setdefault(idx, now)
            else:
                yawn_since.pop(idx, None)
            yawning = wide and (now - yawn_since.get(idx, now)) >= YAWN_MIN_S
            mouth_col = (0, 0, 255) if yawning else ((0, 165, 255) if wide else (0, 255, 0))
            mt = (int(xy[MOUTH_TOP_BOTTOM[0]][0]), int(xy[MOUTH_TOP_BOTTOM[0]][1]))
            mb = (int(xy[MOUTH_TOP_BOTTOM[1]][0]), int(xy[MOUTH_TOP_BOTTOM[1]][1]))
            ml = (int(xy[MOUTH_CORNERS[0]][0]), int(xy[MOUTH_CORNERS[0]][1]))
            mr = (int(xy[MOUTH_CORNERS[1]][0]), int(xy[MOUTH_CORNERS[1]][1]))
            cv2.line(frame, mt, mb, mouth_col, 2)
            cv2.line(frame, ml, mr, mouth_col, 2)
            for pt in (mt, mb, ml, mr):
                cv2.circle(frame, pt, 2, mouth_col, -1)

            # tot ce trimitem in state pentru aceasta fata (soferul se alege mai jos)
            face_info[idx] = {"gaze": g, "away": away, "eyes_open": not closed,
                              "one_eye": one_eye, "yawning": yawning}

            # --- texte deasupra fetei ---
            head_col = (0, 0, 255) if away else (255, 255, 255)
            head_txt = f"Cap yaw {yaw_d:+.0f} pitch {pitch_d:+.0f}"
            if away:
                head_txt += f" - NU PRIVESTE DRUMUL {away_sec:.1f}s"
            if turned:
                eye_state = "n/a (cap intors)"
            elif closed:
                eye_state = "AMBII INCHISI"
            elif one_eye:
                eye_state = "un ochi inchis (clipit/wink?)"
            else:
                eye_state = "deschisi"
            if op is not None:
                eye_val = f"deschidere R{op[0]:.2f} L{op[1]:.2f}"
            else:
                eye_val = f"blink {blink:.2f}" if blink is not None else f"EAR {ear:.2f}"
            mouth_txt = "CASCA!" if yawning else ("gura deschisa" if wide else "gura inchisa")
            mouth_val = f"jaw {jaw:.2f}" if jaw is not None else "MAR"
            gaze_txt = ""
            if g:
                d = gaze.direction(g[0], g[1], GAZE_CAL)
                gaze_txt = f"Privire: {d or 'pe drum'}"

            cv2.putText(frame, f"Fata {idx + 1}", (x0, y0 - 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(frame, head_txt, (x0, y0 - 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_col, 2)
            cv2.putText(frame, f"Ochi {eye_val} - {eye_state}",
                        (x0, y0 - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_col, 2)
            cv2.putText(frame, gaze_txt, (x0, y0 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Gura {mouth_val} MAR {mar:.2f} - {mouth_txt}",
                        (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_col, 2)

            if idx in closed_start:
                closure_txt = f"OCHI INCHISI {now - closed_start[idx]:.1f}s"
            elif idx in last_closure:
                end, dur = last_closure[idx]
                closure_txt = f"Ultima inchidere: {dur:.1f}s, acum {now - end:.0f}s"
            else:
                closure_txt = "Ochii nu au fost detectati inchisi inca"
            cv2.putText(frame, closure_txt, (x0, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_col, 2)

        for d in (closed_frames, closed_start, open_since, last_closure, eye_hist, yawn_since, away_since):
            for gone in [k for k in d if k >= n_faces]:
                d.pop(gone, None)

        # soferul = fata cea mai mare
        driver = max(range(n_faces), key=lambda k: areas[k]) if n_faces else None
        driver_away_sec = (now - away_since[driver]
                           if driver is not None and driver in away_since else 0.0)
        driver_closed_sec = (now - closed_start[driver]
                             if driver is not None and driver in closed_start else 0.0)
        driver_open_sec = (now - open_since[driver]
                           if driver is not None and driver in open_since else 0.0)
        driver_since_blink = (now - last_closure[driver][0]      # secunde de la ultima clipire
                              if driver is not None and driver in last_closure else driver_open_sec)

        # --- IRIS: hranim tracker-ul DOAR cu soferul + desenam panoul (NOU) ---
        if driver is not None:
            di = face_info[driver]
            if driver in blink_ended:
                gaze_tracker.note_blink(now)
            gm = gaze_tracker.update(now, di["gaze"], head_away=di["away"], eyes_open=di["eyes_open"])
            draw_gaze_panel(frame, gm, w - 206, 72)
        else:
            di, gm = None, None

        # --- maini ---
        n_hands = len(hand_res.hand_landmarks)
        for idx, pts in enumerate(hand_res.hand_landmarks):
            xy = [(int(p.x * w), int(p.y * h)) for p in pts]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, xy[a], xy[b], (230, 230, 230), 2)
            for x, y in xy:
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
            label = "Mana"
            if hand_res.handedness:
                label = hand_res.handedness[idx][0].category_name
            cv2.putText(frame, label, (xy[0][0] - 10, xy[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # ====== umplem starea comuna (state.py) - toate alertele soferului ======
        state = new_state()
        state["face_found"] = n_faces > 0
        if driver is not None:
            state["eyes_closed_sec"] = round(driver_closed_sec, 1)   # ambii ochi inchisi
            state["eyes_open_sec"] = round(driver_open_sec, 1)        # de cat timp sunt ochii deschisi
            state["since_last_blink_sec"] = round(driver_since_blink, 1)  # secunde de la ultima clipire
            state["one_eye_closed"] = di["one_eye"]                   # doar un ochi (clipit/wink)
            state["yawning"] = di["yawning"]                          # cascat
        if gm:
            state["looking_away_sec"] = gm["away_sec"]               # nu se uita la drum (cap sau privire)
            state["gaze"] = gm["direction"] or "on_road"
            state["on_road_ratio"] = gm["on_road_ratio"]
            state["vats_s"] = gm["vats_s"]
            state["vats_alert"] = gm["vats_alert"]
            state["mirror_checks"] = gm["mirror_checks"]
            state["fixation_s"] = gm["fixation_s"]                   # fixatie
            state["hypnosis"] = gm["hypnosis"]                       # hipnoza de autostrada

        if now - last_emit > 0.2:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            print(state)
            last_emit = now
        # ==================================================

        # --- info sus (2 randuri) ---
        cv2.rectangle(frame, (0, 0), (w, 50), (25, 25, 25), -1)
        cv2.putText(frame, f"Fete: {n_faces}   Maini: {n_hands}   CLAHE: {'ON' if USE_CLAHE else 'OFF'}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        eye_thr = (f"blink>{BLINK_CLOSED:.2f}" if EYE_METRIC == "blink" else f"EAR<{EAR_CLOSED:.2f}")
        cv2.putText(frame, f"ochi {eye_thr}  jaw>{JAW_OPEN:.2f}  "
                           f"[c] recalibrare  [l] CLAHE  [q] iesire",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        frames += 1
        if now - t0 >= 1:
            fps, frames, t0 = frames / (now - t0), 0, now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        prof_txt = "  ".join(f"{k} {v:.0f}" for k, v in prof.items()) + " ms"
        cv2.putText(frame, prof_txt, (110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        tick("desen", t_draw)
        if key == ord("q"):
            break
        if key == ord("c"):
            calibrate(cap, face)
            gaze_tracker = gaze.GazeTracker(GAZE_CAL)           # NOU: reia cu noua calibrare
        if key == ord("l"):
            USE_CLAHE = not USE_CLAHE

    cap.release()
    face.close()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

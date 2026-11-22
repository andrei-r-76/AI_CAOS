"""Calibration of one driver: his normal head pose, eye / yawn thresholds and gaze zones.

Poses: LOOK AT THE ROAD -> each zone in ZONE_POSES -> EYES CLOSED. Each pose starts with a short pause
and a beep, and ends with a beep; c cancels. The values are saved per driver (calib_<name>.json).
"""
import json
import os
import time

import cv2
import numpy as np

import gaze
from camera import next_ts, to_mp_image
from eyewear import SUNGLASSES_REL, eye_cheek_ratio
from head import angle_diff, circular_mean, head_angles
from tracking import blendshapes, box_area, box_of, to_pixels
from utils import beep, top_bar

ZONE_POSES = [             # TUNE: gaze zones learned at calibration (look the way you would while driving)
    (gaze.ZONE_LEFT, "Look LEFT (mirror / window)"),
    (gaze.ZONE_RIGHT, "Look RIGHT (mirror / window)"),
    (gaze.ZONE_UP, "Look UP (rear-view mirror)"),
    (gaze.ZONE_DOWN, "Look DOWN (dashboard)"),
]
CALIB_VERSION = 9          # calib files of this version load (extra keys are optional)
KNOWN_ZONES = {gaze.ROAD_ZONE} | {name for name, _ in ZONE_POSES}   # older files with other zone names recalibrate
CALIB_SECONDS = 4          # TUNE: ROAD pose duration (eyes open, neutral)
CALIB_CLOSED_SECONDS = 3   # TUNE: last pose: eyes closed (between two beeps)
POSE_SECONDS = 2.5         # TUNE: each gaze-zone pose
PREP_SECONDS = 3.0         # TUNE: pause before each pose (shows what comes next)
JAW_K = 0.5                # TUNE: yawn threshold = jaw_base + K * (1 - jaw_base). Lower = easier yawn
JAW_MIN, JAW_MAX = 0.30, 0.80   # TUNE: limits for the yawn threshold
EYE_OPEN_K = 0.4           # TUNE: closed threshold = closed + K * (open - closed). Lower = fewer false "closed"
EYE_OPEN_MIN, EYE_OPEN_MAX = 0.20, 0.70   # TUNE: limits for the closed threshold
CANCELLED = "Calibration cancelled, keeping the old values."
NEXT_COLOR = (9, 83, 180)  # BGR of the dashboard's amber (--warn): "get ready" screens
POSE_COLOR = (28, 28, 185) # BGR of the dashboard's red (--danger): pose being measured
CANCEL_KEYS = (ord("c"), ord("q"))   # c - cancel (q also comes when the window is closed)


def separation(open_vals, closed_vals):
    """How well openness separates open from closed eyes (median difference / sum of std devs)."""
    if len(open_vals) < 5 or len(closed_vals) < 5:
        return None
    diff = float(np.median(open_vals) - np.median(closed_vals))
    return diff / (float(np.std(open_vals)) + float(np.std(closed_vals)) + 1e-6)


def _prepare(cap, face, show, seconds, text):
    """Pause before a pose: preview + "next: ..." + countdown (nothing measured). False if cancelled."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = face.detect_for_video(to_mp_image(frame), next_ts())   # keeps MediaPipe tracking warm
        if res.face_landmarks:
            for p in res.face_landmarks[0]:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 200, 255), -1)
        left = int(t_end - time.time()) + 1
        top_bar(frame, f"NEXT: {text}", NEXT_COLOR, f"starts in {left}s   |   c - cancel")
        if show(frame) in CANCEL_KEYS:
            return False
    return True


def _collect(cap, face, show, seconds, text, skip=0.0):
    """Measurements of every frame for `seconds` (the first `skip` seconds are ignored). None if cancelled."""
    data = {"jaw": [], "yaw": [], "pitch": [], "roll": [], "gaze_h": [], "gaze_v": [], "open": [], "dark": []}
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
            xy = to_pixels(res.face_landmarks[i], w, h)
            dark = eye_cheek_ratio(frame, xy, box_of(xy))   # before drawing the dots
            for x, y in xy:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            if time.time() - t_start >= skip:
                if dark is not None:
                    data["dark"].append(dark)
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
                op = gaze.eye_openness(xy)                   # both eyes pooled
                if op is not None:
                    data["open"].extend(op)
        left = int(t_end - time.time()) + 1
        if res.face_landmarks:
            top_bar(frame, text, POSE_COLOR, f"{left}s left   |   c - cancel")
        else:
            top_bar(frame, text, POSE_COLOR, f"FACE NOT DETECTED - move into view   |   {left}s left", POSE_COLOR)
        if show(frame) in CANCEL_KEYS:
            return None
    return data


class Calibration:
    """The current driver's calibrated values (defaults until loaded or calibrated)."""

    def __init__(self):
        self.path = "calib.json"                    # set per driver (drivers.calib_file)
        self.jaw_open = 0.5                         # yawn threshold (jawOpen)
        self.yaw_base = self.pitch_base = self.roll_base = 0.0   # head pose when looking at the road
        self.eye_open_closed = gaze.EYE_OPEN_CLOSED  # eye-closed threshold
        self.zone_model = gaze.ZoneModel.default()  # gaze zones
        self.dark_base = None                       # normal "dark" value (sunglasses check)
        self.calibrated = False

    def head_pose(self, face_res, idx):
        """(yaw, pitch, roll) of the face relative to the road pose; zeros if unknown."""
        ang = head_angles(face_res, idx)
        if not ang:
            return 0.0, 0.0, 0.0
        return (angle_diff(ang[0], self.yaw_base), angle_diff(ang[1], self.pitch_base),
                angle_diff(ang[2], self.roll_base))

    def load(self):
        """Load self.path. False if missing / old / damaged (then calibrate)."""
        self.calibrated = False                     # until this driver's calibration loads
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                c = json.load(f)
            if c.get("version") != CALIB_VERSION or not c.get("zones") or not set(c["zones"]) <= KNOWN_ZONES:
                print(f"{self.path} is from an older version -> calibrating again.")
                return False
            self.jaw_open = c["jaw_open"]
            self.yaw_base, self.pitch_base = c["yaw_base"], c["pitch_base"]
            self.roll_base = c.get("roll_base", 0.0)
            self.eye_open_closed = c["eye_open_closed"]
            self.zone_model = gaze.ZoneModel(c["zones"])
            self.dark_base = c.get("dark_base")     # older calibrations: None -> fixed sunglasses threshold
        except (KeyError, ValueError, TypeError):
            print(f"{self.path} is damaged -> calibrating again.")
            return False
        except OSError as e:                        # locked (antivirus, another program) or unreadable
            print(f"Cannot read {self.path} ({e}) -> calibrating again.")
            return False
        self.calibrated = True
        print(f"Calibration loaded: eyes closed below {self.eye_open_closed:.2f}  jaw > {self.jaw_open:.2f}  "
              f"road head pose: yaw {self.yaw_base:+.0f} pitch {self.pitch_base:+.0f} roll {self.roll_base:+.0f}  "
              f"zones: {', '.join(self.zone_model.zones)}")
        return True

    def run(self, cap, face, show):
        """Calibrate the driver in front of the camera and save to self.path.
        At each zone look as you would while driving (head, eyes or both): head yaw/pitch + iris are saved."""
        print("\n=== CALIBRATION ===")
        n_steps = len(ZONE_POSES) + 2
        poses = [("front", CALIB_SECONDS, f"1/{n_steps} LOOK AT THE ROAD (straight), eyes open, mouth closed")]
        poses += [(name, POSE_SECONDS, f"{i + 2}/{n_steps} {text}") for i, (name, text) in enumerate(ZONE_POSES)]
        got = {}
        for name, secs, text in poses:
            print(text)
            if not _prepare(cap, face, show, PREP_SECONDS, text):
                print(CANCELLED)
                return
            beep()                                   # "start now"
            got[name] = _collect(cap, face, show, secs, text, skip=0.5)
            if got[name] is None:
                print(CANCELLED)
                return
            beep()                                   # "pose done"
            if name == "front" and not got[name]["yaw"]:
                print("Calibration failed (no face). Keeping the old values.")
                return
        front = got["front"]

        text = f"{n_steps}/{n_steps} At the BEEP CLOSE YOUR EYES, open them at the 2nd BEEP"
        print(f"{text} ({CALIB_CLOSED_SECONDS}s)")
        if not _prepare(cap, face, show, PREP_SECONDS, text):
            print(CANCELLED)
            return
        beep()                                       # "close your eyes"
        closed = _collect(cap, face, show, CALIB_CLOSED_SECONDS,
                          f"{n_steps}/{n_steps} keep your EYES CLOSED until the 2nd beep", skip=0.7)
        beep()                                       # "pose done: open your eyes"
        if closed is None:
            print(CANCELLED)
            return

        med = lambda lst, d: float(np.median(lst)) if lst else d

        # --- mouth / head / iris on the road (ROAD pose) ---
        if front["jaw"]:
            jaw_base = med(front["jaw"], 0.0)
            self.jaw_open = float(np.clip(jaw_base + JAW_K * (1 - jaw_base), JAW_MIN, JAW_MAX))
        self.yaw_base, self.pitch_base = circular_mean(front["yaw"]), circular_mean(front["pitch"])
        self.roll_base = circular_mean(front["roll"])
        h0, v0 = med(front["gaze_h"], 0.5), med(front["gaze_v"], 0.5)
        self.dark_base = med(front["dark"], None)   # normal eye/cheek brightness (calibrate WITHOUT sunglasses)
        if self.dark_base is not None:
            print(f"Normal dark {self.dark_base:.2f} -> sunglasses below {self.dark_base * SUNGLASSES_REL:.2f}")

        # --- eye-closed threshold: ROAD (open) vs CLOSED ---
        sep = separation(front["open"], closed["open"])
        if sep is not None and sep > 0:
            open_med, closed_med = med(front["open"], 0.9), med(closed["open"], 0.15)
            self.eye_open_closed = float(np.clip(closed_med + EYE_OPEN_K * (open_med - closed_med),
                                                 EYE_OPEN_MIN, EYE_OPEN_MAX))
            print(f"Eye openness: open {open_med:.2f}  closed {closed_med:.2f}  "
                  f"-> threshold {self.eye_open_closed:.2f}  (separation {sep:.1f})")
            if sep < 2:
                print("WARNING: small separation -> press Recalibrate in better light.")
        else:
            self.eye_open_closed = gaze.EYE_OPEN_CLOSED
            print(f"WARNING: open / closed look the same -> default threshold {self.eye_open_closed:.2f}")

        # --- gaze zones: head yaw/pitch (vs the road pose) + iris ---
        zones = {gaze.ROAD_ZONE: (0.0, 0.0, h0, v0)}
        for name, _ in ZONE_POSES:
            z = got[name]
            if len(z["yaw"]) < 10:
                print(f"WARNING: zone {name}: only {len(z['yaw'])} frames with a face -> skipped (Recalibrate).")
                continue
            zy = angle_diff(circular_mean(z["yaw"]), self.yaw_base)
            zp = angle_diff(circular_mean(z["pitch"]), self.pitch_base)
            zh, zv = med(z["gaze_h"], None), med(z["gaze_v"], None)
            iris_ok = (zh is not None and zv is not None and abs(zy) <= gaze.YAW_EYES_MAX
                       and 0.0 <= zh <= 1.0 and 0.0 <= zv <= 1.0)
            if not iris_ok:
                zh = zv = None      # head turned too far / iris out of range -> head only
            zones[name] = (zy, zp, zh, zv)
            print(f"  zone {name:6s} head yaw {zy:+5.0f} pitch {zp:+5.0f}   iris "
                  + (f"h{zh:.2f} v{zv:.2f}" if zh is not None else "-"))
        self.zone_model = gaze.ZoneModel(zones)
        self.calibrated = True

        try:
            with open(self.path, "w") as f:
                json.dump({"version": CALIB_VERSION, "jaw_open": self.jaw_open,
                           "yaw_base": self.yaw_base, "pitch_base": self.pitch_base, "roll_base": self.roll_base,
                           "eye_open_closed": self.eye_open_closed, "dark_base": self.dark_base,
                           "zones": self.zone_model.to_json()}, f, indent=2)
        except OSError as e:
            print(f"WARNING: cannot save {self.path} ({e}) - used for this session only.")
        print(f"Done. eyes < {self.eye_open_closed:.2f}  jaw > {self.jaw_open:.2f}  "
              f"road head pose: yaw {self.yaw_base:+.0f} pitch {self.pitch_base:+.0f} roll {self.roll_base:+.0f}\n")

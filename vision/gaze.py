"""Gaze zones, iris metrics and eye fatigue - all from MediaPipe Face Landmarker points.

The face_landmarker.task model outputs 478 points: the last 10 are the irises (468-477).
  - 468 = right iris center, ring 469-472
  - 473 = left iris center, ring 474-477

What lives here:
  1) ZoneModel: WHERE the driver looks (road, left, right, up, down), from head yaw/pitch + iris.
     The iris alone only says where the eyes point INSIDE THE HEAD: with the head turned a bit
     while looking at the road, the eyes compensate and the iris says "left". So at calibration
     the driver looks naturally (head + eyes) at each zone and the nearest zone wins live.
  2) GazeTracker: attention metrics over time (eyes-on-road ratio, long glance).
  3) eye_openness + BlinkTracker: eyes closed, how long the eyes stay open, blinks, microsleep.
     BlinkTracker is the ONLY place where closures and blinks are timed and counted.

TUNING NOTE: every line marked "TUNE:" is a value you can adjust. All times are in seconds.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# --- landmarks (MediaPipe Face Mesh 478 indices) ---
EYES = [
    {"corners": (33, 133), "lids": (159, 145), "iris": 468, "ring": [469, 470, 471, 472]},   # right eye
    {"corners": (362, 263), "lids": (386, 374), "iris": 473, "ring": [474, 475, 476, 477]},  # left eye
]

# --- eyes closed from the iris ---
EYE_OPEN_CLOSED = 0.45     # default eye-closed threshold; face_basic.py calibrates it per driver

# --- gaze zones (head + eyes) ---
# zone names - defined ONLY here; face_basic.ZONE_POSES and state.json use these
ROAD_ZONE = "drum"
ZONE_LEFT, ZONE_RIGHT, ZONE_UP, ZONE_DOWN = "stanga", "dreapta", "sus", "jos"
# calibrated zones and the generic fallback labels use the SAME names; ZONE_LEFT / ZONE_DOWN also
# tell which head-yaw sign is left and which head-pitch sign is down
ZONE_SCALE = {"yaw": 8.0, "pitch": 6.0, "h": 0.06, "v": 0.10}   # TUNE: how much each feature may vary
                           #   inside one zone (degrees for the head, iris ratio for the eyes).
                           #   Bigger = that feature matters less. Raise h/v if the iris is noisy.
ZONE_MAX_DIST = 2.5        # TUNE: farther than this from every calibrated zone -> label from the head alone
                           #   (stanga/dreapta/sus/jos), or the road if the head is roughly straight
ZONE_HEAD_YAW = 20.0       # TUNE: for the generic label: head yaw beyond this = looking sideways
ZONE_HEAD_PITCH = 15.0     # TUNE: for the generic label: head pitch beyond this = looking up/down
ZONE_SMOOTH = 5            # TUNE: the zone is the most common one over the last N frames (removes flicker)

# --- attention time windows ---
LONG_GLANCE_S = 3.0        # TUNE: one continuous glance away this long = long distraction
ONROAD_WINDOW = 30.0       # window for the eyes-on-road ratio


@dataclass
class GazeCalib:
    """Iris position when looking at the road (from calibration)."""
    h0: float = 0.5
    v0: float = 0.5


class ZoneModel:
    """Nearest calibrated gaze zone from head yaw/pitch (relative to the road pose) + iris (h, v)."""

    def __init__(self, zones):
        # zones: {name: (yaw, pitch, h, v)}; yaw/pitch in degrees relative to the road pose,
        # h/v = iris ratios (None if the eyes were not measured)
        self.zones = {k: tuple(v) for k, v in zones.items()}
        self.calibrated = len(self.zones) > 1
        # which head-yaw sign means "left" and which pitch sign means "down", learned from the zones
        left, right = self.zones.get(ZONE_LEFT), self.zones.get(ZONE_RIGHT)
        down = self.zones.get(ZONE_DOWN)
        if left and abs(left[0]) > 3:
            self.left_sign = float(np.sign(left[0]))
        elif right and abs(right[0]) > 3:
            self.left_sign = -float(np.sign(right[0]))
        else:
            self.left_sign = 1.0
        self.down_sign = float(np.sign(down[1])) if down and abs(down[1]) > 3 else 1.0

    @classmethod
    def default(cls):
        """No calibration yet: only the road, everything else from the head (left/right sign unknown)."""
        return cls({ROAD_ZONE: (0.0, 0.0, None, None)})

    def classify(self, yaw, pitch, gaze=None):
        """Returns (zone_name, distance). gaze=None (eyes closed / no iris) -> head only."""
        best, best_d = ROAD_ZONE, 1e9
        for name, (zy, zp, zh, zv) in self.zones.items():
            d = ((yaw - zy) / ZONE_SCALE["yaw"]) ** 2 + ((pitch - zp) / ZONE_SCALE["pitch"]) ** 2
            if gaze is not None and zh is not None and zv is not None:
                d += ((gaze[0] - zh) / ZONE_SCALE["h"]) ** 2 + ((gaze[1] - zv) / ZONE_SCALE["v"]) ** 2
            d = math.sqrt(d)
            if d < best_d:
                best, best_d = name, d
        if best_d <= ZONE_MAX_DIST:
            return best, best_d
        # far from every calibrated zone: generic direction from the head alone
        if abs(yaw) >= ZONE_HEAD_YAW and abs(yaw) >= abs(pitch):
            return (ZONE_LEFT if yaw * self.left_sign > 0 else ZONE_RIGHT), best_d
        if abs(pitch) >= ZONE_HEAD_PITCH:
            return (ZONE_DOWN if pitch * self.down_sign > 0 else ZONE_UP), best_d
        return ROAD_ZONE, best_d                 # head roughly straight: trust the road (iris noise)

    def to_json(self):
        return {k: list(v) for k, v in self.zones.items()}


def gaze_ratios(xy):
    """(h, v) = average iris position inside the eye, 0..1 (0.5 = centered). None if points are missing."""
    if len(xy) < 478:
        return None
    hs, vs = [], []
    for e in EYES:
        (ca, cb), (lt, lb), iris = e["corners"], e["lids"], e["iris"]
        left, right = min(xy[ca][0], xy[cb][0]), max(xy[ca][0], xy[cb][0])
        top, bottom = min(xy[lt][1], xy[lb][1]), max(xy[lt][1], xy[lb][1])
        ix, iy = xy[iris][0], xy[iris][1]
        if right - left > 1e-3:
            hs.append((ix - left) / (right - left))
        if bottom - top > 1e-3:
            vs.append((iy - top) / (bottom - top))
    if not hs or not vs:
        return None
    return float(np.mean(hs)), float(np.mean(vs))


def eye_openness(xy):
    """Per eye: vertical opening / iris diameter. Returns (right, left) or None."""
    if len(xy) < 478:
        return None
    out = []
    for e in EYES:
        lt, lb = e["lids"]
        center = np.array(xy[e["iris"]])
        lid = float(np.hypot(xy[lt][0] - xy[lb][0], xy[lt][1] - xy[lb][1]))
        diam = 2.0 * float(np.mean([np.hypot(xy[r][0] - center[0], xy[r][1] - center[1])
                                    for r in e["ring"]]))
        out.append(lid / diam if diam > 1e-3 else 0.0)
    return out[0], out[1]


class GazeTracker:
    """Attention metrics for the DRIVER (a single person), over time windows."""

    def __init__(self):
        self.samples = deque()          # (t, on_road) for the eyes-on-road ratio
        self.zone_hist = deque(maxlen=ZONE_SMOOTH)
        self.ep = None                  # start time of the current look away from the road, or None

    def update(self, t, zone):
        """zone: raw zone from ZoneModel.classify for this frame (head + eyes, or head only)."""
        # zone, smoothed: the most common one over the last ZONE_SMOOTH frames
        self.zone_hist.append(zone or ROAD_ZONE)
        zone = max(set(self.zone_hist), key=list(self.zone_hist).count)
        away = zone != ROAD_ZONE
        on_road = not away

        # --- how long the driver has been looking away from the road ---
        if away:
            self.ep = self.ep if self.ep is not None else t
        else:
            self.ep = None
        away_sec = (t - self.ep) if self.ep is not None else 0.0

        self.samples.append((t, on_road))
        while self.samples and t - self.samples[0][0] > ONROAD_WINDOW:
            self.samples.popleft()

        # --- eyes-on-road ratio over the last ONROAD_WINDOW seconds ---
        onroad_t = total_t = 0.0
        prev_t = None
        for tt, orr in self.samples:
            if prev_t is not None:
                dt = tt - prev_t
                total_t += dt
                if orr:
                    onroad_t += dt
            prev_t = tt
        on_road_ratio = (onroad_t / total_t) if total_t > 0 else 1.0

        return {
            "direction": "" if zone == ROAD_ZONE else zone,   # '' = on the road
            "zone": zone,
            "on_road": on_road, "away_sec": round(away_sec, 2),
            "on_road_ratio": round(on_road_ratio, 3),
            "long_glance": away_sec >= LONG_GLANCE_S,
        }


# ============================================================================
#  FATIGUE from the eyes: closures, blinks, microsleep
# ============================================================================
MICROSLEEP_S = 1.0         # TUNE: eyes closed 1-3 s = microsleep
SLEEP_S = 3.0              # TUNE: eyes closed over 3 s = asleep
SLOW_BLINK_S = 0.40        # TUNE: blinks longer than this = slow/heavy (a sign of fatigue)
WARMUP_S = 20.0            # blink-rate based judgements need at least this much data


class BlinkTracker:
    """Fatigue metrics from the (confirmed) eye state + eye openness. The single source of truth
    for closures: eyes_closed_sec, since_last_blink, blink counts, microsleep."""

    def __init__(self, closed_threshold=None):
        # closed_threshold is no longer used (it was for squinting); kept so face_basic.py needs no change
        self.blinks = deque()        # (t_end, duration) from the last 60 s
        self.closed_since = None
        self.last_blink_end = None
        self.t0 = None

    def update(self, t, closed, openness=None):
        """closed: eyes closed (debounced, from face_basic.py). openness is no longer used (kept for compatibility)."""
        if self.t0 is None:
            self.t0 = t

        # closure episodes (each one is a blink, or a microsleep if long)
        blink_ended = False
        if closed:
            self.closed_since = self.closed_since if self.closed_since is not None else t
        elif self.closed_since is not None:
            self.blinks.append((t, t - self.closed_since))
            self.closed_since = None
            self.last_blink_end = t
            blink_ended = True
        while self.blinks and t - self.blinks[0][0] > 60.0:
            self.blinks.popleft()
        closed_sec = (t - self.closed_since) if self.closed_since is not None else 0.0
        since_blink = t - (self.last_blink_end if self.last_blink_end is not None else self.t0)

        durs = [d for _, d in self.blinks]
        blink_rate = len(durs)                       # blinks in the last minute (low during the first minute)
        elapsed = min(60.0, t - self.t0)
        blinks_per_min = blink_rate * 60.0 / elapsed if elapsed > 0 else 0.0   # scaled during warm-up
        avg_blink = float(np.mean(durs)) if durs else 0.0
        slow_blinks = sum(1 for d in durs if d > SLOW_BLINK_S)

        # microsleep / sleep from the duration of the current closure
        level = "sleep" if closed_sec >= SLEEP_S else \
                ("microsleep" if closed_sec >= MICROSLEEP_S else "awake")

        return {
            "closed_sec": closed_sec,
            "since_last_blink": since_blink,
            "eyes_open_sec": round(since_blink, 1) if not closed else 0.0,   # eyes open continuously, s
            "blink_ended": blink_ended,
            "blink_rate": blink_rate,
            "blinks_per_min": round(blinks_per_min, 1),
            "warmed_up": (t - self.t0) >= WARMUP_S,
            "avg_blink_sec": round(avg_blink, 2),
            "slow_blinks": slow_blinks,
            "microsleep": level == "microsleep",
            "sleep": level == "sleep",
        }
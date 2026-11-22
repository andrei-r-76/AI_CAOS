"""Gaze zones, iris metrics and eye fatigue, from the MediaPipe Face Landmarker points.

The model gives 478 points; the last 10 are the irises: 468 = right iris center (ring 469-472),
473 = left iris center (ring 474-477).

  1) ZoneModel: WHERE the driver looks (road, left, right, up, down) from head yaw/pitch + iris.
     The iris alone only says where the eyes point inside the head (head turned a bit + eyes on the
     road = iris says "left"), so at calibration the driver looks naturally at each zone and the
     nearest zone wins.
  2) GazeTracker: attention over time (eyes-on-road ratio, long glance).
  3) eye_openness + EyeTracker: are the eyes closed (debounced, relaxed when looking down).
  4) BlinkTracker: blinks and microsleep (the ONLY place closures are timed).

Every "TUNE:" line can be adjusted. Times are in seconds.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

# --- landmarks ---
EYES = [
    {"corners": (33, 133), "lids": (159, 145), "iris": 468, "ring": [469, 470, 471, 472]},   # right eye
    {"corners": (362, 263), "lids": (386, 374), "iris": 473, "ring": [474, 475, 476, 477]},  # left eye
]

IRIS_CENTERS = (468, 473)  # right / left iris center

# --- closed eyes ---
EYE_OPEN_CLOSED = 0.45     # default eye-closed threshold (calibrated per driver)
CLOSED_MIN_S = 0.07        # TUNE: eyes below the threshold this long = closed...
CLOSED_MIN_FRAMES = 2      # TUNE: ...and never fewer frames than this (single-frame noise)
YAW_EYES_MAX = 45.0        # TUNE: eyes are checked up to this head yaw (beyond it one eye is hidden)
LOOK_DOWN_BS = 0.45        # TUNE: eyeLookDown blendshape above this = looking down (phone / dashboard)
LOOK_DOWN_RELAX = 0.6      # TUNE: looking down: an eye is closed only below threshold * this

# --- gaze zones (head + eyes) ---
# zone names are defined ONLY here (calibration, screen and state.json use them)
ROAD_ZONE = "road"
ZONE_LEFT, ZONE_RIGHT, ZONE_UP, ZONE_DOWN = "left", "right", "up", "down"
ZONE_SCALE = {"yaw": 8.0, "pitch": 6.0, "h": 0.06, "v": 0.10}   # TUNE: how much each feature may vary in one
                           #   zone (degrees for the head, iris ratio for the eyes). Bigger = matters less.
ZONE_MAX_DIST = 2.5        # TUNE: farther than this from every zone -> label from the head alone
ZONE_HEAD_YAW = 20.0       # TUNE: head-only label: yaw beyond this = looking sideways
ZONE_HEAD_PITCH = 15.0     # TUNE: head-only label: pitch beyond this = looking up / down
ZONE_SMOOTH = 5            # TUNE: zone = most common one over the last N frames (no flicker)

# --- attention ---
LONG_GLANCE_S = 3.0        # TUNE: one look away this long = long glance
ONROAD_WINDOW = 30.0       # window for the eyes-on-road ratio


class ZoneModel:
    """Nearest calibrated zone from head yaw/pitch (vs the road pose) + iris (h, v)."""

    def __init__(self, zones):
        # zones: {name: (yaw, pitch, h, v)}; h/v = iris ratios, None if the eyes were not measured
        self.zones = {k: tuple(v) for k, v in zones.items()}
        # which yaw sign is "left" and which pitch sign is "down", learned from the zones
        left, right, down = self.zones.get(ZONE_LEFT), self.zones.get(ZONE_RIGHT), self.zones.get(ZONE_DOWN)
        if left and abs(left[0]) > 3:
            self.left_sign = float(np.sign(left[0]))
        elif right and abs(right[0]) > 3:
            self.left_sign = -float(np.sign(right[0]))
        else:
            self.left_sign = 1.0
        self.down_sign = float(np.sign(down[1])) if down and abs(down[1]) > 3 else 1.0

    @classmethod
    def default(cls):
        """No calibration yet: only the road; everything else from the head."""
        return cls({ROAD_ZONE: (0.0, 0.0, None, None)})

    def classify(self, yaw, pitch, gaze=None):
        """(zone, distance). gaze=None (eyes closed / no iris) -> head only."""
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
        # far from every zone: direction from the head alone
        if abs(yaw) >= ZONE_HEAD_YAW and abs(yaw) >= abs(pitch):
            return (ZONE_LEFT if yaw * self.left_sign > 0 else ZONE_RIGHT), best_d
        if abs(pitch) >= ZONE_HEAD_PITCH:
            return (ZONE_DOWN if pitch * self.down_sign > 0 else ZONE_UP), best_d
        return ROAD_ZONE, best_d                 # head roughly straight: trust the road (iris noise)

    def to_json(self):
        return {k: list(v) for k, v in self.zones.items()}


def gaze_ratios(xy):
    """(h, v) = average iris position inside the eyes, 0..1 (0.5 = centered), or None."""
    if len(xy) < 478:
        return None
    hs, vs = [], []
    for e in EYES:
        (ca, cb), (lt, lb), iris = e["corners"], e["lids"], e["iris"]
        left, right = min(xy[ca][0], xy[cb][0]), max(xy[ca][0], xy[cb][0])
        top, bottom = min(xy[lt][1], xy[lb][1]), max(xy[lt][1], xy[lb][1])
        ix, iy = xy[iris]
        if right - left > 1e-3:
            hs.append((ix - left) / (right - left))
        if bottom - top > 1e-3:
            vs.append((iy - top) / (bottom - top))
    if not hs or not vs:
        return None
    return float(np.mean(hs)), float(np.mean(vs))


def eye_openness(xy):
    """Per eye: lid gap / iris diameter. (right, left) or None."""
    if len(xy) < 478:
        return None
    out = []
    for e in EYES:
        lt, lb = e["lids"]
        center = np.array(xy[e["iris"]])
        lid = float(np.hypot(xy[lt][0] - xy[lb][0], xy[lt][1] - xy[lb][1]))
        diam = 2.0 * float(np.mean([np.hypot(xy[r][0] - center[0], xy[r][1] - center[1]) for r in e["ring"]]))
        out.append(lid / diam if diam > 1e-3 else 0.0)
    return out[0], out[1]


class EyeTracker:
    """Eyes closed = BOTH eyes below the threshold for CLOSED_MIN_S (relaxed while looking down).
    Not checked when the eyes cannot be seen: head turned too far, sunglasses, camera covered, hand on the eyes."""

    def __init__(self):
        self.closed_frames = 0       # frames in a row with both eyes below the threshold

    def update(self, xy, yaw, bs, covered, fps, threshold):
        """yaw: head yaw vs the road pose; bs: blendshapes; covered: eyes hidden by sunglasses / a hand /
        a covered camera; threshold: calibrated eye-closed threshold."""
        hidden = abs(yaw) > YAW_EYES_MAX
        look_down_score = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2 if bs else 0.0
        op = eye_openness(xy)
        checked = op is not None and not hidden and not covered
        thr = threshold * (LOOK_DOWN_RELAX if look_down_score > LOOK_DOWN_BS else 1.0)
        below = checked and op[0] < thr and op[1] < thr
        self.closed_frames = self.closed_frames + 1 if below else 0
        need = max(CLOSED_MIN_FRAMES, round(CLOSED_MIN_S * fps)) if fps else CLOSED_MIN_FRAMES
        return {"openness": op, "closed": self.closed_frames >= need, "hidden": hidden, "covered": covered,
                "checked": checked}

    def lost(self):
        """The face is not visible: start counting again when it comes back."""
        self.closed_frames = 0


class GazeTracker:
    """Attention metrics for the driver over time."""

    def __init__(self):
        self.samples = deque()          # (t, on_road) for the eyes-on-road ratio
        self.zone_hist = deque(maxlen=ZONE_SMOOTH)
        self.away_since = None          # start of the current look away, or None

    def update(self, t, zone):
        """zone: this frame's zone from ZoneModel.classify."""
        self.zone_hist.append(zone or ROAD_ZONE)
        zone = max(set(self.zone_hist), key=list(self.zone_hist).count)   # smoothed
        on_road = zone == ROAD_ZONE
        if on_road:
            self.away_since = None
        elif self.away_since is None:
            self.away_since = t
        away_sec = t - self.away_since if self.away_since is not None else 0.0

        self.samples.append((t, on_road))
        while self.samples and t - self.samples[0][0] > ONROAD_WINDOW:
            self.samples.popleft()
        onroad_t = total_t = 0.0                    # time on the road over the last ONROAD_WINDOW
        for (t1, _), (t2, orr) in zip(self.samples, list(self.samples)[1:]):
            total_t += t2 - t1
            onroad_t += (t2 - t1) if orr else 0.0

        return {
            "direction": "" if on_road else zone,   # "" = on the road
            "away_sec": round(away_sec, 2),
            "on_road_ratio": round(onroad_t / total_t, 3) if total_t > 0 else 1.0,
            "long_glance": away_sec >= LONG_GLANCE_S,
        }


# ============================================================================
#  FATIGUE from the eyes: closures, blinks, microsleep
# ============================================================================
MICROSLEEP_S = 1.0         # TUNE: eyes closed 1-3 s = microsleep
SLEEP_S = 3.0              # TUNE: eyes closed over 3 s = asleep
SLOW_BLINK_S = 0.40        # TUNE: blinks longer than this = slow / heavy (fatigue)
BLINK_WINDOW_S = 60.0      # blinks are counted over the last minute


class BlinkTracker:
    """Closures and blinks of the (debounced) eye state: the single source of truth for them.
    A blink = the eyes closed then opened again; its duration is the closed time
    (longer than SLOW_BLINK_S = slow / heavy blink)."""

    def __init__(self):
        self.blinks = deque()        # (t_end, duration) over the last 60 s
        self.closed_since = None
        self.last_blink_end = None
        self.t0 = None
        self.watch_start = None      # since when the eyes have been visible without a break

    def drop(self):
        """The eyes cannot be seen (face lost, sunglasses, camera covered, hand on the eyes): the current
        closure is not timed (no false microsleep) and the blink rate needs a new full minute."""
        self.closed_since = None
        self.watch_start = None

    def update(self, t, closed, checked=True):
        """closed: eyes closed (debounced); checked: the eyes could be seen in this frame."""
        if not checked:
            self.drop()
            closed = False
        if self.t0 is None:
            self.t0 = t
        if self.watch_start is None:
            self.watch_start = t
        if closed:
            self.closed_since = self.closed_since if self.closed_since is not None else t
        elif self.closed_since is not None:          # a closure ended = one blink (or a microsleep)
            self.blinks.append((t, t - self.closed_since))
            self.closed_since = None
            self.last_blink_end = t
        while self.blinks and t - self.blinks[0][0] > BLINK_WINDOW_S:
            self.blinks.popleft()

        closed_sec = t - self.closed_since if self.closed_since is not None else 0.0
        since_blink = t - (self.last_blink_end if self.last_blink_end is not None else self.t0)
        durs = [d for _, d in self.blinks]
        return {
            "closed_sec": closed_sec,
            "since_last_blink": since_blink,
            "eyes_open_sec": round(since_blink, 1) if not closed else 0.0,
            "blink_rate": len(durs),                 # blinks in the last minute
            "blink_rate_valid": t - self.watch_start >= BLINK_WINDOW_S,   # eyes seen a full minute
            "slow_blinks": sum(1 for d in durs if d > SLOW_BLINK_S),
            "microsleep": MICROSLEEP_S <= closed_sec < SLEEP_S,
            "sleep": closed_sec >= SLEEP_S,
        }

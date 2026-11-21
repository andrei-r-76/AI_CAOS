"""Gaze zones, iris metrics and eye fatigue - all from MediaPipe Face Landmarker points.

The face_landmarker.task model outputs 478 points: the last 10 are the irises (468-477).
  - 468 = right iris center, ring 469-472
  - 473 = left iris center, ring 474-477

What lives here:
  1) ZoneModel: WHERE the driver looks (road, mirrors, dashboard), from head yaw/pitch + iris.
     The iris alone only says where the eyes point INSIDE THE HEAD: with the head turned a bit
     while looking at the road, the eyes compensate and the iris says "left". So at calibration
     the driver looks naturally (head + eyes) at each zone and the nearest zone wins live.
  2) GazeTracker: attention metrics over time (eyes-on-road ratio, VATS, long glance, mirror
     checks, fixation / highway hypnosis).
  3) eye_openness + BlinkTracker: eyes closed, blinks, PERCLOS, microsleep, squinting.
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
ROAD_ZONE = "drum"
MIRROR_ZONES = ("oglinda_stanga", "oglinda_dreapta", "oglinda_mijloc")
ZONE_SCALE = {"yaw": 8.0, "pitch": 6.0, "h": 0.06, "v": 0.10}   # TUNE: how much each feature may vary
                           #   inside one zone (degrees for the head, iris ratio for the eyes).
                           #   Bigger = that feature matters less. Raise h/v if the iris is noisy.
ZONE_MAX_DIST = 2.5        # TUNE: farther than this from every zone -> generic label from the head
                           #   (stanga/dreapta/sus/jos), or the road if the head is roughly straight
ZONE_HEAD_YAW = 20.0       # TUNE: for the generic label: head yaw beyond this = looking sideways
ZONE_HEAD_PITCH = 15.0     # TUNE: for the generic label: head pitch beyond this = looking up/down
ZONE_SMOOTH = 5            # TUNE: the zone is the most common one over the last N frames (removes flicker)

# --- attention time windows ---
# VATS = visual time-sharing: REPEATED SHORT glances at the phone / dashboard (the "texting" pattern).
# Each glance is too short for long_glance, but together they keep the eyes off the road.
VATS_WINDOW = 30.0         # TUNE: window for visual time-sharing
VATS_ZONES = ("bord", "jos")   # TUNE: zones that count (phone / dashboard). Mirrors and side looks
                           #       (blind spot) are normal driving; long looks have their own alert
VATS_MIN_GLANCE_S = 0.5    # TUNE: shorter glances are ignored (zone flicker / noise, not a real glance)
VATS_MIN_GLANCES = 3       # TUNE: at least this many glances in the window (repeated, not one look)...
VATS_SUM_ALERT = 7.0       # TUNE: ...adding up to at least this many seconds -> vats_alert
LONG_GLANCE_S = 3.0        # TUNE: one continuous glance away this long = long distraction
MIRROR_MIN_S = 0.25        # TUNE: shortest glance that can be a mirror check
MIRROR_ZONE_MAX_S = 2.0    # TUNE: a mirror glance up to this long = mirror check; longer = mirror_too_long
MIRROR_DEBOUNCE_S = 0.8    # minimum time between two counted mirror checks
ONROAD_WINDOW = 30.0       # window for the eyes-on-road ratio
FIX_WINDOW = 1.5           # window used to decide whether the gaze is still moving
FIXATION_MOVE = 0.03       # TUNE: gaze spread below this (raw ratio) = gaze is fixed
HYPNOSIS_S = 8.0           # TUNE: fixation this long (+ rare blinks) = highway hypnosis.
                           #       8 s is good for a demo; 20-30 s is more realistic on the road
HYPNOSIS_BLINKS_MIN = 8    # TUNE: fewer blinks per minute than this counts as "rare blinking"


@dataclass
class GazeCalib:
    """Iris position when looking at the road (from calibration); used for fixation."""
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
        left, right = self.zones.get("oglinda_stanga"), self.zones.get("oglinda_dreapta")
        down = self.zones.get("bord")
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
            return ("stanga" if yaw * self.left_sign > 0 else "dreapta"), best_d
        if abs(pitch) >= ZONE_HEAD_PITCH:
            return ("jos" if pitch * self.down_sign > 0 else "sus"), best_d
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

    def __init__(self, cal: GazeCalib | None = None):
        self.cal = cal or GazeCalib()
        self.smooth = deque(maxlen=3)   # last 3 iris samples, averaged to reduce jitter
        self.samples = deque()          # (t, dh, dv, on_road, has_gaze)
        self.zone_hist = deque(maxlen=ZONE_SMOOTH)
        self.ep = None                  # current episode away from the road (dict) or None
        self.last_mirror_t = -1e9
        self.episodes = deque()         # (t_end, duration) of finished SHORT glances at VATS_ZONES
        self.mirrors = deque()          # times of counted mirror checks
        self.fix_since = None

    @staticmethod
    def _vats_glance(ep, dur):
        """A VATS glance: mostly at a VATS zone, long enough to be real, shorter than a long glance."""
        return (ep["n"] >= 2 and ep["vats_n"] >= 0.6 * ep["n"]
                and VATS_MIN_GLANCE_S <= dur < LONG_GLANCE_S)

    def update(self, t, gaze, eyes_open, zone, blinks_per_min=None, warmed_up=False):
        """gaze: iris (h, v) or None; zone: raw zone from ZoneModel.classify for this frame;
        blinks_per_min / warmed_up: from BlinkTracker (used for highway hypnosis)."""
        cal = self.cal
        has_gaze = gaze is not None and eyes_open
        if has_gaze:
            self.smooth.append(gaze)
            sh = float(np.mean([g[0] for g in self.smooth]))
            sv = float(np.mean([g[1] for g in self.smooth]))
            dh, dv = sh - cal.h0, sv - cal.v0
        else:
            self.smooth.clear()
            dh = dv = 0.0

        # zone (head + eyes; works with the eyes closed too, from the head), smoothed
        self.zone_hist.append(zone or ROAD_ZONE)
        zone = max(set(self.zone_hist), key=list(self.zone_hist).count)
        away = zone != ROAD_ZONE
        on_road = not away

        # --- episode away from the road ---
        if away:
            if self.ep is None:
                self.ep = {"start": t, "n": 0, "mirror_n": 0, "vats_n": 0}
            self.ep["n"] += 1
            if zone in MIRROR_ZONES:
                self.ep["mirror_n"] += 1
            if zone in VATS_ZONES:
                self.ep["vats_n"] += 1
        elif self.ep is not None:
            ep, dur = self.ep, t - self.ep["start"]
            mostly_mirror = ep["n"] >= 2 and ep["mirror_n"] >= 0.6 * ep["n"]
            if mostly_mirror and MIRROR_MIN_S <= dur <= MIRROR_ZONE_MAX_S:
                if t - self.last_mirror_t > MIRROR_DEBOUNCE_S:
                    self.mirrors.append(t)
                    self.last_mirror_t = t
            elif self._vats_glance(ep, dur):
                self.episodes.append((t, dur))
            self.ep = None

        away_sec = (t - self.ep["start"]) if self.ep is not None else 0.0
        in_mirror = (self.ep is not None and self.ep["mirror_n"] >= 0.6 * self.ep["n"])
        mirror_too_long = in_mirror and away_sec > MIRROR_ZONE_MAX_S

        self.samples.append((t, dh, dv, on_road, has_gaze))
        while self.samples and t - self.samples[0][0] > ONROAD_WINDOW:
            self.samples.popleft()
        while self.episodes and t - self.episodes[0][0] > VATS_WINDOW:
            self.episodes.popleft()
        while self.mirrors and t - self.mirrors[0] > 60.0:
            self.mirrors.popleft()

        # --- fixation / highway hypnosis ---
        recent = [(hh, vv) for tt, hh, vv, _, hg in self.samples if hg and t - tt <= FIX_WINDOW]
        if len(recent) >= 5:
            spread = math.hypot(float(np.std([r[0] for r in recent])),
                                float(np.std([r[1] for r in recent])))
            moving = spread > FIXATION_MOVE
        else:
            moving = True
        if has_gaze and on_road and not moving:
            self.fix_since = self.fix_since or t
        else:
            self.fix_since = None
        fixation_s = (t - self.fix_since) if self.fix_since is not None else 0.0
        hypnosis = (warmed_up and blinks_per_min is not None and fixation_s >= HYPNOSIS_S
                    and blinks_per_min < HYPNOSIS_BLINKS_MIN)

        # --- eyes-on-road ratio over the last ONROAD_WINDOW seconds ---
        onroad_t = total_t = 0.0
        prev_t = None
        for tt, _, _, orr, _ in self.samples:
            if prev_t is not None and t - tt <= ONROAD_WINDOW:
                dt = tt - prev_t
                total_t += dt
                if orr:
                    onroad_t += dt
            prev_t = tt
        on_road_ratio = (onroad_t / total_t) if total_t > 0 else 1.0

        # VATS: finished short glances at the phone/dashboard + the current one if it qualifies so far
        cur = self.ep is not None and self._vats_glance(self.ep, away_sec)
        vats_s = sum(d for _, d in self.episodes) + (away_sec if cur else 0.0)
        vats_glances = len(self.episodes) + (1 if cur else 0)
        vats_alert = vats_glances >= VATS_MIN_GLANCES and vats_s >= VATS_SUM_ALERT

        return {
            "direction": "" if zone == ROAD_ZONE else zone,   # '' = on the road
            "zone": zone,
            "on_road": on_road, "away_sec": round(away_sec, 2),
            "on_road_ratio": round(on_road_ratio, 3),
            "vats_s": round(vats_s, 1), "vats_glances": vats_glances, "vats_alert": vats_alert,
            "long_glance": away_sec >= LONG_GLANCE_S,
            "mirror_checks": len(self.mirrors), "mirror_too_long": mirror_too_long,
            "fixation_s": round(fixation_s, 1), "hypnosis": hypnosis,
        }


# ============================================================================
#  FATIGUE from the eyes: closures, blinks, microsleep, PERCLOS, squinting
# ============================================================================
MICROSLEEP_S = 1.0         # TUNE: eyes closed 1-3 s = microsleep
SLEEP_S = 3.0              # TUNE: eyes closed over 3 s = asleep
PERCLOS_WINDOW = 60.0      # PERCLOS over the last 60 s
PERCLOS_DROWSY = 0.15      # TUNE: more than 15% of the time with eyes closed = drowsy
SLOW_BLINK_S = 0.40        # TUNE: blinks longer than this = slow/heavy (a sign of fatigue)
SQUINT_FRAC = 0.72         # TUNE: eyes below 72% of their normal opening = squinting
SQUINT_MIN_S = 0.5         # TUNE: ...sustained this long
OPEN_REF_WINDOW = 60.0     # window used to estimate "how open the eyes normally are"
OPEN_REF_PCT = 90          # percentile used as the normal opening (robust to outliers, unlike max)
WARMUP_S = 20.0            # blink-rate based judgements (hypnosis) need at least this much data


class BlinkTracker:
    """Fatigue metrics from the (confirmed) eye state + eye openness. The single source of truth
    for closures: eyes_closed_sec, since_last_blink, blink counts, microsleep."""

    def __init__(self, closed_threshold=EYE_OPEN_CLOSED):
        self.closed_threshold = closed_threshold
        self.perclos_win = deque()   # (t, dt, closed)
        self.blinks = deque()        # (t_end, duration) from the last 60 s
        self.open_win = deque()      # (t, openness) for the "normally open" reference
        self.closed_since = None
        self.last_blink_end = None
        self.squint_since = None
        self.last_t = None
        self.t0 = None

    def update(self, t, closed, openness):
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.last_t is None else min(0.2, t - self.last_t)
        self.last_t = t

        # PERCLOS: share of the last 60 s with eyes closed
        self.perclos_win.append((t, dt, closed))
        while self.perclos_win and t - self.perclos_win[0][0] > PERCLOS_WINDOW:
            self.perclos_win.popleft()
        total = sum(d for _, d, _ in self.perclos_win) or 1e-6
        perclos = sum(d for _, d, c in self.perclos_win if c) / total

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

        # squinting: eyes partly closed (below the normal-open reference) but not closed, sustained.
        if openness is not None:
            self.open_win.append((t, openness))
            while self.open_win and t - self.open_win[0][0] > OPEN_REF_WINDOW:
                self.open_win.popleft()
            open_ref = float(np.percentile([o for _, o in self.open_win], OPEN_REF_PCT))
            squint_now = (not closed) and self.closed_threshold < openness < SQUINT_FRAC * open_ref
        else:
            squint_now = False
        self.squint_since = (self.squint_since or t) if squint_now else None
        squinting = self.squint_since is not None and (t - self.squint_since) >= SQUINT_MIN_S

        return {
            "closed_sec": closed_sec,
            "since_last_blink": since_blink,
            "blink_ended": blink_ended,
            "perclos": round(perclos, 3),
            "perclos_drowsy": perclos >= PERCLOS_DROWSY,
            "blink_rate": blink_rate,
            "blinks_per_min": round(blinks_per_min, 1),
            "warmed_up": (t - self.t0) >= WARMUP_S,
            "avg_blink_sec": round(avg_blink, 2),
            "slow_blinks": slow_blinks,
            "microsleep": level == "microsleep",
            "sleep": level == "sleep",
            
            "squinting": squinting,
        }
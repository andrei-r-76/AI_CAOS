"""Gaze zones, iris metrics and eye fatigue - all from MediaPipe Face Landmarker points.

The face_landmarker.task model outputs 478 points: the last 10 are the irises (468-477).
  - 468 = right iris center, ring 469-472
  - 473 = left iris center, ring 474-477

What lives here:
  1) ZoneModel: WHERE the driver looks (road, mirrors, dashboard), from head yaw/pitch + iris.
     The iris alone only says where the eyes point INSIDE THE HEAD: with the head turned a bit
     while looking at the road, the eyes compensate and the iris says "left". So at calibration
     the driver looks naturally (head + eyes) at each zone and the nearest zone wins live.
  2) GazeTracker: attention metrics over time (eyes-on-road ratio, long glance, mirror checks,
     fixation / highway hypnosis).
  (All eye opening / closing / blink logic lives in eyes.py.)

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

# --- gaze zones (head + eyes) ---
ROAD_ZONE = "drum"
MIRROR_ZONES = ("oglinda_stanga", "oglinda_dreapta", "oglinda_mijloc")
ZONE_SCALE = {"yaw": 8.0, "pitch": 6.0, "h": 0.06, "v": 0.10}   # TUNE: how the features are weighed
                           #   (degrees for the head, iris ratio for the eyes). Bigger = that feature
                           #   matters less. Raise h/v if the iris is noisy.
# A zone is chosen by DIRECTION, not by the nearest point: real glances are quicker and smaller than
# the deliberate calibration pose (head turned 33 deg at calibration, maybe 15 deg while driving).
ZONE_REACH = 0.45          # TUNE: the gaze must go at least this fraction of the way from the road to the
                           #   calibrated zone (0.45 = less than half of the calibration movement is enough)
ZONE_KEEP = 0.30           # TUNE: once in a zone, stay there until the gaze is back below this fraction
ZONE_COS_MIN = 0.75        # TUNE: the movement must point towards the zone (cosine; 0.75 = within ~40 deg)
ZONE_HEAD_YAW = 20.0       # TUNE: for the generic label: head yaw beyond this = looking sideways
ZONE_HEAD_PITCH = 15.0     # TUNE: for the generic label: head pitch beyond this = looking up/down
MIRROR_CONFIRM_S = 0.15    # TUNE: mirror glances are quick (0.3-0.8 s): a mirror zone is confirmed sooner
ZONE_CONFIRM_S = 0.35      # TUNE: "awareness delay": a new zone must hold this long before the gaze is
                           #   considered moved there (no instant jumps bord -> oglinda -> drum). The
                           #   glance timing is backdated to when it really started, so durations stay exact

# --- attention time windows ---
LONG_GLANCE_S = 3.0        # TUNE: one continuous glance away this long = long distraction
MIRROR_MIN_S = 0.20        # TUNE: shortest glance that can be a mirror check
MIRROR_ZONE_MAX_S = 2.0    # TUNE: a mirror glance up to this long = mirror check; longer = mirror_too_long
MIRROR_DEBOUNCE_S = 0.8    # minimum time between two counted mirror checks
ONROAD_WINDOW = 30.0       # window for the eyes-on-road ratio
FIX_WINDOW = 1.5           # window used to decide whether the gaze is still moving
FIXATION_MOVE = 0.03       # TUNE: gaze spread below this (raw ratio) = gaze is fixed
HYPNOSIS_S = 8.0           # TUNE: fixation this long (+ rare blinks, eyes OPEN) = highway hypnosis.
                           #       8 s is good for a demo; 20-30 s is more realistic on the road
FIX_BLINK_GRACE_S = 0.6    # TUNE: a closure shorter than this (a normal blink) does not reset the fixation;
                           #       a longer one does (that is drowsiness, handled by microsleep, not hypnosis)
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

    def _vec(self, yaw, pitch, h, v, use_iris):
        """Scaled feature vector relative to the road pose (iris parts only when both sides have them)."""
        ry, rp, rh, rv = self.zones[ROAD_ZONE]
        out = [(yaw - ry) / ZONE_SCALE["yaw"], (pitch - rp) / ZONE_SCALE["pitch"]]
        if use_iris:
            out += [(h - rh) / ZONE_SCALE["h"], (v - rv) / ZONE_SCALE["v"]]
        return np.array(out)

    def classify(self, yaw, pitch, gaze=None, prefer=None):
        """Returns (zone_name, reach). reach = how far the gaze went towards that zone (1.0 = like at
        calibration). gaze=None (eyes closed / no iris) -> head only.
        A zone wins when the movement from the road points towards it (cosine >= ZONE_COS_MIN) and covers
        at least ZONE_REACH of the calibrated movement; prefer (= the current zone) only needs ZONE_KEEP."""
        road = self.zones[ROAD_ZONE]
        best, best_cos, best_reach = ROAD_ZONE, -1.0, 0.0
        for name, (zy, zp, zh, zv) in self.zones.items():
            if name == ROAD_ZONE:
                continue
            use_iris = gaze is not None and None not in (zh, zv, road[2], road[3])
            z = self._vec(zy, zp, zh, zv, use_iris)
            u = self._vec(yaw, pitch, *(gaze if use_iris else (0.0, 0.0)), use_iris)
            zz = float(z @ z)
            if zz < 1e-6:
                continue
            reach = float(u @ z) / zz                     # projection: fraction of the way to the zone
            cos = float(u @ z) / (np.linalg.norm(u) * math.sqrt(zz) + 1e-9)
            need = ZONE_KEEP if name == prefer else ZONE_REACH
            if reach >= need and cos >= ZONE_COS_MIN and cos > best_cos:
                best, best_cos, best_reach = name, cos, reach
        if best != ROAD_ZONE:
            return best, best_reach
        # no calibrated zone in that direction: generic direction from the head alone
        if abs(yaw) >= ZONE_HEAD_YAW and abs(yaw) >= abs(pitch):
            return ("stanga" if yaw * self.left_sign > 0 else "dreapta"), 0.0
        if abs(pitch) >= ZONE_HEAD_PITCH:
            return ("jos" if pitch * self.down_sign > 0 else "sus"), 0.0
        return ROAD_ZONE, 0.0

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


class GazeTracker:
    """Attention metrics for the DRIVER (a single person), over time windows."""

    def __init__(self, cal: GazeCalib | None = None):
        self.cal = cal or GazeCalib()
        self.smooth = deque(maxlen=3)   # last 3 iris samples, averaged to reduce jitter
        self.samples = deque()          # (t, dh, dv, on_road, has_gaze)
        self.zone = ROAD_ZONE           # confirmed zone (after the awareness delay)
        self.cand = None                # zone waiting to be confirmed
        self.cand_since = None
        self.fix_closed_since = None    # eyes closed during a fixation (blink grace)
        self.ep = None                  # current episode away from the road (dict) or None
        self.last_mirror_t = -1e9
        self.mirrors = deque()          # times of counted mirror checks
        self.fix_since = None

    def update(self, t, gaze, eyes_open, zone, blinks_per_min=None, warmed_up=False):
        """gaze: iris (h, v) or None; zone: raw zone from ZoneModel.classify for this frame;
        blinks_per_min / warmed_up: from eyes.EyeTracker (used for highway hypnosis)."""
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

        # zone (head + eyes; from the head alone while the eyes are closed) with the awareness delay:
        # a new zone becomes the current one only after it held for ZONE_CONFIRM_S; the switch is
        # backdated to when the new zone first appeared, so glance durations stay exact
        raw = zone or ROAD_ZONE
        te = t                                    # effective time of a switch made on this frame
        if raw == self.zone:
            self.cand = None
        else:
            if raw != self.cand:
                self.cand, self.cand_since = raw, t
            confirm = MIRROR_CONFIRM_S if self.cand in MIRROR_ZONES else ZONE_CONFIRM_S
            if t - self.cand_since >= confirm:
                self.zone, te, self.cand = self.cand, self.cand_since, None
        zone = self.zone
        away = zone != ROAD_ZONE
        on_road = not away

        # --- episode away from the road ---
        if away:
            if self.ep is None:
                self.ep = {"start": te, "n": 0, "mirror_n": 0}
            self.ep["n"] += 1
            if zone in MIRROR_ZONES:
                self.ep["mirror_n"] += 1
        elif self.ep is not None:
            ep, dur = self.ep, te - self.ep["start"]
            mostly_mirror = ep["n"] >= 2 and ep["mirror_n"] >= 0.6 * ep["n"]
            if mostly_mirror and MIRROR_MIN_S <= dur <= MIRROR_ZONE_MAX_S:
                if t - self.last_mirror_t > MIRROR_DEBOUNCE_S:
                    self.mirrors.append(t)
                    self.last_mirror_t = t
            self.ep = None

        # while a return to the road is waiting for confirmation, the glance already ended at cand_since
        ep_end = self.cand_since if (self.cand == ROAD_ZONE and self.cand_since is not None) else t
        away_sec = (ep_end - self.ep["start"]) if self.ep is not None else 0.0
        in_mirror = (self.ep is not None and self.ep["mirror_n"] >= 0.6 * self.ep["n"])
        mirror_too_long = in_mirror and away_sec > MIRROR_ZONE_MAX_S

        self.samples.append((t, dh, dv, on_road, has_gaze))
        while self.samples and t - self.samples[0][0] > ONROAD_WINDOW:
            self.samples.popleft()
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
        if not eyes_open:
            # a normal blink keeps the fixation going; a longer closure is drowsiness, not hypnosis
            self.fix_closed_since = self.fix_closed_since or t
            if t - self.fix_closed_since > FIX_BLINK_GRACE_S:
                self.fix_since = None
        else:
            self.fix_closed_since = None
            if has_gaze and on_road and not moving:
                self.fix_since = self.fix_since or t
            else:
                self.fix_since = None
        fixation_s = (t - self.fix_since) if self.fix_since is not None else 0.0
        # highway hypnosis = staring with the eyes OPEN (right now), on the road, rarely blinking
        hypnosis = (eyes_open and warmed_up and blinks_per_min is not None
                    and fixation_s >= HYPNOSIS_S and blinks_per_min < HYPNOSIS_BLINKS_MIN)

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

        return {
            "direction": "" if zone == ROAD_ZONE else zone,   # '' = on the road
            "zone": zone,
            "on_road": on_road, "away_sec": round(away_sec, 2),
            "on_road_ratio": round(on_road_ratio, 3),
            "long_glance": away_sec >= LONG_GLANCE_S,
            "mirror_checks": len(self.mirrors), "mirror_too_long": mirror_too_long,
            "fixation_s": round(fixation_s, 1), "hypnosis": hypnosis,
        }

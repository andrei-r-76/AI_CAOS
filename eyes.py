"""Eyes - all the eye logic in one place (rebuilt from scratch).

From the MediaPipe Face Landmarker iris points (468-477) + eyelid points, per eye:

  openness  = vertical eyelid gap / iris diameter (scale-free: works near and far)
  r         = the driver's own scale: 0 = closed like at calibration, 1 = open like at calibration
                r = (openness - closed_ref) / (open_ref - closed_ref)

  state of each eye:   open (r > R_HALF)  |  half (R_CLOSED < r <= R_HALF)  |  closed (r <= R_CLOSED)
  closed (both eyes):  both r <= R_CLOSED for CLOSED_MIN_FRAMES frames; stays closed until an eye opens
                       past R_REOPEN (hysteresis: no flicker on the border)
  one eye closed:      one closed and the other open (iris), or one eyeBlink blendshape high and the other
                       low (wink: the blendshape catches it even when the iris gap drops less)
  half closed:         both eyes "half" for PARTIAL_MIN_S (heavy eyelids)
  blink:               both eyes dip below R_BLINK (one frame is enough: a 0.1 s blink is often caught
                       half-closed) or both eyeBlink high, then reopen within BLINK_MAX_S
  microsleep / sleep:  eyes closed for >= MICROSLEEP_S / SLEEP_S
  PERCLOS:             share of the last 60 s with the eyes closed

Looking down lowers the upper eyelids, exactly like tired eyes. So "looking down" is NOT read from the
eyelids: it comes from the iris position relative to the eye CORNERS (lid-independent) and the head pitch.
While looking down, the "open" reference becomes the opening measured at the "bord" calibration pose,
so looking at the phone is not "half closed". Closed eyes stay closed either way (r ~ 0 against both).

TUNING NOTE: every line marked "TUNE:" is a value you can adjust. All times are in seconds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# right eye (on the left of the image), left eye
EYES = [
    {"corners": (33, 133), "lids": (159, 145), "iris": 468, "ring": [469, 470, 471, 472]},
    {"corners": (362, 263), "lids": (386, 374), "iris": 473, "ring": [474, 475, 476, 477]},
]

# --- thresholds on the driver's own scale r (0 = closed, 1 = open, as at calibration) ---
R_CLOSED = 0.25            # TUNE: an eye at or below this is closed
R_REOPEN = 0.35            # TUNE: closed eyes count as open again only above this (hysteresis)
R_HALF = 0.55              # TUNE: at or below this (and above R_CLOSED) the eye is half closed
R_BLINK = 0.50             # TUNE: both eyes below this, even for one frame = a blink in progress
CLOSED_MIN_FRAMES = 2      # TUNE: frames in a row with both eyes closed before "closed" (filters noise)
BLINK_BS = 0.60            # TUNE: both eyeBlink blendshapes above this = blink in progress (not while looking down)
WINK_BS = 0.55             # TUNE: one eyeBlink above this...
WINK_BS_OTHER = 0.30       # TUNE: ...while the other is below this = one eye closed (wink)
PARTIAL_MIN_S = 0.5        # TUNE: both eyes half closed this long = eyes_partially_closed
BLINK_MAX_S = 0.5          # TUNE: a dip up to this long = a blink; longer only if the eyes were really closed
MICROSLEEP_S = 1.0         # TUNE: eyes closed 1-3 s = microsleep
SLEEP_S = 3.0              # TUNE: eyes closed over 3 s = asleep
PERCLOS_WINDOW = 60.0      # PERCLOS over the last 60 s
PERCLOS_DROWSY = 0.15      # TUNE: more than 15% of the time with the eyes closed = drowsy
SLOW_BLINK_S = 0.40        # TUNE: blinks longer than this = slow / heavy (fatigue)
WARMUP_S = 20.0            # blink-rate judgements (hypnosis) need at least this much data
LOOK_DOWN_FRAC = 0.5       # TUNE: iris lowered (vs the eye corners) at least this fraction of the way to the
                           #       calibrated "bord" pose = looking down

# without calibration (estimated live)
REF_WINDOW = 30.0          # the normal opening = 90th percentile of the last 30 s
DEFAULT_CLOSED_FRAC = 0.30 # closed opening ~ 30% of the normal one
DEFAULT_DOWN_FRAC = 0.65   # looking down ~ 65% of the normal one


def eye_openness(xy):
    """Per eye: vertical eyelid gap / iris diameter. Returns (right, left) or None."""
    if len(xy) < 478:
        return None
    out = []
    for e in EYES:
        lt, lb = e["lids"]
        cx, cy = xy[e["iris"]]
        lid = float(np.hypot(xy[lt][0] - xy[lb][0], xy[lt][1] - xy[lb][1]))
        diam = 2.0 * float(np.mean([np.hypot(xy[r][0] - cx, xy[r][1] - cy) for r in e["ring"]]))
        out.append(lid / diam if diam > 1e-3 else 0.0)
    return out[0], out[1]


def iris_drop(xy):
    """How low the iris sits relative to the line between the eye corners, / eye width (both eyes).
    Grows when looking DOWN; does not change when the eyelids droop (tired eyes). None if missing."""
    if len(xy) < 478:
        return None
    vals = []
    for e in EYES:
        (a, b), iris = e["corners"], e["iris"]
        width = float(np.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]))
        if width > 1e-3:
            vals.append((xy[iris][1] - (xy[a][1] + xy[b][1]) / 2) / width)
    return float(np.mean(vals)) if vals else None


@dataclass
class EyeCalib:
    """The driver's eyes, measured at calibration (None = not calibrated -> estimated live)."""
    open_ref: float = None      # opening with the eyes open, looking at the road
    closed_ref: float = None    # opening with the eyes closed
    down_ref: float = None      # opening while looking at the dashboard / phone
    drop_road: float = None     # iris_drop looking at the road
    drop_down: float = None     # iris_drop looking at the dashboard / phone


class EyeTracker:
    def __init__(self, cal: EyeCalib | None = None):
        self.cal = cal or EyeCalib()
        self.ref_win = deque()       # (t, mean openness) for the live reference without calibration
        self.closed = False
        self.closed_frames = 0
        self.closed_since = None
        self.one_eye_since = None
        self.partial_since = None
        self.dip_since = None
        self.dip_closed = False
        self.last_blink_end = None
        self.blinks = deque()        # (t_end, duration), last 60 s
        self.perclos_win = deque()   # (t, dt, closed)
        self.last_t = None
        self.t0 = None

    def _refs(self, t, op, looking_down):
        """(open reference, closed reference) for this frame."""
        c = self.cal
        if c.open_ref and c.closed_ref is not None and c.open_ref - c.closed_ref > 0.05:
            o, cl = c.open_ref, c.closed_ref
            if looking_down:
                o = c.down_ref if (c.down_ref and c.down_ref - cl > 0.05) else cl + DEFAULT_DOWN_FRAC * (o - cl)
            return o, cl
        # not calibrated: the normal opening = 90th percentile of the recent past
        if op is not None:
            self.ref_win.append((t, (op[0] + op[1]) / 2))
        while self.ref_win and t - self.ref_win[0][0] > REF_WINDOW:
            self.ref_win.popleft()
        o = float(np.percentile([v for _, v in self.ref_win], 90)) if self.ref_win else 0.8
        cl = DEFAULT_CLOSED_FRAC * o
        if looking_down:
            o = cl + DEFAULT_DOWN_FRAC * (o - cl)
        return o, cl

    def is_looking_down(self, drop, head_down=False):
        """Looking down from the iris position vs the eye corners (calibrated) or the head pitch."""
        c = self.cal
        if head_down:
            return True
        if drop is None or c.drop_road is None or c.drop_down is None:
            return False
        span = c.drop_down - c.drop_road
        return abs(span) > 0.01 and (drop - c.drop_road) / span >= LOOK_DOWN_FRAC

    def update(self, t, op, blink_bs=None, drop=None, head_down=False, hidden=False, turned=False):
        """op: (right, left) eye_openness or None; blink_bs: (eyeBlinkLeft, eyeBlinkRight) or None;
        drop: iris_drop; head_down: head pitched down like at the "bord" pose;
        hidden: head turned so far that an eye is hidden; turned: head turned enough that a single
        "closed" eye is just the far eye being hidden."""
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.last_t is None else min(0.2, t - self.last_t)
        self.last_t = t
        looking_down = self.is_looking_down(drop, head_down)
        o_ref, c_ref = self._refs(t, op, looking_down)

        r = states = None
        blink_now = one_eye = half_now = False
        if op is not None and not hidden:
            r = [float(np.clip((v - c_ref) / (o_ref - c_ref), -0.5, 1.5)) for v in op]
            states = ["closed" if x <= R_CLOSED else "half" if x <= R_HALF else "open" for x in r]

            # both eyes closed: debounce to enter, hysteresis to leave
            if self.closed:
                self.closed = max(r) < R_REOPEN           # stays closed until an eye opens past R_REOPEN
            else:
                self.closed_frames = self.closed_frames + 1 if max(r) <= R_CLOSED else 0
                self.closed = self.closed_frames >= CLOSED_MIN_FRAMES
            if not self.closed and max(r) > R_CLOSED:
                self.closed_frames = 0

            # one eye closed (wink): from the iris, or from the blendshapes (one high, one low)
            wink_bs = False
            if blink_bs is not None:
                hi, lo = max(blink_bs), min(blink_bs)
                wink_bs = hi >= WINK_BS and lo < WINK_BS_OTHER
            wink_iris = min(r) <= R_CLOSED and max(r) > R_HALF
            one_eye = (wink_iris or wink_bs) and not self.closed and not turned
            if one_eye and wink_bs and not wink_iris:    # mark the eye the blendshape says is closed
                k = int(np.argmin(r))
                states[k] = "closed"

            half_now = (not self.closed and not one_eye and all(s == "half" for s in states))

            # blink in progress: both eyes dip (one frame is enough), or both blendshapes high
            bs_blink = (blink_bs is not None and not looking_down and min(blink_bs) >= BLINK_BS)
            blink_now = not one_eye and (max(r) <= R_BLINK or bs_blink)
        else:
            self.closed = False
            self.closed_frames = 0

        # timers
        self.closed_since = (self.closed_since or t) if self.closed else None
        closed_sec = (t - self.closed_since) if self.closed_since is not None else 0.0
        self.one_eye_since = (self.one_eye_since or t) if one_eye else None
        self.partial_since = (self.partial_since or t) if half_now else None
        partial = self.partial_since is not None and t - self.partial_since >= PARTIAL_MIN_S

        # blinks: short dips, or longer ones if the eyes were really closed (slow blink / microsleep)
        blink_ended = False
        if blink_now or self.closed:
            self.dip_since = self.dip_since if self.dip_since is not None else t
            self.dip_closed = self.dip_closed or self.closed
        elif self.dip_since is not None:
            dur = t - self.dip_since
            if dur <= BLINK_MAX_S or self.dip_closed:
                self.blinks.append((t, dur))
                self.last_blink_end = t
                blink_ended = True
            self.dip_since, self.dip_closed = None, False
        while self.blinks and t - self.blinks[0][0] > 60.0:
            self.blinks.popleft()

        # PERCLOS
        self.perclos_win.append((t, dt, self.closed))
        while self.perclos_win and t - self.perclos_win[0][0] > PERCLOS_WINDOW:
            self.perclos_win.popleft()
        total = sum(d for _, d, _ in self.perclos_win) or 1e-6
        perclos = sum(d for _, d, c in self.perclos_win if c) / total

        durs = [d for _, d in self.blinks]
        elapsed = min(60.0, t - self.t0)
        level = "sleep" if closed_sec >= SLEEP_S else "microsleep" if closed_sec >= MICROSLEEP_S else "awake"
        pct = int(round(np.clip(np.mean(r), 0, 1.2) * 100)) if r is not None else None
        return {
            "r": r,                                   # per eye (right, left), 0 closed .. 1 open
            "states": states,                         # per eye: open | half | closed (None if not visible)
            "pct": pct,                               # eye opening % of normal (both eyes)
            "closed": self.closed,
            "closed_sec": closed_sec,
            "one_eye": one_eye,
            "one_eye_sec": (t - self.one_eye_since) if self.one_eye_since is not None else 0.0,
            "partial": partial,
            "looking_down": looking_down,
            "blink_now": blink_now,
            "blink_ended": blink_ended,
            "since_last_blink": t - (self.last_blink_end if self.last_blink_end is not None else self.t0),
            "blink_rate": len(durs),                  # blinks in the last minute
            "blinks_per_min": round(len(durs) * 60.0 / elapsed, 1) if elapsed > 0 else 0.0,
            "warmed_up": t - self.t0 >= WARMUP_S,
            "slow_blinks": sum(1 for d in durs if d > SLOW_BLINK_S),
            "perclos": round(perclos, 3),
            "perclos_drowsy": perclos >= PERCLOS_DROWSY,
            "microsleep": level == "microsleep",
            "sleep": level == "sleep",
        }

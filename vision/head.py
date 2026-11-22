"""Head pose (yaw / pitch / roll) + head drop, nodding and tilt.

  - head_drop:  the head falls forward fast (nodding off) with the eyes closed, or very deep.
                Looking down at a phone is slower and does not count.
  - head_nods:  drops that come back up quickly, counted over 60 s.
  - head_tilt:  head tilted sideways for a while (slumped / falling asleep).
"""
import math
from collections import deque

import numpy as np

from utils import Held

HEAD_DROP_DEG = 20.0       # TUNE: head goes down at least this much (degrees vs the road pose)...
HEAD_DROP_WINDOW = 0.7     # TUNE: ...within this many seconds (fast = falling, not looking at a phone)
HEAD_DROP_HARD = 35.0      # TUNE: a drop this deep counts even with the eyes open
HEAD_RECOVER_FRAC = 0.4    # the drop ends when the head is back above HEAD_DROP_DEG * this
HEAD_NOD_RECOVER_S = 2.5   # TUNE: back up within this time = a nod
HEAD_NOD_WINDOW = 60.0     # nods are counted over this window
ROLL_MAX = 20.0            # TUNE: tilted sideways more than this (degrees)...
ROLL_MIN_S = 2.0           # TUNE: ...for this long = head_tilt


def angle_diff(a, b):
    """a - b in degrees, wrapped to -180..180."""
    return (a - b + 180.0) % 360.0 - 180.0


def circular_mean(deg_values):
    """Mean of angles in degrees (works across the -180/180 wrap)."""
    if len(deg_values) == 0:
        return 0.0
    rad = np.radians(deg_values)
    return math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


def head_angles(face_res, idx):
    """(yaw, pitch, roll) in degrees from MediaPipe's face transformation matrix, or None."""
    mats = face_res.facial_transformation_matrixes
    if not mats or idx >= len(mats):
        return None
    r = np.array(mats[idx])[:3, :3]
    r = r / np.linalg.norm(r, axis=0)          # rotation only, no scale
    yaw = math.degrees(math.asin(float(np.clip(-r[2, 0], -1.0, 1.0))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class HeadTracker:
    def __init__(self):
        self.hist = deque()          # (t, down_deg, eyes_closed) over HEAD_DROP_WINDOW
        self.drop_since = None
        self.nods = deque()
        self.tilt = Held(ROLL_MIN_S)

    def update(self, t, yaw, pitch, roll, eyes_closed, down_sign=1.0):
        """Angles relative to the calibrated road pose; down_sign comes from the gaze ZoneModel."""
        down = pitch * down_sign                     # positive = head lower than on the road
        self.hist.append((t, down, eyes_closed))
        while self.hist and t - self.hist[0][0] > HEAD_DROP_WINDOW:
            self.hist.popleft()

        if self.drop_since is None:
            drop = down - min(d for _, d, _ in self.hist)
            eyes_were_closed = any(c for _, _, c in self.hist)
            if drop >= HEAD_DROP_DEG and down >= HEAD_DROP_DEG * 0.75 and \
                    (eyes_were_closed or down >= HEAD_DROP_HARD):
                self.drop_since = t
        elif down < HEAD_DROP_DEG * HEAD_RECOVER_FRAC:   # head came back up
            if t - self.drop_since <= HEAD_NOD_RECOVER_S:
                self.nods.append(t)
            self.drop_since = None
        while self.nods and t - self.nods[0] > HEAD_NOD_WINDOW:
            self.nods.popleft()

        return {
            "head_drop": self.drop_since is not None,
            "head_down_sec": round(t - self.drop_since, 1) if self.drop_since is not None else 0.0,
            "head_nods_1min": len(self.nods),
            "head_tilt": self.tilt.update(t, abs(roll) > ROLL_MAX),
        }

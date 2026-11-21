"""Head pose: yaw / pitch / roll from the face transformation matrix + head-drop / nodding / tilt.

  - head_drop:  the head falls forward quickly (typical when nodding off), while the eyes are
                closed or the drop is very deep. Looking down at a phone is slower and does not count.
  - head_nods:  head drops that come back up quickly ("picat din cap"), counted over 60 s.
  - head_tilt:  head tilted sideways (roll) for a while: slumped posture or falling asleep.
"""
import math
from collections import deque

import numpy as np

HEAD_DROP_DEG = 20.0       # TUNE: the head must go down at least this much (degrees, vs the road pose)...
HEAD_DROP_WINDOW = 0.7     # TUNE: ...within this many seconds (fast = falling, not looking at the phone)
HEAD_DROP_HARD = 35.0      # TUNE: a drop this deep counts even with the eyes open
HEAD_RECOVER_FRAC = 0.4    # the drop ends when the head is back above HEAD_DROP_DEG * this
HEAD_NOD_RECOVER_S = 2.5   # TUNE: a drop that comes back within this time counts as a nod
HEAD_NOD_WINDOW = 60.0     # nods are counted over this window
ROLL_MAX = 20.0            # TUNE: head tilted sideways more than this (degrees)...
ROLL_MIN_S = 2.0           # TUNE: ...for this long = head_tilt


def head_angles(face_res, idx):
    """(yaw, pitch, roll) in degrees from the facial transformation matrix (R = Rz(roll) Ry(yaw) Rx(pitch))."""
    mats = face_res.facial_transformation_matrixes
    if not mats or idx >= len(mats):
        return None
    r = np.array(mats[idx])[:3, :3]
    r = r / np.linalg.norm(r, axis=0)          # remove scale, keep rotation only
    yaw = math.degrees(math.asin(float(np.clip(-r[2, 0], -1.0, 1.0))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class HeadTracker:
    def __init__(self):
        self.hist = deque()          # (t, down_deg, eyes_closed) over HEAD_DROP_WINDOW
        self.drop_since = None
        self.nods = deque()
        self.tilt_since = None

    def update(self, t, yaw, pitch, roll, eyes_closed, down_sign=1.0):
        """yaw/pitch/roll relative to the calibrated road pose; down_sign from ZoneModel."""
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

        self.tilt_since = (self.tilt_since or t) if abs(roll) > ROLL_MAX else None

        return {
            "head_drop": self.drop_since is not None,
            "head_down_sec": round(t - self.drop_since, 1) if self.drop_since is not None else 0.0,
            "head_nods_1min": len(self.nods),
            "head_tilt": self.tilt_since is not None and t - self.tilt_since >= ROLL_MIN_S,
        }

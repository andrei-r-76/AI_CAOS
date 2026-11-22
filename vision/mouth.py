"""Mouth: yawning, mouth open and talking.

  - yawn:       jawOpen blendshape above the calibrated threshold for YAWN_MIN_S
  - mouth open: the widest of 3 inner-lip gaps / face height (one side stays visible with the head turned)
  - talking:    that gap keeps going up and down (not during or right after a yawn)
"""
import math
from collections import deque

from utils import Held

YAWN_MIN_S = 1.2           # TUNE: mouth wide open this long = yawn (not speech)
YAWN_RELEASE = 0.7         # TUNE: a yawn ends only when jaw < threshold * this
MOUTH_GAPS = ((13, 14), (82, 87), (312, 317))   # inner lips: center, left, right (upper, lower)
FACE_HEIGHT = (10, 152)    # forehead, chin: the lip gap is a fraction of the face height
MOUTH_OPEN_GAP = 0.03      # TUNE: lip gap above this = mouth open (closes below 70% of it)
TALK_WINDOW_S = 1.0        # TUNE: talking is judged over the last N seconds
TALK_BAND = 0.006          # TUNE: lip moves under +-this are jitter (raise it if "talking" shows when silent)
TALK_SMOOTH = 3            # TUNE: lip gap averaged over N frames first (removes jitter)
TALK_SWINGS = 2            # TUNE: lips turn up/down this many times in the window = talking (yawn = 1 turn)
MOUTH_DELAY_S = 0.5        # TUNE: from "mouth closed", wait this long before showing open / talking


def count_swings(vals, band):
    """How many times the values turn from up to down (or back), ignoring moves under +-band.
    Averaged over TALK_SMOOTH frames first, so landmark jitter does not count."""
    if not vals:
        return 0
    vals = [sum(vals[max(0, i - TALK_SMOOTH + 1):i + 1]) / len(vals[max(0, i - TALK_SMOOTH + 1):i + 1])
            for i in range(len(vals))]
    swings, direction, ext = 0, 0, vals[0]     # direction: 0 = not moving yet, 1 = up, -1 = down
    for v in vals[1:]:
        if direction == 0:
            if abs(v - ext) > band:
                direction, ext = (1 if v > ext else -1), v
        elif direction > 0:
            if v > ext:
                ext = v
            elif ext - v > band:               # was going up, came down more than band
                swings, direction, ext = swings + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext > band:               # was going down, went up more than band
                swings, direction, ext = swings + 1, 1, v
    return swings


class MouthTracker:
    def __init__(self):
        self.yawn_since = None
        self.open = False                    # lip gap above MOUTH_OPEN_GAP
        self.gap_hist = deque()              # (t, lip gap) over the last TALK_WINDOW_S
        self.delay = Held(MOUTH_DELAY_S)     # delay from "mouth closed" to open / talking
        self.last_wide = -1e9                # last wide-open mouth (yawn): no "talking" right after it

    def update(self, t, xy, jaw, jaw_open):
        """xy: 478 face points; jaw: jawOpen blendshape (or None); jaw_open: calibrated yawn threshold."""
        # yawn (jawOpen)
        if jaw is not None:
            wide = jaw > (jaw_open * YAWN_RELEASE if self.yawn_since is not None else jaw_open)
        else:
            wide = False
        self.yawn_since = (self.yawn_since or t) if wide else None
        yawning = wide and t - self.yawn_since >= YAWN_MIN_S

        # mouth open / talking (lips)
        face_h = math.hypot(xy[FACE_HEIGHT[0]][0] - xy[FACE_HEIGHT[1]][0],
                            xy[FACE_HEIGHT[0]][1] - xy[FACE_HEIGHT[1]][1]) or 1.0
        gap = max(math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]) for a, b in MOUTH_GAPS) / face_h
        self.open = gap > (MOUTH_OPEN_GAP * 0.7 if self.open else MOUTH_OPEN_GAP)
        self.gap_hist.append((t, gap))
        while self.gap_hist and t - self.gap_hist[0][0] > TALK_WINDOW_S:
            self.gap_hist.popleft()
        swings = count_swings([v for _, v in self.gap_hist], TALK_BAND)
        if wide:
            self.last_wide = t
        started = self.delay.update(t, self.open or swings >= 1)     # MOUTH_DELAY_S from closed
        talking = (started and swings >= TALK_SWINGS and not wide
                   and t - self.last_wide > 2 * TALK_WINDOW_S)      # a yawn's closing moves the lips too
        return {"yawning": yawning, "wide": wide, "open": self.open and started, "started": started,
                "talking": talking, "jaw": jaw, "gap": gap, "swings": swings}

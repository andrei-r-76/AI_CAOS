"""Where the driver's hands are relative to the face (MediaPipe Hands + Face Landmarker points).

  - hand_near_face: a hand in (or just around) the face box
  - hand_at_ear:    a hand held at the ear -> phone call without hands-free
  - hand_at_eyes:   fingers at the eyes    -> rubbing the eyes (fatigue)
  - hand_at_mouth:  a hand at the mouth    -> eating / smoking / covering a yawn
"""
import math

from gaze import IRIS_CENTERS
from utils import Held

HAND_POINTS = (0, 4, 8, 12, 16, 20)   # wrist + 5 fingertips
FINGERTIPS = (4, 8, 12, 16, 20)
EAR_POINTS = (234, 454)               # face points at the sides of the face, near the ears
MOUTH_POINTS = (13, 14, 61, 291)      # upper lip, lower lip, mouth corners

FACE_MARGIN = 0.20         # TUNE: face box grown by this fraction = "near the face" (objects.py uses it too)
EAR_RADIUS = 0.30          # TUNE: hand point this close to an ear (x face width) = at the ear
EYE_RADIUS = 0.22          # TUNE: fingertip this close to an eye = at the eyes
MOUTH_RADIUS = 0.15        # TUNE: any hand point this close to a mouth point = at the mouth
NEAR_FACE_MIN_S = 0.3      # TUNE: how long each condition must last before it is reported
AT_EAR_MIN_S = 1.0         #       (a call is held, a scratch is not)
AT_EYES_MIN_S = 0.5
AT_MOUTH_MIN_S = 0.5

HAND_MIN_SCORE = 0.80      # TUNE: minimum hand confidence
GHOST_MAX_SIZE = 0.50      # TUNE: a "hand" fully inside the face box and smaller than this x face width is a
                           #       ghost (the model seeing a hand in the face). Real hands stick out of the box.
HAND_CONFIRM_FRAMES = 3    # TUNE: a hand must be seen this many frames in a row


def grow_box(box, margin=FACE_MARGIN):
    """The face box grown by margin x its size on every side ("near the face")."""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return x0 - mx, y0 - my, x1 + mx, y1 + my


def is_ghost(hand, face_box):
    """True if the hand is fully inside the face box and small (a false hand on the face)."""
    if face_box is None:
        return False
    x0, y0, x1, y1 = face_box
    inside = all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in hand)
    hx, hy = [p[0] for p in hand], [p[1] for p in hand]
    return inside and max(max(hx) - min(hx), max(hy) - min(hy)) < GHOST_MAX_SIZE * (x1 - x0)


class HandFilter:
    """Keeps only believable hands: confident, not a ghost, seen a few frames in a row."""

    def __init__(self):
        self.streak = 0

    def update(self, hands_px, scores, face_box):
        good = [h for h, sc in zip(hands_px, scores) if sc >= HAND_MIN_SCORE and not is_ghost(h, face_box)]
        self.streak = self.streak + 1 if good else 0
        return good if self.streak >= HAND_CONFIRM_FRAMES else []


class HandFaceTracker:
    def __init__(self):
        self.held = {"near_face": Held(NEAR_FACE_MIN_S), "at_ear": Held(AT_EAR_MIN_S),
                     "at_eyes": Held(AT_EYES_MIN_S), "at_mouth": Held(AT_MOUTH_MIN_S)}

    def update(self, t, hands_px, face_xy, face_box):
        """hands_px: hands as lists of 21 (x, y); face_xy: 478 face points or None; face_box or None."""
        now = {"near_face": False, "at_ear": False, "at_eyes": False, "at_mouth": False}
        if face_xy is not None and face_box is not None and hands_px:
            fw = face_box[2] - face_box[0]
            box = grow_box(face_box)

            def near(pt, ref, radius):
                return math.hypot(pt[0] - ref[0], pt[1] - ref[1]) <= radius * fw

            for hand in hands_px:
                pts = [hand[i] for i in HAND_POINTS]
                tips = [hand[i] for i in FINGERTIPS]
                if any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in pts):
                    now["near_face"] = True
                if any(near(p, face_xy[e], EAR_RADIUS) for p in pts for e in EAR_POINTS):
                    now["at_ear"] = True
                if any(near(p, face_xy[e], EYE_RADIUS) for p in tips for e in IRIS_CENTERS):
                    now["at_eyes"] = True
                if any(near(p, face_xy[m], MOUTH_RADIUS) for p in hand for m in MOUTH_POINTS):   # all 21 points
                    now["at_mouth"] = True

        if now["at_eyes"]:
            now["at_ear"] = False     # fingers at the eyes = rubbing them, not a phone
        return {"hand_" + k: self.held[k].update(t, active) for k, active in now.items()}

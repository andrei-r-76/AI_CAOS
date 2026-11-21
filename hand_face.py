"""Where the driver's hands are relative to the face (from MediaPipe Hands + Face Landmarker points).

  - hand_near_face: a hand inside (or just around) the face box
  - hand_at_ear:    a hand held at the ear  -> likely a phone call without hands-free
  - hand_at_eyes:   fingers at the eyes     -> rubbing the eyes (a fatigue sign)
  - hand_at_mouth:  a hand at the mouth     -> eating / drinking / smoking / covering a yawn
"""
import math

HAND_POINTS = (0, 4, 8, 12, 16, 20)   # wrist + the 5 fingertips
FINGERTIPS = (4, 8, 12, 16, 20)
EAR_POINTS = (234, 454)               # face-mesh points at the sides of the face, near the ears
EYE_CENTERS = (468, 473)              # iris centers
MOUTH_CENTER = 13                     # upper inner lip

FACE_MARGIN = 0.20         # TUNE: face box expanded by this fraction for "near the face"
EAR_RADIUS = 0.30          # TUNE: hand point within this * face width of an ear point = "at the ear"
EYE_RADIUS = 0.22          # TUNE: fingertip within this * face width of an eye = "at the eyes"
MOUTH_RADIUS = 0.25        # TUNE: hand point within this * face width of the mouth = "at the mouth"
NEAR_FACE_MIN_S = 0.3      # TUNE: each condition must last this long before it is reported
AT_EAR_MIN_S = 1.0         # TUNE: (a phone call is held, not a quick scratch)
AT_EYES_MIN_S = 0.5
AT_MOUTH_MIN_S = 0.5


HAND_MIN_SCORE = 0.80      # TUNE: minimum "this is a hand" score (handedness confidence) to accept a hand
GHOST_MAX_SIZE = 0.70      # TUNE: a "hand" that lies COMPLETELY inside the face box and is smaller than this
                           #       * face width is a ghost (the hand model seeing a hand in the face).
                           #       A real hand at the face sticks out of the face box (wrist, fingers).
HAND_CONFIRM_FRAMES = 3    # TUNE: a hand must be seen this many frames in a row before it is used


def is_ghost(hand, face_xy):
    """True if the detected hand is fully inside the face box and small: a false hand on the face."""
    if face_xy is None:
        return False
    xs = [p[0] for p in face_xy]
    ys = [p[1] for p in face_xy]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    fw = x1 - x0
    inside = all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in hand)
    hx, hy = [p[0] for p in hand], [p[1] for p in hand]
    size = max(max(hx) - min(hx), max(hy) - min(hy))
    return inside and size < GHOST_MAX_SIZE * fw


class HandFilter:
    """Keeps only believable hands: confident, not a ghost inside the face, seen a few frames in a row."""

    def __init__(self):
        self.streak = 0

    def update(self, hands_px, scores, face_xy):
        good = [h for h, sc in zip(hands_px, scores)
                if sc >= HAND_MIN_SCORE and not is_ghost(h, face_xy)]
        self.streak = self.streak + 1 if good else 0
        return good if self.streak >= HAND_CONFIRM_FRAMES else []


class HandFaceTracker:
    def __init__(self):
        self.since = {"near_face": None, "at_ear": None, "at_eyes": None, "at_mouth": None}

    def update(self, t, hands_px, face_xy):
        """hands_px: list of hands, each a list of 21 (x, y) pixel points; face_xy: 478 (x, y) points or None."""
        now = {"near_face": False, "at_ear": False, "at_eyes": False, "at_mouth": False}
        if face_xy is not None and hands_px:
            xs = [p[0] for p in face_xy]
            ys = [p[1] for p in face_xy]
            fw, fh = max(xs) - min(xs), max(ys) - min(ys)
            mx, my = fw * FACE_MARGIN, fh * FACE_MARGIN
            box = (min(xs) - mx, min(ys) - my, max(xs) + mx, max(ys) + my)

            def near(pt, ref, radius):
                return math.hypot(pt[0] - ref[0], pt[1] - ref[1]) <= radius * fw

            for hand in hands_px:
                pts = [hand[i] for i in HAND_POINTS]
                tips = [hand[i] for i in FINGERTIPS]
                if any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in pts):
                    now["near_face"] = True
                if any(near(p, face_xy[e], EAR_RADIUS) for p in pts for e in EAR_POINTS):
                    now["at_ear"] = True
                if any(near(p, face_xy[e], EYE_RADIUS) for p in tips for e in EYE_CENTERS):
                    now["at_eyes"] = True
                if any(near(p, face_xy[MOUTH_CENTER], MOUTH_RADIUS) for p in pts):
                    now["at_mouth"] = True

        if now["at_eyes"]:
            now["at_ear"] = False     # fingers at the eyes = rubbing them, not a phone at the ear
        min_s = {"near_face": NEAR_FACE_MIN_S, "at_ear": AT_EAR_MIN_S,
                 "at_eyes": AT_EYES_MIN_S, "at_mouth": AT_MOUTH_MIN_S}
        out = {}
        for k, active in now.items():
            self.since[k] = (self.since[k] or t) if active else None
            out["hand_" + k] = self.since[k] is not None and t - self.since[k] >= min_s[k]
        return out

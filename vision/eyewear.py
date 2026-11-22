"""Glasses and sunglasses on the driver, measured from the camera image.

  - glasses:    the frame bridge makes a dark band on the nose between the eyes ("bridge" value)
  - sunglasses: the eye area is much darker than the cheeks ("dark" value, vs the driver's normal value)
  - glasses_missing: the driver's profile says glasses, but none are seen
"""
import cv2

from gaze import IRIS_CENTERS
from utils import Held

SUNGLASSES_REL = 0.5       # TUNE: sunglasses = "dark" below this x the driver's calibrated value
SUNGLASSES_RATIO = 0.25    # TUNE: fixed threshold for old calibrations without a "dark" value
SUNGLASSES_MIN_S = 1.0     # TUNE: ...for this long (a blink or a shadow is shorter)
SUNGLASSES_PATCH = 0.06    # size of the measured patches (x face width)
CHEEK_POINTS = (50, 280)   # cheeks, just below the eyes
GLASSES_DIP = 0.25         # TUNE: nose-bridge band this much darker than the skin around it = glasses frame
GLASSES_MIN_S = 1.0        # TUNE: ...for this long
GLASSES_MISSING_S = 1.0    # TUNE: profile says glasses, none seen this long = warn
BRIDGE_POINTS = (168, 6)   # top of the nose bridge (where the frame bridge sits)
INNER_EYE = (133, 362)     # inner eye corners (the strip is centered between them)


def eye_cheek_ratio(frame, xy, face_box):
    """Eye brightness / cheek brightness (small gray patches): ~0.6-0.9 normally, much lower with
    sunglasses. Call BEFORE drawing. None if a patch is outside the image."""
    h, w = frame.shape[:2]
    r = max(3, int(SUNGLASSES_PATCH * (face_box[2] - face_box[0])))

    def patch_mean(i):
        cx, cy = int(xy[i][0]), int(xy[i][1])
        x0, x1, y0, y1 = max(0, cx - r), min(w, cx + r), max(0, cy - r), min(h, cy + r)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return float(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean())

    eyes = [patch_mean(i) for i in IRIS_CENTERS]      # iris centers are still placed behind dark lenses
    cheeks = [patch_mean(i) for i in CHEEK_POINTS]
    if None in eyes or None in cheeks:
        return None
    cheek = (cheeks[0] + cheeks[1]) / 2
    return (eyes[0] + eyes[1]) / 2 / cheek if cheek > 5 else None


def glasses_dip(frame, xy, face_box):
    """Glasses frame: how much darker the darkest row of a thin strip on the nose bridge is than the
    skin above and below it (0 = no band). Call BEFORE drawing. None if outside the image."""
    h, w = frame.shape[:2]
    fw = face_box[2] - face_box[0]
    cx = (xy[INNER_EYE[0]][0] + xy[INNER_EYE[1]][0]) / 2
    y0, y1 = int(xy[BRIDGE_POINTS[0]][1] - 0.06 * fw), int(xy[BRIDGE_POINTS[1]][1] + 0.03 * fw)
    x0, x1 = int(cx - max(2, 0.04 * fw)), int(cx + max(2, 0.04 * fw))
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or y1 - y0 < 6:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    rows = gray.mean(axis=1)                       # brightness of each row, top to bottom
    m = int(rows.argmin())
    if m == 0 or m == len(rows) - 1:               # darkest at an end = no band in the middle
        return 0.0
    skin = min(rows[:m].max(), rows[m:].max())     # the skin above and below the band
    return float(1 - rows[m] / skin) if skin > 5 else None


class EyewearTracker:
    def __init__(self):
        self.sunglasses_held = Held(SUNGLASSES_MIN_S)
        self.glasses_held = Held(GLASSES_MIN_S)
        self.missing_held = Held(GLASSES_MISSING_S)

    def update(self, t, frame, xy, face_box, dark_base):
        """Glasses / sunglasses on the driver's face (call BEFORE drawing on the frame).
        dark_base: the driver's normal "dark" value from calibration (None = use the fixed threshold)."""
        dark = eye_cheek_ratio(frame, xy, face_box)
        dark_thr = dark_base * SUNGLASSES_REL if dark_base else SUNGLASSES_RATIO
        sunglasses = self.sunglasses_held.update(t, dark is not None and dark < dark_thr)
        bridge = glasses_dip(frame, xy, face_box)
        glasses = self.glasses_held.update(t, bridge is not None and bridge > GLASSES_DIP) and not sunglasses
        return {"glasses": glasses, "sunglasses": sunglasses, "dark": dark, "bridge": bridge}

    def missing(self, t, should_wear):
        """True once glasses have been missing for GLASSES_MISSING_S (should_wear = profile says glasses,
        the face is visible and no glasses / sunglasses are seen)."""
        return self.missing_held.update(t, should_wear)

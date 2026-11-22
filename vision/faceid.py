"""Who is the driver: face recognition with OpenCV SFace.

SFace turns an aligned face into 128 numbers (a "face print"). Prints of the same person point the same
way (cosine ~0.5-0.9), different people ~0.0-0.3. Each profile keeps a few prints (learned when the driver
is added); every ID_EVERY_S the current face is compared with all of them.

The model (~37 MB) downloads once, next to this file. ~1 check per second -> no FPS loss.
Face prints are stored only locally, in profiles.json.
"""

import os
from collections import deque

import cv2
import numpy as np

from utils import download

SFACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_recognition_sface_2021dec.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")

MATCH_COS = 0.55        # TUNE: similarity above this = recognized (OpenCV's own threshold: 0.363)
KEEP_COS = 0.30         # TUNE: a recognized driver is kept while his similarity stays above this (glasses off,
                        #       other light); below it (a stranger is ~0.0-0.3) he becomes "unknown"
ID_EVERY_S = 0.5        # TUNE: how often the face is checked (also how often the match bar updates)
ID_VOTES = 3            # TUNE: same answer this many checks in a row before switching (~1.5 s)
FRONTAL_YAW = 25        # only faces turned less than this (degrees): side views match badly
ENROLL_SAMPLES = 5      # face prints saved for a new driver...
ENROLL_EVERY_S = 0.4    # ...taken this far apart
LOST_RESET_S = 2.0      # face gone this long -> the next face is recognized from scratch

ALIGN_POINTS = (468, 473, 1, 61, 291)   # in SFace's order: right eye, left eye, nose tip, mouth R, mouth L


class DriverID:
    def __init__(self):
        self.available, self.status = False, ""
        try:
            download(SFACE_MODEL, SFACE_URL)
            self.rec = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
            self.available, self.status = True, "on"
        except Exception as e:                 # no internet / old OpenCV -> picking by hand still works
            self.status = f"off ({e})"
        print(f"Driver recognition (SFace): {self.status}")
        self.last_t = -1e9
        self.votes = deque(maxlen=ID_VOTES)
        self.enroll_name, self.enroll_feats = None, []
        self.current = None                    # recognized / picked driver, kept while his face stays
        self.manual = False                    # current driver was picked by hand (kept even with a low match)
        self.last_seen = -1e9
        self.last_score = None

    def face_print(self, frame, xy, face_box):
        """128 numbers for the face (unit length). Call BEFORE drawing on the frame."""
        x0, y0, x1, y1 = face_box
        row = [x0, y0, x1 - x0, y1 - y0]
        for i in ALIGN_POINTS:
            row += [xy[i][0], xy[i][1]]
        crop = self.rec.alignCrop(frame, np.array(row + [1.0], np.float32))
        f = self.rec.feature(crop).flatten()
        return f / (np.linalg.norm(f) or 1.0)

    @staticmethod
    def similarity(f, driver):
        """Best similarity between the face print and one driver's saved prints (-1 if none)."""
        return max((float(np.dot(f, p)) for p in driver.get("faces", [])), default=-1.0)

    @staticmethod
    def best_match(f, drivers):
        """(name, similarity) of the closest profile; name is None if nobody is close enough."""
        best, best_cos = None, -1.0
        for name, d in drivers.items():
            for p in d.get("faces", []):
                c = float(np.dot(f, p))
                if c > best_cos:
                    best, best_cos = name, c
        return (best if best_cos >= MATCH_COS else None), best_cos

    def start_enroll(self, name):
        self.enroll_name, self.enroll_feats = name, []
        self.votes.clear()

    def pick(self, name):
        """Driver picked by hand: kept even if his match is low, but if the face is another SAVED driver
        (ID_VOTES checks in a row), recognition switches to him."""
        self.current, self.manual = name, True
        self.votes.clear()

    def update(self, t, frame, xy, face_box, yaw, drivers):
        """Call every frame (it only works every ID_EVERY_S / ENROLL_EVERY_S). Returns:
          ("enrolled", name, prints)  finished learning a new driver's face
          ("driver", name)            same answer ID_VOTES times (None = unknown face)
          None                        nothing new
        """
        if not self.available:
            return None
        if xy is None:                                     # no face
            if t - self.last_seen > LOST_RESET_S:
                self.current, self.manual = None, False
                self.votes.clear()
            return None
        self.last_seen = t
        if abs(yaw) > FRONTAL_YAW:
            return None
        if not self.enroll_name and not any(d.get("faces") for d in drivers.values()):
            return None                                    # nobody learned yet

        if self.enroll_name:                               # learning a new face
            if t - self.last_t < ENROLL_EVERY_S:
                return None
            self.last_t = t
            self.enroll_feats.append([round(float(v), 4) for v in self.face_print(frame, xy, face_box)])
            if len(self.enroll_feats) >= ENROLL_SAMPLES:
                name, feats = self.enroll_name, self.enroll_feats
                self.enroll_name, self.enroll_feats = None, []
                return ("enrolled", name, feats)
            return None

        if t - self.last_t < ID_EVERY_S:
            return None
        self.last_t = t
        f = self.face_print(frame, xy, face_box)
        name, self.last_score = self.best_match(f, drivers)
        if self.current and name is None and (
                self.manual or self.similarity(f, drivers.get(self.current, {})) >= KEEP_COS):
            self.votes.clear()                             # still him: picked by hand, or only a bit lower match
            return None
        self.votes.append(name)
        if len(self.votes) == ID_VOTES and len(set(self.votes)) == 1:
            self.current, self.manual = name, False
            return ("driver", name)
        return None

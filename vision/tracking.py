"""Faces in the frame: which one is the driver, plus small helpers for MediaPipe face results.

DriverPicker keeps the SAME driver even when other faces enter: every face gets a stable ID (nearest face
from the last frames), the driver is the largest face, and if he is briefly out of view we wait for him.
"""
import math

import numpy as np

TRACK_MAX_JUMP = 0.25      # TUNE: max face move between frames (x frame width) to keep its ID
TRACK_GRACE_S = 2.0        # TUNE: driver missing less than this keeps his ID (we wait for him)


def to_pixels(landmarks, w, h):
    """MediaPipe landmarks (0..1) -> [(x, y), ...] in pixels."""
    return [(p.x * w, p.y * h) for p in landmarks]


def box_of(xy):
    """(x0, y0, x1, y1) around the points."""
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def box_area(landmarks):
    """Face size (area of its box, in 0..1 units)."""
    return ((max(p.x for p in landmarks) - min(p.x for p in landmarks)) *
            (max(p.y for p in landmarks) - min(p.y for p in landmarks)))


def blendshapes(face_res, idx):
    """{name: score} of the face's blendshapes ({} if missing)."""
    if face_res.face_blendshapes and idx < len(face_res.face_blendshapes):
        return {c.category_name: c.score for c in face_res.face_blendshapes[idx]}
    return {}


class DriverPicker:
    def __init__(self):
        self.tracks = {}        # face ID -> (center_x, center_y, last_seen)
        self.next_id = 0
        self.driver_id = None   # ID of the driver's face

    def _assign_ids(self, centers, now, w):
        """Stable face IDs: each face gets the ID of the nearest face from the last frames
        (MediaPipe can swap the face order, so the driver could jump to a passenger)."""
        for tid in [k for k, v in self.tracks.items() if now - v[2] > TRACK_GRACE_S]:
            del self.tracks[tid]
        pairs = sorted((math.hypot(c[0] - p[0], c[1] - p[1]), i, tid)
                       for i, c in enumerate(centers) for tid, p in self.tracks.items())
        ids, used = [None] * len(centers), set()
        for d, i, tid in pairs:                     # closest pairs first
            if d <= TRACK_MAX_JUMP * w and ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, c in enumerate(centers):
            if ids[i] is None:                      # new face
                ids[i], self.next_id = self.next_id, self.next_id + 1
            self.tracks[ids[i]] = (c[0], c[1], now)
        return ids

    def pick(self, faces, now, w, h):
        """Index of the driver in faces (MediaPipe face_landmarks), or None."""
        centers = [(float(np.mean([p.x for p in pts])) * w, float(np.mean([p.y for p in pts])) * h)
                   for pts in faces]
        ids = self._assign_ids(centers, now, w)
        if not faces:
            return None
        if self.driver_id in ids:
            return ids.index(self.driver_id)        # same person as before, whatever other faces do
        if self.driver_id is None or self.driver_id not in self.tracks:
            # no driver yet, or gone longer than TRACK_GRACE_S: take the largest face
            driver = max(range(len(faces)), key=lambda k: box_area(faces[k]))
            self.driver_id = ids[driver]
            return driver
        return None                                 # driver briefly out of view: wait, never jump

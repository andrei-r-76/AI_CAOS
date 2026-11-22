"""Phone detection with YOLO (ultralytics), on a background thread.

Optional: without `ultralytics` everything else still works.   Install: uv pip install ultralytics
YOLO runs every OBJ_EVERY_S on the newest frame, so the face FPS stays high. Results vs the driver:
  - phone_detected:  a phone is visible
  - phone_in_hand:   a phone box touches a hand
  - phone_near_face: a phone box touches the (grown) face box -> call / looking at the phone
"""
import os
import threading
import time

from hand_face import grow_box      # same "near the face" for hands and objects

try:
    from ultralytics import YOLO
except Exception:              # not installed, or torch missing
    YOLO = None

OBJ_MODEL = "yolo26n.pt"   # TUNE: YOLO26 nano (fast on CPU); "yolo26s.pt" = more accurate, slower.
                           #       Looked for next to this file, then in the run folder, else downloaded once.
OBJ_CLASSES = {67: "phone"}   # COCO class id
OBJ_CONF = 0.35            # TUNE: minimum confidence
OBJ_IMGSZ = 416            # TUNE: input size (320 faster, 416 finds phones better, 640 small / far phones)
OBJ_EVERY_S = 0.5          # TUNE: run at most this often (lower = faster reaction, fewer FPS)
OBJ_THREADS = 1            # TUNE: CPU threads for YOLO (torch takes all cores by default -> MediaPipe slows)
OBJ_KEEP_S = 1.0           # TUNE: a detection stays valid this long


def _model_path(name):
    """The model next to this file, else in the run folder, else just the name (auto-download)."""
    for folder in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return name


class ObjectDetector:
    def __init__(self, enabled=True):
        self.available = enabled and YOLO is not None
        self.status = "off" if not enabled else ("ultralytics not installed" if YOLO is None else "loading")
        self.lock = threading.Lock()
        self.frame = None
        self.dets, self.dets_t = [], 0.0
        self.running = self.available
        if self.available:
            threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        """Give the newest frame (BEFORE drawing on it). Copies only when YOLO is idle."""
        if self.available:
            with self.lock:
                if self.frame is None:
                    self.frame = frame.copy()

    def _loop(self):
        try:
            import torch
            torch.set_num_threads(OBJ_THREADS)
        except Exception:
            pass
        try:
            path = _model_path(OBJ_MODEL)
            model = YOLO(path)
            self.status = "on"
            print(f"YOLO on ({path})")
        except Exception as e:                      # e.g. no internet for the first download
            self.status = f"error: {e}"
            self.available = False
            print(f"YOLO cannot start: {e}")
            return
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            try:
                res = model.predict(frame, imgsz=OBJ_IMGSZ, conf=OBJ_CONF, classes=list(OBJ_CLASSES),
                                    verbose=False)[0]
                dets = [(OBJ_CLASSES[int(c)], float(p), *map(float, box))
                        for c, p, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                             res.boxes.xyxy.tolist())]
            except Exception as e:
                self.status = f"error: {e}"
                dets = []
            with self.lock:
                self.dets, self.dets_t = dets, time.time()
            time.sleep(max(0.0, OBJ_EVERY_S - (time.time() - t0)))

    def latest(self, now):
        """[(label, conf, x1, y1, x2, y2), ...] in pixels; [] if too old."""
        with self.lock:
            return list(self.dets) if now - self.dets_t <= OBJ_KEEP_S else []

    def stop(self):
        self.running = False


def _overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def relate(dets, hands_px, face_box):
    """Phones vs the driver's face box (or None) and hands (lists of (x, y))."""
    out = {"phone_detected": False, "phone_in_hand": False, "phone_near_face": False}
    near_face = grow_box(face_box) if face_box is not None else None
    for _, _, x1, y1, x2, y2 in dets:
        out["phone_detected"] = True
        if near_face and _overlap((x1, y1, x2, y2), near_face):
            out["phone_near_face"] = True
        for hand in hands_px or []:
            if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in hand):
                out["phone_in_hand"] = True
    return out

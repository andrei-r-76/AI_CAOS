"""Phone / bottle / cup detection with YOLO (ultralytics), on a background thread.

Optional: if `ultralytics` is not installed, everything else keeps working (object_detection = False).
    Install:  uv pip install ultralytics      (the model file downloads automatically the first time)

YOLO runs on its own thread every OBJ_EVERY_S on the newest frame, so the face/eye detection FPS
stays high. Detections are then related to the driver's face and hands:
  - phone_detected:  a phone is visible
  - phone_in_hand:   a phone box touches a hand
  - phone_near_face: a phone box touches the (expanded) face box -> call / looking at the phone
  - drinking:        a bottle or cup at the mouth
"""
import threading
import time

from hand_face import FACE_MARGIN   # one definition of "near the face" for hands AND objects

try:
    from ultralytics import YOLO
except Exception:              # not installed, or torch missing
    YOLO = None

OBJ_MODEL = "yolo26n.pt"   # TUNE: YOLO26 nano (fast on CPU). "yolo26s.pt" = more accurate, slower
OBJ_CLASSES = {67: "phone", 39: "bottle", 41: "cup"}   # COCO class ids
OBJ_CONF = 0.35            # TUNE: minimum confidence for a detection
OBJ_IMGSZ = 320            # TUNE: YOLO input size (320 = faster, 416/640 = better for small or far phones)
OBJ_EVERY_S = 0.5          # TUNE: run YOLO at most this often (seconds). Lower = faster reaction, fewer FPS
OBJ_THREADS = 1            # TUNE: CPU threads for YOLO. By default torch takes ALL cores and the face
                           #       detection FPS drops; 1-2 keeps MediaPipe fast
OBJ_KEEP_S = 1.0           # TUNE: a detection stays valid this long (YOLO runs less often than the camera)
MOUTH_MARGIN = 0.15        # cup/bottle box expanded by this fraction (of the face width) for "at the mouth"


class ObjectDetector:
    def __init__(self, enabled=True):
        self.available = enabled and YOLO is not None
        self.status = "dezactivat" if not enabled else ("ultralytics lipseste" if YOLO is None else "se incarca")
        self.lock = threading.Lock()
        self.frame = None
        self.dets, self.dets_t = [], 0.0
        self.running = self.available
        if self.available:
            threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        """Give the newest camera frame (call BEFORE drawing on it). Cheap: copies only when YOLO is idle."""
        if self.available:
            with self.lock:
                if self.frame is None:
                    self.frame = frame.copy()

    def _loop(self):
        try:
            import torch
            torch.set_num_threads(OBJ_THREADS)    # leave the other cores to MediaPipe
        except Exception:
            pass
        try:
            model = YOLO(OBJ_MODEL)
            self.status = "activ"
        except Exception as e:                      # e.g. no internet for the first download
            self.status = f"eroare: {e}"
            self.available = False
            return
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            try:
                res = model.predict(frame, imgsz=OBJ_IMGSZ, conf=OBJ_CONF,
                                    classes=list(OBJ_CLASSES), verbose=False)[0]
                dets = [(OBJ_CLASSES[int(c)], float(p), *map(float, box))
                        for c, p, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                             res.boxes.xyxy.tolist())]
            except Exception as e:
                self.status = f"eroare: {e}"
                dets = []
            with self.lock:
                self.dets, self.dets_t = dets, time.time()
            time.sleep(max(0.0, OBJ_EVERY_S - (time.time() - t0)))

    def latest(self, now):
        """[(label, conf, x1, y1, x2, y2), ...] in frame pixels; [] if stale."""
        with self.lock:
            return list(self.dets) if now - self.dets_t <= OBJ_KEEP_S else []

    def stop(self):
        self.running = False


def _overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def relate(dets, face_xy, hands_px, face_box):
    """Relate detections to the driver's face (478 points or None), its box (x0, y0, x1, y1, computed once
    per frame in face_basic.py; None = no face) and hands (lists of (x, y))."""
    out = {"phone_detected": False, "phone_in_hand": False, "phone_near_face": False,
           "drinking": False, "objects": sorted({d[0] for d in dets})}
    mouth = None
    fw = 0.0
    if face_xy is not None and face_box is not None:
        x0, y0, x1, y1 = face_box
        fw, fh = x1 - x0, y1 - y0
        face_box = (x0 - fw * FACE_MARGIN, y0 - fh * FACE_MARGIN, x1 + fw * FACE_MARGIN, y1 + fh * FACE_MARGIN)
        mouth = face_xy[13]
    for label, _, x1, y1, x2, y2 in dets:
        box = (x1, y1, x2, y2)
        if label == "phone":
            out["phone_detected"] = True
            if face_box and _overlap(box, face_box):
                out["phone_near_face"] = True
            for hand in hands_px or []:
                if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in hand):
                    out["phone_in_hand"] = True
        elif mouth is not None:                    # bottle / cup at the mouth
            m = fw * MOUTH_MARGIN
            if x1 - m <= mouth[0] <= x2 + m and y1 - m <= mouth[1] <= y2 + m:
                out["drinking"] = True
    return out

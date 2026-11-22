"""Camera input.

  - Camera: capture on its own thread (while the models work on one frame, the next one is read)
  - to_mp_image: frame -> MediaPipe image (downscaled, brightness measured, CLAHE in low light)
  - next_ts: timestamps for MediaPipe VIDEO mode
  - CameraHealth: camera covered / too dark (from the same brightness measurement)
"""
import os
import threading
import time

import cv2
import mediapipe as mp

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30   # TUNE: camera resolution / FPS
CAM_DSHOW = True           # Windows: DirectShow + MJPG (faster camera start)
CAM_FAIL_LIMIT = 30        # failed reads in a row before the camera thread gives up
CAM_REOPEN_TRIES = 5       # attempts to reopen a stopped camera before quitting
DETECT_MAX_WIDTH = 640     # TUNE: wider frames are downscaled before detection

# --- light: CLAHE turns on by itself in low light ---
CLAHE_ON_BELOW = 90        # TUNE: brightness (0-255) below this -> CLAHE on...
CLAHE_OFF_ABOVE = 110      # TUNE: ...off only above this (no flicker)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_clahe_on = False
_brightness = 0.0          # last measured brightness 0-255
_contrast = 0.0            # last measured contrast (std of the gray image)

# --- camera health ---
BLOCKED_MEAN = 18          # TUNE: brightness below this = camera covered
BLOCKED_STD = 6            # TUNE: or almost no contrast = covered / dirty
LOW_LIGHT_MEAN = 60        # TUNE: brightness below this = low light
HEALTH_MIN_S = 1.0         # TUNE: must last this long before it is reported


def open_capture(index):
    cap = None
    if os.name == "nt" and CAM_DSHOW:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    return cap


class Camera:
    """read() returns the newest frame not returned yet."""

    def __init__(self, index=0):
        self.index = index
        self.cap = open_capture(index)
        self.ok, self.frame, self.id, self.last_read = False, None, 0, 0
        self.cond = threading.Condition()
        self.running = self.cap.isOpened()
        self.thread = None
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        fails = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails < CAM_FAIL_LIMIT:          # short glitch: retry
                    time.sleep(0.01)
                    continue
            else:
                fails = 0
            with self.cond:
                self.ok, self.frame = ok, frame
                self.id += 1
                if not ok:
                    self.running = False
                self.cond.notify_all()

    def isOpened(self):
        return self.running

    def read(self, timeout=2.0):
        with self.cond:
            got = self.cond.wait_for(lambda: self.id > self.last_read or not self.running, timeout)
            if not got or self.id == self.last_read:
                return False, None
            self.last_read = self.id
            return self.ok, self.frame

    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()


def reopen_camera(cap):
    """The camera stopped: try to reopen it a few times instead of ending the program."""
    cap.release()
    for attempt in range(1, CAM_REOPEN_TRIES + 1):
        print(f"Camera stopped, retrying ({attempt}/{CAM_REOPEN_TRIES}) ...")
        time.sleep(1.0)
        new = Camera(cap.index)
        if new.isOpened():
            return new
        new.release()
    return None


# ============================================================================
#  FRAMES FOR MEDIAPIPE
# ============================================================================
_last_ts = 0


def next_ts():
    """Strictly increasing timestamp in ms (MediaPipe VIDEO mode needs it)."""
    global _last_ts
    _last_ts = max(int(time.monotonic() * 1000), _last_ts + 1)
    return _last_ts


def measure_light(frame):
    """Brightness + contrast of a tiny gray copy, once per frame (for CLAHE and CameraHealth)."""
    global _brightness, _contrast
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60), interpolation=cv2.INTER_AREA)
    _brightness, _contrast = float(small.mean()), float(small.std())


def enhance(frame):
    """CLAHE on the L channel (LAB), on by itself in low light."""
    global _clahe_on
    if _clahe_on and _brightness > CLAHE_OFF_ABOVE:
        _clahe_on = False
    elif not _clahe_on and _brightness < CLAHE_ON_BELOW:
        _clahe_on = True
    if not _clahe_on:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def to_mp_image(frame):
    """Camera frame -> MediaPipe image (also measures the light)."""
    h, w = frame.shape[:2]
    if w > DETECT_MAX_WIDTH:
        frame = cv2.resize(frame, (DETECT_MAX_WIDTH, int(h * DETECT_MAX_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
    measure_light(frame)
    rgb = cv2.cvtColor(enhance(frame), cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class CameraHealth:
    """Camera covered / dirty / too dark, from measure_light()."""

    def __init__(self):
        self.blocked_since = None
        self.dark_since = None

    def update(self, t):
        blocked_now = _brightness < BLOCKED_MEAN or _contrast < BLOCKED_STD
        dark_now = _brightness < LOW_LIGHT_MEAN and not blocked_now
        self.blocked_since = (self.blocked_since or t) if blocked_now else None
        self.dark_since = (self.dark_since or t) if dark_now else None
        return {
            "camera_blocked": self.blocked_since is not None and t - self.blocked_since >= HEALTH_MIN_S,
            "low_light": self.dark_since is not None and t - self.dark_since >= HEALTH_MIN_S,
        }

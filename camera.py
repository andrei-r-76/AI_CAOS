"""Webcam capture on a separate thread + camera health (covered / too dark).

While the models run on one frame, the camera thread already reads the next one (higher FPS).
"""
import os
import threading
import time

import cv2
import numpy as np

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30   # TUNE: camera resolution / FPS
CAM_DSHOW = True           # Windows: use DirectShow + MJPG (faster camera startup)
CAM_FAIL_LIMIT = 30        # consecutive failed reads before the camera thread gives up
CAM_REOPEN_TRIES = 5       # attempts to reopen a camera that stopped, before quitting

# --- camera health ---
BLOCKED_MEAN = 18          # TUNE: average brightness (0-255) below this = camera covered / black
BLOCKED_STD = 6            # TUNE: or almost no contrast (uniform image) = camera covered / dirty
LOW_LIGHT_MEAN = 60        # TUNE: average brightness below this = low light (turn on CLAHE)
HEALTH_MIN_S = 1.0         # TUNE: a condition must last this long before it is reported


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
    """Reads the camera on a separate thread; read() returns the newest frame not yet returned."""

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
                if fails < CAM_FAIL_LIMIT:          # transient glitch: retry instead of stopping
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
    """The camera stopped: try to reopen it a few times instead of ending the demo."""
    cap.release()
    for attempt in range(1, CAM_REOPEN_TRIES + 1):
        print(f"Camera s-a oprit, reincerc ({attempt}/{CAM_REOPEN_TRIES}) ...")
        time.sleep(1.0)
        new = Camera(cap.index)
        if new.isOpened():
            return new
        new.release()
    return None


class CameraHealth:
    """Camera covered / dirty / too dark, from the brightness and contrast of a tiny gray copy."""

    def __init__(self):
        self.blocked_since = None
        self.dark_since = None

    def update(self, t, frame):
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60),
                           interpolation=cv2.INTER_AREA)
        mean, std = float(np.mean(small)), float(np.std(small))
        blocked_now = mean < BLOCKED_MEAN or std < BLOCKED_STD
        dark_now = mean < LOW_LIGHT_MEAN and not blocked_now
        self.blocked_since = (self.blocked_since or t) if blocked_now else None
        self.dark_since = (self.dark_since or t) if dark_now else None
        return {
            "brightness": round(mean, 1),
            "camera_blocked": self.blocked_since is not None and t - self.blocked_since >= HEALTH_MIN_S,
            "low_light": self.dark_since is not None and t - self.dark_since >= HEALTH_MIN_S,
        }

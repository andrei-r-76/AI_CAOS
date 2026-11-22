"""Small helpers shared by the vision modules."""
import os
import urllib.request

import cv2


class Held:
    """True only after a condition stays true for min_s seconds without a break (filters short blips).

        tilted = Held(2.0).update(t, abs(roll) > 20)   # True once the head stays tilted 2 s
    """

    def __init__(self, min_s):
        self.min_s = min_s
        self.since = None            # when the condition became true (None = false now)

    def update(self, t, active):
        if not active:
            self.since = None
        elif self.since is None:
            self.since = t
        return self.since is not None and t - self.since >= self.min_s


def download(path, url):
    """Download a model file once (to a temp file first, renamed only when complete)."""
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        print("Done.")


def top_bar(frame, title, color, subtitle=None, subtitle_color=(77, 77, 77)):
    """White bar at the top of the image, like the dashboard's top bar: a title (shrunk to fit the width)
    and an optional small line under it (gray by default). Colors are BGR."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 50), (255, 255, 255), -1)
    cv2.line(frame, (0, 50), (w, 50), (219, 213, 209), 1)
    scale = 0.75 if subtitle is None else 0.6
    while scale > 0.35 and cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 20:
        scale -= 0.05
    cv2.putText(frame, title, (10, 32 if subtitle is None else 22), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    if subtitle:
        cv2.putText(frame, subtitle, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, subtitle_color, 1)


def beep():
    """Short beep (Windows), else the terminal bell."""
    try:
        import winsound
        winsound.Beep(1000, 250)
    except Exception:
        print("\a", end="", flush=True)

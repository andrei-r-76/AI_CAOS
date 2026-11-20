"""Quick check that the webcam, OpenCV and MediaPipe work.

    python scripts/webcam_test.py            # webcam + MediaPipe face landmarks
    python scripts/webcam_test.py --cv-only  # webcam + OpenCV only
    python scripts/webcam_test.py --camera 1 # another camera

Press q to quit.
"""
import argparse
import time
import urllib.request
from pathlib import Path

import cv2

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODEL_PATH.exists():
        print(f"Downloading face model to {MODEL_PATH} ...")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cv-only", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    mp, landmarker = (None, None) if args.cv_only else make_landmarker()
    start, prev = time.monotonic(), time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        status = "OpenCV only"
        if landmarker:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(image, ts_ms)
            if result.face_landmarks:
                h, w = frame.shape[:2]
                for lm in result.face_landmarks[0]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
                status = f"face: {len(result.face_landmarks[0])} landmarks"
            else:
                status = "no face"

        now = time.monotonic()
        fps = 1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:.0f} FPS | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("webcam test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

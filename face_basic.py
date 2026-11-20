"""MediaPipe - detectie de fete (minim 2) + maini din webcam.

Instalare:   uv sync
Rulare:      uv run python face_basic.py
Iesire:      tasta q

Foloseste API-ul nou `mediapipe.tasks` (mp.solutions din tutorialele vechi nu mai exista).
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# --- modele (se descarca automat prima data) ---
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# indici ochi (pentru evidentiere)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# scheletul mainii (ce puncte se leaga intre ele)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FACE_COLORS = [(0, 255, 0), (255, 150, 0)]   # culoare diferita pentru fiecare fata


def ensure(path, url):
    if not os.path.exists(path):
        print(f"Descarc {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Gata.")


def main():
    ensure(FACE_MODEL, FACE_URL)
    ensure(HAND_MODEL, HAND_URL)

    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,                     # <-- minim 2 fete
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,                     # <-- pana la 2 maini
    ))

    cap = cv2.VideoCapture(0)            # 0 = prima camera; pune 1 daca ai alta
    if not cap.isOpened():
        raise SystemExit("Nu pot deschide camera.")

    print("Pornit. Apasa q ca sa iesi.")
    t0, frames, fps = time.time(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)     # timestamp crescator (mod VIDEO)

        face_res = face.detect_for_video(mp_image, ts)
        hand_res = hands.detect_for_video(mp_image, ts)

        # --- fete ---
        n_faces = len(face_res.face_landmarks)
        for idx, pts in enumerate(face_res.face_landmarks):
            col = FACE_COLORS[idx % len(FACE_COLORS)]
            xs = [p.x * w for p in pts]
            ys = [p.y * h for p in pts]
            cv2.rectangle(frame, (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))), col, 2)
            for x, y in zip(xs, ys):
                cv2.circle(frame, (int(x), int(y)), 1, col, -1)
            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, (int(xs[i]), int(ys[i])), 2, (0, 255, 255), -1)
            cv2.putText(frame, f"Fata {idx + 1}", (int(min(xs)), int(min(ys)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # --- maini ---
        n_hands = len(hand_res.hand_landmarks)
        for idx, pts in enumerate(hand_res.hand_landmarks):
            xy = [(int(p.x * w), int(p.y * h)) for p in pts]
            for a, b in HAND_CONNECTIONS:          # scheletul
                cv2.line(frame, xy[a], xy[b], (230, 230, 230), 2)
            for x, y in xy:                        # incheieturile
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
            label = "Mana"
            if hand_res.handedness:                # stanga / dreapta
                label = hand_res.handedness[idx][0].category_name
            cv2.putText(frame, label, (xy[0][0] - 10, xy[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # --- info sus ---
        cv2.rectangle(frame, (0, 0), (w, 35), (25, 25, 25), -1)
        cv2.putText(frame, f"Fete: {n_faces}   Maini: {n_hands}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- FPS ---
        frames += 1
        if time.time() - t0 >= 1:
            fps, frames, t0 = frames / (time.time() - t0), 0, time.time()
        cv2.putText(frame, f"{fps:.1f} FPS", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("MediaPipe - fete + maini", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    face.close()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""MediaPipe - fete (2) + maini + ochi (EAR) + gura (MAR) cu praguri CALIBRATE pe fata ta.

In loc de praguri fixe, la pornire sistemul se calibreaza: masoara ochii tai deschisi
si gura ta relaxata timp de cateva secunde, apoi:
  - ochi INCHISI  = sub 60% din EAR-ul tau normal  (merge si cu ochelari / ochi mici)
  - CASCAT        = gura deschisa peste (MAR-ul tau relaxat + o marja) si sustinut in timp

Instalare:  uv pip install mediapipe opencv-python
Rulare:     uv run python face_basic.py
Taste:      q = iesire,  c = recalibrare
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FACE_COLORS = [(0, 255, 0), (255, 150, 0)]

# --- parametrii calibrarii (relativi la fata ta, nu valori absolute) ---
CALIB_SECONDS = 4.0        # cat tine calibrarea
EYE_RATIO = 0.60           # ochi inchisi daca EAR < baseline_deschis * 0.60
MAR_MARGIN = 0.30          # cascat daca MAR > baseline_relaxat + 0.30
YAWN_MIN_S = 0.8           # ...si sustinut atat (ca sa nu confunde cu vorbitul)

# praguri implicite (se inlocuiesc dupa calibrare)
cal = {"active": True, "start": None, "ears": [], "mars": [],
       "ear_closed": 0.20, "mar_yawn": 0.60, "base_ear": None, "base_mar": None}
yawn_since = {}


def ensure(path, url):
    if not os.path.exists(path):
        print(f"Descarc {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Gata.")


def eye_aspect_ratio(xy, idx):
    p1, p2, p3, p4, p5, p6 = (np.array(xy[i]) for i in idx)
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = 2.0 * np.linalg.norm(p1 - p4)
    return vert / horiz if horiz else 0.0


def mouth_aspect_ratio(xy):
    top, bottom = np.array(xy[MOUTH_TOP_BOTTOM[0]]), np.array(xy[MOUTH_TOP_BOTTOM[1]])
    left, right = np.array(xy[MOUTH_CORNERS[0]]), np.array(xy[MOUTH_CORNERS[1]])
    width = np.linalg.norm(left - right)
    return np.linalg.norm(top - bottom) / width if width else 0.0


def start_calibration():
    cal.update(active=True, start=None, ears=[], mars=[])


def finish_calibration():
    """Din esantioanele adunate calculeaza pragurile personale."""
    if len(cal["ears"]) >= 10:
        cal["base_ear"] = float(np.median(cal["ears"]))
        cal["base_mar"] = float(np.median(cal["mars"]))
        cal["ear_closed"] = cal["base_ear"] * EYE_RATIO
        cal["mar_yawn"] = cal["base_mar"] + MAR_MARGIN
        print(f"Calibrat: EAR normal={cal['base_ear']:.3f} -> inchis<{cal['ear_closed']:.3f} | "
              f"MAR relaxat={cal['base_mar']:.3f} -> cascat>{cal['mar_yawn']:.3f}")
    cal["active"] = False


def main():
    ensure(FACE_MODEL, FACE_URL)
    ensure(HAND_MODEL, HAND_URL)

    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=2))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Nu pot deschide camera.")

    print("Pornit. Calibrare la inceput - priveste camera normal. (q=iesire, c=recalibrare)")
    t0, frames, fps = time.time(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)

        face_res = face.detect_for_video(mp_image, ts)
        hand_res = hands.detect_for_video(mp_image, ts)

        # pornim cronometrul calibrarii la prima fata vazuta
        if cal["active"] and cal["start"] is None and face_res.face_landmarks:
            cal["start"] = time.time()

        for idx, pts in enumerate(face_res.face_landmarks):
            col = FACE_COLORS[idx % len(FACE_COLORS)]
            xy = [(p.x * w, p.y * h) for p in pts]
            xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
            x0, y0 = int(min(xs)), int(min(ys))
            cv2.rectangle(frame, (x0, y0), (int(max(xs)), int(max(ys))), col, 2)

            ear = (eye_aspect_ratio(xy, LEFT_EYE) + eye_aspect_ratio(xy, RIGHT_EYE)) / 2
            mar = mouth_aspect_ratio(xy)

            # in timpul calibrarii strangem valorile de la fata principala (idx 0)
            if cal["active"] and idx == 0 and cal["start"] is not None:
                cal["ears"].append(ear)
                cal["mars"].append(mar)

            closed = ear < cal["ear_closed"]
            wide = mar > cal["mar_yawn"]
            if wide:
                yawn_since.setdefault(idx, time.time())
            else:
                yawn_since.pop(idx, None)
            yawning = wide and (time.time() - yawn_since.get(idx, time.time())) >= YAWN_MIN_S

            # desen ochi
            eye_col = (0, 0, 255) if closed else (0, 255, 255)
            for eye in (LEFT_EYE, RIGHT_EYE):
                poly = np.array([[int(xy[i][0]), int(xy[i][1])] for i in eye], np.int32)
                cv2.polylines(frame, [poly], True, eye_col, 1)
            # desen gura
            mouth_col = (0, 0, 255) if yawning else ((0, 165, 255) if wide else (0, 255, 0))
            mt = (int(xy[13][0]), int(xy[13][1])); mb = (int(xy[14][0]), int(xy[14][1]))
            ml = (int(xy[61][0]), int(xy[61][1])); mr = (int(xy[291][0]), int(xy[291][1]))
            cv2.line(frame, mt, mb, mouth_col, 2); cv2.line(frame, ml, mr, mouth_col, 2)

            cv2.putText(frame, f"Fata {idx + 1}", (x0, y0 - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(frame, f"EAR {ear:.2f} - {'INCHISI' if closed else 'deschisi'}",
                        (x0, y0 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_col, 2)
            mouth_txt = "CASCA!" if yawning else ("deschisa" if wide else "inchisa")
            cv2.putText(frame, f"MAR {mar:.2f} - {mouth_txt}",
                        (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_col, 2)

        # maini
        n_hands = len(hand_res.hand_landmarks)
        for idx, pts in enumerate(hand_res.hand_landmarks):
            xyh = [(int(p.x * w), int(p.y * h)) for p in pts]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, xyh[a], xyh[b], (230, 230, 230), 2)
            for x, y in xyh:
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
            label = hand_res.handedness[idx][0].category_name if hand_res.handedness else "Mana"
            cv2.putText(frame, label, (xyh[0][0] - 10, xyh[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # gestionare calibrare (banner + finalizare)
        if cal["active"]:
            prog = 0.0 if cal["start"] is None else min(1.0, (time.time() - cal["start"]) / CALIB_SECONDS)
            cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
            msg = ("Calibrare: priveste camera normal, ochii deschisi, gura relaxata..."
                   if cal["start"] else "Calibrare: apropie-te in cadru...")
            cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, (10, 42), (10 + int(prog * (w - 20)), 60), (0, 220, 120), -1)
            cv2.rectangle(frame, (10, 42), (w - 10, 60), (200, 200, 200), 1)
            if cal["start"] and prog >= 1.0:
                finish_calibration()
        else:
            cv2.rectangle(frame, (0, 0), (w, 35), (25, 25, 25), -1)
            info = f"Praguri PERSONALE: ochi<{cal['ear_closed']:.2f}  cascat>{cal['mar_yawn']:.2f}   Maini: {n_hands}"
            cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        frames += 1
        if time.time() - t0 >= 1:
            fps, frames, t0 = frames / (time.time() - t0), 0, time.time()
        cv2.putText(frame, f"{fps:.1f} FPS", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("MediaPipe - fete + maini + ochi + gura (calibrat)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            start_calibration()

    cap.release(); face.close(); hands.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# test 
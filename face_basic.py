"""MediaPipe - detectie de fete (minim 2) + maini din webcam.

Instalare:   uv pip install mediapipe opencv-python
Rulare:      uv run python face_basic.py
Iesire:      tasta q

Foloseste API-ul nou `mediapipe.tasks` (mp.solutions din tutorialele vechi nu mai exista).
"""
import json          # <-- NOU: pentru a scrie starea in state.json
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from state import new_state          # <-- NOU: starea comuna (state.py, acelasi folder)

# --- modele (se descarca automat prima data) ---
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# indici ochi: [colt_ext, sus1, sus2, colt_int, jos2, jos1] - ordine pentru formula EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_CLOSED = 0.20          # sub acest prag -> ochi inchisi (aproximativ, fara calibrare)

# gura: colturi (orizontal) + buza sus/jos (vertical), pentru MAR
MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
MAR_OPEN = 0.6             # gura larg deschisa (candidat de cascat)
YAWN_MIN_S = 0.8           # trebuie sa stea deschisa atat ca sa fie cascat (nu vorbit)
yawn_since = {}            # per fata: de cand e gura larg deschisa
# scheletul mainii (ce puncte se leaga intre ele)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FACE_COLORS = [(0, 255, 0), (255, 150, 0)]   # culoare diferita pentru fiecare fata
STATE_FILE = "state.json"  # NOU: puntea catre dashboard/llm (starea scrisa ca JSON)


def ensure(path, url):
    if not os.path.exists(path):
        print(f"Descarc {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Gata.")


def eye_aspect_ratio(xy, idx):
    """EAR: inaltimea ochiului / latimea lui. Mic = inchis, mare = deschis."""
    p1, p2, p3, p4, p5, p6 = (np.array(xy[i]) for i in idx)
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = 2.0 * np.linalg.norm(p1 - p4)
    return vert / horiz if horiz else 0.0


def mouth_aspect_ratio(xy):
    """MAR: deschiderea gurii (sus-jos) / latimea ei (colt-colt). Mare = gura deschisa."""
    top = np.array(xy[MOUTH_TOP_BOTTOM[0]])
    bottom = np.array(xy[MOUTH_TOP_BOTTOM[1]])
    left = np.array(xy[MOUTH_CORNERS[0]])
    right = np.array(xy[MOUTH_CORNERS[1]])
    width = np.linalg.norm(left - right)
    return np.linalg.norm(top - bottom) / width if width else 0.0


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
    last_emit = 0.0                      # NOU: cand am scris ultima data starea

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
            xy = [(p.x * w, p.y * h) for p in pts]
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            x0, y0 = int(min(xs)), int(min(ys))
            cv2.rectangle(frame, (x0, y0), (int(max(xs)), int(max(ys))), col, 2)
            for x, y in xy:
                cv2.circle(frame, (int(x), int(y)), 1, col, -1)

            # --- ochi: EAR + stare deschis/inchis ---
            ear = (eye_aspect_ratio(xy, LEFT_EYE) + eye_aspect_ratio(xy, RIGHT_EYE)) / 2
            closed = ear < EAR_CLOSED
            eye_col = (0, 0, 255) if closed else (0, 255, 255)
            for eye in (LEFT_EYE, RIGHT_EYE):
                poly = np.array([[int(xy[i][0]), int(xy[i][1])] for i in eye], np.int32)
                cv2.polylines(frame, [poly], True, eye_col, 1)   # conturul ochiului
                for i in eye:
                    cv2.circle(frame, (int(xy[i][0]), int(xy[i][1])), 2, eye_col, -1)

            # --- gura: MAR + cascat (deschisa + sustinut in timp) ---
            mar = mouth_aspect_ratio(xy)
            wide = mar > MAR_OPEN
            if wide:
                yawn_since.setdefault(idx, time.time())
            else:
                yawn_since.pop(idx, None)
            yawning = wide and (time.time() - yawn_since.get(idx, time.time())) >= YAWN_MIN_S
            mouth_col = (0, 0, 255) if yawning else ((0, 165, 255) if wide else (0, 255, 0))
            # conturul gurii (deschidere verticala + latime)
            mt = (int(xy[MOUTH_TOP_BOTTOM[0]][0]), int(xy[MOUTH_TOP_BOTTOM[0]][1]))
            mb = (int(xy[MOUTH_TOP_BOTTOM[1]][0]), int(xy[MOUTH_TOP_BOTTOM[1]][1]))
            ml = (int(xy[MOUTH_CORNERS[0]][0]), int(xy[MOUTH_CORNERS[0]][1]))
            mr = (int(xy[MOUTH_CORNERS[1]][0]), int(xy[MOUTH_CORNERS[1]][1]))
            cv2.line(frame, mt, mb, mouth_col, 2)
            cv2.line(frame, ml, mr, mouth_col, 2)
            for pt in (mt, mb, ml, mr):
                cv2.circle(frame, pt, 2, mouth_col, -1)

            cv2.putText(frame, f"Fata {idx + 1}", (x0, y0 - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(frame, f"EAR {ear:.2f} - {'INCHISI' if closed else 'deschisi'}",
                        (x0, y0 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_col, 2)
            mouth_txt = "CASCA!" if yawning else ("gura deschisa" if wide else "gura inchisa")
            cv2.putText(frame, f"MAR {mar:.2f} - {mouth_txt}",
                        (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_col, 2)

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

        # ====== NOU: umplem starea comuna (state.py) ======
        # Deocamdata e activa o singura cheie in state.py: face_found. Aici o umplem.
        # Cand decomentezi alte chei in state.py, le umpli tot aici (vezi liniile comentate).
        state = new_state()
        state["face_found"] = n_faces > 0
        # state["eyes_closed_sec"] = ...   # cand adaugi un timer pe `closed`
        # state["perclos"] = ...           # cand adaugi fereastra de 60s
        # state["looking_away_sec"] = ...  # cand adaugi unghiul capului
        # state["hand_near_face"] = ...    # cand verifici mana langa fata

        # puntea catre dashboard/llm: scriem starea ca JSON de ~5 ori/sec
        if time.time() - last_emit > 0.2:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            print(state)                   # vezi starea live in terminal (scoate-l cand ai dashboard)
            last_emit = time.time()
        # ==================================================

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

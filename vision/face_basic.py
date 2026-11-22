"""Driver monitoring from the webcam -> state.json (main program; see README.md for every module).

Run:   uv run python face_basic.py
Keys:  q - quit   f - face drawing on / off   c - cancel a calibration (started with Recalibrate in the panel)

Every frame: camera -> MediaPipe (face + hands) -> pick the driver -> eyes / mouth / head / gaze / glasses
-> hands and phone -> alerts -> state.json -> drawing -> app window.
Only the DRIVER is analysed: the largest face, kept even when other faces enter the frame.
Every "TUNE:" line (here and in the modules) can be adjusted.
"""
import json
import os
import time

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import drivers
import faceid
import gaze
import objects
from alerts import LEVEL_OF, AlertEngine
from calibration import Calibration
from camera import Camera, CameraHealth, next_ts, reopen_camera, to_mp_image
from eyewear import EyewearTracker
from hand_face import HandFaceTracker, HandFilter
from head import HeadTracker
from mouth import MouthTracker
from state import internal_state, public
from tracking import DriverPicker, blendshapes, box_of, to_pixels
from utils import beep, download, top_bar

# --- models (downloaded on the first run) ---
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# --- speed / output ---
HAND_EVERY = 2             # TUNE: hand detection every N frames
STATE_EMIT_S = 0.2         # TUNE: write state.json every N seconds
PRINT_STATE = False        # TUNE: also print the state (noisy, slower)
DRAW_FACE = True           # TUNE: draw face points, box, eye and mouth lines (key f; text always stays)
OBJECT_DETECTION = True    # TUNE: YOLO phone detection (needs ultralytics; off = faster)
HAND_DETECT_CONF = 0.6     # TUNE: lower (0.5) if hands are missed, higher if ghost hands appear
HAND_PRESENCE_CONF = 0.6   # TUNE
HAND_TRACK_CONF = 0.6      # TUNE

# --- drawing ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
IRIS_POINTS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
DRIVER_COLOR = (0, 255, 0)
STATE_FILE = "state.json"
WINDOW = "Driver Recognition"
UI_SHOW = None             # the app window's show(); None = plain OpenCV window


def write_state(state):
    """Write state.json atomically (temp file + swap): readers never see a half-written file.
    If another program locks state.json for a moment, this write is skipped."""
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(public(state), f)
        os.replace(tmp, STATE_FILE)
    except PermissionError:
        pass


# ============================================================================
#  DRAWING
# ============================================================================
def show(frame):
    """Show the frame (app window, else an OpenCV window); returns the key pressed."""
    if UI_SHOW:
        return UI_SHOW(frame)
    cv2.imshow(WINDOW, frame)
    return cv2.waitKey(1) & 0xFF


def put(frame, text, org, col, scale=0.5, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)


def gaze_text(direction):
    """Gaze for the screen ("" = on the road)."""
    return direction or "on road"


def draw_gaze_panel(frame, gm, x, y):
    """Gaze + eyes-on-road ratio, in the top bar (right side)."""
    lines = [
        f"Gaze: {gaze_text(gm['direction'])}",
        f"Eyes on road: {gm['on_road_ratio'] * 100:.0f}%",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (77, 77, 77), 1)


def draw_unknown(frame, box):
    """A face that is not a saved driver (not monitored)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if DRAW_FACE:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (140, 140, 140), 2)
    put(frame, "UNKNOWN FACE - not monitored", (x0, y0 - 10), (170, 170, 170))
    put(frame, "add him in the panel", (x0, y1 + 20), (170, 170, 170))


def draw_driver(frame, xy, face_box, pose, eyes, mouth, wear, bm, gm):
    """The driver's face: points, box, eye and mouth lines (key f) + text above and under the face."""
    x0, y0, x1, y1 = (int(v) for v in face_box)
    yaw, pitch, roll = pose
    op, closed = eyes["openness"], eyes["closed"]
    eye_col = (150, 150, 150) if eyes["hidden"] else ((0, 0, 255) if closed else (0, 255, 255))
    mouth_col = ((0, 0, 255) if mouth["yawning"] else (0, 165, 255) if mouth["wide"] else
                 (255, 255, 0) if (mouth["talking"] or mouth["open"]) else (0, 255, 0))
    if DRAW_FACE:
        cv2.rectangle(frame, (x0, y0), (x1, y1), DRIVER_COLOR, 2)
        for x, y in xy:
            cv2.circle(frame, (int(x), int(y)), 1, DRIVER_COLOR, -1)
        for eye in (LEFT_EYE, RIGHT_EYE):
            poly = np.array([[int(xy[i][0]), int(xy[i][1])] for i in eye], np.int32)
            cv2.polylines(frame, [poly], True, eye_col, 1)
        for i in IRIS_POINTS:
            cv2.circle(frame, (int(xy[i][0]), int(xy[i][1])), 1, (255, 255, 0), -1)
        for a, b in (MOUTH_TOP_BOTTOM, MOUTH_CORNERS):
            cv2.line(frame, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), mouth_col, 2)

    # above the face: head, eyes, blink / closed time, gaze
    eye_state = ("eyes not detected (sunglasses)" if wear["sunglasses"] else
                 "eyes covered" if eyes["covered"] else
                 "eyes not detected" if (eyes["hidden"] or op is None) else
                 "CLOSED" if closed else "open")
    if wear["glasses"]:
        eye_state += " (glasses)"
    eye_val = f"{(op[0] + op[1]) / 2:.2f}" if op is not None else "-"      # both eyes, average
    if wear["dark"] is not None:
        eye_val += f"  dark {wear['dark']:.2f}"            # sunglasses check
    if wear["bridge"] is not None:
        eye_val += f"  bridge {wear['bridge']:.2f}"        # glasses check
    closure_txt = (f"EYES CLOSED {bm['closed_sec']:.1f}s" if bm["closed_sec"] > 0 else
                   f"Last blink {bm['since_last_blink']:.0f}s ago")
    put(frame, f"Driver head  yaw {yaw:+.0f} pitch {pitch:+.0f} roll {roll:+.0f}", (x0, y0 - 68), (255, 255, 255))
    put(frame, f"Eyes {eye_val} - {eye_state}", (x0, y0 - 48), eye_col)
    put(frame, closure_txt, (x0, y0 - 28), eye_col)
    put(frame, f"Gaze: {gaze_text(gm['direction'])}", (x0, y0 - 8), (255, 255, 0))

    # under the mouth
    mouth_txt = ("YAWNING!" if mouth["yawning"] else "talking" if mouth["talking"] else
                 "mouth open" if (mouth["open"] or (mouth["wide"] and mouth["started"])) else "mouth closed")
    mouth_bottom = int(max(xy[MOUTH_TOP_BOTTOM[1]][1], xy[MOUTH_CORNERS[0]][1], xy[MOUTH_CORNERS[1]][1]))
    put(frame, f"Mouth jaw {mouth['jaw'] or 0:.2f} gap {mouth['gap']:.3f} swings {mouth['swings']} - {mouth_txt}",
        (x0, mouth_bottom + 28), mouth_col)


def draw_hands(frame, hands_px):
    for hand in hands_px:
        ipts = [(int(x), int(y)) for x, y in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, ipts[a], ipts[b], (230, 230, 230), 2)
        for p in ipts:
            cv2.circle(frame, p, 3, (255, 0, 255), -1)


def draw_phones(frame, dets):
    for label, conf, x1, y1, x2, y2 in dets:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        put(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 6), (0, 0, 255), 0.5)


def draw_top_bar(frame, gm):
    """White top bar: title + gaze on the right."""
    top_bar(frame, "DRIVER CAMERA", (0, 0, 0))
    if gm:
        draw_gaze_panel(frame, gm, frame.shape[1] - 206, 20)


# ============================================================================
#  MAIN LOOP
# ============================================================================
def main():
    global DRAW_FACE, UI_SHOW
    download(FACE_MODEL, FACE_URL)
    download(HAND_MODEL, HAND_URL)
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=HAND_DETECT_CONF,
        min_hand_presence_confidence=HAND_PRESENCE_CONF,
        min_tracking_confidence=HAND_TRACK_CONF,
    ))

    cap = Camera(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open the camera.")

    panel = drivers.DriverPanel(WINDOW)       # one window: camera + drivers panel
    UI_SHOW = panel.show
    idf = faceid.DriverID()                   # who the driver is (switches the profile by itself)
    calib = Calibration()
    calib.path = drivers.calib_file(panel.active)
    print(f"Driver: {panel.active or '- (none, add one in the Drivers window)'}  calibration: {calib.path}")
    if not calib.load():
        calib.run(cap, face, show)

    detector = objects.ObjectDetector(OBJECT_DETECTION)
    print(f"Object detection (YOLO): {detector.status}")

    def new_trackers():
        """Per-driver trackers, started again after a calibration or a driver change."""
        return (gaze.EyeTracker(), gaze.BlinkTracker(), gaze.GazeTracker(), HeadTracker(), MouthTracker())

    eye_tracker, blink_tracker, gaze_tracker, head_tracker, mouth_tracker = new_trackers()
    picker, eyewear = DriverPicker(), EyewearTracker()
    hand_tracker, hand_filter, health, alerts = HandFaceTracker(), HandFilter(), CameraHealth(), AlertEngine()

    print("Started. q - quit, f - face drawing on / off.")
    t_start = time.time()
    t0, frames, fps = t_start, 0, 0.0
    last_emit = 0.0
    frame_no, hand_res = 0, None
    last_face_seen = t_start      # for face_lost_sec
    last_yaw = 0.0                # driver's head yaw in the last frame (recognition skips side views)
    hand_at_eyes = False          # a hand was on the eyes in the last frame (eyes not checked)
    was_glasses_missing = False

    while True:
        ok, frame = cap.read()
        if not ok:
            if cap.isOpened():
                continue                          # just a slow frame: wait for the next one
            cap = reopen_camera(cap)              # camera stopped: try to bring it back
            if cap is None:
                break
            continue
        h, w = frame.shape[:2]
        now = time.time()
        detector.submit(frame)

        # ---------- detection ----------
        mp_image = to_mp_image(frame)             # also measures brightness / contrast
        cam = health.update(now)
        ts = next_ts()
        face_res = face.detect_for_video(mp_image, ts)
        if hand_res is None or frame_no % HAND_EVERY == 0:
            hand_res = hands.detect_for_video(mp_image, ts)
        frame_no += 1
        hands_px = [to_pixels(pts, w, h) for pts in hand_res.hand_landmarks]
        faces = face_res.face_landmarks
        driver = picker.pick(faces, now, w, h)   # index of the driver's face, or None

        state = internal_state()
        state["ts"] = round(now, 2)
        state["calibrated"] = calib.calibrated
        state.update(cam)
        gm = None
        xy = face_box = id_res = None
        wear = {"glasses": False, "sunglasses": False}

        # ---------- unknown face (not a saved driver) -> not monitored ----------
        unknown_face = driver is not None and panel.how == "unknown face"
        if unknown_face:
            uxy = to_pixels(faces[driver], w, h)
            ubox = box_of(uxy)
            # keep checking: if he is a saved driver after all, monitoring starts by itself
            id_res = idf.update(now, frame, uxy, ubox, calib.head_pose(face_res, driver)[0],
                                panel.profiles["drivers"])
            draw_unknown(frame, ubox)
            driver = None                         # below: treated as "no driver"
        state["face_found"] = driver is not None

        # ---------- driver: full analysis ----------
        if driver is not None:
            last_face_seen = now
            xy = to_pixels(faces[driver], w, h)
            face_box = box_of(xy)
            wear = eyewear.update(now, frame, xy, face_box, calib.dark_base)    # image checks BEFORE drawing
            id_res = idf.update(now, frame, xy, face_box, last_yaw, panel.profiles["drivers"])
            bs = blendshapes(face_res, driver)
            pose = yaw, pitch, roll = calib.head_pose(face_res, driver)
            last_yaw = yaw

            covered = wear["sunglasses"] or cam["camera_blocked"] or hand_at_eyes
            eyes = eye_tracker.update(xy, yaw, bs, covered, fps, calib.eye_open_closed)
            g = gaze.gaze_ratios(xy)              # gaze: head + eyes (head only while the eyes are closed)
            zone, _ = calib.zone_model.classify(yaw, pitch, g if (g and not eyes["closed"]) else None)
            mouth = mouth_tracker.update(now, xy, bs.get("jawOpen") if bs else None, calib.jaw_open)
            bm = blink_tracker.update(now, eyes["closed"], eyes["checked"])
            gm = gaze_tracker.update(now, zone)
            hm = head_tracker.update(now, yaw, pitch, roll, eyes["closed"], calib.zone_model.down_sign)

            state.update({
                "eyes_closed_sec": round(bm["closed_sec"], 1),
                "eyes_open_sec": bm["eyes_open_sec"],
                "blink_rate": bm["blink_rate"],
                "blink_rate_valid": bm["blink_rate_valid"],
                "slow_blinks_1min": bm["slow_blinks"],
                "microsleep": bm["microsleep"],
                "sleep": bm["sleep"],
                "yawning": mouth["yawning"],
                "talking": mouth["talking"],
                "head_yaw": round(yaw, 1), "head_pitch": round(pitch, 1), "head_roll": round(roll, 1),
                **hm,
                "gaze": gm["direction"] or "on_road",
                "looking_away_sec": gm["away_sec"],
                "on_road_ratio": gm["on_road_ratio"],
                "long_glance": gm["long_glance"],
            })
            draw_driver(frame, xy, face_box, pose, eyes, mouth, wear, bm, gm)
        else:
            eye_tracker.lost()                    # no driver: closed eyes / blinks are not timed
            blink_tracker.drop()
            if not unknown_face:
                idf.update(now, frame, None, None, 0.0, {})   # face gone: the next face is recognized from scratch

        # ---------- driver name, glasses ----------
        state["face_lost_sec"] = 0.0 if driver is not None else round(now - last_face_seen, 1)
        # the name only once recognized / picked / just added; without recognition: the panel's driver
        known = panel.how in ("recognized", "picked by hand", "new driver") or not idf.available
        state["driver"] = (panel.active if known else None) or "unknown"
        state["face_learned"] = known and panel.has_face(panel.active)
        state["glasses"], state["sunglasses"] = wear["glasses"], wear["sunglasses"]
        # profile says glasses, face visible, no glasses (sunglasses count as glasses)
        state["glasses_missing"] = eyewear.missing(
            now, driver is not None and panel.wears_glasses and not (wear["glasses"] or wear["sunglasses"]))
        if state["glasses_missing"] and not was_glasses_missing:
            beep()                                # each time the warning (re)appears
        was_glasses_missing = state["glasses_missing"]

        # ---------- hands and phone (relative to the driver's face) ----------
        hand_scores = [hd[0].score if hd else 0.0 for hd in (hand_res.handedness or [[]] * len(hands_px))]
        hands_px = hand_filter.update(hands_px, hand_scores, face_box)   # drop ghost / unsure hands
        state.update(hand_tracker.update(now, hands_px, xy, face_box))
        hand_at_eyes = state["hand_at_eyes"]
        draw_hands(frame, hands_px)
        dets = detector.latest(now)
        state.update(objects.relate(dets, hands_px, face_box))
        draw_phones(frame, dets)

        # ---------- alerts, state.json ----------
        alerts.update(state, now)
        if now - last_emit > STATE_EMIT_S:
            write_state(state)
            if PRINT_STATE:
                print(state)
            last_emit = now

        # ---------- screen: top bar, FPS, warnings under the image ----------
        draw_top_bar(frame, gm)
        panel.set_alerts([(code, LEVEL_OF[code]) for code in state["active_alerts"]])
        frames += 1
        if now - t0 >= 1:
            fps, frames, t0 = frames / (now - t0), 0, now
        put(frame, f"{fps:.1f} FPS", (10, h - 15), (200, 200, 200), 0.6, 1)

        key = show(frame)
        if key == ord("q"):
            break
        if key == ord("f"):
            DRAW_FACE = not DRAW_FACE

        # ---------- panel events: picked driver -> his calibration; new driver -> calibrate + learn face ----------
        reset = False
        panel.tick(wear["glasses"] or wear["sunglasses"])
        for ev, name in panel.pop_events():
            if ev == "enroll":
                idf.start_enroll(name)
                continue
            calib.path = drivers.calib_file(name)
            print(f"Driver: {name or '- (none)'}  calibration: {calib.path}")
            if ev in ("add", "calibrate") or not calib.load():
                calib.run(cap, face, show)
            if ev == "select":
                idf.pick(name)                    # recognition still checks him
            if ev == "add" or (ev == "select" and not panel.has_face(name)):
                idf.start_enroll(name)            # learn his face
            reset = True

        # ---------- automatic recognition ----------
        if id_res and id_res[0] == "enrolled":
            panel.save_faces(id_res[1], id_res[2])
            print(f"Face learned: {id_res[1]}")
        elif id_res and id_res[0] == "driver" and panel.set_recognized(id_res[1]):
            calib.path = drivers.calib_file(panel.active)
            print(f"Recognized driver: {panel.active}  calibration: {calib.path}")
            if not calib.load():
                print("  he has no calibration yet -> press Recalibrate in the panel")
            reset = True
        panel.set_status(f"Learning face {len(idf.enroll_feats)}/{faceid.ENROLL_SAMPLES} - look at the camera"
                         if idf.enroll_name else
                         f"Recognition {idf.status}" + (f"  ·  match {idf.last_score:.2f}"
                                                        if idf.last_score is not None else ""))
        panel.set_match(idf.last_score, faceid.MATCH_COS)
        if reset:
            eye_tracker, blink_tracker, gaze_tracker, head_tracker, mouth_tracker = new_trackers()

    panel.close()
    detector.stop()
    cap.release()
    face.close()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

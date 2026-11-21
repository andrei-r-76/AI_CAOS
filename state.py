"""The driver state shared by vision -> llm -> dashboard.

Vision fills it every frame, llm.py reads it, the dashboard shows it (it's a plain dict, so it is JSON-ready).
Adding a key: OK, tell the team. Renaming or removing a key: ask everyone first.
"""


def new_state():
    """A driver state with every key set to its 'nothing detected' value."""
    return {
        "face_found": False,          # is a face visible

        # --- eyes / drowsiness (from iris, per eye) ---
        "eyes_closed_sec": 0.0,       # seconds BOTH eyes closed continuously (microsleep/sleep)
        "one_eye_closed": False,      # only one eye closed (likely blink/wink, not drowsiness)
        "yawning": False,             # sustained wide-open mouth = yawn

        # --- head / gaze attention (from iris + head pose) ---
        "looking_away_sec": 0.0,      # seconds not looking at the road (head or gaze), continuous
        "gaze": "on_road",            # on_road | stanga | dreapta | sus | jos | cap intors
        "on_road_ratio": 1.0,         # 0..1, share of the last 30 s looking at the road
        "vats_s": 0.0,                # seconds of short glances away in the last 30 s
        "vats_alert": False,          # too much visual time-sharing (distraction)
        "mirror_checks": 0,           # mirror glances in the last 60 s (positive attention)
        "fixation_s": 0.0,            # seconds gaze held fixed on one spot (tunnel vision)
        "hypnosis": False,            # highway hypnosis: gaze fixed + rare blinking

        # --- not produced yet ---
#        "perclos": 0.0,              # 0..1, share of last 60 s with eyes closed (needs a 60 s window)
#        "hand_near_face": False,     # phone-use proxy (needs a hand-near-face check)
    }

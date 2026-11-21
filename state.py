"""The driver state shared by vision -> llm -> dashboard.

state.json contains ONLY what is useful for the LLM (R2) and the dashboard (R4): new_state() below.
Vision also keeps a few internal values (internal_state()) that feed the alert rules and scores but
are not written out.
Adding a key: OK, tell the team. Renaming or removing a key: ask everyone first.

For the LLM: react to `alert_level` / `alert_reason` CHANGES, not to every update. The other keys are
the details for the message (how long, where the driver looks, how tired, ...).
state.json is written with a plain write: if a read fails (empty / half-written file), read again.
"""


def new_state():
    """The public state (state.json), every key at its 'nothing detected' value."""
    return {
        # --- system ---
        "ts": 0.0,                    # unix time of this state (if it stops changing, vision is not running)
        "calibrated": False,          # a driver calibration is loaded
        "camera_blocked": False,      # camera covered / black image
        "low_light": False,           # image too dark

        # --- alerts: what the LLM reacts to ---
        "alert_level": "none",        # none | info | warning | critical
        "alert_reason": "",           # most important active alert code (see alerts.RULES), "" if none
        "active_alerts": [],          # all active alert codes, most important first
        "events": [],                 # last alerts that started: {ts, time, type, level}
        "fatigue_level": "ok",        # ok | tired | drowsy

        # --- people ---
        "face_found": False,          # the driver's face is visible
        "face_lost_sec": 0.0,         # seconds the driver's face has been missing
        "passenger_count": 0,         # other faces in view

        # --- eyes / drowsiness ---
        "eyes_closed_sec": 0.0,       # seconds both eyes closed continuously
        "eye_openness_pct": 100,      # eye opening as % of the driver's normal (calibrated) opening
        "eyes_partially_closed": False,   # opening <= 50% of normal, sustained (heavy eyelids)
        "microsleep": False,          # eyes closed 1-3 s
        "sleep": False,               # eyes closed over 3 s
        "microsleep_count_10min": 0,  # microsleep episodes in the last 10 minutes
        "perclos": 0.0,               # 0..1, share of the last 60 s with the eyes closed (0 is normal when awake)
        "blink_rate": 0,              # blinks in the last minute

        # --- head ---
        "head_yaw": 0.0,  # degrees, left/right relative to the calibrated road pose
        "head_pitch": 0.0,  # degrees, up/down relative to the calibrated road pose
        "head_roll": 0.0,  # degrees, sideways tilt

        # --- mouth ---
        "yawning": False,             # yawning right now
        "yawn_count_5min": 0,         # yawns in the last 5 minutes

        # --- gaze / attention ---
        "gaze": "on_road",            # on_road | oglinda_stanga | oglinda_dreapta | oglinda_mijloc | bord |
                                      #   stanga | dreapta | sus | jos
        "looking_away_sec": 0.0,      # seconds not looking at the road, continuous
        "long_glance": False,         # one look away >= 3 s
        "on_road_ratio": 1.0,         # 0..1, share of the last 30 s looking at the road
        "mirror_checks": 0,           # mirror glances in the last 60 s (good driving)
        "mirror_too_long": False,     # looking at a mirror > 2 s
        "hypnosis": False,            # highway hypnosis: eyes open, staring, rarely blinking

        # --- head ---
        "head_drop": False,           # head fell forward quickly (nodding off)
        "head_nods_1min": 0,          # head drops that came back up, last minute
        "head_tilt": False,           # head tilted sideways for a while

        # --- hands / objects ---
        "hand_near_face": False,      # a hand at the face
        "hand_at_ear": False,         # a hand held at the ear (phone call)
        "hand_at_eyes": False,        # rubbing the eyes (fatigue)
        "phone_use": False,           # phone at the ear / in the hand / at the face (hand or YOLO)
        "phone_detected": False,      # a phone is visible (YOLO)
        "drinking": False,            # bottle / cup at the mouth (YOLO)
    }


def internal_state():
    """new_state() + internal values used only by the alert rules / scores (not written to state.json)."""
    s = new_state()
    s.update({
        "one_eye_closed": False,      # wink / one eye closed (not drowsiness)
        "perclos_drowsy": False,      # PERCLOS above the drowsy threshold
        "slow_blinks": 0,             # slow/heavy blinks in the last minute
        "phone_in_hand": False,       # YOLO phone box touching a hand
        "phone_near_face": False,     # YOLO phone box at the face
        "hand_at_mouth": False,       # hand at the mouth
    })
    return s


def public(s):
    """The part of an internal state that goes to state.json."""
    return {k: s[k] for k in new_state()}

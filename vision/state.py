"""The driver state shared by vision -> LLM -> dashboard (written to state.json several times a second).

new_state() is the JSON schema: every key with its "nothing detected" value.
internal_state() adds a few values used only by the alert rules (not written to state.json).

For the LLM: react to CHANGES of alert_level / alert_reason (not to every update); the other keys are the
details for the message. If reading state.json fails (half-written file), read it again.
"""


def new_state():
    """The public state (state.json)."""
    return {
        # --- system ---
        "ts": 0.0,                    # unix time (if it stops changing, vision is not running)
        "camera_blocked": False,      # camera covered / black image
        "low_light": False,           # image too dark

        # --- alerts: what the LLM reacts to ---
        "alert_level": "none",        # none | info | warning | critical
        "alert_reason": "",           # most important active alert code (alerts.RULES), "" if none
        "active_alerts": [],          # all active alert codes, most important first
        "events": [],                 # last alerts that started: {ts, time, type, level}

        # --- driver ---
        "driver": "unknown",          # driver's name (as added in the panel), "unknown" until recognized
        "face_learned": False,        # the driver's face is learned (he can be recognized automatically)
        "calibrated": False,          # the driver's calibration succeeded / is loaded
        "face_found": False,          # the driver's face is visible
        "face_lost_sec": 0.0,         # seconds the face has been missing

        # --- eyes ---
        "eyes_closed_sec": 0.0,       # seconds both eyes have been closed
        "eyes_open_sec": 0.0,         # seconds the eyes have been open since the last blink
        "blink_rate": 0,              # blinks in the last minute
        "slow_blinks_1min": 0,        # slow / heavy blinks in the last minute
        "microsleep": False,          # eyes closed 1-3 s
        "sleep": False,               # eyes closed over 3 s
        "microsleep_count_10min": 0,  # microsleeps in the last 10 min

        # --- glasses ---
        "glasses": False,             # clear glasses on
        "sunglasses": False,          # sunglasses on (closed eyes cannot be checked)
        "glasses_missing": False,     # the profile says glasses, none seen

        # --- head ---
        "head_yaw": 0.0,              # degrees left / right vs the calibrated road pose
        "head_pitch": 0.0,            # degrees up / down vs the road pose
        "head_roll": 0.0,             # degrees of sideways tilt
        "head_drop": False,           # head fell forward fast (nodding off)
        "head_down_sec": 0.0,         # seconds the head has stayed down after a drop
        "head_nods_1min": 0,          # drops that came back up, last minute
        "head_tilt": False,           # head tilted sideways for a while

        # --- mouth ---
        "yawning": False,             # yawning now
        "yawn_count_5min": 0,         # yawns in the last 5 min

        # --- gaze ---
        "gaze": "on_road",            # on_road | left | right | up | down
        "looking_away_sec": 0.0,      # seconds looking away from the road
        "long_glance": False,         # one look away >= 3 s
        "on_road_ratio": 1.0,         # 0..1, share of the last 30 s looking at the road
        "distraction": False,         # talking while looking away from the road

        # --- hands / phone ---
        "hand_near_face": False,      # a hand at the face
        "hand_at_ear": False,         # a hand held at the ear (phone call)
        "hand_at_eyes": False,        # rubbing the eyes now
        "eye_rub_count_5min": 0,      # eye rubs in the last 5 min
        "phone_detected": False,      # a phone is visible (YOLO)
        "phone_use": False,           # phone in the hand / at the face / hand at the ear
    }


def internal_state():
    """new_state() + values used only by the alert rules (not written to state.json)."""
    s = new_state()
    s.update({
        "blink_rate_valid": False,    # a full minute watched: blink_rate is meaningful
        "talking": False,             # the driver is talking (lips moving)
        "phone_in_hand": False,       # a phone touching a hand (YOLO)
        "phone_near_face": False,     # a phone at the face (YOLO)
        "hand_at_mouth": False,       # a hand at the mouth
    })
    return s


def public(s):
    """The part of an internal state that goes to state.json."""
    return {k: s[k] for k in new_state()}

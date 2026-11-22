"""Alert engine: turns the detections in the state into counters, ONE alert level and an event log.
This is what the LLM and the dashboard react to.

  - alert_level:   none | info | warning | critical (highest active alert)
  - alert_reason:  most important active alert code (see RULES), "" if none
  - active_alerts: all active alert codes, most important first
  - events:        the last EVENTS_MAX alerts that started (time + level)

Three kinds of rules:
  - instant:   something happening now (eyes closed, phone, looking away ...)
  - repeated:  something counted over a time window (3 yawns in 5 min, 3 eye rubs in 5 min ...)
  - combined:  "drowsy" = at least DROWSY_SIGNS different fatigue signs at the same time,
               "distraction" = talking while looking away from the road

Call the LLM when alert_level or alert_reason changes, not on every update.
"""
import time
from collections import deque

ALERT_HOLD_S = 2.0         # TUNE: an alert stays active this long after its condition ends (no flicker)
EVENTS_MAX = 5             # recent events kept in the state

# --- thresholds ---
FACE_LOST_ALERT_S = 3.0    # TUNE: driver's face missing this long = critical
YAWNS_WINDOW = 300.0       # yawns counted over 5 min...
YAWNS_WARN = 3             # TUNE: ...this many = frequent yawning
EYE_RUBS_WINDOW = 300.0    # eye rubs counted over 5 min...
EYE_RUBS_WARN = 3          # TUNE: ...this many = frequent eye rubbing
MICROSLEEP_WINDOW = 600.0  # microsleeps counted over 10 min...
MICROSLEEPS_CRIT = 2       # TUNE: ...the 2nd one (and later ones) = repeated microsleep, shown while it happens
NODS_WARN = 2              # TUNE: head nods per minute = nodding
SLOW_BLINKS_WARN = 3       # TUNE: slow / heavy blinks per minute = slow blinking
LOW_BLINK_RATE = 5         # TUNE: fewer blinks per minute = staring / low blink rate (normal: ~15-20)
DROWSY_SIGNS = 2           # TUNE: this many different fatigue signs at once = drowsy
DISTRACTION_MIN_S = 1.0    # TUNE: talking while looking away from the road this long = distraction

LEVELS = ["none", "info", "warning", "critical"]

# fatigue signs counted for the combined "drowsy" alert
FATIGUE_SIGNS = ("frequent_yawning", "frequent_eye_rubbing", "slow_blinking", "low_blink_rate", "head_nodding")

# (code, level, condition) - ORDER = PRIORITY: the first active rule is alert_reason.
# "drowsy" has no condition here: it is computed from FATIGUE_SIGNS in AlertEngine.update.
RULES = [
    # critical
    ("sleep",                "critical", lambda s: s["sleep"]),
    ("repeated_microsleep",  "critical", lambda s: (s["microsleep"] or s["sleep"])
                                                   and s["microsleep_count_10min"] >= MICROSLEEPS_CRIT),
    ("microsleep",           "critical", lambda s: s["microsleep"]),
    ("head_drop",            "critical", lambda s: s["head_drop"]),
    ("drowsy",               "critical", None),
    ("driver_missing",       "critical", lambda s: s["face_lost_sec"] >= FACE_LOST_ALERT_S
                                                   and not s["camera_blocked"]),   # covered camera: only that
    # warning
    ("phone_use",            "warning",  lambda s: s["phone_use"]),
    ("long_glance",          "warning",  lambda s: s["long_glance"]),
    ("distraction",          "warning",  lambda s: s["distraction"]),
    ("glasses_missing",      "warning",  lambda s: s["glasses_missing"]),
    ("head_nodding",         "warning",  lambda s: s["head_nods_1min"] >= NODS_WARN),
    ("frequent_yawning",     "warning",  lambda s: s["yawn_count_5min"] >= YAWNS_WARN),
    ("frequent_eye_rubbing", "warning",  lambda s: s["eye_rub_count_5min"] >= EYE_RUBS_WARN),
    ("slow_blinking",        "warning",  lambda s: s["slow_blinks_1min"] >= SLOW_BLINKS_WARN),
    ("low_blink_rate",       "warning",  lambda s: s["blink_rate_valid"] and s["blink_rate"] < LOW_BLINK_RATE),
    ("camera_blocked",       "warning",  lambda s: s["camera_blocked"]),
    # info
    ("yawning",              "info",     lambda s: s["yawning"]),
    ("rubbing_eyes",         "info",     lambda s: s["hand_at_eyes"]),
    ("hand_at_mouth",        "info",     lambda s: s["hand_at_mouth"]),
    ("phone_visible",        "info",     lambda s: s["phone_detected"]),
    ("head_tilt",            "info",     lambda s: s["head_tilt"]),
    ("hand_near_face",       "info",     lambda s: s["hand_near_face"]),
    ("low_light",            "info",     lambda s: s["low_light"]),
]
LEVEL_OF = {code: level for code, level, _ in RULES}


class RisingCounter:
    """How many times a flag went from False to True within a time window."""

    def __init__(self, window):
        self.window, self.times, self.prev = window, deque(), False

    def update(self, t, flag):
        if flag and not self.prev:
            self.times.append(t)
        self.prev = bool(flag)
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()
        return len(self.times)


class AlertEngine:
    def __init__(self):
        self.yawns = RisingCounter(YAWNS_WINDOW)
        self.eye_rubs = RisingCounter(EYE_RUBS_WINDOW)
        self.micro = RisingCounter(MICROSLEEP_WINDOW)
        self.last_seen = {}          # code -> last time its condition was true
        self.active_prev = set()
        self.events = deque(maxlen=EVENTS_MAX)

    def update(self, s, t):
        """Adds the counters, alert_level / alert_reason / active_alerts / events to the state s."""
        s["phone_use"] = bool(s["phone_in_hand"] or s["phone_near_face"] or s["hand_at_ear"])
        s["yawn_count_5min"] = self.yawns.update(t, s["yawning"])
        s["eye_rub_count_5min"] = self.eye_rubs.update(t, s["hand_at_eyes"])
        s["microsleep_count_10min"] = self.micro.update(t, s["microsleep"] or s["sleep"])
        s["distraction"] = bool(s["talking"] and s["looking_away_sec"] >= DISTRACTION_MIN_S)

        now = {code: cond(s) for code, _, cond in RULES if cond}
        now["drowsy"] = sum(now[c] for c in FATIGUE_SIGNS) >= DROWSY_SIGNS
        for code, active in now.items():
            if active:
                self.last_seen[code] = t
        active = [(code, lvl) for code, lvl, _ in RULES
                  if code in self.last_seen and t - self.last_seen[code] <= ALERT_HOLD_S]
        codes = [c for c, _ in active]

        for code, lvl in active:                     # newly started alerts -> events
            if code not in self.active_prev:
                self.events.append({"ts": round(t, 2), "time": time.strftime("%H:%M:%S", time.localtime(t)),
                                    "type": code, "level": lvl})
        self.active_prev = set(codes)

        s["alert_level"] = max((lvl for _, lvl in active), key=LEVELS.index, default="none")
        s["alert_reason"] = codes[0] if codes else ""
        s["active_alerts"] = codes
        s["events"] = list(self.events)
        return s

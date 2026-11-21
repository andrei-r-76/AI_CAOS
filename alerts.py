"""Alert engine: turns the raw detections in the state into scores, counters, ONE alert level and an
event log. This is what the LLM (R2) and the dashboard (R4) should react to.

  - alert_level:   none | info | warning | critical (the highest active alert)
  - alert_reason:  code of the most important active alert (see RULES), "" if none
  - active_alerts: all active alert codes, most important first
  - events:        the last EVENTS_MAX alerts that started, with time and level
  - drowsiness_score / fatigue_level: combined fatigue (ok | tired | drowsy)
  - attention_score: combined attention to the road (0-100)

Call the LLM when alert_level or alert_reason changes, not on every state update.
"""
import time
from collections import deque

FACE_LOST_ALERT_S = 3.0    # TUNE: the driver's face missing this long = critical (left the seat, slumped, blocked)
ALERT_HOLD_S = 2.0         # TUNE: an alert stays active this long after its condition ends (no flicker)
EVENTS_MAX = 5             # how many recent events are kept in the state (context for the LLM)
YAWNS_WINDOW = 300.0       # yawns are counted over 5 minutes
YAWNS_WARN = 3             # TUNE: this many yawns in 5 minutes = frequent yawning
MICROSLEEP_WINDOW = 600.0  # microsleeps are counted over 10 minutes
NODS_WARN = 2              # TUNE: this many head nods per minute = warning


LEVELS = ["none", "info", "warning", "critical"]

# (code, level, condition) - ORDER = PRIORITY: the first active rule becomes alert_reason
RULES = [
    ("sleep",            "critical", lambda s: s["sleep"]),
    ("microsleep",       "critical", lambda s: s["microsleep"]),
    ("head_drop",        "critical", lambda s: s["head_drop"]),
    ("driver_missing",   "critical", lambda s: s["face_lost_sec"] >= FACE_LOST_ALERT_S),
    ("phone_use",        "warning",  lambda s: s["phone_use"]),
    ("long_glance",      "warning",  lambda s: s["long_glance"]),
    ("fatigue_high",     "warning",  lambda s: s["fatigue_level"] == "drowsy"),
    ("fatigue_medium",   "info",     lambda s: s["fatigue_level"] == "tired"),
    ("head_nodding",     "warning",  lambda s: s["head_nods_1min"] >= NODS_WARN),
    ("perclos",          "warning",  lambda s: s["perclos_drowsy"]),
    ("hypnosis",         "warning",  lambda s: s["hypnosis"]),
    ("frequent_yawning", "warning",  lambda s: s["yawn_count_5min"] >= YAWNS_WARN),
    ("mirror_too_long",  "warning",  lambda s: s["mirror_too_long"]),
    ("camera_blocked",   "warning",  lambda s: s["camera_blocked"]),
    ("yawning",          "info",     lambda s: s["yawning"]),
    ("rubbing_eyes",     "info",     lambda s: s["hand_at_eyes"]),
    ("drinking",         "info",     lambda s: s["drinking"]),
    ("phone_visible",    "info",     lambda s: s["phone_detected"]),
    ("head_tilt",        "info",     lambda s: s["head_tilt"]),
    ("eyes_partially_closed", "info", lambda s: s["eyes_partially_closed"]),
    ("hand_near_face",   "info",     lambda s: s["hand_near_face"]),
    ("low_light",        "info",     lambda s: s["low_light"]),
]


class RisingCounter:
    """Counts how many times a flag switched from False to True within a time window."""

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
        self.micro = RisingCounter(MICROSLEEP_WINDOW)
        self.last_seen = {}          # code -> last time its condition was true
        self.active_prev = set()
        self.events = deque(maxlen=EVENTS_MAX)
        self.counts = {}             # code -> how many times it started since launch

    def update(self, s, t):
        """Adds counters, scores, alert_level / alert_reason / active_alerts / events to the state s."""
        s["phone_use"] = bool(s["phone_in_hand"] or s["phone_near_face"] or s["hand_at_ear"])
        s["yawn_count_5min"] = self.yawns.update(t, s["yawning"])
        s["microsleep_count_10min"] = self.micro.update(t, s["microsleep"] or s["sleep"])

        for code, _, cond in RULES:
            if cond(s):
                self.last_seen[code] = t
        active = [(code, lvl) for code, lvl, _ in RULES
                  if code in self.last_seen and t - self.last_seen[code] <= ALERT_HOLD_S]
        codes = [c for c, _ in active]

        for code, lvl in active:                     # newly started alerts -> events
            if code not in self.active_prev:
                self.counts[code] = self.counts.get(code, 0) + 1
                self.events.append({"ts": round(t, 2), "time": time.strftime("%H:%M:%S", time.localtime(t)),
                                    "type": code, "level": lvl})
        self.active_prev = set(codes)

        s["alert_level"] = max((lvl for _, lvl in active), key=LEVELS.index, default="none")
        s["alert_reason"] = codes[0] if codes else ""
        s["active_alerts"] = codes
        s["events"] = list(self.events)
        return s

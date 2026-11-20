"""Shared data contract between modules. Changes need a PR approved by every module owner."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum


class Level(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriverState:
    """Produced by vision.py once per frame."""
    timestamp: float = field(default_factory=time.time)
    face_detected: bool = False
    ear: float = 0.0               # eye aspect ratio (avg of both eyes)
    eyes_closed: bool = False
    eyes_closed_sec: float = 0.0   # how long eyes have been continuously closed
    perclos: float = 0.0           # 0..1, share of the last window with eyes closed
    yaw: float = 0.0               # head pose, degrees (left/right)
    pitch: float = 0.0             # head pose, degrees (up/down)
    looking_away: bool = False
    looking_away_sec: float = 0.0
    drowsy: bool = False
    distracted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alert:
    """Produced by llm.py, consumed by tts.py and the dashboard."""
    level: Level
    reason: str                    # e.g. "drowsy", "distracted"
    message: str                   # text to show and speak
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d

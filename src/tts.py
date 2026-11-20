"""Voice alerts (owner: Cristi).

TODO phase 1:
- speak Alert.message with pyttsx3 in a background thread (never block the camera loop)
- short warning beep before CRITICAL alerts
- don't overlap: drop or replace a message if one is already playing
"""
from src.events import Alert


class Speaker:
    def say(self, alert: Alert) -> None:
        print(f"[TTS] {alert.level.value}: {alert.message}")  # stub

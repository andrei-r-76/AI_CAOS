"""R1: driver detection from the webcam (owner: Cristi).

TODO phase 1:
- MediaPipe Face Landmarker -> landmarks
- EAR per eye -> eyes_closed, eyes_closed_sec
- PERCLOS over config.PERCLOS_WINDOW_SEC
- head pose (yaw/pitch) via cv2.solvePnP -> looking_away, looking_away_sec
"""
from src.events import DriverState


class DriverMonitor:
    def process(self, frame) -> DriverState:
        """Analyse one BGR frame and return the current driver state."""
        return DriverState()  # stub

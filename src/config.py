"""All tunable thresholds and settings in one place."""
import os

# Camera
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Eyes / drowsiness
EAR_CLOSED_THRESHOLD = 0.20     # below this = eye closed
EYES_CLOSED_ALERT_SEC = 1.5     # microsleep
PERCLOS_WINDOW_SEC = 60
PERCLOS_DROWSY = 0.15

# Head pose / distraction
YAW_AWAY_DEG = 30
PITCH_DOWN_DEG = 20
LOOKING_AWAY_ALERT_SEC = 2.0

# Alerts
ALERT_COOLDOWN_SEC = 8          # don't repeat the same alert more often than this

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # ollama | anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# Dashboard
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8000"))

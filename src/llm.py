"""R2: turn the driver state into an in-car alert using an LLM (owner: Teammate 2).

TODO phase 1:
- decide if an alert is needed (drowsy / distracted + cooldown from config)
- ask the LLM (config.LLM_PROVIDER) for a short, calm, spoken-style message
- fall back to a fixed message if the LLM is slow or unavailable
"""
from __future__ import annotations

from src.events import Alert, DriverState


class AlertEngine:
    def evaluate(self, state: DriverState) -> Alert | None:
        """Return an Alert if the driver needs one, otherwise None."""
        return None  # stub

"""The driver state shared by vision -> llm -> dashboard.

Vision fills it every frame, llm.py reads it, the dashboard shows it (it's a plain dict, so it is JSON-ready).
Adding a key: OK, tell the team. Renaming or removing a key: ask everyone first.
"""


def new_state():
    """A driver state with every key set to its 'nothing detected' value."""
    return {
        "face_found": False,        # is a face visible
        "eyes_closed_sec": 0.0,     # seconds eyes closed continuously (0 when open)
        "perclos": 0.0,             # 0..1, share of the last 60 s with eyes closed
        "looking_away_sec": 0.0,    # seconds head turned away/down continuously
        "hand_near_face": False,    # phone-use proxy
    }

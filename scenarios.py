"""Simulated events outside the car, triggered by hand during the demo.

Each event has: what the LLM is told, words the alert must mention, and a fixed fallback message.
"""

EVENTS = {
    "pedestrian": {
        "situation": "a pedestrian is crossing the road ahead, not at a crosswalk",
        "must_mention": ["pedestrian", "person", "someone", "crossing"],
        "fallback": "Pedestrian crossing ahead! Slow down.",
    },
    "car_braking": {
        "situation": "the car ahead is braking hard",
        "must_mention": ["car", "brak", "vehicle", "slow"],
        "fallback": "The car ahead is braking! Slow down.",
    },
    "cyclist": {
        "situation": "a cyclist is riding close on the right side",
        "must_mention": ["cyclist", "bike", "bicycle", "rider"],
        "fallback": "Cyclist on your right. Keep your distance.",
    },
    "animal": {
        "situation": "an animal is on the road ahead",
        "must_mention": ["animal", "deer", "dog"],
        "fallback": "Animal on the road ahead! Slow down.",
    },
}

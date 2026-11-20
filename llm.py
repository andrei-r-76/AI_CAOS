"""LLM alerts: turn the driver state into one short spoken warning.

Run the test:  uv run python llm.py      (needs Ollama running + `ollama pull llama3.2:1b`)

Vision decides WHEN to alert. The LLM only decides WHAT to say.
If the LLM is slow, down or says something unsafe, a fixed message is used instead.
"""
import re

import ollama

MODEL = "llama3.2:1b"
TIMEOUT_SEC = 3

SYSTEM = ("You are a car safety assistant. You speak to the driver to keep them awake "
          "and focused. Reply with ONE short spoken sentence, max 12 words.")
EXAMPLES = [
    ("eyes closed for 2 seconds", "Wake up! Open your eyes and focus on the road."),
    ("looking away from the road for 3 seconds", "Eyes back on the road, please."),
    ("holding a phone while driving", "Please put the phone down and focus on driving."),
    ("eyes often closing, getting drowsy", "You seem tired. Let's find a place to take a break."),
]
FALLBACK = {
    "drowsy": "Wake up! Open your eyes and focus on the road.",
    "distracted": "Eyes back on the road, please.",
    "phone": "Please put the phone down and focus on driving.",
}
UNSAFE = ["keep your eyes closed", "close your eyes", "go to sleep", "get some sleep", "nap",  # dangerous advice
          "i cannot", "i can't", "sorry", "i don't", "as an ai"]          # refusals / nonsense
# the alert must mention at least one of these, or it's off-topic
RELEVANT = {
    "drowsy": ["wake", "eyes", "tired", "break", "awake", "rest stop", "open"],
    "distracted": ["road", "eyes", "focus", "look", "ahead"],
    "phone": ["phone"],
}

client = ollama.Client(timeout=TIMEOUT_SEC)


def describe(state):
    """Return (reason, situation text) or None if the driver is fine."""
    if state.get("eyes_closed_sec", 0) >= 1.5:
        return "drowsy", f"eyes closed for {state['eyes_closed_sec']:.0f} seconds"
    if state.get("perclos", 0) >= 0.15:
        return "drowsy", "eyes often closing, getting drowsy"
    if state.get("looking_away_sec", 0) >= 2:
        return "distracted", f"looking away from the road for {state['looking_away_sec']:.0f} seconds"
    if state.get("hand_near_face"):
        return "phone", "holding a phone while driving"
    return None


def clean(text):
    """Keep only the first sentence, without quotes or 'Situation:' leftovers."""
    text = re.sub(r"(?i)situation:.*?->", "", text).strip()
    quoted = re.findall(r"['\"“](.+?)['\"”]", text)
    if quoted:
        text = quoted[0]
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip()


def is_ok(text, reason):
    low = text.lower()
    return (2 <= len(text.split()) <= 20
            and not any(p in low for p in UNSAFE)
            and any(k in low for k in RELEVANT[reason]))


def warm_up():
    """Load the model into RAM at startup so the first real alert is fast."""
    try:
        client.generate(model=MODEL, prompt="hi", options={"num_predict": 1}, keep_alive="30m")
    except Exception as e:
        print(f"[llm] warm-up failed ({type(e).__name__}): alerts will use fixed messages")


def get_alert(state):
    """Return the alert text to speak, or None if no alert is needed."""
    found = describe(state)
    if not found:
        return None
    reason, situation = found

    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in EXAMPLES:
        messages += [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": situation})

    try:
        reply = client.chat(model=MODEL, messages=messages,
                            options={"num_predict": 30}, keep_alive="30m")
        text = clean(reply["message"]["content"])
        if is_ok(text, reason):
            return text
        print(f"[llm] rejected {text!r}, using fallback")
    except Exception as e:  # LLM down, timeout, model missing...
        print(f"[llm] fallback ({type(e).__name__}: {e})")
    return FALLBACK[reason]


if __name__ == "__main__":
    import time

    from state import new_state

    def fake(**values):
        s = new_state()
        s.update(face_found=True, **values)
        return s

    warm_up()
    fake_states = [
        fake(perclos=0.05),                      # all good -> None
        fake(eyes_closed_sec=2.1, perclos=0.2),  # drowsy
        fake(looking_away_sec=3.4),              # distracted
        fake(hand_near_face=True),               # phone
    ]
    for s in fake_states * 3:
        t = time.time()
        print(f"{get_alert(s)!r:70} ({time.time() - t:.1f}s)")

"""Main loop: webcam -> vision -> llm -> tts + dashboard (owner: Teammate 2)."""
import cv2

from src import config
from src.llm import AlertEngine
from src.tts import Speaker
from src.vision import DriverMonitor


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    monitor, engine, speaker = DriverMonitor(), AlertEngine(), Speaker()
    # TODO: start dashboard server and push state/alerts to it
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            state = monitor.process(frame)
            alert = engine.evaluate(state)
            if alert:
                speaker.say(alert)
            cv2.imshow("AI_CAOS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

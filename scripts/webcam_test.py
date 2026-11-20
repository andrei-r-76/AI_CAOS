"""Webcam test: shows the camera feed. Press q to quit."""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Webcam test (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

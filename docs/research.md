# Research: existing driver-safety mechanisms

Team research notes for the AI Drive track. Each teammate adds their findings under their own section at the bottom.

## Why it matters
- **EU General Safety Regulation (EU) 2019/2144** requires:
  - **DDAW** (Driver Drowsiness and Attention Warning): mandatory for new vehicles since 2022/2024.
  - **ADDW** (Advanced Driver Distraction Warning, camera-based): mandatory for all new registrations from July 2026.
- **Euro NCAP 2026** scores driver monitoring: distraction, drowsiness, unresponsive driver, phone use.
- Human error (distraction, fatigue, phone) causes most crashes.

## Production systems
| System | Sensing | Detects | Reaction |
|---|---|---|---|
| Mercedes Attention Assist, Bosch drowsiness detection | steering-input patterns, time driven | fatigue | coffee-cup icon, chime, break suggestion |
| Subaru DriverFocus, Mazda DMS, BMW Driving Assistant Pro, GM Super Cruise, Ford BlueCruise | IR driver-facing camera | eyes closed, gaze off road | escalating visual → audio → seat/wheel haptics → hands-on request → controlled stop |
| Volvo / Mercedes emergency stop assist | camera + no steering input | unresponsive driver | hazard lights, slows to a stop in lane, emergency call |
| Tier-1 suppliers: Seeing Machines, Smart Eye, Cipia, Tobii | NIR camera + neural nets | gaze zones, PERCLOS, phone, smoking, seatbelt, occupancy | defined by the carmaker |
| Lane Departure Warning / Lane Keep Assist | front camera | lane drift (indirect sign of fatigue) | steering-wheel vibration |

### The common alert pattern: an escalation ladder
1. **L1:** icon on the cluster
2. **L2:** chime + text message
3. **L3:** haptic (seat or steering-wheel vibration) + voice
4. **L4:** intervention: slow down, hazard lights, SOS call

Cooldowns between alerts prevent alarm fatigue (drivers ignore or disable systems that nag).

## Computer-vision metrics (all work with a laptop webcam)
| Metric | How | Typical threshold |
|---|---|---|
| **EAR** (eye aspect ratio) | ratio of vertical to horizontal eye-landmark distances | < ~0.2 = eye closed. Closed > ~1.5 s = microsleep |
| **PERCLOS** | % of time eyes are ≥80% closed over a rolling 60 s window | > 15% = drowsy (industry standard) |
| Blink rate and duration | from the EAR time series | long, slow blinks = fatigue |
| **MAR** (mouth aspect ratio) | mouth-landmark distances | yawn = MAR high for > ~1 s |
| **Head pose** | yaw/pitch/roll via `cv2.solvePnP` on face landmarks | looking away or down > ~2 s = distraction. Pitch nodding = drowsiness |
| Gaze | iris landmarks (MediaPipe) | off-road gaze zone |
| Phone use | YOLO (COCO class `cell phone`) + hand near the ear/face | phone visible near the driver |
| Posture | MediaPipe Pose: head/shoulder slump compared with a baseline | slumping |

**Per-driver calibration** (record a 10 s baseline of EAR and head pose at startup) makes every threshold much more robust across people and camera positions.

## Tech-stack choices
- Python 3.11, OpenCV, MediaPipe Face Landmarker (later: Hands, Pose, YOLOv8n).
- Dashboard: FastAPI + WebSocket, single HTML page.
- LLM: behind one interface. Default is **Ollama** running locally (free, offline, low latency), with an optional API key for a hosted model.
  Note: Claude Pro and ChatGPT Pro subscriptions do **not** include API access.
- TTS: `pyttsx3` (offline) or the browser's `speechSynthesis`.
- Simulated wheel vibration: shaking wheel animation on the dashboard + low-frequency rumble sound.

## Where we can differentiate (innovation)
- The LLM chooses the alert's tone and urgency from the driver state + context (time driving, time of day, repeated events), instead of a fixed beep.
- **Conversational alertness check:** the assistant asks a short question. Reply latency and coherence feed back into the attention score.
- Break suggestions with a natural-language explanation of *why*.
- Optional RAG over a small "car manual / traffic rules" text for passenger questions.

---

## Teammate research
<!-- Each person: add a "### <name>" section below with your findings and sources. -->

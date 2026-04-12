# Pysimverse

Example **Python drone missions** built on the **`pysimverse`** SDK (`from pysimverse import Drone`). Scripts connect to a drone (simulator or hardware, depending on how your environment provides `Drone`), run takeoff / flight / land flows, and some missions combine **OpenCV**, **keyboard** input, **MediaPipe** vision, or **cvzone** helpers.

## Prerequisites

- Python **3.10+** is a reasonable default (match whatever your `pysimverse` and MediaPipe builds support).
- The **`pysimverse`** package installed and configured so `Drone` can connect (follow your course or simulator vendor instructions).
- A webcam for vision-based scripts (`body_follower_level1.py`, `hand_gesture_level1.py`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install Python dependencies used across the repo (versions are not pinned here):

```bash
pip install pysimverse opencv-python keyboard mediapipe cvzone
```

On Windows, the `keyboard` library may need elevated permissions in some setups; if keys are not detected, run the terminal as administrator or check the library’s documentation.

## Layout

| Path | Purpose |
|------|---------|
| `*.py` (repo root) | **Level / mission** scripts you run directly. |
| `templates/` | Smaller examples and **shared MediaPipe helpers** loaded by some root scripts. |
| `models/` | MediaPipe **task model** files (`.task`) for pose and hand landmarks. |
| `image_capture_screenshots/` | Default output folder for `image_capture_level1.py` screenshots. |

## Root missions (`python <script>.py`)

| Script | What it does |
|--------|----------------|
| `garage_level1.py` | Scripted flight: takeoff, set speed, rotate, move forward, land. |
| `image_capture_level1.py` | Live video, keyboard RC (WASD / arrows / Q–E), **Z** saves a PNG, **X** or **ESC** lands and exits. |
| `body_follower_level1.py` | Webcam **pose** via MediaPipe; detects a **jump** and sends a short upward RC pulse. Uses helpers from `templates/body_follower_mediapipe.py`. Press **q** or **ESC** to quit. |
| `hand_gesture_level1.py` | Webcam **hand** landmarks; **left / center / right** zones drive lateral RC. Uses `templates/hand_gesture_mediapipe.py`. |

## Templates (`python templates/<script>.py` from repo root, or open in your IDE)

| Script | What it does |
|--------|----------------|
| `first_flight.py` | Minimal connect → takeoff → move → rotate → land. |
| `rc_control.py` | Short loop sending `send_rc_control(...)` then land. |
| `keyboard_control.py` | Continuous RC from keys (**WASD**, **arrows**, **Q/E** yaw); **X** / **ESC** exits. |
| `drone_video_stream.py` | `streamon`, show `get_frame()` in a window (no land in loop; adjust as needed for your session). |
| `color_detection.py` | Drone camera + **cvzone** `ColorFinder` HSV example (stacked view); **q** quits. |
| `body_follower_mediapipe.py` | Pose landmarker utilities and drawing (imported by `body_follower_level1.py`). |
| `hand_gesture_mediapipe.py` | Hand landmarker utilities and drawing (imported by `hand_gesture_level1.py`). |

## MediaPipe models

Vision scripts expect landmark models under `models/` (for example `pose_landmarker_lite.task`, `hand_landmarker.task`). The template modules download or verify paths as needed; keep this folder populated if you run offline.

## Minimal API smoke test

```python
import time
from pysimverse import Drone

drone = Drone()
drone.connect()
drone.take_off()
time.sleep(3)
drone.land()
```

Use `templates/first_flight.py` for a slightly richer scripted path example (`set_speed`, `move_forward`, `rotate`).

"""
Live hand detection and landmark visualization using MediaPipe Hand Landmarker
and an OpenCV camera feed.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand landmarker uses the same 21-landmark topology as legacy MediaPipe Hands.
HAND_CONNECTIONS = frozenset(
    [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "../models/hand_landmarker.task"


def ensure_model(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand skeleton and joints on a copy of the RGB frame."""
    annotated = rgb_image.copy()
    if not detection_result.hand_landmarks:
        return annotated
    h, w, _ = annotated.shape
    for hand_landmarks in detection_result.hand_landmarks:
        pts = []
        for lm in hand_landmarks:
            pts.append((int(lm.x * w), int(lm.y * h)))
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(annotated, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
        for px, py in pts:
            cv2.circle(annotated, (px, py), 4, (255, 0, 0), -1)
    return annotated


def main() -> None:
    model_path = ensure_model(DEFAULT_MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera (index 0).")
        raise SystemExit(1)

    print("Hand detection running. Press 'q' or ESC to quit.")

    t0 = time.monotonic()
    last_ts_ms = -1
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - t0) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            result = detector.detect_for_video(mp_image, ts_ms)

            annotated_rgb = draw_landmarks_on_image(rgb, result)
            out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hand landmarks (MediaPipe)", out_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()

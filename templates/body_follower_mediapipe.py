"""
Live body pose detection and landmark visualization using MediaPipe Pose Landmarker
and an OpenCV camera feed.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# BlazePose / Pose Landmarker topology (33 landmarks).
POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "../models/pose_landmarker_lite.task"


def ensure_model(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose skeleton and joints on a copy of the RGB frame."""
    annotated = rgb_image.copy()
    if not detection_result.pose_landmarks:
        return annotated
    h, w, _ = annotated.shape
    for pose_landmarks in detection_result.pose_landmarks:
        pts = []
        for lm in pose_landmarks:
            pts.append((int(lm.x * w), int(lm.y * h)))
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(annotated, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
        for px, py in pts:
            cv2.circle(annotated, (px, py), 4, (255, 0, 0), -1)
    return annotated


def main() -> None:
    model_path = ensure_model(DEFAULT_MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera (index 0).")
        raise SystemExit(1)

    print("Body pose detection running. Press 'q' or ESC to quit.")

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
            cv2.imshow("Body pose landmarks (MediaPipe)", out_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()

"""
One-hand horizontal zones: left / deadzone / right, using MediaPipe from
templates/hand_gesture_mediapipe.py. Drives pysimverse RC strafe (same axis as
templates/keyboard_control.py: A/D). Prints "left" or "right" on zone change;
deadzone / no hand sends zero lateral RC.
"""

import importlib.util
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pysimverse import Drone

_ROOT = Path(__file__).resolve().parent
_TEMPLATE = _ROOT / "templates" / "hand_gesture_mediapipe.py"
_spec = importlib.util.spec_from_file_location("hand_gesture_mediapipe", _TEMPLATE)
_hgm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hgm)

ensure_model = _hgm.ensure_model
draw_landmarks_on_image = _hgm.draw_landmarks_on_image
DEFAULT_MODEL_PATH = _hgm.DEFAULT_MODEL_PATH

# Half-width of center deadzone in mirrored normalized x (total band = 2 * this).
DEADZONE_HALF = 0.08

# Wrist landmark index (stable for horizontal position).
WRIST_IDX = 0

# Lateral RC speed (-100..100); sign matches keyboard_control (A = negative, D = positive).
RC_SPEED = 100


def zone_from_x(x_norm: float) -> str | None:
    """Return 'left', 'right', or None (deadzone). x_norm in [0, 1] (image coords).

    Uses mirrored x so labels match a typical mirrored webcam preview.
    """
    x_m = 1.0 - x_norm
    if x_m < 0.5 - DEADZONE_HALF:
        return "left"
    if x_m > 0.5 + DEADZONE_HALF:
        return "right"
    return None


def draw_zones(bgr_image, w: int, h: int) -> None:
    """Side bands + narrow center deadzone; tints match zone_from_x (mirrored)."""
    x1 = max(0, int((0.5 - DEADZONE_HALF) * w))
    x2 = min(w, int((0.5 + DEADZONE_HALF) * w))

    overlay = bgr_image.copy()
    # Logical right = low raw x; logical left = high raw x.
    cv2.rectangle(overlay, (0, 0), (x1, h), (0, 0, 80), -1)
    cv2.rectangle(overlay, (x2, 0), (w, h), (80, 0, 0), -1)
    cv2.rectangle(overlay, (x1, 0), (x2, h), (60, 60, 60), -1)
    cv2.addWeighted(overlay, 0.25, bgr_image, 0.75, 0, bgr_image)

    cv2.line(bgr_image, (x1, 0), (x1, h), (0, 255, 255), 2)
    cv2.line(bgr_image, (x2, 0), (x2, h), (0, 255, 255), 2)


def main() -> None:
    model_path = ensure_model(DEFAULT_MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera (index 0).")
        detector.close()
        raise SystemExit(1)

    drone: Drone | None = None
    print(
        "Mirrored: left = red band, right = blue, narrow gray center = deadzone "
        "(zero strafe). Press 'q' or ESC to quit."
    )

    t0 = time.monotonic()
    last_ts_ms = -1
    prev_zone: str | None = None

    try:
        drone = Drone()
        drone.connect()
        drone.take_off(10)

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - t0) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            result = detector.detect_for_video(mp_image, ts_ms)

            zone: str | None = None
            if result.hand_landmarks:
                wrist = result.hand_landmarks[0][WRIST_IDX]
                zone = zone_from_x(wrist.x)

            if zone == "left":
                left_right = -RC_SPEED
            elif zone == "right":
                left_right = RC_SPEED
            else:
                left_right = 0

            drone.send_rc_control(left_right, 0, 0, 0)

            annotated_rgb = draw_landmarks_on_image(rgb, result)
            out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            draw_zones(out_bgr, w, h)

            if zone != prev_zone:
                if zone == "left":
                    print("left")
                elif zone == "right":
                    print("right")
                prev_zone = zone

            label = zone or "deadzone"
            cv2.putText(
                out_bgr,
                f"zone: {label}  RC lr: {left_right}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand gesture level 1 (left | deadzone | right)", out_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if drone is not None:
            try:
                drone.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            try:
                drone.land()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()

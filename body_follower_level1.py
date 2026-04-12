"""
Jump detection from a webcam using MediaPipe Pose Landmarker
(templates/body_follower_mediapipe.py).

Ways to detect a jump from pose landmarks (pick one or combine):

1. **Hip height vs baseline (implemented)** — In image coordinates, y grows
   downward, so the body moving up means hip y *decreases*. Track a slow EMA of
   mid-hip y while grounded; a sudden drop below that baseline signals takeoff;
   return near baseline signals landing. Add cooldown to avoid double counts.

2. **Vertical velocity / acceleration** — Compute per-frame delta of hip (or
   nose) y; a large negative spike is upward motion. Smooth with a short EMA and
   threshold; peak-picking on -vy works well for crisp events.

3. **World landmarks** — `pose_world_landmarks` gives metric-ish coordinates
   with hip origin; world hip or nose y can be more consistent across zoom than
   normalized y (still noisy without calibration).

4. **Leg geometry** — During a jump, ankle y often rises toward hip y (shorter
   hip–ankle vertical gap) or knee angles change; useful as a secondary check to
   reject arm-waving false positives.

5. **Temporal template / classifier** — Feed a short history of features (hip y,
   velocities, knee angles) to a small model or DTW; heavier but robust.

This script uses (1) plus a short-term upward velocity check, time-based baseline
smoothing, and a baseline freeze while the hips move up (reduces missed jumps when
the loop is fast or the baseline was chasing the body). On each detected jump it
sends a short upward RC pulse to the drone (same API as templates/rc_control.py:
send_rc_control(left_right, forward_backward, up_down, yaw)). Press 'q' or ESC to quit.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pysimverse import Drone

_ROOT = Path(__file__).resolve().parent
_TEMPLATE = _ROOT / "templates" / "body_follower_mediapipe.py"
_spec = importlib.util.spec_from_file_location("body_follower_mediapipe", _TEMPLATE)
_bfm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bfm)

ensure_model = _bfm.ensure_model
draw_landmarks_on_image = _bfm.draw_landmarks_on_image
DEFAULT_MODEL_PATH = _bfm.DEFAULT_MODEL_PATH

# BlazePose indices: left_hip=23, right_hip=24
IDX_L_HIP = 23
IDX_R_HIP = 24

# Mid-hip must move up by at least this much (normalized y) vs baseline to start a jump.
JUMP_UP_THRESHOLD = 0.040
# Upward speed (normalized y per second): hip y decreasing => positive. Catches fast
# takeoff when the baseline EMA has not lagged enough yet (common with high FPS).
VEL_THRESH_UP = 0.42
# Hysteresis: hip y must come back within this of baseline to accept landing.
LAND_HYSTERESIS = 0.042
# Baseline time constant (seconds): larger = stabler standing reference, less chasing
# the body during the upward phase (see moving_up gate below).
TAU_BASELINE_S = 0.65
# If hip y drops more than this vs previous frame, skip baseline blending (freeze
# "standing" level during takeoff so displacement stays large enough to fire).
FREEZE_BASELINE_DELTA_Y = 0.0014
# Minimum landmark visibility [0,1] to trust hips.
MIN_VIS = 0.45
# Ignore jump re-triggers for this long after a detected jump (seconds).
JUMP_COOLDOWN_S = 0.55

# Upward RC while a jump pulse is active (-100..100); matches keyboard up arrow.
RC_UP_SPEED = 100
# How long to hold up_down after each jump (seconds).
UP_PULSE_S = 0.45


def mid_hip_y(landmarks) -> float | None:
    """Average hip y in normalized image space, or None if not confident."""
    a, b = landmarks[IDX_L_HIP], landmarks[IDX_R_HIP]
    if a.visibility < MIN_VIS or b.visibility < MIN_VIS:
        return None
    return 0.5 * (a.y + b.y)


class JumpStateMachine:
    """grounded -> airborne on hip moving up; airborne -> grounded on return + cooldown."""

    def __init__(self) -> None:
        self.state = "grounded"
        self.baseline_y: float | None = None
        self.last_jump_time = 0.0
        self.prev_hip_y: float | None = None
        self._last_t: float | None = None

    def update(self, hip_y: float | None, now: float) -> tuple[bool, float, float]:
        """
        Returns (fired, displacement_up, upward_vel) for HUD.
        displacement_up = baseline_y - hip_y (positive means hips above baseline).
        upward_vel = (prev_hip_y - hip_y) / dt (positive means moving up in the frame).
        """
        if hip_y is None:
            return False, 0.0, 0.0

        if self.baseline_y is None:
            self.baseline_y = hip_y
            self.prev_hip_y = hip_y
            self._last_t = now
            return False, 0.0, 0.0

        dt = max(1e-4, now - self._last_t if self._last_t is not None else (1.0 / 30.0))
        self._last_t = now

        up_vel = 0.0
        if self.prev_hip_y is not None:
            up_vel = (self.prev_hip_y - hip_y) / dt

        disp = self.baseline_y - hip_y
        fired = False

        if self.state == "grounded":
            moving_up = (
                self.prev_hip_y is not None
                and hip_y < self.prev_hip_y - FREEZE_BASELINE_DELTA_Y
            )
            if not moving_up:
                alpha = 1.0 - math.exp(-dt / TAU_BASELINE_S)
                self.baseline_y = (1.0 - alpha) * self.baseline_y + alpha * hip_y
                disp = self.baseline_y - hip_y

            crossed = disp >= JUMP_UP_THRESHOLD or up_vel >= VEL_THRESH_UP
            if crossed and (now - self.last_jump_time) >= JUMP_COOLDOWN_S:
                self.state = "airborne"
                self.last_jump_time = now
                fired = True
        else:
            if hip_y >= self.baseline_y - LAND_HYSTERESIS:
                self.state = "grounded"

        self.prev_hip_y = hip_y
        return fired, disp, up_vel


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
        detector.close()
        raise SystemExit(1)

    print(
        "Jump detection (mid-hip vs baseline); each jump -> drone RC up briefly. "
        "Press 'q' or ESC to quit."
    )

    t0 = time.monotonic()
    last_ts_ms = -1
    jump_fsm = JumpStateMachine()
    jump_count = 0
    up_pulse_until = 0.0
    drone: Drone | None = None

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

            now = time.monotonic()
            hip_y: float | None = None
            if result.pose_landmarks:
                hip_y = mid_hip_y(result.pose_landmarks[0])

            jumped, disp_up, up_vel = jump_fsm.update(hip_y, now)
            if jumped:
                jump_count += 1
                print(f"jump ({jump_count})")
                up_pulse_until = now + UP_PULSE_S

            if drone is not None:
                if now < up_pulse_until:
                    drone.send_rc_control(0, 0, RC_UP_SPEED, 0)
                else:
                    drone.send_rc_control(0, 0, 0, 0)

            annotated_rgb = draw_landmarks_on_image(rgb, result)
            out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            baseline = jump_fsm.baseline_y
            pulsing = now < up_pulse_until
            line1 = f"state: {jump_fsm.state}  jumps: {jump_count}  RC up: {pulsing}"
            if hip_y is not None and baseline is not None:
                line2 = (
                    f"hip_y: {hip_y:.3f}  base: {baseline:.3f}  disp: {disp_up:.3f}  v_up: {up_vel:.2f}"
                )
            else:
                line2 = "hips: low visibility or missing"

            cv2.putText(
                out_bgr,
                line1,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out_bgr,
                line2,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Body follower level 1 — jump detection", out_bgr)

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

import cv2
import keyboard
import time
from datetime import datetime
from pathlib import Path
from pysimverse import Drone


# RC Speeds
SPEED = 50
ROTATION_SPEED = 1

# Directory for screenshots
CAPTURE_DIR = Path("image_capture_screenshots")


def main():
    drone = Drone()
    drone.connect()

    drone.streamon()
    drone.take_off(20)

    CAPTURE_DIR.mkdir(exist_ok=True, parents=True)

    print("Keyboard control + video active!")
    print("Z = screenshot, X or ESC = land and quit.")

    while True:
        # get frame and show video
        frame, is_success = drone.get_frame()

        if is_success:
            cv2.imshow("Drone Feed", frame)

        # Land and Quit
        if keyboard.is_pressed("x") or keyboard.is_pressed("ESC"):
            break

        # Screenshot when Z is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("z") or keyboard.is_pressed("z"):
            if is_success:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = CAPTURE_DIR / f"capture_{timestamp}.png"
                cv2.imwrite(str(path), frame)
                print(f"Screenshot saved to {path}")

        # Keyboard RC control
        forward_backward = SPEED if (keyboard.is_pressed("w")) else 0
        if keyboard.is_pressed("s"): forward_backward = -SPEED

        left_right = SPEED if keyboard.is_pressed("d") else 0
        if keyboard.is_pressed("a"): left_right = -SPEED

        up_down = SPEED if keyboard.is_pressed("up") else 0
        if keyboard.is_pressed("down"): up_down = -SPEED

        yaw = ROTATION_SPEED if keyboard.is_pressed("e") else 0
        if keyboard.is_pressed("q"): yaw = -ROTATION_SPEED

        drone.send_rc_control(left_right, forward_backward, up_down, yaw)

        cv2.waitKey(1)
        time.sleep(0.02)

    cv2.destroyAllWindows()
    drone.streamoff()
    drone.land()
    exit(0)


if __name__ == "__main__":
    main()
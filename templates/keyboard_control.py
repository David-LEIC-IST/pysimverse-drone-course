from pysimverse import Drone
import time
import keyboard


# RC value range
MIN_RC = -100
MAX_RC = 100
SPEED = 100
ROTATION_SPEED = 1


def main():
    drone = Drone()
    drone.connect()

    drone.take_off(10)

    print("Keyboard control active!")

    while True:
        if keyboard.is_pressed("x") or keyboard.is_pressed("esc"):
            break

        # Poll key state and set RC values (hold to move)
        forward_backward = SPEED if keyboard.is_pressed("w") else 0
        if keyboard.is_pressed("s"): forward_backward = -SPEED

        left_right = SPEED if keyboard.is_pressed("d") else 0
        if keyboard.is_pressed("a"): left_right = -SPEED

        up_down = SPEED if keyboard.is_pressed("up") else 0
        if keyboard.is_pressed("down"): up_down = -SPEED

        yaw = ROTATION_SPEED if keyboard.is_pressed("e") else 0
        if keyboard.is_pressed("q"): yaw = -ROTATION_SPEED

        drone.send_rc_control(left_right, forward_backward, up_down, yaw)
        time.sleep(0.05)

    drone.land()


if __name__ == "__main__":
    main()

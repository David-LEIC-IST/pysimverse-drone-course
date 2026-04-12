from pysimverse import Drone

left_right = 0
forward_backward = 0
up_down = 0
yaw = 0

drone = Drone()
drone.connect()

drone.take_off()

i = 0

while i < 20:
    drone.send_rc_control(left_right, forward_backward, up_down, yaw)
    i += 1

drone.land()

from pysimverse import Drone
import cv2

drone = Drone()
drone.connect()

drone.streamon()
drone.take_off()

while True:
    frame, is_success = drone.get_frame()

    cv2.imshow("Drone Feed", frame)
    cv2.waitKey(1)

drone.land()

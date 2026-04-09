import time
from pysimverse import Drone

drone = Drone()
drone.connect()

drone.take_off()

time.sleep(3)

drone.land()

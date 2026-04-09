from pysimverse import Drone

drone = Drone()
drone.connect()

drone.take_off()

drone.set_speed(50)
drone.move_forward(150)

drone.rotate(192)  # do not understand why drone.rotate(180) is not rotating 180 degrees

drone.move_forward(150)

drone.land()

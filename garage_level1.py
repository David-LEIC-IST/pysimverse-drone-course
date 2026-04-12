from pysimverse import Drone

drone = Drone()
drone.connect()

drone.take_off(1)

drone.set_speed(100)
drone.rotate(50)
drone.move_forward(300)


drone.land()



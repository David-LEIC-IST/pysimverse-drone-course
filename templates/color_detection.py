from cvzone.ColorModule import ColorFinder
import cv2
import cvzone
from pysimverse import Drone

# Create an instance of the ColorFinder class with trackBar set to False.
myColorFinder = ColorFinder(trackBar=False)

drone = Drone()
drone.connect()
drone.streamon()
drone.take_off()

# Custom color values for detecting orange.
# 'hmin', 'smin', 'vmin' are the minimum values for Hue, Saturation, and Value.
# 'hmax', 'smax', 'vmax' are the maximum values for Hue, Saturation, and Value.
hsvVals = {'hmin': 40, 'smin': 42, 'vmin': 69, 'hmax': 52, 'smax': 139, 'vmax': 255}

# Main loop: drone camera feed (same pattern as drone_video_stream.py).
while True:
    img, is_success = drone.get_frame()
    if not is_success or img is None or img.size == 0:
        continue

    # Use the update method from the ColorFinder class to detect the color.
    # It returns the masked color image and a binary mask.
    imgOrange, mask = myColorFinder.update(img, hsvVals)

    # Stack the original image, the masked color image, and the binary mask.
    imgStack = cvzone.stackImages([img, imgOrange, mask], 3, 1)

    # Show the stacked images.
    cv2.imshow("Image Stack", imgStack)

    # Break the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
drone.streamoff()
drone.land()

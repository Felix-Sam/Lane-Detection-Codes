import cv2
import numpy as np
from scipy import ndimage
from tensorflow import keras

# Load the trained model
model = keras.models.load_model(r'C:\Users\felix\PycharmProjects\TECHWATT\files\model.h5')


class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    small_image = ndimage.zoom(image, (80 / image.shape[0], 160 / image.shape[1], 1))
    small_image = np.array(small_image)
    small_image = small_image[None, :, :, :]
    prediction = model.predict(small_image)[0] * 255
    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
    blank = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lanedrawn = np.dstack((blank, lanes.avg_fit, blank))
    lane_image = ndimage.zoom(lanedrawn, (720 / lanedrawn.shape[0], 1280 / lanedrawn.shape[1], 1))

    # Convert lane_image to the same data type as the image
    lane_image = lane_image.astype(image.dtype)
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result


lanes = Lanes()
video_input = r'C:\Users\felix\PycharmProjects\TECHWATT\files\lanes_clip.mp4'
# Open a connection to the camera (0 is usually the built-in camera)
cap = cv2.VideoCapture(video_input)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = road_lines(frame)

    # Display the processed frame
    cv2.imshow('Lane Detection', processed_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

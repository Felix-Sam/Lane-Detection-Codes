import cv2
import numpy as np
from scipy import ndimage
from moviepy.editor import VideoFileClip
from tensorflow import keras

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

    # Ensure both image and lane_image have the same shape
    lane_image_resized = cv2.resize(lane_image, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(image, 1, lane_image_resized, 1, 0)
    return result


lanes = Lanes()
video_input = VideoFileClip(r'C:\Users\felix\PycharmProjects\TECHWATT\files\roadvid.mp4')
video_output = 'lanes_vid_out.mp4'

video_clip = video_input.fl_image(road_lines)
video_clip.write_videofile(video_output, audio=False)  # Disable audio to avoid audio-related issues

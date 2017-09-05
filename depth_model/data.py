import pickle
from sync_frames import Frame
import cv2
from scipy.misc import imread
import numpy as np

# main_path = "/storage/nyu_v2/"
main_path = "/Users/imac05/Desktop/nyu_v2_examples/"


def load_train_data():
    with open("all_frames_list.pkl", 'rb') as f:
        data = pickle.load(f)
        print(str(len(data)))

    depth_images = []
    rgb_images = []
    # do i need accelerometer files?
    for frame in data:
        image = cv2.imread(frame.rgb_filename)
        rgb_images.append(image)
        depth = cv2.imread(frame.depth_filename, flags=cv2.IMREAD_GRAYSCALE)
        # depth = cv2.
        depth_images.append(depth)

    return rgb_images, depth_images


def load_test_data():
    return [], []

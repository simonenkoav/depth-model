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
    i = 0
    for frame in data:
        i += 1
        if i % 10000 == 0:
            print("Data in process: frame " + str(i))
        image = cv2.imread(frame.rgb_filename)
        rgb_images.append(image)
        depth = cv2.imread(frame.depth_filename, flags=cv2.IMREAD_GRAYSCALE)
        # depth = cv2.
        depth_images.append(depth)

    return rgb_images, depth_images


def load_test_data():
    return [], []


def load_train_filenames():
    train_data = []
    # with open("my_all_frames_list.pkl", 'rb') as f:
    with open("all_frames_list.pkl", 'rb') as f:
        data = pickle.load(f)

    print(len(data))

    for d in data:
        train_sample = {"image": d.rgb_filename, "depth": d.depth_filename}
        train_data.append(train_sample)

    return train_data


def load_not_none():
    with open("not_none_data.pkl", 'rb') as f:
        data = pickle.load(f)

    return data


import cv2
import numpy as np

img_rows = 96
img_cols = 96

delta_x = int(img_rows/2)
delta_y = int(img_cols/2)

def preprocess(image):
    image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    x_center = int(image.shape[0] / 2)
    y_center = int(image.shape[1] / 2)
    image = image[x_center - delta_x: x_center + delta_x,
                y_center - delta_y: y_center + delta_y]
    return image


def read_rgb_image(filename):
    image = cv2.imread(filename)
    image = preprocess(image)
    image = np.true_divide(image, 127.5)
    image -= 1
    return image


def read_depth_image(filename):
    image = cv2.imread(filename)
    image = preprocess(image)
    normal_depth = image[:, :, 0]
    normal_depth = normal_depth[:, :, np.newaxis]
    normal_depth = np.true_divide(normal_depth, 127.5)
    normal_depth -= 1
    return normal_depth


def read_sample(img_filenames):
    image = read_rgb_image(img_filenames["image"])
    depth = read_depth_image(img_filenames["depth"])
    return image, depth


def reverse_depth_image(depth_output):
    depth_output += 1
    depth_output *= 127.5
    depth_output = np.repeat(depth_output, 3, axis=2)
    return depth_output
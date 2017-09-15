import pickle
import cv2
from data import load_test_filenames

data = load_test_filenames()


def is_image_none(img_filename):
    img = cv2.imread(img_filename)
    return img is None


def is_sample_none(sample):
    return is_image_none(sample["image"]) or is_image_none(sample["depth"])


not_none = []

for d in data:
    if not is_sample_none(d):
        not_none.append(d)


with open("test_not_none_data.pkl", 'wb') as f:
    pickle.dump(not_none, f, pickle.HIGHEST_PROTOCOL)


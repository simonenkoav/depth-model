import pickle
import cv2
from data import load_train_filenames

train_data = load_train_filenames()


def is_image_none(img_filename):
    img = cv2.imread(img_filename)
    return img is None


def is_sample_none(sample):
    return is_image_none(sample["image"]) or is_image_none(sample["depth"])


not_none = []

for d in train_data:
    if not is_sample_none(d):
        not_none.append(d)


with open("not_none_data.pkl", 'wb') as f:
    pickle.dump(not_none, f, pickle.HIGHEST_PROTOCOL)

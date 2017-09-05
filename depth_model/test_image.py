import cv2
import pickle
import os
from sync_frames import Frame

# image = cv2.imread("/Users/imac05/Desktop/hand/dataset/train/rgb_2_0001234.png")
# print("image = " + str(image.shape))

path = "/Users/imac05/Workspace/work/depth_model/frame_lists/"
files = os.listdir(path)

print("files[0] = " + str(files[0]))

with open(path + files[0], 'rb') as f:
    data = pickle.load(f)

print(str(len(data)))

for i in range(len(data)):
    print(str(data[i].time_diff))
    print(str(data[i].depth_filename))
    print(str(data[i].rgb_filename))
    print(str(data[i].accel_filename))
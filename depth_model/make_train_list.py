import os
import pickle
import sys
from sync_frames import Frame

# path = "/Users/imac05/Workspace/work/depth_model/frame_lists/"
assert (len(sys.argv) == 2)
path = sys.argv[1]

files = os.listdir(path)
files_path = "files/"
frames_list = []

for file in files:
    with open(path + file, 'rb') as f:
        data = pickle.load(f)
        frames_list = frames_list + data

    with open(files_path + "test_frames_list.pkl", 'wb') as f:
        pickle.dump(frames_list, f, pickle.HIGHEST_PROTOCOL)

# with open("all_frames_list.pkl", 'rb') as f:
#     data = pickle.load(f)
#     print(str(len(data)))
#     for i in range(10):
#         print("____ I = " + str(i) + " ____")
#         print("depth_filename = " + str(data[i].depth_filename))
#         print("rgb_filename = " + str(data[i].rgb_filename))
#         print("accel_filename = " + str(data[i].accel_filename))


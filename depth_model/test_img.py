import cv2
import numpy as np

img = cv2.imread("/Users/imac05/Desktop/nyu_v2_examples/path1/d-1294886893.419866-627792608.pgm")
normal_depth = img[:, :, np.newaxis]

print(str(normal_depth.shape))

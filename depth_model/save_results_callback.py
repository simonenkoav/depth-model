import cv2
import random
from keras.callbacks import Callback
from image_processing import read_sample, reverse_depth_image, preprocess
from make_net import get_unet
import os
import glob
import numpy as np
import shutil
from operator import itemgetter 


class SaveResults(Callback):
    train_data = []
    results_dir = ""
    weights_dir = ""
    
    EPOCH_DIR_NAME = "epoch_"
    
    def __init__(self, tr_data, res_directory, weights_directory):
        self.train_data = tr_data
        self.results_dir = res_directory
        self.weights_dir = weights_directory
        
     
    def make_epoch_dir(self, epoch):
        dir_name = self.results_dir + self.EPOCH_DIR_NAME + str(epoch) + "/"
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        return dir_name
    
    
    def get_epoch_weights(self, epoch):
        weight_re = "./" + self.weights_dir + "weights.0" + str(epoch) + "*"
        weights_file = glob.glob(weight_re)[0]
        return weights_file
        
        
    def save_images(self, rgb_img, depth_img, predicted_depth_img, index, epoch, epoch_dir):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cur_dir + "/" + epoch_dir)
        cv2.imwrite('rgb' + str(index) + '.ppm', rgb_img)
        cv2.imwrite('real_depth' + str(index) + '.pgm', depth_img)
        cv2.imwrite('predicted_depth' + str(index) + '.pgm', predicted_depth_img)
        os.chdir(cur_dir)
        
    
    def on_epoch_end(self, epoch, logs={}):
        some_indexes = random.sample(range(0, len(self.train_data)), 3)
        for_save_samples = list(itemgetter(*some_indexes)(self.train_data))
        
        model = get_unet()
        model.load_weights(self.get_epoch_weights(epoch))
        
        epoch_dir = self.make_epoch_dir(epoch)
        
        for i in range(len(for_save_samples)):
            sample = for_save_samples[i]
            rgb_img, depth_img = read_sample(sample)
            depth_img = reverse_depth_image(depth_img)
            rgb_img = rgb_img[np.newaxis, :, :, :]
            predicted_depth_img = reverse_depth_image(model.predict(rgb_img)[0])
            rgb_img = cv2.imread(sample["image"])
            rgb_img = preprocess(rgb_img)
            self.save_images(rgb_img, depth_img, predicted_depth_img, i, epoch, epoch_dir)
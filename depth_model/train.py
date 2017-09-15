import os
from sys import stdout
import shutil

import numpy as np
from random import shuffle

import cv2

from keras.callbacks import ModelCheckpoint
from make_net import get_unet
from image_processing import read_sample, reverse_depth_image, img_rows, img_cols
from save_results_callback import SaveResults

from data import load_test_not_none, load_not_none


model_filename = "model.h5"


def std_print(str):
    stdout.write(str + "\n")
    stdout.flush()


def image_generator(data, read_sample, shuffle=False):
    if shuffle:
        np.random.shuffle(data)
    for img_filenames in data:
        img_real, depth_real = read_sample(img_filenames)
        yield img_real, depth_real


def batch_generator(img_generator, batch_size=32):
    while True:
        cur_batch_x = []
        cur_batch_y = []
        img_gen = img_generator()
        for image, depth in img_gen:
            cur_batch_x.append(image)
            cur_batch_y.append(depth)
            if len(cur_batch_x) == batch_size:
                yield (np.array(cur_batch_x), np.array(cur_batch_y))
                cur_batch_x = []
                cur_batch_y = []
                
                
def make_logs_dirs():
    results_dir = "epochs_results/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    
    checkpoint_dir = "snapshot/"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    
    return results_dir, checkpoint_dir
      
        
        
def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    train_data = load_not_none()
    test_data = load_test_not_none()
    shuffle(test_data)
    test_data = test_data[:1000]

    img_generator = lambda: image_generator(train_data, read_sample, shuffle=True)
    train_generator = batch_generator(img_generator, 8)
    
    test_img_gen = lambda: image_generator(test_data, read_sample, shuffle=True)
    test_generator = batch_generator(test_img_gen, 8)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    print(str(model.summary()))

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    
    results_dir, checkpoint_dir = make_logs_dirs()
  
    checkpoint = ModelCheckpoint(checkpoint_dir + '/weights.{epoch:02d}-loss_{loss:.3f}.hdf5', monitor='loss', verbose=0,
                                 save_best_only=False, mode='auto')
    
    save_results = SaveResults(train_data, results_dir, checkpoint_dir)
    
    model.fit_generator(generator=train_generator, steps_per_epoch=11300, verbose=1, epochs=10, 
                        callbacks=[checkpoint, save_results], validation_data=test_generator, validation_steps=125)
    model.save(model_filename)


def predict():
    model = load_model(model_filename)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    test_filenames = load_test_not_none()

    shuffle(test_filenames)

    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    real_dir = 'real'
    if not os.path.exists(real_dir):
        os.mkdir(real_dir)

    for i in range(10):
        print("test[" + str(i) + "]")
        print("rgb = " + str(test_filenames[i]["image"]))
        print("depth = " + str(test_filenames[i]["depth"]))
        rgb_img = read_rgb_image(test_filenames[i]["image"])
        depth_img = cv2.imread(test_filenames[i]["depth"])
        depth_img = preprocess(depth_img)
        cv2.imwrite(real_dir + '/real_depth' + str(i) + '.pgm', depth_img)
        rgb_img = rgb_img[np.newaxis, :, :, :]
        predicted_depth = reverse_depth_image(model.predict(rgb_img)[0])
        cv2.imwrite(pred_dir + '/predicted_depth' + str(i) + '.pgm', predicted_depth)


if __name__ == '__main__':
    train()
    # predict()

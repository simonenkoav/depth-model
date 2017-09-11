from __future__ import print_function

import os
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

from data import load_train_filenames, load_test_data

from sys import stdout

import cv2

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96

delta_x = int(img_rows/2)
delta_y = int(img_cols/2)

smooth = 1.

mean = 0
std = 0

model_filename = "model.h5"


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate(inputs=[Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate(inputs=[Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate(inputs=[Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate(inputs=[Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def std_print(str):
    stdout.write(str + "\n")
    stdout.flush()


def preprocess(image):
    image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    x_center = int(image.shape[0] / 2)
    y_center = int(image.shape[1] / 2)
    image = image[x_center - delta_x: x_center + delta_x,
                y_center - delta_y: y_center + delta_y]
    return image


def read_rgb_image(filename):
    image = cv2.imread(filename)
    if image is None:
        std_print("None image = " + str(filename))
    image = preprocess(image)
    return image


def read_depth_image(filename):
    image = cv2.imread(filename)
    if image is None:
        std_print("None image = " + str(filename))
    image = preprocess(image)
    normal_depth = image[:, :, 0]
    normal_depth = normal_depth[:, :, np.newaxis]
    return normal_depth


def read_sample(img_filenames):
    image = read_rgb_image(img_filenames["image"])
    depth = read_depth_image(img_filenames["depth"])
    return image, depth


def image_generator(data, read_sample, shuffle=False):
    if shuffle:
        np.random.shuffle(data)
    for img_filenames in data:
        img_real, depth_real = read_sample(img_filenames)
        yield img_real, depth_real


def batch_generator(img_generator, batch_size=32):
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


def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    train_data = load_train_filenames()

    img_generator = lambda: image_generator(train_data, read_sample, shuffle=True)
    # train_generator = batch_generator(img_generator, 38)
    train_generator = batch_generator(img_generator, 4)

    for val in train_generator:
        std_print("val = " + str(val))
    quit()
    # train_generator = batch_generator(img_generator, 2)

    # imgs = np.array(preprocess(imgs))
    # depths = np.array(preprocess(depths))

    # depths = get_normal_depth(depths)

    # imgs = imgs.astype('float32')
    # mean = np.mean(imgs)  # mean for data centering
    # std = np.std(imgs)  # std for data normalization

    # imgs -= mean
    # imgs /= std

    # depths = depths.astype('float32')
    # depths /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    print(str(model.summary()))
    # model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    # model.fit(imgs, depths, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
    #           validation_split=0.2,
    #           callbacks=[model_checkpoint])

    # model.fit_generator(train_generator, 2394, verbose=1)
    model.fit_generator(train_generator, 3, verbose=1)
    model.save(model_filename)


def predict():
    model = load_model(model_filename)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_filenames()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    # train_and_predict()
    train()
    # predict()

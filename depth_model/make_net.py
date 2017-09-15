from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.models import load_model
from image_processing import img_rows, img_cols
from instance_normalization import InstanceNormalization


def get_unet():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = InstanceNormalization()(conv1)
    conv1_ = Conv2D(32, (3, 3), activation='relu', strides=(2,2), padding='same')(conv1)
    conv1_ = InstanceNormalization()(conv1_)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_)
    conv2 = InstanceNormalization()(conv2)
    conv2_ = Conv2D(64, (3, 3), activation='relu', strides=(2,2), padding='same')(conv2)
    conv2_ = InstanceNormalization()(conv2_)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_)
    conv3 = InstanceNormalization()(conv3)
    conv3_ = Conv2D(128, (3, 3), activation='relu', strides=(2,2), padding='same')(conv3)
    conv3_ = InstanceNormalization()(conv3_)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_)
    conv4 = InstanceNormalization()(conv4)
    conv4_ = Conv2D(256, (3, 3), activation='relu', strides=(2,2), padding='same')(conv4)
    conv4_ = InstanceNormalization()(conv4_)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = InstanceNormalization()(conv5)

    up6 = concatenate(inputs=[Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = InstanceNormalization()(conv6)

    up7 = concatenate(inputs=[Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = InstanceNormalization()(conv7)

    up8 = concatenate(inputs=[Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = concatenate(inputs=[Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = InstanceNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='tanh')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='mse')

    return model
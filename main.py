import os
import cv2
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import tensorflow as tf

import numpy as np
from keras.backend import int_shape
from keras.models import Model
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate



import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from keras.utils import plot_model

im_width = 768
im_height = 496
channels = 1
border = 5

train_img = './Aug_Dataset/train/images/'
train_mask = './Aug_Dataset/train/mask/'

val_img = './Aug_Dataset/val/images/'
val_mask = './Aug_Dataset/val/mask/'

# val = './data/'

train_imgs = os.listdir(train_img)
train_masks = os.listdir(train_mask)

val_imgs = os.listdir(val_img)
val_masks = os.listdir(val_mask)
data = os.listdir(val_img)
# print(len(imgs))

X_train = np.zeros((len(train_imgs), im_height, im_width, 1), dtype=np.float32)
y_train = np.zeros((len(train_imgs), im_height, im_width, 1), dtype=np.float32)


X_val = np.zeros((len(val_imgs), im_height, im_width, 1), dtype=np.float32)
y_val = np.zeros((len(val_imgs), im_height, im_width, 1), dtype=np.float32)

print(len(y_train), len(y_val))


def iou(y_true, y_pred):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    print(intersection, type(intersection))
    print(union, type(union))
    return 1. * intersection / union


def dice_coef(y_true, y_pred, eps=1e-3):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + eps) / (K.sum(y_true) + K.sum(y_pred) + eps)


for i, img in enumerate(train_imgs):

    img = load_img(train_img+img, color_mode='grayscale')
    x_img = img_to_array(img)
    # x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    # # Load masks
    # mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
    # # Save images
    X_train[i] = x_img/255



for i, msk in enumerate(train_masks):

    mask = load_img(train_mask+msk, color_mode='grayscale')
    y = img_to_array(mask)
    y_train[i] = y/255

for i, img in enumerate(val_imgs):
    # print(val+img)
    img = load_img(val_img+img, grayscale=True)
    x_img = img_to_array(img)
    # x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    # # Load masks
    # mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
    # # Save images
    X_val[i] = x_img/255


for i, msk in enumerate(val_masks):
    mask = load_img(val_mask + msk, color_mode='grayscale')
    y = img_to_array(mask)
    y_val[i] = y/255


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x



def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model"""

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
# model.load_weights('./experiments/model1_50.h5')
model.compile(optimizer=Adam(lr=0.0002), loss="binary_crossentropy", metrics=["accuracy", iou, dice_coef])
model.summary()

results = model.fit(X_train, y_train, batch_size=4, epochs=100, validation_data=(X_val, y_val))

preds_val = model.predict(X_val, batch_size=4)
preds_val = (preds_val > 0.5).astype(np.uint8)
preds_val_t = preds_val*255


id = random.randint(0, len(X_val))

print(len(X_val), len(preds_val_t))
for i in range(len(X_val)):
    print('./overlays/overlay' + str(i) + '.png')
    img = array_to_img(X_val[i])
    main = np.array(img)
    mask = array_to_img(preds_val_t[i])
    # cv2.imwrite('./overlays/mask' + str(i) + '.png')
    seg = np.array(mask)
    # print(type(main), type(seg))
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
    # main.show()
    # seg.show()

    print(seg.shape)
    # sys.exit(0)
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    RGBforLabel = {1: (0, 0, 255), 2: (0, 255, 255)}

    for i,c in enumerate(contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(seg.shape, np.uint8)
        cv2.drawContours(mask,[c],-1,255, -1)
        # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
        mean, _, _, _ = cv2.mean(seg, mask=mask)
        # DEBUG: print(f"i: {i}, mean: {mean}")

        # Get appropriate colour for this label
        label = 2 if mean > 1.0 else 1
        colour = RGBforLabel.get(label)
        # DEBUG: print(f"Colour: {colour}")

        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(main, [c], -1, colour, 1)

    cv2.imwrite('./overlays/overlay' + str(i) + '.png', main)
    cv2.imwrite('./overlays/mask'+str(i)+'.png', main)

for i in range(len(preds_val_t)):
    img = array_to_img(preds_val_t[i])
    save_img('./graders/prediction/'+data[i], preds_val[i], scale=True)

model.save_weights('./experiments/model2_100.h5')


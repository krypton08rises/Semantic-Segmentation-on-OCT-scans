import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# %matplotlib inline

from PIL import Image
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img

im_width = 768
im_height = 496
channels = 1
border = 5

train_img = './finalDataset/train/images/'
train_mask = './finalDataset/train/mask/'

val_img = './finalDataset/val/images/'
val_mask = './finalDataset/val/mask/'

train_imgs = os.listdir(train_img)
train_masks = os.listdir(train_mask)

val_imgs = os.listdir(val_img)
val_masks = os.listdir(val_mask)
# print(len(imgs))

X_train = np.zeros((len(train_imgs), im_height, im_width, 1), dtype=np.float32)
y_train = np.zeros((len(train_imgs), im_height, im_width, 1), dtype=np.float32)

X_val = np.zeros((len(val_imgs), im_height, im_width, 1), dtype=np.float32)
y_val = np.zeros((len(val_imgs), im_height, im_width, 1), dtype=np.float32)

# tqdm is used to display the progress bar
# for n, id_ in tqdm_notebook(enumerate(imgs), total=len(imgs)):
    # Load images

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


# Visualize any random image along with the mask
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0 # salt indicator
has_mask = has_mask*255
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

ax1.imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask: # if salt
    # draw a boundary(contour) in the original image separating salt and non-salt areas
    ax1.contour(y_train[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
ax1.set_title('Seismic')

ax2.imshow(y_train[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
ax2.set_title('Salt')
# print('reached here!')
# time.sleep(10)


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
model.load_weights('./model1.h5')
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model1_40.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
#
#
results = model.fit(X_train, y_train, batch_size=4, epochs=20, callbacks=callbacks,
                    validation_data=(X_val, y_val))

preds_val = model.predict(X_val, batch_size=4)
preds_val_t = preds_val*255


id = random.randint(0, len(X_val))
img = array_to_img(X_val[id])
mask = array_to_img(preds_val_t[id])
img.show()
mask.show()

for i in range(len(preds_val_t)):
    img = array_to_img(preds_val_t[i])
    save_img('./predictions/'+str(i)+'.png', preds_val[i], scale=True)

model.save_weights('./model1_40.h5')


def plot_sample(X, y, preds, binary_preds, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Salt Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Predicted binary');



plot_sample(X_val, y_val, preds_val, preds_val_t, ix=14)
import os
import scipy.io
import numpy as np
import cv2 as cv
from PIL import Image

path = './2015_BOE_Chiu/'
imgs ='./finalDataset/images/'
mask = './finalDataset/mask/'

patient = 0
for mat_file in os.listdir(path):
    mat_file = path+mat_file
    mat = scipy.io.loadmat(mat_file)
    final_mat = {key: mat[key] for key in mat if key not in ['__header__', '__version__', '__globals__']}
    # print("looking at file ", j)
    for key, val in final_mat.items():
        print(key, "corresponds to ", np.shape(val))

    images = final_mat['images']
    manual_fluid1 = final_mat['manualFluid1']
    manual_fluid2 = final_mat['manualFluid2']
    # automaticFluidDME = final_mat['automaticFluidDME']

    # for i in range(61):
    #     cv.imshow('image', images[:, :, i])
    #     cv.imshow('fluid', automaticFluidDME[:,:,i])
    #     cv.waitKey(0)
    # 'images' has 61 gray level images with dimension 496, 768
    # break
    j=0
    _, _, total_images = np.shape(images)
    print('Total number of images in file: ', total_images)
    for i in range(total_images):

        img0 = images[:, :, i]
        img1 = manual_fluid1[:, :, i]
        img2 = manual_fluid2[:, :, i]
        if np.isnan(np.sum(img1))==True:
            continue
        for col in range(768):
            manual_fluid1[:, col, i] = [255 if x > 0 else 0 for x in manual_fluid1[:, col, i]]


        for col in range(768):
            manual_fluid2[:, col, i] = [255 if x > 0 else 0 for x in manual_fluid2[:, col, i]]
        # print(imgs + 'subject' + str(patient) + 'image' + str(j) + '.png')

        cv.imwrite(imgs + 'patient' + str(patient) + '_image' + str(j) + '.png', img0)
        cv.imwrite(mask + 'patient' + str(patient) + '_image' + str(j) + '.png', img1)
        j += 1
        cv.imwrite(imgs + 'patient' + str(patient) + '_image' + str(j) + '.png', img0)
        cv.imwrite(mask + 'patient' + str(patient) + '_image' + str(j) + '.png', img2)
        j += 1
    patient += 1


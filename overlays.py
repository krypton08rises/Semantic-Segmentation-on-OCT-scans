import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import color
from skimage import segmentation


images = './data/'
grad0 = './graders/grader0/'
grad1 = './graders/grader1/'
pred = './graders/predictions/'

output = './overlays/grad1_pred/'
outgrad0 = './graders/grader0_overlays/'
outgrad1 = './graders/grader1_overlays/'

for img in os.listdir(images):

    seg0 = cv2.imread(grad0 + img, 0)
    seg1 = cv2.imread(grad1 + img, 0)
    main = cv2.imread(images + img, 0)
    seg = cv2.imread(pred + img, 0)


    # print(main.shape)
    # seg = io.imread(pred+img, as_gray=True)
    # image = io.imread(images + img, as_gray=True)

    # seg = cv2.imread('segmented.png', cv2.IMREAD_GRAYSCALE)
    # main = cv2.imread('main.png', cv2.IMREAD_GRAYSCALE)
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
    # image2 = np.zeros((main.shape))

    # cv2.imshow('image1', main)
    # cv2.waitKey(0)

    # Dictionary giving RGB colour for label (segment label) - label 1 in red, label 2 in yellow
    RGBforLabel = {1: (0, 0, 255), 2: (0, 255, 0), 3:(255, 0, 255)}

    # Find external contours
    contours, _ = cv2.findContours(seg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate over all contours
    for i, c in enumerate(contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(seg1.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
        mean, _, _, _ = cv2.mean(seg1, mask=mask)
        # DEBUG: print(f"i: {i}, mean: {mean}")

        # Get appropriate colour for this label
        label = 2 if mean > 1.0 else 1
        colour = RGBforLabel.get(label)
        # DEBUG: print(f"Colour: {colour}")

        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(main, [c], -1, colour, -1)

    # main = cv2.addWeighted(main, 0.8, image1, 0.2, 0)
    # cv2.imshow('image1', main)
    # cv2.waitKey(0)
    # Save result
    cv2.imwrite(outgrad1+img, main)


    # contours, _ = cv2.findContours(seg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for i, c in enumerate(contours):
    #     # Find mean colour inside this contour by doing a masked mean
    #     mask = np.zeros(seg1.shape, np.uint8)
    #     cv2.drawContours(mask, [c], -1, 255, -1)
    #     # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
    #     mean, _, _, _ = cv2.mean(seg1, mask=mask)
    #     # DEBUG: print(f"i: {i}, mean: {mean}")
    #
    #     # Get appropriate colour for this label
    #     if mean > 1.0:
    #         label = 3
    #     colour = RGBforLabel.get(label)
    #     # DEBUG: print(f"Colour: {colour}")
    #
    #     # Outline contour in that colour on main image, line thickness=1
    #
    #     cv2.drawContours(main, [c], -1, colour, 1)

    cv2.imwrite(output+img, main)

    # cv2.imshow('image2', main)
    # cv2.waitKey(0)

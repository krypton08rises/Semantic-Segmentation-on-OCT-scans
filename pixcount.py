import os
import cv2
import numpy as np
import pandas as pd
from grader_check import getIOU

path = './graders/'
# grader0 = './graders/grader0/'
# grader1 = './graders/grader1/'
# predictions = './graders/pred/'
df = pd.DataFrame(columns=['image',
                           'pix_count_grader0',
                           'pix_count_grader1',
                           'pix_count_predictions',
                           'iou_graders',
                           'iou_grader0',
                           'iou_grader1'])

imgs = []
for img in os.listdir('./data/'):
    imgs.append(img)

# pix_grad0 = np.zeros((len(os.listdir(grader0))))
# pix_grad1 = np.zeros((len(os.listdir(grader1))))
# pix_pred = np.zeros((len(os.listdir(predictions))))

def count_pixels():
    pixels = []
    for files in os.listdir(path):
        file = path + files + '/'
        a = []
        for img in os.listdir(file):
            img = file + img
            print(img)
            image = cv2.imread(img, 0)
            a.append(np.sum(image)/255)
        pixels.append(a)
    return pixels

if __name__=='__main__':
    count = count_pixels()
    # print(count[2])
    df['image'] = imgs
    df['pix_count_grader0'] = count[0]
    df['pix_count_grader1'] = count[1]
    df['pix_count_predictions'] = count[2]

    iou_grader0, iou_grader1, iou_graders = getIOU()
    df['iou_graders'] = iou_graders
    df['iou_grader1'] = iou_grader1
    df['iou_grader0'] = iou_grader0
    print(df.head())
    df.to_csv('pix_count.csv')
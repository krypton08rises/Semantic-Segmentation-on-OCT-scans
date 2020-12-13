import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
# from keras.preprocessing.image import img_to_array, load_img

im_width = 768
im_height = 496
channels = 1

predictions = './graders/predictions/'
grader0 = './graders/grader0/'
grader1 = './graders/grader1/'

dataset = './UNET_dataset/images/'
imgs = os.listdir(grader1)


def iou(y_true, y_pred):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    # print(intersection)
    # print(union)
    return 1. * intersection / union


def dice_coef(y_true, y_pred, eps=1e-3):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + eps) / (np.sum(y_true) + np.sum(y_pred) + eps)


def box_plot(x, y):

    dice = []
    ioU = []
    df = pd.DataFrame(columns=['iou', 'dice_score', 'q1', 'q2', 'q3'])
    countds = 0
    for i in range(len(x)):
        if math.isnan(iou(x[i], y[i])) or (np.sum(x[i])<1000 and np.sum(y[i])<1000):
            continue
            # nan images are both unannotated

        a = iou(x[i], y[i])
        b = dice_coef(x[i], y[i])
        if a!=0 and b!=0:
            ioU.append(a)
            dice.append(b)
            countds+=1
        # print('Iou is :', ioU[i])
        # print('Dice Coefficient is :', dice[i])
        # break
    print(np.sum(dice)/countds)

    jc = np.array(ioU)
    dice = np.array(dice)
    return dice, jc
    # df['iou']=jc
    # df['dice_score']=dice
    # print(df['iou'])


    # df['q2'] = df[df.iou >= q2[0]]
    # df['q3'] = df[df.iou >= q3[0]]
    # df_q1 = df.loc[:,'q1'].values
    # print(df_q1)
    # df_q2 = df.loc[:, 'q2'].values
    # df_q3 = df.loc[:, 'q3'].values
    # data = [iou, df_q1, df_q2, df_q3]
    # # # print(len(df_q1.index))
    #
    # # print(df_q1.head())
    # plt.show()



    # fig1 = plt.figure(figsize=(10, 7))

    # fig2 = plt.figure(figsize=(10, 7))
    # data = [ioU, dice]
    # plt.boxplot(data)
    # plt.show()




# if __name__ == '__main__':
def getIOU():

    x = np.zeros((len(imgs), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(imgs), im_height, im_width, 1), dtype=np.float32)
    pred = np.zeros((len(imgs), im_height, im_width, 1), dtype=np.float32)


    # dice = np.zeros(len(imgs))
    # ioU = np.zeros(len(imgs))

    # for i, img in enumerate(os.listdir(predictions)):
    #     img = Image.open(predictions + img)
    #     img = np.asarray(img)
    #     img = img.reshape(496, 768, 1)
    #     # print(img.shape)
    #     # img = load_img(predictions + img, color_mode='grayscale')
    #     # img = img_to_array(img)
    #     pred[i] = img/255.

    for i, img in enumerate(imgs):
        if img[:8]!="patient9":
            continue
        # print(img)
        img0 = Image.open(grader0 + img)
        img2 = Image.open(predictions + img)


        img2 = np.asarray(img2)
        img2 = img2.reshape(496, 768, 1)
        # img1 = load_img(grader0 + img, color_mode='grayscale')
        # img1.show()
        # sys.exit(0)
        x_img = np.asarray(img0)
        x_img = x_img.reshape(496, 768, 1)

        # x_img = img_to_array(img1)
        img1 = Image.open(grader1 + img)
        # img2 = load_img(grader1 + img, color_mode='grays  cale')
        y_img = np.asarray(img1)
        y_img = y_img.reshape(496, 768, 1)

        # y_img = img_to_array(img1)
        x[i] = x_img / 255
        y[i] = y_img / 255
        pred[i] = img2 / 255.

    iou_graders = []
    iou_grader0 = []
    iou_grader1 = []
    dice_score = []


    # for i in range(len(x)):
    #
    #     dice_score.append(dice_coef(x[i], y[i]))
    #     dice_score.append(dice_coef(x[i], y[i]))
    #     dice_score.append(dice_coef(x[i], y[i]))
    # print(dice_score)
    #
    #     iou_graders.append(iou(x[i], y[i]))
    #     iou_grader0.append(iou(pred[i], x[i]))
    #     iou_grader1.append(iou(y[i], pred[i]))
    jc_1, _ = box_plot(x, y)
    jc_2, _ = box_plot(x, pred)
    jc_3, _ = box_plot(pred, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Grader 1 and 2
    q1 = np.quantile(jc_1, 0.25)
    q2 = np.quantile(jc_1, 0.5)
    q3 = np.quantile(jc_1, 0.75)
    iou_q1 = jc_1[jc_1 > q1]
    iou_q2 = jc_1[jc_1 > q2]
    iou_q3 = jc_1[jc_1 > q3]

    print(q1, iou_q1)
    print(q2, iou_q2)
    print(q3, iou_q3)

    data1 = [jc_1, iou_q1, iou_q2, iou_q3]
    # plt.boxplot(data1)
    # plt.show()

#Grader 1 and predictions
    q1 = np.quantile(jc_2, 0.25)
    q2 = np.quantile(jc_2, 0.5)
    q3 = np.quantile(jc_2, 0.75)
    iou_q1 = jc_2[jc_2 > q1]
    iou_q2 = jc_2[jc_2 > q2]
    iou_q3 = jc_2[jc_2 > q3]

    print(q1, iou_q1)
    print(q2, iou_q2)
    print(q3, iou_q3)
    data2 = [jc_2, iou_q1, iou_q2, iou_q3]

## grader 2 and predictions:
    q1 = np.quantile(jc_3, 0.25)
    q2 = np.quantile(jc_3, 0.5)
    q3 = np.quantile(jc_3, 0.75)
    iou_q1 = jc_3[jc_3 > q1]
    iou_q2 = jc_3[jc_3 > q2]
    iou_q3 = jc_3[jc_3 > q3]

    print(q1, iou_q1)
    print(q2, iou_q2)
    print(q3, iou_q3)
    data3 = [jc_3, iou_q1, iou_q2, iou_q3]

    colors = ['#00FF00', '#00FF00', '#00FF00', '#00FF00',
              '#FFFF00', '#FFFF00', '#FFFF00', '#FFFF00',
              '#0000FF', '#0000FF', '#0000FF', '#0000FF']

    for arr in data2:
        data1.append(arr)
    for arr in data3:
        data1.append(arr)

    ax.set_yticklabels(['graders1&2', 'q2', 'q3', 'q4',
                        'grader1&model', 'q2', 'q3', 'q4',
                        'grader2&model', 'q2', 'q3', 'q4'])

    plt.xlim(0, 1)
    bp = ax.boxplot(data1, patch_artist=True, vert=0)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.title("Dice Score - Validation")
    plt.show()
    return iou_grader0, iou_grader1, iou_graders
    print(x.shape)

getIOU()


import os

import cv2
import numpy as np
from numpy import genfromtxt
from constants import SEQUENCE_PADDING
from constants import NUMBER_OF_KEYPOINTS
import csv
from matplotlib import pyplot as plt
from numpy import savetxt
from PIL import Image


def get_hands_csv(hands_list):
    hands = []
    if len(hands_list) == 1:
        hands.append(hands_list[0].get_timestep_data())
        zeros = [0] * 19 * 3
        hands.append(','.join(str(v) for v in zeros))
    elif len(hands_list) == 2:
        hands.append(hands_list[0].get_timestep_data())
        hands.append(hands_list[1].get_timestep_data())
    return ','.join(hands)


def load_data(number_of_classes, data_type):
    X_train = []
    Y_train = []
    for i in range(0, number_of_classes):
        all_files = os.listdir("data/" + data_type + "/" + str(i))
        for file in all_files:
            file_np_array = genfromtxt("data/" + data_type + "/" + str(i) + "/" + str(file), delimiter=',',
                                       usecols=np.arange(0, NUMBER_OF_KEYPOINTS)).tolist()
            X_train.append(file_np_array)
            Y_train.append(i)
    return X_train, np.array(Y_train, dtype='float64')


def convert_to_img(number_of_classes, data_type):
    X_train = []
    Y_train = []
    for i in range(0, number_of_classes):
        all_files = os.listdir("data/" + data_type + "/" + str(i))

        for file in all_files:
            arr = np.empty([50, 50], dtype='float64')
            file_np_array = genfromtxt("data/" + data_type + "/" + str(i) + "/" + str(file), delimiter=',',
                                       usecols=np.arange(0, NUMBER_OF_KEYPOINTS)).tolist()
            # X_train.append(file_np_array)
            j = 0
            while j < 114:
                if (j + 1) % 3 != 0:
                    data = np.zeros((50, 50), dtype=np.uint8)
                    for e in file_np_array:
                        x_pos = (int(e[j + 1] * 50)) + 25
                        y_pos = (int(e[j] * 50)) + 25
                        try:
                            if x_pos >= 50:
                                x_pos = 49
                            if y_pos >= 50:
                                y_pos = 49
                            data[x_pos, y_pos] = 1
                        except:
                            print("Could not save point " + str((int(e[j + 1] * 50)) + 25) + ", " + str(
                                int(e[j] * 50) + 25) + " for file: " + str(file))
                            print("Could not save point " + str((e[j + 1])) + ", " + str(
                                e[j]) + " for file: " + str(file))

                    plt.imshow(data, cmap='gray', vmin=0, vmax=1)
                    plt.show()
                    arr = np.dstack((arr, data))
                    j = j + 2
                else:
                    j = j + 1
            #   0,1   2   3,4   5   6,7   8   9,10   11   12,13
            arr = np.delete(arr, 0, axis=2)

            X_train.append(arr)
            Y_train.append(i)
    return X_train, np.array(Y_train, dtype='float64')


def convert_to_img_debug(image_size):
    arr = np.empty([80, 80], dtype='float64')
    file_np_array = genfromtxt("data/recordings/TEST2.csv", delimiter=',',
                               usecols=np.arange(0, NUMBER_OF_KEYPOINTS)).tolist()
    j = 0
    while j < 114:
        if (j + 1) % 3 != 0:
            data = np.zeros((80, 80), dtype=np.uint8)
            for e in file_np_array:
                try:
                    data[(int(e[j + 1] * 80)) + (40), (int(e[j] * 80)) + (40)] = 1
                except:
                    print("c")
                    # print("Could not save point " + str((int(e[j+1] * 80))) + ", " + str(int(e[j] * 80)) + " for file: " + str(file))
                    # print("Could not save point " + str((int(e[j+1]))) + ", " + str(int(e[j])) + " for file: " + str(file))

            # plt.imshow(data, cmap='gray', vmin=0, vmax=1)
            # plt.show()
            arr = np.dstack((arr, data))
            j = j + 2
        else:
            j = j + 1
    #   0,1   2   3,4   5   6,7   8   9,10   11   12,13
    arr = np.delete(arr, 0, axis=2)
    return arr


def pad_sequence(seq, maxlen):
    for j, s in enumerate(seq):
        for i in range(maxlen - len(seq[j])):
            seq[j] = np.insert(seq[j], 0, np.zeros(NUMBER_OF_KEYPOINTS), axis=0)
    return reshape_sequence(seq)


def reshape_sequence(seq):
    arr = np.empty([SEQUENCE_PADDING, NUMBER_OF_KEYPOINTS], dtype='float64')
    for i in seq:
        arr = np.dstack((arr, i))
    arr = np.swapaxes(arr, 0, 2)
    arr = np.swapaxes(arr, 1, 2)
    return np.delete(arr, 0, axis=0)

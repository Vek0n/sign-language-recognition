import os
import numpy as np
from numpy import genfromtxt
from constants import SEQUENCE_PADDING
from constants import NUMBER_OF_KEYPOINTS


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


import os


def delete_short_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if the file is a regular file
        if os.path.isfile(filepath):
            # Open the file and count the number of lines
            with open(filepath, 'r') as file:
                num_lines = sum(1 for line in file)

            # Delete the file if it has less than 10 lines
            if num_lines > 50 or num_lines < 10:
                os.remove(filepath)
                print(f"{filename} deleted because it had less than 10 lines.")


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

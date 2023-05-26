import cv2
import numpy as np
from os import listdir
from os import getcwd
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Conv3D, Dropout, Dense, MaxPooling2D, Flatten, Conv2D, Dense, Dropout, Conv1D, GlobalMaxPooling1D
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical

from training import record_sign
from numpy import genfromtxt
from constants import NUMBER_OF_KEYPOINTS
import os

from utils import convert_to_img_debug, convert_to_img


#0 dobry
#1 dziekuje
#2 czesc
#3 prosze
#4 widzieć


#
# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
#     # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs
#
#     # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res
#
#
#
# def build_model(
#     input_shape,
#     head_size,
#     num_heads,
#     ff_dim,
#     num_transformer_blocks,
#     mlp_units,
#     dropout=0,
#     mlp_dropout=0,
# ):
#     inputs = keras.Input(shape=input_shape)
#     x = inputs
#     for _ in range(num_transformer_blocks):
#         x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
#
#     x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
#     outputs = layers.Dense(5, activation="softmax")(x)
#     return keras.Model(inputs, outputs)
#
#
# # def load_data(type):
# #     x = []
# #     y = []
# #     for i in range(0,4):
# #         path = getcwd() + "/data/" + str(type) + "/"+str(i)
# #         for f in listdir(path):
# #             entry = np.loadtxt(path + "/" + f, delimiter=",")
# #             concatenated = np.array([])
# #             for e in entry:
# #                 concatenated = np.concatenate((concatenated, e))
# #             x.append(concatenated)
# #             y.append(i)
# #     return x, y
#
#
# def readucr(filename):
#     data = np.loadtxt(filename, delimiter="\t")
#     y = data[:, 0]
#     x = data[:, 1:]
#     return x, y.astype(int)
#
#
# def pad_sequence(seq, maxlen):
#     for j, s in enumerate(seq):
#         # for i in range(maxlen - len(seq[j])):
#         seq[j] = np.insert(seq[j], len(seq[j]), np.zeros(maxlen - len(seq[j])), axis=0)
#     return reshape_sequence(seq)
#
# def reshape_sequence(seq):
#     arr = np.empty([1, 3400], dtype='float64')
#     for i in seq:
#         arr = np.vstack((arr, i))
#     return np.delete(arr, 0, axis=0)


# def load_data(data_type):
#     X_train = []
#     Y_train = []
#     all_files = os.listdir("data/" + data_type + "/2")
#     for file in all_files:
#         file_np_array = genfromtxt("data/" + data_type + "/2/" + str(file), delimiter=',',
#                                    usecols=np.arange(0, NUMBER_OF_KEYPOINTS)).tolist()
#         X_train.append(file_np_array)
#     return X_train, np.array(Y_train, dtype='float64')


def load_temp_data():
    X_train = []
    Y_train = []
    file_np_array = genfromtxt("data/temp.csv", delimiter=',',
                               usecols=np.arange(0, NUMBER_OF_KEYPOINTS)).tolist()
    X_train.append(file_np_array)
    return X_train, np.array(Y_train, dtype='float64')

def read_data(base_path):
    X = []
    y = []

    for class_number in range(4):
        class_path = os.path.join(base_path, str(class_number))
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)

            with open(file_path, 'r') as file:
                time_series = file.readline().strip().split(',')
                time_series = [float(val) for val in time_series]
                X.append(time_series)
                y.append(class_number)

    return np.array(X), np.array(y)

if __name__ == '__main__':
    X_train, y_train = read_data('data/train')
    X_test, y_test = read_data('data/test')

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Convert labels to categorical data
    y_train = to_categorical(y_train, num_classes=16)
    y_test = to_categorical(y_test, num_classes=16)

    # Build the 1D-CNN model
    model = models.Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=25)

    # Evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("Training Accuracy: %.2f%%" % (train_acc * 100))
    print("Test Accuracy: %.2f%%" % (test_acc * 100))





















    # data = np.zeros((80, 80), dtype=np.uint8)
    # x_tr, y_tr = load_data(5, "train")
    #
    # for example in x_tr:
    #     for e in example:
    #         data[(int(e[1] * 80) + 40), (int(e[0] * 80)) + 40] = 1
    #     plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    #     plt.show()
    #     data = np.zeros((80, 80), dtype=np.uint8)

    # x_train, y_train = convert_to_img(6, "train")
    # x_trainn = tf.stack(x_train)
    # y_train = to_categorical(y_train)
    # y_trainn = tf.stack(y_train)
    # opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    #
    # s = 1
    # model = models.Sequential()
    # model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(30, 30, 38)))
    # # model.add(MaxPooling2D((2, 2)))
    # # model.add(Conv2D(16, (5, 5), activation='relu'))
    # model.add(Flatten())
    # # model.add(Dense(16, activation='relu'))
    # model.add(Dense(6))
    # model.compile(optimizer=opt,
    #               loss="categorical_crossentropy",
    #               metrics=['accuracy'])
    #
    # history = model.fit(x_trainn, y_trainn, epochs=50, batch_size=10)

    # for j in range(10):
    #     xx = x_train[0][:, :, j].squeeze()
    #     xx = np.flip(xx, 1)
    #     rescaled = (255.0 / xx.max() * (xx - xx.min())).astype(np.uint8)
        # im = Image.fromarray(rescaled)
        # im.save('test.png')
        # cv2.imwrite(str(j) + ".png", rescaled)
        # plt.imshow(xx, cmap='gray', vmin=0, vmax=1)
        # plt.show()







    # x_tr, y_tr = load_data("train")
    # x_tst, y_tst = load_data("test")
    #
    # x_train = pad_sequence(x_tr, 3400)
    # x_test = pad_sequence(x_tst, 3400)
    # y_train = np.array(y_tr)
    # y_test = np.array(y_tst)
    #
    # sh0 = x_train.shape[0]
    # sh1 = x_train.shape[1]
    #
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    #
    # input_shape = x_train.shape[1:]
    #
    # model = build_model(
    #     input_shape,
    #     head_size=1,
    #     num_heads=2,
    #     ff_dim=4,
    #     num_transformer_blocks=4,
    #     mlp_units=[4],
    #     mlp_dropout=0.4,
    #     dropout=0.25,
    # )
    #
    # model.compile(
    #     loss="sparse_categorical_crossentropy",
    #     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #     metrics=["sparse_categorical_accuracy"],
    # )
    # model.summary()
    #
    # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    #
    # model.fit(
    #     x_train,
    #     y_train,
    #     validation_split=0.2,
    #     epochs=5,
    #     batch_size=10,
    #     callbacks=callbacks,
    # )
    #
    # model.evaluate(x_test, y_test, verbose=1)


    # record_sign(file_name="TEST.csv")
    # record_sign(file_name="recording_5_test.csv")

    # generate_training_examples_from_recording("recording_5.csv", 5, 0, "train")
    # generate_training_examples_from_recording("recording_5_test.csv", 5, 0, "test")

    # train(number_of_classes=5, epochs=5, batch_size=30, model_name="model_test.h5")

    # model = tf.keras.models.load_model('models/model.h5')
    # start(model=model)





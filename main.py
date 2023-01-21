import tensorflow as tf
from training import record_sign, train, generate_training_examples_from_recording
from hand_tracker import start
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from os import listdir
from os import getcwd
from matplotlib import pyplot as plt
from utils import load_data, convert_to_img, convert_to_img_debug
from numpy import genfromtxt
from constants import NUMBER_OF_KEYPOINTS
import os


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

if __name__ == '__main__':
    # data = np.zeros((80, 80), dtype=np.uint8)
    # x_tr, y_tr = load_data(5, "train")
    #
    # for example in x_tr:
    #     for e in example:
    #         data[(int(e[1] * 80) + 40), (int(e[0] * 80)) + 40] = 1
    #     plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    #     plt.show()
    #     data = np.zeros((80, 80), dtype=np.uint8)

    a = convert_to_img_debug()
    for j in range(10):
        xx = a[:, :, j].squeeze()
        plt.imshow(xx, cmap='gray', vmin=0, vmax=1)
        plt.show()



    x_tr, y_tr = convert_to_img_debug(5, "train")

    arr = np.empty([80, 80, 39, 1], dtype='float64')
    for i in x_tr:
        i = np.expand_dims(i, axis=3)
        arr = np.concatenate((arr, i), axis=3)
        for j in range(10):
            xx = i[:, :, j].squeeze()
            plt.imshow(xx, cmap='gray', vmin=0, vmax=1)
            plt.show()




    print(x_tr[0][0])
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


    # record_sign(file_name="recording_5.csv")
    # record_sign(file_name="recording_5_test.csv")

    # generate_training_examples_from_recording("recording_5.csv", 5, 0, "train")
    # generate_training_examples_from_recording("recording_5_test.csv", 5, 0, "test")

    # train(number_of_classes=5, epochs=5, batch_size=30, model_name="model_test.h5")

    # model = tf.keras.models.load_model('models/model.h5')
    # start(model=model)





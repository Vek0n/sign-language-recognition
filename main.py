import os
import cv2
import numpy as np
from numpy import genfromtxt
import mediapipe as mp
from Point import Point
from Hand import Hand
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_PADDING = 23
NUMBER_OF_KEYPOINTS = 114

def record_sign(file_name):
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        with mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image_height, image_width, _ = image.shape

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(image)
                face_results = face_detection.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                x_nose_pos = 0
                y_nose_pos = 0

                if face_results.detections:
                    for detection in face_results.detections:
                        mp_drawing.draw_detection(image, detection)
                        x_nose_pos = detection.location_data.relative_keypoints[2].x
                        y_nose_pos = detection.location_data.relative_keypoints[2].y

                nose_point = Point(x_nose_pos, y_nose_pos, 0)
                hands_list = []
                with open(file_name, 'a') as file1:
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                            hands_list.append(Hand(hand_landmarks, nose_point, image_height, image_width))

                        data_csv = get_hands_csv(hands_list)
                        file1.write(data_csv + '\n')
                    else:
                        file1.write('\n')

                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()


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


def generate_training_examples_from_recording(recording_file_name, class_number, start, data_type):
    with open(recording_file_name, 'r') as file2:
        end_of_file = False
        last_line_empty = True
        i = start
        for line in file2:
            with open("data/"+ data_type + "/" + str(class_number) + "/" + str(i) + "_" + str(class_number) + ".txt", 'a') as file3:
                if line.rstrip() == "":
                    end_of_file = True
                    if not last_line_empty:
                        i = i + 1
                if not end_of_file:
                    file3.write(line.rstrip() + '\n')
                    last_line_empty = False
                else:
                    file3.close()
                    end_of_file = False
                    last_line_empty = True


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


def train_model(trainX, trainY, testX, testY, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(SEQUENCE_PADDING, NUMBER_OF_KEYPOINTS)))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batch_size)
    model.save("model.h5")
    return model


if __name__ == '__main__':
    # Gather training data
    # record_sign(file_name="recording_class_3.csv")
    # generate_training_examples_from_recording(recording_file_name="recording_class_3.csv", class_number=1, start=1, data_type="test")

    # Train model
    X_train, Y_train = load_data(number_of_classes=2, data_type="train")
    X_train = pad_sequence(X_train, SEQUENCE_PADDING)
    Y_train = to_categorical(Y_train)

    X_test, Y_test = load_data(number_of_classes=2, data_type="test")
    X_test = pad_sequence(X_test, SEQUENCE_PADDING)
    Y_test = to_categorical(Y_test)
    model = train_model(X_train, Y_train, X_test, Y_test, epochs=15, batch_size=5)


    # Predict model
    # X_test = [genfromtxt("data/test/1/0_1.txt", delimiter=',',
    #                      usecols=np.arange(0, NUMBER_OF_KEYPOINTS))]
    # X_test = pad_sequence(X_test, SEQUENCE_PADDING)
    # prediction = model.predict(X_test)
    print("d")
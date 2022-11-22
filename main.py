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


def record_sign(file_name, class_number):
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
                        file1.write(data_csv + ',' + str(class_number) + '\n')
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


def generate_training_examples_from_recording(recording_file_name, class_number, start):
    with open(recording_file_name, 'r') as file2:
        end_of_file = False
        last_line_empty = True
        i = start
        for line in file2:
            with open("data/" + str(class_number) + "/" + str(i) + "_" + str(class_number) + ".txt", 'a') as file3:
                if line.rstrip() == "":
                    end_of_file = True
                    if last_line_empty == False:
                        i = i + 1
                if end_of_file == False:
                    file3.write(line.rstrip() + '\n')
                    last_line_empty = False
                else:
                    file3.close()
                    end_of_file = False
                    last_line_empty = True


def load_data():
    X_train = []
    Y_train = []
    num_of_classes = 2
    for i in range(0, num_of_classes):
        all_files = os.listdir("data/" + str(i))
        for file in all_files:
            file_np_array = genfromtxt("data/" + str(i) + "/" + str(file), delimiter=',',
                                       usecols=np.arange(0, 76)).tolist()
            X_train.append(file_np_array)
            Y_train.append(i)
    return X_train, Y_train


def pad_sequence(seq, maxlen):
    for j, s in enumerate(seq):
        for i in range(maxlen - len(seq[j])):
            seq[j] = np.insert(seq[j], 0, np.zeros(76), axis=0)
    return seq


def train_model(trainX, trainY):
    epochs, batch_size = 15, 5
    model = Sequential()
    model.add(LSTM(100, input_shape=(20, 76)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    model.save("model.h5")
    # evaluate model
    # _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return model

if __name__ == '__main__':
    # record_sign(file_name="recording_class_01.csv", class_number=0)
    # generate_training_examples_from_recording(recording_file_name="recording_class_1.csv", class_number=1, start=10)

    X_train, Y_train = load_data()
    X_train = pad_sequence(X_train, 20)
    arr = np.empty([20, 76], dtype='float64') #TODO delete empty sample so tensor is 18,20,76
    for i in X_train:                         #TODO fix number of features with one hand (line 84)
        arr = np.dstack((arr, i))
    Y_train.append(1.0)

    Y_train = np.array(Y_train, dtype='float64')
    Y_train = to_categorical(Y_train)
    arr = np.swapaxes(arr, 0, 2)
    arr = np.swapaxes(arr, 1, 2)

    model = train_model(arr, Y_train)
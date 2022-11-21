import os
import cv2
import numpy as np
from numpy import genfromtxt
import mediapipe as mp
from Point import Point
from Hand import Hand
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

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
    # data = ""
    if len(hands_list) == 1:
        hands.append(hands_list[0].get_timestep_data())
        zeros = [0]*19
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
                        i=i+1
                if end_of_file == False:
                    file3.write(line.rstrip()+'\n')
                    last_line_empty = False
                else:
                    file3.close()
                    end_of_file = False
                    last_line_empty = True


def load_data():
    X_train = []
    Y_train = []
    num_of_classes = 2
    for i in range(0,num_of_classes):
        all_files = os.listdir("data/" + str(i))
        for file in all_files:
            file_np_array = genfromtxt("data/" + str(i) + "/" + str(file), delimiter=',', usecols=np.arange(0, 76))#.tolist()
            X_train.append(file_np_array)
            # X_train[y] = file_np_array
            # y=y+1
            Y_train.append(i)
    return np.array(X_train), np.array(Y_train, dtype='float64')

def pad_sequence(seq, maxlen):
    # zeros = [0.0] * 76
    # for l in seq:
    for j,s in enumerate(seq):
        for i in range(maxlen - len(seq[j])):
            # l.insert(0, zeros)
             seq[j] = np.insert(seq[j], 0, np.zeros(76), axis=0)
    return seq

if __name__ == '__main__':
    # record_sign(file_name="recording_class_01.csv", class_number=0)
    # generate_training_examples_from_recording(recording_file_name="recording_class_1.csv", class_number=1, start=10)
    X_train2, Y_train2 = load_data()
    X_train2 = pad_sequence(X_train2, 20)

    # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
    # X_train = sequence.pad_sequences(X_train, maxlen=500)
    print(X_train2)

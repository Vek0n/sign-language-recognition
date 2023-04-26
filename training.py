import cv2
import mediapipe as mp
from point import Point
from hand import Hand
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils import get_hands_csv
from utils import pad_sequence
from utils import load_data

from constants import SEQUENCE_PADDING
from constants import NUMBER_OF_KEYPOINTS

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection


def generate_training_examples_from_recording(recording_file_name, class_number, start, data_type):
    with open("data/recordings/" + recording_file_name, 'r') as file2:
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


def record_sign(file_name):
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75) as hands:
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
                with open("data/recordings/" + file_name, 'a') as file1:
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


def __train_model(trainX, trainY, testX, testY, epochs, batch_size, number_of_classes):
    model = Sequential()
    model.add(LSTM(5, return_sequences=True,
                   input_shape=(SEQUENCE_PADDING, NUMBER_OF_KEYPOINTS)))
    model.add(Dropout(0.2))
    model.add(LSTM(5))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batch_size)
    return model


def train(number_of_classes, epochs, batch_size, model_name):
    X_train, Y_train = load_data(number_of_classes=number_of_classes, data_type="train")
    print("Processing training data")
    X_train = pad_sequence(X_train, SEQUENCE_PADDING)
    Y_train = to_categorical(Y_train)

    X_test, Y_test = load_data(number_of_classes=number_of_classes, data_type="test")
    print("Processing test data")
    X_test = pad_sequence(X_test, SEQUENCE_PADDING)
    Y_test = to_categorical(Y_test)
    model = __train_model(X_train, Y_train, X_test, Y_test, epochs=epochs, batch_size=batch_size, number_of_classes=number_of_classes)
    model.save("models/" + model_name)
    print("Model saved to " + model_name)

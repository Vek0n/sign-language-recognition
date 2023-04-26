import time
import cv2
import mediapipe as mp
import numpy as np
from point import Point
from hand import Hand
from numpy import genfromtxt
from utils import pad_sequence
from utils import get_hands_csv
from constants import SEQUENCE_PADDING
from constants import NUMBER_OF_KEYPOINTS

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection


def argmax_to_response(argmax):
    if argmax == 0:
        return "Dobry"
    elif argmax == 1:
        return "Dziekuje"
    elif argmax == 2:
        return "Czesc"
    elif argmax == 3:
        return "Prosze"
    elif argmax == 4:
        return "Widziec"
    elif argmax == 5:
        return "Nazwisko"
    return argmax


def predict(model):
    data = [genfromtxt("data/temp.csv", delimiter=',',
                       usecols=np.arange(0, NUMBER_OF_KEYPOINTS))]
    data = pad_sequence(data, SEQUENCE_PADDING)
    prediction = model.predict(data, verbose=0)
    argmax = np.argmax(prediction)
    return argmax_to_response(argmax)


def start(model):
    last_prediction = time.time()
    last_frame = False
    prediction = ""
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75) as hands:
        with mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                if time.time() - last_prediction > 3:
                    prediction = ""
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
                with open("data/temp.csv", 'a') as temp:
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
                        temp.write(data_csv + '\n')
                        last_frame = True
                    else:
                        if last_frame:
                            prediction = predict(model)
                            last_prediction = time.time()
                        last_frame = False
                        temp.close()
                        open('data/temp.csv', 'w').close()

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (160, 450)
                fontScale = 2
                fontColor = (255, 255, 255)
                thickness = 4
                lineType = 2
                im = cv2.flip(image, 1)
                cv2.putText(im, prediction,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                cv2.imshow('MediaPipe Hands', im)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()

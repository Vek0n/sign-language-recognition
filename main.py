import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from constants import SEQUENCE_PADDING
from hand_tracker import start
import training
import utils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

class_names = [
'dobry',    #0
'dzień',    #1
'wieczór',  #2
'cześć',    #3
'jak się czujesz',#4
'ja',       #5
'ty',       #6
'bardzo',   #7
'do widzenia',#8
'dobranoc', #9
'jak',      #10
'dziękuję', #11
'zmęczony', #12
'lubić',    #13
'chcieć',   #14
'nie lubić',#15
'co?',      #16
'gdzie?',   #17
'ile?',     #18
'jest',     #19
'mój',      #20
'twój',     #21
'mam',      #22
'masz',     #23
'imie',     #24
'nazwisko', #25
'jeśli',    #26
'dużo',     #27
'duży',     #28
'mały',     #29
'mało',     #30
'umieć',    #31
'A',        #32
'Ą',        #33
'B',        #34
'C',        #35
'Ć',         #36
'CH',        #37
'CZ',       #38
'D',         #39
'E',        #40
'F',         #41
'G',         #42
'H',         #43
'I',         #44
'J',         #45
'K',         #46
'L',         #47
'M',         #48
'N',        #49
'O',         #50
'P',         #51
'R',         #52
'S',         #53
'T',         #54
'U',         #55
'V',         #56
'W',         #57
'X',         #58
'Y',         #59
'Z',         #60
'Ś',         #61
'SZ',         #62
'Ń',         #63
'Ó',         #64
]

class Estimator:
  _estimator_type = ''
  classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred


def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

if __name__ == '__main__':
    class_number = 65
    class_category = "train"
    # training.record_sign(file_name="recording_" + str(class_number) + "_" + class_category + ".csv")
    # training.generate_training_examples_from_recording("recording_" + str(class_number) + "_" + class_category + ".csv", class_number, 0, class_category)
    # training.delete_short_files("data/" + class_category + "/" + str(class_number))

    #
    # training.train(number_of_classes=65, epochs=35, batch_size=64, model_name="model16.h5")
    #
    model = tf.keras.models.load_model('models/model65_2.h5')
    # start(model=model)
    #
    # model.summary()
    X_test, Y_test = utils.load_data(number_of_classes=65, data_type="test")
    print("Processing test data")
    X_test = utils.pad_sequence(X_test, SEQUENCE_PADDING)
    Y_test = tf.keras.utils.to_categorical(Y_test)
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print("F1 score:", f1)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    print("Precision:", precision)
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print("Recall:", recall)


    # unique_class_labels = np.unique(true_labels)
    # cm = confusion_matrix(true_labels, predicted_labels, labels=unique_class_labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot()
    # plt.xticks(rotation=75)
    # plt.tight_layout()
    # plt.show()

import tensorflow as tf
from hand_tracker import start
import training

#0 dobry
#1 dziekuje
#2 czesc
#3 prosze
#4 widzieć


if __name__ == '__main__':
    training.record_sign(file_name="recording_6.csv")
    # record_sign(file_name="recording_6_test.csv")

    # generate_training_examples_from_recording("recording_6.csv", 5, 0, "train")
    # generate_training_examples_from_recording("recording_6_test.csv", 5, 0, "test")

    # training.train(number_of_classes=6, epochs=30, batch_size=5, model_name="model_test.h5")

    # model = tf.keras.models.load_model('models/model_test.h5')
    # start(model=model)



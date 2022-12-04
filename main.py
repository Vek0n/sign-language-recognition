import tensorflow as tf
from training import record_sign, train, generate_training_examples_from_recording
from hand_tracker import start

#0 dobry
#1 dziekuje
#2 czesc
#3 prosze
#4 widzieć
if __name__ == '__main__':

    record_sign(file_name="recording_5.csv")
    # record_sign(file_name="recording_5_test.csv")

    # generate_training_examples_from_recording("recording_5.csv", 5, 0, "train")
    # generate_training_examples_from_recording("recording_5_test.csv", 5, 0, "test")

    # train(number_of_classes=5, epochs=5, batch_size=30, model_name="model_test.h5")

    model = tf.keras.models.load_model('models/model.h5')
    start(model=model)



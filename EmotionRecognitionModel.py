import tensorflow as tf
import cv2
import numpy as np


class EmotionRecognitionModel:

    def __init__(self):
        self.model = tf.keras.models.load_model('./ResNet50_checkpoint_best.hdf5')
        self.face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        self.emotions = {0: 'anger',
                         1: 'contempt',
                         2: 'disgust',
                         3: 'fear',
                         4: 'happy',
                         5: 'neutral',
                         6: 'sad',
                         7: 'surprise',
                         8: 'uncertain'}

    def predict(self, pic):
        return self.emotions[np.argmax(self.model.predict(pic))]

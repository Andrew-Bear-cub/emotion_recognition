import tensorflow as tf
from keras_vggface import utils
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


# Функция предобработки изображения перед подачей в модель
def preprocess_image(img):
    processed = cv2.resize(img, (224, 224))  # Меняем размер картинки в нужный для подачи в модель
    processed = np.expand_dims(image.img_to_array(processed),
                               axis=0)  # Переводим картинку в массив и разворачиваем его
    processed = utils.preprocess_input(processed,
                                       version=2)  # Предобрпбатывамем картинку, веса resnet50 - поэтому вторая
    # версия (из документации)
    return processed


class Predictor:

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

        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FPS, 60)  # Частота кадров
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # Ширина кадров в видеопотоке.
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # Высота кадров в видеопотоке.

    def make_predict(self, pic):
        return self.emotions[np.argmax(self.model.predict(pic))]

    def video_stream(self):
        while True:
            # Считываем кадр
            _, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Обнаруживаем лица
            faces = self.face_cascade.detectMultiScale(gray, 1.15, 5)

            # Рисуем рамку вокруг каждого лица
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # Предсказываем эмоцию
                predict = self.make_predict(preprocess_image(img[y:y + h, x:x + w]))
                # Размещаем название эмоции возле рамки
                cv2.putText(img, predict, (y + 200, x - 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2)
            # Выводим картинку на экран
            cv2.imshow('img', img)
            # Останавливаем программу по нажатии клавиши Esc
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


# Если запускается как самостоятельный модуль - создаем нужные объекты и начинаем распознавание
if __name__ == '__main__':
    pred = Predictor()
    pred.video_stream()
    pred.cam.release()
    cv2.destroyAllWindows()

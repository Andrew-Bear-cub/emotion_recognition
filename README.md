# Emotion recognition with VGG-Face CNN and OpenCV face detector using Haar feature-based cascade classifier
Diploma project for Skillbox.
Implementation of an algorithm for recognizing human emotions. Capturing video from a camera using the OpenCV library, face recognition using Haar cascades,
then the inference of emotion from the trained model.

[This](https://github.com/rcmalli/keras-vggface) keras-vggface implementation is used as a neural network.
Model training takes place in google collab.
The results of my training are available at [link](https://drive.google.com/file/d/1x691GZaU66tb16OCodMjC45DlXAiGh10/view?usp=sharing)
Inference and image output are in recognition.py file.
The file with the Haar cascade (haarcascade_frontalface_default.xml) and the saved model from the link above must be placed in the root of the project in PyCharm.
The binaries required to run are installed from the requirements.txt file


Дипломный проект для образовательной платформы Скиллбокс. 
Реализация алгоритма для распознавания эмоций. захват видео с камеры с помощью библиотеки OpenCV, распознавание лиц с использованием каскадов Хаара,
затем инференс эмоций из обученной ранее модели.

В качестве нейронной сети используется вот [эта](https://github.com/rcmalli/keras-vggface) реализация keras-vggface.
Обучение моделей проиходит в google collab в приложенном ноутбуке. 
Результаты моего обучения доступны по [ссылке](https://drive.google.com/file/d/1x691GZaU66tb16OCodMjC45DlXAiGh10/view?usp=sharing) 
Инференс и вывод изображения в файле recognition.py . 
В корень проекта в PyCharm необходимо подложить файл с каскадом Хаара (haarcascade_frontalface_default.xml) и сохраненную модель по ссылке выше.
Необходимые для запуска билбиотеки ставятся из файла requirements.txt
![image](https://user-images.githubusercontent.com/37875675/141759276-27d71f6c-5f9c-42f5-8320-4c4a23095b17.png)

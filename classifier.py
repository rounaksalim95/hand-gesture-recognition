import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report

# Imports

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = keras.models.load_model("./FinalModel/GestureRecognitionModel.tfl")

# Read test images and generate test_labels

test_images = []

def read_images(file_path):
  for i in range(901, 1201):
    image = cv2.imread(f"./hand-gesture-dataset-copy/{file_path}/{i}.jpg")
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(bw_image.reshape(50, 50, 1))


read_images("test/test/ok-sign")
read_images("test/test/l-sign")
read_images("test/test/victory-sign")
read_images("test/test/fingers-crossed")
read_images("test/test/thumbs-up")

print(len(test_images))


test_labels = []
for i in range(300):
  test_labels.append([1, 0, 0, 0, 0])

for i in range(300):
  test_labels.append([0, 1, 0, 0, 0])

for i in range(300):
  test_labels.append([0, 0, 1, 0, 0])

for i in range(300):
  test_labels.append([0, 0, 0, 1, 0])

for i in range(300):
  test_labels.append([0, 0, 0, 0, 1])

# classification report
pred = model.predict(np.array(test_images))
print(pred.shape)

y_pred=np.argmax(pred, axis=1)
y_test=np.argmax(np.array(test_labels), axis=1)

print(classification_report(y_test, y_pred))
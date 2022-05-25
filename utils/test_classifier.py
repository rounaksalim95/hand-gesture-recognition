import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# model = keras.models.load_model("./FinalModel/GestureRecognitionModel.tfl")
model = keras.models.load_model(
    "../FinalModel/CustomDataGestureRecognitionModelBig.tfl"
)

test_images = []

image = cv2.imread("../webcam-images/3.png")
# image = cv2.imread(f"./hand-gesture-dataset-copy/test/test/ok-sign/910.jpg")
image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
test_images.append(bw_image.reshape(50, 50, 1))

start = time.time()
pred = model.predict(np.array(test_images))
end = time.time()
score = tf.nn.softmax(pred)

print(f"Time taken for prediction: {end - start} seconds")
print("Prediction is:", score)
print(np.argmax(score))
print("amax", np.amax(score))

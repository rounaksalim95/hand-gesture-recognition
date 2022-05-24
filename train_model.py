"""
Script to train a CNN model using Tensorflow along with Keras API
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

DATASET_PATH = "./custom_dataset"
CLASSES = 5
IMAGE_HEIGHT = 50
IMAGE_WIDTH = 50


def read_images(file_path, start, end, image_arr):
    """
    Read images from given file path and push image to image_arr
    Args:
        file_path: Path to directory containing images to read
        image_arr: List to push images to
    """
    for i in range(start, end):
        image = cv2.imread(f"{DATASET_PATH}/{file_path}/{i}.png")
        bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arr.append(bw_image.reshape(50, 50, 1))

def main():
    train_images = []
    test_images = []

    # Read train images
    read_images("train/0_ok_sign", 1, 2401, train_images)
    read_images("train/1_l_sign", 1, 2401, train_images)
    read_images("train/2_victory_sign", 1, 2401, train_images)
    read_images("train/3_fingers_crossed", 1, 2401, train_images)
    read_images("train/4_thumbs_up", 1, 2401, train_images)

    # Read test images
    read_images("test/0_ok_sign", 2401, 3201, test_images)
    read_images("test/1_l_sign", 2401, 3201, test_images)
    read_images("test/2_victory_sign", 2401, 3201, test_images)
    read_images("test/3_fingers_crossed", 2401, 3201, test_images)
    read_images("test/4_thumbs_up", 2401, 3201, test_images)

    # Generate train labels
    train_labels = []

    for i in range(2400):
        train_labels.append([1, 0, 0, 0, 0])

    for i in range(2400):
        train_labels.append([0, 1, 0, 0, 0])

    for i in range(2400):
        train_labels.append([0, 0, 1, 0, 0])

    for i in range(2400):
        train_labels.append([0, 0, 0, 1, 0])

    for i in range(2400):
        train_labels.append([0, 0, 0, 0, 1])


    # Generate test labels
    test_labels = []

    for i in range(800):
        test_labels.append([1, 0, 0, 0, 0])

    for i in range(800):
        test_labels.append([0, 1, 0, 0, 0])

    for i in range(800):
        test_labels.append([0, 0, 1, 0, 0])

    for i in range(800):
        test_labels.append([0, 0, 0, 1, 0])

    for i in range(800):
        test_labels.append([0, 0, 0, 0, 1])


    # Convert to numpy arrays
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Shuffle data
    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    # Create model
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPool2D(padding="same"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(padding="same"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(padding="same"),
        layers.Dropout(.15),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(CLASSES)
    ])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = opt,
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics = ["accuracy"])

    epochs = 10
    history = model.fit(x = train_images, y = train_labels, validation_data = (test_images, test_labels), epochs = epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # classification report
    
    pred = model.predict(test_images)

    y_pred=np.argmax(pred, axis=1)
    y_test=np.argmax(test_labels, axis=1)

    print(classification_report(y_test, y_pred))

    model.save("FinalModel/CustomDataGestureRecognitionModelBig.tfl")


if __name__ == '__main__':
    main()

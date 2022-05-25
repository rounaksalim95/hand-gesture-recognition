import cv2
import numpy as np
import tensorflow as tf

from absl import app, logging
from PIL import Image
from tensorflow import keras

CONFIDENCE_THRESHOLD = 0.99

# model = keras.models.load_model("./FinalModel/GestureRecognitionModel.tfl")
model = keras.models.load_model("./FinalModel/CustomDataGestureRecognitionModelBig.tfl")


def resizeImage(imageName):
    finalwidth = 50
    img = Image.open(imageName)
    wpercent = finalwidth / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((finalwidth, hsize), Image.ANTIALIAS)
    print(f"Image shape is {img}")
    img.save(imageName)


def resize_image(image):
    # image = cv2.resize(image, (1,50,50,1), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    return image


def getPrediction():
    img = cv2.imread("temp.png")
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([bw_img.reshape(50, 50, 1)])
    return np.argmax(prediction), (
        np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2])
    )


def get_prediction(image):
    # img = cv2.imread('temp.png')
    # bw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict(np.array([image.reshape(50, 50, 1)]))
    
    score = tf.nn.softmax(prediction)
    pred = np.argmax(score)
    confidence = np.amax(score)

    # Get the prediction class
    prediction_class = ""
    if pred == 0:
        prediction_class = "OK sign"
    elif pred == 1:
        prediction_class = "L sign"
    elif pred == 2:
        prediction_class = "Victory sign"
    elif pred == 3:
        prediction_class = "Fingers crossed"
    elif pred == 4:
        prediction_class = "Thumbs up"

    logging.info(f"Predicted {prediction_class} with confidence {confidence * 100}%")

    if confidence < CONFIDENCE_THRESHOLD:
        return "", confidence

    return prediction_class, confidence


def main(argv):
    # index depends on your devices
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1)

    # define frame corners
    # top, right, bottom, left = 10, 350, 225, 590
    # top, right, bottom, left = 10, 350, 260, 600
    top, right, bottom, left = 10, 900, 260, 1200

    while cap.isOpened():
        image, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        # thresh = gray
        cv2.imwrite("temp_full.png", thresh)
        crp_img = thresh[top:bottom, right:left]

        # cv2.imwrite('temp.png', crp_img)
        # resize_image('temp.png')

        # temp_img = cv2.imread("temp.png")
        # print("IMAGE SHAPE: ", temp_img.shape)

        resized_img = resize_image(crp_img)
        cv2.imwrite("temp.png", resized_img)

        # prediction, confidence = getPrediction()
        prediction, confidence = get_prediction(resized_img)
        # cv2.imshow("After processing", 'temp.png')

        if prediction:
            cv2.putText(
                frame,
                "Pedicted Class : " + prediction + " Confidence : " + str(confidence * 100) + "%",
                (right, bottom),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_4,
            )

        cv2.rectangle(frame, (right, top), (left, bottom), (255, 255, 255), 2)
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)

import cv2
import numpy as np
from matplotlib import pyplot as plt

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
from PIL import Image

model = keras.models.load_model("./FinalModel/GestureRecognitionModel.tfl")

def resizeImage(imageName):
    finalwidth = 50
    img = Image.open(imageName)
    wpercent = (finalwidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((finalwidth,hsize), Image.ANTIALIAS)
    print(f"Image shape is {img}")
    img.save(imageName)

def resize_image(image):
    # image = cv2.resize(image, (1,50,50,1), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (50,50), interpolation = cv2.INTER_AREA)
    return image
    
def getPrediction():
    img = cv2.imread('temp.png')
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([bw_img.reshape(50,50,1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def get_prediction(image):
    # img = cv2.imread('temp.png')
    # bw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict(np.array([image.reshape(50,50,1)]))
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence):
    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "OK sign"
    elif predictedClass == 1:
        className = "L sign"
    elif predictedClass == 2:
        className = "Victory sign"
    elif predictedClass == 3:
        className = "Fingers crossed"
    elif predictedClass == 4:
        className = "Thumbs up"

    cv2.putText(textImage,"Pedicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    return className

def main():
    #index depends on your devices
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1)
    
    #define frame corners
    #top, right, bottom, left = 10, 350, 225, 590
    top, right, bottom, left = 10, 350, 260, 600
    
    i=0 #number of images
    
    while cap.isOpened() :
        image, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.Canny(gray, 100, 200)
        thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        #thresh = gray
        cv2.imwrite('temp_full.png', thresh)
        crp_img = thresh[top:bottom,right:left]
        
        # cv2.imwrite('temp.png', crp_img)
        # resize_image('temp.png')

        # temp_img = cv2.imread("temp.png")
        # print("IMAGE SHAPE: ", temp_img.shape)

        resized_img = resize_image(crp_img)
        cv2.imwrite('temp.png', resized_img)

        # prediction, confidence = getPrediction()
        prediction, confidence = get_prediction(resized_img)
        tf.print(prediction)
        pred_str = showStatistics(prediction, confidence)
        #cv2.imshow("After processing", 'temp.png')
        
        cv2.rectangle(frame, (right,top), (left,bottom), (255, 255, 255), 2)
        cv2.putText(frame, str(pred_str), (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.imshow("image",frame)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')): break
        i+=1
        #if (i>50): break
            
    cap.release()
    #cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

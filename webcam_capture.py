import cv2

cam = cv2.VideoCapture(0)

# Read the image
result, image = cam.read()

if result:
    cv2.imshow("test", image)

    # Save image in local storage
    cv2.imwrite("test.png", image)

    # Press key to destroy window
    cv2.waitKey(0)
    cv2.destroyWindow("test")

else:
    print("Error in capturing image")

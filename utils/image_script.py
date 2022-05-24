"""
Utility script to capture images from webcam and save them to a folder to build custom dataset
"""

import cv2


def resize_image(image):
    # image = cv2.resize(image, (1,50,50,1), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    return image


def main():
    # index depends on your devices
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1)

    # define frame corners
    # top, right, bottom, left = 10, 350, 225, 590
    # top, right, bottom, left = 10, 350, 260, 600
    top, right, bottom, left = 10, 900, 260, 1200

    i = 0  # number of images
    j = 0

    while cap.isOpened():
        image, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.Canny(gray, 100, 200)
        thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
        # thresh = gray
        crp_img = thresh[top:bottom, right:left]
        resized_img = resize_image(crp_img)
        cv2.rectangle(frame, (right, top), (left, bottom), (255, 255, 255), 2)
        # cv2.putText(frame, str(pred_str), (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.imshow("image", frame)

        if i % 5 == 0:
            cv2.imwrite(f"l-sign-0/{j}.png", resized_img)
            j += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        i += 1


if __name__ == "__main__":
    main()

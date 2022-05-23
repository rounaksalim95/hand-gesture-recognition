import cv2

def resize_image(image):
    # image = cv2.resize(image, (1,50,50,1), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (50,50), interpolation = cv2.INTER_AREA)
    return image

def main():
    #index depends on your devices
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1)
    
    #define frame corners
    #top, right, bottom, left = 10, 350, 225, 590
    top, right, bottom, left = 10, 350, 260, 600
    
    i=0 #number of images
    
    while cap.isOpened() :

        if i < 20 :
            continue
        
        image, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.Canny(gray, 100, 200)
        thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        #thresh = gray
        crp_img = thresh[top:bottom,right:left]
        resized_img = resize_image(crp_img)
        cv2.rectangle(frame, (right,top), (left,bottom), (255, 255, 255), 2)
        #cv2.putText(frame, str(pred_str), (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.imshow("image",frame)        
        
        if i%5 == 0:
            cv2.imwrite(f'thumbs_up/{i}.png', resized_img)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')): break

        i+= 1


if __name__ == "__main__":
    main()
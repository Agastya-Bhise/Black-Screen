import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = cv2.imread("place.webp", cv2.IMREAD_UNCHANGED)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    img = np.flip(img, axis=1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0,0,0])
    upper_black = np.array([111,111,111])

    mask_1 = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    final_output = cv2.addWeighted(res_1,1,res_2,1,0)
    output_file.write(final_output)
    
    cv2.imshow("Happy Holidays!", final_output)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()  


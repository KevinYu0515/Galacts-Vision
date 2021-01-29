import cv2
import numpy as np
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import argparse
import imutils

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def nothing():
    pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 480)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Beta", "Trackbars", 0, 20, nothing)
cv2.createTrackbar("Eroseion","Trackbars", 0, 100, nothing)
cap = cv2.VideoCapture(1)


while True:
    sucess, frame = cap.read()

    if not sucess:
        print('Can’t recive frame(stream end?). Exiting ...')
        break

    
    #ret, thresh = cv2.threshold(hsv, 127, 255, cv2.THRESH_BINARY)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    alpha = 2
    beta = cv2.getTrackbarPos("Beta","Trackbars")
    eroseion = cv2.getTrackbarPos("Eroseion", "Trackbars")
    
    lower_yellow = np.array([l_h, l_s, l_v])
    upper_yellow = np.array([u_h, u_s, u_v])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black = np.zeros(hsv.shape, hsv.dtype)
    dst = cv2.addWeighted(hsv, alpha, black, 1-alpha, beta)
    
    kernel = np.ones((3,3), np.uint8)

    mask = cv2.inRange(dst, lower_yellow, upper_yellow)
    eroseion = cv2.erode(mask,kernel,iterations=eroseion)

    result = cv2.bitwise_and(hsv, frame, mask=eroseion)
    Gaussian = cv2.GaussianBlur(result,(0,0), sigmaX=2)
    canny = cv2.Canny(Gaussian, 20, 150)

    (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixelsPerMetric = None

    frame_ = frame.copy()

    cv2.drawContours(frame_, cnts, -1, (0, 255, 0), 2)

    for c in cnts: 
        if cv2.contourArea(c)<100:
            continue
        else:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if area > 100 :
                (x, y), radius = cv2.minEnclosingCircle(c)
                center, radius = (int(x), int(y)), int(radius)  # for the minimum enclosing circle
                frame_ = cv2.circle(frame_, center, radius, (0, 0, 255), 4)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            

            
    print("dx,dy,area",cx,cy,area)

    imgStack = stackImages(0.4,[frame,dst,eroseion,Gaussian,canny,frame_])
    cv2.imshow("Main", imgStack)
    if cv2.waitKey(10) == ord('q'):
        break
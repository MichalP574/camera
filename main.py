# import libraries
import time

import cv2
# import numpy as np
# import time
# import request
import circle
# distance between fiducial markers (width - W (mm), height - H (mm):
SCALE = 4
W = 285
H = 160

is_vertical = False


def empty(a):
    pass


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # set capture device, 0 - default, 1 - USB Webcam, -1 - to get a list of available devices
cv2.namedWindow("MeasuringInstrument")
# cv2.resizeWindow("MeasuringInstrument", 640, 160)
cv2.createTrackbar("Threshold1", "MeasuringInstrument", 120, 255, empty)
cv2.createTrackbar("Threshold2", "MeasuringInstrument", 140, 255, empty)
# minimal and maximal radius = minimal + 30 for fiducial markers
cv2.createTrackbar("minRadius", "MeasuringInstrument", 10, 255, empty)

while True:
    timeStart = time.perf_counter()
    success, frame = cap.read()
    imgCopy = frame.copy()
    threshold1 = cv2.getTrackbarPos("Threshold1", "MeasuringInstrument")
    threshold2 = cv2.getTrackbarPos("Threshold2", "MeasuringInstrument")
    minRadius = cv2.getTrackbarPos("minRadius", "MeasuringInstrument")
    maxRadius = minRadius + 30
    imgProcessed = circle.imageProcessing(imgCopy, threshold1, threshold2)
    imgToWarp, fiducialMarkers = circle.getFiducialMarkersToWarp(imgProcessed, draw=True)
    if len(fiducialMarkers) == 4:
        imgWarped = circle.warpImg(imgToWarp, fiducialMarkers, W*SCALE, H*SCALE)
        # imgWarpedWithCircles = circle.getCircle(imgWarped, minRadius, maxRadius)
        imgPostProcessed, points = circle.getPendulumFM(imgWarped)
        imgStack = circle.stackImages(1, [frame, imgToWarp, imgPostProcessed])
        if len(points) == 2 and is_vertical:
            angle = circle.calculateAngle(points)
        cv2.imshow("MeasuringInstrument", imgStack)
        if not is_vertical:
            # CAPS LOCK OFF!!!
            if cv2.pollKey() & 0xFF == ord("v"):
                REFERENCE_POINTS, X, Y = circle.setVertical(points)
                print(f"Reference points: {REFERENCE_POINTS}, X: {X}, Y: {Y}")
                is_vertical = True
    # else:
    #     cv2.imshow("Test", frame)
    timeEnd = time.perf_counter()
    totalTime = timeEnd-timeStart
    print(f"Total time: {totalTime:0.4f}")
    if cv2.waitKey(1) & 0xFF == ord('q'):  # to terminate the programme press "q" provided that "v" key had been pressed
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#komentarz

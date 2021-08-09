# import libraries
import cv2
import numpy as np
import time
import imutils
import math

SCALE = 4

REFERENCE_POINTS = []
X = []
Y = []
# the minimal area of the rectangles (fiducial markers) indispensable to warp image
MIN_AREA_RECTANGLES = 10


def imageProcessing(img, threshold1=120, threshold2=140):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDil, kernel, iterations=2)
    return imgThre


def getFiducialMarkersToWarp(img, minArea=MIN_AREA_RECTANGLES, filter=4, draw=False):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangleContours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            # length of arc, True == closed
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # bounding box
            bbox = cv2.boundingRect(approx)
            # rectangles = 4 vertices
            if len(approx) == filter:
                rectangleContours.append([len(approx), area, approx, bbox, i])
    # while drawing on a binary (1 channel) image only the first number from RGB tuple is taken
    fiducial_markers = []
    if draw and len(rectangleContours) > 0:
        for contour in rectangleContours:
            # cv2.drawContours(img, contour[4], -1, (127,0,0), 3)
            ((x, y), radius) = cv2.minEnclosingCircle(contour[4])
            cv2.circle(img, (int(x), int(y)), int(radius), (127, 0, 0), 2)
            M = cv2.moments(contour[4])
            center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
            cv2.circle(img, center, 5, (127, 0, 0), -1)
            fiducial_markers.append(center)
            # print(fiducial_markers)
    return img, fiducial_markers


def reorderFiducialMarkers(fiducialMarkers):
    newMarkers = np.zeros_like(fiducialMarkers)
    fiducialMarkers = np.array(fiducialMarkers)
    #sum of each column(0) and sum of each row(1) of a given array
    add = fiducialMarkers.sum(axis=1)
    newMarkers[0] = fiducialMarkers[np.argmin(add)]
    newMarkers[3] = fiducialMarkers[np.argmax(add)]
    diff = np.diff(fiducialMarkers, axis=1)
    newMarkers[1] = fiducialMarkers[np.argmin((diff))]
    newMarkers[2] = fiducialMarkers[np.argmax((diff))]
    return newMarkers


def warpImg(img, points, w, h, pad=70):
    points = reorderFiducialMarkers(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp


def getCircle(img, minRadius=10, maxRadius=40, ):
    imgC = img.copy()
    circles = cv2.HoughCircles(imgC, cv2.HOUGH_GRADIENT, 1, 20, param1=60, param2=40, minRadius=minRadius,
                               maxRadius=maxRadius)
    # ensure at least some circles were found
    if circles is not None:
        # covert x, y coordinates and radius into integers
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(imgC, (x, y), 2, (127, 0, 0), 1)
    return imgC


def getPendulumFM(img, minArea=10, filter=7):
    imgThreCopy = img.copy()
    contours = cv2.findContours(imgThreCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(contours)
    center = None
    points = []
    rescaledPoints2 = []
    if len(cnts) > 0:
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > minArea:
                # length of arc, True == closed
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # bounding box
                bbox = cv2.boundingRect(approx)
                if len(approx) > filter:
                    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    M = cv2.moments(cnt)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    # only proceed if the radius meets a minimum size
                    if radius > 5:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(imgThreCopy, (int(x), int(y)), int(radius),
                                   (127, 0, 0), 2)
                        cv2.circle(imgThreCopy, center, 5, (127, 0, 0), -1)
                        points.append([int(x), int(y)])

                        rescaledPoints2.append([int(x // SCALE), int(y // SCALE), int(radius // SCALE)])
                    if len(points) > 1:
                        cv2.line(imgThreCopy, (points[0][0], points[0][1]), (points[1][0], points[1][1]),
                                 color=(127, 0, 0), thickness=4)
    return imgThreCopy, rescaledPoints2


def setVertical(verticalPoints):
    global REFERENCE_POINTS, X, Y
    referencePoints = np.zeros_like(verticalPoints)
    verticalPoints = np.array(verticalPoints)
    referencePoints = verticalPoints[verticalPoints[:, 2].argsort()]
    REFERENCE_POINTS = referencePoints
    XYR = np.diff(REFERENCE_POINTS, axis=0)
    X = XYR[0, 0]
    Y = XYR[0, 1]
    return REFERENCE_POINTS, X, Y


def calculateAngle(pendulumPoints):
    # https://www.euclideanspace.com/maths/algebra/vectors/angleBetween/ atan2
    pendulumPoints = np.array(pendulumPoints)
    newPenMarkers = np.zeros_like(pendulumPoints)
    newPenMarkers = pendulumPoints[pendulumPoints[:, 2].argsort()]
    global X, Y
    x = X
    y = Y
    # the relative angle between two angles: [atan2(v2.y,v2.x) - atan2(v1.y,v1.x)] [pendulum - vertical_line]
    theta = math.atan2(-(newPenMarkers[1, 1] - newPenMarkers[0, 1]), newPenMarkers[1, 0] - newPenMarkers[0, 0]) - \
            math.atan2(-(y), x)
    if theta < 0:
        theta = theta + 2*math.pi
    theta_degree = round(math.degrees(theta))
    # print(f"Angle: {theta_degree}")
    return theta_degree


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

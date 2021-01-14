
import cv2 as cv
import numpy as np
import imutils
import math

def capture_histogram(source):
    cap = cv.VideoCapture(source)
    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (1000, 600))

        font = cv.FONT_HERSHEY_COMPLEX
        cv.putText(frame, "Place region of hand inside this box and press A",
                   (5,50), font, 0.7, (255,255,255), 2, cv.LINE_AA)
        cv.rectangle(frame, (500,100), (580,180), (105,105,105), 2)
        box = frame[105:175, 505:575]
    #    object_color = box

        cv.imshow("capture histogram", frame)
        key = cv.waitKey(10)
        if key == 97:
            object_color = box
            cv.destroyAllWindows()
            break

        if key == ord('q'):
            cv.destroyAllWindows()
            cap.release()
            break

    object_color_hsv = cv.cvtColor(object_color, cv.COLOR_BGR2HSV)
    object_hist = cv.calcHist([object_color_hsv], [0,1], None,
                                  [12,15], [0,180,0,256])

    cv.normalize(object_hist, object_hist, 0, 255, cv.NORM_MINMAX)

    return object_hist


def locate_object(frame, object_hist):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    object_segment = cv.calcBackProject([hsv_frame], [0, 1], object_hist,
                                        [0, 180, 0, 256], 1)

    _, segment_thresh = cv.threshold(object_segment, 70, 255,
                                     cv.THRESH_BINARY)

    kernel = None
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    filtered = cv.filter2D(segment_thresh, -1, disc)

    eroded = cv.erode(filtered, kernel, iterations=2)
    dilated = cv.dilate(eroded, kernel, iterations=2)
    closing = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

    masked = cv.bitwise_and(frame, frame, mask=closing)

    return closing, masked, segment_thresh


def detect_hand(frame, hist):
    return_value = {}

    detected_hand, masked, raw = locate_object(frame, hist)
    return_value["binary"] = detected_hand
    return_value["masked"] = masked
    return_value["raw"] = raw

    contours, hierarchy = cv.findContours(detected_hand,
                                          cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)

    palm_area = 0
    flag = None
    cnt = None

    for (i,c) in enumerate(contours):
        area = cv.contourArea(c)
        if area > palm_area:
            palm_area = area
            flag = i

    if flag is not None and palm_area > 5000:
        cnt = contours[flag]
        return_value["contours"] = cnt
        cpy = frame.copy()
        cv.drawContours(cpy, [cnt], 0, (0, 255, 0), 2)
        return_value["boundaries"] = cpy
        return True, return_value

    else:
        return  False, return_value

    return return_value

def extract_fingertips(hand):
    cnt = hand["contours"]
    points = []
    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        end = tuple(cnt[e][0])
        points.append(end)

    filtered = filter_points(points, 50)
    filtered.sort(key=lambda point: point[1])

    return [pt for idx, pt in zip(range(5), filtered)]


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (b[1]-a[1])**2)

def filter_points(points, filterValue):
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if points[i] and points[j] and dist(points[i], points[j]) < filterValue:
                points[j] = None

    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)

    return filtered

def plot (frame, points):
    radius = 8
    colour = (0, 0, 255)
    thickness = -1
    for point in points:
        cv.circle(frame, point, radius, colour, thickness)


cap = cv.VideoCapture(0)
hist = capture_histogram(0)
screen = np.zeros((600, 1000))

curr = None
prev = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hand_detected, hand = detect_hand(frame, hist)
    if hand_detected:
        hand_image = hand["boundaries"]
        fingertips = extract_fingertips(hand)
        plot(hand_image, fingertips)

        prev = curr
        curr = fingertips[0]

        if prev and curr:

            cv.line(screen, prev, curr, (255, 0, 0), 5)

        cv.imshow("Drawing", screen)
        cv.imshow("Hand Detector", hand_image)

    else:
        cv.imshow("Hand Detector", frame)

    k = cv.waitKey(10)
    if k == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
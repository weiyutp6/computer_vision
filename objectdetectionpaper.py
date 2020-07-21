from pyimagesearch.shapedetector import ShapeDetector
import imutils
from cv2 import *
import numpy as np


def greenImage(img):
    return split(img)[1]


def redImage(img):
    return split(img)[2]


def blueImage(img):
    return split(img)[0]


def YUY2(img):
    return split(cvtColor(img, COLOR_RGB2YUV))[0]


previous_contours = []


def find_contour(binary, img, color):
    cont = imutils.grab_contours(findContours(binary, RETR_TREE, CHAIN_APPROX_SIMPLE))
    cont_used = []
    pos = []
    sd = ShapeDetector()
    for a in range(1, len(cont)):
        c = cont[a]
        # compute the center of the contour
        M = moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        shape = sd.detect(c)
        isSame = False
        for contour in previous_contours:
            contour_dist = (contour[0] - cX) ** 2 + (contour[1] - cY) ** 2
            if contour_dist < 2500:
                isSame = True
        if isSame:
            continue
        else:
            previous_contours.append([cX, cY])
            cont_used.append(c)
        if color == 'black totem' and shape != 'rectangle':
            color = 'light bouy'
        b, g, r = img[cY, cX]
        drawContours(img, [c], -1, (0, 255, 0), 2)
        putText(img, color, (cX - 20, cY - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        pos.append([color, [cX, cY]])
    return pos


def get_color(img, lower_range, upper_range, color):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # imshow('image', img)
    # waitKey(0)

    lower_range = np.array([lower_range])
    upper_range = np.array([upper_range])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    kernel = np.ones((5, 5), np.uint8)
    opening = morphologyEx(mask, MORPH_OPEN, kernel)
    not_opening = bitwise_not(opening)
    # imshow('image', not_opening)
    # waitKey(0)
    return find_contour(not_opening, img, color)


def object_detection(diff, img, color):
    difference = diff - YUY2(img)
    filtered = medianBlur(difference, 25)
    binary = threshold(filtered, 200, 255, THRESH_OTSU)[1]
    # imshow('image', binary)
    # waitKey(0)
    return find_contour(binary, img, color)


def return_objects(img):
    # redImg = object_detection(redImage(img), img)
    # greenImg = object_detection(greenImage(img), img)
    black = get_color(img, (0, 0, 0), (3, 3, 50), 'black totem')
    red = get_color(img, (0, 200, 50), (3, 255, 255), 'red totem')
    yellow = get_color(img, (25, 200, 50), (35, 255, 255), 'yellow totem')
    green = get_color(img, (55, 200, 50), (65, 255, 255), 'green totem')
    blue = get_color(img, (115, 200, 50), (125, 255, 255), 'blue totem')
    # white = get_color(img, (0, 0, 40), (3, 3, 50), 'white')
    surmark_950400 = get_color(img, (74, 75, 50), (82, 108, 255), 'surmark_950400')
    surmark_950410 = get_color(img, (0, 100, 50), (4, 160, 255), 'surmark_950410')
    other = object_detection(redImage(img), img, 'surmark_46104')
    other1 = object_detection(greenImage(img), img, 'surmark_46104')

    total = black
    total.extend(red)
    total.extend(yellow)
    total.extend(green)
    total.extend(blue)
    # total.extend(white)
    total.extend(surmark_950400)
    total.extend(surmark_950410)
    total.extend(other)
    total.extend(other1)
    return total


if __name__ == '__main__':
    image = imread('vrx_objs.png')
    print(return_objects(image))
    imshow('image', image)
    waitKey(0)

from pyimagesearch.shapedetector import ShapeDetector
import imutils
from cv2 import *
import numpy as np


class ObjectDetection:
    def __init__(self, img):
        self.__img = img
        self.__previous_contours = []

    # get gray image which uses green value as luminosity
    def greenImage(self):
        return split(self.__img)[1]

    # get gray image which uses red value as luminosity
    def redImage(self):
        return split(self.__img)[2]

    # get gray image which uses blue value as luminosity
    def blueImage(self):
        return split(self.__img)[0]

    # get y in yuy2 encoding
    def YUY2(self):
        return split(cvtColor(self.__img, COLOR_RGB2YUV))[0]

    # find contours of objects in binary image
    def find_contour(self, binary, color):
        cont = imutils.grab_contours(findContours(binary, RETR_TREE, CHAIN_APPROX_SIMPLE))
        cont_used = []
        pos = ""
        sd = ShapeDetector()
        for a in range(1, len(cont)):
            c = cont[a]
            # compute the center of the contour
            M = moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape = sd.detect(c)
            isSame = False
            for contour in self.__previous_contours:
                contour_dist = (contour[0] - cX) ** 2 + (contour[1] - cY) ** 2
                if contour_dist < 2500:
                    isSame = True
            if isSame:
                continue
            else:
                self.__previous_contours.append([cX, cY])
                cont_used.append(c)
            if color == 'black_totem' and shape != 'rectangle':
                color = 'light_bouy'
            b, g, r = self.__img[cY, cX]
            drawContours(self.__img, [c], -1, (0, 255, 0), 2)
            putText(self.__img, color, (cX - 20, cY - 20),
                    FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            pos += color + ":" + "(" + str(cX) + "," + str(cY) + ")" + "\t"
        return pos

    # use a specific range of colors as mask and transfer image to binary image
    def get_color(self, lower_range, upper_range, color):
        hsv = cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV)
        # imshow('image', self)
        # waitKey(0)

        lower_range = np.array([lower_range])
        upper_range = np.array([upper_range])

        mask = cv2.inRange(hsv, lower_range, upper_range)
        kernel = np.ones((5, 5), np.uint8)
        opening = morphologyEx(mask, MORPH_OPEN, kernel)
        not_opening = bitwise_not(opening)
        # imshow('image', not_opening)
        # waitKey(0)
        return self.find_contour(not_opening, color)

    # find binary image from subtracting a single color value gray image and y value then threshold using otsu algorithm
    def object_detection(self, diff, color):
        difference = diff - self.YUY2()
        filtered = medianBlur(difference, 25)
        binary = threshold(filtered, 200, 255, THRESH_OTSU)[1]
        # imshow('image', binary)
        # waitKey(0)
        return self.find_contour(binary, color)

    # find vrx_target objects
    # output: 'obj_name:(x,y)'
    def return_objects(self):
        # redimg = object_detection(redImage(self), self)
        # greenimg = object_detection(greenImage(self), self)
        black = self.get_color((0, 0, 0), (3, 3, 50), 'black_totem')
        red = self.get_color((0, 200, 50), (3, 255, 255), 'red_totem')
        yellow = self.get_color((25, 200, 50), (35, 255, 255), 'yellow_totem')
        green = self.get_color((55, 200, 50), (65, 255, 255), 'green_totem')
        blue = self.get_color((115, 200, 50), (125, 255, 255), 'blue_totem')
        # white = get_color(self, (0, 0, 40), (3, 3, 50), 'white')
        surmark_950400 = self.get_color((74, 75, 50), (82, 108, 255), 'surmark_950400')
        surmark_950410 = self.get_color((0, 100, 50), (4, 160, 255), 'surmark_950410')
        surmark_46104 = self.object_detection(self.redImage(), 'surmark_46104')
        # other1 = object_detection(greenImage(self), self, 'surmark_46104')

        total = black + red + yellow + green + blue + surmark_950400 + surmark_950410 + surmark_46104
        return total.strip("\t")


# 範例
if __name__ == '__main__':
    image = imread('vrx_objs.png')
    # return_objects回傳資料用\t隔開的字串
    od = ObjectDetection(image)
    print(od.return_objects().split("\t"))
    imshow('image', image)
    waitKey(0)

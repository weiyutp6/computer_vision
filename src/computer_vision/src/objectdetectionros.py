from shapedetector import ShapeDetector
import imutils
from cv2 import *
import numpy as np

import rospy
from std_msgs.msg import Int32MultiArray, Float64, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from time import sleep

bridge = CvBridge()

contour_dist_max = 2500
contour_color = (0, 255, 0)
text_color = (255, 255, 255)
blur_radius = 25
struct_ele = (5, 5)


class ObjectDetection:
    def __init__(self):
        self.on_startup()
        self.__img = []
        self.__previous_contours = []
        self.middle_cam_sub = rospy.Subscriber("/wamv/sensors/cameras/front_left_camera/image_raw", Image, self.middle_callBack, queue_size=1)

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
                if contour_dist < contour_dist_max:
                    isSame = True
            if isSame:
                continue
            else:
                self.__previous_contours.append([cX, cY])
                cont_used.append(c)
            if color == 'black_totem' and shape != 'rectangle':
                color = 'light_bouy'
            b, g, r = self.__img[cY, cX]
            drawContours(self.__img, [c], -1, contour_color, 2)
            putText(self.__img, color, (cX - 20, cY - 20),
                    FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            pos += color + ":" + "(" + str(cX) + "," + str(cY) + ")" + "\t"
        return pos

    # use a specific range of colors as mask and transfer image to binary image
    def get_color(self, lower_range, upper_range, color):
        hsv = cvtColor(self.__img, COLOR_BGR2HSV)
        # imshow('image', self)
        # waitKey(0)

        lower_range = np.array([lower_range])
        upper_range = np.array([upper_range])

        mask = inRange(hsv, lower_range, upper_range)
        kernel = np.ones(struct_ele, np.uint8)
        opening = morphologyEx(mask, MORPH_OPEN, kernel)
        not_opening = bitwise_not(opening)
        # imshow('image', not_opening)
        # waitKey(0)
        return self.find_contour(not_opening, color)

    # find binary image from subtracting a single color value gray image and y value then threshold using otsu algorithm
    def object_detection(self, diff, color):
        difference = diff - self.YUY2()
        filtered = medianBlur(difference, blur_radius)
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
        if pub_message_sum != "":
            img_item_pub = rospy.Publisher("ntu/img_item_pub", String, queue_size=10)
            img_item_pub.publish(total.strip("\t"))

    def on_startup(self):
        global contour_dist_max
        global text_color
        global contour_color
        global blur_radius
        global struct_ele
        if rospy.has_param('~contour_dist_max'):
            contour_dist_max = rospy.get_param('~contour_dist_max')
        if rospy.has_param('~contour_color'):
            contour_color = rospy.get_param('~contour_color')
        if rospy.has_param('~text_color'):
            text_color = rospy.get_param('~text_color')
        if rospy.has_param('~blur_radius'):
            blur_radius = rospy.get_param('~blur_radius')
        if rospy.has_param('~struct_ele'):
            struct_ele = rospy.get_param('~struct_ele')

    def middle_callBack(self, data):
        try:
            img = bridge.imgmsg_to_cv2(data, "bgr8")
            print(img)
        except CvBridgeError as e:
            print(e)
            print("CvBridge error")

        self.__img = img
        self.return_objects()
 
    def onShutdown(self):
        rospy.loginfo("CV Shutdown.")

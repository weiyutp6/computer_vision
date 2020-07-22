#!/usr/bin/env python

from objectdetectionros import *

if __name__ == '__main__':
    rospy.init_node('objectdetectionmain', anonymous=True)
    # return_objects returns data parsed by \t
    od = ObjectDetection()
    
    rospy.on_shutdown(od.onShutdown)
    rospy.spin()

#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn import linear_model
import random
import math
from matplotlib import pyplot as plt


bridge = CvBridge()

# debugging publishers
purple_dots_pub = None
green_dots_pub = None
red_dots_pub = None
blue_dots_pub = None
all_dots_pub = None

# for debugging purposes; in order to publish which pixels were recognized by the mask
def publish_img(img, publisher):
    transport_img = bridge.cv2_to_imgmsg(img, "rgb8")
    publisher.publish(transport_img)


def image_callback(img_msg):
    img_rgb = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define color ranges
    lower_red = np.array([171, 80, 50])  # done
    upper_red = np.array([180, 255, 255])

    lower_red2 = np.array([0, 80, 50])  # done
    upper_red2 = np.array([20, 255, 255])

    lower_blue = np.array([80, 120, 80])
    upper_blue = np.array([120, 255, 255])

    lower_green = np.array([30, 80, 50])  # done
    upper_green = np.array([70, 255, 255])

    lower_purple = np.array([121, 80, 50])
    upper_purple = np.array([170, 255, 255])

    # Get masks
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    mask_red = cv2.medianBlur(mask_red, 3)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red2 = cv2.medianBlur(mask_red2, 3)
    mask_red = np.bitwise_or(mask_red, mask_red2)

    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_green = cv2.medianBlur(mask_green, 3)

    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask_blue = cv2.medianBlur(mask_blue, 3)

    mask_purple = cv2.inRange(img_hsv, lower_purple, upper_purple)
    mask_purple = cv2.medianBlur(mask_purple, 3)

    mask_all = np.bitwise_or(np.bitwise_or(np.bitwise_or(mask_red, mask_green), mask_blue), mask_purple)

    selected_red = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_red)
    publish_img(selected_red, red_dots_pub)

    selected_green = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_green)
    publish_img(selected_green, green_dots_pub)

    selected_blue = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_blue)
    publish_img(selected_blue, blue_dots_pub)

    selected_purple = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_purple)
    publish_img(selected_purple, purple_dots_pub)

    selected_all = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_all)
    publish_img(selected_all, all_dots_pub)

    # get clusters, 10 Cluster
    # calculate mean value from all (H, S, V) in the form of
    # [(H,S,V) for cluster 1, (H,S,V) for cluster 2, ...],

    # for each color:
    # compute [abs(targetH - clusterH)weightH + abs(targetS - clusterS)*weightS + abs(targetV - clusterV)*weightV,
    #           for cluster 2 also, for cluster 3 also...]
        # -> get min value from array -> this cluster is our winner -> get mean coordinates


def init():
    global purple_dots_pub, green_dots_pub, red_dots_pub, blue_dots_pub, all_dots_pub

    rospy.init_node('indoor_localization', anonymous=True)
    rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    purple_dots_pub = rospy.Publisher('/debug_image/magenta', Image, queue_size=10)
    green_dots_pub = rospy.Publisher('/debug_image/green', Image, queue_size=10)
    red_dots_pub = rospy.Publisher('/debug_image/red', Image, queue_size=10)
    blue_dots_pub = rospy.Publisher('/debug_image/blue', Image, queue_size=10)
    all_dots_pub = rospy.Publisher('/debug_image/all_colors', Image, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    try:
        init()
    except rospy.ROSInterruptException:
        pass

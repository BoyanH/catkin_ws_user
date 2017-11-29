#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
rgb_bw_pub = None
hsv_bw_pub = None
yuv_bw_pub = None

def publish_img(img, publisher):
    transport_img = bridge.cv2_to_imgmsg(img, "rgb8")
    publisher.publish(transport_img)


def image_callback(img_msg):
    img_rgb = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)

    # RGB Transform
    rgb_threshold = 190
    lower_rgb = np.array([rgb_threshold, rgb_threshold, rgb_threshold])
    upper_rgb = np.array([255, 255, 255])
    mask_rgb = cv2.inRange(img_rgb, lower_rgb, upper_rgb)
    mask_rgb = cv2.medianBlur(mask_rgb, 3)
    res_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_rgb)
    publish_img(res_rgb, rgb_bw_pub)

    # HSV Transform
    lower_hsv = np.array([0, 0, 204])
    upper_hsv = np.array([179, 51, 255])
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    mask_hsv = cv2.medianBlur(mask_hsv, 3)
    res_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_hsv)
    res_hsv_in_rgb = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2RGB)
    publish_img(res_hsv_in_rgb, hsv_bw_pub)

    # YUV Transform
    lower_yuv = np.array([154, 108, 108])
    # lower_yuv = np.array([0, 0, 108])
    upper_yuv = np.array([255, 148, 148])
    mask_yuv = cv2.inRange(img_yuv, lower_yuv, upper_yuv)
    mask_yuv = cv2.medianBlur(mask_yuv, 3)
    res_yuv = cv2.bitwise_and(img_yuv, img_yuv, mask=mask_yuv)
    res_yuv_in_rgb = cv2.cvtColor(res_yuv, cv2.COLOR_YUV2RGB)
    # res_yuv_in_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_yuv)
    publish_img(res_yuv_in_rgb, yuv_bw_pub)

def init():
    global rgb_bw_pub, hsv_bw_pub, yuv_bw_pub

    rospy.init_node('line_detector', anonymous=True)
    rospy.Subscriber('/app/camera/rgb/image_raw', Image, image_callback)
    rgb_bw_pub = rospy.Publisher('/img/binarized/rgb', Image, queue_size=10)
    hsv_bw_pub = rospy.Publisher('/img/binarized/hsv', Image, queue_size=10)
    yuv_bw_pub = rospy.Publisher('/img/binarized/yuv', Image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    try:
        init()
    except rospy.ROSInterruptException:
        pass

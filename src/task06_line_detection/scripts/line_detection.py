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
rgb_bw_pub = None
hsv_bw_pub = None
yuv_bw_pub = None
with_lines_pub = None

max_iterations = 100

def publish_img(img, publisher):
    transport_img = bridge.cv2_to_imgmsg(img, "rgb8")
    publisher.publish(transport_img)

def get_m_b(a, b):
    m = (a[1] - b[1]) / (a[0] - b[0] + np.nextafter(0, 1))
    b = a[1] - m * a[0]

    return m, b

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
    res_hsv = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_hsv)
    publish_img(res_hsv, hsv_bw_pub)

    # YUV Transform
    lower_yuv = np.array([154, 108, 108])
    # lower_yuv = np.array([0, 0, 108])
    upper_yuv = np.array([255, 148, 148])
    mask_yuv = cv2.inRange(img_yuv, lower_yuv, upper_yuv)
    mask_yuv = cv2.medianBlur(mask_yuv, 3)
    res_yuv = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_yuv)
    publish_img(res_yuv, yuv_bw_pub)

    # TODO: vectorize
    white_points = np.where(np.array(mask_hsv))
    line_x = white_points[1]
    line_y = white_points[0]

    # find_lines(line_points)

    # line_points = np.array(line_points)

    line_x = np.matrix(line_x).reshape(-1,1)
    line_y = np.array(line_y)
    ra = linear_model.RANSACRegressor(stop_n_inliers=len(line_x)*0.4, max_trials=200, residual_threshold=50)
    ra.fit(line_x, line_y)

    inverted_inliers_mask = np.invert(ra.inlier_mask_)
    inliers_first_X = line_x[ra.inlier_mask_]
    outliers_first_X = line_x[inverted_inliers_mask]
    outliers_first_y = line_y[inverted_inliers_mask]

    ra2 = linear_model.RANSACRegressor(stop_n_inliers=len(outliers_first_X)*0.8, max_trials=200, residual_threshold=50)
    ra2.fit(outliers_first_X, outliers_first_y)

    if len(np.where(ra2.inlier_mask_)[0]) < len(outliers_first_X)*0.6:
        return

    first_a = (inliers_first_X.min(), ra.predict(inliers_first_X.min()))
    first_b = (inliers_first_X.max(), ra.predict(inliers_first_X.max()))

    second_a = (outliers_first_X.min(), ra2.predict(outliers_first_X.min()))
    second_b = (outliers_first_X.max(), ra2.predict(outliers_first_X.max()))

    m1, b1 = get_m_b(first_a, first_b)
    m2, b2 = get_m_b(second_a, second_b)

    img_with_lines = cv2.line(img_rgb, first_a, first_b, (255,0,0))
    img_with_lines = cv2.line(img_with_lines, second_a, second_b, (255,0,0))

    publish_img(img_with_lines, with_lines_pub)

    # plt.show()


def init():
    global rgb_bw_pub, hsv_bw_pub, yuv_bw_pub, with_lines_pub

    rospy.init_node('line_detector', anonymous=True)
    rospy.Subscriber('/app/camera/rgb/image_raw', Image, image_callback)
    rgb_bw_pub = rospy.Publisher('/img/binarized/rgb', Image, queue_size=10)
    hsv_bw_pub = rospy.Publisher('/img/binarized/hsv', Image, queue_size=10)
    yuv_bw_pub = rospy.Publisher('/img/binarized/yuv', Image, queue_size=10)
    with_lines_pub = rospy.Publisher('/img/with/lines', Image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    try:
        init()
    except rospy.ROSInterruptException:
        pass

























def find_intercept_point(m, b, point):
    x = (point[0] + m * point[1] - m * b) / (1 + m ** 2)
    y = (m * point[0] + (m ** 2) * point[1] - (m ** 2) * b) / (1 + m ** 2) + b

    return x, y


def get_dist_between_points(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


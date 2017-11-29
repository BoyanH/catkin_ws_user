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

max_iterations = 100

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
    line_x = []
    line_y = []
    for x, row in enumerate(mask_hsv):
        for y, cell in enumerate(row):
            if cell != 0:
                line_x.append(x)
                line_y.append(y)

    # find_lines(line_points)

    # line_points = np.array(line_points)

    line_x = np.matrix(line_x).reshape(-1,1)
    line_y = np.array(line_y)
    ra = linear_model.RANSACRegressor(stop_n_inliers=len(line_x)*0.3, )
    ra.fit(line_x, line_y)

    line_X = np.arange(np.array(line_x).min(), np.array(line_x).max()).reshape(-1,1)
    line_y_ransac = ra.predict(line_X)

    inverted_inliers_mask = np.invert(ra.inlier_mask_)
    outliers_first_X = line_x[inverted_inliers_mask]
    outliers_first_y = line_y[inverted_inliers_mask]

    ra2 = linear_model.RANSACRegressor(stop_n_inliers=len(outliers_first_X)*0.6)
    ra2.fit(outliers_first_X, outliers_first_y)
    line_X2 = np.arange(np.array(outliers_first_X).min(), np.array(outliers_first_X).max()).reshape(-1, 1)
    line_y_ransac2 = ra2.predict(line_X2)
    fig, ax = plt.subplots()

    ax.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=10,
             label='RANSAC regressor')
    ax.plot(line_X2, line_y_ransac2, color='cornflowerblue', linewidth=10,
             label='RANSAC regressor')

    plt.show()


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


























def find_lines(data):
    for i in range(max_iterations):
        a, b = random.sample(data, k=2)
        rospy.loginfo(a)

        # y = mx + b
        # a_x*m + b = a_y
        # b_x*m + b = b_y
        # a_x*m - b_x*m = a_y - b_y
        # m = (a_y - b_y)/(a_x - b_x)
        # rospy.loginfo(a)

        m = (a[1] - b[1])/(a[0] - b[0] + np.nextafter(0, 1))
        b = a[1] - m*a[0]
        rospy.loginfo("y = {}x + {}".format(m ,b))

def find_intercept_point(m, b, point):
    x = (point[0] + m * point[1] - m * b) / (1 + m ** 2)
    y = (m * point[0] + (m ** 2) * point[1] - (m ** 2) * b) / (1 + m ** 2) + b

    return x, y


def get_dist_between_points(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
import random
import math
from matplotlib import pyplot as plt

bridge = CvBridge()

number_of_clusters = 4

# debugging publishers
purple_dots_pub = None
green_dots_pub = None
red_dots_pub = None
blue_dots_pub = None
all_dots_pub = None
recognized_pub = None
odom_pub = None

weight_h = 7
weight_s = 4
weight_v = 4


# for debugging purposes; in order to publish which pixels were recognized by the mask
def publish_img(img, publisher):
    transport_img = bridge.cv2_to_imgmsg(img, "rgb8")
    publisher.publish(transport_img)


def mirror_coords(coords, max_x, max_y):
    return [[max_x - x, max_y - y] for x, y in coords]


def get_img_coords(img_msg):
    img_rgb = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define color ranges
    lower_red = np.array([171, 20, 100])  # done
    upper_red = np.array([180, 255, 255])

    lower_red2 = np.array([0, 20, 100])  # done
    upper_red2 = np.array([20, 255, 255])

    lower_blue = np.array([80, 210, 220])
    upper_blue = np.array([120, 255, 255])

    lower_green = np.array([21, 0, 50])  # done
    upper_green = np.array([79, 255, 255])

    lower_purple = np.array([125, 150, 250])
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

    # get clusters, 4 Cluster
    # calculate mean value from all (H, S, V) in the form of
    # [(H,S,V) for cluster 1, (H,S,V) for cluster 2, ...],

    # for each color:
    # compute [abs(targetH - clusterH)weightH + abs(targetS - clusterS)*weightS + abs(targetV - clusterV)*weightV,
    #           for cluster 2 also, for cluster 3 also...]
    # -> get min value from array -> this cluster is our winner -> get mean coordinates

    masked_pixels = np.where(np.array(mask_all))
    # get all (x,y) pair of coordinates in a 2d array (matrix)
    masked_points_coordinates = np.array(list(zip(masked_pixels[1], masked_pixels[0])))

    clustered = KMeans(n_clusters=number_of_clusters).fit(masked_points_coordinates)

    points_per_cluster = [[] for x in range(number_of_clusters)]

    for idx, label in enumerate(clustered.labels_):
        points_per_cluster[label].append(masked_points_coordinates[idx])

    # mean_colors_per_cluster = np.vectorize(lambda x: get_mean_color(x, img_hsv),
    #                                        signature='(m)->(m)')(points_per_cluster)
    mean_colors_per_cluster = [get_mean_color(x, img_hsv) for x in points_per_cluster]

    # TODO: get really precise expectancy colors
    red_expected_color = np.array([175, 190, 150])
    blue_expected_color = np.array([110, 230, 230])
    green_expected_color = np.array([50, 250, 100])
    # green_expected_color = np.array([70, 100, 100])
    purple_expected_color = np.array([140, 190, 255])

    img_coords_red = get_image_coords(mean_colors_per_cluster, points_per_cluster, red_expected_color)
    img_coords_blue = get_image_coords(mean_colors_per_cluster, points_per_cluster, blue_expected_color)
    img_coords_green = get_image_coords(mean_colors_per_cluster, points_per_cluster, green_expected_color)
    img_coords_purple = get_image_coords(mean_colors_per_cluster, points_per_cluster, purple_expected_color)

    cv2.circle(img_rgb, tuple(img_coords_red), 10, (255, 0, 0), 1)
    cv2.circle(img_rgb, tuple(img_coords_blue), 10, (0, 0, 255), 1)
    cv2.circle(img_rgb, tuple(img_coords_green), 10, (0, 255, 0), 1)
    cv2.circle(img_rgb, tuple(img_coords_purple), 10, (148, 0, 211), 1)
    publish_img(img_rgb, recognized_pub)

    center = math.floor(len(img_hsv[0]) / 2), math.floor(len(img_hsv) / 2)

    return mirror_coords([img_coords_green, img_coords_red, img_coords_blue, img_coords_purple, center],
                         len(img_hsv[0]), len(img_hsv))


def get_dists_coords_and_angles(img_msg):
    img_coords_green, img_coords_red, img_coords_blue, img_coords_purple, center = get_img_coords(img_msg)
    real_coords_green, real_coords_red, real_coords_blue, real_coords_purple = np.array(
        [[2.29, 1.14], [3.55, 3.03], [4.18, 1.77], [2.29, 2.4]]
    )

    angle_g_c_b = get_angle_between_points_and_center(img_coords_green, center, img_coords_blue)
    angle_c_g_b = get_angle_between_points_and_center(center, img_coords_green, img_coords_blue)
    angle_c_b_g = np.pi - angle_g_c_b - angle_c_g_b

    # TODO: check if that is right with the vector divisions
    angle_r_c_p = get_angle_between_points_and_center(img_coords_red, center, img_coords_purple)
    angle_c_r_p = get_angle_between_points_and_center(center, img_coords_red, img_coords_purple)
    angle_c_p_r = np.pi - angle_r_c_p - angle_c_r_p

    distance_g_b_real = np.linalg.norm(real_coords_green - real_coords_blue)
    distance_r_p_real = np.linalg.norm(real_coords_red - real_coords_purple)

    distance_green = distance_g_b_real * np.sin(angle_c_b_g) / np.sin(angle_g_c_b)
    distance_blue = distance_g_b_real * np.sin(angle_c_g_b) / np.sin(angle_g_c_b)

    distance_red = distance_r_p_real * np.sin(angle_c_r_p) / np.sin(angle_r_c_p)
    distance_purple = distance_r_p_real * np.sin(angle_c_p_r) / np.sin(angle_r_c_p)

    distances = distance_green, distance_blue, distance_red, distance_purple
    coordinates = [real_coords_green, real_coords_blue, real_coords_red, real_coords_purple]

    angle_g_c_p = get_angle_between_points_and_center(img_coords_green, center, img_coords_purple)
    angles = [angle_g_c_b, angle_g_c_p]

    return distances, coordinates, angles


def get_angle_between_points_and_center(a, center, b, ):
    a_vec = np.array(a) - np.array(center)
    b_vec = np.array(b) - np.array(center)

    return np.arccos(a_vec.dot(b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))


def get_mean_color(coordinates, img):
    # img = [ row, row, [col, col, [h, s, v], ..], ..]

    color_for_coordinates = np.array([img[x[1]][x[0]] for x in coordinates])
    mean_color = np.mean(color_for_coordinates, axis=0)
    return mean_color


def get_image_coords(mean_cluster_colors, points_per_cluster, expected_color):
    distances_per_cluster = [get_distance_to_cluster(
        mean_cluster_colors[i], expected_color) for i in range(len(mean_cluster_colors))]
    winner_cluster = distances_per_cluster.index(min(distances_per_cluster))
    average_cluster_coords = np.array(points_per_cluster[winner_cluster]).mean(0)

    return np.array(average_cluster_coords, dtype=int)


def get_distance_to_cluster(mean_color, expected_color):
    return (min(abs(mean_color[0] - expected_color[0]), abs(mean_color[0] + expected_color[0]) % 180) * weight_h +
            abs(mean_color[1] - expected_color[1]) * weight_s +
            abs(mean_color[2] - expected_color[2]) * weight_v)


def image_callback(scan_msg):
    distances, coords, angles = get_dists_coords_and_angles(scan_msg)

    dist_1 = distances[0]
    x_1, y_1 = coords[0]

    dist_coords_k = list(zip(distances, coords))[1:]

    # a = X2 - X1, Y2 - Y1
    # b = (dist1^2 - dist2^2 - X1^2 - Y1^2 + X2^2 + Y2^2) / 2
    a = np.array([[x_k - x_1, y_k - y_1] for x_k, y_k in coords[1:]])
    b = np.array(
        [(dist_1 ** 2 - dist_k ** 2 - x_1 ** 2 - y_1 ** 2 + x_k ** 2 + y_k ** 2) / 2 for dist_k, (x_k, y_k) in
         dist_coords_k])

    (x, y) = np.linalg.lstsq(a, b)[0]

    rospy.loginfo('x: {}; y: {}'.format(x, y))

    # TODO: magic
    yaw = angles[0]

    yaw_quaternion = yaw_to_quaternion(yaw)
    odometry = Odometry()
    odometry.header.frame_id = 'map'
    odometry.header.seq = scan_msg.header.seq
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.orientation = yaw_quaternion

    odom_pub.publish(odometry)


def yaw_to_quaternion(yaw):
    """ convert a yaw angle (in radians) into a Quaternion message """
    return Quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))

def init():
    global purple_dots_pub, green_dots_pub, red_dots_pub, blue_dots_pub, all_dots_pub, recognized_pub, odom_pub

    rospy.init_node('indoor_localization', anonymous=True)
    rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    purple_dots_pub = rospy.Publisher('/debug_image/magenta', Image, queue_size=10)
    green_dots_pub = rospy.Publisher('/debug_image/green', Image, queue_size=10)
    red_dots_pub = rospy.Publisher('/debug_image/red', Image, queue_size=10)
    blue_dots_pub = rospy.Publisher('/debug_image/blue', Image, queue_size=10)
    all_dots_pub = rospy.Publisher('/debug_image/all_colors', Image, queue_size=10)
    recognized_pub = rospy.Publisher('/debug_image/recognized', Image, queue_size=10)
    odom_pub = rospy.Publisher('/odom_gps', Odometry, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    try:
        init()
    except rospy.ROSInterruptException:
        pass

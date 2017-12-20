#!/usr/bin/env python

# x size
# 6m 1.6cm = 6.016
# y size
# 4m 1cm = 4.01

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray, Marker
from sklearn.cluster import KMeans
import math
bridge = CvBridge()

number_of_clusters = 4

inf = float('inf')
none_coords = np.array([inf, inf])
# debugging publishers
purple_dots_pub = None
green_dots_pub = None
red_dots_pub = None
blue_dots_pub = None
all_dots_pub = None
recognized_pub = None

marker_array_pub = None
odom_pub = None


field_x_size = 6.016
field_y_size = 4.01

x_noise_size = math.sqrt(0.001)
y_noise_size = math.sqrt(0.001)
yaw_noise_size = math.sqrt(np.pi/64)

odom_msg_queue = []
last_image_cb_stamp = None

# -------------------------------------- LAMPS RECOGNITION -------------------------------------------------------------

weight_h = 7
weight_s = 2
weight_v = 2


# for debugging purposes; in order to publish which pixels were recognized by the mask
def publish_img(img, publisher):
    transport_img = bridge.cv2_to_imgmsg(img, "rgb8")
    publisher.publish(transport_img)


def mirror_coords(coords, max_x, max_y):
    return np.array([[max_x - x, max_y - y] if x != inf else none_coords for x, y in coords])


def get_lamp_coords(img_msg):
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
    # for regions on image which have the same color as one of the lamps
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

    # combine masks
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


    # using all the pixels which match with the color of any lamp, we can then cluster them all
    # and for each lamp in the real world pick the best fit from a cluster (cluster with mean color that best
    # fits the desired one) and return the mean x,y position of the cluster as location

    # !IMPORTANT: here we also use weights for h,s,v to determine how "close" one colour is to another

    masked_pixels = np.where(np.array(mask_all))
    # get all (x,y) pair of coordinates in a 2d array (matrix)
    masked_points_coordinates = np.array(list(zip(masked_pixels[1], masked_pixels[0])))

    clustered = KMeans(n_clusters=number_of_clusters).fit(masked_points_coordinates)

    points_per_cluster = [[] for x in range(number_of_clusters)]

    for idx, label in enumerate(clustered.labels_):
        points_per_cluster[label].append(masked_points_coordinates[idx])

    mean_colors_per_cluster = [get_mean_color(x, img_hsv) for x in points_per_cluster]
    expected_colors = [
        np.array([175, 190, 150]),
        np.array([110, 230, 230]),
        np.array([50, 250, 100]),
        np.array([140, 190, 255])
    ]
    img_coords = [get_coords_in_image(mean_colors_per_cluster,
                                      points_per_cluster, expected_color) for expected_color in expected_colors]

    for i, coord in enumerate(img_coords):
        for j in range(i+1, len(img_coords)):
            if img_coords[i][0] != inf and img_coords[j][0] != inf and \
                            np.linalg.norm(img_coords[i] - img_coords[j]) <= 20:
                if (get_distance_to_cluster(img_hsv[img_coords[i][1]][img_coords[i][0]], expected_colors[i]) <
                        get_distance_to_cluster(img_hsv[img_coords[j][1]][img_coords[j][0]], expected_colors[j])):
                    img_coords[j] = none_coords
                else:
                    img_coords[i] = none_coords


    img_coords_red, img_coords_blue, img_coords_green, img_coords_purple = img_coords
    raw_colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (148, 0, 211)
    ]

    for idx, colour in enumerate(raw_colors):
        if img_coords[idx][0] != inf:
            cv2.circle(img_rgb, tuple(img_coords[idx]), 10, colour, 1)
    publish_img(img_rgb, recognized_pub)

    center = math.floor(len(img_hsv[0]) / 2), math.floor(len(img_hsv) / 2)

    return mirror_coords([img_coords_green, img_coords_red, img_coords_blue, img_coords_purple, center],
                         len(img_hsv[0]), len(img_hsv))



def get_angle_between_vectors(vec_1, vec_2):
    dot_product = vec_1.dot(vec_2)
    det = np.linalg.det(np.array([vec_1, vec_2]))
    return np.arctan2(det, dot_product)



def get_mean_color(coordinates, img):
    # img = [ row, row, [col, col, [h, s, v], ..], ..]

    color_for_coordinates = np.array([img[x[1]][x[0]] for x in coordinates])
    mean_color = np.mean(color_for_coordinates, axis=0)
    return mean_color


def get_coords_in_image(mean_cluster_colors, points_per_cluster, expected_color):
    distances_per_cluster = [get_distance_to_cluster(
        mean_cluster_colors[i], expected_color) for i in range(len(mean_cluster_colors))]
    winner_cluster = distances_per_cluster.index(min(distances_per_cluster))
    average_cluster_coords = np.array(points_per_cluster[winner_cluster]).mean(0)

    return np.array(average_cluster_coords, dtype=int)


def get_distance_to_cluster(mean_color, expected_color):
    return (min(abs(mean_color[0] - expected_color[0]), abs(mean_color[0] + expected_color[0]) % 180) * weight_h +
            abs(mean_color[1] - expected_color[1]) * weight_s +
            abs(mean_color[2] - expected_color[2]) * weight_v)


# ---------------------------------- END OF LAMPS RECOGNITION ----------------------------------------------------------

marker_array = None


def initialize_particle_cloud():
    global marker_array

    marker_array = MarkerArray()

    for i in range(100):
        new_marker = Marker()
        new_marker.header.frame_id = 'map'
        new_marker.header.stamp = rospy.Time.now()

        new_marker.id = i
        new_marker.ns = 'point_cloud_marker'

        x, y, yaw_quaternion = get_random_marker_point()
        new_marker.scale.x = .5
        new_marker.scale.y = .1
        new_marker.scale.z = .1

        new_marker.color.a = 1
        new_marker.color.r = 0
        new_marker.color.g = 1
        new_marker.color.b = 0

        new_marker.type = 0

        new_marker.pose.position.x = x
        new_marker.pose.position.y = y
        new_marker.pose.orientation = yaw_quaternion
        marker_array.markers.append(new_marker)

def get_random_marker_point():
    x = np.random.random_sample() * field_x_size
    y = np.random.random_sample() * field_y_size
    yaw = np.random.random_sample() * 2*np.pi - np.pi

    return x, y, yaw_to_quaternion(yaw)


def get_seen_angles(img_msg):
    img_coords_red, img_coords_blue, img_coords_green, img_coords_purple, center = get_lamp_coords(img_msg)

    # TODO calculate
    return 1,2,3,4


def get_x_y_orientation(odom_msg):
    pose = odom_msg.pose.pose
    position = pose.position
    orientation = pose.orientation

    return position.x, position.y, get_orientation_angle(orientation)


def unpack_msg(odom_msg):
    return get_x_y_orientation(odom_msg), odom_msg.header.stamp


def get_odom_velocity_and_yaw_change_speed():
    if len(odom_msg_queue) != 2:
        return None, None

    odom_last_msg, odom_current_msg = odom_msg_queue
    (x_last, y_last, yaw_last), t_last = unpack_msg(odom_last_msg)
    (x_crnt, y_crnt, yaw_crnt), t_current = unpack_msg(odom_current_msg)

    pos_last = np.array([x_last, y_last])
    pos_current = np.array([x_crnt, y_crnt])
    distance = np.linalg.norm(pos_last - pos_current)
    time_passed = (t_last - t_current).to_nsec() * 1.0
    velocity = distance / time_passed
    yaw_diff = (yaw_crnt - yaw_last)
    # rospy.loginfo('current: {}; last: {}; difference: {}'.format(yaw_crnt, yaw_last, yaw_crnt - yaw_last))

    if yaw_diff < -np.pi:
        yaw_diff = 2 * np.pi + yaw_diff
    elif yaw_diff > np.pi:
        yaw_diff = -2 * np.pi + yaw_diff

    yaw_change_speed = yaw_diff / time_passed

    return velocity, -yaw_change_speed


def propagate(time_passed):
    odom_velocity, yaw_change_speed = get_odom_velocity_and_yaw_change_speed()
    velocity_time = odom_velocity * time_passed
    yaw_time = yaw_change_speed * time_passed

    for i, marker in enumerate(marker_array.markers):
        m_x = marker.pose.position.x
        m_y = marker.pose.position.y
        m_yaw = get_orientation_angle(marker.pose.orientation)

        new_x = m_x + np.cos(m_yaw) * velocity_time + (np.random.random_sample() - 0.5) * x_noise_size
        new_y = m_y + np.sin(m_yaw) * velocity_time + (np.random.random_sample() - 0.5) * y_noise_size
        new_yaw = yaw_to_quaternion(normalize_angle(m_yaw + yaw_time + (np.random.random_sample() - 0.5)*yaw_noise_size))
        marker_array.markers[i].pose.position.x = new_x
        marker_array.markers[i].pose.position.y = new_y
        marker_array.markers[i].pose.orientation = new_yaw


def image_callback(img_msg):
    global last_image_cb_stamp

    if marker_array is None:
        last_image_cb_stamp = img_msg.header.stamp
        return initialize_particle_cloud()

    if len(odom_msg_queue) != 2:
        return

    time_passed = (img_msg.header.stamp - last_image_cb_stamp).to_nsec() * 1.0
    propagate(time_passed)
    marker_array_pub.publish(marker_array)
    seen_red, seen_blue, seen_green, seen_purple = get_seen_angles(img_msg)

    x = 1
    y = 1
    yaw = 0

    yaw_quaternion = yaw_to_quaternion(yaw)
    odometry = Odometry()
    odometry.header.frame_id = 'odom'
    odometry.header.seq = img_msg.header.seq
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.orientation = yaw_quaternion
    odometry.header.stamp = rospy.Time.now()

    odom_pub.publish(odometry)
    last_image_cb_stamp = img_msg.header.stamp

def odom_callback(odom_msg):
    global odom_msg_queue

    unpacked_messages = [unpack_msg(x) for x in odom_msg_queue]
    angles_in_queue = [x[0][2] for x in unpacked_messages]
    if (len(odom_msg_queue) > 0 and
            (get_orientation_angle(odom_msg.pose.pose.orientation) in angles_in_queue)):
        # got ya, giving me the same message again...
        return

    odom_msg_queue = [odom_msg] + odom_msg_queue
    odom_msg_queue = odom_msg_queue[:2]


def yaw_to_quaternion(yaw):
    """ convert a yaw angle (in radians) into a Quaternion message """
    return Quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))

def get_orientation_angle(quaternion):
    return np.arccos(quaternion.w) * 2 * np.sign(quaternion.z)

def normalize_angle(angle):
    if angle > np.pi:
        return -2*np.pi + angle
    elif angle < -np.pi:
        return 2*np.pi + angle

    return angle

def init():
    global purple_dots_pub, green_dots_pub, red_dots_pub,\
        blue_dots_pub, all_dots_pub, recognized_pub, odom_pub, marker_array_pub

    rospy.init_node('particle_filter', anonymous=True)
    rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    purple_dots_pub = rospy.Publisher('/debug_image/magenta', Image, queue_size=10)
    green_dots_pub = rospy.Publisher('/debug_image/green', Image, queue_size=10)
    red_dots_pub = rospy.Publisher('/debug_image/red', Image, queue_size=10)
    blue_dots_pub = rospy.Publisher('/debug_image/blue', Image, queue_size=10)
    all_dots_pub = rospy.Publisher('/debug_image/all_colors', Image, queue_size=10)
    recognized_pub = rospy.Publisher('/debug_image/recognized', Image, queue_size=10)
    odom_pub = rospy.Publisher('/odom_gps', Odometry, queue_size=10)
    marker_array_pub = rospy.Publisher('/mcmarkerarray', MarkerArray, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    try:
        init()
    except rospy.ROSInterruptException:
        pass

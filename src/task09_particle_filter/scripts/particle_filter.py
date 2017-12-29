#!/usr/bin/env python

# x size
# 6m 1.6cm = 6.016
# y size
# 4m 1cm = 4.01

import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from helpers import yaw_to_quaternion, get_orientation_angle, normalize_angle, lv_sample, get_angle_difference
from beacons import get_lamp_coords
from odom import get_odom_velocity_and_yaw_change_speed
import math

# debugging publishers
purple_dots_pub = None
green_dots_pub = None
red_dots_pub = None
blue_dots_pub = None
all_dots_pub = None
recognized_pub = None

marker_array_pub = None
odom_pub = None
last_pos_and_orientation = None

# measured from real world field, this is the size of the carpet/track
field_x_size = 6.016
field_y_size = 4.01

# real_coords_red, real_coords_blue, real_coords_green, real_coords_purple
real_coords = np.array([[3.55, 3.03], [4.18, 1.77], [2.29, 1.14], [2.29, 2.4]])

x_noise_size = math.sqrt(0.0001)
y_noise_size = math.sqrt(0.0001)
yaw_noise_size = math.sqrt(np.pi / 256)

# copied from console, the logs of a measured angle (I think it was for red, but doesn't matter anyways)
# measured in the beginning of the bag file while the car still doesn't move
# we could do this in code, but wanted to save some time
# Note for submission: value used is 9.8823559825076556e-12 ^^
# angle_measurement_var = np.var(np.array([
#     -0.87893677600343323,
#     -0.87893677600343323,
#     -0.88289247655971048,
#     -0.88072588025077903,
#     -0.88072588025077903,
#     -0.88289247655971048,
#     -0.88072588025077903,
#     -0.87893677600343323,
#     -0.88072588025077903,
#     -0.88506681588861058,
#     -0.88072588025077903,
#     -0.87893677600343323,
#     -0.88072588025077903,
#     -0.87893677600343323,
#     -0.87893677600343323
# ]))

# work best, for 0.2 radians difference gives a weight of 94.6
angle_measurement_var_sq = (np.pi / 4) ** 2

odom_msg_queue = []
marker_array = None
last_image_cb_stamp = None


def get_angle_between_vectors(vec_1, vec_2):
    dot_product = vec_1.dot(vec_2)
    det = np.linalg.det(np.array([vec_1, vec_2]))
    return np.arctan2(det, dot_product)


# rotates a vector by angle in 2D space
def rotate_vector(vector, angle):
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return rot_matrix.dot(vector)


def create_n_markers(n, x, y, yaw_quaternion, id, mean_weight):
    markers = [None] * n

    markers[0] = create_marker(x, y, yaw_quaternion, id)
    for i in range(1, n):
        x_new = x + (np.random.random_sample() - 0.5) * x_noise_size * (1 - mean_weight) * 10
        y_new = y + (np.random.random_sample() - 0.5) * y_noise_size * (1 - mean_weight) * 10
        yaw_new = get_orientation_angle(yaw_quaternion) + (np.random.random_sample() - 0.5) * yaw_noise_size * (
        1 - mean_weight) * 2
        markers[i] = create_marker(x_new, y_new, yaw_to_quaternion(yaw_new), id + i)

    return markers


def create_marker(x, y, yaw_quaternion, id):
    marker = Marker()
    marker.header.frame_id = 'odom'
    marker.header.stamp = rospy.Time.now()

    marker.id = id
    marker.ns = 'point_cloud_marker'

    marker.scale.x = .5
    marker.scale.y = .1
    marker.scale.z = .1

    marker.color.a = 1
    marker.color.r = 0
    marker.color.g = 1
    marker.color.b = 0

    marker.type = 0

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.orientation = yaw_quaternion

    return marker


def initialize_particle_cloud():
    global marker_array

    marker_array = MarkerArray()

    for i in range(100):
        x, y, yaw_quaternion = get_random_marker_point()
        marker_array.markers.append(create_marker(x, y, yaw_quaternion, i))


def get_random_marker_point():
    x = np.random.random_sample() * field_x_size
    y = np.random.random_sample() * field_y_size
    yaw = np.random.random_sample() * 2 * np.pi - np.pi

    return x, y, yaw_to_quaternion(yaw)


def get_seen_angles(img_msg):
    publishers = red_dots_pub, green_dots_pub, blue_dots_pub, purple_dots_pub, all_dots_pub, recognized_pub

    # img_coords_red, img_coords_blue, img_coords_green, img_coords_purple, center
    img_coords_center = get_lamp_coords(img_msg, publishers)

    seen_angles = [get_angle_between_vectors(coord - img_coords_center[-1],
                                             np.array([0, 1])) for coord in img_coords_center[:-1]]

    return seen_angles


def propagate(time_passed):
    odom_velocity, yaw_change_speed = get_odom_velocity_and_yaw_change_speed(odom_msg_queue)
    velocity_time = odom_velocity * time_passed
    yaw_time = yaw_change_speed * time_passed

    for i, marker in enumerate(marker_array.markers):
        m_x = marker.pose.position.x
        m_y = marker.pose.position.y
        m_yaw = get_orientation_angle(marker.pose.orientation)

        new_x = m_x + np.cos(m_yaw) * velocity_time + (np.random.random_sample() - 0.5) * x_noise_size
        new_y = m_y + np.sin(m_yaw) * velocity_time + (np.random.random_sample() - 0.5) * y_noise_size
        new_yaw = yaw_to_quaternion(
            normalize_angle(m_yaw + yaw_time + (np.random.random_sample() - 0.5) * yaw_noise_size))
        marker_array.markers[i].pose.position.x = new_x
        marker_array.markers[i].pose.position.y = new_y
        marker_array.markers[i].pose.orientation = new_yaw


def get_marker_weights(img_msg):
    # seen_red, seen_blue, seen_green, seen_purple
    seen_angles = get_seen_angles(img_msg)
    marker_weights = []

    for i, marker in enumerate(marker_array.markers):
        # get angle between the vector pointing straight in from the position of a marker
        # towards its orientation and the vector from the marker to a beacon/light bulb

        marker_yaw = get_orientation_angle(marker.pose.orientation)
        marker_coords = marker.pose.position.x, marker.pose.position.y
        marker_orientation = marker_yaw + np.pi
        marker_orientation_vector = rotate_vector(np.array([0, 1]), marker_orientation)
        expected_angles = [get_angle_between_vectors(rc - marker_coords,
                                                     marker_orientation_vector) for rc in real_coords]

        # calculate weight for each light bulb using e^(- (expected - perceived)^2/standardDeviation^2)
        weights = np.array([np.exp(
            -0.5 * ((get_angle_difference(expected, perceived)) / angle_measurement_var_sq) ** 2
        ) for expected, perceived in list(zip(expected_angles, seen_angles))])

        # rospy.loginfo('e: {}; s: {}; w: {}'.format(expected_angles, seen_angles, weights))

        # remove NaN weights for unseen light bulbs
        weights = weights[np.invert(np.isnan(weights))]

        # multiple weights (for multiple light bulbs should be multiplied together)
        marker_weights.append(weights.prod())

    marker_weights = np.array(marker_weights)
    mean_weight = marker_weights.mean()
    # normalize between 0 and 1
    marker_weights = marker_weights / marker_weights.sum()

    return marker_weights, mean_weight


def get_most_probable_markers(img_msg):
    marker_weights, mean_weight = get_marker_weights(img_msg)

    for marker in marker_array.markers:
        marker.color.a = 0

    sampled_elements, sampled_weights = lv_sample(marker_array.markers, marker_weights, 20)

    return sampled_elements, sampled_weights, mean_weight


def next_marker_evolution(sampled_elements, sampled_weights, mean_weight):
    marker_array.markers = list(np.array([create_n_markers(5, s.pose.position.x, s.pose.position.y,
                                                           s.pose.orientation, s.id + i, mean_weight) for i, s in
                                          enumerate(sampled_elements)]).flatten())


def get_pos_and_orientation(markers):
    # grid size is size of field * 10, so in decimeters, double field to allow overflows
    grid = np.zeros((int(field_x_size * 20), int(field_y_size * 20)))

    # matrix, where for each location we have a 2d matrix, where the rows are the
    # individual positions, both with x and y
    # that way, we can easily find the center of gravity with center_of_gravity.mean(axis=2)
    # we defined 100 possible locations, will later on remove the unused ones
    centers_of_gravity = np.zeros((grid.shape[0], grid.shape[1], 100, 2))
    # fill matrix with NaN so we can filter it later on
    centers_of_gravity = centers_of_gravity * float('inf')
    yaw_cos = 0
    yaw_sin = 0

    for marker in markers:
        x = marker.pose.position.x
        y = marker.pose.position.y
        yaw = get_orientation_angle(marker.pose.orientation)

        x_idx = int(x * 10)
        y_idx = int(y * 10)

        if 0 <= x_idx < grid.shape[0] and 0 <= y_idx < grid.shape[1]:
            grid[x_idx][y_idx] += 1
            centers_of_gravity[x_idx][y_idx] = x, y
            yaw_cos += np.cos(yaw)
            yaw_sin += np.sin(yaw)

    winning_cell = np.unravel_index(grid.argmax(), grid.shape)

    x, y = centers_of_gravity[winning_cell].mean(0)
    yaw = np.arctan2(yaw_sin, yaw_cos)

    return x, y, yaw


def image_callback(img_msg):
    global last_image_cb_stamp, last_pos_and_orientation

    if marker_array is None:
        last_image_cb_stamp = img_msg.header.stamp
        initialize_particle_cloud()
        return marker_array_pub.publish(marker_array)

    if len(odom_msg_queue) != 2:
        return

    time_passed = (img_msg.header.stamp - last_image_cb_stamp).to_nsec() * 1.0
    propagate(time_passed)
    # TODO: maybe use marker_weights to determine when alg. converges
    sample_markers, sample_weights, mean_weight = get_most_probable_markers(img_msg)
    x, y, yaw = get_pos_and_orientation(sample_markers)
    next_marker_evolution(sample_markers, sample_weights, mean_weight)
    marker_array_pub.publish(marker_array)
    last_image_cb_stamp = img_msg.header.stamp

    if last_pos_and_orientation is None:
        last_pos_and_orientation = x, y, yaw
        return

    # rospy.loginfo('x: {}; y: {}'.format(x, y))

    yaw_quaternion = yaw_to_quaternion(yaw)
    odometry = Odometry()
    odometry.header.frame_id = 'odom'
    odometry.header.seq = img_msg.header.seq
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.orientation = yaw_quaternion
    odometry.header.stamp = rospy.Time.now()

    odom_pub.publish(odometry)


def odom_callback(odom_msg):
    global odom_msg_queue

    if len(odom_msg_queue) > 0:
        last_stamp = odom_msg_queue[0].header.stamp
        current_stamp = odom_msg.header.stamp
        time_passed = abs((current_stamp - last_stamp).to_nsec())

        if time_passed < 0.2 * 10 ** 9:
            # if we take messages too often, position and angle doesn't change
            # at all, so it's hard to estimate next position
            return

    odom_msg_queue = [odom_msg] + odom_msg_queue
    odom_msg_queue = odom_msg_queue[:2]


def init():
    global purple_dots_pub, green_dots_pub, red_dots_pub, \
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

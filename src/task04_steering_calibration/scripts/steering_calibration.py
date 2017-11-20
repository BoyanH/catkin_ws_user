#!/usr/bin/env python

# --- imports ---
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16
import math
import numpy as np
import json
from sklearn.linear_model import LinearRegression


########################################################################################################################
# ------------------------- HELPER METHODS, NOT IMPORTANT FOR CURRENT TASK ------------------------------------------- #
########################################################################################################################

def range_idx_to_angle(idx, angle_increment, angle_min, angle_max):
    angles = np.arange(angle_min, angle_max, angle_increment)
    return angles[idx]


def angle_to_ranges_idx(angle, angle_increment, angle_min, angle_max):
    # should return index of closest angle to the given one, rounded up
    angles = list(np.arange(angle_min, angle_max + angle_increment, angle_increment))
    last_angle = angles[0]
    for idx, linar_angle in enumerate(angles):
        if linar_angle >= angle:
            if abs(last_angle - linar_angle) < abs(angle - linar_angle):
                return idx - 1
            return idx


def find_smallest_dist_and_idx_in_ranges(ranges):
    min_distance = INFINITY
    idx_of_min = -1

    for idx, distance in enumerate(ranges):
        if distance < min_distance:
            min_distance = distance
            idx_of_min = idx

    return min_distance, idx_of_min


def find_perpendicular_to_wall(scan_msg):
    # start in the range of [-pi/2, pi/2] and find the shortest distance
    # according to our LaserScan data, this should correspond to all angles in the range
    # of to 90deg left and right of the car

    # that means, as the angles go from -pi to pi, we must take the middle 2 quarters of the range

    ranges_len = len(scan_msg.ranges)
    search_in_ranges = [
        scan_msg.ranges[:ranges_len / 4 + 1],  # + 1 as 2. index is exclusive
        scan_msg.ranges[ranges_len * 3 / 4:]
    ]
    smallest_distances_and_angles = list(map(lambda x: find_smallest_dist_and_idx_in_ranges(x), search_in_ranges))
    distance_to_wall, range_idx = min(smallest_distances_and_angles, key=lambda x: x[0])

    if distance_to_wall == smallest_distances_and_angles[1][0]:
        range_idx += range_idx * 3 / 4

    angle_of_perpendicular = range_idx_to_angle(range_idx, scan_msg.angle_increment,
                                                scan_msg.angle_min, scan_msg.angle_max)

    return distance_to_wall, angle_of_perpendicular, range_idx


def get_angle_indexes_and_distance(scan_msg):
    distance_from_wall, angle_of_perpendicular, range_idx = find_perpendicular_to_wall(scan_msg)

    angle_left = angle_of_perpendicular + (-1 if angle_of_perpendicular < 0 else 1) * ALPHA_IN_RAD
    angle_right = angle_of_perpendicular - (-1 if angle_of_perpendicular < 0 else 1) * ALPHA_IN_RAD

    if angle_left < scan_msg.angle_min:
        angle_left = scan_msg.angle_max + (angle_left - scan_msg.angle_min)
    if angle_right > scan_msg.angle_max:
        angle_right = scan_msg.angle_min + (angle_right - scan_msg.angle_max)

    idx_angle_left = angle_to_ranges_idx(angle_left, scan_msg.angle_increment,
                                         scan_msg.angle_min, scan_msg.angle_max)
    idx_angle_right = angle_to_ranges_idx(angle_right, scan_msg.angle_increment,
                                          scan_msg.angle_min, scan_msg.angle_max)

    return idx_angle_left, idx_angle_right, distance_from_wall


########################################################################################################################
# ------------------------------- END OF HELPER METHODS AND EXERCISE IRRELATIVE METHODS ------------------------------ #
########################################################################################################################


INFINITY = float('inf')
DISTANCE_BETWEEN_AXES = 0.50
ALPHA_ANGLE = 10
ALPHA_IN_RAD = ALPHA_ANGLE * math.pi / 180
left_angle_idx = None
right_angle_idx = None
initial_distance_left = None
initial_distance_right = None
initial_distance_to_wall = None

waiting_for_initial_measurement = False
waiting_for_final_measurement = False
last_published_speed = None

initialized = True
steering_angle = None

calibration_angles = [0, 30, 60, 90, 120, 150, 179]
expected_vs_received = {}

map_steering = lambda x: x


# unfortunately the bag data that I have isn't the best for e.g. angle = 150deg
# so take the best you can...
def get_range_safe(ranges, idx):
    _idx_try = None
    _idx = idx
    while (not ranges[_idx] < INFINITY) and _idx < len(ranges) - 1:
        _idx += 1

    _idx_try = _idx

    _idx = idx
    while (not ranges[_idx] < INFINITY) and _idx > 1:
        _idx -= 1

    if abs(_idx - idx) < abs(_idx_try - idx):
        return ranges[_idx]

    return ranges[_idx_try]


def measure_initial(scan_msg):
    global left_angle_idx, right_angle_idx, initial_distance_left, initial_distance_right, initial_distance_to_wall

    # we could also assume -pi + alpha, pi + alpha or something for the 2 initial angles
    # but we thought it might be helpful to get them relatively to a perpendicular line
    # which we find as the shortmeasure_finalest distance in a given range (where we expect the wall to be)

    rospy.loginfo('initial measurement')

    left_angle_idx, right_angle_idx, initial_distance_to_wall = get_angle_indexes_and_distance(scan_msg)
    initial_distance_left = get_range_safe(scan_msg.ranges, left_angle_idx)
    initial_distance_right = get_range_safe(scan_msg.ranges, right_angle_idx)


def measure_final(scan_msg):
    global waiting_for_final_measurement, expected_vs_received
    rospy.loginfo('final measurement')

    distance_l = get_range_safe(scan_msg.ranges, left_angle_idx)
    distance_r = get_range_safe(scan_msg.ranges, right_angle_idx)

    # if, although we used get_range_safe, some range is still infinity, just abort and wait for the next
    # message and hope it is better...
    if not (distance_l < INFINITY and distance_r < INFINITY):
        waiting_for_final_measurement = True
        return

    distance_between_measuring_points_squared = distance_l ** 2 + distance_r ** 2 - 2 * distance_l * distance_r * math.cos(
        ALPHA_IN_RAD * 2)
    distance_between_measuring_points = math.sqrt(distance_between_measuring_points_squared)

    # a / sin a = b / sin b
    # => distance_r / sin_angle_between_dl_and_wall = distance_between_measuring_points / sin(2*ALPHA_IN_RAD)
    sin_angle_between_dl_and_wall = distance_r * math.sin(ALPHA_IN_RAD * 2) / distance_between_measuring_points
    angle_between_dl_and_wall = math.asin(sin_angle_between_dl_and_wall)

    distance_to_wall = sin_angle_between_dl_and_wall * distance_l  # FIXME: formula on sketch is wrong!

    # until now I could easily take find_perpendicular_to_wall() ... ^^

    angle_between_distances_plus_orthogonal = math.pi / 2 - angle_between_dl_and_wall
    angle_between_center_of_car_and_orthogonal = angle_between_distances_plus_orthogonal - ALPHA_IN_RAD

    # Sketch 3 & 4

    curve_radius = (distance_to_wall - initial_distance_to_wall) / math.sin(angle_between_center_of_car_and_orthogonal)

    rospy.loginfo('initial distance to wall: {}'.format(initial_distance_to_wall))
    rospy.loginfo('distance to wall: {}'.format(distance_to_wall))
    rospy.loginfo('distance between measuring points (where lidar bounces of): {}'.format(
        distance_between_measuring_points))
    rospy.loginfo('Curve radius: {}'.format(curve_radius))
    rospy.loginfo('angle_between_center_of_car_and_orthogonal: {}'.format(
        angle_between_center_of_car_and_orthogonal * 180 / math.pi))
    # we assume the car's distance between the front and the rear axle is
    # DISTANCE_BETWEEN_AXES m, unfortunately we have no better data
    turning_angle = math.asin(DISTANCE_BETWEEN_AXES / curve_radius)

    turning_angle_degrees = turning_angle * 180 / math.pi

    # we find the angle relative to the orthogonal, we need to map it now into the range [0, 180)
    if distance_l > distance_r:
        turning_angle_degrees = 90 - turning_angle_degrees
    else:
        turning_angle_degrees = 90 + turning_angle_degrees

    rospy.loginfo('Expected turning angle: {}; Turning angle: {}'.format(steering_angle, turning_angle_degrees))

    expected_vs_received[steering_angle] = turning_angle_degrees

    rospy.loginfo('Experiments finished: {}/{}; Results: {}'.format(len(expected_vs_received.keys()),
                                                                    len(calibration_angles), expected_vs_received))
    if len(expected_vs_received.keys()) == len(calibration_angles):
        with open('calibration.json', 'w') as f:
            f.write(json.dumps(expected_vs_received, ensure_ascii=False))


def scan_callback(scan_msg):
    global waiting_for_initial_measurement, waiting_for_final_measurement

    if waiting_for_initial_measurement:
        waiting_for_initial_measurement = False
        measure_initial(scan_msg)
    elif waiting_for_final_measurement:
        waiting_for_final_measurement = False
        rospy.sleep(1)  # wait for car to stop
        measure_final(scan_msg)


# I didn't have enough time to finish the task in the lab, so I took 7 rosbags home
# therefore, I am using the speed topic to tell me when to measure
# I find this approach very cool, as one can do it with bags or with the car, so I left it like this
def speed_callback(msg):
    global initialized, waiting_for_initial_measurement, waiting_for_final_measurement, last_published_speed

    if not initialized:
        initialized = True
        return start_calibration()

    if last_published_speed == msg.data:
        return

    last_published_speed = msg.data

    if last_published_speed == 0:
        waiting_for_final_measurement = True
    else:
        waiting_for_initial_measurement = True


def steering_callback(msg):
    global steering_angle
    steering_angle = msg.data


# NOTE: didn't test this behavior, as I could only get bags from the lab
# I recreated it from the script I used for recording
def start_calibration():
    rate = rospy.Rate(10)  # 10hz
    angles = calibration_angles

    for angle in angles:
        for i in range(10):
            pub_steering.publish(angle)
            rate.sleep()
        for i in range(50):
            pub_speed.publish(-200)  # backwards, 200
            pub_steering.publish(angle)
            rate.sleep()
        for i in range(10):
            pub_speed.publish(0)
            rate.sleep()

        # make sure everything is straight before next experiment
        for i in range(10):
            pub_steering.publish(90)
            rate.sleep()


def prepare_mapping_function():
    global expected_vs_received, map_steering

    # can also read from saved json
    if len(expected_vs_received.keys()) == 0:
        with open('calibration.json') as json_data:
            expected_vs_received = json.load(json_data)

    rospy.loginfo(expected_vs_received)
    lm = LinearRegression()
    y_train = list(map(lambda x: float(x), expected_vs_received.keys()))
    X_train = list(map(lambda x: [float(expected_vs_received[x])], expected_vs_received.keys()))

    rospy.loginfo(y_train)
    rospy.loginfo(X_train)
    lm.fit(X_train, y_train)

    map_steering = lambda x: lm.predict(x)[0]


# --- main ---
rospy.init_node("steering_calibration")

rospy.Subscriber("scan", LaserScan, scan_callback, queue_size=100)
rospy.Subscriber("/manual_control/speed", Int16, speed_callback, queue_size=100)
rospy.Subscriber("/manual_control/steering", Int16, steering_callback, queue_size=100)

pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100)
pub_steering = rospy.Publisher("/manual_control/steering", Int16, queue_size=100)

prepare_mapping_function()

rospy.loginfo(map_steering(104))


rospy.spin()

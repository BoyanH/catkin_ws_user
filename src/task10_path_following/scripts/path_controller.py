#!/usr/bin/env python

import os
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int16
import math

first_run = True

steer_pub = None
speed_pub = None
start_stop_pub = None
dir = os.path.dirname(os.path.abspath(__file__))
lane1_matrix = np.load(os.path.join(dir, './matrixDynamic_lane2.npy'))
lane2_matrix = np.load(os.path.join(dir, './matrixDynamic_lane2.npy'))
meter2res_factor = 10

forward_speed = 350
backward_speed = -350
calibrated_forward_angle = 100

kp = 4

speed = None
speed_change_threshold = 0.1


def get_calibrated_steering(angle):
    ackerman_angles = [-45, 0, 45]
    steering_angles = [0, 100, 170]

    return np.interp(angle, ackerman_angles, steering_angles)


def get_orientation_angle(quaternion):
    return np.arccos(quaternion.w) * 2 * np.sign(quaternion.z)


def get_x_y_orientation(odom_msg):
    pose = odom_msg.pose.pose
    position = pose.position
    orientation = pose.orientation

    return position.x, position.y, get_orientation_angle(orientation)


def unpack_msg(odom_msg):
    return get_x_y_orientation(odom_msg)


def get_steering_speed_fxy_car_map(x, y, yaw):
    global speed
    # rospy.loginfo('x: {}; y: {}'.format(x, y))

    x_in_map = int(np.round(x * meter2res_factor))
    y_in_map = int(np.round(y * meter2res_factor))

    if x_in_map < 0:
        x_in_map = 0
    elif x_in_map >= lane1_matrix.shape[0]:
        x_in_map = lane1_matrix.shape[0] - 1

    if y_in_map < 0:
        y_in_map = 0
    elif y_in_map >= lane1_matrix.shape[1]:
        y_in_map = lane1_matrix.shape[1] - 1

    f_x_map, f_y_map = lane1_matrix[x_in_map, y_in_map, :]
    f_x_car = np.cos(yaw) * f_x_map + np.sin(yaw) * f_y_map
    f_y_car = - np.sin(yaw) * f_x_map + np.cos(yaw) * f_y_map

    if (speed is None and f_x_car >= 0) or f_x_car > speed_change_threshold:
        speed = forward_speed
    elif (speed is None and f_x_car < 0) or f_x_car < -speed_change_threshold:
        speed = backward_speed

    steering = kp * np.arctan(f_y_car / (2.5 * f_x_car))

    if steering > np.pi / 4:
        steering = np.pi / 4
    elif steering < - np.pi / 4:
        steering = - np.pi / 4

    if speed == backward_speed:
        if f_y_car > 0:
            steering = -np.pi / 4
        if f_y_car < 0:
            steering = np.pi / 4

    control_steering = get_calibrated_steering(np.array([steering * 180 / np.pi]))[0]

    return control_steering, speed, f_x_car, f_y_car, f_x_map, f_y_map, steering


def get_steering_and_speed(x, y, yaw):
    return get_steering_speed_fxy_car_map(x, y, yaw)[:2]


def kalman_callback(odom_msg):
    global speed, first_run

    rospy.loginfo('in callback')

    # to plot circle
    # if first_run:
    #     first_run = False
    # else:
    #     return

    x, y, yaw = unpack_msg(odom_msg)

    control_steering, speed = get_steering_and_speed(x, y, yaw)

    start_stop_pub.publish(0)
    steer_pub.publish(control_steering)
    speed_pub.publish(speed)

    rospy.loginfo('here')

    # rospy.loginfo('f_x: {}'.format(f_x_car))
    # rospy.loginfo('steering: {}'.format(control_steering))
    # rospy.loginfo('f_x: {}; f_y: {}'.format(f_x, f_y))


def init():
    global steer_pub, speed_pub, start_stop_pub

    rospy.init_node("path_controller", anonymous=True)

    rospy.Subscriber('/odom_gps', Odometry, kalman_callback)
    steer_pub = rospy.Publisher('/manual_control/steering', Int16, queue_size=10)
    speed_pub = rospy.Publisher('/manual_control/speed', Int16, queue_size=10)
    start_stop_pub = rospy.Publisher('/manual_control/stop_start', Int16, queue_size=10)


if __name__ == '__main__':
    try:
        init()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("interrupt")

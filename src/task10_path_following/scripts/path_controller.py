#!/usr/bin/env python

import os
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int16
import math

first_run = True

steer_pub = None
speed_pub = None
start_stop_pub = None
dir = os.path.dirname(os.path.abspath(__file__))
lane1_matrix = np.load(os.path.join(dir, './matrixDynamic_lane1.npy'))
lane2_matrix = np.load(os.path.join(dir, './matrixDynamic_lane2.npy'))
meter2res_factor = 10

forward_speed = -150
backward_speed = 150
calibrated_forward_angle = 100

kp = 6
max_steering_angle = np.pi / 4

speed = None
speed_change_threshold = 0.1


first_speed = None
first_steering = None

ackerman_angles = [45, 0, -45]
steering_angles = [0, 100, 179]
# polynomial of 3rd degree to best map the steering angle
# as explained in the lectures, autos tend to have a more precise steering near the straight ahead angle,
# so more like
#
#                                 *
#                                 *
#                                *
#                              *
#                            *
#                      * * *
#                   *
#                 *
#                *
#                *   Well at least I tried ^^, note my awesome ASCII function drawing skills
steer_map_p = np.poly1d(np.polyfit(ackerman_angles, steering_angles, 3))


def get_calibrated_steering(angle):
    return steer_map_p(angle)


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

    # formula from assignment, kp unchanged (for now)
    steering = kp * np.arctan(f_y_car / (2.5 * f_x_car))

    # the car can only steer max_steering_angle amount
    # so make sure we don't give any other commands
    if steering > max_steering_angle:
        steering = max_steering_angle
    elif steering < - max_steering_angle:
        steering = - max_steering_angle

    if speed == backward_speed:
        if f_y_car > 0:
            steering = -max_steering_angle
        if f_y_car < 0:
            steering = max_steering_angle

    # should get a mapping to the steering angle of the car
    # we used linear interpolation to map [-45, 45] to [0,179]
    # also added a point 0, 100 as the car we used was heading straight with a control angle of 100
    # note: if the function is using something else than 100, probably we changed the car at some point :D
    control_steering = get_calibrated_steering(np.degrees(steering))
    return control_steering, speed, f_x_car, f_y_car, f_x_map, f_y_map, steering


def get_steering_and_speed(x, y, yaw):
    return get_steering_speed_fxy_car_map(x, y, yaw)[:2]


def kalman_callback(odom_msg):
    global speed, first_run, first_speed, first_steering

    rospy.loginfo('in callback')

    if first_run:
        first_run = False
        start_stop_pub.publish(0)
    else:
        # pass

        # to plot circle
        steer_pub.publish(first_steering)
        speed_pub.publish(first_speed)
        return

    x, y, yaw = unpack_msg(odom_msg)

    control_steering, speed = get_steering_and_speed(x, y, yaw)

    first_speed = speed
    first_steering = control_steering

    steer_pub.publish(control_steering)
    speed_pub.publish(speed)

    rospy.loginfo('controlling; speed: {}; angle: {}'.format(speed, control_steering))

    # rospy.loginfo('f_x: {}'.format(f_x_car))
    # rospy.loginfo('steering: {}'.format(control_steering))
    # rospy.loginfo('f_x: {}; f_y: {}'.format(f_x, f_y))


def init():
    global steer_pub, speed_pub, start_stop_pub

    rospy.init_node("path_controller", anonymous=True)

    rospy.Subscriber('/assignment6/odom', Odometry, kalman_callback)
    steer_pub = rospy.Publisher('/manual_control/steering', Int16, queue_size=10)
    speed_pub = rospy.Publisher('/manual_control/speed', Int16, queue_size=10)
    start_stop_pub = rospy.Publisher('/manual_control/stop_start', Int16, queue_size=10)


if __name__ == '__main__':
    try:
        init()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("interrupt")

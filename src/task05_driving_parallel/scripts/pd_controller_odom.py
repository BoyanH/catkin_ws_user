#!/usr/bin/env python

import rospy
import sys
import math
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16
# import matplotlib.pyplot as plt

experiment_started = False
desired_y = None
desired_y_delta = 0.2
initial_time = None
last_steering_delta = 0
Kp = 0.6
Kd = 0.6

def get_calibrated_steering():
    return 100


def odom_callback(odom_msg):
    pose = odom_msg.pose.pose
    position = pose.position  # x, y, z
    current_y = position.y
    # orientation = pose.orientation  # x, y, z, w

    global initial_time, desired_y, last_steering_delta

    if desired_y is None:
        desired_y = current_y + desired_y_delta

    calibrated_angle = get_calibrated_steering(90)
    steering_delta = current_y - desired_y
    steer_angle = Kp * steering_delta + + Kd * (steering_delta - last_steering_delta) + calibrated_angle
    last_steering_delta = steering_delta

    if initial_time is None:
        initial_time = rospy.get_time()
        start_experiment()
    elif rospy.get_time() - initial_time > 10:
        pub_speed.publish(0)
        rospy.signal_shutdown("done")

    pub_steering.publish(steer_angle)
    pub_y_coord.publish(current_y)
    start_experiment()


def start_experiment():
    global experiment_started

    if experiment_started:
        return

    experiment_started = True

    pub_stop_start.publish(0)
    rospy.sleep(1)
    pub_speed.publish(-300)
    rospy.sleep(1)


# --- main ---
rospy.init_node("pd_controller")

rospy.loginfo('started')

rospy.Subscriber("/odom", Odometry, odom_callback, queue_size=100)

pub_y_coord = rospy.Publisher("/log/y_coord", Int16, queue_size=100)
pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100)
pub_stop_start = rospy.Publisher("/manual_control/stop_start", Int16, queue_size=100)
pub_steering = rospy.Publisher("/manual_control/steering", Int16, queue_size=100)
rospy.sleep(1)

rospy.spin()

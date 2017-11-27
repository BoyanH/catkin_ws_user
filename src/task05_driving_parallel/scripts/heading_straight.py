#!/usr/bin/env python

import rospy
import sys
import math
import numpy as np
from std_msgs.msg import Float32, Int16
# import matplotlib.pyplot as plt

experiment_started = False
squared_errors = []
times = []
Kp = -5
firstCall = True
desired_angle = 0


def get_calibrated_steering():
    return 100


def yaw_callback(yaw_msg):
    global squared_errors
    global desired_angle
    global firstCall

    if firstCall:
        desired_angle = yaw_msg.data + 20
        firstCall = False

    current_angle = yaw_msg.data
    #rospy.loginfo('current: {}; desired: {}'.format(current_angle, desired_angle))
    calibrated_angle = get_calibrated_steering()
    squared_error = (desired_angle - current_angle) ** 2
    pub_squared_error.publish(squared_error)
    squared_errors.append(squared_error)

    current_time = rospy.get_time()
    times.append(current_time)

    if current_time - times[0] >= 13:
        squared_errors_avg = np.array(squared_errors).mean()
        pub_speed.publish(0)
        rospy.loginfo('average squared error: {}'.format(squared_errors_avg))
        rospy.signal_shutdown("done")

    steering = Kp * (desired_angle - current_angle) + calibrated_angle
    rospy.loginfo(steering)
    pub_steering.publish(steering)
    start_experiment()


def start_experiment():
    global experiment_started

    if experiment_started:
        return

    experiment_started = True

    pub_stop_start.publish(0)
    rospy.sleep(1)
    pub_steering.publish(get_calibrated_steering())
    rospy.sleep(1)
    pub_speed.publish(-300)
    rospy.sleep(1)


# --- main ---
rospy.init_node("p_controller")

rospy.loginfo('started')

rospy.Subscriber("/model_car/yaw", Float32, yaw_callback, queue_size=100)

pub_squared_error = rospy.Publisher("/squared_error", Int16, queue_size=100)
pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100)
pub_stop_start = rospy.Publisher("/manual_control/stop_start", Int16, queue_size=100)
pub_steering = rospy.Publisher("/manual_control/steering", Int16, queue_size=100)
rospy.sleep(1)

rospy.spin()

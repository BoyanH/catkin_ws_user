#!/usr/bin/env python

import rospy
import sys
import math
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16
# import matplotlib.pyplot as plt

experiment_started = False
times = []

def get_calibrated_steering():
    return 100


def odom_callback(odom_msg):
    #steering = Kp * (desired_angle - current_angle) + calibrated_angle


    #stop after 10 seconds
    current_time = rospy.get_time()
    times.append(current_time)

    if current_time - times[0] >= 13:
        pub_speed.publish(0)
        rospy.signal_shutdown("done")


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
rospy.init_node("pd_controller")

rospy.loginfo('started')

rospy.Subscriber("/odom", Odometry, odom_callback, queue_size=100)

#pub_squared_error = rospy.Publisher("/squared_error", Int16, queue_size=100)
pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100)
pub_stop_start = rospy.Publisher("/manual_control/stop_start", Int16, queue_size=100)
pub_steering = rospy.Publisher("/manual_control/steering", Int16, queue_size=100)
rospy.sleep(1)

rospy.spin()

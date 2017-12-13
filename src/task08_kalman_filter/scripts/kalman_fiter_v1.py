#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
import math

odom_kalman_pub = None
odom_velocity = None
odom_msg_queue = []
last_kalman_odom = None

w_sensory_x = 0.5
w_sensory_y = 0.5
w_sensory_yaw = 0.5


def get_orientation_angle(quaternion):
    return np.arccos(quaternion.w) * 2 * np.sign(quaternion.z)


def get_x_y_orientation(odom_msg):
    pose = odom_msg.pose.pose
    position = pose.position
    orientation = pose.orientation

    return position.x, position.y, get_orientation_angle(orientation)


def unpack_msg(odom_msg):
    return get_x_y_orientation(odom_msg), odom_msg.header.stamp


def yaw_to_quaternion(yaw):
    """ convert a yaw angle (in radians) into a Quaternion message """
    return Quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))


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


def odom_gps_callback(odom_msg):
    global last_kalman_odom

    odom_velocity, yaw_change_speed = get_odom_velocity_and_yaw_change_speed()

    if odom_velocity is None or last_kalman_odom is None:
        kalman_odom = Odometry()
        kalman_odom.pose = odom_msg.pose
        last_kalman_odom = kalman_odom
        last_kalman_odom.header.stamp = rospy.Time.now()
        return odom_kalman_pub.publish()

    # apply kalman filter
    (x_gps, y_gps, yaw_gps), time_gps = unpack_msg(odom_msg)
    (last_k_x, last_k_y, last_k_yaw), last_k_time = unpack_msg(last_kalman_odom)

    time_passed = (time_gps - last_k_time).to_nsec()
    velocity_time = odom_velocity * time_passed
    predicted_x = last_k_x + np.cos(last_k_yaw) * velocity_time
    predicted_y = last_k_y + np.sin(last_k_yaw) * velocity_time
    predicted_yaw = last_k_yaw + (yaw_change_speed * time_passed)

    kalman_x = w_sensory_x * x_gps + ((1 - w_sensory_x) * predicted_x)
    kalman_y = w_sensory_y * y_gps + ((1 - w_sensory_y) * predicted_y)
    kalman_yaw = w_sensory_yaw * yaw_gps + ((1 - w_sensory_yaw) * predicted_yaw)

    kalman_odom = Odometry()
    kalman_odom.header.frame_id = 'odom'
    kalman_odom.header.seq = odom_msg.header.seq
    kalman_odom.pose.pose.position.x = kalman_x
    kalman_odom.pose.pose.position.y = kalman_y
    kalman_odom.pose.pose.orientation = yaw_to_quaternion(kalman_yaw)
    kalman_odom.header.stamp = rospy.Time.now()

    last_kalman_odom = kalman_odom
    odom_kalman_pub.publish(kalman_odom)


def init():
    global odom_kalman_pub

    rospy.init_node('kalman_filter_v1', anonymous=True)
    rospy.Subscriber('/odom_gps', Odometry, odom_gps_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    odom_kalman_pub = rospy.Publisher('/odom_kalman_v1', Odometry, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    try:
        init()
    except rospy.ROSInterruptException:
        pass

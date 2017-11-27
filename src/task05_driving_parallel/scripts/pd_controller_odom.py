#!/usr/bin/env python

import rospy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16, Float32

experiment_started = False
initial_y = None
desired_y_delta = 0.2
initial_time = None
last_steering_delta = 0
Kp = 80
Kd = 600
dist_axles = 0.3
lookahead_distance = 0.5
desired_dist_wall = 0.4

current_yaw = 0
initial_yaw = None


def get_calibrated_steering(angle):
    return 72


def odom_callback(odom_msg):
    pose = odom_msg.pose.pose
    position = pose.position  # x, y, z
    current_y = position.y
    # orientation = pose.orientation  # x, y, z, w

    # rospy.loginfo('y: {}'.format(current_y))
    # rospy.loginfo('x: {}'.format(position.x))
    global initial_time, initial_y, last_steering_delta

    if initial_y is None:
        initial_y = current_y

    calibrated_angle = get_calibrated_steering(90)
    distance = current_y - initial_y
    rospy.loginfo('distance: {}'.format(distance))
    steering_delta = get_delta_heading_from_dis_and_angle(0.5 + distance, current_yaw, dist_axles,
                                                          lookahead_distance, 0.5 + 0.2)
    # rospy.loginfo('steering delta: {}'.format(steering_delta))
    steer_angle = Kp * steering_delta + Kd * (steering_delta - last_steering_delta)*10 + calibrated_angle
    last_steering_delta = steering_delta

    if initial_time is None:
        initial_time = rospy.get_time()
        start_experiment()
    elif rospy.get_time() - initial_time > 15:
        pub_speed.publish(0)
        rospy.signal_shutdown("done")

    if steer_angle < 0:
        steer_angle = 0
    elif steer_angle > 179:
        steer_angle = 179

    # rospy.loginfo('yaw: {}'.format(current_yaw))
    pub_steering.publish(steer_angle - current_yaw)
    pub_y_coord.publish(current_y)


def get_delta_heading_from_dis_and_angle(dist_wall, curve_angle, dist_axles, dist_lookahead, desired_dist_wall):
    center_axis_y = dist_wall + math.sin(curve_angle) * dist_axles
    # rospy.loginfo('curve angle: {}'.format(curve_angle * 180/math.pi))
    return float(math.atan((desired_dist_wall - center_axis_y) / dist_lookahead))


def yaw_callback(yaw_msg):
    global current_yaw, initial_yaw

    returned_yaw = yaw_msg.data
    # rospy.loginfo(returned_yaw)

    if initial_yaw is None:
        initial_yaw = returned_yaw

    calculated_yaw = returned_yaw - initial_yaw
    if calculated_yaw < -math.pi:
        calculated_yaw = 2 * math.pi - calculated_yaw

    if calculated_yaw != 0:
        current_yaw = calculated_yaw


def start_experiment():
    rospy.sleep(1)
    for i in range(10):
        pub_stop_start.publish(0)
        pub_speed.publish(-200)



# --- main ---
rospy.init_node("pd_controller")

rospy.loginfo('started')

rospy.Subscriber("/odom", Odometry, odom_callback, queue_size=100)

pub_y_coord = rospy.Publisher("/log/y_coord", Int16, queue_size=100)
pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100)
pub_stop_start = rospy.Publisher("/manual_control/stop_start", Int16, queue_size=100)
pub_steering = rospy.Publisher("/manual_control/steering", Int16, queue_size=100)
pub_steer_delta = rospy.Publisher("/logging/steer_delta", Float32, queue_size=100)
rospy.Subscriber("/model_car/yaw", Float32, yaw_callback, queue_size=100)
rospy.sleep(1)

rospy.spin()

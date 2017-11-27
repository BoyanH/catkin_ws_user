#!/usr/bin/env python

# --- imports ---
import rospy
from sensor_msgs.msg import LaserScan
import math
import sys

from std_msgs.msg import Int16, Float32


# --- global variables

# on car 104
# 90deg is right wall, 180deg is behind, 270 deg is left, 360 is front
angle_far = 110
angle_near = 70
# in radians
angle_between_measurements = (angle_far - angle_near) * math.pi / 180
dist_axles = 0.3
lookahead_distance = 0.5
desired_dist_wall = 0.4
inited = False
Kp = 2.5
Kd = 1 # TODO: adjust, when Kp adjustment is ready
initial_time = None
last_steering_delta = 0
infinity = float('inf')

# --- definitions ---


def measure_distances(ranges):
    return float(ranges[angle_near]), float(ranges[angle_far])


def measure_distance_and_angle(dist_l2, dist_r2, angle_between_measurements):
    pub_left.publish(dist_l2 * 100)
    pub_right.publish(dist_r2 * 100)
    # rospy.loginfo('distance left: {}; distance right: {}'.format(dist_l2, dist_r2))
    # Law of cosines: a**2 = b**2 + c**2 - 2bc*cos(alpha)
    a = math.sqrt(dist_l2 ** 2 + dist_r2 ** 2 - 2 * dist_l2 * dist_r2 * math.cos(angle_between_measurements))
    # Law of sines: sin(alpha)/a = sin(gamma)/c -> sin(alpha)*c/a = sin(gamma)
    angle_gamma = math.asin((math.sin(angle_between_measurements) * dist_r2) / a)

    distance_to_wall = math.sin(angle_gamma) * dist_l2
    theta_l2 = math.pi/2 - angle_gamma

    curve_angle = theta_l2 - angle_between_measurements / 2

    pub_distance.publish(distance_to_wall * 100) # hope this distance is meant to be plotted, measured from rear axle
    return distance_to_wall, curve_angle


def get_delta_heading(scan_msg):
    ranges = scan_msg.ranges

    # get distances
    dist_l2, dist_r2 = measure_distances(ranges)

    if not (dist_l2 < infinity and dist_r2 < infinity):
        return None

    dist_to_wall, curve_angle = measure_distance_and_angle(dist_l2, dist_r2, angle_between_measurements)
    delta_heading = get_delta_heading_from_dis_and_angle(dist_to_wall, curve_angle, dist_axles,
                                                lookahead_distance, desired_dist_wall)

    return delta_heading


def get_delta_heading_from_dis_and_angle(dist_wall, curve_angle, dist_axles, dist_lookahead, desired_dist_wall):
    center_axis_y = dist_wall + math.sin(curve_angle) * dist_axles
    # rospy.loginfo('curve angle: {}'.format(curve_angle * 180/math.pi))
    return float(math.atan((desired_dist_wall - center_axis_y) / dist_lookahead))


def scan_callback(scan_msg):
    global inited, initial_time, last_steering_delta

    if not inited:
        inited = True
        start_experiment()
        initial_time = rospy.get_time()
    elif rospy.get_time() - initial_time > 20:
        pub_speed.publish(0)
        # rospy.signal_shutdown("done")

    calibrated_angle = get_calibrated_steering(90)
    delta_heading = get_delta_heading(scan_msg)

    if delta_heading is None:
        return

    delta_in_deg = delta_heading * 180 / math.pi

    pub_theta.publish(delta_in_deg)

    # if car is pointing away from the wall (as in sketch) angle will be negative, we need to steer right though
    steering_delta = delta_in_deg

    if math.isnan(delta_heading):
        return

    # rospy.loginfo(delta_in_deg)
    steer_angle = Kp * steering_delta + Kd*(steering_delta - last_steering_delta) + calibrated_angle

    if steer_angle > 179:
        steer_angle = 179
    elif steer_angle < 0:
        steer_angle = 0

    # rospy.loginfo(steer_angle)
    last_steering_delta = steering_delta
    pub_steer.publish(steer_angle)


def start_experiment():
    rospy.sleep(1)
    for i in range(10):
        pub_unlock.publish(0)
        pub_speed.publish(-250)


def get_calibrated_steering(angle):
    return 100


if __name__ == "__main__":
    rospy.init_node("driving_parallel")

    pub_steer = rospy.Publisher("/manual_control/steering", Int16, queue_size=100)
    pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100)
    pub_unlock = rospy.Publisher("/manual_control/stop_start", Int16, queue_size=100)
    pub_distance = rospy.Publisher("/logging/distance", Float32, queue_size=100)
    pub_theta = rospy.Publisher("/logging/angle_theta", Float32, queue_size=100)


    pub_left = rospy.Publisher("/logging/left", Float32, queue_size=100)
    pub_right = rospy.Publisher("/logging/right", Float32, queue_size=100)



    rospy.Subscriber("scan", LaserScan, scan_callback, queue_size=100)
    rospy.spin()

import numpy as np
from helpers import get_orientation_angle


def get_odom_velocity_and_yaw_change_speed(odom_msg_queue):
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

    if yaw_diff < -np.pi:
        yaw_diff = 2 * np.pi + yaw_diff
    elif yaw_diff > np.pi:
        yaw_diff = -2 * np.pi + yaw_diff

    yaw_change_speed = yaw_diff / time_passed

    return velocity, yaw_change_speed


def unpack_msg(odom_msg):
    return get_x_y_orientation(odom_msg), odom_msg.header.stamp


def get_x_y_orientation(odom_msg):
    pose = odom_msg.pose.pose
    position = pose.position
    orientation = pose.orientation

    return position.x, position.y, get_orientation_angle(orientation)

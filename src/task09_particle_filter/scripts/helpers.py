import math
import numpy as np
from geometry_msgs.msg import Quaternion
import rospy


def yaw_to_quaternion(yaw):
    """ convert a yaw angle (in radians) into a Quaternion message """
    return Quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))


def get_orientation_angle(quaternion):
    return np.arccos(quaternion.w) * 2 * np.sign(quaternion.z)


def get_angle_difference(angle_a, angle_b):
    return np.pi - abs(abs(angle_a - angle_b) - np.pi)


def normalize_angle(angle):
    if angle > np.pi:
        return -2 * np.pi + angle
    elif angle < -np.pi:
        return 2 * np.pi + angle

    return angle


def lv_sample(elements, weights, k):
    M = k
    r = np.random.random_sample() * 1.0/M
    sampled = []
    sampled_weights = []
    weights_sum = 0
    el_idx = -1

    for m in range(M):
        new_weight = r + m * (1.0/M)

        while weights_sum < new_weight:
            el_idx += 1
            weights_sum += weights[el_idx]

        # i = el_idx % len(elements)

        sampled.append(elements[el_idx])
        sampled_weights.append(weights[el_idx])

    # should converge to the percentage of particles sampled
    # rospy.loginfo('ow: {}; sw: {}'.format(np.array(weights).sum(), np.array(sampled_weights).sum()))

    return sampled, sampled_weights

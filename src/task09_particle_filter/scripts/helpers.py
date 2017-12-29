import math
import numpy as np
from geometry_msgs.msg import Quaternion
import copy


def yaw_to_quaternion(yaw):
    """ convert a yaw angle (in radians) into a Quaternion message """
    return Quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))


def get_orientation_angle(quaternion):
    return np.arccos(quaternion.w) * 2 * np.sign(quaternion.z)


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
        new_weight = r + (1.0/M) * m

        while weights_sum < new_weight or weights_sum == 0:
            weights_sum += weights[el_idx]
            el_idx += 1

        i = el_idx % len(elements)
        sampled.append(copy.copy(elements[i]))
        sampled_weights.append(weights[i])

    return sampled, sampled_weights

import unittest
from driving_parallel import get_delta_heading_from_dis_and_angle, measure_distance_and_angle
import math

class driving_parallel_spec(unittest.TestCase):
    def test_get_delta_heading_from_dis_and_angle(self):
        dist_to_wall = 0.6
        curve_angle = 0.2
        dist_axles = 0.2
        dist_lookahead = 0.5
        desired_dist_wall = 0.4
        result = get_delta_heading_from_dis_and_angle(dist_to_wall, curve_angle, dist_axles,
                                                      dist_lookahead, desired_dist_wall)
        expected_result = math.atan((0.4 - 0.64) / 0.5)
        difference = abs(expected_result - result)

        self.assertTrue(difference < 0.0005)

    def test_measure_distance_and_angle(self):
        dist_l2, dist_r2 = 0.6, 0.4
        angle_between_measurements = 40 * (math.pi / 180)
        result_dist, result_angle = measure_distance_and_angle(dist_l2, dist_r2, angle_between_measurements)
        expected_dist, expected_angle = 0.3953, 0.5025
        diff_dist = abs(result_dist - expected_dist)
        diff_angle = abs(result_angle - expected_angle)

        self.assertTrue(diff_dist < 0.0005)
        self.assertTrue(diff_angle < 0.0005)


ds = driving_parallel_spec()
ds.test_get_delta_heading_from_dis_and_angle()
ds.test_measure_distance_and_angle()

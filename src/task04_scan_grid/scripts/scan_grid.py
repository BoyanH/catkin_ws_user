#!/usr/bin/env python

# --- imports ---
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import math
import numpy as np

OCCUPANCY_FREE = 1
OCCUPANCY_UNKNOWN = 50
OCCUPANCY_OCCUPIED = 100
INFINITY = float('inf')
FLAG_NO_OBSTACLE_MEANS_FREE = True


# --- definitions ---
def reset_grid():
    global occupancy_grid

    for i in range(len(occupancy_grid.data)):
        if FLAG_NO_OBSTACLE_MEANS_FREE:   # set all values to "FREE"
            occupancy_grid.data[i] = OCCUPANCY_FREE
        else:  # set all values to "UNKNOWN"
            occupancy_grid.data[i] = OCCUPANCY_UNKNOWN


# to a given cartesian x,y coordinate, mark the corresponding cell in the grid as "OCCUPIED"
def set_cell(x, y, occupancy):
    global occupancy_grid

    res = occupancy_grid.info.resolution
    x_scaled = (x * 1.0 / res) + occupancy_grid.info.width / 2
    y_scaled = (y * 1.0 / res) + occupancy_grid.info.height / 2

    if x_scaled >= occupancy_grid.info.width or x_scaled < 0 or y_scaled >= occupancy_grid.info.height or y_scaled < 0:
        return

    offset = (int(round(x_scaled)) - 1) * occupancy_grid.info.height
    occupancy_grid.data[int(offset) + int(round(y_scaled) - 1)] = occupancy


def get_coords_from_angle_and_distance(angle, distance):
    x_offset = math.cos(angle) * distance
    y_offset = math.sin(angle) * distance

    # we noticed x grows correctly in the grid but y not, so we mirrored it
    return x_offset, -y_offset


def range_idx_to_angle(idx, angle_increment, angle_min, angle_max):
    angles = np.arange(angle_min, angle_max, angle_increment)
    return angles[idx]


def scan_callback(scan_msg):
    global occupancy_grid
    reset_grid()

    for idx, distance in enumerate(scan_msg.ranges):
        if distance > 0 and distance < INFINITY:
            angle = range_idx_to_angle(idx, scan_msg.angle_increment, scan_msg.angle_min, scan_msg.angle_max)
            # we rotate the given coordinate system 90 deg counter clock-wise
            # to understand the problem more easily (so 0 degrees is point to the right, 90deg is in front etc.)
            mapped_angle = angle - math.pi / 2

            res = occupancy_grid.info.resolution

            if FLAG_NO_OBSTACLE_MEANS_FREE:
                max_horizontal = occupancy_grid.info.width / 2
                max_vertical = occupancy_grid.info.height / 2
                max_distance = math.sqrt(max_horizontal**2 + max_vertical**2)
                longer_ranges = np.arange(distance + res, max_distance, res)
                for longer_range in longer_ranges:
                    x_l, y_l = get_coords_from_angle_and_distance(mapped_angle, longer_range)
                    set_cell(x_l, y_l, OCCUPANCY_UNKNOWN)
            else:
                shorter_ranges = np.arange(distance - res, 0, -res)
                for shorter_range in shorter_ranges:
                    x_s, y_s = get_coords_from_angle_and_distance(mapped_angle, shorter_range)
                    set_cell(x_s, y_s, OCCUPANCY_FREE)

            x, y = get_coords_from_angle_and_distance(mapped_angle, distance)
            set_cell(x, y, OCCUPANCY_OCCUPIED)

    pub_grid.publish(occupancy_grid)


# --- main ---
rospy.init_node("scan_grid")

# init occupancy grid
occupancy_grid = OccupancyGrid()
occupancy_grid.header.frame_id = "laser"
occupancy_grid.info.resolution = 0.03  # in m/cell

# width x height cells
occupancy_grid.info.width = 300
occupancy_grid.info.height = 300

# origin is shifted at half of cell size * resolution
occupancy_grid.info.origin.position.x = int(-1.0 * occupancy_grid.info.width / 2.0) * occupancy_grid.info.resolution
occupancy_grid.info.origin.position.y = int(-1.0 * occupancy_grid.info.height / 2.0) * occupancy_grid.info.resolution
occupancy_grid.info.origin.position.z = 0
occupancy_grid.info.origin.orientation.x = 0
occupancy_grid.info.origin.orientation.y = 0
occupancy_grid.info.origin.orientation.z = 0
occupancy_grid.info.origin.orientation.w = 1

occupancy_grid.data = range(occupancy_grid.info.width * occupancy_grid.info.height)

rospy.Subscriber("scan", LaserScan, scan_callback, queue_size=100)
pub_grid = rospy.Publisher("scan_grid", OccupancyGrid, queue_size=100)

rospy.spin()

#!/usr/bin/env python2
import numpy as np
import path_parser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree

from path_controller import get_steering_speed_fxy_car_map

map_size_x = 600  # cm
map_size_y = 400  # cm
resolution = 10  # cm
matrix = np.zeros((map_size_x // resolution, map_size_y // resolution, 2), dtype='f')
yaw = -np.pi / 3


def main(map_file):
    xy = np.array(list(path_parser.read_points(map_file)))
    x, y = xy.T

    fig = plt.figure(figsize=(12, 10), facecolor='w')
    plt.plot(x, y, ':o', markersize=2)
    global matrix
    matrix = np.load('matrixDynamic_lane2.npy')

    plt.gca().set_aspect(1, 'datalim')  # keep circles as circles
    plt.tight_layout()

    def show_nearest(target):

        global yaw
        yaw += np.pi / 3  # np.pi/2
        car_length = 0.3

        global matrix
        x1, y1 = target

        control_steering, speed, f_x, f_y, x3, y3, steering = get_steering_speed_fxy_car_map(x1, y1, yaw)

        r = car_length * np.abs(np.tan((np.pi) / 2 - steering))

        if (r > 10):
            r = 10
        print(r)
        if (steering < 0.0):
            r = -r
        xc = x1 - np.sin(yaw) * r
        yc = y1 + np.cos(yaw) * r

        ax = plt.axes()
        ax.arrow(x1, y1, car_length * np.cos(yaw), car_length * np.sin(yaw), width=car_length, head_width=car_length,
                 head_length=0.09, fc='b', ec='b')

        ax = plt.axes()
        ax.arrow(x1, y1, f_x * np.cos(yaw), f_x * np.sin(yaw), head_width=0.01, head_length=0.01, fc='r', ec='r')

        ax = plt.axes()
        ax.arrow(x1, y1, -f_y * np.sin(yaw), f_y * np.cos(yaw), head_width=0.01, head_length=0.01, fc='r', ec='r')

        ax = plt.axes()
        ax.arrow(x1, y1, x3, y3, head_width=0.01, head_length=0.01, fc='g', ec='g')

        plt.scatter(*target, color='r')
        plt.scatter(*(x1 + x3, y1 + y3), color='g')
        circ = plt.Circle((xc, yc), r, color='r', fill=False)
        plt.gcf().gca().add_artist(circ)
        plt.show(block=False)

    def onclick(event):
        show_nearest((event.xdata, event.ydata))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    main('sample_map_origin_map.txt')
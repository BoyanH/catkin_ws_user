import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
weight_h = 7
weight_s = 2
weight_v = 2
number_of_clusters = 4
inf = float('inf')
none_coords = np.array([inf, inf])


# for debugging purposes; in order to publish which pixels were recognized by the mask
def publish_img(img, publisher):
    transport_img = bridge.cv2_to_imgmsg(img, "rgb8")
    publisher.publish(transport_img)


def get_lamp_coords(img_msg, publishers):
    red_dots_pub, green_dots_pub, blue_dots_pub, purple_dots_pub, all_dots_pub, recognized_pub = publishers
    img_rgb = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define color ranges
    lower_red = np.array([171, 20, 100])  # done
    upper_red = np.array([180, 255, 255])

    lower_red2 = np.array([0, 20, 100])  # done
    upper_red2 = np.array([20, 255, 255])

    lower_blue = np.array([80, 210, 220])
    upper_blue = np.array([120, 255, 255])

    lower_green = np.array([21, 0, 50])  # done
    upper_green = np.array([79, 255, 255])

    lower_purple = np.array([125, 150, 250])
    upper_purple = np.array([170, 255, 255])

    # Get masks
    # for regions on image which have the same color as one of the lamps
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    mask_red = cv2.medianBlur(mask_red, 3)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red2 = cv2.medianBlur(mask_red2, 3)
    mask_red = np.bitwise_or(mask_red, mask_red2)

    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_green = cv2.medianBlur(mask_green, 3)

    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask_blue = cv2.medianBlur(mask_blue, 3)

    mask_purple = cv2.inRange(img_hsv, lower_purple, upper_purple)
    mask_purple = cv2.medianBlur(mask_purple, 3)

    # combine masks
    mask_all = np.bitwise_or(np.bitwise_or(np.bitwise_or(mask_red, mask_green), mask_blue), mask_purple)

    selected_red = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_red)
    publish_img(selected_red, red_dots_pub)

    selected_green = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_green)
    publish_img(selected_green, green_dots_pub)

    selected_blue = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_blue)
    publish_img(selected_blue, blue_dots_pub)

    selected_purple = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_purple)
    publish_img(selected_purple, purple_dots_pub)

    selected_all = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_all)
    publish_img(selected_all, all_dots_pub)

    # get clusters, 4 Cluster
    # calculate mean value from all (H, S, V) in the form of
    # [(H,S,V) for cluster 1, (H,S,V) for cluster 2, ...],

    # for each color:
    # compute [abs(targetH - clusterH)weightH + abs(targetS - clusterS)*weightS + abs(targetV - clusterV)*weightV,
    #           for cluster 2 also, for cluster 3 also...]
    # -> get min value from array -> this cluster is our winner -> get mean coordinates


    # using all the pixels which match with the color of any lamp, we can then cluster them all
    # and for each lamp in the real world pick the best fit from a cluster (cluster with mean color that best
    # fits the desired one) and return the mean x,y position of the cluster as location

    # !IMPORTANT: here we also use weights for h,s,v to determine how "close" one colour is to another

    masked_pixels = np.where(np.array(mask_all))
    # get all (x,y) pair of coordinates in a 2d array (matrix)
    masked_points_coordinates = np.array(list(zip(masked_pixels[1], masked_pixels[0])))

    clustered = KMeans(n_clusters=number_of_clusters).fit(masked_points_coordinates)

    points_per_cluster = [[] for x in range(number_of_clusters)]

    for idx, label in enumerate(clustered.labels_):
        points_per_cluster[label].append(masked_points_coordinates[idx])

    mean_colors_per_cluster = [get_mean_color(x, img_hsv) for x in points_per_cluster]
    expected_colors = [
        np.array([175, 190, 150]),
        np.array([110, 230, 230]),
        np.array([50, 250, 100]),
        np.array([140, 190, 255])
    ]
    img_coords = [get_coords_in_image(mean_colors_per_cluster,
                                      points_per_cluster, expected_color) for expected_color in expected_colors]

    for i, coord in enumerate(img_coords):
        for j in range(i + 1, len(img_coords)):
            if img_coords[i][0] != inf and img_coords[j][0] != inf and \
                            np.linalg.norm(img_coords[i] - img_coords[j]) <= 20:
                if (get_distance_to_cluster(img_hsv[img_coords[i][1]][img_coords[i][0]], expected_colors[i]) <
                        get_distance_to_cluster(img_hsv[img_coords[j][1]][img_coords[j][0]], expected_colors[j])):
                    img_coords[j] = none_coords
                else:
                    img_coords[i] = none_coords

    img_coords_red, img_coords_blue, img_coords_green, img_coords_purple = img_coords
    raw_colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (148, 0, 211)
    ]

    for idx, colour in enumerate(raw_colors):
        if img_coords[idx][0] != inf:
            cv2.circle(img_rgb, tuple(img_coords[idx]), 10, colour, 1)
    publish_img(img_rgb, recognized_pub)

    center = math.floor(len(img_hsv[0]) / 2), math.floor(len(img_hsv) / 2)

    return mirror_coords([img_coords_red, img_coords_blue, img_coords_green, img_coords_purple, center],
                         len(img_hsv[0]), len(img_hsv))


def get_coords_in_image(mean_cluster_colors, points_per_cluster, expected_color):
    distances_per_cluster = [get_distance_to_cluster(
        mean_cluster_colors[i], expected_color) for i in range(len(mean_cluster_colors))]
    winner_cluster = distances_per_cluster.index(min(distances_per_cluster))
    average_cluster_coords = np.array(points_per_cluster[winner_cluster]).mean(0)

    return np.array(average_cluster_coords, dtype=int)


def get_distance_to_cluster(mean_color, expected_color):
    return (min(abs(mean_color[0] - expected_color[0]), abs(mean_color[0] + expected_color[0]) % 180) * weight_h +
            abs(mean_color[1] - expected_color[1]) * weight_s +
            abs(mean_color[2] - expected_color[2]) * weight_v)


def get_mean_color(coordinates, img):
    # img = [ row, row, [col, col, [h, s, v], ..], ..]

    color_for_coordinates = np.array([img[x[1]][x[0]] for x in coordinates])
    mean_color = np.mean(color_for_coordinates, axis=0)
    return mean_color


def mirror_coords(coords, max_x, max_y):
    return np.array([[max_x - x, max_y - y] if x != inf else none_coords for x, y in coords])

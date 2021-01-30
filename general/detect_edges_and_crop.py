import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import time
import csv


def crop_image_and_label(path_image, path_mask):
    image_rgb = cv2.imread(path_image)
    label_rgb = cv2.imread(path_mask)
    # print(image_rgb.shape)
    limit_y = 0
    list_min_x = []
    list_max_x = []
    list_min_y = []
    for index_y in range(image_rgb.shape[0]):
        # print(image_rgb[index_y].shape)
        row = image_rgb[index_y]

        perimeter_points = [index_x for index_x, element in enumerate(row)
                            if element[0] != 0 and element[1] != 255 and element[2] != 0]

        if perimeter_points:
            list_min_y.append(index_y)
            limit_y = limit_y + 1
            list_min_x.append(min(perimeter_points))
            list_max_x.append(max(perimeter_points))

    min_x = min(list_min_x)
    min_y = min(list_min_y)
    max_x = max(list_max_x)

    crop_img = image_rgb[min_y:min_y + limit_y, min_x:max_x]
    crop_label = label_rgb[min_y:min_y + limit_y, min_x:max_x]

    return (crop_img, crop_label)


def crop_image(path_image):

    image_rgb = cv2.imread(path_image)
    limit_y = 0
    list_min_x = []
    list_max_x = []
    list_min_y = []
    for index_y in range(image_rgb.shape[0]):
        row = image_rgb[index_y]

        perimeter_points = [index_x for index_x, element in enumerate(row)
                            if element[0] != 0 and element[1] != 255 and element[2]
                            != 0]

        if perimeter_points:
            list_min_y.append(index_y)
            limit_y = limit_y + 1
            list_min_x.append(min(perimeter_points))
            list_max_x.append(max(perimeter_points))

    min_x = min(list_min_x)
    min_y = min(list_min_y)
    max_x = max(list_max_x)

    crop_img = image_rgb[min_y:min_y + limit_y, min_x:max_x]
    return crop_img


def detect_edge(path_image):
    image_rgb = cv2.imread(path_image)
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.01, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    init_time = time.time()

    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120)
    delta = time.time() - init_time
    print(delta)

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 255, 0)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (0, 255, 0)

    return image_rgb, delta

def prepare_data(path_images_folder, output_dir=''):

    image_list = [f for f in os.listdir(path_images_folder)
                  if os.path.isfile(os.path.join(path_images_folder, f))]
    time_stats = []
    for image in image_list:
        print(image)
        path_image = ''.join([path_images_folder, image])
        edge_detected, time = detect_edge(path_image)
        cv2.imwrite(''.join([path_images_folder, 'test/',
                             image]), edge_detected)
        time_stats.append(time)
        # modified_image = crop_image(path_image)

    csv_file_name = path_images_folder + 'time_test.csv'
    with open(name_performance_metrics_file, mode='w') as results_file:
        results_file_writer = csv.writer(csv_file_name, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_file_writer.writerow(time_stats)



def main():
    list_images_dir = '/home/nearlab/Jorge/data/cystoscopy/case_002/urs_pt6/'
    prepare_data(list_images_dir)

if __name__ == '__main__':
    main()
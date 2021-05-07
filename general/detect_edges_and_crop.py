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
import copy

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


def crop_image(image_array, min_x, min_y, max_x, max_y):
    space_y = 10
    space_x = 10
    limit_x, limit_y, z = np.shape(image_array)

    init_y = min_y - space_y
    init_x = min_x - space_x

    if init_y < 0:
       init_y = 0

    if init_x < 0:
       init_x = 0

    end_y = max_y + space_y
    end_x = max_x + space_x

    if end_y > limit_y:
       end_y = limit_y

    if end_x > limit_x:
       end_x = limit_x

    crop_img = image_array[init_y:end_y, init_x:end_x]

    return crop_img


def detect_border_and_crop_image(path_image):

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


def calculate_elipse_hough(image_gray, sigma, high_threshold):
    edges = canny(image_gray, sigma=sigma,
                  low_threshold=0.01, high_threshold=high_threshold)
    init_time = time.time()
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=350)
    delta = time.time() - init_time
    print(delta)
    return result, delta


def detect_edge(path_image):
    image_rgb = cv2.imread(path_image)
    image_gray = color.rgb2gray(image_rgb)
    high_threshold = 0.6
    sigma = 3.0
    result, delta = calculate_elipse_hough(image_gray, sigma, high_threshold)

    counter = 0
    while len(result) == 0 and counter < 3:
        sigma = sigma + 0.05
        high_threshold = high_threshold - 0.05
        result, delta = calculate_elipse_hough(image_gray, sigma, high_threshold)
        counter = counter + 1

    if counter == 3:
        image_rgb = np.zeros([2, 2, 2])
        cropped_image = image_rgb
        delta = 999

    else:

        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        copy_image_rgb = copy.copy(image_rgb)
        image_rgb[cy, cx] = (0, 255, 0)
        cropped_image = crop_image(copy_image_rgb, np.amin(cx), np.amin(cy),
                                np.amax(cx), np.amax(cy))

    return image_rgb, cropped_image, delta


def prepare_data(path_images_folder, output_dir_edges, output_dir_croped):

    image_list = sorted([f for f in os.listdir(path_images_folder)
                  if os.path.isfile(os.path.join(path_images_folder, f))])
    time_stats = []
    for image in image_list:
        print(image)
        path_image = ''.join([path_images_folder, image])
        edge_detected, cropped_img, time = detect_edge(path_image)
        time_stats.append(time)

        if not(np.any(edge_detected)):
            continue
        else:
            # save the detected edges image
            cv2.imwrite(''.join([output_dir_edges, image]), edge_detected)
            # save the croped images
            cv2.imwrite(''.join([cropped_img, image]), edge_detected)


    csv_file_name = path_images_folder + 'time_test.csv'

    with open(csv_file_name, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_file_writer.writerow(time_stats)


def main():
    list_images_dir = '/home/nearlab/Jorge/data/cystoscopy/case_001/frames/'
    output_dir_edges = '/home/nearlab/Jorge/DATASETS/fov_detection/cys/case_001/'
    output_dir_croped = '/home/nearlab/Jorge/DATASETS/tumor_classification/cys/cys_case_001/all/'

    prepare_data(list_images_dir, output_dir_edges, output_dir_croped)

if __name__ == '__main__':
    main()
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from skimage.feature import canny
import os
import time
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import csv


def detect_square(img):
    img_shape = 'none'
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0

    init_time = time.time()
    img = cv2.medianBlur(img, 5)
    edges = cv2.Canny(img, 5, 200, apertureSize=3)
    w, l, d = np.shape(img)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, 150)
    num_lines = 0
    x_values = []
    y_values = []
    if lines is not None:
        for i, element in enumerate(lines[:4]):
            rho = element[0][0]
            theta = element[0][1]
            #print('rho', rho, 'theta', theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = int(rho * a)
            y0 = int(rho * b)

            if theta > -5 * np.pi / 180 and theta < 5 * np.pi / 180:
                # if theta in range(-5, 5) or theta in range(-85, 55):
                num_lines += 1
                draw = True
                x2 = int(x0 - 100 * (-b))
                y2 = w

            if theta > 175 * np.pi / 180 and theta < 185 * np.pi / 180:
                # if theta in range(-5, 5) or theta in range(-85, 55):
                num_lines += 1
                draw = True
                x2 = int(x0 - 100 * (-b))
                y2 = w


            if theta > 85 * np.pi / 180 and theta < 95 * np.pi / 180:
                # if theta in range(-85, 95)
                num_lines += 1
                draw = True
                x2 = l
                y2 = int(y0 + 100 * (a))

            if draw is True:
                #cv2.line(img, (x0, y0), (x2, y2), (0, 255), 2)
                if x2 != l:
                    x_values.append(x2)

                if y2 != w:
                    y_values.append(y2)

        if num_lines == 4:
            img_shape = 'square'

            min_x = np.min(x_values)
            max_x = np.max(x_values)
            min_y = np.min(y_values)
            max_y = np.max(y_values)

        """plt.figure()
        plt.imshow(img)
        plt.show()"""
    delta = time.time() - init_time

    return img_shape, delta, min_x, min_y, max_x, max_y


def calculate_elipse_hough(image_rgb, sigma, high_threshold):
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    shape = 'none'
    edges = cv2.Canny(image_rgb, 5, 200, apertureSize=3)

    #gray_edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    w, l, d = np.shape(image_rgb)
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators

    init_time = time.time()
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=70, max_size=w)
    delta = time.time() - init_time

    if np.shape(result)[0] != 0:

        result.sort(order='accumulator')
        # Estimated parameters for the ellipse
        best = list(result[-1])

        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)

        #image_rgb[cy, cx] = (0, 255, 0)
        min_x = np.min(cx)
        min_y = np.min(cy)
        max_x = np.max(cx)
        max_y = np.max(cy)
        shape = 'ellipse'

        """plt.figure()
        plt.imshow(image_rgb)
        plt.show()"""

    return shape, delta, min_x, min_y, max_x, max_y


def detect_shape(image_rgb):

    img_shape, delta, min_x, min_y, max_x, max_y = detect_square(image_rgb)
    #if img_shape == 'none':
    #    high_threshold = 100
    #    sigma = 3
    #    img_shape, delta, min_x, min_y, max_x, max_y = \
    #        calculate_elipse_hough(image_rgb, sigma, high_threshold)

    return img_shape, delta, min_x, min_y, max_x, max_y


def in_min_range(min_val, padding):
    if min_val-padding >= 0:
        min_val = min_val - padding

    return min_val


def in_max_range(max_val, limit, padding):
    if max_val - padding <= limit:
        max_val = max_val + padding

    return max_val


def crop_img(image_rgb, min_x, min_y, max_x, max_y, padding=5):
    w, l, d = np.shape(image_rgb)

    min_x = in_min_range(min_x, padding)
    min_y = in_min_range(min_y, padding)
    max_x = in_max_range(max_x, w, padding)
    max_y = in_max_range(max_y, l, padding)

    cropped_img = image_rgb[min_y:max_y, min_x:max_x]

    return cropped_img


def prepare_data(data_dir, destination_dir):

    list_images = sorted([f for f in os.listdir(data_dir)
                  if os.path.isfile(os.path.join(data_dir, f))])
    print(len(list_images))
    list_images = [image for image in list_images if np.random.rand() <= 0.05]
    total = len(list_images)
    list_times = []
    list_shapes = []
    list_min_x = []
    list_min_y = []
    list_max_x = []
    list_max_y = []

    for j, img_name in enumerate(list_images):
        print('detecting:', img_name)
        img_dir = data_dir + img_name
        image_rgb = cv2.imread(img_dir)
        img_shape, delta, min_x, min_y, max_x, max_y = detect_shape(image_rgb)
        print(j, total, img_name, delta, img_shape,
              'min_x', min_x, 'min_y', min_y,
              'max_x', max_x, 'max_y', max_y)
        list_times.append(delta)
        list_shapes.append(img_shape)
        list_min_x.append(min_x)
        list_min_y.append(min_y)
        list_max_x.append(max_x)
        list_max_y.append(max_y)
        if img_shape != 'none':
            cropped_image = crop_img(image_rgb, min_x, min_y, max_x, max_y, padding=5)
            destination_dir_image = destination_dir + img_name
            cv2.imwrite(destination_dir_image, cropped_image)

    # save everything in a csv file
    name_file = ''.join([destination_dir, 'record.csv'])
    with open(name_file, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        headers = ['num', 'img_name', 'shape', 'min_x',
                   'min_y', 'max_x', 'max_y', 'time']
        results_file_writer.writerow(headers)
        for i, image_name in enumerate(list_images):
            results_file_writer.writerow([str(i), image_name,
                                          list_shapes[i], list_min_x[i],
                                          list_min_y[i], list_max_x[i],
                                          list_max_y[i], list_times[i]])

def main():
    original_data_dir = '/home/nearlab/Jorge/DATASETS/' \
               'lesion_classification/urs/urs_case_012/' \
               'frames_selected/lesion/'

    destination_dir = '/home/nearlab/Jorge/DATASETS/lesion_classification/urs/' \
                      'urs_case_012/lesion/'

    prepare_data(original_data_dir, destination_dir)


if __name__ == '__main__':
    main()

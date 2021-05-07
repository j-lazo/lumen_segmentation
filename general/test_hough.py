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
import imutils
import copy


def detect_circle(image_name):
    calculate_elipse_hough
    img = cv2.imread(image_name, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=45, maxRadius=80)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    plt.figure()
    plt.imshow(cimg)
    plt.show()

def detect_square(image_name):
    print(image_name)
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 200, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 360, 60)

    if lines is not None:
        list_1 = [value[0][0] for value in lines]
        list_2 = [value[0][1] for value in lines]


        for element in lines[:4]:
            rho = element[0][0]
            theta = element[0][1]
            print('rho', rho, 'theta', theta)
            draw = False
            if theta>-5*np.pi/180 and theta<5*np.pi/180:
            #if theta in range(-5, 5) or theta in range(-85, 55):
                draw = True

            if theta>85*np.pi/180 and theta<95*np.pi/180:
                draw = True

            if draw is True:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))


                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.figure()
        plt.subplot(121)
        plt.plot(list_1, list_2, '*')
        plt.subplot(122)
        plt.imshow(img)
        plt.show()
    
    
    

def calculate_elipse_hough(image_gray, sigma, high_threshold):
    edges = canny(image_gray, sigma=sigma,
                  low_threshold=0.01, high_threshold=high_threshold)
    plt.figure()
    plt.imshow(edges)
    plt.show()
    init_time = time.time()
    shape = detect_shape(edges)
    print(shape)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100)
    delta = time.time() - init_time
    print(type(result))
    print(np.shape(result))
    print(result[0])
    print(result)
    print(delta)
    return result, delta


def detect_shape(img_array_edges):

    #blurred = cv2.GaussianBlur(img_array_edges, (5, 5), 0)
    print(np.unique(img_array_edges))
    th, thresh = cv2.threshold(np.float32(img_array_edges), 10,
                               255, cv2.THRESH_BINARY)

    print(np.shape(thresh))
    print(np.unique(thresh))
    ## Find contours on the binary threshed image

    black_and_white = img_array_edges*255

    print(np.unique(black_and_white))
    plt.figure()
    plt.imshow(black_and_white)
    plt.show()
    cnts, hierarchy = cv2.findContours(np.float32(black_and_white),
                            cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)


    print(np.unique(thresh))
    plt.figure()
    plt.imshow(thresh)
    plt.show()

    cnts = imutils.grab_contours(cnts)

    shape = "unidentified"
    peri = cv2.arcLength(cnts, True)
    approx = cv2.approxPolyDP(cnts, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) > 6 and len(approx) < 12:
        shape = "octagon"
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
    # return the name of the shape
    return shape


def detect_edge(path_image):
    image_rgb = cv2.imread(path_image)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    high_threshold = 100
    sigma = 3
    result, delta = calculate_elipse_hough(image_gray, sigma, high_threshold)
    counter = 0
    """while len(result) == 0 or counter == 3:
        sigma = sigma - 0.1
        high_threshold = high_threshold - 0.1
        result, delta = calculate_elipse_hough(image_gray, sigma, high_threshold)
        counter = counter + 1"""

    # Estimated parameters for the ellipse
    if counter <= 3:
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)

        image_rgb[:, np.min(cx)-15] = (255, 0, 0)
        image_rgb[:, np.max(cx)+15] = (255, 0, 0)
        image_rgb[np.min(cy)-15, :] = (255, 0, 0)
        image_rgb[np.max(cy)+15, :] = (255, 0, 0)

        image_rgb[cy, cx] = (0, 255, 0)


    return image_rgb, delta


def prepare_data(path_images_folder, output_dir=''):

    image_list = sorted([f for f in os.listdir(path_images_folder)
                  if os.path.isfile(os.path.join(path_images_folder, f))])
    time_stats = []
    for image in image_list:
        print(image)
        detect_circle(path_images_folder + image)
        """path_image = ''.join([path_images_folder, image])
        edge_detected, time = detect_edge(path_image)
        cv2.imwrite(''.join([path_images_folder, 'test/',
                             image]), edge_detected)
        time_stats.append(time)"""
        # modified_image = crop_image(path_image)

    csv_file_name = path_images_folder + 'time_test.csv'
    with open(csv_file_name, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_file_writer.writerow(time_stats)



def main():
    list_images_dir = '/home/nearlab/Desktop/test_shapes/'
    prepare_data(list_images_dir)

if __name__ == '__main__':
    main()
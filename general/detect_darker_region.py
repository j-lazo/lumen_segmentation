import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import copy
import time


def determine_convex_hull(contours):

    x_points = []
    y_points = []
    point_sets = np.asarray(contours[0])


    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hulls = []
    new_hulls = []
    new_contours = []
    temp_contours = []
    # calculate points for each contour
    zeros = np.zeros((500, 500))

    #plt.figure()
    points_x = []
    points_y = []
    #for i, contour in enumerate(set):
    #    print(set)
    #    cv2.drawContours(zeros, set, i, (0, 255, 255), -1)  # as opencv stores in BGR format

    if len(point_sets) > 1:
        temp_contours = point_sets[0]
        for i in range(1, len(point_sets)):
            temp_contours = np.concatenate((temp_contours, point_sets[i]), axis=0)

        new_contours.append(temp_contours)
    else:
        new_contours = point_sets

    for i in range(len(new_contours)):
        new_hulls.append(cv2.convexHull(new_contours[i], False))

    if new_hulls == []:
        point_x = 'old'
        point_y = 'old'
    else:
        M = cv2.moments(new_hulls[0])
        if M["m00"] < 0.001:
            M["m00"] = 0.001

        point_x = int(M["m10"] / M["m00"])
        point_y = int(M["m01"] / M["m00"])

    return new_hulls, point_x, point_y


def show_histograms(image, mask):


    n_bins = 256
    grayscale = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = np.multiply(grayscale, mask)

    list_values_grayscale = [value for row in grayscale for value in row if value != 0]

    histogram_gray, bin_edges_gray = np.histogram(list_values_grayscale, bins=n_bins, range=(np.min(list_values_grayscale), np.max(list_values_grayscale)))

    red = image[:, :, 2].copy()
    blue = image[:, :, 0].copy()
    image[:, :, 0] = red
    image[:, :, 2] = blue


    # create the histogram plot, with three lines, one for
    # each color

    red_ch = np.multiply(image[:, :, 0], mask)
    blue_ch = np.multiply(image[:, :, 1], mask)
    green_ch = np.multiply(image[:, :, 2], mask)

    list_values_red = [value for row in red_ch for value in row if value != 0]
    list_values_green = [value for row in blue_ch for value in row if value != 0]
    list_values_blue = [value for row in green_ch for value in row if value != 0]

    histogram_red, bin_edges_red = np.histogram(list_values_red, bins=n_bins, range=(np.min(list_values_red), np.max(list_values_red)))
    histogram_green, bin_edges_green = np.histogram(list_values_green, bins=n_bins, range=(np.min(list_values_green), np.max(list_values_green)))
    histogram_blue, bin_edges_blue = np.histogram(list_values_blue, bins=n_bins, range=(np.min(list_values_blue), np.max(list_values_blue)))

    plt.figure(figsize=(18, 7))
    plt.subplot(2, 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2, 5, 2)
    plt.imshow(grayscale, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(2, 5, 3)
    plt.imshow(red_ch, cmap='Reds', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(2, 5, 4)
    plt.imshow(blue_ch, cmap='Greens', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(2, 5, 5)
    plt.imshow(green_ch, cmap='Blues', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(2, 5, 6)
    plt.imshow(mask)
    plt.axis('off')
    plt.subplot(2, 5, 7)
    plt.plot(bin_edges_gray[0:-1], histogram_gray)
    plt.subplot(2, 5, 8)
    plt.plot(bin_edges_red[0:-1], histogram_red)
    plt.subplot(2, 5, 9)
    plt.plot(bin_edges_green[0:-1], histogram_green)
    plt.subplot(2, 5, 10)
    plt.plot(bin_edges_blue[0:-1], histogram_blue)

    plt.show()


def build_contours(array_of_points):

    contours = []
    for i, y_points in enumerate(array_of_points[0]):
        point = (array_of_points[1][i], y_points)
        point = np.asarray(point)
        contours.append([point])

    return contours


def show_histograms_and_center(image, mask, image_name):

    save_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
               'lumen_data/video_test/painted_frames/darK_center_phantom_003/'

    if not (np.all(mask == 0)):

        percentage = 0.6
        n_bins = 256
        grayscale = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = np.multiply(grayscale, mask)

        list_values_grayscale = [value for row in grayscale for value in row if value != 0]
        histogram_gray, bin_edges_gray = np.histogram(list_values_grayscale, bins=n_bins, range=(
        np.min(list_values_grayscale), np.max(list_values_grayscale)))

        red = image[:, :, 2].copy()
        blue = image[:, :, 0].copy()
        image[:, :, 0] = red
        image[:, :, 2] = blue

        # create the histogram plot, with three lines, one for
        # each color

        red_ch = np.multiply(image[:, :, 0], mask)
        green_ch = np.multiply(image[:, :, 1], mask)
        blue_ch = np.multiply(image[:, :, 2], mask)

        list_values_red = [value for row in red_ch for value in row if value != 0]
        list_values_green = [value for row in blue_ch for value in row if value != 0]
        list_values_blue = [value for row in green_ch for value in row if value != 0]

        histogram_red, bin_edges_red = np.histogram(list_values_red, bins=n_bins,
                                                    range=(np.min(list_values_red), np.max(list_values_red)))
        histogram_green, bin_edges_green = np.histogram(list_values_green, bins=n_bins,
                                                        range=(np.min(list_values_green), np.max(list_values_green)))
        histogram_blue, bin_edges_blue = np.histogram(list_values_blue, bins=n_bins,
                                                      range=(np.min(list_values_blue), np.max(list_values_blue)))

        max_grays = ((np.where(grayscale >= int(percentage * np.amax(grayscale)))))
        max_reds = ((np.where(red_ch >= int(percentage * np.amax(list_values_red)))))
        max_greens = ((np.where(green_ch >= int(percentage * np.amax(list_values_green)))))
        max_blues = ((np.where(blue_ch >= int(percentage * np.amax(list_values_blue)))))

        w, h, d = np.shape(image)

        gray_contours = np.asarray([build_contours(max_grays)])
        gray_convex_hull, gray_x, gray_y = determine_convex_hull(gray_contours)

        red_contours = np.asarray([build_contours(max_reds)])
        red_convex_hull, red_x, red_y = determine_convex_hull(red_contours)

        green_contours = np.asarray([build_contours(max_greens)])
        green_convex_hull, green_x, green_y = determine_convex_hull(green_contours)

        blue_contours = np.asarray([build_contours(max_blues)])
        blue_convex_hull, blue_x, blue_y = determine_convex_hull(blue_contours)

        points_x = []
        points_y = []
        for hull in gray_convex_hull:
            for i, point in enumerate(hull):
                points_x.append(point[0][0])
                points_y.append(point[0][1])

    else:
        max_grays = []
        max_reds = []
        max_greens = []
        max_blues = []

        points_x = []
        points_y = []

        bin_edges_gray = []
        bin_edges_red = []
        bin_edges_green = []
        bin_edges_blue = []

        histogram_gray = []
        histogram_red = []
        histogram_green = []
        histogram_blue = []
        gray_x = 'nAN'
        red_x = 'nAN'
        green_x = 'nAN'
        blue_x = 'nAN'
        gray_y = 'nAN'
        red_y = 'nAN'
        green_y = 'nAN'
        blue_y = 'nAN'

    plt.figure(figsize=(18, 7))
    plt.subplot(2, 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2, 5, 2)
    plt.imshow(cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)

    if type(gray_x) != str:
        plt.plot(max_grays[1], max_grays[0], '*')
        plt.plot([int(w / 2), gray_x], [int(h / 2), gray_y], color='yellow')
        plt.plot(gray_x, gray_y, marker='X', color='yellow')
        plt.plot(points_x, points_y, '*-')
    plt.axis('off')

    plt.subplot(2, 5, 3)
    plt.imshow(image[:, :, 0], cmap='Reds', vmin=0, vmax=255)

    if type(red_x) != str:
        plt.plot(max_reds[1], max_reds[0], 'ro')
        plt.plot([int(w / 2), red_x], [int(h / 2), red_y], color='yellow')
        plt.plot(red_x, red_y, marker='X', color='yellow')
    plt.axis('off')

    plt.subplot(2, 5, 4)
    plt.imshow(image[:, :, 1], cmap='Greens', vmin=0, vmax=255)

    if type(green_x) != str:
        plt.plot(max_greens[1], max_greens[0], 'go')
        plt.plot([int(w / 2), green_x], [int(h / 2), green_y], color='yellow')
        plt.plot(green_x, green_y, marker='X', color='yellow')
    plt.axis('off')

    plt.subplot(2, 5, 5)
    plt.imshow(image[:, :, 2], cmap='Blues', vmin=0, vmax=255)

    if type(blue_x) != str:
        plt.plot(max_blues[1], max_blues[0], 'bo')
        plt.plot([int(w / 2), blue_x], [int(h / 2), blue_y], color='yellow')
        plt.plot(blue_x, blue_y, marker='X', color='yellow')
    plt.axis('off')

    plt.subplot(2, 5, 6)
    plt.imshow(mask)
    plt.axis('off')
    plt.subplot(2, 5, 7)
    if histogram_gray != []:
        plt.plot(bin_edges_gray[0:-1], histogram_gray)
    plt.subplot(2, 5, 8)
    if histogram_red != []:
        plt.plot(bin_edges_red[0:-1], histogram_red)
    plt.subplot(2, 5, 9)
    if histogram_green != []:
        plt.plot(bin_edges_green[0:-1], histogram_green)
    plt.subplot(2, 5, 10)
    if histogram_blue != []:
        plt.plot(bin_edges_blue[0:-1], histogram_blue)
    # plt.show()
    plt.savefig(save_dir + image_name)
    plt.close()


    return [gray_x, red_x, green_x, blue_x], [gray_y, red_y, green_y, blue_y]


def clean_mask(mask, plot_result=False):
    # remove the small areas appearing in the image if there is a considerable big one
    w, d = np.shape(mask)

    print(type(mask))
    print(np.shape(mask))
    print(np.unique(mask))
    plt.figure()
    plt.imshow(mask)
    plt.show()

    areas = []
    remove_small_areas = False
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_TREE, cv2.cvStartFindContours_Impl)

    for contour in contours:
        areas.append(cv2.contourArea(contour))

    if len(contours) > 1:
        # sort the areas from bigger to smaller
        sorted_areas = sorted(areas, reverse=True)
        index_remove = np.ones(len(areas))
        for i in range(len(sorted_areas)-1):
            # if an area is 1/4 smaller than the bigger area, mark to remove
            if sorted_areas[i+1] < 0.15 * sorted_areas[0]:
                index_remove[areas.index(sorted_areas[i+1])] = 0
                remove_small_areas = True

    if remove_small_areas is True:
        #new_mask = np.zeros((w, d))
        new_mask = copy.copy(mask)
        for index, remove in enumerate(index_remove):
            if remove == 0:
                # replace the small areas with 0
                cv2.drawContours(new_mask, contours, index, (0, 0, 0), -1)  # as opencv stores in BGR format
    else:
        new_mask = mask
    if plot_result is True:
        plt.figure()
        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(new_mask)
        plt.show()
    return new_mask


def detect_dark_region(image, mask, image_name):
    w_mask, h_mask, d_mask = np.shape(mask)
    image = cv2.resize(image, (w_mask, h_mask), interpolation=cv2.INTER_AREA)
    mask = cv2.cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cl_mask = clean_mask(mask)
    gt_mask = np.zeros((w_mask, h_mask))
    gt_mask[cl_mask == 255] = 1
    point_x, point_y = show_histograms_and_center(image, gt_mask, image_name)

    return point_x, point_y


def check_point_list(c_list):

    fixed_list = copy.copy(c_list)
    if type(c_list[0]) == str:
        fixed_list[0] = 0.0

    for i in range(1, len(fixed_list)):
        if type(fixed_list[i]) == str:
            fixed_list[i] = fixed_list[i-1]

    return fixed_list


def main():
    #mask_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
    #          'lumen_data/video_test/result_masks/phantom_001_pt2/phantom_001_pt_02_0315.png'
    #img_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
    #          'lumen_data/video_test/phantom_001_pt2/phantom_001_pt_02_0315.png'
    masks_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/video_test/result_masks/phantom_003/'
    imgs_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/video_test/phantom_003/'
    list_imgs = sorted(os.listdir(imgs_dir))
    list_masks = sorted(os.listdir(masks_dir))

    gray_points_x = []
    red_points_x = []
    green_points_x = []
    blue_points_x = []

    gray_points_y = []
    red_points_y = []
    green_points_y = []
    blue_points_y = []

    for j, image_name in enumerate(list_imgs[:]):
        print(j, image_name)
        image_dir = imgs_dir + image_name
        mask_dir = masks_dir + image_name
        mask = cv2.imread(mask_dir)
        image = cv2.imread(image_dir)
        point_x, point_y = detect_dark_region(image, mask, image_name)

        gray_points_x.append(point_x[0])
        red_points_x.append(point_x[1])
        green_points_x.append(point_x[2])
        blue_points_x.append(point_x[3])

        gray_points_y.append(point_y[0])
        red_points_y.append(point_y[1])
        green_points_y.append(point_y[2])
        blue_points_y.append(point_y[3])

    gray_points_x = check_point_list(gray_points_x)
    gray_points_y = check_point_list(gray_points_y)
    red_points_x = check_point_list(red_points_x)
    red_points_y = check_point_list(red_points_y)
    green_points_x = check_point_list(green_points_x)
    green_points_y = check_point_list(green_points_y)
    blue_points_x = check_point_list(blue_points_x)
    blue_points_y = check_point_list(blue_points_y)

    plt.figure()
    plt.subplot(4, 2, 1)
    plt.title('Gray')
    plt.plot(gray_points_x, '-o', color='grey')
    plt.ylabel('$\Delta y$')
    plt.subplot(4, 2, 2)
    plt.plot(gray_points_y, '-o', color='grey')
    plt.ylabel('$\Delta x$')

    plt.subplot(4, 2, 3)
    plt.title('Red')
    plt.plot(red_points_x, '-o', color='red')
    plt.ylabel('$\Delta y$')
    plt.subplot(4, 2, 4)
    plt.plot(red_points_y, '-o', color='red')
    plt.ylabel('$\Delta x$')

    plt.subplot(4, 2, 5)
    plt.title('Green')
    plt.plot(green_points_x, '-o', color='green')
    plt.ylabel('$\Delta y$')
    plt.subplot(4, 2, 6)
    plt.plot(green_points_y, '-o', color='green')
    plt.ylabel('$\Delta x$')

    plt.subplot(4, 2, 7)
    plt.title('Blue')
    plt.plot(blue_points_x, '-o', color='blue')
    plt.ylabel('$\Delta y$')
    plt.subplot(4, 2, 8)
    plt.plot(blue_points_y, '-o', color='blue')
    plt.ylabel('$\Delta x$')

    plt.show()


if __name__ == "__main__":
    main()
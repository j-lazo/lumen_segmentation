import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time


def determine_convex_hull(image_dir):

    src = cv2.imread(image_dir, 1)  # read input image
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blur = cv2.blur(gray, (3, 3))  # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # Finding contours for the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hulls = []
    new_hulls = []
    new_contours = []
    temp_contours = []
    # calculate points for each contour

    if len(contours) > 1:
        temp_contours = contours[0]
        for i in range(1, len(contours)):
            temp_contours = np.concatenate((temp_contours, contours[i]), axis=0)

        new_contours.append(temp_contours)
    else:
        new_contours = contours

    for i in range(len(new_contours)):
        #print('new_contour', type(new_contours[i]))
        #print(new_contours[i])
        new_hulls.append(cv2.convexHull(new_contours[i], False))


    M = cv2.moments(new_contours[0])
    point_x = int(M["m10"] / M["m00"])
    point_y = int(M["m01"] / M["m00"])
    print('x:', point_x, 'y:', point_y)
    return new_hulls, point_x, point_y


def main():
    dir_mask = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
               'lumen_data/video_test/result_masks/phantom_001_pt2/'

    dir_imgs = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
               'lumen_data/video_test/painted_frames/phantom_001_pt2/'

    save_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
               'lumen_data/video_test/painted_frames/center_phantom_001_pt2/'

    list_images = sorted(os.listdir(dir_mask))
    all_centers_x = []
    all_centers_y = []
    for i, image in enumerate(list_images):
        print(i, image)
        mask_dir = dir_mask + image
        hulls, center_x, center_y = determine_convex_hull(mask_dir)
        all_centers_x.append(center_x)
        all_centers_y.append(center_y)
        points_x = []
        points_y = []
        for hull in hulls:
            for i, point in enumerate(hull):
                points_x.append(point[0][0])
                points_y.append(point[0][1])

        image_dir = dir_imgs + image
        img = cv2.imread(image_dir)
        red = img[:, :, 2].copy()
        blue = img[:, :, 0].copy()
        img[:, :, 0] = red
        img[:, :, 2] = blue
        w, h, d = np.shape(img)

        #plt.figure()
        #plt.imshow(img)
        #plt.plot([center_x, int(w/2)], [center_y, int(h/2)], marker='D')
        #plt.plot(int(w/2), int(h/2), 'gx')
        #plt.plot(center_x, center_y, 'ro')
        #plt.plot(points_x, points_y, 'b-')
        #plt.axis('off')
        #plt.savefig(save_dir + image)
        #plt.show()
        #plt.close()

    plt.figure()
    plt.subplot(211)
    plt.plot(all_centers_x, '-o')
    plt.subplot(212)
    plt.plot(all_centers_y, '-o')
    plt.show()


if __name__ == "__main__":
    main()
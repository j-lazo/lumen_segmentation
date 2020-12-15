import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import os
import ast


def read_csv(file_path):
    points = []
    name_imgs = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            points.append((row[5]))
            name_imgs.append((row[0]))

        return name_imgs, points


def generate_mask_from_points(img, mask_points):
    mask_points = np.array(mask_points, dtype=np.int32)
    img_shape = np.shape(img)
    mask = np.zeros([img_shape[0], img_shape[1]])
    drawing = cv2.fillPoly(mask, mask_points, color=255)
    return drawing


def convert_cvs_data_to_imgs(directory_csv_file, directory_imgs, output_directory='masks'):

    if not(os.path.isdir(output_directory)):
        os.mkdir(output_directory)

    list_images = [file for file in os.listdir(directory_imgs) if file.endswith('.png')]
    list_name_imgs, list_points = read_csv(directory_csv_file)
    for img in list_images:
        if img in list_name_imgs:
            print('img:', img)
            indexes = [i for i, element in enumerate(list_name_imgs) if element == img]
            contours = []
            for index in indexes:
                list_points[index]
                res = ast.literal_eval(list_points[index])
                points_x = res.get('all_points_x')
                points_y = res.get('all_points_y')
                contour = []
                for i, x in enumerate(points_x):
                    contour.append([[x, points_y[i]]])
                    array_contour = np.array(contour, dtype=np.int32)

                contours.append(array_contour)
                image = cv2.imread(directory_imgs + img)
                mask = generate_mask_from_points(image, contours)
                cv2.imwrite(output_directory + '/' + img, mask)

def main():

    path_csv = '/path_/to/file/file.csv'
    path_imgs = '/path/to/images/'
    path_output = '/path/to/output/mask/folder/'
    convert_cvs_data_to_imgs(path_csv, path_imgs, path_output)

if __name__ == "__main__":
    main()
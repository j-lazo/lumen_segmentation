import numpy as np
import cv2
import os
import csv
import shutil

def read_results_csv(file_path, row_id=0):
    selected_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            selected_values.append(float(row[row_id]))

        return selected_values


def select_samples(source_folder, destination_folder, samples_list):
    destination_folder = samples_list + 'samples/'
    if not(os.path.isdir(destination_folder)):
        os.mkdir(destination_folder)

    """
    Copy tuples of images and labels in 1 step
    :param source_folder:
    :param destination_folder:
    :param samples_list
    :return:
    """

    images_list = os.listdir(source_folder)

    for counter, image in enumerate(samples_list):
        if os.path.isfile(source_folder + samples_list):
            shutil.copy(source_folder + image, destination_folder + image)


def main():

    base_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
               'data/old_lumen_data/results/'

    model = 'comparison_MaskRCNN_vs_ResUnet_grayscale/'

    test_folder = 'test_01/'

    source_folder = ''.jopin(base_dir, model, 'predictions', test_folder)

    csv_file_samples = '/home/nearlab/Jorge/current_work/' \
                       'lumen_segmentation/data/lumen_data' \
                       'list_samples.csv'

    list_samples = read_results_csv(csv_file_samples)
    select_samples(source_folder, source_folder, list_samples)


if __name__ == '__main__':
    main()
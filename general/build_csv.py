import csv
import cv2
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import os


def read_file_txt(file_path, row_id=0):
    color = []
    point_1 = []
    point_2 = []
    point_3 = []
    point_4 = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            line = row[0].split()
            color.append(float(line[0]))
            point_1.append(float(line[1]))
            point_2.append(float(line[2]))
            point_3.append(float(line[3]))
            point_4.append(float(line[4]))

        return color, point_1, point_2, point_3, point_4


def main():

    dir_csv_files = '/home/nearlab/Jorge/data/EAD_2019/trainingData_detection/trainingData_detection/'
    dir_images = '/home/nearlab/Jorge/DATASETS/EAD_2019/image/'


if __name__ == '__main__':
    main()
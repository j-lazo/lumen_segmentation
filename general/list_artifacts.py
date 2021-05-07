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


def build_rectangle(center_x, center_y, width, height):

    x_min = center_x - int(width/2)
    x_max = center_x + int(width / 2)

    y_min = center_y - int(height / 2)
    y_max = center_y + int(height / 2)

    points_x = [x_min, x_max, x_max, x_min, x_min]
    points_y = [y_min, y_min, y_max, y_max, y_min]

    return points_x, points_y


def remove_duplicates(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    print(final_list)
    return final_list


def prepare_data(dir_images, dir_csv_files, plot=False):
    name_test_csv_file = 'name_file.csv'
    list_csv_files = os.listdir(dir_csv_files)
    list_csv_files = sorted([file[:-4] for file in list_csv_files if file.endswith('.txt')])
    list_imgs =sorted([file for file in os.listdir(dir_images) if file.endswith('.png')])
    unique_colors = []
    unique_imgs = []
    for image in list_imgs:
        print(image)
        if image[:-4] in list_csv_files:
            img = cv2.imread(dir_images + image)
            w, h, d = np.shape(img)
            colours, pnts_xmax, pnts_ymax, pnts_xmin, pnts_ymin = read_file_txt(dir_csv_files + image[:-4] + '.txt')

            unique_imgs.append(image)
            unique_colors.append(remove_duplicates(colours))

            pnts_ymax = [int(point*w) for point in pnts_ymax]
            pnts_xmax = [int(point*h) for point in pnts_xmax]
            pnts_ymin = [int(point*w) for point in pnts_ymin]
            pnts_xmin = [int(point*h) for point in pnts_xmin]


    with open(name_test_csv_file, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_file_writer.writerow(['num', 'name',
                                      'specularity', 'saturation',
                                      'artifact', 'blur', 'bubbles'])
        for i, file in enumerate(unique_imgs):
            if 0.0 in unique_colors[i]:
                specularity = 1
            else:
                specularity = 0

            if 1.0 in unique_colors[i]:
                saturation = 1
            else:
                saturation = 0

            if 2.0 in unique_colors[i]:
                artifact = 1
            else:
                artifact = 0

            if 3.0 in unique_colors[i]:
                blur = 1
            else:
                blur = 0

            if 5.0 in unique_colors[i]:
                bubbles = 1
            else:
                bubbles = 0

            results_file_writer.writerow([str(i), file,
                                          specularity,
                                          saturation,
                                          artifact,
                                          blur,
                                          bubbles])




            if plot is True:

                plt.figure()
                plt.imshow(img)
                for j, color in enumerate(colours):
                    print(color)
                    if color == 0.0:
                        # specularity
                        col = 'red'
                    elif color == 1.0:
                        # saturation
                        col = 'green'
                    elif color == 2.0:
                        # artifact
                        col = 'purple'
                    elif color == 3.0:
                        # blur
                        col = 'pink'
                    elif color == 4.0:
                        # contrast
                        col = 'yellow'
                    elif color == 5.0:
                        # bubles
                        col = 'orange'
                    elif color == 6.0:
                        # insturment
                        col = 'white'
                    else:
                        # blood
                        col = 'black'

                    pnts_x, pnts_y = build_rectangle(pnts_xmax[j],
                                                     pnts_ymax[j],
                                                     pnts_xmin[j],
                                                     pnts_ymin[j])

                    plt.plot(pnts_xmax[j], pnts_ymax[j], '*')
                    plt.plot(pnts_x, pnts_y, color=col)
                plt.show()


    #print(list_csv_files)
    #print(list_imgs)

def main():

    dir_csv_files = '/home/nearlab/Jorge/data/EAD_2019/trainingData_detection/trainingData_detection/'
    dir_images = '/home/nearlab/Jorge/DATASETS/EAD_2019/image/'
    prepare_data(dir_images, dir_csv_files)


if __name__ == '__main__':
    main()

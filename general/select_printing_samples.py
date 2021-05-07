import os
import csv
import shutil


def read_results_csv(file_path, row_id=0):
    selected_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            selected_values.append((row[row_id]))

        return selected_values


def select_samples(source_folder, destination_folder, samples_list):
    destination_folder = destination_folder + 'samples/'
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
        print(source_folder)
        if os.path.isfile(source_folder + image):
            print(image)
            shutil.copy(source_folder + image, destination_folder + image)


def main():

    base_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
               'lumen_data/results/'

    model = 'ResUnet_lr_0.001_bs_8_grayscale_26_11_2020_20_38/'

    test_folder = 'test_02/'
    source_folder = ''.join([base_dir, model, 'predictions/', test_folder])

    csv_file_samples = '/home/nearlab/Jorge/current_work/' \
                       'lumen_segmentation/data/lumen_data/' \
                       'list_samples.csv'

    destination_folder = ''.join([base_dir, 'compare_3Dvs2D/latest_results/',
                                  model, test_folder])

    #list_samples = read_results_csv(csv_file_samples)
    list_samples = sorted(os.listdir('/home/nearlab/Jorge/current_work/'
                                     'lumen_segmentation/data/lumen_data/'
                                     'results/compare_3Dvs2D/3DMaskRCNN_grayscale_'))
    select_samples(source_folder, destination_folder, list_samples)


if __name__ == '__main__':
    main()
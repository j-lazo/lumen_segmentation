import os
import random
import shutil


def gather_all_data(source_folders, destination_folder, exceptions):

    """
    :param source_folders:
    :param destination_folder:
    :param exceptions:
    :return:
    """

    current_directory = '/home/nearlab/Jorge/ICPR2020/data/lumen_enlarged_dataset_rgb/augmented_data/'
    files_path_images = "".join([current_directory, 'all/image/'])
    files_path_labels = "".join([current_directory, 'all/label/'])
    original_images = os.listdir(files_path_images)
    label_images = os.listdir(files_path_labels)

    print(original_images)
    print(label_images)

    training_dir = current_directory + 'train/'
    validation_dir = current_directory + 'val/'

    # all_training = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/all_training/'
    # all_validation = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/all_validation/'

    for count_i, image in enumerate(original_images):
        if random.random() <= training_percentage:
            if image in label_images:
                print(image, 'image and label exists')
                shutil.copy(files_path_images + image, "".join([training_dir, 'image/', image]))
                shutil.copy(files_path_labels + image, "".join([training_dir, 'label/', image]))
            else:
                print('.')
                # print(image, 'the pair doesn not exists')
            # shutil.copy(files_path_positives + image, "".join([all_training, image]))
        else:
            if image in label_images:
                print(image, 'image and label exists')
                shutil.copy(files_path_images + image, "".join([validation_dir, 'image/', image]))
                shutil.copy(files_path_labels + image, "".join([validation_dir, 'label/', image]))
            else:
                print('.')
                # print(image, 'the pair doesn not exists')
            # shutil.copy(files_path_positives + image, "".join([all_validation, image]))


def main():
    source_folders = ''
    destination_folder = ''
    exceptions = []

    gather_all_data():

if __name__ == '__main__':
    main()

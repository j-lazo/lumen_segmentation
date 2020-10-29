import os
import random
import shutil


def gather_all_data(source_folder, destination_folder, exceptions):

    """
    This function gathers all data from different folders and put it all together in a single folder called "all"
    :param source_folders:
    :param destination_folder:
    :param exceptions:
    :return:
    """

    folder_list = set(os.listdir(source_folder)) - set(exceptions)
    folder_list = sorted([element for element in folder_list if
                          os.path.isdir(''.join([source_folder, element]))])

    for folder in folder_list[:]:
        print(folder)
        files_path_images = "".join([source_folder, folder, '/image/'])
        files_path_labels = "".join([source_folder, folder, '/label/'])
        images_list = os.listdir(files_path_images)
        labels_list = os.listdir(files_path_labels)

        #image_subfolder = sorted([element for element in images_list if os.path.isdir(''.join([source_folder, files_path_images]))])
        labels_subfolder = sorted([element for element in labels_list if
                                  os.path.isdir(''.join([source_folder, files_path_labels]))])

        if not(labels_subfolder):

            destination_image_folder = "".join([destination_folder, 'image/'])
            destination_label_folder = "".join([destination_folder, 'label/'])

            if not (os.path.isdir(destination_image_folder)):
                os.mkdir(destination_image_folder)

            if not (os.path.isdir(destination_label_folder)):
                os.mkdir(destination_label_folder)

            for counter, image in enumerate(images_list[:]):
                shutil.copy(files_path_images + image, destination_image_folder + image)
                shutil.copy(files_path_labels + image, destination_label_folder + image)

        else:
            for sub_folder in labels_subfolder:
                #2Do complete this option and the funciotn copy_images_and_label
                copy_images_and_label(source_folder, destination_folder, sub_folder)

def copy_images_and_label(source_folder, destination_folder, folder=''):

    """
    Copy tuples of images and labels in 1 step
    :param original_folder:
    :param destination_folder:
    :return:
    """

    source_folder = "".join([source_folder, '/', folder, '/'])
    destination_folder = "".join([destination_folder, '/', folder, '/'])

    files_path_images = "".join([source_folder,  '/image/'])
    files_path_labels = "".join([source_folder, '/label/'])
    images_list = os.listdir(files_path_images)
    labels_list = os.listdir(files_path_labels)

    destination_image_folder = "".join([destination_folder, 'image/'])
    destination_label_folder = "".join([destination_folder, 'label/'])

    if not (os.path.isdir(destination_image_folder)):
        os.mkdir(destination_image_folder)

    if not (os.path.isdir(destination_label_folder)):
        os.mkdir(destination_label_folder)

    for counter, image in enumerate(images_list):
        shutil.copy(files_path_images + image, destination_image_folder + image)
        shutil.copy(files_path_labels + image, destination_label_folder + image)

    return 0

def main():

    source_folders = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/3D_volume_data/all_data/patient_cases/'
    destination_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/3D_volume_data/all_data/all/'
    exceptions = ['all']
    gather_all_data(source_folders, destination_folder, exceptions)

if __name__ == '__main__':
    main()

import os
import random
import shutil


def copy_images_and_label(source_folder, destination_folder, list_imgs):

    source_folder_img = source_folder + 'image/'
    source_folder_label = source_folder + 'label/'

    destination_image_folder = destination_folder + 'image/'
    destination_label_folder = destination_folder + 'label/'

    for counter, image in enumerate(list_imgs):
        shutil.copy(source_folder_img + image, destination_image_folder + image)
        shutil.copy(source_folder_label + image, destination_label_folder + image)

def main():

    source_folders = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/phantom_001_pt2/data/'
    destination_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/phantom_001_pt2/original_data/'
    source_img_lists = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/phantom_001_pt2/label/'
    list_imgs = sorted(os.listdir(source_img_lists))

    copy_images_and_label(source_folders, destination_folder, list_imgs)


if __name__ == '__main__':
    main()

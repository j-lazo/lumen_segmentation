import os
import random
import shutil


def generate_training_and_validation_sets(current_directory, output_directory, training_percentage=0.5):

    """
    :param current_directory:
    :param output_directory:
    :param training_percentage:
    :return:
    """

    files_path_images = "".join([current_directory, 'image/'])
    files_path_labels = "".join([current_directory, 'label/'])

    original_images = os.listdir(files_path_images)
    label_images = os.listdir(files_path_labels)

    training_dir = output_directory + 'train/'
    validation_dir = output_directory + 'val/'

    original_images = [image[:-4] for image in original_images]
    label_images = [image[:-4] for image in label_images]

    for count_i, image in enumerate(original_images):
        if random.random() <= training_percentage:

            if image in label_images:
                print(image, 'image and label exists')
                shutil.copy(files_path_images + image + '.png', "".join([training_dir, 'image/', image, '.png']))
                shutil.copy(files_path_labels + image + '.png', "".join([training_dir, 'label/', image, '.png']))
            else:
                print(image, 'the pair does not exists')

        else:

            if image in label_images:
                print(image, 'image and label exists')
                shutil.copy(files_path_images + image + '.png', "".join([validation_dir, 'image/', image, '.png']))
                shutil.copy(files_path_labels + image + '.png', "".join([validation_dir, 'label/', image, '.png']))
            else:
                print(image, 'the pair does not exists')


def main():

    data_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'lumen_data/test/phantom_001_pt1/original_data/'

    output_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/'


    generate_training_and_validation_sets(data_directory,
                                          output_directory,
                                          training_percentage=0.65)


if __name__ == '__main__':
    main()

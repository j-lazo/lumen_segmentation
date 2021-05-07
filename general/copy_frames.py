import numpy as np
import shutil
import os
import cv2
import random


def copy_1():
    dir_imgs = '/home/nearlab/Jorge/current_work/' \
               'tumor_classification/data/all/cys/cys_case_006/no_lesion/'
    list_imgs = sorted(os.listdir(dir_imgs))
    other_dir_imgs = '/home/nearlab/Jorge/data/cystoscopy/case_009/pt11/'

    other_list = sorted(os.listdir(other_dir_imgs))

    for i, image in enumerate(list_imgs):
        if image in other_list:
            print(image)
            shutil.copy(other_dir_imgs + image, dir_imgs + image)


def copy_2():

    fold = 'fold_3/'
    data = 'val/'
    clas = 'no_lesion/'

    dir_imgs = '/home/nearlab/Jorge/current_work/' \
               'tumor_classification/data/all/cys/all_cases/' + fold + data + clas

    list_imgs = sorted(os.listdir(dir_imgs))
    #list_imgs = [image for image in list_imgs if np.random.rand() >= 0.5]

    destination_dir = '/home/nearlab/Jorge/current_work/' \
                     'tumor_classification/data/all/all/' + fold + data + clas

    for i, image in enumerate(list_imgs):
        print(image)
        shutil.copy(dir_imgs + image, destination_dir + image)


def copy_3():
    width = 300
    height = 300
    dim = (width, height)
    directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                'lumen_data/all_data/patient_cases/phantom_003/'

    #destination_dir = '/home/nearlab/Jorge/current_work/' \
    #                  'tumor_classification/data/all/all/fold_3/test/'

    destination_dir = directory

    sub_dirs = [f for f in os.listdir(directory)]
    for sub_dir in sub_dirs:
        images = [f for f in os.listdir(''.join([directory, sub_dir, '/']))]
        for image in images:
            print(sub_dir, image)
            img = cv2.imread(''.join([directory, sub_dir, '/', image]))
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            output_file_name = ''.join([destination_dir, sub_dir, '/', image])
            cv2.imwrite(output_file_name, resized)


def copy_4():

    cases = ['urs_case_002', 'urs_case_005', 'urs_case_006',
             'urs_case_007', 'urs_case_009', 'urs_case_011',
             'urs_case_012', 'urs_case_013']

    training_percentage = 0.68
    width = 300
    height = 300
    dim = (width, height)
    base_dir = '/home/nearlab/Jorge/current_work/' \
                   'tumor_classification/data/all/urs/'

    destination_dir = '/home/nearlab/Jorge/current_work/' \
                      'tumor_classification/data/all/urs/All_urs/fold_3/'

    for case in cases:

        directory = ''.join([base_dir, case, '/'])


        sub_dirs = [f for f in os.listdir(directory)]
        for sub_dir in sub_dirs:
            images = [f for f in os.listdir(''.join([directory, sub_dir, '/']))]
            for image in images:
                print(case, sub_dir, image)
                img = cv2.imread(''.join([directory, sub_dir, '/', image]))
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                if random.random() <= training_percentage:
                    output_file_name = ''.join([destination_dir, 'train/', sub_dir, '/', image])
                else:
                    output_file_name = ''.join([destination_dir, 'val/', sub_dir, '/', image])

                print(output_file_name)
                cv2.imwrite(output_file_name, resized)


if __name__ == '__main__':
    copy_3()
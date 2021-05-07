import os
import numpy as np
import cv2


def main():
    main_dir = '/home/nearlab/Jorge/current_work/tumor_classification/data/urs/all_cases/fold_1/'
    list_dirs = os.listdir(main_dir)
    if 'all_casses' in list_dirs:
        list_dirs.remove('all_cases')
    #list_dirs = [folder for folder in main_dir if os.path.isdir(main_dir)]

    for dir in list_dirs:
        list_sub_folders = ['lesion', 'no_lesion']
        for sub_folder in list_sub_folders:
            print(dir, sub_folder)
            new_base_dir = ''.join([main_dir, dir, '/', sub_folder, '/'])
            list_imgs = os.listdir(new_base_dir)
            for image in list_imgs:
                img = cv2.imread(new_base_dir + image)
                print(np.shape(img))
                reshaped_img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
                cv2.imwrite(new_base_dir + image, reshaped_img)


if __name__ == '__main__':
    main()
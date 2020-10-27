import os
import random
import cv2
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
from os import listdir

def load_test_data(X_test_path, Y_test_path):
    print('-' * 30)
    print('load test images...')
    print('-' * 30)
    test_dir = listdir(X_test_path)
    test_dir.sort()
    test_mask_dir = listdir(Y_test_path)
    test_mask_dir.sort()
    imgs_test = np.empty([0, 4, 128, 128, 3], dtype=np.float32)
    mask_test = np.empty([0, 4, 128, 128, 3], dtype=np.float32)
    for t in tqdm(test_dir[:6]):
        X_vol = np.load(os.path.join(X_test_path, t), allow_pickle=True)
        #X_vol = np.moveaxis(X_vol, -2, 0)
        #print(np.shape(X_vol))
        #imgs_test = np.append(imgs_test, [X_vol / 255], axis=0)
        imgs_test = np.append(imgs_test, [X_vol/255], axis=0)

    for tm in tqdm(test_mask_dir[:]):
        Y_vol = np.load(os.path.join(Y_test_path, tm), allow_pickle=True)
        #Y_vol = np.moveaxis(Y_vol, -2, 0)
        #print('Y_vol', np.shape(Y_vol))
        #print(np.amin(Y_vol), np.amax(Y_vol))
        mask_test = np.append(mask_test, [Y_vol / 255], axis=0)
    #    y_new = np.empty([256, 256, 1], dtype=np.float32)
    #    Y_vol[:, :, 0] = cv2.cvtColor(Y_vol[:, :, :], cv2.COLOR_RGB2GRAY)
    #    y_new[:, :, 0] = cv2.threshold(Y_vol[:, :, 0], 127, 255, cv2.THRESH_BINARY)[1] / 255
    #    mask_test = np.append(mask_test, [y_new], axis=0)

    return imgs_test, mask_test


def main():

    path_images = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'quick_test/p_007_pt4/new_folder/train/image/'
    path_labels = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'quick_test/p_007_pt4/new_folder/train/label/'

    images, labels = load_test_data(path_images, path_labels)
    print(np.shape(images), np.shape(labels))
    index = 0

    image_tes = images[index][0]
    mask_test = labels[index][0]

    print(np.shape(image_tes), np.shape(image_tes))
    print(np.shape(mask_test), np.shape(mask_test))
    plt.figure()
    plt.subplot(121)
    plt.imshow(image_tes)
    plt.subplot(122)
    plt.imshow(mask_test)
    plt.show()


if __name__ == "__main__":
    main()
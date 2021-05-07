import os
import random
import cv2
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
from os import listdir


def load_npy_data(X_test_path, Y_test_path):
    print('-' * 30)
    print('load test images...')
    print('-' * 30)
    test_dir = listdir(X_test_path)
    test_dir.sort()
    test_mask_dir = listdir(Y_test_path)
    test_mask_dir.sort()
    img_size = 256
    volume_size = 3
    imgs_test = np.empty([0, volume_size, img_size, img_size, 3], dtype=np.float32)
    mask_test = np.empty([0, volume_size, img_size, img_size, 3], dtype=np.float32)
    for t in tqdm(test_dir[:]):
        X_vol = np.load(os.path.join(X_test_path, t), allow_pickle=True)
        print(np.shape(X_vol))
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

    path = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
           '3D_volume_data/train/'
    path_images = path + 'image/'
    path_labels = path + 'label/'

    images, labels = load_npy_data(path_images, path_labels)
    print(np.shape(images), np.shape(labels))
    index = 25

    image_tes_1 = images[index][0]
    mask_test_1 = labels[index][0]

    image_tes_2 = images[index][1]
    mask_test_2 = labels[index][1]

    image_tes_3 = images[index][2]
    mask_test_3 = labels[index][2]


    plt.figure()

    plt.subplot(421)
    plt.imshow(image_tes_1)
    plt.subplot(422)
    plt.imshow(mask_test_1)

    plt.subplot(423)
    plt.imshow(image_tes_2)
    plt.subplot(424)
    plt.imshow(mask_test_2)

    plt.subplot(425)
    plt.imshow(image_tes_3)
    plt.subplot(426)
    plt.imshow(mask_test_3)


    plt.show()


if __name__ == "__main__":
    main()
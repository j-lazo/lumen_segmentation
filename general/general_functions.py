import os
import cv2
import skimage.io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import copy


def convert_mask(dir_folder):
    masks_folder_dir = dir_folder
    masks_folder = os.listdir(masks_folder_dir)

    for mask in masks_folder:
        print(dir_folder, mask)
        img = cv2.imread(''.join([dir_folder, mask]))
        new_image = copy.copy(img)
        # new_image = np.zeros(np.shape(img))
        # print(np.shape(img))
        """h,w,d = np.shape(img)
        for i in range(0,h):
            for j in range(0,w):
                print(img[i,j,2])
                if img[i,j,0] != 0:
                    print(img[i,j,0])
                    new_image[i,j,0] = 255"""

        new_image[:, :, 0] = new_image[:, :, 1]
        # new_image[i,j,2] = new_image[i,j,0]
        # img[:,:,1] = img[:,:,0]
        # img[:,:,2] = img[:,:,0]
        # img[:] = img[:] / 255
        # new_image = np.zeros(np.shape(img))
        # new_image = copy.copy(img)
        # img[img[:,:,0] > 0.9] = 255
        # new_image[:,:,0] = (img[:,:,0] > 0.9) * 255
        # new_image[:,:,1] = (img[:,:,1] > 0.9) * 255
        # new_image[:,:,2] = (img[:,:,2] > 0.9) * 255
        # print(new_image[:,:,0].all() == new_image[:,:,1].all() == new_image[:,:,2].all())
        cv2.imwrite(''.join([dir_folder, mask]), new_image)


def convert_to_polar(dir_folder):
    images_folder_dir = ''.join([dir_folder, 'image/'])
    masks_folder_dir = ''.join([dir_folder, 'label/'])

    images_folder = os.listdir(images_folder_dir)
    masks_folder = os.listdir(masks_folder_dir)

    for image in images_folder:
        img = cv2.imread(''.join([dir_folder, 'image/', image]))

        value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
        polar_image = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
        cv2.imwrite(''.join([dir_folder, 'image/', image]), polar_image)

    for mask in masks_folder:
        img = cv2.imread(''.join([dir_folder, 'label/', mask]))
        value_2 = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
        polar_mask = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value_2, cv2.WARP_FILL_OUTLIERS)
        cv2.imwrite(''.join([dir_folder, 'label/', mask]), polar_mask)


"""    
def convert_to_cartesian(dir_folder):
    print('ola ke ase')
    images_folder_dir = ''.join([dir_folder, 'image/'])
    masks_folder_dir = ''.join([dir_folder, 'label/'])

    images_folder = os.listdir(images_folder_dir)
    masks_folder = os.listdir(masks_folder_dir)

    for image in images_folder: 
        img = cv2.imread(''.join([dir_folder, 'image/', image]))
        img64_float = img.astype(np.float64)
        Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))
        cartisian_image = cv2.linearPolar(img, (img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)
        cartisian_image = cartisian_image/200
        plt.figure()
        plt.imshow(cartisian_image)
        plt.show()
        #cv2.imwrite(''.join([dir_folder, 'image/', image]), cartisian_image)


    for mask in masks_folder: 

        img = cv2.imread(''.join([dir_folder, 'label/', mask]))
        img64_float = img.astype(np.float64)
        Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))
        cartisian_image = cv2.linearPolar(img, (img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)
        cartisian_image = cartisian_image/200

        cv2.imwrite(''.join([dir_folder, 'label/', mask]), cartisian_image)"""



def check_size(dir_folder):

    print(dir_folder)
    # dir_folder = '/home/nearlab/Jorge/ICPR2020/data/polyps_mask_rcnn/train/'
    images_folder_dir = ''.join([dir_folder, 'image/'])
    masks_folder_dir = ''.join([dir_folder, 'label/'])

    images_folder = os.listdir(images_folder_dir)
    masks_folder = os.listdir(masks_folder_dir)

    print('Images')
    for image in images_folder:
        i = cv2.imread(''.join([dir_folder, 'image/', image]))
        # i = skimage.io.imread(os.path.join(dir_folder + 'train/')).astype(np.bool)
        print(np.shape(i), image)

    print('labels')
    for mask in masks_folder:
        i = cv2.imread(''.join([dir_folder, 'label/', mask]))
        # i = skimage.io.imread(os.path.join(dir_folder + 'train/')).astype(np.bool)
        print(np.shape(i), image)


def resize(dir_folder, new_size):
    # dir_folder = '/home/nearlab/Jorge/ICPR2020/data/polyps_mask_rcnn/train/'
    new_size = int(new_size)
    images_folder_dir = ''.join([dir_folder, 'image/'])
    masks_folder_dir = ''.join([dir_folder, 'label/'])

    images_folder = os.listdir(images_folder_dir)
    masks_folder = os.listdir(masks_folder_dir)

    for image in images_folder:
        img = cv2.imread(''.join([dir_folder, 'image/', image]))
        resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
        print(np.shape(resized))
        cv2.imwrite(''.join([dir_folder, 'image/', image]), resized)

    for mask in masks_folder:
        img = cv2.imread(''.join([dir_folder, 'label/', mask]))
        resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
        print(np.shape(resized))
        cv2.imwrite(''.join([dir_folder, 'label/', mask]), resized)


def convert_grayscale(dir_folder):

    images_folder_dir = ''.join([dir_folder, 'image/'])
    images_folder = os.listdir(images_folder_dir)

    #masks_folder_dir = ''.join([dir_folder, 'label/'])
    #masks_folder = os.listdir(masks_folder_dir)

    for image in images_folder:
        img = cv2.imread(''.join([dir_folder, 'image/', image]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(''.join([dir_folder, 'image/', image]), gray)

    #for mask in masks_folder:
    #    img = cv2.imread(''.join([dir_folder, 'label/', mask]))
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    cv2.imwrite(''.join([dir_folder, 'label/', mask]), gray)


def main():

    path_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/val/augmented_data/image/'
    convert_grayscale(path_directory)


if __name__ == "__main__":
    main()
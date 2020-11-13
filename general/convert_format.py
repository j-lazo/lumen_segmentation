import cv2
import os

def convert_format(dir_folder, destination_foler):

    imgs_folder_dir = dir_folder
    list_images = sorted(os.listdir(imgs_folder_dir))

    for image in list_images:
        print(dir_folder, image)
        img = cv2.imread(''.join([dir_folder, image]))
        cv2.imwrite(''.join([destination_foler, image[:-4], '.png']), img)


def main():

    base_dir = '/home/nearlab/Jorge/data/CVC-EndoSceneStill/CVC-300/gtlumen/'

    destination_foler = '/home/nearlab/Jorge/current_work/' \
                        'lumen_segmentation/data/lumen_colonoscopy/label/'

    convert_format(base_dir, destination_foler)


if __name__ == '__main__':
    main()
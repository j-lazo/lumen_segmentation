import cv2
import os
import numpy as np

def paint_imgs(img_dir, mask_dir, output_dir):

    image_list = sorted(os.listdir(img_dir))
    mask_list = sorted(os.listdir(mask_dir))

    for image in image_list:
        if image in mask_list:

            img = cv2.imread(img_dir + image)
            mask = cv2.imread(mask_dir + image)

            if np.shape(img) != np.shape(mask):
                img = cv2.resize(img, (np.shape(mask)[0], np.shape(mask)[1]))


            print(image)
            for i in range(np.shape(mask)[0]):
                for j in range(np.shape(mask)[1]):
                    if mask[i, j, 1] == 255:
                        img[i, j, 0] = 180

            cv2.imwrite(output_dir + image, img)


def main():

    path_imgs = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
                'data/lumen_data/video_test/phantom_001_pt2/'

    path_masks = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                 'lumen_data/video_test/result_masks/phantom_001_pt2/'

    path_output = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/video_test/painted_frames/phantom_001_pt2/'

    paint_imgs(path_imgs, path_masks, path_output)

if __name__ == "__main__":
    main()
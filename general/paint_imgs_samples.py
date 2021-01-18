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
                new_img = cv2.resize(img, (np.shape(mask)[0], np.shape(mask)[1]))
            else:
                new_img = img

            print(image)
            for i in range(np.shape(mask)[0]):
                for j in range(np.shape(mask)[1]):
                    if mask[i, j, 1] == 255:
                        new_img[i, j, 1] = 100

        cv2.imwrite(output_dir + image, new_img)


def main():

    path_imgs = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
                'data/lumen_data/video_test/original_frames/'
    path_masks = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
                 'data/lumen_data/video_test/result_masks/UNet/'
    path_output = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
                  'data/lumen_data/video_test/painted_frames/UNet/'
    paint_imgs(path_imgs, path_masks, path_output)

if __name__ == "__main__":
    main()
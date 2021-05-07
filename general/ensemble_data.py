import os
import cv2
import numpy as np
from scipy import stats

def read_img(im_dir):
    original_img = cv2.imread(im_dir)
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    return img


def preapre_data(output_dir, dir_1, dir_2, dir_3, dir_4):
    list_img_1 = os.listdir(dir_1)
    list_img_2 = os.listdir(dir_2)
    list_img_3 = os.listdir(dir_3)
    list_img_4 = os.listdir(dir_4)
    #print(list_img_3)
    for image in list_img_1:
        #print(image in list_img_3)
        if image in list_img_3:
            print(image)
            avrage_im = average_ensemble(
                            read_img(os.path.join(dir_1, image)),
                            read_img(os.path.join(dir_2, image)),
                            read_img(os.path.join(dir_3, image)),
                            read_img(os.path.join(dir_4, image)))

            #max_vote_img = max_vote(
            #    read_img(os.path.join(dir_1, image)),
            #    read_img(os.path.join(dir_2, image)),
            #    read_img(os.path.join(dir_3, image)),
            #    read_img(os.path.join(dir_4, image)))
            print(os.path.join(output_dir, 'Ensemble', image))
            cv2.imwrite(os.path.join(output_dir, 'Ensemble', image), avrage_im)
            #cv2.imwrite(os.path.join(output_dir, 'max_vote', image), max_vote_img)

def average_ensemble(image1, image2, image3, image4):

    if np.shape(image1) != np.shape(image2):
        print('no same size', np.shape(image1), np.shape(image2))
    else:
        #average_img = (image1 + image2 + image3 + image4) / 4
        #average_img = (image1 + image2) / 2
        average_img = (image1 + image2 + image3) / 3
        average_img[average_img > 0.5] = 255

    return average_img

def max_vote(img1, img2, img3, img4):
    output_img = np.zeros(np.shape(img1))

    w,h,d = np.shape(img1)

    for i in range(w):
        for j in range(h):
            for k in range(d):
                values = [img1[i,j,k], img2[i,j,k],
                          img3[i,j,k], img4[i,j,k]]
                output_img[i,j,k] = max(set(values), key=values.count)

    return output_img


def main():

    output_dir = '/home/nearlab/Jorge/current_work/' \
                 'lumen_segmentation/data/polyps/results/' \
                 'ensemble_2/'

    #path_files_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
    #               'lumen_data/results/compare_3Dvs2D/' \
    #               'ResUnet_lr_0.001_bs_8_grayscale_03_11_2020_20_08/'

    path_files_1 = '/home/nearlab/Jorge/current_work/' \
                   'lumen_segmentation/data/polyps/' \
                   'results/test_05/predictions/'

    path_files_2 = '/home/nearlab/Jorge/current_work/' \
                   'lumen_segmentation/data/polyps/results/' \
                   'test_07/predictions/'

    path_files_3 = '/home/nearlab/Jorge/current_work/' \
                   'lumen_segmentation/data/polyps/results/' \
                   'test_08/predictions/'

    path_files_4 = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
                   'data/lumen_data/video_test/' \
                   'result_masks/3DMaskRCNN/'

    preapre_data(output_dir, path_files_1,
                 path_files_2, path_files_3,
                 path_files_2)


if __name__ == "__main__":
    main()
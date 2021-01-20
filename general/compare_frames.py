import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_comparison(img_1, img_2, img_3, img_4, img_5, output_name):

    fig = plt.figure()
    # 2 rows, 4 columns
    gs = GridSpec(2, 4)
    # First row, first column
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('$m_1$')
    ax1.imshow(img_1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # First row, second column
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('$m_2$')
    ax2.imshow(img_2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('$M_1$')
    ax3.imshow(img_3)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('$M_2$')
    ax4.imshow(img_4)
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = fig.add_subplot(gs[:, 2:4])
    ax5.set_title('Proposed')
    ax5.imshow(img_5)
    ax5.set_xticks([])
    ax5.set_yticks([])

    fig.savefig(output_name)
    plt.close()
    #plt.show()


def prepare_data(dir_files_1, dir_files_2, dir_files_3,
                 dir_files_4, dir_files_5, output_dir):

    list_files_dir1 = sorted(os.listdir(dir_files_1))
    list_files_dir2 = sorted(os.listdir(dir_files_2))
    list_files_dir3 = sorted(os.listdir(dir_files_3))
    list_files_dir4 = sorted(os.listdir(dir_files_4))
    list_files_dir5 = sorted(os.listdir(dir_files_5))

    for j, image in enumerate(list_files_dir1[:]):
        if (image in list_files_dir2) and (image in list_files_dir3) and (image in list_files_dir4) and (image in list_files_dir5):

            print(image)
            img_1 = cv2.imread(dir_files_1 + image)
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
            img_2 = cv2.imread(dir_files_2 + image)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
            img_3 = cv2.imread(dir_files_3 + image)
            img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
            img_4 = cv2.imread(dir_files_4 + image)
            img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB)
            img_5 = cv2.imread(dir_files_5 + image)
            img_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2RGB)
            output_name = output_dir + image
            plot_comparison(img_1, img_2, img_3, img_4,
                            img_5, output_name)


def crop_images(image_directory, roi, string_to_add=''):

    image_list = [f for f in os.listdir(image_directory)
                  if os.path.isfile(os.path.join(image_directory, f))]

    for image in image_list:
        print('resizing', image)
        path_image = ''.join([image_directory, image])
        original_img = cv2.imread(path_image)
        cropped_img = original_img[roi[1]:roi[3], roi[0]:roi[2]]
        new_name = ''.join([image_directory, string_to_add, image])
        cv2.imwrite(new_name, cropped_img)

def main():

    base_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/video_test/painted_frames/'
    dir_files_1 = base_dir + 'UNet/'
    dir_files_2 = base_dir + 'MaskRCNN/'
    dir_files_3 = base_dir + '3DUNet/'
    dir_files_4 = base_dir + '3DMaskRCNN/'
    dir_files_5 = base_dir + 'Ensemble/'
    output_dir = base_dir + 'comparison/'

    prepare_data(dir_files_1, dir_files_2, dir_files_3,
                 dir_files_4, dir_files_5, output_dir)

    new_size = [68, 61, 584, 408]
    crop_images(output_dir, new_size)


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import glob
import os

name_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
              'lumen_data/video_test/painted_frames/' \
              'darK_center_phantom_003/'

img_array = []
images_folder = sorted(os.listdir(name_folder))

for filename in images_folder:
    img = cv2.imread(name_folder + filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


name_video ='/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
            'lumen_data/video_test/something.mp4'


fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('out.mp4', fourcc, 15, size)

#out = cv2.VideoWriter(name_video, cv2.VideoWriter_fourcc(*'avi'), 15, size)

for i in range(len(img_array)):
    print(i)
    out.write(img_array[i])
out.release()
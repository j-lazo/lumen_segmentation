import cv2
from matplotlib import  pyplot as plt
import numpy as np
file_name = '/home/nearlab/Downloads/' \
            'trainingData_semanticSegmentation/' \
            'trainingData_semanticSegmentation/' \
            '00000.tif'

image = cv2.imread(file_name)
print(np.unique(image))

plt.figure()
plt.subplot(131)
plt.imshow(image[:, :, 0])
plt.subplot(132)
plt.imshow(image[:, :, 1])
plt.subplot(133)
plt.imshow(image[:, :, 2])
plt.show()
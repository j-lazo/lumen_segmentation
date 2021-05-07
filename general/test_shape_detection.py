import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from skimage.feature import canny


def detect(c):

    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices

    if len(approx) == 3:
        shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
        # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
        # return the name of the shape

    return shape




image_dir = '/home/nearlab/Downloads/test_shapes/example_2.png'

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(image_dir)
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

print(np.unique(gray))

edges = canny(gray, sigma=3.5,
                  low_threshold=0.01, high_threshold=0.6)



x_lim = np.shape(edges)[0]
y_lim = np.shape(edges)[1]

print(x_lim, y_lim)

for x in range(0, x_lim):
    for y in range(0, y_lim):

        if edges[x, y] == True:
            print(x, y, edges[x, y])
            edges[x, y] = 255
        else:
            edges[x, y] = 0


print('edges', np.unique(edges))

thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
print(np.unique(thresh))

# find contours in the thresholded image and initialize the
# shape detector

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

print('edges', type(edges), np.shape(edges), np.amin(edges), np.amax(edges))
print('thres, ', type(thresh), np.shape(thresh), np.amin(thresh), np.amax(thresh))



for c in cnts:
    #print(cnts[0][1][0])
    #plt.figure()
    #cnts_0 = [numbers for numbers in cnts[0][0]]
    #print(cnts_0)
    #plt.plot(cnts[0])
    #plt.show()
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    #M = cv2.moments(cv2.UMat(cnts))
    #cX = int((M["m10"] / M["m00"]) * ratio)
    #cY = int((M["m01"] / M["m00"]) * ratio)
    shape = detect(c)
    print(shape)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #    0.5, (255, 255, 255), 2)

# show the output image
plt.figure()
plt.subplot(221)
plt.imshow(thresh)
plt.subplot(222)
plt.imshow(edges)
plt.subplot(223)
plt.imshow(gray)
plt.subplot(224)
plt.imshow(image)

plt.figure()
plt.imshow(image)
plt.show()





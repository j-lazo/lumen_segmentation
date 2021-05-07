from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


def calculate_area_and_circunference(dir_folder):
    mask_list = sorted(os.listdir(dir_folder))

    list_areas = []
    list_circuference = []
    list_circularities = []

    size_x = []
    size_y = []

    for mask in mask_list[:]:
        name_mask = ''.join([dir_folder, mask])

        arc_len, area = findArc(name_mask)
        if area != 0:
            circulatiry = 1.0*(arc_len**2)/(4*np.pi*area)
            list_circularities.append(circulatiry)


        list_areas.append(area)
        list_circuference.append(arc_len)

        #size_x.append(np.amax(list_x_pixels) - np.amin(list_x_pixels))
        #size_y.append(np.amax(list_y_pixels) - np.amin(list_y_pixels))

    return list_areas, list_circuference, list_circularities


def calculateDistance(x1, y1, X, Y):

    dist_vector = []
    for index, x2, in enumerate(X):
        y2 = Y[index]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_vector.append(dist)

    return dist_vector


def findArc(image, th=200):
    img = cv2.imread(image)
    res = img.copy()
    ## convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## threshold the gray
    th, threshed = cv2.threshold(gray, th, 255,  cv2.THRESH_BINARY)
    ## Find contours on the binary threshed image
    cnts = cv2.findContours(threshed,
                            cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]


    ## calcualte
    for cnt in cnts:
        arclen = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        cv2.drawContours(res, [cnt], -1, (0,255,0), 3, cv2.LINE_AA)
        #print("Length: {:.3f}\nArea: {:.3f}".format(arclen, area))

    cnt = cnts[0]
    pnts_x = [point[0][0] for point in cnt]
    pnts_y = [point[0][1] for point in cnt]

    moments = cv2.moments(cnt)
    print(np.shape(moments))
    print(moments)

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    c1 = int(moments['m20'] / moments['m00'])
    c2 = int(moments['m11'] / moments['m00'])
    c3 = int(moments['m02'] / moments['m00'])
    c4 = int(moments['m30'] / moments['m00'])
    c5 = int(moments['m21'] / moments['m00'])
    c6 = int(moments['m12'] / moments['m00'])
    c7 = int(moments['m03'] / moments['m00'])



    distances = calculateDistance(cx, cy, pnts_x, pnts_y)
    print('cx:', cx, 'cy:', cy)

    print('c1:', c1, 'c3:', c3)
    print('c2:', c2, 'c4:', c4)
    print('c5:', c5, 'c6:', c6)

    print(1.0*(arclen**2)/(4*np.pi*area))
    print(arclen, area)
    print(np.min(distances), np.amax(distances))

    fig, ax = plt.subplots()
    ax.plot(cx, cy, 'ro')
    ax.plot(c1, c3, 'go')
    ax.plot(c5, c6, 'bo')
    ax.add_artist(plt.Circle((cx, cy), np.min(distances), color='g', fill=False))
    ax.add_artist(plt.Circle((cx, cy), np.max(distances), color='b', fill=False))

    plt.imshow(img)
    plt.show()

    return arclen, area


def other():
    #directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/samples_masks/'
    directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/all_data/patient_cases/p_002_pt1/label/'
    calculate_area_and_circunference(directory)

def main():

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_001_pt1/label/'

    size_x, size_y, circ_1 = calculate_area_and_circunference(directory)
    plt.figure()
    plt.plot(size_x, size_y, 'ro')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_001_pt2/label/'
    size_x, size_y, circ_2 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'ro', label='patient 1')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_006_pt1/label/'
    size_x, size_y, circ_3 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'g*', label='patient 6')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_003_pt1/label/'
    size_x, size_y, circ_4 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'bo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_003_pt2/label/'
    size_x, size_y, circ_5 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'bo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_003_pt3/label/'
    size_x, size_y, circ_6 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'bo', label='patient 3')
    plt.legend(loc='best')
    plt.xlabel('Contour Perimeter ')
    plt.ylabel('Area')

    data_patient_1 = circ_1 + circ_2
    data_patient_2 = circ_3
    data_patient_3 = circ_4 + circ_5 + circ_6

    plt.figure()
    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_004_pt1/label/'
    size_x, size_y, circ_7 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'r*', label='patient 4')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_002_pt1/label/'
    size_x, size_y, circ_8 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'bo', label='patient 2')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_005_pt1/label/'
    size_x, size_y, circ_9 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'g*')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_005_pt2/label/'
    size_x, size_y, circ_10 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'g*', label='patient 5')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt1/label/'
    size_x, size_y, circ_11 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'yo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt2/label/'
    size_x, size_y, circ_12 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'yo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt3/label/'
    size_x, size_y, circ_13 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'yo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt4/label/'
    size_x, size_y, circ_14 = calculate_area_and_circunference(directory)
    plt.plot(size_x, size_y, 'yo', label='patient 7')

    plt.legend(loc='best')
    plt.xlabel('Contour Perimeter ')
    plt.ylabel('Area')


    plt.figure(3)
    fig, axs = plt.subplots(1, 3)
    axs[0].boxplot(data_patient_1, 1, 'gD')
    axs[0].set_title('patient 1')
    axs[0].set_ylim(0.8, 8.9)
    axs[1].boxplot(data_patient_2, 1, 'gD')
    axs[1].set_title('patient 6')
    axs[1].set_ylim(0.8, 8.9)
    axs[2].boxplot(data_patient_3, 1, 'gD')
    axs[2].set_title('patient 3')
    axs[2].set_ylim(0.8, 8.9)


    data_patient_7 = circ_11 + circ_12 + circ_13 + circ_14
    plt.figure(4)
    fig, axs = plt.subplots(1, 4)
    axs[0].boxplot(circ_7, 1, 'gD')
    axs[0].set_title('patient 4')
    axs[0].set_ylim(0.8, 11)
    axs[1].boxplot(circ_8, 1, 'gD')
    axs[1].set_title('patient 1')
    axs[1].set_ylim(0.8, 11)
    axs[2].boxplot(circ_9 + circ_10, 1, 'gD')
    axs[2].set_title('patient 5')
    axs[2].set_ylim(0.8, 11)
    axs[3].boxplot(data_patient_7, 1, 'gD')
    axs[3].set_title('patient 7')
    axs[3].set_ylim(0.8, 11)

    plt.show()


if __name__ == '__main__':
    #main()
    other()
import time
import numpy as np
import cv2
from glob import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import copy
import os
import csv
import matplotlib.pyplot as plt
from Unet_based import ResUnet
from Unet_based import Transpose_Unet
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import datetime
import Unet_based as un


def load_data(path, image_modality):
    print(path)
    path_images = ''.join([path, 'image/*'])
    path_labels = ''.join([path, "label/*"])
    images = sorted(glob(path_images))
    masks = sorted(glob(path_labels))
    total_size_images = len(images)
    total_size_labels = len(masks)
    print('total size images:', total_size_images, path_images)
    print('total size labels:', total_size_labels, path_labels)
    return (images, masks)


def load_data_only_imgs(path):
    print(path)
    path_images = ''.join([path, "/*"])
    images = sorted(glob(path_images))
    total_size_images = len(images)
    print('total size images:', total_size_images, path_images)
    return (images, images)


def read_image_test(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask_test(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x


def read_image(path):

    path = path.decode()
    x = cv2.imread(path, 1)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y


def tf_dataset(x, y, batch=8, img_modality='rgb'):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def iou(y_true, y_pred, smooth=1e-15):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def build_model(model_name):

    size = 256
    num_filters = [16, 32, 48, 64]
    # num_filters = [64, 48, 32, 16]
    # num_filters = [64, 128, 256, 512]
    inputs = Input((size, size, 3))

    if model_name == 'ResUnet':
        model = ResUnet.build_model()

    elif model_name == 'Transpose_Unet':
        model = Transpose_Unet.build_model()

    return model


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def read_image_test(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask_test(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x


def get_mcc(groundtruth_list, predicted_list):
    """Return mcc covering edge cases"""

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = -1

    elif _mcc_denominator_zero(tn, fp, fn, tp) is True:
        mcc = -1

    # Finally calculate MCC
    else:
        mcc = ((tp * tn) - (fp * fn)) / (
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return mcc


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)


def calculate_rates(image_1, image_2):
    image_1 = np.asarray(image_1).astype(np.bool)
    image_2 = np.asarray(image_2).astype(np.bool)
    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    if image_1.shape != image_2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    accuracy_value = accuracy_score(image_1, image_2)

    if (np.unique(image_1) == [False]).all() and (np.unique(image_1) == [False]).all():
        recall_value = 1.
        precision_value = 1.

    else:
        recall_value = recall_score(image_1, image_2)
        precision_value = average_precision_score(image_1, image_2)

    return precision_value, recall_value, accuracy_value


def dice(im1, im2, smooth=0.001):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    if (np.unique(im1) == [False]).all() and (np.unique(im2) == [False]).all():
        dsc = 1.
    else:
        dsc = 2. * (intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)

    return dsc
    # return 2. * (intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)


def read_img(dir_image):
    original_img = cv2.imread(dir_image)
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    return img


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))

        return dice_values


def evaluate_and_predict(model, directory_to_evaluate,
                         image_modality, results_directory, output_name, new_results_id):

    output_directory = 'predictions/' + output_name + '/'
    batch_size = 8
    (test_x, test_y) = load_data(directory_to_evaluate, image_modality)
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
    test_steps = (len(test_x)//batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1

    # evaluate the model in the test dataset
    model.evaluate(test_dataset, steps=test_steps)
    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1
    times = []
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        #print(i, x)
        directory_image = x
        x = read_image_test(x)
        #y = read_mask_test(y)
        init_time = time.time()
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        delta = time.time() - init_time
        times.append(delta)
        name_original_file = directory_image.replace(''.join([directory_to_evaluate, 'image/']), '')
        results_name = ''.join([results_directory, output_directory, name_original_file])
        cv2.imwrite(results_name, y_pred * 255.0)

    # save the results of the test dataset in a CSV file
    ground_truth_imgs_dir = directory_to_evaluate + 'image/'
    result_mask_dir = results_directory + output_directory

    ground_truth_image_list = [file for file in os.listdir(ground_truth_imgs_dir) if
                               os.path.isfile(os.path.join(ground_truth_imgs_dir, file))]
    results_image_list = [file for file in os.listdir(result_mask_dir) if os.path.isfile(os.path.join(result_mask_dir, file))]
    results_dice = []
    results_sensitivity = []
    results_specificity = []
    output_directory = 'predictions/' + output_name + '/'
    batch_size = 16
    (test_x, test_y) = load_data(directory_to_evaluate, image_modality)
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
    test_steps = (len(test_x) // batch_size)

    # save the results of the test dataset in a CSV file
    ground_truth_imgs_dir = directory_to_evaluate + 'image/'
    ground_truth_labels_dir = directory_to_evaluate + 'label/'
    result_mask_dir = results_directory + output_directory

    ground_truth_image_list = [file for file in os.listdir(ground_truth_imgs_dir) if
                               os.path.isfile(os.path.join(ground_truth_imgs_dir, file))]
    results_image_list = [file for file in os.listdir(result_mask_dir) if os.path.isfile(
        os.path.join(result_mask_dir, file))]
    results_dice = []
    results_sensitivity = []
    results_specificity = []
    results_accuracy = []

    for image in ground_truth_image_list[:]:

        result_image = [name for name in results_image_list if image[-12:] == name[-12:]][0]
        if result_image is not None:

            original_mask = read_img(''.join([ground_truth_labels_dir, image]))
            predicted_mask = read_img(''.join([result_mask_dir, result_image]))
            dice_val = dice(original_mask, predicted_mask)
            results_dice.append(dice_val)
            sensitivity, specificity, accuracy = calculate_rates(original_mask, predicted_mask)
            results_sensitivity.append(sensitivity)
            results_specificity.append(specificity)
            results_accuracy.append(accuracy)

        else:
            print(image, 'not found in results list')

    name_test_csv_file = ''.join([results_directory, 'results_evaluation_',
                                  output_name,
                                  '_',
                                  new_results_id,
                                  '_.csv'])

    with open(name_test_csv_file, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, file in enumerate(ground_truth_image_list):
            results_file_writer.writerow(
                [str(i), file, results_dice[i],
                 results_sensitivity[i],
                 results_specificity[i],
                 results_accuracy[i]])

    if len(test_x) % batch_size != 0:
        test_steps += 1
        # evaluate the model in the test dataset
    model.evaluate(test_dataset, steps=test_steps)
    print('Average inference times and std:')
    print(np.average(times), np.std(times))
    return name_test_csv_file


def predict_mask(model, input_image):

    y_pred = model.predict(np.expand_dims(input_image, axis=0))[0] >= 0.5

    return y_pred


def paint_imgs(img, mask):

    if np.shape(img) != np.shape(mask):
        img = cv2.resize(img, (np.shape(mask)[0], np.shape(mask)[1]))

    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i, j, 0] == True:
                img[i, j, 1] = 100

    return img


def build_contours(array_of_points):

    contours = []
    for i, y_points in enumerate(array_of_points[0]):
        point = (array_of_points[1][i], y_points)
        point = np.asarray(point)
        contours.append([point])

    return contours


def calc_histograms_and_center(mask, image):

    if not (np.all(mask == 0)):

        percentage = 0.6
        #grayscale = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = image[:, :, 2]
        grayscale = np.multiply(grayscale, mask)

        #list_values_grayscale = [value for row in grayscale for value in row if value != 0]
        # create the histogram plot, with three lines, one for
        # each color
        max_grays = ((np.where(grayscale >= int(percentage * np.amax(grayscale)))))
        gray_contours = np.asarray([build_contours(max_grays)])
        gray_convex_hull, gray_x, gray_y = determine_convex_hull(gray_contours)

        points_x = []
        points_y = []
        for hull in gray_convex_hull:
            for i, point in enumerate(hull):
                points_x.append(point[0][0])
                points_y.append(point[0][1])

    else:
        gray_x = 'nAN'
        gray_y = 'nAN'

    return gray_x, gray_y


def detect_dark_region(mask, image):
    w_image, h_image, d_image = np.shape(image)
    w_mask, h_mask, d_mask = np.shape(mask)
    temp_mask = np.zeros((w_mask, h_mask, 3), dtype="float32")
    temp_mask[:, :, 0] = mask[:, :, 0]
    temp_mask[:, :, 1] = mask[:, :, 0]
    temp_mask[:, :, 2] = mask[:, :, 0]

    if w_image != w_mask or h_image != h_mask:
       mask = cv2.resize(temp_mask, (w_image, h_image))

    gt_mask = np.zeros((w_image, h_image))

    if np.any(mask):

        #temp_mask = cv2.cv2.cvtColor(temp_mask*1, cv2.COLOR_BGR2GRAY)
        cl_mask = clean_mask(mask*1)
        gt_mask[cl_mask == 1.0] = 1

    point_x, point_y = calc_histograms_and_center(gt_mask, image)

    return point_x, point_y


def clean_mask(mask):

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    u8 = mask.astype(np.uint8)
    # remove the small areas appearing in the image if there is a considerable big one
    areas = []
    remove_small_areas = False
    contours, hierarchy = cv2.findContours(u8,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        areas.append(cv2.contourArea(contour))

    if len(contours) > 1:
        # sort the areas from bigger to smaller
        sorted_areas = sorted(areas, reverse=True)
        index_remove = np.ones(len(areas))
        for i in range(len(sorted_areas)-1):
            # if an area is 1/4 smaller than the bigger area, mark to remove
            if sorted_areas[i+1] < 0.15 * sorted_areas[0]:
                index_remove[areas.index(sorted_areas[i+1])] = 0
                remove_small_areas = True

    if remove_small_areas is True:
        #new_mask = np.zeros((w, d))
        new_mask = copy.copy(mask)
        for index, remove in enumerate(index_remove):
            if remove == 0:
                # replace the small areas with 0
                cv2.drawContours(new_mask, contours, index, (0, 0, 0), -1)  # as opencv stores in BGR format
    else:
        new_mask = mask

    return new_mask


def determine_convex_hull(contours):

    point_sets = np.asarray(contours[0])
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    new_hulls = []
    new_contours = []
    # calculate points for each contour
    if len(point_sets) > 1:
        temp_contours = point_sets[0]
        for i in range(1, len(point_sets)):
            temp_contours = np.concatenate((temp_contours, point_sets[i]), axis=0)

        new_contours.append(temp_contours)
    else:
        new_contours = point_sets

    for i in range(len(new_contours)):
        new_hulls.append(cv2.convexHull(new_contours[i], False))

    if new_hulls == []:
        point_x = 'old'
        point_y = 'old'
    else:
        M = cv2.moments(new_hulls[0])
        if M["m00"] < 0.001:
            M["m00"] = 0.001

        point_x = int(M["m10"] / M["m00"])
        point_y = int(M["m01"] / M["m00"])

    return new_hulls, point_x, point_y


def save_history(name_performance_metrics_file, model_history):

    with open(name_performance_metrics_file, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        list_keys = [str(item) for item in model_history.history.keys()]
        list_keys.insert(0, 'epoch')
        results_file_writer.writerow(list_keys)
        for i, element in enumerate(model_history.history['loss']):
            results_file_writer.writerow([str(i), element,
                                          model_history.history[list_keys[2]][i],
                                          model_history.history[list_keys[3]][i],
                                          model_history.history[list_keys[4]][i],
                                          model_history.history[list_keys[5]][i],
                                          model_history.history[list_keys[6]][i],
                                          model_history.history[list_keys[7]][i],
                                          model_history.history[list_keys[8]][i],
                                          model_history.history[list_keys[9]][i],
                                          model_history.history[list_keys[10]][i],
                                          model_history.history[list_keys[11]][i],
                                          model_history.history[list_keys[12]][i],
                                         model_history.history[list_keys[13]][i]])


def save_plots(model_history, results_directory, new_results_id):

    # summarize history for DSC
    plt.figure()
    plt.plot(model_history.history['dice_coef'], '-o')
    plt.plot(model_history.history['val_dice_coef'], '-o')
    plt.title('model DSC history')
    plt.ylabel('DSC')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(''.join([results_directory, 'DSC_history_', new_results_id, '_.svg']))
    plt.close()

    # summarize history for accuracy
    plt.figure()
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy history')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(''.join([results_directory, 'Accuracy_history_', new_results_id, '_.svg']))
    plt.close()

    # summarize history for loss
    plt.figure()
    plt.plot(model_history.history['loss'], '-o')
    plt.plot(model_history.history['val_loss'], '-o')
    plt.title('model loss history')
    plt.ylabel('DSC loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valtest'], loc='upper left')
    plt.savefig(''.join([results_directory, 'DSC_loss_history_', new_results_id, '_.svg']))
    plt.close()

    print('Plots of the history saved at: results_directory')


def main(project_folder):
    name_model = 'Transpose_Unet'
    # Hyper-parameters:
    batch = 16
    lr = 1e-5
    epochs = 1
    # optimizer:
    opt = tf.keras.optimizers.Adam(lr)

    # image modality of the data
    image_modality = 'rgb'
    augmented = True
    if augmented is True:
        amount_data = '/augmented_data/'
    else:
        amount_data = '/original_data/'

    # Define training and validation data
    path = project_folder
    train_data_used = ''.join([project_folder, 'train', amount_data])

    (train_x, train_y) = load_data(train_data_used, image_modality)
    print('Data training: ', train_data_used)

    val_data_used = ''.join([project_folder, 'val', amount_data])
    (valid_x, valid_y) = load_data(val_data_used, image_modality)
    print('Data validation: ', val_data_used)

    train_dataset = tf_dataset(train_x, train_y, batch=batch,
                               img_modality=image_modality)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch,
                               img_modality=image_modality)

    # metrics list:
    metrics = ["acc", tf.keras.metrics.Recall(),
               tf.keras.metrics.Precision(), dice_coef, iou]

    model = build_model(name_model)
    model.summary()
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=metrics)
    training_starting_time = datetime.datetime.now()

    # determine if also perform analysis of the training and validation dataset
    analyze_validation_set = False
    evaluate_train_dir = False
    # ID name for the folder and results

    new_results_id = ''.join([name_model,
                              '_lr_',
                              str(lr),
                              '_bs_',
                              str(batch),
                              '_', image_modality, '_',
                              training_starting_time.strftime("%d_%m_%Y_%H_%M"),
                              ])
    results_directory = ''.join([project_folder, 'results/', name_model,
                                 '/', new_results_id, '/'])
    # if results directory doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    callbacks = [
        ModelCheckpoint(results_directory + new_results_id + "_model.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
        CSVLogger(results_directory + new_results_id + "_data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    start_time = datetime.datetime.now()

    # Train the network
    model_history = model.fit(train_dataset,
                              validation_data=valid_dataset,
                              epochs=epochs,
                              steps_per_epoch=train_steps,
                              validation_steps=valid_steps,
                              callbacks=callbacks)
    # save the model
    model.save(results_directory + new_results_id + '_model')
    print('Total Training TIME:', (datetime.datetime.now() - start_time))
    print('METRICS Considered:')
    print(model_history.history.keys())

    name_performance_metrics_file = ''.join([results_directory,
                                             'performance_metrics_',
                                             training_starting_time.strftime("%d_%m_%Y_%H_%M"),
                                             '_.csv'])

    save_history(name_performance_metrics_file, model_history)
    save_plots(model_history, results_directory, new_results_id)
    # make directory for the predictions
    os.mkdir(results_directory + 'predictions/')
    # Evaluate and predict in the test dataset(s)
    list_of_test_sets = sorted(os.listdir(project_folder + 'test/'))

    for folder in list_of_test_sets:
        names_csv_files = []
        os.mkdir(''.join([results_directory, 'predictions/', folder, '/']))
        evaluation_directory = ''.join([project_folder, 'test/', folder, '/'])
        name_test_csv_file = evaluate_and_predict(model, evaluation_directory,
                                                  image_modality,
                                                  results_directory, folder, new_results_id)
        names_csv_files.append(name_test_csv_file)

    if analyze_validation_set is True:
        os.mkdir(results_directory + 'predictions/val/')
        evaluation_directory_val = project_folder + "val/original_data/"
        name_test_csv_file = evaluate_and_predict(model, evaluation_directory_val,
                                                    image_modality, results_directory,
                                                  'val', new_results_id)

    if evaluate_train_dir is True:
        os.mkdir(results_directory + 'predictions/train/')
        evaluation_directory_val = project_folder + "train/original_data/"
        name_test_csv_file = evaluate_and_predict(model, evaluation_directory_val,
                                                  image_modality, results_directory,
                                                  'train', new_results_id)


if __name__ == "__main__":
    project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/phantom_lumen/'
    main(project_folder)



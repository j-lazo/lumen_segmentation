project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/'
image_modality = 'rgb'
augmented = True

if augmented is True:
    amount_data = '/augmented_data/'
else:
    amount_data = '/original/'

analyze_validation_set = False
evaluate_train_dir = False

import sys
sys.path.append(project_folder)

import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.backend import sum as suma
from tensorflow.keras.backend import mean
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.utils import CustomObjectScope

import os.path
from os import path
from PIL import Image
from os import listdir
from os.path import isfile, join
from datetime import datetime
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


def load_data(path):
    path_images = ''.join([path, 'image/', image_modality, "/*"])
    path_labels = ''.join([path, "label/*"])
    images = sorted(glob(path_images))
    masks = sorted(glob(path_labels))
    total_size_images = len(images)
    total_size_labels = len(masks)
    print('total size images:', total_size_images, path_images)
    print('total size labels:', total_size_labels, path_labels)
    return (images, masks)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, 1)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
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


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


"""def dice_coef(y_true, y_pred, smooth=1):
    def f (y_true, y_pred):
        intersection = suma(y_true * y_pred, axis=[1,2,3])
        union = suma(y_true, axis=[1,2,3]) + suma(y_pred, axis=[1,2,3])
        x = mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        #x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)"""


def dice_coef(y_true, y_pred, smooth=0.00001):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    skip = Conv2D(num_filters, (3, 3), padding="same")(x)
    #skip = Activation("relu")(skip)
    skip = BatchNormalization()(skip)


    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = tf.math.add_n([x, skip])
    x = Activation("relu")(x)

    return x


def build_model():
    size = 256
    num_filters = [16, 32, 48, 64]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


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

def calculae_rates(image_1, image_2):

    image_1 = np.asarray(image_1).astype(np.bool)
    image_2 = np.asarray(image_2).astype(np.bool)
    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    if image_1.shape != image_2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    precision_value = average_precision_score(image_1, image_2)
    recall_value = recall_score(image_1, image_2)

    return precision_value, recall_value


def dice(im1, im2, smooth=1):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)


def read_img(dir_image):
    original_img = cv2.imread(dir_image)
    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))
    return img


def make_predictions():
    return 0


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))

        return dice_values

def evaluate_and_predict(model, directory_to_evaluate, results_directory, output_name):

    output_directory = 'predictions/' + output_name + '/'
    batch_size = 16
    (test_x, test_y) = load_data(directory_to_evaluate)
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
    test_steps = (len(test_x)//batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1

    # evaluate the model in the test dataset
    model.evaluate(test_dataset, steps=test_steps)
    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        print(i, x)
        directory_image = x
        x = read_image_test(x)
        y = read_mask_test(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        print(directory_to_evaluate + image_modality + '/')
        name_original_file = directory_image.replace(''.join([directory_to_evaluate, 'image/', image_modality, '/']), '')
        print(name_original_file)
        results_name = ''.join([results_directory, output_directory, name_original_file])
        print(results_name)
        cv2.imwrite(results_name, y_pred * 255.0)

    # save the results of the test dataset in a CSV file
    ground_truth_imgs_dir = directory_to_evaluate + 'image/' + image_modality + '/'
    result_mask_dir = results_directory + output_directory

    ground_truth_image_list = [file for file in listdir(ground_truth_imgs_dir) if
                               isfile(join(ground_truth_imgs_dir, file))]
    results_image_list = [file for file in listdir(result_mask_dir) if isfile(join(result_mask_dir, file))]
    results_dice = []
    results_sensitivity = []
    results_specificity = []

    for image in ground_truth_image_list[:]:

        result_image = [name for name in results_image_list if image[-12:] == name[-12:]][0]
        if result_image is not None:
            original_mask = read_img(''.join([ground_truth_imgs_dir, image]))
            predicted_mask = read_img(''.join([result_mask_dir, result_image]))
            dice_val = dice(original_mask, predicted_mask)
            results_dice.append(dice_val)
            sensitivity, specificity = calculae_rates(original_mask, predicted_mask)
            results_sensitivity.append(sensitivity)
            results_specificity.append(specificity)

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
                [str(i), file, results_dice[i], results_sensitivity[i], results_specificity[i]])

    return name_test_csv_file

filepath_models = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/old_lumen_data/results' \
                  '/ResUnet_lr_0.001_bs_16_grayscale_23_10_2020_13_31' \
                  'ResUnet_lr_0.001_bs_16_grayscale_23_10_2020_13_31_model.h5'

# ------------------- Hyperparameters -----------------------------------
batch = 16
lr = 1e-3
epochs = 10

opt = tf.keras.optimizers.Adam(lr)
metrics = ["acc", tf.keras.metrics.Recall(),
           tf.keras.metrics.Precision(), dice_coef, iou]

model = build_model()
model.compile(optimizer=opt, loss=dice_coef_loss, metrics=metrics)

model.load_weights('/home/nearlab/Jorge/current_work/lumen_segmentation/'
                   'data/old_lumen_data/results/'
                   'ResUnet_lr_0.001_bs_16_grayscale_23_10_2020_15_04/'
                   'ResUnet_lr_0.001_bs_16_grayscale_23_10_2020_15_04_model.h5')

model.summary()

training_time = datetime.now()
new_results_id = ''.join(['ResUnet',
                           '_lr_',
                           str(lr),
                           '_bs_',
                           str(batch),
                           '_', image_modality, '_',
                           training_time.strftime("%d_%m_%Y_%H_%M"),
                           ])

results_directory = ''.join([project_folder, 'results/', new_results_id, '/'])
os.mkdir(results_directory)


os.mkdir(results_directory + 'predictions/')
os.mkdir(results_directory + 'predictions/test_01/')
os.mkdir(results_directory + 'predictions/test_02/')
os.mkdir(results_directory + 'predictions/test_03/')

evaluation_directory_01 = project_folder + "test/test_01/"
evaluation_directory_02 = project_folder + "test/test_02/"
evaluation_directory_03 = project_folder + "test/test_03/"
name_test_csv_file_1 = evaluate_and_predict(model, evaluation_directory_01, results_directory, 'test_01')
name_test_csv_file_2 = evaluate_and_predict(model, evaluation_directory_02, results_directory, 'test_02')
name_test_csv_file_3 = evaluate_and_predict(model, evaluation_directory_03, results_directory, 'test_03')

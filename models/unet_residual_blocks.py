project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/'
image_modality = 'grayscale'
augmented = True
if augmented is True:
    amount_data = '/augmented_data/'
else:
    amount_data = '/'

analyze_validation_set = False
evaluate_train_dir = False

import sys
sys.path.append(project_folder)

import os
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
    print(path)
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

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred) 


def conv_block(x, num_filters):
    
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    skip = Conv2D(num_filters, (3, 3), padding="same")(x)
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
    x = x/255.0
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
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True :
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
    confusion_matrix_arrs['tn'] = np.logical_and(groundtru/home/nearlab/Jorge/ICPR2020/data/lumen_enlarged_dataset_grayscaleth_inverse, predicted_inverse)
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


# ------------------- Define training and validation data -----------------------------------
path = project_folder 
train_data_used = ''.join([project_folder,  'train', amount_data])

(train_x, train_y) = load_data(train_data_used)
print('Data training: ', train_data_used)

val_data_used = ''.join([project_folder,  'val', amount_data])
(valid_x, valid_y) = load_data(val_data_used)
print('Data validation: ', val_data_used)

# ------------------- Hyperparameters -----------------------------------
batch = 16
lr = 1e-3
epochs = 35

train_dataset = tf_dataset(train_x, train_y, batch=batch)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

opt = tf.keras.optimizers.Adam(lr)
metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), dice_coef, iou]

model = build_model()
model.summary()
model.compile(optimizer=opt, loss=dice_coef_loss, metrics=metrics)
#model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

training_time = datetime.now()
new_results_id = ''.join(['ResUnet',
                           '_lr_',
                           str(lr),
                           '_bs_',
                           str(batch),
                           '_', image_modality, '_',
                           training_time.strftime("%d_%m_%Y %H_%M"),
                           ])

results_directory = ''.join([project_folder, 'results/', new_results_id, '/'])
os.mkdir(results_directory)

callbacks = [
    ModelCheckpoint(results_directory + new_results_id + "_model.h5"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
    CSVLogger(results_directory + new_results_id + "_data.csv"),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

#model_checkpoint = [ModelCheckpoint(project_folder + 'unet_training_' + training_time.strftime("%d_%m_%Y %H_%M") +'_.hdf5'),
#                   EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
#class_1_weight = 0.55
#class_2_weight = 5.02
#class_weight = [class_1_weight, class_2_weight, 1.0, 3.4]

#weight_for_0 = 0.55
#weight_for_1 = 5.02
#class_weight = {0:weight_for_0, 1:weight_for_1}

#with CustomObjectScope({'iou': iou}):
#    model_dir = project_folder + "files/model.h5"
#    model = tf.keras.models.load_model(model_dir, compile=False)
#    model.compile(optimizer=opt, metrics = [["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]])
#dependencies = {'iou': iou}

#dependencies = {'dice_coef': dice_coef}
#with CustomObjectScope({'dice_coef': dice_coef}):
#        model = tf.keras.models.load_model(project_folder + "files/model.hdf5",
#                                           custom_objects = dependencies)
        #model.compile(optimizer=opt, loss='binary_crossentropy', metrics = [["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]])

train_steps = len(train_x)//batch
valid_steps = len(valid_x)//batch

if len(train_x) % batch != 0:
    train_steps += 1
if len(valid_x) % batch != 0:
    valid_steps += 1

model_history = model.fit(train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks)
    #class_weight=class_weight)

print('METRICS')
print(model_history.history.keys())

name_performance_metrics_file = ''.join([results_directory,
                                         'performance_metrics',
                                         new_results_id,
                                         '_.csv'])

name_performance_metrics_file = ''.join([project_folder, 'performance_metrics', training_time.strftime("%d_%m_%Y %H_%M"), '_.csv'])

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
                                      model_history.history[list_keys[11]][i]])

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
    print(results_image_list)
    print(ground_truth_imgs_dir)
    print(ground_truth_image_list)

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


# evaluate and predict in the test dataset(s)
os.mkdir(results_directory + 'predictions/')
os.mkdir(results_directory + 'predictions/test_01/')
os.mkdir(results_directory + 'predictions/test_02/')
os.mkdir(results_directory + 'predictions/test_03/')

evaluation_directory_01 = project_folder + "test/test_01/"
evaluation_directory_02 = project_folder + "test/test_02/"
evaluation_directory_03 = project_folder + "test/test_03/"
name_test_csv_file_1 = evaluate_and_predict(model, evaluation_directory_01, results_directory, 'test_01')
name_test_csv_file_2 = evaluate_and_predict(model, evaluation_directory_01, results_directory, 'test_02')
name_test_csv_file_3 = evaluate_and_predict(model, evaluation_directory_01, results_directory, 'test_03')

# evaluate in the validation dataset
os.mkdir(results_directory + 'predictions/val/')
(valid_x, valid_y) = load_data(project_folder + "val/")
val_dataset = tf_dataset(valid_x, valid_y, batch=16)

if len(valid_x) % 16 != 0:
    valid_steps += 1
print('Evaluation Validation Dataset')
model.evaluate(val_dataset, steps=valid_steps)

# evaluate in the train dataset
(train_x, train_y) = load_data(project_folder + "train/")
train_dataset = tf_dataset(train_x, train_y, batch=16)

if len(train_x) % 16 != 0:
    train_steps += 1

print('Evaluation Train Dataset')
model.evaluate(train_dataset, steps=train_steps)

if evaluate_train_dir is True:
    for i, (x, y) in tqdm(enumerate(zip(train_x, train_y)), total=len(train_x)):
        print(i, x)
        directory_image = x
        x = read_image_test(x)
        y = read_mask_test(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        height, width, _ = x.shape

        name_original_file = directory_image.replace(project_folder + 'train/image/', '')
        results_name = ''.join([project_folder, 'predictions_train/', name_original_file])
        cv2.imwrite(results_name, y_pred * 255.0)

# save the results of the validation dataset in a CSV file
if analyze_validation_set is True:

    now=datetime.now()
    #predictions in the validation dataset
    for i, (x, y) in tqdm(enumerate(zip(valid_x, valid_y)), total=len(valid_x)):
        print(i, x)
        directory_image = x
        x = read_image_test(x)
        y = read_mask_test(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5

        height, width, _ = x.shape

        name_original_file = directory_image.replace(project_folder + 'val/image/', '')
        results_name = ''.join([project_folder, 'predictions_val/', name_original_file])
        cv2.imwrite(results_name, y_pred * 255.0)

    ground_truth_imgs_dir = results_directory + 'val/label/'
    result_mask_dir = results_directory + 'predictions_val/'

    ground_truth_image_list = [file for file in listdir(ground_truth_imgs_dir) if isfile(join(ground_truth_imgs_dir, file))]

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

            # print(dice_val, 1-dice_val)
            results_dice.append(dice_val)

            sensitivity, specificity = calculae_rates(original_mask, predicted_mask)
            results_sensitivity.append(sensitivity)
            results_specificity.append(specificity)

            # print(sensitivity, specificity)
        else:
            print(image, 'not found in results list')

    name_validation_csv_file = ''.join([project_folder, 'results_validation_', now.strftime("%d_%m_%Y %H_%M"), '_.csv'])
    with open(name_validation_csv_file, mode='w') as results_file:
        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, file in enumerate(ground_truth_image_list):
            results_file_writer.writerow([str(i), file, results_dice[i], results_sensitivity[i], results_specificity[i]])

plt.figure()
plt.plot(model_history.history['dice_coef'], '-*')
plt.plot(model_history.history['val_dice_coef'], '-*')
plt.title('model DSC history')
plt.ylabel('DSC')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

# summarize history for accuracy
plt.figure()
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy history')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')


# summarize history for loss
plt.figure()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss history')
plt.ylabel('DSC loss')
plt.xlabel('epoch')
plt.legend(['train', 'valtest'], loc='upper left')


path_file_1= name_test_csv_file_1
path_file_2 = name_test_csv_file_2
path_file_3 = name_test_csv_file_3

print('Dice Coef.')
list_dice_values_file_1 = read_results_csv(path_file_1, 2)
list_dice_values_file_2 = read_results_csv(path_file_2, 2)
list_dice_values_file_3 = read_results_csv(path_file_3, 2)

data = [list_dice_values_file_1, list_dice_values_file_2, list_dice_values_file_3]

fig, axs = plt.subplots(1, 3)
axs[0].boxplot(data[0], 1, 'gD')
axs[0].set_title('Test dataset 1')

axs[1].boxplot(data[1], 1, 'gD')
axs[1].set_title('Test dataset 2')

axs[2].boxplot(data[2], 1, 'gD')
axs[2].set_title('Test dataset 3')

# sensitivity 
print('Sensitivity')
list_dice_values_file_1 = read_results_csv(path_file_1, 3)
list_dice_values_file_2 = read_results_csv(path_file_2, 3)
list_dice_values_file_3 = read_results_csv(path_file_3, 3)

data = [list_dice_values_file_1, list_dice_values_file_2, list_dice_values_file_3]

fig, axs = plt.subplots(1, 3)

axs[0].boxplot(data[0], 1, 'gD')
axs[0].set_title('Test dataset 1')
axs[1].boxplot(data[1], 1, 'gD')
axs[1].set_title('Test dataset 2')
axs[2].boxplot(data[2], 1, 'gD')
axs[2].set_title('Test dataset 3')

# specificity   
print('Specifficity')
list_dice_values_file_1 = read_results_csv(path_file_1, 4)
list_dice_values_file_2 = read_results_csv(path_file_2, 4)
list_dice_values_file_3 = read_results_csv(path_file_3, 4)

data = [list_dice_values_file_1, list_dice_values_file_2, list_dice_values_file_3]

fig, axs = plt.subplots(1, 3)

axs[0].boxplot(data[0], 1, 'gD')
axs[0].set_title('Test dataset 1')

axs[1].boxplot(data[1], 1, 'gD')
axs[1].set_title('Test dataset 2')

axs[2].boxplot(data[2], 1, 'gD')
axs[2].set_title('Test dataset 3')

plt.show()



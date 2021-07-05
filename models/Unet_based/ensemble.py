from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


def conv_block(tensor, num_filters):
    tensor = Conv2D(num_filters, (3, 3), padding="same")(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation("relu")(tensor)

    skip = Conv2D(num_filters, (3, 3), padding="same")(tensor)
    skip = Activation("relu")(skip)
    tensor = BatchNormalization()(skip)

    tensor = Conv2D(num_filters, (3, 3), padding="same")(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation("relu")(tensor)

    tensor = tf.math.add_n([tensor, skip])
    tensor = Activation("relu")(tensor)

    return tensor


def build_model(input_size=256, num_blocks=3):
    num_filters = [16, 32, 48, 64]
    # num_filters = [16, 32, 64, 128]
    volume_input = Input((3, input_size, input_size, 3))
    # --- volume branch ---
    skip_volume_branch = []
    num_3d_filters = 16
    volume_branch = tf.keras.layers.Conv3D(num_3d_filters, kernel_size=(3, 3, 3),
                               activation='relu',
                               strides=(1, 1, 1),
                               padding='valid')(volume_input)
    volume_branch = BatchNormalization()(volume_branch)
    paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
    volume_branch = tf.pad(volume_branch, paddings, "CONSTANT")
    volume_branch = tf.reshape(volume_branch, [-1, 256, 256, num_3d_filters])
    ## Encoder
    for f in num_filters:
        volume_branch = conv_block(volume_branch, f)
        skip_volume_branch.append(volume_branch)
        volume_branch = MaxPool2D((2, 2))(volume_branch)
    ## Bridge
    volume_branch = conv_block(volume_branch, num_filters[-1])
    num_filters.reverse()
    skip_volume_branch.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        volume_branch = UpSampling2D((2, 2))(volume_branch)
        xs1 = skip_volume_branch[i]
        volume_branch = Concatenate()([volume_branch, xs1])
        volume_branch = conv_block(volume_branch, f)
    ## Output
    output_1 = Conv2D(1, (1, 1), padding="same")(volume_branch)

    # --- single frame branch ---
    singe_frame = Input((input_size, input_size, 3))
    skip_single_branch = []
    single_branch = singe_frame
    for f in num_filters:
        single_branch = conv_block(single_branch, f)
        skip_single_branch.append(single_branch)
        single_branch = MaxPool2D((2, 2))(single_branch)
    ## Bridge
    single_branch = conv_block(single_branch, num_filters[-1])
    num_filters.reverse()
    skip_single_branch.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        single_branch = UpSampling2D((2, 2))(single_branch)
        xs = skip_single_branch[i]
        single_branch = Concatenate()([single_branch, xs])
        single_branch = conv_block(single_branch, f)
    ## Output
    output_2 = Conv2D(1, (1, 1), padding="same")(single_branch)
    avg = tf.keras.layers.Average()([output_1, output_2])
    output_layer = Activation("sigmoid")(avg)

    model = tf.keras.models.Model(inputs=[volume_input, singe_frame],
                                  outputs=output_layer,
                                  name='four_frames_ensemble')
    return model
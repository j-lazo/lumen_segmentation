from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import Model


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


def build_model():
    size = 256
    num_filters = [16, 32, 48, 64]
    # num_filters = [16, 32, 64, 128]
    # num_filters = [64, 128, 256, 512]
    input_layer = Input((3, size, size, 3))
    skip_x = []
    x = input_layer
    num_3d_filters = 16
    x = tf.keras.layers.Conv3D(num_3d_filters, kernel_size=(3, 3, 3), activation='relu',
                               strides=(1, 1, 1),
                               padding='valid')(x)
    x = BatchNormalization()(x)
    print('output shape 3D')
    print(str(x.shape.as_list()))
    paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
    # x = keras.layers.ZeroPadding3D(padding=2)
    x = tf.pad(x, paddings, "CONSTANT")
    print(str(x.shape.as_list()))
    x = tf.reshape(x, [-1, 256, 256, num_3d_filters])
    print(str(x.shape.as_list()))
    # y = tf.reshape(y, [5, 254, 254, 12])
    # x = tf.reshape(-1, tf.shape(x)[1] * tf.shape(x)[2])
    # x = tf.reshape(x)[0]
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        print(str(x.shape.as_list()))
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
    output_layer = Activation("sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='3_Continuous_frames_ResUnet')
    return model
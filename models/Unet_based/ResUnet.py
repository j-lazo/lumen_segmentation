from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import Model


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    skip = Conv2D(num_filters, (3, 3), padding="same")(x)
    skip = Activation("relu")(skip)
    skip = BatchNormalization()(skip)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = tf.math.add_n([x, skip])
    x = Activation("relu")(x)

    return x


def build_model(input_size=256, num_filters=[16, 32, 48, 64]):

    #num_filters = [16, 32, 48, 64]
    # num_filters = [64, 48, 32, 16]
    # num_filters = [64, 128, 256, 512]
    input_layer = Input((input_size, input_size, 3))

    skip_x = []
    x = input_layer
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
    output_layer = Activation("sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='ResUnet')
    return model

import os

from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.layers import Dense, Flatten, Reshape, Input, Lambda, Dropout, BatchNormalization, Convolution3D, \
    MaxPooling3D, ZeroPadding3D, Permute
from keras.models import Model
import keras.backend as K


#################
# Custom Layer

def output_of_stack_channels(input_shape):
    return input_shape[0], input_shape[1], input_shape[2], 3


def stack_channels(x):
    stacked = K.stack([x, x, x], axis=3)
    squeezed = K.squeeze(stacked, axis=-1)
    return squeezed


def output_of_stack_channels_3d(input_shape):
    return input_shape[0], input_shape[1], input_shape[2], input_shape[3], 3


def stack_channels_3d(x):
    stacked = K.stack([x, x, x], axis=4)
    squeezed = K.squeeze(stacked, axis=-1)
    return squeezed


#################
# VGG 16 model
DEFAULT_CONFIG_VGG16 = {
    "num_features": 384,
    "dropout_rate": 0.5,
    "pretrained": True
}


def create_model_vgg(out_shape, config, in_shape=(120, 160, 1)):
    out_height, out_width = out_shape

    # increase channels from 1 -> 3
    input_layer = Input(in_shape)
    input_stacked = Lambda(stack_channels, output_shape=output_of_stack_channels)(input_layer)

    # learn a normalization, roughly map distributions Kinect data -> RGB
    input_normalized = BatchNormalization()(input_stacked)

    # pre-trained VGG16 feature extraction
    weights = 'imagenet' if config["pretrained"] else None
    base_model = VGG16(weights=weights, include_top=False)  # , input_tensor=input_normalized)
    # x = base_model.output
    features = Model(inputs=input_layer, outputs=base_model(input_normalized))
    x = features.output

    # add MLP on top
    x = Flatten()(x)
    x = Dense(config["num_features"])(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(out_height * out_width)(x)
    x = Reshape((out_height, out_width))(x)

    # combine into single model object
    model = Model(inputs=input_layer, outputs=x)

    # freeze VGG16 layers
    if config["pretrained"]:
        for layer in base_model.layers:
            layer.trainable = False

    return model


#######################
# C3D model
DEFAULT_CONFIG_C3D = {
    "num_features": 384,
    "dropout_rate": 0.5,
}


def create_model_c3d(out_shape, in_shape=(112, 112, 15, 1), pretrained=False, config=DEFAULT_CONFIG_C3D):
    out_height, out_width = out_shape

    # increase channels from 1 -> 3
    input_layer = Input(in_shape)
    input_stacked = Lambda(stack_channels_3d, output_shape=output_of_stack_channels_3d)(input_layer)

    # permute channels from (H, W, T, C) -> (T, H, W, C)
    input_reorderd = Permute((2, 0, 1, 3))(input_stacked)

    # add 1 frame to beginning of clip (our data has 15 frames per clip, networks expects 16)
    input_padded = ZeroPadding3D(((1, 0), (0, 0), (0, 0)))(input_reorderd)

    # learn a normalization, roughly map distributions Kinect data -> RGB
    input_normalized = BatchNormalization()(input_padded)

    # pre-trained VGG16 feature extraction
    base_model = C3D()
    # x = base_model.output
    features = Model(inputs=input_layer, outputs=base_model(input_normalized))
    x = features.output

    # add MLP on top
    x = Flatten()(x)
    x = Dense(config["num_features"])(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(out_height * out_width)(x)
    x = Reshape((out_height, out_width))(x)

    # combine into single model object
    model = Model(inputs=input_layer, outputs=x)

    # freeze C3D layers
    if pretrained:
        for layer in base_model.layers:
            layer.trainable = False

    return model

#
# def PretrainedC3D():
#     model_dir =
#     model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
#     model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')


def C3D(summary=False, backend='tf', weights=None):
    """ Return the Keras model of the network
    Src: https://github.com/axon-research/c3d-keras/blob/master/c3d_model.py
    """
    model = Sequential()
    if backend == 'tf':
        input_shape = (16, 112, 112, 3)  # l, h, w, c
    else:
        input_shape = (3, 16, 112, 112)  # c, l, h, w
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())

    return model


if __name__ == "__main__":
    # C3D(True)
    # create_model_c3d(out_shape=(60, 80))
    #
    conf_vgg = {
        "num_features": 2384,
        "dropout_rate": 0.5,
        "pretrained": False
    }
    model = create_model_vgg(out_shape=(24, 32), config=conf_vgg)
    #
    model.summary()
    #
    # # create sample for testing
    # _input_shape = (1, 120, 160, 1)
    # x = np.random.rand(*_input_shape)
    #
    # # x = preprocess_input(x)  # TODO: needed?
    # features = model.predict(x, batch_size=1)
    # print(features.shape)
    #
    #

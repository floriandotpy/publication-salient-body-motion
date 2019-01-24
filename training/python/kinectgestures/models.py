from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv3D, Activation, MaxPool2D, MaxPooling3D, Flatten, Reshape, Dropout, \
    Conv3DTranspose, UpSampling3D

DEFAULT_CONFIG_3D = {
    "num_features": 384,
    "dropout_rate": 0.5,
    "kernel_base": 8
}

DEFAULT_CONFIG_2D = {
    "num_features": 384,
    "dropout_rate": 0.5,
    "kernel_base": 8
}


def create_model_2d(out_shape, config, in_shape=(120, 160, 1)):
    out_height, out_width = out_shape
    kernel_base = config["kernel_base"]

    model = Sequential()
    model.add(Conv2D(kernel_base, (3, 3), input_shape=in_shape, padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(2 * kernel_base, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(4 * kernel_base, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(config["num_features"]))
    model.add(Dense(out_height * out_width))
    model.add(Reshape((out_height, out_width)))

    return model


def create_model_3d(out_shape, config, in_shape=(120, 160, 15, 1)):

    out_height, out_width = out_shape

    # feature extraction
    model = create_feature_extractor_3d(in_shape, config)

    model.add(Flatten())

    # generate output
    model.add(Dense(config["num_features"]))
    model.add(Dropout(config["dropout_rate"]))
    model.add(Dense(out_height * out_width))
    model.add(Reshape((out_height, out_width)))

    return model


def create_feature_extractor_3d(in_shape, config):
    kernel_base = config["kernel_base"]
    model = Sequential()
    model.add(Conv3D(kernel_base, (3, 3, 3), input_shape=in_shape, padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling3D())
    model.add(Conv3D(kernel_base * 2, (3, 3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling3D())
    model.add(Conv3D(kernel_base * 4, (3, 3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling3D())
    return model


def create_model_3d_deconv(out_shape, in_shape=(120, 160, 15, 1), config=DEFAULT_CONFIG_3D):
    kernel_base = config["kernel_base"]

    # feature extraction
    model = create_feature_extractor_3d(in_shape, config)

    # deconv part
    model.add(Conv3DTranspose(kernel_base, 3))
    model.add(UpSampling3D())
    model.add(Conv3DTranspose(kernel_base, 3))
    model.add(UpSampling3D())
    model.add(Conv3DTranspose(kernel_base, 3))
    # model.add(UpSampling3D())

    return model


if __name__ == "__main__":
    model = create_model_3d_deconv((24, 32))
    model.summary()


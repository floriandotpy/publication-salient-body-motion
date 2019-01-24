import os

import numpy as np
from keras.callbacks import ModelCheckpoint

from kinectgestures.preprocessing import default_evaluation_preprocessing, default_training_preprocessing
from kinectgestures.data import GestureDataset

from kinectgestures.models import create_model_2d, create_model_3d
from kinectgestures.test import test
from kinectgestures.transfer import create_model_vgg
from kinectgestures.util import save_history, save_config, get_checkpoint_filepath
from kinectgestures.visuals import plot_history
from kinectgestures.util import checkpoint_dir_exists, make_or_get_checkpoint_dir, ask_yes_no_question
from kinectgestures.metrics import motion_metric
import kinectgestures.metrics as metrics


def test_dummy_prediction(model):
    # specify batch_dimension
    batch_shape = (1,) + model.input_shape[1:]
    print(batch_shape)

    # random sample
    sample = np.random.rand(*batch_shape)
    print("Testing forward pass...")
    result = model.predict(sample, batch_size=1)
    print(result.shape)
    print("[DONE] Tested forward pass.")


def train(config):
    #####################
    ## Dataset
    is_2d_model = config['model'] in ("cnn2d", "vgg16")
    dataset_train = GestureDataset(config["dataset_path"],
                                   which_split='train',
                                   last_frame_only=is_2d_model,
                                   batch_size=config["batch_size"])
    dataset_validation = GestureDataset(config["dataset_path"],
                                        which_split='validation',
                                        last_frame_only=is_2d_model,
                                        batch_size=config["batch_size"])

    #####################
    ## Model
    kwargs = dict(out_shape=config["out_shape"],
                        in_shape=config["in_shape"],
                        config=config)
    if config['model'] == "cnn2d":
        model = create_model_2d(**kwargs)
    elif config['model'] == "cnn3d":
        model = create_model_3d(**kwargs)
    elif config['model'] == "vgg16":
        model = create_model_vgg(**kwargs)
    else:
        raise ValueError("Unknown model {}".format(config["model"]))

    model.summary()

    #####################
    ## Data augmentation
    dataset_train_augmented = default_training_preprocessing(config, dataset_train)
    dataset_validation_prepared = default_evaluation_preprocessing(config, dataset_validation)

    #####################
    ## Training setup
    metrics.BATCH_SIZE = config["batch_size"]
    model.compile(optimizer='adam', loss='mse', metrics=[motion_metric])

    #####################
    # Callbacks
    filepath = get_checkpoint_filepath(config, pattern='weights.hdf5')
    checkpoint_saver = ModelCheckpoint(filepath,
                                       monitor='val_motion_metric',
                                       save_best_only=True,  # only overwrite if model is better
                                       mode='max'  # higher is better for this metric
                                       )

    #####################
    ## Go!
    history = model.fit_generator(generator=dataset_train_augmented,
                                  validation_data=dataset_validation_prepared,
                                  callbacks=[checkpoint_saver],
                                  epochs=config["epochs"],
                                  verbose=2  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                  )

    return model, history


def run_experiment(config):

    if not os.path.exists(config["dataset_path"]):
        raise FileNotFoundError("Dataset not found at {}".format(config["dataset_path"]))

    if checkpoint_dir_exists(config):
        should_overwrite = ask_yes_no_question("[PROMPT] Overwrite existing checkpoint? {}".format(config['checkpoint_path']))
        if should_overwrite:
            make_or_get_checkpoint_dir(config)
        else:
            print("Skipping experiment...")
            return

    print("===================================")
    print("Starting experiment for model {}".format(config["model"]))
    print("===================================")

    # store visuals and files
    model, hist = train(config)
    history_dict = hist.history
    plot_history(config, history_dict)
    save_history(config, history_dict)
    save_config(config)

    # test data and write results
    test(model, config, store_output=True, evaluate_splits=True)


def main():
    pass

if __name__ == "__main__":
    main()

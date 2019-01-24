import argparse
import os

from kinectgestures.metrics import motion_metric
from kinectgestures.util import load_config
from kinectgestures.test import test
import kinectgestures.metrics as metrics

from keras.models import load_model


def test_checkpoint(checkpoint_path, evaluate_splits=True, store_output=False):
    config = load_config(checkpoint_path)
    metrics.BATCH_SIZE = config["batch_size"]  # set before the metric is compiled
    model = load_model(os.path.join(checkpoint_path, "weights.hdf5"), custom_objects={'motion_metric': motion_metric})
    test(model, config, evaluate_splits=evaluate_splits, store_output=store_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate checkpoint on test set and calculate metrics")
    parser.add_argument("path", help="Path to checkpoint directory")
    args = parser.parse_args()
    test_checkpoint(args.path)

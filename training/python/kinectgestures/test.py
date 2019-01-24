import json
import os
from pprint import pprint
from time import time

import numpy as np

from kinectgestures.data import GestureDataset
from kinectgestures.preprocessing import default_evaluation_preprocessing
from kinectgestures.util import make_output_dir, save_singlechannel, get_output_dir


def test(model, config, store_output=False, evaluate_splits=False):
    #####################
    # Run complete test set
    metrics = {
        "overall":  test_subset(config, model, store_output)
    }

    if not evaluate_splits:
        save_metrics_to_checkpoint(config, metrics)
        return

    #####################
    # Split test set by people count
    counts = [0, 1, 2, 3]
    results_by_count = {}
    for count in counts:
        results_by_count[count] = test_subset(config, model, store_output, filter_count=count)
    metrics["count"] = results_by_count

    #####################
    # Split test set by category of main gestures
    gestures = ['abort', 'check', 'chi', 'circle', 'point', 'stop', 'wave', 'x']
    categories = gestures + ['empty', 'passive']
    results_by_category = {}
    for category in categories:
        results_by_category[category] = test_subset(config, model, store_output, filter_category=category)
    metrics["category"] = results_by_category

    save_metrics_to_checkpoint(config, metrics)


def save_metrics_to_checkpoint(config, metrics):
    pprint(metrics)
    json_path = os.path.join(config["checkpoint_path"], "metrics_test.json")
    with open(json_path, "w") as fp:
        json.dump(metrics, fp, indent=4)
        print("Written to {}".format(json_path))


def test_subset(config, model, should_store_output, filter_category=None, filter_count=None):
    out_path = make_output_dir(config) if should_store_output else get_output_dir(config)
    is_2d_model = config['model'] in ("cnn2d", "vgg16")
    dataset_test = GestureDataset(config["dataset_path"],
                                  which_split='test',
                                  batch_size=config["batch_size"],
                                  last_frame_only=is_2d_model,
                                  filter_category=filter_category,
                                  filter_count=filter_count,
                                  skip_incomplete_batch=True  # FIXME: workaround bc the metric fails on smaller batches
                                  )
    dataset_test = default_evaluation_preprocessing(config, dataset_test)
    print("[BUSY] Running test set...")
    start_time = time()
    motion_metric = []
    for i in range(len(dataset_test)):
        batch, teachers = dataset_test[i]

        loss, motion_metric_batch = model.test_on_batch(batch, teachers)
        motion_metric.append(motion_metric_batch)

        if should_store_output:
            result = model.predict(batch, batch_size=config["batch_size"])

            # iterate through result batch and save each sample
            save_network_output(config, i, out_path, result, teachers)

    print("[DONE] took {:.2f}s".format(time() - start_time))
    motion_metric = np.mean(motion_metric)
    print("Motion metric (mean): {}".format(motion_metric))

    # convert to Python float to make sure it's serializable
    return float(motion_metric)


def save_network_output(config, i, out_path, result, teachers):
    for j in range(len(result)):
        index = i * config["batch_size"] + j
        path_output = os.path.join(out_path, "test-{}-out.png".format(index))
        path_teacher = os.path.join(out_path, "test-{}-teacher.png".format(index))
        save_singlechannel(path_output, result[j])
        save_singlechannel(path_teacher, teachers[j])

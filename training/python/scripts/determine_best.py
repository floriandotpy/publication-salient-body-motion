import os
import glob
import json

from kinectgestures.util import load_history, load_config


def determine_best_motion(hist_dict):
    return max(hist_dict['val_motion_metric'])


if __name__ == "__main__":
    checkpoint_glob = "/home/flo/checkpoints/experiments-find-best-3dcnn/*"
    checkpoints = glob.glob(checkpoint_glob)

    print("=================")
    print("Displaying BEST motion metric on validation set from training histories")
    print("=================")

    results = []

    for checkpoint_path in checkpoints:

        if not os.path.exists(os.path.join(checkpoint_path, "history.json")):
            continue

        hist_dict = load_history(checkpoint_path)
        config = load_config(checkpoint_path)

        with open(os.path.join(checkpoint_path, "metrics_test.json"), "r") as fp:
            metric = json.load(fp)
            motion_test = metric["overall"]
        best_motion = determine_best_motion(hist_dict)

        results.append((motion_test, best_motion, config['model'], checkpoint_path))

    results = sorted(results, key=lambda k: k[0], reverse=True)

    for r in results:
        print("test {:.4f}: val {:.4f}, ({}): {}".format(r[0], r[1], r[2], r[3]))


import datetime
from pprint import pprint

from kinectgestures.experiments import ExperimentGenerator
from kinectgestures.notify import notify_all_trainings_done, notify_training_crashed, notify_start_trainings
from kinectgestures.train import run_experiment

base_config = {
    "dataset_path": "/datasets/kinect-gestures-v1-240x320",
    "out_shape": [60, 80],  # exactly half the input size, original: [120, 160]
    "in_shape": [120, 160],
    "preprocessing_scale": [180, 220],
    "preprocessing_scale_teacher": [60, 80],
    "batch_size": 16,
    "epochs": 50,
}


def generate_basic_experiments():
    vgg16 = {
        "checkpoint_path": "/checkpoints/vgg16-normalized-more-dropout",
        "model": "vgg16",
        **base_config
    }
    cnn2d = {
        "checkpoint_path": "/checkpoints/cnn2d",
        "model": "cnn2d",
        **base_config
    }
    cnn3d = {
        "checkpoint_path": "/checkpoints/cnn3d",
        "model": "cnn3d",
        **base_config
    }

    return [vgg16, cnn2d, cnn3d]


def generate_scale_experiments():
    configs = []
    checkpoint_root = "/checkpoints/experiments-scaling"
    combinations = {
        "checkpoint_path": checkpoint_root,
        "model": ["vgg16", "cnn2d", "cnn3d"],
        "preprocessing_scale": list(map(lambda s: [int(s[0]), int(s[1])], [
            # [240 * 1.0, 320 * 1.0],
            # [240 * 0.9, 320 * 0.9],
            # [240 * 0.8, 320 * 0.8],
            # [240 * 0.7, 320 * 0.7],
            # [240 * 0.6, 320 * 0.6]
            [240 * 0.5, 320 * 0.5]  # smallest scale will match exactly the model input, so _no_ cropping will happen
        ]))
    }

    generator = ExperimentGenerator(base_config, combinations,
                                    checkpoint_path_pattern="{model}-scale-{preprocessing_scale[0]}x{preprocessing_scale[1]}")
    generated = generator.generate()

    # some manual settings after automatic combinations
    for config in generated:
        config["in_shape"] = [120, 160, 15, 1] if "3d" in config["model"] else [120, 160, 1]
        configs.append(config)

    return configs


def generate_find_best_3dcnn_experiments():
    variations = {
        "checkpoint_path": "/checkpoints/experiments-find-best-3dcnn",
        "batch_size": 4,
        "model": ["cnn3d"],
        "kernel_base": [4, 8, 16],
        "num_features": [int(0.5 * 768), 1 * 768, 2 * 768],
        "dropout_rate": [0.5],
        "preprocessing_scale": list(map(lambda s: [int(s[0]), int(s[1])], [
            # [240 * 1.0, 320 * 1.0],
            # [240 * 0.9, 320 * 0.9],
            # [240 * 0.8, 320 * 0.8],
            # [240 * 0.7, 320 * 0.7]
            [240 * 0.6, 320 * 0.6]
            # [240 * 0.5, 320 * 0.5]  # smallest scale will match exactly the model input, so _no_ cropping will happen
        ]))
    }

    generator = ExperimentGenerator(base_config, variations,
                                    checkpoint_path_pattern="{model}-scale-{preprocessing_scale[0]}x{preprocessing_scale[1]}-kb-{kernel_base}-ft-{num_features}-bs-{batch_size}-drop-{dropout_rate}")
    generated = generator.generate()

    # some manual settings after automatic combinations
    configs = []
    for config in generated:
        config["in_shape"] = [120, 160, 15, 1] if "3d" in config["model"] else [120, 160, 1]
        configs.append(config)

    return configs


def generate_final_architecture_runs(num_runs=3):
    """
    <num_runs> runs each of the final configurations used in the journal paper
    for the 2 architectures: 2dcnn and 3dcnn.

    :return: List of experiment configs
    """

    configs = []
    for n in range(num_runs):
        config_2dcnn = {
            "dataset_path": "/datasets/kinect-gestures-v1-240x320",
            "out_shape": [60, 80],
            "in_shape": [120, 160, 1],
            "preprocessing_scale": [144, 192],
            "preprocessing_scale_teacher": [60, 80],
            "batch_size": 4,
            "epochs": 50,
            "checkpoint_path": "/checkpoints/experiments-2dvs3d/cnn2d-run-{}".format(
                n),
            "model": "cnn2d",
            "kernel_base": 16,
            "num_features": 384,
            "dropout_rate": 0.5
        }
        configs.append(config_2dcnn)

    for n in range(num_runs):
        config_3dcnn = {
            "dataset_path": "/datasets/kinect-gestures-v1-240x320",
            "out_shape": [60, 80],
            "in_shape": [120, 160, 15, 1],
            "preprocessing_scale": [144, 192],
            "preprocessing_scale_teacher": [60, 80],
            "batch_size": 4,
            "epochs": 50,
            "checkpoint_path": "/checkpoints/experiments-2dvs3d/cnn3d-run-{}".format(n),
            "model": "cnn3d",
            "kernel_base": 16,
            "num_features": 384,
            "dropout_rate": 0.5
        }
        configs.append(config_3dcnn)
    return configs


def generate_final_vgg16_runs(num_runs=3):
    configs = [{
        "dataset_path": "/datasets/kinect-gestures-v1-240x320",
        "out_shape": [
            60,
            80
        ],
        "in_shape": [
            120,
            160,
            1
        ],
        "preprocessing_scale": [
            144,
            192
        ],
        "preprocessing_scale_teacher": [
            60,
            80
        ],
        "batch_size": 4,
        "epochs": 50,
        "checkpoint_path": "/checkpoints/experiments-vgg16-runs/vgg16-run-{}".format(
            n),
        "model": "vgg16",
        "num_features": 384,
        "dropout_rate": 0.5,
        "pretrained": True
    } for n in range(num_runs)]
    return configs


def generate_final_vgg16_from_scratch_runs():
    configs = generate_final_vgg16_runs()
    for conf in configs:
        conf["pretrained"] = False
        conf["checkpoint_path"] = conf["checkpoint_path"].replace("experiments-vgg16-runs",
                                                                  "experiments-vgg16-from-scratch-runs")
    return configs


def run_experiments(which):
    configs = which()
    notify_start_trainings(configs)

    for i, config in enumerate(configs):
        datetime_start = datetime.datetime.now()
        print("=== Starting experiment {} of total {} | {} ===".format(i, len(configs), datetime_start.strftime("%c")))

        pprint(config)
        run_experiment(config)
        datetime_end = datetime.datetime.now()
        minutes_diff = (datetime_end - datetime_start).total_seconds() / 60.0
        print("=== End experiment | took {:.1f} minutes ===".format(minutes_diff))

    notify_all_trainings_done(configs)


if __name__ == "__main__":

    try:

        # 1: Determine best hyper params for 3d CNN
        # run_experiments(generate_find_best_3dcnn_experiments)

        # --- !! now, look at results of 1. and update hyper parameters for 2

        # 2: run all final architectures 3 times each
        # generate_final_architecture_runs -> done
        for which in [generate_final_vgg16_runs,
                      generate_final_vgg16_from_scratch_runs]:
            run_experiments(which)

    except Exception as e:
        notify_training_crashed(e)
        raise e

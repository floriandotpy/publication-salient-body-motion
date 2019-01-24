import json
import os
import sys

import numpy as np
import PIL

from kinectgestures.data import inv_normalize


def save_singlechannel(path, arr):
    arr8 = inv_normalize(arr).astype(np.uint8)
    img = PIL.Image.fromarray(arr8)
    img.save(path)


def make_output_dir(config):
    out_path = get_output_dir(config)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def get_output_dir(config):
    out_path = os.path.join(config["checkpoint_path"], "out")
    return out_path


def ask_yes_no_question(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Src: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input#3041990
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def checkpoint_dir_exists(config):
    return os.path.exists(config["checkpoint_path"])


def make_or_get_checkpoint_dir(config):
    os.makedirs(config["checkpoint_path"], exist_ok=True)
    return config["checkpoint_path"]


def get_checkpoint_dir(config):
    return make_or_get_checkpoint_dir(config)


def get_checkpoint_filepath(config, pattern):
    return os.path.join(make_or_get_checkpoint_dir(config), pattern)


def save_history(config, history_dict):
    target_dir = get_checkpoint_dir(config)
    target_path = os.path.join(target_dir, "history.json")
    save_to_json(history_dict, target_path)


def load_history(checkpoint_path):
    history_path = os.path.join(checkpoint_path, "history.json")
    return load_json(history_path)


def load_config(checkpoint_path):
    config_path = os.path.join(checkpoint_path, "config.json")
    return load_json(config_path)


def save_config(config):
    target_dir = get_checkpoint_dir(config)
    target_path = os.path.join(target_dir, "config.json")
    save_to_json(config, target_path)


def save_to_json(obj, target_path):
    with open(target_path, "w") as fp:
        json.dump(obj, fp, indent=4)


def load_json(path):
    with open(path, "r") as fp:
        return json.load(fp)
import argparse
from kinectgestures.util import load_json


from scripts.run_experiments import run_experiments


def train(config_path):
    config = load_json(config_path)
    run_experiments(lambda: [config])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run training for a given config file")
    parser.add_argument("path", help="Path to config json")
    args = parser.parse_args()
    train(args.path)

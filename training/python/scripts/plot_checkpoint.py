import argparse
from kinectgestures.util import load_config, load_history
from kinectgestures.visuals import plot_history


def create_plots(checkpoint_path, save_figures=False):
    config = load_config(checkpoint_path)
    history = load_history(checkpoint_path)

    if not save_figures:
        print("[Notice] Not saving files, only showing figures interactively.")
    else:
        print("[Notice] Writing figures to directory: {}".format(checkpoint_path))

    plot_history(config, history, show=not save_figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create plots from an existing checkpoint directory")
    parser.add_argument("path", help="Path to checkpoint directory")
    args = parser.parse_args()
    create_plots(args.path)
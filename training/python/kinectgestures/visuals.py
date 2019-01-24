import os

from kinectgestures.util import get_checkpoint_dir
import matplotlib.pyplot as plt


def clear_plot():
    plt.gcf().clear()


def plot_history(config, history_dict, show=False):

    # prepare paths
    target_dir = get_checkpoint_dir(config)
    path_fig_metric = os.path.join(target_dir, "plot_motion.png")
    path_fig_loss = os.path.join(target_dir, "plot_loss.png")

    # Plot training & validation accuracy values
    plt.plot(history_dict['motion_metric'])
    plt.plot(history_dict['val_motion_metric'])
    plt.title('Motion metric: {}'.format(config["model"]))
    plt.ylabel('Motion')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    if show:
        plt.show()
    else:
        plt.savefig(path_fig_metric)
        # savefig('foo.png', bbox_inches='tight')
    clear_plot()

    # Plot training & validation loss values
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('Loss: {}'.format(config["model"]))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    if show:
        plt.show()
    else:
        plt.savefig(path_fig_loss)
    clear_plot()


import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

colors = {
    "3d": "green",
    "2d": "red"
}

COLOR_2D = '#3bc9db'
COLOR_3D = '#40c057'
COLOR_VGG16SCR = '#bbbbbb'
COLOR_VGG16PRE = '#888888'
COLOR_RAND = '#ffa94d'


def autolabel(ax, rects, labels, size, number_inside=False):
    """
    Attach a text label above each bar displaying its height
    """
    font = matplotlib.font_manager.FontProperties()
    font.set_size('16')
    # if number_inside:
    #     font.set_weight('bold')
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if labels[i] == 0:
            continue
        pos = 0.05 if number_inside else height
        ax.text(rect.get_x() + rect.get_width() / 2., pos,
                '%.2f' % labels[i],
                fontproperties=font,
                ha='center', va='bottom')


def plot_results_grouped_args(main_label, plot_labels, groups, values_2d, values_3d, narrow=False, yerr_2d=None,
                              yerr_3d=None, legend_outside=False):
    n_groups = len(groups)
    offset = 0.2
    ind = np.arange(n_groups) * 1.2 + offset  # the x locations for the groups
    width = 0.4  # the width of the bars

    size = (16, 7) if not narrow else (10, 5)
    fig = plt.figure(figsize=size)
    fig.add_subplot()
    ax = plt.gca()
    # fig.set_size_inches(10, 2)
    # ax.errorbar(ind, values_2d, yerr=yerr_2d, xerr=0.1, fmt='o', capthick=2)
    rects1 = ax.bar(ind, values_2d, width, color=COLOR_2D, yerr=yerr_2d)
    rects2 = ax.bar(ind + width, values_3d, width, color=COLOR_3D, yerr=yerr_3d)

    ax.set_ylabel('Scores')
    ax.set_title(main_label, loc='left')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(groups)
    # ax.set_ylim([0, 1.2])

    autolabel(ax, rects2, values_3d, 'x-small', number_inside=True)
    autolabel(ax, rects1, values_2d, 'x-small', number_inside=True)

    # shrink main plot
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    box = None
    if legend_outside:
        box = (1, 1)
    ax.legend((rects1[0], rects2[0]), plot_labels, loc='upper right', ncol=2, framealpha=0, bbox_to_anchor=box)
    plt.show()


def get_scores_from_checkpoint(checkpoint_dir, which, keys):
    assert which == "category" or which == "count"

    # in graph, we show it as 'doublewave' while in the data it's called 'chi'
    if which == "category":
        keys = [c if c != "doublewave" else "chi" for c in keys]

    with open(os.path.join(checkpoint_dir, "metrics_test.json"), "r") as fp:
        metrics = json.load(fp)

    return [metrics[which][str(k)] for k in keys]


def get_means_from_checkpoints(checkpoint_dirs, which, keys):
    return get_stats_from_checkpoints(checkpoint_dirs, which, keys, stat_fn=np.mean)


def get_std_from_checkpoints(checkpoint_dirs, which, keys):
    return get_stats_from_checkpoints(checkpoint_dirs, which, keys, stat_fn=np.std)


def get_stats_from_checkpoints(checkpoint_dirs, which, keys, stat_fn):
    assert stat_fn == np.mean or stat_fn == np.std
    assert which == "category" or which == "count"

    if type(checkpoint_dirs) != list:
        checkpoint_dirs = list(checkpoint_dirs)

    # collect list of scores from all checkpoints
    scores = []
    for checkpoint_dir in checkpoint_dirs:
        scores.append(get_scores_from_checkpoint(checkpoint_dir, which, keys))

    # group all scores for the same group together
    scores = zip(*scores)

    # compute statistic over each group of values
    stats = [stat_fn(values) for values in scores]

    return stats


def plot_results_grouped(checkpoints_2d, checkpoints_3d):
    font = {'family': 'normal', 'weight': 'normal', 'size': '19'}

    matplotlib.rc('font', **font)
    matplotlib.rcParams['lines.linewidth'] = 4

    # general
    plots = ('2D-Net', '3D-Net')
    gestures = ('abort', 'check', 'doublewave', 'circle', 'point', 'stop', 'wave', 'x')
    KEYS_CATEGORIES = gestures + ('empty', 'passive')
    KEYS_COUNTS = [0, 1, 2, 3]
    WHICH_CATEGORY = "category"
    WHICH_COUNT = "count"

    # Plot 1: motion by gesture
    motion_label = 'Network performance by gesture'
    motion_2d_means = get_means_from_checkpoints(checkpoints_2d,
                                                 which=WHICH_CATEGORY,
                                                 keys=KEYS_CATEGORIES)
    motion_2d_stds = get_std_from_checkpoints(checkpoints_2d,
                                              which=WHICH_CATEGORY,
                                              keys=KEYS_CATEGORIES)

    motion_3d_means = get_means_from_checkpoints(checkpoints_3d,
                                                 which=WHICH_CATEGORY,
                                                 keys=KEYS_CATEGORIES)
    motion_3d_stds = get_std_from_checkpoints(checkpoints_3d,
                                              which=WHICH_CATEGORY,
                                              keys=KEYS_CATEGORIES)

    plot_results_grouped_args(motion_label, plots, KEYS_CATEGORIES, motion_2d_means, motion_3d_means,
                              yerr_2d=motion_2d_stds,
                              yerr_3d=motion_3d_stds, legend_outside=True)

    # plot 2: motion by count
    motion_label = 'Network performance by scene configuration'
    counts = ['%d %s' % (c, 'person' if c == 1 else 'people') for c in (0, 1, 2, 3)]
    motion_2d_means = get_means_from_checkpoints(checkpoints_2d,
                                                 which=WHICH_COUNT,
                                                 keys=KEYS_COUNTS)
    motion_2d_stds = get_std_from_checkpoints(checkpoints_2d,
                                              which=WHICH_COUNT,
                                              keys=KEYS_COUNTS)

    motion_3d_means = get_means_from_checkpoints(checkpoints_3d,
                                                 which=WHICH_COUNT,
                                                 keys=KEYS_COUNTS)
    motion_3d_stds = get_std_from_checkpoints(checkpoints_3d,
                                              which=WHICH_COUNT,
                                              keys=KEYS_COUNTS)

    plot_results_grouped_args(motion_label, plots, counts, motion_2d_means, motion_3d_means, narrow=True,
                              yerr_2d=motion_2d_stds, yerr_3d=motion_3d_stds, legend_outside=True)


def get_overall_stats_from_checkpoints(checkpoint_dirs):
    if type(checkpoint_dirs) != list:
        checkpoint_dirs = list(checkpoint_dirs)

    values = []
    for checkpoint_dir in checkpoint_dirs:
        with open(os.path.join(checkpoint_dir, "metrics_test.json"), "r") as fp:
            metrics = json.load(fp)
        values.append(metrics["overall"])

    return np.mean(values), np.std(values)


def plot_results(checkpoints_2d,
                 checkpoints_3d,
                 checkpoints_vgg16_pretrained,
                 checkpoints_vgg16_from_scratch):
    font = {'family': 'normal', 'weight': 'normal', 'size': '16'}

    matplotlib.rc('font', **font)
    matplotlib.rcParams['lines.linewidth'] = 4

    model_names = ('Guessing', 'VGG16', 'VGG16pre', '2D-Net', '3D-Net')
    model_random = 0.05
    model_2d, model_2d_std = get_overall_stats_from_checkpoints(checkpoints_2d)
    model_3d, model_3d_std = get_overall_stats_from_checkpoints(checkpoints_3d)
    model_vgg16pre, model_vgg16pre_std = get_overall_stats_from_checkpoints(checkpoints_vgg16_pretrained)
    model_vgg16scr, model_vgg16scr_std = get_overall_stats_from_checkpoints(checkpoints_vgg16_from_scratch)

    ind = np.arange(1) * 1.5  # the x locations for the groups
    width = 0.45  # the width of the bars
    offset = 0.15

    fig, ax = plt.subplots()

    rects1 = ax.bar(offset + ind, model_random, width * 0.8, color=COLOR_RAND)
    rects2 = ax.bar(offset + ind + width, model_vgg16scr, width * 0.8, color=COLOR_VGG16SCR, yerr=model_vgg16scr_std)
    rects3 = ax.bar(offset + ind + width * 2, model_vgg16pre, width * 0.8, color=COLOR_VGG16PRE, yerr=model_vgg16pre_std)
    rects4 = ax.bar(offset + ind + width * 3, model_2d, width * 0.8, color=COLOR_2D, yerr=model_2d_std)
    rects5 = ax.bar(offset + ind + width * 4, model_3d, width * 0.8, color=COLOR_3D,
                    yerr=model_3d_std)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Performance')
    ax.set_title('Network performance overall')
    # ax.set_xticks(offset + ind + width)
    ax.set_xticklabels(('', ''))
    ax.set_ylim([0, 1])

    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), model_names, loc='upper center', ncol=3)

    autolabel(ax, rects1, [model_random], 'medium', number_inside=True)
    autolabel(ax, rects2, [model_vgg16scr], 'medium', number_inside=True)
    autolabel(ax, rects3, [model_vgg16pre], 'medium', number_inside=True)
    autolabel(ax, rects4, [model_2d], 'medium', number_inside=True)
    autolabel(ax, rects5, [model_3d], 'medium', number_inside=True)

    plt.show()


def main():

    def join_root(path):
        return os.path.join("/Users/flo/checkpoints/kinect-gestures/", path)

    checkpoints_2d = [join_root(path) for path in [
        "experiments-2dvs3d/cnn2d-run-0",
        "experiments-2dvs3d/cnn2d-run-1",
        "experiments-2dvs3d/cnn2d-run-2"]
    ]

    checkpoints_3d = [join_root(path) for path in [
        "experiments-2dvs3d/cnn3d-run-0",
        "experiments-2dvs3d/cnn3d-run-1",
        "experiments-2dvs3d/cnn3d-run-2"]
                      ]
    checkpoints_vgg16_pretrained = [join_root(path) for path in
        ["experiments-vgg16-runs/vgg16-run-0",
         "experiments-vgg16-runs/vgg16-run-1",
         "experiments-vgg16-runs/vgg16-run-2"]]

    checkpoints_vgg16_from_scratch = [join_root(path) for path in
                                    ["experiments-vgg16-from-scratch-runs/vgg16-run-0",
                                     "experiments-vgg16-from-scratch-runs/vgg16-run-1",
                                     "experiments-vgg16-from-scratch-runs/vgg16-run-2"]]


    plot_results_grouped(checkpoints_2d, checkpoints_3d)
    plot_results(checkpoints_2d,
                 checkpoints_3d,
                 checkpoints_vgg16_pretrained,
                 checkpoints_vgg16_from_scratch)


if __name__ == "__main__":
    main()

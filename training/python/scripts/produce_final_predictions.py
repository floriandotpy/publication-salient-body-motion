"""
This script feeds the test set through the final model instances and saves their resulting output to image files
which can then be used for visualization in the paper.
"""
import os
import shutil
from collections import Counter

from keras.engine.saving import load_model

from kinectgestures import metrics
from kinectgestures.data import GestureDataset
from kinectgestures.metrics import motion_metric
from kinectgestures.preprocessing import default_evaluation_preprocessing
from kinectgestures.util import load_config, save_singlechannel


def get_dataset(config):
    is_2d_model = config["model"] in ("cnn2d", "vgg16")
    return GestureDataset(config["dataset_path"],
                          which_split='test',
                          batch_size=1,
                          last_frame_only=is_2d_model,
                          filter_category=None,
                          filter_count=None,
                          skip_incomplete_batch=True  # FIXME: workaround bc the metric fails on smaller batches
                          )


def main():
    target_dir = "/home/flo/datasets/testset-outputs"
    path_checkpoint_2d = "/home/flo/checkpoints/experiments-2dvs3d/cnn2d-run-0"
    path_checkpoint_3d = "/home/flo/checkpoints/experiments-2dvs3d/cnn3d-run-0"
    path_checkpoint_vgg = "/home/flo/checkpoints/experiments-vgg16-runs/vgg16-run-0"

    config2d = load_config(path_checkpoint_2d)
    config3d = load_config(path_checkpoint_3d)
    metrics.BATCH_SIZE = 1  # set before the metric is compiled

    # init all 3 model instances
    model2d = load_model(os.path.join(path_checkpoint_2d, "weights.hdf5"),
                         custom_objects={'motion_metric': motion_metric})
    model3d = load_model(os.path.join(path_checkpoint_3d, "weights.hdf5"),
                         custom_objects={'motion_metric': motion_metric})
    model_vgg = load_model(os.path.join(path_checkpoint_vgg, "weights.hdf5"),
                           custom_objects={'motion_metric': motion_metric})

    # we need 2 different dataset instances becase 2d vs 3d models take different input shapes
    dataset_2d = get_dataset(config2d)
    dataset_3d = get_dataset(config3d)

    # to actually feed to the network, input samples have to be preprocessed
    dataset_2d_preprocessed = default_evaluation_preprocessing(config2d, dataset_2d)
    dataset_3d_preprocessed = default_evaluation_preprocessing(config3d, dataset_3d)

    category_counter = Counter()  # used to add unique number to each sample from the same category
    for i in range(len(dataset_2d)):

        # Which category (gesture/action) does this sample belong to?
        category = dataset_2d.get_category_of(i)
        category_counter[category] += 1
        print("Sample with category/class: {}{}".format(category, category_counter[category]))

        # Define all filenames we want to write to
        fname_core = "{}{}".format(category, category_counter[category])
        fname_teacher = os.path.join(target_dir, "{}-teacher.png".format(fname_core))
        fname_still = os.path.join(target_dir, "{}-still.png".format(fname_core))
        fname_2d = os.path.join(target_dir, "{}-result-2d.png".format(fname_core))
        fname_3d = os.path.join(target_dir, "{}-result-3d.png".format(fname_core))
        fname_vgg = os.path.join(target_dir, "{}-result-vgg.png".format(fname_core))

        # copy still frame and teacher to target location
        teacher_src = dataset_2d.get_filename_teacher(i)
        still_src = dataset_2d.get_filename_still(i)
        shutil.copy(teacher_src, fname_teacher)
        shutil.copy(still_src, fname_still)

        # Save 2d prediction to target location
        batch2d, teachers2d = dataset_2d_preprocessed[i]
        result2d = model2d.predict(batch2d, batch_size=1)
        save_singlechannel(fname_2d, result2d[0])

        # Save vgg prediction to target location
        result_vgg = model_vgg.predict(batch2d, batch_size=1)
        save_singlechannel(fname_vgg, result_vgg[0])

        # Save 3d prediction to target location
        batch3d, teachers = dataset_3d_preprocessed[i]
        result3d = model3d.predict(batch3d, batch_size=1)
        save_singlechannel(fname_3d, result3d[0])


if __name__ == '__main__':
    main()

import glob
import os

import numpy as np
import scipy
from keras.utils import Sequence


class GestureDataset(Sequence):

    def __init__(self, path, batch_size, which_split, last_frame_only=True, skip_incomplete_batch=False,
                 filter_category=None, filter_count=None):
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.last_frame_only = last_frame_only
        self.skip_incomplete_batch = skip_incomplete_batch

        if which_split == 'train':
            fold_ids = "012"
        elif which_split == "validation":
            fold_ids = "3"
        elif which_split == "test":
            fold_ids = "4"
        else:
            raise ValueError("Wrong value, specify one of the following dataset splits: train, validation, test")

        # only grab samples for this split
        self.dirs = glob.glob("{}/fold_[{}]/*".format(self.path, fold_ids))

        if filter_category and filter_count:
            raise ValueError("Cannot filter by category AND person count at the same time")

        if filter_category:
            self.dirs = [d for d in self.dirs if GestureDataset.is_sample_in_category(d, filter_category)]
            print("== Filter by category {}, samples matching: {}".format(filter_category, len(self.dirs)))

        if filter_count is not None:
            self.dirs = [d for d in self.dirs if GestureDataset.has_people_count(d, filter_count)]
            print("== Filter by count {}, samples matching: {}".format(filter_count, len(self.dirs)))

    def __len__(self):
        "Returns the number of batches"
        round_fn = np.floor if self.skip_incomplete_batch else np.ceil
        TMP_OFFSET = 1  # FIXME: workaround to avoid the incomplete last batch
        return int(round_fn(len(self.dirs) / float(self.batch_size))) - TMP_OFFSET

    def __getitem__(self, idx):
        "Returns a batch of a given index"
        batch_dirs = self.dirs[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([self.read_sequence(dirname, self.last_frame_only) for dirname in batch_dirs]), \
               np.array([self.read_teacherframe(dirname) for dirname in batch_dirs])

    def read_sequence(self, folder, last_frame_only):
        """
        Read image sequence and teacher signal from a given folder.

        :param folder: Path to folder
        :param last_frame_only If set to True, only last frame will be read,
                               otherwise the whole sequence of frames
        :return: tuple of sequence and teacher image:
            (sequence, teacher)
            | sequence is a numpy array of images
            | teacher is a numpy array
        """

        if last_frame_only:  # only read last frame
            fn_frame = os.path.join(folder, "depth-14.png")
            sequence = normalize(scipy.ndimage.imread(fn_frame))

        else:  # read 15 frames
            files = [os.path.join(folder, "depth-%d.png" % i) for i in range(15)]

            sequence = np.array([normalize(scipy.ndimage.imread(fn)) for fn in files])
            sequence = np.swapaxes(np.swapaxes(sequence, 0, 1), 2,
                                   1)  # reorder so that (480, 640, 15) -> (15, 480, 640)

        sequence = np.expand_dims(sequence, 3)  # add channel dimension (for now it's 1)
        return sequence

    def read_teacherframe(self, dirname):
        file_teacher = os.path.join(dirname, '_teacher.png')
        teacher = normalize(scipy.ndimage.imread(file_teacher))
        # teacher = np.expand_dims(teacher, 3)  # add channel dimension (for now it's 1)
        return teacher

    @staticmethod
    def is_sample_in_category(path, category):
        folder = os.path.basename(path)
        has_active_person = any("-{}-".format(category) in folder for category in
                                ['abort', 'check', 'chi', 'circle', 'point', 'stop', 'wave', 'x'])

        if category in ['abort', 'check', 'chi', 'circle', 'point', 'stop', 'wave', 'x']:
            return "-{}-".format(category) in folder
        elif category == "empty":
            return folder[-1] == '_' and not has_active_person
        elif category == "passive":
            has_passive_person = '-standing-' in folder or '-still-' in folder or '-passive-' in folder
            return has_passive_person and not has_active_person
        else:
            raise ValueError("Unknown category filter: {}".format(category))

    @staticmethod
    def has_people_count(path, filter_count):
        if not filter_count in [0, 1, 2, 3]:
            raise ValueError("Unknown value for filter_count: {}".format(filter_count))
        folder = os.path.basename(path)
        return folder.count("depth") == filter_count

    def get_category_of(self, i):
        path = self.dirs[i]
        for category in ['abort', 'check', 'chi', 'circle', 'point', 'stop', 'wave', 'x', 'empty', 'passive']:
            if self.is_sample_in_category(path, category):
                return category
        raise ValueError("Weird. Current sample not in any known category. Looks like a bug: {}".format(path))

    def get_filename_teacher(self, i):
        dirname = self.dirs[i]
        return os.path.join(dirname, '_teacher.png')

    def get_filename_still(self, i):
        dirname = self.dirs[i]
        return os.path.join(dirname, 'depth-14.png')


def normalize(arr):
    """
    Normalize an array. Values in range 0 .. 255 are mapped to range -1.0 .. +1.0
    :param arr: Array
    :return: Normalized array of identical dimensions
    """
    arr = arr / (256.0 / 2.0)  # FIXME: this looks wrong. I _think_ I should just replace with batchnorm anyways
    arr = arr - 1.0
    return arr


def inv_normalize(arr):
    """
    Invert the normalization, for displayable images. Maps range -1.0 .. +1.0 to range 0 .. 255
    :param arr: Normalized array
    :return: Array with values 0 .. 255 (aka "image")
    """
    arr[arr < -1.0] = -1.0
    arr[arr > 1.0] = 1.0
    arr = arr + 1.0
    arr = arr * (256.0 / 2.0)
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr

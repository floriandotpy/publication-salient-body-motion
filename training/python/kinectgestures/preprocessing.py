from keras.utils import Sequence
import numpy as np
import cv2


def default_training_preprocessing(config, dataset):
    crop_size = config['in_shape'][0:2]  # remove channel dim
    return VideoDataPreparer(scale_size_before_cropping=config["preprocessing_scale"],
                             crop_size=crop_size,
                             crop_strategy="random",
                             scale_size_teacher=config["preprocessing_scale_teacher"],
                             dataset=dataset)


def default_evaluation_preprocessing(config, dataset):
    crop_size = config['in_shape'][0:2]  # remove channel dim
    return VideoDataPreparer(scale_size_before_cropping=config["preprocessing_scale"],
                             crop_size=crop_size,
                             crop_strategy="center",
                             scale_size_teacher=config["preprocessing_scale_teacher"],
                             dataset=dataset)


class VideoDataPreparer(Sequence):
    def __init__(self, dataset, scale_size_before_cropping=None, crop_size=None, crop_strategy='random', scale_size_teacher=None):
        self.dataset = dataset

        # augmentation operations
        self.ops = []

        if scale_size_before_cropping is not None:
            self.ops.append(ScaleOp(scale_size_before_cropping))

        if crop_size is not None and crop_strategy == 'random':
            self.ops.append(RandomCropOp(crop_size))

        if crop_size is not None and crop_strategy == 'center':
            self.ops.append(CenterCropOp(crop_size))

        if scale_size_teacher is not None:
            self.ops.append(ScaleTeacherOp(scale_size_teacher))

    def add(self, op):
        self.ops.append(op)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, teacher = self.dataset[idx]

        for op in self.ops:
            sample, teacher = op(sample, teacher)

        return sample, teacher


def resize_single_frame(frame, size):

    # flip size tuple for cv
    height, width = size
    size = tuple((width, height))

    has_single_channel = len(frame.shape) >= 3 and frame.shape[2] == 1
    frame = cv2.resize(frame, size)
    if has_single_channel:
        frame = np.expand_dims(frame, axis=-1)
    return frame


def resize_single_video(video, size):
    video = video.transpose((2, 0, 1, 3))  # move time to front
    video = np.array([resize_single_frame(frame, size) for frame in video])
    video = video.transpose((1, 2, 0, 3))  # restore shape
    return video


def scale_videos(batch_of_videos, batch_of_teachers, size, teacher_only=False):
    if not teacher_only:
        batch_of_videos = np.array([resize_single_video(batch_item, size) for batch_item in batch_of_videos])
    batch_of_teachers = np.array([resize_single_frame(batch_item, size) for batch_item in batch_of_teachers])
    return batch_of_videos, batch_of_teachers


def scale_frames(batch_of_frames, batch_of_teachers, size, teacher_only=False):
    if not teacher_only:
        batch_of_frames = np.array([resize_single_frame(batch_item, size) for batch_item in batch_of_frames])
    batch_of_teachers = np.array([resize_single_frame(batch_item, size) for batch_item in batch_of_teachers])
    return batch_of_frames, batch_of_teachers


def center_crop(sample, teacher, crop_size):
    target_height, target_width = crop_size
    assert sample.shape[1] >= target_height
    assert sample.shape[2] >= target_width
    assert sample.shape[1] == teacher.shape[1]
    assert sample.shape[2] == teacher.shape[2]
    x = (sample.shape[2] - target_width) // 2
    y = (sample.shape[1] - target_height) // 2
    sample = sample[:, y:y + target_height, x:x + target_width, :]
    teacher = teacher[:, y:y + target_height, x:x + target_width]
    return sample, teacher


def random_crop(sample, teacher, crop_size):
    """
    Randomly crop a sample (video) and a teacher frame (image) according
    to the same random crop parameters (s.t. they still ligh up afterwards)

    :param sample: Sample (video), a 4d array: (height, width, time, channels)
    :param teacher: Teacher frame, a 2d array: (heigh, width)
    :param crop_size: The desired crop size, a 2d tuple: (heght, width)
    :return:
    """

    height, width = crop_size
    # shapes: (batch, height, width, time?, channels)
    assert sample.shape[1] >= height
    assert sample.shape[2] >= width
    assert sample.shape[1] == teacher.shape[1]
    assert sample.shape[2] == teacher.shape[2]
    x = np.random.randint(0, sample.shape[2] - width) if sample.shape[2] != width else 0
    y = np.random.randint(0, sample.shape[1] - height) if sample.shape[1] != height else 0
    sample = sample[:, y:y + height, x:x + width, :]
    teacher = teacher[:, y:y + height, x:x + width]
    return sample, teacher


class RandomCropOp(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample, teacher):
        return random_crop(sample, teacher, self.crop_size)


class CenterCropOp(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample, teacher):
        return center_crop(sample, teacher, self.crop_size)


class ScaleOp(object):
    def __init__(self, size, teacher_only=False):
        self.size = size
        self.teacher_only = teacher_only

    def __call__(self, sample, teacher):
        if len(sample.shape) == 4:
            return scale_frames(sample, teacher, self.size, teacher_only=self.teacher_only)
        elif len(sample.shape) == 5:
            return scale_videos(sample, teacher, self.size, teacher_only=self.teacher_only)


class ScaleTeacherOp(ScaleOp):
    def __init__(self, size):
        super().__init__(size, teacher_only=True)
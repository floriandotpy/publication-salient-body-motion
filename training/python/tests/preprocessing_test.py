import logging

# disable excessive tf log messages
logging.getLogger('tensorflow').disabled = True

from unittest.mock import Mock
import numpy as np
from kinectgestures.preprocessing import RandomCropOp, VideoDataPreparer, ScaleOp, CenterCropOp
from nose.tools import assert_equals


def generate_video(shape):
    return np.zeros(shape)


def test_random_crop_op_returns_correct_shape():
    video_before = generate_video((8, 32, 32, 3))
    teacher_before = generate_video((8, 32, 32))
    target_shape_sample = (8, 16, 16, 3)
    target_shape_teacher = (8, 16, 16)
    crop_size = (16, 16)

    op = RandomCropOp(crop_size=crop_size)
    video_after, teacher_after = op(video_before, teacher_before)

    assert_equals(video_after.shape, target_shape_sample)
    assert_equals(teacher_after.shape, target_shape_teacher)


def test_random_crop_op_keeps_sample_and_teacher_aligned():
    video_before = generate_video((8, 7, 7, 3))
    teacher_before = generate_video((8, 7, 7))
    crop_size = (4, 4)

    # mark center pixel white
    px_value = 255.0
    video_before[:, 3, 3, :] = px_value
    teacher_before[:, 3, 3] = px_value

    op = RandomCropOp(crop_size=crop_size)
    video_after, teacher_after = op(video_before, teacher_before)

    # find transformed location in teacher frame
    _, y, x = np.where(teacher_after == px_value)

    # only look at first sample in batch
    y = y[0]
    x = x[0]

    # in video this must still be the same location
    assert_equals(video_after[0, y, x, 0], px_value)


def test_video_data_augmentor_inits_with_crop_op():
    crop_size = (16, 16)

    dataset = Mock()
    augmentor_with_crop = VideoDataPreparer(crop_size=crop_size, dataset=dataset)
    augmentor_without_crop = VideoDataPreparer(dataset=dataset)

    assert_equals(len(augmentor_with_crop.ops), 1)
    assert_equals(len(augmentor_without_crop.ops), 0)
    assert_equals(augmentor_with_crop.ops[0].crop_size, crop_size)


def test_scale_returns_correct_shape():
    video_before = generate_video((8, 64, 64, 3))
    teacher_before = generate_video((8, 64, 64))

    target_size = (16, 32)

    target_shape_video = (8, 16, 32, 3)
    target_shape_teacher = (8, 16, 32)

    op = ScaleOp(target_size)
    video_after, teacher_after = op(video_before, teacher_before)

    assert_equals(video_after.shape, target_shape_video)
    assert_equals(teacher_after.shape, target_shape_teacher)


def test_center_crop_returns_correct_shape():
    video_before = generate_video((8, 64, 64, 3))
    teacher_before = generate_video((8, 64, 64))

    target_size = (32, 32)

    target_shape_video = (8, 32, 32, 3)
    target_shape_teacher = (8, 32, 32)

    op = CenterCropOp(target_size)
    video_after, teacher_after = op(video_before, teacher_before)

    assert_equals(video_after.shape, target_shape_video)
    assert_equals(teacher_after.shape, target_shape_teacher)


def test_different_crop_sizes():
    # 1. test when cropping the exact size of the video itself
    video_before = generate_video((8, 32, 32, 3))
    teacher_before = generate_video((8, 32, 32))

    target_size = (32, 32)
    target_shape_video = (8, 32, 32, 3)
    target_shape_teacher = (8, 32, 32)

    # 1a) center crop
    op = CenterCropOp(target_size)
    video_after_center_crop, teacher_after_center_crop = op(video_before, teacher_before)
    assert_equals(video_after_center_crop.shape, target_shape_video)
    assert_equals(teacher_after_center_crop.shape, target_shape_teacher)

    # 1b) random crop
    op = RandomCropOp(target_size)
    video_after_random_crop, teacher_after_random_crop = op(video_before, teacher_before)
    assert_equals(video_after_random_crop.shape, target_shape_video)
    assert_equals(teacher_after_random_crop.shape, target_shape_teacher)


def test_scale_works_on_video():
    num_frames = 15
    video_before = generate_video((8, 64, 64, num_frames, 1))
    teacher_before = generate_video((8, 64, 64))

    target_size = (16, 32)

    target_shape_video = (8, 16, 32, num_frames, 1)
    target_shape_teacher = (8, 16, 32)

    op = ScaleOp(target_size)
    video_after, teacher_after = op(video_before, teacher_before)

    assert_equals(video_after.shape, target_shape_video)
    assert_equals(teacher_after.shape, target_shape_teacher)


def test_crop_works_on_video():
    num_frames = 15
    video_before = generate_video((8, 64, 64, num_frames, 1))
    teacher_before = generate_video((8, 64, 64))

    target_size = (32, 32)

    target_shape_video = (8, 32, 32, num_frames, 1)
    target_shape_teacher = (8, 32, 32)

    op = CenterCropOp(target_size)
    video_after, teacher_after = op(video_before, teacher_before)

    assert_equals(video_after.shape, target_shape_video)
    assert_equals(teacher_after.shape, target_shape_teacher)

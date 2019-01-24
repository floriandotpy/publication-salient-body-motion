import logging

# disable excessive tf log messages
logging.getLogger('tensorflow').disabled = True

from kinectgestures.metrics import motion_metric, motion_metric_single_element, contains_motion
import keras.backend as K
from nose.tools import assert_equals, assert_true, assert_false

DEFAULT_THRESHOLD = -0.3


def test_motion_metric_batch_1():
    # CASE 1: 1 sample correct, 1 sample wrong: metric should produce 0.5
    y_true = [
        [[-1, -1], [0, 1]],
        [[0, 1], [0, -1]]
    ]
    y_pred = [
        [[0, 0], [0, 1]],  # <- 1, because max value at same location, and above threshold
        [[-1, -1], [-1, -1]]  # <- 0, because max value from above does not pass threshold here
    ]
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    actual_result = K.eval(motion_metric(y_true, y_pred, batch_size=2))

    assert_equals(actual_result, 0.5)


def test_motion_metric_batch_all_correct():
    # CASE 2: all "passive" pixels, metric should not be bothered and produce 1
    y_true = [
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]]
    ]
    y_pred = y_true
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    actual_result = K.eval(motion_metric(y_true, y_pred, batch_size=4))
    assert_equals(actual_result, 1)


def test_motion_metric_batch_one_of_four_is_incorrect():
    # CASE 2: all "passive" pixels, metric should not be bothered and produce 1
    y_true = [
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]]
    ]
    y_pred = [
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, -1, -1]],
        [[1, 1, 1], [1, 1, 1]]
    ]
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    actual_result = K.eval(motion_metric(y_true, y_pred, batch_size=4))
    assert_equals(actual_result, 0.75)


def test_motion_metric_single_when_pred_is_correct():
    y_true = [[-1, -1, 1], [-1, -1, -1]]
    y_pred = [[-1, -1, 0], [-1, -1, -1]]  # "0" is not the same value, but still above threshold
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    assert_equals(K.eval(motion_metric_single_element(y_true, y_pred, DEFAULT_THRESHOLD)), 1.0)


def test_motion_metric_single_when_pred_is_incorrect():
    y_true = [[-1, -1, 1], [-1, -1, -1]]
    y_pred = [[-1, -1, -1], [1, -1, -1]]
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    assert_equals(K.eval(motion_metric_single_element(y_true, y_pred, DEFAULT_THRESHOLD)), 0.0)


def test_motion_metric_single_close_to_threshold_pred_is_incorrect():
    y_true = [[0, 1], [0, -1]]
    y_pred = [[-1, -1], [-1, -1]]
    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)
    assert_equals(K.eval(motion_metric_single_element(y_true, y_pred, DEFAULT_THRESHOLD)), 0.0)


def test_contains_motion_positive():
    frame = K.variable([[-1, -1, -1], [0, 0, 0]])
    assert_true(K.eval(contains_motion(frame, DEFAULT_THRESHOLD)))

    frame = K.variable([[-1, 1, 1], [0, 0, 0]])
    assert_true(K.eval(contains_motion(frame, DEFAULT_THRESHOLD)))

    frame = K.variable([[0, 0, 0], [0, 0, 0]])
    assert_true(K.eval(contains_motion(frame, DEFAULT_THRESHOLD)))

    frame = K.variable([[DEFAULT_THRESHOLD + 0.01, -1, -1], [-1, -1, -1]])
    assert_true(K.eval(contains_motion(frame, DEFAULT_THRESHOLD)))


def test_contains_motion_negative():
    frame = K.variable([[-1, -1, -1], [-1, -1, -1]])
    assert_false(K.eval(contains_motion(frame, DEFAULT_THRESHOLD)))

    frame = K.variable([[-0.4, -0.4, -0.4], [-1, -1, -1]])
    assert_false(K.eval(contains_motion(frame, DEFAULT_THRESHOLD)))

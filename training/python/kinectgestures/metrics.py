import tensorflow as tf
from keras import backend as K

# set this from outside before evaluating the metric, to allow for other batch sizes. it's a bit hacky, I know
BATCH_SIZE = 16


def is_motion_at_right_location(y_true, y_pred, threshold):
    # flatten frame representation for easier indexing: (height, width) -> (height*width)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # where is the "active" actor located? in the network output?
    active_locations = tf.argmax(y_pred_flat, axis=0)

    # is this also "active" in the teacher?
    actual_values = tf.gather(y_true_flat, active_locations)

    return tf.greater(actual_values, tf.constant(threshold))


def contains_motion(frame, threshold):
    return tf.greater(tf.count_nonzero(frame > threshold), 0)


def motion_metric_single_element_motion_in_teacher(y_true, y_pred, threshold):
    """
    Apply the motion metric to a single element, NOT a full batch.
    This method assumes that the teacher frame HAS motion in it.
    """

    cond_active_at_right_location = tf.cond(is_motion_at_right_location(y_true, y_pred, threshold),
                                            lambda: 1.0,
                                            lambda: 0.0)

    return tf.cond(contains_motion(y_pred, threshold),  # if motion in prediction...
                   lambda: cond_active_at_right_location,  # then: need to check if it's at the right location
                   lambda: 0.0)  # else: motion in teacher, but not in prediction? wrong


def motion_metric_single_element_no_motion_in_teacher(y_true, y_pred, threshold):
    """
    Apply the motion metric to a single element, NOT a full batch.
    This method assumes that the teacher frame HAS NO motion in it.
    """

    # no motion in the teacher: make sure result also contains none
    cond_no_motion_in_teacher = tf.equal(tf.count_nonzero(y_true > threshold), 0)
    cond_no_motion_in_prediction = tf.equal(tf.count_nonzero(y_pred > threshold), 0)

    return tf.cond(tf.logical_and(cond_no_motion_in_teacher, cond_no_motion_in_prediction),
                   lambda: 1.0,
                   lambda: 0.0)


def motion_metric_single_element(y_true, y_pred, threshold):
    motion_in_teacher = contains_motion(y_true, threshold)

    return tf.cond(motion_in_teacher,
                   lambda: motion_metric_single_element_motion_in_teacher(y_true, y_pred, threshold),
                   lambda: motion_metric_single_element_no_motion_in_teacher(y_true, y_pred, threshold))


def motion_metric(y_true, y_pred, threshold=-0.3, batch_size=None):
    """
    Custom metric to evaluate if the network has corrected motion in the correct location of the frame
    :param y_true: The teacher frame / ground truth
    :param y_pred: The actual generated output by the network
    :param threshold: Threshold value. Above this, a value is considered "active", otherwise "passive" or background
    :return: 1.0 if motion correctly localized, 0.0 elsewise
    """

    # not set as default parameter value, because function head is evaluated too early
    if batch_size is None:
        batch_size = BATCH_SIZE

    results = []
    for i in range(batch_size):
        r = motion_metric_single_element(y_true[i], y_pred[i], threshold)
        results.append(r)

    results = K.stack(results)
    return K.mean(results)

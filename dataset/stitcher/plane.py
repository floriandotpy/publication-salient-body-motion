"""
Detect and remove the ground plane from a Kinect depth frame using PCA.

Based on: http://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
"""

import numpy as np
import cv2

BOTTOM_ROWS = 60


def read_png_single(filename):
    # be careful to properly read in 16bit image without messing up the values. misc.imread for example doesnt work
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:  # libpng might fail, but doesnt throw an exception.
        raise IOError("Reading %s failed. Invalid file?" % filename)
    return img


def transform_16to8(arr):
    """
    Takes a 16 bit numpy array and returns it as an 8 bit array.
    Values are remapped so that the visible range in Kinect frames
    are preserved maximally.

    :param arr: 16 bit numpy array, dtype=np.uint16
    :return: 8 bit numpy array, dtype=np.uint8
    """
    expected_min = 500.0  # TODO: get actual minimum (except outlier, except 0) from data
    expected_max = 12000.0  # TODO: get actual maximum (except outlier) from data
    result = cv2.convertScaleAbs(arr, alpha=(expected_min / expected_max))
    assert result.dtype == np.uint8  # just to be sure...
    return result


def find_good_value(frame, row):
    values = frame[row]
    return np.median(values)


class SimplePlane(object):
    def __init__(self, row, val, gradient):
        self.row = row
        self.row_val = val
        self.gradient = gradient

    def belongs_to_plane(self, row, val, threshold):
        distance = self.row - row
        expected = self.row_val + distance * self.gradient

        return abs(expected - val) < threshold

    def get_mask(self, frame, threshold):
        mask = np.zeros_like(frame, dtype=np.uint16)
        for row_i in range(frame.shape[0]):
            for col_i in range(frame.shape[1]):
                mask[row_i, col_i] = self.belongs_to_plane(row_i, frame[row_i, col_i], threshold)

        return mask

    def get_mask_inside_box(self, frame, box, threshold, skip=0.7):
        col_from, row_from, col_to, row_to = box
        col_from += int((col_to - col_from) * skip)  # the floor is most likely only in lowest part. skip the top
        mask = np.zeros_like(frame, dtype=np.uint16)
        for row_i in range(row_from, row_to):
            for col_i in range(col_from, col_to):
                mask[row_i, col_i] = self.belongs_to_plane(row_i, frame[row_i, col_i], threshold)

        return mask


def plane_from_frame(frame):
    row = 450
    distance = 20

    val_lower = find_good_value(frame, row)
    val_upper = find_good_value(frame, row - distance)

    gradient = float(val_upper - val_lower) / float(distance)
    gradient *= 1.7  # FIXME: somehow, the gradient is mysteriously too low. this adjusts it in the right direction

    plane = SimplePlane(row, val_lower, gradient)

    return plane


def remove_ground(frame, threshold=400):
    """
    Remove the ground plane from a Kinect frame

    :param frame: Kinect frame, numpy array
    :param threshold: Tolerance threshold of what to still define as "close". Value e.g. 400 (raw frames) or 20 (8bit frames)
    :return: The frame where the floor pixels have been changed to black (=0)
    """

    plane = plane_from_frame(frame)
    mask = plane.get_mask(frame, threshold)
    frame[mask == True] = 0
    return frame


def remove_ground_inside_box(frame, box, threshold=400):
    """
    Removes the ground plane in a given bounding box. More efficient because it has to go through less pixels
    """

    plane = plane_from_frame(frame)
    mask = plane.get_mask_inside_box(frame, box, threshold)
    frame[mask == True] = 0
    return frame


def remove_background(frame):
    frame[frame > 220] = 0
    return frame


if __name__ == '__main__':

    import glob
    import random
    import matplotlib.pyplot as plt

    frames = glob.glob("/mnt/igloo/videos/s*/*/depth-100.png")

    random.shuffle(frames)

    fig = plt.figure()

    count = 5

    for i, fname in enumerate(frames[:count]):

        for threshold in (20,):
            frame = read_png_single(fname)
            frame = transform_16to8(frame)

            # frame = remove_background(frame)
            with_ground = np.copy(frame)

            frame = remove_ground(frame, threshold)

            a = fig.add_subplot(count, 2, 2 * i + 1)
            a.set_title("With ground plane")
            plt.imshow(with_ground, cmap="Greys_r")

            b = fig.add_subplot(count, 2, 2 * i + 2)
            b.set_title("Removed (threshold %d)" % threshold)
            plt.imshow(frame, cmap="Greys_r")

    plt.show()

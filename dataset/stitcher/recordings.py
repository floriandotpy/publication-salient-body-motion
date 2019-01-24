import random
import sys
import stitcher
import stitcher.stitching
# from stitcher.stitching import generate_time_windows

import os
import json
import cv2
import glob
import numpy as np

# OpenCV 3.X
CONST_PNG = cv2.IMWRITE_PNG_COMPRESSION
CONST_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT


def generate_png_filelist(dir, limit=None):
    filelist = []

    files = glob.glob(os.path.join(dir, "depth-*.png"))  # not sorted correctly, thats why we use range() instead

    if limit is None:
        limit = len(files)

    for i in range(limit):

        fn = os.path.join(dir, "depth-%d.png" % i)
        if not os.path.exists(fn):
            print("File not found, skipping: %s" % fn)
            continue

        filelist.append(fn)

    return filelist


def read_png_single(filename):
    # be careful to properly read in 16bit image without messing up the values. misc.imread for example doesnt work
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:  # libpng might fail, but doesnt throw an exception.
        raise IOError("Reading %s failed. Invalid file?" % filename)
    return img


def read_png_sequence(dir, limit=None):
    filelist = generate_png_filelist(dir, limit=limit)
    return [read_png_single(fn) for fn in filelist]


def read_video_sequence(filename):
    vc = cv2.VideoCapture(filename)

    framecount = int(vc.get(CONST_FRAME_COUNT))

    frames = []
    for i in range(framecount):
        _, frame = vc.read(framecount)
        frames.append(frame)

    # cleanup
    vc.release()

    return frames


def read_static_image(path, sequence_length=1):
    """
    Read in a static image an return it as a (non-changing) sequence multiple times.
    :param path: Path to image file
    :param sequence_length: Length of sequence to be returned (Default: 1)
    :return:
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return [img for _ in range(sequence_length)]


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


class Recording(object):
    def __init__(self, path_video):

        self.basename = os.path.splitext(os.path.basename(path_video))[0]
        self.dirname = os.path.dirname(path_video)
        self.filebase = os.path.join(self.dirname, self.basename.replace('-img', '-depth'))

        self.path_video = path_video
        self.path_rgb = path_video.replace('-depth', '-img')
        self.path_depth = "%s" % os.path.join(self.dirname, self.basename.replace('-depth', '-img-depth'))
        self.path_times = "%s.json" % self.filebase
        self.path_boxes = "%s.txt" % self.filebase

    def has_video(self):
        return os.path.exists(self.path_video)

    def has_times(self):
        return os.path.exists(self.path_times)

    def has_boxes(self):
        return os.path.exists(self.path_boxes)

    def has_depth(self):
        return os.path.exists(self.path_depth) and os.path.isdir(self.path_depth)

    def get_slug(self):
        return self.basename

    def is_labeled(self):
        return self.has_times() and self.has_boxes() and self.has_depth()

    def get_depth_filenames(self):
        return generate_png_filelist(self.path_depth)

    def get_gesture(self):
        available = ('passive', 'abort', 'check', 'chi', 'circle', 'point', 'stop', 'wave', 'x')
        for gesture in available:
            if gesture in self.basename:
                return gesture
            if 'still' in self.basename or 'standing' in self.basename:  # map all of these to passive
                return 'passive'
        return None

    def get_times(self):
        """
        Read and return time labels for this recording.
        :return: List with the time labels
        """

        with open(self.path_times, 'r') as fp:
            return json.load(fp)

    def get_annotations(self):
        """
        Read and return annotations (bounding boxes) for this recording.
        :return: Dictionary with the loaded annotations
        """

        annotations = {}  # dict: frame_number -> annotation_dict

        with open(self.path_boxes, 'r') as fp:

            for line in fp:

                parts = line.rstrip('\n').split(' ')

                identifier = parts[0]
                xtl, ytl, xbr, ybr = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                frame = int(parts[5])
                lost, occluded, generated = parts[6], parts[7], parts[8]
                label = parts[9].replace('"', '')

                # CASE A: Single actor in this video
                if label == "head" or label == "body":

                    if frame not in annotations:
                        annotations[frame] = {'body': None, 'head': None}

                    annotations[frame][label] = (xtl, ytl, xbr, ybr)

                # CASE B: Multiple actors in this video:
                if label in ("body_active", "body_passive", "head_active", "head_passive"):
                    if frame not in annotations:
                        annotations[frame] = {'body_active': [], 'body_passive': [], 'head_active': [],
                                              'head_passive': []}

                    coords = (xtl, ytl, xbr, ybr)
                    annotations[frame][label].append(coords)

        return annotations


class Recordings(object):
    def __init__(self, path):
        self.directory = path

        self.active = [self.full_path(path) for path in ["s1/s1-abort-depth.avi",
                                                         "s1/s1-check-depth.avi",
                                                         "s1/s1-chi-depth.avi",
                                                         "s1/s1-circle-depth.avi",
                                                         "s1/s1-point-left-depth.avi",
                                                         "s1/s1-point-right-depth.avi",
                                                         "s1/s1-stop-depth.avi",
                                                         # "s1/s1-turn-depth.avi",
                                                         "s1/s1-wave-depth.avi",
                                                         "s1/s1-x-depth.avi",
                                                         "s2/s2-abort-depth.avi",
                                                         "s2/s2-check-depth.avi",
                                                         "s2/s2-chi-depth.avi",
                                                         "s2/s2-circle-depth.avi",
                                                         "s2/s2-circle-outwards-depth.avi",
                                                         "s2/s2-point-left-depth.avi",
                                                         "s2/s2-point-right-depth.avi",
                                                         "s2/s2-stop-depth.avi",
                                                         "s2/s2-wave-depth.avi",
                                                         "s2/s2-x-depth.avi",
                                                         "s3/s3-abort-depth.avi",
                                                         "s3/s3-check-depth.avi",
                                                         "s3/s3-chi-depth.avi",
                                                         "s3/s3-circle-depth.avi",
                                                         "s3/s3-point-left-depth.avi",
                                                         "s3/s3-point-right-depth.avi",
                                                         "s3/s3-stop-depth.avi",
                                                         "s3/s3-wave-depth.avi",
                                                         "s3/s3-x-depth.avi",
                                                         "s4/s4-abort-depth.avi",
                                                         "s4/s4-check-depth.avi",
                                                         "s4/s4-chi-depth.avi",
                                                         "s4/s4-circle-depth.avi",
                                                         "s4/s4-point-left-depth.avi",
                                                         "s4/s4-point-right-depth.avi",
                                                         "s4/s4-stop-depth.avi",
                                                         "s4/s4-wave-depth.avi",
                                                         "s4/s4-x-depth.avi",
                                                         "s5/s5-abort-depth.avi",
                                                         "s5/s5-check-depth.avi",
                                                         "s5/s5-chi-depth.avi",
                                                         "s5/s5-circle-depth.avi",
                                                         "s5/s5-point-left-depth.avi",
                                                         "s5/s5-point-right-depth.avi",
                                                         "s5/s5-stop-depth.avi",
                                                         "s5/s5-wave-depth.avi",
                                                         "s5/s5-x-depth.avi",
                                                         "s6/s6-abort-depth.avi",
                                                         "s6/s6-check-depth.avi",
                                                         "s6/s6-chi-depth.avi",
                                                         "s6/s6-circle-depth.avi",
                                                         "s6/s6-point-left-depth.avi",
                                                         "s6/s6-point-right-depth.avi",
                                                         "s6/s6-stop-depth.avi",
                                                         "s6/s6-wave-depth.avi",
                                                         "s6/s6-x-depth.avi"]]

        self.passive = [self.full_path(path) for path in ["s1/s1-background-walking-depth.avi",
                                                          "s1/s1-standing-depth.avi",
                                                          "s1/s1-still-depth.avi",
                                                          "s1/s1-walking-left-depth.avi",
                                                          "s1/s1-walking-right-depth.avi",
                                                          "s2/s2-standing-depth.avi",
                                                          "s2/s2-still-depth.avi",
                                                          "s3/s3-standing-depth.avi",
                                                          "s3/s3-still-depth.avi",
                                                          "s4/s4-standing-depth.avi",
                                                          "s4/s4-standing-sideways-depth.avi",
                                                          "s5/s5-standing-arms-crossed-depth.avi",
                                                          "s5/s5-standing-depth.avi",
                                                          "s5/s5-standing-sideways-depth.avi",
                                                          "s6/s6-standing-arms-crossed-depth.avi",
                                                          "s6/s6-standing-depth.avi",
                                                          "s6/s6-standing-sideways-depth.avi"]]

    def full_path(self, short_path):
        return os.path.join(self.directory, short_path)

    def check(self):
        """
        Checks the availability of labels and time annotations for the registered videos.
        Prints a table.
        :return:
        """

        for path in self.active + self.passive:
            recording = Recording(path)

            # terminal colours
            color_green = '\033[92m'
            color_fail = '\033[91m'
            color_end = '\033[0m'

            # pretty message for boolean value
            m = lambda v: {True: color_green + 'YES' + color_end, False: color_fail + 'NO ' + color_end}[v]

            print("[check] Boxes: %s  Times: %s  Depth: %s  |  %s" % (
                m(recording.has_boxes()), m(recording.has_times()), m(recording.has_depth()), recording.get_slug()))

    def get_labeled(self, include_active=True, include_passive=True):

        clips = []

        if include_active:
            clips += self.active
        if include_passive:
            clips += self.passive

        clips = [Recording(path) for path in clips]
        clips = filter(lambda r: r.is_labeled(), clips)

        return map(lambda r: r.path_video, clips)


class BackgroundHelper(object):
    def __init__(self, window_size, video_root):

        self.background_videos = ['%s/%s' % (video_root, v) for v in
                                  [
                                      # 's1/s1-background-walking-2-img-depth',
                                      # 's1/s1-background-walking-img-depth',
                                      'rooms/room-1-1-img-depth', 'rooms/room-1-2-img-depth',
                                      'rooms/room-1-3-img-depth',
                                      'rooms/room-2-1-img-depth', 'rooms/room-2-2-img-depth',
                                      'rooms/room-3-1-img-depth', 'rooms/room-3-2-img-depth',
                                      'rooms/room-3-3-img-depth']]

        self.window_size = window_size

        # generate a list of lists, with available frames in each sublist
        self.video_frames = [generate_png_filelist(video_path) for video_path in self.background_videos]

        # generate a list of lists, with time windows in each sublist
        self.times = [stitcher.stitching.generate_time_windows([(0, len(paths) - 1)], window_size=window_size) for paths in
                      self.video_frames]

        # initial values
        self.index_video = 0
        self.index_times = 0

    def get_next_sequence(self):

        # get current time window boundaries
        start, end = self.times[self.index_video][self.index_times]

        # print("[info] Using background sequence start / end: %d / %d" % (start, end))

        # fetch filenames of corresponding frames
        frame_paths = self.video_frames[self.index_video][start:end]

        # move to next time window, and potentially next video. repeat when end is reached
        self.index_times = (self.index_times + 1) % len(self.times[self.index_video])
        if self.index_times is 0:  # skip to next video
            self.index_video = (self.index_video + 1) % len(self.video_frames)

        return [read_png_single(path) for path in frame_paths]

    def get_random_sequence(self):

        self.index_video = random.randrange(len(self.video_frames))

        start, end = self._sample_random_window()
        # print("[info] Using background sequence start / end: %d / %d" % (start, end))

        frame_paths = self.video_frames[self.index_video][start:end]

        try:
            return [read_png_single(path) for path in frame_paths]
        except IOError as e:
            return self.get_random_sequence()  # TODO: is recursive call a good idea?

    def _sample_random_window(self):
        start = random.randrange(len(self.video_frames[self.index_video]) - self.window_size)
        end = start + self.window_size
        return start, end

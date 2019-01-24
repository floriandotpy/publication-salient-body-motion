import itertools
import json
import os
import random
import time

import cv2
import numpy as np
import scipy.misc

from stitcher.utils import eprint
from stitcher.plane import remove_ground_inside_box
from stitcher.recordings import BackgroundHelper, read_static_image, Recording, read_png_single, transform_16to8
from stitcher.teacher import Gauss2DTeacher, MaskTeacher

# OpenCV 3.X
CONST_PNG = cv2.IMWRITE_PNG_COMPRESSION
CONST_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT


# Some global stuff to verify that there is no overlap across folds

verify_folds = None


def init_verify_folds_if_needed(fold_count):
    global verify_folds
    if verify_folds is None:
        verify_folds = {k: [] for k in range(fold_count)}


def assert_verify_folds():
    # checks that there are no sequences shared across folds.
    print("checking clean fold separation...")

    for l1, l2 in itertools.product(verify_folds.values(), repeat=2):
        if l1 == l2:
            continue
        shared = set(l1) & set(l2)
        if len(shared) > 1:
            print("detected overlap of sequences across folds. exiting")
            exit()


class Stitch(object):
    MODE_GAUSS = 1
    MODE_MASK = 2

    def __init__(self, actors, args, video_path, fold_count):
        """
        Create the Stitcher tool and load the specified files.

        :param actors: List of actors dicts with video, position and definition if the actor is considered active
            Example: {'path': 'video.avi', 'active': True, 'position': (100, 300)}
            Position is a tuple of the form (row_index, col_index)
            Note about path: Link to file used to extract an actor (.avi file or folder with png files),
            Expects annotation files of the name: video.avi -> video.txt (annotations)
            and video.json (start/end delimiters of gestures)
        :param args: Configuration parameters
        """

        self.actors = [Actor(arguments) for arguments in actors]
        self.mode = args['mode']
        self.destination = args['destination']
        self.video_path = video_path
        self.fold_count = fold_count
        init_verify_folds_if_needed(fold_count)

        defaults = {
            'scale_frames': 1.0,
            'scale_teacher': 1.0
        }
        self.args = dict(defaults, **args)  # merge dictionaries

        # how long the generated sequences should be
        self.window = 15

        # defaults
        self.frames_background = None
        self.depth_frames_background = None

        # check some implementations before execution
        self.test_copy_if_greater()

        self.gauss2d = Gauss2DTeacher()
        self.teacher = MaskTeacher()
        self.bg_helper = BackgroundHelper(self.window, self.video_path)

    def sample_background_sequence(self, strategy, length):
        """
        Loads a new an random background sequence.
        :param strategy: Loading strategy. Possible values:
            'random': will load a new random moving sequence (this is the best strategy!)
            'moving': uses a moving background (continuously selected sequences)
            'static': uses a static background image for every frame (only use if you know what you do)
            'black': uses a black background image for every frame (only use if you know what you do)
        :param length: Length of sequence to be loaded
        :return:
        """

        if strategy == 'random':
            return self.bg_helper.get_random_sequence()
        elif strategy == 'moving':
            return self.bg_helper.get_next_sequence()
        elif strategy == 'static':  # should only be used for testing
            self.background_file = '%s/empty-room.png' % self.video_path
            return read_static_image(self.background_file, length)
        elif strategy == 'black':  # should only be used for testing
            self.background_file = '%s/black.png' % self.video_path
            return read_static_image(self.background_file, length)
        else:
            raise ValueError("Unknown background generation strategy: %s" % strategy)

    def generate_jobs(self, fold_id, count):

        # load all files and annotations
        for a in self.actors:
            a.load()

        time_windows_per_actor = [actor.get_windows(self.window, fold_id, fold_count=self.fold_count) for actor in
                                  self.actors]
        all_window_combinations = list(itertools.product(*time_windows_per_actor))
        random.shuffle(all_window_combinations)
        all_window_combinations = all_window_combinations[:count]
        job_count = 0

        for combination in all_window_combinations:

            bg_sequence = self.sample_background_sequence(strategy=self.args['bg_strategy'], length=self.window)
            job = SequenceJob(bg_sequence)

            for times, actor in zip(combination, self.actors):

                start, end = times

                # body coordinates to collect "max box"
                b_xtl, b_ytl, b_xbr, b_ybr = (
                    640, 480, 0, 0)  # invalid on purpose, so that max() and min() work in loop

                # head position in last frame
                h_xtl, h_ytl, h_xbr, h_ybr = (640, 480, 0, 0)  # invalid on purpose, updated anyway
                try:
                    for frame_i in range(start, end):
                        # always enlarge body bounding box, so that whole gesture execution fits in the box
                        xtl, ytl, xbr, ybr = actor.args['annotations'][frame_i]['body']
                        b_xtl, b_ytl = min(xtl, b_xtl), min(b_ytl, ytl)
                        b_xbr, b_ybr = max(b_xbr, xbr), max(b_ybr, ybr)

                        # only interested in final head position anyway (network target signal centered here)
                        h_xtl, h_ytl, h_xbr, h_ybr = actor.args['annotations'][frame_i]['head']
                except KeyError as e:
                    eprint("Error, no annotation for body or head box")
                    eprint(actor.recording.path_boxes)
                    eprint("Start %d, end %d" % (start, end))
                    continue
                except TypeError as e:
                    eprint("Error, 'None' annotation for box")
                    eprint(actor.recording.path_boxes)
                    eprint("Start %d, end %d" % (start, end))
                    continue

                # sample position
                player_width = xbr - xtl
                player_height = ybr - ytl
                player_col = random.randrange(0, 640 - player_width)
                player_row = random.randrange(0, 480 - player_height)

                item = SequenceItem(actor)
                item.is_active = actor.is_active
                item.position = player_row, player_col
                item.start = start
                item.end = end
                item.player_box = (b_xtl, b_ytl, b_xbr, b_ybr)
                item.head_box = (h_xtl, h_ytl, h_xbr, h_ybr)
                item.head_relative = (h_xtl - b_xtl, h_ytl - b_ytl, h_xbr - b_xtl, h_ybr - b_ytl)
                job.add_item(item)
            job_count += 1

            self.run_job(job, fold_id)

        return job_count

    def run_job(self, sequence_job, fold_id):

        frames = []

        for item in sequence_job.items:
            item.init_shift()

        for i in range(sequence_job.length):

            canvas = np.copy(sequence_job.get_background_frame(i))
            frames.append(canvas)

            for item in sequence_job.items:
                hash = sequence_hash(item.actor.recording.filebase, item.start, item.end)
                verify_folds[fold_id].append(hash)

                item_depth = np.copy(item.get_frame(i))
                item_depth = remove_ground_inside_box(item_depth, item.player_box)
                item_depth = item.shift(item_depth)
                col_from, row_from, col_to, row_to = item.player_box
                row_offset, col_offset = item.position

                copy_if_greater(item_depth, canvas, row_from, row_to, col_from, col_to,
                                row_offset,
                                col_offset)

        # save sequence
        fold_dir = "fold_%d" % fold_id
        target_dir = sequence_job.get_destination_name()
        target_dir = os.path.join(self.destination, fold_dir, target_dir)
        self.save_sequence(frames, target_dir)

        # create teacher signal
        teacher = None
        if self.mode == Stitch.MODE_GAUSS:
            teacher = self.gauss2d.teachersignal(sequence_job.items)
        elif self.mode == Stitch.MODE_MASK:
            teacher = self.teacher.mask_from_sequence_items(sequence_job.items)

        teacher_fn = os.path.join(target_dir, '_teacher.png')

        # downscale if specified
        teacher = scipy.misc.imresize(teacher, self.args['scale_teacher'])

        cv2.imwrite(teacher_fn, teacher, (CONST_PNG, 0))

    def save_sequence(self, frames, directory):

        out_depth = os.path.join(directory, 'depth-%d.png')
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(len(frames)):
            visible = transform(frames[i])

            # downscale if specified
            visible = scipy.misc.imresize(visible, self.args['scale_frames'])

            # be careful to properly write 16bit png. misc.imsave doesnt seem to work.
            cv2.imwrite(out_depth % i, visible,
                        (CONST_PNG, 0))


    def test_copy_if_greater(self):

        array_from = np.array([
            [2, 3, 0],
            [3, 2, 3],
            [3, 3, 2]])
        array_to = np.array([
            [3, 2, 3],
            [2, 3, 2],
            [3, 3, 3]])
        array_expected = np.array([
            [3, 2, 3],
            [2, 2, 2],
            [3, 3, 2]])
        row_from = 1
        row_to = 3
        col_from = 0
        col_to = 3

        copy_if_greater(array_from, array_to, row_from, row_to, col_from, col_to)

        np.testing.assert_array_equal(array_expected, array_to)


class SequenceGenerator(object):
    def __init__(self, from_recording, scale_frames=1.0, scale_teacher=1.0):
        self.recording = Recording(from_recording)
        self.scale_frames = scale_frames
        self.scale_teacher = scale_teacher
        self.teachergenerator = MaskTeacher()

    def run(self, destination):

        windows = generate_time_windows(self.recording.get_times(), 15)
        frames = self.recording.get_depth_filenames()
        annotations = self.recording.get_annotations()

        for (start, end) in windows:

            print("Go from start %d to end %d" % (start, end))

            frames = [read_png_single(frames[i]) for i in range(start, end)]

            # Destination folder handling
            folder = '%s_start%d-stop%d' % (self.recording.basename, start, end)
            fullpath = os.path.join(destination, folder)
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)

            # Generate and save teacher
            lastframe = frames[-1]
            annotation = annotations[end]
            teacher = self.teachergenerator.mask_from_single_frame(lastframe, annotation)
            teacher_fn = os.path.join(fullpath, '_teacher.png')
            teacher = scipy.misc.imresize(teacher, self.scale_teacher)
            cv2.imwrite(teacher_fn, teacher, (CONST_PNG, 0))

            # Transform to 8 bit, downscale, store frames
            frames = [scipy.misc.imresize(transform(f), self.scale_frames) for f in frames]
            for i, frame in enumerate(frames):
                filename = os.path.join(fullpath, 'depth-%d.png' % i)
                cv2.imwrite(filename, frame, (CONST_PNG, 0))


class Scrambler(object):
    def __init__(self, lst):
        self.lst = list(lst)  # clone
        self.i = 0

    def next(self):
        if self.i == 0:
            random.shuffle(self.lst)
        self.i = (self.i + 1) % len(self.lst)
        return self.lst[self.i]

    def current(self):
        return self.lst[self.i]

    def random(self):
        return random.choice(self.lst)


def generate_time_windows(times_in, window_size):
    """
    Given: list of tuples (start, end) and a window_size of n frames.

    Returns: list of tuples (start_i, end_i) so that
    end_i - start_i == window_size
    and every the number (start_i, end_i) are always in between (start, end) delimiters.

    Example 1:
    times_in = [(10, 100)], window_size = 20
    => times_out = [(10, 30), (30, 50), (50, 70), (70, 90)]

    Example 2:
    times_in = [(20, 50), (110, 155)], window_size = 20
    => times_out = [(20, 40), (110, 130), (130, 150)]

    """

    times_out = []
    for (start, end) in times_in:
        for i in range((end - start) // window_size):
            if start + (i + 1) * window_size <= end:
                times_out.append((start + i * window_size, start + (i + 1) * window_size))

    return times_out


class SequenceItem(object):
    """
    A single item that will be rendered in a sequence
    :param object: An actor object that will be rendered according to the other parameters provided
    :return:  None
    """

    def __init__(self, actor):
        self.shiftoffset = None
        self.actor = actor
        self.is_active = False
        self.position = None
        self.start = None  # index of start frame
        self.end = None  # index of last frame
        self.player_box = None  # bounding box of player in end frame, tuple
        self.head_box = None  # bounding box of head in end frame, tuple
        self.head_relative = None  # bounding box of head in end frame, relative from origin of player's bounding box

    def get_frame(self, frame_number):
        return self.actor.get_frame(self.start + frame_number)

    def get_slug(self):
        return self.actor.recording.get_slug()

    def init_shift(self):
        self.shiftoffset = random.randint(-300, 300)  # TODO: negative values cause trouble

    def shift(self, depth):
        """
        Randomly shift an item in its depth coordinate. This corresponds to placing
        an actor closer or farther from the camera.
        """

        if self.shiftoffset < 0:
            depth[depth < abs(self.shiftoffset)] = 0  # prevent shifts to negative

        depth[depth > 0] = depth[depth > 0] + self.shiftoffset

        return depth


class SequenceJob(object):
    def __init__(self, background_sequence):
        self.length = 15  # FIXME do not hardcode
        self.background_sequence = background_sequence  # renderable background object
        self.items = []  # a sequence is defined by a number of items it should render.

    def add_item(self, sequence_item):
        self.items.append(sequence_item)

    def get_background_frame(self, frame_index):
        return self.background_sequence[frame_index]

    def get_destination_name(self):
        millis = int(round(time.time() * 1000))
        slug = "_".join([item.get_slug() for item in self.items])
        return "%d_%s" % (millis, slug)


class Actor(object):
    def __init__(self, args):
        self.args = args
        self.is_active = args['active']
        self.recording = Recording(path_video=args['path'])

    def load(self):
        """
        Loads annotations and depth frames for this actor in memory.
        Afterwards, these will be available as follows:
            self.args['annotations']  : dict of annotations for each frame
            self.args['times']        : list of time labels, each a tuple (start_frame, end_frame)
            self.args['depth_frames'] : list of numpy arrays, each corresponding to a frame

        :return: None
        """
        print("[info] Using %s" % self.recording.path_depth)

        self.args['times'] = self.recording.get_times()
        self.args['annotations'] = self.recording.get_annotations()
        self.args['depth_frames'] = self.recording.get_depth_filenames()

    def get_frame(self, index):
        if index >= len(self.args['depth_frames']):
            eprint("Requesting frame %d that does not exists... return fallback frame instead" % index)
            index = len(self.args['depth_frames']) - 1  # fall back to last frame
        return read_png_single(self.args['depth_frames'][index])

    def get_windows(self, length, fold_filter, fold_count):
        windows = generate_time_windows(self.args['times'], length)
        if fold_filter is None:
            return windows
        else:
            return windows[fold_filter::fold_count]  # every n-th element


def copy_if_greater(array_from, array_to, row_from, row_to, col_from, col_to, row_offset=None,
                    col_offset=None):
    # by default: copy and paste with same origin coordinates
    if row_offset is None or col_offset is None:
        row_offset = row_from
        col_offset = col_from

    # calculate the offset for correct placement
    col_diff = col_from - col_offset
    row_diff = row_from - row_offset

    # assuming 640 x 480 images
    ROW_MIN = 0
    ROW_MAX = 480 - 1
    COL_MIN = 0
    COL_MAX = 640 - 1

    # fix cases where result will be out of bounds:
    if row_from - row_diff < ROW_MIN:  # top would be cropped
        correctby = abs(row_from - row_diff)
        row_from += correctby
    if row_to - row_diff > ROW_MAX:  # bottom would be cropped
        correctby = (row_to - row_diff) - ROW_MAX
        row_to -= correctby
    if col_from - col_diff < COL_MIN:  # left would be cropped
        correctby = abs(col_from - col_diff)
        col_from += correctby
    if col_to - col_diff > COL_MAX:  # right would be cropped
        correctby = (col_to - col_diff) - COL_MAX
        col_to -= correctby

    # grab the two slices
    slice_from = array_from[row_from:row_to, col_from:col_to]
    slice_to = array_to[row_from - row_diff:row_to - row_diff, col_from - col_diff:col_to - col_diff]

    # find the element wise closest pixels in the two slices
    slice_result = np.minimum(slice_from, slice_to)

    # 0 is "less", but we still prefer real pixel value from body parts
    slice_result[slice_result == 0] = slice_from[slice_result == 0]

    # however, things that are still 0, are probably floor. put floor pixels back in
    slice_result[slice_result == 0] = slice_to[slice_result == 0]

    # paste the result, modifies the original (good practice?)
    array_to[row_from - row_diff:row_to - row_diff, col_from - col_diff:col_to - col_diff] = slice_result


def transform(arr):
    return arr  # TODO: find better way to disable temporary
    return transform_16to8(arr)


def sequence_hash(recording, start, end):
    return '%d-%d-%s' % (start, end, recording)

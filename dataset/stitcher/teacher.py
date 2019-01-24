# import pylab
import numpy as np
from stitcher import plane

def eprint(msg):
    color_fail = '\033[91m'
    color_end = '\033[0m'
    print(color_fail + '[error] ' + color_end + msg)


class MaskTeacher(object):
    def __init__(self):
        self.amplitudes = {True: 255, False: 255 * 0.3}

    def mask_from_single_frame(self, frame, desc):

        if not len(desc['head_active']) == len(desc['body_active']) and \
                        len(desc['head_passive']) == len(desc['body_passive']):
            raise ValueError("Missing labels. Needed is the same number of labeled heads and bodies in each frame.")

        # TODO: actually find partners by comparing coordinates, not just by looking for same index
        actives = [{
                       'head': desc['head_active'][i],
                       'body': desc['body_active'][i]
                   } for i in range(len(desc['head_active']))]

        passives = [{
                        'head': desc['head_passive'][i],
                        'body': desc['body_passive'][i]
                    } for i in range(len(desc['head_passive']))]

        masks_active = [self._isolate(frame, label['head'], label['body'], True, label['body'][1], label['body'][0]) for
                        label in actives]
        masks_passive = [self._isolate(frame, label['head'], label['body'], False, label['body'][1], label['body'][0])
                         for label in passives]

        return self._sumMasks(masks_active + masks_passive)

    def mask_from_sequence_item(self, item):
        last_frame = item.actor.get_frame(item.end)
        head = item.head_box
        body = item.player_box
        offset_row = item.position[0]
        offset_col = item.position[1]

        return self._isolate(last_frame, head, body, item.is_active, offset_row, offset_col)

    def mask_from_sequence_items(self, items):
        return self._sumMasks([self.mask_from_sequence_item(item) for item in items])

    def mask_from_scene_description(self, actors):

        def _do(actor):
            frame_no = actor.args['job']['stop']
            frame = actor.get_frame(frame_no)
            labels = actor.args['annotations'][frame_no]
            is_active = actor.args['active']
            offset_row = actor.args['position'][0]
            offset_col = actor.args['position'][1]

            return self._isolate(frame, labels['head'], labels['body'], is_active, offset_row, offset_col)

        masks = [_do(actor) for actor in actors]

        canvas = self._sumMasks(masks)

        return canvas

    def _sumMasks(self, masks):
        canvas = np.zeros(shape=(480, 640), dtype=np.uint8)
        for mask in masks:
            canvas += mask
            canvas[canvas < mask] = self.amplitudes[True]  # make sure all overflows are set to MAX_VALUE
        return canvas

    def _isolate(self, frame, label_head, label_body, is_active, offset_row=0, offset_col=0):

        frame = plane.remove_ground(np.copy(frame))

        head_tlx, head_tly, head_brx, head_bry = label_head
        body_tlx, body_tly, body_brx, body_bry = label_body

        slice_body = frame[body_tly:body_bry, body_tlx:body_brx]

        # make relative to work for coordinates inside slice
        head_tlx -= body_tlx
        head_tly -= body_tly
        head_brx -= body_tlx
        head_bry -= body_tly

        slice_head = slice_body[head_tly:head_bry, head_tlx:head_brx]

        if not slice_head.size:
            eprint("Invalid label for head. Using empty mask instead")
            return np.zeros_like(frame)

        depth_head = np.median(slice_head)

        depth_window = 400  # experimental value

        mask = np.zeros(shape=slice_body.shape, dtype=np.uint8)
        mask[slice_body < (depth_head + depth_window)] = self.amplitudes[is_active]
        mask[slice_body == 0] = 0  # remove some parts again that we wrongly included in the mask

        # get "moved" box of player
        # body_tlx_ = actor.args['position'][1]
        # body_tly_ = actor.args['position'][0]
        body_tlx_ = offset_col
        body_tly_ = offset_row
        body_brx_ = body_tlx_ + (body_brx - body_tlx)  # keep in valid range
        body_bry_ = body_tly_ + (body_bry - body_tly)  # keep in valid range

        # find out if cropping occurs
        crop_x = max(0, body_brx_ - 640)
        crop_y = max(0, body_bry_ - 480)

        # now transform mask back to original reference frame
        frame_mask = np.zeros(shape=frame.shape, dtype=np.uint8)
        frame_mask[body_tly_:body_bry_ - crop_y, body_tlx_:body_brx_ - crop_x] = mask[:mask.shape[0] - crop_y,
                                                                                 :mask.shape[1] - crop_x]

        return frame_mask


class Gauss2DTeacher(object):
    """
    Generate 2D Gauss curves by itself or generate 2D teacher signal which consists of multiple overlapping
    2D Gauss curves with different positions and paximum amplitudes
    """

    def __init__(self, scale_by=1):
        """
        Create a Gauss2D object.
        :param scale_by: Scale down final image by this number. Example: scale_by=4 with a 640x480 signal will
                        scale down to 160x120 output.
        """
        self.scale_by = scale_by  # divide dimensions by this number

    def gauss2d(self, x, y, width=640, height=480, amplitude=255.0, center_x=None, center_y=None):
        """
        Returns the value of a single pixel in a 2D Gauss bell according to the provided parameters.

        :param x: x-coordinate
        :param y: y-coordinate
        :param width: Width of the generated 2d Gauss distribution
        :param height: Height of the generated 2d Gauss distribution
        :param amplitude: Amplitude of the Gauss distribution at its center
        :param center_x: x-coordinate of the distribution center
        :param center_y: y-coordinate of the distribution center
        :return: A numpy array, potentially scaled down, dimensions: (height/self.scale_by, width/self.scale_by)
        """

        # by default: center
        if center_x is None:
            center_x = width / 2.0  # center x
        if center_y is None:
            center_y = height / 2.0  # center y

        # arbitrary value for standard deviation, might be worth experimenting with
        sigma_x = width / 6.0
        sigma_y = height / 6.0

        q_1 = ((x - center_x) ** 2) / (2 * sigma_x ** 2)
        q_2 = ((y - center_y) ** 2) / (2 * sigma_y ** 2)

        return amplitude * np.exp(-(q_1 + q_2))

    def teachersignalSingle(self, bounding_box, amplitude):
        '''
        Create the distribution for a single given bounding box and amplitude.

        :param bounding_box: Tuple of bounding box around actor's head, (col_topleft, row_topleft, col_bottomright, row_bottomright)
        :param amplitude: Amplitude for this actor's Gauss distribution
        :return: Numpy array with the resulting distribution at its specified location
        '''

        # evenly spaced grid of coordinates, TODO: dont re-generate every single time, also use parameters with img dims
        xx, yy = np.meshgrid(
            np.linspace(0, 640, 640 / self.scale_by),
            np.linspace(0, 480, 480 / self.scale_by))

        col_topleft, row_topleft, col_bottomright, row_bottomright = bounding_box

        center_col = (col_topleft + col_bottomright) / 2.0
        center_row = (row_topleft + row_bottomright) / 2.0

        zz = np.zeros(xx.shape, dtype=np.uint8)

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = self.gauss2d(xx[i, j],
                                        yy[i, j],
                                        center_x=center_col,
                                        center_y=center_row,
                                        amplitude=amplitude)

        return zz

    def teachersignal(self, actors):
        """
        Generate the complete teacher signal (one frame) for a given list of head bounding boxes with their amplitudes.

        :param actors: List of dicts, where each is of the form: {'head_box': (x_tl, y_tl, x_br, y_br), 'active': <bool>}
        :return: Numpy array with the generated teacher signal frame
        """
        amplitudes = {True: 255, False: 255 * 0.3}

        canvas = np.zeros(shape=(480 / self.scale_by, 640 / self.scale_by), dtype=np.uint8)

        for a in actors:
            curve = self.teachersignalSingle(a['head_box'], amplitudes[a['active']])
            canvas += curve
            canvas[canvas < curve] = amplitudes[True]  # make sure all overflows are set to MAX_VALUE
            # TODO: normalize in the end? (max value 255?)

        return canvas
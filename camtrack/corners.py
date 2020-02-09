#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
# params for ShiTomasi corner detection
maxCorners = 1500
minDistance = 7
max_diff = 0.2
feature_params = dict(maxCorners=maxCorners,
                      qualityLevel=0.12,
                      minDistance=minDistance,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def mask_current_corners(points_, shape):
    mask = np.ones_like(shape, dtype=np.uint8)
    for x, y in points_:
        cv2.circle(mask, (x, y), minDistance, 0, -1)
    return mask * 255


class CornersParams:
    def __init__(self, ids, points, sizes):
        self.ids_ = ids
        self.points_ = points
        self.sizes_ = sizes

    def extend(self, id, point):
        self.ids_ = np.concatenate([self.ids_, [id]])
        self.points_ = np.concatenate([self.points_, [point]])
        self.sizes_ = np.concatenate([self.sizes_, [10]])

    def get(self):
        return self.ids_, self.points_, self.sizes_

    def get_good_tracks_masked(self, i1, i2, ids_, points_, sizes_):
        forward = cv2.calcOpticalFlowPyrLK(i1, i2, points_, None, **lk_params)[0].squeeze()
        backward = cv2.calcOpticalFlowPyrLK(i2, i1, forward, None, **lk_params)[0].squeeze()
        mask = np.abs(points_ - backward).max(-1) < max_diff
        self.ids_, self.points_, self.sizes_ = ids_[mask], forward[mask], sizes_[mask]
        return self.points_

def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    frame_sequence = list(map(lambda t: (np.array(t) * 255.0).astype(np.uint8), frame_sequence))
    cur_image = frame_sequence[0]
    initial_points = cv2.goodFeaturesToTrack(cur_image, **feature_params).squeeze(axis=1)
    ptr = len(initial_points)
    params = CornersParams(np.arange(ptr), initial_points, np.full(ptr, 10))
    builder.set_corners_at_frame(0, FrameCorners(*params.get()))
    idx = 0

    for next_image in frame_sequence[1:]:
        idx += 1
        points_ = params.get_good_tracks_masked(cur_image, next_image, *params.get())

        if len(points_) < maxCorners:
            next_features = cv2.goodFeaturesToTrack(next_image, mask=mask_current_corners(points_, next_image), **feature_params)
            next_features = next_features.squeeze(axis=1) if next_features is not None else []
            for pnt in next_features[:maxCorners - len(points_)]:
                params.extend(ptr, pnt)
                ptr += 1

        builder.set_corners_at_frame(idx, FrameCorners(*params.get()))
        cur_image = next_image


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

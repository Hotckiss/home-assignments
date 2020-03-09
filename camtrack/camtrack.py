#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
import click
from _camtrack import *

params = TriangulationParameters(
    max_reprojection_error=2.2,
    min_triangulation_angle_deg=1.0,
    min_depth=0.1
)


def init_track_with_known_views(corner_storage: CornerStorage,
                                known_view_1: Optional[Tuple[int, Pose]],
                                known_view_2: Optional[Tuple[int, Pose]]):
    result = [None] * len(corner_storage)
    result[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    result[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    return result


def add_points(storage, track, intrinsic_mat, current, i, j):
    corr = build_correspondences(storage[i], storage[j])
    if corr.points_1.size == 0 or corr.points_2.size == 0:
        return current

    pts, ids, _ = triangulate_correspondences(corr, track[i], track[j], intrinsic_mat, params)

    print(f'new pts count = {len(list(filter(lambda t: current[t] is None, ids)))}')
    return apply(pts, ids, current)


def fill_track(N, track):
    for i in range(N):
        track[i] = track[i] if track[i] is not None else track[i - 1]

    return track


def filter_poses(inliers, current, ids):
    print(f'inliers count = {len(inliers)}')

    for i in ids:
        current[i] = None if i not in inliers else current[i]

    print(f'cloud size = {len(list(filter(lambda t: t is not None, current)))}')
    return current


def apply(pts, ids, current):
    for pt, i in zip(pts, ids):
        if current[i] is None:
            current[i] = pt

    return current


def calculate_matrix(frame, current, intrinsic_mat):
    ids = frame.ids.squeeze()
    mask = np.array([current[ids[i]] is not None for i in range(len(ids))])

    if frame.points[mask].shape[0] < 5:
        return current, None

    rv, rvec, tvec, inliers = cv2.solvePnPRansac(np.array([current[ind] for ind in ids[mask]]),
                                                 frame.points[mask], intrinsic_mat, None)

    if not rv:
        return current, None

    return filter_poses(inliers.squeeze(), current, ids), rodrigues_and_translation_to_view_mat3x4(rvec, tvec)


def try_update(N, track, poses, intrinsic_mat, storage, i):
    print(f'Process {i + 1}/{N}')
    poses, track[i] = calculate_matrix(storage[i], poses, intrinsic_mat)

    if track[i] is not None:
        for j in range(N):
            if i != j and track[j] is not None:
                poses = add_points(storage, track, intrinsic_mat, poses, i, j)
        return True

    return False


def track_(N, track, current, intrinsic_mat, storage):
    updated = True
    while updated:
        updated = False
        with click.progressbar(range(N), label='Process frames', length=N) as progress_bar:
            for i in progress_bar:
                if track[i] is None:
                    updated = updated or try_update(N, track, current, intrinsic_mat, storage, i)

    return fill_track(N, track), current


def validate_mat(corr, mask):
    _, trust = cv2.findHomography(corr.points_1, corr.points_2, method=cv2.RANSAC, confidence=0.999, ransacReprojThreshold=2.2)

    return np.count_nonzero(mask) >= np.count_nonzero(trust)


def initialize(storage, track, intrinsic_mat):
    best_pose, best_size, best_idx = None, -1, 0

    for i in range(1, len(storage)):
        print("process frame " + str(i + 1))
        cur_pose, cur_size = test_frames(intrinsic_mat, storage[0], storage[i])
        print("size " + str(cur_size))
        if best_size < cur_size:
            best_size, best_idx, best_pose = cur_size, i, cur_pose

    track[0], track[best_idx] = eye3x4(), pose_to_view_mat3x4(best_pose)

    return 0, best_idx


def test_frames(intrinsic_mat, fr1, fr2):
    corr = build_correspondences(fr1, fr2)
    if corr.points_1.shape[0] < 5:
        return None, 0

    mat, mask = cv2.findEssentialMat(corr.points_1, corr.points_2, intrinsic_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if mat is None or mat.shape != (3, 3) or not validate_mat(corr, mask):
        return None, 0

    # https://kite.com/python/docs/cv2.decomposeEssentialMat
    R1, R2, t = cv2.decomposeEssentialMat(mat)
    poses = [Pose(R1.T, R1.T @ t), Pose(R2.T, R2.T @ t), Pose(R1.T, R1.T @ (-t)), Pose(R2.T, R2.T @ (-t))]
    best_size, best_idx = -1, 0

    for i, pose in enumerate(poses):
        points, _, _ = triangulate_correspondences(remove_correspondences_with_ids(corr, np.argwhere(mask == 0)), eye3x4(), pose_to_view_mat3x4(pose), intrinsic_mat, params)
        if points.shape[0] > best_size:
            best_size, best_idx = points.shape[0], i

    return poses[best_idx], best_size

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    known_view_1 = None
    known_view_2 = None

    if known_view_1 is None or known_view_2 is None:
        track = [None] * len(corner_storage)
        i, j = initialize(corner_storage, track, intrinsic_mat)
        poses = add_points(corner_storage, track, intrinsic_mat, [None] * (corner_storage.max_corner_id() + 1),
                           i, j)
    else:
        track = init_track_with_known_views(corner_storage, known_view_1, known_view_2)

        poses = add_points(corner_storage, track, intrinsic_mat, [None] * (corner_storage.max_corner_id() + 1),
                           known_view_1[0], known_view_2[0])





    track, poses = track_(len(corner_storage), track, poses, intrinsic_mat, corner_storage)
    point_cloud_builder = PointCloudBuilder(
            ids=np.array([i for i, point in enumerate(poses) if point is not None]),
            points=np.array([point for point in poses if point is not None]))

    view_mats = np.array(track)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

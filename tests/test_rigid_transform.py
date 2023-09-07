"""Test ``pymagsac.findRigidTransformation``."""
import os
import pymagsac
import numpy as np
import pytest

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

CORRESPONDENCES_PATH = os.path.join(
    THIS_PATH,
    "..",
    "graph-cut-ransac/build/data/rigid_pose_example/rigid_pose_example_points.txt",
)
POSE_PATH = os.path.join(
    THIS_PATH,
    "..",
    "graph-cut-ransac/build/data/rigid_pose_example/rigid_pose_example_gt.txt",
)

CORRESPONDENCES = np.loadtxt(CORRESPONDENCES_PATH)
GT_POSE = np.loadtxt(POSE_PATH)
GROUND_TRUTH_T = GT_POSE[:4, :]
THRESHOLD = 0.03

# Translating the points so there are no negative coordinates.
# This is only important if the space partitioning technique is used to
# accelerate the robust estimation, or when the spatial coherence term is >0.
MIN_COORDINATES = np.min(CORRESPONDENCES, axis=0)
T1 = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-MIN_COORDINATES[0], -MIN_COORDINATES[1], -MIN_COORDINATES[2], 1],
    ]
)
T2INV = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [MIN_COORDINATES[3], MIN_COORDINATES[4], MIN_COORDINATES[5], 1],
    ]
)
TRANSFORMED_CORRESPONDENCES = CORRESPONDENCES - MIN_COORDINATES


def verify_magsac(
    corrs,
    threshold,
    sampler_id=0,
    use_magsac_plus_plus=True,
    min_iters=1000,
    max_iters=5000,
):
    n = len(corrs)

    pose, mask = pymagsac.findRigidTransformation(
        np.ascontiguousarray(corrs),
        probabilities=[],
        min_iters=min_iters,
        max_iters=max_iters,
        sampler=sampler_id,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=threshold,
    )

    return pose, mask


def tranform_points(corrs, T):
    n = len(corrs)
    points1 = np.float32([corrs[i][0:3] for i in np.arange(n)]).reshape(-1, 3)
    points2 = np.float32([corrs[i][3:6] for i in np.arange(n)]).reshape(-1, 3)

    transformed_corrs = np.zeros((corrs.shape[0], 6))

    for i in range(n):
        p1 = np.append(correspondences[i][:3], 1)
        p2 = p1.dot(T)
        transformed_corrs[i][:3] = p2[:3]
        transformed_corrs[i][3:] = corrs[i][3:]
    return transformed_corrs


def calculate_error(gt_pose, est_pose):
    R2R1 = np.dot(gt_pose[:3, :3].T, est_pose[:3, :3])
    cos_angle = max(-1.0, min(1.0, 0.5 * (R2R1.trace() - 1.0)))

    err_R = np.arccos(cos_angle) * 180.0 / np.pi
    err_t = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])

    return err_R, err_t


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_magsac(use_magsac_plus_plus):
    gc_t, gc_mask = verify_magsac(
        CORRESPONDENCES,
        THRESHOLD,
        min_iters=5000,
        max_iters=5000,
        use_magsac_plus_plus=use_magsac_plus_plus,
    )
    if gc_t is None:
        gc_t = np.eye(4)
    else:
        gc_t = gc_t.T

    err_r, err_t = calculate_error(GROUND_TRUTH_T, gc_t)
    assert err_r < 4  # rotation less than 4 degrees
    assert err_t < 0.2  # translation less than 0.2 cm

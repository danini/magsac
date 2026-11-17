"""Test the reference threshold of the Python functions (``reference_th``)."""
import os

import numpy as np
import pymagsac
import pytest

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS_PATH, "..", "graph-cut-ransac/build/data/rigid_pose_example")
CORRESPONDENCES = np.loadtxt(os.path.join(DATA, "rigid_pose_example_points.txt"))
GT_POSE = np.loadtxt(os.path.join(DATA, "rigid_pose_example_gt.txt"))[:4, :]
THRESHOLD = 0.03  # metres


def rotation_error(pose):
    """Rotation error of an estimated pose (as returned, transposed), in degrees."""
    R = GT_POSE[:3, :3].T @ pose.T[:3, :3]
    return np.degrees(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)))


def estimate_rigid(scale):
    return pymagsac.findRigidTransformation(
        np.ascontiguousarray(CORRESPONDENCES * scale),
        probabilities=[],
        sampler=0,
        sigma_th=THRESHOLD * scale,
        min_iters=50,
        max_iters=5000,
    )


def test_rigid_transformation_in_millimetres_as_in_metres():
    # The default reference threshold of findRigidTransformation is sigma_th, so the
    # same data give a model in millimetres as in metres (with a fixed 1.0, in
    # millimetres, no model was found).
    pose_m, mask_m = estimate_rigid(1.0)
    pose_mm, mask_mm = estimate_rigid(1000.0)
    assert pose_m is not None and pose_mm is not None
    assert rotation_error(pose_m) < 4  # degrees
    assert rotation_error(pose_mm) < 4
    assert abs(int(mask_mm.sum()) - int(mask_m.sum())) < 0.2 * mask_m.sum()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_invalid_reference_threshold_raises_value_error(value):
    with pytest.raises(ValueError):
        pymagsac.findRigidTransformation(
            np.ascontiguousarray(CORRESPONDENCES), probabilities=[], sampler=0,
            sigma_th=THRESHOLD, reference_th=value)


def test_reference_threshold_of_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        pymagsac.findRigidTransformation(
            np.ascontiguousarray(CORRESPONDENCES), probabilities=[], sampler=0,
            sigma_th=THRESHOLD, reference_th="1")


def has_seed():
    return "seed" in (pymagsac.findHomography.__doc__ or "")


@pytest.mark.skipif(not has_seed(), reason="needs the seed argument to compare two runs")
def test_default_of_pixel_functions_is_min_of_sigma_th_and_1_px():
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 640, (300, 2))
    H = np.array([[1.1, 0.05, 20.0], [-0.03, 0.95, 10.0], [1e-4, 5e-5, 1.0]])
    x2 = np.c_[x1, np.ones(len(x1))] @ H.T
    x2 = x2[:, :2] / x2[:, 2:] + rng.normal(0, 0.5, x1.shape)
    x2[:100] = rng.uniform(0, 640, (100, 2))  # outliers
    c = np.ascontiguousarray(np.c_[x1, x2])
    for sigma_th in [0.5, 3.0]:
        args = (c, 640, 480, 640, 480, np.zeros(0))
        kw = dict(sampler=0, sigma_th=sigma_th, seed=7)
        H0, m0 = pymagsac.findHomography(*args, **kw)
        H1, m1 = pymagsac.findHomography(*args, reference_th=min(sigma_th, 1.0), **kw)
        assert np.array_equal(H0, H1) and np.array_equal(m0, m1)

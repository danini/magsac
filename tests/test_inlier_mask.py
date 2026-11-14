"""The inlier mask returned by findHomography and findLine2D.

A point must be in the mask exactly when its residual for the returned model
is at most sigma_th, as in the other estimators.
"""
import numpy as np
import pymagsac
import pytest

SIGMA_TH = 3.0  # The bug is visible only for sigma_th != 1.


def _expected_mask(residuals):
    return residuals <= SIGMA_TH


def test_homography_mask():
    rng = np.random.default_rng(0)
    H = np.array([[1.1, 0.05, 20.0], [-0.03, 0.95, 10.0], [1e-4, 2e-5, 1.0]])
    src = rng.uniform(0, 640, (400, 2))
    p = np.c_[src, np.ones(len(src))] @ H.T
    dst = p[:, :2] / p[:, 2:]
    # Half of the points are moved by 2 to 8 pixels: above sigma_th, but
    # many below sigma_th**2.
    angle = rng.uniform(0, 2 * np.pi, 200)
    offset = rng.uniform(2.0, 8.0, 200)
    dst[200:] += offset[:, None] * np.c_[np.cos(angle), np.sin(angle)]

    H_est, mask = pymagsac.findHomography(
        np.ascontiguousarray(np.c_[src, dst]), 640, 640, 640, 640,
        np.zeros(0), sampler=0, sigma_th=SIGMA_TH)
    assert H_est is not None

    p = np.c_[src, np.ones(len(src))] @ H_est.T
    residuals = np.linalg.norm(dst - p[:, :2] / p[:, 2:], axis=1)
    np.testing.assert_array_equal(mask, _expected_mask(residuals))
    assert mask.sum() < 300  # Not most of the moved points


# findLine2D crashes on some random samples (danini/graph-cut-ransac#57), which
# kills pytest. Remove the skip once the submodule includes the fix.
@pytest.mark.skip(reason="findLine2D can segfault: needs danini/graph-cut-ransac#52 in the submodule")
def test_line_mask():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 640, 400)
    points = np.c_[t, 0.5 * t + 100.0]
    points[:, 1] += rng.uniform(-8.0, 8.0, 400)

    line, mask = pymagsac.findLine2D(
        np.ascontiguousarray(points), 640, 640, np.zeros(0), sampler=0,
        sigma_th=SIGMA_TH)
    assert line is not None

    residuals = np.abs(points @ line[:2] + line[2])
    np.testing.assert_array_equal(mask, _expected_mask(residuals))

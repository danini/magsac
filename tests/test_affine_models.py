"""Test ``pymagsac.findHomographyAffine`` and ``pymagsac.findFundamentalMatrixAffine``."""
import pymagsac
import numpy as np
import pytest


def generate_affine_homography_data(H_gt, n_inliers=100, n_outliers=30, seed=42):
    """Generate affine correspondences from a known homography.

    Returns (N, 8) array: [x1, y1, x2, y2, a11, a12, a21, a22].
    The affine part is the local Jacobian of the homography at each point.
    """
    rng = np.random.default_rng(seed)
    src = rng.uniform(-1, 1, (n_inliers, 2))

    # Project points
    src_h = np.hstack([src, np.ones((n_inliers, 1))])
    dst_h = (H_gt @ src_h.T).T
    dst = dst_h[:, :2] / dst_h[:, 2:3]

    # Compute local affine (Jacobian of homography at each point)
    data = np.zeros((n_inliers, 8))
    for i in range(n_inliers):
        x, y = src[i]
        denom = H_gt[2, 0] * x + H_gt[2, 1] * y + H_gt[2, 2]
        # Jacobian of the homography projection
        J = np.zeros((2, 2))
        num_x = H_gt[0, 0] * x + H_gt[0, 1] * y + H_gt[0, 2]
        num_y = H_gt[1, 0] * x + H_gt[1, 1] * y + H_gt[1, 2]
        J[0, 0] = (H_gt[0, 0] * denom - num_x * H_gt[2, 0]) / (denom * denom)
        J[0, 1] = (H_gt[0, 1] * denom - num_x * H_gt[2, 1]) / (denom * denom)
        J[1, 0] = (H_gt[1, 0] * denom - num_y * H_gt[2, 0]) / (denom * denom)
        J[1, 1] = (H_gt[1, 1] * denom - num_y * H_gt[2, 1]) / (denom * denom)
        data[i] = [src[i, 0], src[i, 1], dst[i, 0], dst[i, 1],
                   J[0, 0], J[0, 1], J[1, 0], J[1, 1]]

    # Outliers with random affine
    outliers = np.zeros((n_outliers, 8))
    outliers[:, :4] = rng.uniform(-2, 2, (n_outliers, 4))
    outliers[:, 4:] = rng.uniform(-2, 2, (n_outliers, 4))

    return np.vstack([data, outliers])


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_homography_affine_basic(use_magsac_plus_plus):
    """Test homography from affine correspondences."""
    angle = np.radians(5)
    H_gt = np.array([
        [np.cos(angle), -np.sin(angle), 0.05],
        [np.sin(angle), np.cos(angle), -0.02],
        [0.0, 0.0, 1.0]
    ])

    corrs = generate_affine_homography_data(H_gt, n_inliers=200, n_outliers=50)

    result, mask = pymagsac.findHomographyAffine(
        np.ascontiguousarray(corrs),
        probabilities=[],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=0.01,
        max_iters=5000,
    )

    assert result is not None, "Homography affine fitting failed"
    n_inliers_found = np.sum(mask)
    assert n_inliers_found > 100, f"Only {n_inliers_found} inliers found"


def test_homography_affine_insufficient_points():
    """Test that homography affine raises with fewer than 2 points."""
    corrs = np.random.rand(1, 8)
    with pytest.raises(Exception):
        pymagsac.findHomographyAffine(np.ascontiguousarray(corrs), probabilities=[])


def test_fundamental_affine_insufficient_points():
    """Test that fundamental affine raises with fewer than 4 points."""
    corrs = np.random.rand(3, 8)
    corrs[:, 4] = 1.0  # Ensure q1 is non-zero (solver divides by q1)
    with pytest.raises(Exception):
        pymagsac.findFundamentalMatrixAffine(np.ascontiguousarray(corrs), probabilities=[])


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_fundamental_affine_basic(use_magsac_plus_plus):
    """Test fundamental matrix from SIFT correspondences.

    Data format: [x1, y1, x2, y2, q1, q2, o1, o2]
    where q1,q2 are SIFT scales and o1,o2 are orientations (radians).
    """
    rng = np.random.default_rng(42)

    # Generate correspondences from a homography (epipolar geometry)
    angle = np.radians(5)
    H_gt = np.array([
        [np.cos(angle), -np.sin(angle), 0.05],
        [np.sin(angle), np.cos(angle), -0.03],
        [0.002, -0.001, 1.0]
    ])

    n_inliers = 200
    src = rng.uniform(-1, 1, (n_inliers, 2))
    src_h = np.hstack([src, np.ones((n_inliers, 1))])
    dst_h = (H_gt @ src_h.T).T
    dst = dst_h[:, :2] / dst_h[:, 2:3]

    # Generate SIFT-style scale and orientation
    # For a homography, the scale ratio and orientation change can be derived
    # from the Jacobian, but for testing we use consistent values
    data = np.zeros((n_inliers, 8))
    for i in range(n_inliers):
        x, y = src[i]
        d = H_gt[2, 0] * x + H_gt[2, 1] * y + H_gt[2, 2]
        # Scale: approximate as 1/d (projective scaling)
        q1 = 1.0 + rng.normal(0, 0.01)
        q2 = q1 / d
        # Orientation: rotation angle of the homography at this point
        o1 = rng.uniform(0, 2 * np.pi)
        o2 = o1 + angle  # Consistent with the rotation
        data[i] = [src[i, 0], src[i, 1], dst[i, 0], dst[i, 1], q1, q2, o1, o2]

    # Outliers with valid (non-zero) scales
    n_outliers = 80
    outliers = np.zeros((n_outliers, 8))
    outliers[:, :4] = rng.uniform(-2, 2, (n_outliers, 4))
    outliers[:, 4] = rng.uniform(0.5, 2.0, n_outliers)  # q1 > 0
    outliers[:, 5] = rng.uniform(0.5, 2.0, n_outliers)  # q2 > 0
    outliers[:, 6] = rng.uniform(0, 2 * np.pi, n_outliers)  # o1
    outliers[:, 7] = rng.uniform(0, 2 * np.pi, n_outliers)  # o2

    corrs = np.vstack([data, outliers])

    result, mask = pymagsac.findFundamentalMatrixAffine(
        np.ascontiguousarray(corrs),
        probabilities=[],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=0.05,
        max_iters=5000,
    )

    assert result is not None, "Fundamental matrix affine fitting failed"
    n_inliers_found = np.sum(mask)
    assert n_inliers_found > 50, f"Only {n_inliers_found} inliers found"

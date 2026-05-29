"""Test ``pymagsac.findRadialHomography``."""
import pymagsac
import numpy as np
import pytest


def generate_homography_data(H_gt, n_inliers=100, n_outliers=30, noise_std=0.5, seed=42):
    """Generate point correspondences from a known homography.

    Returns (N, 4) array: [x1, y1, x2, y2].
    """
    rng = np.random.default_rng(seed)

    # Random source points
    src = rng.uniform(-1, 1, (n_inliers, 2))

    # Apply homography: p2 = H * p1 (homogeneous)
    src_h = np.hstack([src, np.ones((n_inliers, 1))])
    dst_h = (H_gt @ src_h.T).T
    dst = dst_h[:, :2] / dst_h[:, 2:3]

    # Add noise
    dst += rng.normal(0, noise_std / 1000, dst.shape)

    inliers = np.hstack([src, dst])

    # Outliers
    outlier_src = rng.uniform(-2, 2, (n_outliers, 2))
    outlier_dst = rng.uniform(-2, 2, (n_outliers, 2))
    outliers = np.hstack([outlier_src, outlier_dst])

    return np.vstack([inliers, outliers])


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_radial_homography_basic(use_magsac_plus_plus):
    """Test radial homography with a simple planar transformation."""
    # Simple homography (rotation + translation in image plane)
    angle = np.radians(10)
    H_gt = np.array([
        [np.cos(angle), -np.sin(angle), 0.1],
        [np.sin(angle), np.cos(angle), -0.05],
        [0.0, 0.0, 1.0]
    ])

    corrs = generate_homography_data(H_gt, n_inliers=200, n_outliers=50)

    result, mask = pymagsac.findRadialHomography(
        np.ascontiguousarray(corrs),
        probabilities=[],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=0.01,
        conf=0.99,
        min_iters=100,
        max_iters=5000,
    )

    assert result is not None, "Radial homography fitting failed"
    # Check that enough inliers were found
    n_inliers_found = np.sum(mask)
    assert n_inliers_found > 100, f"Only {n_inliers_found} inliers found"


def test_radial_homography_insufficient_points():
    """Test that radial homography raises with fewer than 5 points."""
    corrs = np.random.rand(4, 4)
    with pytest.raises(Exception):
        pymagsac.findRadialHomography(np.ascontiguousarray(corrs), probabilities=[])

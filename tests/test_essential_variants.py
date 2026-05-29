"""Test ``pymagsac.findEssentialMatrixPlanar`` and ``pymagsac.findEssentialMatrixGravity``."""
import pymagsac
import numpy as np
import pytest


def generate_essential_data(R_gt, t_gt, K, n_inliers=100, n_outliers=30, seed=42):
    """Generate 2D-2D correspondences from a known relative pose.

    Returns (N, 4) array: [x1, y1, x2, y2] in pixel coordinates.
    """
    rng = np.random.default_rng(seed)

    # Random 3D points in front of both cameras
    points_3d = rng.uniform(-3, 3, (n_inliers, 3))
    points_3d[:, 2] = np.abs(points_3d[:, 2]) + 3  # Ensure positive depth

    # Project into camera 1 (identity pose)
    proj1 = (K @ points_3d.T).T
    img1 = proj1[:, :2] / proj1[:, 2:3]

    # Project into camera 2 (R, t)
    points_cam2 = (R_gt @ points_3d.T).T + t_gt
    proj2 = (K @ points_cam2.T).T
    img2 = proj2[:, :2] / proj2[:, 2:3]

    # Add small noise
    img1 += rng.normal(0, 0.5, img1.shape)
    img2 += rng.normal(0, 0.5, img2.shape)

    inliers = np.hstack([img1, img2])

    # Outliers
    outliers = np.hstack([
        rng.uniform(0, 640, (n_outliers, 2)),
        rng.uniform(0, 640, (n_outliers, 2))
    ])

    return np.vstack([inliers, outliers])


def rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_essential_planar_basic(use_magsac_plus_plus):
    """Test planar essential matrix estimation (2-point, planar motion)."""
    # Planar motion: rotation around Y axis only, translation in XZ plane
    R_gt = rotation_matrix([0, 1, 0], 5)
    t_gt = np.array([0.5, 0.0, 0.1])
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)

    corrs = generate_essential_data(R_gt, t_gt, K, n_inliers=200, n_outliers=50)

    E, mask = pymagsac.findEssentialMatrixPlanar(
        np.ascontiguousarray(corrs),
        K1=K, K2=K,
        probabilities=[],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=2.0,
        max_iters=5000,
    )

    assert E is not None, "Planar essential matrix estimation failed"
    n_inliers_found = np.sum(mask)
    assert n_inliers_found > 100, f"Only {n_inliers_found} inliers found"


def test_essential_planar_insufficient_points():
    """Test that planar essential matrix raises with fewer than 2 points."""
    K = np.eye(3)
    corrs = np.random.rand(1, 4)
    with pytest.raises(Exception):
        pymagsac.findEssentialMatrixPlanar(np.ascontiguousarray(corrs), K1=K, K2=K, probabilities=[])


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_essential_gravity_basic(use_magsac_plus_plus):
    """Test gravity-assisted essential matrix estimation (3-point)."""
    # General motion with known gravity direction
    R_gt = rotation_matrix([0.1, 1, 0.2], 10)
    t_gt = np.array([0.3, -0.1, 0.5])
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)

    corrs = generate_essential_data(R_gt, t_gt, K, n_inliers=200, n_outliers=50)

    # Gravity rotation matrices (align gravity to Y axis)
    # In practice these come from IMU; here we use identity (gravity already aligned)
    gravity_src = np.eye(3)
    gravity_dst = np.eye(3)

    E, mask = pymagsac.findEssentialMatrixGravity(
        np.ascontiguousarray(corrs),
        K1=K, K2=K,
        gravity_src=gravity_src,
        gravity_dst=gravity_dst,
        probabilities=[],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=2.0,
        max_iters=5000,
    )

    assert E is not None, "Gravity essential matrix estimation failed"
    n_inliers_found = np.sum(mask)
    assert n_inliers_found > 50, f"Only {n_inliers_found} inliers found"


def test_essential_gravity_insufficient_points():
    """Test that gravity essential matrix raises with fewer than 3 points."""
    K = np.eye(3)
    G = np.eye(3)
    corrs = np.random.rand(2, 4)
    with pytest.raises(Exception):
        pymagsac.findEssentialMatrixGravity(np.ascontiguousarray(corrs), K1=K, K2=K,
                                            gravity_src=G, gravity_dst=G, probabilities=[])

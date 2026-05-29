"""Test ``pymagsac.findPnP``."""
import pymagsac
import numpy as np
import pytest


def generate_pnp_data(R_gt, t_gt, n_inliers=100, n_outliers=30, noise_std=0.5, seed=42):
    """Generate 2D-3D correspondences from a known camera pose.

    Returns (N, 5) array: [u, v, x, y, z] where (u,v) is the normalized
    image point and (x,y,z) is the 3D world point.
    """
    rng = np.random.default_rng(seed)

    # Random 3D points in front of camera
    world_points = rng.uniform(-5, 5, (n_inliers, 3))
    # Ensure points are in front of camera (positive z after transform)
    world_points[:, 2] = np.abs(world_points[:, 2]) + 2

    # Project: p = R * X + t, then normalize
    cam_points = (R_gt @ world_points.T).T + t_gt
    # Normalized image coordinates
    image_points = cam_points[:, :2] / cam_points[:, 2:3]

    # Add noise to image points
    image_points += rng.normal(0, noise_std / 1000, image_points.shape)

    inliers = np.hstack([image_points, world_points])

    # Outliers: random image points with random 3D points
    outlier_img = rng.uniform(-2, 2, (n_outliers, 2))
    outlier_3d = rng.uniform(-10, 10, (n_outliers, 3))
    outliers = np.hstack([outlier_img, outlier_3d])

    correspondences = np.vstack([inliers, outliers])
    return correspondences


def rotation_matrix(axis, angle_deg):
    """Create rotation matrix from axis-angle."""
    angle = np.radians(angle_deg)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rotation_error(R1, R2):
    """Rotation error in degrees."""
    R = R1 @ R2.T
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_angle))


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_pnp_basic(use_magsac_plus_plus):
    """Test PnP with a simple camera pose."""
    R_gt = rotation_matrix([0, 1, 0], 15)
    t_gt = np.array([0.5, -0.3, 0.1])

    corrs = generate_pnp_data(R_gt, t_gt, n_inliers=150, n_outliers=50)

    pose, mask = pymagsac.findPnP(
        np.ascontiguousarray(corrs),
        probabilities=[],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=0.01,
        conf=0.99,
        min_iters=100,
        max_iters=5000,
    )

    assert pose is not None, "PnP failed"

    R_est = pose[:3, :3]
    t_est = pose[:3, 3]

    err_R = rotation_error(R_gt, R_est)
    err_t = np.linalg.norm(t_gt - t_est)

    assert err_R < 5.0, f"Rotation error {err_R:.2f} degrees"
    assert err_t < 0.5, f"Translation error {err_t:.3f}"


def test_pnp_identity_pose():
    """Test PnP with near-identity rotation and small translation."""
    R_gt = rotation_matrix([1, 0, 0], 2)  # 2 degree rotation
    t_gt = np.array([0.01, 0.02, 0.0])

    corrs = generate_pnp_data(R_gt, t_gt, n_inliers=200, n_outliers=50, noise_std=0.1)

    pose, mask = pymagsac.findPnP(
        np.ascontiguousarray(corrs),
        probabilities=[],
        sigma_th=0.005,
        max_iters=5000,
    )

    assert pose is not None, "PnP failed"
    R_est = pose[:3, :3]
    err_R = rotation_error(R_gt, R_est)
    assert err_R < 3.0, f"Rotation error {err_R:.2f} degrees"


def test_pnp_insufficient_points():
    """Test that PnP raises with fewer than 3 points."""
    corrs = np.array([[0.0, 0.0, 1.0, 0.0, 5.0],
                      [0.1, 0.0, 0.0, 1.0, 5.0]])
    with pytest.raises(Exception):
        pymagsac.findPnP(np.ascontiguousarray(corrs), probabilities=[])

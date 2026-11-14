"""Test ``pymagsac.findPlane3D``."""
import pymagsac
import numpy as np
import pytest


def generate_plane_data(
    normal, offset, n_inliers=200, n_outliers=50, noise_std=0.01, seed=42
):
    """Generate 3D points on a plane with noise and outliers.

    The plane is defined as normal . x + offset = 0.
    """
    rng = np.random.default_rng(seed)

    # Generate random points in a tangent basis
    # Find two vectors orthogonal to the normal
    normal = np.array(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Pick an arbitrary vector not parallel to normal
    if abs(normal[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(normal, v)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    # Generate inlier points on the plane
    coords = rng.uniform(-10, 10, (n_inliers, 2))
    # Point on the plane closest to origin
    base_point = -offset * normal
    inliers = base_point + coords[:, 0:1] * t1 + coords[:, 1:2] * t2
    # Add Gaussian noise
    inliers += rng.normal(0, noise_std, inliers.shape)

    # Generate outlier points randomly in a cube
    outliers = rng.uniform(-15, 15, (n_outliers, 3))

    points = np.vstack([inliers, outliers])
    gt_inlier_mask = np.zeros(len(points), dtype=bool)
    gt_inlier_mask[:n_inliers] = True

    return points, gt_inlier_mask


def verify_plane(
    points,
    sigma_th=0.1,
    sampler_id=0,
    use_magsac_plus_plus=True,
    min_iters=500,
    max_iters=5000,
):
    plane, mask = pymagsac.findPlane3D(
        np.ascontiguousarray(points),
        probabilities=[],
        min_iters=min_iters,
        max_iters=max_iters,
        sampler=sampler_id,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=sigma_th,
    )
    return plane, mask


def angle_between_normals(n1, n2):
    """Angle in degrees between two plane normals (handles sign ambiguity)."""
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    cos_angle = np.clip(abs(np.dot(n1, n2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_plane_fitting_basic(use_magsac_plus_plus):
    """Test plane fitting with a simple horizontal plane z=5."""
    gt_normal = np.array([0.0, 0.0, 1.0])
    gt_offset = -5.0  # plane: 0*x + 0*y + 1*z - 5 = 0

    points, gt_mask = generate_plane_data(
        gt_normal, gt_offset, n_inliers=200, n_outliers=50, noise_std=0.01
    )

    plane, mask = verify_plane(
        points,
        sigma_th=0.1,
        use_magsac_plus_plus=use_magsac_plus_plus,
    )

    assert plane is not None, "Plane fitting failed"

    # Check normal direction (up to sign)
    est_normal = plane[:3]
    angle_err = angle_between_normals(gt_normal, est_normal)
    assert angle_err < 2.0, f"Normal angle error {angle_err:.2f} degrees"

    # Check that most inliers are found
    n_inliers_found = np.sum(mask)
    assert n_inliers_found > 150, f"Only {n_inliers_found} inliers found"


@pytest.mark.parametrize(
    "normal",
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.3, -0.7, 0.5],
    ],
)
def test_plane_fitting_orientations(normal):
    """Test plane fitting with various orientations."""
    points, gt_mask = generate_plane_data(
        normal, offset=3.0, n_inliers=300, n_outliers=100, noise_std=0.005
    )

    plane, mask = verify_plane(points, sigma_th=0.05)

    assert plane is not None, "Plane fitting failed"

    est_normal = plane[:3]
    angle_err = angle_between_normals(normal, est_normal)
    assert angle_err < 2.0, f"Normal angle error {angle_err:.2f} degrees"


def test_plane_fitting_high_outlier_ratio():
    """Test plane fitting with 50% outliers."""
    gt_normal = np.array([0.0, 0.0, 1.0])
    points, gt_mask = generate_plane_data(
        gt_normal, offset=0.0, n_inliers=200, n_outliers=200, noise_std=0.01
    )

    plane, mask = verify_plane(points, sigma_th=0.1, max_iters=10000)

    assert plane is not None, "Plane fitting failed with 50% outliers"

    est_normal = plane[:3]
    angle_err = angle_between_normals(gt_normal, est_normal)
    assert angle_err < 3.0, f"Normal angle error {angle_err:.2f} degrees"


def test_plane_fitting_insufficient_points():
    """Test that plane fitting raises with fewer than 3 points."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(Exception):
        verify_plane(points)

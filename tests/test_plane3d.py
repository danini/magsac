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


def check_plane(plane, mask, normal, offset, n_inliers):
    """Checks the plane against normal . x + offset = 0 (up to sign), and the inlier mask.
    Tolerances: about five times the largest errors over 300 runs of each test."""
    assert plane is not None, "Plane fitting failed"
    normal = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    assert abs(np.linalg.norm(plane[:3]) - 1) < 1e-9, "The normal is not a unit vector"
    angle_err = angle_between_normals(normal, plane[:3])
    assert angle_err < 0.1, f"Normal angle error {angle_err:.4f} degrees"
    sign = np.sign(np.dot(normal, plane[:3]))
    offset_err = abs(sign * plane[3] - offset)
    assert offset_err < 0.005, f"Offset error {offset_err:.5f}"
    assert mask[:n_inliers].sum() >= 0.95 * n_inliers, f"Only {mask[:n_inliers].sum()} inliers found"
    n_outliers = len(mask) - n_inliers
    assert mask[n_inliers:].sum() <= 0.05 * n_outliers, f"{mask[n_inliers:].sum()} outliers accepted"


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
    check_plane(plane, mask, gt_normal, gt_offset, 200)


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
    check_plane(plane, mask, normal, 3.0, 300)


def test_plane_fitting_high_outlier_ratio():
    """Test plane fitting with 50% outliers."""
    gt_normal = np.array([0.0, 0.0, 1.0])
    points, gt_mask = generate_plane_data(
        gt_normal, offset=0.0, n_inliers=200, n_outliers=200, noise_std=0.01
    )

    plane, mask = verify_plane(points, sigma_th=0.1, max_iters=10000)
    check_plane(plane, mask, gt_normal, 0.0, 200)


@pytest.mark.parametrize("sampler", [1, 3, 4])
def test_plane_fitting_samplers(sampler):
    """PROSAC (rows sorted by quality), NG-RANSAC and AR-Sampler, with one probability per row."""
    points, gt_mask = generate_plane_data([0.3, -0.7, 0.5], offset=3.0, n_inliers=300, n_outliers=100, noise_std=0.005)
    probabilities = np.where(gt_mask, 0.9, 0.1)
    plane, mask = pymagsac.findPlane3D(points, probabilities, sampler=sampler, sigma_th=0.05, min_iters=500, max_iters=5000)
    check_plane(plane, mask, [0.3, -0.7, 0.5], 3.0, 300)


@pytest.mark.parametrize("sampler", [1, 3])
def test_plane_fitting_samples_with_a_repeated_point(sampler):
    """Samples that repeat a point define no plane and must be rejected. PROSAC's first sample
    repeats a point in about one run in three, and NG-RANSAC's samples often do when one point
    has most of the probability (here 1000 of about 1280)."""
    points, gt_mask = generate_plane_data([0.3, -0.7, 0.5], offset=3.0, n_inliers=300, n_outliers=100, noise_std=0.005)
    probabilities = np.where(gt_mask, 0.9, 0.1)
    probabilities[0] = 1000
    for _ in range(30 if sampler == 1 else 10):
        plane, mask = pymagsac.findPlane3D(points, probabilities, sampler=sampler, sigma_th=0.05, min_iters=500, max_iters=5000)
        check_plane(plane, mask, [0.3, -0.7, 0.5], 3.0, 300)


def test_plane_fitting_insufficient_points():
    """Test that plane fitting raises with fewer than 3 points."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        verify_plane(points)


@pytest.mark.parametrize("sampler, probabilities", [
    (3, []), (4, []), (3, [0.5] * 10),
    (3, np.zeros(20)), (4, np.zeros(20)),
    (3, np.r_[-0.5, np.ones(19)]), (4, np.r_[-0.5, np.ones(19)]),
    (3, np.r_[np.nan, np.ones(19)]), (4, np.r_[np.inf, np.ones(19)])])
def test_plane_fitting_invalid_probabilities(sampler, probabilities):
    """Samplers 3 and 4 need one probability per row, finite and non-negative, at least one positive."""
    points = np.random.default_rng(0).uniform(-1, 1, (20, 3))
    with pytest.raises(ValueError):
        pymagsac.findPlane3D(points, probabilities, sampler=sampler)


@pytest.mark.parametrize("sampler", [2, 5, -1])
def test_plane_fitting_unsupported_sampler(sampler):
    """Sampler 2 (P-NAPSAC) needs the image sizes, which findPlane3D doesn't take; 5 and -1 don't exist."""
    points = np.random.default_rng(0).uniform(-1, 1, (20, 3))
    with pytest.raises(ValueError):
        pymagsac.findPlane3D(points, np.ones(20), sampler=sampler)

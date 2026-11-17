"""Test ``pymagsac.findPlanarEssentialMatrix`` and ``pymagsac.findGravityEssentialMatrix``."""
import pymagsac
import numpy as np
import pytest

K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
N_INLIERS, N_OUTLIERS = 200, 100


def rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    S = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * S + (1 - np.cos(angle)) * S @ S


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def generate_correspondences(R, t, seed=42):
    """Pixel correspondences [x1, y1, x2, y2] of points seen by cameras [I | 0] and [R | t],
    with 0.5 px noise, followed by uniformly random outliers."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (N_INLIERS, 3))
    X[:, 2] = np.abs(X[:, 2]) + 4
    x1 = X @ K.T
    x2 = (X @ R.T + t) @ K.T
    inliers = np.hstack([x1[:, :2] / x1[:, 2:], x2[:, :2] / x2[:, 2:]])
    inliers += rng.normal(0, 0.5, inliers.shape)
    outliers = rng.uniform(0, 640, (N_OUTLIERS, 4))
    return np.vstack([inliers, outliers])


def check(E, mask, R, t):
    assert E is not None, "No essential matrix found"
    E_gt = skew(t) @ R
    E, E_gt = E / np.linalg.norm(E), E_gt / np.linalg.norm(E_gt)
    error = min(np.linalg.norm(E - E_gt), np.linalg.norm(E + E_gt))
    assert error < 0.2, f"Essential matrix error {error:.3f}"
    assert mask[:N_INLIERS].sum() > 0.9 * N_INLIERS, f"Only {mask[:N_INLIERS].sum()} inliers found"
    assert mask[N_INLIERS:].sum() < 0.1 * N_OUTLIERS, f"{mask[N_INLIERS:].sum()} outliers accepted"


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_essential_planar(use_magsac_plus_plus):
    """Planar motion: rotation about the y axis, translation in the xz plane."""
    R, t = rotation_matrix([0, 1, 0], 5), np.array([0.5, 0.0, 0.1])
    E, mask = pymagsac.findPlanarEssentialMatrix(
        generate_correspondences(R, t), K1=K, K2=K, probabilities=[],
        use_magsac_plus_plus=use_magsac_plus_plus, sigma_th=2.0, min_iters=500, max_iters=5000)
    check(E, mask, R, t)


def test_essential_planar_insufficient_points():
    with pytest.raises(ValueError):
        pymagsac.findPlanarEssentialMatrix(np.random.rand(1, 4), K1=K, K2=K, probabilities=[])


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_essential_gravity(use_magsac_plus_plus):
    """Cameras tilted by known rotations; once these are undone, the rotation is about the y axis."""
    G1, G2 = rotation_matrix([1, 0, 0.3], 8), rotation_matrix([0.2, 0, 1], -6)
    R = G2.T @ rotation_matrix([0, 1, 0], 10) @ G1
    t = np.array([0.3, -0.1, 0.5])
    # Positional arguments, in the order of pygcransac's findGravityEssentialMatrix.
    E, mask = pymagsac.findGravityEssentialMatrix(
        generate_correspondences(R, t), G1, G2, K, K, [],
        use_magsac_plus_plus=use_magsac_plus_plus, sigma_th=2.0, min_iters=500, max_iters=5000)
    check(E, mask, R, t)


def test_essential_gravity_insufficient_points():
    G = np.eye(3)
    with pytest.raises(ValueError):
        pymagsac.findGravityEssentialMatrix(np.random.rand(2, 4), source_gravity=G, destination_gravity=G,
                                            K1=K, K2=K, probabilities=[])

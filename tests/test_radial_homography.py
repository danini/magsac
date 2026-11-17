"""Test ``pymagsac.findRadialHomography``."""
import pymagsac
import numpy as np
import pytest

N_INLIERS, N_OUTLIERS = 200, 100


def distort(points, lam):
    """Inverse of the division model: finds x_d with x_d / (1 + lam |x_d|^2) = x_u."""
    r_u = np.linalg.norm(points, axis=1, keepdims=True)
    if lam == 0:
        return points.copy()
    # r_u * lam * r_d^2 - r_d + r_u = 0, the root that tends to r_u when lam -> 0
    r_d = (1 - np.sqrt(1 - 4 * lam * r_u**2)) / (2 * lam * np.maximum(r_u, 1e-12))
    return points * r_d / np.maximum(r_u, 1e-12)


def generate_correspondences(H, lam1, lam2, noise_std, seed=42):
    """Correspondences [x1, y1, x2, y2] in coordinates centred on the distortion centre and
    scaled to about [-1, 1], followed by uniformly random outliers."""
    rng = np.random.default_rng(seed)
    u1 = rng.uniform(-0.8, 0.8, (N_INLIERS, 2))
    u2 = np.c_[u1, np.ones(N_INLIERS)] @ H.T
    u2 = u2[:, :2] / u2[:, 2:]
    inliers = np.hstack([distort(u1, lam1), distort(u2, lam2)])
    inliers += rng.normal(0, noise_std, inliers.shape)
    outliers = rng.uniform(-1, 1, (N_OUTLIERS, 4))
    return np.vstack([inliers, outliers])


def test_radial_homography():
    # MAGSAC++ only: the original MAGSAC sometimes returns a poor distortion estimate on this data.
    angle = np.radians(10)
    H_gt = np.array([[np.cos(angle), -np.sin(angle), 0.05],
                     [np.sin(angle), np.cos(angle), -0.03],
                     [0.02, -0.01, 1.0]])
    lam1, lam2 = -0.2, -0.1
    corrs = generate_correspondences(H_gt, lam1, lam2, noise_std=0.0005)

    H, distortion, mask = pymagsac.findRadialHomography(
        corrs, probabilities=[], sigma_th=0.005, min_iters=500, max_iters=5000)

    assert H is not None, "No model found"
    np.testing.assert_allclose(distortion, [lam1, lam2], atol=0.04)
    np.testing.assert_allclose(H / H[2, 2], H_gt, atol=0.003)
    assert mask[:N_INLIERS].sum() > 0.9 * N_INLIERS, f"Only {mask[:N_INLIERS].sum()} inliers found"
    assert mask[N_INLIERS:].sum() < 0.1 * N_OUTLIERS, f"{mask[N_INLIERS:].sum()} outliers accepted"


def test_radial_homography_insufficient_points():
    with pytest.raises(ValueError):
        pymagsac.findRadialHomography(np.random.rand(4, 4), probabilities=[])

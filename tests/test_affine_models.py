"""Test ``pymagsac.findHomographyAffine``."""
import pymagsac
import numpy as np
import pytest

N_INLIERS, N_OUTLIERS = 200, 100


def generate_affine_correspondences(H, seed=42):
    """Affine correspondences [x1, y1, x2, y2, a11, a12, a21, a22] in pixels, where A is the
    Jacobian of H at (x1, y1), with 0.5 px noise on (x2, y2), followed by random outliers."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 640, (N_INLIERS, 2))
    p = np.c_[x1, np.ones(N_INLIERS)] @ H.T
    x2 = p[:, :2] / p[:, 2:]
    # d(x2)/d(x1) = (H[:2, :2] - x2 H[2, :2]) / p_z
    A = (H[None, :2, :2] - x2[:, :, None] * H[None, 2:3, :2]) / p[:, 2, None, None]
    x2 += rng.normal(0, 0.5, x2.shape)
    inliers = np.hstack([x1, x2, A.reshape(-1, 4)])
    outliers = np.hstack([rng.uniform(0, 640, (N_OUTLIERS, 4)), rng.normal(1, 0.3, (N_OUTLIERS, 4))])
    return np.vstack([inliers, outliers])


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_homography_affine(use_magsac_plus_plus):
    H_gt = np.array([[1.1, 0.2, 30.0], [-0.1, 0.9, -20.0], [1e-4, 2e-4, 1.0]])
    H, mask = pymagsac.findHomographyAffine(
        generate_affine_correspondences(H_gt), probabilities=[],
        use_magsac_plus_plus=use_magsac_plus_plus, sigma_th=3.0, min_iters=500)

    assert H is not None, "No homography found"
    # Compare the mapped points rather than the entries of H, whose scales differ.
    x = np.array([[0, 0, 1], [640, 0, 1], [0, 480, 1], [640, 480, 1.0]])
    y, y_gt = x @ H.T, x @ H_gt.T
    error = np.linalg.norm(y[:, :2] / y[:, 2:] - y_gt[:, :2] / y_gt[:, 2:], axis=1).max()
    # Largest error over 300 runs: 1.2 px with MAGSAC++, 4.3 px with MAGSAC.
    assert error < (3.0 if use_magsac_plus_plus else 8.0), f"Corner transfer error {error:.2f} px"
    assert mask[:N_INLIERS].sum() > 0.9 * N_INLIERS, f"Only {mask[:N_INLIERS].sum()} inliers found"
    assert mask[N_INLIERS:].sum() < 0.1 * N_OUTLIERS, f"{mask[N_INLIERS:].sum()} outliers accepted"


def test_homography_affine_insufficient_points():
    with pytest.raises(ValueError):
        pymagsac.findHomographyAffine(np.random.rand(1, 8), probabilities=[])

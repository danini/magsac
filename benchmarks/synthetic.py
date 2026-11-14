"""Synthetic problems with known ground truth, and the residuals used to score them.

Each generator returns (data, clean, inlier): `data` holds the noisy inliers and
uniform outliers in random order, `inlier` marks the inliers, and `clean` holds
the same inliers without noise, so the residual of an estimated model on `clean`
measures its distance to the true model.
"""
import numpy as np

W = 1000.0  # image width and height, in pixels
K = np.array([[1000.0, 0, W / 2], [0, 1000.0, W / 2], [0, 0, 1.0]])


def _rotation(rv):
    th = np.linalg.norm(rv)
    a = rv / th
    skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.cos(th) * np.eye(3) + np.sin(th) * skew + (1 - np.cos(th)) * np.outer(a, a)


def _add_noise_and_outliers(rng, clean, outlier_ratio, noise):
    n_out = int(round(len(clean) * outlier_ratio / (1 - outlier_ratio)))
    noisy = clean + rng.normal(0, noise, clean.shape)
    outliers = rng.uniform(0, W, (n_out, clean.shape[1]))
    order = rng.permutation(len(clean) + n_out)
    inlier = order < len(clean)
    return np.ascontiguousarray(np.vstack([noisy, outliers])[order]), clean[order[inlier]], inlier


def gen_fundamental(rng, n, outlier_ratio, noise):
    """Two views of random 3D points (small rotation, random baseline)."""
    R = _rotation(rng.normal(0, 0.15, 3))
    t = rng.normal(0, 1.0, 3)
    t[2] = abs(t[2]) + 1.0
    X = np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(4, 8, n)])
    x1 = X @ K.T
    x2 = (X @ R.T + t) @ K.T
    clean = np.c_[x1[:, :2] / x1[:, 2:], x2[:, :2] / x2[:, 2:]]
    return _add_noise_and_outliers(rng, clean, outlier_ratio, noise)


def gen_homography(rng, n, outlier_ratio, noise):
    """Points of the first image mapped by a random homography close to identity."""
    H = np.eye(3) + np.r_[rng.normal(0, 0.1, 6), rng.normal(0, 1e-4, 2), 0].reshape(3, 3)
    H[:2, 2] += rng.normal(0, 30, 2)
    p1 = rng.uniform(0, W, (n, 2))
    p2 = np.c_[p1, np.ones(n)] @ H.T
    clean = np.c_[p1, p2[:, :2] / p2[:, 2:]]
    return _add_noise_and_outliers(rng, clean, outlier_ratio, noise)


# Residuals of a model, in pixels.

def sampson(corrs, F):
    """Sampson distance to a fundamental matrix."""
    x1 = np.c_[corrs[:, :2], np.ones(len(corrs))]
    x2 = np.c_[corrs[:, 2:4], np.ones(len(corrs))]
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / den)


def transfer(corrs, H):
    """Distance between H x1 and x2."""
    p = np.c_[corrs[:, :2], np.ones(len(corrs))] @ H.T
    return np.hypot(*(p[:, :2] / p[:, 2:] - corrs[:, 2:4]).T)


def rmse(residuals):
    return float(np.sqrt(np.mean(np.square(residuals))))


def f1(estimated, true):
    """F1 score of an estimated inlier mask."""
    tp = np.sum(estimated & true)
    return float(2 * tp / (np.sum(estimated) + np.sum(true))) if tp else 0.0

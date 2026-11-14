"""Test that ``pymagsac.findEssentialMatrix`` returns its refined model.

The function ends with a bundle adjustment of the relative pose that minimizes
the squared Sampson errors of the inliers. The returned essential matrix should
therefore be a minimum of that cost: a few Gauss-Newton steps from it should
barely decrease it.
"""
import numpy as np
import pymagsac


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rotation(w):
    angle = np.linalg.norm(w)
    if angle < 1e-15:
        return np.eye(3)
    k = _skew(w / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _sampson(E, x1, x2):
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.sum(x2 * Ex1, axis=1)
    return num / np.sqrt(Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2)


def _relative_cost_decrease(E, x1, x2, steps=5):
    """Relative decrease of the sum of squared Sampson errors by Gauss-Newton."""
    U, _, Vt = np.linalg.svd(E)
    # E = U diag(1, 1, 0) Vt: perturb U and V by rotations (5 degrees of
    # freedom once the rotation about the third axis of both is removed).
    def essential(p):
        Up = U @ _rotation(p[:3])
        Vp = Vt.T @ _rotation(np.r_[p[3:5], -p[2]])
        return Up @ np.diag([1.0, 1.0, 0.0]) @ Vp.T

    p = np.zeros(5)
    cost0 = np.sum(_sampson(E, x1, x2)**2)
    for _ in range(steps):
        r = _sampson(essential(p), x1, x2)
        J = np.empty((len(r), 5))
        for i in range(5):
            dp = np.zeros(5)
            dp[i] = 1e-7
            J[:, i] = (_sampson(essential(p + dp), x1, x2) - r) / 1e-7
        p = p + np.linalg.lstsq(J, -r, rcond=None)[0]
    cost = np.sum(_sampson(essential(p), x1, x2)**2)
    return (cost0 - cost) / cost0


def _scene(rng, n_inliers=200, n_outliers=100, noise=0.5):
    """Matches between two 640x480 views, with outliers far from the epipolar lines."""
    K = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    R = _rotation(np.radians(10.0) * np.array([0.6, 0.8, 0.0]))
    t = np.array([1.0, 0.2, 0.1])
    X = np.c_[rng.uniform(-3, 3, (4 * n_inliers, 2)), rng.uniform(4, 10, 4 * n_inliers)]
    X2 = X @ R.T + t
    p1 = X @ K.T
    p2 = X2 @ K.T
    p1 = p1[:, :2] / p1[:, 2:]
    p2 = p2[:, :2] / p2[:, 2:]
    visible = np.all((p1 >= 0) & (p1 < [640, 480]) & (p2 >= 0) & (p2 < [640, 480]), axis=1)
    inliers = np.c_[p1, p2][visible][:n_inliers] + rng.normal(0, noise, (n_inliers, 4))

    E = _skew(t) @ R
    outliers = rng.uniform(0, 1, (20 * n_outliers, 4)) * [640, 480, 640, 480]
    x1 = np.c_[outliers[:, :2], np.ones(len(outliers))] @ np.linalg.inv(K).T
    x2 = np.c_[outliers[:, 2:], np.ones(len(outliers))] @ np.linalg.inv(K).T
    outliers = outliers[np.abs(_sampson(E, x1, x2)) * 800 > 50][:n_outliers]
    return np.r_[inliers, outliers], K


def test_returned_model_is_refined():
    rng = np.random.default_rng(0)
    for _ in range(5):
        correspondences, K = _scene(rng)
        E, mask = pymagsac.findEssentialMatrix(
            np.ascontiguousarray(correspondences), K, K, 640, 480, 640, 480,
            np.zeros(0), sampler=0, sigma_th=5.0)
        assert E is not None
        np.testing.assert_array_equal(mask, np.arange(len(mask)) < 200)

        x1 = np.c_[correspondences[:, :2], np.ones(len(mask))] @ np.linalg.inv(K).T
        x2 = np.c_[correspondences[:, 2:], np.ones(len(mask))] @ np.linalg.inv(K).T
        assert _relative_cost_decrease(E, x1[mask], x2[mask]) < 1e-5

"""Test the ``seed`` argument of the estimators."""
import numpy as np
import pymagsac
import pytest

N = 300
RNG = np.random.default_rng(0)
K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
PROBABILITIES = RNG.uniform(0, 1, N)


def project(P, X):
    x = np.c_[X, np.ones(len(X))] @ P.T
    return x[:, :2] / x[:, 2:]


def with_outliers(points, low, high, ratio=0.6):
    """Replaces the first points (the given ratio of them) by uniform outliers."""
    points = points.copy()
    n = int(ratio * len(points))
    points[:n] = RNG.uniform(low, high, size=(n, points.shape[1]))
    return points


X1 = RNG.uniform([0, 0], [640, 480], size=(N, 2))
H = np.array([[1.1, 0.05, 10], [-0.03, 0.95, 5], [1e-4, -5e-5, 1]])
HOMOGRAPHY = with_outliers(np.c_[X1, project(H, X1) + RNG.normal(0, 1, (N, 2))], 0, 640)

X = RNG.uniform([-2, -2, 4], [2, 2, 8], size=(N, 3))
R = np.array([[0.995, 0, 0.0998], [0, 1, 0], [-0.0998, 0, 0.995]])
T = np.array([1.0, 0.1, 0.0])
TWO_VIEW = with_outliers(
    np.c_[project(K @ np.eye(3, 4), X), project(K @ np.c_[R, T], X) + RNG.normal(0, 1, (N, 2))],
    0, 640)
# Fewer outliers: with 60%, 5 iterations rarely find a rigid transformation
RIGID = with_outliers(np.c_[X, X @ R.T + T + RNG.normal(0, 0.01, (N, 3))], -3, 9, ratio=0.3)

# Few iterations, so that different samples give different results
ITERS = dict(min_iters=5, max_iters=5)

ESTIMATORS = {
    "findHomography": lambda sampler, seed: pymagsac.findHomography(
        HOMOGRAPHY, 640, 480, 640, 480, PROBABILITIES, sampler=sampler,
        sigma_th=3.0, seed=seed, **ITERS),
    "findFundamentalMatrix": lambda sampler, seed: pymagsac.findFundamentalMatrix(
        TWO_VIEW, 640, 480, 640, 480, PROBABILITIES, sampler=sampler,
        sigma_th=3.0, seed=seed, **ITERS),
    "findEssentialMatrix": lambda sampler, seed: pymagsac.findEssentialMatrix(
        TWO_VIEW, K, K, 640, 480, 640, 480, PROBABILITIES, sampler=sampler,
        sigma_th=3.0, seed=seed, **ITERS),
    "findRigidTransformation": lambda sampler, seed: pymagsac.findRigidTransformation(
        RIGID, PROBABILITIES, sampler=sampler, sigma_th=0.05, seed=seed, **ITERS),
    # findLine2D isn't tested: it crashes, with or without a seed, until
    # danini/graph-cut-ransac#52 (a null sample in the line solver) is merged.
}


def result(estimator, sampler, seed):
    model, mask = ESTIMATORS[estimator](sampler, seed)
    return (None if model is None else np.asarray(model).tobytes(), np.asarray(mask).tobytes())


# Samplers: 0 uniform, 1 PROSAC, 2 P-NAPSAC, 3 importance sampling, 4 AR-Sampler.
# The rigid transformation estimator has no P-NAPSAC.
CASES = [(estimator, sampler) for estimator in ESTIMATORS for sampler in range(5)
         if (estimator, sampler) != ("findRigidTransformation", 2)]


@pytest.mark.parametrize("estimator, sampler", CASES)
def test_same_seed_same_result(estimator, sampler):
    assert result(estimator, sampler, 7) == result(estimator, sampler, 7)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_different_seeds_different_results(estimator):
    assert len({result(estimator, 0, seed) for seed in range(5)}) > 1


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_no_seed_after_a_seed_is_random(estimator):
    result(estimator, 0, 7)
    assert len({result(estimator, 0, None) for _ in range(5)}) > 1


def test_negative_seed_is_rejected():
    with pytest.raises(TypeError):
        result("findHomography", 0, -1)

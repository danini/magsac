"""Samplers 3 (importance) and 4 (AR-Sampler) need one probability per point, finite and non-negative,
at least one positive."""
import numpy as np
import pymagsac
import pytest

N = 50
RNG = np.random.default_rng(0)
K = np.eye(3)

FUNCTIONS = {
    "findHomography": lambda d, p, s: pymagsac.findHomography(
        d, 100, 100, 100, 100, probabilities=p, sampler=s),
    "findFundamentalMatrix": lambda d, p, s: pymagsac.findFundamentalMatrix(
        d, 100, 100, 100, 100, probabilities=p, sampler=s),
    "findEssentialMatrix": lambda d, p, s: pymagsac.findEssentialMatrix(
        d, K, K, 100, 100, 100, 100, probabilities=p, sampler=s),
    "findLine2D": lambda d, p, s: pymagsac.findLine2D(
        np.ascontiguousarray(d[:, :2]), 100, 100, probabilities=p, sampler=s),
    "findRigidTransformation": lambda d, p, s: pymagsac.findRigidTransformation(
        np.ascontiguousarray(np.c_[d, d[:, :2]]), probabilities=p, sampler=s),
}


@pytest.mark.parametrize("sampler", [3, 4])
@pytest.mark.parametrize("probabilities", [[], np.ones(N - 1)])
@pytest.mark.parametrize("name", FUNCTIONS)
def test_wrong_number_of_probabilities(name, probabilities, sampler):
    data = np.ascontiguousarray(RNG.uniform(0, 100, (N, 4)))
    with pytest.raises(ValueError, match="one probability per point"):
        FUNCTIONS[name](data, probabilities, sampler)


@pytest.mark.parametrize("sampler", [3, 4])
@pytest.mark.parametrize("probabilities, message", [
    (np.zeros(N), "positive probability"),
    (np.r_[-0.5, np.ones(N - 1)], "non-negative"),
    (np.r_[np.nan, np.ones(N - 1)], "non-negative"),
    (np.r_[np.inf, np.ones(N - 1)], "non-negative")])
@pytest.mark.parametrize("name", FUNCTIONS)
def test_invalid_probabilities(name, probabilities, message, sampler):
    data = np.ascontiguousarray(RNG.uniform(0, 100, (N, 4)))
    with pytest.raises(ValueError, match=message):
        FUNCTIONS[name](data, probabilities, sampler)


@pytest.mark.parametrize("sampler", [3, 4])
def test_one_probability_per_point(sampler):
    data = np.ascontiguousarray(RNG.uniform(0, 100, (N, 4)))
    FUNCTIONS["findHomography"](data, np.ones(N), sampler)

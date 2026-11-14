"""MAGSAC++ on data that the model fits exactly (zero residuals)."""
import numpy as np
import pymagsac
import pytest


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_line2d_on_exact_points(use_magsac_plus_plus):
    # Points exactly on the line x = 3: every residual is zero
    rng = np.random.default_rng(0)
    points = np.c_[np.full(100, 3.0), rng.uniform(0, 100, 100)]

    line, mask = pymagsac.findLine2D(
        points, 100, 100, probabilities=[],
        use_magsac_plus_plus=use_magsac_plus_plus)

    assert line is not None
    line = line / line[0]
    np.testing.assert_allclose(line, [1, 0, -3], atol=1e-9)
    assert mask.all()

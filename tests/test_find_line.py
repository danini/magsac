"""Test ``pymagsac.findLine2D`` on real edge points."""

import numpy as np
import cv2
import pytest
import pymagsac


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_find_line2d_edges(use_magsac_plus_plus):
    # Load test image
    img = cv2.imread("../graph-cut-ransac/build/data/adam/adam1.png")
    assert img is not None, "Test image not found"

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    # Extract edge points
    edge_points = np.argwhere(edges == 255)[:, ::-1]  # Swap x,y
    assert len(edge_points) > 0, "No edges detected"

    # Run MAGSAC line fitting
    line, mask = pymagsac.findLine2D(
        np.ascontiguousarray(edge_points),
        w1=img.shape[1],
        h1=img.shape[0],
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=3.0,
        conf=0.99,
        min_iters=500,
        max_iters=2000,
        partition_num=1,
    )

    assert line is not None, "Line fitting failed"
    assert mask.shape[0] == edge_points.shape[0], "Mask size mismatch"

    # Check that there are some inliers
    inlier_count = np.sum(mask)
    assert inlier_count > 0, "No inliers detected"

    # Optionally verify line parameters roughly make sense
    a, b, c = line
    # The line should not be degenerate
    assert not (a == 0 and b == 0)


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_find_line2d_point_cloud(use_magsac_plus_plus):
    np.random.seed(42)

    # Generate points along y = 0.5 * x + 2 with noise
    n_points = 200
    x = np.linspace(0, 100, n_points)
    y = 0.5 * x + 2 + np.random.normal(0, 1.0, n_points)  # Gaussian noise
    points = np.column_stack([x, y])

    # Add some outliers
    n_outliers = 20
    outliers = np.random.uniform(low=0, high=100, size=(n_outliers, 2))
    points_with_outliers = np.vstack([points, outliers])

    # Run MAGSAC line fitting
    line, mask = pymagsac.findLine2D(
        np.ascontiguousarray(points_with_outliers),
        w1=100.0,
        h1=100.0,
        probabilities=np.array([], dtype=np.float64),
        sampler=0,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=1.0,
        conf=0.99,
        min_iters=500,
        max_iters=2000,
        partition_num=1,
    )

    # Check outputs
    assert line is not None, "Line fitting failed"
    assert mask.shape[0] == points_with_outliers.shape[0], "Mask size mismatch"
    inlier_count = np.sum(mask)
    assert inlier_count >= n_points * 0.7, "Too few inliers detected"

    # Optionally, verify slope approximately correct
    a, b, c = line
    slope_estimated = -a / b if b != 0 else np.inf
    assert np.isclose(slope_estimated, 0.5, atol=0.1), (
        f"Slope mismatch: {slope_estimated}"
    )

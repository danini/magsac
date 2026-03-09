import pytest
from pathlib import Path
import numpy as np
import cv2
import pymagsac

THIS_PATH = Path(__file__).resolve().parent
DATA_PATH = THIS_PATH / ".." / "data"


def detect_and_match(img1, img2, snn_threshold=0.85):
    """
    Detect SIFT features, match with BFMatcher and ratio test, and sort by SNN ratio.

    Parameters
    ----------
    img1 : np.ndarray
        First RGB image.
    img2 : np.ndarray
        Second RGB image.
    snn_threshold : float, optional
        Lowe's ratio test threshold. Default is 0.85.

    Returns
    -------
    pts1 : np.ndarray
        Keypoints from the first image, shape (N, 2).
    pts2 : np.ndarray
        Keypoints from the second image, shape (N, 2).
    correspondences : np.ndarray
        Combined points for MAGSAC, shape (N, 4).
    probabilities : np.ndarray
        Probabilities based on SNN ranking.
    """
    det = cv2.SIFT_create(8000)
    kps1, descs1 = det.detectAndCompute(img1, None)
    kps2, descs2 = det.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1, descs2, k=2)

    tentatives = []
    snn_ratios = []
    for m, n in matches:
        if m.distance < snn_threshold * n.distance:
            tentatives.append(m)
            snn_ratios.append(m.distance / n.distance)

    sorted_idx = np.argsort(snn_ratios)
    tentatives = list(np.array(tentatives)[sorted_idx])

    pts1 = np.float32([kps1[m.queryIdx].pt for m in tentatives])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in tentatives])
    correspondences = np.hstack([pts1, pts2]).astype(np.float64)

    # Probabilities for MAGSAC++ (SNN ranking)
    probabilities = 1.0 - np.arange(len(tentatives)) / len(tentatives)
    return pts1, pts2, correspondences, probabilities


def reprojection_error(E, pts1, pts2, K1, K2) -> float:
    """
    Compute median reprojection error of points given an essential matrix.

    Parameters
    ----------
    E : np.ndarray
        Essential matrix, shape (3, 3).
    pts1 : np.ndarray
        Points in the first image, shape (N, 2).
    pts2 : np.ndarray
        Points in the second image, shape (N, 2).
    K1 : np.ndarray
        Intrinsic matrix of the first camera, shape (3, 3).
    K2 : np.ndarray
        Intrinsic matrix of the second camera, shape (3, 3).

    Returns
    -------
    float
        Median epipolar constraint error.
    """
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    errs = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
    return float(np.median(errs))


@pytest.mark.parametrize("use_magsac_plus_plus", [True, False])
def test_find_essential_matrix(use_magsac_plus_plus: bool) -> None:
    # Load images and intrinsics
    img1 = cv2.cvtColor(
        cv2.imread(DATA_PATH / "essential_matrix/fountain1.jpg"), cv2.COLOR_BGR2RGB
    )
    img2 = cv2.cvtColor(
        cv2.imread(DATA_PATH / "essential_matrix/fountain2.jpg"), cv2.COLOR_BGR2RGB
    )
    K1 = np.loadtxt(DATA_PATH / "essential_matrix/fountain1.K")
    K2 = np.loadtxt(DATA_PATH / "essential_matrix/fountain2.K")

    pts1, pts2, correspondences, probabilities = detect_and_match(img1, img2)

    # OpenCV baseline
    cv_E, cv_mask = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=K1, method=cv2.RANSAC, prob=0.99, threshold=1.0
    )
    cv_inliers = int(cv_mask.sum()) if cv_mask is not None else 0
    cv_err = reprojection_error(cv_E, pts1, pts2, K1, K2)

    # MAGSAC++ estimation
    mag_E, mag_mask = pymagsac.findEssentialMatrix(
        correspondences,
        K1,
        K2,
        img1.shape[1],
        img1.shape[0],
        img2.shape[1],
        img2.shape[0],
        probabilities=probabilities,
        sampler=4,
        use_magsac_plus_plus=use_magsac_plus_plus,
        sigma_th=1.5,
    )
    mag_inliers = int(mag_mask.sum())
    mag_err = reprojection_error(mag_E, pts1, pts2, K1, K2)

    # MAGSAC++ should be at least as good as OpenCV
    assert mag_inliers >= cv_inliers, (
        f"MAGSAC++ inliers {mag_inliers} < OpenCV {cv_inliers}"
    )
    assert mag_err <= cv_err, (
        f"MAGSAC++ reprojection error {mag_err:.3f} > OpenCV {cv_err:.3f}"
    )

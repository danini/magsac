import numpy as np
from numpy.typing import NDArray
from typing import Optional

def adaptiveInlierSelection(
    x1y1: NDArray[np.float64],
    x2y2: NDArray[np.float64],
    modelParameters: NDArray[np.float64],
    maximumThreshold: float,
    problemType: int,
    minimumInlierNumber: int = 20,
) -> tuple[NDArray[np.float64], int, float]: ...

# returns: (inliers mask [bool, 1D], inlier count, best threshold)

def findEssentialMatrix(
    correspondences: NDArray[np.float64],
    K1: NDArray[np.float64],
    K2: NDArray[np.float64],
    w1: float,
    h1: float,
    w2: float,
    h2: float,
    probabilities: NDArray[np.float64] = np.array([]),
    sampler: int = 4,
    use_magsac_plus_plus: bool = True,
    sigma_th: float = 1.0,
    conf: float = 0.99,
    min_iters: int = 50,
    max_iters: int = 1000,
    partition_num: int = 5,
) -> tuple[NDArray[np.float64] | None, NDArray[bool]]: ...

# returns: (3x3 Essential matrix or None, inliers mask)

def findFundamentalMatrix(
    correspondences: NDArray[np.float64],
    w1: float,
    h1: float,
    w2: float,
    h2: float,
    probabilities: NDArray[np.float64] = np.array([]),
    sampler: int = 4,
    use_magsac_plus_plus: bool = True,
    sigma_th: float = 1.0,
    conf: float = 0.99,
    min_iters: int = 50,
    max_iters: int = 1000,
    partition_num: int = 5,
) -> tuple[NDArray[np.float64] | None, NDArray[bool]]: ...

# returns: (3x3 Fundamental matrix or None, inliers mask)

def findRigidTransformation(
    correspondences: NDArray[np.float64],
    probabilities: NDArray[np.float64] = np.array([]),
    sampler: int = 4,
    use_magsac_plus_plus: bool = True,
    sigma_th: float = 1.0,
    conf: float = 0.99,
    min_iters: int = 50,
    max_iters: int = 1000,
    partition_num: int = 5,
) -> tuple[NDArray[np.float64] | None, NDArray[bool]]: ...

# returns: (4x4 transformation matrix or None, inliers mask)

def findHomography(
    correspondences: NDArray[np.float64],
    w1: float,
    h1: float,
    w2: float,
    h2: float,
    probabilities: NDArray[np.float64] = np.array([]),
    sampler: int = 4,
    use_magsac_plus_plus: bool = True,
    sigma_th: float = 1.0,
    conf: float = 0.99,
    min_iters: int = 50,
    max_iters: int = 1000,
    partition_num: int = 5,
) -> tuple[NDArray[np.float64] | None, NDArray[bool]]: ...

# returns: (3x3 Homography or None, inliers mask)

def findLine2D(
    points: NDArray[np.float64],
    w1: float,
    h1: float,
    probabilities: NDArray[np.float64] = np.array([]),
    sampler: int = 0,
    use_magsac_plus_plus: bool = True,
    sigma_th: float = 1.0,
    conf: float = 0.99,
    min_iters: int = 50,
    max_iters: int = 1000,
    partition_num: int = 5,
) -> tuple[NDArray[np.float64] | None, NDArray[bool]]: ...

# returns: (line parameters [a,b,c] or None, inliers mask)

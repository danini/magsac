"""Loader for the annotated scenes in the repository's data/ directory.

Each '<scene>_pts.txt' holds one correspondence per row, in the format read by
examples/cpp_example.cpp:
  'x1 y1 1 x2 y2 1 label'                    (data/fundamental_matrix, data/homography)
  'x1 y1 x2 y2 s1 s2 detector descriptor label' (data/homography/extremeview)
Rows with label > 0 are the ground-truth inliers.
"""
import glob
import os
import numpy as np


def load_annotated_pts(path):
    """Returns (correspondences N x 4 [x1 y1 x2 y2], inlier mask)."""
    corrs, labels = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) == 9:
                corrs.append([float(v) for v in t[:4]])
            elif len(t) == 7:
                corrs.append([float(t[i]) for i in (0, 1, 3, 4)])
            else:
                continue
            labels.append(float(t[-1]))
    return np.ascontiguousarray(corrs, dtype=np.float64), np.array(labels) > 0


def iter_scenes(directory):
    """Yields (scene name, correspondences, inlier mask) for every '<scene>_pts.txt'
    in directory (not in its subdirectories)."""
    for path in sorted(glob.glob(os.path.join(directory, "*_pts.txt"))):
        yield (os.path.basename(path)[:-len("_pts.txt")], *load_annotated_pts(path))

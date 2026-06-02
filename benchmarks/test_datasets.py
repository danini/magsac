#!/usr/bin/env python3
"""Unit tests for datasets.py loaders — tiny synthetic fixtures in each expected
on-disk format, so parsing is proven without downloading the full datasets.

Run:  PYTHONPATH=. python3 test_datasets.py   (needs scipy; cv2 path not covered)
"""
import os
import tempfile
import numpy as np
from scipy.io import savemat
import datasets as ds


def test_mat():
    d = tempfile.mkdtemp(); N = 5
    data = np.zeros((6, N))
    data[0] = [10, 20, 30, 40, 50]; data[1] = [11, 21, 31, 41, 51]; data[2] = 1
    data[3] = [12, 22, 32, 42, 52]; data[4] = [13, 23, 33, 43, 53]; data[5] = 1
    savemat(os.path.join(d, "scene.mat"), {"data": data, "label": np.array([1, 1, 1, 0, 2])})
    name, corrs, mask = ds.load_mat(os.path.join(d, "scene.mat"))
    assert corrs.shape == (5, 4) and corrs[0].tolist() == [10, 11, 12, 13]
    assert mask.tolist() == [True, True, True, False, False]          # dominant label == 1


def test_gt_h_pair():
    d = tempfile.mkdtemp()
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    p1 = np.array([[100, 100], [200, 150], [50, 300], [400, 400]], float)
    p2 = (H @ np.c_[p1, np.ones(4)].T).T; p2 = p2[:, :2] / p2[:, 2:]
    np.savetxt(os.path.join(d, "pair.corr"), np.vstack([np.c_[p1, p2], [10, 10, 999, 999]]))
    np.savetxt(os.path.join(d, "pair.H"), H)
    _, c2, m2 = ds.load_gt_h_pair(os.path.join(d, "pair.corr"), os.path.join(d, "pair.H"), gt_thresh=1.0)
    assert c2.shape == (5, 4) and m2.tolist() == [True, True, True, True, False]
    assert {n for n, *_ in ds.iter_datasets(d, task="homography", gt_thresh=1.0)} == {"pair"}


def test_annotated_and_essential():
    d = tempfile.mkdtemp()
    # F/H annotated format: x1 y1 s x2 y2 s label  (readAnnotatedPoints)
    open(os.path.join(d, "scene_pts.txt"), "w").write(
        "10 20 1 12 22 1 1\n30 40 1 99 99 1 0\n50 60 1 52 63 1 1\n")
    name, corrs, mask = ds.load_annotated_pts(os.path.join(d, "scene_pts.txt"))
    assert name == "scene" and corrs.shape == (3, 4)
    assert corrs[0].tolist() == [10, 20, 12, 22]          # x1 y1 x2 y2
    assert mask.tolist() == [True, False, True]
    # essential format: leading count then 'x1 y1 x2 y2' (no labels) + intrinsics
    e = tempfile.mkdtemp()
    open(os.path.join(e, "fountain_pts.txt"), "w").write("2\n10 20 12 22\n30 40 32 41\n")
    K = np.array([[700.0, 0, 320], [0, 700, 240], [0, 0, 1]])
    np.savetxt(os.path.join(e, "fountain1.K"), K); np.savetxt(os.path.join(e, "fountain2.K"), K)
    recs = list(ds.iter_datasets(e, task="essential"))
    assert len(recs) == 1 and len(recs[0]) == 4
    nm, c, m, meta = recs[0]
    assert nm == "fountain" and c.shape == (2, 4) and c[0].tolist() == [10, 20, 12, 22]
    assert not m.any() and meta["dims"] == (640.0, 480.0, 640.0, 480.0)


if __name__ == "__main__":
    test_mat(); test_gt_h_pair(); test_annotated_and_essential()
    print("ALL LOADER TESTS PASSED")

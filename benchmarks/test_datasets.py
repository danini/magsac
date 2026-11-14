"""Tests of datasets.py on small files in both formats, and on the bundled data.

Run: python -m pytest test_datasets.py
"""
import os
import datasets

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def test_both_formats(tmp_path):
    d = str(tmp_path)
    with open(os.path.join(d, "a_pts.txt"), "w") as f:
        f.write("10 20 1 12 22 1 1\n30 40 1 99 99 1 0\n")
    with open(os.path.join(d, "b_pts.txt"), "w") as f:
        f.write("1 2 3 4 0.1 0.1 HessianAffine RootSIFT 0\n"
                "5 6 7 8 0.2 0.2 HessianAffine RootSIFT 1\n")
    (a, corrs_a, mask_a), (b, corrs_b, mask_b) = datasets.iter_scenes(d)
    assert (a, b) == ("a", "b")
    assert corrs_a.tolist() == [[10, 20, 12, 22], [30, 40, 99, 99]]
    assert mask_a.tolist() == [True, False]
    assert corrs_b.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert mask_b.tolist() == [False, True]


def test_bundled_data():
    for directory, scenes in (("fundamental_matrix", 47), ("homography", 16),
                              (os.path.join("homography", "extremeview"), 15)):
        loaded = list(datasets.iter_scenes(os.path.join(DATA, directory)))
        assert len(loaded) == scenes
        assert all(len(c) == len(m) > 0 and m.any() for _, c, m in loaded)

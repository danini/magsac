# Accuracy benchmark for pymagsac

`run_benchmark.py` measures the accuracy of `pymagsac.findFundamentalMatrix` and
`pymagsac.findHomography`, with MAGSAC++ (default) or MAGSAC (`--magsac`), on:

- **synthetic problems** with known ground truth, for several inlier noise levels
  (0.5, 1 and 2 px) and outlier ratios (0 to 70%);
- **the annotated scenes in `data/`**: `data/fundamental_matrix` (47 scenes, of
  which `examples/cpp_example.cpp` uses 36), `data/homography` (the 16 scenes of
  the homogr dataset) and `data/homography/extremeview` (the 15 scenes of EVD).

The synthetic problems are the same on every run, so the results of two builds
(for example before and after a change to the scoring) can be compared with
`--compare`. The estimator's own sampling is random: each setting is the median
of 100 runs (`--runs`). Between two runs of the same build, the medians differed
by up to 0.4 px at 2 px noise or 70% outliers, and by less than 0.05 px elsewhere.

## Usage

Build and install pymagsac first (see the main README), then, from this directory:

```bash
pip install -r requirements.txt
python run_benchmark.py                       # synthetic, about a minute
python run_benchmark.py --quick               # fewer settings, a few seconds
python run_benchmark.py --task homography --data ../data/homography
python run_benchmark.py --task fundamental --data ../data/fundamental_matrix

# Compare two builds:
python run_benchmark.py --out a.csv           # with build A installed
python run_benchmark.py --out b.csv           # with build B installed
python run_benchmark.py --compare a.csv b.csv # B minus A, per setting
# Use the same options for both runs: the CSV files don't record them.
```

## Columns

All errors are in pixels: the RMSE of the model's residual (Sampson distance for
fundamental matrices, transfer error for homographies).

- `err`: RMSE on the ground-truth inliers. On synthetic data, these are the noisy
  inliers; on real data, the correspondences labelled as inliers in
  `<scene>_pts.txt` (`cpp_example.cpp` also adds the inliers of a model fitted to
  them; this benchmark doesn't).
- `gt_err` (synthetic only): RMSE on the same inliers without noise, i.e. the
  distance between the estimated model and the true one.
- `F1` (synthetic only): F1 score of the returned inlier mask.
- `fail`: fraction of runs that returned no model.
- `time_ms`: median run time of the estimator call.

`python -m pytest test_datasets.py` tests the loader of `<scene>_pts.txt` files.

# Accuracy benchmark for pymagsac

`run_benchmark.py` measures the accuracy of `pymagsac.findFundamentalMatrix` and
`pymagsac.findHomography`, with MAGSAC++ (default) or MAGSAC (`--magsac`), on:

- **synthetic problems** with known ground truth, for several inlier noise levels
  (0.5, 1 and 2 px) and outlier ratios (0 to 70%);
- **the annotated scenes in `data/`**, the ones `examples/cpp_example.cpp` uses:
  `data/fundamental_matrix` (47 scenes), `data/homography` (16 scenes of the
  homogr dataset) and `data/homography/extremeview` (15 scenes of EVD).

The synthetic problems are the same on every run, so the results of two builds
(for example before and after a change to the scoring) can be compared with
`--compare`. The estimator's own sampling is random: each setting is the median
of 100 runs (`--runs`), which keeps the run-to-run differences of the median
errors below about 0.3 px.

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
```

## Columns

All errors are in pixels: the RMSE of the model's residual (Sampson distance for
fundamental matrices, transfer error for homographies), as in `cpp_example.cpp`.

- `err`: RMSE on the ground-truth inliers. On synthetic data, these are the noisy
  inliers; on real data, the correspondences labelled as inliers in
  `<scene>_pts.txt`.
- `gt_err` (synthetic only): RMSE on the same inliers without noise, i.e. the
  distance between the estimated model and the true one.
- `F1` (synthetic only): F1 score of the returned inlier mask.
- `fail`: fraction of runs that returned no model.
- `time_ms`: median run time of the estimator call.

`python -m pytest test_datasets.py` tests the loader of `<scene>_pts.txt` files.

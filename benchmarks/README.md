# MAGSAC++ benchmark suite

Self-contained accuracy benchmarks for `pymagsac`, with **synthetic ground truth**
and **real datasets**. Useful to quantify accuracy and to A/B two builds when a
scoring/weight change is made (the model is run on identical data, so harness and
matcher choices cancel out).

Estimators are resolved by name in `pymagsac` **at runtime**, so the same command
runs on the upstream `master` build and on builds that add more estimators
(e.g. `personal/pip-installable`, which adds `findPlane3D`, PnP, …). Unavailable
estimators are detected (`hasattr`) and **skipped with a note**, so you can A/B
the two builds directly. Currently wired:

| `--task` | pymagsac fn | DoF | on master? | real data |
|----------|-------------|:---:|:----------:|-----------|
| `fundamental` | `findFundamentalMatrix` | 4 | yes | AdelaideRMF/kusvod2 |
| `homography`  | `findHomography`        | 2 | yes | homogr/EVD/Oxford |
| `line2d`      | `findLine2D`            | 1 | yes | — (synthetic) |
| `plane3d`     | `findPlane3D`           | 1 | **no** (our addition) | — (synthetic) |
| `essential`   | `findEssentialMatrix`   | 5 | yes | strecha (`data/`, real-only) |

`--task all` runs every estimator the current build provides. Per problem we
report, over the **ground-truth inliers**: `err`, the **RMSE** of the estimated
model's point-to-model residual — exactly the metric the repo's
`examples/cpp_example.cpp` uses (`sqrt(mean(squaredResidual over GT inliers))`);
(`essential` has no ground truth in the repo, so — like `cpp_example` — it reports
only `inliers`, the inlier ratio of the estimated E, plus failure/time.)
the inlier-mask `F1`, the no-model failure fraction (`none`), the median
wall-clock `time_ms`, and `gt_err`, the error of the estimated model vs a **GT
model**: for synthetic problems the synthetic GT (SGD for F, reprojection for H,
normal angle for line/plane); for **real homography** scenes that ship a
`<scene>_vpts.mat` (e.g. magsac's bundled homogr), the reprojection discrepancy of
the estimated H vs the **GT homography** in that file — a robust check independent
of the sparse inlier labels.

### Backends & baselines (`--backend`)
| value | method |
|-------|--------|
| `magsac++` (default) | pymagsac, MAGSAC++ (marginalized scoring) |
| `magsac` | pymagsac, MAGSAC (fixed-σ) — the in-repo baseline |
| `cv2:NAME` | OpenCV `find{FundamentalMat,Homography}` with `method=cv2.NAME` — the paper's RANSAC/LMedS/USAC baselines (`cv2:RANSAC`, `cv2:LMEDS`, `cv2:USAC_MAGSAC`, `cv2:USAC_ACCURATE`, …); needs `opencv-python`; F/H only |
| `py:MODULE` | any **pymagsac-compatible pure-python module** (e.g. `py:garfield_sfm.magsac`), imported via `importlib`; MAGSAC++ mode |

So `--compare` of e.g. `cv2:RANSAC` vs `magsac++` (or `magsac` vs `magsac++`, or
`py:MODULE` vs `magsac++`) on identical data reproduces the paper-style "MAGSAC++
vs baselines" comparison. Pass `--seed N` to make both sides deterministic for a
fair A/B (e.g. comparing a pure-python re-implementation against the C++ binding).

### Threshold sensitivity & sampler showcase
- `--sweep-sigma 0.5,1,2,3,5,8,12,20` — fix a config and vary `sigma_th`; MAGSAC++'s
  `gt_err` stays low across a wide range while fixed-threshold baselines degrade
  (the paper's key plot). Compare a MAGSAC++ sweep CSV against a `cv2:RANSAC` one.
- `--coherent` clusters inliers spatially and `--sampler 2` selects **P-NAPSAC**
  (0=uniform, 1=PROSAC, 2=P-NAPSAC, 3=importance, 4=adaptive); on clustered data
  P-NAPSAC typically lowers `time_ms`. The synthetic sweep also includes hard
  regimes (up to 70% outliers, σ up to 2 px).

## Dependencies
- `numpy` — synthetic benchmarks (always).
- `scipy` — `.mat` dataset loaders (AdelaideRMF / kusvod2).
- `opencv-python` — for the `cv2:*` baseline backends and the image-pair loader
  (SIFT). The pure-pymagsac path and pre-computed loaders need no OpenCV.

## Usage
```bash
python run_benchmark.py --task all --quick                  # every estimator this build has
python run_benchmark.py --task plane3d                      # skipped on master, runs on pip-installable
python run_benchmark.py --task fundamental --backend magsac # MAGSAC baseline (vs default magsac++)
python run_benchmark.py --task fundamental --backend cv2:RANSAC   # OpenCV baseline (needs cv2)
python run_benchmark.py --task homography --sweep-sigma 0.5,1,2,3,5,8,12,20   # threshold sensitivity
python run_benchmark.py --task fundamental --coherent --sampler 2            # P-NAPSAC on clustered inliers
python run_benchmark.py --task homography --data DIR        # real datasets
python run_benchmark.py --task essential --data data/essential_matrix   # bundled essential scene (inlier ratio)
python run_benchmark.py --task essential --data data/essential_matrix   # bundled essential scene (inlier ratio)

# A/B (two builds, or MAGSAC++ vs a baseline) on identical data:
PYTHONPATH=/build_master   python run_benchmark.py --task all --out master.csv
PYTHONPATH=/build_improved python run_benchmark.py --task all --out improved.csv
python run_benchmark.py --compare master.csv improved.csv   # d_err/d_gt_err<0, d_F1>0 = improved better
```

## Real datasets (bundled, or downloaded separately)
`--data DIR` discovers only the datasets relevant to `--task` (so a folder that
mixes F/H/E files won't cross-load). It runs **out-of-the-box on the scenes
already in magsac's `data/`** (47 fundamental, 16 homography, 1 essential), via
`datasets.py`:

| input | files | inliers | source |
|-------|-------|---------|--------|
| repo datasets (F: kusvod2/AdelaideRMF/Multi-H · H: homogr/EVD) | `<scene>_pts.txt` (annotated correspondences + label, cpp_example format) | `label > 0` | bundled in magsac `data/` |
| essential (strecha/fountain) | `<scene>_pts.txt` (count + `x1 y1 x2 y2`, **no labels**) + `<scene>1.K`/`<scene>2.K` | n/a — no GT, reports inlier ratio | bundled in magsac `data/` |
| AdelaideRMF / kusvod2 (fundamental) | `*.mat` (`data` 6×N homogeneous, `label`) | dominant structure | cs.adelaide.edu.au/~hwong · cmp.felk.cvut.cz/wbs |
| homogr / EVD / Oxford (homography) | `<stem>.corr` (`x1 y1 x2 y2`) + `<stem>.H` (3×3) | transfer err < `--gt-thresh` | cmp.felk.cvut.cz/wbs |
| bundled homogr (homography) | `<scene>_pts.txt` + `<scene>_vpts.mat` (GT homography `model`, `imsize`) | `label > 0`; GT H drives `gt_err` + P-NAPSAC dims | magsac `data/homography` |
| image pairs (homography) | `<stem>A.* <stem>B.*` + `<stem>_model.txt` (3×3 H) | transfer err < `--gt-thresh` | (any; needs OpenCV) |

The image-pair path computes SIFT + Lowe-ratio matches (as magsac's own example
notebooks do) and is skipped with a hint if OpenCV is absent. Loaders are
unit-tested in `test_datasets.py` against tiny synthetic fixtures.

## Reproducing the paper (Table 1 + figures)
`reproduce_paper.py` renders the MAGSAC++ (CVPR 2020) **Table 1** layout — median
error / failure% / time per method×dataset — with the **paper's published numbers
as reference rows**, and, where datasets + backends are available, **our
reproduced rows** alongside. It also draws the paper's figures (CDF of errors,
Fig. 3/4; error vs threshold, Fig. 5) plus a runtime-vs-threshold panel, with the
same method legend, and prints the swept error/time numbers. Runs are seeded for
reproducibility.

```bash
python reproduce_paper.py                                   # paper Table 1 only
python reproduce_paper.py --datasets-h DIR --plots out/     # + reproduce homography + figures
python reproduce_paper.py --datasets-f DIR --datasets-h DIR
```bash
python reproduce_paper.py                                   # paper Table 1 only
python reproduce_paper.py --datasets-h data/homography --plots out/   # reproduce homogr+EVD from bundled data
python reproduce_paper.py --datasets-h data/homography --py-impl garfield_sfm.magsac   # + a pure-python MAGSAC++ row
python reproduce_paper.py --datasets-f DIR --datasets-h DIR
```
Method→backend: MAGSAC++→pymagsac, RANSAC/LMedS→`cv2:*`, GC-RANSAC→pygcransac (if
present), MSAC→paper-only. `--datasets-h`/`--datasets-f` accept either a
subdir-per-dataset layout (`homogr/`, `EVD/`, `KITTI/`, …) **or** magsac's bundled
`data/homography` directly (top-level `_pts.txt` = homogr, `extremeview/` = EVD).
`--sigma` defaults to `cpp_example.cpp`'s σ_max per task (homography 50, fundamental
5); MAGSAC marginalises over σ∈[0,σ_max], so the result is threshold-insensitive.
Anything unavailable (a backend, OpenCV, matplotlib, or a dataset) is auto-skipped
and the paper row is still shown.

On the bundled `data/homography` (default σ_max), the H reproduction tracks the
paper closely: MAGSAC++ homogr 1.5 px (paper 1.3) and EVD 5.6 px (paper 12.6),
RANSAC homogr ~1.2–1.9 px (paper 1.1), LMedS EVD ~94 px (paper 89.9 — LMedS fails
on EVD in both). homogr median stays in 1.3–1.7 px across σ_max 3→100 (the paper's
threshold-insensitivity; `--sweep-sigma` / the Fig-5 plot show this directly).
Caveat: `ours` rows use `err` = RMSE of the residual over the GT-labelled inliers
(the metric in the repo's `cpp_example.cpp`), which equals the paper's H metric.
The paper's F column instead uses SGD vs a GT fundamental matrix, which needs
GT-pose datasets (KITTI/TUM/T&T/CPC); the repo's bundled F datasets
(kusvod2/AdelaideRMF/Multi-H) ship labels, not GT F, so F shows paper rows only.

## Files
- `synthetic.py` — GT generators + residual/F1 metrics (no external reference).
- `datasets.py` — real-dataset loaders (pre-computed `.mat`/text + optional cv2 images).
- `run_benchmark.py` — unified runner (`--task`, synthetic / `--data` / `--compare`).
- `reproduce_paper.py` — paper Table 1 (with reference rows) + CDF / threshold figures.
- `test_datasets.py` — loader unit tests.

## A gcransac counterpart
This suite drives `pymagsac`. An analogous benchmark for **graph-cut-ransac**
could be added in that repo driving `pygcransac` (which exposes the same
estimators), reusing the synthetic generators and dataset loaders here; it would
additionally cover gcransac's own scoring path.

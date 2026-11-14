#!/usr/bin/env python3
"""Accuracy benchmark for pymagsac, on synthetic data or on the scenes in data/.

Synthetic data (default): for each task, inlier noise and outlier ratio, the
estimator runs on several random problems; the same problems are generated on
every run, so two builds can be compared on identical data. Columns (medians over
the problems):
  err      RMSE of the estimated model's residual on the noisy inliers (pixels)
  gt_err   the same on the noise-free inliers: the distance to the true model
  F1       F1 score of the returned inlier mask against the true inliers
  fail     fraction of problems where no model was returned
  time_ms  run time of the estimator

Real data (--data DIR, with --task fundamental or homography): every
'<scene>_pts.txt' in DIR: data/fundamental_matrix, data/homography (the homogr
dataset) or data/homography/extremeview (EVD). `err` is then the RMSE over the
manually labelled inliers, and the medians are over runs.

Examples
  python run_benchmark.py --quick
  python run_benchmark.py --task homography --data ../data/homography/extremeview
  python run_benchmark.py --out a.csv            # with one build of pymagsac
  python run_benchmark.py --out b.csv            # with another
  python run_benchmark.py --compare a.csv b.csv  # differences, B minus A
"""
import argparse
import csv
import time
import numpy as np

import datasets
import synthetic as syn

# task -> (pymagsac's estimator, generator, residual)
TASKS = {
    "fundamental": ("findFundamentalMatrix", syn.gen_fundamental, syn.sampson),
    "homography": ("findHomography", syn.gen_homography, syn.transfer),
}
# Maximum sigma used for the scenes of data/ by examples/cpp_example.cpp.
REAL_SIGMA = {"fundamental": 5.0, "homography": 50.0}
METRICS = ["err", "gt_err", "F1", "fail", "time_ms"]


def estimate(task, data, args, sigma):
    """Runs pymagsac; returns (3x3 model or None, inlier mask, time in ms)."""
    import pymagsac  # here, so that --compare works without pymagsac
    estimator = getattr(pymagsac, TASKS[task][0])
    options = dict(sampler=args.sampler, use_magsac_plus_plus=not args.magsac,
                   sigma_th=sigma, conf=0.99, min_iters=50, max_iters=args.max_iters)
    # Image sizes are only used by the P-NAPSAC and AR samplers' grids.
    size = np.ceil(data.max(axis=0)) + 1
    start = time.perf_counter()
    model, mask = estimator(data, *size, np.array([]), **options)
    ms = (time.perf_counter() - start) * 1e3
    return model, mask, ms


def summarize(rows):
    for r in rows:  # a degenerate model can give non-finite residuals: count it as a failure
        if r["err"] is not None and not np.isfinite(r["err"]):
            r.update({k: None for k in ("err", "gt_err", "F1") if k in r})

    def median(key):
        values = [r[key] for r in rows if r[key] is not None]
        return round(float(np.median(values)), 4) if values else float("nan")
    out = {m: median(m) for m in ("err", "gt_err", "F1", "time_ms") if m in rows[0]}
    out["fail"] = round(sum(r["err"] is None for r in rows) / len(rows), 3)
    return out


def run_synthetic(task, args):
    _, generate, residual = TASKS[task]
    sigma = 3.0 if args.sigma is None else args.sigma
    noises = [1.0] if args.quick else [0.5, 1.0, 2.0]
    ratios = [0.0, 0.5] if args.quick else [0.0, 0.3, 0.5, 0.7]
    for noise in noises:
        for ratio in ratios:
            runs = []
            for seed in range(args.runs):
                rng = np.random.default_rng(seed)
                data, clean, true = generate(rng, args.inliers, ratio, noise)
                model, mask, ms = estimate(task, data, args, sigma)
                ok = model is not None
                runs.append({"err": syn.rmse(residual(data[true], model)) if ok else None,
                             "gt_err": syn.rmse(residual(clean, model)) if ok else None,
                             "F1": syn.f1(mask, true) if ok else None, "time_ms": ms})
            yield {"task": task, "noise": noise, "outliers": ratio, **summarize(runs)}


def run_real(task, args):
    sigma = REAL_SIGMA[task] if args.sigma is None else args.sigma
    residual = TASKS[task][2]
    for name, data, true in datasets.iter_scenes(args.data):
        runs = []
        for _ in range(args.runs):
            model, _, ms = estimate(task, data, args, sigma)
            err = syn.rmse(residual(data[true], model)) if model is not None else None
            runs.append({"err": err, "time_ms": ms})
        yield {"task": task, "scene": name, **summarize(runs)}


def print_rows(rows):
    keys = [k for k in rows[0] if k not in METRICS]
    metrics = [m for m in METRICS if m in rows[0]]
    print(" ".join(f"{k:>18}" for k in keys) + " | " + " ".join(f"{m:>8}" for m in metrics))
    for r in rows:
        print(" ".join(f"{r[k]!s:>18}" for k in keys) + " | "
              + " ".join(f"{r[m]:>8}" for m in metrics))


def compare(path_a, path_b):
    def load(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        keys = [k for k in rows[0] if k not in METRICS]
        return keys, {tuple(r[k] for k in keys): r for r in rows}
    keys, a = load(path_a)
    _, b = load(path_b)
    metrics = [m for m in METRICS if m in next(iter(a.values()))]
    print(f"B - A, with A = {path_a} and B = {path_b}")
    if a.keys() ^ b.keys():
        print(f"warning: {len(a.keys() ^ b.keys())} rows are in only one of the files")
    print(" ".join(f"{k:>18}" for k in keys) + " | " + " ".join(f"{m:>8}" for m in metrics))
    for k in sorted(a.keys() & b.keys()):
        diffs = []
        for m in metrics:
            try:
                diffs.append(f"{float(b[k][m]) - float(a[k][m]):>8.3f}")
            except ValueError:
                diffs.append(f"{'':>8}")
        print(" ".join(f"{v:>18}" for v in k) + " | " + " ".join(diffs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--data", help="directory of '<scene>_pts.txt' files (real data)")
    ap.add_argument("--magsac", action="store_true", help="MAGSAC instead of MAGSAC++")
    ap.add_argument("--sigma", type=float,
                    help="maximum sigma (default: 3 on synthetic data; on real data, "
                         "the values of examples/cpp_example.cpp: 5 for F, 50 for H)")
    ap.add_argument("--sampler", type=int, default=0, choices=[0, 1, 2],
                    help="0 uniform, 1 PROSAC, 2 P-NAPSAC")
    ap.add_argument("--max-iters", type=int, default=10000)
    ap.add_argument("--inliers", type=int, default=200, help="inliers per synthetic problem")
    ap.add_argument("--runs", type=int, default=100,
                    help="synthetic problems per setting, or runs per real scene")
    ap.add_argument("--quick", action="store_true", help="fewer synthetic settings")
    ap.add_argument("--out", help="write the rows to this CSV file")
    ap.add_argument("--compare", nargs=2, metavar=("A.csv", "B.csv"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.data:
        if args.task == "all":
            ap.error("--data needs --task fundamental or --task homography")
        rows = list(run_real(args.task, args))
        if not rows:
            ap.error(f"no '<scene>_pts.txt' file in {args.data}")
        print_rows(rows)
        print(f"{len(rows)} scenes, median err {np.nanmedian([r['err'] for r in rows]):.2f} px")
    else:
        tasks = list(TASKS) if args.task == "all" else [args.task]
        rows = [row for task in tasks for row in run_synthetic(task, args)]
        print_rows(rows)
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

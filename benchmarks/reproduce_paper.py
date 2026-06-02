#!/usr/bin/env python3
"""Reproduce the MAGSAC++ (CVPR 2020) results and present them the paper's way.

Renders the paper's Table 1 layout — median geometric error (eps_med), failure
rate (fail%), runtime (t, ms), per method x dataset — with the **paper's published
numbers as reference rows**, and, alongside, **our reproduced rows** computed by
running the corresponding backends (via run_benchmark) on the datasets you point
it at. It also draws the paper's figures: CDF of errors (Fig. 3/4) and avg error
vs inlier-outlier threshold (Fig. 5), using the same method legend.

Method -> backend:
  MAGSAC++  -> pymagsac (magsac++)      RANSAC -> cv2:RANSAC
  MAGSAC    -> pymagsac (magsac)        LMedS  -> cv2:LMEDS
  GC-RANSAC -> pygcransac (if present)  MSAC   -> (no OpenCV equivalent; paper-only)

What is reproducible here depends on availability (auto-skipped, paper row still
shown): pymagsac build, opencv-python (RANSAC/LMedS), pygcransac (GC-RANSAC),
matplotlib (figures), and the datasets themselves (downloaded separately).
Note: for homography our `err` is the reprojection RMSE over GT inliers (the
paper's H metric), so the H columns are directly comparable. For fundamental the
paper uses SGD over a GT F (needs GT-pose datasets KITTI/TUM/T&T/CPC); our
correspondence loaders instead give RMSE over GT inliers (as in cpp_example.cpp),
so the F column shows the paper's numbers as reference-only.

Usage
  python reproduce_paper.py                              # paper Table 1 only
  python reproduce_paper.py --datasets-h data/homography # + reproduce homogr+EVD from the bundled data
  python reproduce_paper.py --datasets-h DIR --datasets-f DIR --plots out/  # subdirs=datasets also work
"""
import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmark as rb

# Paper Table 1: problem -> dataset -> method -> (eps_med, fail%, time_ms)
PAPER = {
    "fundamental": {
        "KITTI": {"MAGSAC++": (3.6, 2.8, 117), "GC-RANSAC": (3.7, 2.3, 111), "RANSAC": (3.8, 2.7, 119), "LMedS": (3.6, 2.7, 111), "MSAC": (3.8, 2.6, 110)},
        "TUM":   {"MAGSAC++": (3.7, 17.7, 18), "GC-RANSAC": (4.1, 25.1, 11), "RANSAC": (5.4, 22.1, 11), "LMedS": (4.3, 23.9, 12), "MSAC": (5.5, 36.2, 11)},
        "T&T":   {"MAGSAC++": (4.2, 0.7, 267), "GC-RANSAC": (4.5, 2.2, 126), "RANSAC": (6.3, 2.6, 133), "LMedS": (4.9, 1.1, 166), "MSAC": (7.0, 2.2, 133)},
        "CPC":   {"MAGSAC++": (17.0, 17.8, 261), "GC-RANSAC": (17.5, 12.1, 144), "RANSAC": (16.9, 29.5, 151), "LMedS": (10.7, 17.8, 187), "MSAC": (16.5, 33.8, 153)},
    },
    "homography": {
        "homogr": {"MAGSAC++": (1.3, 10.8, 32), "GC-RANSAC": (1.1, 10.0, 25), "RANSAC": (1.1, 10.0, 26), "LMedS": (1.5, 12.5, 31), "MSAC": (1.1, 10.0, 24)},
        "EVD":    {"MAGSAC++": (12.6, 12.0, 426), "GC-RANSAC": (12.6, 18.3, 166), "RANSAC": (14.0, 26.1, 168), "LMedS": (89.9, 60.0, 182), "MSAC": (13.2, 23.7, 164)},
    },
}
METHODS = ["MAGSAC++", "GC-RANSAC", "RANSAC", "LMedS", "MSAC"]          # paper legend order
BACKEND = {"MAGSAC++": "magsac++", "RANSAC": "cv2:RANSAC", "LMedS": "cv2:LMEDS",
           "GC-RANSAC": None, "MSAC": None}                             # None = paper-only here
COLORS = {"MAGSAC++": "C3", "GC-RANSAC": "C0", "RANSAC": "C1", "LMedS": "C2", "MSAC": "C4"}
# Default MAGSAC sigma_max per task, as used by the repo's examples/cpp_example.cpp
# ("fairly high maximum threshold"): MAGSAC marginalises over sigma in [0, sigma_max].
SIGMA_MAX = {"homography": 50.0, "fundamental": 5.0}
# Sampler = uniform. The paper / cpp_example use P-NAPSAC (sampler=2), a
# PROSAC-family progressive sampler that needs quality-ordered correspondences.
# Even where that holds (EVD/extremeview `_pts.txt` are SNN-sorted), P-NAPSAC
# measured WORSE than uniform on the bundled data against the GT homography
# (homogr GT-H err 6.5 vs 3.4 px; EVD RMSE 7.9 vs 4.8) -- the bundled demo scenes
# aren't the paper's benchmark regime. So we use uniform. Real image dimensions
# are provided per scene, so `--sampler 2` works correctly on properly-ordered data.
SAMPLER = 0


def _stems(d):  # immediate <scene>_pts.txt stems in dir d (non-recursive)
    return {os.path.basename(p)[:-len("_pts.txt")] for p in glob.glob(os.path.join(d, "*_pts.txt"))}


def _dataset_dirs(task, root):
    """Map each paper dataset name -> (path, members|None). Handles a subdir-per-dataset
    layout (root/homogr/, root/KITTI/, ...) and the repo's bundled data/homography
    (top-level _pts.txt = homogr, extremeview/ = EVD). members filters run_data's
    recursive output to the scenes that belong to this dataset (None = keep all)."""
    subdirs = {d.lower(): os.path.join(root, d)
               for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    alias = {"evd": "extremeview"}
    out, used_root = {}, False
    for d in PAPER[task]:
        key = d.lower()
        if key in subdirs:
            out[d] = (subdirs[key], None)
        elif alias.get(key) in subdirs:
            out[d] = (subdirs[alias[key]], None)
        elif not used_root and _stems(root):                 # top-level files = one dataset
            out[d] = (root, _stems(root)); used_root = True
    return out


def collect(task, root, gt_thresh, sigma):
    """Run each method's backend per paper dataset under `root`.
    Returns ours[dataset][method] = (eps_med, fail%, time_ms) and
    errs[dataset][method] = per-pair error list (for the CDF)."""
    ours, errs, gts = {}, {}, {}
    for name, (path, members) in _dataset_dirs(task, root).items():
        ours[name], errs[name], gts[name] = {}, {}, {}
        for method in METHODS:
            backend = BACKEND.get(method)
            if backend is None:
                continue
            ok, why = rb.backend_ok(task, backend)
            if not ok:
                print(f"  [{task}/{name}] {method}: {why}"); continue
            rows = [r for r in rb.run_data(task, path, gt_thresh, sigma, backend, SAMPLER, 0)
                    if not np.isnan(r["err"]) and (members is None or r["name"] in members)]
            if not rows:
                continue
            e = [r["err"] for r in rows]
            ours[name][method] = (float(np.median(e)),
                                  100.0 * float(np.mean([r["none"] for r in rows])),
                                  float(np.median([r["time_ms"] for r in rows])))
            errs[name][method] = e
            g = [r["gt_err"] for r in rows if not np.isnan(r.get("gt_err", float("nan")))]
            if g:                                            # GT-model (vs _vpts.mat GT homography)
                gts[name][method] = float(np.median(g))
    return ours, errs, gts


def render_gt_h(gts):
    """Print median error vs the GT homography (from `_vpts.mat`), where available.
    This is a robustness cross-check independent of the sparse inlier labels; it is
    NOT the paper's metric (the table's `eps` is reproj-RMSE over GT inliers)."""
    items = [(d, m) for d, m in gts.items() if m]
    if not items:
        return
    print("  GT-model check — median reprojection vs the _vpts.mat GT homography (px):")
    for d, methods in items:
        print(f"    {d}: " + "  ".join(f"{m} {v:.2f}" for m, v in methods.items()))


def render_table(task, ours):
    datasets = list(PAPER[task])
    print(f"\n=== {task.capitalize()} — Table 1 (paper) vs reproduced (ours) ===")
    print(f"{'method':<20}" + "".join(f"| {d:^20} " for d in datasets))
    print(f"{'':<20}" + "".join(f"| {'eps':>5} {'fail%':>6} {'t':>5} " for _ in datasets))
    print("-" * (20 + 23 * len(datasets)))
    for m in METHODS:
        in_paper = any(m in PAPER[task][d] for d in datasets)
        for tag, src in (("paper", PAPER[task]), ("ours", ours)):
            if tag == "paper" and not in_paper:
                continue                       # e.g. the pure-python impl has no paper row
            cells = ""
            any_val = False
            for d in datasets:
                v = src.get(d, {}).get(m) if tag == "ours" else PAPER[task][d].get(m)
                if v:
                    any_val = True
                    cells += f"| {v[0]:>5.1f} {v[1]:>6.1f} {v[2]:>5.0f} "
                else:
                    cells += f"| {'-':>5} {'-':>6} {'-':>5} "
            if tag == "paper" or any_val:
                print(f"{m + ' (' + tag + ')':<20}{cells}")
    print("note: 'ours' err = RMSE of the residual over the GT-labelled inliers "
          "(as in the repo's cpp_example.cpp). The paper's F column uses SGD vs a GT "
          "fundamental matrix, which needs GT-pose datasets (KITTI/TUM/T&T/CPC).")


def plot_cdf(task, errs, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for d, per in errs.items():
        if not per:
            continue
        plt.figure()
        for m in METHODS:
            if m in per and per[m]:
                x = np.sort(per[m]); y = np.arange(1, len(x) + 1) / len(x)
                plt.plot(x, y, label=m, color=COLORS[m])
        plt.xlabel("error"); plt.ylabel("CDF"); plt.title(f"{task} — {d}"); plt.legend()
        p = os.path.join(out, f"cdf_{task}_{d}.png"); plt.savefig(p, dpi=120); plt.close()
        print(f"  wrote {p}")


def plot_threshold(task, root, gt_thresh, out, sigmas=(0.5, 1, 2, 3, 5, 8, 12, 20, 50)):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dd = _dataset_dirs(task, root)
    name, (path, members) = next(iter(dd.items())) if dd else (os.path.basename(root), (root, None))
    err_curves, time_curves = {}, {}
    for m in METHODS:
        b = BACKEND.get(m)
        if b is None or not rb.backend_ok(task, b)[0]:
            continue
        es, ts = [], []
        for sg in sigmas:
            rows = [r for r in rb.run_data(task, path, gt_thresh, sg, b, SAMPLER, 0)
                    if not np.isnan(r["err"]) and (members is None or r["name"] in members)]
            es.append(float(np.median([r["err"] for r in rows])) if rows else np.nan)
            ts.append(float(np.median([r["time_ms"] for r in rows])) if rows else np.nan)
        err_curves[m], time_curves[m] = es, ts
    # Print the swept numbers (for the report), then draw the two Fig-5 panels.
    print(f"\n# {task} threshold sweep on {name} (median over scenes; sampler=uniform, seed=0)")
    print("sigma_max  " + "  ".join(f"{m:>16}" for m in err_curves))
    for j, sg in enumerate(sigmas):
        print(f"{sg:>9}  " + "  ".join(f"{err_curves[m][j]:>6.2f}px/{time_curves[m][j]:>6.1f}ms"
                                       for m in err_curves))
    for ylab, curves, fname, logy in (("median error (px)", err_curves, f"threshold_{task}.png", True),
                                       ("median time (ms)", time_curves, f"time_{task}.png", False)):
        plt.figure()
        for m in curves:
            (plt.semilogy if logy else plt.plot)(sigmas, curves[m], label=m, color=COLORS[m], marker="o")
        plt.xlabel("inlier-outlier threshold sigma_max (px)"); plt.ylabel(ylab); plt.legend()
        plt.title(f"{task} — {ylab.split(' (')[0]} vs threshold, {name}")
        p = os.path.join(out, fname); plt.savefig(p, dpi=120); plt.close(); print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-f", default=None, help="dir of fundamental datasets (subdirs = datasets)")
    ap.add_argument("--datasets-h", default=None, help="homography datasets dir; subdirs=datasets, or "
                    "magsac's bundled data/homography (top-level=homogr, extremeview/=EVD)")
    ap.add_argument("--gt-thresh", type=float, default=3.0)
    ap.add_argument("--sigma", type=float, default=None,
                    help="MAGSAC sigma_max; default per cpp_example (homography 50, fundamental 5)")
    ap.add_argument("--plots", default=None, help="output dir for CDF + threshold PNGs")
    ap.add_argument("--py-impl", default=None, metavar="MODULE",
                    help="also run a pymagsac-compatible pure-python module (e.g. garfield_sfm.magsac) "
                         "as an extra 'MAGSAC++(py)' row")
    a = ap.parse_args()
    if a.py_impl:                              # inject an extra pure-python MAGSAC++ method
        m = "MAGSAC++(py)"; METHODS.append(m); BACKEND[m] = f"py:{a.py_impl}"; COLORS[m] = "C5"
    jobs = [("fundamental", a.datasets_f), ("homography", a.datasets_h)]
    if a.plots:
        os.makedirs(a.plots, exist_ok=True)
    for task, root in jobs:
        sigma = a.sigma if a.sigma is not None else SIGMA_MAX.get(task, 3.0)
        ours, errs, gts = (collect(task, root, a.gt_thresh, sigma) if root and os.path.isdir(root)
                           else ({}, {}, {}))
        render_table(task, ours)
        render_gt_h(gts)
        if a.plots and root:
            try:
                plot_cdf(task, errs, a.plots); plot_threshold(task, root, a.gt_thresh, a.plots)
            except ImportError:
                print("  [plots] matplotlib not installed — `pip install matplotlib` to draw figures.")


if __name__ == "__main__":
    main()

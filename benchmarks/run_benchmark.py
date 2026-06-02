#!/usr/bin/env python3
"""MAGSAC++ benchmark runner (synthetic ground truth + real datasets).

Estimators/backends are resolved at runtime, so the same command runs on the
upstream `master` build and on builds that add estimators (e.g. pip-installable's
findPlane3D). Unavailable ones are detected and skipped.

Per problem we report, over the TRUE inliers, the median residual of the estimated
model (`err`), the inlier-mask `F1`, the no-model failure fraction (`none`), the
median wall-clock `time_ms`, and (synthetic only) `gt_err` — the error of the
estimated model vs the synthetic ground-truth model (SGD for F, reprojection for
H, normal angle for line/plane), mirroring the paper's metrics.

Backends (`--backend`):
  magsac++   pymagsac, use_magsac_plus_plus=True   (default)
  magsac     pymagsac, use_magsac_plus_plus=False  (the MAGSAC baseline)
  cv2:NAME   OpenCV cv2.find{FundamentalMat,Homography} with method=cv2.NAME,
             e.g. cv2:RANSAC, cv2:LMEDS, cv2:USAC_MAGSAC, cv2:USAC_ACCURATE
             (the paper's RANSAC/LMedS/… baselines; needs opencv-python; F/H only)
  py:MODULE  any pymagsac-compatible pure-python module (e.g. py:garfield_sfm.magsac),
             imported via importlib; MAGSAC++ mode. Use --seed for reproducible A/B.

Examples
  python run_benchmark.py --task all --quick
  python run_benchmark.py --task fundamental --backend cv2:RANSAC --out ransac.csv
  python run_benchmark.py --task fundamental --backend magsac++   --out mpp.csv
  python run_benchmark.py --compare ransac.csv mpp.csv
  python run_benchmark.py --task homography --sweep-sigma 0.5,1,2,3,5,8,12,20   # threshold sensitivity
  python run_benchmark.py --task fundamental --coherent --sampler 2             # P-NAPSAC on clustered inliers
  python run_benchmark.py --task fundamental --data DIR
"""
import argparse
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic as syn
import datasets as ds

W = syn.W
METRICS = {"err", "gt_err", "F1", "inliers", "none", "time_ms"}

# task -> (pymagsac fn, generator, residual metric, GT-model metric)
# essential has no synthetic generator (real-data only); its residual/SGD reuse the
# fundamental ones since _run converts the estimated E to a pixel-space F = K2^-T E K1^-1.
TASKS = {
    "fundamental": ("findFundamentalMatrix", syn.gen_fundamental, syn.sampson,      syn.sgd),
    "homography":  ("findHomography",        syn.gen_homography,  syn.transfer_err, syn.h_model_err),
    "line2d":      ("findLine2D",            syn.gen_line2d,      syn.line_err,     syn.normal_angle),
    "plane3d":     ("findPlane3D",           syn.gen_plane3d,     syn.plane_err,    syn.normal_angle),
    "essential":   ("findEssentialMatrix",   None,                syn.sampson,      syn.sgd),
}
DATA_TASKS = {"fundamental", "homography", "essential"}   # have real-dataset loaders
CV2_TASKS = {"fundamental", "homography"}    # OpenCV exposes these (essential needs K -> skip)


def backend_ok(task, backend):
    if backend.startswith("cv2:"):
        if task not in CV2_TASKS:
            return False, f"{task}: OpenCV backend supports only {sorted(CV2_TASKS)}"
        try:
            import cv2
        except ImportError:
            return False, "OpenCV (cv2) not installed — `pip install opencv-python`"
        if not hasattr(cv2, backend[4:]):
            return False, f"cv2 has no method '{backend[4:]}'"
        return True, ""
    if backend.startswith("py:"):
        import importlib
        try:
            _m = importlib.import_module(backend[3:])
        except Exception as e:
            return False, f"module '{backend[3:]}' not importable ({e})"
        fn = TASKS[task][0]
        return (hasattr(_m, fn), f"{task}: '{fn}' not in {backend[3:]}")
    import pymagsac
    fn = TASKS[task][0]
    return (hasattr(pymagsac, fn), f"{task}: '{fn}' not in this pymagsac build")


def _run(task, data, sigma, backend, sampler, meta=None, seed=-1):
    if backend.startswith("cv2:"):
        import cv2
        m = getattr(cv2, backend[4:])
        p1 = np.ascontiguousarray(data[:, :2]); p2 = np.ascontiguousarray(data[:, 2:4])
        if task == "fundamental":
            M, mask = cv2.findFundamentalMat(p1, p2, m, sigma, 0.99)
        else:
            M, mask = cv2.findHomography(p1, p2, m, sigma)
        if M is None or mask is None or M.shape[0] < 3:
            return None, None
        return np.asarray(M[:3], float), np.asarray(mask, bool).ravel()
    if backend.startswith("py:"):                       # pymagsac-compatible pure-python module
        import importlib
        _m = importlib.import_module(backend[3:]); mpp = True
    else:
        import pymagsac as _m
        mpp = backend != "magsac"
    fn = getattr(_m, TASKS[task][0])
    common = dict(sampler=sampler, use_magsac_plus_plus=mpp, sigma_th=sigma,
                  conf=0.99, min_iters=50, max_iters=1000, seed=seed)
    if task == "essential":
        K1, K2 = meta["K1"], meta["K2"]; w1, h1, w2, h2 = meta["dims"]
        E, mask = fn(data, K1, K2, w1, h1, w2, h2, np.array([]), **common)  # K-normalized internally
        E = np.asarray(E, float)
        if E.size < 9:
            return None, None
        F = np.linalg.inv(K2).T @ E.reshape(3, 3) @ np.linalg.inv(K1)  # pixel-space equivalent
        return F, np.asarray(mask, bool)
    if task in ("fundamental", "homography"):
        w1, h1, w2, h2 = meta["dims"] if (meta and meta.get("dims")) else (W, W, W, W)
        M, mask = fn(data, w1, h1, w2, h2, np.array([]), **common); need = 9
    elif task == "line2d":
        M, mask = fn(data, W, W, np.array([]), **common); need = 3
    else:  # plane3d
        M, mask = fn(data, np.array([]), **common); need = 4
    M = np.asarray(M, float)
    if M.size < need:
        return None, None
    return (M.reshape(3, 3) if need == 9 else M), np.asarray(mask, bool)


def _eval(task, data, tmask, model):
    # RMSE of the model's point-to-model residual over the GT inliers (as in the
    # repo's cpp_example.cpp: rmse = sqrt(mean(squaredResidual over GT inliers))).
    r = TASKS[task][2](data[tmask], model)
    return float(np.sqrt(np.mean(np.square(r))))


def _trial(task, gen, n, noise, o, sigma, backend, sampler, coherent, n_seeds):
    errs, gts, f1s, times, none = [], [], [], [], 0
    for s in range(n_seeds):
        data, tmask, gt = gen(n, o, noise, seed=s, coherent=coherent)
        t0 = time.perf_counter()
        model, emask = _run(task, data, sigma, backend, sampler, seed=s)
        times.append((time.perf_counter() - t0) * 1e3)
        if model is None:
            none += 1; continue
        errs.append(_eval(task, data, tmask, model))
        gts.append(float(TASKS[task][3](model, gt)))
        f1s.append(syn.f1(emask, tmask))
    med = lambda a: round(float(np.median(a)), 4) if a else float("nan")
    return {"err": med(errs), "gt_err": med(gts), "F1": med(f1s),
            "none": round(none / n_seeds, 3), "time_ms": round(float(np.median(times)), 2)}


def run_synthetic(task, quick, sigma, backend, sampler, coherent):
    gen = TASKS[task][1]
    ns = [200] if quick else [200, 1000]
    noises = [1.0] if quick else [0.5, 1.0, 2.0]
    outs = [0.0, 0.5] if quick else [0.0, 0.3, 0.5, 0.7]
    n_seeds = 3 if quick else 5
    for n in ns:
        for noise in noises:
            for o in outs:
                row = {"n": n, "noise": noise, "out": o}
                row.update(_trial(task, gen, n, noise, o, sigma, backend, sampler, coherent, n_seeds))
                yield row


def run_sweep_sigma(task, sigmas, backend, sampler, coherent, n=500, noise=1.0, o=0.5, n_seeds=5):
    gen = TASKS[task][1]
    for sg in sigmas:
        row = {"sigma": sg}
        row.update(_trial(task, gen, n, noise, o, sg, backend, sampler, coherent, n_seeds))
        yield row


def run_data(task, root, gt_thresh, sigma, backend, sampler, seed=-1):
    for rec in ds.iter_datasets(root, task, gt_thresh):
        name, data, tmask = rec[0], rec[1], rec[2]
        meta = rec[3] if len(rec) > 3 else None
        t0 = time.perf_counter()
        model, emask = _run(task, data, sigma, backend, sampler, meta, seed)
        dt = (time.perf_counter() - t0) * 1e3
        if task == "essential":          # no GT in the data (like cpp_example): inlier ratio only
            inl = round(float(emask.mean()), 4) if model is not None else float("nan")
            yield {"name": name, "inliers": inl, "none": float(model is None), "time_ms": round(dt, 2)}
            continue
        if model is None or not tmask.any():
            yield {"name": name, "err": float("nan"), "F1": float("nan"),
                   "none": 1.0, "time_ms": round(dt, 2)}; continue
        row = {"name": name, "err": round(_eval(task, data, tmask, model), 4),  # RMSE over GT inliers
               "F1": round(syn.f1(emask, tmask), 4), "none": 0.0, "time_ms": round(dt, 2)}
        if meta and meta.get("gt_model") is not None:           # error vs the GT homography
            w, h = meta["dims"][0], meta["dims"][1]
            row["gt_err"] = round(float(syn.h_model_err(model, meta["gt_model"], W=w, H=h)), 4)
        yield row


def compare(a_path, b_path):
    def load(p):
        with open(p) as f:
            r = csv.DictReader(f); keys = [c for c in r.fieldnames if c not in METRICS]
            return keys, [c for c in r.fieldnames if c in METRICS], \
                {tuple(row[k] for k in keys): row for row in r}
    (ka, ma, A), (_, _, B) = load(a_path), load(b_path)
    print(f"A={a_path}  B={b_path}   (d* = B - A; d_err/d_gt_err<0, d_F1>0 = B better)")
    cols = [m for m in ("err", "gt_err", "F1", "inliers", "none", "time_ms") if m in ma]
    print(f"{'/'.join(ka):>24} | " + " ".join(f"d_{m:>7}" for m in cols))
    print("-" * (26 + 10 * len(cols)))
    for k in sorted(A.keys() & B.keys()):
        a, b = A[k], B[k]
        def d(m):
            try: return f"{float(b[m]) - float(a[m]):>9.3f}"
            except (ValueError, KeyError): return f"{'':>9}"
        print(f"{'/'.join(map(str, k)):>24} | " + " ".join(d(m) for m in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="fundamental")
    ap.add_argument("--backend", default="magsac++", help="magsac++ | magsac | cv2:NAME")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--coherent", action="store_true", help="spatially clustered inliers")
    ap.add_argument("--sampler", type=int, default=0, help="0=uniform 1=PROSAC 2=P-NAPSAC 3=importance 4=AR")
    ap.add_argument("--sweep-sigma", default=None, help="comma list of sigma_th to sweep (threshold sensitivity)")
    ap.add_argument("--data", default=None, help="real-dataset dir (fundamental/homography/essential)")
    ap.add_argument("--gt-thresh", type=float, default=3.0)
    ap.add_argument("--sigma", type=float, default=3.0, help="MAGSAC++ sigma_th / reproj threshold")
    ap.add_argument("--seed", type=int, default=-1, help="estimator seed (-1=random); set for reproducible runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare", nargs=2, metavar=("A.csv", "B.csv"))
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    tasks = list(TASKS) if a.task == "all" else [a.task]
    multi = len(tasks) > 1
    all_rows = []
    for task in tasks:
        ok, why = backend_ok(task, a.backend)
        if not ok:
            print(f"[{a.label}] {why} — skipped."); continue
        if a.data and task not in DATA_TASKS:
            print(f"[{a.label}] {task}: no real-dataset loader — skipped."); continue
        if not a.data and TASKS[task][1] is None:
            print(f"[{a.label}] {task}: real-data only (no synthetic generator) — use --data."); continue
        if a.sweep_sigma:
            sigmas = [float(x) for x in a.sweep_sigma.split(",")]
            rows = list(run_sweep_sigma(task, sigmas, a.backend, a.sampler, a.coherent))
            lead_h, lead = f"{'sigma':>7}", lambda r: f"{r['sigma']:>7}"
        elif a.data:
            rows = list(run_data(task, a.data, a.gt_thresh, a.sigma, a.backend, a.sampler, a.seed))
            lead_h, lead = f"{'name':>20}", lambda r: f"{r['name']:>20}"
        else:
            rows = list(run_synthetic(task, a.quick, a.sigma, a.backend, a.sampler, a.coherent))
            lead_h, lead = (f"{'n':>5} {'noise':>6} {'out':>5}",
                            lambda r: f"{r['n']:>5} {r['noise']:>5}  {int(r['out']*100):>3}%")
        cols = [c for c in ("err", "gt_err", "F1", "inliers", "none", "time_ms") if c in rows[0]]
        print(f"[{a.label}] {a.backend} {task}" + ("  (coherent)" if a.coherent else ""))
        print(f"{lead_h} | " + " ".join(f"{c:>8}" for c in cols))
        print("-" * (len(lead_h) + 3 + 9 * len(cols)))
        for r in rows:
            print(f"{lead(r)} | " + " ".join(f"{r.get(c, ''):>8}" for c in cols))
        if multi:
            for r in rows:
                r["task"] = task
        all_rows += rows
    if a.out and all_rows:
        cols = (["task"] if multi else []) + [c for c in all_rows[0] if c != "task"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(all_rows)
        print(f"\nWrote {a.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Synthetic problems with known ground truth for the MAGSAC++ benchmark.

Each generator returns (data, true_mask, gt_model):
  fundamental : data = corrs Nx4 [x1,y1,x2,y2] (px),  gt = F (3x3)
  homography  : data = corrs Nx4 [x1,y1,x2,y2] (px),  gt = H (3x3)
  plane       : data = pts   Nx3 [x,y,z],            gt = (normal(3), d)

No external reference is needed: the synthetic GT model is the reference, so the
error metrics below are exact. Used by run_benchmark.py for synthetic sweeps and
shipped-vs-fixed A/B.
"""
import math
import numpy as np

W = 1000.0
K = np.array([[1000.0, 0, W / 2], [0, 1000.0, W / 2], [0, 0, 1.0]])
_Kinv = np.linalg.inv(K)


def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def _n_out(n_in, ratio):
    return int(round(n_in * ratio / (1 - ratio))) if 0 < ratio < 1 else 0


def gen_fundamental(n_inliers, outlier_ratio, noise_px, seed=0, coherent=False):
    rng = np.random.default_rng(seed)
    rv = rng.normal(0, 0.15, 3); th = np.linalg.norm(rv) + 1e-9; a = rv / th
    R = (math.cos(th) * np.eye(3) + math.sin(th) * _skew(a) + (1 - math.cos(th)) * np.outer(a, a))
    t = rng.normal(0, 1.0, 3); t[2] = abs(t[2]) + 1.0
    F = _Kinv.T @ _skew(t) @ R @ _Kinv; F /= np.linalg.norm(F)
    h = 0.6 if coherent else 2.0                       # coherent: inliers in a small 3D patch
    c = rng.uniform(-1.4, 1.4, 2) if coherent else np.zeros(2)
    X = np.column_stack([rng.uniform(c[0] - h, c[0] + h, n_inliers),
                         rng.uniform(c[1] - h, c[1] + h, n_inliers), rng.uniform(4, 8, n_inliers)])
    x1 = (K @ X.T).T; x1 = x1[:, :2] / x1[:, 2:]
    X2 = (R @ X.T).T + t; x2 = (K @ X2.T).T; x2 = x2[:, :2] / x2[:, 2:]
    x1 += rng.normal(0, noise_px, x1.shape); x2 += rng.normal(0, noise_px, x2.shape)
    data, mask = _assemble(rng, np.c_[x1, x2], n_inliers, outlier_ratio, W, signed=False)
    return data, mask, F


def gen_homography(n_inliers, outlier_ratio, noise_px, seed=0, coherent=False):
    rng = np.random.default_rng(seed)
    H = np.eye(3) + np.r_[rng.normal(0, 0.1, 6), rng.normal(0, 1e-4, 2), [0]].reshape(3, 3)
    H[0, 2] += rng.normal(0, 30); H[1, 2] += rng.normal(0, 30); H /= H[2, 2]
    win = W / 3 if coherent else W                     # coherent: inliers in a sub-window
    o = rng.uniform(0, W - win, 2) if coherent else np.zeros(2)
    p1 = o + rng.uniform(0, win, (n_inliers, 2))
    p2h = (H @ np.c_[p1, np.ones(n_inliers)].T).T; p2 = p2h[:, :2] / p2h[:, 2:]
    p1 = p1 + rng.normal(0, noise_px, p1.shape); p2 = p2 + rng.normal(0, noise_px, p2.shape)
    data, mask = _assemble(rng, np.c_[p1, p2], n_inliers, outlier_ratio, W, signed=False)
    return data, mask, H


def gen_line2d(n_inliers, outlier_ratio, noise_px, seed=0, coherent=False):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, math.pi); nrm = np.array([math.cos(th), math.sin(th)])  # unit normal
    c = -float(nrm @ (rng.uniform(0, W, 2)))                                     # line: nrm.p + c = 0
    tdir = np.array([-nrm[1], nrm[0]])
    sr = W / 6 if coherent else W / 2                  # coherent: inliers on a short segment
    sc = rng.uniform(-W / 3, W / 3) if coherent else 0.0
    s = sc + rng.uniform(-sr, sr, (n_inliers, 1))
    pts = (-c) * nrm + s * tdir + rng.normal(0, noise_px, (n_inliers, 1)) * nrm
    data, mask = _assemble(rng, pts, n_inliers, outlier_ratio, W, signed=False)
    return data, mask, np.array([nrm[0], nrm[1], c])


def gen_plane3d(n_inliers, outlier_ratio, noise, seed=0, extent=10.0, coherent=False):
    """3D plane fitting (findPlane3D, DoF=1). GT plane = (unit normal, offset d):
    nrm.p + d = 0. Not in upstream master — only our new estimators build."""
    rng = np.random.default_rng(seed)
    nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm); d = rng.uniform(-1, 1)
    b = np.eye(3)[int(np.argmin(np.abs(nrm)))]
    u = np.cross(nrm, b); u /= np.linalg.norm(u); v = np.cross(nrm, u)
    er = extent / 3 if coherent else extent            # coherent: inliers in a patch
    ec = rng.uniform(-extent * 0.6, extent * 0.6, 2) if coherent else np.zeros(2)
    uv = ec + rng.uniform(-er, er, (n_inliers, 2))
    pts = uv[:, :1] * u + uv[:, 1:] * v - d * nrm + rng.normal(0, noise, (n_inliers, 1)) * nrm
    data, mask = _assemble(rng, pts, n_inliers, outlier_ratio, extent, signed=True)
    return data, mask, np.array([nrm[0], nrm[1], nrm[2], d])


def _assemble(rng, inl, n_in, ratio, box, signed):
    n_out = _n_out(n_in, ratio)
    mask = np.r_[np.ones(n_in, bool), np.zeros(n_out, bool)]
    if n_out:
        lo = -box if signed else 0.0
        inl = np.vstack([inl, rng.uniform(lo, box, (n_out, inl.shape[1]))])
    return np.ascontiguousarray(inl, dtype=np.float64), mask


# ---- metrics (estimate vs ground truth / labels) -------------------------------

def sampson(corrs, F):
    x1 = np.c_[corrs[:, :2], np.ones(len(corrs))]; x2 = np.c_[corrs[:, 2:4], np.ones(len(corrs))]
    Fx1 = (F @ x1.T).T; Ftx2 = (F.T @ x2.T).T
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    return np.sqrt(num / den)


def transfer_err(corrs, H):
    x1 = np.c_[corrs[:, :2], np.ones(len(corrs))]
    p = (H @ x1.T).T; p = p[:, :2] / p[:, 2:]
    return np.hypot(*(p - corrs[:, 2:4]).T)


def line_err(pts, line, true_mask=None):
    """Median point-line distance (px) of `line`=(a,b,c) over `pts` (unit normal)."""
    a, b, c = np.asarray(line, float).ravel()[:3]
    nrm = math.hypot(a, b) + 1e-12
    return np.abs(pts[:, 0] * a + pts[:, 1] * b + c) / nrm


def plane_err(pts, plane, true_mask=None):
    """Point-plane distance of `plane`=(a,b,c,d) over 3D `pts` (ax+by+cz+d=0)."""
    v = np.asarray(plane, float).ravel()[:4]
    nrm = np.linalg.norm(v[:3]) + 1e-12
    return np.abs(pts @ v[:3] + v[3]) / nrm


def f1(est, true):
    tp = int(np.sum(est & true)); fp = int(np.sum(est & ~true)); fn = int(np.sum(~est & true))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ---- ground-truth MODEL error (estimate vs the synthetic GT model) -------------
# These take (model_est, model_gt) and need the GT model, so they are only used in
# synthetic runs. They mirror the paper's metrics (F: SGD; H: reprojection).

def _border_pts(W, H, n):
    t = np.linspace(0, 1, n)
    e = np.vstack([np.c_[t * W, np.zeros(n)], np.c_[t * W, np.full(n, H)],
                   np.c_[np.zeros(n), t * H], np.c_[np.full(n, W), t * H]])
    return np.c_[e, np.ones(len(e))]


def sgd(F_est, F_gt, W=W, H=W, n=25):
    """SGD-style symmetric epipolar distance between two F (Zhang): for image-border
    points, take the point on the GT epipolar line nearest the image centre and
    measure its distance to the estimated epipolar line (orientation-robust)."""
    c = np.array([W / 2.0, H / 2.0])

    def side(Fe, Fg, pts):
        lg = (Fg @ pts.T).T; le = (Fe @ pts.T).T
        ng = np.hypot(lg[:, 0], lg[:, 1]) + 1e-12
        lgn = lg / ng[:, None]                                  # normalize: a^2+b^2=1
        sdist = lgn[:, 0] * c[0] + lgn[:, 1] * c[1] + lgn[:, 2]
        q = c[None, :] - sdist[:, None] * lgn[:, :2]            # point on GT line near centre
        qh = np.c_[q, np.ones(len(q))]
        ne = np.hypot(le[:, 0], le[:, 1]) + 1e-12
        return np.abs(np.sum(le * qh, axis=1)) / ne            # dist of that point to est line
    p = _border_pts(W, H, n)
    return float(np.median(np.r_[side(F_est, F_gt, p), side(F_est.T, F_gt.T, p)]))


def h_model_err(H_est, H_gt, W=W, H=W, n=15):
    """Mean reprojection error (px) between H_est and H_gt over a grid of points."""
    gx, gy = np.meshgrid(np.linspace(0, W, n), np.linspace(0, H, n))
    p = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
    a = (H_est @ p.T).T; a = a[:, :2] / a[:, 2:]
    b = (H_gt @ p.T).T; b = b[:, :2] / b[:, 2:]
    return float(np.mean(np.hypot(*(a - b).T)))


def normal_angle(model_est, model_gt):
    """Angle (deg) between the normals of two line(a,b,c) or plane(a,b,c,d) models."""
    k = 2 if len(np.ravel(model_gt)) == 3 else 3
    ne = np.asarray(model_est, float).ravel()[:k]; ng = np.asarray(model_gt, float).ravel()[:k]
    ne = ne / (np.linalg.norm(ne) + 1e-12); ng = ng / (np.linalg.norm(ng) + 1e-12)
    return math.degrees(math.acos(min(1.0, abs(float(ne @ ng)))))



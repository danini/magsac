#!/usr/bin/env python3
"""Real-dataset loaders for the MAGSAC++ benchmark.

Each loader yields (name, corrs, true_mask): corrs is float64 Nx4 [x1,y1,x2,y2]
in pixels, true_mask is a bool array of ground-truth inliers. Datasets are
downloaded separately (not vendored). Two input regimes are supported:

PRE-COMPUTED correspondences (no feature matching, no OpenCV):
  * '<scene>_pts.txt' — the repo's own cpp_example format: annotated correspondences
        'x1 y1 s x2 y2 s label' (extremeview/EVD: 'x1 y1 x2 y2 s s tag tag label'),
        label>0 = GT inlier. Used for kusvod2 / AdelaideRMF / Multi-H (F) and
        homogr / EVD (H). Bundled under magsac's data/ directory.
  * AdelaideRMF / kusvod2 — '*.mat' with
        data  : 6xN homogeneous [x1;y1;1;x2;y2;1]
        label : 1xN (0 = outlier, k>=1 = structure)  -> dominant structure = inliers
    AdelaideRMF: cs.adelaide.edu.au/~hwong   kusvod2: cmp.felk.cvut.cz/wbs/
  * '<stem>.corr' (rows "x1 y1 x2 y2") + '<stem>.H' (3x3 GT homography)
    -> inliers = symmetric transfer error < gt_thresh px (homogr / EVD / Oxford).

IMAGE PAIRS (features computed with OpenCV SIFT — needs `pip install opencv-python`):
  * '<stem>A.<ext>' + '<stem>B.<ext>' + '<stem>_model.txt' (3x3 GT H).
    SIFT + Lowe ratio test build the correspondences; inliers from the GT H.
    This mirrors magsac's example notebooks (which also use cv2 SIFT + a 3x3 H).
"""
import os
import glob
import functools
import numpy as np


@functools.lru_cache(maxsize=None)
def _image_dims(stem):
    """(w1,h1,w2,h2) in pixels from '<stem>A.*'/'<stem>B.*' images (for the
    P-NAPSAC grid, matching cpp_example which reads image.cols/rows). None if the
    images or OpenCV are unavailable. cv2 shape is (rows=h, cols=w)."""
    try:
        import cv2
    except ImportError:
        return None
    a = sorted(glob.glob(stem + "A.*")); b = sorted(glob.glob(stem + "B.*"))
    if not a or not b:
        return None
    ia, ib = cv2.imread(a[0]), cv2.imread(b[0])
    if ia is None or ib is None:
        return None
    return (float(ia.shape[1]), float(ia.shape[0]), float(ib.shape[1]), float(ib.shape[0]))


def load_mat(mat_path):
    """AdelaideRMF / kusvod2 .mat with pre-computed correspondences + labels.
    Returns None if the file isn't this format (e.g. *_vpts.mat with other keys)."""
    from scipy.io import loadmat
    m = loadmat(mat_path)
    if "data" not in m or "label" not in m:
        return None
    data = np.asarray(m["data"], dtype=np.float64)
    label = np.asarray(m["label"]).ravel().astype(int)
    corrs = data[[0, 1, 3, 4], :].T
    nz = label[label != 0]
    mask = (label == np.bincount(nz).argmax()) if nz.size else np.zeros(len(corrs), bool)
    return os.path.splitext(os.path.basename(mat_path))[0], np.ascontiguousarray(corrs), mask


def _h_inliers(corrs, H, gt_thresh):
    x1 = np.c_[corrs[:, :2], np.ones(len(corrs))]; x2 = np.c_[corrs[:, 2:4], np.ones(len(corrs))]
    p12 = (H @ x1.T).T; p12 = p12[:, :2] / p12[:, 2:]
    p21 = (np.linalg.inv(H) @ x2.T).T; p21 = p21[:, :2] / p21[:, 2:]
    return (np.hypot(*(p12 - corrs[:, 2:4]).T) + np.hypot(*(p21 - corrs[:, :2]).T)) < gt_thresh


def load_gt_h_pair(corr_path, h_path, gt_thresh=3.0):
    corrs = np.loadtxt(corr_path, dtype=np.float64).reshape(-1, 4)
    H = np.loadtxt(h_path, dtype=np.float64).reshape(3, 3)
    return os.path.splitext(os.path.basename(corr_path))[0], np.ascontiguousarray(corrs), \
        _h_inliers(corrs, H, gt_thresh)


def load_image_pair(img1_path, img2_path, h_path, gt_thresh=3.0, snn=0.85):
    """Image pair + 3x3 GT H -> SIFT/ratio-test correspondences + GT inliers.
    Requires OpenCV (cv2); raises ImportError otherwise."""
    import cv2
    H = np.loadtxt(h_path, dtype=np.float64).reshape(3, 3)
    i1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE); i2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    det = cv2.SIFT_create()
    k1, d1 = det.detectAndCompute(i1, None); k2, d2 = det.detectAndCompute(i2, None)
    pairs = []
    for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2):
        if m.distance < snn * n.distance:
            pairs.append((k1[m.queryIdx].pt, k2[m.trainIdx].pt))
    corrs = np.ascontiguousarray([[a[0], a[1], b[0], b[1]] for a, b in pairs], dtype=np.float64)
    name = os.path.commonprefix([os.path.basename(img1_path), os.path.basename(img2_path)]) or "pair"
    return name, corrs, _h_inliers(corrs, H, gt_thresh)


def load_annotated_pts(path):
    """Repo cpp_example.cpp '<scene>_pts.txt': annotated correspondences + inlier labels.
    Default rows are 'x1 y1 s x2 y2 s label'; extremeview/EVD rows are
    'x1 y1 x2 y2 s s tag tag label'. label>0 => GT inlier (getSubsetFromLabeling(.,1))."""
    ev = "extremeview" in path.lower()  # EVD files live in an extremeview/ dir (not the basename)
    corrs, labels = [], []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if len(t) < (9 if ev else 7):
                continue
            c = (t[0], t[1], t[2], t[3], t[8]) if ev else (t[0], t[1], t[3], t[4], t[6])
            corrs.append([float(c[0]), float(c[1]), float(c[2]), float(c[3])])
            labels.append(float(c[4]))
    name = os.path.basename(path)[:-len("_pts.txt")] or "pts"
    return name, np.ascontiguousarray(corrs, dtype=np.float64), np.array(labels) > 0


def load_essential_pts(pts_path, k1_path, k2_path):
    """Essential scene: '<scene>_pts.txt' in the repo's readPoints<4> format — an
    optional leading count then rows 'x1 y1 x2 y2' (NO inlier labels) — plus 3x3
    intrinsics '<scene>1.K','<scene>2.K'. There is no ground truth for essential in
    the repo (cpp_example only counts inliers), so the returned mask is all-False;
    the 4th meta element carries {K1,K2,dims} (image size proxied from each K)."""
    toks = open(pts_path).read().split()
    off = 1 if (toks and toks[0].lstrip("-").isdigit() and len(toks) == 1 + 4 * int(toks[0])) else 0
    corrs = np.array(toks[off:], dtype=np.float64).reshape(-1, 4)
    K1 = np.loadtxt(k1_path, dtype=np.float64).reshape(3, 3)
    K2 = np.loadtxt(k2_path, dtype=np.float64).reshape(3, 3)
    dims = (2 * K1[0, 2], 2 * K1[1, 2], 2 * K2[0, 2], 2 * K2[1, 2])
    name = os.path.basename(pts_path)[:-len("_pts.txt")] or "pts"
    return name, np.ascontiguousarray(corrs), np.zeros(len(corrs), bool), \
        {"K1": K1, "K2": K2, "dims": dims}


def _load_vpts(stem):
    """GT homography (image1->image2) + (w,h) from '<stem>_vpts.mat', or None.
    The .mat `validation.model` maps image2->image1 (verified 0px on its points),
    so we invert it; `imsize` is [h,w]. Used as the homography GT reference."""
    p = stem + "_vpts.mat"
    if not os.path.exists(p):
        return None
    try:
        from scipy.io import loadmat
        v = loadmat(p)["validation"][0, 0]
        h21 = np.asarray(v["model"], dtype=np.float64)
        h_, w_ = (float(x) for x in np.asarray(v["imsize"]).ravel()[:2])
        return np.linalg.inv(h21), (w_, h_)
    except Exception:
        return None


def iter_datasets(root, task="fundamental", gt_thresh=3.0):
    """Discover only the datasets appropriate for `task` under root:
       essential   : '<scene>_pts.txt' + '<scene>1.K'/'<scene>2.K' (or '<scene>.K')
       fundamental : '<scene>_pts.txt', AdelaideRMF/kusvod2 '*.mat'
       homography  : '<scene>_pts.txt', '<stem>.corr'+'<stem>.H', '<stem>A/B'+'<stem>_model.txt'."""
    if not root or not os.path.isdir(root):
        print(f"[datasets] '{root}' not found — see README for layout + URLs."); return
    pts_files = sorted(glob.glob(os.path.join(root, "**", "*_pts.txt"), recursive=True))
    found = False
    if task == "essential":
        for pts in pts_files:
            stem = pts[:-len("_pts.txt")]
            k1, k2 = stem + "1.K", stem + "2.K"
            if not (os.path.exists(k1) and os.path.exists(k2)):
                k1 = k2 = stem + ".K"
            if os.path.exists(k1) and os.path.exists(k2):
                found = True
                try: yield load_essential_pts(pts, k1, k2)
                except Exception as e: print(f"[datasets] skip {pts}: {e}")
        if not found:
            print(f"[datasets] no essential datasets (need <scene>_pts.txt + .K) under '{root}'.")
        return
    for pts in pts_files:                                        # generic F/H annotated points
        found = True
        try:
            rec = load_annotated_pts(pts)                        # (name, corrs, mask)
            stem = pts[: -len("_pts.txt")]
            meta = {}
            vp = _load_vpts(stem) if task == "homography" else None
            if vp:                                               # GT homography + imsize (homogr)
                meta["gt_model"], (w, h) = vp
                meta["dims"] = (w, h, w, h)
            else:
                dims = _image_dims(stem)                         # P-NAPSAC grid dims, else None
                if dims:
                    meta["dims"] = dims
            yield (*rec, meta) if meta else rec
        except Exception as e: print(f"[datasets] skip {pts}: {e}")
    seen = {os.path.basename(p)[:-len("_pts.txt")] for p in pts_files}
    if task == "fundamental":
        for mat in sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
            try:
                r = load_mat(mat)
                if r: found = True; yield r
            except Exception as e: print(f"[datasets] skip {mat}: {e}")
    if task == "homography":
        for corr in sorted(glob.glob(os.path.join(root, "**", "*.corr"), recursive=True)):
            h = os.path.splitext(corr)[0] + ".H"
            if os.path.exists(h):
                found = True
                try: yield load_gt_h_pair(corr, h, gt_thresh)
                except Exception as e: print(f"[datasets] skip {corr}: {e}")
        for model in sorted(glob.glob(os.path.join(root, "**", "*_model.txt"), recursive=True)):
            stem = model[: -len("_model.txt")]
            if os.path.basename(stem) in seen:          # annotated GT already used for this scene
                continue
            imgs = sorted(glob.glob(stem + "A.*") + glob.glob(stem + "B.*"))
            if len(imgs) >= 2:
                found = True
                try: yield load_image_pair(imgs[0], imgs[1], model, gt_thresh)
                except ImportError:
                    print(f"[datasets] {os.path.basename(stem)}: image pair needs OpenCV "
                          "(pip install opencv-python) — skipped."); break
                except Exception as e: print(f"[datasets] skip {stem}: {e}")
    if not found:
        print(f"[datasets] no {task} datasets under '{root}'.")

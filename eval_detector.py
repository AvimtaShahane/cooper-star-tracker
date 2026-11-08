"""
eval_detector.py -- Does the learned detector actually beat thresholding?

Trained loss going down proves the network learned SOMETHING. It does not prove
the something is useful. This measures the thing that matters to the system:

  RECALL     what fraction of real stars did it find?
  PRECISION  what fraction of its detections were real?
  ACCURACY   how close are the sub-pixel centroids to truth?

against the classical sigma-threshold detector, across the noise sweep. The
renderer knows every true star position exactly, so this is a clean comparison
with no hand-labelling anywhere.

Usage:
    python eval_detector.py                      # cpu
    python eval_detector.py --device mps --trials 40
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from classical import detect_centroids
from model import StarDetector
from render import (CameraModel, NoiseConfig, StarCatalog, StarFieldRenderer,
                    random_quaternion)

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
MATCH_RADIUS_PX = 2.0


def score(found: np.ndarray, truth: np.ndarray, radius=MATCH_RADIUS_PX):
    """Greedy nearest-neighbour matching -> (n_matched, mean px error).

    Each true star can be claimed once, so duplicate detections on one star
    count as false positives rather than inflating recall.
    """
    if len(found) == 0 or len(truth) == 0:
        return 0, float("nan")
    d = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)
    used, errs = set(), []
    for i in np.argsort(d.min(axis=1)):          # closest detections first
        j = int(np.argmin(d[i]))
        if j in used or d[i, j] > radius:
            continue
        used.add(j)
        errs.append(d[i, j])
    return len(errs), (float(np.mean(errs)) if errs else float("nan"))


def run(trials=30, device="cpu", weights="detector.pt", threshold=0.5):
    cat = StarCatalog.build()
    cam = CameraModel()
    rnd = StarFieldRenderer(cat, cam)

    net = StarDetector().to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"loaded {weights} on {device}\n")

    header = (f"{'noise':>6} {'method':>10} {'recall':>8} {'precision':>10} "
              f"{'centroid_err':>13} {'found/true':>12}")
    print(header)
    print("-" * len(header))

    summary = {}
    for level in LEVELS:
        cfg = NoiseConfig.at_level(level)
        rng = np.random.default_rng(int(level * 1000))
        acc = {m: {"tp": 0, "n_found": 0, "n_true": 0, "errs": []}
               for m in ("classical", "learned")}

        for _ in range(trials):
            q = random_quaternion(rng)
            img, truth = rnd.render(q, rng=rng, noise=cfg, return_truth=True)
            true_uv = truth["uv"]
            if len(true_uv) == 0:
                continue

            c_found, _ = detect_centroids(img, max_stars=120)
            l_found, _ = net.detect(img, threshold=threshold, device=device,
                                    max_stars=120)

            for name, found in (("classical", c_found), ("learned", l_found)):
                tp, err = score(found, true_uv)
                a = acc[name]
                a["tp"] += tp
                a["n_found"] += len(found)
                a["n_true"] += len(true_uv)
                if np.isfinite(err):
                    a["errs"].append(err)

        summary[level] = {}
        for name in ("classical", "learned"):
            a = acc[name]
            recall = a["tp"] / max(1, a["n_true"])
            precision = a["tp"] / max(1, a["n_found"])
            cerr = float(np.mean(a["errs"])) if a["errs"] else float("nan")
            summary[level][name] = (recall, precision, cerr)
            print(f"{level:6.2f} {name:>10} {recall*100:7.1f}% "
                  f"{precision*100:9.1f}% {cerr:12.3f}px "
                  f"{a['n_found']/trials:6.1f}/{a['n_true']/trials:.1f}")
        print()

    print("=" * 62)
    print("RECALL: learned minus classical, by noise level")
    for level in LEVELS:
        c = summary[level]["classical"][0]
        l = summary[level]["learned"][0]
        delta = (l - c) * 100
        arrow = "better" if delta > 1 else ("worse" if delta < -1 else "same")
        print(f"  noise {level:.2f}:  classical {c*100:5.1f}%   "
              f"learned {l*100:5.1f}%   -> {delta:+5.1f} pts  ({arrow})")
    print("\nMore stars found means more catalog matches, which means a better")
    print("attitude solve. That is how this feeds the rest of the system.")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--weights", default="detector.pt")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    run(a.trials, a.device, a.weights, a.threshold)

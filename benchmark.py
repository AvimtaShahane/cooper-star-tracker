"""
benchmark.py -- The evidence. This produces the plot your pitch is built on.

Sweeps sensor degradation from clean to severe and measures, for each method:
  * accuracy   median angular error, in degrees, over solved frames
  * robustness fraction of frames solved to better than 1 degree
  * latency    milliseconds per frame

Two results tend to come out of this, and both are worth saying plainly:

  1. The classical pipeline is MORE ACCURATE where it works. Sub-arcminute.
     Do not pretend otherwise -- a judge who knows the field will catch it,
     and the honest version is a better story anyway.

  2. The classical pipeline DEGRADES IN LATENCY, not just accuracy. As noise
     rises it must try more triangle hypotheses before one verifies, so solve
     time climbs by an order of magnitude and eventually it fails outright.
     The network is constant-time and always answers.

That is the real finding: they fail differently, so the right system uses the
network as a fast always-available prior and the classical solver for
precision when it can verify. Run with --hybrid to measure that combination.

Usage
-----
    python benchmark.py --trials 30                    # classical only
    python benchmark.py --trials 30 --model attitude_net.pt
    python benchmark.py --trials 30 --model attitude_net.pt --hybrid
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from classical import ClassicalTracker, PairDatabase
from render import (CameraModel, NoiseConfig, StarCatalog, StarFieldRenderer,
                    angular_error_deg, random_quaternion)

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def bootstrap_ci(values, n_boot=2000, pct=95, seed=0):
    """Confidence interval on the median by bootstrap.

    Error distributions here are heavily skewed -- a few total failures sit
    next to many sub-arcminute solves -- so a mean +/- std would be misleading.
    Report the median with a bootstrap interval instead.
    """
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    lo, hi = (100 - pct) / 2, 100 - (100 - pct) / 2
    return float(np.percentile(meds, lo)), float(np.percentile(meds, hi))


def load_model(path, device="cpu"):
    """Load a trained AttitudeNet, or return None if unavailable."""
    if not path:
        return None, device
    try:
        import torch
        from model import AttitudeNet
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        net = AttitudeNet(pretrained=False).to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        print(f"loaded model from {path} (device={device})")
        return net, device
    except Exception as exc:
        print(f"could not load model ({exc}); benchmarking classical only")
        return None, device


def run(trials=30, levels=LEVELS, model_path=None, hybrid=False, seed=0,
        device="cpu", out_json="benchmark.json", out_png="benchmark.png"):
    cat = StarCatalog.build()
    cam = CameraModel()
    rnd = StarFieldRenderer(cat, cam)
    tracker = ClassicalTracker(cat, cam, PairDatabase(cat))
    net, device = load_model(model_path, device)

    methods = ["classical"] + (["learned"] if net else []) + \
              (["hybrid"] if net and hybrid else [])
    results = {m: {"level": [], "median": [], "ci_lo": [], "ci_hi": [],
                   "frac_sub1": [], "frac_sub01": [], "solve_rate": [],
                   "ms": []} for m in methods}

    print(f"\n{'level':>6} {'method':>10} {'median_deg':>12} {'<1deg':>8} "
          f"{'solved':>8} {'ms/frame':>10}")
    print("-" * 60)

    for level in levels:
        cfg = NoiseConfig.at_level(level)
        rng = np.random.default_rng(seed + int(level * 1000))
        per = {m: {"err": [], "ms": [], "solved": 0} for m in methods}

        for _ in range(trials):
            q_true = random_quaternion(rng)
            img = rnd.render(q_true, rng=rng, noise=cfg)

            t0 = time.perf_counter()
            res = tracker.solve(img)
            dt_c = (time.perf_counter() - t0) * 1000
            per["classical"]["ms"].append(dt_c)
            if res["quat"] is not None:
                per["classical"]["solved"] += 1
                per["classical"]["err"].append(angular_error_deg(q_true, res["quat"]))
            else:
                per["classical"]["err"].append(np.nan)

            if net is not None:
                t0 = time.perf_counter()
                q_pred, sigma = net.predict(img, device=device)
                dt_l = (time.perf_counter() - t0) * 1000
                per["learned"]["ms"].append(dt_l)
                per["learned"]["solved"] += 1  # always answers
                per["learned"]["err"].append(angular_error_deg(q_true, q_pred))

                if hybrid:
                    # Prefer the classical solve when it verifies; fall back to
                    # the network when it does not. This is the system you
                    # would actually fly.
                    use = res["quat"] if res["quat"] is not None else q_pred
                    per["hybrid"]["ms"].append(dt_c + dt_l)
                    per["hybrid"]["solved"] += 1
                    per["hybrid"]["err"].append(angular_error_deg(q_true, use))

        for m in methods:
            e = np.array(per[m]["err"], dtype=float)
            finite = e[np.isfinite(e)]
            lo, hi = bootstrap_ci(finite)
            r = results[m]
            r["level"].append(level)
            r["median"].append(float(np.median(finite)) if len(finite) else float("nan"))
            r["ci_lo"].append(lo)
            r["ci_hi"].append(hi)
            # Counted over ALL trials, so an unsolved frame counts as a failure.
            r["frac_sub1"].append(float(np.nansum(e < 1.0) / trials))
            r["frac_sub01"].append(float(np.nansum(e < 0.1) / trials))
            r["solve_rate"].append(per[m]["solved"] / trials)
            r["ms"].append(float(np.mean(per[m]["ms"])))
            print(f"{level:6.1f} {m:>10} {r['median'][-1]:12.4f} "
                  f"{r['frac_sub1'][-1]*100:7.0f}% {r['solve_rate'][-1]*100:7.0f}% "
                  f"{r['ms'][-1]:10.1f}")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")
    try:
        make_plot(results, out_png)
        print(f"wrote {out_png}")
    except Exception as exc:
        print(f"plot skipped ({exc})")
    return results


def make_plot(results, path="benchmark.png"):
    """Three panels: accuracy, robustness, latency. This is the pitch slide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"classical": "#d9534f", "learned": "#0275d8", "hybrid": "#5cb85c"}

    for m, r in results.items():
        c = colors.get(m, "#777777")
        x = r["level"]
        axes[0].plot(x, r["median"], "o-", color=c, label=m, lw=2)
        axes[0].fill_between(x, r["ci_lo"], r["ci_hi"], color=c, alpha=0.18)
        axes[1].plot(x, np.array(r["frac_sub1"]) * 100, "o-", color=c,
                     label=m, lw=2)
        axes[2].plot(x, r["ms"], "o-", color=c, label=m, lw=2)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("median attitude error (deg)")
    axes[0].set_title("Accuracy (95% CI)")
    axes[1].set_ylabel("frames solved to <1 deg (%)")
    axes[1].set_title("Robustness")
    axes[1].set_ylim(-5, 105)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("ms per frame")
    axes[2].set_title("Latency")

    for ax in axes:
        ax.set_xlabel("sensor degradation level")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Star tracker: learned vs classical attitude determination",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--model", default=None, help="path to attitude_net.pt")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-json", default="benchmark.json")
    p.add_argument("--out-png", default="benchmark.png")
    a = p.parse_args()
    run(a.trials, LEVELS, a.model, a.hybrid, device=a.device,
        out_json=a.out_json, out_png=a.out_png)

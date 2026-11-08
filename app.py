"""
app.py -- The demo. Run with:  streamlit run app.py

Four tabs, in the order you should walk a judge through them:

  LIVE       one frame, both pipelines, with the failure sliders exposed.
             HAND THE LAPTOP OVER AND LET THEM DRAG THE SLIDERS. An
             interactive failure demo cannot be accused of cherry-picking,
             and it beats any static chart you could show.
  TRACKING   the EKF coasting through blacked-out frames.
  BENCHMARK  the accuracy / robustness / latency sweep.
  ABOUT      how it works, for the judge who asks.

Everything is synthetic and runs offline. No camera, no network, no hardware.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from classical import ClassicalTracker, PairDatabase, detect_centroids
from render import (CameraModel, NoiseConfig, StarCatalog, StarFieldRenderer,
                    angular_error_deg, quat_multiply, quat_normalize,
                    quat_to_matrix, random_quaternion)
from tracking import run_sequence, small_angle_quat

st.set_page_config(page_title="Star Tracker", layout="wide",
                   initial_sidebar_state="expanded")


# ----------------------------------------------------------------------------
# Cached heavy objects
# ----------------------------------------------------------------------------

@st.cache_resource
def get_system():
    cat = StarCatalog.build()
    cam = CameraModel()
    rnd = StarFieldRenderer(cat, cam)
    tracker = ClassicalTracker(cat, cam, PairDatabase(cat))
    return cat, cam, rnd, tracker


@st.cache_resource
def get_model(path="attitude_net.pt"):
    """Load the trained network if it exists. The app works fine without it."""
    try:
        import torch
        from model import AttitudeNet
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = AttitudeNet(pretrained=False)
        net.load_state_dict(ckpt["model"])
        net.eval()
        return net
    except Exception:
        return None


cat, cam, renderer, tracker = get_system()
net = get_model()


def overlay(img, centroids=None, predicted=None):
    """Grayscale frame -> RGB with detection and reprojection markers."""
    import cv2
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_NEAREST)
    if centroids is not None:
        for u, v in centroids:
            cv2.circle(rgb, (int(u * 2), int(v * 2)), 7, (0, 220, 255), 1)
    if predicted is not None:
        for u, v in predicted:
            cv2.drawMarker(rgb, (int(u * 2), int(v * 2)), (120, 255, 120),
                           cv2.MARKER_CROSS, 9, 1)
    return rgb


# ----------------------------------------------------------------------------
# Sidebar: the failure injection controls
# ----------------------------------------------------------------------------

st.sidebar.title("Sensor conditions")
st.sidebar.caption("Drag these. Watch what breaks and what holds.")

level = st.sidebar.slider("Overall degradation", 0.0, 1.0, 0.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.caption("Or override individual effects:")
manual = st.sidebar.checkbox("Manual control", False)

if manual:
    read_noise = st.sidebar.slider("Read noise (sigma)", 0.0, 25.0, 4.0, 0.5)
    blur = st.sidebar.slider("Motion blur (px)", 0.0, 5.0, 0.0, 0.1)
    dropout = st.sidebar.slider("Star dropout", 0.0, 0.8, 0.0, 0.05)
    hot = st.sidebar.slider("Hot pixel rate (1e-4)", 0.0, 20.0, 0.0, 0.5)
    false_stars = st.sidebar.slider("False detections", 0, 25, 0)
    maglim = st.sidebar.slider("Limiting magnitude", 4.0, 6.5, 6.5, 0.1)
    cfg = NoiseConfig(psf_sigma=(0.9, 1.3), read_noise=(read_noise, read_noise),
                      background=(4.0, 8.0), hot_pixel_rate=(hot * 1e-4, hot * 1e-4),
                      dropout_prob=(dropout, dropout),
                      false_star_count=(false_stars, false_stars),
                      motion_blur_px=(blur, blur), mag_limit=(maglim, maglim),
                      gain=(1.0, 1.0), shot_noise=True)
else:
    cfg = NoiseConfig.at_level(level)

st.sidebar.markdown("---")
if net is None:
    st.sidebar.warning("No trained model found. Classical pipeline only.\n\n"
                       "Train one with `python train.py --task attitude`.")
else:
    st.sidebar.success("Trained attitude network loaded.")


# ----------------------------------------------------------------------------
tab_live, tab_track, tab_bench, tab_about = st.tabs(
    ["Live", "Tracking", "Benchmark", "About"])


# ----------------------------------------------------------------------------
# LIVE
# ----------------------------------------------------------------------------

with tab_live:
    st.subheader("Single-frame attitude determination")
    c1, c2 = st.columns([1, 1])
    if "seed" not in st.session_state:
        st.session_state.seed = 7
    if c1.button("New random attitude", use_container_width=True):
        st.session_state.seed = int(np.random.randint(0, 10 ** 6))

    rng = np.random.default_rng(st.session_state.seed)
    q_true = random_quaternion(rng)
    img = renderer.render(q_true, rng=np.random.default_rng(st.session_state.seed + 1),
                          noise=cfg)

    t0 = time.perf_counter()
    res = tracker.solve(img)
    ms_classical = (time.perf_counter() - t0) * 1000

    cents = res.get("centroids")
    pred_uv = None
    if res["quat"] is not None:
        uv, vis = cam.project(cat.vectors @ quat_to_matrix(res["quat"]).T)
        pred_uv = uv[vis]

    left, right = st.columns([1, 1])
    with left:
        st.image(overlay(img, cents, pred_uv), caption=(
            "circles = detected centroids, crosses = catalog stars "
            "reprojected under the solved attitude"), use_container_width=True)
    with right:
        st.markdown("**Classical pipeline**")
        if res["quat"] is None:
            st.error(f"FAILED to solve  ({res['n_detected']} stars detected, "
                     f"{ms_classical:.0f} ms spent searching)")
        else:
            err = angular_error_deg(q_true, res["quat"])
            a, b, c = st.columns(3)
            a.metric("Attitude error", f"{err*3600:.0f} arcsec" if err < 0.5
                     else f"{err:.2f} deg")
            b.metric("Stars matched", f"{res['n_matched']}/{res['n_detected']}")
            c.metric("Solve time", f"{ms_classical:.0f} ms")
            st.caption(f"mean reprojection residual "
                       f"{res['residual_deg']*3600:.1f} arcsec")

        st.markdown("---")
        st.markdown("**Learned pipeline**")
        if net is None:
            st.info("Train the network to populate this panel.")
        else:
            t0 = time.perf_counter()
            q_pred, sigma = net.predict(img)
            ms_learned = (time.perf_counter() - t0) * 1000
            err_l = angular_error_deg(q_true, q_pred)
            a, b, c = st.columns(3)
            a.metric("Attitude error", f"{err_l:.3f} deg")
            b.metric("Self-reported sigma", f"{np.degrees(sigma):.3f} deg")
            c.metric("Inference time", f"{ms_learned:.0f} ms")
            st.caption("Constant time regardless of noise: no search, no "
                       "catalog lookup, always returns an answer.")

    with st.expander("What am I looking at?"):
        st.markdown(f"""
This frame was rendered from a real star catalog ({len(cat):,} stars down to
magnitude 6.5) seen through a {cam.fov_deg:.0f}-degree field of view, with a
pinhole camera model, Gaussian point spread functions, Poisson photon noise and
sensor read noise. The true attitude is known exactly because we generated it.

Turn the degradation slider up. The classical solver stays accurate right up
until it stops working, and its solve time climbs steeply along the way as it
has to test more and more star-triangle hypotheses before one verifies.
""")


# ----------------------------------------------------------------------------
# TRACKING
# ----------------------------------------------------------------------------

with tab_track:
    st.subheader("Temporal tracking: coasting through sensor loss")
    st.caption("A single frame gives an attitude. A sequence lets the filter "
               "learn the spacecraft's rotation rate and survive losing the "
               "sensor entirely.")

    c1, c2, c3 = st.columns(3)
    n_frames = c1.slider("Frames", 30, 120, 60, 10)
    rate = c2.slider("Slew rate (deg/s)", 0.5, 10.0, 3.0, 0.5)
    gap_len = c3.slider("Blackout length (frames)", 0, 30, 10)

    if st.button("Run sequence", use_container_width=True):
        dt = 0.1
        rng = np.random.default_rng(3)
        q = random_quaternion(rng)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        step = np.deg2rad(rate) * dt
        dq = np.concatenate([[np.cos(step / 2)], axis * np.sin(step / 2)])

        truth, meas, sigmas = [], [], []
        prog = st.progress(0.0, "rendering and solving frames...")
        for i in range(n_frames):
            truth.append(q.copy())
            im = renderer.render(q, rng=rng, noise=cfg)
            r = tracker.solve(im)
            if r["quat"] is not None:
                meas.append(r["quat"])
                sigmas.append(max(r["residual_deg"], 0.005))
            else:
                meas.append(None)
                sigmas.append(1.0)
            q = quat_normalize(quat_multiply(dq, q))
            prog.progress((i + 1) / n_frames)
        prog.empty()

        start = n_frames // 2
        gap = list(range(start, start + gap_len))
        out = run_sequence(meas, np.array(truth), dt, sigmas, dropout=gap)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(n_frames)
        ax.plot(x, out["raw_deg"], ".", color="#d9534f", alpha=0.6,
                label="per-frame solve")
        ax.plot(x, out["filtered_deg"], "-", color="#0275d8", lw=2,
                label="EKF filtered")
        ax.plot(x, out["sigma_deg"], "--", color="#999", lw=1,
                label="filter's own sigma")
        if gap:
            ax.axvspan(gap[0], gap[-1], color="#333", alpha=0.18)
            ax.text(np.mean(gap), ax.get_ylim()[1] * 0.85, "SENSOR LOST",
                    ha="center", fontsize=9, color="#333")
        ax.set_yscale("log")
        ax.set_xlabel("frame")
        ax.set_ylabel("attitude error (deg)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        a, b, c = st.columns(3)
        a.metric("Recovered slew rate", f"{out['final_rate_deg_s']:.2f} deg/s",
                 f"true {rate:.1f}")
        if gap:
            b.metric("Drift over blackout",
                     f"{out['filtered_deg'][gap[-1]]:.3f} deg",
                     f"from {out['filtered_deg'][gap[0]-1]:.3f} deg")
            after = min(gap[-1] + 3, n_frames - 1)
            c.metric("Error 3 frames after", f"{out['filtered_deg'][after]:.3f} deg")


# ----------------------------------------------------------------------------
# BENCHMARK
# ----------------------------------------------------------------------------

with tab_bench:
    st.subheader("Degradation sweep")
    st.caption("Accuracy, robustness and latency as the sensor gets worse. "
               "Bootstrap confidence intervals on the median, because the "
               "error distribution is heavily skewed.")
    trials = st.slider("Trials per level", 5, 50, 15, 5)
    if st.button("Run benchmark", use_container_width=True):
        import benchmark as bm
        with st.spinner("sweeping..."):
            results = bm.run(trials=trials, model_path=None,
                             out_json="/tmp/bench.json",
                             out_png="/tmp/bench.png")
        st.image("/tmp/bench.png", use_container_width=True)
        st.markdown("**Read the latency panel.** The classical solver does not "
                    "just get less accurate as conditions worsen, it gets "
                    "dramatically slower, because it must test more star "
                    "triangles before one verifies against the catalog.")
    try:
        st.image("benchmark.png", caption="last saved run",
                 use_container_width=True)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# ABOUT
# ----------------------------------------------------------------------------

with tab_about:
    st.markdown(f"""
### What this is

A **star tracker**: the sensor a spacecraft uses to work out which way it is
pointing. It photographs the sky, identifies the stars, and solves for its
orientation to arcsecond precision. Every satellite and deep-space probe
carries one.

### The machine learning problem

You cannot collect training data for this. No human can look at a star field
and write down the correct quaternion, and we have no orbital camera. So the
data **is** the project:

- A physics-based renderer generates star fields from a real catalog
  ({len(cat):,} stars to magnitude 6.5) with a pinhole camera model, Gaussian
  point spread functions, Poisson photon noise, read noise, motion blur and
  hot pixels.
- Every image comes with an exact label, for free, at ~150 images/second.
- Sensor parameters are randomised per image (**domain randomisation**), so the
  network must learn the rotation-invariant geometry rather than memorise one
  sensor's quirks.

### Three learned components

1. **Attitude regression** — ResNet-18 to a unit quaternion. Trained with a
   geodesic loss, because quaternions double-cover SO(3): `q` and `-q` are the
   same rotation, and plain MSE penalises a perfect prediction.
2. **Star detection** — a U-Net heatmap regressor supervised by the renderer's
   exact sub-pixel positions, replacing threshold-based centroiding.
3. **Uncertainty** — a heteroscedastic variance head. The model reports how
   confident it is, and that number is fed to the Kalman filter as the
   measurement covariance.

### The classical baseline we compete against

Sub-pixel centroiding, then star identification by **triangle matching** (one
inter-star angle matches ~1000 catalog pairs, so three mutual angles are needed
to disambiguate), then **Wahba's problem** solved by SVD, then RANSAC-style
verification by reprojection.

It reaches about **0.01 degrees — under one arcminute**. It is a strong
baseline and we did not weaken it.

### The honest result

The classical solver is more accurate where it works. But as conditions degrade
its solve time climbs by more than an order of magnitude and it eventually
fails outright, while the network is constant-time and always answers. **They
fail differently**, so the right system is a hybrid: the network as a fast
always-available prior, the classical solver for precision whenever it can
verify itself.
""")

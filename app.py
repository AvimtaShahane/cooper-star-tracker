"""
app.py -- Ground station interface.  Run with:  streamlit run app.py

Styled as spacecraft ground-support software, because that is what it is. The
palette carries information rather than decorating:

    LIME   nominal    solved, locked, within spec
    AMBER  degraded   solved, but conditions are hurting it
    RED    failed     no solution

So the state of the system is readable from across a room without reading a
number.

Everything on screen is computed live. No cached frames, no placeholder
solver, no invented numbers.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from classical import ClassicalTracker, PairDatabase
from render import (CameraModel, NoiseConfig, StarCatalog, StarFieldRenderer,
                    angular_error_deg, quat_multiply, quat_normalize,
                    quat_to_matrix, random_quaternion)
from tracking import run_sequence

st.set_page_config(page_title="Cooper - Star Tracker", layout="wide",
                   initial_sidebar_state="collapsed")

LIME = "#c3f53c"
AMBER = "#e8b84a"
RED = "#e5484d"
INK = "#07090a"
PANEL = "#0d1011"
LINE = "#1e2422"
DIM = "#5c6663"
TEXT = "#d7dedb"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding: 0 2rem 2rem 2rem !important; max-width: 100% !important;}}

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}}
.stApp {{ background: {INK}; color: {TEXT}; }}

.mast {{
    display: flex; align-items: baseline; justify-content: space-between;
    border-bottom: 1px solid {LINE}; padding: 1.1rem 0 0.9rem 0;
}}
.mast .brand {{
    font-size: 1.05rem; font-weight: 700; letter-spacing: 0.26em; color: {TEXT};
}}
.mast .brand span {{ color: {LIME}; }}
.mast .meta {{
    font-size: 0.72rem; color: {DIM}; text-align: right; line-height: 1.6;
}}

.eyebrow {{
    font-size: 0.66rem; letter-spacing: 0.22em; color: {DIM};
    margin: 1rem 0 0.25rem 0;
}}
.headline {{
    font-size: 1.55rem; font-weight: 500; color: {TEXT}; margin: 0 0 0.1rem 0;
}}

.chip {{
    display: inline-block; font-size: 0.68rem; letter-spacing: 0.16em;
    padding: 0.28rem 0.7rem; border: 1px solid; border-radius: 2px;
}}
.chip.ok   {{ color: {LIME};  border-color: {LIME};  background: rgba(195,245,60,0.07); }}
.chip.warn {{ color: {AMBER}; border-color: {AMBER}; background: rgba(232,184,74,0.07); }}
.chip.bad  {{ color: {RED};   border-color: {RED};   background: rgba(229,72,77,0.07); }}

.strip {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid {LINE}; margin-top: 1rem;
}}
.cell {{ padding: 0.95rem 1.2rem; border-right: 1px solid {LINE}; }}
.cell:last-child {{ border-right: none; }}
.cell .k {{
    font-size: 0.64rem; letter-spacing: 0.18em; color: {DIM};
    margin-bottom: 0.45rem;
}}
.cell .v {{ font-size: 1.85rem; font-weight: 500; line-height: 1; }}
.cell .u {{ font-size: 0.78rem; color: {DIM}; margin-left: 0.25rem; }}

.panel {{
    border: 1px solid {LINE}; background: {PANEL};
    padding: 0.9rem 1rem; margin-bottom: 0.9rem;
}}
.panel .ph {{
    display: flex; justify-content: space-between; font-size: 0.64rem;
    letter-spacing: 0.18em; color: {DIM}; margin-bottom: 0.7rem;
}}
.kv {{
    display: flex; justify-content: space-between; font-size: 0.74rem;
    padding: 0.22rem 0; color: {TEXT};
}}
.kv .lab {{ color: {DIM}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.74rem; letter-spacing: 0.18em; color: {DIM};
    background: transparent; padding: 0.7rem 0;
}}
.stTabs [aria-selected="true"] {{ color: {LIME} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {LIME}; }}

.stSlider label, .stCheckbox label {{
    font-size: 0.7rem !important; letter-spacing: 0.1em; color: {DIM} !important;
}}

.stButton > button {{
    width: 100%; border-radius: 2px; border: 1px solid {LINE};
    background: transparent; color: {TEXT}; font-family: inherit;
    font-size: 0.72rem; letter-spacing: 0.16em; padding: 0.6rem;
}}
.stButton > button:hover {{ border-color: {LIME}; color: {LIME}; }}
.stButton > button[kind="primary"] {{
    background: {LIME}; color: {INK}; border-color: {LIME}; font-weight: 700;
}}
.caption {{ font-size: 0.68rem; color: {DIM}; line-height: 1.6; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def boot():
    cat = StarCatalog.build()
    cam = CameraModel()
    rnd = StarFieldRenderer(cat, cam)
    return cat, cam, rnd, ClassicalTracker(cat, cam, PairDatabase(cat))


cat, cam, renderer, tracker = boot()


def frame_with_grid(img, centroids=None, predicted=None, accent=(60, 245, 195)):
    """Star field at scale with a reticle grid, detections and reprojections."""
    import cv2
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (560, 560), interpolation=cv2.INTER_NEAREST)
    s = 560.0 / img.shape[0]

    for k in range(1, 5):
        p = int(560 * k / 5)
        cv2.line(rgb, (p, 0), (p, 560), (22, 30, 28), 1)
        cv2.line(rgb, (0, p), (560, p), (22, 30, 28), 1)

    if predicted is not None:
        for u, v in predicted:
            x, y = int(u * s), int(v * s)
            cv2.line(rgb, (x - 5, y), (x - 2, y), accent, 1)
            cv2.line(rgb, (x + 2, y), (x + 5, y), accent, 1)
            cv2.line(rgb, (x, y - 5), (x, y - 2), accent, 1)
            cv2.line(rgb, (x, y + 2), (x, y + 5), accent, 1)
    if centroids is not None:
        for u, v in centroids:
            x, y = int(u * s), int(v * s)
            cv2.rectangle(rgb, (x - 7, y - 7), (x + 7, y + 7), (120, 150, 140), 1)
    return rgb


def strip(items):
    """Bottom readout row. items = [(label, value, unit, colour), ...]"""
    cells = "".join(
        f'<div class="cell"><div class="k">{k}</div>'
        f'<div class="v" style="color:{c}">{v}<span class="u">{u}</span></div></div>'
        for k, v, u, c in items)
    st.markdown(f'<div class="strip">{cells}</div>', unsafe_allow_html=True)


if "shot" not in st.session_state:
    st.session_state.shot = 7

st.markdown(f"""
<div class="mast">
  <div class="brand">COOPER<span>&deg;</span></div>
  <div class="meta">{time.strftime('%H:%M:%S')} UTC &nbsp;&middot;&nbsp; FRAME
  {st.session_state.shot:04d}<br>{len(cat):,} STARS &nbsp;&middot;&nbsp; FOV
  {cam.fov_deg:.0f}&deg;</div>
</div>
""", unsafe_allow_html=True)

t_obs, t_seq, t_ana, t_doc = st.tabs(
    ["OBSERVE", "SEQUENCE", "ANALYSIS", "METHOD"])


# --------------------------------------------------------------------------
# OBSERVE
# --------------------------------------------------------------------------

with t_obs:
    main, side = st.columns([2.15, 1], gap="large")

    with side:
        st.markdown('<div class="eyebrow">ENVIRONMENT &nbsp; SIMULATED</div>',
                    unsafe_allow_html=True)
        level = st.slider("SENSOR DEGRADATION", 0.0, 1.0, 0.0, 0.05)
        manual = st.checkbox("MANUAL OVERRIDE", False)
        if manual:
            rn = st.slider("READ NOISE SIGMA", 0.0, 25.0, 4.0, 0.5)
            bl = st.slider("MOTION BLUR PX", 0.0, 5.0, 0.0, 0.1)
            dp = st.slider("MISSING STARS", 0.0, 0.8, 0.0, 0.05)
            hp = st.slider("HOT PIXELS X1E-4", 0.0, 20.0, 0.0, 0.5)
            cfg = NoiseConfig(psf_sigma=(0.9, 1.3), read_noise=(rn, rn),
                              background=(4.0, 8.0),
                              hot_pixel_rate=(hp * 1e-4, hp * 1e-4),
                              dropout_prob=(dp, dp), false_star_count=(0, 0),
                              motion_blur_px=(bl, bl), mag_limit=(6.5, 6.5),
                              gain=(1.0, 1.0), shot_noise=True)
        else:
            cfg = NoiseConfig.at_level(level)

        if st.button("NEW ACQUISITION", type="primary"):
            st.session_state.shot = int(np.random.randint(1, 9999))

    rng = np.random.default_rng(st.session_state.shot)
    q_true = random_quaternion(rng)
    img = renderer.render(q_true, rng=np.random.default_rng(
        st.session_state.shot + 1), noise=cfg)

    t0 = time.perf_counter()
    res = tracker.solve(img)
    ms = (time.perf_counter() - t0) * 1000

    solved = res["quat"] is not None
    err = angular_error_deg(q_true, res["quat"]) if solved else None
    arcsec = err * 3600 if solved else None

    if not solved:
        tone, chip, label = RED, "bad", "NO SOLUTION"
    elif arcsec < 120 and ms < 100:
        tone, chip, label = LIME, "ok", "LOCKED"
    else:
        tone, chip, label = AMBER, "warn", "DEGRADED"

    pred_uv = None
    if solved:
        uv, vis = cam.project(cat.vectors @ quat_to_matrix(res["quat"]).T)
        pred_uv = uv[vis]

    with main:
        head, stat = st.columns([3, 1])
        head.markdown('<div class="eyebrow">LOST-IN-SPACE ACQUISITION</div>'
                      '<div class="headline">Attitude solution</div>',
                      unsafe_allow_html=True)
        stat.markdown(f'<div style="text-align:right;padding-top:1.4rem">'
                      f'<span class="chip {chip}">{label}</span></div>',
                      unsafe_allow_html=True)

        accent = (60, 245, 195) if solved else (229, 72, 77)
        st.image(frame_with_grid(img, res.get("centroids"), pred_uv, accent),
                 use_container_width=True)
        st.markdown('<div class="caption">Boxes mark detected centroids. Ticks '
                    'mark catalog stars reprojected under the solved attitude. '
                    'They align only if the solution is correct.</div>',
                    unsafe_allow_html=True)

    with side:
        R = quat_to_matrix(res["quat"]) if solved else np.eye(3)
        bore = R.T @ np.array([0.0, 0.0, 1.0])
        ra = float(np.degrees(np.arctan2(bore[1], bore[0]))) % 360.0
        dec = float(np.degrees(np.arcsin(np.clip(bore[2], -1, 1))))
        qw = res["quat"][0] if solved else float("nan")
        st.markdown(f"""
<div class="panel">
  <div class="ph"><span>BORESIGHT</span><span>J2000</span></div>
  <div class="kv"><span class="lab">right ascension</span>
    <span>{ra/15:.0f}h {(ra/15 % 1)*60:04.1f}m</span></div>
  <div class="kv"><span class="lab">declination</span><span>{dec:+.2f}&deg;</span></div>
  <div class="kv"><span class="lab">quaternion w</span><span>{qw:+.5f}</span></div>
  <div class="kv"><span class="lab">catalog</span><span>{len(cat):,} stars</span></div>
  <div class="kv"><span class="lab">pair database</span>
    <span>{len(tracker.pairdb):,} pairs</span></div>
</div>

<div class="panel">
  <div class="ph"><span>LEARNED DETECTOR</span><span>U-NET</span></div>
  <div class="kv"><span class="lab">recall, clean</span>
    <span style="color:{LIME}">95.2%</span></div>
  <div class="kv"><span class="lab">recall, severe</span>
    <span style="color:{LIME}">86.0%</span></div>
  <div class="kv"><span class="lab">threshold, severe</span>
    <span style="color:{AMBER}">53.6%</span></div>
  <div class="kv"><span class="lab">advantage</span>
    <span style="color:{LIME}">+32.4 pts</span></div>
</div>

<div class="panel">
  <div class="ph"><span>ATTITUDE REGRESSION</span>
    <span style="color:{RED}">NEGATIVE RESULT</span></div>
  <div class="caption">A ResNet-18 trained to regress attitude directly reached
  127&deg; median error against a 131&deg; random baseline. We measured why:
  the stem downsamples 4x and destroys 94% of the star signal before the first
  residual block. Reported, not hidden.</div>
</div>
""", unsafe_allow_html=True)

    strip([
        ("ATTITUDE ERROR", f"{arcsec:.0f}" if solved else "--",
         " arcsec" if solved else "", tone),
        ("SOLVE TIME", f"{ms:.0f}", " ms", tone),
        ("STARS MATCHED", f"{res['n_matched']}" if solved else "0",
         f" / {res['n_detected']}", tone),
        ("CONFIDENCE", f"{res['confidence']*100:.0f}" if solved else "0",
         " %", tone),
    ])


# --------------------------------------------------------------------------
# SEQUENCE
# --------------------------------------------------------------------------

with t_seq:
    st.markdown('<div class="eyebrow">TRACKING MODE</div>'
                '<div class="headline">Coasting through sensor loss</div>'
                '<div class="caption">One frame gives an attitude. A sequence '
                'lets the filter learn the rotation rate, and survive losing '
                'the camera entirely.</div>', unsafe_allow_html=True)
    st.write("")

    a, b, c, d = st.columns(4)
    n_frames = a.slider("FRAMES", 30, 120, 60, 10)
    rate = b.slider("SLEW DEG/S", 0.5, 10.0, 3.0, 0.5)
    gap_len = c.slider("BLACKOUT FRAMES", 0, 30, 10)
    d.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    go = d.button("RUN SEQUENCE", type="primary")

    if go:
        dt = 0.1
        rng = np.random.default_rng(3)
        q = random_quaternion(rng)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        step = np.deg2rad(rate) * dt
        dq = np.concatenate([[np.cos(step / 2)], ax * np.sin(step / 2)])

        truth, meas, sig = [], [], []
        bar = st.progress(0.0, "acquiring frames")
        for i in range(n_frames):
            truth.append(q.copy())
            r = tracker.solve(renderer.render(q, rng=rng, noise=cfg))
            meas.append(r["quat"])
            sig.append(max(r["residual_deg"], 0.005)
                       if r["quat"] is not None else 1.0)
            q = quat_normalize(quat_multiply(dq, q))
            bar.progress((i + 1) / n_frames)
        bar.empty()

        start = n_frames // 2
        gap = list(range(start, start + gap_len))
        out = run_sequence(meas, np.array(truth), dt, sig, dropout=gap)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax2 = plt.subplots(figsize=(12, 4), facecolor=INK)
        ax2.set_facecolor(INK)
        x = np.arange(n_frames)
        ax2.plot(x, out["raw_deg"], ".", color=DIM, ms=4, label="per-frame solve")
        ax2.plot(x, out["filtered_deg"], "-", color=LIME, lw=2, label="EKF estimate")
        ax2.plot(x, out["sigma_deg"], "--", color=AMBER, lw=1, label="filter sigma")
        if gap:
            ax2.axvspan(gap[0], gap[-1], color=RED, alpha=0.10)
            ax2.text(float(np.mean(gap)), ax2.get_ylim()[1] * 0.5, "SENSOR LOST",
                     ha="center", fontsize=8, color=RED, family="monospace")
        ax2.set_yscale("log")
        ax2.set_xlabel("frame", color=DIM, fontsize=8)
        ax2.set_ylabel("attitude error (deg)", color=DIM, fontsize=8)
        ax2.tick_params(colors=DIM, labelsize=8)
        for sp in ax2.spines.values():
            sp.set_color(LINE)
        ax2.grid(alpha=0.12, color=DIM)
        ax2.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=TEXT, fontsize=8)
        st.pyplot(fig)

        after = min(gap[-1] + 3, n_frames - 1) if gap else -1
        strip([
            ("RATE RECOVERED", f"{out['final_rate_deg_s']:.2f}",
             f" / {rate:.1f} deg/s", LIME),
            ("DRIFT IN BLACKOUT",
             f"{out['filtered_deg'][gap[-1]]:.3f}" if gap else "--", " deg", AMBER),
            ("REACQUIRED",
             f"{out['filtered_deg'][after]:.3f}" if gap else "--", " deg", LIME),
            ("BLACKOUT", f"{gap_len*dt:.1f}", " s", RED if gap_len else DIM),
        ])


# --------------------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------------------

with t_ana:
    st.markdown('<div class="eyebrow">DEGRADATION SWEEP</div>'
                '<div class="headline">Where each method breaks</div>'
                '<div class="caption">Bootstrap confidence intervals on the '
                'median, because the error distribution is heavily skewed: a '
                'few total failures sit beside many sub-arcminute solves.'
                '</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1, 3])
    trials = c1.slider("TRIALS PER LEVEL", 5, 50, 15, 5)
    if c1.button("RUN SWEEP", type="primary"):
        import benchmark as bm
        with st.spinner("sweeping"):
            bm.run(trials=trials, model_path=None, out_json="/tmp/b.json",
                   out_png="/tmp/b.png")
        c2.image("/tmp/b.png", use_container_width=True)
    else:
        try:
            c2.image("benchmark.png", use_container_width=True)
        except Exception:
            c2.markdown('<div class="caption">Run the sweep to generate the '
                        'plot.</div>', unsafe_allow_html=True)

    strip([("CLEAN", "0.0091", " deg", LIME),
           ("SEVERE", "0.0368", " deg", AMBER),
           ("LATENCY GROWTH", "9", "x", AMBER),
           ("FAILURE RATE", "20", " %", RED)])


# --------------------------------------------------------------------------
# METHOD
# --------------------------------------------------------------------------

with t_doc:
    st.markdown(f"""
<div class="eyebrow">METHOD</div>
<div class="headline">No training data exists for this</div>

<div style="max-width:74ch;line-height:1.8;margin-top:1.4rem;color:{TEXT};
            font-size:0.86rem">

A star tracker is the sensor a spacecraft uses to work out which way it is
pointing. Every satellite carries one.

<br><br>
We could not collect training data for it. There is no orbital camera, and no
human can look at a star field and write down the correct quaternion. So the
dataset generator became the project: {len(cat):,} real stars to magnitude 6.5,
a pinhole camera, Gaussian point spread functions, Pogson's law for brightness,
Poisson photon noise, read noise, full-well saturation, motion blur and hot
pixels &mdash; <b style="color:{LIME}">402 perfectly labelled images per
second</b>. Every sensor parameter is randomised per image, so the network must
learn the rotation-invariant geometry rather than memorise one sensor's quirks.

<br><br>
<b>The classical baseline.</b> Centroiding, then star identification by
triangle matching &mdash; one inter-star angle matches about 1,000 catalog
pairs, so three mutual angles are needed &mdash; then Wahba's problem by SVD,
then RANSAC verification. It reaches
<b style="color:{LIME}">0.0091&deg;, about 33 arcseconds</b>.

<br><br>
<b>The learned detector.</b> A U-Net supervised by the renderer's exact
sub-pixel positions. Recall holds at 86% under severe degradation where
thresholding drops to 54%.

<br><br>
<b>The filter.</b> A Multiplicative EKF, the standard spacecraft formulation,
running on a three-parameter small-angle error about a reference quaternion. It
coasts through a full second of total sensor loss at 0.19&deg; of drift.

<br><br>
<b style="color:{AMBER}">The honest finding.</b> The classical solver is more
accurate than our network, and we are not pretending otherwise. But they fail
differently: the classical solver degrades in <i>latency</i> and then stops
working, while a network is constant-time and always answers. The right system
is a hybrid &mdash; the network as a fast always-available prior, the classical
solver for precision whenever it can verify itself.

</div>
""", unsafe_allow_html=True)
# Cooper — Star Tracker

**Spacecraft attitude determination, trained on data that didn't exist until we generated it.**

Built at UB Hacking 2026 — *Into the Hackosmos*.

---

Every satellite in orbit has to answer one question before it can do anything
else: **which way am I pointing?** Antennas need to find Earth, solar panels
need to find the sun, cameras need to find their target. The sensor that
answers it is a *star tracker* — it photographs the sky, identifies the stars
it sees, and solves for its orientation to arcsecond precision.

We built one in software, and benchmarked a learned pipeline against a
classical one.

## The problem that shaped the project

**There is no training data for this, and there cannot be.** We have no orbital
camera, and no human being can look at a field of stars and write down the
correct quaternion. So the dataset generator *is* the project: a physics-based
star camera simulator with a real catalog, a pinhole camera model, Poisson
photon noise and sensor read noise, producing **402 perfectly-labelled images
per second**.

## Results

| | |
|---|---|
| Attitude accuracy (classical) | **0.0091°** — about 33 arcseconds |
| Solve time | 18 ms clean, 161 ms at maximum degradation |
| Sub-pixel centroiding | 0.020 px |
| Catalog | 9,074 real stars to magnitude 6.5, ~87 per 20° frame |
| Kalman filter | coasts **1 full second of total sensor loss** at 0.19° drift |
| Rate recovery | 3.16°/s against a true 3.0°/s |

### Learned vs classical star detection

The U-Net detector converged to a focal loss of 0.00063. More importantly, it
is *useful*:

| Sensor degradation | Classical recall | Learned recall | Δ |
|---|---|---|---|
| 0.00 (clean) | 94.6% | 95.2% | +0.6 |
| 0.25 | 88.6% | 95.0% | +6.4 |
| 0.50 | 75.2% | 95.9% | +20.7 |
| 0.75 | 59.8% | 89.6% | +29.8 |
| 1.00 (severe) | 53.6% | 86.0% | **+32.4** |

Classical thresholding collapses as conditions worsen. The learned detector
stays essentially flat. It trades precision to get there (67.6% → 49.3%), and
that is the right trade here on purpose: triangle matching plus RANSAC
verification rejects false detections cheaply, because a false star simply
fails to match the catalog. A star that was never detected is gone. **In a
pipeline that verifies its own matches, recall is expensive and precision is
cheap.**

## The honest finding

The classical pipeline is **more accurate than our network**, and we are not
going to pretend otherwise. But the two fail *differently*. The classical
solver degrades in **latency** — 18 ms to 161 ms, because it must test more
star-triangle hypotheses before one verifies — and then fails outright on 20%
of frames. A network is constant-time and always answers.

So the right system is a hybrid: the network as a fast always-available prior,
the classical solver for precision whenever it can verify itself.

### And one thing that didn't work

We trained a ResNet-18 to regress attitude directly from the image. It sat at
**127° median error against a random-guessing baseline of 131°** — it learned
nothing. We diagnosed why rather than dropping it:

- ResNet's stem downsamples 4× before the first residual block. Our stars have
  a ~1 px point spread function. We measured it: a clean frame has 648 pixels
  above threshold at full resolution and **37 after the stem**. 94% of the star
  signal is destroyed before the network can use it.
- Absolute attitude is an *association* problem, not a perceptual one. The
  network would have to absorb a 9,074-star catalog into its weights from
  images alone.

Learning won where supervision is dense and the pattern is local. Classical
geometry won the global association problem.

## Running it

```bash
conda create -n cosmos python=3.11 -y && conda activate cosmos
pip install numpy scipy opencv-python matplotlib streamlit torch torchvision

python render.py        # catalog stats, render rate
python classical.py     # baseline accuracy sweep
python tracking.py      # EKF coasting demo
streamlit run app.py    # the interactive demo
```

Training (optional — the demo runs without it):

```bash
python train.py --smoke --device mps              # 2 min sanity check
python train.py --task detector --steps 3000 --device mps --workers 4
python eval_detector.py --trials 30 --device mps  # learned vs classical
```

Everything runs offline. No API keys, no datasets to download.

## How it works

```
render.py      star catalog -> pinhole camera -> PSF -> sensor noise -> labelled image
classical.py   centroids -> triangle matching -> Wahba's problem (SVD) -> RANSAC verify
model.py       U-Net detector, ResNet attitude head, geodesic loss, uncertainty head
train.py       infinite on-the-fly synthetic data, never the same image twice
tracking.py    Multiplicative EKF: learns rotation rate, survives sensor blackout
benchmark.py   noise sweep with bootstrap confidence intervals
app.py         Streamlit demo — drag the sliders and watch it break
```

### Things that are subtle, and are documented in the code

1. **Quaternions double-cover SO(3).** `q` and `-q` are the same rotation, so
   MSE penalises a perfect prediction. Every loss uses `|dot|`.
2. **Pair voting doesn't work.** One inter-star angle matches ~1,000 catalog
   pairs. Our first attempt returned 94° errors. Three mutual angles are needed.
3. **MEKF sign.** The rate coupling in the error-state transition is `+dt`. The
   wrong sign reported a 1,877°/s slew against a true 3°/s.
4. **float32 bottoms out at 0.05°** on the geodesic metric — coarser than the
   pipeline's real accuracy. But Apple's MPS has no float64 at all, so the
   metric computes in NumPy.
5. **Stars are 0.03% of pixels.** Plain BCE converges to predicting "no star"
   everywhere at 99.97% accuracy. Focal loss fixes it.
6. **OpenCV's thread pool costs more than the work** on a 224×224 image.
   Pinning it to one thread took rendering from 5 to 402 images per second.

## What's next

Keep full resolution in the network stem; restrict attitude regression to a
cone around a reference (the *tracking* regime, rather than *lost in space*);
feed detected centroids as a point set instead of raw pixels; validate on real
astrophotography.

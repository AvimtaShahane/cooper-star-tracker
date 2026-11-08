# Star Tracker — Build Plan (software only)

**What it is:** the sensor a spacecraft uses to determine its orientation. It photographs the sky, identifies the stars, and solves for attitude to arcsecond precision. Every satellite carries one.

**Why it wins the AI/ML track:** there is no training data for this and none can be collected, so the dataset generator *is* the project. That is a real methodology story, not an LLM wrapper.

---

## Status: the critical path is already built and tested

| File | What it does | Verified |
|---|---|---|
| `render.py` | Star catalog, camera model, PSF, sensor noise | 9,074 stars, 87 per frame, **151 img/s** |
| `classical.py` | Centroids → triangle match → Wahba/SVD | **0.0091° (33 arcsec)**, 100% solve rate |
| `model.py` | Geodesic loss, ResNet attitude net, U-Net detector | 11.3M + 118K params, gradients finite |
| `train.py` | On-the-fly dataset, training loops | Smoke test: **131° → 0.006°** |
| `tracking.py` | Multiplicative EKF | Coasts 1 s of total sensor loss at 0.19° |
| `benchmark.py` | Noise sweep, bootstrap CIs, 3-panel plot | Full sweep runs |
| `app.py` | Streamlit demo, judge-controlled sliders | Starts clean |

**Everything above runs right now with zero training.** You already have a demoable project. What remains is training the network and polishing.

---

## Setup (10 minutes)

```bash
pip install numpy scipy opencv-python matplotlib torch torchvision streamlit
cd cosmos
python render.py        # catalog stats, render speed
python classical.py     # baseline accuracy sweep
python tracking.py      # EKF coasting demo
streamlit run app.py    # the demo
```

No API keys, no downloads, no internet needed after install. That matters at 3 AM.

---

## Hour-by-hour

### Hours 0–1 — Orient
All four of you read `PLAN.md` and run the four commands above. Confirm everyone sees the same numbers. Agree the data contract out loud:

> Images are `uint8 (224,224)`. Attitudes are `float64 (4,)` unit quaternions in `[w,x,y,z]`. Always normalized.

Interface drift between teammates at hour 14 is the most common way four-person teams lose a hackathon.

### Hours 1–2 — Split and launch training
| Person | Owns |
|---|---|
| **A** | Training: run the smoke test, then launch the long attitude run |
| **B** | Detector: train the U-Net, compare against OpenCV centroiding |
| **C** | App: Streamlit polish, 3D attitude view, judge sliders |
| **D** | Benchmark, ablations, Devpost draft, video |

Person A runs this first, always:
```bash
python train.py --smoke          # 2 min — must PASS before anything long
python train.py --task attitude --steps 8000 --batch-size 64
```
If the smoke test fails, the bug is in the code, not the hyperparameters. You find it in two minutes instead of four hours.

### Hours 2–6 — Train and build in parallel
Attitude training on a laptop GPU: roughly 1–2 hours for 8,000 steps. While it runs, B trains the detector (`--task detector --steps 3000`, much faster), C builds the UI, D runs the classical-only benchmark and writes the Devpost draft.

**Expectation management:** full-sky attitude regression is genuinely hard. A single network will likely land in the 1–10° range, not arcseconds. That is not a failure — see *The honest result* below. If you want a headline accuracy number from the learned side, train a **tracking-mode** variant instead: restrict to attitudes within ±10° of a reference (`NoiseConfig` plus a narrowed `random_quaternion`), which is the regime a real star tracker spends most of its life in, and accuracy improves dramatically.

### Hours 6–11 — Integrate
Wire the trained model into `app.py` (the loader is already there — drop `attitude_net.pt` next to it and the panel populates). Wire the detector into the classical pipeline as a swappable front end so you can benchmark learned-vs-threshold centroiding.

**Post the Devpost draft by 11 PM.** Rough is fine; you can edit later. Missing it disqualifies you entirely.

### Hours 11–15 — Depth
Run the full benchmark with the model, including `--hybrid`. Add the uncertainty-vs-error correlation plot: bin predictions by predicted σ, plot actual error per bin. If the line is monotone, the model knows when it is wrong, and that is the single most commercially interesting slide you will have.

### Hours 15–19 — Ablations
This is what separates "we trained a model" from "we did ML". Train short variants (1,500 steps each is enough to rank them):

| Variant | Flag | Expected finding |
|---|---|---|
| No domain randomization | pin `noise_level=0.0` | collapses on noisy test data |
| MSE instead of geodesic | swap the loss | training stalls |
| From scratch | `--no-pretrained` | slower convergence |
| Smaller backbone | `backbone="resnet34"` vs 18 | diminishing returns |

Report as a table. Judges notice tables.

### Hours 19–22 — Polish
3D attitude visualization (Plotly in Streamlit), final benchmark plots, README.

### Hours 22–24 — Ship
Film the 2-minute video. Attend the ACV/Alan Hunt session on this first — it is mandatory for a complete submission and most teams rush it.

---

## The demo script (90 seconds)

1. **Open on the Live tab, clean conditions.** "This is a simulated star camera. It just determined its orientation to 33 arcseconds, from scratch, knowing nothing beforehand."
2. **Hand the laptop to the judge.** "Drag that slider." Noise climbs, hot pixels appear, stars drop out. The solve time counter climbs from 30 ms toward 800 ms. Eventually it fails.
3. **Tracking tab.** Run a sequence, black out ten frames. "We just lost the sensor completely for a full second. The filter coasted on its learned rotation rate and drifted 0.19 degrees."
4. **Benchmark tab.** "Here is the whole picture, with bootstrap confidence intervals."

Handing over the controls is the strongest move available to you. It cannot be accused of cherry-picking.

---

## The honest result (say this out loud)

The classical pipeline is **more accurate** than the network where it works — about 0.01°. Do not claim otherwise; a judge who knows the field will catch it, and the honest version is the better story:

> They fail **differently**. The classical solver degrades in *latency* — 30 ms to 800 ms, a 25× climb — and then fails outright at 15% of frames. The network is constant-time and always answers, with calibrated uncertainty. So the right system is a hybrid: the network as a fast always-available prior, the classical solver for precision whenever it can verify itself.

That is an engineering conclusion drawn from your own benchmark. It is worth far more than an inflated accuracy claim.

---

## Track positioning

- **Best AI/ML** — three learned components, synthetic data generation, domain randomization, ablations, calibrated uncertainty, benchmarked against a strong classical baseline.
- **Radial Ventures ($1000)** — the slide says *"hardware, sensors, cameras, real-world measurements, **or simulation**."* Lead with the physics: real catalog, pinhole optics, Pogson's law, Poisson shot noise, full-well saturation. "This is how aerospace teams validate attitude algorithms before launch," which is true.
- **Most Over Engineered** — two full pipelines, an EKF, and a bootstrap-CI benchmark harness for a problem nobody asked you to solve.
- **Spirit of the Hackathon** — spacecraft attitude determination at a space-themed hackathon.

---

## Traps already handled for you

These are in the code with comments explaining them. Know them, because judges ask.

1. **Quaternion double cover.** `q` and `-q` are the same rotation. MSE scores a perfect prediction at 1.0. Every loss here uses `|dot|`.
2. **arccos gradient blowup.** Train on `1 - |dot|` (bounded gradient), report geodesic degrees (interpretable). Using arccos as the loss destabilizes training exactly when the model starts doing well.
3. **Float32 metric floor.** In float32 the geodesic metric bottoms out at 0.05°, coarser than the classical pipeline's real accuracy — you would be measuring your own rounding error. The metric is computed in float64.
4. **Pair voting does not work.** One inter-star angle matches ~1,000 catalog pairs, so votes drown in false matches. You need the *triangle* constraint. We tried the naive version first and it returned 94° errors.
5. **MEKF sign error.** The rate coupling in the error-state transition is `+dt`, not `−dt`. The wrong sign made the filter report a 1877°/s slew rate against a true 3°/s.
6. **Focal loss for the heatmap.** Stars are 0.03% of pixels. Plain BCE converges to predicting "no star" everywhere at 99.97% accuracy.
7. **Uniform random rotations.** `random_quaternion` uses Shoemake's method. Random Euler angles cluster at the poles and would bias your training set.

---

## If you have spare time

- **Validate on real astrophotos.** Download a few public night-sky images and run the pipeline. Trained purely on synthetic data, generalizing to real photons, is your sim-to-real proof without a camera. One slide, high impact.
- **Point-at-a-star mode.** `catalog.lookup("Vega")` already works. Compute the rotation needed to center it and animate the slew.
- **Learned sensor fusion.** Let a small network gate between the classical and learned estimates instead of the hand-written rule.

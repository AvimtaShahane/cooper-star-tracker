# Technical reference

The vocabulary this project uses, and where each piece lives in the code.

---

## Core concepts

**Attitude** — orientation. Not position. Position says where a satellite is;
attitude says how it is rotated. Three degrees of freedom: pitch, yaw, roll.

**Lost-in-space** — determining attitude with no prior estimate at all. The
hard case, and what `classical.py` solves. The easy case is *tracking mode*,
where you already have a rough answer from the previous frame.

**Boresight** — the direction the camera is looking. The +Z axis of the camera
frame.

**Arcsecond** — 1/3600 of a degree. The unit attitude error is reported in,
because degrees are too coarse to be meaningful here. Our 0.0091° is 33
arcseconds. Flight star trackers are 1–10.

**Right ascension / declination** — longitude and latitude on the celestial
sphere. How the star catalog stores positions.

**J2000** — the epoch the catalog coordinates are fixed to. Stars drift, so
positions need a reference date.

---

## Representing rotations

**Quaternion** — four numbers `[w, x, y, z]` with unit length, representing a
3D rotation. Used throughout instead of Euler angles (which suffer gimbal
lock) or rotation matrices (nine numbers, six constraints).

**Double cover** — `q` and `-q` are the *same* rotation. Four numbers for three
degrees of freedom means every rotation has two valid representations. This is
why mean squared error is wrong as a loss: it penalises a perfect prediction.
Every loss in `model.py` uses the absolute cosine.

**SO(3)** — the mathematical space of all 3D rotations. Random guessing on
SO(3) gives a median error of 131°, which is the baseline our failed attitude
network was measured against.

**Geodesic distance** — the true angular difference between two rotations,
`2·arccos(|q₁·q₂|)`. Used as the reporting metric. Not used as the training
loss, because arccos has unbounded gradient near zero error.

---

## The classical pipeline (`classical.py`)

**Centroiding** — finding the sub-pixel centre of each star. Background
estimate, sigma threshold, connected components, intensity-weighted first
moment. Accurate to 0.020 px.

**Star identification** — working out which catalog star each detection is.

**Triangle matching** — the method used. A single inter-star angle matches
about 1,000 catalog pairs, so it is useless alone. Three mutual angles between
a triple of stars are near-unique. Requires a neighbour index to ask "which
star sits at angle X from A and angle Y from B".

**Pair database** — 77,394 precomputed inter-star angles from 2,833 guide
stars, sorted for range lookup.

**Wahba's problem** — given matched vector pairs in two frames, find the
rotation between them. Posed in 1965, solved optimally by SVD (the Kabsch
algorithm) in about six lines.

**RANSAC verification** — after an initial solve, reproject the whole catalog
and refit on everything that matches. A wrong triangle explains only its own
three stars; a correct one explains nearly every detection in frame. This is
what delivers the final accuracy.

---

## The renderer (`render.py`)

**Pinhole camera model** — projects 3D directions onto a 2D sensor plane. Focal
length derived from the field of view.

**Point spread function (PSF)** — a real optic spreads a point source over
several pixels as a Gaussian. Without this, sub-pixel centroiding is
impossible.

**Pogson's law** — the magnitude scale. Each step of 1 magnitude is a factor of
10^0.4 ≈ 2.512 in brightness. Magnitude 6.5 is roughly the naked-eye limit and
our catalog cutoff.

**Poisson shot noise** — photon arrival is a random process. This noise is
irreducible; no better sensor removes it.

**Read noise** — Gaussian noise added by the readout electronics.

**Full-well saturation** — pixels stop accumulating once the well fills, which
is why bright stars render as flat discs rather than tall peaks.

**Domain randomization** — randomising every sensor parameter per image, so the
network must learn the rotation-invariant geometry rather than memorise one
camera's characteristics. The mechanism behind synthetic-to-real transfer.

---

## The learned components (`model.py`, `train.py`)

**U-Net** — encoder–decoder with skip connections, outputs at input
resolution. Used here to produce a star-probability heatmap.

**Heatmap regression** — predicting a soft Gaussian bump at each star instead
of a hard binary mask. Gives the network a gradient to follow when it is close
but not exact.

**Focal loss** — down-weights easy examples. Necessary because stars are 0.03%
of pixels, so plain cross-entropy converges to predicting "no star" everywhere
at 99.97% accuracy.

**Recall / precision** — recall is the fraction of real stars found; precision
is the fraction of detections that were real. In this pipeline recall is the
expensive quantity: a false detection is rejected cheaply by triangle
matching, but a missed star is gone.

**Heteroscedastic uncertainty** — the model predicting its own error variance,
trained jointly with the main task. Feeds the Kalman filter as a measurement
covariance.

**Overfit smoke test** — before any long training run, confirm the loop can
drive loss to zero on ~24 fixed samples. If it cannot, the bug is in the code,
not the hyperparameters.

---

## The filter (`tracking.py`)

**EKF** — Extended Kalman Filter. Recursive state estimation for nonlinear
systems.

**Multiplicative EKF (MEKF)** — the spacecraft-standard variant. A quaternion
has four components but only three degrees of freedom, so a filter running on
the raw components has a singular covariance. The MEKF keeps the full attitude
as a reference quaternion and runs the filter on a three-parameter small-angle
*error* about it.

**Error state** — the small-angle deviation being estimated, rather than the
attitude itself.

**Innovation** — the difference between a measurement and the filter's
prediction of it.

**Joseph form** — a covariance update that stays positive-definite under
floating point, where the textbook `(I − KH)P` form can go indefinite and
diverge.

**Coasting** — continuing to propagate an estimate with no measurements. Ours
survives one full second of total sensor loss at 0.19° of drift.

---

## Evaluation (`benchmark.py`, `eval_detector.py`)

**Bootstrap confidence interval** — resampling to estimate uncertainty on the
median. Used because the error distribution is heavily skewed: a handful of
total failures sit beside many sub-arcminute solves, so mean ± standard
deviation would mislead.

**Degradation sweep** — a single 0-to-1 dial interpolating every sensor
imperfection from clean to severe, so accuracy, robustness and latency can be
plotted against one axis.
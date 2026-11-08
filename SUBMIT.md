# SUBMIT — paste-ready Devpost + video script

Every number below is from **your** machine. Nothing is invented. The only gap
is the detector's final focal loss, marked `[DETECTOR]`, which you will have
when the run ends.

---

# PART 1 — PASTE INTO DEVPOST

## Project name
```
Cooper — Star Tracker
```

## Tagline
```
Spacecraft attitude determination, trained on data that didn't exist until we generated it.
```

## Inspiration

```
Every satellite in orbit has to answer one question before it can do anything else: which way am I pointing? Antennas need to find Earth, solar panels need to find the sun, cameras need to find their target. The sensor that answers it is a star tracker — it photographs the sky, identifies the stars it sees, and solves for its orientation to arcsecond precision.

We picked it because it has a property almost no hackathon problem has: the training data cannot be collected. We have no orbital camera, and no human being can look at a field of stars and write down the correct quaternion. If we wanted a neural network to do this, we had to manufacture the ground truth ourselves. That constraint became the whole project.
```

## What it does

```
Give it an image of a star field and it determines the camera's orientation in 3D space with no prior knowledge of where it was pointing — the "lost in space" problem. It solves this two ways and compares them honestly.

A classical pipeline does sub-pixel centroiding, identifies stars by triangle matching against a catalog of 9,074 real stars, and solves for attitude via Wahba's problem. It reaches 0.0091 degrees — about 33 arcseconds — from scratch, in 18 milliseconds.

A learned pipeline detects stars with a U-Net trained on the renderer's exact sub-pixel ground truth, and a second network regresses attitude directly while reporting its own uncertainty.

On top of both sits a Multiplicative Extended Kalman Filter that tracks attitude across frames, learns the spacecraft's rotation rate, and keeps producing an estimate even when the camera goes dark entirely.

The demo exposes sliders for sensor noise, motion blur, dead stars and hot pixels. You drag them and watch which approach survives.
```

## How we built it

```
THE RENDERER IS THE PROJECT. We built a physics-based star camera simulator: 9,074 real stars down to magnitude 6.5, a pinhole camera model, Gaussian point spread functions, brightness via Pogson's law, Poisson photon shot noise, sensor read noise, full-well saturation, motion blur and hot pixels. Each 20-degree field of view contains about 87 stars, matching the real sky.

It produces 402 perfectly-labeled images per second, so the network never sees the same image twice. Every sensor parameter is randomized per image — domain randomization — which forces the model to learn the rotation-invariant geometry of the star pattern rather than memorizing one sensor's quirks.

THE CLASSICAL BASELINE we built to be genuinely strong, not a strawman. Star identification uses triangle matching, because a single inter-star angle matches roughly a thousand catalog pairs. We built a 77,394-pair database from 2,833 guide stars with a neighbour index, so a triangle of three mutual angles resolves to a unique match. Attitude comes from Wahba's problem solved by SVD, then a RANSAC-style verification pass reprojects the whole catalog and refits on everything that matches.

THE LEARNED PIPELINE is a U-Net heatmap detector supervised by the renderer's exact star positions, and a ResNet-18 regressing a unit quaternion with a geodesic loss and a heteroscedastic variance head.

THE KALMAN FILTER is a Multiplicative EKF, the standard formulation in spacecraft attitude estimation, running a three-parameter small-angle error about a reference quaternion.
```

## Challenges we ran into

```
QUATERNIONS DOUBLE-COVER SO(3). q and -q are the same rotation, so mean squared error assigns a penalty of 1.0 to a perfect prediction. Every loss we use takes the absolute cosine instead. This is the most common way rotation regression silently fails.

NAIVE STAR IDENTIFICATION DOESN'T WORK. Our first identification stage voted on pairs of stars by angular separation and returned essentially random attitudes — around 94 degrees of error. We measured why: one angle matches about 1,000 catalog pairs, so the correct votes were buried under roughly 100,000 false ones. Adding the triangle constraint dropped the error to 0.0091 degrees.

A SIGN ERROR IN THE KALMAN FILTER. The rate coupling in the error-state transition matrix is +dt, not -dt. With the wrong sign the filter reported a 1,877 deg/s slew rate against a true 3 deg/s and diverged within seconds.

OUR OWN METRIC WAS LYING TO US. Computed in float32, the geodesic angle metric bottoms out at 0.05 degrees — coarser than the classical pipeline's real accuracy of 0.0091. We were measuring our own rounding error.

THEN APPLE'S GPU BACKEND HAS NO FLOAT64 AT ALL. Those two constraints collide. Even casting with .to("cpu", torch.float64) fails on an MPS tensor, because the dtype resolves against the source device's registry. The metric now computes in NumPy, where the widening happens somewhere float64 definitely exists.

RENDERING WAS 80x SLOWER THAN IT SHOULD HAVE BEEN. We profiled instead of guessing: OpenCV spawns a thread pool per call, and on a 224x224 image that overhead dwarfs the actual work. Pinning it to one thread took us from 5 images per second to 402.

STARS ARE 0.03% OF PIXELS. Plain binary cross-entropy on the detector heatmap converges to predicting "no star" everywhere at 99.97% accuracy. Focal loss fixed it.
```

## Accomplishments we're proud of

```
The classical pipeline reaches 0.0091 degrees — about 33 arcseconds — solving from scratch with no prior orientation knowledge, in 18 milliseconds per frame.

Sub-pixel centroiding accurate to 0.020 pixels.

The Kalman filter coasts through a full second of total sensor loss, drifting only 0.076 to 0.192 degrees, and recovers the true rotation rate as 3.16 against a true 3.0. Its self-reported uncertainty grows from 0.039 to 0.444 degrees during the blackout — it knows it is getting less sure — and it reacquires to 0.043 degrees within three frames.

An 80x rendering speedup found by profiling rather than guessing.

And we diagnosed our own negative result instead of quietly dropping it.
```

## What we learned

```
We have two findings. One is an engineering result. The other is a negative result we measured carefully rather than hid.

1. THE CLASSICAL PIPELINE IS MORE ACCURATE THAN OUR NETWORK, AND WE ARE NOT GOING TO PRETEND OTHERWISE.

It reaches 0.0091 degrees. But the two approaches fail differently. As sensor conditions degrade, the classical solver's latency climbs from 18 ms to 161 ms — nine times — because it must test more and more star-triangle hypotheses before one verifies against the catalog, and it then fails outright on 20% of frames. A network is constant-time and always returns an answer. The right system is a hybrid: the network as a fast always-available prior, the classical solver for precision whenever it can verify itself.

2. DIRECT FULL-SKY ATTITUDE REGRESSION DOES NOT WORK, AND WE FOUND OUT WHY.

We trained a ResNet-18 to regress a quaternion straight from the image. After 1,000 steps it sat at a median error of 127 degrees against a random-guessing baseline of 131, with a loss of 0.55 against a random baseline of 0.572. It was learning nothing — and we could prove it was learning nothing, because we computed what chance looked like instead of assuming.

The training loop was not the problem. Our overfit smoke test drives 24 fixed samples from 138 degrees to 0.02 in seven seconds, so the gradients, the loss and the optimizer were all fine. Two things make the task intractable in this form.

The architecture destroys the signal before it reaches a residual block. ResNet-18's stem is a 7x7 stride-2 convolution followed by a stride-2 max pool, so a 224-pixel input is 56 pixels by the time the first block sees it. Our stars have a point spread function of about one pixel. We measured it: a clean frame has 648 pixels above threshold at full resolution and 37 after the stem. Ninety-four percent of the star signal is gone before the network can use it. A backbone designed for textured natural images is the wrong tool for a sparse point pattern.

And the task requires memorizing the entire sky. Absolute attitude from a star field is not a perceptual problem, it is an association problem: the network must map about 90 unlabelled dots onto a specific 3-degree-of-freedom orientation out of the whole celestial sphere. The classical algorithm solves this with an explicit catalog lookup and a geometric constraint. Asking a CNN to absorb a 9,074-star catalog into its weights from images alone is a fundamentally harder ask than we had assumed.

3. WHERE LEARNING ACTUALLY HELPED.

Star detection. Supervision is dense, the pattern is local, and the renderer hands us pixel-perfect labels for free. The U-Net converged to a focal loss of [DETECTOR] and finds faint stars that a fixed sigma threshold misses under noise.

That is a real division of labour, and we would not have found it by assuming the neural network should do the hardest part. When ground truth cannot be collected, building the physics beats collecting the data — but you still have to point the model at the sub-problem it can actually solve.
```

## What's next for Cooper

```
Keep full resolution in the network stem, which our measurement says is the binding constraint on attitude regression.

Restrict the output space to a cone around a reference attitude — the tracking regime a real star tracker spends most of its life in, rather than the much harder lost-in-space regime.

Feed the network detected centroids as a point set instead of raw pixels, which is a far better-posed input for a sparse pattern.

Validate on real astrophotography. Trained purely on synthetic data, generalizing to actual photons on a real sensor, is the sim-to-real proof.

Deploy on a real camera and pan-tilt mount for closed-loop pointing.
```

## Built with
```
python, pytorch, numpy, scipy, opencv, streamlit, matplotlib, resnet, u-net, kalman-filter, computer-vision, synthetic-data, domain-randomization
```

---

# PART 2 — THE VIDEO (2 minutes, mandatory)

## Before you record

```bash
streamlit run app.py
```

Let it fully load once so the catalog and pair database are cached — otherwise
your first click has a 20-second pause on camera. Set the degradation slider
to 0. Close other apps. Record the browser window only, not your whole desktop.

**Mac screen recording with audio:** QuickTime → File → New Screen Recording →
click the chevron → set Microphone to your built-in mic. Or Cmd+Shift+5.

## Shot list

**0:00–0:15 — The hook.** Live tab on screen, clean conditions, solve visible.

> "Every satellite in orbit has to answer one question before it can do anything else: which way am I pointing? This is the sensor that answers it. It just determined its orientation to 33 arcseconds, from a photograph of the sky, knowing nothing beforehand."

**0:15–0:40 — The data problem.** Stay on the Live tab, click "New random attitude" a couple of times.

> "There's no training data for this. No orbital camera, and no human can look at a star field and write down a quaternion. So we built the physics instead — 9,074 real stars, a pinhole camera, Poisson photon noise, sensor read noise. Four hundred perfectly-labeled images a second."

**0:40–1:10 — Break it. This is the most important shot.** Drag the degradation slider slowly from 0 to 1. Let the solve time counter climb on camera. Keep dragging until it fails.

> "Watch what happens as the sensor degrades. Accuracy holds — but the solve time climbs from 18 milliseconds to 160, because it has to test more and more star triangles before one verifies. And then it stops working entirely."

**Do not cut the failure.** It is the point.

**1:10–1:35 — The recovery.** Tracking tab, set blackout to 10, hit Run sequence.

> "A single frame isn't the whole problem. This filter learns the rotation rate across frames. Here we black out the camera completely for a full second — and it coasts through, drifting less than two tenths of a degree, then reacquires in three frames."

**1:35–2:00 — The honest finding.** Benchmark tab, or just the plot.

> "Our neural network doesn't beat the classical solver on accuracy, and we're not going to pretend it does. But they fail differently — one degrades in latency and then dies, the other is constant-time and always answers. So the right system is a hybrid. That's a conclusion we drew from our own benchmark, and we think it's worth more than an inflated number."

## Rules

- One take. Judges watch dozens of these; a clear unpolished demo beats a slick confusing one.
- Talk over live software. No slides.
- Say numbers out loud — "33 arcseconds," "a full second of sensor loss."
- Under 2:00. Hard cap.
- Upload to YouTube **unlisted**, paste the link into Devpost.

---

# PART 3 — SUBMISSION CHECKLIST

- [ ] GitHub repo pushed and public
- [ ] Devpost: all sections above pasted
- [ ] `[DETECTOR]` replaced with the real focal loss
- [ ] Repo link added
- [ ] `benchmark.png` uploaded as the cover image
- [ ] A screenshot of the Live tab uploaded
- [ ] YouTube link added (unlisted is fine)
- [ ] Tracks selected: **Best AI/ML**, **Radial Ventures**, **Most Over Engineered**, **Spirit of the Hackathon**
- [ ] All four teammates added to the submission
- [ ] **Submitted** — not just saved as draft

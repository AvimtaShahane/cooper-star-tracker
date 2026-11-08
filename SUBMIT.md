## Project name

Cooper — Star Tracker


## Tagline

A spacecraft attitude sensor, built on training data we had to manufacture ourselves.


---

## Inspiration


We wanted a problem where we couldn't just download a dataset.

A star tracker is the sensor a satellite uses to figure out which way it's pointing. It takes a photo of the sky, works out which stars are in frame, and from that computes its orientation. Every satellite has one, because nothing else works — you can't point an antenna at Earth if you don't know which way you're facing.

What drew us in is that you cannot collect training data for it. We don't have a camera in orbit. And even if someone handed us a thousand photos of the night sky, nobody could label them: no human can look at a star field and write down the right quaternion. The labels don't exist and can't be made by hand.

So the question stopped being "can we train a model" and became "where does the data come from at all."


## What it does


You give it a photo of a star field. It tells you how the camera is oriented in 3D, with no prior knowledge of where it was pointing. In the literature that's called the lost-in-space problem.

It does this two different ways so we could compare them.

The classical pipeline finds star centroids to a fiftieth of a pixel, identifies them by matching triangles of inter-star angles against a catalog of 9,074 real stars, and solves for orientation with an SVD. It lands at 0.0091 degrees — 33 arcseconds — in 18 milliseconds.

The learned pipeline uses a U-Net to find the stars instead, trained on the exact sub-pixel positions our renderer knows for free.

A Kalman filter sits on top of both. It tracks orientation across frames, works out how fast the spacecraft is rotating, and keeps producing an answer even when the camera stops returning anything.

The demo has sliders for read noise, motion blur, dead stars and hot pixels. You drag them and watch which parts hold up.


## How we built it


Most of the work went into the renderer, because the renderer is where the data comes from.

It takes 9,074 catalog stars down to magnitude 6.5, projects them through a pinhole camera, and draws each one as a Gaussian blob with brightness from Pogson's law. Then it adds the things a real sensor does to an image: Poisson noise from photon arrival, Gaussian read noise, full-well saturation on the bright stars, motion blur if the spacecraft is slewing, and the occasional hot pixel. A 20-degree field comes out with about 87 stars in it, which matches the real sky.

It runs at 402 images a second, and every image carries an exact label because we chose the orientation before rendering it. We randomise the sensor settings on every single frame, so the model can't learn one camera's quirks — it has to learn the geometry.

For the classical side we wanted a baseline worth beating, not a strawman. Star identification turned out to be the hard part. Our first version matched pairs of stars by the angle between them, which failed badly. We built a 77,394-pair database with a neighbour index so we could ask "which star sits at angle X from A and angle Y from B" — the triangle constraint — and that worked. After an initial solve we reproject the whole catalog and refit on everything that matches, which is what gets the accuracy down to arcseconds.

The learned side is a U-Net heatmap detector, plus a ResNet-18 we tried to use for attitude regression (see below), plus a variance head so the model reports its own confidence.

The filter is a Multiplicative EKF. Quaternions have four numbers for three degrees of freedom, so a normal Kalman filter on the raw components has a singular covariance and falls apart. The MEKF runs the filter on a small-angle error about a reference instead.


## Challenges we ran into


Quaternion double cover. q and -q are the same rotation, so mean squared error gives a penalty of 1.0 to a prediction that's exactly right. Every loss we wrote uses the absolute cosine. This one is easy to miss because the training still runs, it just never converges.

Star identification by pair voting. Our first attempt returned attitudes that were off by about 94 degrees, which is roughly random. We checked why rather than guessing: a single inter-star angle matches around a thousand catalog pairs, so with 20 detected stars the real votes were buried under about 100,000 wrong ones. Switching to triangles fixed it.

A sign error in the Kalman filter. The rate coupling in the error-state transition should be +dt. With -dt the filter decided the spacecraft was spinning at 1,877 degrees per second when the truth was 3, and diverged in about five seconds.

Our accuracy metric was wrong. In float32 the geodesic angle calculation bottoms out around 0.05 degrees, which is coarser than what the pipeline actually achieves. We were measuring our own rounding error. Then it turned out Apple's GPU backend doesn't support float64 at all — even casting through .to("cpu", torch.float64) fails, because the dtype gets resolved against the source device. We moved the metric into NumPy.

Rendering was 80x slower than it should have been. We profiled instead of assuming: OpenCV starts a thread pool on every call, and for a 224x224 image that costs more than the actual filtering. Setting it to one thread took us from 5 images per second to 402.

Class imbalance in the detector. Stars are 0.03% of pixels. Ordinary cross-entropy learns to say "no star" everywhere and reports 99.97% accuracy. Focal loss fixed it.


## Accomplishments that we're proud of


0.0091 degrees — 33 arcseconds — from a single frame with no prior information, in 18 milliseconds. Real flight star trackers are in the 1 to 10 arcsecond range with proper optics and years of calibration, so we're within an order of magnitude on a laptop.

Centroids accurate to 0.020 pixels.

The Kalman filter survives a full second with no camera at all. It drifts from 0.076 to 0.192 degrees over the blackout, recovers the rotation rate as 3.16 against a true 3.0, and is back to 0.043 degrees three frames after the sensor returns. Its own uncertainty estimate grows from 0.039 to 0.444 degrees while it's blind, so it knows it's guessing.

Finding the 80x rendering bug by profiling rather than guessing at it.

And writing up the thing that didn't work, instead of leaving it out.


## What we learned


The classical pipeline is more accurate than our network. We want to say that plainly rather than bury it.

But the two fail in different ways, and that turned out to be the interesting part. As conditions get worse the classical solver doesn't gradually lose precision — it gets slower. Solve time goes from 18 milliseconds to 161, because with fewer usable stars it has to test more and more triangle hypotheses before one of them verifies against the catalog. Then at the far end it stops solving at all, on about one frame in five. A network has none of that behaviour: constant time, always returns something. So the sensible system isn't one or the other, it's the network as a fast prior with the classical solver stepping in for precision whenever it can verify itself.

The second thing we learned came from a failure.

We trained a ResNet-18 to predict orientation straight from the image. After 1,000 steps it was at 127 degrees median error. Random guessing on SO(3) is 131 degrees, and the loss sat at 0.55 against a random baseline of 0.572. It had learned essentially nothing. We only knew that for certain because we computed what chance looks like instead of assuming a falling loss curve meant progress.

The training loop was fine — our overfit test takes 24 fixed images from 138 degrees down to 0.02 in seven seconds. So we looked at the architecture. ResNet's stem is a stride-2 convolution followed by a stride-2 max pool, which means a 224-pixel input is 56 pixels before the first residual block. Our stars are about one pixel across. We measured it: 648 pixels above threshold at full resolution, 37 after the stem. Most of the signal is gone before the network can use any of it.

There's a deeper issue too. Working out absolute orientation from a star field isn't really perception, it's lookup. The network would have to hold a 9,074-star catalog in its weights and match against it, which is what the classical algorithm does explicitly with a database and a geometric constraint.

Where learning did help was detection. The U-Net got to a focal loss of 0.00063, and more to the point it's useful: under severe noise it finds 86% of the stars in frame where threshold detection finds 54%. On clean frames they're even. The gap opens as conditions degrade, which is exactly where you need it.

It gets there by accepting more false positives — precision drops from 68% to 49%. For this pipeline that's the right trade and we made it on purpose. A false detection costs almost nothing because triangle matching rejects it; it just doesn't match anything. A star you never found is gone. When the system verifies its own matches, recall is the expensive quantity.

So: dense local supervision works. Global memorisation from sparse points doesn't. We wouldn't have found that by assuming the network should handle the hardest part of the problem.

## What's next for Cooper


Keep full resolution through the network stem. Our measurement says that's what's blocking attitude regression, so it's the first thing to try.

Train on a restricted range of orientations instead of the whole sky. Real star trackers spend most of their life in tracking mode, where they already have a rough estimate — a much easier problem than lost-in-space, and a more honest one to point a network at.

Feed the network detected star positions as a point set rather than raw pixels. A list of 30 coordinates is a far better input for this than a mostly-empty image.

Run it on real astrophotos. If something trained only on our simulator works on actual photons, that's the proof the whole approach rests on.

Then put it on a real camera and a pan-tilt mount and close the loop.


## Built with

python, pytorch, numpy, scipy, opencv, streamlit, matplotlib, resnet, u-net, kalman-filter, computer-vision, synthetic-data, domain-randomization


---

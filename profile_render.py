"""profile_render.py -- find out why rendering is slow on this machine.

Run:  python profile_render.py
"""
import time
import numpy as np
import cv2
from render import (NoiseConfig, StarCatalog, StarFieldRenderer, _motion_blur,
                    random_quaternion)


def bench(fn, n=30):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


cat = StarCatalog.build()
r = StarFieldRenderer(cat)
rng = np.random.default_rng(0)
q = random_quaternion(rng)
cfg = NoiseConfig()

print(f"cv2 {cv2.__version__}, threads = {cv2.getNumThreads()}")
print(f"numpy {np.__version__}\n")

full = bench(lambda: r.render(q, rng=rng, noise=cfg))
print(f"{'full render':24} {full:8.2f} ms   -> {1000/full:.0f} img/s")

uv, mags, _ = r.visible_stars(q)
flux = r._flux(mags)
canvas = np.zeros((224, 224))
print(f"{'  visible_stars':24} {bench(lambda: r.visible_stars(q)):8.2f} ms")
print(f"{'  _draw_psf':24} {bench(lambda: r._draw_psf(canvas.copy(), uv, flux, 1.0)):8.2f} ms")
print(f"{'  poisson':24} {bench(lambda: rng.poisson(np.abs(canvas) + 5)):8.2f} ms")
print(f"{'  _motion_blur (cv2)':24} {bench(lambda: _motion_blur(canvas, 1.2, 0.5)):8.2f} ms")
print(f"{'  gaussian read noise':24} {bench(lambda: rng.normal(0, 4.0, canvas.shape)):8.2f} ms")

print("\nIf _motion_blur dominates, cv2 threading is the problem.")
print("If poisson dominates, that is expected (~2-3 ms) and fine.")
print("Target: 5-10 ms per render, i.e. 100-200 img/s.")

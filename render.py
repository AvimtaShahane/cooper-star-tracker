"""
render.py -- Physics-based synthetic star field generator.

This is the CRITICAL PATH of the project. Everything downstream consumes its
output. It turns a real star catalog plus a random spacecraft attitude into a
realistic camera image, with the attitude as a perfectly known label.

CONTRACT (do not change without telling the whole team):
    image    : np.uint8, shape (H, W), grayscale, H = W = 224 by default
    attitude : np.float64, shape (4,), unit quaternion, [w, x, y, z] order
               Rotates INERTIAL vectors into the CAMERA frame:
                   v_camera = R(q) @ v_inertial
               Camera boresight is +Z. Image x is +X, image y is +Y.

Physical modelling, in order of application:
    1. Catalog star unit vectors from right ascension / declination
    2. Rotation into the camera frame by the sampled attitude
    3. Pinhole projection with the stated focal length
    4. Optional barrel distortion
    5. Magnitude -> photon flux via the Pogson ratio (2.512 per magnitude)
    6. Point spread function: each star is a 2D Gaussian, not a single pixel
    7. Motion blur from angular rate during exposure
    8. Poisson shot noise (photon arrival statistics)
    9. Gaussian read noise, dark current offset, hot pixels
   10. Quantisation to 8-bit

Every one of those steps corresponds to a real physical effect. That is what
makes this a simulation of hardware rather than a drawing of dots.
"""

from __future__ import annotations

import numpy as np

# OpenCV spins up a thread pool per call. On a 224x224 image that overhead
# dwarfs the actual work -- it can cost 30x more than running single-threaded,
# and it gets worse under DataLoader workers, where every worker spawns its own
# pool and they fight each other for cores. One thread is correct here.
try:
    import cv2 as _cv2
    _cv2.setNumThreads(1)
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Star catalog
# ----------------------------------------------------------------------------

# The brightest real stars: (name, right ascension deg, declination deg, Vmag).
# Positions are J2000. These make the demo concrete -- you can ask the system to
# point at Vega and it means something.
BRIGHT_STARS = [
    ("Sirius", 101.287, -16.716, -1.46), ("Canopus", 95.988, -52.696, -0.74),
    ("Rigil Kentaurus", 219.902, -60.834, -0.27), ("Arcturus", 213.915, 19.182, -0.05),
    ("Vega", 279.234, 38.784, 0.03), ("Capella", 79.172, 45.998, 0.08),
    ("Rigel", 78.634, -8.202, 0.13), ("Procyon", 114.826, 5.225, 0.34),
    ("Achernar", 24.429, -57.237, 0.46), ("Betelgeuse", 88.793, 7.407, 0.50),
    ("Hadar", 210.956, -60.373, 0.61), ("Altair", 297.696, 8.868, 0.77),
    ("Acrux", 186.650, -63.099, 0.77), ("Aldebaran", 68.980, 16.509, 0.85),
    ("Antares", 247.352, -26.432, 0.96), ("Spica", 201.298, -11.161, 1.04),
    ("Pollux", 116.329, 28.026, 1.14), ("Fomalhaut", 344.413, -29.622, 1.16),
    ("Deneb", 310.358, 45.280, 1.25), ("Mimosa", 191.930, -59.689, 1.25),
    ("Regulus", 152.093, 11.967, 1.35), ("Adhara", 104.656, -28.972, 1.50),
    ("Castor", 113.650, 31.888, 1.57), ("Gacrux", 187.791, -57.113, 1.63),
    ("Shaula", 263.402, -37.104, 1.63), ("Bellatrix", 81.283, 6.350, 1.64),
    ("Elnath", 81.573, 28.608, 1.65), ("Miaplacidus", 138.300, -69.717, 1.67),
    ("Alnilam", 84.053, -1.202, 1.69), ("Alnair", 332.058, -46.961, 1.74),
    ("Alnitak", 85.190, -1.943, 1.77), ("Alioth", 193.507, 55.960, 1.77),
    ("Dubhe", 165.932, 61.751, 1.79), ("Mirfak", 51.081, 49.861, 1.79),
    ("Wezen", 107.098, -26.393, 1.84), ("Kaus Australis", 276.043, -34.385, 1.85),
    ("Sargas", 264.330, -42.998, 1.86), ("Avior", 125.628, -59.509, 1.86),
    ("Alkaid", 206.885, 49.313, 1.86), ("Menkalinan", 89.882, 44.947, 1.90),
    ("Atria", 252.166, -69.028, 1.91), ("Alhena", 99.428, 16.399, 1.93),
    ("Peacock", 306.412, -56.735, 1.94), ("Delta Velorum", 131.176, -54.709, 1.96),
    ("Mirzam", 95.675, -17.956, 1.98), ("Alphard", 141.897, -8.659, 1.98),
    ("Polaris", 37.955, 89.264, 1.98), ("Hamal", 31.793, 23.462, 2.00),
    ("Diphda", 10.897, -17.987, 2.04), ("Nunki", 283.816, -26.297, 2.05),
    ("Alpheratz", 2.097, 29.090, 2.06), ("Menkent", 211.671, -36.370, 2.06),
    ("Mirach", 17.433, 35.621, 2.07), ("Tiaki", 340.667, -46.885, 2.07),
    ("Rasalhague", 263.734, 12.560, 2.08), ("Algieba", 154.993, 19.841, 2.08),
    ("Kochab", 222.676, 74.156, 2.08), ("Saiph", 86.939, -9.670, 2.09),
    ("Almach", 30.975, 42.330, 2.10), ("Algol", 47.042, 40.956, 2.12),
    ("Denebola", 177.265, 14.572, 2.14), ("Naos", 120.896, -40.003, 2.21),
    ("Aspidiske", 139.273, -59.275, 2.21), ("Alphecca", 233.672, 26.715, 2.22),
    ("Muhlifain", 190.379, -48.960, 2.20), ("Suhail", 136.999, -43.433, 2.23),
    ("Mintaka", 83.002, -0.299, 2.23), ("Sadr", 305.557, 40.257, 2.23),
    ("Eltanin", 269.152, 51.489, 2.23), ("Schedar", 10.127, 56.537, 2.23),
    ("Mizar", 200.981, 54.925, 2.23), ("Caph", 2.295, 59.150, 2.27),
    ("Dschubba", 240.083, -22.622, 2.29), ("Larawag", 252.541, -34.293, 2.29),
    ("Merak", 165.460, 56.382, 2.37), ("Izar", 221.247, 27.074, 2.37),
    ("Ankaa", 6.571, -42.306, 2.38), ("Enif", 326.046, 9.875, 2.38),
    ("Girtab", 265.622, -39.030, 2.39), ("Scheat", 345.944, 28.083, 2.42),
    ("Sabik", 257.595, -15.725, 2.43), ("Phecda", 178.458, 53.695, 2.44),
    ("Aludra", 111.024, -29.303, 2.45), ("Markeb", 140.528, -55.011, 2.47),
    ("Navi", 14.177, 60.717, 2.47), ("Markab", 346.190, 15.205, 2.49),
    ("Aljanah", 311.552, 33.970, 2.48), ("Acrab", 241.359, -19.805, 2.56),
]

# Cumulative star counts follow log10 N(<m) = 0.506 m + 0.669, fitted to the real
# sky (48 stars brighter than mag 2, about 9100 brighter than mag 6.5).
_COUNT_SLOPE = 0.506
_COUNT_INTERCEPT = 0.669


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Right ascension / declination in degrees -> unit vectors, shape (N, 3)."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


class StarCatalog:
    """Real bright stars plus a statistically faithful faint population.

    If you manage to download the Yale Bright Star Catalog, use from_csv(). The
    procedural fill exists so the project never blocks on a download at 3 AM.
    """

    def __init__(self, vectors, magnitudes, names=None):
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.magnitudes = np.asarray(magnitudes, dtype=np.float64)
        self.names = names if names is not None else [""] * len(self.magnitudes)
        order = np.argsort(self.magnitudes)
        self.vectors = self.vectors[order]
        self.magnitudes = self.magnitudes[order]
        self.names = [self.names[i] for i in order]

    def __len__(self) -> int:
        return len(self.magnitudes)

    @classmethod
    def build(cls, mag_limit: float = 6.5, seed: int = 0) -> "StarCatalog":
        """Bright named stars, topped up procedurally to the magnitude limit."""
        rng = np.random.default_rng(seed)

        names = [s[0] for s in BRIGHT_STARS]
        ra = np.array([s[1] for s in BRIGHT_STARS])
        dec = np.array([s[2] for s in BRIGHT_STARS])
        mags = np.array([s[3] for s in BRIGHT_STARS])
        vectors = radec_to_unit(ra, dec)

        faint_start = float(mags.max())
        n_total = 10.0 ** (_COUNT_SLOPE * mag_limit + _COUNT_INTERCEPT)
        n_already = 10.0 ** (_COUNT_SLOPE * faint_start + _COUNT_INTERCEPT)
        n_fill = int(max(0, n_total - n_already))

        if n_fill > 0:
            # Inverse-transform sample magnitudes from the cumulative count law.
            u = rng.uniform(0.0, 1.0, n_fill)
            cum = n_already + u * (n_total - n_already)
            fill_mags = (np.log10(cum) - _COUNT_INTERCEPT) / _COUNT_SLOPE

            # Uniform on the sphere: z uniform, azimuth uniform.
            z = rng.uniform(-1.0, 1.0, n_fill)
            phi = rng.uniform(0.0, 2.0 * np.pi, n_fill)
            r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
            fill_vecs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)

            vectors = np.concatenate([vectors, fill_vecs], axis=0)
            mags = np.concatenate([mags, fill_mags], axis=0)
            names = names + [""] * n_fill

        return cls(vectors, mags, names)

    @classmethod
    def from_csv(cls, path: str, ra_col="RA", dec_col="Dec", mag_col="Vmag",
                 name_col=None, mag_limit: float = 6.5) -> "StarCatalog":
        """Load a real catalog CSV (Yale BSC, Hipparcos, whatever you found)."""
        import pandas as pd
        df = pd.read_csv(path)
        df = df.dropna(subset=[ra_col, dec_col, mag_col])
        df = df[df[mag_col] <= mag_limit]
        vectors = radec_to_unit(df[ra_col].to_numpy(), df[dec_col].to_numpy())
        names = df[name_col].astype(str).tolist() if name_col else None
        return cls(vectors, df[mag_col].to_numpy(), names)

    def lookup(self, name: str):
        """Unit vector of a named star, for 'point at Vega' style demos."""
        target = name.strip().lower()
        for i, n in enumerate(self.names):
            if n.lower() == target:
                return self.vectors[i]
        return None


# ----------------------------------------------------------------------------
# Quaternion helpers -- [w, x, y, z], Hamilton convention
# ----------------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w,x,y,z], w >= 0 by convention."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = quat_normalize(q)
    return -q if q[0] < 0 else q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation (Shoemake's method). Not the same as random Euler
    angles -- those cluster at the poles and would bias your training set."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    q = np.array([s2 * np.cos(2 * np.pi * u3), s1 * np.sin(2 * np.pi * u2),
                  s1 * np.cos(2 * np.pi * u2), s2 * np.sin(2 * np.pi * u3)])
    return -q if q[0] < 0 else q


def angular_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle between two attitudes, in degrees.

    The abs() handles quaternion double cover: q and -q are the SAME rotation.
    Forgetting this is the number one source of nonsense error metrics.
    """
    d = abs(float(np.dot(quat_normalize(q_a), quat_normalize(q_b))))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))


# ----------------------------------------------------------------------------
# Camera and noise configuration
# ----------------------------------------------------------------------------

class CameraModel:
    """Pinhole camera with optional radial distortion."""

    def __init__(self, width=224, height=224, fov_deg=20.0, k1=0.0):
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.k1 = float(k1)
        # Focal length in pixels from the horizontal field of view.
        self.focal = (self.width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def project(self, vecs_cam: np.ndarray):
        """Camera-frame unit vectors -> (pixels Nx2, visibility mask N)."""
        z = vecs_cam[:, 2]
        in_front = z > 1e-6
        z_safe = np.where(in_front, z, 1.0)
        xn = vecs_cam[:, 0] / z_safe
        yn = vecs_cam[:, 1] / z_safe

        if self.k1 != 0.0:
            r2 = xn * xn + yn * yn
            scale = 1.0 + self.k1 * r2
            xn, yn = xn * scale, yn * scale

        u = self.focal * xn + self.cx
        v = self.focal * yn + self.cy
        inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return np.stack([u, v], axis=-1), in_front & inside

    def pixel_to_unit(self, uv: np.ndarray) -> np.ndarray:
        """Pixel coordinates -> camera-frame unit vectors. Inverse of project()."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        xn = (uv[:, 0] - self.cx) / self.focal
        yn = (uv[:, 1] - self.cy) / self.focal
        if self.k1 != 0.0:
            r2 = xn * xn + yn * yn
            scale = 1.0 / (1.0 + self.k1 * r2)
            xn, yn = xn * scale, yn * scale
        v = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


class NoiseConfig:
    """Sensor imperfections. Every field maps to a real physical effect.

    Randomising these per-sample is DOMAIN RANDOMISATION: it forces the network
    to learn the geometric star pattern rather than memorising clean renders.
    That is the mechanism that makes synthetic training transfer to real images.
    """

    def __init__(self, psf_sigma=(0.6, 1.6), read_noise=(1.0, 8.0),
                 background=(3.0, 18.0), hot_pixel_rate=(0.0, 3e-4),
                 dropout_prob=(0.0, 0.25), false_star_count=(0, 6),
                 motion_blur_px=(0.0, 1.5), mag_limit=(5.5, 6.5),
                 gain=(0.8, 1.4), shot_noise=True):
        self.psf_sigma = psf_sigma
        self.read_noise = read_noise
        self.background = background
        self.hot_pixel_rate = hot_pixel_rate
        self.dropout_prob = dropout_prob
        self.false_star_count = false_star_count
        self.motion_blur_px = motion_blur_px
        self.mag_limit = mag_limit
        self.gain = gain
        self.shot_noise = shot_noise

    @classmethod
    def clean(cls) -> "NoiseConfig":
        """Ideal sensor. Use for sanity checks and the best-case benchmark point."""
        return cls(psf_sigma=(1.0, 1.0), read_noise=(0.0, 0.0), background=(2.0, 2.0),
                   hot_pixel_rate=(0.0, 0.0), dropout_prob=(0.0, 0.0),
                   false_star_count=(0, 0), motion_blur_px=(0.0, 0.0),
                   mag_limit=(6.5, 6.5), gain=(1.0, 1.0), shot_noise=False)

    @classmethod
    def at_level(cls, level: float) -> "NoiseConfig":
        """Interpolate clean -> severe by a single 0..1 dial.

        This is what the benchmark sweeps and what the judge's slider controls.
        """
        t = float(np.clip(level, 0.0, 1.0))
        lerp = lambda a, b: a + t * (b - a)
        return cls(
            psf_sigma=(lerp(1.0, 0.6), lerp(1.0, 2.2)),
            read_noise=(0.0, lerp(0.0, 16.0)),
            background=(2.0, lerp(2.0, 28.0)),
            hot_pixel_rate=(0.0, lerp(0.0, 8e-4)),
            dropout_prob=(0.0, lerp(0.0, 0.45)),
            false_star_count=(0, int(round(lerp(0, 12)))),
            motion_blur_px=(0.0, lerp(0.0, 3.0)),
            mag_limit=(lerp(6.5, 5.0), 6.5),
            gain=(lerp(1.0, 0.6), lerp(1.0, 1.6)),
            shot_noise=True,
        )


# ----------------------------------------------------------------------------
# The renderer
# ----------------------------------------------------------------------------

class StarFieldRenderer:
    """Attitude in, realistic camera image out."""

    def __init__(self, catalog: StarCatalog = None, camera: CameraModel = None,
                 noise: NoiseConfig = None, zero_mag_flux: float = 40000.0,
                 full_well: float = 255.0):
        self.catalog = catalog if catalog is not None else StarCatalog.build()
        self.camera = camera if camera is not None else CameraModel()
        self.noise = noise if noise is not None else NoiseConfig()
        # Electrons collected from a magnitude 0 star during one exposure.
        # Calibrated so the faintest catalog star (mag 6.5) peaks a few counts
        # above read noise -- exactly how a real star tracker is specified.
        self.zero_mag_flux = float(zero_mag_flux)
        # Pixels stop accumulating once the well fills. Bright stars therefore
        # render as saturated discs, which is what they look like on real
        # sensors and is why centroiding them needs care.
        self.full_well = float(full_well)

    # -- physics -----------------------------------------------------------

    def _flux(self, mags: np.ndarray) -> np.ndarray:
        """Pogson's law: each magnitude step is a factor of 10^0.4 = 2.512."""
        return self.zero_mag_flux * np.power(10.0, -0.4 * mags)

    def visible_stars(self, q: np.ndarray, mag_limit: float = 6.5):
        """Stars inside the field of view for attitude q.

        Returns (pixels Nx2, magnitudes N, catalog indices N).
        """
        R = quat_to_matrix(q)
        keep = self.catalog.magnitudes <= mag_limit
        vecs_cam = self.catalog.vectors[keep] @ R.T
        uv, mask = self.camera.project(vecs_cam)
        idx = np.flatnonzero(keep)[mask]
        return uv[mask], self.catalog.magnitudes[keep][mask], idx

    # -- rendering ---------------------------------------------------------

    def _draw_psf(self, canvas, uv, flux, sigma):
        """Additively draw 2D Gaussian point spread functions.

        A real optic spreads a point source over several pixels. Sub-pixel
        centroid accuracy depends on this -- a single-pixel star carries no
        sub-pixel information at all.
        """
        h, w = canvas.shape
        radius = max(2, int(np.ceil(3.0 * sigma)))
        two_s2 = 2.0 * sigma * sigma
        norm = 1.0 / (2.0 * np.pi * sigma * sigma)

        for (u, v), f in zip(uv, flux):
            u0, v0 = int(np.floor(u)), int(np.floor(v))
            x_lo, x_hi = max(0, u0 - radius), min(w, u0 + radius + 1)
            y_lo, y_hi = max(0, v0 - radius), min(h, v0 + radius + 1)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            xs = np.arange(x_lo, x_hi, dtype=np.float64) + 0.5 - u
            ys = np.arange(y_lo, y_hi, dtype=np.float64) + 0.5 - v
            gx = np.exp(-(xs * xs) / two_s2)
            gy = np.exp(-(ys * ys) / two_s2)
            canvas[y_lo:y_hi, x_lo:x_hi] += f * norm * np.outer(gy, gx)

    def render(self, q: np.ndarray, rng: np.random.Generator = None,
               noise: NoiseConfig = None, return_truth: bool = False):
        """Render the sky as seen from attitude q.

        Returns a uint8 image, or (image, truth_dict) when return_truth is set.
        truth_dict carries the exact sub-pixel star positions -- free, perfect
        supervision for the learned centroid detector.
        """
        rng = rng if rng is not None else np.random.default_rng()
        cfg = noise if noise is not None else self.noise
        pick = lambda lo_hi: rng.uniform(lo_hi[0], lo_hi[1]) if lo_hi[1] > lo_hi[0] else lo_hi[0]

        mag_limit = pick(cfg.mag_limit)
        sigma = max(0.35, pick(cfg.psf_sigma))
        gain = pick(cfg.gain)

        uv, mags, idx = self.visible_stars(q, mag_limit)

        # Dim stars the sensor simply misses this frame.
        if len(uv) and cfg.dropout_prob[1] > 0:
            p = pick(cfg.dropout_prob)
            keep = rng.uniform(size=len(uv)) > p
            uv, mags, idx = uv[keep], mags[keep], idx[keep]

        canvas = np.zeros((self.camera.height, self.camera.width), dtype=np.float64)
        canvas += pick(cfg.background)

        if len(uv):
            self._draw_psf(canvas, uv, self._flux(mags) * gain, sigma)

        # Cosmic ray hits and warm pixels read out as false point sources.
        n_false = int(round(pick(cfg.false_star_count)))
        if n_false > 0:
            fu = rng.uniform(0, self.camera.width, n_false)
            fv = rng.uniform(0, self.camera.height, n_false)
            ff = self._flux(rng.uniform(3.0, 6.0, n_false)) * gain
            self._draw_psf(canvas, np.stack([fu, fv], -1), ff, sigma)

        # Smear from angular rate during the exposure.
        blur = pick(cfg.motion_blur_px)
        if blur > 0.25:
            canvas = _motion_blur(canvas, blur, rng.uniform(0, np.pi))

        # Wells fill before readout, so clip here rather than at the very end.
        canvas = np.minimum(canvas, self.full_well)

        # Photon arrival is a Poisson process; this noise is irreducible.
        if cfg.shot_noise:
            canvas = rng.poisson(np.clip(canvas, 0, None)).astype(np.float64)

        # Readout electronics noise.
        sigma_read = pick(cfg.read_noise)
        if sigma_read > 0:
            canvas += rng.normal(0.0, sigma_read, canvas.shape)

        # Stuck-bright pixels.
        hp_rate = pick(cfg.hot_pixel_rate)
        if hp_rate > 0:
            n_hot = rng.poisson(hp_rate * canvas.size)
            if n_hot:
                ys = rng.integers(0, canvas.shape[0], n_hot)
                xs = rng.integers(0, canvas.shape[1], n_hot)
                canvas[ys, xs] += rng.uniform(120, 255, n_hot)

        image = np.clip(canvas, 0, 255).astype(np.uint8)

        if not return_truth:
            return image
        return image, {
            "uv": uv, "mags": mags, "catalog_idx": idx,
            "attitude": quat_normalize(q), "psf_sigma": sigma,
            "mag_limit": mag_limit,
        }

    def sample(self, rng: np.random.Generator = None, noise: NoiseConfig = None,
               return_truth: bool = False):
        """One random (image, attitude) training pair."""
        rng = rng if rng is not None else np.random.default_rng()
        q = random_quaternion(rng)
        out = self.render(q, rng=rng, noise=noise, return_truth=return_truth)
        return (out[0], q, out[1]) if return_truth else (out, q)

    def sequence(self, n_frames: int, rate_deg_s: float = 2.0, dt: float = 0.1,
                 rng: np.random.Generator = None, noise: NoiseConfig = None):
        """A smooth slew: frames and attitudes for the Kalman filter demo.

        Blank out a few frames from this and watch the filter coast through.
        """
        rng = rng if rng is not None else np.random.default_rng()
        q = random_quaternion(rng)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        step = np.deg2rad(rate_deg_s) * dt
        dq = np.concatenate([[np.cos(step / 2)], axis * np.sin(step / 2)])

        frames, attitudes = [], []
        for _ in range(n_frames):
            frames.append(self.render(q, rng=rng, noise=noise))
            attitudes.append(q.copy())
            q = quat_normalize(quat_multiply(dq, q))
        return np.stack(frames), np.stack(attitudes)


def _motion_blur(img: np.ndarray, length: float, angle_rad: float) -> np.ndarray:
    """Convolve with a line kernel: the smear a rotating spacecraft produces."""
    try:
        import cv2
    except ImportError:
        return img
    k = max(3, int(round(length)) | 1)
    kernel = np.zeros((k, k), dtype=np.float64)
    cx = cy = k // 2
    for t in np.linspace(-length / 2, length / 2, max(8, 4 * k)):
        x = int(round(cx + t * np.cos(angle_rad)))
        y = int(round(cy + t * np.sin(angle_rad)))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] += 1.0
    s = kernel.sum()
    if s <= 0:
        return img
    return cv2.filter2D(img, -1, kernel / s)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    cat = StarCatalog.build()
    print(f"catalog: {len(cat)} stars, mag {cat.magnitudes.min():.2f} "
          f"to {cat.magnitudes.max():.2f}")

    r = StarFieldRenderer(cat)
    rng = np.random.default_rng(42)

    counts = []
    for _ in range(200):
        q = random_quaternion(rng)
        uv, _, _ = r.visible_stars(q)
        counts.append(len(uv))
    print(f"stars in a {r.camera.fov_deg:.0f} deg FOV: "
          f"mean {np.mean(counts):.1f}, min {np.min(counts)}, max {np.max(counts)}")

    t0 = time.time()
    for _ in range(100):
        r.sample(rng)
    dt = time.time() - t0
    print(f"render rate: {100/dt:.0f} images/sec ({dt*10:.1f} ms each)")

    q = random_quaternion(rng)
    print(f"round trip quat->matrix->quat err: "
          f"{angular_error_deg(q, matrix_to_quat(quat_to_matrix(q))):.3e} deg")
    print(f"double cover check (q vs -q): {angular_error_deg(q, -q):.3e} deg")

    v = cat.lookup("Vega")
    print(f"Vega unit vector: {np.round(v, 4)}")
"""
classical.py -- The non-learned baseline. This is what the CNN has to beat.

Three stages, all pure geometry and linear algebra:

  1. CENTROIDING   image -> sub-pixel star positions
                   Background estimate, sigma threshold, connected components,
                   intensity-weighted centroid.

  2. IDENTIFICATION centroids -> which catalog star is which
                   The classic "lost in space" problem: you know nothing about
                   your orientation. Solved by angular-distance pair voting.
                   The angle between two stars is rotation invariant, so it is
                   a fingerprint you can look up.

  3. ATTITUDE      matched vector pairs -> rotation
                   This is WAHBA'S PROBLEM (1965), solved optimally by SVD
                   (the Kabsch algorithm). Six lines of NumPy.

Keeping this honest matters: if you weaken the baseline, beating it proves
nothing. This implementation uses verify-and-refine, which is what real flight
software does.
"""

from __future__ import annotations

import numpy as np

from render import (CameraModel, StarCatalog, matrix_to_quat, quat_to_matrix)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------
# Stage 1: centroiding
# ----------------------------------------------------------------------------

def detect_centroids(image: np.ndarray, sigma_threshold: float = 4.0,
                     min_area: int = 2, max_stars: int = 30,
                     return_mask: bool = False):
    """Find stars in an image to sub-pixel accuracy.

    Returns (centroids Nx2 as (u, v) pixels, fluxes N), brightest first.

    min_area >= 2 is deliberate: single stuck-bright pixels are hot pixels, not
    stars. A real PSF always covers several pixels. That one filter removes most
    sensor artefacts for free.
    """
    img = image.astype(np.float64)

    # Robust background: median and a median-absolute-deviation sigma, so a few
    # bright stars cannot drag the threshold up.
    bg = np.median(img)
    mad = np.median(np.abs(img - bg))
    noise = max(1.4826 * mad, 0.5)
    threshold = bg + sigma_threshold * noise

    mask = (img > threshold).astype(np.uint8)
    if mask.sum() == 0:
        empty = (np.zeros((0, 2)), np.zeros(0))
        return (*empty, mask) if return_mask else empty

    if cv2 is not None:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        regions = [(labels == k, stats[k, cv2.CC_STAT_AREA]) for k in range(1, n_labels)]
    else:  # pragma: no cover
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        regions = [(labels == k, int((labels == k).sum())) for k in range(1, n + 1)]

    sub = img - bg
    centroids, fluxes = [], []
    for region, area in regions:
        if area < min_area:
            continue
        ys, xs = np.nonzero(region)
        w = sub[ys, xs]
        w = np.clip(w, 0, None)
        total = w.sum()
        if total <= 0:
            continue
        # Intensity-weighted first moment, +0.5 for pixel centre convention.
        centroids.append([(w * (xs + 0.5)).sum() / total,
                          (w * (ys + 0.5)).sum() / total])
        fluxes.append(total)

    if not centroids:
        empty = (np.zeros((0, 2)), np.zeros(0))
        return (*empty, mask) if return_mask else empty

    centroids = np.array(centroids, dtype=np.float64)
    fluxes = np.array(fluxes, dtype=np.float64)
    order = np.argsort(-fluxes)[:max_stars]
    out = (centroids[order], fluxes[order])
    return (*out, mask) if return_mask else out


# ----------------------------------------------------------------------------
# Stage 2: identification by angular-distance pair voting
# ----------------------------------------------------------------------------

class PairDatabase:
    """Angular distances between catalog star pairs, sorted for range lookup.

    The angle between two stars does not change when you rotate the camera, so
    a measured angle is a searchable fingerprint. This is the core trick behind
    every lost-in-space star identification algorithm.
    """

    def __init__(self, catalog: StarCatalog, max_angle_deg: float = 16.0,
                 mag_limit: float = 5.5):
        self.catalog = catalog
        self.max_angle = np.deg2rad(max_angle_deg)
        self.guide_idx = np.flatnonzero(catalog.magnitudes <= mag_limit)
        vecs = catalog.vectors[self.guide_idx]

        # Pairwise dot products in blocks, keeping only close-enough pairs.
        cos_max = np.cos(self.max_angle)
        a_list, b_list, d_list = [], [], []
        block = 512
        for start in range(0, len(vecs), block):
            chunk = vecs[start:start + block]
            dots = np.clip(chunk @ vecs.T, -1.0, 1.0)
            rows, cols = np.nonzero(dots > cos_max)
            rows_global = rows + start
            keep = rows_global < cols  # upper triangle only, no self-pairs
            rows_global, cols = rows_global[keep], cols[keep]
            if len(rows_global):
                a_list.append(rows_global)
                b_list.append(cols)
                d_list.append(np.arccos(dots[rows[keep], cols]))

        self.pair_a = np.concatenate(a_list) if a_list else np.zeros(0, int)
        self.pair_b = np.concatenate(b_list) if b_list else np.zeros(0, int)
        dists = np.concatenate(d_list) if d_list else np.zeros(0)

        order = np.argsort(dists)
        self.dists = dists[order]
        self.pair_a = self.pair_a[order]
        self.pair_b = self.pair_b[order]

        # Per-star neighbour lists, each sorted by distance. This is what turns
        # the near-useless single-angle lookup into a sharp triangle lookup: it
        # lets us ask "which star sits at angle X from A and angle Y from B".
        n_guide = len(self.guide_idx)
        both_a = np.concatenate([self.pair_a, self.pair_b])
        both_b = np.concatenate([self.pair_b, self.pair_a])
        both_d = np.concatenate([self.dists, self.dists])
        sort = np.lexsort((both_d, both_a))
        both_a, both_b, both_d = both_a[sort], both_b[sort], both_d[sort]
        starts = np.searchsorted(both_a, np.arange(n_guide), "left")
        ends = np.searchsorted(both_a, np.arange(n_guide), "right")
        self._nbr_slice = list(zip(starts, ends))
        self._nbr_idx = both_b
        self._nbr_dist = both_d

    def __len__(self) -> int:
        return len(self.dists)

    def query(self, angle: float, tol: float):
        """Catalog pairs whose separation is angle +/- tol (as guide indices)."""
        lo = np.searchsorted(self.dists, angle - tol, "left")
        hi = np.searchsorted(self.dists, angle + tol, "right")
        return self.pair_a[lo:hi], self.pair_b[lo:hi]

    def neighbours(self, star: int, angle: float, tol: float) -> np.ndarray:
        """Guide stars sitting at angle +/- tol from `star`."""
        s, e = self._nbr_slice[star]
        d = self._nbr_dist[s:e]
        lo = s + np.searchsorted(d, angle - tol, "left")
        hi = s + np.searchsorted(d, angle + tol, "right")
        return self._nbr_idx[lo:hi]


def solve_wahba(body: np.ndarray, inertial: np.ndarray,
                weights: np.ndarray = None) -> np.ndarray:
    """Optimal rotation taking inertial vectors onto body vectors.

    Wahba's problem: minimise sum_i w_i * |b_i - R r_i|^2 over rotations R.
    The SVD solution is exact and takes six lines. The det() correction stops
    the solver returning a reflection instead of a rotation.

    Returns a quaternion [w, x, y, z].
    """
    body = np.atleast_2d(body)
    inertial = np.atleast_2d(inertial)
    w = np.ones(len(body)) if weights is None else np.asarray(weights, float)

    B = (body * w[:, None]).T @ inertial
    U, _, Vt = np.linalg.svd(B)
    M = np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(Vt)])
    R = U @ M @ Vt
    return matrix_to_quat(R)


def _triple_order(n_stars: int, max_stars: int = 9):
    """Candidate star triples, brightest combinations first.

    Detections arrive sorted by flux, so low indices are bright stars: better
    centroids and more likely to be in the guide catalog at all. Trying those
    first means the search usually succeeds on the first few triples.
    """
    import itertools
    m = min(n_stars, max_stars)
    triples = list(itertools.combinations(range(m), 3))
    triples.sort(key=lambda t: (sum(t), t))
    return triples


class ClassicalTracker:
    """Full classical pipeline: image in, attitude out."""

    def __init__(self, catalog: StarCatalog, camera: CameraModel,
                 pairdb: PairDatabase = None, angle_tol_deg: float = 0.10,
                 match_tol_deg: float = 0.20, min_matches: int = 6,
                 accept_matches: int = 10, max_triple_stars: int = 9,
                 max_pair_candidates: int = 3000):
        self.catalog = catalog
        self.camera = camera
        self.pairdb = pairdb if pairdb is not None else PairDatabase(catalog)
        self.angle_tol = np.deg2rad(angle_tol_deg)     # inter-star angle tolerance
        self.match_tol = np.deg2rad(match_tol_deg)     # reprojection match radius
        self.min_matches = min_matches                 # below this, reject outright
        self.accept_matches = accept_matches           # above this, stop searching
        self.max_triple_stars = max_triple_stars       # brightest N used for triples
        self.max_pair_candidates = max_pair_candidates # skip hopelessly ambiguous angles

    def solve(self, image: np.ndarray, max_stars: int = 20,
              sigma_threshold: float = 4.0, verbose: bool = False):
        """Estimate attitude from one frame.

        Returns a dict with keys: quat (or None), n_detected, n_matched,
        confidence, residual_deg.
        """
        centroids, fluxes = detect_centroids(
            image, sigma_threshold=sigma_threshold, max_stars=max_stars)
        out = {"quat": None, "n_detected": len(centroids), "n_matched": 0,
               "confidence": 0.0, "residual_deg": float("nan"),
               "centroids": centroids}
        if len(centroids) < 3:
            return out

        body = self.camera.pixel_to_unit(centroids)
        n = len(body)
        angles = np.arccos(np.clip(body @ body.T, -1.0, 1.0))

        # --- triangle matching ------------------------------------------
        # One inter-star angle matches ~1000 catalog pairs, so voting on pairs
        # alone is hopeless. Three mutual angles are very nearly unique. For a
        # candidate pair (a, b) matching observed (i, j), we ask the neighbour
        # index which catalog star sits at the right angle from BOTH -- the
        # intersection is usually empty or a single star.
        best = None
        triples = _triple_order(n, self.max_triple_stars)

        for (i, j, k) in triples:
            d_ij, d_ik, d_jk = angles[i, j], angles[i, k], angles[j, k]
            if max(d_ij, d_ik, d_jk) > self.pairdb.max_angle:
                continue
            pa, pb = self.pairdb.query(d_ij, self.angle_tol)
            if len(pa) == 0 or len(pa) > self.max_pair_candidates:
                continue

            for a, b in zip(pa, pb):
                # Both orientations of the unordered pair must be tried.
                for (sa, sb) in ((a, b), (b, a)):
                    na = self.pairdb.neighbours(sa, d_ik, self.angle_tol)
                    if len(na) == 0:
                        continue
                    nb = self.pairdb.neighbours(sb, d_jk, self.angle_tol)
                    if len(nb) == 0:
                        continue
                    for sc in np.intersect1d(na, nb, assume_unique=False):
                        cat3 = self.pairdb.guide_idx[[sa, sb, sc]]
                        q = solve_wahba(body[[i, j, k]], self.catalog.vectors[cat3])
                        cand = self._verify(body, q)
                        if cand is not None and (best is None
                                                 or cand[1] > best[1]):
                            best = cand
                    if best is not None and best[1] >= self.accept_matches:
                        break
                if best is not None and best[1] >= self.accept_matches:
                    break
            if best is not None and best[1] >= self.accept_matches:
                break

        if best is None:
            return out

        q, n_matched, residual = best
        out.update(quat=q, n_matched=int(n_matched),
                   residual_deg=float(np.degrees(residual)),
                   confidence=float(n_matched) / max(1, len(centroids)))
        if verbose:
            print(f"detected={out['n_detected']} matched={out['n_matched']} "
                  f"residual={out['residual_deg']:.4f} deg")
        return out

    def _verify(self, body: np.ndarray, q: np.ndarray, refine_passes: int = 3):
        """Reproject the catalog under q and rebuild the fit from what matches.

        This is the RANSAC-style verification step. A wrong triangle explains
        only its own three stars; a correct one explains nearly every detection
        in the frame. Counting matches is therefore a reliable accept test, and
        refitting on all of them is what delivers the final accuracy.
        """
        n_matched, residual = 0, float("nan")
        for _ in range(refine_passes):
            R = quat_to_matrix(q)
            uv, visible = self.camera.project(self.catalog.vectors @ R.T)
            vis_idx = np.flatnonzero(visible)
            if len(vis_idx) < 3:
                return None
            pred = self.camera.pixel_to_unit(uv[visible])

            dots = np.clip(body @ pred.T, -1.0, 1.0)
            nearest = dots.argmax(axis=1)
            seps = np.arccos(np.clip(dots[np.arange(len(body)), nearest], -1, 1))
            good = seps < self.match_tol
            if good.sum() < 3:
                return None

            # Inverse-variance style weighting: tight matches count for more.
            w = 1.0 / (seps[good] + 1e-7)
            q = solve_wahba(body[good], self.catalog.vectors[vis_idx[nearest[good]]], w)
            n_matched, residual = int(good.sum()), float(seps[good].mean())

        if n_matched < self.min_matches:
            return None
        return q, n_matched, residual


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from render import (NoiseConfig, StarFieldRenderer, angular_error_deg,
                        random_quaternion)

    cat = StarCatalog.build()
    cam = CameraModel()
    rnd = StarFieldRenderer(cat, cam)

    t0 = time.time()
    db = PairDatabase(cat, max_angle_deg=16.0, mag_limit=5.5)
    print(f"pair database: {len(db):,} pairs from {len(db.guide_idx):,} guide "
          f"stars, built in {time.time()-t0:.1f}s")

    tracker = ClassicalTracker(cat, cam, db)
    rng = np.random.default_rng(0)

    # Centroid accuracy against known truth.
    img, truth = rnd.render(random_quaternion(rng), rng=np.random.default_rng(3),
                            noise=NoiseConfig.clean(), return_truth=True)
    cents, _ = detect_centroids(img)
    errs = [np.min(np.linalg.norm(truth["uv"] - c, axis=1)) for c in cents]
    print(f"centroiding: {len(cents)} found of {len(truth['uv'])} rendered, "
          f"median error {np.median(errs):.3f} px")

    for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cfg = NoiseConfig.at_level(level)
        errors, fails, times = [], 0, []
        for k in range(25):
            q = random_quaternion(rng)
            im = rnd.render(q, rng=rng, noise=cfg)
            t0 = time.time()
            res = tracker.solve(im)
            times.append(time.time() - t0)
            if res["quat"] is None:
                fails += 1
            else:
                errors.append(angular_error_deg(q, res["quat"]))
        solved = np.array(errors)
        ok = solved[solved < 1.0] if len(solved) else solved
        print(f"noise {level:.2f}: solved {len(solved)}/25, "
              f"accurate(<1deg) {len(ok)}/25, "
              f"median err {np.median(solved) if len(solved) else float('nan'):.4f} deg, "
              f"{np.mean(times)*1000:.0f} ms/frame")

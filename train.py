"""
train.py -- Training on infinite self-generated data.

The defining feature of this project: there is no dataset on disk. Every batch
is rendered fresh, with freshly randomised sensor parameters. Consequences
worth saying out loud in the pitch:

  * The model never sees the same image twice, so it cannot overfit in the
    usual sense -- train and validation loss track each other.
  * Labels are exact. No human could look at a star field and write down the
    quaternion, so hand labelling was never an option.
  * Noise severity is a knob. Curriculum learning (easy -> hard) comes free.

Usage
-----
    python train.py --smoke                  # 2 min: proves the loop learns
    python train.py --task attitude --steps 8000
    python train.py --task detector --steps 3000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import (AttitudeNet, StarDetector, focal_bce, geodesic_angle_deg,
                   heteroscedastic_loss, make_heatmap, quat_cosine_loss)
from render import (CameraModel, NoiseConfig, StarCatalog, StarFieldRenderer,
                    random_quaternion)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class StarDataset(Dataset):
    """Infinite synthetic star fields, rendered on demand.

    Map-style with a virtual length so DataLoader's worker sharding and
    shuffling work normally. Indexing is seeded by index, so a validation set
    built with fixed_seed=True is byte-identical run to run -- essential, or
    your benchmark numbers move for no reason.
    """

    def __init__(self, length: int = 20000, task: str = "attitude",
                 image_size: int = 224, fov_deg: float = 20.0,
                 noise_level: float = None, fixed_seed: bool = False,
                 seed_offset: int = 0, heatmap_sigma: float = 1.2):
        self.length = int(length)
        self.task = task
        self.fixed_seed = fixed_seed
        self.seed_offset = seed_offset
        self.noise_level = noise_level
        self.heatmap_sigma = heatmap_sigma
        self.image_size = image_size
        self.fov_deg = fov_deg
        self._renderer = None  # built lazily, once per worker process

    @property
    def renderer(self) -> StarFieldRenderer:
        if self._renderer is None:
            cat = StarCatalog.build()
            cam = CameraModel(self.image_size, self.image_size, self.fov_deg)
            self._renderer = StarFieldRenderer(cat, cam)
        return self._renderer

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.fixed_seed:
            seed = self.seed_offset + idx
        else:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item()) + wid * 7919
        rng = np.random.default_rng(seed)

        # Sample a noise severity per image unless one was pinned. Training
        # across the whole range is domain randomisation; it is what makes the
        # model robust instead of tuned to one sensor.
        level = (self.noise_level if self.noise_level is not None
                 else rng.uniform(0.0, 1.0))
        cfg = NoiseConfig.at_level(level)

        q = random_quaternion(rng)
        img, truth = self.renderer.render(q, rng=rng, noise=cfg, return_truth=True)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        if self.task == "attitude":
            return x, torch.from_numpy(q.astype(np.float32)), float(level)

        hm = make_heatmap(truth["uv"], img.shape, self.heatmap_sigma)
        return x, torch.from_numpy(hm).unsqueeze(0), float(level)


def make_loaders(task: str, batch_size: int, workers: int, image_size: int,
                 train_len: int, val_len: int = 512):
    train = StarDataset(train_len, task, image_size)
    val = StarDataset(val_len, task, image_size, fixed_seed=True,
                      seed_offset=10_000_000)
    kw = dict(batch_size=batch_size, num_workers=workers,
              pin_memory=torch.cuda.is_available(),
              persistent_workers=workers > 0, drop_last=True)
    return (DataLoader(train, shuffle=False, **kw),
            DataLoader(val, shuffle=False, **kw))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_attitude(steps=8000, batch_size=64, lr=3e-4, workers=4,
                   image_size=224, device=None, out="attitude_net.pt",
                   pretrained=True, log_every=50, val_every=500):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[attitude] device={device} steps={steps} batch={batch_size}")

    net = AttitudeNet(pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    train_loader, val_loader = make_loaders("attitude", batch_size, workers,
                                            image_size, steps * batch_size)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best, step, t0 = float("inf"), 0, time.time()
    net.train()
    for x, q, _ in train_loader:
        if step >= steps:
            break
        x, q = x.to(device, non_blocking=True), q.to(device, non_blocking=True)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            pred, log_var = net(x)
            loss = heteroscedastic_loss(pred, q, log_var).mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % log_every == 0:
            with torch.no_grad():
                deg = geodesic_angle_deg(pred, q).median().item()
            print(f"  step {step:6d}/{steps} loss {loss.item():8.4f} "
                  f"median_err {deg:8.3f} deg  lr {sched.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/step:.2f}s/step)")

        if step % val_every == 0 or step == steps:
            stats = evaluate_attitude(net, val_loader, device)
            print(f"  [val] median {stats['median_deg']:.4f} deg | "
                  f"mean {stats['mean_deg']:.4f} | <1deg {stats['frac_sub1']*100:.1f}% "
                  f"| unc-corr {stats['uncertainty_corr']:+.3f}")
            if stats["median_deg"] < best:
                best = stats["median_deg"]
                torch.save({"model": net.state_dict(), "stats": stats}, out)
                print(f"  saved -> {out}")
            net.train()

    print(f"[attitude] done in {(time.time()-t0)/60:.1f} min, best "
          f"median {best:.4f} deg")
    return net


@torch.no_grad()
def evaluate_attitude(net, loader, device):
    """Accuracy plus the uncertainty-vs-error correlation.

    That correlation is the headline number for the uncertainty head: it is
    the evidence the model knows when it is wrong.
    """
    net.eval()
    errs, sigmas = [], []
    for x, q, _ in loader:
        x, q = x.to(device), q.to(device)
        pred, log_var = net(x)
        errs.append(geodesic_angle_deg(pred, q).cpu().numpy())
        sigmas.append(torch.exp(0.5 * log_var).float().cpu().numpy())
    errs = np.concatenate(errs)
    sigmas = np.concatenate(sigmas)
    corr = (float(np.corrcoef(np.log(sigmas + 1e-9), np.log(errs + 1e-9))[0, 1])
            if errs.std() > 0 and sigmas.std() > 1e-12 else 0.0)
    return {"median_deg": float(np.median(errs)), "mean_deg": float(errs.mean()),
            "frac_sub1": float((errs < 1.0).mean()),
            "frac_sub01": float((errs < 0.1).mean()),
            "uncertainty_corr": corr}


def train_detector(steps=3000, batch_size=16, lr=1e-3, workers=4,
                   image_size=224, device=None, out="detector.pt",
                   log_every=50, val_every=500):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[detector] device={device} steps={steps} batch={batch_size}")

    net = StarDetector().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    train_loader, val_loader = make_loaders("detector", batch_size, workers,
                                            image_size, steps * batch_size, 128)

    best, step, t0 = float("inf"), 0, time.time()
    net.train()
    for x, hm, _ in train_loader:
        if step >= steps:
            break
        x, hm = x.to(device), hm.to(device)
        loss = focal_bce(net(x), hm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1

        if step % log_every == 0:
            print(f"  step {step:6d}/{steps} focal {loss.item():.6f} "
                  f"({(time.time()-t0)/step:.2f}s/step)")
        if step % val_every == 0 or step == steps:
            vl = float(np.mean([focal_bce(net(a.to(device)), b.to(device)).item()
                                for a, b, _ in val_loader]))
            print(f"  [val] focal {vl:.6f}")
            if vl < best:
                best = vl
                torch.save({"model": net.state_dict()}, out)
                print(f"  saved -> {out}")
            net.train()

    print(f"[detector] done in {(time.time()-t0)/60:.1f} min")
    return net


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

def smoke_test(device=None, n=24, steps=150, image_size=112):
    """Can the loop overfit a tiny fixed batch?

    This is the standard first check on any training pipeline, and it is worth
    running BEFORE a long job. If a model cannot drive the loss to near zero on
    24 samples it has memorised, the bug is in the code, not the data or the
    hyperparameters -- and you have found it in two minutes instead of four
    hours. Do this at hour 7, not hour 16.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] overfitting {n} fixed samples on {device}; "
          f"loss must collapse\n")

    ds = StarDataset(n, "attitude", image_size, fixed_seed=True, seed_offset=1234)
    xs = torch.stack([ds[i][0] for i in range(n)]).to(device)
    qs = torch.stack([ds[i][1] for i in range(n)]).to(device)

    net = AttitudeNet(pretrained=False, predict_uncertainty=False).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    net.train()
    t0 = time.time()
    first = None
    for s in range(1, steps + 1):
        pred, _ = net(xs)
        loss = quat_cosine_loss(pred, qs).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        if s % 25 == 0 or s == 1:
            deg = geodesic_angle_deg(pred, qs).median().item()
            print(f"  step {s:4d}  loss {loss.item():.6f}  median_err {deg:8.3f} deg")

    final = loss.item()
    ok = final < 0.5 * first
    print(f"\n[smoke] {first:.4f} -> {final:.6f} in {time.time()-t0:.0f}s")
    print(f"[smoke] {'PASS - the training loop learns' if ok else 'FAIL - debug before launching a long run'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["attitude", "detector"], default="attitude")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2-minute overfit check")
    a = p.parse_args()

    if a.smoke:
        raise SystemExit(0 if smoke_test(a.device) else 1)

    if a.task == "attitude":
        train_attitude(a.steps, a.batch_size, a.lr, a.workers, a.image_size,
                       a.device, a.out or "attitude_net.pt",
                       pretrained=not a.no_pretrained)
    else:
        train_detector(a.steps, a.batch_size, a.lr, a.workers, a.image_size,
                       a.device, a.out or "detector.pt")

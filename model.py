"""
model.py -- The learned half of the system.

Three learned components, so the ML story is not resting on a single network:

  1. AttitudeNet   image -> unit quaternion + predicted uncertainty
                   ResNet backbone, geodesic loss, heteroscedastic variance
                   head. Constant-time: no search, no catalog lookup.

  2. StarDetector  image -> per-pixel star probability heatmap (U-Net)
                   Supervised by the renderer's exact sub-pixel truth, which
                   costs nothing to produce and is impossible to obtain by hand.

  3. Uncertainty   AttitudeNet's second output, trained jointly
                   A system that knows when it is wrong is deployable; one that
                   fails silently is not. It also feeds the Kalman filter as a
                   measurement covariance, linking the learned and classical
                   halves of the project.

THE LOSS IS THE PART TO GET RIGHT. Quaternions double-cover SO(3): q and -q are
the same rotation. Plain MSE punishes a perfect prediction (see the demo in
__main__). Everything here uses |dot| based losses instead.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Rotation losses and metrics
# ----------------------------------------------------------------------------

def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """Project raw network output onto the unit sphere."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - |cos| between quaternions. Per-sample, shape (B,).

    This is the TRAINING loss. It is monotone in the true angular error and,
    unlike arccos, has a bounded gradient everywhere -- arccos has infinite
    slope as the prediction approaches perfect, which destabilises training
    exactly when the model starts doing well.

    The abs() is the double-cover fix. Remove it and the model fights itself.
    """
    pred = normalize_quat(pred)
    target = normalize_quat(target)
    # clamp_max(1.0): float rounding can push |dot| a hair above 1 and make the
    # loss negative, which looks alarming in logs for no real reason.
    return 1.0 - (pred * target).sum(-1).abs().clamp(max=1.0)


def geodesic_angle_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """True angular error in DEGREES. Per-sample, shape (B,).

    This is the REPORTING metric, not the training loss. Report degrees in your
    pitch: a judge can feel 0.05 degrees, nobody can feel a loss value of 0.003.

    Computed in float64 on purpose. In float32 the arccos clamp puts a floor of
    about 0.05 deg on this metric -- coarser than the classical pipeline's real
    accuracy, so you would be measuring your own rounding error.

    Computed in NumPy rather than torch. Apple's MPS backend has no float64 at
    all, and even `.to("cpu", torch.float64)` fails on an MPS tensor because the
    dtype is resolved against the source device's type registry. Dropping to
    NumPy sidesteps the whole problem: .cpu().numpy() always yields float32 on
    the host, and the widening happens where float64 definitely exists. This is
    a metric and never a loss, so the round trip costs nothing.

    Returns a CPU torch tensor so callers can still use .median(), .item(),
    .cpu().numpy() unchanged.
    """
    p = pred.detach().cpu().numpy().astype(np.float64)
    t = target.detach().cpu().numpy().astype(np.float64)
    p /= np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-12)
    t /= np.linalg.norm(t, axis=-1, keepdims=True).clip(1e-12)
    d = np.abs((p * t).sum(-1)).clip(max=1.0 - 1e-15)
    return torch.from_numpy(np.degrees(2.0 * np.arccos(d)))


def heteroscedastic_loss(pred: torch.Tensor, target: torch.Tensor,
                         log_var: torch.Tensor) -> torch.Tensor:
    """Attitude loss weighted by the model's own predicted uncertainty.

    L = exp(-s) * cos_loss + 0.5 * s      where s = log(variance)

    The model may excuse a hard sample by raising s, but pays 0.5*s for doing
    so. It cannot cheat by always claiming uncertainty. After training, s
    correlates with real error -- that correlation is the plot that sells this
    feature to judges (Kendall & Gal, 2017).
    """
    base = quat_cosine_loss(pred, target)
    return torch.exp(-log_var) * base + 0.5 * log_var


# ----------------------------------------------------------------------------
# Attitude regression network
# ----------------------------------------------------------------------------

class AttitudeNet(nn.Module):
    """Star image -> attitude quaternion + log-variance.

    Backbone: ResNet-18 adapted to a single grayscale channel. Pretrained
    ImageNet weights help even though stars look nothing like photographs --
    the early edge and blob filters transfer, and it converges in far fewer
    epochs than training from scratch, which matters on a 24-hour budget.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 predict_uncertainty: bool = True):
        super().__init__()
        self.predict_uncertainty = predict_uncertainty

        try:
            import torchvision.models as tvm
            weights = "DEFAULT" if pretrained else None
            net = getattr(tvm, backbone)(weights=weights)
            feat_dim = net.fc.in_features
            # Collapse RGB weights to one channel by summing, which preserves
            # the learned filter response instead of discarding two thirds.
            old = net.conv1.weight.data
            net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            if pretrained:
                net.conv1.weight.data = old.sum(dim=1, keepdim=True)
            net.fc = nn.Identity()
            self.backbone = net
        except Exception as exc:  # offline, or torchvision missing
            print(f"[AttitudeNet] torchvision unavailable ({exc}); "
                  f"using built-in compact ResNet")
            self.backbone = _CompactResNet()
            feat_dim = self.backbone.out_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))
        if predict_uncertainty:
            self.var_head = nn.Sequential(
                nn.Linear(feat_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):
        """x: (B, 1, H, W) float in [0, 1]. Returns (quat (B,4), log_var (B,))."""
        f = self.backbone(x)
        q = normalize_quat(self.head(f))
        if not self.predict_uncertainty:
            return q, torch.zeros(len(q), device=q.device)
        # Clamp keeps exp(-log_var) from exploding early in training.
        return q, self.var_head(f).squeeze(-1).clamp(-8.0, 6.0)

    @torch.no_grad()
    def predict(self, image: np.ndarray, device="cpu"):
        """Single uint8 image -> (quaternion as numpy [w,x,y,z], sigma)."""
        self.eval()
        x = torch.from_numpy(image.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(device)
        q, log_var = self(x)
        return q[0].cpu().numpy().astype(np.float64), float(
            torch.exp(0.5 * log_var[0]).cpu())


class _CompactResNet(nn.Module):
    """Fallback backbone for when torchvision cannot download weights."""

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [nn.Conv2d(1, widths[0], 7, 2, 3, bias=False),
                        nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, 2, 1)], widths[0]
        for c_out in widths:
            layers += [_ResBlock(c_in, c_out, stride=1 if c_in == c_out else 2),
                       _ResBlock(c_out, c_out)]
            c_in = c_out
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)
        self.out_dim = c_in

    def forward(self, x):
        return self.net(x).flatten(1)


class _ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.skip = (nn.Sequential() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                   nn.BatchNorm2d(c_out)))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(y)) + self.skip(x), inplace=True)


# ----------------------------------------------------------------------------
# Learned centroid detector
# ----------------------------------------------------------------------------

class StarDetector(nn.Module):
    """Small U-Net: image -> star-probability heatmap, same resolution.

    Replaces OpenCV thresholding. The renderer knows every star's exact
    sub-pixel position, so supervision is free and pixel-perfect. Under heavy
    noise this finds faint stars a fixed sigma threshold misses, which is the
    comparison to show.
    """

    def __init__(self, base: int = 16):
        super().__init__()
        self.e1 = _ConvBlock(1, base)
        self.e2 = _ConvBlock(base, base * 2)
        self.e3 = _ConvBlock(base * 2, base * 4)
        self.d2 = _ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = _ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        up = F.interpolate(e3, size=e2.shape[-2:], mode="nearest")
        d2 = self.d2(torch.cat([up, e2], 1))
        up = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.d1(torch.cat([up, e1], 1))
        return self.out(d1)  # logits; apply sigmoid outside

    @torch.no_grad()
    def detect(self, image: np.ndarray, threshold: float = 0.5, device="cpu",
               max_stars: int = 30):
        """uint8 image -> (centroids Nx2, scores N), sub-pixel, brightest first."""
        self.eval()
        x = torch.from_numpy(image.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(device)
        prob = torch.sigmoid(self(x))[0, 0].cpu().numpy()
        return heatmap_to_centroids(prob, threshold, max_stars)


class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


def make_heatmap(uv: np.ndarray, shape, sigma: float = 1.2) -> np.ndarray:
    """Ground-truth target: a Gaussian bump at each true star position.

    Regressing a soft heatmap rather than a hard 0/1 mask is standard in
    keypoint detection -- it gives the network a gradient to follow when it is
    close but not exact, and it preserves sub-pixel information.
    """
    h, w = shape
    target = np.zeros((h, w), dtype=np.float32)
    if len(uv) == 0:
        return target
    radius = max(2, int(np.ceil(3 * sigma)))
    two_s2 = 2.0 * sigma * sigma
    for u, v in uv:
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        x_lo, x_hi = max(0, u0 - radius), min(w, u0 + radius + 1)
        y_lo, y_hi = max(0, v0 - radius), min(h, v0 + radius + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        xs = np.arange(x_lo, x_hi, dtype=np.float32) + 0.5 - u
        ys = np.arange(y_lo, y_hi, dtype=np.float32) + 0.5 - v
        bump = np.outer(np.exp(-(ys ** 2) / two_s2), np.exp(-(xs ** 2) / two_s2))
        np.maximum(target[y_lo:y_hi, x_lo:x_hi], bump,
                   out=target[y_lo:y_hi, x_lo:x_hi])
    return target


def heatmap_to_centroids(prob: np.ndarray, threshold: float = 0.5,
                         max_stars: int = 30):
    """Probability map -> sub-pixel centroids, by weighted moments per blob."""
    try:
        import cv2
        mask = (prob > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return np.zeros((0, 2)), np.zeros(0)
        n, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
        regions = range(1, n)
    except ImportError:  # pragma: no cover
        from scipy import ndimage
        mask = prob > threshold
        labels, n = ndimage.label(mask)
        regions = range(1, n + 1)

    cents, scores = [], []
    for k in regions:
        ys, xs = np.nonzero(labels == k)
        w = prob[ys, xs]
        total = w.sum()
        if total <= 0:
            continue
        cents.append([(w * (xs + 0.5)).sum() / total,
                      (w * (ys + 0.5)).sum() / total])
        scores.append(total)
    if not cents:
        return np.zeros((0, 2)), np.zeros(0)
    cents, scores = np.array(cents), np.array(scores)
    order = np.argsort(-scores)[:max_stars]
    return cents[order], scores[order]


def focal_bce(logits: torch.Tensor, target: torch.Tensor,
              alpha: float = 0.95, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for the heatmap.

    Stars occupy well under 1% of pixels, so plain BCE converges to predicting
    "no star" everywhere at 99% accuracy. Focal loss down-weights the easy
    empty background and keeps the gradient on the pixels that matter.
    """
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1 - prob)
    w = torch.where(target > 0.5, alpha, 1 - alpha)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (w * (1 - pt).pow(gamma) * bce).mean()


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== why not MSE: quaternion double cover ===")
    q = normalize_quat(torch.randn(1, 4))
    print(f"  MSE(q, -q)            = {F.mse_loss(q, -q).item():.4f}  (should be 0!)")
    print(f"  cosine_loss(q, -q)    = {quat_cosine_loss(q, -q).item():.3e}")
    print(f"  geodesic_deg(q, -q)   = {geodesic_angle_deg(q, -q).item():.3e} deg")

    print("\n=== loss is monotone in true angular error ===")
    base = normalize_quat(torch.randn(256, 4))
    for eps in [0.001, 0.01, 0.1, 0.5]:
        pert = normalize_quat(base + eps * torch.randn_like(base))
        print(f"  noise {eps:5.3f} -> cos_loss {quat_cosine_loss(base, pert).mean():.6f}"
              f"   angle {geodesic_angle_deg(base, pert).mean():7.3f} deg")

    print("\n=== AttitudeNet forward/backward ===")
    net = AttitudeNet(pretrained=False)
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(4, 1, 224, 224)
    qp, lv = net(x)
    print(f"  params {n_params/1e6:.2f}M | quat {tuple(qp.shape)} "
          f"| norms {qp.norm(dim=-1).detach().numpy().round(4)}")
    target = normalize_quat(torch.randn(4, 4))
    loss = heteroscedastic_loss(qp, target, lv).mean()
    loss.backward()
    gnorm = sum(p.grad.norm().item() ** 2 for p in net.parameters()
                if p.grad is not None) ** 0.5
    print(f"  hetero loss {loss.item():.4f} | grad norm {gnorm:.4f} (finite: "
          f"{np.isfinite(gnorm)})")

    print("\n=== uncertainty head can excuse hard samples but pays for it ===")
    bad = normalize_quat(torch.randn(1, 4))
    tgt = normalize_quat(torch.randn(1, 4))
    for s in [-4.0, -1.0, 0.0, 2.0]:
        lvv = torch.tensor([s])
        print(f"  log_var {s:5.1f} -> loss "
              f"{heteroscedastic_loss(bad, tgt, lvv).item():7.4f}")

    print("\n=== StarDetector + heatmap target ===")
    det = StarDetector()
    print(f"  params {sum(p.numel() for p in det.parameters())/1e3:.0f}K")
    logits = det(torch.randn(2, 1, 224, 224))
    print(f"  output {tuple(logits.shape)} (matches input resolution)")
    uv = np.array([[50.3, 60.7], [120.9, 33.1], [200.0, 200.0]])
    hm = make_heatmap(uv, (224, 224))
    print(f"  heatmap max {hm.max():.3f}, positive pixel fraction "
          f"{(hm>0.5).mean()*100:.3f}%  <-- why focal loss is needed")
    rec, _ = heatmap_to_centroids(hm, 0.3)
    err = [np.min(np.linalg.norm(uv - c, axis=1)) for c in rec]
    print(f"  heatmap round-trip centroid error: max {max(err):.4f} px")
    fl = focal_bce(logits, torch.from_numpy(hm)[None, None].repeat(2, 1, 1, 1))
    print(f"  focal loss {fl.item():.5f}")
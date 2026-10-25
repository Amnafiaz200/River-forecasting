"""Shared code for Question 2 (Motion or Memory?): data, transforms, windows, models, metrics."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- data
TRAIN = slice(0, 8000)
VAL = slice(8000, 9000)
TEST = slice(9000, 10000)


def load_videos(root="data"):
    """All 10,000 videos as a uint8 tensor (N, 20, 1, 64, 64); divide by 255 per batch (see to_float)."""
    import torchvision
    ds = torchvision.datasets.MovingMNIST(root=root, split=None, download=True)
    return ds.data


def to_float(v):
    return v.to(torch.float32) / 255.0 if v.dtype == torch.uint8 else v


# ----------------------------------------------------------------------------- transforms
def mirror(x):
    """(M x)[i, j] = x[i, 63 - j]: flip the last (column) axis."""
    return torch.flip(x, dims=(-1,))


def rot(x, a=1):
    """R^a: a * 90 deg anticlockwise, torch.rot90(x, k=a, dims=(-2, -1))."""
    return torch.rot90(x, k=a % 4, dims=(-2, -1))


# G = {R^a, M R^a}: element index g in 0..7;  g < 4 -> R^g,  g >= 4 -> M R^(g-4) (rotate first, then mirror)
def g_apply(x, g):
    y = rot(x, g % 4)
    return mirror(y) if g >= 4 else y


def g_inverse(x, g):
    """Inverse of g_apply: R^a -> R^-a;  M R^a -> R^-a M  (apply M first, then R^-a)."""
    if g >= 4:
        return rot(mirror(x), -(g % 4))
    return rot(x, -(g % 4))


# ----------------------------------------------------------------------------- windows
def window(video, t, k, n_in=5, n_out=5):
    """Window of `video` (20, ...) starting at frame t (1-based) at speed k.
    X = (x_t, x_{t+k}, ..., x_{t+4k}),  Y = (x_{t+5k}, ..., x_{t+9k}) keeping only frames <= 20.
    Returns X, Y, frame numbers of X, frame numbers of Y."""
    xi = [t + k * s for s in range(n_in)]
    yi = [t + k * s for s in range(n_in, n_in + n_out) if t + k * s <= 20]
    assert xi[-1] <= 20, "input frames must exist"
    return video[[i - 1 for i in xi]], video[[i - 1 for i in yi]], xi, yi


def allowed_starts(k, augmented):
    """Allowed start frames t. Normal training: all five targets exist (k=1 only): t=1..11.
    Speed-augmented training: 5 input frames and >= 2 target frames exist."""
    if not augmented:
        return list(range(1, 11 + 1)) if k == 1 else None
    last_needed = lambda t: t + 6 * k  # 5 inputs + 2 targets -> frame t + 6k  (index 0..6)
    return [t for t in range(1, 21) if t + 6 * k <= 20]


def batch_windows(videos, ts, ks):
    """Stack windows for a batch. Returns X (B,5,1,64,64), Y (B,5,1,64,64) zero-padded, mask (B,5)."""
    B = len(videos)
    X = torch.empty(B, 5, 1, 64, 64)
    Y = torch.zeros(B, 5, 1, 64, 64)
    M = torch.zeros(B, 5)
    for b in range(B):
        x, y, _, _ = window(videos[b], int(ts[b]), int(ks[b]))
        X[b] = to_float(x)
        Y[b, : len(y)] = to_float(y)
        M[b, : len(y)] = 1
    return X, Y, M


def make_eval(videos, k=1, t=1):
    return batch_windows(videos, [t] * len(videos), [k] * len(videos))


# ----------------------------------------------------------------------------- models
class ConvLSTMCell(nn.Module):
    """One ConvLSTM layer (Shi et al. 2015 without peepholes): one conv produces [i, f, o, g]."""

    def __init__(self, c_in, c_hid, kernel=3, dilation=1):
        super().__init__()
        self.c_hid = c_hid
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, kernel, 1, pad, dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], 1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Model A (defaults): pixel_unshuffle(patch) -> 2 ConvLSTM layers -> 1x1 conv -> pixel_shuffle -> sigmoid.
    Runs 9 steps; steps 6..9 are fed their own previous prediction (no teacher forcing).
    Output = predictions of steps 5..9, shape (B, 5, 1, 64, 64)."""

    def __init__(self, patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1)):
        super().__init__()
        self.patch = patch
        c0 = patch * patch
        self.l1 = ConvLSTMCell(c0, hidden, kernels[0], dilations[0])
        self.l2 = ConvLSTMCell(hidden, hidden, kernels[1], dilations[1])
        self.out = nn.Conv2d(hidden, c0, 1, bias=True)
        self.hidden = hidden

    def forward(self, X):  # X: (B, 5, 1, 64, 64)
        B = X.shape[0]
        hw = 64 // self.patch
        z = lambda: torch.zeros(B, self.hidden, hw, hw, device=X.device, dtype=X.dtype)
        s1, s2 = (z(), z()), (z(), z())
        u, preds = None, []
        for step in range(9):
            frame = X[:, step] if step < 5 else u
            s1 = self.l1(F.pixel_unshuffle(frame, self.patch), s1)
            s2 = self.l2(s1[0], s2)
            u = torch.sigmoid(F.pixel_shuffle(self.out(s2[0]), self.patch))
            if step >= 4:
                preds.append(u)
        return torch.stack(preds, 1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def symmetrised(f):
    """f_bar(X) = 1/8 sum_g g^-1( f( g(X) ) ).  X: (B, 5, 1, 64, 64)."""

    def fbar(X):
        return sum(g_inverse(f(g_apply(X, g)), g) for g in range(8)) / 8.0

    return fbar


# ----------------------------------------------------------------------------- metrics
def per_horizon_mse(pred, target, mask=None):
    """(B, 5) mean squared error over the 64x64 pixels for each forecast horizon."""
    return ((pred - target) ** 2).mean(dim=(2, 3, 4))


@torch.no_grad()
def predict(f, X, bs=32):
    out = []
    for i in range(0, len(X), bs):
        out.append(f(X[i : i + bs]))
    return torch.cat(out)


def masked_loss(pred, Y, M):
    """Squared error averaged over pixels and available target frames of each video, then over the batch."""
    se = ((pred - Y) ** 2).mean(dim=(2, 3, 4))  # (B, 5)
    return ((se * M).sum(1) / M.sum(1)).mean()


# ----------------------------------------------------------------------------- model registry
# Model D is chosen on the validation split only (see select_d.py); filled in once chosen.
D_CONFIG = dict(patch=8, hidden=64, kernels=(3, 3), dilations=(1, 1))


def build_model(name):
    if name in ("A", "B", "C"):
        return ConvLSTMForecaster()
    if name in ("D", "DB"):
        return ConvLSTMForecaster(**D_CONFIG)
    raise ValueError(name)

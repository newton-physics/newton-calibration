# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Chamfer distance between a projected cable axis and a cable mask.

The score of one frame is

    0.5 * (mean over axis samples of the distance to the nearest mask pixel
           + mean over mask pixels of the distance to the projected axis) / extent

in cable lengths on screen: ``extent`` is the mask's length along its principal axis
[px]. Both directions are needed: the first alone is minimized by a cable that curls
into a short piece inside the mask, the second alone by a cable that covers the mask
and runs past it. Distances are to the cable axis, not its surface, so a perfect fit
scores a small positive value set by the cable's width on screen.
"""

import functools
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

# Axis samples per frame, at uniform arc length along the projected axis.
AXIS_SAMPLES = 512
# Largest distance [px] one sample or pixel contributes.
MAX_DISTANCE_PX = 1024.0
# Frame score when no part of the axis is in front of the camera.
MISS = 2.0
# Smallest depth [m] in front of the camera at which a point is projected.
MIN_DEPTH_M = 1e-3


def _rotation(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class Camera:
    """Pinhole camera with a pose in the drive frame.

    Args:
        block: ``position_m``, ``quaternion_xyzw`` (Newton camera frame: +X right, +Y up,
            -Z forward) and ``intrinsics`` (``fx``, ``fy``, ``cx``, ``cy`` [px]).
    """

    def __init__(self, block):
        self.position = np.asarray(block["position_m"], dtype=np.float64)
        self.rotation = _rotation(block["quaternion_xyzw"])
        k = block["intrinsics"]
        self.fx, self.fy, self.cx, self.cy = (float(k[name]) for name in ("fx", "fy", "cx", "cy"))

    def project(self, points):
        """Pixel coordinates (u right, v down) of ``points`` [m], shape (n, 2).

        Points less than ``MIN_DEPTH_M`` in front of the camera are NaN.
        """
        local = (np.asarray(points, dtype=np.float64) - self.position) @ self.rotation
        depth = -local[:, 2]
        uv = np.full((len(local), 2), np.nan)
        front = depth > MIN_DEPTH_M
        uv[front, 0] = self.cx + self.fx * local[front, 0] / depth[front]
        uv[front, 1] = self.cy - self.fy * local[front, 1] / depth[front]
        return uv


class MaskTarget:
    """One frame's cable mask, prepared for scoring.

    Args:
        mask: Image whose nonzero pixels are the cable.
    """

    def __init__(self, mask):
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        if len(xs) < 2:
            raise ValueError("the mask needs at least two cable pixels")
        self.pixels = np.stack([xs, ys], axis=1).astype(np.float64)
        self.tree = cKDTree(self.pixels)
        centered = self.pixels - self.pixels.mean(axis=0)
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        t = centered @ direction
        self.extent = max(1.0, float(t.max() - t.min()))


@functools.cache
def load_mask_target(path):
    """Read a mask image and prepare it; cached, since every candidate batch uses it."""
    with Image.open(Path(path)) as image:
        return MaskTarget(np.asarray(image.convert("L")))


def _axis_samples(segments, count):
    """``count`` points at uniform arc length along the segments (s, 2, 2)."""
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    total = lengths.sum()
    if total <= 0.0:
        return segments[:, 0]
    targets = (np.arange(count) + 0.5) / count * total
    ends = np.cumsum(lengths)
    index = np.minimum(np.searchsorted(ends, targets), len(segments) - 1)
    start = ends[index] - lengths[index]
    frac = np.where(lengths[index] > 0.0, (targets - start) / np.maximum(lengths[index], 1e-12), 0.0)
    return segments[index, 0] + frac[:, None] * (segments[index, 1] - segments[index, 0])


def _distance_to_segments(points, segments):
    """Distance [px] from each point (m, 2) to the nearest of the segments (s, 2, 2)."""
    a = segments[:, 0][None]
    v = (segments[:, 1] - segments[:, 0])[None]
    w = points[:, None] - a
    length2 = (v * v).sum(axis=2)
    t = np.clip((w * v).sum(axis=2) / np.maximum(length2, 1e-12), 0.0, 1.0)
    d = w - t[..., None] * v
    return np.sqrt((d * d).sum(axis=2)).min(axis=1)


def chamfer(uv, target):
    """Chamfer score of one frame, in cable lengths on screen.

    Args:
        uv: Projected axis nodes [px], shape (n, 2), NaN where not in front of the camera.
        target: The frame's :class:`MaskTarget`.
    """
    valid = np.isfinite(uv).all(axis=1)
    keep = valid[:-1] & valid[1:]
    if not keep.any():
        return MISS
    segments = np.stack([uv[:-1][keep], uv[1:][keep]], axis=1)
    samples = _axis_samples(segments, AXIS_SAMPLES)
    to_goal = np.minimum(target.tree.query(samples)[0], MAX_DISTANCE_PX).mean()
    to_axis = np.minimum(_distance_to_segments(target.pixels, segments), MAX_DISTANCE_PX).mean()
    return float(0.5 * (to_goal + to_axis) / target.extent)

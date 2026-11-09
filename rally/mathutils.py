"""Small numeric helpers and the value-noise field used for the hills.

Pure numpy, no engine imports, so this is safe to use from tests.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

Number = Union[float, np.ndarray]


def clamp(value: Number, low: float, high: float) -> Number:
    """Clamp a scalar or an array into [low, high]."""
    if isinstance(value, np.ndarray):
        return np.clip(value, low, high)
    return max(low, min(high, value))


def lerp(a: Number, b: Number, t: Number) -> Number:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: Number) -> Number:
    """Hermite fade from 0 at edge0 to 1 at edge1."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sign(x: float) -> float:
    return math.copysign(1.0, x) if x else 0.0


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# Value noise
# --------------------------------------------------------------------------- #
def _hash2(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random value in [0, 1) per integer lattice point."""
    n = np.sin(xi * 127.1 + yi * 311.7) * 43758.5453123
    return n - np.floor(n)


def value_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smoothly interpolated value noise over a 2D grid."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)

    a = _hash2(xi, yi)
    b = _hash2(xi + 1.0, yi)
    c = _hash2(xi, yi + 1.0)
    d = _hash2(xi + 1.0, yi + 1.0)

    top = a * (1.0 - u) + b * u
    bottom = c * (1.0 - u) + d * u
    return top * (1.0 - v) + bottom * v


def fbm(x: np.ndarray, y: np.ndarray, octaves: int = 4, gain: float = 0.5,
        lacunarity: float = 2.07) -> np.ndarray:
    """Fractional Brownian motion: stacked octaves of value noise, normalised to 0..1."""
    total = np.zeros_like(np.asarray(x, dtype=np.float64))
    amplitude, frequency, norm = 1.0, 1.0, 0.0
    for _ in range(octaves):
        total = total + value_noise(np.asarray(x) * frequency, np.asarray(y) * frequency) * amplitude
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm

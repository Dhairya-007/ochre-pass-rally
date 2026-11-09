"""Procedural audio synthesis: engine, tyres, surface and wind loops.

No engine import here - this writes WAV files with numpy and the presentation
layer plays them. The loops are generated once into the user cache folder.

Engine: two loops at a fixed 55 Hz fundamental, played back at a rate set by
rpm. "on" is the loaded, throttle-open sound (strong odd harmonics, intake
roar); "off" is the overrun burble (softer, with irregular pops). The game
cross-fades between them on throttle position, which is most of what makes a
synthesised engine sound like it is working rather than humming.

Every loop's length is a whole number of fundamental periods, or cross-faded
at the seam, so nothing clicks when it wraps.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Callable, Dict

import numpy as np

SAMPLE_RATE = 22050
SYNTH_VERSION = 3          # bump to regenerate cached loops after a change


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _one_pole(signal: np.ndarray, coefficient: float, highpass: bool = False) -> np.ndarray:
    """Single-pole IIR filter (scipy is not a dependency)."""
    out = np.empty_like(signal)
    state = 0.0
    for i, value in enumerate(signal):
        state += coefficient * (value - state)
        out[i] = value - state if highpass else state
    return out


def _seamless(signal: np.ndarray, fade_seconds: float = 0.05) -> np.ndarray:
    blend = int(SAMPLE_RATE * fade_seconds)
    ramp = np.linspace(0.0, 1.0, blend)
    signal = signal.copy()
    signal[:blend] = signal[:blend] * ramp + signal[-blend:] * (1.0 - ramp)
    return signal[:-blend]


def _normalise(signal: np.ndarray, peak: float = 0.85) -> np.ndarray:
    top = np.max(np.abs(signal))
    return signal / top * peak if top > 0 else signal


def engine_on(fundamental: float = 55.0, seconds: float = 1.0, seed: int = 1) -> np.ndarray:
    count = int(SAMPLE_RATE * seconds)
    t = np.arange(count) / SAMPLE_RATE
    out = np.zeros(count)
    # Four-cylinder firing: strong 2nd and 4th orders, odd orders for rasp.
    for harmonic, amplitude in {1: 0.55, 2: 1.0, 3: 0.50, 4: 0.72, 5: 0.30, 6: 0.34,
                                7: 0.16, 8: 0.20, 10: 0.10, 12: 0.07, 14: 0.04}.items():
        phase = (harmonic * 0.61) % 1.0 * math.tau
        out += amplitude * np.sin(math.tau * fundamental * harmonic * t + phase)
    # Mild saturation, like an exhaust under load.
    out = np.tanh(out * 0.9)
    rng = np.random.default_rng(seed)
    roar = _one_pole(rng.uniform(-1, 1, count), 0.18) * 0.35
    # Modulate the intake noise with the firing pulse so it sounds attached.
    out += roar * (0.6 + 0.4 * np.sin(math.tau * fundamental * 2 * t))
    return _normalise(out)


def engine_off(fundamental: float = 55.0, seconds: float = 1.0, seed: int = 2) -> np.ndarray:
    count = int(SAMPLE_RATE * seconds)
    t = np.arange(count) / SAMPLE_RATE
    out = np.zeros(count)
    for harmonic, amplitude in {1: 1.0, 2: 0.65, 3: 0.22, 4: 0.30, 6: 0.10, 8: 0.05}.items():
        out += amplitude * np.sin(math.tau * fundamental * harmonic * t + harmonic)
    rng = np.random.default_rng(seed)
    # Overrun pops: sparse decaying bursts on firing pulses.
    period = int(SAMPLE_RATE / (fundamental * 2))
    for start in range(0, count - period, period):
        if rng.random() < 0.12:
            burst = rng.uniform(-1, 1, period) * np.exp(-np.linspace(0, 6, period))
            out[start:start + period] += burst * rng.uniform(0.6, 1.4)
    return _normalise(_one_pole(out, 0.55), 0.7)


def noise_loop(seconds: float, low: float, high: float, seed: int,
               grain: float = 0.0) -> np.ndarray:
    """Band-limited noise. `grain` adds random crackle (gravel pinging the arches)."""
    count = int(SAMPLE_RATE * seconds) + int(SAMPLE_RATE * 0.05)
    rng = np.random.default_rng(seed)
    signal = rng.uniform(-1.0, 1.0, count)
    signal = _one_pole(signal, low)
    signal = _one_pole(signal, high, highpass=True)
    if grain:
        clicks = (rng.random(count) < grain).astype(float) * rng.uniform(-1, 1, count)
        signal += _one_pole(clicks, 0.6) * 3.0
    return _normalise(_seamless(signal), 0.8)


LOOPS: Dict[str, Callable[[], np.ndarray]] = {
    "engine_on": engine_on,
    "engine_off": engine_off,
    "skid": lambda: noise_loop(2.0, low=0.50, high=0.25, seed=11),
    "gravel": lambda: noise_loop(2.0, low=0.30, high=0.05, seed=29, grain=0.004),
    "snow": lambda: noise_loop(2.0, low=0.12, high=0.02, seed=41),
    "wind": lambda: noise_loop(3.0, low=0.05, high=0.004, seed=53),
}


def ensure_loops(folder: Path) -> Dict[str, Path]:
    """Generate any missing loops into `folder`; return name -> path."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = folder / f".synth-v{SYNTH_VERSION}"
    fresh = not stamp.exists()
    paths = {}
    for name, make in LOOPS.items():
        path = folder / f"{name}.wav"
        if fresh or not path.exists():
            _write_wav(path, make())
        paths[name] = path
    stamp.touch()
    return paths

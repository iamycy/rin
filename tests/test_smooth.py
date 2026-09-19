"""End-to-end tests for rin.smooth with the default estimator and solver."""

import numpy as np

from rin import smooth

SR = 16000
HOP = int(0.02 * SR)


def _chirp_clip(dur=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    freq = 200.0 + 50.0 * t
    x = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    n_frames = 1 + len(x) // HOP
    ft = np.arange(n_frames) * HOP / SR
    true_f0 = 200.0 + 50.0 * ft
    noisy_f0 = true_f0 * 2 ** (rng.normal(0, 40, n_frames) / 1200)
    strength = np.full(n_frames, 0.9)
    return x, true_f0, noisy_f0, strength


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_smooth_reduces_noise():
    x, true_f0, noisy_f0, strength = _chirp_clip()
    f0_sm, voicing = smooth(noisy_f0, strength, x, SR, hops=(1, 2, 3), return_voicing=True)
    assert f0_sm.shape == true_f0.shape
    assert _rmse(f0_sm, true_f0) < _rmse(noisy_f0, true_f0)
    assert voicing.shape == true_f0.shape
    assert bool(((voicing >= 0) & (voicing <= 1)).all())


def test_smooth_tolerates_nan_unvoiced():
    x, true_f0, noisy_f0, strength = _chirp_clip()
    noisy_f0[10:15] = np.nan  # unvoiced region
    f0_sm = smooth(noisy_f0, strength, x, SR, hops=(1, 2))
    assert f0_sm.shape == true_f0.shape
    assert bool(np.isfinite(f0_sm).all())

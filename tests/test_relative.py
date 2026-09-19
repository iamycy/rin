"""Tests for rin.relative.vqt_diff_calculator on synthetic audio."""

import numpy as np

from rin.relative import vqt_diff_calculator

SR = 16000
HOP = int(0.02 * SR)


def _sine(f0=220.0, dur=2.0):
    t = np.arange(int(SR * dur)) / SR
    return 0.5 * np.sin(2 * np.pi * f0 * t)


def _chirp(f0=200.0, f1=300.0, dur=2.0):
    t = np.arange(int(SR * dur)) / SR
    freq = f0 + (f1 - f0) * t / dur
    return 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR), freq


def test_steady_tone_has_near_zero_diff():
    x = _sine()
    edges, est, conf, _ = vqt_diff_calculator(x, SR, HOP, (1, 2), bins_per_octave=36, n_bins=252)
    assert np.median(np.abs(est)) < 5.0  # cents
    assert np.median(conf) > 0.5
    # edges reference valid frame pairs
    M = 1 + len(x) // HOP
    assert bool(((edges[:, 0] >= 0) & (edges[:, 1] < M)).all())


def test_chirp_diff_tracks_instantaneous_change():
    x, freq = _chirp()
    edges, est, conf, _ = vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=36, n_bins=252)
    n_frames = 1 + len(x) // HOP
    t = np.arange(n_frames) * HOP / SR
    f = 200.0 + 100.0 * t / 2.0
    expected = 1200 * np.log2(f[1:] / f[:-1])
    # trim VQT edge effects
    lo, hi = 5, len(est) - 5
    err = np.abs(est[lo:hi] - expected[edges[lo:hi, 0]])
    assert np.median(err) < 15.0  # cents


def test_unknown_weighting_fails_fast():
    x = _sine(dur=0.2)
    try:
        vqt_diff_calculator(x, SR, HOP, (1,), weighting="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown weighting")

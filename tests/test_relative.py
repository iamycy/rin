"""Tests for rin.relative.vqt_diff_calculator on synthetic audio."""

import numpy as np
import pytest

from rin.relative import _vqt_xcorr_setup, compute_vqt, hop_diff, vqt_diff_calculator

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
    edges, est, conf = vqt_diff_calculator(x, SR, HOP, (1, 2), bins_per_octave=36, n_bins=252)
    assert np.median(np.abs(est)) < 5.0  # cents
    assert np.median(conf) > 0.5
    # edges reference valid frame pairs
    M = 1 + len(x) // HOP
    assert bool(((edges[:, 0] >= 0) & (edges[:, 1] < M)).all())


def test_chirp_diff_tracks_instantaneous_change():
    x, freq = _chirp()
    edges, est, conf = vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=36, n_bins=252)
    n_frames = 1 + len(x) // HOP
    t = np.arange(n_frames) * HOP / SR
    f = 200.0 + 100.0 * t / 2.0
    expected = 1200 * np.log2(f[1:] / f[:-1])
    # trim VQT edge effects
    lo, hi = 5, len(est) - 5
    err = np.abs(est[lo:hi] - expected[edges[lo:hi, 0]])
    assert np.median(err) < 15.0  # cents


def test_hop_diff_without_setup_matches():
    # hop_diff computes the xcorr setup itself when none is passed.
    x = _sine()
    V = compute_vqt(x, SR, HOP)
    max_diff_bins = int(600.0 / (1200 / 36))
    ref = hop_diff(V, 2, max_diff_bins, 1200 / 36, setup=_vqt_xcorr_setup(V, max_diff_bins))
    out = hop_diff(V, 2, max_diff_bins, 1200 / 36)
    for a, b in zip(ref, out):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_vqt_uses_paper_n_bins_by_default():
    # n_bins=252 is the default; overriding still works.
    x = _sine(dur=0.5)
    V = compute_vqt(x, SR, HOP)
    assert V.shape[0] == 252
    assert compute_vqt(x, SR, HOP, n_bins=84).shape[0] == 84


def test_invalid_inputs_fail_fast():
    x = _sine(dur=0.5)
    with pytest.raises(ValueError):
        vqt_diff_calculator(x.reshape(1, -1), SR, HOP, (1,))  # stereo
    with pytest.raises(ValueError):
        vqt_diff_calculator(x, SR, HOP, ())  # empty hops
    with pytest.raises(ValueError):
        vqt_diff_calculator(x, SR, HOP, (0,))  # non-positive hop
    with pytest.raises(ValueError):
        vqt_diff_calculator(x, SR, HOP, (1.5,))  # non-integer hop

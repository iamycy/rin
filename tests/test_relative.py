"""Tests for rin.relative.vqt_diff_calculator on synthetic audio."""

import numpy as np
import pytest

from rin.relative import _compute_vqt, _hop_diff, _vqt_xcorr_setup, vqt_diff_calculator

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
    # _hop_diff computes the xcorr setup itself when none is passed.
    x = _sine()
    V = _compute_vqt(x, SR, HOP)
    max_diff_bins = int(600.0 / (1200 / 36))
    ref = _hop_diff(V, 2, max_diff_bins, 1200 / 36, setup=_vqt_xcorr_setup(V, max_diff_bins))
    out = _hop_diff(V, 2, max_diff_bins, 1200 / 36)
    for a, b in zip(ref, out):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_vqt_uses_paper_n_bins_by_default():
    # n_bins=252 is the default; overriding still works.
    x = _sine(dur=0.5)
    V = _compute_vqt(x, SR, HOP)
    assert V.shape[0] == 252
    assert _compute_vqt(x, SR, HOP, n_bins=84).shape[0] == 84


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
    with pytest.raises(ValueError, match="bins_per_octave"):
        vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=0)
    with pytest.raises(ValueError, match="bins_per_octave"):
        vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=-12)
    with pytest.raises(ValueError, match="max_diff_cents"):
        vqt_diff_calculator(x, SR, HOP, (1,), max_diff_cents=0.0)
    with pytest.raises(ValueError, match="max_diff_cents"):
        vqt_diff_calculator(x, SR, HOP, (1,), max_diff_cents=-100.0)
    with pytest.raises(ValueError, match="max_diff_cents"):
        vqt_diff_calculator(x, SR, HOP, (1,), max_diff_cents=10.0)  # < 1 bin


def test_hop_diff_rejects_nonpositive_jump():
    x = _sine(dur=0.5)
    V = _compute_vqt(x, SR, HOP)
    with pytest.raises(ValueError, match="jump"):
        _hop_diff(V, 0, 18, 1200 / 36)
    with pytest.raises(ValueError, match="jump"):
        _hop_diff(V, 1.5, 18, 1200 / 36)


def test_max_diff_cents_beyond_vqt_range_rejected():
    # Regression: the search radius was validated from below (>= 1 bin) but
    # not against n_bins. Once it reaches n_bins the overlap length K_tau hits
    # zero, the Pearson normalization divides by it, and NaN confidences
    # escaped the estimator, surfacing much later as an unrelated
    # "confidences must be finite" error from the solver.
    x = _sine(dur=0.5)
    with pytest.raises(ValueError, match="max_diff_cents"):
        # 600 cents at 36 bins/octave is an 18-bin radius; the VQT has 12.
        vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=36, n_bins=12)
    with pytest.raises(ValueError, match="max_diff_cents"):
        vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=36, n_bins=18)  # radius == n_bins


def test_narrowest_valid_vqt_range_stays_finite():
    # One bin wider than the search radius is the tightest legal setting;
    # it must still produce finite confidences rather than NaN.
    x = _sine(dur=0.5)
    _, est, conf = vqt_diff_calculator(x, SR, HOP, (1,), bins_per_octave=36, n_bins=19)
    assert bool(np.isfinite(est).all())
    assert bool(np.isfinite(conf).all())
    assert bool(((conf >= 0) & (conf <= 1)).all())


def _two_sided_vqt(max_diff_bins, F=80, base=24):
    """VQT whose correlation column has lobes at *both* ends of the shift axis.

    Frame 0 carries two peaks ``2 * max_diff_bins`` apart and of unequal
    height; frame 1 carries one peak centred between them, so shifting by
    ``+max_diff_bins`` aligns the lower pair and ``-max_diff_bins`` the upper.
    The peak lands on index 0 while the opposite end stays non-zero, exactly
    the shape the ``idx - 1 == -1`` wrap used to read from.
    """
    V = np.full((F, 2), 1e-6)
    hi = base + 2 * max_diff_bins
    V[base - 1 : base + 2, 0] = [0.5, 1.0, 0.5]
    V[hi - 1 : hi + 2, 0] = [0.4, 0.8, 0.4]
    V[base + max_diff_bins - 1 : base + max_diff_bins + 2, 1] = [0.5, 1.0, 0.5]
    return V


def test_boundary_peak_does_not_wrap_around_the_shift_axis():
    # Regression: the stencil clamp was one-sided (np.minimum), so at idx == 0
    # the index -1 wrapped to the opposite end of the shift axis and the
    # parabolic offset was fitted across an unrelated shift. With a lobe at
    # both ends this pushed the reported difference outside the
    # +-max_diff_cents window the estimator promises (204.2 from a +-200 one).
    max_diff_bins, diff_unit = 6, 1200 / 36
    limit = max_diff_bins * diff_unit
    V = _two_sided_vqt(max_diff_bins)
    setup = _vqt_xcorr_setup(V, max_diff_bins)
    _, est, conf = _hop_diff(V, 1, max_diff_bins, diff_unit, setup=setup)

    assert bool(np.isfinite(est).all())
    assert bool((np.abs(est) <= limit + 1e-9).all()), f"estimate outside +-{limit} cents"
    # the peak is pinned to the boundary, so it carries no weight and reports
    # the search limit itself, with no interpolated sub-bin offset
    assert conf[0] == 0
    np.testing.assert_allclose(est[0], limit)


def test_correlation_reduction_axes_stay_contiguous(monkeypatch):
    # Guards the np.asfortranarray calls in relative.py: dropping either costs
    # ~4.7x, and no accuracy test would notice.
    V = np.ascontiguousarray(_compute_vqt(_sine(), SR, HOP))
    assert not V.flags.f_contiguous

    sliding_V = _vqt_xcorr_setup(V, 18)[0]
    assert sliding_V.strides[-1] == V.itemsize

    seen = []
    real_vecdot = np.linalg.vecdot

    def spy(a, b, **kwargs):
        seen.append((a.strides[-1], b.strides[-1]))
        return real_vecdot(a, b, **kwargs)

    monkeypatch.setattr(np.linalg, "vecdot", spy)
    _hop_diff(V, 1, 18, 1200 / 36)
    assert seen == [(V.itemsize, V.itemsize)]


def test_identical_frames_keep_confidences_finite():
    # Guards the upper clip in _dot_weight. The parabola vertex sits above the
    # discrete peak, so corr_max reaches 1 + 1e-15 when two frames match
    # exactly and arcsin returns NaN; the column is interior, so hit_boundary
    # does not mask it and the NaN would reach the solver.
    col = np.exp(-(((np.arange(60) - 22) / 5.0) ** 2)) + 1e-9
    V = np.asfortranarray(np.stack([col, col], axis=1))
    _, est, conf = _hop_diff(V, 1, 6, 1200 / 36)
    assert bool(np.isfinite(est).all())
    assert bool(np.isfinite(conf).all())
    assert bool(((conf >= 0) & (conf <= 1)).all())

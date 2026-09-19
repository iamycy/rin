"""End-to-end tests: the high-level pipeline and manual core wiring.

``smooth_pitch`` is the paper's version in one call; the manual test below
shows the intended custom-wiring pattern -- the package does no caching and
no file loading, each caller manages their own pipeline around the cores.
"""

import numpy as np
import pytest

from rin import estimate_voicing, lp_smoother, smooth_pitch, vqt_diff_calculator

SR = 16000
HOP = int(0.02 * SR)


def _hz2cent(f):
    return 1200.0 * np.log2(f)


def _cent2hz(c):
    return 2.0 ** (c / 1200.0)


def _noisy_chirp(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * 2.0)) / SR
    freq = 200.0 + 25.0 * t
    x = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    n_frames = 1 + len(x) // HOP
    true_f0 = 200.0 + 25.0 * np.arange(n_frames) * HOP / SR
    noisy_f0 = true_f0 * 2 ** (rng.normal(0, 40, n_frames) / 1200)
    noisy_f0[30:40] = np.nan
    strength = np.full(n_frames, 0.9)
    return x, true_f0, noisy_f0, strength


def test_pipeline_reduces_noise():
    x, true_f0, noisy_f0, strength = _noisy_chirp()
    f0_smooth, voicing = smooth_pitch(x, noisy_f0, strength, SR, HOP)

    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    voiced = np.isfinite(noisy_f0)
    assert rmse(f0_smooth[voiced], true_f0[voiced]) < rmse(noisy_f0[voiced], true_f0[voiced])
    assert bool(np.isfinite(f0_smooth).all())
    assert bool(((voicing >= 0) & (voicing <= 1)).all())


def test_smooth_pitch_matches_manual_wiring():
    # The high-level function must equal the three cores wired by hand.
    x, _, noisy_f0, strength = _noisy_chirp()
    ref_hz, ref_voicing = smooth_pitch(x, noisy_f0, strength, SR, HOP, hops=(1, 2, 3))

    edges, estimates, confidences = vqt_diff_calculator(x, SR, HOP, hops=(1, 2, 3))
    voiced = np.isfinite(noisy_f0)
    f0_cents = _hz2cent(np.where(voiced, noisy_f0, 1.0))
    abs_conf = np.where(voiced, strength, 0.0)
    smooth_cents = lp_smoother(f0_cents, abs_conf, edges, estimates, confidences)
    voicing = estimate_voicing(abs_conf, edges, confidences)

    np.testing.assert_allclose(ref_hz, _cent2hz(smooth_cents), rtol=1e-12)
    np.testing.assert_allclose(ref_voicing, voicing, rtol=1e-12)


def test_smooth_pitch_rejects_mismatched_frames():
    x, _, noisy_f0, strength = _noisy_chirp()
    with pytest.raises(ValueError):
        smooth_pitch(x, noisy_f0[:-1], strength, SR, HOP)  # length mismatch
    with pytest.raises(ValueError):
        smooth_pitch(x, noisy_f0, strength[:-5], SR, HOP)
    with pytest.raises(ValueError):
        smooth_pitch(x, noisy_f0, strength, SR, HOP * 2)  # wrong grid

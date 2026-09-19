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


def _noisy_chirp(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * 2.0)) / SR
    freq = 200.0 + 25.0 * t
    x = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    n_frames = 1 + len(x) // HOP
    true_cents = 1200 * np.log2(200.0 + 25.0 * np.arange(n_frames) * HOP / SR)
    noisy_cents = true_cents + rng.normal(0, 40, n_frames)
    noisy_cents[30:40] = np.nan
    strength = np.full(n_frames, 0.9)
    return x, true_cents, noisy_cents, strength


def test_pipeline_reduces_noise():
    x, true_cents, noisy_cents, strength = _noisy_chirp()
    f0_smooth, voicing = smooth_pitch(x, noisy_cents, strength, SR, HOP)

    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    voiced = np.isfinite(noisy_cents)
    assert rmse(f0_smooth[voiced], true_cents[voiced]) < rmse(
        noisy_cents[voiced], true_cents[voiced]
    )
    assert bool(np.isfinite(f0_smooth).all())
    assert bool(((voicing >= 0) & (voicing <= 1)).all())


def test_smooth_pitch_matches_manual_wiring():
    # The high-level function must equal the three cores wired by hand.
    x, _, noisy_cents, strength = _noisy_chirp()
    ref, ref_voicing = smooth_pitch(x, noisy_cents, strength, SR, HOP, hops=(1, 2, 3))

    edges, estimates, confidences = vqt_diff_calculator(x, SR, HOP, hops=(1, 2, 3))
    voiced = np.isfinite(noisy_cents)
    abs_est = np.where(voiced, noisy_cents, 0.0)
    abs_conf = np.where(voiced, strength, 0.0)
    smooth_cents = lp_smoother(abs_est, abs_conf, edges, estimates, confidences)
    voicing = estimate_voicing(abs_conf, edges, confidences)

    np.testing.assert_allclose(ref, smooth_cents, rtol=1e-12)
    np.testing.assert_allclose(ref_voicing, voicing, rtol=1e-12)


def test_smooth_pitch_rejects_mismatched_frames():
    x, _, noisy_cents, strength = _noisy_chirp()
    with pytest.raises(ValueError):
        smooth_pitch(x, noisy_cents[:-1], strength, SR, HOP)  # length mismatch
    with pytest.raises(ValueError):
        smooth_pitch(x, noisy_cents, strength[:-5], SR, HOP)
    with pytest.raises(ValueError):
        smooth_pitch(x, noisy_cents, strength, SR, HOP * 2)  # wrong grid

"""End-to-end test: a caller wires the three cores directly.

This is the intended usage pattern -- the package does no caching and no
file loading; each caller manages their own pipeline around the three core
functions.
"""

import numpy as np

from rin import estimate_voicing, lp_smoother, vqt_diff_calculator

SR = 16000
HOP = int(0.02 * SR)


def _hz2cent(f):
    return 1200.0 * np.log2(f)


def _cent2hz(c):
    return 2.0 ** (c / 1200.0)


def test_pipeline_reduces_noise():
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 2.0)) / SR
    freq = 200.0 + 25.0 * t
    x = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)

    n_frames = 1 + len(x) // HOP
    true_f0 = 200.0 + 25.0 * np.arange(n_frames) * HOP / SR
    noisy_f0 = true_f0 * 2 ** (rng.normal(0, 40, n_frames) / 1200)
    noisy_f0[30:40] = np.nan
    strength = np.full(n_frames, 0.9)

    # caller-side wiring: estimate diffs, fuse with the LP, convert units.
    # Unvoiced (NaN) frames get zero absolute confidence, so the LP bridges
    # them through the relative edges alone.
    edges, estimates, confidences = vqt_diff_calculator(x, SR, HOP, hops=(1, 2, 3))
    voiced = np.isfinite(noisy_f0)
    f0_cents = _hz2cent(np.where(voiced, noisy_f0, 1.0))
    abs_conf = np.where(voiced, strength, 0.0)
    smooth_cents = lp_smoother(f0_cents, abs_conf, edges, estimates, confidences)
    voicing = estimate_voicing(abs_conf, edges, confidences)
    f0_smooth = _cent2hz(smooth_cents)

    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    assert rmse(f0_smooth[voiced], true_f0[voiced]) < rmse(noisy_f0[voiced], true_f0[voiced])
    assert bool(np.isfinite(f0_smooth).all())
    assert bool(((voicing >= 0) & (voicing <= 1)).all())

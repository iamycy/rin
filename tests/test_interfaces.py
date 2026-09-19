"""Plugin-interface tests: custom DifferenceEstimator and Solver through smooth()."""

import numpy as np
import pytest

from rin import smooth
from rin.interfaces import DifferenceEstimator, Solver

SR = 16000
HOP = int(0.02 * SR)


class ZeroEstimator:
    """Degenerate baseline: exact hop edges, zero relative pitch, full confidence."""

    def estimate(self, x, sr, hop_length, hops):
        M = 1 + len(x) // hop_length
        edges, est, conf = [], [], []
        for h in hops:
            edges.append(np.stack([np.arange(M - h), np.arange(h, M)], axis=1))
            est.append(np.zeros(M - h))
            conf.append(np.ones(M - h))
        return np.concatenate(edges), np.concatenate(est), np.concatenate(conf)


class IdentitySolver:
    """Pass-through solver: smoothed = absolute, voicing = strength."""

    def solve(self, abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
        return abs_estimates.copy(), np.clip(abs_confidences, 0, 1)


def _sine_clip(dur=1.0):
    t = np.arange(int(SR * dur)) / SR
    x = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    n_frames = 1 + len(x) // HOP
    f0 = np.full(n_frames, 220.0)
    strength = np.full(n_frames, 0.9)
    return x, f0, strength


def test_protocols_are_satisfied_structurally():
    # No inheritance needed: structural subtyping is enough.
    assert isinstance(ZeroEstimator(), DifferenceEstimator)
    assert isinstance(IdentitySolver(), Solver)
    assert not isinstance(object(), DifferenceEstimator)
    assert not isinstance(object(), Solver)


def test_custom_estimator_and_solver_roundtrip():
    x, f0, strength = _sine_clip()
    f0_sm, voicing = smooth(
        f0,
        strength,
        x,
        SR,
        hops=(1, 2),
        estimator=ZeroEstimator(),
        solver=IdentitySolver(),
        return_voicing=True,
    )
    np.testing.assert_allclose(f0_sm, f0, rtol=1e-12)
    np.testing.assert_allclose(voicing, strength, rtol=1e-12)


def test_custom_estimator_with_default_solver():
    # Degenerate baseline (zero relative pitch) still runs through the LP.
    x, f0, strength = _sine_clip()
    f0_sm = smooth(f0, strength, x, SR, hops=(1,), estimator=ZeroEstimator())
    assert f0_sm.shape == f0.shape
    assert bool(np.isfinite(f0_sm).all())


def test_nonconforming_plugins_rejected():
    x, f0, strength = _sine_clip()
    with pytest.raises(TypeError):
        smooth(f0, strength, x, SR, estimator=object())
    with pytest.raises(TypeError):
        smooth(f0, strength, x, SR, solver=object())

"""Plugin-interface tests: plain-function estimator and solver through smooth()."""

import functools

import numpy as np
import pytest

from rin import smooth

SR = 16000
HOP = int(0.02 * SR)


def zero_estimator(x, sr, hop_length, hops):
    """Degenerate baseline: exact hop edges, zero relative pitch, full confidence."""
    M = 1 + len(x) // hop_length
    edges, est, conf = [], [], []
    for h in hops:
        edges.append(np.stack([np.arange(M - h), np.arange(h, M)], axis=1))
        est.append(np.zeros(M - h))
        conf.append(np.ones(M - h))
    return np.concatenate(edges), np.concatenate(est), np.concatenate(conf)


def identity_solver(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
    """Pass-through solver: smoothed = absolute, voicing = strength."""
    return abs_estimates.copy(), np.clip(abs_confidences, 0, 1)


class CallableSolver:
    """A callable object also satisfies the contract -- no plain function needed."""

    def __call__(self, abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
        return identity_solver(
            abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences
        )


def _sine_clip(dur=1.0):
    t = np.arange(int(SR * dur)) / SR
    x = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    n_frames = 1 + len(x) // HOP
    f0 = np.full(n_frames, 220.0)
    strength = np.full(n_frames, 0.9)
    return x, f0, strength


def test_custom_estimator_and_solver_roundtrip():
    x, f0, strength = _sine_clip()
    f0_sm, voicing = smooth(
        f0,
        strength,
        x,
        SR,
        hops=(1, 2),
        estimator=zero_estimator,
        solver=identity_solver,
        return_voicing=True,
    )
    np.testing.assert_allclose(f0_sm, f0, rtol=1e-12)
    np.testing.assert_allclose(voicing, strength, rtol=1e-12)


def test_any_callable_shape_is_accepted():
    # lambda, functools.partial, and callable objects all obey the contract.
    x, f0, strength = _sine_clip()
    f0_sm = smooth(
        f0,
        strength,
        x,
        SR,
        hops=(1,),
        estimator=lambda x_, sr_, hop_, hops_: zero_estimator(x_, sr_, hop_, hops_),
        solver=functools.partial(identity_solver),
    )
    np.testing.assert_allclose(f0_sm, f0, rtol=1e-12)
    f0_sm = smooth(
        f0, strength, x, SR, hops=(1,), estimator=zero_estimator, solver=CallableSolver()
    )
    np.testing.assert_allclose(f0_sm, f0, rtol=1e-12)


def test_custom_estimator_with_default_solver():
    # Degenerate baseline (zero relative pitch) still runs through the LP.
    x, f0, strength = _sine_clip()
    f0_sm = smooth(f0, strength, x, SR, hops=(1,), estimator=zero_estimator)
    assert f0_sm.shape == f0.shape
    assert bool(np.isfinite(f0_sm).all())


def test_noncallable_plugins_rejected():
    x, f0, strength = _sine_clip()
    with pytest.raises(TypeError):
        smooth(f0, strength, x, SR, estimator=object())
    with pytest.raises(TypeError):
        smooth(f0, strength, x, SR, solver=42)

"""End-to-end tests: the high-level pipeline and manual core wiring.

``smooth_pitch`` is the paper's version in one call; the manual test below
shows the intended custom-wiring pattern -- the package does no caching and
no file loading, each caller manages their own pipeline around the cores.
"""

from functools import partial

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


def test_smooth_pitch_uses_injected_callables():
    # smooth_pitch is a higher-order function: each stage can be swapped
    # for any callable obeying the rin.interfaces contracts.
    x, _, noisy_cents, strength = _noisy_chirp()
    calls = {}

    estimator = partial(vqt_diff_calculator, max_diff_cents=500.0)

    def recording_estimator(x_, sr_, hop_, hops_):
        calls["estimator"] = tuple(hops_)
        return estimator(x_, sr_, hop_, hops_)

    def recording_solver(abs_est, abs_conf, edges, estimates, confidences):
        calls["solver"] = (abs_est.shape, edges.shape)
        return lp_smoother(abs_est, abs_conf, edges, estimates, confidences)

    def recording_voicing(abs_conf, edges, confidences):
        calls["voicing"] = abs_conf.shape
        return estimate_voicing(abs_conf, edges, confidences)

    f0_smooth, voicing = smooth_pitch(
        x,
        noisy_cents,
        strength,
        SR,
        HOP,
        hops=(1, 2),
        difference_estimator=recording_estimator,
        solver=recording_solver,
        voicing_estimator=recording_voicing,
    )

    assert set(calls) == {"estimator", "solver", "voicing"}
    assert calls["estimator"] == (1, 2)
    n_frames = len(noisy_cents)
    assert calls["solver"][0] == (n_frames,)
    assert calls["voicing"] == (n_frames,)
    assert bool(np.isfinite(f0_smooth).all())
    assert bool(((voicing >= 0) & (voicing <= 1)).all())


def test_smooth_pitch_custom_solver_replaces_lp():
    # A degenerate solver (ignore relative edges, return absolutes) must
    # flow through unchanged, proving the built-in LP is not hard-wired.
    x, _, noisy_cents, strength = _noisy_chirp()

    def passthrough_solver(abs_est, abs_conf, edges, estimates, confidences):
        return abs_est.copy()

    f0_smooth, _ = smooth_pitch(x, noisy_cents, strength, SR, HOP, solver=passthrough_solver)
    voiced = np.isfinite(noisy_cents)
    expected = np.where(voiced, noisy_cents, 0.0)
    np.testing.assert_allclose(f0_smooth, expected, rtol=1e-12)


def test_estimator_may_leave_trailing_frames_unconnected():
    # Regression: the frame count was inferred as edges.max() + 1, which is
    # only a lower bound. An estimator that leaves trailing frames unconnected
    # (one needing lookahead, say) made a correctly sized f0 fail with a
    # misleading "both must use the same hop_length grid".
    x, _, noisy_cents, strength = _noisy_chirp()
    n_frames = len(noisy_cents)

    def lookahead_estimator(x_, sr_, hop_, hops_):
        # conforms to DifferenceEstimator but never references the last 3 frames
        u = np.arange(n_frames - 4)
        return np.stack([u, u + 1], axis=1), np.zeros(len(u)), np.ones(len(u))

    f0_smooth, voicing = smooth_pitch(
        x, noisy_cents, strength, SR, HOP, difference_estimator=lookahead_estimator
    )
    assert f0_smooth.shape == (n_frames,)
    assert voicing.shape == (n_frames,)
    assert bool(np.isfinite(f0_smooth).all())


def test_out_of_range_edges_from_estimator_rejected():
    # The other half of the grid check: an edge pointing past the end of f0
    # would be silently dropped by the solver, so it must fail loudly.
    x, _, noisy_cents, strength = _noisy_chirp()
    n_frames = len(noisy_cents)

    def overshooting_estimator(x_, sr_, hop_, hops_):
        u = np.arange(n_frames)
        return np.stack([u, u + 1], axis=1), np.zeros(n_frames), np.ones(n_frames)

    with pytest.raises(ValueError, match="edge referencing frame"):
        smooth_pitch(
            x, noisy_cents, strength, SR, HOP, difference_estimator=overshooting_estimator
        )


def test_negative_edge_endpoints_rejected():
    # The lower half of the grid check: a negative endpoint is dropped just as
    # silently by _edge_keep_mask as an overshooting one, so it must fail loudly.
    x, _, noisy_cents, strength = _noisy_chirp()
    n_frames = len(noisy_cents)

    def negative_estimator(x_, sr_, hop_, hops_):
        u = np.arange(-1, n_frames - 2)
        return np.stack([u, u + 1], axis=1), np.zeros(len(u)), np.ones(len(u))

    with pytest.raises(ValueError, match=r"referencing frame -1, outside"):
        smooth_pitch(x, noisy_cents, strength, SR, HOP, difference_estimator=negative_estimator)


def test_edgeless_estimator_still_requires_the_full_grid():
    # With edges present, a too-short f0 is caught because some edge indexes
    # past its end. With no edges nothing constrains it from below, so the hop
    # grid is required exactly rather than silently accepting any shorter f0.
    x, _, noisy_cents, strength = _noisy_chirp()

    def edgeless_estimator(x_, sr_, hop_, hops_):
        return np.zeros((0, 2), dtype=int), np.zeros(0), np.zeros(0)

    f0_smooth, _ = smooth_pitch(
        x, noisy_cents, strength, SR, HOP, difference_estimator=edgeless_estimator
    )
    assert f0_smooth.shape == noisy_cents.shape

    with pytest.raises(ValueError, match="both must use the same hop_length grid"):
        smooth_pitch(
            x, noisy_cents[:-3], strength[:-3], SR, HOP,
            difference_estimator=edgeless_estimator,
        )

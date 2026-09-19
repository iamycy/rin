"""Tests for rin.lp.lp_smoother on synthetic contours."""

import numpy as np
import pytest

from rin.lp import lp_smoother


def _synth(M=50, seed=0):
    rng = np.random.default_rng(seed)
    truth_hz = 200 + 10 * np.sin(np.linspace(0, 4 * np.pi, M))
    truth_c = 1200 * np.log2(truth_hz)
    abs_est = truth_c + rng.normal(0, 30, M)  # noisy absolute estimates, in cents
    abs_conf = np.full(M, 0.8)
    edges = np.stack([np.arange(M - 1), np.arange(1, M)], axis=1)
    rel_est = np.diff(truth_c)  # exact hop-1 relative differences
    rel_conf = np.full(M - 1, 0.9)
    return truth_c, abs_est, abs_conf, edges, rel_est, rel_conf


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_dual_and_primal_agree():
    truth_c, abs_est, abs_conf, edges, rel_est, rel_conf = _synth()
    args = (abs_est, abs_conf, edges, rel_est, rel_conf)
    sm_dual = lp_smoother(*args, dual_form=True)
    sm_primal = lp_smoother(*args, dual_form=False)
    # Same LP, two formulations: agreement is limited by HiGHS numerics.
    # ~7.5 cents max deviation on a ~9200-cent scale (0.08%) was observed.
    np.testing.assert_allclose(sm_dual, sm_primal, rtol=1e-3, atol=5.0)


def test_smoothing_beats_noisy_input():
    truth_c, abs_est, abs_conf, edges, rel_est, rel_conf = _synth()
    sm = lp_smoother(abs_est, abs_conf, edges, rel_est, rel_conf)
    assert _rmse(sm, truth_c) < _rmse(abs_est, truth_c)


def test_zero_weight_edges_are_ignored():
    truth_c, abs_est, abs_conf, edges, rel_est, rel_conf = _synth()
    sm_ref = lp_smoother(abs_est, abs_conf, edges, rel_est, rel_conf)
    # append garbage edges with zero confidence: must not change the optimum
    bad_edges = np.array([[0, 49], [10, 40]])
    bad_est = np.array([5000.0, -5000.0])
    bad_conf = np.array([0.0, 0.0])
    sm = lp_smoother(
        abs_est,
        abs_conf,
        np.concatenate([edges, bad_edges]),
        np.concatenate([rel_est, bad_est]),
        np.concatenate([rel_conf, bad_conf]),
    )
    np.testing.assert_allclose(sm, sm_ref, rtol=1e-6, atol=1e-6)


def test_out_of_range_edges_are_dropped():
    _, abs_est, abs_conf, edges, rel_est, rel_conf = _synth(M=50)
    sm_ref = lp_smoother(abs_est, abs_conf, edges, rel_est, rel_conf)
    # frame 50..60 do not exist; negative indices are invalid too.
    # They must be dropped, leaving the optimum unchanged.
    bad_edges = np.array([[0, 50], [49, 60], [-1, 5]])
    sm = lp_smoother(
        abs_est,
        abs_conf,
        np.concatenate([edges, bad_edges]),
        np.concatenate([rel_est, [10.0, 20.0, 30.0]]),
        np.concatenate([rel_conf, [0.9, 0.9, 0.9]]),
    )
    np.testing.assert_allclose(sm, sm_ref, rtol=1e-9, atol=1e-9)


def test_empty_edges_returns_absolute():
    # No relative constraints: each frame is independent, so the optimum is
    # the absolute estimate itself.
    _, abs_est, abs_conf, _, _, _ = _synth(M=50)
    sm = lp_smoother(
        abs_est,
        abs_conf,
        np.zeros((0, 2), dtype=int),
        np.zeros(0),
        np.zeros(0),
    )
    np.testing.assert_allclose(sm, abs_est, rtol=1e-9, atol=1e-9)


def test_invalid_inputs_fail_fast():
    _, abs_est, abs_conf, edges, rel_est, rel_conf = _synth(M=50)
    with pytest.raises(ValueError):
        lp_smoother(abs_est, -abs_conf, edges, rel_est, rel_conf)  # negative conf
    with pytest.raises(ValueError):
        lp_smoother(abs_est, abs_conf, edges, rel_est, -rel_conf)
    with pytest.raises(ValueError):
        lp_smoother(abs_est, abs_conf, edges.reshape(-1), rel_est, rel_conf)
    with pytest.raises(ValueError):
        lp_smoother(abs_est, abs_conf, edges, rel_est[:-1], rel_conf)  # length mismatch
    with pytest.raises(ValueError):
        bad = abs_est.copy()
        bad[0] = np.inf
        lp_smoother(bad, abs_conf, edges, rel_est, rel_conf)

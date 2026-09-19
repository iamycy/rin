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


def _primal_smoother(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
    """Explicit primal LP (|.| split into slacks) as a reference for the dual."""
    from scipy.optimize import linprog

    abs_estimates = np.asarray(abs_estimates, dtype=float)
    abs_confidences = np.asarray(abs_confidences, dtype=float)
    rel_edges = np.asarray(rel_edges, dtype=int)
    rel_estimates = np.asarray(rel_estimates, dtype=float)
    rel_confidences = np.asarray(rel_confidences, dtype=float)
    M = abs_estimates.shape[0]
    E = rel_edges.shape[0]
    B = np.zeros((E, M))
    rows = np.arange(E)
    # np.add.at: duplicate (row, col) pairs accumulate, so self-loops
    # correctly produce an all-zero row (as in the package's csr_matrix).
    np.add.at(B, (rows, rel_edges[:, 0]), -1.0)
    np.add.at(B, (rows, rel_edges[:, 1]), 1.0)
    I_M, I_E = np.eye(M), np.eye(E)
    Z_ME, Z_EM = np.zeros((M, E)), np.zeros((E, M))
    # x = [sA (M), sR (E), f (M)]; constraints: +/-(f - a) <= sA, +/-(Bf - r) <= sR
    A_ub = np.block(
        [
            [-I_M, Z_ME, I_M],
            [-I_M, Z_ME, -I_M],
            [Z_EM, -I_E, B],
            [Z_EM, -I_E, -B],
        ]
    )
    b_ub = np.concatenate([abs_estimates, -abs_estimates, rel_estimates, -rel_estimates])
    c = np.concatenate([abs_confidences, rel_confidences, np.zeros(M)])
    bounds = [(0, None)] * (M + E) + [(None, None)] * M
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.success, res.message
    return res.x[M + E :]


def test_dual_matches_primal():
    """The dual min-cost flow must equal the explicit primal LP optimum,
    including adversarial graphs with backward edges, self-loops, and
    duplicate edges."""
    rng = np.random.default_rng(11)
    for _ in range(5):
        M = int(rng.integers(4, 10))
        abs_est = rng.normal(6900.0, 50.0, M)
        abs_conf = rng.uniform(0.1, 1.0, M)
        E = int(rng.integers(0, 2 * M))
        edges = (
            np.stack([rng.integers(0, M, E), rng.integers(0, M, E)], axis=1)
            if E
            else np.zeros((0, 2), dtype=int)
        )
        rel_est = rng.normal(0.0, 30.0, E)
        rel_conf = rng.uniform(0.1, 1.0, E)
        dual = lp_smoother(abs_est, abs_conf, edges, rel_est, rel_conf)
        primal = _primal_smoother(abs_est, abs_conf, edges, rel_est, rel_conf)
        np.testing.assert_allclose(dual, primal, rtol=1e-6, atol=1e-6)


def test_malformed_empty_edges_rejected():
    _, abs_est, abs_conf, _, _, _ = _synth()
    with pytest.raises(ValueError, match="rel_edges"):
        lp_smoother(abs_est, abs_conf, np.zeros((0, 3)), np.zeros(0), np.zeros(0))


def test_smoothing_is_deterministic():
    truth_c, abs_est, abs_conf, edges, rel_est, rel_conf = _synth()
    args = (abs_est, abs_conf, edges, rel_est, rel_conf)
    np.testing.assert_allclose(lp_smoother(*args), lp_smoother(*args), rtol=1e-12)


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
        lp_smoother(abs_est[:-1], abs_conf, edges, rel_est, rel_conf)  # length mismatch
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
    with pytest.raises(ValueError):
        lp_smoother(abs_est, abs_conf, edges.astype(float), rel_est, rel_conf)  # non-integer

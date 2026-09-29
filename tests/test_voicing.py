"""Tests for rin.lp.estimate_voicing on synthetic confidences."""

import numpy as np
import pytest

from rin.lp import estimate_voicing


def _synth(M=50, seed=0):
    rng = np.random.default_rng(seed)
    abs_conf = rng.uniform(0.3, 1.0, M)
    edges = np.stack([np.arange(M - 1), np.arange(1, M)], axis=1)
    rel_conf = np.full(M - 1, 0.9)
    return abs_conf, edges, rel_conf


def test_voicing_in_unit_range():
    abs_conf, edges, rel_conf = _synth()
    voicing = estimate_voicing(abs_conf, edges, rel_conf)
    assert voicing.shape == abs_conf.shape
    assert bool(((voicing >= 0) & (voicing <= 1)).all())


def test_empty_edges_gives_zero_voicing():
    # No relative evidence: the geometric mean with an empty RMS is zero.
    abs_conf, _, _ = _synth(M=20)
    voicing = estimate_voicing(abs_conf, np.zeros((0, 2), dtype=int), np.zeros(0))
    np.testing.assert_allclose(voicing, np.zeros(20))


def test_out_of_range_edges_are_dropped():
    abs_conf, edges, rel_conf = _synth(M=50)
    ref = estimate_voicing(abs_conf, edges, rel_conf)
    bad_edges = np.array([[0, 50], [49, 60], [-1, 5]])
    voicing = estimate_voicing(
        abs_conf,
        np.concatenate([edges, bad_edges]),
        np.concatenate([rel_conf, [0.9, 0.9, 0.9]]),
    )
    np.testing.assert_allclose(voicing, ref, rtol=1e-9, atol=1e-9)


def test_zero_weight_edges_are_ignored():
    abs_conf, edges, rel_conf = _synth(M=50)
    ref = estimate_voicing(abs_conf, edges, rel_conf)
    bad_edges = np.array([[0, 49], [10, 40]])
    voicing = estimate_voicing(
        abs_conf,
        np.concatenate([edges, bad_edges]),
        np.concatenate([rel_conf, [0.0, 0.0]]),
    )
    np.testing.assert_allclose(voicing, ref, rtol=1e-9, atol=1e-9)


def test_stronger_relative_confidence_raises_voicing():
    abs_conf, edges, _ = _synth(M=30)
    low = estimate_voicing(abs_conf, edges, np.full(29, 0.1))
    high = estimate_voicing(abs_conf, edges, np.full(29, 0.9))
    assert bool((high >= low).all())
    assert bool((high[1:-1] > low[1:-1]).all())


def test_invalid_inputs_fail_fast():
    abs_conf, edges, rel_conf = _synth(M=50)
    with pytest.raises(ValueError):
        estimate_voicing(-abs_conf, edges, rel_conf)  # negative conf
    with pytest.raises(ValueError):
        estimate_voicing(abs_conf, edges, -rel_conf)
    with pytest.raises(ValueError):
        estimate_voicing(abs_conf, edges.reshape(-1), rel_conf)
    with pytest.raises(ValueError):
        estimate_voicing(abs_conf, edges.astype(float), rel_conf)  # non-integer
    with pytest.raises(ValueError):
        estimate_voicing(abs_conf, edges, rel_conf[:-1])  # length mismatch
    with pytest.raises(ValueError):
        bad = abs_conf.copy()
        bad[0] = np.inf
        estimate_voicing(bad, edges, rel_conf)


def _sparse_voicing(abs_confidences, rel_edges, rel_confidences):
    """Reference: the sparse-adjacency form estimate_voicing was built from.

    Kept as an oracle for the np.bincount accumulation that replaced it, the
    way test_lp.py keeps an explicit primal LP for the dual solver.
    """
    from scipy.sparse import csr_matrix

    M = len(abs_confidences)
    keep = np.all((rel_edges >= 0) & (rel_edges < M), axis=1) & (rel_confidences > 0)
    e, c = rel_edges[keep], rel_confidences[keep]
    d = 1.0 / np.maximum(np.abs(e[:, 1] - e[:, 0]), 1).astype(float) ** 2
    num_s = csr_matrix((d * c**2, (e[:, 0], e[:, 1])), shape=(M, M))
    den_s = csr_matrix((d, (e[:, 0], e[:, 1])), shape=(M, M))
    num = (num_s + num_s.T).sum(axis=1).A1
    den = (den_s + den_s.T).sum(axis=1).A1
    rms = np.sqrt(np.divide(num, np.maximum(den, 1e-10)))
    return np.sqrt(np.clip(abs_confidences, 0.0, 1.0) * np.clip(rms, 0.0, 1.0))


def test_bincount_matches_the_sparse_form():
    """The endpoint accumulation must equal the sparse form it replaced,
    including duplicate edges, self-loops and backward edges."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        M = int(rng.integers(1, 30))
        E = int(rng.integers(0, 3 * M))
        edges = np.stack([rng.integers(0, M, E), rng.integers(0, M, E)], axis=1)
        rel_conf = rng.uniform(0.0, 1.0, E)
        abs_conf = rng.uniform(0.0, 1.0, M)
        np.testing.assert_allclose(
            estimate_voicing(abs_conf, edges, rel_conf),
            _sparse_voicing(abs_conf, edges, rel_conf),
            rtol=1e-12,
            atol=1e-12,
        )


def test_relative_evidence_is_undirected():
    # An edge is evidence for the frames at both ends, whichever way round it
    # was given, so direction and redundant listings must not move the result.
    M = 6
    fwd = np.stack([np.arange(M - 1), np.arange(1, M)], axis=1)
    rev = fwd[:, ::-1].copy()
    conf = np.full(M - 1, 0.5)
    abs_conf = np.full(M, 1.0)
    ref = estimate_voicing(abs_conf, fwd, conf)

    np.testing.assert_allclose(estimate_voicing(abs_conf, rev, conf), ref)
    for redundant in (np.concatenate([fwd, fwd]), np.concatenate([fwd, rev])):
        np.testing.assert_allclose(
            estimate_voicing(abs_conf, redundant, np.concatenate([conf, conf])), ref
        )


def test_self_loop_is_well_behaved():
    # A self-loop adds its weight to both "ends" of the same frame, so it
    # stays in range rather than double-counting into nonsense.
    voicing = estimate_voicing(np.ones(3), np.array([[1, 1]]), np.array([0.64]))
    assert bool(((voicing >= 0) & (voicing <= 1)).all())
    np.testing.assert_allclose(voicing[1], 0.8)  # sqrt(1.0 * 0.64)
    np.testing.assert_allclose(voicing[[0, 2]], 0.0)  # no incident edges


def test_confidences_above_one_are_clamped():
    # Nothing validates confidences <= 1, so the upper bounds in estimate_voicing
    # are the only thing holding the documented [0, 1] return contract.
    voicing = estimate_voicing(np.full(3, 4.0), np.array([[0, 1], [1, 2]]), np.full(2, 4.0))
    assert bool(((voicing >= 0) & (voicing <= 1)).all())

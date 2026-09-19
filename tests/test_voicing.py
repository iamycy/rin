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

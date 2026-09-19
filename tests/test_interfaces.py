"""Contract tests: custom functions can replace the three cores.

The package exposes exactly three core functions; each obeys one of the
``rin.interfaces`` signatures. A caller swaps in their own implementation
by calling their own function instead -- no wrapper, no registration.
"""

import functools
import inspect

import numpy as np

from rin import estimate_voicing, lp_smoother, vqt_diff_calculator

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
    """Pass-through solver: smoothed = absolute."""
    return abs_estimates.copy()


def dummy_voicing(abs_confidences, rel_edges, rel_confidences):
    """Trivial voicing: just the absolute confidences."""
    return np.clip(abs_confidences, 0, 1)


class CallableSolver:
    """A callable object also satisfies the contract -- no plain function needed."""

    def __call__(self, abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
        return identity_solver(
            abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences
        )


def test_builtin_cores_match_documented_signatures():
    est_params = list(inspect.signature(vqt_diff_calculator).parameters)
    assert est_params[:4] == ["x", "sr", "hop_length", "hops"]
    solver_params = list(inspect.signature(lp_smoother).parameters)
    assert solver_params[:5] == [
        "abs_estimates",
        "abs_confidences",
        "rel_edges",
        "rel_estimates",
        "rel_confidences",
    ]
    voicing_params = list(inspect.signature(estimate_voicing).parameters)
    assert voicing_params == ["abs_confidences", "rel_edges", "rel_confidences"]


def test_caller_wires_custom_functions():
    rng = np.random.default_rng(0)
    n_frames = 40
    x = rng.standard_normal(n_frames * HOP)
    f0_cents = 6900.0 + rng.normal(0, 5, n_frames)
    strength = np.full(n_frames, 0.9)

    edges, est, conf = zero_estimator(x, SR, HOP, (1, 2))
    sm_cents = identity_solver(f0_cents, strength, edges, est, conf)
    voicing = dummy_voicing(strength, edges, conf)
    np.testing.assert_allclose(sm_cents, f0_cents, rtol=1e-12)
    np.testing.assert_allclose(voicing, strength, rtol=1e-12)


def test_any_callable_shape_is_accepted():
    # lambda, functools.partial, and callable objects all obey the contract.
    rng = np.random.default_rng(1)
    n_frames = 30
    f0_cents = np.full(n_frames, 6900.0)
    strength = np.full(n_frames, 0.9)
    x = rng.standard_normal(n_frames * HOP)

    est = functools.partial(zero_estimator)
    solve = CallableSolver()
    edges, _, _ = est(x, SR, HOP, (1,))
    sm_cents = solve(f0_cents, strength, edges, np.zeros(0), np.zeros(0))
    np.testing.assert_allclose(sm_cents, f0_cents, rtol=1e-12)
    np.testing.assert_allclose(dummy_voicing(strength, edges, np.zeros(0)), strength)

    lam = lambda xc, s, h, hp: zero_estimator(xc, s, h, hp)  # noqa: E731
    edges, _, _ = lam(x, SR, HOP, (1,))
    assert edges.shape[1] == 2

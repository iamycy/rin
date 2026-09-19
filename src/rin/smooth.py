"""RIN pitch smoothing: fuse absolute pitch estimates with multi-hop relative
pitch differences through the network-flow LP smoother.

The entry point is :func:`smooth`: absolute F0 (Hz) plus a mono waveform in,
smoothed F0 (Hz) plus voicing probabilities out. The absolute estimates may
come from any tracker; ``NaN`` marks unvoiced frames.

Both stages are pluggable: pass a :class:`rin.interfaces.DifferenceEstimator`
as ``estimator`` to replace the built-in VQT cross-correlation, and/or a
:class:`rin.interfaces.Solver` as ``solver`` to replace the built-in LP.
"""

import numpy as np

from ._utils import cent2hz, hz2cent
from .interfaces import DifferenceEstimator, Solver
from .lp import lp_smoother
from .relative import (
    DEFAULT_CORR_MODE,
    DEFAULT_WEIGHTING,
    MAX_DIFF_CENTS,
    VQT_BINS_PER_OCT,
    VQT_N_BINS,
    _vqt_xcorr_setup,
    compute_vqt,
    hop_diff,
)

# Frame hop in seconds shared by the VQT grid and the output contour.
PERIOD = 0.02

# The relative offset set the paper reports RIN with.
DEFAULT_HOPS = (1, 2, 3, 5)


def build_hop_cache(
    x: np.ndarray,
    sr: int,
    hops,
    *,
    period: float = PERIOD,
    max_diff_cents: float = MAX_DIFF_CENTS,
    vqt_bins_per_octave: int = VQT_BINS_PER_OCT,
    vqt_n_bins: int = VQT_N_BINS,
    weighting: str = DEFAULT_WEIGHTING,
    corr_mode: str = DEFAULT_CORR_MODE,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the VQT once, then per-hop ``(edges, estimates, confidences)``.

    Returns ``{hop: (edges[int], estimates[cents], confidences)}``. ``edges``
    are frame-index pairs into the ``period``-spaced frame grid.
    """
    hop_length = int(period * sr)
    V = compute_vqt(x, sr, hop_length, bins_per_octave=vqt_bins_per_octave, n_bins=vqt_n_bins)
    diff_unit = 1200 / vqt_bins_per_octave
    max_diff_bins = int(max_diff_cents / diff_unit)
    setup = _vqt_xcorr_setup(V, max_diff_bins, corr_mode=corr_mode)
    cache = {}
    for h in hops:
        edges, est, conf, _ = hop_diff(
            V, h, max_diff_bins, diff_unit, setup=setup, weighting=weighting
        )
        cache[int(h)] = (edges.astype(int), est, conf)
    return cache


def _check_estimator(estimator) -> DifferenceEstimator:
    if not isinstance(estimator, DifferenceEstimator):
        raise TypeError(
            "estimator must implement rin.interfaces.DifferenceEstimator: "
            "an estimate(x, sr, hop_length, hops) method returning "
            "(edges, estimates_cents, confidences)."
        )
    return estimator


def _check_solver(solver) -> Solver:
    if not isinstance(solver, Solver):
        raise TypeError(
            "solver must implement rin.interfaces.Solver: "
            "a solve(abs_estimates, abs_confidences, rel_edges, rel_estimates, "
            "rel_confidences) method returning (smooth_pitch_cents, voicing)."
        )
    return solver


def rin_smooth(
    f0: np.ndarray,
    strength: np.ndarray,
    hop_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    hops,
    *,
    solver: Solver | None = None,
    dual_form: bool = True,
    return_voicing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Fuse absolute F0 with the relative edges of ``hops``.

    Args:
        f0: (M,) absolute pitch in Hz; ``NaN`` marks unvoiced frames.
        strength: (M,) voicing strength / confidence in [0, 1].
        hop_cache: per-hop ``(edges, estimates, confidences)`` as returned by
            :func:`build_hop_cache`, on the same frame grid as ``f0``.
        hops: subset of ``hop_cache`` keys to fuse.
        solver: custom :class:`rin.interfaces.Solver`; defaults to the
            built-in network-flow LP (:func:`rin.lp.lp_smoother`).
        dual_form: solve the dual network-circulation LP (default) instead of
            the primal slack-variable LP. Only used with the default solver.
        return_voicing: also return the fused voicing probabilities.

    Returns:
        Smoothed F0 in Hz, or ``(f0_smooth, voicing)`` if ``return_voicing``.
    """
    f0 = np.nan_to_num(np.asarray(f0, dtype=float), nan=1.0)
    strength = np.asarray(strength, dtype=float)
    edges = np.concatenate([hop_cache[h][0] for h in hops], axis=0)
    est = np.concatenate([hop_cache[h][1] for h in hops])
    conf = np.concatenate([hop_cache[h][2] for h in hops])
    if solver is None:
        smooth_cents, voicing, _ = lp_smoother(
            hz2cent(f0), strength, edges, est, conf, dual_form=dual_form
        )
    else:
        smooth_cents, voicing = _check_solver(solver).solve(hz2cent(f0), strength, edges, est, conf)
    f0_smooth = cent2hz(smooth_cents)
    if return_voicing:
        return f0_smooth, voicing
    return f0_smooth


def smooth(
    f0: np.ndarray,
    strength: np.ndarray,
    x: np.ndarray,
    sr: int,
    hops=DEFAULT_HOPS,
    *,
    estimator: DifferenceEstimator | None = None,
    solver: Solver | None = None,
    period: float = PERIOD,
    max_diff_cents: float = MAX_DIFF_CENTS,
    vqt_bins_per_octave: int = VQT_BINS_PER_OCT,
    vqt_n_bins: int = VQT_N_BINS,
    weighting: str = DEFAULT_WEIGHTING,
    corr_mode: str = DEFAULT_CORR_MODE,
    dual_form: bool = True,
    return_voicing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Smooth an absolute F0 contour with RIN.

    Args:
        f0: (M,) absolute pitch in Hz on a ``period``-spaced grid;
            ``NaN`` marks unvoiced frames.
        strength: (M,) voicing strength / confidence in [0, 1].
        x: mono input audio.
        sr: sample rate in Hz.
        hops: relative offset set, e.g. ``(1, 2, 3, 5)``.
        estimator: custom :class:`rin.interfaces.DifferenceEstimator`;
            defaults to the built-in VQT cross-correlation.
        solver: custom :class:`rin.interfaces.Solver`; defaults to the
            built-in network-flow LP.
        period: frame hop in seconds.
        max_diff_cents: maximum VQT pitch-shift search in cents.
        vqt_bins_per_octave: VQT resolution.
        vqt_n_bins: number of VQT bins.
        weighting: confidence weighting scheme for relative estimates.
        corr_mode: ``"pearson"`` or ``"cosine"`` VQT cross-correlation.
        dual_form: solve the dual network-circulation LP (default).
            Only used with the default solver.
        return_voicing: also return the fused voicing probabilities.

    Returns:
        Smoothed F0 in Hz, or ``(f0_smooth, voicing)`` if ``return_voicing``.
    """
    hop_length = int(period * sr)
    hops = tuple(hops)
    if estimator is None:
        hop_cache = build_hop_cache(
            x,
            sr,
            hops,
            period=period,
            max_diff_cents=max_diff_cents,
            vqt_bins_per_octave=vqt_bins_per_octave,
            vqt_n_bins=vqt_n_bins,
            weighting=weighting,
            corr_mode=corr_mode,
        )
        edges = np.concatenate([hop_cache[h][0] for h in hops], axis=0)
        rel_est = np.concatenate([hop_cache[h][1] for h in hops])
        rel_conf = np.concatenate([hop_cache[h][2] for h in hops])
    else:
        edges, rel_est, rel_conf = _check_estimator(estimator).estimate(
            np.asarray(x), int(sr), hop_length, hops
        )

    f0_cents = hz2cent(np.nan_to_num(np.asarray(f0, dtype=float), nan=1.0))
    strength = np.asarray(strength, dtype=float)
    if solver is None:
        smooth_cents, voicing, _ = lp_smoother(
            f0_cents, strength, edges, rel_est, rel_conf, dual_form=dual_form
        )
    else:
        smooth_cents, voicing = _check_solver(solver).solve(
            f0_cents, strength, edges, rel_est, rel_conf
        )
    f0_smooth = cent2hz(smooth_cents)
    if return_voicing:
        return f0_smooth, voicing
    return f0_smooth

"""High-level RIN pipeline: the paper's version in one call.

Chains three stages -- a difference estimator, a solver, and a voicing
estimator -- with the paper's fixed settings by default:

1. :func:`rin.relative.vqt_diff_calculator` -- Pearson normalized
   cross-correlation of VQT magnitude slices, arcsin x peak2mean confidence
   weighting.
2. :func:`rin.lp.lp_smoother` -- dual min-cost circulation LP fusion.
3. :func:`rin.lp.estimate_voicing` -- distance-weighted RMS voicing fusion.

Each stage is an injectable callable obeying the contracts in
:mod:`rin.interfaces` (any function, lambda, ``functools.partial``, or
callable object works), so :func:`smooth_pitch` doubles as a swappable
pipeline: pass your own implementations to replace any stage while keeping
the one-call convenience.

The package never converts pitch units: absolute and relative estimates
must share one pitch domain (the built-in estimator outputs cents), and
converting to or from the caller's own domain (Hz, MIDI, ...) is the
caller's responsibility.
"""

from collections.abc import Sequence

import numpy as np

from .interfaces import DifferenceEstimator, Solver, VoicingEstimator
from .lp import estimate_voicing, lp_smoother
from .relative import vqt_diff_calculator

# Hop set reported in the paper.
HOPS = (1, 2, 3, 5)


def smooth_pitch(
    x: np.ndarray,
    f0: np.ndarray,
    strength: np.ndarray,
    sr: int,
    hop_length: int,
    hops: Sequence[int] = HOPS,
    difference_estimator: DifferenceEstimator = vqt_diff_calculator,
    solver: Solver = lp_smoother,
    voicing_estimator: VoicingEstimator = estimate_voicing,
    **estimator_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth absolute pitch estimates with RIN, as reported in the paper.

    Chains a difference estimator, a solver, and a voicing estimator in
    one call. The defaults reproduce the paper; pass your own callables
    (obeying the contracts in :mod:`rin.interfaces`) to swap any stage.

    Parameters
    ----------
    x : np.ndarray [shape=(..., n)]
        Mono input audio signal.
    f0 : np.ndarray [shape=(M,)]
        Absolute pitch per frame, in cents; NaN (or inf) marks unvoiced frames.
    strength : np.ndarray [shape=(M,)]
        Tracker voicing confidence in [0, 1].
    sr : int
        Sample rate in Hz.
    hop_length : int
        Hop length in samples (shared by the estimator and ``f0``).
    hops : sequence of int
        Hop differences for the relative estimator.
    difference_estimator : DifferenceEstimator, optional
        ``(x, sr, hop_length, hops, **estimator_kwargs) -> (edges,
        estimates, confidences)``. Defaults to
        :func:`rin.relative.vqt_diff_calculator`.
    solver : Solver, optional
        ``(abs_estimates, abs_confidences, rel_edges, rel_estimates,
        rel_confidences) -> smooth_pitch``. Defaults to
        :func:`rin.lp.lp_smoother`.
    voicing_estimator : VoicingEstimator, optional
        ``(abs_confidences, rel_edges, rel_confidences) -> voicing``.
        Defaults to :func:`rin.lp.estimate_voicing`.
    **estimator_kwargs
        Extra keyword arguments forwarded to ``difference_estimator``
        (e.g. ``max_diff_cents``, ``bins_per_octave``, ``n_bins`` for the
        built-in VQT estimator).

    Returns
    -------
    f0_smooth : np.ndarray [shape=(M,)]
        Smoothed pitch, in cents.
    voicing : np.ndarray [shape=(M,)]
        Voicing probabilities in [0, 1].

    Raises
    ------
    ValueError
        If ``f0``/``strength`` are not 1-D of equal length, or their length
        does not match the estimator's frame count.
    """
    f0 = np.asarray(f0, dtype=float)
    strength = np.asarray(strength, dtype=float)
    if f0.ndim != 1 or strength.shape != f0.shape:
        raise ValueError("f0 and strength must be 1-D arrays of the same length")

    edges, estimates, confidences = difference_estimator(
        x, sr, hop_length, hops, **estimator_kwargs
    )
    # The tracker frames must line up 1:1 with the estimator's frames;
    # out-of-range edges would otherwise be silently dropped downstream.
    # (With fewer than two frames there are no edges; fall back to the VQT
    # frame count directly.)
    n_frames = int(edges.max()) + 1 if edges.size else 1 + len(x) // hop_length
    if len(f0) != n_frames:
        raise ValueError(
            f"f0 has {len(f0)} frames but the estimator produced "
            f"{n_frames}; both must use the same hop_length grid"
        )

    # Unvoiced (non-finite) frames get zero absolute confidence, so the LP
    # bridges them through the relative edges alone.
    voiced = np.isfinite(f0)
    abs_est = np.where(voiced, f0, 0.0)
    abs_conf = np.where(voiced, strength, 0.0)

    f0_smooth = solver(abs_est, abs_conf, edges, estimates, confidences)
    voicing = voicing_estimator(abs_conf, edges, confidences)
    return f0_smooth, voicing

"""High-level RIN pipeline: the paper's version in one call.

Chains a difference estimator, a solver and a voicing estimator, defaulting
to the paper's three:

1. :func:`rin.relative.vqt_diff_calculator`: Pearson normalized
   cross-correlation of VQT magnitude slices, arcsin x peak2mean confidence
   weighting.
2. :func:`rin.lp.lp_smoother`: dual min-cost circulation LP fusion.
3. :func:`rin.lp.estimate_voicing`: distance-weighted RMS voicing fusion.

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
HOPS = (1, 5)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth absolute pitch estimates with RIN, as reported in the paper.

    Chains a difference estimator, a solver, and a voicing estimator in
    one call. The defaults reproduce the paper; pass your own callables
    (obeying the contracts in :mod:`rin.interfaces`) to swap any stage.
    Configure a stage with :func:`functools.partial`, e.g.
    ``difference_estimator=partial(vqt_diff_calculator,
    max_diff_cents=500.0)``; ``smooth_pitch`` itself takes no
    stage-specific keyword arguments.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Mono (1-D) input audio signal.
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
        Called as ``difference_estimator(x, sr, hop_length, hops)``;
        returns ``(edges, estimates, confidences)``. Defaults to
        :func:`rin.relative.vqt_diff_calculator`.
    solver : Solver, optional
        ``(abs_estimates, abs_confidences, rel_edges, rel_estimates,
        rel_confidences) -> smooth_pitch``. Defaults to
        :func:`rin.lp.lp_smoother`.
    voicing_estimator : VoicingEstimator, optional
        ``(abs_confidences, rel_edges, rel_confidences) -> voicing``.
        Defaults to :func:`rin.lp.estimate_voicing`.

    Returns
    -------
    f0_smooth : np.ndarray [shape=(M,)]
        Smoothed pitch, in cents.
    voicing : np.ndarray [shape=(M,)]
        Voicing probabilities in [0, 1].

    Raises
    ------
    ValueError
        If ``f0``/``strength`` are not 1-D of equal length, if an edge from
        the estimator references a frame outside ``[0, len(f0))``, or if
        ``f0`` is longer than the ``hop_length`` grid allows.
    """
    f0 = np.asarray(f0, dtype=float)
    strength = np.asarray(strength, dtype=float)
    if f0.ndim != 1 or strength.shape != f0.shape:
        raise ValueError("f0 and strength must be 1-D arrays of the same length")

    edges, estimates, confidences = difference_estimator(x, sr, hop_length, hops)
    # ``f0`` must sit on the estimator's frame grid: every edge must index
    # into it (an out-of-range one is silently dropped downstream), and it may
    # not be longer than the hop grid holds (e.g. computed at a different
    # hop_length). Not an equality check against ``edges.max() + 1``, which is
    # only a lower bound, since an estimator may leave trailing frames
    # unconnected, so it would reject a correctly sized ``f0``. When there are
    # no edges at all, though, nothing constrains ``f0`` from below (the
    # upper-bound check on edges is what does that in the general case), so
    # the hop grid is required exactly.
    n_samples = np.shape(x)[-1]
    grid_frames = 1 + n_samples // hop_length
    if edges.size:
        lo, hi = int(edges.min()), int(edges.max())
        if lo < 0 or hi >= len(f0):
            bad = hi if hi >= len(f0) else lo
            raise ValueError(
                f"the estimator produced an edge referencing frame {bad}, outside "
                f"[0, {len(f0)}); both must use the same hop_length grid"
            )
    if len(f0) > grid_frames or (not edges.size and len(f0) != grid_frames):
        raise ValueError(
            f"f0 has {len(f0)} frames but hop_length={hop_length} over "
            f"{n_samples} samples gives {grid_frames}; "
            "both must use the same hop_length grid"
        )

    # Unvoiced (non-finite) frames get zero absolute confidence, so the LP
    # positions them from the relative edges. (``lp_smoother`` floors absolute
    # weights at 1e-6, leaving a residual pull toward 0.0 that is negligible
    # beside real edge weights.)
    voiced = np.isfinite(f0)
    abs_est = np.where(voiced, f0, 0.0)
    abs_conf = np.where(voiced, strength, 0.0)

    f0_smooth = solver(abs_est, abs_conf, edges, estimates, confidences)
    voicing = voicing_estimator(abs_conf, edges, confidences)
    return f0_smooth, voicing

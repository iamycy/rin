"""High-level RIN pipeline: the paper's version in one call.

Chains the three core functions with the paper's fixed settings:

1. :func:`rin.relative.vqt_diff_calculator` -- Pearson normalized
   cross-correlation of VQT magnitude slices, arcsin x peak2mean confidence
   weighting.
2. :func:`rin.lp.lp_smoother` -- dual min-cost circulation LP fusion.
3. :func:`rin.lp.estimate_voicing` -- distance-weighted RMS voicing fusion.

For custom behavior, call the three cores directly instead (see
:mod:`rin.interfaces` for the contracts).
"""

from collections.abc import Sequence

import numpy as np

from .lp import estimate_voicing, lp_smoother
from .relative import vqt_diff_calculator

# Hop set reported in the paper.
HOPS = (1, 2, 3, 5)


def smooth_pitch(
    x: np.ndarray,
    f0_hz: np.ndarray,
    strength: np.ndarray,
    sr: int,
    hop_length: int,
    hops: Sequence[int] = HOPS,
    **vqt_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a tracker's F0 with RIN, exactly as reported in the paper.

    Args:
        x: mono input audio signal.
        f0_hz: (M,) absolute pitch in Hz on the ``hop_length`` grid;
            NaN marks unvoiced frames.
        strength: (M,) tracker voicing confidence in [0, 1].
        sr: sample rate in Hz.
        hop_length: hop length in samples (shared by the VQT and ``f0_hz``).
        hops: hop differences for the relative estimator.
        vqt_kwargs: extra keyword arguments forwarded to
            :func:`rin.relative.vqt_diff_calculator` (e.g. ``max_diff_cents``,
            ``bins_per_octave``, ``n_bins``).

    Returns:
        f0_smooth_hz: (M,) smoothed pitch in Hz.
        voicing: (M,) voicing probabilities in [0, 1].

    Raises:
        ValueError: if ``f0_hz``/``strength`` are not 1-D of equal length,
            or their length does not match the estimator's frame count.
    """
    f0_hz = np.asarray(f0_hz, dtype=float)
    strength = np.asarray(strength, dtype=float)
    if f0_hz.ndim != 1 or strength.shape != f0_hz.shape:
        raise ValueError("f0_hz and strength must be 1-D arrays of the same length")

    edges, estimates, confidences = vqt_diff_calculator(x, sr, hop_length, hops=hops, **vqt_kwargs)
    # The tracker frames must line up 1:1 with the estimator's frames;
    # out-of-range edges would otherwise be silently dropped downstream.
    n_frames = int(edges.max()) + 1 if edges.size else 0
    if len(f0_hz) != n_frames:
        raise ValueError(
            f"f0_hz has {len(f0_hz)} frames but the estimator produced "
            f"{n_frames}; both must use the same hop_length grid"
        )

    # Unvoiced (NaN) frames get zero absolute confidence, so the LP bridges
    # them through the relative edges alone.
    voiced = np.isfinite(f0_hz)
    f0_cents = 1200.0 * np.log2(np.where(voiced, f0_hz, 1.0))
    abs_conf = np.where(voiced, strength, 0.0)

    smooth_cents = lp_smoother(f0_cents, abs_conf, edges, estimates, confidences)
    voicing = estimate_voicing(abs_conf, edges, confidences)
    return 2.0 ** (smooth_cents / 1200.0), voicing

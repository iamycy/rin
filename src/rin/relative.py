"""Multi-hop relative pitch estimation from a magnitude VQT.

For each hop ``m`` in a hop set, every frame pair ``(n, n+m)`` is scored by
normalized cross-correlation of their VQT magnitude slices over a bounded
pitch-shift window; the peak (with parabolic interpolation) gives the relative
pitch difference in cents, and the peak's shape gives a confidence weight.
"""

from collections.abc import Sequence

import numpy as np
from librosa import vqt

from ._utils import parabolic_interpolation

# --------------------------------------------------------------------------
# Fixed estimation path (the paper's reported setup)
# --------------------------------------------------------------------------

VQT_N_BINS = 252
VQT_BINS_PER_OCT = 36
MAX_DIFF_CENTS = 600.0

# Pearson (mean-subtracted) normalized cross-correlation of VQT magnitude
# slices; confidence weighting is arcsin (dot-based) x peak2mean
# (flatness-based). These are not configurable -- this is the paper's path.


def _dot_weight(corr_max: np.ndarray) -> np.ndarray:
    """Dot-based component: decompressed match quality (angular distance)."""
    corr_max_pos = np.clip(corr_max, 0, 1)
    return 2 / np.pi * np.arcsin(corr_max_pos)


def _flat_weight(dots: np.ndarray, corr_max: np.ndarray) -> np.ndarray:
    """Flatness-based component: how sharp/distinct the correlation peak is."""
    dots_pos = np.maximum(dots, 0.0)
    denom = np.maximum(corr_max, 1e-10)
    return 1 - np.mean(dots_pos, axis=0) / denom


def compute_vqt(
    x: np.ndarray,
    sr: int,
    hop_length: int,
    bins_per_octave: int = VQT_BINS_PER_OCT,
    **vqt_kwargs,
) -> np.ndarray:
    """Magnitude VQT spectrogram, factored out so it can be computed once and
    reused across many diff-hops.
    """
    return np.abs(
        vqt(
            x,
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            **vqt_kwargs,
        )
    )


def _vqt_xcorr_setup(V: np.ndarray, max_diff_bins: int):
    """Precompute the padded VQT, sliding-window norms and sliding-window view
    used by every hop's Pearson normalized cross-correlation. Depends only on
    ``V`` and ``max_diff_bins`` (not on the hop), so it is computed once per clip.
    """
    F = V.shape[0]
    padded_V = np.pad(
        V,
        ((max_diff_bins + 1, max_diff_bins), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    cumsum_V2 = np.cumsum(padded_V**2, axis=0)
    sliding_V = np.lib.stride_tricks.sliding_window_view(padded_V[1:, :], F, axis=0)

    cumsum_V = np.cumsum(padded_V, axis=0)
    shifts = np.arange(max_diff_bins, -max_diff_bins - 1, -1)
    K_tau = (F - np.abs(shifts))[:, None]

    window_sum = cumsum_V[F:, :] - cumsum_V[:-F, :]
    window_sum2 = cumsum_V2[F:, :] - cumsum_V2[:-F, :]
    sliding_V_norm = np.sqrt(np.maximum(window_sum2 - (window_sum**2) / K_tau, 1e-10))
    return sliding_V, sliding_V_norm, window_sum, K_tau


def hop_diff(
    V: np.ndarray,
    jump: int,
    max_diff_bins: int,
    diff_unit: float,
    setup=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation pitch differences for a *single* hop ``jump``.

    Returns ``(edges, estimates, confidences)`` for this hop. ``edges`` are
    frame-index pairs, ``estimates`` are in cents.
    """
    if setup is None:
        setup = _vqt_xcorr_setup(V, max_diff_bins)
    sliding_V, sliding_V_norm, window_sum, K_tau = setup
    M = V.shape[1]

    raw_dot = np.linalg.vecdot(sliding_V[:, :-jump, :], V[:, jump:].T, axis=-1)
    cov_dot = raw_dot - (window_sum[:, :-jump] * window_sum[::-1, jump:]) / K_tau
    dots = cov_dot / (sliding_V_norm[:, :-jump] * sliding_V_norm[::-1, jump:])

    # Correlations are rectified before the peak is picked. A real fraction of
    # the Pearson correlations are negative; a frame whose whole column is
    # negative collapses to argmax 0, which hit_boundary already zeroes the
    # weight for.
    np.maximum(dots, 0.0, out=dots)

    idx = np.argmax(dots, axis=0)
    hit_boundary = (idx == 0) | (idx == dots.shape[0] - 1)

    abc = -np.take_along_axis(
        dots,
        np.minimum(np.stack([idx - 1, idx, idx + 1], axis=0), dots.shape[0] - 1),
        axis=0,
    )
    interpolated_dot, p = parabolic_interpolation(*abc)

    pitch_diffs = -(idx + p - max_diff_bins) * diff_unit
    # fixed paper weighting: arcsin (dot-based) x peak2mean (flatness-based)
    corr_max = -interpolated_dot
    diff_probs = _dot_weight(corr_max) * _flat_weight(dots, corr_max)
    diff_probs[hit_boundary] = 0

    indexes = np.arange(M)
    edges = np.stack([indexes[:-jump], indexes[jump:]], axis=1)
    return edges, pitch_diffs, diff_probs


def vqt_diff_calculator(
    x: np.ndarray,
    sr: int,
    hop_length: int,
    hops: Sequence[int] = (1,),
    max_diff_cents: float = MAX_DIFF_CENTS,
    bins_per_octave: int = VQT_BINS_PER_OCT,
    **vqt_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-hop relative pitch differences from a waveform.

    Uses the paper's fixed estimation path: Pearson (mean-subtracted)
    normalized cross-correlation of VQT magnitude slices, with the
    arcsin x peak2mean confidence weighting.

    Args:
        x: mono input audio signal.
        sr: sample rate in Hz.
        hop_length: hop length for VQT, in samples.
        hops: hop differences, e.g. ``(1,)`` for adjacent frames,
            ``(1, 2)`` for adjacent and next-adjacent frames.
        max_diff_cents: maximum allowed pitch difference in cents.
        bins_per_octave: bins per octave for VQT.
        vqt_kwargs: additional keyword arguments for VQT, e.g. ``n_bins``.

    Returns:
        edges (*, 2): frame-index pairs for every pitch difference.
        estimates (*,): pitch differences in cents.
        confidences (*,): confidence for each pitch difference.
    """
    assert all(h > 0 for h in hops), "hops must be positive integers"

    V = compute_vqt(x, sr, hop_length, bins_per_octave=bins_per_octave, **vqt_kwargs)

    diff_unit = 1200 / bins_per_octave
    max_diff_bins = int(max_diff_cents / diff_unit)

    setup = _vqt_xcorr_setup(V, max_diff_bins)

    diff_pitch_estimates = []
    diff_pitch_confidences = []
    edges = []
    # iterate through hop differences to calculate pitch differences
    for jump in hops:
        hop_edges, pitch_diffs, diff_probs = hop_diff(
            V,
            jump,
            max_diff_bins,
            diff_unit,
            setup=setup,
        )
        diff_pitch_estimates.append(pitch_diffs)
        diff_pitch_confidences.append(diff_probs)
        edges.append(hop_edges)

    return (
        np.concatenate(edges, axis=0),
        np.concatenate(diff_pitch_estimates),
        np.concatenate(diff_pitch_confidences),
    )

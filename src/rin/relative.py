"""Multi-hop relative pitch estimation from a magnitude VQT.

For each hop ``m`` in a hop set, every frame pair ``(n, n+m)`` is scored by
normalized cross-correlation of their VQT magnitude slices over a bounded
pitch-shift window; the peak (with parabolic interpolation) gives the relative
pitch difference in cents, and the peak's shape gives a confidence weight.

The estimation path is fixed -- see :func:`vqt_diff_calculator` -- and only
the VQT's shape is a parameter.
"""

from collections.abc import Sequence
from operator import index as _index

import numpy as np
from librosa import vqt

# The paper's reported VQT settings.
VQT_N_BINS = 252
VQT_BINS_PER_OCT = 36
MAX_DIFF_CENTS = 600.0


def _parabolic_interpolation(a, b, c):
    """Interpolate a peak over the three-point stencil ``(-1, 0, +1)``.

    Returns the interpolated peak value and its offset from the center bin.

    Notes
    -----
    Kept local on purpose: librosa's equivalent (``_parabolic_interpolation``
    in ``librosa.core.pitch``) is private, takes the full array instead of a
    stencil triple, and has different edge semantics, so it is not a drop-in
    replacement.
    """
    denom = a - 2 * b + c
    offset = np.divide(
        0.5 * (a - c),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=np.abs(denom) > 1e-10,
    )
    return b - 0.25 * (a - c) * offset, offset


def _dot_weight(corr_max: np.ndarray) -> np.ndarray:
    """Dot-based component: decompressed match quality (angular distance)."""
    return 2 / np.pi * np.arcsin(np.clip(corr_max, 0, 1))


def _flat_weight(dots: np.ndarray, corr_max: np.ndarray) -> np.ndarray:
    """Flatness-based component: how sharp/distinct the correlation peak is."""
    return 1 - np.mean(dots, axis=0) / np.maximum(corr_max, 1e-10)


def compute_vqt(
    x: np.ndarray,
    sr: int,
    hop_length: int,
    bins_per_octave: int = VQT_BINS_PER_OCT,
    n_bins: int = VQT_N_BINS,
    **vqt_kwargs,
) -> np.ndarray:
    """Compute the magnitude VQT spectrogram.

    Factored out so it can be computed once and reused across many diff-hops.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Mono (1-D) input audio signal.
    sr : int
        Sample rate in Hz.
    hop_length : int
        Hop length for the VQT, in samples.
    bins_per_octave : int
        Bins per octave for the VQT.
    n_bins : int
        Total number of VQT bins.
    **vqt_kwargs
        Additional keyword arguments passed to :func:`librosa.vqt`.

    Returns
    -------
    np.ndarray [shape=(n_bins, n_frames)]
        Magnitude VQT spectrogram.
    """
    return np.abs(
        vqt(
            x,
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins,
            **vqt_kwargs,
        )
    )


def _vqt_xcorr_setup(V: np.ndarray, max_diff_bins: int):
    """Precompute the padded VQT, sliding-window norms, and sliding-window view.

    Used by every hop's Pearson normalized cross-correlation. Depends only on
    ``V`` and ``max_diff_bins`` (not on the hop), so it is computed once per clip.

    Parameters
    ----------
    V : np.ndarray [shape=(n_bins, n_frames)]
        Magnitude VQT spectrogram.
    max_diff_bins : int
        Maximum pitch-shift search radius, in VQT bins.

    Returns
    -------
    sliding_V : np.ndarray
        Sliding-window view of the padded VQT.
    sliding_V_norm : np.ndarray
        Norm of each sliding window (mean-subtracted).
    window_sum : np.ndarray
        Windowed sums of the VQT magnitudes, for mean subtraction.
    K_tau : np.ndarray
        Overlap lengths per shift, for mean subtraction.
    """
    # Load-bearing: the correlation's reduction axes are contiguous only under
    # F order, which librosa.vqt already gives. C order costs 4.7x.
    V = np.asfortranarray(V)
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
    """Cross-correlation pitch differences for a single hop.

    Parameters
    ----------
    V : np.ndarray [shape=(n_bins, n_frames)]
        Magnitude VQT spectrogram.
    jump : int
        Frame offset between correlated frame pairs.
    max_diff_bins : int
        Maximum pitch-shift search radius, in VQT bins.
    diff_unit : float
        Cents per VQT bin.
    setup : tuple or None
        Precomputed values from ``_vqt_xcorr_setup``. If None, they are
        computed from ``V`` and ``max_diff_bins``.

    Returns
    -------
    edges : np.ndarray [shape=(n_frames - jump, 2)]
        Frame-index pairs ``(n, n + jump)``.
    estimates : np.ndarray [shape=(n_frames - jump,)]
        Pitch differences in cents.
    confidences : np.ndarray [shape=(n_frames - jump,)]
        Confidence weight for each pitch difference.

    Raises
    ------
    ValueError
        If ``jump`` is not a positive integer.
    """
    try:
        jump = _index(jump)
    except TypeError:
        raise ValueError("jump must be an integer") from None
    if jump < 1:
        raise ValueError("jump must be positive")
    # Before the setup call, so V[:, jump:].T shares its layout -- see _vqt_xcorr_setup.
    V = np.asfortranarray(V)
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

    # At a boundary peak, idx - 1 == -1 indexes from the far end of the shift
    # axis, so the stencil is not a neighbourhood of the peak. Harmless: both
    # of its consumers are zeroed at hit_boundary -- p just below, and
    # diff_probs further down.
    abc = -np.take_along_axis(
        dots,
        np.minimum(np.stack([idx - 1, idx, idx + 1], axis=0), dots.shape[0] - 1),
        axis=0,
    )
    interpolated_dot, p = _parabolic_interpolation(*abc)
    # Keeps every estimate inside the +-max_diff_cents window: a boundary
    # peak's parabola fit is meaningless, so drop the sub-bin offset and
    # report the search limit itself. These edges are zero-weighted, so this
    # only affects what a custom solver sees.
    p = np.where(hit_boundary, 0.0, p)

    pitch_diffs = -(idx + p - max_diff_bins) * diff_unit
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

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Mono (1-D) input audio signal.
    sr : int
        Sample rate in Hz.
    hop_length : int
        Hop length for the VQT, in samples.
    hops : sequence of int
        Hop differences, e.g. ``(1,)`` for adjacent frames, ``(1, 2)`` for
        adjacent and next-adjacent frames.
    max_diff_cents : float
        Maximum allowed pitch difference in cents.
    bins_per_octave : int
        Bins per octave for the VQT.
    **vqt_kwargs
        Additional keyword arguments for :func:`librosa.vqt`, e.g. ``n_bins``
        (defaults to 252, the paper's setting).

    Returns
    -------
    edges : np.ndarray [shape=(E, 2)]
        Frame-index pairs for every pitch difference.
    estimates : np.ndarray [shape=(E,)]
        Pitch differences in cents.
    confidences : np.ndarray [shape=(E,)]
        Confidence for each pitch difference.

    Raises
    ------
    ValueError
        If ``x`` is not a mono (1-D) waveform, ``hops`` is empty or contains
        non-positive values, ``bins_per_octave`` is not positive, or
        ``max_diff_cents`` allows less than one VQT bin of search or more
        than ``n_bins`` of it.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a mono (1-D) waveform")
    try:
        hops = tuple(_index(h) for h in hops)
    except TypeError:
        raise ValueError("hops must be integers") from None
    if not hops:
        raise ValueError("hops must be non-empty")
    if any(h <= 0 for h in hops):
        raise ValueError("hops must be positive integers")
    if bins_per_octave <= 0:
        raise ValueError("bins_per_octave must be positive")

    V = compute_vqt(x, sr, hop_length, bins_per_octave=bins_per_octave, **vqt_kwargs)

    diff_unit = 1200 / bins_per_octave
    max_diff_bins = int(max_diff_cents / diff_unit)
    if max_diff_bins < 1:
        raise ValueError("max_diff_cents must allow at least one VQT bin of search")
    if max_diff_bins >= V.shape[0]:
        # K_tau = n_bins - |shift| is a division denominator in the Pearson
        # normalization; once the search radius reaches n_bins it hits zero and
        # the correlations come back NaN.
        raise ValueError(
            f"max_diff_cents={max_diff_cents} needs a search radius of "
            f"{max_diff_bins} bins but the VQT has only {V.shape[0]}; "
            "increase n_bins or reduce max_diff_cents"
        )

    setup = _vqt_xcorr_setup(V, max_diff_bins)

    diff_pitch_estimates = []
    diff_pitch_confidences = []
    edges = []
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

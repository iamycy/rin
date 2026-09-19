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
# Published defaults (match the paper's reported setup)
# --------------------------------------------------------------------------

VQT_N_BINS = 252
VQT_BINS_PER_OCT = 36
MAX_DIFF_CENTS = 600.0

# Similarity used for the VQT cross-correlation. "pearson" subtracts the slice
# mean first; "cosine" is the uncentered normalized cross-correlation.
DEFAULT_CORR_MODE = "pearson"

# Confidence weighting scheme for the pitch difference estimates.
DEFAULT_WEIGHTING = "arcsin_peak2mean"

_DOT_WEIGHTS = ("raw", "arcsin")
_FLAT_WEIGHTS = ("flatness", "peak2mean", "gm_peak")
_HYBRID_WEIGHTS = frozenset(f"{d}_{f}" for d in _DOT_WEIGHTS for f in _FLAT_WEIGHTS)


def _dot_weight(corr_max: np.ndarray, name: str) -> np.ndarray:
    """Dot-based component: how well the two frames match at the best shift."""
    corr_max_pos = np.clip(corr_max, 0, 1)
    if name == "raw":
        return corr_max_pos
    # "arcsin": decompressed match quality (angular distance)
    return 2 / np.pi * np.arcsin(corr_max_pos)


def _flat_weight(dots: np.ndarray, corr_max: np.ndarray, name: str) -> np.ndarray:
    """Flatness-based component: how sharp/distinct the correlation peak is."""
    dots_pos = np.maximum(dots, 0.0)
    denom = np.maximum(corr_max, 1e-10)
    if name == "flatness":
        flatness = -20 * (
            np.log10(np.mean(dots_pos, axis=0) + 1e-10)
            - np.mean(np.log10(dots_pos + 1e-10), axis=0)
        )
        return 1 - 10 ** (flatness / 20)
    if name == "peak2mean":
        return 1 - np.mean(dots_pos, axis=0) / denom
    # "gm_peak"
    gm = 10 ** np.mean(np.log10(dots_pos + 1e-10), axis=0)
    return 1 - gm / denom


def _resolve_weighting(name: str):
    """Split a weighting scheme into its (dot-based, flatness-based) terms;
    either may be None. Raises on an unknown scheme."""
    if name in _DOT_WEIGHTS:
        return name, None
    if name in _FLAT_WEIGHTS:
        return None, name
    if name in _HYBRID_WEIGHTS:
        return name.split("_", 1)
    raise ValueError(f"Unknown weighting scheme: {name}")


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


def _vqt_xcorr_setup(V: np.ndarray, max_diff_bins: int, corr_mode: str = DEFAULT_CORR_MODE):
    """Precompute the padded VQT, sliding-window norms and sliding-window view
    used by every hop's normalized cross-correlation. Depends only on ``V``,
    ``max_diff_bins``, and ``corr_mode`` (not on the hop), so it is computed once per clip.
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

    if corr_mode == "pearson":
        cumsum_V = np.cumsum(padded_V, axis=0)
        shifts = np.arange(max_diff_bins, -max_diff_bins - 1, -1)
        K_tau = (F - np.abs(shifts))[:, None]

        window_sum = cumsum_V[F:, :] - cumsum_V[:-F, :]
        window_sum2 = cumsum_V2[F:, :] - cumsum_V2[:-F, :]
        sliding_V_norm = np.sqrt(np.maximum(window_sum2 - (window_sum**2) / K_tau, 1e-10))
        return sliding_V, sliding_V_norm, window_sum, K_tau, corr_mode
    elif corr_mode == "cosine":
        sliding_V_norm = np.sqrt(np.maximum(cumsum_V2[F:, :] - cumsum_V2[:-F, :], 1e-10))
        return sliding_V, sliding_V_norm, None, None, corr_mode
    else:
        raise ValueError(f"Unknown corr_mode: {corr_mode}")


def hop_diff(
    V: np.ndarray,
    jump: int,
    max_diff_bins: int,
    diff_unit: float,
    setup=None,
    weighting: str = DEFAULT_WEIGHTING,
    corr_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation pitch differences for a *single* hop ``jump``.

    Returns ``(edges, estimates, confidences, voicing_contrib)`` for this hop,
    where ``voicing_contrib`` is the per-edge ``diff_probs * (-interpolated_dot)``
    term. ``edges`` are frame-index pairs, ``estimates`` are in cents.
    """
    if setup is None:
        setup = _vqt_xcorr_setup(
            V,
            max_diff_bins,
            corr_mode=DEFAULT_CORR_MODE if corr_mode is None else corr_mode,
        )
    # Both branches of _vqt_xcorr_setup return a 5-tuple, so the mode the setup was
    # built with is the one that decides. A caller that passes both a setup and a
    # conflicting corr_mode would otherwise silently get the setup's mode.
    sliding_V, sliding_V_norm, window_sum, K_tau, mode = setup
    if corr_mode is not None and corr_mode != mode:
        raise ValueError(
            f"corr_mode={corr_mode!r} but setup was built for {mode!r}; "
            "the setup decides, so this call would compute the other one."
        )
    M = V.shape[1]

    raw_dot = np.linalg.vecdot(sliding_V[:, :-jump, :], V[:, jump:].T, axis=-1)
    if mode == "pearson":
        cov_dot = raw_dot - (window_sum[:, :-jump] * window_sum[::-1, jump:]) / K_tau
        dots = cov_dot / (sliding_V_norm[:, :-jump] * sliding_V_norm[::-1, jump:])
    else:
        dots = raw_dot / (sliding_V_norm[:, :-jump] * sliding_V_norm[::-1, jump:])

    # Correlations are rectified before the peak is picked. Under "cosine"
    # V >= 0 and this is a strict no-op; under "pearson" a real fraction of the
    # correlations are negative. A frame whose whole column is negative collapses
    # to argmax 0, which hit_boundary already zeroes the weight for.
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
    # confidence weighting scheme for the pitch difference estimate
    dot_name, flat_name = _resolve_weighting(weighting)
    corr_max = -interpolated_dot
    if dot_name is None:
        diff_probs = _flat_weight(dots, corr_max, flat_name)
    elif flat_name is None:
        diff_probs = _dot_weight(corr_max, dot_name)
    else:
        diff_probs = _dot_weight(corr_max, dot_name) * _flat_weight(dots, corr_max, flat_name)
    diff_probs[hit_boundary] = 0

    indexes = np.arange(M)
    edges = np.stack([indexes[:-jump], indexes[jump:]], axis=1)
    voicing_contrib = diff_probs * (-interpolated_dot)
    return edges, pitch_diffs, diff_probs, voicing_contrib


def vqt_diff_calculator(
    x: np.ndarray,
    sr: int,
    hop_length: int,
    hops: Sequence[int] = (1,),
    max_diff_cents: float = MAX_DIFF_CENTS,
    bins_per_octave: int = VQT_BINS_PER_OCT,
    weighting: str = DEFAULT_WEIGHTING,
    corr_mode: str = DEFAULT_CORR_MODE,
    **vqt_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-hop relative pitch differences from a waveform.

    Args:
        x: mono input audio signal.
        sr: sample rate in Hz.
        hop_length: hop length for VQT, in samples.
        hops: hop differences, e.g. ``(1,)`` for adjacent frames,
            ``(1, 2)`` for adjacent and next-adjacent frames.
        max_diff_cents: maximum allowed pitch difference in cents.
        bins_per_octave: bins per octave for VQT.
        weighting: confidence weighting scheme, one of ``"raw"``,
            ``"arcsin"``, ``"flatness"``, ``"peak2mean"``, ``"gm_peak"``,
            or a multiplicative hybrid of a dot-based and a flatness-based
            term, e.g. ``"arcsin_flatness"``.
        corr_mode: ``"cosine"`` (standard normalized cross-correlation) or
            ``"pearson"`` (mean-subtracted cross-correlation).
        vqt_kwargs: additional keyword arguments for VQT, e.g. ``n_bins``.

    Returns:
        edges (*, 2): frame-index pairs for every pitch difference.
        estimates (*,): pitch differences in cents.
        confidences (*,): confidence for each pitch difference.
    """
    assert all(h > 0 for h in hops), "hops must be positive integers"
    # fail fast on an unknown scheme, before the expensive VQT
    _resolve_weighting(weighting)

    V = compute_vqt(x, sr, hop_length, bins_per_octave=bins_per_octave, **vqt_kwargs)

    diff_unit = 1200 / bins_per_octave
    max_diff_bins = int(max_diff_cents / diff_unit)

    setup = _vqt_xcorr_setup(V, max_diff_bins, corr_mode=corr_mode)

    diff_pitch_estimates = []
    diff_pitch_confidences = []
    edges = []
    # iterate through hop differences to calculate pitch differences
    for jump in hops:
        hop_edges, pitch_diffs, diff_probs, _ = hop_diff(
            V,
            jump,
            max_diff_bins,
            diff_unit,
            setup=setup,
            weighting=weighting,
            corr_mode=corr_mode,
        )
        diff_pitch_estimates.append(pitch_diffs)
        diff_pitch_confidences.append(diff_probs)
        edges.append(hop_edges)

    return (
        np.concatenate(edges, axis=0),
        np.concatenate(diff_pitch_estimates),
        np.concatenate(diff_pitch_confidences),
    )
